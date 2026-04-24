"""Learner — weekly Bayesian / EMA threshold updater (Task 25).

Railway Cron friday 12:00 MSK (``0 9 * * 5`` UTC). Reads last-7-days
concluded hypotheses (``state IN ('confirmed','rejected')``), rolls up
the evidence per ``hypothesis_type`` + kill-switch family, and writes
the data-driven overlay into ``sda_state[learned_thresholds]`` as a
JSONB document. Downstream consumers (``decision_engine.evaluate`` and
the 5 learnable guards in :mod:`agent_runtime.tools.kill_switches`) read
that row at tick time and merge it over hardcoded defaults.

**Why EMA-in-MVP (Decision recorded in decisions.md):** a full Bayesian
update (beta prior + likelihood) needs ≥ 200 confirmed/rejected samples
to give a stable posterior. On the 14-day shadow + 60-day assisted ramp
we accumulate ~50-70 confirmed + ~20-30 rejected — statistically thin.
Rolling EMA (α=0.3) with a ±30 % clamp per weekly tick is the safe MVP;
the strategy is pluggable via ``settings.SDA_LEARNER_STRATEGY`` (``ema``
default | ``bayesian`` stub) so we can flip without a rewrite once the
sample count justifies it.

**Safety invariants grep-auditable here:**

* ``LEARNABLE_KILL_SWITCHES`` excludes ``neg_kw_floor`` and
  ``conversion_integrity`` — those are security invariants, never
  learned (enforced by unit test
  ``test_learnable_kill_switches_excludes_security_invariants``).
* ``MIN_SAMPLES_PER_THRESHOLD = 3`` — fewer evidence points → threshold
  stays unchanged (``EMAUpdater.update`` returns ``current``).
* ``CLAMP_PCT = 0.30`` — maximum relative move per weekly iteration,
  caps divergence at 1.3×/0.7× per week.
* ``dry_run=True`` computes the new thresholds and returns them **but
  does not write** ``sda_state``.
* Job is idempotent: the 7-day rolling window gives the same answer for
  a repeat invocation inside the same minute.

**TODO(integration):**

1. Register in ``agent_runtime/jobs/__init__.py::JOB_REGISTRY`` as
   ``"learner": learner.run``.
2. Add Railway Cron row in ``railway.toml``: ``schedule = "0 9 * * 5"``
   (UTC = 12:00 МСК), HTTP-trigger to ``/run/learner`` guarded by
   ``SDA_INTERNAL_API_KEY`` (Task 5b bearer).
3. ``agent_runtime.config.Settings`` — add
   ``SDA_LEARNER_STRATEGY: Literal["ema", "bayesian"] = "ema"``.
4. ``agent_runtime.decision_engine`` — export
   ``DEFAULT_ENGINE_THRESHOLDS`` + add ``async
   _load_learned_overlay(pool)`` that reads
   ``sda_state.learned_thresholds.decision_engine`` and merges it over
   the hardcoded map (fall back to defaults when missing or
   ``updated_at < now() - interval '14 days'``).
5. ``agent_runtime.tools.kill_switches`` — extend
   ``KillSwitchContext`` with ``learned_overlay: dict`` (today's class
   attribute constants ``SURGE_MULTIPLIER`` / ``P90_MULTIPLIER`` /
   ``PRODUCTIVITY_FLOOR`` / ``WEEKLY_CAP_FRACTION`` /
   ``JACCARD_FLOOR`` consult ``context.learned_overlay.get(self.name,
   {})`` before falling back to their ClassVar default).
6. ``agent_runtime.main`` — register ``POST /run/learner`` (bearer-auth)
   delegating to :func:`run`.

Until integration steps 3-6 land, this module is standalone and
harmless — it reads, it writes ``sda_state.learned_thresholds``, and
anyone reading that key gets a well-typed document. No other module
crashes just because the overlay is present but unconsumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, Protocol

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


# --- tunables (module constants, grep-auditable) ----------------------------

MIN_SAMPLES_PER_THRESHOLD: int = 3
EMA_ALPHA: float = 0.3
CLAMP_PCT: float = 0.30
_HYPOTHESES_WINDOW_DAYS: int = 7
_HYPOTHESES_LIMIT: int = 500
_OVERLAY_STATE_KEY: str = "learned_thresholds"

LearnerStrategy = Literal["ema", "bayesian"]


# ---- single source of truth for thresholds the learner owns -----------------
# We deliberately duplicate small numbers here rather than reach into
# decision_engine / kill_switches module internals. Rationale:
#   * This module is Wave 4; those modules are Wave 1.
#   * If the Wave 1 defaults change we want a deterministic failure at the
#     integration step, not a silent number swap.
# When the integration PR lands, Wave 1 will export DEFAULT_ENGINE_THRESHOLDS /
# DEFAULT_GUARD_THRESHOLDS and this block collapses to `from ... import ...`.

DEFAULT_ENGINE_THRESHOLDS: dict[str, dict[str, float]] = {
    # per-hypothesis_type overridables; keys MUST match
    # agent_runtime.models.HypothesisType values.
    "ad": {"affected_budget_pct": 0.10, "min_confident_samples": 30.0},
    "neg_kw": {"affected_budget_pct": 0.05, "min_confident_samples": 20.0},
    "image": {"affected_budget_pct": 0.10, "min_confident_samples": 30.0},
    "landing": {"affected_budget_pct": 0.20, "min_confident_samples": 50.0},
    "new_camp": {"affected_budget_pct": 0.30, "min_confident_samples": 50.0},
    "format_change": {"affected_budget_pct": 0.30, "min_confident_samples": 80.0},
    "strategy_switch": {"affected_budget_pct": 0.40, "min_confident_samples": 100.0},
    "account_level": {"affected_budget_pct": 0.10, "min_confident_samples": 30.0},
}

DEFAULT_GUARD_THRESHOLDS: dict[str, dict[str, float]] = {
    "budget_cap": {"daily_multiplier": 1.5},  # matches BudgetCap.SURGE_MULTIPLIER
    "cpc_ceiling": {"p90_multiplier": 1.3},  # matches CPCCeiling.P90_MULTIPLIER
    "qs_guard": {"min_productivity": 6.0},  # matches QSGuard.PRODUCTIVITY_FLOOR
    "budget_balance": {"weekly_redistribution_pct": 0.20},  # BudgetBalance.WEEKLY_CAP_FRACTION
    "query_drift": {"jaccard_floor": 0.50},  # QueryDrift.JACCARD_FLOOR
}

# Security invariants — these guards are NEVER learned. Putting them here in
# a frozenset and asserting membership in the unit test prevents a future
# contributor from "opening up" the set without reviewer awareness.
LEARNABLE_KILL_SWITCHES: frozenset[str] = frozenset(
    {"budget_cap", "cpc_ceiling", "qs_guard", "budget_balance", "query_drift"}
)
_NEVER_LEARNED: frozenset[str] = frozenset({"neg_kw_floor", "conversion_integrity"})


# --- data types --------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisEvidence:
    """Projection of ``hypotheses`` row used for threshold updates."""

    id: str
    hypothesis_type: str
    outcome: str  # 'positive' | 'negative' | 'neutral'
    actions: list[dict[str, Any]]
    metrics_before: dict[str, Any]
    metrics_after: dict[str, Any]
    state: str
    created_at: datetime


@dataclass(frozen=True)
class LearnerReport:
    """Structured result returned by :func:`run` (mirrors JSON response)."""

    status: str
    strategy: LearnerStrategy
    samples_count: int
    written: bool
    dry_run: bool
    thresholds: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "strategy": self.strategy,
            "samples_count": self.samples_count,
            "written": self.written,
            "dry_run": self.dry_run,
            "thresholds": self.thresholds,
            "reason": self.reason,
        }


# --- Protocol + strategies ---------------------------------------------------


class ThresholdUpdater(Protocol):
    name: ClassVar[str]

    def update(
        self,
        current: float,
        evidence: list[float],
        hypothesis_type: str,
        threshold_name: str,
    ) -> float: ...


class EMAUpdater:
    """Exponential moving average — MVP default.

    * ``α = 0.3`` — moderate responsiveness; a clean week nudges the
      threshold by at most ``CLAMP_PCT`` (30 %).
    * Below ``MIN_SAMPLES_PER_THRESHOLD`` evidence items → returns
      ``current`` unchanged.  We explicitly prefer "no move" over "move
      by small sample" because a single bad week outweighs four normal
      ones otherwise.
    * Clamp is absolute on ``current`` (NOT on the post-EMA value) — the
      pre-move value is the stable reference.
    """

    name: ClassVar[str] = "ema"

    def update(
        self,
        current: float,
        evidence: list[float],
        hypothesis_type: str,  # unused, accepted for Protocol compliance
        threshold_name: str,  # unused, accepted for Protocol compliance
    ) -> float:
        del hypothesis_type, threshold_name  # silence "unused" linters
        n = len(evidence)
        if n < MIN_SAMPLES_PER_THRESHOLD:
            return current
        mean = sum(evidence) / n
        proposed = EMA_ALPHA * mean + (1.0 - EMA_ALPHA) * current
        lo = current * (1.0 - CLAMP_PCT)
        hi = current * (1.0 + CLAMP_PCT)
        # Handle pathological current<=0: clamp collapses to 0; we then only
        # allow proposed up to |CLAMP_PCT| above 0.
        if current <= 0:
            return max(0.0, min(proposed, abs(CLAMP_PCT)))
        return max(lo, min(proposed, hi))


class BayesianUpdater:
    """Beta-prior + likelihood stub. Fail-safe: raises on invocation.

    Activates only after ≥200 accumulated confirmed+rejected samples
    (decision in ``decisions.md``). Flipping ``SDA_LEARNER_STRATEGY`` to
    ``bayesian`` early → NotImplementedError → Railway alert → rollback.
    """

    name: ClassVar[str] = "bayesian"

    def update(
        self,
        current: float,
        evidence: list[float],
        hypothesis_type: str,
        threshold_name: str,
    ) -> float:
        raise NotImplementedError(
            "BayesianUpdater is a post-MVP upgrade. Switch "
            "SDA_LEARNER_STRATEGY to 'bayesian' only after ≥200 "
            "confirmed+rejected samples land (see decisions.md Task 25 "
            "Post-completion)."
        )


def _updater_factory(strategy: LearnerStrategy) -> ThresholdUpdater:
    if strategy == "ema":
        return EMAUpdater()
    if strategy == "bayesian":
        return BayesianUpdater()
    # Pydantic Literal validation should prevent this; narrow-safe fallback.
    logger.warning("learner: unknown strategy %r, defaulting to EMA", strategy)
    return EMAUpdater()


# --- data loaders ------------------------------------------------------------


async def _load_hypotheses_last_week(
    pool: AsyncConnectionPool,
) -> list[HypothesisEvidence]:
    """Fetch last-7d concluded hypotheses. ``LIMIT 500`` as O(n) safety net.

    ``COALESCE(metrics_after_captured_at, created_at)`` is the "concluded
    at" proxy — ``metrics_after_captured_at`` is set by
    :func:`decision_journal.update_outcome`, so it lands for every
    confirmed/rejected row. Falling back on ``created_at`` covers
    legacy rows (and tests that seed synthetic data without bothering
    to set the column).
    """
    query = """
    SELECT id, hypothesis_type, outcome, actions,
           metrics_before, metrics_after, state, created_at
    FROM hypotheses
    WHERE state IN ('confirmed', 'rejected')
      AND COALESCE(metrics_after_captured_at, created_at)
          >= NOW() - make_interval(days => %s)
    ORDER BY created_at DESC
    LIMIT %s
    """
    rows: list[tuple[Any, ...]] = []
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (_HYPOTHESES_WINDOW_DAYS, _HYPOTHESES_LIMIT))
                rows = list(await cur.fetchall())
    except Exception:
        logger.exception("learner: failed to load hypotheses — degrade to empty set")
        return []

    evidence: list[HypothesisEvidence] = []
    for row in rows:
        try:
            evidence.append(
                HypothesisEvidence(
                    id=str(row[0]),
                    hypothesis_type=str(row[1]),
                    outcome=str(row[2] or ""),
                    actions=list(row[3] or []),
                    metrics_before=dict(row[4] or {}),
                    metrics_after=dict(row[5] or {}),
                    state=str(row[6]),
                    created_at=row[7],
                )
            )
        except (TypeError, ValueError):
            logger.warning("learner: skipping malformed hypothesis row %r", row[0])
            continue
    return evidence


# --- threshold computation ---------------------------------------------------


def _outcome_sign(outcome: str, state: str) -> float:
    """+1 for confirmed/positive, -1 for rejected/negative, 0 otherwise.

    We use the sign to weight evidence: a confirmed positive outcome
    *tightens* the threshold (less tolerance for the guard/engine to
    veto) — the brain got this action through and it paid off; we can
    afford to relax. A rejected action says the guard let through
    something that flopped — tighten.
    """
    out = outcome.lower()
    st = state.lower()
    if st == "confirmed" or out == "positive":
        return 1.0
    if st == "rejected" or out == "negative":
        return -1.0
    return 0.0


def _extract_engine_evidence(
    hypotheses: list[HypothesisEvidence],
    hypothesis_type: str,
    threshold_name: str,
) -> list[float]:
    """Pull raw evidence for one decision_engine threshold.

    * ``affected_budget_pct`` — take the budget delta the action
      actually moved (``params.delta_rub``) normalised to a rough
      fraction. Fallback to the default when fields missing.
    * ``min_confident_samples`` — approximate via the click count we
      observed in ``metrics_before`` (clicks pre-mutation). Low clicks
      → threshold should decrease (we act on less data), high clicks →
      we want more data before relaxing.
    """
    values: list[float] = []
    default = DEFAULT_ENGINE_THRESHOLDS[hypothesis_type][threshold_name]
    for h in hypotheses:
        if h.hypothesis_type != hypothesis_type:
            continue
        sign = _outcome_sign(h.outcome, h.state)
        if threshold_name == "affected_budget_pct":
            for action in h.actions:
                params = action.get("params") or {}
                raw_delta = params.get("delta_rub")
                if raw_delta is None:
                    continue
                try:
                    pct = abs(float(raw_delta)) / 10_000.0  # rough normalisation
                except (TypeError, ValueError):
                    continue
                # weight: confirmed → moves toward current pct;
                # rejected → moves away (add inverse signal).
                values.append(pct if sign >= 0 else max(default * 0.5, default - pct))
        elif threshold_name == "min_confident_samples":
            clicks_raw = h.metrics_before.get("clicks")
            if clicks_raw is None:
                continue
            try:
                clicks = float(clicks_raw)
            except (TypeError, ValueError):
                continue
            if clicks <= 0:
                continue
            # confirmed → threshold moves toward observed clicks (enough
            # to confirm); rejected → require more.
            values.append(clicks if sign >= 0 else clicks * 1.5)
    return values


def _extract_guard_evidence(
    hypotheses: list[HypothesisEvidence],
    guard: str,
    threshold_name: str,
) -> list[float]:
    """Per-guard evidence — read ``metrics_before/after`` hints.

    Kept intentionally simple: production evidence for kill_switches
    lives in ``audit_log`` under ``kill_switch_triggered`` — that
    table is Wave 1. Wiring it into learner needs a join we do not
    want in MVP. For now we infer:

    * confirmed hypothesis → the guard *did* let a good action
      through: we can *relax* its threshold a bit (evidence value
      above current).
    * rejected hypothesis → the guard let a bad action through:
      *tighten* (evidence value below current).
    """
    default = DEFAULT_GUARD_THRESHOLDS[guard][threshold_name]
    values: list[float] = []
    for h in hypotheses:
        sign = _outcome_sign(h.outcome, h.state)
        if sign == 0:
            continue
        # Relax on positive, tighten on negative. Magnitude proportional
        # to |sign|*CLAMP_PCT so single datum cannot move threshold past
        # clamp anyway.
        delta = default * CLAMP_PCT * 0.5 * sign
        values.append(default + delta)
    return values


def _compute_new_thresholds(
    hypotheses: list[HypothesisEvidence],
    updater: ThresholdUpdater,
) -> dict[str, dict[str, dict[str, float]]]:
    """Build the full overlay dict. Empty input → empty dict (caller handles)."""
    if not hypotheses:
        return {"decision_engine": {}, "kill_switches": {}}

    # decision_engine ---------------------------------------------------------
    engine_overlay: dict[str, dict[str, float]] = {}
    for htype, defaults in DEFAULT_ENGINE_THRESHOLDS.items():
        per_type: dict[str, float] = {}
        for thr_name, current in defaults.items():
            evidence = _extract_engine_evidence(hypotheses, htype, thr_name)
            new_val = updater.update(float(current), evidence, htype, thr_name)
            if new_val != current:
                per_type[thr_name] = new_val
        if per_type:
            engine_overlay[htype] = per_type

    # kill_switches -----------------------------------------------------------
    guards_overlay: dict[str, dict[str, float]] = {}
    for guard in LEARNABLE_KILL_SWITCHES:
        assert (
            guard not in _NEVER_LEARNED
        ), f"invariant violated: {guard!r} in both LEARNABLE and _NEVER_LEARNED"
        per_guard: dict[str, float] = {}
        for thr_name, current in DEFAULT_GUARD_THRESHOLDS[guard].items():
            evidence = _extract_guard_evidence(hypotheses, guard, thr_name)
            new_val = updater.update(float(current), evidence, guard, thr_name)
            if new_val != current:
                per_guard[thr_name] = new_val
        if per_guard:
            guards_overlay[guard] = per_guard

    return {"decision_engine": engine_overlay, "kill_switches": guards_overlay}


# --- persistence + reflection ------------------------------------------------


async def _persist_thresholds(
    pool: AsyncConnectionPool,
    payload: dict[str, Any],
    *,
    dry_run: bool,
) -> bool:
    """Write the payload to ``sda_state[learned_thresholds]``.

    Returns True iff the write actually happened. ``dry_run`` logs +
    returns False without touching the DB.
    """
    if dry_run:
        logger.info("learner: dry_run=True — skipping sda_state write")
        return False
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO sda_state (key, value, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (key) DO UPDATE
                        SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    (_OVERLAY_STATE_KEY, Jsonb(payload)),
                )
    except Exception:
        logger.exception("learner: failed to persist learned_thresholds")
        return False
    return True


async def _write_reflection(
    reflection_store: Any,
    new_payload: dict[str, Any],
    old_payload: dict[str, Any],
    samples_count: int,
    strategy: LearnerStrategy,
) -> None:
    """Best-effort audit trail through ``PGReflectionStore``. Never raises."""
    if reflection_store is None:
        return
    try:
        text = (
            f"Learner weekly update: n={samples_count} samples, "
            f"strategy={strategy}; engine keys changed="
            f"{len(new_payload.get('decision_engine') or {})}, "
            f"guard keys changed="
            f"{len(new_payload.get('kill_switches') or {})}"
        )
        await reflection_store.save(
            text=text,
            metadata={
                "job": "learner",
                "strategy": strategy,
                "samples": samples_count,
                "thresholds_before": old_payload,
                "thresholds_after": new_payload,
            },
        )
    except Exception:
        logger.warning("learner: reflection_store.save failed", exc_info=True)


async def _load_previous_overlay(pool: AsyncConnectionPool) -> dict[str, Any]:
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM sda_state WHERE key = %s",
                    (_OVERLAY_STATE_KEY,),
                )
                row = await cur.fetchone()
    except Exception:
        logger.warning("learner: previous overlay read failed; treating as empty", exc_info=True)
        return {}
    if row is None or row[0] is None:
        return {}
    val = row[0]
    if isinstance(val, dict):
        return val
    return {}


# --- public entry point ------------------------------------------------------


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    strategy: LearnerStrategy | None = None,
    reflection_store: Any = None,
) -> dict[str, Any]:
    """Weekly Learner cron tick.

    Args:
        pool: psycopg AsyncConnectionPool — required.
        dry_run: compute + return proposed overlay without writing it.
        strategy: override ``settings.SDA_LEARNER_STRATEGY`` (tests pass
            this explicitly so they don't have to monkey-patch Settings).
        reflection_store: optional PGReflectionStore for audit trail.

    Returns a JSON-serialisable dict suitable for ``/run/learner``.
    """
    chosen_strategy: LearnerStrategy = strategy or _strategy_from_settings()
    updater = _updater_factory(chosen_strategy)
    logger.info(
        "learner start: dry_run=%s strategy=%s",
        dry_run,
        chosen_strategy,
    )

    hypotheses = await _load_hypotheses_last_week(pool)
    samples_count = len(hypotheses)

    if samples_count == 0:
        # Cold start / shadow week / empty table. Do NOT overwrite
        # previous overlay (if any) — keeping stale data is better than
        # regressing toward hardcoded defaults silently. Reflection
        # written to leave a breadcrumb.
        logger.info("learner: no confirmed/rejected hypotheses in last 7d — no-op")
        await _write_reflection(
            reflection_store,
            new_payload={},
            old_payload={},
            samples_count=0,
            strategy=chosen_strategy,
        )
        return LearnerReport(
            status="ok",
            strategy=chosen_strategy,
            samples_count=0,
            written=False,
            dry_run=dry_run,
            thresholds={},
            reason="cold_start_no_outcomes",
        ).as_dict()

    new_thresholds = _compute_new_thresholds(hypotheses, updater)
    now_iso = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "decision_engine": new_thresholds["decision_engine"],
        "kill_switches": new_thresholds["kill_switches"],
        "updated_at": now_iso,
        "samples_count": samples_count,
        "strategy": chosen_strategy,
    }

    old_overlay = await _load_previous_overlay(pool)
    written = await _persist_thresholds(pool, payload, dry_run=dry_run)
    await _write_reflection(
        reflection_store,
        new_payload=payload,
        old_payload=old_overlay,
        samples_count=samples_count,
        strategy=chosen_strategy,
    )

    logger.info(
        "learner done: samples=%d written=%s engine_keys=%d guard_keys=%d",
        samples_count,
        written,
        len(payload["decision_engine"]),
        len(payload["kill_switches"]),
    )

    return LearnerReport(
        status="ok",
        strategy=chosen_strategy,
        samples_count=samples_count,
        written=written,
        dry_run=dry_run,
        thresholds=payload,
    ).as_dict()


def _strategy_from_settings() -> LearnerStrategy:
    """Read ``SDA_LEARNER_STRATEGY`` from settings; default ``ema``.

    Reading lazily (not a module-level call) so tests that do not touch
    Settings at all don't trigger pydantic validation.
    """
    try:
        from agent_runtime.config import Settings  # lazy to avoid import side-effects
    except Exception:
        return "ema"
    try:
        raw = getattr(Settings(), "SDA_LEARNER_STRATEGY", "ema")  # type: ignore[call-arg]
    except Exception:
        return "ema"
    if raw in ("ema", "bayesian"):
        return raw  # type: ignore[return-value]
    logger.warning("learner: SDA_LEARNER_STRATEGY=%r invalid, using 'ema'", raw)
    return "ema"


# CLI entry for Railway Cron (matches impact_tracker_job style).
if __name__ == "__main__":  # pragma: no cover
    import asyncio as _asyncio

    from agent_runtime.config import Settings as _S
    from agent_runtime.db import create_pool as _create_pool

    async def _main() -> None:
        settings = _S()  # type: ignore[call-arg]
        pool = _create_pool(settings.DATABASE_URL)
        await pool.open()
        try:
            result = await run(pool)
            logger.info("learner CLI result: %s", result)
        finally:
            await pool.close()

    _asyncio.run(_main())


__all__ = [
    "CLAMP_PCT",
    "DEFAULT_ENGINE_THRESHOLDS",
    "DEFAULT_GUARD_THRESHOLDS",
    "EMA_ALPHA",
    "LEARNABLE_KILL_SWITCHES",
    "MIN_SAMPLES_PER_THRESHOLD",
    "BayesianUpdater",
    "EMAUpdater",
    "HypothesisEvidence",
    "LearnerReport",
    "LearnerStrategy",
    "ThresholdUpdater",
    "run",
]
