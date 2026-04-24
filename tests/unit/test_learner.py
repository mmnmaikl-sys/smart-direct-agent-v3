"""Unit tests for :mod:`agent_runtime.jobs.learner` (Task 25).

Covers:
* Strategy factory (EMA default, Bayesian fail-safe).
* EMAUpdater math: mean + α blend, ±30% clamp, MIN_SAMPLES=3 guard.
* Security invariant: ``LEARNABLE_KILL_SWITCHES`` excludes
  ``neg_kw_floor`` + ``conversion_integrity``.
* Job behaviour:
    - Empty hypotheses → no-op, written=False, reason=cold_start_no_outcomes.
    - 10 confirmed + 3 rejected → writes overlay with samples_count=13
      and kill_switches keys ⊆ LEARNABLE_KILL_SWITCHES.
    - ``dry_run=True`` → computes but does not persist.
    - Idempotent: two consecutive runs produce the same payload.
    - Reflection store called on both cold-start and live runs.
    - Previous overlay returned when non-empty.

We use a stateful in-memory ``_FakePool`` to avoid pytest-postgresql
overhead in the unit layer (matches the pattern in
``tests/unit/test_bitrix_feedback.py``). Integration with real PG is
covered elsewhere (out of scope for Task 25 unit fan-out).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_runtime.jobs import learner as learner  # noqa: PLC0414 — re-export for tests
from agent_runtime.jobs.learner import (
    CLAMP_PCT,
    DEFAULT_ENGINE_THRESHOLDS,
    DEFAULT_GUARD_THRESHOLDS,
    EMA_ALPHA,
    LEARNABLE_KILL_SWITCHES,
    MIN_SAMPLES_PER_THRESHOLD,
    BayesianUpdater,
    EMAUpdater,
    HypothesisEvidence,
    _compute_new_thresholds,
    _extract_engine_evidence,
    _extract_guard_evidence,
    _outcome_sign,
    _updater_factory,
    run,
)

# --------------------------------------------------------------------- _FakePool


class _FakePool:
    """In-memory stand-in for psycopg_pool.AsyncConnectionPool.

    Interprets only the SQL surface the learner hits:
      * SELECT FROM hypotheses WHERE state IN ('confirmed','rejected') ... LIMIT
      * SELECT value FROM sda_state WHERE key = 'learned_thresholds'
      * INSERT INTO sda_state ... ON CONFLICT DO UPDATE
    """

    def __init__(
        self,
        *,
        hypotheses: list[tuple[Any, ...]] | None = None,
        sda_state: dict[str, Any] | None = None,
        raise_on_hypotheses_read: bool = False,
        raise_on_persist: bool = False,
    ) -> None:
        self.hypotheses_rows: list[tuple[Any, ...]] = list(hypotheses or [])
        self.sda_state: dict[str, Any] = dict(sda_state or {})
        self.sda_state_writes: list[tuple[str, Any]] = []
        self.raise_on_hypotheses_read = raise_on_hypotheses_read
        self.raise_on_persist = raise_on_persist

    def connection(self) -> _FakeConn:
        return _FakeConn(self)


class _FakeConn:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.pool)


class _FakeCursor:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool
        self._fetchone_result: Any = None
        self._fetchall_result: list[Any] = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        normalised = " ".join(sql.split()).lower()

        if "from hypotheses" in normalised and "state in" in normalised:
            if self.pool.raise_on_hypotheses_read:
                raise RuntimeError("simulated PG failure")
            self._fetchall_result = list(self.pool.hypotheses_rows)
            return

        if normalised.startswith("select value from sda_state where key ="):
            assert params is not None
            key = params[0]
            stored = self.pool.sda_state.get(key)
            self._fetchone_result = (stored,) if stored is not None else None
            return

        if normalised.startswith("insert into sda_state") and params is not None:
            if self.pool.raise_on_persist:
                raise RuntimeError("simulated write failure")
            key, value = params[0], params[1]
            unwrapped = _unwrap(value)
            self.pool.sda_state[key] = unwrapped
            self.pool.sda_state_writes.append((key, unwrapped))
            return
        # unknown SQL — accept silently

    async def fetchone(self) -> Any:
        out = self._fetchone_result
        self._fetchone_result = None
        return out

    async def fetchall(self) -> list[Any]:
        out = self._fetchall_result
        self._fetchall_result = []
        return out


def _unwrap(val: Any) -> Any:
    obj = getattr(val, "obj", None)
    return obj if obj is not None else val


# ---------------------------------------------------------------- fixtures


def _make_hypothesis_row(
    *,
    hid: str,
    htype: str,
    outcome: str,
    state: str,
    delta_rub: int = 200,
    clicks_before: int = 100,
    created_at: datetime | None = None,
) -> tuple[Any, ...]:
    created_at = created_at or datetime.now(UTC) - timedelta(days=1)
    actions = [{"type": "add_negatives", "params": {"delta_rub": delta_rub}}]
    return (
        hid,
        htype,
        outcome,
        actions,
        {"clicks": clicks_before, "ctr": 2.0},
        {"clicks": clicks_before + 10, "ctr": 2.3},
        state,
        created_at,
    )


def _seed_10_confirmed_3_rejected() -> list[tuple[Any, ...]]:
    """Spec TDD anchor: 10 confirmed + 3 rejected, varied types."""
    rows: list[tuple[Any, ...]] = []
    types_cycle = ["ad", "neg_kw", "landing", "new_camp", "image", "format_change"]
    for i in range(10):
        rows.append(
            _make_hypothesis_row(
                hid=f"c{i}",
                htype=types_cycle[i % len(types_cycle)],
                outcome="positive",
                state="confirmed",
                delta_rub=200 + i * 50,
                clicks_before=80 + i * 5,
            )
        )
    for i in range(3):
        rows.append(
            _make_hypothesis_row(
                hid=f"r{i}",
                htype=types_cycle[i % len(types_cycle)],
                outcome="negative",
                state="rejected",
                delta_rub=600 + i * 100,
                clicks_before=200 + i * 20,
            )
        )
    return rows


# ============================================================================
# pure helpers / math
# ============================================================================


def test_learnable_kill_switches_excludes_security_invariants() -> None:
    assert "neg_kw_floor" not in LEARNABLE_KILL_SWITCHES
    assert "conversion_integrity" not in LEARNABLE_KILL_SWITCHES
    assert LEARNABLE_KILL_SWITCHES == frozenset(
        {"budget_cap", "cpc_ceiling", "qs_guard", "budget_balance", "query_drift"}
    )
    assert len(LEARNABLE_KILL_SWITCHES) == 5


def test_ema_updater_clamps_to_30_percent_upward() -> None:
    """Big positive evidence should be clamped to +30 %."""
    up = EMAUpdater()
    result = up.update(current=1.5, evidence=[3.0, 3.0, 3.0], hypothesis_type="", threshold_name="")
    assert 1.5 * (1 - CLAMP_PCT) <= result <= 1.5 * (1 + CLAMP_PCT)
    # With evidence far above current and α=0.3, result should sit at the +30% ceiling.
    assert result == pytest.approx(1.5 * 1.30, rel=1e-6)


def test_ema_updater_clamps_downward() -> None:
    """Evidence far below current stays within the −30 % floor.

    EMA proposed = 0.3 * mean(evidence) + 0.7 * current. With current=2.0
    and evidence far below, proposed could be either below the clamp floor
    (then floor applies) or already above it (then proposed wins). Either
    way it MUST NOT go below ``current * (1 − CLAMP_PCT)``.
    """
    up = EMAUpdater()
    current = 2.0
    result = up.update(
        current=current,
        evidence=[-5.0, -5.0, -5.0],
        hypothesis_type="",
        threshold_name="",
    )
    floor = current * (1 - CLAMP_PCT)
    assert result == pytest.approx(floor, rel=1e-6)


def test_ema_updater_below_min_samples_returns_current() -> None:
    up = EMAUpdater()
    # n=2 < MIN_SAMPLES=3 → passthrough
    assert up.update(1.5, [3.0, 3.0], "ad", "x") == 1.5
    # empty evidence
    assert up.update(1.5, [], "ad", "x") == 1.5


def test_ema_updater_exact_math_at_min_samples() -> None:
    up = EMAUpdater()
    current = 1.0
    evidence = [1.05, 1.05, 1.05]  # small positive drift, below clamp band
    expected = EMA_ALPHA * 1.05 + (1 - EMA_ALPHA) * current
    got = up.update(current, evidence, "ad", "x")
    assert got == pytest.approx(expected, rel=1e-9)


def test_ema_updater_zero_current_safe() -> None:
    up = EMAUpdater()
    # pathological current=0 → clamp collapses; result bounded by CLAMP_PCT.
    got = up.update(0.0, [0.2, 0.2, 0.2], "ad", "x")
    assert 0.0 <= got <= CLAMP_PCT


def test_bayesian_updater_raises_not_implemented_in_mvp() -> None:
    b = BayesianUpdater()
    with pytest.raises(NotImplementedError, match="post-MVP"):
        b.update(1.5, [1.0, 1.0, 1.0], "ad", "x")


def test_updater_factory_returns_expected_types() -> None:
    assert isinstance(_updater_factory("ema"), EMAUpdater)
    assert isinstance(_updater_factory("bayesian"), BayesianUpdater)
    # unknown → defaults to EMA (defensive)
    assert isinstance(_updater_factory("unknown"), EMAUpdater)  # type: ignore[arg-type]


def test_outcome_sign_mapping() -> None:
    assert _outcome_sign("positive", "confirmed") == 1.0
    assert _outcome_sign("negative", "rejected") == -1.0
    assert _outcome_sign("neutral", "inconclusive") == 0.0
    # state wins over outcome when one is missing
    assert _outcome_sign("", "confirmed") == 1.0
    assert _outcome_sign("", "rejected") == -1.0


def test_extract_engine_evidence_respects_hypothesis_type() -> None:
    rows = _seed_10_confirmed_3_rejected()
    hypotheses = [
        HypothesisEvidence(
            id=str(r[0]),
            hypothesis_type=str(r[1]),
            outcome=str(r[2]),
            actions=list(r[3]),
            metrics_before=dict(r[4]),
            metrics_after=dict(r[5]),
            state=str(r[6]),
            created_at=r[7],
        )
        for r in rows
    ]
    ad_evidence = _extract_engine_evidence(hypotheses, "ad", "affected_budget_pct")
    assert ad_evidence, "ad type should have evidence from the seed"
    all_evidence = sum(1 for h in hypotheses if h.hypothesis_type == "ad")
    # every ad hypothesis has exactly one action with delta_rub, so:
    assert len(ad_evidence) == all_evidence


def test_extract_guard_evidence_tightens_on_rejected() -> None:
    # one rejected row → guard evidence should be below default.
    rows = [
        _make_hypothesis_row(
            hid="r0", htype="ad", outcome="negative", state="rejected", delta_rub=500
        )
    ]
    hypotheses = [_to_evidence(r) for r in rows]
    evidence = _extract_guard_evidence(hypotheses, "budget_cap", "daily_multiplier")
    default = DEFAULT_GUARD_THRESHOLDS["budget_cap"]["daily_multiplier"]
    assert evidence, "one rejected row must produce evidence"
    assert all(v < default for v in evidence)


def test_extract_guard_evidence_relaxes_on_confirmed() -> None:
    rows = [
        _make_hypothesis_row(
            hid="c0", htype="ad", outcome="positive", state="confirmed", delta_rub=500
        )
    ]
    hypotheses = [_to_evidence(r) for r in rows]
    evidence = _extract_guard_evidence(hypotheses, "budget_cap", "daily_multiplier")
    default = DEFAULT_GUARD_THRESHOLDS["budget_cap"]["daily_multiplier"]
    assert evidence
    assert all(v > default for v in evidence)


def test_compute_new_thresholds_empty_input_returns_empty_overlays() -> None:
    out = _compute_new_thresholds([], EMAUpdater())
    assert out == {"decision_engine": {}, "kill_switches": {}}


def test_compute_new_thresholds_populates_expected_guards() -> None:
    rows = _seed_10_confirmed_3_rejected()
    hypotheses = [_to_evidence(r) for r in rows]
    out = _compute_new_thresholds(hypotheses, EMAUpdater())
    # every guard key in overlay must be LEARNABLE.
    for guard_name in out["kill_switches"].keys():
        assert guard_name in LEARNABLE_KILL_SWITCHES


# ============================================================================
# run() — end-to-end with _FakePool
# ============================================================================


@pytest.mark.asyncio
async def test_run_no_hypotheses_keeps_defaults() -> None:
    pool = _FakePool(hypotheses=[])
    store = AsyncMock()
    result = await run(pool, reflection_store=store)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    assert result["samples_count"] == 0
    assert result["written"] is False
    assert result["reason"] == "cold_start_no_outcomes"
    assert result["thresholds"] == {}
    # sda_state untouched
    assert pool.sda_state_writes == []
    # reflection still fired (audit breadcrumb for silent weeks)
    assert store.save.await_count == 1


@pytest.mark.asyncio
async def test_run_with_10_confirmed_3_rejected_updates_thresholds() -> None:
    """TDD anchor: samples_count=13, engine+kill_switches overlays present."""
    pool = _FakePool(hypotheses=_seed_10_confirmed_3_rejected())
    store = AsyncMock()
    result = await run(pool, reflection_store=store)  # type: ignore[arg-type]

    assert result["status"] == "ok"
    assert result["samples_count"] == 13
    assert result["written"] is True
    assert result["dry_run"] is False
    payload = result["thresholds"]
    assert "updated_at" in payload
    assert payload["samples_count"] == 13
    assert payload["strategy"] == "ema"

    # sda_state got exactly one write to learned_thresholds.
    assert len(pool.sda_state_writes) == 1
    key, stored_value = pool.sda_state_writes[0]
    assert key == "learned_thresholds"
    assert stored_value["samples_count"] == 13
    # kill_switches overlay keys are a subset of LEARNABLE_KILL_SWITCHES.
    ks_overlay = stored_value.get("kill_switches") or {}
    assert set(ks_overlay.keys()).issubset(LEARNABLE_KILL_SWITCHES)

    # reflection got called with metadata.job='learner'
    assert store.save.await_count == 1
    call = store.save.await_args
    assert call.kwargs["metadata"]["job"] == "learner"
    assert call.kwargs["metadata"]["strategy"] == "ema"


@pytest.mark.asyncio
async def test_dry_run_logs_but_does_not_write() -> None:
    pool = _FakePool(hypotheses=_seed_10_confirmed_3_rejected())
    result = await run(pool, dry_run=True)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    assert result["written"] is False
    assert result["dry_run"] is True
    assert result["samples_count"] == 13
    assert result["thresholds"]  # proposed payload is present
    assert pool.sda_state_writes == []


@pytest.mark.asyncio
async def test_idempotent_same_window_same_result() -> None:
    pool = _FakePool(hypotheses=_seed_10_confirmed_3_rejected())
    first = await run(pool)  # type: ignore[arg-type]
    # Drop updated_at because it embeds `now()` which trivially differs.
    second = await run(pool)  # type: ignore[arg-type]

    first_payload = {k: v for k, v in first["thresholds"].items() if k != "updated_at"}
    second_payload = {k: v for k, v in second["thresholds"].items() if k != "updated_at"}
    assert first_payload == second_payload
    assert first["samples_count"] == second["samples_count"] == 13


@pytest.mark.asyncio
async def test_run_with_bayesian_strategy_raises_from_compute() -> None:
    pool = _FakePool(hypotheses=_seed_10_confirmed_3_rejected())
    # strategy='bayesian' engages BayesianUpdater which raises on .update().
    # The run() orchestrator does not catch NotImplementedError — it should
    # bubble out so a misconfiguration triggers loud alerting, not silent
    # default fallback.
    with pytest.raises(NotImplementedError):
        await run(pool, strategy="bayesian")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_reflection_not_called_when_store_is_none() -> None:
    # Guard against accidental crash when reflection_store is None.
    pool = _FakePool(hypotheses=_seed_10_confirmed_3_rejected())
    result = await run(pool, reflection_store=None)  # type: ignore[arg-type]
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_persist_failure_is_swallowed_as_written_false() -> None:
    """A PG write failure returns written=False but does not crash the job."""
    pool = _FakePool(
        hypotheses=_seed_10_confirmed_3_rejected(),
        raise_on_persist=True,
    )
    result = await run(pool)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    assert result["written"] is False
    assert result["samples_count"] == 13


@pytest.mark.asyncio
async def test_hypotheses_read_failure_degrades_to_cold_start() -> None:
    pool = _FakePool(
        hypotheses=_seed_10_confirmed_3_rejected(),
        raise_on_hypotheses_read=True,
    )
    result = await run(pool)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    assert result["samples_count"] == 0
    assert result["reason"] == "cold_start_no_outcomes"


@pytest.mark.asyncio
async def test_reflection_failure_does_not_crash_run() -> None:
    pool = _FakePool(hypotheses=_seed_10_confirmed_3_rejected())
    store = AsyncMock()
    store.save.side_effect = RuntimeError("reflection DB down")
    # Should NOT raise — reflection is best-effort.
    result = await run(pool, reflection_store=store)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    assert result["written"] is True


@pytest.mark.asyncio
async def test_previous_overlay_read_when_present() -> None:
    """If sda_state already carries learned_thresholds, reflection metadata
    includes the 'before' snapshot so ops can diff."""
    prev = {
        "decision_engine": {"ad": {"affected_budget_pct": 0.08}},
        "kill_switches": {"budget_cap": {"daily_multiplier": 1.45}},
        "updated_at": "2026-04-01T00:00:00+00:00",
        "samples_count": 5,
        "strategy": "ema",
    }
    pool = _FakePool(
        hypotheses=_seed_10_confirmed_3_rejected(),
        sda_state={"learned_thresholds": prev},
    )
    store = AsyncMock()
    result = await run(pool, reflection_store=store)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    call = store.save.await_args
    assert call.kwargs["metadata"]["thresholds_before"] == prev


@pytest.mark.asyncio
async def test_kill_switches_overlay_never_contains_security_invariants() -> None:
    """Security invariant enforced end-to-end, not just at the constant."""
    pool = _FakePool(hypotheses=_seed_10_confirmed_3_rejected())
    result = await run(pool)  # type: ignore[arg-type]
    ks = result["thresholds"].get("kill_switches") or {}
    assert "neg_kw_floor" not in ks
    assert "conversion_integrity" not in ks


# ---------------------------------------------------------------- helpers


def _to_evidence(row: tuple[Any, ...]) -> HypothesisEvidence:
    return HypothesisEvidence(
        id=str(row[0]),
        hypothesis_type=str(row[1]),
        outcome=str(row[2]),
        actions=list(row[3]),
        metrics_before=dict(row[4]),
        metrics_after=dict(row[5]),
        state=str(row[6]),
        created_at=row[7],
    )


def test_default_thresholds_cover_all_hypothesis_types() -> None:
    """Contract with agent_runtime.models.HypothesisType — adding a new
    hypothesis type must bump DEFAULT_ENGINE_THRESHOLDS in the same PR."""
    expected = {
        "ad",
        "neg_kw",
        "image",
        "landing",
        "new_camp",
        "format_change",
        "strategy_switch",
        "account_level",
    }
    assert set(DEFAULT_ENGINE_THRESHOLDS.keys()) == expected


def test_module_constants_are_well_formed() -> None:
    assert 0 < EMA_ALPHA < 1
    assert 0 < CLAMP_PCT < 1
    assert MIN_SAMPLES_PER_THRESHOLD >= 3
    # __all__ exports include the big names
    assert "run" in learner.__all__
    assert "EMAUpdater" in learner.__all__
    assert "BayesianUpdater" in learner.__all__
    assert "LEARNABLE_KILL_SWITCHES" in learner.__all__
