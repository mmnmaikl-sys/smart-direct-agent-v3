"""Regression Watch — post-promote drawdown guard (Task 27).

P3 cron (Railway ``0 */6 * * *``). The 72h window in
:mod:`agent_runtime.impact_tracker` classifies a hypothesis at
promote-time; a *confirmed* hypothesis may still regress 7–14 days later
as traffic mix shifts or auction pressure changes. Without this job the
owner only notices the drop via P&L weeks later. Design per
**tech-spec Decision 6**.

Pipeline per tick:

1. ``SELECT ... FROM hypotheses WHERE state='confirmed' AND promoted_at
    > now() - interval '30 days'``. Uses the Wave-1 index
   ``ix_hypotheses_regression_watch``.
2. For each row: fetch *current* metrics (Direct stats over
   ``[promoted_at, now()]`` + optional Bitrix lead count keyed by
   ``UTM_CAMPAIGN=sda_<id>``), diff against the frozen
   ``baseline_at_promote`` snapshot on the target metric for the
   hypothesis type (CTR / CPA / leads).
3. Classify drawdown using the two-tier, budget-calibrated thresholds
   (Decision 6):

   +----------------+----------+----------+
   | budget_cap_rub |  warn%   |  hard%   |
   +================+==========+==========+
   | ≤ 500          |   25     |   40     |
   +----------------+----------+----------+
   | 501 – 2500     |   20     |   30     |
   +----------------+----------+----------+
   | 2501 – 5000    |   15     |   25     |
   +----------------+----------+----------+
   | ≥ 5001         |   10     |   20     |
   +----------------+----------+----------+

4. Route:

   * ``dd < warn`` → log-only.
   * ``warn ≤ dd < hard`` → Telegram warning ("degraded but not terminal").
     DB untouched, no reflection write.
   * ``dd ≥ hard`` → (unless ``dry_run``) call
     :func:`agent_runtime.impact_tracker.rollback`, write a
     :class:`ReflectionStore` lesson, and shoot a CRITICAL Telegram
     alert. If ``rollback`` raises we still emit reflection + alert so
     the owner knows manual intervention is required.

Trust overlay: ``shadow`` / ``FORBIDDEN_LOCK`` downgrades every hard
verdict to NOTIFY only (mirrors bitrix_feedback). ``dry_run`` suppresses
all side effects regardless of trust.

Degraded-noop: when the ``dispatch_job`` path forwards only ``pool`` and
``dry_run``, every downstream client is ``None``. We return
``status='ok' action='degraded_noop'`` exactly like
``bitrix_feedback`` — the real HTTP handler at ``/run/regression_watch``
is responsible for injecting ``direct``, ``bitrix_client``,
``http_client``, ``reflection_store``, ``settings`` from ``app.state``.

TODO(integration): Decision 6 thresholds SHOULD live in
:mod:`agent_runtime.config` as ``REGRESSION_THRESHOLDS`` per AC #2 of
Task 27. Task-27 parallel batch forbids edits to ``config.py`` —
thresholds are therefore mirrored here as module-level constants with
the same *numbers*. Task 27b will move them to ``config.py`` and leave
``regression_thresholds_for`` as the sole reader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

from psycopg_pool import AsyncConnectionPool

from agent_runtime import impact_tracker
from agent_runtime.trust_levels import TrustLevel, get_trust_level

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- Decision 6
# Mirror of the tech-spec table. Format: (budget_upper_bound_inclusive,
# warning_pct, hard_pct). ``float('inf')`` is the catch-all top tier.
# The list is scanned in order so keep it sorted by the bound.

_REGRESSION_THRESHOLDS: list[tuple[float, int, int]] = [
    (500, 25, 40),
    (2_500, 20, 30),
    (5_000, 15, 25),
    (float("inf"), 10, 20),
]


def _regression_thresholds_for(budget_cap_rub: int) -> tuple[int, int]:
    """Return ``(warn_pct, hard_pct)`` for the calibrated tier.

    Linear scan — the table is 4 rows, branch prediction beats a map
    lookup and the code reads exactly like the tech-spec table.
    """
    for upper, warn, hard in _REGRESSION_THRESHOLDS:
        if budget_cap_rub <= upper:
            return warn, hard
    # Unreachable — last row has upper = inf.
    return _REGRESSION_THRESHOLDS[-1][1], _REGRESSION_THRESHOLDS[-1][2]


# Which metric we diff for each hypothesis type. Mirrors
# ``impact_tracker._TARGET_METRIC`` so the 72h and 30d guards agree on
# the axis.

_TARGET_METRIC_BY_TYPE: dict[str, str] = {
    "ad": "ctr",
    "neg_kw": "cpa",
    "landing": "cr",
    "new_camp": "leads",
    "image": "ctr",
    "format_change": "ctr",
    "strategy_switch": "cpa",
    "account_level": "leads",
}

# CTR / leads / cr → higher is better (drop = regression).
# CPA / cost → lower is better (growth = regression).
_DRAWDOWN_DIRECTION_BY_METRIC: dict[str, Literal["higher_is_better", "lower_is_better"]] = {
    "ctr": "higher_is_better",
    "cr": "higher_is_better",
    "leads": "higher_is_better",
    "clicks": "higher_is_better",
    "cpa": "lower_is_better",
    "cpc": "lower_is_better",
    "cost": "lower_is_better",
}

_MIN_CLICKS_FOR_SIGNAL = 50  # below this the window is too noisy to act on


# --------------------------------------------------------------- protocols


class _DirectLike(Protocol):
    async def get_campaign_stats(
        self, campaign_id: int, date_from: str, date_to: str
    ) -> dict[str, Any]: ...
    async def pause_group(self, ad_group_id: int) -> dict[str, Any]: ...
    async def resume_group(self, ad_group_id: int) -> dict[str, Any]: ...
    async def pause_campaign(self, campaign_id: int) -> dict[str, Any]: ...
    async def resume_campaign(self, campaign_id: int) -> dict[str, Any]: ...
    async def set_bid(
        self, keyword_id: int, bid_rub: int, context_bid_rub: int | None = None
    ) -> dict[str, Any]: ...
    async def add_negatives(self, campaign_id: int, phrases: list[str]) -> dict[str, Any]: ...


class _BitrixLike(Protocol):
    async def get_leads_count_by_utm(self, utm_campaign: str, date_from: str) -> int: ...


class _TelegramLike(Protocol):
    async def send_message(self, *, text: str, priority: str = ...) -> None: ...


class _ReflectionStoreLike(Protocol):
    async def save(self, text: str, metadata: dict[str, Any]) -> None: ...


# --------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class _ConfirmedRow:
    id: str
    agent: str
    hypothesis_type: str
    budget_cap_rub: int
    campaign_id: int | None
    ad_group_id: int | None
    actions: list[dict[str, Any]]
    baseline_at_promote: dict[str, Any]
    promoted_at: datetime


@dataclass(frozen=True)
class _RegressionCheckResult:
    hypothesis_id: str
    verdict: Literal["ok", "skip_no_baseline", "skip_low_signal", "warn", "hard"]
    drawdown_pct: float
    target_metric: str
    warn_pct: int
    hard_pct: int
    current_metrics: dict[str, Any]


# --------------------------------------------------------------- SELECT


async def _select_confirmed_in_window(pool: AsyncConnectionPool) -> list[_ConfirmedRow]:
    """Use ``ix_hypotheses_regression_watch`` index (state + promoted_at)."""
    query = """
    SELECT id, agent, hypothesis_type, budget_cap_rub, campaign_id,
           ad_group_id, actions, baseline_at_promote, promoted_at
    FROM hypotheses
    WHERE state = 'confirmed'
      AND promoted_at > NOW() - INTERVAL '30 days'
    ORDER BY promoted_at DESC
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query)
            rows = await cur.fetchall()
    return [
        _ConfirmedRow(
            id=str(r[0]),
            agent=str(r[1] or ""),
            hypothesis_type=str(r[2]),
            budget_cap_rub=int(r[3] or 0),
            campaign_id=int(r[4]) if r[4] is not None else None,
            ad_group_id=int(r[5]) if r[5] is not None else None,
            actions=list(r[6] or []),
            baseline_at_promote=dict(r[7] or {}),
            promoted_at=r[8],
        )
        for r in rows
    ]


# --------------------------------------------------------------- drawdown math


def _compute_drawdown_pct(
    *,
    baseline_value: float,
    current_value: float,
    direction: Literal["higher_is_better", "lower_is_better"],
) -> float:
    """Positive == degraded. Zero baseline → 0.0 (undefined, treat neutral).

    * ``higher_is_better``: drawdown = (baseline - current) / baseline.
      Drop in CTR from 3.0 → 2.25 = +25%.
    * ``lower_is_better``: drawdown = (current - baseline) / baseline.
      CPA growing 1000 → 1400 = +40% (bad).
    """
    if baseline_value <= 0:
        return 0.0
    if direction == "higher_is_better":
        raw = (baseline_value - current_value) / baseline_value
    else:
        raw = (current_value - baseline_value) / baseline_value
    return raw * 100.0


# --------------------------------------------------------------- current metrics


async def _fetch_current_metrics(
    direct: _DirectLike,
    bitrix: _BitrixLike | None,
    row: _ConfirmedRow,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Assemble a dict comparable with ``baseline_at_promote`` keys.

    Swallows per-source exceptions — Direct/Bitrix hiccups degrade a
    single row into ``skip_low_signal`` rather than sinking the cron.
    """
    window_start = row.promoted_at.date().isoformat()
    window_end = now.date().isoformat()

    merged: dict[str, Any] = {}

    if row.campaign_id is not None:
        try:
            stats = await direct.get_campaign_stats(
                row.campaign_id, date_from=window_start, date_to=window_end
            )
            if isinstance(stats, dict):
                merged.update(stats)
        except Exception:
            logger.exception(
                "regression_watch: direct.get_campaign_stats(%s) failed for %s",
                row.campaign_id,
                row.id,
            )

    if bitrix is not None:
        try:
            leads = await bitrix.get_leads_count_by_utm(
                utm_campaign=f"sda_{row.id}", date_from=window_start
            )
            merged["leads"] = int(leads)
        except Exception:
            logger.exception(
                "regression_watch: bitrix lead lookup failed for %s",
                row.id,
            )

    return merged


# --------------------------------------------------------------- messaging


def _format_warning(row: _ConfirmedRow, dd_pct: float, warn_pct: int) -> str:
    return (
        f"<b>⚠️ Regression warning</b>\n"
        f"Hypothesis <code>{row.id}</code> · agent=<code>{row.agent}</code>\n"
        f"Type <code>{row.hypothesis_type}</code>, "
        f"drawdown <b>{dd_pct:.1f}%</b> (warn ≥ {warn_pct}%)\n"
        f"Budget cap: {row.budget_cap_rub}₽"
    )


def _format_critical(row: _ConfirmedRow, dd_pct: float, hard_pct: int) -> str:
    return (
        f"<b>🛑 Regression hard — rollback</b>\n"
        f"Hypothesis <code>{row.id}</code> · agent=<code>{row.agent}</code>\n"
        f"Type <code>{row.hypothesis_type}</code>, "
        f"drawdown <b>{dd_pct:.1f}%</b> (hard ≥ {hard_pct}%)\n"
        f"Budget cap: {row.budget_cap_rub}₽"
    )


def _format_rollback_failed(row: _ConfirmedRow, dd_pct: float, err: str) -> str:
    return (
        f"<b>🚨 Regression rollback FAILED — manual intervention</b>\n"
        f"Hypothesis <code>{row.id}</code> · agent=<code>{row.agent}</code>\n"
        f"Drawdown <b>{dd_pct:.1f}%</b>\n"
        f"Error: <code>{err}</code>"
    )


# --------------------------------------------------------------- core loop


async def _check_one(
    row: _ConfirmedRow,
    *,
    direct: _DirectLike,
    bitrix: _BitrixLike | None,
    now: datetime,
) -> _RegressionCheckResult:
    """Pure per-row classification, no side effects."""
    target = _TARGET_METRIC_BY_TYPE.get(row.hypothesis_type, "ctr")
    direction = _DRAWDOWN_DIRECTION_BY_METRIC.get(target, "higher_is_better")
    warn_pct, hard_pct = _regression_thresholds_for(row.budget_cap_rub)

    if not row.baseline_at_promote:
        # Promoted before Task 27 shipped; nothing to compare against.
        return _RegressionCheckResult(
            hypothesis_id=row.id,
            verdict="skip_no_baseline",
            drawdown_pct=0.0,
            target_metric=target,
            warn_pct=warn_pct,
            hard_pct=hard_pct,
            current_metrics={},
        )

    current = await _fetch_current_metrics(direct, bitrix, row, now=now)

    clicks = int(current.get("clicks", 0) or 0)
    # Low-signal window (<50 clicks including 0 — empty fetch, API timeout)
    # → skip. Rationale: budget_guard already freezes the spend; we don't
    # double-alert, and regression math on 0 clicks is garbage.
    if clicks < _MIN_CLICKS_FOR_SIGNAL:
        return _RegressionCheckResult(
            hypothesis_id=row.id,
            verdict="skip_low_signal",
            drawdown_pct=0.0,
            target_metric=target,
            warn_pct=warn_pct,
            hard_pct=hard_pct,
            current_metrics=current,
        )

    try:
        baseline_value = float(row.baseline_at_promote.get(target, 0) or 0)
    except (TypeError, ValueError):
        baseline_value = 0.0
    try:
        current_value = float(current.get(target, 0) or 0)
    except (TypeError, ValueError):
        current_value = 0.0

    dd_pct = _compute_drawdown_pct(
        baseline_value=baseline_value,
        current_value=current_value,
        direction=direction,
    )

    if dd_pct >= hard_pct:
        verdict: Literal["ok", "skip_no_baseline", "skip_low_signal", "warn", "hard"] = "hard"
    elif dd_pct >= warn_pct:
        verdict = "warn"
    else:
        verdict = "ok"

    return _RegressionCheckResult(
        hypothesis_id=row.id,
        verdict=verdict,
        drawdown_pct=dd_pct,
        target_metric=target,
        warn_pct=warn_pct,
        hard_pct=hard_pct,
        current_metrics=current,
    )


async def _execute_rollback(
    pool: AsyncConnectionPool,
    row: _ConfirmedRow,
    result: _RegressionCheckResult,
    *,
    direct: _DirectLike,
    reflection_store: _ReflectionStoreLike | None,
    telegram: _TelegramLike | None,
) -> dict[str, Any]:
    """Hard verdict → rollback + reflection + critical alert.

    Reflection + alert are **always** attempted even if rollback raises
    (the owner needs to know). Returns a dict describing which steps
    succeeded so ``run`` can build its report accurately.
    """
    rollback_ok = False
    rollback_err: str | None = None
    reflection_ok = False
    alert_ok = False

    try:
        await impact_tracker.rollback(pool, row.id, direct=direct)
        rollback_ok = True
    except Exception as exc:  # noqa: BLE001 — re-shot via Telegram
        rollback_err = f"{type(exc).__name__}: {exc}"
        logger.exception("regression_watch: impact_tracker.rollback failed for %s", row.id)

    if reflection_store is not None:
        try:
            await reflection_store.save(
                text=(
                    f"regression rollback {row.id}: {row.agent} "
                    f"{row.hypothesis_type} drawdown {result.drawdown_pct:.1f}%"
                ),
                metadata={
                    "hypothesis_id": row.id,
                    "outcome": "regressed",
                    "agent": row.agent,
                    "budget_cap_rub": row.budget_cap_rub,
                    "hypothesis_type": row.hypothesis_type,
                    "drawdown_pct": result.drawdown_pct,
                    "target_metric": result.target_metric,
                    "baseline": row.baseline_at_promote,
                    "current": result.current_metrics,
                    "trigger": "regression_watch",
                    "rollback_status": "ok" if rollback_ok else "failed",
                },
            )
            reflection_ok = True
        except Exception:
            logger.exception("regression_watch: reflection_store.save failed for %s", row.id)

    if telegram is not None:
        try:
            if rollback_ok:
                await telegram.send_message(
                    text=_format_critical(row, result.drawdown_pct, result.hard_pct),
                    priority="critical",
                )
            else:
                await telegram.send_message(
                    text=_format_rollback_failed(row, result.drawdown_pct, rollback_err or ""),
                    priority="critical",
                )
            alert_ok = True
        except Exception:
            logger.exception(
                "regression_watch: telegram.send_message(critical) failed for %s",
                row.id,
            )

    return {
        "hypothesis_id": row.id,
        "rollback_ok": rollback_ok,
        "rollback_error": rollback_err,
        "reflection_ok": reflection_ok,
        "alert_ok": alert_ok,
    }


async def _safe_get_trust_level(pool: AsyncConnectionPool) -> TrustLevel:
    try:
        return await get_trust_level(pool)
    except Exception:
        logger.warning(
            "regression_watch: trust_level lookup failed, defaulting shadow",
            exc_info=True,
        )
        return TrustLevel.SHADOW


# --------------------------------------------------------------- run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: _DirectLike | None = None,
    bitrix: _BitrixLike | None = None,
    telegram: _TelegramLike | None = None,
    reflection_store: _ReflectionStoreLike | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Entry point. JOB_REGISTRY-compatible.

    Extra kwargs default to ``None`` so the minimal ``(pool, dry_run)``
    dispatch in :mod:`agent_runtime.jobs.__init__` does not break. The
    real HTTP handler injects clients from ``app.state``.

    ``now`` is injected by tests for determinism; prod leaves it ``None``
    and uses :func:`datetime.now`.
    """
    now = now or datetime.now()
    now_iso = now.isoformat()
    trust = await _safe_get_trust_level(pool)

    if direct is None:
        logger.warning("regression_watch: direct client missing — degraded no-op")
        return {
            "status": "ok",
            "action": "degraded_noop",
            "trust_level": trust.value,
            "checked": 0,
            "warnings": [],
            "rollbacks": [],
            "skipped": [],
            "dry_run": dry_run,
            "checked_at": now_iso,
        }

    rows = await _select_confirmed_in_window(pool)
    logger.info(
        "regression_watch: confirmed-in-window=%d trust=%s dry_run=%s",
        len(rows),
        trust.value,
        dry_run,
    )

    warnings_emitted: list[str] = []
    rollbacks_done: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    mutate_allowed = (
        not dry_run and trust != TrustLevel.SHADOW and trust != TrustLevel.FORBIDDEN_LOCK
    )

    for row in rows:
        try:
            result = await _check_one(row, direct=direct, bitrix=bitrix, now=now)
        except Exception:
            logger.exception("regression_watch: _check_one crashed for %s", row.id)
            skipped.append({"hypothesis_id": row.id, "reason": "check_error"})
            continue

        logger.info(
            "regression_check id=%s agent=%s type=%s target=%s dd=%.1f%% verdict=%s",
            row.id,
            row.agent,
            row.hypothesis_type,
            result.target_metric,
            result.drawdown_pct,
            result.verdict,
        )

        if result.verdict == "skip_no_baseline":
            skipped.append({"hypothesis_id": row.id, "reason": "no_baseline"})
            continue
        if result.verdict == "skip_low_signal":
            skipped.append({"hypothesis_id": row.id, "reason": "low_signal"})
            continue
        if result.verdict == "ok":
            continue

        if result.verdict == "warn":
            # Warnings only; in shadow we still surface them (no DB mutation).
            if not dry_run and telegram is not None:
                try:
                    await telegram.send_message(
                        text=_format_warning(row, result.drawdown_pct, result.warn_pct),
                        priority="warning",
                    )
                except Exception:
                    logger.exception("regression_watch: warn alert failed for %s", row.id)
            warnings_emitted.append(row.id)
            continue

        # verdict == "hard"
        if not mutate_allowed:
            # shadow / dry_run: downgrade to NOTIFY — still alert loudly but
            # do not touch Direct / DB.
            if not dry_run and telegram is not None:
                try:
                    await telegram.send_message(
                        text=_format_critical(row, result.drawdown_pct, result.hard_pct),
                        priority="critical",
                    )
                except Exception:
                    logger.exception("regression_watch: shadow hard-notify failed for %s", row.id)
            warnings_emitted.append(row.id)
            continue

        try:
            rollback_report = await _execute_rollback(
                pool,
                row,
                result,
                direct=direct,
                reflection_store=reflection_store,
                telegram=telegram,
            )
        except Exception:
            logger.exception("regression_watch: _execute_rollback outer crash for %s", row.id)
            skipped.append({"hypothesis_id": row.id, "reason": "rollback_outer_error"})
            continue
        rollbacks_done.append(rollback_report)

    logger.info(
        "regression_watch done: checked=%d warnings=%d rollbacks=%d skipped=%d",
        len(rows),
        len(warnings_emitted),
        len(rollbacks_done),
        len(skipped),
    )
    return {
        "status": "ok",
        "trust_level": trust.value,
        "checked": len(rows),
        "warnings": warnings_emitted,
        "rollbacks": rollbacks_done,
        "skipped": skipped,
        "dry_run": dry_run,
        "checked_at": now_iso,
    }


__all__ = [
    "run",
]
