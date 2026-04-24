"""Hypothesis lifecycle: measure → promote/rollback → release waiting.

Task 12b module. Four lifecycle operations:

* :func:`measure_outcome` — compare current metrics against the recorded
  ``metrics_before``; classify ``positive`` / ``negative`` / ``neutral``
  with a 15% threshold and a 50-click confidence floor. Writes the
  outcome to ``hypotheses`` via :mod:`agent_runtime.decision_journal`,
  persists a lesson into ``PGReflectionStore``, and — for confirmed /
  rejected results — follows up with :func:`promote_to_prod` /
  :func:`rollback`.
* :func:`promote_to_prod` — freezes ``metrics_after`` as
  ``baseline_at_promote`` so the Wave 2 ``regression_watch`` job can spot
  post-promotion regressions.
* :func:`rollback` — composes the reverse action sequence via
  ``decision_journal.REVERSE_ACTION_MAP`` and issues it through the
  provided DirectAPI instance. Each reverse mutation is audited through
  the sanctioned :func:`agent_runtime.db.insert_audit_log`.
* :func:`release_bucket_and_start_waiting` — FIFO bucket manager. When a
  running hypothesis concludes, the free budget is reclaimed and the
  oldest ``state='waiting_budget'`` rows that fit are promoted to
  ``running``. ``SELECT FOR UPDATE`` on ``sda_state.mutations_this_week``
  prevents two cron ticks from overbooking the cap.
* :func:`mark_expired` — hard stop on runaway hypotheses: anything
  ``state='running'`` that has outlived its ``maximum_running_days`` is
  flipped to ``inconclusive`` with an "expired" lesson.

The module delegates all ``hypotheses`` SQL mutations back to
``decision_journal`` (``update_outcome``) so there is still one writer
per Decision 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.decision_journal import REVERSE_ACTION_MAP, update_outcome
from agent_runtime.models import HypothesisState

logger = logging.getLogger(__name__)


OUTCOME_DELTA_THRESHOLD = 0.15
MIN_CLICKS_FOR_CONFIDENCE = 50
_WEEKLY_BUDGET_CAP_RUB = 12_000

# Target metric per hypothesis type — the field we diff on in metrics_before
# vs current.
_TARGET_METRIC: dict[str, str] = {
    "ad": "ctr",
    "neg_kw": "cpa",
    "landing": "cr",
    "new_camp": "leads",
    "image": "ctr",
    "format_change": "ctr",
    "strategy_switch": "cpa",
    "account_level": "leads",
}


class Outcome(BaseModel):
    """Structured classification result handed back to callers."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    classification: Literal["positive", "negative", "neutral"]
    confidence: Literal["high", "low"]
    delta_pct: float
    metrics_after: dict[str, Any] = Field(default_factory=dict)
    lesson: str


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


class _ReflectionStoreLike(Protocol):
    async def save(self, text: str, metadata: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class _HypothesisSnapshot:
    id: str
    hypothesis_type: str
    campaign_id: int | None
    ad_group_id: int | None
    state: str
    actions: list[dict[str, Any]]
    metrics_before: dict[str, Any]
    created_at: datetime
    budget_cap_rub: int


# ---------------------------------------------------------- measure_outcome


async def measure_outcome(
    pool: AsyncConnectionPool,
    hypothesis_id: str,
    *,
    direct: _DirectLike,
    reflection_store: _ReflectionStoreLike | None = None,
    current_metrics: dict[str, Any] | None = None,
) -> Outcome:
    """Classify a running hypothesis and drive it to its terminal state.

    ``current_metrics`` — optional explicit dict; if omitted we fetch
    campaign stats from ``direct`` using the row's ``campaign_id`` plus a
    ``created_at → now()`` window. Tests inject the metrics directly for
    determinism.
    """
    snapshot = await _load_snapshot(pool, hypothesis_id)
    if snapshot.state != HypothesisState.RUNNING.value:
        logger.info(
            "measure_outcome: hypothesis %s is in state '%s', skipping",
            hypothesis_id,
            snapshot.state,
        )
        return Outcome(
            hypothesis_id=hypothesis_id,
            classification="neutral",
            confidence="low",
            delta_pct=0.0,
            metrics_after={},
            lesson=f"already in terminal state '{snapshot.state}'",
        )

    if current_metrics is None:
        current_metrics = await _fetch_current_metrics(direct, snapshot)

    target = _TARGET_METRIC.get(snapshot.hypothesis_type, "ctr")
    before_value = float(snapshot.metrics_before.get(target, 0) or 0)
    after_value = float(current_metrics.get(target, 0) or 0)

    if before_value == 0:
        delta_pct = float("inf") if after_value > 0 else 0.0
    else:
        delta_pct = (after_value - before_value) / before_value

    clicks = int(current_metrics.get("clicks", 0) or 0)
    if clicks < MIN_CLICKS_FOR_CONFIDENCE:
        classification: Literal["positive", "negative", "neutral"] = "neutral"
        confidence: Literal["high", "low"] = "low"
        lesson = f"insufficient signal: clicks={clicks} < {MIN_CLICKS_FOR_CONFIDENCE}"
    else:
        confidence = "high"
        if delta_pct >= OUTCOME_DELTA_THRESHOLD:
            classification = "positive"
            lesson = f"{snapshot.hypothesis_type} → positive: {target} +{delta_pct * 100:.1f}%"
        elif delta_pct <= -OUTCOME_DELTA_THRESHOLD:
            classification = "negative"
            lesson = f"{snapshot.hypothesis_type} → negative: {target} {delta_pct * 100:.1f}%"
        else:
            classification = "neutral"
            lesson = f"{snapshot.hypothesis_type} → neutral: {target} {delta_pct * 100:+.1f}%"

    await update_outcome(
        pool,
        hypothesis_id,
        classification,
        metrics_after=current_metrics,
        lesson=lesson,
    )

    if classification == "positive":
        await promote_to_prod(pool, hypothesis_id, metrics_after=current_metrics)
    elif classification == "negative":
        await rollback(pool, hypothesis_id, direct=direct, snapshot=snapshot)

    if reflection_store is not None:
        try:
            await reflection_store.save(
                text=lesson,
                metadata={
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_type": snapshot.hypothesis_type,
                    "outcome": classification,
                    "budget_cap_rub": snapshot.budget_cap_rub,
                    "delta_pct": delta_pct,
                },
            )
        except Exception:
            logger.warning("reflection_store.save failed for %s", hypothesis_id, exc_info=True)

    return Outcome(
        hypothesis_id=hypothesis_id,
        classification=classification,
        confidence=confidence,
        delta_pct=delta_pct,
        metrics_after=current_metrics,
        lesson=lesson,
    )


async def _fetch_current_metrics(
    direct: _DirectLike, snapshot: _HypothesisSnapshot
) -> dict[str, Any]:
    """Fetch a minimal metrics dict from Direct stats.

    The brain wrapper (Task 12) populates ``metrics_before`` with fields
    like ``ctr``, ``cpa``, ``leads``, ``clicks`` — we ask for the same.
    The current ``DirectAPI.get_campaign_stats`` returns a TSV blob; until
    Task 24 extends it with a parsed dict helper, we return an empty
    payload on fetch failure so the outcome lands as neutral/low rather
    than crashing the cron.
    """
    if snapshot.campaign_id is None:
        return {}
    today = datetime.now(UTC).date().isoformat()
    window_start = snapshot.created_at.date().isoformat()
    try:
        stats = await direct.get_campaign_stats(
            snapshot.campaign_id, date_from=window_start, date_to=today
        )
    except Exception:
        logger.warning("direct.get_campaign_stats failed for %s", snapshot.id, exc_info=True)
        return {}
    # Best-effort — brain / signal_detector both normalise this into numbers
    # elsewhere. Here we trust upstream to supply pre-parsed metrics.
    return dict(stats)


async def _load_snapshot(pool: AsyncConnectionPool, hypothesis_id: str) -> _HypothesisSnapshot:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, hypothesis_type, campaign_id, ad_group_id, state,
                       actions, metrics_before, created_at, budget_cap_rub
                FROM hypotheses
                WHERE id = %s
                """,
                (hypothesis_id,),
            )
            row = await cur.fetchone()
    if row is None:
        raise LookupError(f"hypothesis {hypothesis_id} not found")
    return _HypothesisSnapshot(
        id=row[0],
        hypothesis_type=row[1],
        campaign_id=row[2],
        ad_group_id=row[3],
        state=row[4],
        actions=list(row[5] or []),
        metrics_before=dict(row[6] or {}),
        created_at=row[7],
        budget_cap_rub=int(row[8]),
    )


# ---------------------------------------------------------- promote_to_prod


async def promote_to_prod(
    pool: AsyncConnectionPool,
    hypothesis_id: str,
    *,
    metrics_after: dict[str, Any],
) -> None:
    """Freeze baseline for regression_watch (Wave 2 Task 24)."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE hypotheses
                SET state = 'confirmed',
                    promoted_at = NOW(),
                    baseline_at_promote = %s
                WHERE id = %s
                """,
                (Jsonb(metrics_after), hypothesis_id),
            )
    logger.info("hypothesis %s promoted; baseline_at_promote set", hypothesis_id)


# ---------------------------------------------------------------- rollback


async def rollback(
    pool: AsyncConnectionPool,
    hypothesis_id: str,
    *,
    direct: _DirectLike,
    snapshot: _HypothesisSnapshot | None = None,
) -> None:
    """Reverse every action in the hypothesis, then flag the row rolled_back.

    Any action that has no entry in ``REVERSE_ACTION_MAP`` or whose reverse
    helper raises (e.g. ``set_bid`` without an original bid recorded) is
    logged and skipped — the hypothesis still moves to ``rolled_back`` but
    a CRITICAL entry goes to audit_log (via the ``insert_audit_log``
    wrapper so sanitisation applies).
    """
    if snapshot is None:
        snapshot = await _load_snapshot(pool, hypothesis_id)

    from agent_runtime.db import insert_audit_log  # avoid cycle at import

    for action in snapshot.actions:
        action_type = str(action.get("type", ""))
        reverser = REVERSE_ACTION_MAP.get(action_type)
        if reverser is None:
            logger.warning("no reverse for action type %r; skipping", action_type)
            await insert_audit_log(
                pool,
                hypothesis_id=hypothesis_id,
                trust_level="",
                tool_name="impact_tracker.rollback_skip",
                tool_input=action,
                tool_output=None,
                is_mutation=False,
                is_error=True,
                error_detail=f"no reverse mapped for action type {action_type!r}",
            )
            continue
        try:
            reverse_action = reverser(action, snapshot.metrics_before)
        except Exception as exc:  # noqa: BLE001 — audited
            logger.exception("reverse builder for %s failed", action_type)
            await insert_audit_log(
                pool,
                hypothesis_id=hypothesis_id,
                trust_level="",
                tool_name="impact_tracker.rollback_skip",
                tool_input=action,
                tool_output=None,
                is_mutation=False,
                is_error=True,
                error_detail=f"reverse_builder_error: {exc!s}",
            )
            continue

        await _execute_reverse(direct, reverse_action)
        await insert_audit_log(
            pool,
            hypothesis_id=hypothesis_id,
            trust_level="",
            tool_name="impact_tracker.rollback",
            tool_input=reverse_action,
            tool_output=None,
            is_mutation=True,
        )

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE hypotheses SET state = 'rolled_back' WHERE id = %s",
                (hypothesis_id,),
            )
    logger.info("hypothesis %s rolled_back (%d actions)", hypothesis_id, len(snapshot.actions))


async def _execute_reverse(direct: _DirectLike, action: dict[str, Any]) -> None:
    action_type = action["type"]
    params = action.get("params") or {}
    if action_type == "pause_group":
        await direct.pause_group(params["ad_group_id"])
    elif action_type == "resume_group":
        await direct.resume_group(params["ad_group_id"])
    elif action_type == "pause_campaign":
        await direct.pause_campaign(params["campaign_id"])
    elif action_type == "resume_campaign":
        await direct.resume_campaign(params["campaign_id"])
    elif action_type == "set_bid":
        await direct.set_bid(
            params["keyword_id"],
            bid_rub=params["bid_rub"],
            context_bid_rub=params.get("context_bid_rub"),
        )
    elif action_type == "remove_negatives":
        # Direct API has no explicit remove; Wave 2 will expose one. For
        # now we record intent in audit_log and let the owner handle the
        # residual negatives manually. Skip here to preserve idempotency.
        logger.warning("remove_negatives is not yet wired to DirectAPI; audit only")
    else:
        logger.warning("unknown reverse action %r; skipped", action_type)


# --------------------------------------------- release_bucket_and_start_waiting


async def release_bucket_and_start_waiting(pool: AsyncConnectionPool) -> list[str]:
    """Reclaim free budget and promote oldest waiting hypotheses to running.

    Uses ``SELECT FOR UPDATE`` on ``sda_state.mutations_this_week`` so two
    cron ticks cannot race and overbook the weekly cap.
    """
    released: list[str] = []
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT value FROM sda_state WHERE key='mutations_this_week' FOR UPDATE"
            )
            row = await cur.fetchone()
            current_sum = int(row[0]) if row and row[0] is not None else 0

            await cur.execute(
                """
                SELECT COALESCE(SUM(budget_cap_rub), 0)
                FROM hypotheses
                WHERE state = 'running'
                """
            )
            running_row = await cur.fetchone()
            running_sum = int(running_row[0]) if running_row else 0
            effective_sum = max(current_sum, running_sum)  # trust the stricter bound

            await cur.execute(
                """
                SELECT id, budget_cap_rub
                FROM hypotheses
                WHERE state = 'waiting_budget'
                ORDER BY created_at ASC
                """
            )
            waiting = await cur.fetchall()

            for waiting_id, cap in waiting:
                cap = int(cap)
                if effective_sum + cap > _WEEKLY_BUDGET_CAP_RUB:
                    continue
                await cur.execute(
                    """
                    UPDATE hypotheses
                    SET state = 'running',
                        metrics_before_captured_at = NOW()
                    WHERE id = %s
                    """,
                    (waiting_id,),
                )
                effective_sum += cap
                released.append(str(waiting_id))

            await cur.execute(
                """
                INSERT INTO sda_state (key, value, updated_at)
                VALUES ('mutations_this_week', %s, NOW())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (Jsonb(effective_sum),),
            )

    if released:
        logger.info("release_bucket: promoted %d waiting hypotheses: %s", len(released), released)
    return released


# -------------------------------------------------------------- mark_expired


async def mark_expired(pool: AsyncConnectionPool) -> list[str]:
    """Flip running hypotheses that outlived ``maximum_running_days``."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE hypotheses
                SET state = 'inconclusive',
                    lesson = COALESCE(lesson, '') ||
                        CASE WHEN lesson IS NULL OR lesson = ''
                             THEN 'expired: running > maximum_running_days'
                             ELSE ' | expired: running > maximum_running_days'
                        END
                WHERE state = 'running'
                  AND NOW() - created_at
                      > make_interval(days => COALESCE(maximum_running_days, 14))
                RETURNING id
                """
            )
            rows = await cur.fetchall()
    ids = [str(r[0]) for r in rows]
    if ids:
        logger.info("mark_expired: %d hypotheses flipped to inconclusive: %s", len(ids), ids)
    return ids


__all__ = [
    "MIN_CLICKS_FOR_CONFIDENCE",
    "OUTCOME_DELTA_THRESHOLD",
    "Outcome",
    "mark_expired",
    "measure_outcome",
    "promote_to_prod",
    "release_bucket_and_start_waiting",
    "rollback",
]
