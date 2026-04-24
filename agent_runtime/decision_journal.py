"""CRUD layer over the ``hypotheses`` table (Task 12b).

Single writer: brain.reason() calls :func:`record_hypothesis` to INSERT;
impact_tracker calls :func:`update_outcome` on the same row through its
lifecycle. Nothing else should INSERT into the table directly — keeps the
Pydantic ``HypothesisDraft`` schema and the SQL CHECK constraints in lock-
step (a new ``hypothesis_type`` value needs both layers to agree).

``detect_flip_flop`` caps the candidate row pool at 1000 so the naive
O(n²) reverse-pair scan never explodes — the agent concludes at most a
few hypotheses per campaign per day, so 1000 rows is multiple quarters of
activity on a single target and well beyond any realistic window.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime.models import (
    HYPOTHESIS_BUDGET_CAP,
    HypothesisDraft,
    HypothesisState,
    Signal,
)

logger = logging.getLogger(__name__)


_MAX_FLIPFLOP_ROWS = 1000
_WEEKLY_BUDGET_CAP_RUB = 12_000  # Decision 4: total weekly hypothesis budget


# --- reverse-action map (shared with impact_tracker via import) --------------


def _reverse_pause_group(action: dict[str, Any], _before: dict[str, Any]) -> dict[str, Any]:
    return {"type": "resume_group", "params": {"ad_group_id": action["params"]["ad_group_id"]}}


def _reverse_resume_group(action: dict[str, Any], _before: dict[str, Any]) -> dict[str, Any]:
    return {"type": "pause_group", "params": {"ad_group_id": action["params"]["ad_group_id"]}}


def _reverse_pause_campaign(action: dict[str, Any], _before: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "resume_campaign",
        "params": {"campaign_id": action["params"]["campaign_id"]},
    }


def _reverse_resume_campaign(action: dict[str, Any], _before: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "pause_campaign",
        "params": {"campaign_id": action["params"]["campaign_id"]},
    }


def _reverse_set_bid(action: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    # metrics_before MUST carry the pre-mutation bid under ``original_bid``.
    # See impact_tracker.rollback for the fallback when it is missing.
    original = before.get("original_bid")
    if original is None:
        raise ValueError("metrics_before.original_bid missing — cannot reverse set_bid")
    return {
        "type": "set_bid",
        "params": {
            "keyword_id": action["params"].get("keyword_id"),
            "ad_group_id": action["params"].get("ad_group_id"),
            "bid_rub": int(original),
        },
    }


def _reverse_add_negatives(action: dict[str, Any], _before: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "remove_negatives",
        "params": {
            "campaign_id": action["params"]["campaign_id"],
            "phrases": action["params"]["phrases"],
        },
    }


REVERSE_ACTION_MAP: dict[str, Any] = {
    "pause_group": _reverse_pause_group,
    "resume_group": _reverse_resume_group,
    "pause_campaign": _reverse_pause_campaign,
    "resume_campaign": _reverse_resume_campaign,
    "set_bid": _reverse_set_bid,
    "add_negatives": _reverse_add_negatives,
}


# --- public result types -----------------------------------------------------


@dataclass(frozen=True)
class HypothesisRow:
    """Minimal projection of ``hypotheses`` we return to callers."""

    id: str
    hypothesis_type: str
    agent: str
    campaign_id: int | None
    ad_group_id: int | None
    state: str
    actions: list[dict[str, Any]]
    metrics_before: dict[str, Any]
    budget_cap_rub: int
    created_at: datetime
    maximum_running_days: int


@dataclass(frozen=True)
class FlipFlopEvent:
    pair_ids: tuple[str, str]
    action_pair: tuple[str, str]
    interval_days: float
    severity: Literal["warn", "critical"]


# --- record_hypothesis ------------------------------------------------------


async def record_hypothesis(
    pool: AsyncConnectionPool,
    draft: HypothesisDraft,
    signals: list[Signal],
    metrics_before: dict[str, Any],
    *,
    agent: str = "brain",
    autonomy_level: str = "ASK",
    risk_score: float = 50.0,
) -> str:
    """INSERT a new hypothesis. State depends on available weekly budget.

    Uses ``SELECT FOR UPDATE`` on ``sda_state.mutations_this_week`` so two
    concurrent callers cannot overbook the 12 000₽ weekly cap (Decision 4).
    Returns the generated hypothesis id (uuid4 hex prefix, 8 chars).
    """
    hypothesis_id = uuid.uuid4().hex[:8]
    budget_cap = HYPOTHESIS_BUDGET_CAP[draft.hypothesis_type]
    signals_payload = [
        {"type": s.type.value, "severity": s.severity, "data": s.data, "ts": s.ts.isoformat()}
        for s in signals
    ]

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT value FROM sda_state WHERE key='mutations_this_week' FOR UPDATE"
            )
            row = await cur.fetchone()
            current_sum = int(row[0]) if row and row[0] is not None else 0
            free = _WEEKLY_BUDGET_CAP_RUB - current_sum
            new_state = (
                HypothesisState.RUNNING.value
                if budget_cap <= free
                else HypothesisState.WAITING_BUDGET.value
            )
            updated_sum = current_sum + (
                budget_cap if new_state == HypothesisState.RUNNING.value else 0
            )

            await cur.execute(
                """
                INSERT INTO hypotheses (
                    id, agent, hypothesis_type, signals, hypothesis, reasoning,
                    actions, expected_outcome, budget_cap_rub, ad_group_id,
                    campaign_id, autonomy_level, risk_score, state,
                    metrics_before
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s
                )
                """,
                (
                    hypothesis_id,
                    agent,
                    draft.hypothesis_type.value,
                    Jsonb(signals_payload),
                    draft.hypothesis,
                    draft.reasoning,
                    Jsonb(draft.actions),
                    draft.expected_outcome,
                    budget_cap,
                    draft.ad_group_id,
                    draft.campaign_id,
                    autonomy_level,
                    risk_score,
                    new_state,
                    Jsonb(metrics_before),
                ),
            )

            # Bump mutations_this_week only for running hypotheses — waiting
            # rows join the queue without consuming budget yet.
            if new_state == HypothesisState.RUNNING.value:
                await cur.execute(
                    """
                    INSERT INTO sda_state (key, value, updated_at)
                    VALUES ('mutations_this_week', %s, NOW())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    (Jsonb(updated_sum),),
                )

    logger.info(
        "hypothesis %s recorded (state=%s type=%s cap=%d)",
        hypothesis_id,
        new_state,
        draft.hypothesis_type.value,
        budget_cap,
    )
    return hypothesis_id


# --- update_outcome ---------------------------------------------------------


_OUTCOME_TO_STATE: dict[str, str] = {
    "positive": HypothesisState.CONFIRMED.value,
    "negative": HypothesisState.REJECTED.value,
    "neutral": HypothesisState.INCONCLUSIVE.value,
}


async def update_outcome(
    pool: AsyncConnectionPool,
    hypothesis_id: str,
    outcome: Literal["positive", "negative", "neutral"],
    metrics_after: dict[str, Any],
    lesson: str,
) -> None:
    """Record the measured outcome and flip the row to the terminal state."""
    state = _OUTCOME_TO_STATE[outcome]
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE hypotheses
                SET outcome = %s,
                    metrics_after = %s,
                    metrics_after_captured_at = NOW(),
                    lesson = %s,
                    state = %s
                WHERE id = %s
                """,
                (outcome, Jsonb(metrics_after), lesson, state, hypothesis_id),
            )
    logger.info("hypothesis %s outcome=%s state=%s", hypothesis_id, outcome, state)


# --- get_pending_checks -----------------------------------------------------


async def get_pending_checks(pool: AsyncConnectionPool) -> list[HypothesisRow]:
    """Running hypotheses older than 72h that we should measure.

    Click-based branch (≥100 new clicks since ``clicks_at_record``) is
    deferred until bitrix_feedback (Wave 2) exposes a fresh clicks count;
    this version filters by elapsed time only.
    """
    query = """
    SELECT id, hypothesis_type, agent, campaign_id, ad_group_id, state,
           actions, metrics_before, budget_cap_rub, created_at,
           COALESCE(maximum_running_days, 14)
    FROM hypotheses
    WHERE state = 'running'
      AND created_at < now() - interval '72 hours'
    ORDER BY created_at ASC
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query)
            rows = await cur.fetchall()
    return [_row_to_hypothesis(r) for r in rows]


def _row_to_hypothesis(row: tuple[Any, ...]) -> HypothesisRow:
    return HypothesisRow(
        id=row[0],
        hypothesis_type=row[1],
        agent=row[2],
        campaign_id=row[3],
        ad_group_id=row[4],
        state=row[5],
        actions=list(row[6] or []),
        metrics_before=dict(row[7] or {}),
        budget_cap_rub=int(row[8]),
        created_at=row[9],
        maximum_running_days=int(row[10]),
    )


# --- get_actions_today ------------------------------------------------------


async def get_actions_today(pool: AsyncConnectionPool) -> list[dict[str, Any]]:
    """Mutating audit_log entries for today — Telegram digest input."""
    query = """
    SELECT id, ts, hypothesis_id, trust_level, tool_name, tool_input,
           tool_output, is_mutation, is_error
    FROM audit_log
    WHERE ts::date = current_date
      AND is_mutation = true
    ORDER BY ts DESC
    LIMIT 100
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query)
            rows = await cur.fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            "hypothesis_id": r[2],
            "trust_level": r[3],
            "tool_name": r[4],
            "tool_input": r[5],
            "tool_output": r[6],
            "is_mutation": r[7],
            "is_error": r[8],
        }
        for r in rows
    ]


# --- detect_flip_flop -------------------------------------------------------


_REVERSE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("pause_group", "resume_group"),
        ("resume_group", "pause_group"),
        ("pause_campaign", "resume_campaign"),
        ("resume_campaign", "pause_campaign"),
        ("add_negatives", "remove_negatives"),
        ("remove_negatives", "add_negatives"),
    }
)


async def detect_flip_flop(
    pool: AsyncConnectionPool,
    *,
    campaign_id: int | None = None,
    ad_group_id: int | None = None,
    window_days: int = 30,
) -> list[FlipFlopEvent]:
    """Find reverse-action pairs on the same target within ``window_days``.

    O(n²) scan over up to 1000 candidate rows — fine because the agent
    produces at most a few concluded hypotheses per target per day.
    """
    if campaign_id is None and ad_group_id is None:
        return []

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, actions, created_at
                FROM hypotheses
                WHERE (campaign_id IS NOT DISTINCT FROM %s
                       OR ad_group_id IS NOT DISTINCT FROM %s)
                  AND created_at > now() - make_interval(days => %s)
                  AND state IN ('confirmed', 'rejected', 'rolled_back')
                  AND (campaign_id IS NOT NULL OR ad_group_id IS NOT NULL)
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (campaign_id, ad_group_id, window_days, _MAX_FLIPFLOP_ROWS),
            )
            rows = await cur.fetchall()

    events: list[FlipFlopEvent] = []
    for i, (id_i, actions_i, created_i) in enumerate(rows):
        for id_j, actions_j, created_j in rows[i + 1 :]:
            for a_i in actions_i or []:
                for a_j in actions_j or []:
                    pair = (str(a_i.get("type", "")), str(a_j.get("type", "")))
                    if pair in _REVERSE_PAIRS:
                        delta_days = (created_j - created_i).total_seconds() / 86400
                        severity: Literal["warn", "critical"] = (
                            "critical" if delta_days < 7 else "warn"
                        )
                        events.append(
                            FlipFlopEvent(
                                pair_ids=(str(id_i), str(id_j)),
                                action_pair=pair,
                                interval_days=round(delta_days, 2),
                                severity=severity,
                            )
                        )
    return events


__all__ = [
    "FlipFlopEvent",
    "HypothesisRow",
    "REVERSE_ACTION_MAP",
    "detect_flip_flop",
    "get_actions_today",
    "get_pending_checks",
    "record_hypothesis",
    "update_outcome",
]
