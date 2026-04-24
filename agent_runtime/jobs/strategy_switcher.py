"""Strategy Switcher — weekly ASK-producer downstream of Strategy Gate (Task 26).

Runs Mondays 11:00 МСК (``0 8 * * 1`` UTC). Reads the ``strategy_gate_state``
row from ``sda_state`` (written by :mod:`agent_runtime.jobs.strategy_gate`,
Task 17) and, **only** when the state is ``ready_to_switch``:

1. Drafts an ``hypotheses`` row of ``hypothesis_type='strategy_switch'`` that
   describes the proposed Direct auto-strategy flip
   (``WB_MAX_CLICKS → WB_MAXIMUM_CONVERSION_RATE``).
2. Inserts the paired ``ask_queue`` row with a human question so Task 23
   (Telegram inline buttons) surfaces it to the owner.

This job itself performs **no** Direct API mutations. The actual
``update_strategy`` call happens through :mod:`agent_runtime.jobs.smart_optimizer`
after the owner presses ``Approve`` in Telegram (Task 23 → ask_queue resolve
→ smart_optimizer executor). This split keeps the weekly cron dumb and
auditable — one row per gate tick, idempotent, no surprise mutations.

**Trust overlay**: the job never touches Direct, so ``shadow`` / ``assisted`` /
``autonomous`` do NOT change its behaviour. What they change is how Task 23
handles the resulting ASK (shadow: NOTIFY-only, assisted: inline buttons,
autonomous: ignored — ``ready_to_switch`` always requires owner confirmation
because strategy swap is in :data:`trust_levels.DANGER_ACTIONS`).

**Idempotency**: :func:`_has_open_ask_for_gate_tick` checks for an
unresolved ``ask_queue`` row whose ``options->>'gate_entered_at'`` matches
the current ``strategy_gate_state.entered_at``. Re-running the same week
against the same gate tick returns ``skip_duplicate`` without creating
anything. A *new* gate tick (entered_at changes after a
``learning → ready_to_switch`` bounce) passes the guard.

**dry_run=True** computes the same payload but explicitly rolls back
the transaction — caller sees the draft ``hypothesis_id`` + ``question``
without a PG row. Telegram is never touched in dry_run either; notification
in prod is Task 23's responsibility once the row is live.

TODO(integration):
  1. Register ``"strategy_switcher": strategy_switcher.run`` in
     ``agent_runtime/jobs/__init__.py::JOB_REGISTRY`` (Task 26 forbidden
     here — see the round-4 scope).
  2. Add Railway Cron entry ``0 8 * * 1`` for
     ``/run/strategy_switcher`` with bearer auth (``railway.toml``,
     also out-of-scope for this file).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log

logger = logging.getLogger(__name__)


# Target strategy — module-level for testability.
PROPOSED_STRATEGY: str = "WB_MAXIMUM_CONVERSION_RATE"
CURRENT_STRATEGY: str = "WB_MAX_CLICKS"
_STATE_KEY: str = "strategy_gate_state"
_HYPOTHESIS_TYPE: str = "strategy_switch"
_TOOL_NAME: str = "strategy_switcher"
_ASK_OPTIONS: tuple[str, ...] = ("approve", "reject", "defer_24h")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_question(
    *,
    campaign_ids: list[int],
    current_cpa_rub: float | None,
    gate_entered_at: str,
) -> str:
    """Plain-text owner question, no HTML — Telegram layer adds formatting.

    The gate_entered_at suffix embeds which ready_to_switch tick produced
    this ask, so the answerer has traceable context and so idempotency
    can match it (see :func:`_has_open_ask_for_gate_tick`).
    """
    camp_list = ", ".join(str(c) for c in campaign_ids) if campaign_ids else "(none)"
    cpa_txt = f"{int(current_cpa_rub)}₽" if current_cpa_rub else "n/a"
    return (
        f"Перевести {len(campaign_ids)} кампаний ({camp_list}) "
        f"с ручных ставок ({CURRENT_STRATEGY}) на автостратегию "
        f"{PROPOSED_STRATEGY}? "
        f"Текущий 7d CPA: {cpa_txt}. "
        f"Strategy gate: 4/4 signals зелёные. "
        f"Gate tick: {gate_entered_at}."
    )


async def _load_gate_state(pool: AsyncConnectionPool) -> dict[str, Any] | None:
    """Read-only lookup of strategy_gate_state. None if missing/empty."""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM sda_state WHERE key = %s",
                    (_STATE_KEY,),
                )
                row = await cur.fetchone()
    except Exception:
        logger.exception("strategy_switcher: sda_state read failed")
        return None
    if row is None or row[0] is None:
        return None
    raw = row[0]
    if isinstance(raw, dict):
        return raw
    # psycopg JSONB decode normally returns a dict; string fallback just in case.
    try:
        import json

        decoded = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


async def _has_open_ask_for_gate_tick(
    pool: AsyncConnectionPool,
    gate_entered_at: str,
) -> bool:
    """Return True iff an unresolved ask_queue row for this gate tick exists.

    Idempotency guarantee: once a strategy_switcher has fired for a given
    ``strategy_gate_state.entered_at`` the function returns True until the
    owner resolves the ask in Telegram or the gate bounces back to
    ``learning`` and later re-enters ``ready_to_switch`` (which updates
    ``entered_at``).
    """
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT 1
                    FROM ask_queue
                    WHERE resolved_at IS NULL
                      AND options->>'gate_entered_at' = %s
                      AND options->>'kind' = %s
                    LIMIT 1
                    """,
                    (gate_entered_at, _TOOL_NAME),
                )
                row = await cur.fetchone()
        return row is not None
    except Exception:
        logger.exception("strategy_switcher: dedupe lookup failed")
        # Fail-closed: if we can't verify idempotency, treat as already-asked
        # so we never double-post.
        return True


def _active_campaigns_from_state(state: dict[str, Any]) -> list[int]:
    """Best-effort read of the cached active campaign set from gate state.

    ``strategy_gate`` does not currently populate ``active_campaigns`` (it
    reasons off Settings.PROTECTED_CAMPAIGN_IDS), but the field is part of
    the spec's future schema — accept either shape so the switcher stays
    forward-compatible.
    """
    raw = state.get("active_campaigns")
    if isinstance(raw, list):
        out: list[int] = []
        for item in raw:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _current_cpa_from_state(state: dict[str, Any]) -> float | None:
    """Mean CPA from the cpa_stability_7d signal, if strategy_gate persisted it."""
    signals = state.get("signals") or {}
    cpa = signals.get("cpa_stability_7d") or {}
    mean = cpa.get("mean")
    if mean is None:
        return None
    try:
        value = float(mean)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


# --- core ---------------------------------------------------------------------


async def _draft_payload(
    pool: AsyncConnectionPool,
    settings: Settings,
) -> dict[str, Any]:
    """Assemble the hypothesis + ask_queue draft without touching PG writes."""
    state = await _load_gate_state(pool)
    if state is None:
        return {
            "action": "skip",
            "state": "missing",
            "reason": "sda_state[strategy_gate_state] is empty",
        }
    status = state.get("status") or "learning"
    if status != "ready_to_switch":
        return {"action": "skip", "state": status}

    gate_entered_at = str(state.get("entered_at") or _now_iso())

    if await _has_open_ask_for_gate_tick(pool, gate_entered_at):
        return {
            "action": "skip_duplicate",
            "state": status,
            "gate_entered_at": gate_entered_at,
        }

    campaigns_from_state = _active_campaigns_from_state(state)
    campaigns = campaigns_from_state or list(settings.PROTECTED_CAMPAIGN_IDS)
    current_cpa = _current_cpa_from_state(state)
    hypothesis_id = f"strategy-switch-{uuid.uuid4()}"
    question = _build_question(
        campaign_ids=campaigns,
        current_cpa_rub=current_cpa,
        gate_entered_at=gate_entered_at,
    )
    ask_options_payload: dict[str, Any] = {
        "kind": _TOOL_NAME,
        "gate_entered_at": gate_entered_at,
        "campaigns": campaigns,
        "from_strategy": CURRENT_STRATEGY,
        "to_strategy": PROPOSED_STRATEGY,
        "current_cpa_rub": current_cpa,
        "options": list(_ASK_OPTIONS),
    }
    return {
        "action": "ask_drafted",
        "state": status,
        "gate_entered_at": gate_entered_at,
        "campaigns": campaigns,
        "hypothesis_draft_id": hypothesis_id,
        "question": question,
        "ask_queue_row_draft": ask_options_payload,
    }


async def _persist_ask(
    pool: AsyncConnectionPool,
    settings: Settings,
    *,
    draft: dict[str, Any],
) -> dict[str, Any]:
    """INSERT hypotheses + ask_queue rows atomically and return enriched result."""
    campaigns: list[int] = list(draft["campaigns"])
    hypothesis_id: str = draft["hypothesis_draft_id"]
    question: str = draft["question"]
    gate_entered_at: str = draft["gate_entered_at"]
    ask_options_payload: dict[str, Any] = draft["ask_queue_row_draft"]
    current_cpa = ask_options_payload.get("current_cpa_rub")

    signals_summary: list[dict[str, Any]] = [
        {"gate_entered_at": gate_entered_at, "status": draft["state"]}
    ]
    actions_payload = [
        {
            "type": "update_strategy",
            "campaign_ids": campaigns,
            "from": CURRENT_STRATEGY,
            "to": PROPOSED_STRATEGY,
        }
    ]

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO hypotheses (
                    id, agent, hypothesis_type, signals, hypothesis, reasoning,
                    actions, expected_outcome, budget_cap_rub, autonomy_level,
                    risk_score, state, metrics_before
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    hypothesis_id,
                    _TOOL_NAME,
                    _HYPOTHESIS_TYPE,
                    Jsonb(signals_summary),
                    (
                        f"switch {len(campaigns)} campaigns {CURRENT_STRATEGY}"
                        f" -> {PROPOSED_STRATEGY}"
                    ),
                    "strategy_gate reported 4/4 signals green; owner confirmation required",
                    Jsonb(actions_payload),
                    "lower CPA under auto-bidding; rollback path = manual_switch back to learning",
                    settings.DAILY_BUDGET_LIMIT,
                    "ASK",
                    0.7,
                    "running",
                    Jsonb({"cpa_mean_rub": current_cpa}),
                ),
            )
            await cur.execute(
                """
                INSERT INTO ask_queue (hypothesis_id, question, options)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (hypothesis_id, question, Jsonb(ask_options_payload)),
            )
            row = await cur.fetchone()

    ask_id = int(row[0]) if row else None
    return {
        **draft,
        "action": "ask_created",
        "hypothesis_id": hypothesis_id,
        "ask_queue_id": ask_id,
    }


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: Any | None = None,  # kept in signature for JOB_REGISTRY symmetry
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Cron entry. Read gate state → maybe INSERT hypothesis+ask row.

    Mutations:
      * ``dry_run=False`` → ``INSERT INTO hypotheses`` + ``INSERT INTO ask_queue``.
        No Direct API calls, no Telegram sends (Task 23 handler covers that).
      * ``dry_run=True`` → only computes the draft and returns it; no PG writes.

    Degraded paths:
      * ``settings`` is None → return ``degraded_noop`` (keep the cron alive
        so health checks stay green; the dispatcher is expected to inject).
      * Gate state missing → ``{"action": "skip", "state": "missing"}``.
      * State != ready_to_switch → ``{"action": "skip", "state": <current>}``.
      * Same gate_tick already asked → ``{"action": "skip_duplicate"}``.
    """
    _ = direct  # unused — kept for registry-compatible signature
    _ = http_client

    if settings is None:
        logger.warning("strategy_switcher: settings missing — degraded no-op")
        return {
            "action": "degraded_noop",
            "dry_run": dry_run,
            "reason": "settings_missing",
        }

    logger.info("strategy_switcher start (dry_run=%s)", dry_run)

    try:
        draft = await _draft_payload(pool, settings)
    except Exception as exc:
        logger.exception("strategy_switcher: draft failed")
        return {
            "action": "error",
            "dry_run": dry_run,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if draft["action"] != "ask_drafted":
        logger.info(
            "strategy_switcher end (skip): action=%s state=%s",
            draft["action"],
            draft.get("state"),
        )
        return {**draft, "dry_run": dry_run}

    if dry_run:
        logger.info(
            "strategy_switcher dry_run: would create hypothesis=%s",
            draft["hypothesis_draft_id"],
        )
        return {**draft, "dry_run": True}

    try:
        result = await _persist_ask(pool, settings, draft=draft)
    except Exception as exc:
        logger.exception("strategy_switcher: persist failed")
        return {
            "action": "error",
            "dry_run": dry_run,
            "hypothesis_draft_id": draft["hypothesis_draft_id"],
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        await insert_audit_log(
            pool,
            hypothesis_id=result["hypothesis_id"],
            trust_level="n/a",
            tool_name=_TOOL_NAME,
            tool_input={
                "gate_entered_at": draft["gate_entered_at"],
                "dry_run": False,
            },
            tool_output={
                "ask_queue_id": result.get("ask_queue_id"),
                "campaigns": draft["campaigns"],
                "to_strategy": PROPOSED_STRATEGY,
            },
            is_mutation=False,  # draft-only; real mutation on Approve
        )
    except Exception:
        logger.warning("strategy_switcher: audit_log write failed", exc_info=True)

    logger.info(
        "strategy_switcher end: ask_queue_id=%s hypothesis_id=%s",
        result.get("ask_queue_id"),
        result["hypothesis_id"],
    )
    return {**result, "dry_run": False}


__all__ = [
    "CURRENT_STRATEGY",
    "PROPOSED_STRATEGY",
    "run",
]
