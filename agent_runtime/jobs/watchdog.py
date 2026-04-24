"""Watchdog — dead-man's-switch for all cron jobs (Task 13).

Every other job calls :func:`heartbeat` on entry (or exit) to bump
``watchdog_heartbeat.last_beat_at``. This job runs every hour and:

1. Self-beats first — if anything below explodes, the next tick still
   sees the watchdog as fresh and won't false-positive it to itself.
2. Selects every non-watchdog row where ``last_beat_at`` is older than
   ``HEARTBEAT_STALE_MINUTES`` (90 min — 60 min cron + 30 min slack).
3. If stale services exist, emits a CRITICAL Telegram alert and, in
   ``autonomous`` trust level, suspends every campaign in
   ``PROTECTED_CAMPAIGN_IDS`` as the last line of defence against a
   silently broken guard burning budget.

In ``shadow`` the job is NOTIFY-only; in ``assisted`` a row lands in
``ask_queue`` for the owner to confirm. ``dry_run=True`` keeps the
Telegram alert but skips every mutation and queue insert so the smoke
path can validate alert delivery without touching prod.

Failure path is fail-loud: any exception inside :func:`run` re-raises
after trying to send a ``WATCHDOG CRASHED`` Telegram alert — better to
surface the 500 than to fail silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI
from agent_runtime.trust_levels import TrustLevel, get_trust_level

logger = logging.getLogger(__name__)


HEARTBEAT_STALE_MINUTES = 90  # 60 min cron + 30 min slack
_VERIFY_RETRY_ATTEMPTS = 3
_VERIFY_RETRY_SLEEP_SEC = 2.0


@dataclass(frozen=True)
class StaleService:
    service: str
    last_beat_at: datetime
    minutes_stale: float


class WatchdogResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str  # "ok" | "stale_detected" | "error"
    stale_services: list[dict[str, Any]] = Field(default_factory=list)
    suspended_campaigns: list[int] = Field(default_factory=list)
    ask_id: int | None = None
    alerted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ------------------------------------------------------------------ heartbeat


async def heartbeat(pool: AsyncConnectionPool, service: str) -> None:
    """Upsert ``watchdog_heartbeat`` so stale-check sees this service fresh.

    Every Wave 2 cron job should call this on entry. Intentionally cheap
    (single UPSERT) and idempotent; audit_log is NOT touched — heartbeats
    would flood the table.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO watchdog_heartbeat (service, last_beat_at)
                VALUES (%s, NOW())
                ON CONFLICT (service) DO UPDATE SET last_beat_at = NOW()
                """,
                (service,),
            )


async def get_stale_services(
    pool: AsyncConnectionPool,
    threshold_minutes: int = HEARTBEAT_STALE_MINUTES,
) -> list[StaleService]:
    """Return every service (except watchdog itself) older than ``threshold``."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT service,
                       last_beat_at,
                       EXTRACT(EPOCH FROM (NOW() - last_beat_at)) / 60
                FROM watchdog_heartbeat
                WHERE service <> 'watchdog'
                  AND last_beat_at < NOW() - make_interval(mins => %s)
                ORDER BY last_beat_at ASC
                """,
                (threshold_minutes,),
            )
            rows = await cur.fetchall()
    return [
        StaleService(service=str(r[0]), last_beat_at=r[1], minutes_stale=float(r[2])) for r in rows
    ]


# ------------------------------------------------------- trust + side effects


async def _suspend_protected_campaigns(direct: DirectAPI, settings: Settings) -> list[int]:
    """Best-effort: pause every PROTECTED campaign. Returns the successes.

    One failing campaign does not block the others — better to suspend
    3/4 than 0/4 when DirectAPI wobbles under load.
    """
    suspended: list[int] = []
    for campaign_id in settings.PROTECTED_CAMPAIGN_IDS:
        try:
            await direct.pause_campaign(campaign_id)
        except Exception:
            logger.exception("watchdog: pause_campaign(%d) failed", campaign_id)
            continue
        # Direct is eventually consistent — three quick re-verifies give it
        # a few seconds to catch up before we log a warning.
        verified = False
        for attempt in range(_VERIFY_RETRY_ATTEMPTS):
            try:
                verified = await direct.verify_campaign_paused(campaign_id)
                if verified:
                    break
            except Exception:
                logger.warning(
                    "watchdog: verify_campaign_paused(%d) attempt %d errored",
                    campaign_id,
                    attempt + 1,
                    exc_info=True,
                )
            await _sleep_for_verify()
        if verified:
            suspended.append(int(campaign_id))
        else:
            logger.warning("watchdog: verify_campaign_paused(%d) never returned True", campaign_id)
    return suspended


async def _sleep_for_verify() -> None:
    """Isolated for monkeypatching in tests."""
    import asyncio

    await asyncio.sleep(_VERIFY_RETRY_SLEEP_SEC)


async def _create_ask_queue_row(
    pool: AsyncConnectionPool, stale_services: list[StaleService]
) -> int | None:
    """Queue an owner confirmation for the assisted trust level.

    Uses the ``admin`` sentinel hypothesis row (Task 5b seeded it) so the
    existing ``ask_queue.hypothesis_id`` FK is satisfied without creating
    a new row per alert.
    """
    question = (
        f"Watchdog detected {len(stale_services)} stale services "
        f"(>90 min idle). Suspend PROTECTED campaigns?"
    )
    payload = {
        "question": question,
        "services": [s.service for s in stale_services],
    }
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Ensure the admin sentinel exists — same pattern as Task 5b.
            await cur.execute(
                """
                INSERT INTO hypotheses (
                    id, agent, hypothesis_type, signals, hypothesis, reasoning,
                    actions, expected_outcome, budget_cap_rub, autonomy_level,
                    risk_score, metrics_before
                ) VALUES (
                    'admin', 'admin', 'account_level', '[]'::jsonb,
                    'admin action gate', 'admin gate', '[]'::jsonb,
                    'owner approval', 1, 'ASK', 0.0, '{}'::jsonb
                ) ON CONFLICT (id) DO NOTHING
                """
            )
            await cur.execute(
                """
                INSERT INTO ask_queue (hypothesis_id, question, options)
                VALUES ('admin', %s, %s)
                RETURNING id
                """,
                (question, Jsonb(payload)),
            )
            row = await cur.fetchone()
    return int(row[0]) if row else None


def _format_alert(
    stale: list[StaleService],
    trust_level: TrustLevel,
    planned_action: str,
) -> str:
    lines = [
        "<b>WATCHDOG ALERT</b>",
        "",
        f"{len(stale)} service(s) stale over {HEARTBEAT_STALE_MINUTES} min:",
        "",
    ]
    for s in stale:
        lines.append(
            f"• <b>{s.service}</b>: last beat "
            f"<code>{s.last_beat_at.isoformat()}</code> "
            f"({s.minutes_stale:.0f} min ago)"
        )
    lines.append("")
    lines.append(f"Trust: <code>{trust_level.value}</code>")
    lines.append(f"Action: {planned_action}")
    return "\n".join(lines)


# ------------------------------------------------------------------- run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Watchdog cron entry. Returns ``WatchdogResult.model_dump()``.

    ``direct`` / ``http_client`` / ``settings`` default to ``None`` so the
    JOB_REGISTRY wrapper (which passes only ``pool`` + ``dry_run``) still
    works in dry-run / health-check mode. The FastAPI handler at
    ``/run/watchdog`` injects them from ``app.state`` for real runs.
    """
    try:
        # Self-beat FIRST — if something below crashes, next tick still sees
        # the watchdog as fresh and does not false-positive itself.
        await heartbeat(pool, "watchdog")
        stale = await get_stale_services(pool)
    except Exception as exc:
        await _fail_loud(http_client, settings, exc)
        raise

    if not stale:
        logger.info("watchdog: all services fresh")
        return WatchdogResult(status="ok").model_dump(mode="json")

    try:
        trust_level = await get_trust_level(pool)
    except Exception:
        logger.warning("watchdog: trust_level lookup failed, defaulting shadow", exc_info=True)
        trust_level = TrustLevel.SHADOW

    suspended: list[int] = []
    ask_id: int | None = None

    if trust_level == TrustLevel.AUTONOMOUS and not dry_run:
        if direct is not None and settings is not None:
            suspended = await _suspend_protected_campaigns(direct, settings)
        planned_action = f"auto-suspended {suspended}"
    elif trust_level == TrustLevel.ASSISTED and not dry_run:
        ask_id = await _create_ask_queue_row(pool, stale)
        planned_action = f"ASK queue created (id={ask_id})"
    elif trust_level == TrustLevel.FORBIDDEN_LOCK:
        planned_action = "NOTIFY only — trust is FORBIDDEN_LOCK"
    elif trust_level == TrustLevel.SHADOW:
        planned_action = "NOTIFY only — shadow trust"
    else:
        planned_action = "NOTIFY only (dry_run=true)" if dry_run else "NOTIFY only"

    if http_client is not None and settings is not None:
        try:
            await telegram_tools.send_message(
                http_client,
                settings,
                text=_format_alert(stale, trust_level, planned_action),
            )
        except Exception:
            logger.exception("watchdog: telegram alert failed")

    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level=trust_level.value,
            tool_name="watchdog",
            tool_input={"stale_services": [s.service for s in stale]},
            tool_output={
                "action": planned_action,
                "suspended": suspended,
                "ask_id": ask_id,
            },
            is_mutation=(trust_level == TrustLevel.AUTONOMOUS and not dry_run),
            is_error=False,
        )
    except Exception:
        logger.exception("watchdog: audit_log write failed")

    result = WatchdogResult(
        status="stale_detected",
        stale_services=[
            {
                "service": s.service,
                "last_beat_at": s.last_beat_at.isoformat(),
                "minutes_stale": s.minutes_stale,
            }
            for s in stale
        ],
        suspended_campaigns=suspended,
        ask_id=ask_id,
    )
    return result.model_dump(mode="json")


async def _fail_loud(
    http_client: httpx.AsyncClient | None,
    settings: Settings | None,
    exc: BaseException,
) -> None:
    """Try to alert before re-raising; never shadow the underlying error."""
    logger.exception("watchdog CRASHED")
    if http_client is None or settings is None:
        return
    try:
        await telegram_tools.send_message(
            http_client,
            settings,
            text=f"<b>WATCHDOG CRASHED</b>: {type(exc).__name__}: {exc}",
        )
    except Exception:
        logger.exception("watchdog: could not deliver fail-loud Telegram alert")


__all__ = [
    "HEARTBEAT_STALE_MINUTES",
    "StaleService",
    "WatchdogResult",
    "get_stale_services",
    "heartbeat",
    "run",
]
