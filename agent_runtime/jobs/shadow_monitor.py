"""Shadow Invariant Monitor — emergency contingency (Task 16b).

Checks the single invariant of the 14-day shadow period: **there must be
zero rows in ``audit_log`` with ``is_mutation=TRUE`` and
``trust_level='shadow'``**. If any exist, the system is unsafe and must
be moved to ``FORBIDDEN_LOCK`` within the next 5-minute tick — catching
bugs in 5 minutes instead of 14 days.

Runs every 5 minutes via Railway Cron (``*/5 * * * *``) *only while*
``sda_state.trust_level = 'shadow'``. On other trust levels the job
no-ops (``{"skipped": true}``) so a legitimate transition to
``assisted`` / ``autonomous`` does not trigger retroactive locks from
pre-existing shadow-era rows.

Happy path: updates ``watchdog_heartbeat[service='shadow_monitor']`` so
Task 13 watchdog can detect a silent-failure regression in the guard
itself.

Violation path:

1. Compose a CRITICAL Telegram alert with up to 10 offending rows.
2. :func:`trust_levels.set_trust_level` → ``FORBIDDEN_LOCK`` with
   ``actor='shadow_monitor'``. This is the **emergency bypass** of
   :func:`trust_levels.allowed_action` — otherwise we would have a
   logical ring (the overlay forbids mutation → can't lock the system
   → can't enforce the invariant). Task 10's ``_ALLOWED_TRANSITIONS``
   explicitly permits ``* → FORBIDDEN_LOCK``.
3. Record a second ``audit_log`` row (``tool_name='shadow_monitor.lock'``,
   ``is_mutation=FALSE``) with the violating ids for forensics.

Telegram failure still locks: priority is ``lock > alert``. A missed
Telegram is acceptable; a runaway unlocked system is not.

``dry_run=True`` keeps the alert + state transitions off — useful for
manual diagnostic hits against prod audit_log without toggling trust.

See Decision 7 (trust state machine) and this task in ``tech-spec.md``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.trust_levels import TrustLevel, get_trust_level, set_trust_level

logger = logging.getLogger(__name__)


_LOCK_ACTOR = "shadow_monitor"
_ALERT_ROW_CAP = 10
_VIOLATION_FETCH_LIMIT = 50


# ---------------------------------------------------------------- data access


async def _fetch_violation_count(pool: AsyncConnectionPool) -> int:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT COUNT(*)
                FROM audit_log
                WHERE is_mutation = TRUE AND trust_level = 'shadow'
                """
            )
            row = await cur.fetchone()
    return int(row[0]) if row else 0


async def _fetch_violations(
    pool: AsyncConnectionPool, limit: int = _VIOLATION_FETCH_LIMIT
) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, ts, tool_name, hypothesis_id
                FROM audit_log
                WHERE is_mutation = TRUE AND trust_level = 'shadow'
                ORDER BY ts DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
    return [
        {
            "id": int(r[0]),
            "ts": r[1],
            "tool_name": str(r[2]),
            "hypothesis_id": r[3],
        }
        for r in rows
    ]


async def _beat_heartbeat(pool: AsyncConnectionPool) -> None:
    """UPSERT watchdog_heartbeat[service='shadow_monitor']. Never raises."""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO watchdog_heartbeat (service, last_beat_at)
                    VALUES ('shadow_monitor', NOW())
                    ON CONFLICT (service) DO UPDATE SET last_beat_at = NOW()
                    """
                )
    except Exception:
        logger.exception("shadow_monitor: heartbeat UPSERT failed (non-fatal)")


# ----------------------------------------------------------------- formatting


def _format_alert(violations_count: int, violations: list[dict[str, Any]]) -> str:
    lines = [
        "<b>SHADOW INVARIANT VIOLATED — LOCKING SYSTEM</b>",
        "",
        f"Violations: <b>{violations_count}</b>",
        "Top offenders:",
    ]
    shown = violations[:_ALERT_ROW_CAP]
    for v in shown:
        ts = v["ts"]
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, datetime) else str(ts)
        lines.append(
            f"  - #{v['id']} {ts_str} tool=<code>{v['tool_name']}</code> "
            f"hyp={v['hypothesis_id'] or '-'}"
        )
    if violations_count > _ALERT_ROW_CAP:
        lines.append(f"  ... and {violations_count - _ALERT_ROW_CAP} more")
    lines.append("")
    lines.append("System moved to <b>FORBIDDEN_LOCK</b>.")
    lines.append("Manual unlock via runbook (decisions.md, actor='owner-unlock').")
    return "\n".join(lines)


# ----------------------------------------------------------------- run_impl


async def _run_impl(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool,
    http_client: Any,
    settings: Settings | None,
) -> dict[str, Any]:
    current = await get_trust_level(pool)
    if current != TrustLevel.SHADOW:
        logger.info(
            "shadow_monitor: no-op, trust_level=%s (invariant only applies to shadow)",
            current.value,
        )
        return {
            "trust_level": current.value,
            "skipped": True,
            "reason": f"trust_level is not shadow (current={current.value})",
            "dry_run": dry_run,
        }

    violations_count = await _fetch_violation_count(pool)
    if violations_count == 0:
        await _beat_heartbeat(pool)
        logger.info("shadow_monitor: invariant OK (0 violations)")
        return {
            "trust_level": current.value,
            "violations_count": 0,
            "alert_sent": False,
            "locked": False,
            "dry_run": dry_run,
        }

    violations = await _fetch_violations(pool)
    violating_ids = [v["id"] for v in violations]

    if dry_run:
        alert_text = _format_alert(violations_count, violations)
        logger.warning(
            "shadow_monitor: dry_run — would lock, violations=%d, alert=%s",
            violations_count,
            alert_text,
        )
        return {
            "trust_level": current.value,
            "violations_count": violations_count,
            "violating_rows": violating_ids,
            "alert_sent": False,
            "locked": False,
            "dry_run": True,
        }

    alert_text = _format_alert(violations_count, violations)
    alert_sent = False
    if http_client is not None and settings is not None:
        try:
            await telegram_tools.send_message(http_client, settings, text=alert_text)
            alert_sent = True
        except Exception:
            logger.warning(
                "shadow_monitor: Telegram send failed; locking anyway",
                exc_info=True,
            )

    locked = False
    try:
        await set_trust_level(
            pool,
            TrustLevel.FORBIDDEN_LOCK,
            actor=_LOCK_ACTOR,
            reason=(
                f"shadow invariant violated: {violations_count} mutation(s) "
                f"in audit_log (top_ids={violating_ids[:_ALERT_ROW_CAP]})"
            ),
        )
        locked = True
    except ValueError as exc:
        # Concurrent run already locked; not an error.
        logger.info("shadow_monitor: already locked by concurrent run (%s)", exc)

    if locked:
        try:
            await insert_audit_log(
                pool,
                hypothesis_id=None,
                trust_level=TrustLevel.FORBIDDEN_LOCK.value,
                tool_name="shadow_monitor.lock",
                tool_input={
                    "violations_count": violations_count,
                    "violating_ids": violating_ids[:_ALERT_ROW_CAP],
                },
                tool_output={"alert_sent": alert_sent},
                is_mutation=False,
                kill_switch_triggered="shadow_invariant",
            )
        except Exception:
            logger.exception("shadow_monitor: forensic audit_log INSERT failed")

    logger.critical(
        "shadow_monitor: LOCKED system — violations=%d, alert_sent=%s",
        violations_count,
        alert_sent,
    )
    return {
        "trust_level": TrustLevel.FORBIDDEN_LOCK.value if locked else current.value,
        "violations_count": violations_count,
        "violating_rows": violating_ids,
        "alert_sent": alert_sent,
        "locked": locked,
        "dry_run": False,
    }


# ---------------------------------------------------------------- public run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: Any = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Cron entry + /run/shadow_monitor handler.

    Wrapped with a blanket try/except so a surprise (DB outage, schema
    drift, Telegram transport error we didn't anticipate) never takes
    the endpoint down — we still want the monitor to respond, even to
    say "I failed this tick, here's why". A best-effort fallback
    Telegram is attempted inside the crash handler; failure there is
    also swallowed so we don't exception-chain into the logs.
    """
    try:
        return await _run_impl(pool, dry_run=dry_run, http_client=http_client, settings=settings)
    except Exception as exc:
        logger.exception("shadow_monitor: crashed")
        if http_client is not None and settings is not None:
            try:
                await telegram_tools.send_message(
                    http_client,
                    settings,
                    text=(f"<b>SHADOW MONITOR CRASHED</b>\n{type(exc).__name__}: {exc}"),
                )
            except Exception:
                logger.exception("shadow_monitor: could not deliver crash Telegram alert")
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "alert_sent": False,
            "locked": False,
            "dry_run": dry_run,
        }


__all__ = ["run"]
