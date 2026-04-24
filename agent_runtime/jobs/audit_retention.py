"""Weekly ``audit_log`` retention — deletes rows older than 90 days.

Triggered by Railway Cron every Sunday 00:00 UTC (= 03:00 МСК) via
``POST /run/audit_retention?dry_run=false`` with the internal Bearer. The
handler dispatches here through ``agent_runtime.jobs.dispatch_job``.

Decision 13 retention:
    audit_log — 90 days
    hypotheses — 365 days (handled by a separate job, not here)
    ask_queue — 30 days (handled by a separate job, not here)
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

RETENTION_DAYS = 90


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete (or count, in dry-run) rows older than ``RETENTION_DAYS``.

    DELETE is atomic per row — concurrent INSERTs are safe. The whole
    operation runs in a single transaction; at current Wave 1 volume
    (≤10k rows/month) it completes in well under a second.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            if dry_run:
                await cur.execute(
                    "SELECT count(*) FROM audit_log WHERE ts < now() - make_interval(days => %s)",
                    (RETENTION_DAYS,),
                )
                row = await cur.fetchone()
                deleted = int(row[0]) if row else 0
                logger.info("audit_retention dry-run: would_delete=%d", deleted)
                return {"deleted": deleted, "dry_run": True}

            await cur.execute(
                "DELETE FROM audit_log WHERE ts < now() - make_interval(days => %s)",
                (RETENTION_DAYS,),
            )
            deleted = cur.rowcount
    logger.info("audit_retention: deleted=%d", deleted)
    return {"deleted": deleted, "dry_run": False}
