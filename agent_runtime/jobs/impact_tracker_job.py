"""Cron orchestrator for the hypothesis lifecycle (Task 12b).

Runs every 6h (Railway Cron ``0 */6 * * *``) and drives:

1. ``get_pending_checks`` → ``measure_outcome`` per hypothesis.
2. ``mark_expired`` to hard-cap runaway rows at ``maximum_running_days``.
3. ``release_bucket_and_start_waiting`` to promote queued hypotheses.

Each ``measure_outcome`` is wrapped in its own try/except so one bad row
cannot sink the whole cron tick. ``dry_run=True`` executes steps 1–3 but
skips the mutating ``update_outcome`` / ``promote_to_prod`` / ``rollback``
call sites — useful for operator smoke-checks.

The job is registered in :mod:`agent_runtime.jobs` under the name
``impact_tracker`` so Railway Cron can POST ``/run/impact_tracker``
through the internal Bearer gate.
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg_pool import AsyncConnectionPool

from agent_runtime import decision_journal, impact_tracker

logger = logging.getLogger(__name__)


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: Any = None,
    reflection_store: Any = None,
) -> dict[str, Any]:
    """Execute one impact_tracker cron cycle and return a structured report.

    ``direct`` and ``reflection_store`` default to ``None`` so the JOB_REGISTRY
    wrapper (which only forwards ``pool`` + ``dry_run``) does not have to
    plumb every dependency. In dry-run the Direct client is never called;
    in a real run the FastAPI handler at ``/run/impact_tracker`` should
    inject them from ``app.state`` before forwarding here.
    """
    pending = await decision_journal.get_pending_checks(pool)
    outcomes: list[dict[str, Any]] = []

    for row in pending:
        if dry_run:
            outcomes.append(
                {
                    "hypothesis_id": row.id,
                    "skipped": True,
                    "reason": "dry_run",
                }
            )
            continue
        if direct is None:
            logger.warning(
                "impact_tracker_job: direct client absent, skipping measure for %s",
                row.id,
            )
            outcomes.append(
                {"hypothesis_id": row.id, "skipped": True, "reason": "no_direct_client"}
            )
            continue
        try:
            outcome = await impact_tracker.measure_outcome(
                pool,
                row.id,
                direct=direct,
                reflection_store=reflection_store,
            )
            outcomes.append(outcome.model_dump())
        except Exception:
            logger.exception("measure_outcome failed for hypothesis %s", row.id)
            outcomes.append({"hypothesis_id": row.id, "error": True, "reason": "measure_failed"})

    if dry_run:
        expired: list[str] = []
        released: list[str] = []
    else:
        try:
            expired = await impact_tracker.mark_expired(pool)
        except Exception:
            logger.exception("mark_expired failed")
            expired = []
        try:
            released = await impact_tracker.release_bucket_and_start_waiting(pool)
        except Exception:
            logger.exception("release_bucket_and_start_waiting failed")
            released = []

    report: dict[str, Any] = {
        "status": "ok",
        "dry_run": dry_run,
        "pending_measured": len(pending),
        "expired": expired,
        "released": released,
        "outcomes": outcomes,
    }
    logger.info(
        "impact_tracker_job done: pending=%d expired=%d released=%d dry_run=%s",
        len(pending),
        len(expired),
        len(released),
        dry_run,
    )
    return report


__all__ = ["run"]
