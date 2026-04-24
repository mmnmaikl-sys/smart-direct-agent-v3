"""Job registry for ``/run/{job}`` dispatch.

Each job module exposes ``async def run(pool, *, dry_run: bool = False) -> dict``.
The registry maps a public slug (used in the cron URL / Railway schedule) to
the callable. Jobs arrive in Wave 1 / Wave 2 tasks — this module starts with
``audit_retention`` (Task 5c) and grows as later tasks register their jobs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from psycopg_pool import AsyncConnectionPool

from agent_runtime.jobs import audit_retention, budget_guard, impact_tracker_job, watchdog

JobCallable = Callable[..., Awaitable[dict[str, Any]]]

JOB_REGISTRY: dict[str, JobCallable] = {
    "audit_retention": audit_retention.run,
    "budget_guard": budget_guard.run,
    "impact_tracker": impact_tracker_job.run,
    "watchdog": watchdog.run,
}


async def dispatch_job(
    name: str,
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Look up ``name`` and invoke with the shared DB pool.

    Raises :class:`KeyError` when the job is not registered — the HTTP layer
    translates that into a 404.
    """
    fn = JOB_REGISTRY[name]
    return await fn(pool, dry_run=dry_run)


__all__ = ["JOB_REGISTRY", "dispatch_job"]
