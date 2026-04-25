"""Job registry for ``/run/{job}`` dispatch.

Each job module exposes ``async def run(pool, *, dry_run: bool = False, **deps)``.
The registry maps a public slug (used in the cron URL / Railway schedule) to
the callable.

Task 28a integration: :class:`JobContext` carries every shared resource the
FastAPI lifespan builds once (DirectAPI, LLMClient, ToolRegistry, signers,
ReflectionStore, http_client, settings). :func:`dispatch_job` introspects
each job's signature and passes only the kwargs that job declares — older
jobs that don't accept a given resource simply don't get it. This closes
the Task 28 audit critical: jobs no longer fall to ``degraded_noop`` for
missing DI when the lifespan has built everything.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Any

from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    import httpx
    from agents_core.llm.client import LLMClient
    from agents_core.memory.reflection import PGReflectionStore
    from agents_core.tools.registry import ToolRegistry

    from agent_runtime.auth.signing import HMACSigner
    from agent_runtime.config import Settings
    from agent_runtime.tools.direct_api import DirectAPI


@dataclass
class JobContext:
    """Shared resources available to every job.

    Constructed once in the FastAPI lifespan and stored on
    ``app.state.job_ctx``. Each field is independently optional so the
    JOB_REGISTRY default-path (``(pool, dry_run)``) and tests can pass a
    minimal context without breaking.
    """

    settings: Settings | None = None
    http_client: httpx.AsyncClient | None = None
    direct: DirectAPI | None = None
    llm_client: LLMClient | None = None
    tool_registry: ToolRegistry | None = None
    signer: HMACSigner | None = None
    reflection_store: PGReflectionStore | None = None


# Submodule imports are below the JobContext dataclass to keep type
# annotations resolving cleanly under TYPE_CHECKING. The noqa silences
# ruff's E402 module-level-import ordering rule which doesn't account
# for forward-reference dataclasses defined above.
from agent_runtime.jobs import (  # noqa: E402
    audience_sync,
    audit_retention,
    auto_resume,
    autotargeting_manager,
    bfl_rf_lead_poller,
    bfl_rf_watchdog,
    bitrix_feedback,
    budget_guard,
    form_checker,
    health_checker,
    impact_tracker_job,
    learner,
    offline_conversions,
    query_analyzer,
    regression_watch,
    shadow_monitor,
    smart_optimizer,
    strategic_advisor,
    strategy_gate,
    strategy_switcher,
    telegram_digest,
    watchdog,
)

JobCallable = Callable[..., Awaitable[dict[str, Any]]]

JOB_REGISTRY: dict[str, JobCallable] = {
    "audit_retention": audit_retention.run,
    "audience_sync": audience_sync.run,
    "auto_resume": auto_resume.run,
    "autotargeting_manager": autotargeting_manager.run,
    "bfl_rf_watchdog": bfl_rf_watchdog.run,
    "bitrix_feedback": bitrix_feedback.run,
    "budget_guard": budget_guard.run,
    "form_checker": form_checker.run,
    "health_checker": health_checker.run,
    "impact_tracker": impact_tracker_job.run,
    "learner": learner.run,
    "lead_poller": bfl_rf_lead_poller.run,
    "offline_conversions": offline_conversions.run,
    "query_analyzer": query_analyzer.run,
    "regression_watch": regression_watch.run,
    "shadow_monitor": shadow_monitor.run,
    "smart_optimizer": smart_optimizer.run,
    "strategic_advisor": strategic_advisor.run,
    "strategy_gate": strategy_gate.run,
    "strategy_switcher": strategy_switcher.run,
    "telegram_digest": telegram_digest.run,
    "watchdog": watchdog.run,
}


_DI_FIELDS: tuple[str, ...] = (
    "direct",
    "http_client",
    "settings",
    "llm_client",
    "tool_registry",
    "signer",
    "reflection_store",
)


async def dispatch_job(
    name: str,
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    ctx: JobContext | None = None,
) -> dict[str, Any]:
    """Look up ``name`` and invoke with the shared DB pool + selective DI.

    Each registered ``run(pool, *, dry_run, ...)`` declares the shared
    resources it accepts via keyword-only parameters with ``None`` default.
    We inspect the function's signature once and pass only the matching
    ones from ``ctx`` — preventing ``TypeError`` for jobs that don't
    declare a given dep, and giving every job a meaningful DI when the
    lifespan has built the resource. Without ``ctx`` (legacy / tests),
    we still pass ``(pool, dry_run)`` and the job falls into its
    ``degraded_noop`` path as before.

    Raises :class:`KeyError` when the job is not registered — the HTTP
    layer translates that into a 404.
    """
    fn = JOB_REGISTRY[name]
    kwargs: dict[str, Any] = {"dry_run": dry_run}
    if ctx is not None:
        sig = signature(fn)
        for field in _DI_FIELDS:
            if field in sig.parameters:
                value = getattr(ctx, field, None)
                if value is not None:
                    kwargs[field] = value
    return await fn(pool, **kwargs)


__all__ = ["JOB_REGISTRY", "JobContext", "dispatch_job"]
