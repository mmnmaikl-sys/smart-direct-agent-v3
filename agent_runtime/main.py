"""FastAPI entrypoint for smart-direct-agent-v3.

Wave 1 surface:
* ``GET /health``              — unauth, only public endpoint (Decision 11)
* ``POST /run/{job}``          — HTTPBearer, rate-limited 30/min (stub)
* ``POST /webhook/bitrix``     — HMAC-SHA256, rate-limited 100/min (stub)
* ``POST /webhook/lead``       — HMAC-SHA256, rate-limited 100/min (stub)
* ``POST /admin/trust_level``  — HTTPBearer + Telegram confirmation gate (stub)

Real job handlers land in Tasks 9-18; webhook handlers in Task 8; admin-confirm
resolver in Task 11. This file owns the DB pool lifecycle and wires auth
dependencies onto every mutating endpoint from day one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from agents_core.llm.client import LLMClient
from agents_core.memory.reflection import PGReflectionStore
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from agent_runtime.auth import (
    build_rate_limiter,
    require_admin_confirmation,
    require_internal_key,
    verify_webhook_signature,
)
from agent_runtime.auth.signing import HMACSigner
from agent_runtime.config import Settings, get_settings
from agent_runtime.db import create_pool, db_ping, run_migrations
from agent_runtime.jobs import JOB_REGISTRY, JobContext, dispatch_job
from agent_runtime.jobs.bfl_rf_lead_poller import LeadPoller, _lead_poller_loop
from agent_runtime.tools.direct_api import DirectAPI
from agent_runtime.tools.registry import build_registry

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _make_lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = create_pool(
            settings.DATABASE_URL,
            min_size=settings.DB_POOL_MIN_SIZE,
            max_size=settings.DB_POOL_MAX_SIZE,
        )
        await pool.open()
        app.state.pool = pool
        app.state.settings = settings
        app.state.agents_count = 0
        app.state.jobs_count = 0

        # One shared httpx client for Bitrix / Metrika / Telegram tools.
        # Connection pool kept modest — these endpoints rate-limit harder
        # than they hang under concurrent load.
        app.state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=60.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )

        reflection_store = PGReflectionStore(settings.DATABASE_URL)
        await reflection_store.ensure_schema()
        applied = await run_migrations(pool, MIGRATIONS_DIR)
        logger.info(
            "lifespan startup: pool opened, reflections schema ok, migrations applied: %d",
            len(applied),
        )

        # Task 28a JobContext — build every shared resource ONCE so each
        # cron-triggered job runs with full DI rather than the legacy
        # ``degraded_noop`` fallback. DirectAPI uses the shared httpx
        # client (no __aenter__ side-effect needed). LLMClient is built
        # lazily by the brain when first invoked, but we plumb a singleton
        # here so the registry + smart_optimizer share it.
        direct = DirectAPI(settings, http_client=app.state.http_client)
        try:
            llm_client: LLMClient | None = LLMClient(
                anthropic_api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
                http_client=app.state.http_client,
            )
        except Exception:
            # Missing/invalid key on import → run without brain; jobs that
            # need it will degrade gracefully, others (watchdog,
            # shadow_monitor, audit_retention, ...) keep working.
            logger.warning(
                "lifespan: LLMClient init failed; brain-dependent jobs will degrade",
                exc_info=True,
            )
            llm_client = None
        tool_registry = build_registry(settings, direct=direct, http_client=app.state.http_client)
        signer = HMACSigner(settings.HYPOTHESIS_HMAC_SECRET)
        app.state.job_ctx = JobContext(
            settings=settings,
            http_client=app.state.http_client,
            direct=direct,
            llm_client=llm_client,
            tool_registry=tool_registry,
            signer=signer,
            reflection_store=reflection_store,
        )
        app.state.jobs_count = len(JOB_REGISTRY)
        logger.info(
            "lifespan startup: JobContext ready (jobs=%d, llm=%s)",
            len(JOB_REGISTRY),
            "ok" if llm_client is not None else "missing",
        )

        # Lead poller — in-process asyncio loop (Decision 9). Only started
        # when a UTM whitelist is configured; otherwise we let CI / dev
        # environments boot without background noise.
        poller_task: asyncio.Task[None] | None = None
        if settings.LEAD_POLLER_UTM_WHITELIST:
            poller = LeadPoller(pool, app.state.http_client, settings)
            poller_task = asyncio.create_task(_lead_poller_loop(poller))
            app.state.lead_poller_task = poller_task

        try:
            yield
        finally:
            if poller_task is not None:
                poller_task.cancel()
                await asyncio.gather(poller_task, return_exceptions=True)
                logger.info("Lead poller stopped")
            await app.state.http_client.aclose()
            await pool.close()
            logger.info("lifespan shutdown: http client + pool closed")

    return lifespan


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded"},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory used by uvicorn and tests (so tests can pass a fake Settings)."""
    settings = settings or get_settings()
    app = FastAPI(
        title="smart-direct-agent-v3",
        version=settings.APP_VERSION,
        lifespan=_make_lifespan(settings),
    )

    limiter = build_rate_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        pool = getattr(request.app.state, "pool", None)
        db_ok = await db_ping(pool) if pool is not None else False
        payload = {
            "status": "ok" if db_ok else "degraded",
            "version": settings.APP_VERSION,
            "agents_count": getattr(request.app.state, "agents_count", 0),
            "jobs_count": getattr(request.app.state, "jobs_count", 0),
            "db": "ok" if db_ok else "error",
        }
        return JSONResponse(payload, status_code=200)

    @app.post("/run/{job}", dependencies=[Depends(require_internal_key)])
    @limiter.limit("30/minute")
    async def run_job(request: Request, job: str, dry_run: bool = False) -> JSONResponse:
        # Unregistered jobs return 200 with `executed=False` so auth/rate-limit
        # tests keep working without a live registry for every job name. Jobs
        # that ARE registered (starting with audit_retention in Task 5c) run
        # for real.
        if job not in JOB_REGISTRY:
            return JSONResponse({"status": "ok", "job": job, "executed": False})
        pool = getattr(request.app.state, "pool", None)
        if pool is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "error", "detail": "DB pool not initialised"},
            )
        ctx = getattr(request.app.state, "job_ctx", None)
        result = await dispatch_job(job, pool, dry_run=dry_run, ctx=ctx)
        return JSONResponse({"status": "ok", "job": job, "executed": True, "result": result})

    @app.post("/webhook/bitrix")
    @limiter.limit("100/minute")
    async def webhook_bitrix(
        request: Request,
        body: bytes = Depends(verify_webhook_signature),
    ) -> JSONResponse:
        # Real lead parsing lands in Task 8. Stub acks receipt.
        return JSONResponse({"status": "received", "bytes": len(body)})

    @app.post("/webhook/lead")
    @limiter.limit("100/minute")
    async def webhook_lead(
        request: Request,
        body: bytes = Depends(verify_webhook_signature),
    ) -> JSONResponse:
        return JSONResponse({"status": "received", "bytes": len(body)})

    @app.post(
        "/admin/trust_level",
        dependencies=[Depends(require_internal_key)],
    )
    @limiter.limit("30/minute")
    async def admin_trust_level(
        request: Request,
        ask_id: int = Depends(require_admin_confirmation),
    ) -> JSONResponse:
        # 202: owner must approve via Telegram (Task 11) before the action
        # actually runs. Endpoint is a pure gate here.
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "pending_confirmation",
                "ask_id": ask_id,
                "detail": (
                    "Confirmation sent to Telegram. Poll /admin/confirm?ask_id=X after approval."
                ),
            },
        )

    return app


# Uvicorn entrypoint: ``uvicorn agent_runtime.main:app``.
# Importing this module without DATABASE_URL / secrets set raises — intentional, fail-fast.
app = create_app()
