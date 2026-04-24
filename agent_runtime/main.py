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

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

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
from agent_runtime.config import Settings, get_settings
from agent_runtime.db import create_pool, db_ping, run_migrations

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

        reflection_store = PGReflectionStore(settings.DATABASE_URL)
        await reflection_store.ensure_schema()
        applied = await run_migrations(pool, MIGRATIONS_DIR)
        logger.info(
            "lifespan startup: pool opened, reflections schema ok, migrations applied: %d",
            len(applied),
        )

        try:
            yield
        finally:
            await pool.close()
            logger.info("lifespan shutdown: pool closed")

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
    async def run_job(request: Request, job: str) -> JSONResponse:
        # Real dispatch lands in Tasks 9-18. Stub returns the job name so
        # smoke tests and auth tests can verify the gate without needing a
        # live registry.
        return JSONResponse({"status": "ok", "job": job, "executed": False})

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
