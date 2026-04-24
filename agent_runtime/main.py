"""FastAPI entrypoint for smart-direct-agent-v3.

Wave 1 surface: only ``GET /health`` (unauth per Decision 11). Auth layer and
mutating endpoints arrive in Task 5b. This file owns the DB pool lifecycle via
the FastAPI lifespan and runs migrations + reflection-store schema on startup.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from agents_core.memory.reflection import PGReflectionStore
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory used by uvicorn and tests (so tests can pass a fake Settings)."""
    settings = settings or get_settings()
    app = FastAPI(
        title="smart-direct-agent-v3",
        version=settings.APP_VERSION,
        lifespan=_make_lifespan(settings),
    )

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

    return app


# Uvicorn entrypoint: ``uvicorn agent_runtime.main:app``.
# Importing this module without DATABASE_URL set raises — intentional, fail-fast.
app = create_app()
