"""Async PostgreSQL helpers: pool factory, idempotent migration runner, ping.

The pool is created but **not opened** by ``create_pool``; ownership lives in
the FastAPI lifespan (``agent_runtime.main``) which calls ``await pool.open()``
on startup and ``await pool.close()`` on shutdown. No module-level globals —
makes lifecycle explicit and testable.

Migrations are idempotent at the SQL level (``CREATE TABLE IF NOT EXISTS``),
so there is no ``schema_migrations`` ledger — per Decision 3, this is simpler
than we need for Wave 1 (single migration file, single-tenant deployment).
"""

from __future__ import annotations

import logging
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

ROLLBACK_SUFFIX = "_rollback.sql"


def create_pool(
    dsn: str,
    *,
    min_size: int = 2,
    max_size: int = 10,
) -> AsyncConnectionPool:
    """Return an un-opened AsyncConnectionPool.

    Caller must ``await pool.open()`` before use and ``await pool.close()``
    on shutdown. Kept un-opened here so callers control lifecycle explicitly.
    """
    return AsyncConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        open=False,
    )


async def run_migrations(
    pool: AsyncConnectionPool,
    migrations_dir: Path,
) -> list[str]:
    """Apply every ``*.sql`` in ``migrations_dir`` except ``*_rollback.sql``.

    Files are applied in lexicographic order inside a single transaction each.
    Idempotent: re-running is safe because DDL uses ``IF NOT EXISTS``.
    Returns the list of applied filenames (for logging / observability).
    """
    # One-shot startup path: the filesystem stats and glob here run once during
    # lifespan init, not in a request-handling hot loop, so the tiny blocking
    # syscall does not justify pulling in trio.Path / anyio.
    if not migrations_dir.exists():  # noqa: ASYNC240
        raise FileNotFoundError(f"migrations dir does not exist: {migrations_dir}")

    files = sorted(
        p
        for p in migrations_dir.glob("*.sql")  # noqa: ASYNC240
        if not p.name.endswith(ROLLBACK_SUFFIX)
    )
    applied: list[str] = []
    for path in files:
        sql_text = path.read_text(encoding="utf-8")
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql_text)  # type: ignore[arg-type]
        logger.info("run_migrations applied: %s", path.name)
        applied.append(path.name)
    return applied


async def db_ping(pool: AsyncConnectionPool) -> bool:
    """Return True iff ``SELECT 1`` round-trip succeeds. Never raises."""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                row = await cur.fetchone()
                return row is not None and row[0] == 1
    except Exception:
        logger.warning("db_ping failed", exc_info=True)
        return False
