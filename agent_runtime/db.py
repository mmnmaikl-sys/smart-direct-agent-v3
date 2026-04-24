"""Async PostgreSQL helpers: pool factory, idempotent migration runner, ping.

The pool is created but **not opened** by ``create_pool``; ownership lives in
the FastAPI lifespan (``agent_runtime.main``) which calls ``await pool.open()``
on startup and ``await pool.close()`` on shutdown. No module-level globals —
makes lifecycle explicit and testable.

Migrations are idempotent at the SQL level (``CREATE TABLE IF NOT EXISTS``),
so there is no ``schema_migrations`` ledger — per Decision 3, this is simpler
than we need for Wave 1 (single migration file, single-tenant deployment).

SECURITY: :func:`insert_audit_log` is the ONLY sanctioned entry point into
``audit_log``. Direct ``INSERT INTO audit_log`` anywhere else skips the PII
sanitiser (Decision 13) and ships raw Bitrix PII into Postgres. Callers must
route through ``insert_audit_log`` even for agent-internal state — the
wrapper cost is microseconds and the legal exposure is 300K–1M₽ per
incident (КоАП 13.11).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime.pii import sanitize_audit_payload

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


async def insert_audit_log(
    pool: AsyncConnectionPool,
    *,
    hypothesis_id: str | None,
    trust_level: str,
    tool_name: str,
    tool_input: Any,
    tool_output: Any,
    is_mutation: bool,
    is_error: bool = False,
    error_detail: str | None = None,
    user_confirmed: bool = False,
    kill_switch_triggered: str | None = None,
) -> int:
    """Sanitise ``tool_input``/``tool_output`` and INSERT into ``audit_log``.

    SECURITY: only entry point. See module docstring. Any new call site MUST
    use this function; greppable assertion is in
    ``tests/unit/test_pii_no_direct_insert.py``.

    Returns the new ``audit_log.id`` (BIGSERIAL).
    """
    safe_input = sanitize_audit_payload(tool_input)
    safe_output = sanitize_audit_payload(tool_output)
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO audit_log (
                    hypothesis_id, trust_level, tool_name,
                    tool_input, tool_output, is_mutation,
                    is_error, error_detail, user_confirmed,
                    kill_switch_triggered
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    hypothesis_id,
                    trust_level,
                    tool_name,
                    Jsonb(safe_input),
                    Jsonb(safe_output) if safe_output is not None else None,
                    is_mutation,
                    is_error,
                    error_detail,
                    user_confirmed,
                    kill_switch_triggered,
                ),
            )
            row = await cur.fetchone()
    if row is None:
        raise RuntimeError("audit_log INSERT did not return id")
    return int(row[0])
