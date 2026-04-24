"""Database layer tests.

Migration tests use ``pytest-postgresql`` which spins up a local ephemeral PG
instance. When the binary is unavailable locally the tests are skipped; CI
installs ``postgresql-contrib`` so they run there.

``db_ping`` tests use a mocked pool — we only need to verify the contract
(True on 1, False on exception) not the real round-trip.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.db import create_pool, db_ping, run_migrations

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


def _make_pool_mock(
    fetchone_return=(1,),
    execute_raises: Exception | None = None,
):
    pool = MagicMock()
    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=execute_raises)
    cursor.fetchone = AsyncMock(return_value=fetchone_return)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)

    pool.connection = MagicMock(return_value=conn)
    return pool


@pytest.mark.asyncio
async def test_db_ping_returns_true_on_healthy_connection() -> None:
    pool = _make_pool_mock(fetchone_return=(1,))
    assert await db_ping(pool) is True


@pytest.mark.asyncio
async def test_db_ping_returns_false_on_connection_error() -> None:
    pool = _make_pool_mock(execute_raises=RuntimeError("connection refused"))
    assert await db_ping(pool) is False


@pytest.mark.asyncio
async def test_db_ping_returns_false_on_unexpected_result() -> None:
    pool = _make_pool_mock(fetchone_return=None)
    assert await db_ping(pool) is False


def test_create_pool_returns_unopened_pool() -> None:
    pool = create_pool("postgresql://x:y@localhost/z", min_size=1, max_size=2)
    assert pool.min_size == 1
    assert pool.max_size == 2


@pytest.mark.asyncio
async def test_run_migrations_errors_when_dir_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        await run_migrations(MagicMock(), missing)


@pytest.mark.asyncio
async def test_run_migrations_filters_rollback_files(tmp_path: Path) -> None:
    (tmp_path / "001_initial.sql").write_text("-- noop")
    (tmp_path / "001_initial_rollback.sql").write_text("-- noop")

    pool = _make_pool_mock()
    applied = await run_migrations(pool, tmp_path)
    assert applied == ["001_initial.sql"]


@pytest.mark.asyncio
async def test_run_migrations_applies_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "002_second.sql").write_text("-- noop")
    (tmp_path / "001_first.sql").write_text("-- noop")
    (tmp_path / "010_tenth.sql").write_text("-- noop")

    pool = _make_pool_mock()
    applied = await run_migrations(pool, tmp_path)
    assert applied == ["001_first.sql", "002_second.sql", "010_tenth.sql"]


# ---------------------------------------------------------------------------
# Real-PG integration tests (require pytest-postgresql binary).
# ---------------------------------------------------------------------------

import shutil as _shutil  # noqa: E402

_HAS_PG_BINARY = _shutil.which("pg_config") is not None

if _HAS_PG_BINARY:
    from pytest_postgresql import factories as pg_factories  # noqa: E402

    postgresql_proc = pg_factories.postgresql_proc(port=None)
    postgresql = pg_factories.postgresql("postgresql_proc")


_SKIP_NO_PG = pytest.mark.skipif(
    not _HAS_PG_BINARY, reason="local PostgreSQL (pg_config) not available"
)


def _dsn_from_postgresql(pg) -> str:
    info = pg.info
    return f"postgresql://{info.user}:{info.password or ''}@{info.host}:{info.port}/{info.dbname}"


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_migrations_apply_creates_all_tables(postgresql) -> None:  # noqa: ANN001
    dsn = _dsn_from_postgresql(postgresql)
    pool = create_pool(dsn, min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' ORDER BY table_name"
                )
                rows = await cur.fetchall()
        tables = {r[0] for r in rows}
        assert {
            "sda_state",
            "hypotheses",
            "audit_log",
            "creative_patterns",
            "ask_queue",
            "watchdog_heartbeat",
        }.issubset(tables)
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_migrations_are_idempotent(postgresql) -> None:  # noqa: ANN001
    dsn = _dsn_from_postgresql(postgresql)
    pool = create_pool(dsn, min_size=1, max_size=2)
    await pool.open()
    try:
        first = await run_migrations(pool, MIGRATIONS_DIR)
        second = await run_migrations(pool, MIGRATIONS_DIR)
        assert first == second  # same files applied, second run must not raise
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_hypothesis_attribution_check_constraint_blocks_null(
    postgresql,  # noqa: ANN001
) -> None:
    dsn = _dsn_from_postgresql(postgresql)
    pool = create_pool(dsn, min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        import psycopg.errors  # noqa: PLC0415

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                with pytest.raises(psycopg.errors.CheckViolation):
                    await cur.execute(
                        """
                        INSERT INTO hypotheses (
                            id, agent, hypothesis_type, signals, hypothesis,
                            reasoning, actions, expected_outcome, budget_cap_rub,
                            autonomy_level, risk_score, metrics_before
                        ) VALUES (
                            'h1', 'smoke', 'ad', '[]'::jsonb, 'x', 'y',
                            '[]'::jsonb, 'z', 500, 'AUTO', 0.1, '{}'::jsonb
                        )
                        """
                    )
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_hypothesis_attribution_allows_account_level(
    postgresql,  # noqa: ANN001
) -> None:
    dsn = _dsn_from_postgresql(postgresql)
    pool = create_pool(dsn, min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO hypotheses (
                        id, agent, hypothesis_type, signals, hypothesis,
                        reasoning, actions, expected_outcome, budget_cap_rub,
                        autonomy_level, risk_score, metrics_before
                    ) VALUES (
                        'h2', 'smoke', 'account_level', '[]'::jsonb, 'x', 'y',
                        '[]'::jsonb, 'z', 500, 'AUTO', 0.1, '{}'::jsonb
                    )
                    """
                )
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_rollback_drops_all_tables(postgresql) -> None:  # noqa: ANN001
    dsn = _dsn_from_postgresql(postgresql)
    pool = create_pool(dsn, min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        rollback_sql = (MIGRATIONS_DIR / "001_initial_rollback.sql").read_text()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(rollback_sql)
                await cur.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                )
                rows = await cur.fetchall()
        remaining = {r[0] for r in rows}
        for name in (
            "sda_state",
            "hypotheses",
            "audit_log",
            "creative_patterns",
            "ask_queue",
            "watchdog_heartbeat",
        ):
            assert name not in remaining
    finally:
        await pool.close()
