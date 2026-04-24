"""Integration: ensure ``insert_audit_log`` actually strips PII before write.

Uses ``pytest-postgresql`` for an ephemeral PG — skipped locally when the
``pg_config`` binary is absent. CI installs ``postgresql-contrib`` so these
run there.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent_runtime.db import create_pool, insert_audit_log, run_migrations

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"

_HAS_PG = shutil.which("pg_config") is not None

_SKIP_NO_PG = pytest.mark.skipif(not _HAS_PG, reason="local PostgreSQL (pg_config) not available")

if _HAS_PG:
    from pytest_postgresql import factories as pg_factories

    postgresql_proc = pg_factories.postgresql_proc(port=None)
    postgresql = pg_factories.postgresql("postgresql_proc")


_BITRIX_LEAD_SAMPLE = {
    "ID": "12345",
    "TITLE": "New lead from landing",
    "NAME": "Иван",
    "LAST_NAME": "Иванов",
    "PHONE": [{"VALUE": "+79991234567", "VALUE_TYPE": "WORK"}],
    "EMAIL": [{"VALUE": "ivan@example.com", "VALUE_TYPE": "WORK"}],
    "SOURCE_DESCRIPTION": "Иван Иванов +79991234567 просит перезвонить",
    "STATUS_ID": "NEW",
    "CURRENCY_ID": "RUB",
    "OPPORTUNITY": "50000",
}


def _dsn(pg) -> str:  # noqa: ANN001
    i = pg.info
    return f"postgresql://{i.user}:{i.password or ''}@{i.host}:{i.port}/{i.dbname}"


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_bitrix_lead_sanitized_on_insert(postgresql) -> None:  # noqa: ANN001
    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        audit_id = await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="shadow",
            tool_name="bitrix.get_lead",
            tool_input={"lead_id": 12345},
            tool_output=_BITRIX_LEAD_SAMPLE,
            is_mutation=False,
        )
        assert audit_id > 0

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT tool_input, tool_output FROM audit_log WHERE id = %s",
                    (audit_id,),
                )
                row = await cur.fetchone()
        assert row is not None
        rendered = json.dumps(row, ensure_ascii=False, default=str)
        # None of the raw PII may appear anywhere in the stored row
        assert "+79991234567" not in rendered
        assert "Иван" not in rendered
        assert "Иванов" not in rendered
        assert "ivan@example.com" not in rendered
        assert "просит перезвонить" not in rendered
        # Non-PII fields preserved
        assert "12345" in rendered
        assert "RUB" in rendered
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_audit_retention_deletes_old_rows(postgresql) -> None:  # noqa: ANN001
    from agent_runtime.jobs.audit_retention import RETENTION_DAYS, run

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        # Seed: one fresh row, one 100-day-old row
        fresh_id = await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="shadow",
            tool_name="test.fresh",
            tool_input={},
            tool_output=None,
            is_mutation=False,
        )
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO audit_log (ts, trust_level, tool_name, tool_input, is_mutation) "
                    "VALUES (now() - interval '100 days', 'shadow', 'test.stale', "
                    "'{}'::jsonb, false) RETURNING id"
                )
                stale_row = await cur.fetchone()
        stale_id = int(stale_row[0])  # type: ignore[index]

        # Dry run first — no deletion, count=1
        dry = await run(pool, dry_run=True)
        assert dry == {"deleted": 1, "dry_run": True}

        # Real run — 1 deleted
        real = await run(pool, dry_run=False)
        assert real == {"deleted": 1, "dry_run": False}

        # Verify the fresh row survived, stale row gone
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT id FROM audit_log ORDER BY id")
                ids = [r[0] for r in await cur.fetchall()]
        assert fresh_id in ids
        assert stale_id not in ids
        _ = RETENTION_DAYS  # assert symbol is exported
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_audit_retention_noop_on_empty_table(postgresql) -> None:  # noqa: ANN001
    from agent_runtime.jobs.audit_retention import run

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        result = await run(pool, dry_run=False)
        assert result == {"deleted": 0, "dry_run": False}
    finally:
        await pool.close()
