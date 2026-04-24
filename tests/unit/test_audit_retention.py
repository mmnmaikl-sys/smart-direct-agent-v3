"""Unit tests for audit_retention job — mock pool, no live PG."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.jobs.audit_retention import RETENTION_DAYS, run


def _make_pool(rowcount: int = 0, fetchone_return=(0,)):
    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=fetchone_return)
    cursor.rowcount = rowcount
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, cursor


@pytest.mark.asyncio
async def test_dry_run_returns_count_without_delete() -> None:
    pool, cursor = _make_pool(fetchone_return=(7,))
    result = await run(pool, dry_run=True)
    assert result == {"deleted": 7, "dry_run": True}
    # Only SELECT count executed, no DELETE
    call_args = cursor.execute.await_args_list[0].args[0]
    assert "SELECT count(*)" in call_args
    assert "DELETE" not in call_args


@pytest.mark.asyncio
async def test_real_run_returns_rowcount_of_delete() -> None:
    pool, cursor = _make_pool(rowcount=12)
    result = await run(pool, dry_run=False)
    assert result == {"deleted": 12, "dry_run": False}
    call_args = cursor.execute.await_args_list[0].args[0]
    assert "DELETE FROM audit_log" in call_args


@pytest.mark.asyncio
async def test_uses_decision_13_retention_window() -> None:
    pool, cursor = _make_pool()
    await run(pool, dry_run=False)
    call = cursor.execute.await_args_list[0]
    sql = call.args[0]
    params = call.args[1]
    assert "make_interval(days => %s)" in sql
    assert params == (RETENTION_DAYS,)
    assert RETENTION_DAYS == 90  # Decision 13 value
