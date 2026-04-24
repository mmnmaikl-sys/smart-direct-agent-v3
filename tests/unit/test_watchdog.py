"""Unit tests for agent_runtime.jobs.watchdog (Task 13).

Tests use mocked pool + mocked DirectAPI + a ``SimpleNamespace`` stub for
HTTP + Telegram so nothing hits the wire. The critical flow tested is the
three trust levels: shadow (NOTIFY only), assisted (ASK queue insert),
autonomous (suspend all PROTECTED campaigns).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import watchdog
from agent_runtime.jobs.watchdog import (
    HEARTBEAT_STALE_MINUTES,
    StaleService,
    get_stale_services,
    heartbeat,
    run,
)

_PROTECTED = [708978456, 708978457, 708978458, 709307228]


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=_PROTECTED,
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )


def _mock_pool(
    fetchone_sequence=None,
    fetchall_sequence=None,
) -> tuple[MagicMock, MagicMock]:
    one_iter = iter(fetchone_sequence or [])
    all_iter = iter(fetchall_sequence or [])

    async def _fetchone():
        try:
            return next(one_iter)
        except StopIteration:
            # Default covers audit_log INSERT ... RETURNING id and ask_queue
            return (1,)

    async def _fetchall():
        try:
            return next(all_iter)
        except StopIteration:
            return []

    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(side_effect=_fetchone)
    cursor.fetchall = AsyncMock(side_effect=_fetchall)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, cursor


def _direct_stub() -> SimpleNamespace:
    return SimpleNamespace(
        pause_campaign=AsyncMock(return_value={}),
        verify_campaign_paused=AsyncMock(return_value=True),
    )


# ---- heartbeat & get_stale_services ---------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_upserts_row() -> None:
    pool, cursor = _mock_pool()
    await heartbeat(pool, "budget_guard")
    call = cursor.execute.await_args_list[0]
    assert "INSERT INTO watchdog_heartbeat" in call.args[0]
    assert "ON CONFLICT (service) DO UPDATE" in call.args[0]
    assert call.args[1] == ("budget_guard",)


@pytest.mark.asyncio
async def test_get_stale_services_returns_empty() -> None:
    pool, _ = _mock_pool(fetchall_sequence=[[]])
    assert await get_stale_services(pool) == []


@pytest.mark.asyncio
async def test_get_stale_services_parses_rows() -> None:
    ts = datetime.now(UTC) - timedelta(hours=2)
    pool, _ = _mock_pool(fetchall_sequence=[[("budget_guard", ts, 120.0)]])
    result = await get_stale_services(pool, threshold_minutes=90)
    assert len(result) == 1
    assert result[0].service == "budget_guard"
    assert result[0].minutes_stale == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_get_stale_services_sql_excludes_watchdog() -> None:
    pool, cursor = _mock_pool(fetchall_sequence=[[]])
    await get_stale_services(pool)
    sql = cursor.execute.await_args_list[0].args[0]
    assert "service <> 'watchdog'" in sql


# ---- run ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_clean_case_returns_ok() -> None:
    pool, cursor = _mock_pool(fetchall_sequence=[[]])
    direct = _direct_stub()
    send = AsyncMock(return_value=1)
    with patch("agent_runtime.tools.telegram.send_message", new=send):
        result = await run(
            pool,
            direct=direct,
            http_client=MagicMock(),
            settings=_settings(),
        )
    assert result["status"] == "ok"
    assert result["stale_services"] == []
    assert result["suspended_campaigns"] == []
    send.assert_not_awaited()
    direct.pause_campaign.assert_not_awaited()
    # Self-beat executed (first INSERT on watchdog_heartbeat)
    first = cursor.execute.await_args_list[0]
    assert "watchdog_heartbeat" in first.args[0]
    assert first.args[1] == ("watchdog",)


@pytest.mark.asyncio
async def test_run_self_beat_before_stale_check() -> None:
    ts = datetime.now(UTC) - timedelta(hours=2)
    pool, cursor = _mock_pool(
        fetchone_sequence=[("shadow",)],  # trust_level lookup
        fetchall_sequence=[[("budget_guard", ts, 120.0)]],
    )
    with patch("agent_runtime.tools.telegram.send_message", new=AsyncMock()):
        await run(pool, direct=_direct_stub(), http_client=MagicMock(), settings=_settings())
    sqls = [call.args[0] for call in cursor.execute.await_args_list]
    heartbeat_idx = next(i for i, s in enumerate(sqls) if "INSERT INTO watchdog_heartbeat" in s)
    stale_idx = next(i for i, s in enumerate(sqls) if "EXTRACT(EPOCH" in s)
    assert heartbeat_idx < stale_idx


@pytest.mark.asyncio
async def test_run_shadow_notifies_without_mutation() -> None:
    ts = datetime.now(UTC) - timedelta(hours=2)
    pool, _ = _mock_pool(
        fetchone_sequence=[("shadow",)],
        fetchall_sequence=[[("budget_guard", ts, 120.0)]],
    )
    direct = _direct_stub()
    send = AsyncMock(return_value=1)
    with patch("agent_runtime.tools.telegram.send_message", new=send):
        result = await run(pool, direct=direct, http_client=MagicMock(), settings=_settings())
    assert result["status"] == "stale_detected"
    assert result["suspended_campaigns"] == []
    send.assert_awaited_once()
    direct.pause_campaign.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_assisted_creates_ask_queue_row() -> None:
    ts = datetime.now(UTC) - timedelta(hours=2)
    # Fetchone: trust_level, then ask_queue RETURNING id
    pool, cursor = _mock_pool(
        fetchone_sequence=[("assisted",), (77,)],
        fetchall_sequence=[[("budget_guard", ts, 120.0)]],
    )
    direct = _direct_stub()
    send = AsyncMock(return_value=1)
    with patch("agent_runtime.tools.telegram.send_message", new=send):
        result = await run(pool, direct=direct, http_client=MagicMock(), settings=_settings())
    assert result["status"] == "stale_detected"
    assert result["ask_id"] == 77
    direct.pause_campaign.assert_not_awaited()
    # Exact ask_queue INSERT SQL was executed
    sqls = [call.args[0] for call in cursor.execute.await_args_list]
    assert any("INSERT INTO ask_queue" in s for s in sqls)
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_autonomous_suspends_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watchdog, "_sleep_for_verify", AsyncMock())
    ts = datetime.now(UTC) - timedelta(hours=2)
    pool, _ = _mock_pool(
        fetchone_sequence=[("autonomous",)],
        fetchall_sequence=[[("budget_guard", ts, 120.0)]],
    )
    direct = _direct_stub()
    send = AsyncMock(return_value=1)
    with patch("agent_runtime.tools.telegram.send_message", new=send):
        result = await run(pool, direct=direct, http_client=MagicMock(), settings=_settings())
    assert result["status"] == "stale_detected"
    assert set(result["suspended_campaigns"]) == set(_PROTECTED)
    assert direct.pause_campaign.await_count == len(_PROTECTED)
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_dry_run_skips_suspend_and_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watchdog, "_sleep_for_verify", AsyncMock())
    ts = datetime.now(UTC) - timedelta(hours=2)
    pool, cursor = _mock_pool(
        fetchone_sequence=[("autonomous",)],
        fetchall_sequence=[[("budget_guard", ts, 120.0)]],
    )
    direct = _direct_stub()
    send = AsyncMock(return_value=1)
    with patch("agent_runtime.tools.telegram.send_message", new=send):
        result = await run(
            pool,
            direct=direct,
            http_client=MagicMock(),
            settings=_settings(),
            dry_run=True,
        )
    # Telegram alert still fires; Direct mutations and ask_queue inserts don't
    send.assert_awaited_once()
    direct.pause_campaign.assert_not_awaited()
    assert result["suspended_campaigns"] == []


@pytest.mark.asyncio
async def test_run_autonomous_best_effort_on_pause_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watchdog, "_sleep_for_verify", AsyncMock())
    ts = datetime.now(UTC) - timedelta(hours=2)
    pool, _ = _mock_pool(
        fetchone_sequence=[("autonomous",)],
        fetchall_sequence=[[("budget_guard", ts, 120.0)]],
    )
    # First pause succeeds, second raises, remaining succeed.
    direct = SimpleNamespace(
        pause_campaign=AsyncMock(
            side_effect=[
                {},
                RuntimeError("Direct flaky"),
                {},
                {},
            ]
        ),
        verify_campaign_paused=AsyncMock(return_value=True),
    )
    with patch("agent_runtime.tools.telegram.send_message", new=AsyncMock()):
        result = await run(pool, direct=direct, http_client=MagicMock(), settings=_settings())
    # Three out of four suspended — the one that raised is skipped.
    assert len(result["suspended_campaigns"]) == 3


@pytest.mark.asyncio
async def test_run_autonomous_verify_failure_skips_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watchdog, "_sleep_for_verify", AsyncMock())
    ts = datetime.now(UTC) - timedelta(hours=2)
    pool, _ = _mock_pool(
        fetchone_sequence=[("autonomous",)],
        fetchall_sequence=[[("budget_guard", ts, 120.0)]],
    )
    direct = SimpleNamespace(
        pause_campaign=AsyncMock(return_value={}),
        verify_campaign_paused=AsyncMock(return_value=False),
    )
    with patch("agent_runtime.tools.telegram.send_message", new=AsyncMock()):
        result = await run(pool, direct=direct, http_client=MagicMock(), settings=_settings())
    # Every pause call went through but none verified → no campaigns counted.
    assert result["suspended_campaigns"] == []


@pytest.mark.asyncio
async def test_run_forbidden_lock_only_notifies() -> None:
    ts = datetime.now(UTC) - timedelta(hours=2)
    pool, _ = _mock_pool(
        fetchone_sequence=[("FORBIDDEN_LOCK",)],
        fetchall_sequence=[[("budget_guard", ts, 120.0)]],
    )
    direct = _direct_stub()
    send = AsyncMock(return_value=1)
    with patch("agent_runtime.tools.telegram.send_message", new=send):
        result = await run(pool, direct=direct, http_client=MagicMock(), settings=_settings())
    assert result["status"] == "stale_detected"
    assert result["suspended_campaigns"] == []
    direct.pause_campaign.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_exception_fails_loud_and_reraises() -> None:
    pool, _ = _mock_pool()

    async def broken_get_stale(*_args: Any, **_kwargs: Any) -> list[StaleService]:
        raise RuntimeError("db gone")

    send = AsyncMock(return_value=1)
    with (
        patch("agent_runtime.tools.telegram.send_message", new=send),
        patch.object(watchdog, "get_stale_services", new=broken_get_stale),
        pytest.raises(RuntimeError, match="db gone"),
    ):
        await run(
            pool,
            direct=_direct_stub(),
            http_client=MagicMock(),
            settings=_settings(),
        )
    # Fail-loud Telegram alert was attempted before the re-raise.
    send.assert_awaited_once()
    assert "CRASHED" in send.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_run_registered_in_job_registry() -> None:
    from agent_runtime.jobs import JOB_REGISTRY

    assert "watchdog" in JOB_REGISTRY
    assert JOB_REGISTRY["watchdog"] is watchdog.run


@pytest.mark.asyncio
async def test_run_result_model_structure() -> None:
    pool, _ = _mock_pool(fetchall_sequence=[[]])
    with patch("agent_runtime.tools.telegram.send_message", new=AsyncMock()):
        result = await run(
            pool,
            direct=_direct_stub(),
            http_client=MagicMock(),
            settings=_settings(),
        )
    for key in ("status", "stale_services", "suspended_campaigns", "alerted_at"):
        assert key in result
    assert result["status"] in {"ok", "stale_detected", "error"}


def test_heartbeat_stale_minutes_constant() -> None:
    assert HEARTBEAT_STALE_MINUTES == 90
