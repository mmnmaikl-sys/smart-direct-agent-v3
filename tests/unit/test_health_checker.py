"""Unit tests for ``agent_runtime.jobs.health_checker`` (Task 26).

Strategy: stub the three Metrika helpers at import site
(``metrika_tools.get_bounce_by_campaign`` / ``get_stats``) and the Telegram
sender, so no real I/O happens. The SUT runs the 3 checks in parallel via
``asyncio.gather(return_exceptions=True)``, so per-check isolation is
exercised by having one of the stubs raise.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import health_checker
from agent_runtime.jobs.health_checker import (
    BOUNCE_RED_THRESHOLD,
    BOUNCE_YELLOW_THRESHOLD,
    _bounce_flag,
    _render_summary,
    run,
)

_PROTECTED = [708978456, 708978457]


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
        METRIKA_COUNTER_ID=107734488,
        METRIKA_OAUTH_TOKEN="metrika-token",  # noqa: S106 (test-only secret)
    )


def _mock_pool(state_row: dict[str, Any] | None = None) -> MagicMock:
    one_iter = iter([(state_row,)] if state_row is not None else [None])

    async def _fetchone():
        try:
            return next(one_iter)
        except StopIteration:
            return None

    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(side_effect=_fetchone)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool


def _device_stats_payload(shares: dict[str, int]) -> dict[str, Any]:
    """Mimic the shape that metrika.get_stats returns for deviceCategory."""
    return {
        "data": [
            {"dimensions": [{"name": name}], "metrics": [visits]} for name, visits in shares.items()
        ]
    }


def _landing_stats_payload(rows: list[tuple[str, int, float]]) -> dict[str, Any]:
    return {
        "data": [
            {"dimensions": [{"name": url}], "metrics": [visits, bounce]}
            for url, visits, bounce in rows
        ]
    }


# ---------------------------------------------------------------- flag helper


def test_bounce_flag_thresholds() -> None:
    assert _bounce_flag(BOUNCE_RED_THRESHOLD + 1) == "🔴"
    assert _bounce_flag(BOUNCE_RED_THRESHOLD) == "🔴"
    assert _bounce_flag(BOUNCE_YELLOW_THRESHOLD) == "🟡"
    assert _bounce_flag(BOUNCE_RED_THRESHOLD - 1) == "🟡"
    assert _bounce_flag(BOUNCE_YELLOW_THRESHOLD - 1) == "🟢"
    assert _bounce_flag(0.0) == "🟢"


# ---------------------------------------------------------------- happy paths


@pytest.mark.asyncio
async def test_high_bounce_triggers_red_flag() -> None:
    pool = _mock_pool()
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={_PROTECTED[0]: 72.0, _PROTECTED[1]: 45.0}),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value=_device_stats_payload({"desktop": 100})),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=12345),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    assert result["action"] == "sent"
    summary = result["summary"]
    assert "🔴" in summary
    assert "72%" in summary
    # 45 % is below the yellow threshold (<50) → 🟢.
    assert "45%" in summary


@pytest.mark.asyncio
async def test_device_breakdown_in_summary() -> None:
    pool = _mock_pool()
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={}),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value=_device_stats_payload({"mobile": 70, "desktop": 30})),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=777),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    summary = result["summary"]
    assert "mobile" in summary
    assert "desktop" in summary
    assert "70%" in summary
    assert "30%" in summary


@pytest.mark.asyncio
async def test_dry_run_returns_text_no_send() -> None:
    pool = _mock_pool()
    send_stub = AsyncMock(return_value=0)
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={_PROTECTED[0]: 30.0}),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value=_device_stats_payload({"desktop": 1})),
        ),
        patch.object(health_checker.telegram_tools, "send_message", send_stub),
    ):
        result = await run(pool, dry_run=True, http_client=MagicMock(), settings=_settings())
    assert result["dry_run"] is True
    assert result["action"] == "drafted"
    assert "*Health check*" in result["summary"]
    send_stub.assert_not_called()


@pytest.mark.asyncio
async def test_no_data_yesterday_shows_placeholder() -> None:
    pool = _mock_pool()
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={}),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value={"data": []}),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=1),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    assert "нет данных за вчера" in result["summary"]


# ---------------------------------------------------------------- no mutations


@pytest.mark.asyncio
async def test_no_mutations_on_direct_api() -> None:
    """The checker must not touch Direct even when a client is injected."""
    pool = _mock_pool()
    direct = MagicMock()
    direct.update_strategy = AsyncMock()
    direct.pause_campaign = AsyncMock()
    direct.resume_campaign = AsyncMock()
    direct.set_bid = AsyncMock()
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={}),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value={"data": []}),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=1),
        ),
    ):
        await run(
            pool,
            direct=direct,
            http_client=MagicMock(),
            settings=_settings(),
        )
    direct.update_strategy.assert_not_called()
    direct.pause_campaign.assert_not_called()
    direct.resume_campaign.assert_not_called()
    direct.set_bid.assert_not_called()


# ---------------------------------------------------------------- isolation


@pytest.mark.asyncio
async def test_parallel_isolation_bounce_failure() -> None:
    """One failing check must not sink the other two."""
    pool = _mock_pool()
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(side_effect=RuntimeError("metrika 500")),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value=_device_stats_payload({"desktop": 1})),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=1),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    assert result["action"] == "sent"
    assert result["bounce"]["ok"] is False
    assert "metrika 500" in result["bounce"]["error"]
    # Devices still rendered.
    assert result["devices"]["ok"] is True
    assert result["devices"]["rows"][0]["device"] == "desktop"


@pytest.mark.asyncio
async def test_parallel_isolation_devices_failure() -> None:
    pool = _mock_pool()

    async def _boom_stats(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise RuntimeError("devices down")

    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={_PROTECTED[0]: 40.0}),
        ),
        patch.object(health_checker.metrika_tools, "get_stats", _boom_stats),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=1),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    assert result["bounce"]["ok"] is True
    assert result["devices"]["ok"] is False
    assert result["landings"]["ok"] is False
    # Bounce section still shows the one campaign row.
    assert result["bounce"]["rows"][0]["campaign_id"] == _PROTECTED[0]


@pytest.mark.asyncio
async def test_telegram_failure_does_not_raise() -> None:
    """If Telegram fails on live run, summary is still returned with error marker."""
    pool = _mock_pool()
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={}),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value={"data": []}),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(side_effect=RuntimeError("tg down")),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    assert result["action"] == "drafted"  # no message_id returned
    assert result["telegram_message_id"] is None
    assert result["telegram_error"] is not None
    assert "tg down" in result["telegram_error"]


# ---------------------------------------------------------------- degraded


@pytest.mark.asyncio
async def test_degraded_noop_when_http_missing() -> None:
    pool = _mock_pool()
    result = await run(pool, http_client=None, settings=_settings())
    assert result["action"] == "degraded_noop"


@pytest.mark.asyncio
async def test_degraded_noop_when_settings_missing() -> None:
    pool = _mock_pool()
    result = await run(pool, http_client=MagicMock(), settings=None)
    assert result["action"] == "degraded_noop"


# ---------------------------------------------------------------- summary


def test_render_summary_caps_length() -> None:
    from agent_runtime.jobs.health_checker import (
        BounceSection,
        DeviceSection,
        LandingSection,
    )

    # A huge bounce section that would blow past TELEGRAM_MAX_LEN.
    rows = [(i, 90.0) for i in range(10_000)]
    summary = _render_summary(
        date1="2026-04-23",
        bounce=BounceSection(ok=True, rows=rows),
        device=DeviceSection(ok=True, rows=[]),
        landing=LandingSection(ok=True, rows=[]),
    )
    assert len(summary) <= health_checker.TELEGRAM_MAX_LEN
    assert "truncated" in summary


@pytest.mark.asyncio
async def test_landings_section_rendered_in_summary() -> None:
    """Top-landings get URL + bounce-flag + visits on screen."""
    pool = _mock_pool()
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={}),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(
                side_effect=[
                    # devices call
                    _device_stats_payload({"desktop": 1}),
                    # top-landings call
                    _landing_stats_payload(
                        [
                            ("https://24bankrotsttvo.ru/pages/ad/a.html", 200, 75.0),
                            ("https://24bankrotsttvo.ru/pages/ad/b.html", 100, 30.0),
                        ]
                    ),
                ]
            ),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=1),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    assert result["landings"]["ok"] is True
    assert len(result["landings"]["rows"]) == 2
    summary = result["summary"]
    assert "/pages/ad/a.html" in summary
    assert "200 visits" in summary
    assert "🔴" in summary  # 75 % bounce


@pytest.mark.asyncio
async def test_active_campaigns_from_json_string_decoded() -> None:
    """Some drivers hand the JSONB column back as a string; accept both shapes."""
    pool = _mock_pool(state_row='{"status":"auto_pilot","active_campaigns":[555]}')
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(return_value={555: 40.0, 999: 88.0}),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value={"data": []}),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=1),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    cids = {row["campaign_id"] for row in result["bounce"]["rows"]}
    assert cids == {555}


@pytest.mark.asyncio
async def test_run_impl_crash_captured() -> None:
    """Unexpected crash inside _run_impl surfaces action=error, never raises."""
    pool = _mock_pool()

    async def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("unresolvable")

    with patch.object(health_checker, "_run_impl", _boom):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    assert result["action"] == "error"
    assert "unresolvable" in result["error"]


@pytest.mark.asyncio
async def test_active_campaigns_from_gate_state_used() -> None:
    """When sda_state has active_campaigns list, filter bounce rows to it."""
    custom_ids = [111, 222]
    pool = _mock_pool(state_row={"status": "auto_pilot", "active_campaigns": custom_ids})
    with (
        patch.object(
            health_checker.metrika_tools,
            "get_bounce_by_campaign",
            AsyncMock(
                return_value={
                    111: 50.0,
                    222: 65.0,
                    999: 80.0,  # not in the whitelist → must be filtered out
                }
            ),
        ),
        patch.object(
            health_checker.metrika_tools,
            "get_stats",
            AsyncMock(return_value={"data": []}),
        ),
        patch.object(
            health_checker.telegram_tools,
            "send_message",
            AsyncMock(return_value=1),
        ),
    ):
        result = await run(pool, http_client=MagicMock(), settings=_settings())
    cids = {row["campaign_id"] for row in result["bounce"]["rows"]}
    assert cids == {111, 222}
    assert 999 not in cids
