"""Tests for agent_runtime.tools.registry.build_registry."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.tools.kill_switches import ALL_GUARDS
from agent_runtime.tools.registry import build_registry


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
    )


def _fake_direct() -> SimpleNamespace:
    return SimpleNamespace(
        get_campaigns=AsyncMock(return_value=[]),
        get_adgroups=AsyncMock(return_value=[]),
        get_keywords=AsyncMock(return_value=[]),
        set_bid=AsyncMock(return_value={}),
        add_negatives=AsyncMock(return_value={}),
        pause_group=AsyncMock(return_value={}),
        resume_group=AsyncMock(return_value={}),
        pause_campaign=AsyncMock(return_value={}),
        resume_campaign=AsyncMock(return_value={}),
    )


def test_build_registry_all_tools_present() -> None:
    registry = build_registry(_settings(), direct=_fake_direct(), http_client=MagicMock())
    names = {tool.name for tool in registry}

    # Direct read + write
    for name in (
        "direct.get_campaigns",
        "direct.get_adgroups",
        "direct.get_keywords",
        "direct.set_bid",
        "direct.add_negatives",
        "direct.pause_group",
        "direct.resume_group",
        "direct.pause_campaign",
        "direct.resume_campaign",
    ):
        assert name in names, f"missing direct tool: {name}"

    # Bitrix + Metrika + Telegram
    for name in (
        "bitrix.get_lead_list",
        "bitrix.get_deal_list",
        "metrika.get_bounce_by_campaign",
        "metrika.get_conversions",
        "telegram.send_message",
    ):
        assert name in names, f"missing tool: {name}"

    # 7 kill-switches present as danger-tier placeholders
    for guard_cls in ALL_GUARDS:
        assert f"killswitch.{guard_cls.name}" in names


def test_tier_assignments() -> None:
    registry = build_registry(_settings(), direct=_fake_direct(), http_client=MagicMock())
    read = {t.name for t in registry.filter(tiers=["read"])}
    write = {t.name for t in registry.filter(tiers=["write"])}
    danger = {t.name for t in registry.filter(tiers=["danger"])}

    # Reads
    assert "direct.get_campaigns" in read
    assert "bitrix.get_lead_list" in read
    assert "metrika.get_bounce_by_campaign" in read

    # Writes
    assert "direct.set_bid" in write
    assert "direct.pause_campaign" in write
    assert "telegram.send_message" in write

    # Danger = exactly 7 kill-switches
    assert len(danger) == 7
    assert danger == {f"killswitch.{g.name}" for g in ALL_GUARDS}


def test_for_api_filters_by_tier_and_omits_handler() -> None:
    registry = build_registry(_settings(), direct=_fake_direct(), http_client=MagicMock())
    api = registry.for_api(tiers=["read"])
    assert api, "read-tier tools should appear in for_api output"
    for entry in api:
        assert "name" in entry
        assert "description" in entry
        assert "input_schema" in entry
        assert "handler" not in entry  # API projection strips callables


def test_no_for_tier_method_on_registry() -> None:
    """Regression: agents-core registry exposes filter/for_api, never for_tier."""
    registry = build_registry(_settings(), direct=_fake_direct(), http_client=MagicMock())
    assert not hasattr(registry, "for_tier")


def test_write_tools_require_verify_flag() -> None:
    registry = build_registry(_settings(), direct=_fake_direct(), http_client=MagicMock())
    for tool in registry.filter(tiers=["write"]):
        if tool.name.startswith("direct."):
            # Every direct.* mutation should advertise requires_verify so the
            # caller cannot forget to pair with verify_*.
            assert tool.requires_verify is True, f"{tool.name} missing requires_verify"


@pytest.mark.asyncio
async def test_read_tool_handler_invokes_underlying() -> None:
    direct = _fake_direct()
    direct.get_campaigns = AsyncMock(return_value=[{"Id": 123}])
    registry = build_registry(_settings(), direct=direct, http_client=MagicMock())
    result = await registry.call("direct.get_campaigns", ids=[123])
    assert result == [{"Id": 123}]
    direct.get_campaigns.assert_awaited_once_with([123])


@pytest.mark.asyncio
async def test_killswitch_handler_returns_metadata() -> None:
    registry = build_registry(_settings(), direct=_fake_direct(), http_client=MagicMock())
    # Take any killswitch tool and confirm its handler signals "not callable".
    names = [t.name for t in registry.filter(tiers=["danger"])]
    result = await registry.call(names[0])
    assert "guard" in result
    assert "metadata" in result["message"]


@pytest.mark.asyncio
async def test_every_direct_handler_delegates() -> None:
    direct = _fake_direct()
    registry = build_registry(_settings(), direct=direct, http_client=MagicMock())

    await registry.call("direct.get_campaigns", ids=[1])
    direct.get_campaigns.assert_awaited_once_with([1])

    await registry.call("direct.get_adgroups", campaign_id=5, ids=[9])
    direct.get_adgroups.assert_awaited_once_with(campaign_id=5, ids=[9])

    await registry.call("direct.get_keywords", ad_group_ids=[7])
    direct.get_keywords.assert_awaited_once_with([7])

    await registry.call("direct.set_bid", keyword_id=11, bid_rub=50)
    direct.set_bid.assert_awaited_once_with(11, bid_rub=50, context_bid_rub=None)

    await registry.call("direct.add_negatives", campaign_id=12, phrases=["бесплатн"])
    direct.add_negatives.assert_awaited_once_with(12, ["бесплатн"])

    await registry.call("direct.pause_group", ad_group_id=13)
    direct.pause_group.assert_awaited_once_with(13)

    await registry.call("direct.resume_group", ad_group_id=13)
    direct.resume_group.assert_awaited_once_with(13)

    await registry.call("direct.pause_campaign", campaign_id=14)
    direct.pause_campaign.assert_awaited_once_with(14)

    await registry.call("direct.resume_campaign", campaign_id=14)
    direct.resume_campaign.assert_awaited_once_with(14)


@pytest.mark.asyncio
async def test_bitrix_metrika_telegram_handlers_delegate() -> None:
    http = MagicMock()
    registry = build_registry(_settings(), direct=_fake_direct(), http_client=http)
    with (
        patch(
            "agent_runtime.tools.bitrix.get_lead_list",
            new=AsyncMock(return_value=[]),
        ) as b_leads,
        patch(
            "agent_runtime.tools.bitrix.get_deal_list",
            new=AsyncMock(return_value=[]),
        ) as b_deals,
        patch(
            "agent_runtime.tools.metrika.get_bounce_by_campaign",
            new=AsyncMock(return_value={}),
        ) as m_bounce,
        patch(
            "agent_runtime.tools.metrika.get_conversions",
            new=AsyncMock(return_value={}),
        ) as m_conv,
        patch(
            "agent_runtime.tools.telegram.send_message",
            new=AsyncMock(return_value=1),
        ) as tg_send,
    ):
        await registry.call("bitrix.get_lead_list", filter={"A": 1}, max_total=5)
        await registry.call("bitrix.get_deal_list", filter={"B": 2}, max_total=10)
        await registry.call("metrika.get_bounce_by_campaign", date1="d1", date2="d2")
        await registry.call("metrika.get_conversions", goal_ids=[1], date1="d1", date2="d2")
        await registry.call("telegram.send_message", text="ping")
    b_leads.assert_awaited_once()
    b_deals.assert_awaited_once()
    m_bounce.assert_awaited_once()
    m_conv.assert_awaited_once()
    tg_send.assert_awaited_once()
