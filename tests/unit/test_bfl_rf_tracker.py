"""Unit tests for agent_runtime.tools.bfl_rf_tracker (Task 20).

Covers:

* TSV parsing: aggregates Impressions / Clicks / Cost / Conversions across
  multiple rows, ignores Total row and header noise, tolerates missing
  Conversions column.
* 3-layer isolation: if one ``asyncio.gather`` branch raises, the other
  layers still return sane dicts and the failing one is reported as
  ``{"error": "..."}``.
* Economics safe-divide: zero leads / zero won → 0 cpa_lead / cpa_won.
* Metrika Protocol stub support: None client → ``error='no_client'``;
  stub client returns a normalised shape.
* ``collect`` end-to-end: direct + metrika + bitrix layers wired together.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.tools import bfl_rf_tracker


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )


def _make_tsv(rows: list[dict[str, int]], include_conversions: bool = True) -> str:
    """Build a CAMPAIGN_PERFORMANCE_REPORT-style TSV.

    ``Cost`` values are in rubles; we multiply to micro-rubles for you.
    """
    if include_conversions:
        header = "Date\tCampaignId\tImpressions\tClicks\tCost\tConversions"
    else:
        header = "Date\tCampaignId\tImpressions\tClicks\tCost"
    lines = ["Title line", header]
    for r in rows:
        cells = [
            r.get("date", "2026-04-24"),
            str(r.get("campaign_id", 709307228)),
            str(r["impressions"]),
            str(r["clicks"]),
            str(int(r["cost"] * 1_000_000)),
        ]
        if include_conversions:
            cells.append(str(r.get("conversions", 0)))
        lines.append("\t".join(cells))
    # Trailing Total row — parser must skip.
    if include_conversions:
        lines.append("Total\t1\t9999\t999\t999000000\t9")
    else:
        lines.append("Total\t1\t9999\t999\t999000000")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- TSV parser


def test_parse_direct_report_sums_rows_and_skips_total() -> None:
    tsv = _make_tsv(
        [
            {"impressions": 1000, "clicks": 50, "cost": 500.0, "conversions": 2},
            {"impressions": 2000, "clicks": 80, "cost": 800.0, "conversions": 3},
        ]
    )
    totals = bfl_rf_tracker._parse_direct_report(tsv)
    assert totals["impressions"] == 3000.0
    assert totals["clicks"] == 130.0
    assert totals["cost"] == 1300.0
    assert totals["conversions"] == 5.0


def test_parse_direct_report_empty_tsv_returns_zeros() -> None:
    totals = bfl_rf_tracker._parse_direct_report("")
    assert totals == {"impressions": 0.0, "clicks": 0.0, "cost": 0.0, "conversions": 0.0}


def test_parse_direct_report_missing_header_returns_zeros() -> None:
    totals = bfl_rf_tracker._parse_direct_report("some garbage\nline without columns\n")
    assert totals["impressions"] == 0.0
    assert totals["clicks"] == 0.0


def test_parse_direct_report_without_conversions_column() -> None:
    tsv = _make_tsv([{"impressions": 500, "clicks": 25, "cost": 250.0}], include_conversions=False)
    totals = bfl_rf_tracker._parse_direct_report(tsv)
    assert totals["impressions"] == 500.0
    assert totals["clicks"] == 25.0
    assert totals["cost"] == 250.0
    assert totals["conversions"] == 0.0


# ----------------------------------------------------------- economics derive


def test_economics_safe_divide_on_zero_leads() -> None:
    direct = {"cost": 5000.0}
    bitrix = {"leads": 0, "deals": {"stages": {"won": 0}}}
    eco = bfl_rf_tracker._economics(direct, bitrix)
    assert eco["cpa_lead"] == 0.0
    assert eco["cpa_won"] == 0.0
    assert eco["cost"] == 5000.0
    assert eco["leads"] == 0
    assert eco["won_deals"] == 0


def test_economics_cpa_computed_when_leads_present() -> None:
    direct = {"cost": 6000.0}
    bitrix = {"leads": 3, "deals": {"stages": {"won": 2}}}
    eco = bfl_rf_tracker._economics(direct, bitrix)
    assert eco["cpa_lead"] == pytest.approx(2000.0)
    assert eco["cpa_won"] == pytest.approx(3000.0)


def test_economics_tolerates_missing_keys() -> None:
    eco = bfl_rf_tracker._economics({}, {})
    assert eco["cost"] == 0.0
    assert eco["leads"] == 0
    assert eco["cpa_lead"] == 0.0


# ------------------------------------------------------- layer_or_error guard


def test_layer_or_error_translates_exception() -> None:
    result = bfl_rf_tracker._layer_or_error(RuntimeError("boom"), "direct")
    assert "error" in result
    assert "boom" in result["error"]


def test_layer_or_error_passes_through_dict() -> None:
    result = bfl_rf_tracker._layer_or_error({"visits": 42}, "metrika")
    assert result == {"visits": 42}


def test_layer_or_error_rejects_unexpected_shape() -> None:
    result = bfl_rf_tracker._layer_or_error(["not a dict"], "bitrix")
    assert "error" in result
    assert "unexpected" in result["error"]


# ------------------------------------------------------------ metrika layer


@pytest.mark.asyncio
async def test_metrika_layer_returns_error_when_client_none() -> None:
    result = await bfl_rf_tracker._metrika_layer(None, days=2)
    assert result == {"error": "no_client"}


@pytest.mark.asyncio
async def test_metrika_layer_normalises_shape() -> None:
    class _Stub:
        async def get_visit_stats(self, *, utm_campaign: str, days: int) -> dict[str, Any]:
            assert utm_campaign == "bfl-rf"
            assert days == 2
            return {"visits": 123, "bounce": 44.5, "avg_time": 77.0, "page_depth": 2.1}

    result = await bfl_rf_tracker._metrika_layer(_Stub(), days=2)
    assert result["visits"] == 123
    assert result["bounce_rate"] == 44.5
    assert result["avg_duration_s"] == 77.0


# --------------------------------------------------------------- direct layer


@pytest.mark.asyncio
async def test_direct_layer_builds_snapshot() -> None:
    tsv = _make_tsv(
        [
            {"impressions": 4000, "clicks": 200, "cost": 1500.0, "conversions": 5},
        ]
    )
    direct = SimpleNamespace(
        get_campaigns=AsyncMock(return_value=[{"Id": 709307228, "Name": "RF", "State": "ON"}]),
        get_campaign_stats=AsyncMock(return_value={"tsv": tsv}),
    )
    data = await bfl_rf_tracker._direct_layer(direct, 709307228, "2026-04-23", "2026-04-24")
    assert data["impressions"] == 4000.0
    assert data["clicks"] == 200.0
    assert data["cost"] == 1500.0
    assert data["ctr"] == pytest.approx(5.0)
    assert data["cpc"] == pytest.approx(7.5)
    assert data["state"] == "ON"


# --------------------------------------------------------------- bitrix layer


@pytest.mark.asyncio
async def test_bitrix_layer_counts_leads_and_won() -> None:
    leads = [{"ID": str(i)} for i in range(7)]
    deals = [
        {"ID": "d1", "STAGE_ID": "C45:WON", "OPPORTUNITY": 60000},
        {"ID": "d2", "STAGE_ID": "C45:WON", "OPPORTUNITY": 40000},
    ]
    with (
        patch.object(bfl_rf_tracker.bitrix_tools, "get_lead_list", AsyncMock(return_value=leads)),
        patch.object(bfl_rf_tracker.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals)),
    ):
        result = await bfl_rf_tracker._bitrix_layer(
            SimpleNamespace(), _settings(), "2026-04-23T00:00:00+03:00"
        )
    assert result["leads"] == 7
    assert result["deals"]["stages"]["won"] == 2
    assert result["deals"]["revenue_won"] == 100000.0


@pytest.mark.asyncio
async def test_bitrix_layer_handles_unparseable_opportunity() -> None:
    deals = [{"ID": "d1", "STAGE_ID": "C45:WON", "OPPORTUNITY": "не число"}]
    with (
        patch.object(bfl_rf_tracker.bitrix_tools, "get_lead_list", AsyncMock(return_value=[])),
        patch.object(bfl_rf_tracker.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals)),
    ):
        result = await bfl_rf_tracker._bitrix_layer(
            SimpleNamespace(), _settings(), "2026-04-23T00:00:00+03:00"
        )
    assert result["deals"]["revenue_won"] == 0.0
    assert result["deals"]["stages"]["won"] == 1


# -------------------------------------------------- collect (end-to-end)


@pytest.mark.asyncio
async def test_collect_three_layers_ok() -> None:
    tsv = _make_tsv([{"impressions": 3500, "clicks": 120, "cost": 900.0, "conversions": 1}])
    direct = SimpleNamespace(
        get_campaigns=AsyncMock(return_value=[{"Id": 709307228, "Name": "RF", "State": "ON"}]),
        get_campaign_stats=AsyncMock(return_value={"tsv": tsv}),
    )

    class _MetrikaStub:
        async def get_visit_stats(self, *, utm_campaign: str, days: int) -> dict[str, Any]:
            return {"visits": 300, "bounce": 42.0, "avg_time": 85.0}

    with (
        patch.object(
            bfl_rf_tracker.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=[{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]),
        ),
        patch.object(bfl_rf_tracker.bitrix_tools, "get_deal_list", AsyncMock(return_value=[])),
    ):
        data = await bfl_rf_tracker.collect(
            SimpleNamespace(), direct, _settings(), metrika=_MetrikaStub(), days=2
        )

    assert data["direct"]["clicks"] == 120.0
    assert data["metrika"]["visits"] == 300
    assert data["bitrix"]["leads"] == 3
    assert data["economics"]["cpa_lead"] == pytest.approx(300.0)  # 900 / 3
    assert data["economics"]["cpa_won"] == 0.0
    assert data["days"] == 2


@pytest.mark.asyncio
async def test_collect_isolates_failing_layer() -> None:
    """Direct down → direct layer is {"error": ...}, others still populated."""
    direct = SimpleNamespace(
        get_campaigns=AsyncMock(side_effect=RuntimeError("direct 503")),
        get_campaign_stats=AsyncMock(return_value={"tsv": ""}),
    )

    class _MetrikaStub:
        async def get_visit_stats(self, *, utm_campaign: str, days: int) -> dict[str, Any]:
            return {"visits": 10, "bounce": 0, "avg_time": 0}

    with (
        patch.object(
            bfl_rf_tracker.bitrix_tools, "get_lead_list", AsyncMock(return_value=[{"ID": "1"}])
        ),
        patch.object(bfl_rf_tracker.bitrix_tools, "get_deal_list", AsyncMock(return_value=[])),
    ):
        data = await bfl_rf_tracker.collect(
            SimpleNamespace(), direct, _settings(), metrika=_MetrikaStub(), days=2
        )

    assert "error" in data["direct"]
    assert "direct 503" in data["direct"]["error"]
    # Other layers still good
    assert data["metrika"]["visits"] == 10
    assert data["bitrix"]["leads"] == 1
    # Economics safe-divide when cost layer failed
    assert data["economics"]["cost"] == 0.0
    assert data["economics"]["cpa_lead"] == 0.0


@pytest.mark.asyncio
async def test_collect_isolates_bitrix_failure() -> None:
    tsv = _make_tsv([{"impressions": 2000, "clicks": 100, "cost": 500.0}])
    direct = SimpleNamespace(
        get_campaigns=AsyncMock(return_value=[{"Id": 709307228, "Name": "RF", "State": "ON"}]),
        get_campaign_stats=AsyncMock(return_value={"tsv": tsv}),
    )
    with (
        patch.object(
            bfl_rf_tracker.bitrix_tools,
            "get_lead_list",
            AsyncMock(side_effect=RuntimeError("bitrix 500")),
        ),
        patch.object(bfl_rf_tracker.bitrix_tools, "get_deal_list", AsyncMock(return_value=[])),
    ):
        data = await bfl_rf_tracker.collect(
            SimpleNamespace(), direct, _settings(), metrika=None, days=2
        )

    assert data["direct"]["clicks"] == 100.0
    assert "error" in data["bitrix"]
    assert data["metrika"] == {"error": "no_client"}


@pytest.mark.asyncio
async def test_collect_default_campaign_id_used_when_kwarg_omitted() -> None:
    tsv = _make_tsv([{"impressions": 100, "clicks": 5, "cost": 50.0}])
    captured: dict[str, Any] = {}

    async def _get_campaigns(ids: list[int]) -> list[dict[str, Any]]:
        captured["ids"] = ids
        return [{"Id": ids[0], "Name": "RF", "State": "ON"}]

    async def _get_stats(cid: int, date_from: str, date_to: str) -> dict[str, Any]:
        return {"tsv": tsv}

    direct = SimpleNamespace(
        get_campaigns=AsyncMock(side_effect=_get_campaigns),
        get_campaign_stats=AsyncMock(side_effect=_get_stats),
    )
    with (
        patch.object(bfl_rf_tracker.bitrix_tools, "get_lead_list", AsyncMock(return_value=[])),
        patch.object(bfl_rf_tracker.bitrix_tools, "get_deal_list", AsyncMock(return_value=[])),
    ):
        await bfl_rf_tracker.collect(SimpleNamespace(), direct, _settings(), days=1)

    assert captured["ids"] == [bfl_rf_tracker.DEFAULT_BFL_RF_CAMPAIGN_ID]
