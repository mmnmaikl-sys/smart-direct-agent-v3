"""Unit tests for the 7 kill-switches + run_all fail-closed semantics.

Every guard gets (pass / trigger / edge) coverage. Clients are mocked via
trivial async stubs — guards talk to protocols, not concrete classes, so we
can hand them tiny fakes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.tools import kill_switches as _kill_switches_mod
from agent_runtime.tools.kill_switches import (
    ALL_GUARDS,
    Action,
    BudgetBalance,
    BudgetCap,
    ConversionIntegrity,
    CPCCeiling,
    KillSwitchContext,
    KillSwitchResult,
    NegKWFloor,
    QSGuard,
    QueryDrift,
    _jaccard,
    _percentile,
    run_all,
)

# Working MSK time for wall-clock-sensitive tests (BudgetCap has a quiet-
# hours window 00:00-02:00 МСК). 12:00 МСК is safely outside.
_MSK_WORKING_HOURS = datetime(2026, 4, 25, 12, 0, tzinfo=UTC) + timedelta(hours=3)

_PROTECTED = [708978456, 708978457, 708978458, 709014142, 709307228]


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        YANDEX_DIRECT_TOKEN="test-token",
        PROTECTED_CAMPAIGN_IDS=_PROTECTED,
    )


def _mock_pool(fetchone_return: Any = None) -> MagicMock:
    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=fetchone_return)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool


def _context(
    *,
    pool: MagicMock | None = None,
    direct: Any = None,
    metrika: Any = None,
    bitrix: Any = None,
    budget_history: dict | None = None,
    bid_history_by_adgroup: dict | None = None,
    adgroup_productivity: dict | None = None,
    baseline_queries: list[str] | None = None,
    recent_queries: list[str] | None = None,
    weekly_budget_total_rub: int | None = None,
    trust_level: str = "shadow",
) -> KillSwitchContext:
    direct = direct or SimpleNamespace(
        get_campaigns=AsyncMock(return_value=[]),
        get_adgroups=AsyncMock(return_value=[]),
        get_keywords=AsyncMock(return_value=[]),
        get_campaign_stats=AsyncMock(return_value={}),
    )
    return KillSwitchContext(
        pool=pool or _mock_pool(),
        direct=direct,
        metrika=metrika,
        bitrix=bitrix,
        settings=_settings(),
        trust_level=trust_level,
        hypothesis_id="h-test",
        budget_history=budget_history,
        adgroup_productivity=adgroup_productivity,
        baseline_queries=baseline_queries,
        recent_queries=recent_queries,
        bid_history_by_adgroup=bid_history_by_adgroup,
        weekly_budget_total_rub=weekly_budget_total_rub,
    )


# ---------- helpers ---------------------------------------------------------


def test_jaccard_empty_sets_is_one() -> None:
    assert _jaccard(set(), set()) == 1.0


def test_jaccard_disjoint() -> None:
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_percentile_single_value() -> None:
    assert _percentile([50], 0.9) == 50.0


def test_percentile_p90_from_10() -> None:
    data = list(range(1, 11))
    assert _percentile(data, 0.9) >= 9.0


# ---------- BudgetCap -------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_cap_triggers_when_today_cost_exceeds_1_5x_avg() -> None:
    ctx = _context(budget_history={123: {"today_cost": 3000, "daily_avg_7d": 1500}})
    action = Action(type="raise_budget", params={"campaign_id": 123})
    with patch.object(_kill_switches_mod, "_current_msk_time", return_value=_MSK_WORKING_HOURS):
        r = await BudgetCap().check(action, ctx)
    assert r.allow is False
    assert r.switch_name == "budget_cap"
    assert "3000" in r.reason and "1500" in r.reason


@pytest.mark.asyncio
async def test_budget_cap_allows_below_threshold() -> None:
    ctx = _context(budget_history={123: {"today_cost": 2000, "daily_avg_7d": 1500}})
    action = Action(type="raise_budget", params={"campaign_id": 123})
    r = await BudgetCap().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_budget_cap_floor_exemption() -> None:
    # today_cost below protected floor → pass even if ratio would trigger
    ctx = _context(budget_history={123: {"today_cost": 1400, "daily_avg_7d": 500}})
    action = Action(type="raise_budget", params={"campaign_id": 123})
    r = await BudgetCap().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_budget_cap_noop_for_unrelated_action() -> None:
    ctx = _context()
    action = Action(type="add_keyword", params={"ad_group_id": 9})
    r = await BudgetCap().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_budget_cap_no_campaign_id_passes() -> None:
    ctx = _context()
    action = Action(type="raise_budget", params={})
    r = await BudgetCap().check(action, ctx)
    assert r.allow is True


# ---------- CPCCeiling -------------------------------------------------------


@pytest.mark.asyncio
async def test_cpc_ceiling_rejects_bid_above_p90_x_1_3() -> None:
    # 10 bids in range 500-5000; p90 is ~5000 (top decile); ceiling = 6500
    history = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    ctx = _context(bid_history_by_adgroup={9: history})
    action = Action(type="set_bid", params={"ad_group_id": 9, "bid": 9000})
    r = await CPCCeiling().check(action, ctx)
    assert r.allow is False
    assert "p90" in r.reason


@pytest.mark.asyncio
async def test_cpc_ceiling_allows_bid_under_ceiling() -> None:
    history = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    ctx = _context(bid_history_by_adgroup={9: history})
    action = Action(type="set_bid", params={"ad_group_id": 9, "bid": 5000})
    r = await CPCCeiling().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_cpc_ceiling_fallback_for_new_ad_group() -> None:
    ctx = _context(bid_history_by_adgroup={9: [1000]})  # <5 points
    action = Action(
        type="set_bid",
        params={"ad_group_id": 9, "bid": 7000, "effective_bid": 3000},
    )
    r = await CPCCeiling().check(action, ctx)
    # effective_bid × 2 = 6000 ceiling; 7000 > 6000 → reject
    assert r.allow is False


@pytest.mark.asyncio
async def test_cpc_ceiling_no_fallback_no_history_passes() -> None:
    ctx = _context()
    action = Action(type="set_bid", params={"ad_group_id": 9, "bid": 10000})
    r = await CPCCeiling().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_cpc_ceiling_no_bid_field_passes() -> None:
    ctx = _context()
    action = Action(type="set_bid", params={"ad_group_id": 9})
    r = await CPCCeiling().check(action, ctx)
    assert r.allow is True


# ---------- NegKWFloor -------------------------------------------------------


@pytest.mark.asyncio
async def test_neg_kw_floor_rejects_pause_protected_campaign_by_id() -> None:
    ctx = _context()
    action = Action(type="pause_campaign", params={"campaign_id": _PROTECTED[0]})
    r = await NegKWFloor().check(action, ctx)
    assert r.allow is False
    assert "PROTECTED" in r.reason


@pytest.mark.asyncio
async def test_neg_kw_floor_rejects_pause_by_name_substring() -> None:
    ctx = _context()
    action = Action(
        type="pause_campaign",
        params={"campaign_name": "БФЛ Башкортостан RSYA (renamed)"},
    )
    r = await NegKWFloor().check(action, ctx)
    assert r.allow is False
    assert "бфл башкортостан" in r.reason.lower()


@pytest.mark.asyncio
async def test_neg_kw_floor_rejects_remove_protected_keyword() -> None:
    ctx = _context()
    action = Action(type="remove_keyword", params={"keyword": "Банкротство физ лиц"})
    r = await NegKWFloor().check(action, ctx)
    assert r.allow is False
    assert "protected registry" in r.reason


@pytest.mark.asyncio
async def test_neg_kw_floor_rejects_add_neg_of_protected() -> None:
    ctx = _context()
    action = Action(type="add_neg_keyword", params={"phrase": "банкротство физических лиц"})
    r = await NegKWFloor().check(action, ctx)
    assert r.allow is False


@pytest.mark.asyncio
async def test_neg_kw_floor_allows_non_protected_keyword() -> None:
    ctx = _context()
    action = Action(type="remove_keyword", params={"keyword": "дешевое банкротство"})
    r = await NegKWFloor().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_neg_kw_floor_noop_for_unrelated_action() -> None:
    ctx = _context()
    action = Action(type="set_bid", params={"ad_group_id": 9, "bid": 5000})
    r = await NegKWFloor().check(action, ctx)
    assert r.allow is True


# ---------- QSGuard ---------------------------------------------------------


@pytest.mark.asyncio
async def test_qs_guard_rejects_raise_budget_when_productivity_under_6() -> None:
    ctx = _context(adgroup_productivity={9: 5})
    action = Action(type="raise_budget", params={"ad_group_id": 9})
    r = await QSGuard().check(action, ctx)
    assert r.allow is False
    assert "QS=5" in r.reason


@pytest.mark.asyncio
async def test_qs_guard_allows_when_productivity_passes_floor() -> None:
    ctx = _context(adgroup_productivity={9: 7})
    action = Action(type="raise_budget", params={"ad_group_id": 9})
    r = await QSGuard().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_qs_guard_fetches_via_direct_when_not_pre_populated() -> None:
    direct = SimpleNamespace(
        get_campaigns=AsyncMock(return_value=[]),
        get_adgroups=AsyncMock(return_value=[{"Id": 9, "Productivity": 3}]),
        get_keywords=AsyncMock(return_value=[]),
        get_campaign_stats=AsyncMock(return_value={}),
    )
    ctx = _context(direct=direct)
    action = Action(type="increase_bid", params={"ad_group_id": 9})
    r = await QSGuard().check(action, ctx)
    assert r.allow is False


# ---------- BudgetBalance ---------------------------------------------------


@pytest.mark.asyncio
async def test_budget_balance_rejects_when_weekly_delta_exceeds_20pct() -> None:
    pool = _mock_pool(fetchone_return=(20000,))  # existing sum 20k
    ctx = _context(pool=pool, weekly_budget_total_rub=100_000)
    action = Action(type="raise_budget", params={"delta_rub": 5000})
    r = await BudgetBalance().check(action, ctx)
    assert r.allow is False
    assert "exceeds" in r.reason


@pytest.mark.asyncio
async def test_budget_balance_allows_under_cap() -> None:
    pool = _mock_pool(fetchone_return=(10000,))
    ctx = _context(pool=pool, weekly_budget_total_rub=100_000)
    action = Action(type="raise_budget", params={"delta_rub": 5000})
    r = await BudgetBalance().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_budget_balance_unknown_total_passes() -> None:
    pool = _mock_pool(fetchone_return=(0,))
    ctx = _context(pool=pool, weekly_budget_total_rub=0)
    action = Action(type="raise_budget", params={"delta_rub": 5000})
    r = await BudgetBalance().check(action, ctx)
    assert r.allow is True


# ---------- ConversionIntegrity ---------------------------------------------


@pytest.mark.asyncio
async def test_conversion_integrity_skips_without_clients() -> None:
    ctx = _context(bitrix=None, metrika=None)
    action = Action(type="raise_budget_based_on_cpa", params={})
    r = await ConversionIntegrity().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_conversion_integrity_rejects_on_duplicate_visitor_id() -> None:
    now = datetime.now(UTC)
    leads = [
        {"visitor_id": "v1", "created_at": now - timedelta(minutes=5)},
        {"visitor_id": "v1", "created_at": now},
        {"visitor_id": "v2", "created_at": now},
    ]
    bitrix = SimpleNamespace(recent_leads=AsyncMock(return_value=leads))
    metrika = SimpleNamespace(recent_visits=AsyncMock(return_value=[]))
    ctx = _context(bitrix=bitrix, metrika=metrika)
    action = Action(type="raise_budget_based_on_cpa", params={})
    r = await ConversionIntegrity().check(action, ctx)
    assert r.allow is False
    assert "v1" in r.reason


@pytest.mark.asyncio
async def test_conversion_integrity_rejects_on_high_bot_share() -> None:
    bitrix = SimpleNamespace(recent_leads=AsyncMock(return_value=[]))
    visits = [{"is_robot": True}] * 8 + [{"is_robot": False}] * 2  # 80% bots
    metrika = SimpleNamespace(recent_visits=AsyncMock(return_value=visits))
    ctx = _context(bitrix=bitrix, metrika=metrika)
    action = Action(type="raise_budget_based_on_cpa", params={})
    r = await ConversionIntegrity().check(action, ctx)
    assert r.allow is False
    assert "80%" in r.reason


@pytest.mark.asyncio
async def test_conversion_integrity_allows_normal_traffic() -> None:
    bitrix = SimpleNamespace(recent_leads=AsyncMock(return_value=[]))
    visits = [{"is_robot": False}] * 10
    metrika = SimpleNamespace(recent_visits=AsyncMock(return_value=visits))
    ctx = _context(bitrix=bitrix, metrika=metrika)
    action = Action(type="raise_budget_based_on_cpa", params={})
    r = await ConversionIntegrity().check(action, ctx)
    assert r.allow is True


# ---------- QueryDrift ------------------------------------------------------


@pytest.mark.asyncio
async def test_query_drift_rejects_when_jaccard_below_0_5() -> None:
    baseline = ["банкротство", "долги", "кредит", "суд", "процедура", "списание"]
    current = ["дешево", "бесплатно", "купить", "ипотека", "кредит", "быстро"]
    ctx = _context(baseline_queries=baseline, recent_queries=current)
    action = Action(type="add_keyword", params={})
    r = await QueryDrift().check(action, ctx)
    assert r.allow is False
    assert "jaccard" in r.reason


@pytest.mark.asyncio
async def test_query_drift_allows_when_high_overlap() -> None:
    baseline = ["a", "b", "c", "d", "e", "f"]
    current = ["a", "b", "c", "d", "e", "g"]  # 5/7 ≈ 0.71
    ctx = _context(baseline_queries=baseline, recent_queries=current)
    action = Action(type="add_keyword", params={})
    r = await QueryDrift().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_query_drift_allows_when_no_baseline() -> None:
    ctx = _context(baseline_queries=None, recent_queries=["x"])
    action = Action(type="add_keyword", params={})
    r = await QueryDrift().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_query_drift_skips_small_sets() -> None:
    # <5 baseline → pass (noise floor)
    ctx = _context(baseline_queries=["a", "b"], recent_queries=["c", "d", "e", "f", "g"])
    action = Action(type="add_keyword", params={})
    r = await QueryDrift().check(action, ctx)
    assert r.allow is True


# ---------- run_all ---------------------------------------------------------


@pytest.mark.asyncio
async def test_run_all_returns_7_results_for_noop_action() -> None:
    ctx = _context()
    action = Action(type="__noop__", params={})
    results = await run_all(action, ctx)
    assert len(results) == 7
    assert all(isinstance(r, KillSwitchResult) for r in results)
    assert all(r.allow for r in results)


@pytest.mark.asyncio
async def test_run_all_fail_closed_on_exception() -> None:
    # QSGuard will try to fetch via direct.get_adgroups — make it raise.
    direct = SimpleNamespace(
        get_campaigns=AsyncMock(return_value=[]),
        get_adgroups=AsyncMock(side_effect=RuntimeError("boom")),
        get_keywords=AsyncMock(return_value=[]),
        get_campaign_stats=AsyncMock(return_value={}),
    )
    ctx = _context(direct=direct)
    action = Action(type="raise_budget", params={"ad_group_id": 9})
    results = await run_all(action, ctx)
    qs = next(r for r in results if r.switch_name == "qs_guard")
    assert qs.allow is False, "fail-closed on exception"
    assert "exception" in qs.reason


@pytest.mark.asyncio
async def test_run_all_multiple_guards_can_fire_simultaneously() -> None:
    ctx = _context(
        budget_history={123: {"today_cost": 3000, "daily_avg_7d": 1500}},
        adgroup_productivity={9: 3},
    )
    action = Action(type="raise_budget", params={"campaign_id": 123, "ad_group_id": 9})
    with patch.object(_kill_switches_mod, "_current_msk_time", return_value=_MSK_WORKING_HOURS):
        results = await run_all(action, ctx)
    rejects = [r for r in results if not r.allow]
    reject_names = {r.switch_name for r in rejects}
    assert "budget_cap" in reject_names
    assert "qs_guard" in reject_names


def test_all_guards_have_name_attribute() -> None:
    for guard_cls in ALL_GUARDS:
        assert hasattr(guard_cls, "name")
        assert isinstance(guard_cls.name, str)
        assert guard_cls.name  # non-empty


# ---------- additional coverage --------------------------------------------


def test_action_from_dict_roundtrip() -> None:
    a = Action.from_dict({"type": "set_bid", "params": {"ad_group_id": 9, "bid": 5000}})
    assert a.type == "set_bid"
    assert a.params == {"ad_group_id": 9, "bid": 5000}


def test_action_from_dict_defaults() -> None:
    a = Action.from_dict({})
    assert a.type == ""
    assert a.params == {}


@pytest.mark.asyncio
async def test_budget_cap_fetches_history_from_direct() -> None:
    direct = SimpleNamespace(
        get_campaigns=AsyncMock(return_value=[]),
        get_adgroups=AsyncMock(return_value=[]),
        get_keywords=AsyncMock(return_value=[]),
        get_campaign_stats=AsyncMock(return_value={"today_cost": 4000, "daily_avg_7d": 2000}),
    )
    ctx = _context(direct=direct)  # no pre-populated budget_history
    action = Action(type="raise_budget", params={"campaign_id": 123})
    with patch.object(_kill_switches_mod, "_current_msk_time", return_value=_MSK_WORKING_HOURS):
        r = await BudgetCap().check(action, ctx)
    assert r.allow is False


@pytest.mark.asyncio
async def test_neg_kw_floor_loads_registry_from_pool() -> None:
    pool = _mock_pool(fetchone_return=([{"keyword": "мой защищённый ключ", "reason": "seed"}],))
    ctx = _context(pool=pool)
    action = Action(type="remove_keyword", params={"keyword": "Мой защищённый ключ"})
    r = await NegKWFloor().check(action, ctx)
    assert r.allow is False


@pytest.mark.asyncio
async def test_neg_kw_floor_registry_load_exception_falls_back_to_seed() -> None:
    pool = MagicMock()
    # Simulate DB-wide failure — connection() raises.
    pool.connection = MagicMock(side_effect=RuntimeError("pg down"))
    ctx = _context(pool=pool)
    action = Action(type="remove_keyword", params={"keyword": "банкротство физ лиц"})
    r = await NegKWFloor().check(action, ctx)
    # Seed still protects the keyword.
    assert r.allow is False


@pytest.mark.asyncio
async def test_conversion_integrity_handles_iso_string_timestamps() -> None:
    now = datetime.now(UTC)
    leads = [
        {"visitor_id": "v42", "created_at": (now - timedelta(minutes=3)).isoformat()},
        {"visitor_id": "v42", "created_at": now.isoformat()},
    ]
    bitrix = SimpleNamespace(recent_leads=AsyncMock(return_value=leads))
    metrika = SimpleNamespace(recent_visits=AsyncMock(return_value=[]))
    ctx = _context(bitrix=bitrix, metrika=metrika)
    action = Action(type="raise_budget_based_on_cpa", params={})
    r = await ConversionIntegrity().check(action, ctx)
    assert r.allow is False


@pytest.mark.asyncio
async def test_conversion_integrity_skips_leads_without_visitor_id() -> None:
    now = datetime.now(UTC)
    leads = [
        {"visitor_id": None, "created_at": now},
        {"visitor_id": "v1", "created_at": now},
    ]
    bitrix = SimpleNamespace(recent_leads=AsyncMock(return_value=leads))
    metrika = SimpleNamespace(recent_visits=AsyncMock(return_value=[]))
    ctx = _context(bitrix=bitrix, metrika=metrika)
    action = Action(type="raise_budget_based_on_cpa", params={})
    r = await ConversionIntegrity().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_budget_balance_noop_for_unrelated_action() -> None:
    ctx = _context()
    action = Action(type="set_bid", params={"bid": 5000})
    r = await BudgetBalance().check(action, ctx)
    assert r.allow is True


@pytest.mark.asyncio
async def test_query_drift_noop_for_unrelated_action() -> None:
    ctx = _context(baseline_queries=["a"] * 10, recent_queries=["b"] * 10)
    action = Action(type="set_bid", params={"bid": 5000})
    r = await QueryDrift().check(action, ctx)
    assert r.allow is True


def test_percentile_empty_returns_zero() -> None:
    assert _percentile([], 0.9) == 0.0


def test_percentile_handles_low_and_middle_quantiles() -> None:
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # p10 should be near the bottom decile
    assert _percentile(data, 0.1) <= 20.0
    # p50 should land in the middle half
    mid = _percentile(data, 0.5)
    assert 30.0 <= mid <= 80.0
