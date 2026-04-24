"""Unit tests for agent_runtime.jobs.budget_guard (Task 14).

Tests rely on a mocked pool + mocked DirectAPI + SimpleNamespace http stub,
so nothing hits the wire. The critical flows are:

* Breach detection across the three trust levels (shadow / assisted /
  autonomous).
* Hypothesis-aware short-circuit (budget_cap reached = concluded, no alert).
* ``dry_run=True`` suppresses suspend and Telegram while reporting
  ``would_suspend``.
* PROTECTED guard rejection in autonomous — alert still fires, no suspend.
* TSV parsing + absolute-limit OR-gate.
* KB consult is best-effort — failures never break the run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import budget_guard
from agent_runtime.tools.direct_api import ProtectedCampaignError
from agent_runtime.trust_levels import TrustLevel

_CAMP_ACTIVE = 708978456
_CAMP_SUSPENDED = 708978458

_PROTECTED = [_CAMP_ACTIVE, 708978457, _CAMP_SUSPENDED, 709307228]


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
    *,
    trust_level: TrustLevel = TrustLevel.SHADOW,
    running_hypothesis: dict[str, Any] | None = None,
) -> tuple[MagicMock, MagicMock, list[Any]]:
    """Mock pool that answers the two SELECTs budget_guard issues in this order:

    1. ``get_trust_level`` → sda_state.value
    2. ``_find_running_hypothesis_for_campaign`` → hypotheses row (may be None)

    Each subsequent fetchone() after the scripted ones defaults to ``(1,)``
    so INSERT...RETURNING id in audit_log / update_outcome never fails.
    """
    rows: list[Any] = [(trust_level.value,)]
    for _ in _PROTECTED:
        if running_hypothesis is None:
            rows.append(None)
        else:
            rows.append(
                (
                    running_hypothesis["id"],
                    running_hypothesis["budget_cap_rub"],
                    running_hypothesis["created_at"],
                    running_hypothesis.get("metrics_before", {}),
                )
            )

    one_iter = iter(rows)

    async def _fetchone():
        try:
            return next(one_iter)
        except StopIteration:
            return (1,)

    executed: list[Any] = []

    async def _exec(*args: Any, **kwargs: Any) -> None:
        executed.append(args)

    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=_exec)
    cursor.fetchone = AsyncMock(side_effect=_fetchone)
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, cursor, executed


def _direct_stub(
    *,
    today_cost_rub: float = 0,
    daily_avg_rub: float = 1500,
    daily_limit_rub: int = 3000,
    active_campaign_ids: list[int] | None = None,
    pause_raises: BaseException | None = None,
    verify_returns: bool = True,
) -> SimpleNamespace:
    """Build a DirectAPI-like stub driven by numeric scenario parameters."""
    active = active_campaign_ids if active_campaign_ids is not None else [_CAMP_ACTIVE]

    async def get_campaigns(ids: list[int]) -> list[dict[str, Any]]:
        cid = ids[0]
        if cid in active:
            return [
                {
                    "Id": cid,
                    "Name": f"Test-{cid}",
                    "State": "ON",
                    "DailyBudget": {"Amount": daily_limit_rub * 1_000_000},
                }
            ]
        return [{"Id": cid, "Name": f"Test-{cid}", "State": "SUSPENDED"}]

    today = (datetime.now(UTC) + timedelta(hours=3)).date().isoformat()
    past = (datetime.now(UTC) + timedelta(hours=3) - timedelta(days=1)).date().isoformat()
    tsv = (
        "Campaign performance report 'sda_v3_x'\n"
        "Date\tCampaignId\tImpressions\tClicks\tCost\tConversions\n"
        f"{past}\t1\t1000\t50\t{int(daily_avg_rub * 1_000_000)}\t2\n"
        f"{today}\t1\t1000\t50\t{int(today_cost_rub * 1_000_000)}\t1\n"
        "Total\t1\t2000\t100\t0\t3\n"
    )

    async def get_campaign_stats(campaign_id: int, date_from: str, date_to: str):
        return {"tsv": tsv}

    pause_mock = AsyncMock()
    if pause_raises is not None:
        pause_mock.side_effect = pause_raises

    verify_mock = AsyncMock(return_value=verify_returns)

    return SimpleNamespace(
        get_campaigns=AsyncMock(side_effect=get_campaigns),
        get_campaign_stats=AsyncMock(side_effect=get_campaign_stats),
        pause_campaign=pause_mock,
        verify_campaign_paused=verify_mock,
    )


# ----------------------------------------------------------------- TSV parser


def test_parse_costs_skips_header_and_total() -> None:
    tsv = (
        "Header line\n"
        "Date\tCampaignId\tCost\n"
        "2026-04-24\t1\t1500000000\n"
        "2026-04-25\t1\t500000000\n"
        "Total\t1\t2000000000\n"
    )
    parsed = budget_guard._parse_costs(tsv)
    assert parsed == {"2026-04-24": 1500.0, "2026-04-25": 500.0}


def test_parse_costs_empty_tsv() -> None:
    assert budget_guard._parse_costs("") == {}


def test_parse_costs_missing_columns_returns_empty() -> None:
    tsv = "Irrelevant header\ncol1\tcol2\nfoo\tbar\n"
    assert budget_guard._parse_costs(tsv) == {}


# ------------------------------------------------------------------ main run


@pytest.mark.asyncio
async def test_run_no_breach_returns_ok_no_actions() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.SHADOW)
    direct = _direct_stub(
        today_cost_rub=1000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    http_client = SimpleNamespace(post=AsyncMock())
    with patch.object(
        budget_guard.knowledge,
        "consult",
        AsyncMock(return_value={"answer": "x", "citations": ["kb/x.md"]}),
    ):
        result = await budget_guard.run(
            pool, direct=direct, http_client=http_client, settings=_settings()
        )

    assert result["status"] == "ok"
    assert result["trust_level"] == "shadow"
    assert result["breached"] == []
    assert result["notified"] == []
    assert result["suspended"] == []
    assert result["would_suspend"] == []
    assert result["hypothesis_concluded"] == []
    assert result["kb_citation"] is not None


@pytest.mark.asyncio
async def test_run_breach_in_autonomous_triggers_suspend_and_notify() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub(
        today_cost_rub=5000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    telegram_mock = AsyncMock(return_value=1)
    http_client = SimpleNamespace()
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(side_effect=RuntimeError("no anthropic key")),
        ),
    ):
        result = await budget_guard.run(
            pool, direct=direct, http_client=http_client, settings=_settings()
        )

    assert _CAMP_ACTIVE in result["breached"]
    assert _CAMP_ACTIVE in result["suspended"]
    assert _CAMP_ACTIVE in result["notified"]
    # PROTECTED guard does NOT block in the mock — stub's pause_campaign is a
    # plain AsyncMock, not the real guarded client. Test covers the happy
    # autonomous path; PROTECTED rejection is tested separately.
    direct.pause_campaign.assert_awaited_once_with(_CAMP_ACTIVE)
    telegram_mock.assert_awaited()
    assert result["kb_citation"] is None


@pytest.mark.asyncio
async def test_run_breach_in_shadow_notifies_only_no_suspend() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.SHADOW)
    direct = _direct_stub(
        today_cost_rub=5000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    telegram_mock = AsyncMock(return_value=1)
    http_client = SimpleNamespace()
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool, direct=direct, http_client=http_client, settings=_settings()
        )

    assert _CAMP_ACTIVE in result["breached"]
    assert result["suspended"] == []
    assert _CAMP_ACTIVE in result["notified"]
    direct.pause_campaign.assert_not_awaited()
    call_kwargs = telegram_mock.await_args.kwargs
    assert "no auto-suspend" in call_kwargs["text"]


@pytest.mark.asyncio
async def test_run_breach_in_assisted_notifies_only_no_suspend() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.ASSISTED)
    direct = _direct_stub(
        today_cost_rub=5000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    telegram_mock = AsyncMock(return_value=1)
    http_client = SimpleNamespace()
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "x", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool, direct=direct, http_client=http_client, settings=_settings()
        )

    assert _CAMP_ACTIVE in result["breached"]
    assert result["suspended"] == []
    assert _CAMP_ACTIVE in result["notified"]
    direct.pause_campaign.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_hypothesis_at_cap_concludes_without_alert() -> None:
    pool, _, executed = _mock_pool(
        trust_level=TrustLevel.AUTONOMOUS,
        running_hypothesis={
            "id": "hyp-budget-cap",
            "budget_cap_rub": 500,
            "created_at": datetime.now(UTC) - timedelta(hours=2),
            "metrics_before": {"cost_snapshot_today": 100.0},
        },
    )
    direct = _direct_stub(
        today_cost_rub=700,  # delta = 700 - 100 = 600 >= cap 500
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    telegram_mock = AsyncMock(return_value=1)
    http_client = SimpleNamespace()
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool, direct=direct, http_client=http_client, settings=_settings()
        )

    assert "hyp-budget-cap" in result["hypothesis_concluded"]
    assert result["breached"] == []
    assert result["suspended"] == []
    telegram_mock.assert_not_awaited()
    direct.pause_campaign.assert_not_awaited()
    # UPDATE hypotheses ... SET state was issued
    update_sqls = [
        args[0]
        for args in executed
        if args and isinstance(args[0], str) and "UPDATE hypotheses" in args[0]
    ]
    assert update_sqls, "update_outcome UPDATE was not issued"


@pytest.mark.asyncio
async def test_run_hypothesis_below_cap_proceeds_with_normal_breach() -> None:
    pool, _, _ = _mock_pool(
        trust_level=TrustLevel.AUTONOMOUS,
        running_hypothesis={
            "id": "hyp-under-cap",
            "budget_cap_rub": 3000,
            "created_at": datetime.now(UTC) - timedelta(hours=1),
            "metrics_before": {"cost_snapshot_today": 200.0},
        },
    )
    direct = _direct_stub(
        today_cost_rub=5000,  # delta = 4800, below cap 3000? no, above.
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    # adjust to keep delta below cap: snapshot 3000, today 5000 → delta 2000 < cap 3000
    direct = _direct_stub(
        today_cost_rub=5000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    pool, _, _ = _mock_pool(
        trust_level=TrustLevel.AUTONOMOUS,
        running_hypothesis={
            "id": "hyp-under-cap",
            "budget_cap_rub": 3000,
            "created_at": datetime.now(UTC) - timedelta(hours=1),
            "metrics_before": {"cost_snapshot_today": 3000.0},
        },
    )
    telegram_mock = AsyncMock(return_value=1)
    http_client = SimpleNamespace()
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool, direct=direct, http_client=http_client, settings=_settings()
        )

    # hypothesis did NOT reach cap → normal breach path
    assert result["hypothesis_concluded"] == []
    assert _CAMP_ACTIVE in result["breached"]


@pytest.mark.asyncio
async def test_run_dry_run_does_not_call_direct_suspend() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub(
        today_cost_rub=5000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    telegram_mock = AsyncMock(return_value=1)
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool,
            direct=direct,
            http_client=SimpleNamespace(),
            settings=_settings(),
            dry_run=True,
        )

    assert _CAMP_ACTIVE in result["breached"]
    assert result["suspended"] == []
    assert _CAMP_ACTIVE in result["would_suspend"]
    direct.pause_campaign.assert_not_awaited()
    telegram_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_autonomous_protected_guard_blocks_suspend_but_alerts() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub(
        today_cost_rub=5000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
        pause_raises=ProtectedCampaignError("protected"),
    )
    telegram_mock = AsyncMock(return_value=1)
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool,
            direct=direct,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )

    assert _CAMP_ACTIVE in result["breached"]
    assert result["suspended"] == []
    assert _CAMP_ACTIVE in result["notified"]
    telegram_mock.assert_awaited()


@pytest.mark.asyncio
async def test_run_suspended_campaigns_skipped() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub(
        today_cost_rub=5000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[],  # nothing ON
    )
    telegram_mock = AsyncMock(return_value=1)
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool,
            direct=direct,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )

    assert result["checked_campaigns"] == []
    assert result["breached"] == []
    assert result["suspended"] == []
    direct.pause_campaign.assert_not_awaited()
    telegram_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_daily_limit_breach_triggers_even_if_surge_low() -> None:
    """Absolute limit fires even when relative surge × 1.5 would pass."""
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    # surge check: 3100 vs avg 2500 * 1.5 = 3750 → pass. BUT 3100 > limit 3000 → breach.
    direct = _direct_stub(
        today_cost_rub=3100,
        daily_avg_rub=2500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    telegram_mock = AsyncMock(return_value=1)
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool,
            direct=direct,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )

    assert _CAMP_ACTIVE in result["breached"]


@pytest.mark.asyncio
async def test_run_kb_consult_best_effort_on_failure() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.SHADOW)
    direct = _direct_stub(
        today_cost_rub=1000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
    )
    with patch.object(
        budget_guard.knowledge, "consult", AsyncMock(side_effect=RuntimeError("no key"))
    ):
        result = await budget_guard.run(
            pool,
            direct=direct,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )

    assert result["kb_citation"] is None
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_stats_api_failure_skips_tick_not_suspend() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub(active_campaign_ids=[_CAMP_ACTIVE])
    direct.get_campaign_stats = AsyncMock(side_effect=RuntimeError("direct api down"))
    telegram_mock = AsyncMock(return_value=1)
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
    ):
        result = await budget_guard.run(
            pool,
            direct=direct,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )

    assert result["checked_campaigns"] == []
    assert result["suspended"] == []
    telegram_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_without_direct_or_settings_returns_degraded_noop() -> None:
    """dispatch_job passes only pool+dry_run — must not blow up."""
    pool, _, _ = _mock_pool(trust_level=TrustLevel.SHADOW)
    result = await budget_guard.run(pool)
    assert result["status"] == "ok"
    assert result["checked_campaigns"] == []
    assert result["trust_level"] == "shadow"


@pytest.mark.asyncio
async def test_run_registered_in_job_registry() -> None:
    from agent_runtime.jobs import JOB_REGISTRY

    assert "budget_guard" in JOB_REGISTRY
    assert JOB_REGISTRY["budget_guard"] is budget_guard.run


@pytest.mark.asyncio
async def test_run_autonomous_verify_failure_still_alerts() -> None:
    """pause succeeded but verify kept returning False — suspend not counted."""
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub(
        today_cost_rub=5000,
        daily_avg_rub=1500,
        daily_limit_rub=3000,
        active_campaign_ids=[_CAMP_ACTIVE],
        verify_returns=False,
    )
    telegram_mock = AsyncMock(return_value=1)
    with (
        patch.object(budget_guard.telegram_tools, "send_message", telegram_mock),
        patch.object(
            budget_guard.knowledge,
            "consult",
            AsyncMock(return_value={"answer": "", "citations": []}),
        ),
        patch.object(budget_guard, "_sleep_for_verify", AsyncMock()),
    ):
        result = await budget_guard.run(
            pool,
            direct=direct,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )

    assert _CAMP_ACTIVE in result["breached"]
    assert result["suspended"] == []
    assert _CAMP_ACTIVE in result["notified"]
    telegram_mock.assert_awaited()


def test_format_alert_contains_expected_fields() -> None:
    text = budget_guard._format_alert(
        campaign_id=708978456,
        name="Test",
        today_cost=5000.0,
        daily_avg=1500.0,
        daily_limit=3000,
        reason="surge",
        trust_level=TrustLevel.AUTONOMOUS,
        auto_suspended=True,
    )
    assert "AUTO-SUSPEND" in text
    assert "5000" in text
    assert "1500" in text
    assert "autonomous" in text


def test_format_alert_shadow_adds_no_auto_suspend_note() -> None:
    text = budget_guard._format_alert(
        campaign_id=708978456,
        name="Test",
        today_cost=5000.0,
        daily_avg=1500.0,
        daily_limit=None,
        reason="surge",
        trust_level=TrustLevel.SHADOW,
        auto_suspended=False,
    )
    assert "no auto-suspend" in text
    assert "BREACH" in text
