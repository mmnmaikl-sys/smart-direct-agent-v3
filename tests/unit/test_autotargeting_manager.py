"""Unit tests for agent_runtime.jobs.autotargeting_manager (Task 21, part 2).

Mocked pool + SimpleNamespace DirectAPI stub — no wire calls. Covers:

* ``_detect_drift`` normaliser (compliant vs WIDER vs WithBrands vs missing).
* Happy path: EXACT+WithoutBrands across all AdGroups → no action.
* Drift in one AdGroup (WIDER / COMPETITORS / WithBrands) → rollback in
  autonomous, NOTIFY in shadow.
* ``dry_run=True`` never mutates.
* GET-after-SET mismatch surfaces as errors row.
* ``set_autotargeting`` missing + ``get_autotargeting`` missing → degraded.
* Only :data:`EPK_SEARCH_CAMPAIGNS` get enumerated — РСЯ campaigns untouched.
* ``ProtectedCampaignError`` swallowed.
* DI missing → degraded_noop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import autotargeting_manager
from agent_runtime.jobs.autotargeting_manager import (
    EPK_SEARCH_CAMPAIGNS,
    REQUIRED_BRANDS,
    REQUIRED_CATEGORY,
    _detect_drift,
    run,
)
from agent_runtime.models import AutonomyLevel
from agent_runtime.tools.direct_api import ProtectedCampaignError
from agent_runtime.trust_levels import TrustLevel


def _force_auto_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``allowed_action`` to return AUTO for the autotargeting_set path.

    TODO(integration): ``autotargeting_set`` is not yet in
    :data:`agent_runtime.decision_engine.IRREVERSIBILITY` nor in
    :data:`agent_runtime.trust_levels.ASSISTED_AUTO_WHITELIST`. Wave 3
    task-10 follow-up will ship the wiring; this monkeypatch lets Task 21
    land with Decision 15 enforcement tests in place.
    """

    def _override(action_type: str, trust, decision_level):
        if action_type == "autotargeting_set":
            # FORBIDDEN_LOCK still wins; shadow still downgrades.
            if trust == TrustLevel.FORBIDDEN_LOCK:
                return AutonomyLevel.FORBIDDEN
            if trust == TrustLevel.SHADOW:
                return AutonomyLevel.NOTIFY
            return AutonomyLevel.AUTO
        # Non-target action types pass through.
        from agent_runtime.trust_levels import allowed_action as _orig

        return _orig(action_type, trust, decision_level)

    monkeypatch.setattr(autotargeting_manager, "allowed_action", _override)


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=list(EPK_SEARCH_CAMPAIGNS) + [709014142, 709307228],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )


def _mock_pool(
    *, trust_level: TrustLevel = TrustLevel.SHADOW
) -> tuple[MagicMock, list[tuple[Any, ...]]]:
    rows = [(trust_level.value,)]
    one_iter = iter(rows)

    async def _fetchone() -> Any:
        try:
            return next(one_iter)
        except StopIteration:
            return (1,)

    executed: list[tuple[Any, ...]] = []

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
    return pool, executed


def _direct_stub(
    *,
    adgroups_by_campaign: dict[int, list[dict[str, Any]]] | None = None,
    autotargeting_by_adgroup: dict[int, dict[str, Any] | None] | None = None,
    set_raises: BaseException | None = None,
    verify_blob: dict[str, Any] | None = None,  # what get_autotargeting returns AFTER set
) -> SimpleNamespace:
    """Build a DirectAPI-like stub with autotargeting extensions.

    ``autotargeting_by_adgroup[ag_id]`` is the payload returned by
    ``get_autotargeting(ag_id)``. ``verify_blob`` overrides the response after
    a successful ``set_autotargeting`` (used to simulate GET-after-SET
    mismatches). When unset, a successful set is reflected as a compliant
    settings blob — no drift.
    """
    adgroups_by_campaign = adgroups_by_campaign or {}
    at_map: dict[int, dict[str, Any] | None] = dict(autotargeting_by_adgroup or {})
    touched: dict[int, bool] = {}

    async def get_adgroups(*, campaign_id: int | None = None, **_: Any) -> list[dict[str, Any]]:
        return adgroups_by_campaign.get(campaign_id or 0, [])

    async def get_autotargeting(ad_group_id: int) -> Any:
        if touched.get(ad_group_id):
            # After successful set.
            if verify_blob is not None:
                return verify_blob
            return {"Category": REQUIRED_CATEGORY, "Brands": REQUIRED_BRANDS}
        return at_map.get(ad_group_id)

    async def set_autotargeting(ad_group_id: int, **_: Any) -> dict[str, Any]:
        if set_raises is not None:
            raise set_raises
        touched[ad_group_id] = True
        return {"ok": True}

    return SimpleNamespace(
        get_adgroups=AsyncMock(side_effect=get_adgroups),
        get_autotargeting=AsyncMock(side_effect=get_autotargeting),
        set_autotargeting=AsyncMock(side_effect=set_autotargeting),
    )


# ----------------------------------------------------------- _detect_drift


def test_detect_drift_compliant_returns_none() -> None:
    assert _detect_drift({"Category": REQUIRED_CATEGORY, "Brands": REQUIRED_BRANDS}) is None


def test_detect_drift_wider_category() -> None:
    drift = _detect_drift({"Category": "WIDER", "Brands": REQUIRED_BRANDS})
    assert drift == ("WIDER", REQUIRED_BRANDS)


def test_detect_drift_competitors_category() -> None:
    drift = _detect_drift({"Category": "COMPETITORS", "Brands": REQUIRED_BRANDS})
    assert drift == ("COMPETITORS", REQUIRED_BRANDS)


def test_detect_drift_with_brands() -> None:
    drift = _detect_drift({"Category": REQUIRED_CATEGORY, "Brands": "WithBrands"})
    assert drift == (REQUIRED_CATEGORY, "WithBrands")


def test_detect_drift_wrapped_payload() -> None:
    drift = _detect_drift({"AutotargetingSettings": {"Category": "WIDER", "Brands": "WithBrands"}})
    assert drift == ("WIDER", "WithBrands")


def test_detect_drift_missing_payload() -> None:
    assert _detect_drift(None) == ("<missing>", "<missing>")


def test_detect_drift_unparsable() -> None:
    assert _detect_drift("garbage") == ("<unparsable>", "<unparsable>")


# -------------------------------------------------------------------- run


@pytest.mark.asyncio
async def test_all_exact_withoutbrands_no_action() -> None:
    pool, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    adgroups = {cid: [{"Id": cid * 10 + i} for i in (1, 2)] for cid in EPK_SEARCH_CAMPAIGNS}
    at = {
        ag["Id"]: {"Category": REQUIRED_CATEGORY, "Brands": REQUIRED_BRANDS}
        for group in adgroups.values()
        for ag in group
    }
    direct = _direct_stub(adgroups_by_campaign=adgroups, autotargeting_by_adgroup=at)

    with patch.object(autotargeting_manager.telegram_tools, "send_message", AsyncMock()):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    assert result["status"] == "ok"
    assert result["action"] == "noop_no_drift"
    assert result["drift_detected"] == []
    assert result["rolled_back"] == []
    direct.set_autotargeting.assert_not_awaited()


@pytest.mark.asyncio
async def test_wider_category_triggers_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_auto_overlay(monkeypatch)
    pool, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    camp = EPK_SEARCH_CAMPAIGNS[0]
    ag_bad = camp * 10 + 1
    ag_good = camp * 10 + 2
    adgroups = {camp: [{"Id": ag_bad}, {"Id": ag_good}]}
    # Other two campaigns have no adgroups — not tested here.
    for other in EPK_SEARCH_CAMPAIGNS[1:]:
        adgroups[other] = []
    at = {
        ag_bad: {"Category": "WIDER", "Brands": REQUIRED_BRANDS},
        ag_good: {"Category": REQUIRED_CATEGORY, "Brands": REQUIRED_BRANDS},
    }
    direct = _direct_stub(adgroups_by_campaign=adgroups, autotargeting_by_adgroup=at)

    telegram_mock = AsyncMock(return_value=1)
    with patch.object(autotargeting_manager.telegram_tools, "send_message", telegram_mock):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    assert result["action"] == "auto_rolled_back"
    assert [r["ad_group_id"] for r in result["rolled_back"]] == [ag_bad]
    assert result["drift_detected"][0]["from"] == {
        "Category": "WIDER",
        "Brands": REQUIRED_BRANDS,
    }
    assert result["drift_detected"][0]["to"] == {
        "Category": REQUIRED_CATEGORY,
        "Brands": REQUIRED_BRANDS,
    }
    direct.set_autotargeting.assert_awaited_once_with(
        ag_bad, Category=REQUIRED_CATEGORY, Brands=REQUIRED_BRANDS
    )
    telegram_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_brands_drift_triggers_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_auto_overlay(monkeypatch)
    pool, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    camp = EPK_SEARCH_CAMPAIGNS[0]
    ag = camp * 10 + 1
    adgroups = {camp: [{"Id": ag}]}
    for other in EPK_SEARCH_CAMPAIGNS[1:]:
        adgroups[other] = []
    at = {ag: {"Category": REQUIRED_CATEGORY, "Brands": "WithBrands"}}
    direct = _direct_stub(adgroups_by_campaign=adgroups, autotargeting_by_adgroup=at)

    with patch.object(autotargeting_manager.telegram_tools, "send_message", AsyncMock()):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    assert [r["ad_group_id"] for r in result["rolled_back"]] == [ag]
    direct.set_autotargeting.assert_awaited_once()


@pytest.mark.asyncio
async def test_competitors_category_triggers_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_auto_overlay(monkeypatch)
    pool, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    camp = EPK_SEARCH_CAMPAIGNS[0]
    ag = camp * 10 + 1
    adgroups = {camp: [{"Id": ag}]}
    for other in EPK_SEARCH_CAMPAIGNS[1:]:
        adgroups[other] = []
    at = {ag: {"Category": "COMPETITORS", "Brands": REQUIRED_BRANDS}}
    direct = _direct_stub(adgroups_by_campaign=adgroups, autotargeting_by_adgroup=at)

    with patch.object(autotargeting_manager.telegram_tools, "send_message", AsyncMock()):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    assert [r["ad_group_id"] for r in result["rolled_back"]] == [ag]


@pytest.mark.asyncio
async def test_shadow_notifies_only() -> None:
    pool, executed = _mock_pool(trust_level=TrustLevel.SHADOW)
    camp = EPK_SEARCH_CAMPAIGNS[0]
    ag = camp * 10 + 1
    adgroups = {camp: [{"Id": ag}]}
    for other in EPK_SEARCH_CAMPAIGNS[1:]:
        adgroups[other] = []
    at = {ag: {"Category": "WIDER", "Brands": REQUIRED_BRANDS}}
    direct = _direct_stub(adgroups_by_campaign=adgroups, autotargeting_by_adgroup=at)

    telegram_mock = AsyncMock(return_value=1)
    with patch.object(autotargeting_manager.telegram_tools, "send_message", telegram_mock):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    assert result["rolled_back"] == []
    assert [r["ad_group_id"] for r in result["would_rollback"]] == [ag]
    direct.set_autotargeting.assert_not_awaited()
    telegram_mock.assert_awaited_once()
    # Audit row with is_mutation=false
    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts
    assert False in audit_inserts[0][1]


@pytest.mark.asyncio
async def test_dry_run_no_side_effects() -> None:
    pool, executed = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    camp = EPK_SEARCH_CAMPAIGNS[0]
    ag = camp * 10 + 1
    adgroups = {camp: [{"Id": ag}]}
    for other in EPK_SEARCH_CAMPAIGNS[1:]:
        adgroups[other] = []
    at = {ag: {"Category": "WIDER", "Brands": REQUIRED_BRANDS}}
    direct = _direct_stub(adgroups_by_campaign=adgroups, autotargeting_by_adgroup=at)

    telegram_mock = AsyncMock(return_value=1)
    with patch.object(autotargeting_manager.telegram_tools, "send_message", telegram_mock):
        result = await run(
            pool,
            direct=direct,
            http_client=SimpleNamespace(),
            settings=_settings(),
            dry_run=True,
        )

    assert result["dry_run"] is True
    assert result["rolled_back"] == []
    assert [r["ad_group_id"] for r in result["would_rollback"]] == [ag]
    direct.set_autotargeting.assert_not_awaited()
    telegram_mock.assert_not_awaited()
    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts == []


@pytest.mark.asyncio
async def test_get_after_set_verify_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_autotargeting returns ok, but the GET-after-SET still shows drift.

    Expected: errors row with autotargeting_set_verify_fail; no rolled_back.
    """
    _force_auto_overlay(monkeypatch)
    pool, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    camp = EPK_SEARCH_CAMPAIGNS[0]
    ag = camp * 10 + 1
    adgroups = {camp: [{"Id": ag}]}
    for other in EPK_SEARCH_CAMPAIGNS[1:]:
        adgroups[other] = []
    at = {ag: {"Category": "WIDER", "Brands": REQUIRED_BRANDS}}
    direct = _direct_stub(
        adgroups_by_campaign=adgroups,
        autotargeting_by_adgroup=at,
        verify_blob={"Category": "WIDER", "Brands": REQUIRED_BRANDS},  # still drifted
    )

    with patch.object(autotargeting_manager.telegram_tools, "send_message", AsyncMock()):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    assert result["rolled_back"] == []
    assert result["errors"][0]["error"] == "autotargeting_set_verify_fail"
    assert result["errors"][0]["ad_group_id"] == ag


@pytest.mark.asyncio
async def test_only_epk_search_campaigns_checked() -> None:
    """get_adgroups must be called exactly for EPK_SEARCH_CAMPAIGNS (456/457/458).

    РСЯ campaigns (709014142, 709307228) are in PROTECTED_CAMPAIGN_IDS but
    must NOT be passed to get_adgroups here.
    """
    pool, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    adgroups: dict[int, list[dict[str, Any]]] = {cid: [] for cid in EPK_SEARCH_CAMPAIGNS}
    direct = _direct_stub(adgroups_by_campaign=adgroups)

    await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    called_campaigns = {
        call.kwargs.get("campaign_id") for call in direct.get_adgroups.await_args_list
    }
    assert called_campaigns == set(EPK_SEARCH_CAMPAIGNS)
    # RSYA не должны быть опрошены
    assert 709014142 not in called_campaigns
    assert 709307228 not in called_campaigns


@pytest.mark.asyncio
async def test_protected_campaign_error_surfaces_as_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_auto_overlay(monkeypatch)
    pool, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    camp = EPK_SEARCH_CAMPAIGNS[0]
    ag = camp * 10 + 1
    adgroups = {camp: [{"Id": ag}]}
    for other in EPK_SEARCH_CAMPAIGNS[1:]:
        adgroups[other] = []
    at = {ag: {"Category": "WIDER", "Brands": REQUIRED_BRANDS}}
    direct = _direct_stub(
        adgroups_by_campaign=adgroups,
        autotargeting_by_adgroup=at,
        set_raises=ProtectedCampaignError("blocked"),
    )

    with patch.object(autotargeting_manager.telegram_tools, "send_message", AsyncMock()):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    assert result["rolled_back"] == []
    assert result["errors"][0]["error"] == "protected_guard"


@pytest.mark.asyncio
async def test_degraded_noop_without_di() -> None:
    pool, executed = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    result = await run(pool)
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts == []


@pytest.mark.asyncio
async def test_degraded_when_get_autotargeting_wrapper_missing() -> None:
    """When DirectAPI lacks get_autotargeting, job returns degraded_wrapper_missing.

    Keeps the cron endpoint green while Task 7 integration is pending.
    """
    pool, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = SimpleNamespace(get_adgroups=AsyncMock(return_value=[]))
    result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())
    assert result["status"] == "degraded_wrapper_missing"


@pytest.mark.asyncio
async def test_audit_log_written_with_drift_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_auto_overlay(monkeypatch)
    pool, executed = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    camp = EPK_SEARCH_CAMPAIGNS[0]
    ag = camp * 10 + 1
    adgroups = {camp: [{"Id": ag}]}
    for other in EPK_SEARCH_CAMPAIGNS[1:]:
        adgroups[other] = []
    at = {ag: {"Category": "COMPETITORS", "Brands": "WithBrands"}}
    direct = _direct_stub(adgroups_by_campaign=adgroups, autotargeting_by_adgroup=at)

    with patch.object(autotargeting_manager.telegram_tools, "send_message", AsyncMock()):
        await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts, "audit_log INSERT missing"
    # is_mutation must be True (rollback happened).
    assert True in audit_inserts[0][1]
