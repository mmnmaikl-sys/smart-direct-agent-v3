"""Unit tests for DirectAPI — all I/O mocked via httpx.MockTransport.

Three properties we care about here:
    1. PROTECTED_CAMPAIGN_IDS never reaches the wire (counted via transport hits).
    2. Money round-trips through ×1M / ÷1M correctly.
    3. Error-code translation obeys the mapping table in the module docstring.

The AST pairing script is also exercised from Python so regressions show up
in pytest, not only in CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from agent_runtime.config import Settings
from agent_runtime.tools.direct_api import (
    DirectAPI,
    InvalidRequestError,
    ProtectedCampaignError,
    RateLimitError,
    TokenExpiredError,
    _prune_readonly_strategy_fields,
)

_PROTECTED = [708978456, 708978457, 708978458, 709014142, 709307228]


def _make_settings(**overrides) -> Settings:
    base = dict(
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        YANDEX_DIRECT_TOKEN="test-direct-token",
        PROTECTED_CAMPAIGN_IDS=_PROTECTED,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class _Recorder:
    """Route-aware MockTransport responder + call counter."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._handlers: list = []

    def reply(self, matcher, response):
        self._handlers.append((matcher, response))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        service = request.url.path.rsplit("/", 1)[-1]
        self.calls.append({"service": service, "method": body.get("method"), "body": body})
        for matcher, response in self._handlers:
            if matcher(service, body):
                if callable(response):
                    return response(request, body)
                return response
        return httpx.Response(200, json={"result": {}})


def _client_with(recorder: _Recorder, settings: Settings) -> DirectAPI:
    api = DirectAPI(settings)
    # Inject our own httpx client so __aenter__ doesn't build one.
    api._client = httpx.AsyncClient(
        transport=httpx.MockTransport(recorder),
        headers=api._headers(),
        base_url=settings.YANDEX_DIRECT_BASE_URL,
    )
    return api


# --- Protected campaigns -----------------------------------------------------


@pytest.mark.asyncio
async def test_protected_campaign_blocks_pause_campaign() -> None:
    recorder = _Recorder()
    api = _client_with(recorder, _make_settings())
    try:
        with pytest.raises(ProtectedCampaignError):
            await api.pause_campaign(_PROTECTED[0])
    finally:
        await api._client.aclose()
    assert recorder.calls == [], "no HTTP request should leave the process"


@pytest.mark.asyncio
async def test_protected_campaign_blocks_via_adgroup_lookup() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "adgroups" and body["method"] == "get",
        httpx.Response(
            200,
            json={"result": {"AdGroups": [{"Id": 999, "CampaignId": _PROTECTED[0]}]}},
        ),
    )
    api = _client_with(recorder, _make_settings())
    try:
        with pytest.raises(ProtectedCampaignError):
            await api.pause_group(999)
    finally:
        await api._client.aclose()
    # Lookup was fired (GET), but no mutation GET leaked through.
    assert any(c["service"] == "adgroups" and c["method"] == "get" for c in recorder.calls)
    assert not any(c["service"] == "adgroups" and c["method"] == "suspend" for c in recorder.calls)


# --- Money conversions -------------------------------------------------------


@pytest.mark.asyncio
async def test_set_bid_multiplies_by_1M() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "keywords" and body["method"] == "get",
        httpx.Response(200, json={"result": {"Keywords": [{"Id": 111, "AdGroupId": 222}]}}),
    )
    recorder.reply(
        lambda svc, body: svc == "adgroups" and body["method"] == "get",
        httpx.Response(
            200,
            json={"result": {"AdGroups": [{"Id": 222, "CampaignId": 999999999}]}},
        ),
    )
    api = _client_with(recorder, _make_settings())
    try:
        await api.set_bid(111, bid_rub=15)
    finally:
        await api._client.aclose()
    bids_call = next(c for c in recorder.calls if c["service"] == "bids")
    assert bids_call["body"]["params"]["Bids"][0]["SearchBid"] == 15_000_000


@pytest.mark.asyncio
async def test_verify_bid_divides_by_1M() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: (
            svc == "keywords"
            and body["method"] == "get"
            and "Ids" in body["params"]["SelectionCriteria"]
        ),
        httpx.Response(200, json={"result": {"Keywords": [{"Id": 111, "AdGroupId": 222}]}}),
    )
    recorder.reply(
        lambda svc, body: (
            svc == "keywords"
            and body["method"] == "get"
            and "AdGroupIds" in body["params"]["SelectionCriteria"]
        ),
        httpx.Response(
            200,
            json={"result": {"Keywords": [{"Id": 111, "AdGroupId": 222, "Bid": 15_000_000}]}},
        ),
    )
    api = _client_with(recorder, _make_settings())
    try:
        ok = await api.verify_bid(111, expected_bid_rub=15)
    finally:
        await api._client.aclose()
    assert ok is True


@pytest.mark.asyncio
async def test_verify_bid_mismatch_returns_false() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "keywords" and "Ids" in body["params"]["SelectionCriteria"],
        httpx.Response(200, json={"result": {"Keywords": [{"Id": 111, "AdGroupId": 222}]}}),
    )
    recorder.reply(
        lambda svc, body: svc == "keywords" and "AdGroupIds" in body["params"]["SelectionCriteria"],
        httpx.Response(
            200,
            json={"result": {"Keywords": [{"Id": 111, "AdGroupId": 222, "Bid": 15_000_000}]}},
        ),
    )
    api = _client_with(recorder, _make_settings())
    try:
        ok = await api.verify_bid(111, expected_bid_rub=20)
    finally:
        await api._client.aclose()
    assert ok is False


# --- add_negatives merges ----------------------------------------------------


@pytest.mark.asyncio
async def test_add_negatives_merges_with_existing() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns" and body["method"] == "get",
        httpx.Response(
            200,
            json={
                "result": {
                    "Campaigns": [{"Id": 555, "NegativeKeywords": {"Items": ["cat", "dog"]}}]
                }
            },
        ),
    )
    api = _client_with(recorder, _make_settings())
    try:
        await api.add_negatives(555, ["dog", "fish"])
    finally:
        await api._client.aclose()
    update_call = next(c for c in recorder.calls if c["method"] == "update")
    items = update_call["body"]["params"]["Campaigns"][0]["NegativeKeywords"]["Items"]
    assert items == ["cat", "dog", "fish"]


# --- Typed exception mapping -------------------------------------------------


@pytest.mark.asyncio
async def test_token_expired_raises_typed_error() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: True,
        httpx.Response(200, json={"error": {"error_code": 53, "error_string": "Token expired"}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        with pytest.raises(TokenExpiredError):
            await api.get_campaigns([111])
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_invalid_request_raises_typed_error() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: True,
        httpx.Response(200, json={"error": {"error_code": 8300, "error_string": "state conflict"}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        with pytest.raises(InvalidRequestError):
            await api.get_campaigns([111])
    finally:
        await api._client.aclose()


# --- Retry behaviour ---------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_retried_and_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Skip the retry sleeps for speed.
    monkeypatch.setattr("agent_runtime.tools.direct_api.asyncio.sleep", _noop_sleep())
    recorder = _Recorder()
    recorder.reply(lambda svc, body: True, httpx.Response(429, json={}))
    api = _client_with(recorder, _make_settings())
    try:
        with pytest.raises(RateLimitError):
            await api.get_campaigns([111])
    finally:
        await api._client.aclose()
    assert len(recorder.calls) == 3, f"expected 3 attempts, got {len(recorder.calls)}"


@pytest.mark.asyncio
async def test_retryable_5xx_succeeds_on_second_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.tools.direct_api.asyncio.sleep", _noop_sleep())
    # First 500, then 200 with a campaign payload.
    state = {"count": 0}

    def handler(request: httpx.Request, body: dict) -> httpx.Response:
        state["count"] += 1
        if state["count"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"result": {"Campaigns": [{"Id": 111, "Name": "x"}]}})

    recorder = _Recorder()
    recorder.reply(lambda svc, body: True, handler)
    api = _client_with(recorder, _make_settings())
    try:
        campaigns = await api.get_campaigns([111])
    finally:
        await api._client.aclose()
    assert state["count"] == 2
    assert campaigns[0]["Id"] == 111


# --- BidModifiers Levels inside SelectionCriteria ---------------------------


@pytest.mark.asyncio
async def test_bid_modifiers_levels_inside_selection_criteria() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: True, httpx.Response(200, json={"result": {"BidModifiers": []}})
    )
    api = _client_with(recorder, _make_settings())
    try:
        await api.get_bid_modifiers([708978456])
    finally:
        await api._client.aclose()
    body = recorder.calls[0]["body"]["params"]
    assert "Levels" in body["SelectionCriteria"]
    assert "Levels" not in body


# --- Strategy prune helper --------------------------------------------------


def test_prune_readonly_strategy_fields_removes_budget_type() -> None:
    strategy = {
        "Search": {
            "BiddingStrategyType": "AVERAGE_CPA",
            "AverageCpa": {"AverageCpa": 80_000_000},
            "BudgetType": "WEEKLY_BUDGET",
        },
        "Network": {"BiddingStrategyType": "SERVING_OFF", "BudgetType": "WEEKLY_BUDGET"},
        "AttributionModel": "LYDC",
    }
    pruned = _prune_readonly_strategy_fields(strategy)
    assert "BudgetType" not in pruned["Search"]
    assert "BudgetType" not in pruned["Network"]
    assert "AttributionModel" not in pruned
    assert pruned["Search"]["AverageCpa"]["AverageCpa"] == 80_000_000


# --- Config: protected list empty fails fast ---------------------------------


def test_empty_protected_ids_fail_fast() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _make_settings(PROTECTED_CAMPAIGN_IDS=[])


# --- AST pairing script ------------------------------------------------------


ROOT = Path(__file__).resolve().parent.parent.parent


def test_ast_pairing_script_all_set_have_verify() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_verify_pairing.py"),
            "--module",
            str(ROOT / "agent_runtime" / "tools" / "direct_api.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"script stderr: {result.stderr}"
    assert "OK:" in result.stdout


def test_ast_pairing_script_detects_orphan(tmp_path: Path) -> None:
    orphan = tmp_path / "orphan.py"
    orphan.write_text(
        "class Foo:\n    async def set_foo(self):\n        pass\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_verify_pairing.py"),
            "--module",
            str(orphan),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "MUTATION WITHOUT MAPPING" in result.stderr


# ---------------------------------------------------------------------- helpers


def _noop_sleep():
    async def _sleep(_secs: float) -> None:  # pragma: no cover - trivial
        return None

    return _sleep


# --- Additional coverage ----------------------------------------------------


@pytest.mark.asyncio
async def test_aenter_builds_client_and_aexit_closes_it() -> None:
    settings = _make_settings()
    api = DirectAPI(settings)
    assert api._client is None
    async with api as entered:
        assert entered._client is not None
    assert api._client is None


@pytest.mark.asyncio
async def test_call_outside_context_raises() -> None:
    api = DirectAPI(_make_settings())
    with pytest.raises(RuntimeError, match="async with"):
        await api._call("campaigns", "get", {})


def test_missing_token_header_raises() -> None:
    settings = _make_settings(YANDEX_DIRECT_TOKEN="")
    api = DirectAPI(settings)
    with pytest.raises(RuntimeError, match="YANDEX_DIRECT_TOKEN"):
        api._headers()


@pytest.mark.asyncio
async def test_get_campaigns_returns_list() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns",
        httpx.Response(200, json={"result": {"Campaigns": [{"Id": 1}, {"Id": 2}]}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        campaigns = await api.get_campaigns([1, 2])
    finally:
        await api._client.aclose()
    assert [c["Id"] for c in campaigns] == [1, 2]


@pytest.mark.asyncio
async def test_get_adgroups_with_ids_and_campaign_id() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "adgroups",
        httpx.Response(200, json={"result": {"AdGroups": [{"Id": 999}]}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        await api.get_adgroups(campaign_id=123, ids=[999])
    finally:
        await api._client.aclose()
    params = recorder.calls[0]["body"]["params"]["SelectionCriteria"]
    assert params["Ids"] == [999]
    assert params["CampaignIds"] == [123]


@pytest.mark.asyncio
async def test_get_ads_and_keywords_pass_ad_group_ids() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "ads",
        httpx.Response(200, json={"result": {"Ads": [{"Id": 1, "AdGroupId": 9}]}}),
    )
    recorder.reply(
        lambda svc, body: svc == "keywords",
        httpx.Response(200, json={"result": {"Keywords": [{"Id": 2, "AdGroupId": 9}]}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        ads = await api.get_ads([9])
        keywords = await api.get_keywords([9])
    finally:
        await api._client.aclose()
    assert ads[0]["AdGroupId"] == 9
    assert keywords[0]["AdGroupId"] == 9


@pytest.mark.asyncio
async def test_get_campaign_negative_keywords_parses_nested() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns",
        httpx.Response(
            200,
            json={
                "result": {
                    "Campaigns": [
                        {"Id": 1, "NegativeKeywords": {"Items": ["a", "b"]}},
                    ]
                }
            },
        ),
    )
    api = _client_with(recorder, _make_settings())
    try:
        negs = await api.get_campaign_negative_keywords(1)
    finally:
        await api._client.aclose()
    assert negs == ["a", "b"]


@pytest.mark.asyncio
async def test_get_campaign_negative_keywords_empty_when_no_campaign() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns",
        httpx.Response(200, json={"result": {"Campaigns": []}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        negs = await api.get_campaign_negative_keywords(1)
    finally:
        await api._client.aclose()
    assert negs == []


@pytest.mark.asyncio
async def test_verify_negatives_is_subset_check() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns",
        httpx.Response(
            200,
            json={
                "result": {"Campaigns": [{"Id": 1, "NegativeKeywords": {"Items": ["a", "b", "c"]}}]}
            },
        ),
    )
    api = _client_with(recorder, _make_settings())
    try:
        assert await api.verify_negatives_added(1, ["a", "b"]) is True
        assert await api.verify_negatives_added(1, ["x"]) is False
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_pause_and_verify_group_paused() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "adgroups" and body["method"] == "get",
        httpx.Response(
            200,
            json={"result": {"AdGroups": [{"Id": 500, "CampaignId": 999, "Status": "SUSPENDED"}]}},
        ),
    )
    recorder.reply(
        lambda svc, body: svc == "adgroups" and body["method"] == "suspend",
        httpx.Response(200, json={"result": {}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        await api.pause_group(500)
        assert await api.verify_group_paused(500) is True
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_resume_group_and_verify() -> None:
    state = {"status": "SUSPENDED"}

    def handler(req, body):
        svc = req.url.path.rsplit("/", 1)[-1]
        if svc == "adgroups" and body["method"] == "resume":
            state["status"] = "ON"
            return httpx.Response(200, json={"result": {}})
        return httpx.Response(
            200,
            json={
                "result": {"AdGroups": [{"Id": 500, "CampaignId": 999, "Status": state["status"]}]}
            },
        )

    recorder = _Recorder()
    recorder.reply(lambda svc, body: True, handler)
    api = _client_with(recorder, _make_settings())
    try:
        await api.resume_group(500)
        assert await api.verify_group_resumed(500) is True
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_pause_campaign_and_verify() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns" and body["method"] == "suspend",
        httpx.Response(200, json={"result": {}}),
    )
    recorder.reply(
        lambda svc, body: svc == "campaigns" and body["method"] == "get",
        httpx.Response(200, json={"result": {"Campaigns": [{"Id": 12345, "State": "SUSPENDED"}]}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        await api.pause_campaign(12345)
        assert await api.verify_campaign_paused(12345) is True
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_resume_campaign_rejects_draft() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns",
        httpx.Response(
            200,
            json={"result": {"Campaigns": [{"Id": 12345, "State": "OFF", "Status": "DRAFT"}]}},
        ),
    )
    api = _client_with(recorder, _make_settings())
    try:
        with pytest.raises(InvalidRequestError, match="DRAFT"):
            await api.resume_campaign(12345)
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_resume_campaign_happy_path() -> None:
    state = {"State": "SUSPENDED", "Status": "ACCEPTED"}

    def handler(req, body):
        svc = req.url.path.rsplit("/", 1)[-1]
        if svc == "campaigns" and body["method"] == "resume":
            state["State"] = "ON"
            return httpx.Response(200, json={"result": {}})
        return httpx.Response(200, json={"result": {"Campaigns": [{"Id": 12345, **state}]}})

    recorder = _Recorder()
    recorder.reply(lambda svc, body: True, handler)
    api = _client_with(recorder, _make_settings())
    try:
        await api.resume_campaign(12345)
        assert await api.verify_campaign_resumed(12345) is True
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_update_ad_href_and_verify() -> None:
    state = {"href": "https://old.example.com/"}

    def handler(req, body):
        svc = req.url.path.rsplit("/", 1)[-1]
        if svc == "ads" and body["method"] == "update":
            state["href"] = body["params"]["Ads"][0]["TextAd"]["Href"]
            return httpx.Response(200, json={"result": {}})
        if svc == "ads" and body["method"] == "get":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "Ads": [{"Id": 77, "AdGroupId": 88, "TextAd": {"Href": state["href"]}}]
                    }
                },
            )
        if svc == "adgroups":
            return httpx.Response(
                200, json={"result": {"AdGroups": [{"Id": 88, "CampaignId": 999}]}}
            )
        return httpx.Response(200, json={"result": {}})

    recorder = _Recorder()
    recorder.reply(lambda svc, body: True, handler)
    api = _client_with(recorder, _make_settings())
    try:
        await api.update_ad_href(77, "https://new.example.com/")
        assert await api.verify_ad_href(77, "https://new.example.com/") is True
        assert await api.verify_ad_href(77, "https://old.example.com/") is False
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_update_strategy_prunes_readonly() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns",
        httpx.Response(200, json={"result": {}}),
    )
    api = _client_with(recorder, _make_settings())
    strategy = {
        "Search": {
            "BiddingStrategyType": "AVERAGE_CPA",
            "AverageCpa": {"AverageCpa": 80_000_000},
            "BudgetType": "WEEKLY_BUDGET",
        },
        "AttributionModel": "LYDC",
    }
    try:
        await api.update_strategy(12345, strategy)
    finally:
        await api._client.aclose()
    sent = recorder.calls[0]["body"]["params"]["Campaigns"][0]["Strategy"]
    assert "BudgetType" not in sent["Search"]
    assert "AttributionModel" not in sent


@pytest.mark.asyncio
async def test_verify_strategy_ignores_readonly_mismatch() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "campaigns",
        httpx.Response(
            200,
            json={
                "result": {
                    "Campaigns": [
                        {
                            "Id": 12345,
                            "Strategy": {
                                "Search": {
                                    "BiddingStrategyType": "AVERAGE_CPA",
                                    "AverageCpa": {"AverageCpa": 80_000_000},
                                    "BudgetType": "WEEKLY_BUDGET",
                                },
                                "AttributionModel": "LYDC",
                            },
                        }
                    ]
                }
            },
        ),
    )
    api = _client_with(recorder, _make_settings())
    expected = {
        "Search": {
            "BiddingStrategyType": "AVERAGE_CPA",
            "AverageCpa": {"AverageCpa": 80_000_000},
        }
    }
    try:
        assert await api.verify_strategy(12345, expected) is True
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_transport_error_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.tools.direct_api.asyncio.sleep", _noop_sleep())

    def handler(req, body):
        raise httpx.ConnectError("boom")

    recorder = _Recorder()
    recorder.reply(lambda svc, body: True, handler)
    api = _client_with(recorder, _make_settings())
    try:
        from agent_runtime.tools import UnknownDirectAPIError

        with pytest.raises(UnknownDirectAPIError, match="transport error"):
            await api.get_campaigns([1])
    finally:
        await api._client.aclose()


@pytest.mark.asyncio
async def test_adgroup_campaign_lookup_cached() -> None:
    recorder = _Recorder()
    recorder.reply(
        lambda svc, body: svc == "adgroups",
        httpx.Response(200, json={"result": {"AdGroups": [{"Id": 500, "CampaignId": 900000000}]}}),
    )
    api = _client_with(recorder, _make_settings())
    try:
        cid1 = await api._resolve_campaign_for_adgroup(500)
        cid2 = await api._resolve_campaign_for_adgroup(500)
    finally:
        await api._client.aclose()
    assert cid1 == cid2 == 900000000
    # Only one GET despite two lookups — cache hit on the second.
    adgroup_gets = [c for c in recorder.calls if c["service"] == "adgroups"]
    assert len(adgroup_gets) == 1


def test_ttlmap_expires_and_evicts() -> None:
    from agent_runtime.tools.direct_api import _TTLMap

    m = _TTLMap(ttl_sec=10)
    m.set(1, 42, now=100.0)
    assert m.get(1, now=105.0) == 42
    assert m.get(1, now=200.0) is None
    assert m.get(1, now=300.0) is None  # evicted
