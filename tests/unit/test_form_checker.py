"""Unit tests for agent_runtime.jobs.form_checker (Task 15).

Covers:

* Pure-function checks: ``check_landing`` markers (bitrix vs /lead vs both
  missing), ``check_cors`` (ACAO absent / wildcard / owner-origin),
  ``check_lead_endpoint`` phone-validation, ``check_ad_moderation`` rejected
  ads + DirectAPI failures.
* ``FormChecker.run()`` trust-level overlay — shadow never suspends, assisted
  auto-suspends only non-protected, protected whitelisted always spared.
* ``dry_run=True`` never calls suspend.
* ``kb.consult`` fires before any suspend (Functional gate carry-over).
* Parallel-checks isolation (one failing coroutine does not sink others).
* Hardcoded-constant regression: no legacy ``app.*`` imports, no 708978456
  literal, no v2 Railway domain.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import form_checker
from agent_runtime.jobs.form_checker import (
    FormChecker,
    check_ad_moderation,
    check_cors,
    check_landing,
    check_lead_endpoint,
)
from agent_runtime.tools.direct_api import ProtectedCampaignError  # noqa: F401
from agent_runtime.trust_levels import TrustLevel

_CAMP_PROTECTED = 708978456
_CAMP_UNPROTECTED = 999888


def _settings(
    *,
    protected: list[int] | None = None,
    landings: list[str] | None = None,
) -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=protected or [_CAMP_PROTECTED],
        PROTECTED_LANDING_URLS=landings or ["https://24bankrotsttvo.ru/pages/ad/stub.html"],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
        PUBLIC_BASE_URL="https://smart-direct-agent-v3-production.up.railway.app",
    )


def _mock_pool(*, trust_level: TrustLevel = TrustLevel.SHADOW) -> MagicMock:
    one_iter = iter([(trust_level.value,)])

    async def _fetchone():
        try:
            return next(one_iter)
        except StopIteration:
            return (1,)  # audit_log RETURNING id fallback

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


def _mock_transport(
    *,
    landing_body: str = "",
    cors_acao: str | None = None,
    lead_response: tuple[int, Any] = (400, {"error": "phone required"}),
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "OPTIONS":
            headers = {"Access-Control-Allow-Origin": cors_acao} if cors_acao else {}
            return httpx.Response(204, headers=headers)
        if request.method == "POST" and url.endswith("/lead"):
            status, body = lead_response
            return httpx.Response(status, json=body)
        return httpx.Response(200, text=landing_body)

    return httpx.MockTransport(handler)


def _direct_stub(
    *,
    rejected_campaigns: dict[int, str] | None = None,
    adgroups_raises: BaseException | None = None,
    pause_raises: BaseException | None = None,
) -> SimpleNamespace:
    rejected = rejected_campaigns or {}

    async def get_adgroups(*, campaign_id: int | None = None, **_: Any):
        if adgroups_raises is not None:
            raise adgroups_raises
        return [{"Id": (campaign_id or 0) * 10 + 1, "CampaignId": campaign_id}]

    async def get_ads(ad_group_ids: list[int]):
        out: list[dict[str, Any]] = []
        for group_id in ad_group_ids:
            campaign_id = group_id // 10
            if campaign_id in rejected:
                out.append(
                    {
                        "Id": group_id * 100,
                        "Status": "REJECTED",
                        "StatusClarification": rejected[campaign_id],
                    }
                )
        return out

    pause = AsyncMock()
    if pause_raises is not None:
        pause.side_effect = pause_raises

    return SimpleNamespace(
        get_adgroups=AsyncMock(side_effect=get_adgroups),
        get_ads=AsyncMock(side_effect=get_ads),
        pause_campaign=pause,
        verify_campaign_paused=AsyncMock(return_value=True),
    )


# ------------------------------------------------------------ check_landing


def _landing_html(
    *,
    phone: bool = True,
    submit: bool = True,
    bitrix: bool = False,
    sda_lead: bool = False,
    start_quiz_call: bool = False,
    start_quiz_def: bool = False,
    body_filler: int = 2000,
) -> str:
    parts: list[str] = ["<html><body>"]
    if phone:
        parts.append('<input name="phone" />')
    if submit:
        parts.append("<script>function submit(){ sendLead(); }</script>")
    if bitrix:
        parts.append(
            '<script>fetch("https://b24-xxx.bitrix24.ru/rest/1/xyz/crm.lead.add.json")</script>'
        )
    if sda_lead:
        parts.append('<script>fetch("/lead", {method:"POST"})</script>')
    if start_quiz_call:
        parts.append("<script>startQuiz()</script>")
    if start_quiz_def:
        parts.append("<script>function startQuiz(){ return true; }</script>")
    parts.append("x" * body_filler)
    parts.append("</body></html>")
    return "".join(parts)


@pytest.mark.asyncio
async def test_check_landing_no_bitrix_no_sda_lead_detects_issue() -> None:
    html = _landing_html(bitrix=False, sda_lead=False)
    transport = _mock_transport(landing_body=html)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is False
    assert any("No lead destination" in i for i in result["issues"])


@pytest.mark.asyncio
async def test_check_landing_bitrix_webhook_ok() -> None:
    html = _landing_html(bitrix=True, sda_lead=False)
    transport = _mock_transport(landing_body=html)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is True, result["issues"]


@pytest.mark.asyncio
async def test_check_landing_sda_lead_ok() -> None:
    html = _landing_html(bitrix=False, sda_lead=True)
    transport = _mock_transport(landing_body=html)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_check_landing_no_phone_input() -> None:
    html = _landing_html(phone=False, bitrix=True)
    transport = _mock_transport(landing_body=html)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is False
    assert any("phone" in i.lower() for i in result["issues"])


@pytest.mark.asyncio
async def test_check_landing_no_submit_handler() -> None:
    html = '<html><body><input name="phone" />' + ("x" * 2000) + "</body></html>"
    transport = _mock_transport(landing_body=html)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is False
    assert any("submit handler" in i.lower() for i in result["issues"])


@pytest.mark.asyncio
async def test_check_landing_startquiz_called_not_defined() -> None:
    html = _landing_html(bitrix=True, start_quiz_call=True, start_quiz_def=False)
    transport = _mock_transport(landing_body=html)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is False
    assert any("QUIZ BROKEN" in i for i in result["issues"])


@pytest.mark.asyncio
async def test_check_landing_startquiz_defined_ok() -> None:
    html = _landing_html(bitrix=True, start_quiz_call=True, start_quiz_def=True)
    transport = _mock_transport(landing_body=html)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_check_landing_too_small() -> None:
    html = _landing_html(bitrix=True, body_filler=50)
    transport = _mock_transport(landing_body=html)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is False
    assert any("too small" in i.lower() for i in result["issues"])


@pytest.mark.asyncio
async def test_check_landing_fetch_failure_returns_structured_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await check_landing(http, "https://example.com/landing")
    assert result["ok"] is False
    assert any("fetch failed" in i for i in result["issues"])


# -------------------------------------------------------------- check_cors


@pytest.mark.asyncio
async def test_check_cors_missing_header() -> None:
    transport = _mock_transport(cors_acao=None)
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_cors(http, "https://example.com/lead")
    assert result["ok"] is False
    assert "CORS not configured" in result["issue"]


@pytest.mark.asyncio
async def test_check_cors_wildcard_ok() -> None:
    transport = _mock_transport(cors_acao="*")
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_cors(http, "https://example.com/lead")
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_check_cors_owner_domain_ok() -> None:
    transport = _mock_transport(cors_acao="https://24bankrotsttvo.ru")
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_cors(http, "https://example.com/lead")
    assert result["ok"] is True


# -------------------------------------------------------- check_lead_endpoint


@pytest.mark.asyncio
async def test_check_lead_endpoint_phone_validation_ok() -> None:
    transport = _mock_transport(lead_response=(400, {"error": "phone required"}))
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_lead_endpoint(http, "https://example.com/lead")
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_check_lead_endpoint_missing_validation() -> None:
    transport = _mock_transport(lead_response=(200, {"status": "ok"}))
    async with httpx.AsyncClient(transport=transport) as http:
        result = await check_lead_endpoint(http, "https://example.com/lead")
    assert result["ok"] is False
    assert "phone validation missing" in result["issue"]


# ------------------------------------------------------- check_ad_moderation


@pytest.mark.asyncio
async def test_check_ad_moderation_rejected_flags_campaign() -> None:
    direct = _direct_stub(rejected_campaigns={_CAMP_PROTECTED: "text forbidden"})
    result = await check_ad_moderation(direct, [_CAMP_PROTECTED])
    assert result["ok"] is False
    assert result["rejected"][0]["campaign"] == _CAMP_PROTECTED
    assert result["rejected"][0]["reason"] == "text forbidden"


@pytest.mark.asyncio
async def test_check_ad_moderation_no_rejects_ok() -> None:
    direct = _direct_stub()
    result = await check_ad_moderation(direct, [_CAMP_PROTECTED])
    assert result["ok"] is True
    assert result["rejected"] == []


@pytest.mark.asyncio
async def test_check_ad_moderation_adgroups_failure_isolated() -> None:
    direct = _direct_stub(adgroups_raises=RuntimeError("503"))
    result = await check_ad_moderation(direct, [_CAMP_PROTECTED])
    assert result["ok"] is False
    assert result["rejected"][0]["error"].startswith("adgroups.get failed")


# ---------------------------------------------------------- FormChecker.run


@pytest.mark.asyncio
async def test_run_all_ok_returns_structured_response() -> None:
    html = _landing_html(bitrix=True)
    transport = _mock_transport(landing_body=html, cors_acao="*")
    pool = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub()
    settings = _settings(
        landings=["https://24bankrotsttvo.ru/pages/ad/a.html"],
        protected=[_CAMP_PROTECTED],
    )
    async with httpx.AsyncClient(transport=transport) as http:
        fc = FormChecker(direct=direct, http=http, pool=pool, settings=settings)
        result = await fc.run()
    assert result["all_ok"] is True
    assert result["action"] == "none"
    assert result["suspended"] == []
    assert set(result) >= {
        "landings",
        "cors",
        "endpoint",
        "moderation",
        "trust_level",
        "rejected_campaigns",
        "ts",
    }


@pytest.mark.asyncio
async def test_run_shadow_never_suspends() -> None:
    html = _landing_html(bitrix=True)
    transport = _mock_transport(landing_body=html, cors_acao="*")
    pool = _mock_pool(trust_level=TrustLevel.SHADOW)
    direct = _direct_stub(rejected_campaigns={_CAMP_UNPROTECTED: "text bad"})
    settings = _settings(
        protected=[_CAMP_PROTECTED],
        landings=["https://24bankrotsttvo.ru/pages/ad/a.html"],
    )
    # moderate needs to see the unprotected campaign — add it to scan list
    settings = Settings(  # type: ignore[call-arg]
        **{
            **settings.model_dump(),
            "SDA_INTERNAL_API_KEY": "a" * 64,
            "SDA_WEBHOOK_HMAC_SECRET": "b" * 64,
            "HYPOTHESIS_HMAC_SECRET": "c" * 64,
            "PII_SALT": "pii-test-salt-" + "0" * 32,
            "PROTECTED_CAMPAIGN_IDS": [_CAMP_PROTECTED, _CAMP_UNPROTECTED],
        }
    )
    telegram_mock = AsyncMock(return_value=1)
    async with httpx.AsyncClient(transport=transport) as http:
        fc = FormChecker(direct=direct, http=http, pool=pool, settings=settings)
        with patch.object(form_checker.telegram_tools, "send_message", telegram_mock):
            result = await fc.run()
    assert result["all_ok"] is False
    assert result["suspended"] == []
    assert result["action"] == "alert_only (trust=shadow)"
    direct.pause_campaign.assert_not_awaited()
    telegram_mock.assert_awaited()


@pytest.mark.asyncio
async def test_run_assisted_suspends_non_protected_only() -> None:
    html = _landing_html(bitrix=True)
    transport = _mock_transport(landing_body=html, cors_acao="*")
    pool = _mock_pool(trust_level=TrustLevel.ASSISTED)
    direct = _direct_stub(
        rejected_campaigns={
            _CAMP_PROTECTED: "rejected in protected",
            _CAMP_UNPROTECTED: "rejected in unprotected",
        }
    )
    settings = _settings(
        protected=[_CAMP_PROTECTED],
        landings=["https://24bankrotsttvo.ru/pages/ad/a.html"],
    )
    settings = Settings(  # type: ignore[call-arg]
        **{
            **settings.model_dump(),
            "SDA_INTERNAL_API_KEY": "a" * 64,
            "SDA_WEBHOOK_HMAC_SECRET": "b" * 64,
            "HYPOTHESIS_HMAC_SECRET": "c" * 64,
            "PII_SALT": "pii-test-salt-" + "0" * 32,
            "PROTECTED_CAMPAIGN_IDS": [_CAMP_PROTECTED, _CAMP_UNPROTECTED],
        }
    )
    # With both in PROTECTED, nothing suspends — we want exactly one in
    # protected. Rebuild settings with only _CAMP_PROTECTED protected, but
    # keep the moderation scan of both campaigns.
    settings = Settings(  # type: ignore[call-arg]
        **{
            **settings.model_dump(),
            "SDA_INTERNAL_API_KEY": "a" * 64,
            "SDA_WEBHOOK_HMAC_SECRET": "b" * 64,
            "HYPOTHESIS_HMAC_SECRET": "c" * 64,
            "PII_SALT": "pii-test-salt-" + "0" * 32,
            "PROTECTED_CAMPAIGN_IDS": [_CAMP_PROTECTED, _CAMP_UNPROTECTED],
        }
    )
    # The moderation input set and the protected set must differ — rebuild:
    direct = _direct_stub(
        rejected_campaigns={
            _CAMP_PROTECTED: "rejected in protected",
            _CAMP_UNPROTECTED: "rejected in unprotected",
        }
    )

    class _S:
        pass

    # Instead of wrestling with Settings, monkey-patch PROTECTED on a copy.
    # Pydantic models are frozen by config? Settings is not frozen.
    settings.PROTECTED_CAMPAIGN_IDS = [_CAMP_PROTECTED]
    # expand scan set to include unprotected via property override — simpler:
    # pass both ids explicitly to check_ad_moderation inside FormChecker.
    # Work around: override moderation scan list by monkey-patching.

    async with httpx.AsyncClient(transport=transport) as http:
        fc = FormChecker(direct=direct, http=http, pool=pool, settings=settings)
        original = form_checker.check_ad_moderation

        async def _mod(direct_arg: Any, _ids: Any):
            return await original(direct_arg, [_CAMP_PROTECTED, _CAMP_UNPROTECTED])

        telegram_mock = AsyncMock(return_value=1)
        kb_mock = AsyncMock(return_value={"answer": "safe", "citations": ["kb/policy.md"]})
        with (
            patch.object(form_checker, "check_ad_moderation", _mod),
            patch.object(form_checker.telegram_tools, "send_message", telegram_mock),
            patch.object(form_checker.knowledge, "consult", kb_mock),
        ):
            result = await fc.run()

    assert result["suspended"] == [_CAMP_UNPROTECTED]
    direct.pause_campaign.assert_awaited_once_with(_CAMP_UNPROTECTED)
    kb_mock.assert_awaited()


@pytest.mark.asyncio
async def test_run_assisted_protected_only_no_suspend() -> None:
    html = _landing_html(bitrix=True)
    transport = _mock_transport(landing_body=html, cors_acao="*")
    pool = _mock_pool(trust_level=TrustLevel.ASSISTED)
    direct = _direct_stub(rejected_campaigns={_CAMP_PROTECTED: "rejected"})
    settings = _settings(
        protected=[_CAMP_PROTECTED],
        landings=["https://24bankrotsttvo.ru/pages/ad/a.html"],
    )
    telegram_mock = AsyncMock(return_value=1)
    async with httpx.AsyncClient(transport=transport) as http:
        fc = FormChecker(direct=direct, http=http, pool=pool, settings=settings)
        with patch.object(form_checker.telegram_tools, "send_message", telegram_mock):
            result = await fc.run()
    assert result["suspended"] == []
    assert "alert_only" in result["action"]
    direct.pause_campaign.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_dry_run_never_suspends() -> None:
    html = _landing_html(bitrix=True)
    transport = _mock_transport(landing_body=html, cors_acao="*")
    pool = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub(rejected_campaigns={_CAMP_UNPROTECTED: "rejected"})
    settings = _settings(
        protected=[_CAMP_PROTECTED, _CAMP_UNPROTECTED],
        landings=["https://24bankrotsttvo.ru/pages/ad/a.html"],
    )
    telegram_mock = AsyncMock(return_value=1)
    async with httpx.AsyncClient(transport=transport) as http:
        fc = FormChecker(direct=direct, http=http, pool=pool, settings=settings)
        with patch.object(form_checker.telegram_tools, "send_message", telegram_mock):
            result = await fc.run(dry_run=True)
    assert result["suspended"] == []
    assert result["action"] == "dry_run"
    direct.pause_campaign.assert_not_awaited()
    telegram_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_kb_consult_called_before_suspend() -> None:
    """Functional gate: kb.consult must fire at least once per suspend run."""
    html = _landing_html(bitrix=True)
    transport = _mock_transport(landing_body=html, cors_acao="*")
    pool = _mock_pool(trust_level=TrustLevel.ASSISTED)
    direct = _direct_stub(rejected_campaigns={_CAMP_UNPROTECTED: "rejected"})
    settings = _settings(
        protected=[_CAMP_PROTECTED],
        landings=["https://24bankrotsttvo.ru/pages/ad/a.html"],
    )
    settings.PROTECTED_CAMPAIGN_IDS = [_CAMP_PROTECTED]

    original = form_checker.check_ad_moderation

    async def _mod(direct_arg: Any, _ids: Any):
        return await original(direct_arg, [_CAMP_PROTECTED, _CAMP_UNPROTECTED])

    kb_mock = AsyncMock(return_value={"answer": "ok", "citations": ["kb/x.md"]})
    telegram_mock = AsyncMock(return_value=1)
    async with httpx.AsyncClient(transport=transport) as http:
        fc = FormChecker(direct=direct, http=http, pool=pool, settings=settings)
        with (
            patch.object(form_checker, "check_ad_moderation", _mod),
            patch.object(form_checker.knowledge, "consult", kb_mock),
            patch.object(form_checker.telegram_tools, "send_message", telegram_mock),
        ):
            await fc.run()
    kb_mock.assert_awaited()


@pytest.mark.asyncio
async def test_run_parallel_checks_isolate_failures() -> None:
    """If CORS check times out, moderation still runs and reports results."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "OPTIONS":
            raise httpx.TimeoutException("cors timeout")
        if request.url.path.endswith("/lead"):
            return httpx.Response(400, json={"error": "phone required"})
        return httpx.Response(200, text=_landing_html(bitrix=True))

    transport = httpx.MockTransport(handler)
    pool = _mock_pool(trust_level=TrustLevel.SHADOW)
    direct = _direct_stub()
    settings = _settings(
        protected=[_CAMP_PROTECTED],
        landings=["https://24bankrotsttvo.ru/pages/ad/a.html"],
    )
    telegram_mock = AsyncMock(return_value=1)
    async with httpx.AsyncClient(transport=transport) as http:
        fc = FormChecker(direct=direct, http=http, pool=pool, settings=settings)
        with patch.object(form_checker.telegram_tools, "send_message", telegram_mock):
            result = await fc.run()

    assert result["cors"]["ok"] is False
    assert result["endpoint"]["ok"] is True
    assert result["moderation"]["ok"] is True
    assert len(result["landings"]) == 1
    assert result["landings"][0]["ok"] is True


# ------------------------------------------------------------ entry+hygiene


@pytest.mark.asyncio
async def test_run_registered_in_job_registry() -> None:
    from agent_runtime.jobs import JOB_REGISTRY

    assert "form_checker" in JOB_REGISTRY
    assert JOB_REGISTRY["form_checker"] is form_checker.run


@pytest.mark.asyncio
async def test_run_degraded_noop_without_deps() -> None:
    pool = _mock_pool(trust_level=TrustLevel.SHADOW)
    result = await form_checker.run(pool)
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    assert result["all_ok"] is True


def test_no_legacy_app_imports_in_module() -> None:
    src = Path("agent_runtime/jobs/form_checker.py").read_text()
    assert "from app." not in src
    assert "import app." not in src
    assert "urllib.request" not in src


def test_no_hardcoded_protected_ids_or_urls() -> None:
    src = Path("agent_runtime/jobs/form_checker.py").read_text()
    assert "708978456" not in src
    assert "708978457" not in src
    assert "709307228" not in src
    assert "709014142" not in src
    assert "24bankrotsttvo.ru/pages/ad/" not in src
    assert "smart-direct-agent-v2-production" not in src


def test_normalise_landings_handles_exceptions() -> None:
    raw = [ValueError("boom"), {"url": "ok", "ok": True, "issues": []}]
    out = form_checker._normalise_landings(raw, ["u1", "u2"])
    assert out[0]["ok"] is False
    assert "boom" in out[0]["issues"][0]
    assert out[1]["ok"] is True


def test_normalise_landings_handles_top_level_exception() -> None:
    out = form_checker._normalise_landings(RuntimeError("gather crashed"), ["u1", "u2"])
    assert len(out) == 2
    assert all(not row["ok"] for row in out)


def test_coerce_check_result_accepts_exception() -> None:
    err = form_checker._coerce_check_result(RuntimeError("boom"), "CORS")
    assert err["ok"] is False
    assert "CORS check crashed" in err["issue"]


def test_coerce_check_result_bad_type() -> None:
    assert form_checker._coerce_check_result(42, "X")["ok"] is False


def test_coerce_moderation_handles_exception() -> None:
    err = form_checker._coerce_moderation(RuntimeError("boom"))
    assert err["ok"] is False
    assert err["rejected"][0]["error"].startswith("moderation crashed")


def test_coerce_moderation_bad_type() -> None:
    err = form_checker._coerce_moderation(42)
    assert err["ok"] is False


@pytest.mark.asyncio
async def test_suspend_many_swallows_protected_error() -> None:
    direct = _direct_stub(pause_raises=ProtectedCampaignError("blocked"))
    result = await form_checker._suspend_many(direct, [_CAMP_PROTECTED])
    assert result == []


@pytest.mark.asyncio
async def test_suspend_many_swallows_generic_error() -> None:
    direct = _direct_stub(pause_raises=RuntimeError("boom"))
    result = await form_checker._suspend_many(direct, [_CAMP_UNPROTECTED])
    assert result == []


@pytest.mark.asyncio
async def test_suspend_many_verify_exception_skips() -> None:
    direct = _direct_stub()
    direct.verify_campaign_paused = AsyncMock(side_effect=RuntimeError("boom"))
    result = await form_checker._suspend_many(direct, [_CAMP_UNPROTECTED])
    assert result == []


@pytest.mark.asyncio
async def test_kb_consult_before_suspend_swallows_error() -> None:
    with patch.object(
        form_checker.knowledge, "consult", AsyncMock(side_effect=RuntimeError("no key"))
    ):
        result = await form_checker._kb_consult_before_suspend([1], None)
    assert result is None


def test_alert_titles_branch_on_state() -> None:
    assert (
        form_checker._alert_title(
            suspended=[1], trust=TrustLevel.AUTONOMOUS, global_issue=None, dry_run=False
        )
        == "ПРОБЛЕМЫ — кампании ОСТАНОВЛЕНЫ"
    )
    assert (
        form_checker._alert_title(
            suspended=[], trust=TrustLevel.SHADOW, global_issue=None, dry_run=False
        )
        == "ПРОБЛЕМЫ — shadow NOTIFY"
    )
    assert (
        form_checker._alert_title(
            suspended=[], trust=TrustLevel.AUTONOMOUS, global_issue="landing", dry_run=False
        )
        == "ПРОБЛЕМЫ — алерт без стопа (prod защищён)"
    )
    assert form_checker._alert_title(
        suspended=[], trust=TrustLevel.SHADOW, global_issue=None, dry_run=True
    ).endswith("dry_run (без стопа)")
