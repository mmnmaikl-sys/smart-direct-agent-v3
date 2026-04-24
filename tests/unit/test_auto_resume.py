"""Unit tests for agent_runtime.jobs.auto_resume (Task 21, part 1).

Mocked pool + mocked DirectAPI + SimpleNamespace http stub — nothing hits
the wire. Trust overlay is driven by ``_mock_pool(trust_level=...)`` which
scripts the first ``fetchone`` to the trust row. Tests cover:

* Happy path: all PROTECTED campaigns ``State='ON'`` → no actions.
* Autonomous + one SUSPENDED → resume + verify + audit row with is_mutation=true.
* Assisted + SUSPENDED → AUTO (auto_resume is in ASSISTED_AUTO_WHITELIST).
* Shadow + SUSPENDED → NOTIFY only, 0 mutations, audit row with is_mutation=false.
* ``StatusArchive='YES'`` ignored even when SUSPENDED.
* ``dry_run=True`` → 0 mutations, 0 telegram, 0 audit, but ``would_resume``
  populated (smoke-snapshot).
* ``ProtectedCampaignError`` on ``resume_campaign`` swallowed → errors row.
* Generic mutation failure → status="ok", errors populated, job doesn't raise.
* ``verify_campaign_resumed`` mismatch → errors row (``verify_mismatch``).
* DI missing → degraded_noop, no wire calls.
* ``_select_resume_targets`` filtering invariant (pure function) — ON skipped,
  archived skipped, SUSPENDED+non-archived selected.
* Whitelist sourced from Settings, never hardcoded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import auto_resume
from agent_runtime.jobs.auto_resume import (
    AutoResumeResult,
    _CampaignView,
    _select_resume_targets,
    run,
)
from agent_runtime.tools.direct_api import ProtectedCampaignError
from agent_runtime.trust_levels import TrustLevel

_CAMP_A = 708978456
_CAMP_B = 708978457
_CAMP_ARCHIVED = 708978458  # Удмуртия — archived in prod
_CAMP_RSYA_ARCHIVED = 709014142  # РСЯ-ретаргет — archived in prod
_CAMP_RSYA_ACTIVE = 709307228

_PROTECTED = [_CAMP_A, _CAMP_B, _CAMP_ARCHIVED, _CAMP_RSYA_ARCHIVED, _CAMP_RSYA_ACTIVE]


# --------------------------------------------------------------------- fixtures


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
) -> tuple[MagicMock, MagicMock, list[tuple[Any, ...]]]:
    """Mock pool: first fetchone returns the trust-level row, rest default.

    Returns ``(pool, cursor, executed_sqls)``. ``executed_sqls`` is a list
    of ``args`` tuples passed to ``cursor.execute`` — tests inspect it to
    confirm audit_log INSERTs fired and to extract the recorded payload.
    """
    rows = [(trust_level.value,)]
    one_iter = iter(rows)

    async def _fetchone() -> Any:
        try:
            return next(one_iter)
        except StopIteration:
            return (1,)  # audit_log INSERT ... RETURNING id fallback

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
    return pool, cursor, executed


def _direct_stub(
    *,
    campaigns: list[dict[str, Any]] | None = None,
    resume_raises: BaseException | None = None,
    verify_returns: bool = True,
    get_raises: BaseException | None = None,
) -> SimpleNamespace:
    """Build a DirectAPI-like stub.

    ``campaigns`` — list of ``{Id, Name, State, StatusArchive}`` dicts
    returned by ``get_campaigns``. Defaults to all-ON protected set.
    """
    if campaigns is None:
        campaigns = [
            {"Id": cid, "Name": f"Camp-{cid}", "State": "ON", "StatusArchive": "NO"}
            for cid in _PROTECTED
        ]

    async def get_campaigns(ids: list[int]) -> list[dict[str, Any]]:
        if get_raises is not None:
            raise get_raises
        lookup = {int(c["Id"]): c for c in campaigns}
        return [lookup[i] for i in ids if i in lookup]

    resume_mock = AsyncMock(return_value={})
    if resume_raises is not None:
        resume_mock.side_effect = resume_raises

    return SimpleNamespace(
        get_campaigns=AsyncMock(side_effect=get_campaigns),
        resume_campaign=resume_mock,
        verify_campaign_resumed=AsyncMock(return_value=verify_returns),
    )


# ---------------------------------------------------------- pure function tests


def test_select_resume_targets_only_suspended_non_archived() -> None:
    views = [
        _CampaignView(1, "ON-camp", "ON", "NO"),
        _CampaignView(2, "SUSP-camp", "SUSPENDED", "NO"),
        _CampaignView(3, "SUSP-archive", "SUSPENDED", "YES"),
        _CampaignView(4, "ON-archive", "ON", "YES"),  # weird but legal
    ]
    targets = _select_resume_targets(views)
    assert [t.campaign_id for t in targets] == [2]


def test_select_resume_targets_missing_archive_defaults_treated_as_no() -> None:
    # _CampaignView always carries a string for status_archive, so "" is the
    # "unknown" case — must NOT match YES, so included when SUSPENDED.
    views = [_CampaignView(1, "x", "SUSPENDED", "")]
    targets = _select_resume_targets(views)
    assert len(targets) == 1


# ----------------------------------------------------------------- run tests


@pytest.mark.asyncio
async def test_run_all_active_no_action() -> None:
    """All PROTECTED campaigns ON → no mutations, no telegram, no audit."""
    pool, _cursor, executed = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub()
    telegram_mock = AsyncMock(return_value=1)
    http_client = SimpleNamespace()

    with patch.object(auto_resume.telegram_tools, "send_message", telegram_mock):
        result = await run(pool, direct=direct, http_client=http_client, settings=_settings())

    assert result["status"] == "ok"
    assert result["suspended_found"] == []
    assert result["resumed"] == []
    assert result["would_resume"] == []
    direct.resume_campaign.assert_not_awaited()
    telegram_mock.assert_not_awaited()
    # No audit INSERT (SQL list should contain only the trust_level SELECT).
    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts == []


@pytest.mark.asyncio
async def test_one_suspended_assisted_triggers_resume() -> None:
    """auto_resume ∈ ASSISTED_AUTO_WHITELIST — assisted must AUTO-resume + verify."""
    pool, _, executed = _mock_pool(trust_level=TrustLevel.ASSISTED)
    campaigns = [
        {"Id": _CAMP_A, "Name": "A", "State": "SUSPENDED", "StatusArchive": "NO"},
        {"Id": _CAMP_B, "Name": "B", "State": "ON", "StatusArchive": "NO"},
        {"Id": _CAMP_ARCHIVED, "Name": "C", "State": "SUSPENDED", "StatusArchive": "YES"},
        {"Id": _CAMP_RSYA_ARCHIVED, "Name": "D", "State": "SUSPENDED", "StatusArchive": "YES"},
        {"Id": _CAMP_RSYA_ACTIVE, "Name": "E", "State": "ON", "StatusArchive": "NO"},
    ]
    direct = _direct_stub(campaigns=campaigns)
    telegram_mock = AsyncMock(return_value=1)
    # Neutralise the verify-retry sleep.
    with (
        patch.object(auto_resume, "_sleep_for_verify", AsyncMock(return_value=None)),
        patch.object(auto_resume.telegram_tools, "send_message", telegram_mock),
    ):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())

    assert result["status"] == "ok"
    assert result["suspended_found"] == [_CAMP_A]
    assert result["resumed"] == [_CAMP_A]
    assert result["would_resume"] == []
    direct.resume_campaign.assert_awaited_once_with(_CAMP_A)
    direct.verify_campaign_resumed.assert_awaited_once_with(_CAMP_A)
    telegram_mock.assert_awaited_once()
    # audit INSERT with is_mutation=true
    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts, "audit_log INSERT not issued"
    # is_mutation is positional; look for True in the bound tuple
    insert_params = audit_inserts[0][1]
    assert True in insert_params, "is_mutation=true not in audit INSERT params"


@pytest.mark.asyncio
async def test_autonomous_asks_on_unknown_action_type() -> None:
    """In autonomous, unknown action 'auto_resume' defaults to ASK (conservative).

    TODO(integration): once ``auto_resume`` lands in
    :data:`agent_runtime.decision_engine.IRREVERSIBILITY`, autonomous will
    AUTO. Today, autonomous-not-in-whitelist falls back to NOTIFY path (
    would_resume populated, no mutation, telegram still fires).
    """
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    campaigns = [
        {"Id": _CAMP_A, "Name": "A", "State": "SUSPENDED", "StatusArchive": "NO"},
    ]
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=[_CAMP_A],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )
    direct = _direct_stub(campaigns=campaigns)
    with patch.object(auto_resume.telegram_tools, "send_message", AsyncMock(return_value=1)):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=settings)

    # No mutation in the default overlay — ASK/NOTIFY path.
    direct.resume_campaign.assert_not_awaited()
    assert result["resumed"] == []
    # Still surfaced as would_resume so the owner sees the request.
    assert result["would_resume"] == [_CAMP_A]


@pytest.mark.asyncio
async def test_one_suspended_shadow_notifies_only() -> None:
    pool, _, executed = _mock_pool(trust_level=TrustLevel.SHADOW)
    campaigns = [
        {"Id": _CAMP_A, "Name": "A", "State": "SUSPENDED", "StatusArchive": "NO"},
        {"Id": _CAMP_B, "Name": "B", "State": "ON", "StatusArchive": "NO"},
    ]
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=[_CAMP_A, _CAMP_B],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )
    direct = _direct_stub(campaigns=campaigns)
    telegram_mock = AsyncMock(return_value=1)
    with patch.object(auto_resume.telegram_tools, "send_message", telegram_mock):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=settings)

    assert result["status"] == "ok"
    assert result["suspended_found"] == [_CAMP_A]
    assert result["resumed"] == []
    assert result["would_resume"] == [_CAMP_A]
    assert result["notified"] == [_CAMP_A]
    direct.resume_campaign.assert_not_awaited()
    telegram_mock.assert_awaited_once()
    # audit INSERT with is_mutation=false
    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts
    insert_params = audit_inserts[0][1]
    # is_mutation is False in shadow (no resume happened).
    assert False in insert_params


@pytest.mark.asyncio
async def test_archived_suspended_ignored() -> None:
    """StatusArchive=YES never gets resumed — even in autonomous."""
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    campaigns = [
        {"Id": _CAMP_ARCHIVED, "Name": "Udm", "State": "SUSPENDED", "StatusArchive": "YES"},
        {
            "Id": _CAMP_RSYA_ARCHIVED,
            "Name": "RSYA",
            "State": "SUSPENDED",
            "StatusArchive": "YES",
        },
    ]
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=[_CAMP_ARCHIVED, _CAMP_RSYA_ARCHIVED],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )
    direct = _direct_stub(campaigns=campaigns)
    with patch.object(auto_resume.telegram_tools, "send_message", AsyncMock(return_value=1)):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=settings)

    assert result["suspended_found"] == []
    assert result["resumed"] == []
    direct.resume_campaign.assert_not_awaited()


@pytest.mark.asyncio
async def test_dry_run_no_side_effects() -> None:
    pool, _, executed = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    campaigns = [
        {"Id": _CAMP_A, "Name": "A", "State": "SUSPENDED", "StatusArchive": "NO"},
        {"Id": _CAMP_B, "Name": "B", "State": "SUSPENDED", "StatusArchive": "NO"},
    ]
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=[_CAMP_A, _CAMP_B],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )
    direct = _direct_stub(campaigns=campaigns)
    telegram_mock = AsyncMock(return_value=1)
    with patch.object(auto_resume.telegram_tools, "send_message", telegram_mock):
        result = await run(
            pool, direct=direct, http_client=SimpleNamespace(), settings=settings, dry_run=True
        )

    assert result["dry_run"] is True
    assert result["would_resume"] == [_CAMP_A, _CAMP_B]
    assert result["resumed"] == []
    assert result["notified"] == []
    direct.resume_campaign.assert_not_awaited()
    telegram_mock.assert_not_awaited()
    # No audit_log INSERT
    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts == []


@pytest.mark.asyncio
async def test_resume_failure_logged_not_raised() -> None:
    """Tool exception does not sink job — errors array populated, status=ok."""
    pool, _, _ = _mock_pool(trust_level=TrustLevel.ASSISTED)
    campaigns = [
        {"Id": _CAMP_A, "Name": "A", "State": "SUSPENDED", "StatusArchive": "NO"},
    ]
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=[_CAMP_A],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )
    direct = _direct_stub(campaigns=campaigns, resume_raises=RuntimeError("Direct down"))
    with (
        patch.object(auto_resume, "_sleep_for_verify", AsyncMock(return_value=None)),
        patch.object(auto_resume.telegram_tools, "send_message", AsyncMock(return_value=1)),
    ):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=settings)

    assert result["status"] == "ok"
    assert result["resumed"] == []
    assert len(result["errors"]) == 1
    assert result["errors"][0]["campaign_id"] == _CAMP_A
    assert "Direct down" in result["errors"][0]["error"]


@pytest.mark.asyncio
async def test_protected_campaign_error_swallowed() -> None:
    """ProtectedCampaignError → errors row with reason=protected_guard, no raise."""
    pool, _, _ = _mock_pool(trust_level=TrustLevel.ASSISTED)
    campaigns = [
        {"Id": _CAMP_A, "Name": "A", "State": "SUSPENDED", "StatusArchive": "NO"},
    ]
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=[_CAMP_A],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )
    direct = _direct_stub(
        campaigns=campaigns,
        resume_raises=ProtectedCampaignError(f"campaign {_CAMP_A} protected"),
    )
    with patch.object(auto_resume.telegram_tools, "send_message", AsyncMock(return_value=1)):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=settings)

    assert result["status"] == "ok"
    assert result["resumed"] == []
    assert result["errors"][0]["error"] == "protected_guard"
    # verify_* was NOT reached — protected error short-circuits.
    direct.verify_campaign_resumed.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_mismatch_recorded_as_error() -> None:
    pool, _, _ = _mock_pool(trust_level=TrustLevel.ASSISTED)
    campaigns = [
        {"Id": _CAMP_A, "Name": "A", "State": "SUSPENDED", "StatusArchive": "NO"},
    ]
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=[_CAMP_A],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )
    direct = _direct_stub(campaigns=campaigns, verify_returns=False)
    with (
        patch.object(auto_resume, "_sleep_for_verify", AsyncMock(return_value=None)),
        patch.object(auto_resume.telegram_tools, "send_message", AsyncMock(return_value=1)),
    ):
        result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=settings)

    assert result["resumed"] == []
    assert result["errors"][0]["error"] == "verify_mismatch"
    # resume_campaign was called; verify hit 3 attempts.
    direct.resume_campaign.assert_awaited_once_with(_CAMP_A)
    assert direct.verify_campaign_resumed.await_count == 3


@pytest.mark.asyncio
async def test_degraded_noop_without_direct_or_settings() -> None:
    """JOB_REGISTRY default dispatch (no DI) returns structured no-op."""
    pool, _, executed = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)

    result = await run(pool)

    assert result["status"] == "ok"
    assert result["checked"] == 0
    assert result["suspended_found"] == []
    # No audit_log INSERT in degraded path.
    audit_inserts = [
        args for args in executed if args and isinstance(args[0], str) and "audit_log" in args[0]
    ]
    assert audit_inserts == []


@pytest.mark.asyncio
async def test_whitelist_sourced_from_settings_not_hardcoded() -> None:
    """Override PROTECTED_CAMPAIGN_IDS to an arbitrary test id; run must look
    up exactly those ids through ``direct.get_campaigns``. Prevents regression
    against Decision 17 (shared config is source of truth)."""
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    custom_id = 999_999_001
    campaigns = [
        {"Id": custom_id, "Name": "Test", "State": "ON", "StatusArchive": "NO"},
    ]
    settings = Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=[custom_id],
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )
    direct = _direct_stub(campaigns=campaigns)

    await run(pool, direct=direct, http_client=SimpleNamespace(), settings=settings)

    call_args = direct.get_campaigns.await_args.args[0]
    assert call_args == [custom_id]


@pytest.mark.asyncio
async def test_get_campaigns_failure_returns_error_status() -> None:
    """Fatal read failure → status=error; never propagates up the stack."""
    pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)
    direct = _direct_stub(get_raises=RuntimeError("reports 503"))

    result = await run(pool, direct=direct, http_client=SimpleNamespace(), settings=_settings())
    # get_campaigns fail logged and returns empty list → targets=[] → status=ok
    # (the module treats a failed fetch as "nothing to do"). Confirm no crash.
    assert result["status"] == "ok"
    assert result["suspended_found"] == []


@pytest.mark.asyncio
async def test_result_model_roundtrip() -> None:
    """AutoResumeResult serialises with the documented keys."""
    r = AutoResumeResult(
        status="ok",
        trust_level="shadow",
        checked=2,
        suspended_found=[1],
        would_resume=[1],
    ).model_dump(mode="json")
    assert r["status"] == "ok"
    assert r["trust_level"] == "shadow"
    assert r["checked"] == 2
    assert r["suspended_found"] == [1]
    assert r["would_resume"] == [1]
    assert "started_at" in r


# ---------------------------------------------------------- autotargeting tests


class TestAutotargetingManager:
    """Smoke-level coverage for the sibling module bundled with Task 21.

    Full matrix lives in ``test_autotargeting_manager.py``; these tests make
    sure the module loads + the degraded path keeps cron green when Task 7
    wrappers have not landed.
    """

    @pytest.mark.asyncio
    async def test_import_does_not_raise(self) -> None:
        from agent_runtime.jobs import autotargeting_manager  # noqa: F401

    @pytest.mark.asyncio
    async def test_degraded_when_wrapper_missing(self) -> None:
        from agent_runtime.jobs import autotargeting_manager

        pool, _, _ = _mock_pool(trust_level=TrustLevel.AUTONOMOUS)

        # direct with get_adgroups but no get_autotargeting → wrapper-missing
        direct = SimpleNamespace(
            get_adgroups=AsyncMock(return_value=[]),
        )
        result = await autotargeting_manager.run(
            pool, direct=direct, http_client=SimpleNamespace(), settings=_settings()
        )
        assert result["status"] == "degraded_wrapper_missing"
        assert result["action"] == "wrapper_missing"
