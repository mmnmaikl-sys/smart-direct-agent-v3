"""Integration tests against a live Yandex.Direct sandbox campaign.

REQUIREMENTS (skipped when absent):
    - YANDEX_DIRECT_TOKEN_SANDBOX  — OAuth token with write scope on a
      *separate* Direct account used only for v3 integration tests.
    - TEST_SANDBOX_CAMPAIGN_ID     — numeric ID of the sandbox campaign; the
      tests mutate keywords / negatives / adgroup state inside it.

Sandbox setup (run once by the owner):
    1. Create a standalone Yandex.Direct account (NOT the prod one).
    2. Make one TEXT_CAMPAIGN called ``test_sandbox_v3`` with daily budget
       50 RUB. One AdGroup, one keyword (``"тест_ключ_v3"``), one ad.
    3. StartDate in the past so it is not DRAFT.
    4. Generate an OAuth token with scopes
       ``direct:api-direct`` + ``direct:api-reports``. Save as env var.

Each test seeds state in setup, asserts post-mutation state via verify_*,
and restores the original state in teardown. Budget consumed per run is
negligible (≤10 RUB), and teardown leaves the sandbox ready for the next
run.

These tests DO NOT run in CI by default — they need real credentials that
are not part of the shared CI token set. Run locally before merging a PR
that touches ``agent_runtime/tools/direct_api.py``::

    pytest tests/integration/test_direct_api.py -v
"""

from __future__ import annotations

import os

import pytest

from agent_runtime.config import Settings
from agent_runtime.tools import DirectAPI, ProtectedCampaignError

SANDBOX_TOKEN = os.environ.get("YANDEX_DIRECT_TOKEN_SANDBOX")
SANDBOX_CAMPAIGN_ID = os.environ.get("TEST_SANDBOX_CAMPAIGN_ID")

_SKIP_NO_SANDBOX = pytest.mark.skipif(
    not SANDBOX_TOKEN or not SANDBOX_CAMPAIGN_ID,
    reason="sandbox creds missing (YANDEX_DIRECT_TOKEN_SANDBOX, TEST_SANDBOX_CAMPAIGN_ID)",
)

_PROTECTED = [708978456, 708978457, 708978458, 709014142, 709307228]


def _sandbox_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        YANDEX_DIRECT_TOKEN=SANDBOX_TOKEN or "",
        PROTECTED_CAMPAIGN_IDS=_PROTECTED,
    )


@pytest.mark.asyncio
@_SKIP_NO_SANDBOX
async def test_set_and_verify_bid() -> None:
    campaign_id = int(SANDBOX_CAMPAIGN_ID or "0")
    assert campaign_id not in _PROTECTED, "sandbox must not be in PROTECTED list"
    async with DirectAPI(_sandbox_settings()) as api:
        groups = await api.get_adgroups(campaign_id=campaign_id)
        assert groups, "sandbox campaign has no adgroups"
        ad_group_ids = [int(g["Id"]) for g in groups]
        keywords = await api.get_keywords(ad_group_ids)
        assert keywords, "sandbox has no keywords to bid on"
        kw = keywords[0]
        keyword_id = int(kw["Id"])
        original_bid = (int(kw.get("Bid", 0)) // 1_000_000) or 10

        await api.set_bid(keyword_id, bid_rub=15)
        try:
            assert await api.verify_bid(keyword_id, expected_bid_rub=15) is True
        finally:
            await api.set_bid(keyword_id, bid_rub=original_bid)


@pytest.mark.asyncio
@_SKIP_NO_SANDBOX
async def test_add_negatives_and_verify() -> None:
    campaign_id = int(SANDBOX_CAMPAIGN_ID or "0")
    async with DirectAPI(_sandbox_settings()) as api:
        probe_phrases = ["тест_минус_sda_v3_1", "тест_минус_sda_v3_2"]
        original = await api.get_campaign_negative_keywords(campaign_id)
        await api.add_negatives(campaign_id, probe_phrases)
        try:
            assert await api.verify_negatives_added(campaign_id, probe_phrases) is True
        finally:
            # Rewrite without our probes; helper isn't symmetric (it merges),
            # so we use the underlying campaigns.update directly.
            cleaned = [p for p in original]
            await api._call(
                "campaigns",
                "update",
                {"Campaigns": [{"Id": campaign_id, "NegativeKeywords": {"Items": cleaned}}]},
            )


@pytest.mark.asyncio
@_SKIP_NO_SANDBOX
async def test_pause_resume_group_and_verify() -> None:
    campaign_id = int(SANDBOX_CAMPAIGN_ID or "0")
    async with DirectAPI(_sandbox_settings()) as api:
        groups = await api.get_adgroups(campaign_id=campaign_id)
        assert groups, "sandbox has no adgroups"
        ad_group_id = int(groups[0]["Id"])
        await api.pause_group(ad_group_id)
        try:
            assert await api.verify_group_paused(ad_group_id) is True
        finally:
            await api.resume_group(ad_group_id)
            assert await api.verify_group_resumed(ad_group_id) is True


@pytest.mark.asyncio
@_SKIP_NO_SANDBOX
async def test_protected_campaign_integration() -> None:
    """Even with real credentials, PROTECTED guard must short-circuit."""
    async with DirectAPI(_sandbox_settings()) as api:
        with pytest.raises(ProtectedCampaignError):
            await api.pause_campaign(_PROTECTED[0])
