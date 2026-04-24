"""Integration smoke tests for Bitrix / Metrika / Telegram — real creds.

Each test skips when its credential is absent. Not part of the default CI
matrix — run locally or in a dedicated `integration` job with secrets in
scope. VCR fixtures are out of scope for Task 8 (see Post-completion note
in task-08-bitrix-metrika-telegram.md — deferred with PII-filter hooks).
"""

from __future__ import annotations

import os

import httpx
import pytest

from agent_runtime.config import Settings
from agent_runtime.tools import bitrix, metrika, telegram

_BITRIX_URL = os.environ.get("BITRIX_WEBHOOK_URL_INT") or os.environ.get("BITRIX_WEBHOOK_URL")
_BITRIX_TOKEN = os.environ.get("BITRIX_WEBHOOK_TOKEN_INT") or os.environ.get("BITRIX_WEBHOOK_TOKEN")
_METRIKA_OAUTH = os.environ.get("METRIKA_OAUTH_TOKEN_INT") or os.environ.get("METRIKA_OAUTH_TOKEN")
_TELEGRAM_BOT = os.environ.get("TELEGRAM_BOT_TOKEN_INT") or os.environ.get("TELEGRAM_BOT_TOKEN")
_TELEGRAM_CHAT = os.environ.get("TELEGRAM_CHAT_ID_INT") or os.environ.get("TELEGRAM_CHAT_ID")

_SKIP_BITRIX = pytest.mark.skipif(
    not _BITRIX_URL or not _BITRIX_TOKEN,
    reason="BITRIX_WEBHOOK_URL / _TOKEN not set",
)
_SKIP_METRIKA = pytest.mark.skipif(not _METRIKA_OAUTH, reason="METRIKA_OAUTH_TOKEN not set")
_SKIP_TELEGRAM = pytest.mark.skipif(
    not _TELEGRAM_BOT or not _TELEGRAM_CHAT,
    reason="TELEGRAM_BOT_TOKEN / _CHAT_ID not set",
)


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        BITRIX_WEBHOOK_URL=_BITRIX_URL or "",
        BITRIX_WEBHOOK_TOKEN=_BITRIX_TOKEN or "",
        METRIKA_OAUTH_TOKEN=_METRIKA_OAUTH or "",
        METRIKA_COUNTER_ID=107734488,
        TELEGRAM_BOT_TOKEN=_TELEGRAM_BOT or "",
        TELEGRAM_CHAT_ID=int(_TELEGRAM_CHAT) if _TELEGRAM_CHAT else 0,
    )


@pytest.mark.asyncio
@_SKIP_BITRIX
async def test_bitrix_lead_list_real() -> None:
    async with httpx.AsyncClient() as client:
        leads = await bitrix.get_lead_list(client, _settings(), max_total=5)
    assert isinstance(leads, list)
    if leads:
        assert "ID" in leads[0]


@pytest.mark.asyncio
@_SKIP_METRIKA
async def test_metrika_bounce_real() -> None:
    async with httpx.AsyncClient() as client:
        result = await metrika.get_bounce_by_campaign(
            client, _settings(), date1="7daysAgo", date2="today"
        )
    assert isinstance(result, dict)
    for rate in result.values():
        assert 0.0 <= rate <= 1.0


@pytest.mark.asyncio
@_SKIP_TELEGRAM
async def test_telegram_send_and_edit_real() -> None:
    async with httpx.AsyncClient() as client:
        mid = await telegram.send_message(
            client, _settings(), text="[CI TEST] SDA v3 Task 8 — safe to ignore"
        )
        assert mid > 0
        await telegram.edit_message(
            client,
            _settings(),
            message_id=mid,
            text="[CI TEST] SDA v3 Task 8 — edited",
        )
