"""Unit tests for ``agent_runtime.tools.telegram``."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from agent_runtime.auth.signing import HMACSigner
from agent_runtime.config import Settings
from agent_runtime.tools.telegram import (
    InlineButton,
    TelegramAPIError,
    edit_message,
    send_message,
    send_with_inline,
)

_HYPOTHESIS_SECRET = "c" * 64


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET=_HYPOTHESIS_SECRET,
        PII_SALT="pii-test-salt-" + "0" * 32,
        TELEGRAM_BOT_TOKEN="1234:secrettoken",
        TELEGRAM_CHAT_ID=42,
    )


class _Recorder:
    def __init__(self, handler):
        self.calls: list[dict[str, Any]] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        method = str(request.url.path).rsplit("/", 1)[-1]
        self.calls.append({"method": method, "body": body})
        return self._handler(request, body, len(self.calls))


async def _client(handler):
    recorder = _Recorder(handler)
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return client, recorder


@pytest.mark.asyncio
async def test_send_message_returns_message_id() -> None:
    def handler(req, body, n):
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})

    client, rec = await _client(handler)
    try:
        mid = await send_message(client, _settings(), text="hello")
    finally:
        await client.aclose()
    assert mid == 42
    assert rec.calls[0]["body"]["chat_id"] == 42
    assert rec.calls[0]["body"]["text"] == "hello"


@pytest.mark.asyncio
async def test_send_with_inline_callback_data_signed() -> None:
    def handler(req, body, n):
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    client, rec = await _client(handler)
    try:
        await send_with_inline(
            client,
            _settings(),
            text="Approve?",
            buttons=[
                [
                    InlineButton(text="✅", action="approve"),
                    InlineButton(text="❌", action="reject"),
                ]
            ],
            hypothesis_id="h123",
        )
    finally:
        await client.aclose()
    keyboard = rec.calls[0]["body"]["reply_markup"]["inline_keyboard"]
    approve_data = keyboard[0][0]["callback_data"]
    reject_data = keyboard[0][1]["callback_data"]
    # Signer must verify both payloads
    from pydantic import SecretStr

    signer = HMACSigner(SecretStr(_HYPOTHESIS_SECRET))
    assert signer.verify_callback(approve_data) == ("h123", "approve")
    assert signer.verify_callback(reject_data) == ("h123", "reject")


@pytest.mark.asyncio
async def test_callback_data_within_64_bytes() -> None:
    def handler(req, body, n):
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    long_id = "hypothesis-" + "x" * 24  # simulate long id
    client, rec = await _client(handler)
    try:
        await send_with_inline(
            client,
            _settings(),
            text="x",
            buttons=[[InlineButton(text="ok", action="approve")]],
            hypothesis_id=long_id,
        )
    finally:
        await client.aclose()
    data = rec.calls[0]["body"]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    assert len(data.encode("utf-8")) <= 64


@pytest.mark.asyncio
async def test_edit_message_called_correctly() -> None:
    def handler(req, body, n):
        return httpx.Response(200, json={"ok": True, "result": {}})

    client, rec = await _client(handler)
    try:
        await edit_message(client, _settings(), message_id=999, text="updated")
    finally:
        await client.aclose()
    call = rec.calls[0]
    assert call["method"] == "editMessageText"
    assert call["body"]["message_id"] == 999
    assert call["body"]["text"] == "updated"


@pytest.mark.asyncio
async def test_retry_on_429_respects_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", fake_sleep)

    def handler(req, body, n):
        if n == 1:
            return httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 0.05}})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})

    client, rec = await _client(handler)
    try:
        mid = await send_message(client, _settings(), text="x")
    finally:
        await client.aclose()
    assert mid == 7
    assert sleeps and sleeps[0] == pytest.approx(0.05, rel=0.01)
    assert len(rec.calls) == 2


@pytest.mark.asyncio
async def test_telegram_error_raises_typed() -> None:
    def handler(req, body, n):
        return httpx.Response(
            400, json={"ok": False, "error_code": 400, "description": "Bad Request"}
        )

    client, _ = await _client(handler)
    try:
        with pytest.raises(TelegramAPIError) as exc:
            await send_message(client, _settings(), text="x")
    finally:
        await client.aclose()
    assert exc.value.code == 400


@pytest.mark.asyncio
async def test_does_not_log_message_text(caplog: pytest.LogCaptureFixture) -> None:
    def handler(req, body, n):
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    caplog.set_level(logging.INFO, logger="agent_runtime.tools.telegram")
    client, _ = await _client(handler)
    try:
        await send_message(client, _settings(), text="lead +79991234567 Иванов ивент СМ-5")
    finally:
        await client.aclose()
    assert "+79991234567" not in caplog.text
    assert "Иванов" not in caplog.text


@pytest.mark.asyncio
async def test_missing_bot_token_raises() -> None:
    bad = _settings()
    bad.TELEGRAM_BOT_TOKEN = bad.TELEGRAM_BOT_TOKEN.__class__("")

    client, _ = await _client(
        lambda req, body, n: httpx.Response(200, json={"ok": True, "result": {}})
    )
    try:
        with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
            await send_message(client, bad, text="x")
    finally:
        await client.aclose()
