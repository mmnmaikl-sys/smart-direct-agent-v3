"""Unit tests for ``agent_runtime.tools.bitrix`` (Task 8)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from agent_runtime.config import Settings
from agent_runtime.tools.bitrix import (
    BitrixAPIError,
    get_deal_list,
    get_lead_list,
    get_stage_history,
    validate_webhook_token,
)


def _settings(token: str = "test-webhook-token-placeholder") -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        BITRIX_WEBHOOK_URL="https://x.bitrix24.ru/rest/1/TOKEN",
        BITRIX_WEBHOOK_TOKEN=token,
    )


class _Recorder:
    def __init__(self, handler):
        self.calls: list[dict[str, Any]] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append({"url": str(request.url), "body": body})
        return self._handler(request, body, len(self.calls))


async def _client(handler):
    recorder = _Recorder(handler)
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return client, recorder


# ---------- happy paths -----------------------------------------------------


@pytest.mark.asyncio
async def test_get_lead_list_basic() -> None:
    payload = [{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]

    def handler(req, body, n):
        return httpx.Response(200, json={"result": payload, "total": 3})

    client, rec = await _client(handler)
    try:
        leads = await get_lead_list(client, _settings(), filter={"STAGE_ID": "C49:NEW"})
    finally:
        await client.aclose()
    assert [lead["ID"] for lead in leads] == ["1", "2", "3"]
    assert rec.calls[0]["body"]["filter"] == {"STAGE_ID": "C49:NEW"}


@pytest.mark.asyncio
async def test_get_lead_list_pagination() -> None:
    pages = [
        {"result": [{"ID": str(i)} for i in range(50)], "next": 50},
        {"result": [{"ID": str(i)} for i in range(50, 80)]},
    ]

    def handler(req, body, n):
        return httpx.Response(200, json=pages[n - 1])

    client, rec = await _client(handler)
    try:
        leads = await get_lead_list(client, _settings(), max_total=100)
    finally:
        await client.aclose()
    assert len(leads) == 80
    assert len(rec.calls) == 2
    assert rec.calls[1]["body"]["start"] == 50


@pytest.mark.asyncio
async def test_get_lead_list_respects_max_total() -> None:
    def handler(req, body, n):
        return httpx.Response(200, json={"result": [{"ID": str(i)} for i in range(50)], "next": 50})

    client, _ = await _client(handler)
    try:
        leads = await get_lead_list(client, _settings(), max_total=30)
    finally:
        await client.aclose()
    assert len(leads) == 30


@pytest.mark.asyncio
async def test_get_deal_list_won_filter() -> None:
    def handler(req, body, n):
        return httpx.Response(
            200,
            json={
                "result": [
                    {"ID": "100", "STAGE_ID": "C45:WON"},
                    {"ID": "101", "STAGE_ID": "C45:WON"},
                ]
            },
        )

    client, rec = await _client(handler)
    try:
        deals = await get_deal_list(
            client,
            _settings(),
            filter={"STAGE_ID": "C45:WON", ">DATE_CLOSED": "2026-03-01"},
        )
    finally:
        await client.aclose()
    assert len(deals) == 2
    assert rec.calls[0]["body"]["filter"]["STAGE_ID"] == "C45:WON"


@pytest.mark.asyncio
async def test_get_stage_history_won() -> None:
    def handler(req, body, n):
        assert body["entityTypeId"] == 2
        return httpx.Response(
            200,
            json={
                "result": [
                    {"ID": 1, "STAGE_ID": "C45:WON", "CREATED_TIME": "2026-04-01"},
                ]
            },
        )

    client, _ = await _client(handler)
    try:
        rows = await get_stage_history(
            client,
            _settings(),
            entity_type_id=2,
            filter={"STAGE_ID": "C45:WON"},
        )
    finally:
        await client.aclose()
    assert rows[0]["STAGE_ID"] == "C45:WON"


# ---------- retry + errors --------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", _noop_sleep())

    def handler(req, body, n):
        if n <= 2:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, json={})
        return httpx.Response(200, json={"result": [{"ID": "1"}]})

    client, rec = await _client(handler)
    try:
        result = await get_lead_list(client, _settings())
    finally:
        await client.aclose()
    assert result == [{"ID": "1"}]
    assert len(rec.calls) == 3


@pytest.mark.asyncio
async def test_fail_fast_on_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", _noop_sleep())

    def handler(req, body, n):
        return httpx.Response(403, json={"error": "FORBIDDEN", "error_description": "no access"})

    client, rec = await _client(handler)
    try:
        with pytest.raises(BitrixAPIError) as exc:
            await get_lead_list(client, _settings())
    finally:
        await client.aclose()
    assert exc.value.status_code == 403
    assert exc.value.code == "FORBIDDEN"
    # No retry on hard 4xx
    assert len(rec.calls) == 1


@pytest.mark.asyncio
async def test_error_description_clipped() -> None:
    def handler(req, body, n):
        return httpx.Response(
            400,
            json={"error": "BAD", "error_description": "x" * 500},
        )

    client, _ = await _client(handler)
    try:
        with pytest.raises(BitrixAPIError) as exc:
            await get_lead_list(client, _settings())
    finally:
        await client.aclose()
    assert len(exc.value.description) <= 201


# ---------- webhook validator ----------------------------------------------


def test_validate_webhook_token_constant_time() -> None:
    s = _settings(token="expected-xyz-123")
    assert validate_webhook_token("expected-xyz-123", s) is True
    assert validate_webhook_token("wrong", s) is False
    assert validate_webhook_token("", s) is False


# ---------- logs don't leak PII -------------------------------------------


@pytest.mark.asyncio
async def test_does_not_log_pii(caplog: pytest.LogCaptureFixture) -> None:
    payload = [{"ID": "1", "PHONE": [{"VALUE": "+79991234567"}], "NAME": "Иван"}]

    def handler(req, body, n):
        return httpx.Response(200, json={"result": payload})

    caplog.set_level(logging.INFO, logger="agent_runtime.tools.bitrix")
    client, _ = await _client(handler)
    try:
        await get_lead_list(client, _settings())
    finally:
        await client.aclose()
    assert "+79991234567" not in caplog.text
    assert "Иван" not in caplog.text


@pytest.mark.asyncio
async def test_missing_webhook_url_raises() -> None:
    bad_settings = _settings()
    bad_settings.BITRIX_WEBHOOK_URL = ""  # type: ignore[misc]

    client, _ = await _client(lambda req, body, n: httpx.Response(200, json={}))
    try:
        with pytest.raises(RuntimeError, match="BITRIX_WEBHOOK_URL"):
            await get_lead_list(client, bad_settings)
    finally:
        await client.aclose()


# ---------- helpers --------------------------------------------------------


def _noop_sleep():
    async def _sleep(_s: float) -> None:
        return None

    return _sleep
