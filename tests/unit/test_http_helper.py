"""Direct coverage for ``agent_runtime.tools._http.retry_with_backoff``."""

from __future__ import annotations

import httpx
import pytest

from agent_runtime.tools._http import retry_with_backoff


def _noop_sleep():
    async def _sleep(_s: float) -> None:
        return None

    return _sleep


@pytest.mark.asyncio
async def test_transport_error_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", _noop_sleep())
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"ok": True})

    response = await retry_with_backoff(fn, name="test")
    assert response.status_code == 200
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_transport_error_exhausts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", _noop_sleep())

    async def fn():
        raise httpx.ConnectTimeout("boom")

    with pytest.raises(httpx.ConnectTimeout):
        await retry_with_backoff(fn, name="test")


@pytest.mark.asyncio
async def test_returns_last_response_when_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", _noop_sleep())

    async def fn():
        return httpx.Response(503, json={})

    response = await retry_with_backoff(fn, name="test", max_attempts=2)
    # After exhaustion, last retryable response is returned (caller maps to typed err).
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_retry_after_header_parsed_as_float(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", fake_sleep)
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2.5"}, json={})
        return httpx.Response(200, json={})

    await retry_with_backoff(fn, name="test")
    assert sleeps and sleeps[0] == 2.5


@pytest.mark.asyncio
async def test_retry_after_header_invalid_falls_back_to_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", fake_sleep)
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "not-a-number"}, json={})
        return httpx.Response(200, json={})

    await retry_with_backoff(fn, name="test")
    # Default backoff base = 0.5
    assert sleeps and sleeps[0] == 0.5
