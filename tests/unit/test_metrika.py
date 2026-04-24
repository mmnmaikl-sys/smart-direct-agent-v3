"""Unit tests for ``agent_runtime.tools.metrika``."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from agent_runtime.config import Settings
from agent_runtime.tools.metrika import (
    MetrikaAPIError,
    get_bounce_by_campaign,
    get_conversions,
    get_stats,
    get_trajectories,
)


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        METRIKA_OAUTH_TOKEN="test-oauth-token",
        METRIKA_COUNTER_ID=107734488,
    )


class _Recorder:
    def __init__(self, handler):
        self.calls: list[dict[str, Any]] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(
            {
                "url": str(request.url),
                "params": dict(request.url.params),
                "headers": dict(request.headers),
            }
        )
        return self._handler(request, len(self.calls))


async def _client(handler):
    recorder = _Recorder(handler)
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return client, recorder


@pytest.mark.asyncio
async def test_get_stats_basic() -> None:
    def handler(req, n):
        return httpx.Response(
            200,
            json={
                "data": [{"dimensions": [{"id": "123"}], "metrics": [0.42]}],
                "totals": [[0.42]],
            },
        )

    client, rec = await _client(handler)
    try:
        data = await get_stats(
            client,
            _settings(),
            metrics=["ym:s:bounceRate"],
            dimensions=["ym:s:lastSignDirectOrderID"],
            date1="2026-04-01",
            date2="2026-04-07",
        )
    finally:
        await client.aclose()
    assert data["totals"][0][0] == 0.42
    # Auth: OAuth scheme (not Bearer)
    auth = rec.calls[0]["headers"]["authorization"]
    assert auth.startswith("OAuth ")


@pytest.mark.asyncio
async def test_get_bounce_uses_lastSign_attribution() -> None:
    def handler(req, n):
        return httpx.Response(200, json={"data": [], "totals": [[]]})

    client, rec = await _client(handler)
    try:
        await get_bounce_by_campaign(client, _settings(), date1="2026-04-01", date2="2026-04-07")
    finally:
        await client.aclose()
    dimensions = rec.calls[0]["params"]["dimensions"]
    assert (
        "lastSignDirect" in dimensions
    ), "bounce must use lastSign* attribution, not directCampaignID (Metrika gotcha)"


@pytest.mark.asyncio
async def test_get_bounce_parses_rows() -> None:
    def handler(req, n):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"dimensions": [{"id": "708978456"}], "metrics": [0.45]},
                    {"dimensions": [{"id": "708978457"}], "metrics": [0.30]},
                ]
            },
        )

    client, _ = await _client(handler)
    try:
        result = await get_bounce_by_campaign(
            client, _settings(), date1="2026-04-01", date2="2026-04-07"
        )
    finally:
        await client.aclose()
    assert result == {708978456: 0.45, 708978457: 0.30}


@pytest.mark.asyncio
async def test_get_conversions_maps_goal_ids() -> None:
    def handler(req, n):
        return httpx.Response(200, json={"totals": [[5, 12]]})

    client, rec = await _client(handler)
    try:
        result = await get_conversions(
            client,
            _settings(),
            goal_ids=[546909177, 546909178],
            date1="2026-04-01",
            date2="2026-04-07",
        )
    finally:
        await client.aclose()
    assert result == {546909177: 5, 546909178: 12}
    # Metrics list is per-goal
    metrics = rec.calls[0]["params"]["metrics"]
    assert "goal546909177reaches" in metrics
    assert "goal546909178reaches" in metrics


@pytest.mark.asyncio
async def test_get_conversions_empty_goals_shortcircuits() -> None:
    def handler(req, n):
        pytest.fail("should not hit network for empty goal_ids")
        return httpx.Response(200)

    client, _ = await _client(handler)
    try:
        result = await get_conversions(
            client, _settings(), goal_ids=[], date1="2026-04-01", date2="2026-04-07"
        )
    finally:
        await client.aclose()
    assert result == {}


@pytest.mark.asyncio
async def test_get_trajectories_returns_rows() -> None:
    def handler(req, n):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"dimensions": [{"name": "utm_variant_a"}], "metrics": [100, 50]},
                    {"dimensions": [{"name": "utm_variant_b"}], "metrics": [80, 40]},
                ]
            },
        )

    client, _ = await _client(handler)
    try:
        rows = await get_trajectories(
            client,
            _settings(),
            dimension="ym:s:lastSignUTMContent",
            date1="2026-04-01",
            date2="2026-04-07",
        )
    finally:
        await client.aclose()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_retry_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", _noop_sleep())

    def handler(req, n):
        if n == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"})
        return httpx.Response(200, json={"data": [], "totals": [[]]})

    client, rec = await _client(handler)
    try:
        await get_stats(
            client, _settings(), metrics=["ym:s:visits"], date1="2026-04-01", date2="2026-04-02"
        )
    finally:
        await client.aclose()
    assert len(rec.calls) == 2


@pytest.mark.asyncio
async def test_metrika_error_raises_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_runtime.tools._http.asyncio.sleep", _noop_sleep())

    def handler(req, n):
        return httpx.Response(403, json={"code": "FORBIDDEN", "message": "counter access denied"})

    client, _ = await _client(handler)
    try:
        with pytest.raises(MetrikaAPIError) as exc:
            await get_stats(
                client,
                _settings(),
                metrics=["ym:s:visits"],
                date1="2026-04-01",
                date2="2026-04-02",
            )
    finally:
        await client.aclose()
    assert exc.value.code == "FORBIDDEN"
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_missing_oauth_token_raises() -> None:
    bad = _settings()
    bad.METRIKA_OAUTH_TOKEN = bad.METRIKA_OAUTH_TOKEN.__class__("")

    client, _ = await _client(lambda req, n: httpx.Response(200, json={}))
    try:
        with pytest.raises(RuntimeError, match="METRIKA_OAUTH_TOKEN"):
            await get_stats(
                client,
                bad,
                metrics=["ym:s:visits"],
                date1="2026-04-01",
                date2="2026-04-02",
            )
    finally:
        await client.aclose()


def _noop_sleep():
    async def _sleep(_s: float) -> None:
        return None

    return _sleep
