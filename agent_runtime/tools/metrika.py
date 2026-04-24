"""Yandex.Metrika stat/v1 client (Task 8).

Thin async helpers over the shared ``httpx.AsyncClient``. Counter ID comes
from ``Settings.METRIKA_COUNTER_ID`` (Decision 17 — single source of truth,
hardcoded 107734488 for 24bankrotsttvo.ru).

Gotcha: bounce-by-campaign uses ``ym:s:lastSignDirect*`` attribution, NOT
``ym:s:directCampaignID``. The second gives last-click only and diverges from
Direct UI. See ``memory/feedback_metrika_api_attributes.md``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from agent_runtime.config import Settings
from agent_runtime.tools._http import retry_with_backoff

logger = logging.getLogger(__name__)

_SEMAPHORE = asyncio.Semaphore(4)  # ≈30 req/min budget with headroom
_BASE_URL = "https://api-metrika.yandex.net/stat/v1/data"


class MetrikaAPIError(Exception):
    def __init__(self, code: str, message: str, status_code: int | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"[{code}] {message}")


def _auth_headers(settings: Settings) -> dict[str, str]:
    token = settings.METRIKA_OAUTH_TOKEN.get_secret_value()
    if not token:
        raise RuntimeError(
            "METRIKA_OAUTH_TOKEN empty — set in Railway env before calling metrika tools"
        )
    return {"Authorization": f"OAuth {token}"}


async def get_stats(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    metrics: list[str],
    dimensions: list[str] | None = None,
    filters: str | None = None,
    date1: str,
    date2: str,
    limit: int = 1000,
) -> dict[str, Any]:
    """Generic ``stat/v1/data`` query. Returns the full Metrika response dict."""
    params: dict[str, Any] = {
        "ids": settings.METRIKA_COUNTER_ID,
        "metrics": ",".join(metrics),
        "date1": date1,
        "date2": date2,
        "limit": limit,
        "accuracy": "full",
    }
    if dimensions:
        params["dimensions"] = ",".join(dimensions)
    if filters:
        params["filters"] = filters

    async with _SEMAPHORE:
        response = await retry_with_backoff(
            lambda: client.get(
                _BASE_URL,
                params=params,
                headers=_auth_headers(settings),
                timeout=60.0,
            ),
            name="metrika.get_stats",
        )

    if response.status_code >= 400:
        try:
            data = response.json()
        except ValueError:
            data = {}
        code = str(data.get("code") or f"HTTP_{response.status_code}")
        message = str(data.get("message") or response.reason_phrase or "")
        raise MetrikaAPIError(code, message, response.status_code)

    try:
        return dict(response.json())
    except ValueError as exc:
        raise MetrikaAPIError("INVALID_JSON", str(exc), response.status_code) from exc


async def get_bounce_by_campaign(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    date1: str,
    date2: str,
) -> dict[int, float]:
    """Return ``{direct_campaign_id: bounce_rate}`` over [date1, date2].

    Uses ``lastSignDirectOrderID`` attribution — divergence with Direct UI is
    a known Metrika gotcha; ``lastSign*`` aligns with SDA's hypothesis model.
    """
    data = await get_stats(
        client,
        settings,
        metrics=["ym:s:bounceRate"],
        dimensions=["ym:s:lastSignDirectOrderID"],
        date1=date1,
        date2=date2,
    )
    result: dict[int, float] = {}
    for row in data.get("data") or []:
        dims = row.get("dimensions") or []
        metrics = row.get("metrics") or []
        if not dims or not metrics:
            continue
        raw_id = dims[0].get("id") or dims[0].get("name")
        try:
            campaign_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        result[campaign_id] = float(metrics[0])
    logger.info("metrika.bounce: %d campaigns", len(result))
    return result


async def get_conversions(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    goal_ids: list[int],
    date1: str,
    date2: str,
) -> dict[int, int]:
    """Return ``{goal_id: reaches}`` for each goal across the window."""
    if not goal_ids:
        return {}
    metrics = [f"ym:s:goal{gid}reaches" for gid in goal_ids]
    data = await get_stats(
        client,
        settings,
        metrics=metrics,
        date1=date1,
        date2=date2,
    )
    totals = (data.get("totals") or [[]])[0]
    return {goal_id: int(totals[i]) if i < len(totals) else 0 for i, goal_id in enumerate(goal_ids)}


async def get_trajectories(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    dimension: str,
    date1: str,
    date2: str,
) -> list[dict[str, Any]]:
    """Return raw ``data`` rows — caller extracts dimension → metrics map.

    Default use: ``dimension='ym:s:lastSignUTMContent'`` for attribution of
    confirmed hypotheses to specific ad variants.
    """
    data = await get_stats(
        client,
        settings,
        metrics=["ym:s:visits", "ym:s:users"],
        dimensions=[dimension],
        date1=date1,
        date2=date2,
    )
    return list(data.get("data") or [])


__all__ = [
    "MetrikaAPIError",
    "get_bounce_by_campaign",
    "get_conversions",
    "get_stats",
    "get_trajectories",
]
