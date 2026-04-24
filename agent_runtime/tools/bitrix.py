"""Bitrix24 REST client (Task 8).

Pure-async helpers over the shared ``httpx.AsyncClient`` from
``agent_runtime.main`` lifespan. No module-level state other than a per-module
semaphore for Bitrix's 2 rps throttle.

**PII barrier.** Bitrix lead/deal responses include PHONE / NAME /
SOURCE_DESCRIPTION. This module does not write to ``audit_log`` itself —
callers route through ``agent_runtime.db.insert_audit_log`` which runs
``sanitize_audit_payload``. Module-level logs never include values, only
counts / keys.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Any

import httpx

from agent_runtime.config import Settings
from agent_runtime.tools._http import retry_with_backoff

logger = logging.getLogger(__name__)

_SEMAPHORE = asyncio.Semaphore(2)  # Bitrix REST hard limit
_BITRIX_BATCH_SIZE = 50


class BitrixAPIError(Exception):
    """Bitrix returned a non-retryable error or HTTP 4xx."""

    def __init__(self, code: str, description: str, status_code: int | None = None):
        self.code = code
        self.description = _clip(description)
        self.status_code = status_code
        super().__init__(f"[{code}] {self.description}")


def _clip(s: str, limit: int = 200) -> str:
    return s if len(s) <= limit else s[:limit] + "…"


def validate_webhook_token(incoming_token: str, settings: Settings) -> bool:
    """Constant-time compare for inbound Bitrix webhook token."""
    expected = settings.BITRIX_WEBHOOK_TOKEN.get_secret_value()
    return hmac.compare_digest(incoming_token, expected)


async def _call(
    client: httpx.AsyncClient,
    settings: Settings,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    base = settings.BITRIX_WEBHOOK_URL
    if not base:
        raise RuntimeError(
            "BITRIX_WEBHOOK_URL empty — set in Railway env before calling bitrix tools"
        )
    url = f"{base.rstrip('/')}/{method}.json"
    async with _SEMAPHORE:
        response = await retry_with_backoff(
            lambda: client.post(url, json=params, timeout=30.0),
            name=f"bitrix.{method}",
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise BitrixAPIError("INVALID_JSON", str(exc), response.status_code) from exc

    if response.status_code >= 400 or "error" in data:
        code = str(data.get("error") or f"HTTP_{response.status_code}")
        description = str(data.get("error_description") or response.reason_phrase or "")
        # Bitrix signals rate-limit via error_description AFTER retry exhausted.
        raise BitrixAPIError(code, description, response.status_code)

    return data


async def get_lead_list(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    filter: dict[str, Any] | None = None,
    select: list[str] | None = None,
    max_total: int = 1000,
) -> list[dict[str, Any]]:
    """Paginated ``crm.lead.list`` — returns up to ``max_total`` leads."""
    return await _paginated(
        client, settings, "crm.lead.list", filter=filter, select=select, max_total=max_total
    )


async def get_deal_list(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    filter: dict[str, Any] | None = None,
    select: list[str] | None = None,
    max_total: int = 1000,
) -> list[dict[str, Any]]:
    """Paginated ``crm.deal.list``. Common filter: ``{"STAGE_ID": "C45:WON"}``."""
    return await _paginated(
        client, settings, "crm.deal.list", filter=filter, select=select, max_total=max_total
    )


async def get_stage_history(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    entity_type_id: int,
    filter: dict[str, Any] | None = None,
    max_total: int = 1000,
) -> list[dict[str, Any]]:
    """``crm.stagehistory.list`` — entity_type_id: 1=Lead, 2=Deal.

    Used by strategy_gate (Task 17) to count transitions into ``C45:WON``
    over a 30d window.
    """
    params: dict[str, Any] = {"entityTypeId": entity_type_id}
    if filter:
        params["filter"] = filter
    return await _paginated_generic(client, settings, "crm.stagehistory.list", params, max_total)


async def _paginated(
    client: httpx.AsyncClient,
    settings: Settings,
    method: str,
    *,
    filter: dict[str, Any] | None,
    select: list[str] | None,
    max_total: int,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if filter:
        params["filter"] = filter
    if select:
        params["select"] = select
    return await _paginated_generic(client, settings, method, params, max_total)


async def _paginated_generic(
    client: httpx.AsyncClient,
    settings: Settings,
    method: str,
    params: dict[str, Any],
    max_total: int,
) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    cursor: int = 0
    while True:
        page_params = {**params, "start": cursor}
        data = await _call(client, settings, method, page_params)
        items = data.get("result") or []
        if not isinstance(items, list):
            # crm.stagehistory.list wraps rows under result.items
            container = items if isinstance(items, dict) else {}
            items = container.get("items") or []
        aggregated.extend(items)
        logger.info("%s: %d records (page_start=%d)", method, len(items), cursor)
        next_cursor = data.get("next")
        if next_cursor is None or len(aggregated) >= max_total:
            break
        cursor = int(next_cursor)
    return aggregated[:max_total]


__all__ = [
    "BitrixAPIError",
    "get_deal_list",
    "get_lead_list",
    "get_stage_history",
    "validate_webhook_token",
]
