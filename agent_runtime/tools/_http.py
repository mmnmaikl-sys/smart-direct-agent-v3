"""Shared async helpers for Task 8 clients (Bitrix / Metrika / Telegram).

Kept small and internal — single retry loop, single fallthrough exception.
Each call site wraps thrown ``httpx.HTTPError`` into its own typed error so
callers can distinguish Bitrix vs Metrika vs Telegram failures without
string-sniffing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BASE_BACKOFF_SEC = 0.5
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

T = TypeVar("T")


async def retry_with_backoff(
    fn: Callable[[], Awaitable[httpx.Response]],
    *,
    name: str,
    max_attempts: int = _MAX_ATTEMPTS,
) -> httpx.Response:
    """Call ``fn`` up to ``max_attempts`` with exp-backoff on 429/5xx/transport.

    The caller supplies a zero-arg closure so the same retry logic serves
    GET, POST, and different auth schemes without branching here. ``name``
    is used in log lines for operator-side traceability.
    """
    delay = _BASE_BACKOFF_SEC
    last_exc: Exception | None = None
    last_response: httpx.Response | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = await fn()
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            await asyncio.sleep(delay)
            delay *= 2
            continue

        if response.status_code in _RETRYABLE_STATUS:
            last_response = response
            retry_after = _parse_retry_after(response)
            if attempt == max_attempts:
                break
            await asyncio.sleep(retry_after or delay)
            delay *= 2
            continue

        return response

    if last_exc is not None:
        raise last_exc
    # last_response is guaranteed non-None here — we only exit the loop
    # either via transport-error (last_exc) or retryable-status (last_response).
    assert last_response is not None
    return last_response


def _parse_retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            return None
    # Telegram style: body {"parameters": {"retry_after": n}}
    try:
        data = response.json()
    except ValueError:
        return None
    if isinstance(data, dict):
        params = data.get("parameters") or {}
        value = params.get("retry_after")
        if isinstance(value, int | float):
            return float(value)
    return None


__all__ = ["retry_with_backoff"]
