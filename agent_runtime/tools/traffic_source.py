"""Traffic-source classifier — Task 01 from MVP-sprint.

Decides whether a Yandex.Direct campaign_id belongs to **our** Direct cabinet
(``own``), to a contractor's cabinet outside our control (``contractor``), or
has no UTM_CAMPAIGN attribution at all (``organic``).

The decision drives owner-report splits, ZeroLeadsAlarm gating (Task 08b)
and bitrix_feedback's traffic-split JSONB key (Task 02). Without it the agent
mistakes contractor leads for our own — the original 26.04.2026 owner
incident where 5 leads attributed to RSYA_Statya_moshenniki_709224565 were
treated as "our funnel".

Contract::

    from agent_runtime.tools import traffic_source
    from agent_runtime.tools.direct_api import DirectAPI

    async with DirectAPI(settings) as api:
        source = await traffic_source.classify(709353005, api=api)  # "own"
        source = await traffic_source.classify(709224565, api=api)  # "contractor"
        source = await traffic_source.classify(None, api=api)       # "organic"

Cache: per-process LRU+TTL (24h, max 200 keys). Failures (5xx) are NOT
cached so the next call retries; ``TokenExpiredError`` propagates because
that is an infra problem (token rotate), not a classify concern.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Literal

from agent_runtime.tools.direct_api import (
    DirectAPI,
    RateLimitError,
    UnknownDirectAPIError,
)

logger = logging.getLogger(__name__)


TrafficSource = Literal["own", "contractor", "organic"]


_CACHE_MAX = 200
_CACHE_TTL_SEC = 60 * 60 * 24  # 24h


class _TTLCache:
    """Minimal OrderedDict-backed LRU with per-entry expiry.

    Same shape as :class:`agent_runtime.knowledge._TTLCache` but specialised
    to ``str → TrafficSource`` so we never accidentally cache an arbitrary
    payload. Not thread-safe; asyncio is single-threaded per loop.
    """

    def __init__(self, maxsize: int, ttl_sec: int) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_sec
        self._data: OrderedDict[str, tuple[float, TrafficSource]] = OrderedDict()

    def get(self, key: str, *, now: float | None = None) -> TrafficSource | None:
        current = time.monotonic() if now is None else now
        entry = self._data.get(key)
        if entry is None:
            return None
        inserted_at, value = entry
        if current - inserted_at > self._ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: TrafficSource, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (current, value)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


_cache: _TTLCache = _TTLCache(maxsize=_CACHE_MAX, ttl_sec=_CACHE_TTL_SEC)


def invalidate_cache() -> None:
    """Drop every cached classification. Used by tests and by manual rotate."""
    _cache.clear()


def _normalise(campaign_id: int | str | None) -> int | None:
    """Coerce the input to ``int`` if possible; otherwise None.

    Whitespace is stripped from strings. Non-numeric strings, ``None`` and
    empty strings all return ``None`` so ``classify`` short-circuits to
    ``organic`` without touching the API.
    """
    if campaign_id is None:
        return None
    if isinstance(campaign_id, int):
        return campaign_id
    stripped = str(campaign_id).strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


async def classify(
    campaign_id: int | str | None,
    *,
    api: DirectAPI,
) -> TrafficSource:
    """Return ``own`` / ``contractor`` / ``organic`` for ``campaign_id``.

    * ``own``        — :meth:`DirectAPI.get_campaigns` returned a non-empty
                       list (cid is in our Direct cabinet)
    * ``contractor`` — empty list (cid not in our cabinet)
    * ``organic``    — no/invalid campaign_id, OR transient 5xx (not cached)

    ``api`` is injected so tests can pass an ``AsyncMock`` without
    monkeypatching. Callers in production typically reuse the existing
    ``async with DirectAPI(settings) as api:`` context from the surrounding
    job (e.g. ``bitrix_feedback._fetch_spend_per_campaign``).

    Raises:
        TokenExpiredError: propagated — this is an infra problem, not
            something ``classify`` can recover from. Caller should rotate
            the OAuth token and retry.
    """
    cid = _normalise(campaign_id)
    if cid is None:
        return "organic"

    key = str(cid)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        campaigns = await api.get_campaigns([cid])
    except (RateLimitError, UnknownDirectAPIError) as exc:
        # Transient failure — DO NOT cache, the next call should retry.
        logger.warning(
            "traffic_source.classify cid=%s API failure (%s) → organic (no cache)",
            cid,
            exc.__class__.__name__,
        )
        return "organic"

    result: TrafficSource = "own" if campaigns else "contractor"
    logger.info("traffic_source.classify miss cid=%s result=%s", cid, result)
    _cache.set(key, result)
    return result


__all__ = [
    "TrafficSource",
    "classify",
    "invalidate_cache",
]
