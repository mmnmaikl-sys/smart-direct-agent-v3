"""Unit tests for agent_runtime.tools.traffic_source.classify() — Task 01 TDD anchor.

Real Yandex.Direct is never hit here — ``DirectAPI.get_campaigns`` is mocked
per test via ``unittest.mock.AsyncMock``. The classifier separates traffic
into:

  * ``own``        — campaign_id is in our Direct cabinet (get_campaigns
                     returns a non-empty list)
  * ``contractor`` — campaign_id is NOT in our cabinet (get_campaigns
                     returns an empty list)
  * ``organic``    — no campaign_id, bad input, or transient API failure

LRU+TTL cache absorbs duplicate classify() calls; failures (5xx) are NOT
cached so the next call retries.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_runtime.tools import traffic_source
from agent_runtime.tools.direct_api import (
    RateLimitError,
    TokenExpiredError,
    UnknownDirectAPIError,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the module-level LRU between tests so ordering is irrelevant."""
    traffic_source.invalidate_cache()
    yield
    traffic_source.invalidate_cache()


# --- happy-path classification ----------------------------------------------


@pytest.mark.asyncio
async def test_classify_own_nonempty_list() -> None:
    api = AsyncMock()
    api.get_campaigns = AsyncMock(
        return_value=[{"Id": 709353005, "Name": "rabotyaga test", "State": "OFF"}]
    )

    result = await traffic_source.classify(709353005, api=api)

    assert result == "own"
    api.get_campaigns.assert_awaited_once_with([709353005])


@pytest.mark.asyncio
async def test_classify_contractor_empty_list() -> None:
    """Direct API returned empty list → cid not in our cabinet → contractor."""
    api = AsyncMock()
    api.get_campaigns = AsyncMock(return_value=[])

    result = await traffic_source.classify(709224565, api=api)

    assert result == "contractor"
    api.get_campaigns.assert_awaited_once_with([709224565])


# --- organic short-circuits -------------------------------------------------


@pytest.mark.asyncio
async def test_classify_organic_none() -> None:
    """None campaign_id → organic without any API call."""
    api = AsyncMock()
    api.get_campaigns = AsyncMock()

    result = await traffic_source.classify(None, api=api)

    assert result == "organic"
    api.get_campaigns.assert_not_awaited()


@pytest.mark.asyncio
async def test_classify_organic_bad_input() -> None:
    """Non-integer-like string → organic without any API call."""
    api = AsyncMock()
    api.get_campaigns = AsyncMock()

    result = await traffic_source.classify("abc", api=api)

    assert result == "organic"
    api.get_campaigns.assert_not_awaited()


# --- cache behaviour --------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_skips_api_on_second_call() -> None:
    api = AsyncMock()
    api.get_campaigns = AsyncMock(return_value=[{"Id": 709353005, "Name": "rabotyaga test"}])

    first = await traffic_source.classify(709353005, api=api)
    second = await traffic_source.classify(709353005, api=api)

    assert first == "own"
    assert second == "own"
    # Only one API call across both invocations
    api.get_campaigns.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_invalidate_forces_refresh() -> None:
    api = AsyncMock()
    api.get_campaigns = AsyncMock(return_value=[{"Id": 709353005, "Name": "rabotyaga test"}])

    await traffic_source.classify(709353005, api=api)
    traffic_source.invalidate_cache()
    await traffic_source.classify(709353005, api=api)

    # Cache cleared between calls → API hit twice
    assert api.get_campaigns.await_count == 2


# --- failure modes ----------------------------------------------------------


@pytest.mark.asyncio
async def test_organic_on_rate_limit_no_cache() -> None:
    """Transient 5xx → organic but NOT cached, so the next call retries."""
    api = AsyncMock()
    api.get_campaigns = AsyncMock(side_effect=RateLimitError("rate limited"))

    first = await traffic_source.classify(709353005, api=api)
    assert first == "organic"

    # Reset side_effect so a retry could succeed
    api.get_campaigns = AsyncMock(return_value=[{"Id": 709353005, "Name": "rabotyaga test"}])
    second = await traffic_source.classify(709353005, api=api)
    assert second == "own"


@pytest.mark.asyncio
async def test_unknown_5xx_not_cached() -> None:
    """UnknownDirectAPIError (post-retry 5xx) → organic without cache."""
    api = AsyncMock()
    api.get_campaigns = AsyncMock(side_effect=UnknownDirectAPIError("500 internal"))

    result = await traffic_source.classify(709353005, api=api)
    assert result == "organic"

    # Second call must hit API again (not cached)
    api.get_campaigns = AsyncMock(return_value=[{"Id": 709353005, "Name": "x"}])
    second = await traffic_source.classify(709353005, api=api)
    assert second == "own"


@pytest.mark.asyncio
async def test_token_expired_propagates() -> None:
    """TokenExpiredError is infra problem, not a classify concern — propagate."""
    api = AsyncMock()
    api.get_campaigns = AsyncMock(side_effect=TokenExpiredError("token expired"))

    with pytest.raises(TokenExpiredError):
        await traffic_source.classify(709353005, api=api)


# --- input normalisation ----------------------------------------------------


@pytest.mark.asyncio
async def test_classify_string_int_normalised() -> None:
    """String '709353005' should classify same as int 709353005 (cache key shared)."""
    api = AsyncMock()
    api.get_campaigns = AsyncMock(return_value=[{"Id": 709353005, "Name": "rabotyaga test"}])

    int_result = await traffic_source.classify(709353005, api=api)
    str_result = await traffic_source.classify("709353005", api=api)

    assert int_result == "own"
    assert str_result == "own"
    # Cache hit on second call — only 1 API call
    api.get_campaigns.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_string_with_whitespace_stripped() -> None:
    """' 709353005 ' should strip and classify normally."""
    api = AsyncMock()
    api.get_campaigns = AsyncMock(return_value=[{"Id": 709353005, "Name": "rabotyaga test"}])

    result = await traffic_source.classify(" 709353005 ", api=api)
    assert result == "own"
