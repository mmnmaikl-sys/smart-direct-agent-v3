"""Yandex.Direct API v5 wrapper with GET-after-SET invariant (Task 7).

Every mutating public method (``set_*``, ``add_*``, ``pause_*``, ``resume_*``,
``update_*``) ships with a paired ``verify_*`` that reads fresh state through
the API and returns ``bool``. A CI-time AST gate (``scripts/check_verify_pairing.py``)
fails the build if a mutation lacks its verifier — Decision 3 made that an
enforced invariant after v2.1 shipped un-verified bid changes.

Three defences stack in front of every mutation:

1. **PROTECTED_CAMPAIGN_IDS runtime guard** — ``_check_not_protected`` raises
   ``ProtectedCampaignError`` *before* any HTTP request leaves the process.
   Dups LLM-level validation (Decision 12) so a prompt-injected brain still
   cannot touch live-money campaigns.
2. **Typed exceptions** — every documented Direct error code maps to a class.
   Callers switch on the class, not string parsing.
3. **Rate limit + retry** — 10 req/s semaphore; exponential backoff (0.5s ×
   2^n, max 3 attempts) on 429/5xx / transport errors.

Money fields ship in micro-rubles (``rubles × 1_000_000``) per the v5 API
contract — helpers below do the conversion at the boundary so callers speak
plain rubles.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import Any

import httpx

from agent_runtime.config import Settings

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------- constants


_RATE_LIMIT_CONCURRENCY = 10
_RATE_LIMIT_WINDOW_SEC = 1.0
_MAX_RETRIES = 3
_RETRY_BASE_BACKOFF_SEC = 0.5
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_MICRO_MULTIPLIER = 1_000_000
_ADGROUP_CAMPAIGN_TTL_SEC = 60

# Direct error codes that collapse into typed exceptions.
_TOKEN_EXPIRED_CODE = 53


# -------------------------------------------------------------------- exceptions


class DirectAPIError(Exception):
    """Base class for any Direct API failure."""


class ProtectedCampaignError(DirectAPIError):
    """Mutation targeted a campaign in ``PROTECTED_CAMPAIGN_IDS`` — blocked."""


class TokenExpiredError(DirectAPIError):
    """Direct returned error_code=53 (token expired / invalid)."""


class RateLimitError(DirectAPIError):
    """Exhausted retries on HTTP 429."""


class InvalidRequestError(DirectAPIError):
    """Direct returned a 4xx error_code (bad params, state conflict, etc.)."""


class VerifyMismatchError(DirectAPIError):
    """Post-mutation ``verify_*`` call disagreed with expected state.

    Raised by the ReAct tool wrapper (Task 12 brain), *not* by ``verify_*``
    themselves — those return ``bool`` so they can also be called standalone.
    """


class UnknownDirectAPIError(DirectAPIError):
    """Any non-mapped Direct API failure (5xx after retries, unknown codes)."""


# ------------------------------------------------------------- adgroup→campaign


class _TTLMap:
    """Tiny per-instance TTL cache for adgroup_id → campaign_id lookups.

    Used by ``_resolve_campaign_for_adgroup`` so the protected-campaign guard
    does not spend an API call per mutation on a hot path.
    """

    def __init__(self, ttl_sec: int = _ADGROUP_CAMPAIGN_TTL_SEC) -> None:
        self._ttl = ttl_sec
        self._data: dict[int, tuple[float, int]] = {}

    def get(self, key: int, *, now: float | None = None) -> int | None:
        current = time.monotonic() if now is None else now
        entry = self._data.get(key)
        if entry is None:
            return None
        inserted_at, value = entry
        if current - inserted_at > self._ttl:
            del self._data[key]
            return None
        return value

    def set(self, key: int, value: int, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        self._data[key] = (current, value)


# ------------------------------------------------------------------- main class


class DirectAPI:
    """Async Direct v5 client with protected-campaign guard and verify pairs.

    Use as an async context manager::

        async with DirectAPI(settings) as api:
            keywords = await api.get_keywords([12345])
            await api.set_bid(keywords[0]["Id"], bid_rub=20)

    Error-code → exception mapping (used by ``_raise_for_error``):

    ======= =============================== =========================
    code    meaning                         exception
    ======= =============================== =========================
    53      token expired / invalid         TokenExpiredError
    8000    missing required field          InvalidRequestError
    8300    state conflict (DRAFT, etc.)    InvalidRequestError
    400-499 any 4xx                         InvalidRequestError
    500-599 any 5xx (post-retry)            UnknownDirectAPIError
    other   unmapped                        UnknownDirectAPIError
    ======= =============================== =========================
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._protected = frozenset(settings.PROTECTED_CAMPAIGN_IDS)
        self._base_url = settings.YANDEX_DIRECT_BASE_URL.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(_RATE_LIMIT_CONCURRENCY)
        self._adgroup_to_campaign = _TTLMap()

    def _headers(self) -> dict[str, str]:
        token = self._settings.YANDEX_DIRECT_TOKEN.get_secret_value()
        if not token:
            raise RuntimeError(
                "YANDEX_DIRECT_TOKEN is empty — set it in Railway env "
                "before instantiating DirectAPI"
            )
        return {
            "Authorization": f"Bearer {token}",
            "Accept-Language": "ru",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def __aenter__(self) -> DirectAPI:
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers=self._headers(),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------- transport

    async def _call(self, service: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute one JSON v5 request with retry + typed-exception translation."""
        if self._client is None:
            raise RuntimeError("DirectAPI used outside `async with` block")

        url = f"{self._base_url}/{service}"
        body = {"method": method, "params": params}

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with self._semaphore:
                    response = await self._client.post(url, json=body)
            except httpx.TransportError:
                if attempt == _MAX_RETRIES:
                    raise UnknownDirectAPIError(
                        f"transport error after {attempt} attempts: {service}.{method}"
                    ) from None
                await asyncio.sleep(_RETRY_BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
                continue

            if response.status_code in _RETRY_STATUS_CODES:
                if attempt == _MAX_RETRIES:
                    if response.status_code == 429:
                        raise RateLimitError(
                            f"rate-limited after {attempt} attempts: {service}.{method}"
                        )
                    raise UnknownDirectAPIError(
                        f"{service}.{method} returned HTTP {response.status_code} "
                        f"after {attempt} attempts"
                    )
                await asyncio.sleep(_RETRY_BASE_BACKOFF_SEC * (2 ** (attempt - 1)))
                continue

            data = response.json()
            if "error" in data:
                _raise_for_error(data["error"])
            return data.get("result", {})

        raise UnknownDirectAPIError("unreachable")

    # -------------------------------------------------------- protected guard

    async def _resolve_campaign_for_adgroup(self, ad_group_id: int) -> int:
        cached = self._adgroup_to_campaign.get(ad_group_id)
        if cached is not None:
            return cached
        result = await self._call(
            "adgroups",
            "get",
            {"SelectionCriteria": {"Ids": [ad_group_id]}, "FieldNames": ["Id", "CampaignId"]},
        )
        items = result.get("AdGroups") or []
        if not items:
            raise InvalidRequestError(f"adgroup {ad_group_id} not found")
        campaign_id = int(items[0]["CampaignId"])
        self._adgroup_to_campaign.set(ad_group_id, campaign_id)
        return campaign_id

    async def _check_not_protected(
        self,
        *,
        campaign_id: int | None = None,
        ad_group_id: int | None = None,
    ) -> None:
        target = campaign_id
        if target is None and ad_group_id is not None:
            target = await self._resolve_campaign_for_adgroup(ad_group_id)
        if target is None:
            raise InvalidRequestError("mutation requires campaign_id or ad_group_id")
        if target in self._protected:
            raise ProtectedCampaignError(
                f"campaign {target} is in PROTECTED_CAMPAIGN_IDS; mutation forbidden"
            )

    # -------------------------------------------------------------------- GET

    async def get_campaigns(self, ids: list[int]) -> list[dict[str, Any]]:
        result = await self._call(
            "campaigns",
            "get",
            {
                "SelectionCriteria": {"Ids": ids},
                "FieldNames": ["Id", "Name", "State", "Status", "Type"],
            },
        )
        return list(result.get("Campaigns") or [])

    async def get_adgroups(
        self,
        *,
        campaign_id: int | None = None,
        ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        criteria: dict[str, Any] = {}
        if ids:
            criteria["Ids"] = ids
        if campaign_id is not None:
            criteria["CampaignIds"] = [campaign_id]
        result = await self._call(
            "adgroups",
            "get",
            {
                "SelectionCriteria": criteria,
                "FieldNames": ["Id", "Name", "CampaignId", "Status", "Type"],
            },
        )
        return list(result.get("AdGroups") or [])

    async def get_ads(self, ad_group_ids: list[int]) -> list[dict[str, Any]]:
        result = await self._call(
            "ads",
            "get",
            {
                "SelectionCriteria": {"AdGroupIds": ad_group_ids},
                "FieldNames": ["Id", "AdGroupId", "State", "Status", "Type"],
                "TextAdFieldNames": ["Title", "Title2", "Text", "Href"],
            },
        )
        return list(result.get("Ads") or [])

    async def get_keywords(self, ad_group_ids: list[int]) -> list[dict[str, Any]]:
        result = await self._call(
            "keywords",
            "get",
            {
                "SelectionCriteria": {"AdGroupIds": ad_group_ids},
                "FieldNames": ["Id", "AdGroupId", "Keyword", "Bid", "ContextBid", "State"],
            },
        )
        return list(result.get("Keywords") or [])

    async def get_campaign_negative_keywords(self, campaign_id: int) -> list[str]:
        result = await self._call(
            "campaigns",
            "get",
            {
                "SelectionCriteria": {"Ids": [campaign_id]},
                "FieldNames": ["Id", "NegativeKeywordsSharedSet", "NegativeKeywords"],
            },
        )
        campaigns = result.get("Campaigns") or []
        if not campaigns:
            return []
        container = campaigns[0].get("NegativeKeywords") or {}
        return list(container.get("Items") or [])

    async def get_bid_modifiers(self, campaign_ids: list[int]) -> list[dict[str, Any]]:
        # Levels lives INSIDE SelectionCriteria per v5 contract — breaking this
        # returns an opaque 400 and costs half a day of debugging.
        result = await self._call(
            "bidmodifiers",
            "get",
            {
                "SelectionCriteria": {"CampaignIds": campaign_ids, "Levels": ["CAMPAIGN"]},
                "FieldNames": ["Id", "CampaignId", "Type"],
            },
        )
        return list(result.get("BidModifiers") or [])

    async def get_campaign_stats(
        self,
        campaign_id: int,
        date_from: str,
        date_to: str,
    ) -> dict[str, Any]:
        # Reports endpoint returns 201/202 with retryIn while the report is
        # being built. Poll up to 10 times.
        if self._client is None:
            raise RuntimeError("DirectAPI used outside `async with` block")
        url = f"{self._base_url}/reports"
        body = {
            "params": {
                "SelectionCriteria": {
                    "DateFrom": date_from,
                    "DateTo": date_to,
                    "Filter": [
                        {"Field": "CampaignId", "Operator": "EQUALS", "Values": [str(campaign_id)]}
                    ],
                },
                "FieldNames": ["CampaignId", "Impressions", "Clicks", "Cost", "Conversions"],
                "ReportName": f"sda_v3_{campaign_id}_{date_from}_{date_to}",
                "ReportType": "CAMPAIGN_PERFORMANCE_REPORT",
                "DateRangeType": "CUSTOM_DATE",
                "Format": "TSV",
                "IncludeVAT": "NO",
            }
        }
        for attempt in range(10):
            response = await self._client.post(url, json=body)
            if response.status_code == 200:
                return {"tsv": response.text}
            if response.status_code in (201, 202):
                retry_in = int(response.headers.get("retryIn", 5))
                await asyncio.sleep(retry_in)
                continue
            raise UnknownDirectAPIError(
                f"reports returned HTTP {response.status_code} on attempt {attempt + 1}"
            )
        raise DirectAPIError(f"report for campaign {campaign_id} not ready after 10 polls")

    # ------------------------------------------------------------ SET + verify

    async def set_bid(
        self,
        keyword_id: int,
        bid_rub: int,
        context_bid_rub: int | None = None,
    ) -> dict[str, Any]:
        # SetBids does not give us the campaign — look it up via the keyword's
        # adgroup before guarding. The adgroup→campaign cache absorbs the cost
        # when jobs mutate multiple keywords in the same group.
        ad_group_id = await self._adgroup_for_keyword(keyword_id)
        await self._check_not_protected(ad_group_id=ad_group_id)
        bid_entry: dict[str, Any] = {
            "KeywordId": keyword_id,
            "SearchBid": bid_rub * _MICRO_MULTIPLIER,
        }
        if context_bid_rub is not None:
            bid_entry["NetworkBid"] = context_bid_rub * _MICRO_MULTIPLIER
        logger.info(
            "set_bid keyword=%d bid_rub=%d context=%s",
            keyword_id,
            bid_rub,
            context_bid_rub,
        )
        return await self._call("bids", "set", {"Bids": [bid_entry]})

    async def verify_bid(
        self,
        keyword_id: int,
        expected_bid_rub: int,
        expected_context_bid_rub: int | None = None,
    ) -> bool:
        ad_group_id = await self._adgroup_for_keyword(keyword_id)
        keywords = await self.get_keywords([ad_group_id])
        entry = next((k for k in keywords if int(k["Id"]) == keyword_id), None)
        if entry is None:
            return False
        actual_search = int(entry.get("Bid", 0)) // _MICRO_MULTIPLIER
        if actual_search != expected_bid_rub:
            return False
        if expected_context_bid_rub is not None:
            actual_ctx = int(entry.get("ContextBid", 0)) // _MICRO_MULTIPLIER
            if actual_ctx != expected_context_bid_rub:
                return False
        return True

    async def add_negatives(self, campaign_id: int, phrases: list[str]) -> dict[str, Any]:
        await self._check_not_protected(campaign_id=campaign_id)
        existing = await self.get_campaign_negative_keywords(campaign_id)
        merged = list(dict.fromkeys([*existing, *phrases]))  # preserves order, dedups
        logger.info(
            "add_negatives campaign=%d added=%d total=%d",
            campaign_id,
            len(phrases),
            len(merged),
        )
        return await self._call(
            "campaigns",
            "update",
            {
                "Campaigns": [
                    {
                        "Id": campaign_id,
                        "NegativeKeywords": {"Items": merged},
                    }
                ]
            },
        )

    async def verify_negatives_added(self, campaign_id: int, expected_phrases: list[str]) -> bool:
        actual = set(await self.get_campaign_negative_keywords(campaign_id))
        return set(expected_phrases).issubset(actual)

    async def pause_group(self, ad_group_id: int) -> dict[str, Any]:
        await self._check_not_protected(ad_group_id=ad_group_id)
        return await self._call(
            "adgroups", "suspend", {"SelectionCriteria": {"Ids": [ad_group_id]}}
        )

    async def verify_group_paused(self, ad_group_id: int) -> bool:
        groups = await self.get_adgroups(ids=[ad_group_id])
        if not groups:
            return False
        status = groups[0].get("Status")
        return status in {"SUSPENDED", "PAUSED", "STOPPED"}

    async def resume_group(self, ad_group_id: int) -> dict[str, Any]:
        await self._check_not_protected(ad_group_id=ad_group_id)
        return await self._call("adgroups", "resume", {"SelectionCriteria": {"Ids": [ad_group_id]}})

    async def verify_group_resumed(self, ad_group_id: int) -> bool:
        groups = await self.get_adgroups(ids=[ad_group_id])
        if not groups:
            return False
        return groups[0].get("Status") in {"ON", "ACCEPTED", "READY"}

    async def pause_campaign(self, campaign_id: int) -> dict[str, Any]:
        await self._check_not_protected(campaign_id=campaign_id)
        return await self._call(
            "campaigns", "suspend", {"SelectionCriteria": {"Ids": [campaign_id]}}
        )

    async def verify_campaign_paused(self, campaign_id: int) -> bool:
        campaigns = await self.get_campaigns([campaign_id])
        if not campaigns:
            return False
        return campaigns[0].get("State") in {"SUSPENDED", "OFF"}

    async def resume_campaign(self, campaign_id: int) -> dict[str, Any]:
        await self._check_not_protected(campaign_id=campaign_id)
        # Direct rejects `resume` on DRAFT campaigns with error 8300 — fail
        # early with a clear hint instead of letting the API 400 surface.
        campaigns = await self.get_campaigns([campaign_id])
        if campaigns and campaigns[0].get("Status") == "DRAFT":
            raise InvalidRequestError(
                f"campaign {campaign_id} is DRAFT; moderate ads before resume"
            )
        return await self._call(
            "campaigns", "resume", {"SelectionCriteria": {"Ids": [campaign_id]}}
        )

    async def verify_campaign_resumed(self, campaign_id: int) -> bool:
        campaigns = await self.get_campaigns([campaign_id])
        if not campaigns:
            return False
        return campaigns[0].get("State") in {"ON", "CONVERTED"}

    async def update_ad_href(self, ad_id: int, href: str) -> dict[str, Any]:
        ads_data = await self._call(
            "ads",
            "get",
            {
                "SelectionCriteria": {"Ids": [ad_id]},
                "FieldNames": ["Id", "AdGroupId"],
            },
        )
        ads = ads_data.get("Ads") or []
        if not ads:
            raise InvalidRequestError(f"ad {ad_id} not found")
        ad_group_id = int(ads[0]["AdGroupId"])
        await self._check_not_protected(ad_group_id=ad_group_id)
        return await self._call(
            "ads",
            "update",
            {"Ads": [{"Id": ad_id, "TextAd": {"Href": href}}]},
        )

    async def verify_ad_href(self, ad_id: int, expected_href: str) -> bool:
        ads_data = await self._call(
            "ads",
            "get",
            {
                "SelectionCriteria": {"Ids": [ad_id]},
                "FieldNames": ["Id"],
                "TextAdFieldNames": ["Href"],
            },
        )
        ads = ads_data.get("Ads") or []
        if not ads:
            return False
        text_ad = ads[0].get("TextAd") or {}
        return text_ad.get("Href") == expected_href

    async def update_strategy(self, campaign_id: int, strategy: dict[str, Any]) -> dict[str, Any]:
        await self._check_not_protected(campaign_id=campaign_id)
        # BudgetType is read-only on SET — Direct returns an opaque error if
        # we echo it back. AttributionModel=LYDC was deprecated; keep it out
        # of the payload.
        sanitised = _prune_readonly_strategy_fields(strategy)
        return await self._call(
            "campaigns",
            "update",
            {"Campaigns": [{"Id": campaign_id, "Strategy": sanitised}]},
        )

    async def verify_strategy(self, campaign_id: int, expected_strategy: dict[str, Any]) -> bool:
        result = await self._call(
            "campaigns",
            "get",
            {
                "SelectionCriteria": {"Ids": [campaign_id]},
                "FieldNames": ["Id", "Strategy"],
            },
        )
        campaigns = result.get("Campaigns") or []
        if not campaigns:
            return False
        actual = _prune_readonly_strategy_fields(campaigns[0].get("Strategy") or {})
        expected = _prune_readonly_strategy_fields(expected_strategy)
        return actual == expected

    # --------------------------------------------------------------- helpers

    async def _adgroup_for_keyword(self, keyword_id: int) -> int:
        result = await self._call(
            "keywords",
            "get",
            {
                "SelectionCriteria": {"Ids": [keyword_id]},
                "FieldNames": ["Id", "AdGroupId"],
            },
        )
        keywords = result.get("Keywords") or []
        if not keywords:
            raise InvalidRequestError(f"keyword {keyword_id} not found")
        return int(keywords[0]["AdGroupId"])


# ---------------------------------------------------------------- module helpers


def _raise_for_error(err: dict[str, Any]) -> None:
    code = int(err.get("error_code", 0))
    detail = err.get("error_detail") or err.get("error_string") or ""
    message = f"{code}: {detail}"
    if code == _TOKEN_EXPIRED_CODE:
        raise TokenExpiredError(message)
    if 400 <= code < 10000:
        raise InvalidRequestError(message)
    raise UnknownDirectAPIError(message)


_STRATEGY_IGNORED_FIELDS: frozenset[str] = frozenset({"BudgetType", "AttributionModel"})


def _prune_readonly_strategy_fields(strategy: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``strategy`` with API-normalised read-only keys dropped.

    Network / Search sub-dicts carry ``BudgetType`` on GET that SET rejects;
    ``AttributionModel=LYDC`` is silently rewritten by the API with a warning
    so we drop it from comparisons too.
    """
    pruned: dict[str, Any] = {}
    for key, value in strategy.items():
        if key in _STRATEGY_IGNORED_FIELDS:
            continue
        if isinstance(value, dict):
            pruned[key] = _prune_readonly_strategy_fields(value)
        else:
            pruned[key] = value
    return pruned


__all__ = [
    "DirectAPI",
    "DirectAPIError",
    "InvalidRequestError",
    "ProtectedCampaignError",
    "RateLimitError",
    "TokenExpiredError",
    "UnknownDirectAPIError",
    "VerifyMismatchError",
]
