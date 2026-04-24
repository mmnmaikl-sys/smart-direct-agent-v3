"""Signal Detector — ported from v2 sync urllib to async httpx.

Reads Direct / Metrika / Bitrix / live landings every ``/tick`` and emits
typed :class:`~agent_runtime.models.Signal` facts that ``brain.reason()``
then turns into hypotheses. All six checks run in parallel via
``asyncio.gather(return_exceptions=True)`` so one flaky source does not
mask the other five — failures become ``SignalType.API_ERROR`` signals
and the agent still sees the rest of the picture.

Integration notes vs v2:
- ``Signal`` / ``SignalType`` imported from :mod:`agent_runtime.models`
  (Task 5 single source of truth — no local redefinition).
- HTTP goes through the shared ``httpx.AsyncClient`` owned by
  ``agent_runtime.main`` lifespan.
- Direct / Bitrix / Metrika access uses the async helpers from
  :mod:`agent_runtime.tools` (Task 7 / Task 8) — the sync v2 clients
  (``urllib.request.urlopen``) are gone.
- Thresholds (``TARGET_CPA`` / ``DAILY_BUDGET_LIMIT`` / landing URLs /
  Metrika counter) live in :class:`~agent_runtime.config.Settings` so
  they can be tuned via Railway env without a redeploy.

The detector does not write to audit_log or trigger notifications — those
are the brain wrapper's (Task 12) concerns. Emit-and-return only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.models import Signal, SignalType
from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)

MSK = timezone(timedelta(hours=3))

# Thresholds for emitted signal severities. Kept module-level (not settings)
# because they describe detector semantics, not business numbers.
_BUDGET_PCT_WARN = 0.70
_BUDGET_PCT_CRIT = 0.90
_SPEND_NO_CLICKS_MIN = 500  # RUB — ignore shoestring campaigns
_HIGH_BOUNCE_MIN_VISITS = 50
_HIGH_BOUNCE_WARN = 60.0  # percent
_HIGH_BOUNCE_CRIT = 80.0
_LANDING_MIN_SIZE = 1000  # bytes — smaller response = broken template
_LANDING_SLOW_SEC = 5.0
_LANDING_TIMEOUT_SEC = 10.0
_GARBAGE_MIN_CLICKS = 3
_CAMPAIGN_WORK_HOUR_START = 7
_CAMPAIGN_WORK_HOUR_END = 23  # exclusive upper bound
_ZERO_LEADS_WINDOW_DAYS = 3

# Sub-strings that mark a Bitrix lead as coming from our Direct pages.
_OUR_LEAD_MARKERS = ("24bankrotsttvo", "pages/ad")

# Names for telemetry when one of the detectors raises — surfaced as
# ``Signal.data.source``.
_DETECTOR_NAMES: tuple[str, ...] = (
    "budget",
    "bounce",
    "zero_leads",
    "landing_health",
    "garbage_queries",
    "campaign_state",
)


def _now_msk() -> datetime:
    return datetime.now(MSK)


class SignalDetector:
    """Parallel signal reader. All dependencies injected for testability."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        direct: DirectAPI,
        metrika: Any,  # metrika_tools module or equivalent (test stub ok)
        bitrix: Any,
        http: httpx.AsyncClient,
        settings: Settings,
    ) -> None:
        self._pool = pool
        self._direct = direct
        self._metrika = metrika
        self._bitrix = bitrix
        self._http = http
        self._settings = settings

    async def detect_all(self) -> list[Signal]:
        started = time.monotonic()
        coros = (
            self._check_budget(),
            self._check_bounce(),
            self._check_zero_leads(),
            self._check_landing_health(),
            self._check_garbage_queries(),
            self._check_campaign_state(),
        )
        results = await asyncio.gather(*coros, return_exceptions=True)

        signals: list[Signal] = []
        for name, result in zip(_DETECTOR_NAMES, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning("detector '%s' raised: %s", name, result)
                signals.append(
                    Signal(
                        type=SignalType.API_ERROR,
                        severity="warning",
                        data={"source": name, "error": str(result)[:200]},
                        ts=_now_msk(),
                    )
                )
            else:
                signals.extend(result)

        logger.info(
            "signal_detector: %d signals in %.2fs",
            len(signals),
            time.monotonic() - started,
        )
        return signals

    # --------------------------------------------------------- _check_budget

    async def _check_budget(self) -> list[Signal]:
        signals: list[Signal] = []
        today = _now_msk().date().isoformat()
        campaign_ids = list(self._settings.PROTECTED_CAMPAIGN_IDS)
        # Per-campaign so we can attribute the alert precisely.
        for campaign_id in campaign_ids:
            stats = await self._direct.get_campaign_stats(
                campaign_id, date_from=today, date_to=today
            )
            cost = float(stats.get("cost", 0))
            clicks = int(stats.get("clicks", 0))
            if cost >= self._settings.DAILY_BUDGET_LIMIT * _BUDGET_PCT_CRIT:
                severity = "critical"
            elif cost >= self._settings.DAILY_BUDGET_LIMIT * _BUDGET_PCT_WARN:
                severity = "warning"
            else:
                severity = None
            if severity is not None:
                pct = (
                    round(cost / self._settings.DAILY_BUDGET_LIMIT * 100)
                    if self._settings.DAILY_BUDGET_LIMIT
                    else 0
                )
                signals.append(
                    Signal(
                        type=SignalType.BUDGET_THRESHOLD,
                        severity=severity,
                        data={
                            "campaign_id": campaign_id,
                            "cost_today": cost,
                            "limit": self._settings.DAILY_BUDGET_LIMIT,
                            "pct": pct,
                        },
                        ts=_now_msk(),
                    )
                )
            if cost > _SPEND_NO_CLICKS_MIN and clicks == 0:
                signals.append(
                    Signal(
                        type=SignalType.SPEND_NO_CLICKS,
                        severity="warning",
                        data={
                            "campaign_id": campaign_id,
                            "cost_today": cost,
                            "clicks": 0,
                        },
                        ts=_now_msk(),
                    )
                )
        return signals

    # --------------------------------------------------------- _check_bounce

    async def _check_bounce(self) -> list[Signal]:
        today = _now_msk().date().isoformat()
        date_from = (_now_msk() - timedelta(days=7)).date().isoformat()
        data = await self._metrika.get_stats(
            self._http,
            self._settings,
            metrics=["ym:s:visits", "ym:s:bounceRate"],
            dimensions=["ym:s:startURL"],
            filters=(
                "ym:s:lastSignUTMSource=='yandex' "
                "AND ym:s:startURL=~'.*24bankrotsttvo.*/pages/ad/.*'"
            ),
            date1=date_from,
            date2=today,
        )
        signals: list[Signal] = []
        for row in data.get("data") or []:
            dims = row.get("dimensions") or []
            metrics = row.get("metrics") or []
            if not dims or len(metrics) < 2:
                continue
            url = dims[0].get("name") or dims[0].get("id") or ""
            visits = int(metrics[0] or 0)
            bounce = float(metrics[1] or 0)
            if visits < _HIGH_BOUNCE_MIN_VISITS:
                continue
            if bounce >= _HIGH_BOUNCE_CRIT:
                severity: str = "critical"
            elif bounce >= _HIGH_BOUNCE_WARN:
                severity = "warning"
            else:
                continue
            signals.append(
                Signal(
                    type=SignalType.HIGH_BOUNCE,
                    severity=severity,
                    data={"url": url, "visits": visits, "bounce_pct": bounce},
                    ts=_now_msk(),
                )
            )
        return signals

    # ----------------------------------------------------- _check_zero_leads

    async def _check_zero_leads(self) -> list[Signal]:
        today = _now_msk().date().isoformat()
        window_start = (_now_msk() - timedelta(days=_ZERO_LEADS_WINDOW_DAYS)).date().isoformat()

        cost_total = 0.0
        for campaign_id in self._settings.PROTECTED_CAMPAIGN_IDS:
            stats = await self._direct.get_campaign_stats(
                campaign_id, date_from=window_start, date_to=today
            )
            cost_total += float(stats.get("cost", 0))

        leads = await self._bitrix.get_lead_list(
            self._http,
            self._settings,
            filter={
                "UF_CRM_1740791420": 1,  # target lead flag (see funnel_logic_verified)
                ">=DATE_CREATE": window_start,
            },
            max_total=500,
        )
        our_leads = [
            lead
            for lead in leads
            if any(
                marker in (lead.get("SOURCE_DESCRIPTION") or "").lower()
                for marker in _OUR_LEAD_MARKERS
            )
        ]

        signals: list[Signal] = []
        if cost_total > _SPEND_NO_CLICKS_MIN and not our_leads:
            signals.append(
                Signal(
                    type=SignalType.ZERO_LEADS,
                    severity="critical",
                    data={
                        "cost_3d": cost_total,
                        "our_leads_3d": 0,
                        "total_leads_3d": len(leads),
                    },
                    ts=_now_msk(),
                )
            )
        elif our_leads:
            cpa = cost_total / len(our_leads)
            target = self._settings.TARGET_CPA
            if cpa > target * 2:
                signals.append(
                    Signal(
                        type=SignalType.HIGH_CPA,
                        severity="warning",
                        data={
                            "cpa": round(cpa),
                            "target": target,
                            "our_leads_3d": len(our_leads),
                        },
                        ts=_now_msk(),
                    )
                )
        return signals

    # ---------------------------------------------------- _check_landing_health

    async def _check_landing_health(self) -> list[Signal]:
        async def _probe(url: str) -> Signal | None:
            try:
                started = time.monotonic()
                response = await self._http.get(url, timeout=_LANDING_TIMEOUT_SEC)
                elapsed = time.monotonic() - started
                size = len(response.content or b"")
                if response.status_code != 200:
                    return Signal(
                        type=SignalType.LANDING_BROKEN,
                        severity="critical",
                        data={"url": url, "status": response.status_code, "size": size},
                        ts=_now_msk(),
                    )
                if size < _LANDING_MIN_SIZE:
                    return Signal(
                        type=SignalType.LANDING_BROKEN,
                        severity="critical",
                        data={"url": url, "status": 200, "size": size},
                        ts=_now_msk(),
                    )
                if elapsed > _LANDING_SLOW_SEC:
                    return Signal(
                        type=SignalType.LANDING_SLOW,
                        severity="warning",
                        data={"url": url, "elapsed_sec": round(elapsed, 2)},
                        ts=_now_msk(),
                    )
                return None
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                return Signal(
                    type=SignalType.LANDING_BROKEN,
                    severity="critical",
                    data={"url": url, "error": str(exc)[:200]},
                    ts=_now_msk(),
                )

        results = await asyncio.gather(
            *(_probe(url) for url in self._settings.PROTECTED_LANDING_URLS),
            return_exceptions=False,
        )
        return [s for s in results if s is not None]

    # ------------------------------------------------- _check_garbage_queries

    async def _check_garbage_queries(self) -> list[Signal]:
        today = _now_msk().date().isoformat()
        week_ago = (_now_msk() - timedelta(days=7)).date().isoformat()
        report = await self._direct.get_campaign_stats(
            self._settings.PROTECTED_CAMPAIGN_IDS[0],
            date_from=week_ago,
            date_to=today,
        )
        tsv = str(report.get("tsv") or "")
        rows = self._parse_queries_tsv(tsv)
        wasteful: list[dict[str, Any]] = []
        for row in rows:
            cost = float(row.get("cost", 0))
            conversions = int(row.get("conversions", 0))
            clicks = int(row.get("clicks", 0))
            if (
                cost > self._settings.TARGET_CPA
                and conversions == 0
                and clicks >= _GARBAGE_MIN_CLICKS
            ):
                wasteful.append({"query": row.get("query"), "cost": cost, "clicks": clicks})
        if not wasteful:
            return []
        wasteful.sort(key=lambda r: r["cost"], reverse=True)
        top = wasteful[:10]
        total_waste = sum(r["cost"] for r in top)
        severity = "warning" if total_waste < self._settings.TARGET_CPA * 3 else "critical"
        return [
            Signal(
                type=SignalType.GARBAGE_QUERIES,
                severity=severity,
                data={"count": len(top), "top": top, "total_waste": total_waste},
                ts=_now_msk(),
            )
        ]

    @staticmethod
    def _parse_queries_tsv(tsv: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not tsv:
            return rows
        lines = [line for line in tsv.splitlines() if line.strip()]
        if len(lines) < 2:
            return rows
        header = [h.strip().lower() for h in lines[0].split("\t")]
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts, strict=False)))
        return rows

    # -------------------------------------------------- _check_campaign_state

    async def _check_campaign_state(self) -> list[Signal]:
        hour = _now_msk().hour
        if not (_CAMPAIGN_WORK_HOUR_START <= hour < _CAMPAIGN_WORK_HOUR_END):
            return []
        campaigns = await self._direct.get_campaigns(list(self._settings.PROTECTED_CAMPAIGN_IDS))
        signals: list[Signal] = []
        for campaign in campaigns:
            state = campaign.get("State") or ""
            if state in ("SUSPENDED", "OFF"):
                signals.append(
                    Signal(
                        type=SignalType.CAMPAIGN_STOPPED,
                        severity="warning",
                        data={
                            "campaign_id": int(campaign.get("Id", 0)),
                            "state": state,
                            "hour_msk": hour,
                        },
                        ts=_now_msk(),
                    )
                )
        return signals


__all__ = ["SignalDetector"]
