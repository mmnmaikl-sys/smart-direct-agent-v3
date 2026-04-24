"""BFL-RF Tracker — async 3-layer data collector for the ``bfl-rf`` pilot (Task 20).

Async port of ``/tmp/sda-v2/app/agents/bfl_rf_tracker.py``. Returns a dict
with four keys:

* ``direct``  — Impressions / Clicks / Cost / Conversions / CTR for the
  pilot campaign over ``days`` window (from Direct CAMPAIGN_PERFORMANCE_REPORT
  TSV, parsed consistently with ``budget_guard._parse_costs``).
* ``metrika`` — visits / bounce_rate / avg_duration_s for UTM_CAMPAIGN=bfl-rf
  over the same window. Delegates to a ``MetrikaLike`` protocol — the
  concrete :class:`agent_runtime.tools.metrika.MetrikaClient` method does
  not yet exist (Task 8 ships ``get_bounce_by_campaign`` / ``get_conversions``
  but no per-UTM visit roll-up). A structural Protocol keeps this module
  plug-compatible with whatever Task 8's next iteration exposes — see the
  TODO in :func:`collect` for the Settings / client wiring.
* ``bitrix``  — lead count + WON deal count for UTM_CAMPAIGN=bfl-rf in the
  same window, through :func:`bitrix.get_lead_list` /
  :func:`bitrix.get_deal_list` with ``">=DATE_CREATE": iso`` filters.
* ``economics`` — derived ``cpa_lead = cost/leads``, ``cpa_won = cost/won``
  with safe-divide (0 denom → 0 output).

All three layers run in parallel via ``asyncio.gather(return_exceptions=True)``
so a single API failure (Metrika wobbles more than Direct / Bitrix in
practice) does not starve the others. The failing layer becomes
``{"error": "..."}``; the watchdog's :func:`_check_all` is tolerant to
missing metrics (noise-floor thresholds skip rows with 0 values, see
:class:`agent_runtime.jobs.bfl_rf_watchdog`).

No file I/O, no sda_state — this module is stateless; the watchdog owns
the PG state (cooldown + last_run rows).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import httpx

from agent_runtime.config import Settings
from agent_runtime.tools import bitrix as bitrix_tools
from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)

# MSK — matches the Direct "local day" boundary used by budget_guard.
_MSK = timezone(timedelta(hours=3))
_MICRO_TO_RUB = 1_000_000
UTM_CAMPAIGN = "bfl-rf"
# Pilot РСЯ-тест campaign ID from memory (`direct_live_state`, 24.04).
# TODO(task-20-integration): move to ``Settings.BFL_RF_CAMPAIGN_ID`` once
# ``config.py`` is editable in the integration wave — this task is scoped
# strictly to the tracker/watchdog modules.
DEFAULT_BFL_RF_CAMPAIGN_ID = 709307228
_WON_STAGE_ID = "C45:WON"


class MetrikaLike(Protocol):
    """Minimal surface required for the metrika layer.

    Task 8's ``metrika.get_bounce_by_campaign`` returns a bounce map keyed by
    Direct campaign id; we need a per-UTM_CAMPAIGN visits roll-up instead.
    When Task 8 grows a ``get_visit_stats`` method matching this shape, plug
    it in directly; until then the watchdog's DI passes a stub (or ``None``,
    which degrades the metrika layer to ``{"error": "no_client"}``).
    """

    async def get_visit_stats(self, *, utm_campaign: str, days: int) -> dict[str, Any]: ...


# --------------------------------------------------------------- Direct layer


def _parse_direct_report(tsv: str) -> dict[str, float]:
    """Parse CAMPAIGN_PERFORMANCE_REPORT TSV → summed metrics across the window.

    Columns we care about: ``Impressions``, ``Clicks``, ``Cost``, ``Conversions``.
    ``Cost`` ships in micro-rubles (Direct contract); divide once at the
    boundary. CTR is derived (clicks / impressions * 100) because the raw
    report column is per-row, not aggregated.
    """
    totals = {"impressions": 0.0, "clicks": 0.0, "cost": 0.0, "conversions": 0.0}
    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    header_idx = -1
    for idx, line in enumerate(lines):
        cols = line.split("\t")
        if "Impressions" in cols and "Clicks" in cols and "Cost" in cols:
            header_idx = idx
            break
    if header_idx < 0:
        return totals
    header = lines[header_idx].split("\t")
    try:
        i_imp = header.index("Impressions")
        i_clk = header.index("Clicks")
        i_cost = header.index("Cost")
    except ValueError:
        return totals
    i_conv = header.index("Conversions") if "Conversions" in header else -1

    for line in lines[header_idx + 1 :]:
        cols = line.split("\t")
        if not cols or cols[0].startswith("Total"):
            continue
        if len(cols) <= max(i_imp, i_clk, i_cost):
            continue
        try:
            totals["impressions"] += float(cols[i_imp] or 0)
            totals["clicks"] += float(cols[i_clk] or 0)
            totals["cost"] += int(cols[i_cost] or 0) / _MICRO_TO_RUB
        except (ValueError, IndexError):
            continue
        if i_conv >= 0 and i_conv < len(cols):
            try:
                # Direct conversions can arrive as floats (fractional cred.).
                totals["conversions"] += float(cols[i_conv] or 0)
            except ValueError:
                pass
    return totals


async def _direct_layer(
    direct: DirectAPI, campaign_id: int, date_from: str, date_to: str
) -> dict[str, Any]:
    """Fetch Direct metadata + performance report for the pilot window."""
    campaigns = await direct.get_campaigns([campaign_id])
    meta = campaigns[0] if campaigns else {}
    report = await direct.get_campaign_stats(campaign_id, date_from, date_to)
    tsv = str(report.get("tsv", ""))
    totals = _parse_direct_report(tsv)
    impressions = totals["impressions"]
    clicks = totals["clicks"]
    ctr = (clicks / impressions * 100.0) if impressions else 0.0
    cpc = (totals["cost"] / clicks) if clicks else 0.0
    return {
        "campaign_id": campaign_id,
        "name": str(meta.get("Name") or ""),
        "state": str(meta.get("State") or ""),
        "impressions": impressions,
        "clicks": clicks,
        "cost": totals["cost"],
        "conversions": totals["conversions"],
        "ctr": ctr,
        "cpc": cpc,
    }


# -------------------------------------------------------------- Metrika layer


async def _metrika_layer(metrika: MetrikaLike | None, days: int) -> dict[str, Any]:
    """Delegate to ``metrika.get_visit_stats`` if wired; otherwise return an
    empty stub with ``error='no_client'``. The watchdog noise-floor thresholds
    handle the absence gracefully (no false alerts).

    TODO(task-20-integration): when Task 8 ships ``MetrikaClient.get_visit_stats``,
    wire it in the DI layer (app.state.metrika) and drop this protocol.
    """
    if metrika is None:
        return {"error": "no_client"}
    data = await metrika.get_visit_stats(utm_campaign=UTM_CAMPAIGN, days=days)
    # Normalise to our internal shape — the protocol may evolve.
    return {
        "visits": int(data.get("visits", 0) or 0),
        "bounce_rate": float(data.get("bounce", data.get("bounce_rate", 0)) or 0),
        "avg_duration_s": float(data.get("avg_time", data.get("avg_duration_s", 0)) or 0),
        "page_depth": float(data.get("page_depth", 0) or 0),
    }


# --------------------------------------------------------------- Bitrix layer


async def _bitrix_layer(
    http: httpx.AsyncClient, settings: Settings, date_from_iso: str
) -> dict[str, Any]:
    """Lead + WON-deal count for UTM_CAMPAIGN=bfl-rf since ``date_from_iso``.

    Uses the Task 8 ``get_lead_list`` / ``get_deal_list`` helpers (paginated,
    PII-safe logging). Only the count shape is surfaced — PII-bearing fields
    stay inside the clients. The watchdog never logs lead payloads.
    """
    leads = await bitrix_tools.get_lead_list(
        http,
        settings,
        filter={
            ">=DATE_CREATE": date_from_iso,
            "UTM_CAMPAIGN": [UTM_CAMPAIGN],
        },
        select=["ID"],
    )
    won_deals = await bitrix_tools.get_deal_list(
        http,
        settings,
        filter={
            ">=DATE_CREATE": date_from_iso,
            "UTM_CAMPAIGN": [UTM_CAMPAIGN],
            "STAGE_ID": _WON_STAGE_ID,
            "CATEGORY_ID": 45,
        },
        select=["ID", "STAGE_ID", "OPPORTUNITY"],
    )
    revenue_won = 0.0
    for d in won_deals:
        try:
            revenue_won += float(d.get("OPPORTUNITY") or 0)
        except (TypeError, ValueError):
            continue
    return {
        "leads": len(leads),
        "deals": {
            "total": len(won_deals),
            "stages": {"won": len(won_deals)},
            "revenue_won": revenue_won,
        },
    }


# -------------------------------------------------------------------- public


async def collect(
    http_client: httpx.AsyncClient,
    direct: DirectAPI,
    settings: Settings,
    *,
    metrika: MetrikaLike | None = None,
    days: int = 2,
    bfl_rf_campaign_id: int | None = None,
) -> dict[str, Any]:
    """Collect the 3-layer snapshot + derived economics.

    ``days`` is interpreted v2-style: the window is ``[now - (days-1), now]``
    (so ``days=1`` means "today only"), consistent with ``bfl_rf_tracker`` in
    ``/tmp/sda-v2``. Dates are formatted as ``YYYY-MM-DD`` for Direct and ISO
    with МСК offset for Bitrix.

    ``bfl_rf_campaign_id`` defaults to :data:`DEFAULT_BFL_RF_CAMPAIGN_ID`
    until ``Settings.BFL_RF_CAMPAIGN_ID`` lands in Task 20 integration.
    """
    campaign_id = (
        bfl_rf_campaign_id if bfl_rf_campaign_id is not None else DEFAULT_BFL_RF_CAMPAIGN_ID
    )
    now = datetime.now(_MSK)
    date_from = (now - timedelta(days=max(days - 1, 0))).strftime("%Y-%m-%d")
    date_to = now.strftime("%Y-%m-%d")
    iso_from = f"{date_from}T00:00:00+03:00"

    direct_task = _direct_layer(direct, campaign_id, date_from, date_to)
    metrika_task = _metrika_layer(metrika, days)
    bitrix_task = _bitrix_layer(http_client, settings, iso_from)

    results = await asyncio.gather(direct_task, metrika_task, bitrix_task, return_exceptions=True)
    direct_data = _layer_or_error(results[0], "direct")
    metrika_data = _layer_or_error(results[1], "metrika")
    bitrix_data = _layer_or_error(results[2], "bitrix")

    economics = _economics(direct_data, bitrix_data)

    return {
        "date_from": date_from,
        "date_to": date_to,
        "days": days,
        "direct": direct_data,
        "metrika": metrika_data,
        "bitrix": bitrix_data,
        "economics": economics,
    }


def _layer_or_error(result: Any, layer_name: str) -> dict[str, Any]:
    """Translate ``asyncio.gather`` result — either a dict or an exception."""
    if isinstance(result, BaseException):
        logger.warning("bfl_rf_tracker: %s layer failed: %s", layer_name, result)
        return {"error": str(result)}
    if isinstance(result, dict):
        return result
    return {"error": f"unexpected layer shape: {type(result).__name__}"}


def _economics(direct: dict[str, Any], bitrix: dict[str, Any]) -> dict[str, Any]:
    """Derived rubles-per-X, safe-divide to 0 if numerator or denom missing."""
    cost = float(direct.get("cost", 0) or 0)
    leads = int(bitrix.get("leads", 0) or 0)
    won = 0
    deals = bitrix.get("deals") or {}
    stages = deals.get("stages") if isinstance(deals, dict) else None
    if isinstance(stages, dict):
        won = int(stages.get("won", 0) or 0)
    cpa_lead = (cost / leads) if leads else 0.0
    cpa_won = (cost / won) if won else 0.0
    return {
        "cost": cost,
        "leads": leads,
        "won_deals": won,
        "cpa_lead": cpa_lead,
        "cpa_won": cpa_won,
    }


__all__ = [
    "DEFAULT_BFL_RF_CAMPAIGN_ID",
    "MetrikaLike",
    "UTM_CAMPAIGN",
    "collect",
]
