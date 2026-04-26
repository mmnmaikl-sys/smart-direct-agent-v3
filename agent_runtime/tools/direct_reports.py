"""Direct API /reports adapter — provides search-query report for query_analyzer.

The /reports endpoint differs from the rest of Direct API v5:
  * returns plain TSV (not JSON)
  * supports synchronous mode via ``processingMode: online`` header
  * has its own retry semantics (201/202 = report still building)

We expose a single method matching the ``_QueryReportFetcher`` Protocol
that ``query_analyzer.run`` injects:

    async def get_search_query_performance_report(
        self, date_from: str, date_to: str
    ) -> list[dict[str, Any]]

Each row in the returned list has lowercase string keys
(``campaignid``, ``query``, ``clicks``, ``cost``, ...) so it slots into
the existing ``query_analyzer`` consumer (``row.get("query")``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from agent_runtime.config import Settings

logger = logging.getLogger(__name__)


_REPORTS_URL = "https://api.direct.yandex.com/json/v5/reports"


def _parse_tsv_rows(tsv: str) -> list[dict[str, str]]:
    """Convert Direct /reports TSV body to list of dicts with lowercase keys.

    Skips metadata lines (Report name, Date range, etc.), the trailing
    "Total rows: N" footer, and any line that isn't tab-separated data.
    """
    if not tsv or not tsv.strip():
        return []
    raw_lines = [ln.rstrip("\r") for ln in tsv.splitlines() if ln.strip()]

    header_idx = -1
    for idx, line in enumerate(raw_lines):
        if "\t" in line and "Query" in line and "CampaignId" in line:
            header_idx = idx
            break
        # Some reports lead with just "Query<tab>...".
        if "\t" in line and "Query" in line and "Impressions" in line:
            header_idx = idx
            break
    if header_idx < 0:
        return []

    cols = [c.strip().lower() for c in raw_lines[header_idx].split("\t")]
    rows: list[dict[str, str]] = []
    for line in raw_lines[header_idx + 1 :]:
        if not line or "\t" not in line:
            continue
        if line.startswith("Total rows") or line.startswith('"Total rows'):
            continue
        parts = line.split("\t")
        row = {cols[i]: parts[i].strip() for i in range(min(len(cols), len(parts)))}
        rows.append(row)
    return rows


def _resolve_secret(token_obj: Any) -> str:
    if token_obj is None:
        return ""
    if hasattr(token_obj, "get_secret_value"):
        return token_obj.get_secret_value()
    return str(token_obj)


class DirectReportAdapter:
    """Implements the ``_QueryReportFetcher`` Protocol used by query_analyzer.

    Holds a reference to a shared ``httpx.AsyncClient`` and ``Settings`` so
    each cron invocation reuses the connection pool and avoids re-reading
    env vars.
    """

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http_client
        self._settings = settings

    async def get_search_query_performance_report(
        self,
        date_from: str,
        date_to: str,
    ) -> list[dict[str, Any]]:
        token = _resolve_secret(getattr(self._settings, "YANDEX_DIRECT_TOKEN", None))
        if not token:
            logger.warning("direct_reports: YANDEX_DIRECT_TOKEN missing — empty report")
            return []

        body = {
            "params": {
                "SelectionCriteria": {"DateFrom": date_from, "DateTo": date_to},
                "FieldNames": [
                    "CampaignId",
                    "AdGroupId",
                    "Query",
                    "Impressions",
                    "Clicks",
                    "Cost",
                ],
                "ReportName": f"sqr_{date_from}_{date_to}",
                "ReportType": "SEARCH_QUERY_PERFORMANCE_REPORT",
                "DateRangeType": "CUSTOM_DATE",
                "Format": "TSV",
                "IncludeVAT": "NO",
                "IncludeDiscount": "NO",
            }
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept-Language": "ru",
            "Content-Type": "application/json; charset=utf-8",
            # Synchronous mode — single request returns the report or
            # 4xx/5xx; no polling needed for our 5-campaign volume.
            "processingMode": "online",
            "returnMoneyInMicros": "false",
            "skipReportHeader": "true",
            "skipColumnHeader": "false",
            "skipReportSummary": "true",
        }

        try:
            resp = await self._http.post(_REPORTS_URL, json=body, headers=headers, timeout=120)
        except Exception:
            logger.exception("direct_reports: request failed")
            return []

        if resp.status_code in (201, 202):
            # Report queued — pretend empty for now, next cron retries.
            logger.info(
                "direct_reports: report queued (HTTP %d) — will retry next cycle",
                resp.status_code,
            )
            return []

        if resp.status_code != 200:
            logger.warning(
                "direct_reports: HTTP %d body=%s",
                resp.status_code,
                resp.text[:200],
            )
            return []

        return _parse_tsv_rows(resp.text)


__all__ = ["DirectReportAdapter", "_parse_tsv_rows"]
