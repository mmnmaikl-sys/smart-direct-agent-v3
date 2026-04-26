"""Lead Quality Uploader — daily Bitrix deals (cat=49 КЦ) → Metrika target events.

Runs daily at 07:30 UTC (10:30 МСК), 30 minutes after ``offline_conversions``.

Whereas ``offline_conversions`` ships only positive contract events
(C45:5/6/WON), this job ships *quality signals* per Bitrix deal in
category 49 (КЦ — call center qualification stage):

  * ЕПК (UF_CRM_1740791420=true OR moved to C45:* / C49:WON)
    → Metrika target ``bfl_lead_qualified``
  * Junk (C49:5 Брак, C49:7 Дубль, C49:LOSE Не можем помочь,
    C49:4 Отказался / Потерял интерес)
    → Metrika target ``bfl_lead_junk``
  * Other (NEW, in progress) — skipped (will land later as it moves)

Why this matters: Direct's ``WB_MAXIMUM_CONVERSIONS`` strategy learns
from Metrika targets. If only ``lead_form_submit`` is exposed as a
target, Direct optimises for *quantity* of leads regardless of quality.
Sending qualified+junk as separate targets lets the operator (or a
PriorityGoals call) push Direct to optimise for ЕПК specifically.

PII barrier: the deal payload contains PHONE / NAME / SOURCE_DESCRIPTION.
This module never logs raw responses — only stripped rows
(yclid/client_id, target, datetime, external_id) cross the boundary.

Idempotency: ``external_id = bitrix_deal_{ID}_quality_{label}`` — Metrika
dedupes on it so re-runs and stage transitions don't double-count.

Degraded-noop: when DI clients are missing the job returns
``status='ok', action='degraded_noop'`` like the rest of the registry.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import httpx
    from psycopg_pool import AsyncConnectionPool

    from agent_runtime.config import Settings
    from agent_runtime.jobs.offline_conversions import MetrikaOfflineClient

from agent_runtime.tools import bitrix as bitrix_tools

logger = logging.getLogger(__name__)


_MSK = timezone(timedelta(hours=3))
_WINDOW_HOURS = 24
_QUALIFIED_FIELD = "UF_CRM_1740791420"  # ЕПК — funnel_logic_verified.md

# C49 = call-center qualification stage IDs that mark a lead as junk.
# C49:WON / C45:* moves are positive — handled as qualified below.
_JUNK_STAGE_IDS: frozenset[str] = frozenset(
    {
        "C49:5",  # Брак
        "C49:7",  # Дубль
        "C49:LOSE",  # Не можем помочь
        "C49:4",  # Отказался / Потерял интерес
    }
)


def _is_truthy(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"y", "yes", "true", "1"}


@dataclass(frozen=True)
class QualityRow:
    """One quality-signal row ready for Metrika offline upload."""

    external_id: str
    identifier_value: str
    identifier_type: Literal["YCLID", "CLIENT_ID"]
    target: str
    datetime_s: str

    def as_upload_dict(self) -> dict[str, Any]:
        """Metrika offline JSON shape — same contract as offline_conversions."""
        key = "yclid" if self.identifier_type == "YCLID" else "client_id"
        return {
            key: self.identifier_value,
            "target": self.target,
            "datetime": self.datetime_s,
            "external_id": self.external_id,
        }


def _resolve_identifier(
    deal: dict[str, Any],
) -> tuple[str | None, Literal["YCLID", "CLIENT_ID"] | None]:
    """Same priority as offline_conversions: yclid > client_id > skip."""
    yclid = deal.get("UF_CRM_YCLID")
    if isinstance(yclid, str) and yclid.strip():
        return yclid.strip(), "YCLID"
    client_id = deal.get("UF_CRM_CLIENT_ID")
    if isinstance(client_id, str) and client_id.strip():
        return client_id.strip(), "CLIENT_ID"
    return None, None


def classify_deal(
    deal: dict[str, Any],
    *,
    target_qualified: str = "bfl_lead_qualified",
    target_junk: str = "bfl_lead_junk",
) -> QualityRow | None:
    """Pure: deal → QualityRow or None if neither qualified nor junk.

    Qualified takes priority over junk: a deal moved to C45:* (in OP)
    or with UF_CRM_1740791420=true is qualified even if it later marks
    junk — the qualification already happened. C49:WON also counts as
    qualified (КЦ booked a meeting).

    Returns None for:
      * deals still in NEW / in-progress stages (not yet decided)
      * deals without yclid/client_id
      * deals without ID
    """
    stage_id = str(deal.get("STAGE_ID") or "")
    qualified_signal = (
        _is_truthy(deal.get(_QUALIFIED_FIELD))
        or stage_id == "C49:WON"
        or stage_id.startswith("C45:")
    )
    junk_signal = stage_id in _JUNK_STAGE_IDS

    if qualified_signal:
        target = target_qualified
        label = "qualified"
    elif junk_signal:
        target = target_junk
        label = "junk"
    else:
        return None  # still in flight — try again tomorrow

    identifier, identifier_type = _resolve_identifier(deal)
    if identifier is None or identifier_type is None:
        return None
    deal_id = str(deal.get("ID") or "").strip()
    if not deal_id:
        return None

    datetime_s = str(
        deal.get("DATE_MODIFY") or deal.get("DATE_CREATE") or datetime.now(_MSK).isoformat()
    )
    return QualityRow(
        external_id=f"bitrix_deal_{deal_id}_quality_{label}",
        identifier_value=identifier,
        identifier_type=identifier_type,
        target=target,
        datetime_s=datetime_s,
    )


def _split_by_identifier(rows: list[QualityRow]) -> tuple[list[QualityRow], list[QualityRow]]:
    yclid: list[QualityRow] = []
    client_id: list[QualityRow] = []
    for r in rows:
        if r.identifier_type == "YCLID":
            yclid.append(r)
        else:
            client_id.append(r)
    return yclid, client_id


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    bitrix_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    metrika_client: MetrikaOfflineClient | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron daily 07:30 UTC."""
    if bitrix_client is None or settings is None:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "qualified_total": 0,
            "junk_total": 0,
            "skipped_no_identifier": 0,
            "dry_run": dry_run,
        }

    period_to = datetime.now(_MSK)
    period_from = period_to - timedelta(hours=_WINDOW_HOURS)

    # crm.deal.list cat=49 (КЦ) modified within the last 24h. We pull
    # *all* stage transitions in the window — classify_deal decides what
    # is qualified/junk/skip.
    try:
        deals = await bitrix_tools.get_deal_list(
            bitrix_client,
            settings,
            filter={
                "CATEGORY_ID": 49,
                ">=DATE_MODIFY": period_from.isoformat(),
            },
            select=[
                "ID",
                "STAGE_ID",
                "UF_CRM_YCLID",
                "UF_CRM_CLIENT_ID",
                _QUALIFIED_FIELD,
                "DATE_MODIFY",
                "DATE_CREATE",
            ],
        )
    except Exception:
        logger.exception("lead_quality_uploader: get_deal_list failed")
        return {
            "status": "error",
            "action": "fetch_failed",
            "qualified_total": 0,
            "junk_total": 0,
            "skipped_no_identifier": 0,
            "dry_run": dry_run,
        }

    qualified_target = os.environ.get("METRIKA_GOAL_QUALIFIED", "bfl_lead_qualified")
    junk_target = os.environ.get("METRIKA_GOAL_JUNK", "bfl_lead_junk")

    rows: list[QualityRow] = []
    skipped_no_identifier = 0
    skipped_in_flight = 0
    for d in deals or []:
        if not isinstance(d, dict):
            continue
        row = classify_deal(d, target_qualified=qualified_target, target_junk=junk_target)
        if row is None:
            stage_id = str(d.get("STAGE_ID") or "")
            qualified_signal = (
                _is_truthy(d.get(_QUALIFIED_FIELD))
                or stage_id == "C49:WON"
                or stage_id.startswith("C45:")
            )
            if qualified_signal or stage_id in _JUNK_STAGE_IDS:
                skipped_no_identifier += 1
            else:
                skipped_in_flight += 1
            continue
        rows.append(row)

    qualified_rows = [r for r in rows if r.target == qualified_target]
    junk_rows = [r for r in rows if r.target == junk_target]

    if dry_run or metrika_client is None:
        logger.info(
            "lead_quality_uploader: dry=%s metrika=%s q=%d j=%d flight=%d no_id=%d",
            dry_run,
            metrika_client is not None,
            len(qualified_rows),
            len(junk_rows),
            skipped_in_flight,
            skipped_no_identifier,
        )
        return {
            "status": "ok",
            "action": "dry_run" if dry_run else "no_metrika_client",
            "qualified_total": len(qualified_rows),
            "junk_total": len(junk_rows),
            "skipped_in_flight": skipped_in_flight,
            "skipped_no_identifier": skipped_no_identifier,
            "dry_run": dry_run,
        }

    counter_id = int(os.environ.get("METRIKA_COUNTER_ID", "108750395"))
    yclid_rows, client_id_rows = _split_by_identifier(rows)

    uploaded = 0
    upload_errors: list[str] = []
    for kind, group in (("YCLID", yclid_rows), ("CLIENT_ID", client_id_rows)):
        if not group:
            continue
        try:
            await metrika_client.upload_offline_conversions(
                counter_id=counter_id,
                rows=[r.as_upload_dict() for r in group],
                client_id_type=kind,  # type: ignore[arg-type]
            )
            uploaded += len(group)
        except Exception as exc:
            logger.exception("lead_quality_uploader: %s upload failed", kind)
            upload_errors.append(f"{kind}: {type(exc).__name__}: {exc}")

    return {
        "status": "ok" if not upload_errors else "partial",
        "action": "uploaded",
        "qualified_total": len(qualified_rows),
        "junk_total": len(junk_rows),
        "uploaded": uploaded,
        "skipped_in_flight": skipped_in_flight,
        "skipped_no_identifier": skipped_no_identifier,
        "errors": upload_errors,
        "dry_run": dry_run,
    }


__all__ = [
    "QualityRow",
    "_JUNK_STAGE_IDS",
    "classify_deal",
    "run",
]
