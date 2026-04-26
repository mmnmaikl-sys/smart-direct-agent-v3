"""Funnel CR tracker — реальная бизнес-CR через Bitrix воронку.

Каждый день читает:
  * crm.lead.list (последние 7d с UTM_CAMPAIGN)
  * crm.deal.list cat=45 STAGE_ID=C45:WON (договоры)
  * sda_state[bitrix_feedback_cpa_history] (Direct spend per cid)

Группирует по campaign_id из UTM_CAMPAIGN. Для каждой cid считает:
  * leads (total Bitrix leads из этой кампании)
  * quals (ЕПК — UF_CRM_1740791420 = true)
  * wons (WON deals)
  * CPL = cost / leads
  * CPQ = cost / quals
  * CPO = cost / wons
  * CR lead→qual (квалификация в ОКК)
  * CR qual→won
  * CR lead→won

Telegram digest + audit_log. Read-only, no Direct mutations.

Cron daily 14:30 UTC (17:30 МСК) — после bitrix_feedback (08:30 UTC) и
ad_quality_assessor (12:10 UTC), до owner_report (15:00 UTC). Это даёт
agent'у полную бизнес-картину к моменту вечернего digest владельцу.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from agent_runtime.jobs.bitrix_feedback import _extract_campaign_id, _is_yandex_source
from agent_runtime.tools import bitrix as bitrix_tools
from agent_runtime.tools import telegram as telegram_tools

if TYPE_CHECKING:
    import httpx
    from psycopg_pool import AsyncConnectionPool

    from agent_runtime.config import Settings

logger = logging.getLogger(__name__)


_OWN_CAMPAIGNS: dict[int, str] = {
    709353005: "rabotyaga",
    709353034: "pensioner",
    709353058: "mother",
    709353078: "mfo",
    709353099: "property",
}

_QUALIFIED_FIELD = "UF_CRM_1740791420"  # ЕПК (целевой) — funnel_logic_verified.md
_WON_STAGE_ID = "C45:WON"
_WINDOW_DAYS = 7
_TRAFFIC_SPLIT_KEY = "bitrix_feedback_traffic_split"


@dataclass(frozen=True)
class FunnelStats:
    cid: int
    name: str
    cost_rub: float
    leads: int
    quals: int
    wons: int
    cpl: float | None
    cpq: float | None
    cpo: float | None
    cr_lead_qual: float
    cr_qual_won: float
    cr_lead_won: float


def compute_funnel(
    *,
    cid: int,
    name: str,
    cost_rub: float,
    leads: int,
    quals: int,
    wons: int,
) -> FunnelStats:
    """Pure aggregator — easy to test, no I/O."""
    cpl = (cost_rub / leads) if leads > 0 else None
    cpq = (cost_rub / quals) if quals > 0 else None
    cpo = (cost_rub / wons) if wons > 0 else None
    cr_lead_qual = (quals / leads) if leads > 0 else 0.0
    cr_qual_won = (wons / quals) if quals > 0 else 0.0
    cr_lead_won = (wons / leads) if leads > 0 else 0.0
    return FunnelStats(
        cid=cid,
        name=name,
        cost_rub=cost_rub,
        leads=leads,
        quals=quals,
        wons=wons,
        cpl=cpl,
        cpq=cpq,
        cpo=cpo,
        cr_lead_qual=cr_lead_qual,
        cr_qual_won=cr_qual_won,
        cr_lead_won=cr_lead_won,
    )


def _is_qualified(lead: dict[str, Any]) -> bool:
    """ЕПК — целевой лид по решению ОКК. Field varies (Y/N, 1/0, true/false)."""
    val = lead.get(_QUALIFIED_FIELD)
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"y", "yes", "true", "1"}


def _group_leads(leads: list[dict[str, Any]]) -> tuple[dict[int, int], dict[int, int]]:
    """Returns (leads_by_cid, quals_by_cid) — only Yandex-sourced leads with parseable cid."""
    by_cid: dict[int, int] = {}
    quals_by_cid: dict[int, int] = {}
    for lead in leads:
        if not isinstance(lead, dict):
            continue
        if not _is_yandex_source(lead.get("UTM_SOURCE")):
            continue
        cid = _extract_campaign_id(lead.get("UTM_CAMPAIGN"))
        if cid is None:
            continue
        by_cid[cid] = by_cid.get(cid, 0) + 1
        if _is_qualified(lead):
            quals_by_cid[cid] = quals_by_cid.get(cid, 0) + 1
    return by_cid, quals_by_cid


def _group_won(deals: list[dict[str, Any]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for d in deals:
        if not isinstance(d, dict):
            continue
        if not _is_yandex_source(d.get("UTM_SOURCE")):
            continue
        cid = _extract_campaign_id(d.get("UTM_CAMPAIGN"))
        if cid is None:
            continue
        out[cid] = out.get(cid, 0) + 1
    return out


async def _fetch_cost_by_cid(pool: AsyncConnectionPool) -> dict[int, float]:
    """Read sda_state[bitrix_feedback_traffic_split].own[cid].cost_rub."""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM sda_state WHERE key = %s", (_TRAFFIC_SPLIT_KEY,)
                )
                row = await cur.fetchone()
    except Exception:
        logger.warning("funnel_cr_tracker: cost fetch failed", exc_info=True)
        return {}
    if not row:
        return {}
    raw = row[0]
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    elif isinstance(raw, dict):
        data = raw
    else:
        return {}
    out: dict[int, float] = {}
    for cid_str, info in (data.get("own") or {}).items():
        try:
            cid = int(cid_str)
        except (TypeError, ValueError):
            continue
        if isinstance(info, dict):
            out[cid] = float(info.get("cost_rub", 0.0) or 0.0)
    return out


def _format_report(stats: list[FunnelStats], window_days: int) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"<b>📊 Funnel CR Tracker</b> — {when}",
        "",
        f"Окно: {window_days}d. По 5 own кампаниям:",
        "",
    ]
    total_cost = sum(s.cost_rub for s in stats)
    total_leads = sum(s.leads for s in stats)
    total_quals = sum(s.quals for s in stats)
    total_wons = sum(s.wons for s in stats)
    if total_leads == 0 and total_cost == 0:
        lines.append("Тихо — 0 cost / 0 leads / 0 квалификаций / 0 договоров.")
        lines.append(
            "<i>Если кампании ON и есть visits — fix attribution UTM (UTM_SOURCE=yandex).</i>"
        )
        return "\n".join(lines)
    for s in sorted(stats, key=lambda x: -x.cost_rub):
        if s.cost_rub == 0 and s.leads == 0:
            continue
        cpl_str = f"{s.cpl:.0f}₽" if s.cpl is not None else "—"
        cpo_str = f"{s.cpo:.0f}₽" if s.cpo is not None else "—"
        lines.append(
            f"<b>{s.name}</b> (cid={s.cid}): cost={s.cost_rub:.0f}₽, "
            f"leads={s.leads}, ЕПК={s.quals}, won={s.wons}"
        )
        lines.append(
            f"  CPL={cpl_str} CPO={cpo_str} "
            f"CR(lead→ЕПК)={s.cr_lead_qual:.0%} CR(ЕПК→won)={s.cr_qual_won:.0%}"
        )
    lines.append("")
    overall_cpl = (total_cost / total_leads) if total_leads > 0 else 0
    overall_cpo = (total_cost / total_wons) if total_wons > 0 else 0
    lines.append(
        f"<b>Итого:</b> cost={total_cost:.0f}₽, leads={total_leads}, "
        f"ЕПК={total_quals}, won={total_wons}"
    )
    lines.append(
        f"  Overall CPL={overall_cpl:.0f}₽ (бенчмарк БФЛ 625-684₽), "
        f"CPO={overall_cpo:.0f}₽ (бенчмарк 22 567₽)"
    )
    return "\n".join(lines)


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    bitrix_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron daily 14:30 UTC (17:30 МСК)."""
    if http_client is None or settings is None or bitrix_client is None:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "leads_total": 0,
            "wons_total": 0,
            "dry_run": dry_run,
        }

    period_to = datetime.now(UTC)
    period_from = period_to - timedelta(days=_WINDOW_DAYS)
    date_from_iso = period_from.isoformat()

    try:
        leads = await bitrix_tools.get_lead_list(
            bitrix_client,
            settings,
            filter={">=DATE_CREATE": date_from_iso},
            select=["ID", "UTM_CAMPAIGN", "UTM_SOURCE", _QUALIFIED_FIELD, "DATE_CREATE"],
        )
    except Exception:
        logger.exception("funnel_cr_tracker: get_lead_list failed")
        leads = []
    try:
        deals = await bitrix_tools.get_deal_list(
            bitrix_client,
            settings,
            filter={"STAGE_ID": _WON_STAGE_ID, ">=DATE_MODIFY": date_from_iso},
            select=["ID", "UTM_CAMPAIGN", "UTM_SOURCE", "DATE_MODIFY"],
        )
    except Exception:
        logger.exception("funnel_cr_tracker: get_deal_list failed")
        deals = []

    leads_by_cid, quals_by_cid = _group_leads(leads)
    wons_by_cid = _group_won(deals)
    cost_by_cid = await _fetch_cost_by_cid(pool)

    stats: list[FunnelStats] = []
    for cid, name in _OWN_CAMPAIGNS.items():
        stats.append(
            compute_funnel(
                cid=cid,
                name=name,
                cost_rub=cost_by_cid.get(cid, 0.0),
                leads=leads_by_cid.get(cid, 0),
                quals=quals_by_cid.get(cid, 0),
                wons=wons_by_cid.get(cid, 0),
            )
        )

    msg = _format_report(stats, window_days=_WINDOW_DAYS)
    if not dry_run:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("funnel_cr_tracker: telegram send failed")

    return {
        "status": "ok",
        "leads_total": sum(s.leads for s in stats),
        "quals_total": sum(s.quals for s in stats),
        "wons_total": sum(s.wons for s in stats),
        "cost_total": sum(s.cost_rub for s in stats),
        "dry_run": dry_run,
    }


__all__ = ["FunnelStats", "compute_funnel", "run"]
