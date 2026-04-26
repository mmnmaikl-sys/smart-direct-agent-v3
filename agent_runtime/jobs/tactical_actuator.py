"""Tactical actuator — pure-rule autonomous DailyBudget adjuster.

The FIRST autonomous actuator in SDA v3. Replaces a manual ad-hoc
"scale winners, starve losers" loop the owner was running by hand.

Runs every 4h (cron `0 */4 * * *` UTC). For each of the 5 PROTECTED
own campaigns (709353xxx) it:

  1. Reads lifetime ``Statistics.Impressions`` / ``Clicks`` via
     ``DirectAPI.get_campaigns`` → derives CTR.
  2. Applies one of three pure rules:
       * **scale**  — CTR > 5% AND DailyBudget < 2000 → +500₽ (capped 2500)
       * **starve** — CTR < 2% AND DailyBudget > 200 → -200₽ (floor 100)
       * **noop**   — otherwise (insufficient data, mid-CTR, at limit)
  3. Validates total daily cap stays ≤ 5000₽ (refuses plan if exceeded).
  4. Applies via ``campaigns.update`` + GET-after-SET verify.
  5. Sends a Telegram alert with the decisions & verification.
  6. Writes one ``audit_log`` row (sanitised; only numeric aggregates).

Pure rules — no LLM, no kill-switches needed. The action surface is so
narrow (only DailyBudget, only own cids, only ±500/200₽ steps within
strict bounds) that a wrong decision costs at most a few hundred rubles
per cycle and is reversed by the next iteration.

dry_run=True returns the planned decisions without mutating Direct or
sending Telegram — used in smoke tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)


_OWN_CAMPAIGNS: dict[int, str] = {
    709353005: "rabotyaga",
    709353034: "pensioner",
    709353058: "mother",
    709353078: "mfo",
    709353099: "property",
}

# Bounds in rubles.
_DAILY_MIN_RUB = 100
_DAILY_MAX_RUB = 2500
_TOTAL_CAP_RUB = 5000

# Decision thresholds.
_CTR_SCALE_PCT = 5.0  # > 5% → scale up
_CTR_STARVE_PCT = 2.0  # < 2% → starve down
_MIN_IMPRESSIONS_FOR_SIGNAL = 100  # lifetime — below this CTR is not a signal

# Step sizes per cycle (4h).
_STEP_SCALE_RUB = 500
_STEP_STARVE_RUB = 200

# Direct API uses micro-rubles (1₽ = 1_000_000 micro).
_MICRO = 1_000_000


@dataclass(frozen=True)
class Decision:
    cid: int
    name: str
    kind: Literal["scale", "starve", "noop", "early_kill"]
    current_daily_rub: int
    new_daily_rub: int
    reason: str


@dataclass(frozen=True)
class CampaignSignals:
    """Multi-signal snapshot for v2 decisions (last 7d aggregates).

    Sourced from:
      * daily_rub  → DirectAPI.get_campaigns DailyBudget
      * clicks_7d  → DirectAPI.get_campaign_stats(date-7d, today) Clicks
      * cost_7d    → DirectAPI.get_campaign_stats(...) Cost
      * leads_7d   → sda_state[bitrix_feedback_traffic_split].own[cid].leads
                     (real Bitrix crm.lead.list count, 26.04.2026+;
                     falls back to .won for stale snapshots — see
                     _fetch_leads_per_cid)
      * bounce_pct → Metrika ym:s:bounceRate via lastDirectClickOrder

    bounce_pct=None means Metrika returned no data for this cid (cold
    campaign or attribution mismatch); decision rules requiring bounce
    return noop in that case.
    """

    cid: int
    name: str
    daily_rub: int
    clicks_7d: int
    cost_7d: float
    leads_7d: int
    bounce_pct: float | None
    age_days: int = 999  # campaign age since StartDate; default = "old enough"


# v2 thresholds — derived from Deep Research 2026-04 (БФЛ benchmarks).
_V2_SAMPLE_MIN_CLICKS = 50
_V2_EARLY_KILL_CLICKS = 100
_V2_EARLY_KILL_BOUNCE_PCT = 60.0
_V2_STARVE_CPL_RUB = 1500.0
_V2_STARVE_BOUNCE_PCT = 55.0
_V2_SCALE_CPL_RUB = 800.0  # vc.ru benchmark = 625-684, headroom to 800
_V2_SCALE_BOUNCE_PCT = 40.0
_V2_SCALE_FACTOR = 1.30
_V2_STARVE_FACTOR = 0.50

# LEARNING_GUARD — Yandex Direct auto-strategy needs ~14 days + 10 conv/week
# to converge. Touching DailyBudget mid-learning resets the bid model. So
# during the first 14 days we only allow EARLY_KILL (true junk-traffic
# protection) and otherwise NOOP — even if CPL/bounce look bad on day 5.
# eLama 2026: "5 причин не отключать кампании в первые 2 недели".
_LEARNING_GUARD_DAYS = 14


def _ctr_pct(impressions: int, clicks: int) -> float:
    if impressions <= 0:
        return 0.0
    return 100.0 * clicks / impressions


def _budget_rub(camp: dict[str, Any]) -> int:
    db = camp.get("DailyBudget") or {}
    amount = int(db.get("Amount", 0) or 0)
    return amount // _MICRO


def decide_action(camp: dict[str, Any]) -> Decision:
    """Pure decision rule — testable, deterministic."""
    cid = int(camp.get("Id", 0))
    name = str(camp.get("Name", "")).replace("[24bfl] ", "").replace(" test", "").strip() or "?"
    stats = camp.get("Statistics") or {}
    impr = int(stats.get("Impressions", 0) or 0)
    clicks = int(stats.get("Clicks", 0) or 0)
    current = _budget_rub(camp)

    if impr < _MIN_IMPRESSIONS_FOR_SIGNAL:
        return Decision(
            cid=cid,
            name=name,
            kind="noop",
            current_daily_rub=current,
            new_daily_rub=current,
            reason=f"insufficient impressions ({impr} < {_MIN_IMPRESSIONS_FOR_SIGNAL}) — no data",
        )

    ctr = _ctr_pct(impr, clicks)

    if ctr >= _CTR_SCALE_PCT and current < _DAILY_MAX_RUB:
        new = min(current + _STEP_SCALE_RUB, _DAILY_MAX_RUB)
        return Decision(
            cid=cid,
            name=name,
            kind="scale",
            current_daily_rub=current,
            new_daily_rub=new,
            reason=f"CTR={ctr:.1f}% >= {_CTR_SCALE_PCT}%, scale +{new - current}₽",
        )

    if ctr < _CTR_STARVE_PCT and current > _DAILY_MIN_RUB:
        new = max(current - _STEP_STARVE_RUB, _DAILY_MIN_RUB)
        return Decision(
            cid=cid,
            name=name,
            kind="starve",
            current_daily_rub=current,
            new_daily_rub=new,
            reason=f"CTR={ctr:.1f}% < {_CTR_STARVE_PCT}%, starve -{current - new}₽",
        )

    return Decision(
        cid=cid,
        name=name,
        kind="noop",
        current_daily_rub=current,
        new_daily_rub=current,
        reason=f"CTR={ctr:.1f}% in control band [2%, 5%) or already at limit",
    )


def decide_v2(s: CampaignSignals) -> Decision:
    """Multi-signal composite-rule decision (replaces v1 single-CTR).

    Priority order (first match wins):
      1. EARLY_KILL_SWITCH — junk traffic burning budget, drop to floor
      2. STARVE_RED — CPL too high + quality bad
      3. SCALE_GREEN — CPL low + quality good + sample sufficient
      4. NOOP — control band, insufficient sample, or missing Metrika data

    The early-kill is allowed at clicks≥100 even before the regular sample
    floor — that is the "save my budget" path from the owner's 03.04
    incident (10K₽ slittus on autotargeting).
    """
    cpl: float | None = (s.cost_7d / s.leads_7d) if s.leads_7d > 0 else None

    # 1. EARLY_KILL_SWITCH — owner's 03.04 pattern. Allowed even during
    # learning window: 100+ clicks with 0 Bitrix leads + bounce>60% is
    # unambiguous junk traffic — saving the budget outweighs the risk of
    # disturbing strategy learning.
    if (
        s.clicks_7d >= _V2_EARLY_KILL_CLICKS
        and s.leads_7d == 0
        and s.bounce_pct is not None
        and s.bounce_pct > _V2_EARLY_KILL_BOUNCE_PCT
    ):
        return Decision(
            cid=s.cid,
            name=s.name,
            kind="early_kill",
            current_daily_rub=s.daily_rub,
            new_daily_rub=_DAILY_MIN_RUB,
            reason=(
                f"EARLY_KILL: {s.clicks_7d} clicks, 0 leads, "
                f"bounce={s.bounce_pct:.0f}% — мусорный трафик, drop to floor"
            ),
        )

    # 2. LEARNING_GUARD — within 14d of campaign start, do NOT scale or
    # starve. Yandex Direct auto-strategy needs the full 14d window with
    # a stable budget to converge; mid-learning DailyBudget changes reset
    # the bid model and make CPA worse, not better.
    if s.age_days < _LEARNING_GUARD_DAYS:
        return Decision(
            cid=s.cid,
            name=s.name,
            kind="noop",
            current_daily_rub=s.daily_rub,
            new_daily_rub=s.daily_rub,
            reason=(
                f"LEARNING_GUARD: campaign age {s.age_days}d < "
                f"{_LEARNING_GUARD_DAYS}d — не трогаем обучение стратегии"
            ),
        )

    # Sample-size guard for non-kill rules.
    if s.clicks_7d < _V2_SAMPLE_MIN_CLICKS:
        return Decision(
            cid=s.cid,
            name=s.name,
            kind="noop",
            current_daily_rub=s.daily_rub,
            new_daily_rub=s.daily_rub,
            reason=(
                f"insufficient sample ({s.clicks_7d} < {_V2_SAMPLE_MIN_CLICKS} clicks/7d) — wait"
            ),
        )

    # bounce data required for STARVE/SCALE.
    if s.bounce_pct is None:
        return Decision(
            cid=s.cid,
            name=s.name,
            kind="noop",
            current_daily_rub=s.daily_rub,
            new_daily_rub=s.daily_rub,
            reason="no bounce data from Metrika — cannot apply quality-based rules",
        )

    # 2. STARVE_RED — CPL high AND quality bad
    if cpl is not None and cpl > _V2_STARVE_CPL_RUB and s.bounce_pct > _V2_STARVE_BOUNCE_PCT:
        new = max(int(s.daily_rub * _V2_STARVE_FACTOR), _DAILY_MIN_RUB)
        return Decision(
            cid=s.cid,
            name=s.name,
            kind="starve",
            current_daily_rub=s.daily_rub,
            new_daily_rub=new,
            reason=(
                f"STARVE: CPL={cpl:.0f}₽ > {_V2_STARVE_CPL_RUB:.0f}₽, "
                f"bounce={s.bounce_pct:.0f}% > {_V2_STARVE_BOUNCE_PCT:.0f}% — -50%"
            ),
        )

    # 3. SCALE_GREEN — CPL low AND quality good AND sample sufficient
    if (
        cpl is not None
        and cpl < _V2_SCALE_CPL_RUB
        and s.bounce_pct < _V2_SCALE_BOUNCE_PCT
        and s.daily_rub < _DAILY_MAX_RUB
    ):
        new = min(int(s.daily_rub * _V2_SCALE_FACTOR), _DAILY_MAX_RUB)
        return Decision(
            cid=s.cid,
            name=s.name,
            kind="scale",
            current_daily_rub=s.daily_rub,
            new_daily_rub=new,
            reason=(
                f"SCALE: CPL={cpl:.0f}₽ < {_V2_SCALE_CPL_RUB:.0f}₽, "
                f"bounce={s.bounce_pct:.0f}% < {_V2_SCALE_BOUNCE_PCT:.0f}% — +30%"
            ),
        )

    # 4. NOOP — middle ground
    cpl_str = f"CPL={cpl:.0f}₽" if cpl is not None else "no leads"
    return Decision(
        cid=s.cid,
        name=s.name,
        kind="noop",
        current_daily_rub=s.daily_rub,
        new_daily_rub=s.daily_rub,
        reason=f"control band ({cpl_str}, bounce={s.bounce_pct:.0f}%)",
    )


def plan_total_within_cap(
    decisions: list[Decision],
    others_current_total: int,
    *,
    cap_rub: int = _TOTAL_CAP_RUB,
) -> tuple[bool, int]:
    """Return (is_within_cap, total_after_apply)."""
    decisions_total = sum(d.new_daily_rub for d in decisions)
    total = decisions_total + others_current_total
    return (total <= cap_rub, total)


def _format_alert(
    decisions: list[Decision],
    applied: list[Decision],
    dry_run: bool,
    cap_check: tuple[bool, int],
) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    head = "<b>🤖 Tactical Actuator</b> — " + when
    if dry_run:
        head += " <i>(dry_run)</i>"

    lines = [head, ""]
    actionable = [d for d in decisions if d.kind != "noop"]
    if not actionable:
        lines.append("Все 5 кампаний в control band — никаких действий.")
        for d in decisions:
            lines.append(f"  • {d.name} (cid={d.cid}): {d.reason}")
        return "\n".join(lines)

    lines.append(f"Решений к применению: {len(actionable)}")
    for d in actionable:
        emoji = "🔼" if d.kind == "scale" else "🔽"
        lines.append(
            f"{emoji} <b>{d.name}</b> (cid={d.cid}): "
            f"{d.current_daily_rub}₽ → {d.new_daily_rub}₽ ({d.reason})"
        )

    ok, total = cap_check
    lines.append("")
    if not ok:
        lines.append(
            f"❌ Total daily cap exceeded: {total}₽ &gt; {_TOTAL_CAP_RUB}₽ — abort, no apply"
        )
    elif dry_run:
        lines.append(
            f"Total daily after apply: {total}₽ (cap {_TOTAL_CAP_RUB}₽). Dry-run, no mutation."
        )
    else:
        lines.append(
            f"✅ Applied. Total daily: {total}₽ (cap {_TOTAL_CAP_RUB}₽)."
            f" Verified via GET-after-SET ({len(applied)}/{len(actionable)})."
        )

    return "\n".join(lines)


async def _apply_via_direct(
    direct: DirectAPI,
    decisions: list[Decision],
) -> list[Decision]:
    """Build a single campaigns.update call + GET-after-SET verify.

    Returns the subset of decisions that verified successfully on the
    GET response. Failures are logged but do not raise — partial
    progress is acceptable.
    """
    if not decisions:
        return []

    actionable = [d for d in decisions if d.kind != "noop"]
    if not actionable:
        return []

    update_body = {
        "method": "update",
        "params": {
            "Campaigns": [
                {
                    "Id": d.cid,
                    "DailyBudget": {
                        "Amount": d.new_daily_rub * _MICRO,
                        "Mode": "STANDARD",
                    },
                }
                for d in actionable
            ]
        },
    }

    try:
        await direct._call("campaigns", "update", update_body["params"])  # noqa: SLF001
    except Exception:
        logger.exception("tactical_actuator: campaigns.update failed — no apply")
        return []

    # GET-after-SET verify.
    try:
        verify = await direct._call(  # noqa: SLF001
            "campaigns",
            "get",
            {
                "SelectionCriteria": {"Ids": [d.cid for d in actionable]},
                "FieldNames": ["Id", "DailyBudget"],
            },
        )
    except Exception:
        logger.exception("tactical_actuator: GET-after-SET failed")
        return []

    verified_amounts: dict[int, int] = {}
    for c in verify.get("Campaigns") or []:
        cid = int(c.get("Id", 0))
        amount_micro = int((c.get("DailyBudget") or {}).get("Amount", 0) or 0)
        verified_amounts[cid] = amount_micro // _MICRO

    applied: list[Decision] = []
    for d in actionable:
        actual = verified_amounts.get(d.cid)
        if actual == d.new_daily_rub:
            applied.append(d)
        else:
            logger.warning(
                "tactical_actuator: verify mismatch cid=%d expected=%d actual=%s",
                d.cid,
                d.new_daily_rub,
                actual,
            )
    return applied


async def _collect_signals(
    direct: DirectAPI,
    http_client: httpx.AsyncClient,
    settings: Settings,
    pool: AsyncConnectionPool,
    cids: list[int],
) -> list[CampaignSignals]:
    """Pull DailyBudget + 7d Direct stats + 7d Bitrix leads + 7d Metrika bounce.

    Each source is wrapped in try/except so a single broken pipeline
    doesn't blank out the whole signal set — bounce_pct=None is a valid
    "no Metrika data" value and the decision rule handles it.
    """
    today = datetime.now(UTC).date().isoformat()
    week_ago = (datetime.now(UTC).date() - timedelta(days=7)).isoformat()

    # 1. DailyBudget + StartDate — explicit FieldNames so we actually get them.
    # StartDate is needed for LEARNING_GUARD (no scale/starve in first 14d).
    daily_by_cid: dict[int, int] = {}
    name_by_cid: dict[int, str] = {}
    age_by_cid: dict[int, int] = {}
    today_utc = datetime.now(UTC).date()
    try:
        camps_raw = await direct._call(  # noqa: SLF001
            "campaigns",
            "get",
            {
                "SelectionCriteria": {"Ids": cids},
                "FieldNames": ["Id", "Name", "DailyBudget", "StartDate"],
            },
        )
        for c in camps_raw.get("Campaigns") or []:
            cid = int(c.get("Id", 0))
            daily_by_cid[cid] = int((c.get("DailyBudget") or {}).get("Amount", 0)) // _MICRO
            name_by_cid[cid] = str(c.get("Name") or "")
            start_str = c.get("StartDate") or ""
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                age_by_cid[cid] = max(0, (today_utc - start).days)
            except (ValueError, TypeError):
                age_by_cid[cid] = 999  # unknown start = treat as old (no guard)
    except Exception:
        logger.exception("tactical_actuator: DailyBudget/StartDate fetch failed")

    # 2. 7d Direct stats per cid (clicks + cost). Use existing helper from
    # bitrix_feedback if available; here we replicate minimal contract.
    clicks_by_cid: dict[int, int] = {}
    cost_by_cid: dict[int, float] = {}
    for cid in cids:
        try:
            tsv = await direct.get_campaign_stats(cid, week_ago, today)
            clicks_by_cid[cid], cost_by_cid[cid] = _parse_tsv_clicks_cost(
                tsv if isinstance(tsv, str) else ""
            )
        except Exception:
            logger.warning("tactical_actuator: stats(%d) failed", cid, exc_info=True)
            clicks_by_cid[cid] = 0
            cost_by_cid[cid] = 0.0

    # 3. 7d leads (won deals as conservative proxy) per cid from sda_state
    leads_by_cid: dict[int, int] = await _fetch_leads_per_cid(pool)

    # 4. 7d bounce per cid from Metrika via lastDirectClickOrder
    bounce_by_cid: dict[int, float] = await _fetch_bounce_per_cid(
        http_client, settings, week_ago, today
    )

    signals: list[CampaignSignals] = []
    for cid in cids:
        signals.append(
            CampaignSignals(
                cid=cid,
                name=_OWN_CAMPAIGNS.get(cid, name_by_cid.get(cid, str(cid))),
                daily_rub=daily_by_cid.get(cid, 0),
                clicks_7d=clicks_by_cid.get(cid, 0),
                cost_7d=cost_by_cid.get(cid, 0.0),
                leads_7d=leads_by_cid.get(cid, 0),
                bounce_pct=bounce_by_cid.get(cid),
                age_days=age_by_cid.get(cid, 999),
            )
        )
    return signals


def _parse_tsv_clicks_cost(tsv: str) -> tuple[int, float]:
    """Sum Clicks and Cost (micro-RUB → RUB) from CAMPAIGN_PERFORMANCE_REPORT TSV."""
    if not tsv:
        return (0, 0.0)
    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    header_idx = -1
    for idx, line in enumerate(lines):
        if "CampaignId" in line and "Cost" in line:
            header_idx = idx
            break
    if header_idx < 0:
        return (0, 0.0)
    cols = lines[header_idx].split("\t")
    try:
        ci_clicks = cols.index("Clicks")
    except ValueError:
        ci_clicks = -1
    try:
        ci_cost = cols.index("Cost")
    except ValueError:
        ci_cost = -1
    total_clicks = 0
    total_cost_micro = 0
    for ln in lines[header_idx + 1 :]:
        c = ln.split("\t")
        if not c or c[0].startswith("Total") or len(c) <= max(ci_clicks, ci_cost):
            continue
        if ci_clicks >= 0:
            try:
                total_clicks += int(c[ci_clicks])
            except (ValueError, IndexError):
                pass
        if ci_cost >= 0:
            try:
                total_cost_micro += int(c[ci_cost])
            except (ValueError, IndexError):
                pass
    return (total_clicks, total_cost_micro / _MICRO)


async def _fetch_leads_per_cid(pool: AsyncConnectionPool) -> dict[int, int]:
    """Read sda_state[bitrix_feedback_traffic_split].own[cid].leads.

    Prefer the ``leads`` field (real Bitrix lead count, no won-lag) added
    2026-04-26. Fall back to ``won`` for snapshots written by older
    bitrix_feedback versions — keeps the actuator running through the
    deploy gap without crashes. After 1-2 days the new snapshot replaces
    the old one and the fallback path becomes dead.
    """
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM sda_state WHERE key = %s",
                    ("bitrix_feedback_traffic_split",),
                )
                row = await cur.fetchone()
    except Exception:
        logger.warning("tactical_actuator: leads fetch failed", exc_info=True)
        return {}
    if not row:
        return {}
    raw = row[0]
    if isinstance(raw, str):
        import json

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    elif isinstance(raw, dict):
        data = raw
    else:
        return {}
    out: dict[int, int] = {}
    for cid_str, info in (data.get("own") or {}).items():
        try:
            cid = int(cid_str)
        except (TypeError, ValueError):
            continue
        if isinstance(info, dict):
            value = info.get("leads")
            if value is None:
                value = info.get("won", 0)
            out[cid] = int(value or 0)
    return out


async def _fetch_bounce_per_cid(
    http_client: httpx.AsyncClient,
    settings: Settings,
    date1: str,
    date2: str,
) -> dict[int, float]:
    """Direct call to Metrika using the working dimension `lastDirectClickOrder`.

    The existing metrika.get_bounce_by_campaign uses ym:s:lastSignDirectOrderID
    which Metrika rejects with error 4001. Until that helper is fixed, we
    inline a minimal call here.
    """
    counter = getattr(settings, "METRIKA_COUNTER_ID", "")
    token_obj = getattr(settings, "METRIKA_OAUTH_TOKEN", None)
    if token_obj is None:
        token_str = ""
    elif hasattr(token_obj, "get_secret_value"):
        token_str = token_obj.get_secret_value()
    else:
        token_str = str(token_obj)
    if not counter or not token_str:
        logger.warning("tactical_actuator: Metrika settings missing")
        return {}
    params = {
        "ids": str(counter),
        "metrics": "ym:s:bounceRate",
        "dimensions": "ym:s:lastDirectClickOrder",
        "date1": date1,
        "date2": date2,
        "limit": "500",
        "accuracy": "full",
    }
    try:
        resp = await http_client.get(
            "https://api-metrika.yandex.net/stat/v1/data",
            params=params,
            headers={"Authorization": f"OAuth {token_str}"},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("tactical_actuator: Metrika %d %s", resp.status_code, resp.text[:200])
            return {}
        data = resp.json()
    except Exception:
        logger.exception("tactical_actuator: Metrika request failed")
        return {}
    out: dict[int, float] = {}
    for row in data.get("data") or []:
        dims = row.get("dimensions") or []
        m = row.get("metrics") or []
        if not dims or not m:
            continue
        raw_id = dims[0].get("id")
        try:
            cid = int(raw_id)
        except (TypeError, ValueError):
            continue
        out[cid] = float(m[0])
    return out


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    direct: DirectAPI | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron `0 */4 * * *` UTC.

    v2: multi-signal composite rules (Direct stats 7d + Bitrix leads + Metrika
    bounce). See decide_v2 for rules. v1 decide_action kept for backward
    compat (some tests still reference it).
    """
    if direct is None or http_client is None or settings is None:
        logger.warning(
            "tactical_actuator: DI missing (direct=%s http=%s settings=%s) — degraded_noop",
            direct is not None,
            http_client is not None,
            settings is not None,
        )
        return {
            "status": "ok",
            "action": "degraded_noop",
            "decisions": 0,
            "applied": 0,
            "dry_run": dry_run,
        }

    cids = list(_OWN_CAMPAIGNS.keys())
    try:
        signals = await _collect_signals(direct, http_client, settings, pool, cids)
    except Exception as exc:
        logger.exception("tactical_actuator: signal collection failed")
        return {
            "status": "error",
            "step": "collect_signals",
            "detail": str(exc),
            "dry_run": dry_run,
        }

    decisions = [decide_v2(s) for s in signals]
    actionable = [d for d in decisions if d.kind != "noop"]

    others_current_total = sum(d.current_daily_rub for d in decisions if d.kind == "noop")
    cap_check = plan_total_within_cap(actionable, others_current_total)

    applied: list[Decision] = []
    if dry_run:
        logger.info(
            "tactical_actuator: dry_run, %d decisions, %d actionable",
            len(decisions),
            len(actionable),
        )
    elif not cap_check[0]:
        logger.error(
            "tactical_actuator: total cap exceeded (%d > %d) — abort",
            cap_check[1],
            _TOTAL_CAP_RUB,
        )
    else:
        applied = await _apply_via_direct(direct, decisions)

    msg = _format_alert(decisions, applied, dry_run, cap_check)
    if not dry_run:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("tactical_actuator: telegram send failed")

    # Audit trail (numeric aggregates only — no PII risk here).
    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="autonomous",
            tool_name="tactical_actuator",
            tool_input={
                "cids": cids,
                "dry_run": dry_run,
            },
            tool_output={
                "decisions": [
                    {
                        "cid": d.cid,
                        "kind": d.kind,
                        "current": d.current_daily_rub,
                        "new": d.new_daily_rub,
                    }
                    for d in decisions
                ],
                "applied_cids": [d.cid for d in applied],
                "cap_within": cap_check[0],
                "total_after": cap_check[1],
            },
            is_mutation=bool(applied),
        )
    except Exception:
        logger.warning("tactical_actuator: audit_log write failed", exc_info=True)

    return {
        "status": "ok",
        "decisions": len(decisions),
        "actionable": len(actionable),
        "applied": len(applied),
        "cap_within": cap_check[0],
        "total_after": cap_check[1],
        "dry_run": dry_run,
    }


__all__ = [
    "Decision",
    "decide_action",
    "plan_total_within_cap",
    "run",
]
