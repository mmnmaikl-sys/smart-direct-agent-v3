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
from datetime import UTC, datetime
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
    kind: Literal["scale", "starve", "noop"]
    current_daily_rub: int
    new_daily_rub: int
    reason: str


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


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    direct: DirectAPI | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron `0 */4 * * *` UTC."""
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
        camps = await direct.get_campaigns(cids)
    except Exception as exc:
        logger.exception("tactical_actuator: get_campaigns failed")
        return {"status": "error", "step": "get_campaigns", "detail": str(exc), "dry_run": dry_run}

    decisions = [decide_action(c) for c in camps]
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
