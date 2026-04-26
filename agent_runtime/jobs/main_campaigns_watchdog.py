"""Main campaigns watchdog — auto-resume SUSPENDED main campaigns.

Following 26.04.2026 incident: 3 main BFL campaigns (708978456/7/8 —
Башк/Тат/Удм) found SUSPENDED at 15:00 МСК, owner alerted manually,
agent-side mutations limit (decision_engine MAX_MUTATIONS_PER_WEEK=5)
blocked any auto-resume by smart_optimizer. Manual Direct API call
needed to recover.

This watchdog runs every 30 min and:
  1. Reads State + Status + StatusPayment of all main cids
  2. If State == SUSPENDED AND StatusPayment == ALLOWED AND
     Status == ACCEPTED → calls campaigns.resume() directly via
     DirectAPI (bypasses agent-side mutation budget)
  3. Sends Telegram alert with action + before/after states
  4. Audit log entry with kill_switch_triggered = "main_watchdog_resume"

Conservative: only resumes when payment is fine AND status is accepted.
Won't auto-resume if:
  * StatusPayment == DISALLOWED (no money) — alerts owner instead
  * Status != ACCEPTED (moderation issue) — alerts owner
  * State == OFF (manually paused by owner) — only SUSPENDED counts as
    auto-recoverable

Cron: every 30 min via shadow-cron.yml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agent_runtime.db import insert_audit_log
from agent_runtime.tools import telegram as telegram_tools

if TYPE_CHECKING:
    import httpx
    from psycopg_pool import AsyncConnectionPool

    from agent_runtime.config import Settings
    from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)


_MAIN_CAMPAIGNS: dict[int, str] = {}
# 26.04.2026: ЕПК-кампании 708978456/7/8 заглушены владельцем (Вариант 1 аудита).
# Ранее watchdog воскрешал их вопреки решению — поведение отключено через пустой whitelist.
# См. memory/feedback_sda2_campaigns_disabled.md.


@dataclass(frozen=True)
class MainState:
    cid: int
    name: str
    state: str
    status: str
    status_payment: str
    needs_resume: bool
    needs_alert: bool
    alert_reason: str = ""


def assess_main(camp: dict[str, Any]) -> MainState:
    """Pure decision: should we auto-resume / alert / leave alone?

    Auto-resume: State=SUSPENDED + StatusPayment=ALLOWED + Status=ACCEPTED.
    Alert (no resume): payment problem or moderation issue.
    Leave alone: State=ON (working) or State=OFF (manual pause).
    """
    cid = int(camp.get("Id", 0))
    name = _MAIN_CAMPAIGNS.get(cid, str(cid))
    state = str(camp.get("State", "")).upper()
    status = str(camp.get("Status", "")).upper()
    status_payment = str(camp.get("StatusPayment", "")).upper()

    needs_resume = False
    needs_alert = False
    reason = ""

    if state == "SUSPENDED":
        if status_payment != "ALLOWED":
            needs_alert = True
            reason = f"SUSPENDED + StatusPayment={status_payment} — пополнить баланс"
        elif status != "ACCEPTED":
            needs_alert = True
            reason = f"SUSPENDED + Status={status} — модерация"
        else:
            needs_resume = True
            reason = "SUSPENDED + payment OK + accepted → auto-resume"

    return MainState(
        cid=cid,
        name=name,
        state=state,
        status=status,
        status_payment=status_payment,
        needs_resume=needs_resume,
        needs_alert=needs_alert,
        alert_reason=reason,
    )


def _format_alert(states: list[MainState], resumed: list[int], errors: list[str]) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"<b>🛡️ Main Campaigns Watchdog</b> — {when}", ""]

    on = [s for s in states if s.state == "ON"]
    off = [s for s in states if s.state == "OFF"]
    susp = [s for s in states if s.state == "SUSPENDED"]
    alerts = [s for s in states if s.needs_alert]

    if on:
        lines.append(f"🟢 ON ({len(on)}): " + ", ".join(s.name for s in on))
    if off:
        lines.append(f"⏸ OFF ({len(off)}, ручная пауза): " + ", ".join(s.name for s in off))
    if susp:
        lines.append(f"🔴 SUSPENDED ({len(susp)}): " + ", ".join(s.name for s in susp))

    if resumed:
        lines.append("")
        lines.append(f"<b>✅ Auto-resumed:</b> {resumed}")

    if alerts:
        lines.append("")
        lines.append("<b>⚠️ Требует внимания (НЕ auto-resume):</b>")
        for s in alerts:
            lines.append(f"  • <code>{s.cid}</code> ({s.name}): {s.alert_reason}")

    if errors:
        lines.append("")
        lines.append("<b>❌ Ошибки:</b>")
        for e in errors[:5]:
            lines.append(f"  • {e}")

    if not (resumed or alerts or errors):
        lines.append("")
        lines.append("Все ОК.")

    return "\n".join(lines)


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    direct: DirectAPI | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron every 30 min."""
    if direct is None or http_client is None or settings is None:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "resumed": 0,
            "alerts": 0,
            "dry_run": dry_run,
        }

    cids = list(_MAIN_CAMPAIGNS.keys())
    try:
        camps_raw = await direct._call(  # noqa: SLF001
            "campaigns",
            "get",
            {
                "SelectionCriteria": {"Ids": cids},
                "FieldNames": ["Id", "Name", "State", "Status", "StatusPayment"],
            },
        )
    except Exception as exc:
        logger.exception("main_watchdog: get_campaigns failed")
        return {"status": "error", "step": "get", "detail": str(exc), "dry_run": dry_run}

    states = [assess_main(c) for c in (camps_raw.get("Campaigns") or [])]
    to_resume = [s.cid for s in states if s.needs_resume]
    resumed: list[int] = []
    errors: list[str] = []

    if to_resume and not dry_run:
        try:
            res = await direct._call(  # noqa: SLF001
                "campaigns",
                "resume",
                {"SelectionCriteria": {"Ids": to_resume}},
            )
            resumed = [int(r.get("Id", 0)) for r in (res.get("ResumeResults") or [])]
        except Exception as exc:
            logger.exception("main_watchdog: resume failed")
            errors.append(f"resume: {type(exc).__name__}: {exc}")

    msg = _format_alert(states, resumed, errors)

    # Send Telegram only when something actionable happened or alerts present.
    should_send = bool(resumed or any(s.needs_alert for s in states) or errors)
    if should_send and not dry_run:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("main_watchdog: telegram failed")

    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="autonomous",
            tool_name="main_campaigns_watchdog",
            tool_input={"cids": cids, "dry_run": dry_run},
            tool_output={
                "states": [
                    {"cid": s.cid, "state": s.state, "status_payment": s.status_payment}
                    for s in states
                ],
                "resumed": resumed,
                "alerts_count": sum(1 for s in states if s.needs_alert),
                "errors": errors,
            },
            is_mutation=bool(resumed),
        )
    except Exception:
        logger.warning("main_watchdog: audit_log failed", exc_info=True)

    return {
        "status": "ok",
        "checked": len(states),
        "resumed": len(resumed),
        "alerts": sum(1 for s in states if s.needs_alert),
        "errors": len(errors),
        "dry_run": dry_run,
    }


__all__ = ["MainState", "assess_main", "run"]
