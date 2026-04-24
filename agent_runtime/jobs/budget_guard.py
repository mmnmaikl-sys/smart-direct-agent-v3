"""Budget Guard — P1 defence-in-depth cron (Task 14).

Runs every 30 minutes between 8:00-23:30 МСК (Railway cron ``*/30 5-20``
in UTC). For each campaign in ``PROTECTED_CAMPAIGN_IDS`` that is ``State=ON``:

1. Fetch today's cost + 7-day daily average through
   :func:`DirectAPI.get_campaign_stats` (TSV parsed in :func:`_parse_costs`).
2. Short-circuit for an active hypothesis: if a ``running`` row is attached
   to this campaign and cumulative spend since ``cost_snapshot_at_start``
   has reached ``hypotheses.budget_cap_rub`` — call
   :func:`decision_journal.update_outcome` with ``outcome='neutral'`` and
   **skip Telegram alert / suspend**. This is the "cap reached = concluded
   test" path from Decision 4 — the forcing function, not a failure.
3. Otherwise reuse :class:`BudgetCap` (Task 9) to decide breach on the
   relative surge; combine with the absolute ``DailyBudget.Amount`` ceiling
   as a second OR-gate.
4. Trust overlay:
   * ``shadow`` / ``assisted`` / ``FORBIDDEN_LOCK`` → Telegram NOTIFY only.
   * ``autonomous`` + ``not dry_run`` → :func:`DirectAPI.pause_campaign`
     with ``verify_campaign_paused`` retry, then NOTIFY.

``PROTECTED_CAMPAIGN_IDS`` are protected at the DirectAPI layer from brain
mutation (Decision 12). ``pause_campaign`` on them raises
``ProtectedCampaignError`` — caught gracefully: Telegram alert still fires,
no suspend recorded. This is sound: prod campaigns require human ack, the
guard refuses to silently shut down 180K₽/month revenue.

``dry_run=True`` keeps Telegram NOTIFY off (so smoke curls don't spam the
owner) and returns ``would_suspend`` / ``would_notify`` markers instead.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from agent_runtime import decision_journal, knowledge
from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI, ProtectedCampaignError
from agent_runtime.tools.kill_switches import Action, BudgetCap, KillSwitchContext
from agent_runtime.trust_levels import TrustLevel, get_trust_level

logger = logging.getLogger(__name__)


_MICRO_TO_RUB = 1_000_000
_VERIFY_RETRY_ATTEMPTS = 3
_VERIFY_RETRY_SLEEP_SEC = 2.0
_STATS_WINDOW_DAYS = 7


@dataclass(frozen=True)
class _CampaignBudget:
    campaign_id: int
    name: str
    daily_limit_rub: int | None
    today_cost: float
    daily_avg_7d: float


class BudgetGuardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str  # "ok" | "error"
    trust_level: str
    checked_campaigns: list[int] = Field(default_factory=list)
    breached: list[int] = Field(default_factory=list)
    notified: list[int] = Field(default_factory=list)
    suspended: list[int] = Field(default_factory=list)
    would_suspend: list[int] = Field(default_factory=list)
    hypothesis_concluded: list[str] = Field(default_factory=list)
    kb_citation: dict[str, Any] | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --------------------------------------------------------------- stats parsing


def _parse_costs(tsv: str) -> dict[str, float]:
    """Parse Direct CAMPAIGN_PERFORMANCE_REPORT TSV into ``{date: cost_rub}``.

    Skips the header/title lines and the trailing "Total" row. Cost column
    is micro-rubles (Direct contract); divide once at the boundary.
    """
    per_day: dict[str, float] = {}
    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    header_idx = -1
    for idx, line in enumerate(lines):
        if "Date" in line and "Cost" in line:
            header_idx = idx
            break
    if header_idx < 0:
        return per_day
    header_cols = lines[header_idx].split("\t")
    try:
        date_i = header_cols.index("Date")
        cost_i = header_cols.index("Cost")
    except ValueError:
        return per_day
    for line in lines[header_idx + 1 :]:
        cols = line.split("\t")
        if not cols or cols[0].startswith("Total") or len(cols) <= max(date_i, cost_i):
            continue
        try:
            date_str = cols[date_i]
            cost_micro = int(cols[cost_i])
        except (ValueError, IndexError):
            continue
        per_day[date_str] = cost_micro / _MICRO_TO_RUB
    return per_day


async def _fetch_today_and_avg(direct: DirectAPI, campaign_id: int) -> tuple[float, float]:
    """Fetch today_cost (rub) + 7-day daily average (rub) for one campaign."""
    now = datetime.now(UTC) + timedelta(hours=3)  # МСК — matches Direct local day
    today = now.date().isoformat()
    week_ago = (now - timedelta(days=_STATS_WINDOW_DAYS)).date().isoformat()
    raw = await direct.get_campaign_stats(campaign_id, week_ago, today)
    per_day = _parse_costs(str(raw.get("tsv", "")))
    today_cost = per_day.get(today, 0.0)
    historical = [v for k, v in per_day.items() if k != today]
    daily_avg = sum(historical) / len(historical) if historical else 0.0
    return today_cost, daily_avg


# ------------------------------------------------------- hypothesis short-path


@dataclass(frozen=True)
class _RunningHypothesis:
    id: str
    budget_cap_rub: int
    created_at: datetime
    cost_snapshot_rub: float


async def _find_running_hypothesis_for_campaign(
    pool: AsyncConnectionPool, campaign_id: int
) -> _RunningHypothesis | None:
    """Return the earliest ``running`` hypothesis tied to ``campaign_id``.

    ``cost_snapshot_rub`` is read from ``metrics_before.cost_snapshot_today``
    written by :func:`decision_journal.record_hypothesis` when the brain
    locks a test. If missing (old row), returns 0.0 — conservative: the
    guard concludes the hypothesis earlier rather than later.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, budget_cap_rub, created_at, metrics_before
                FROM hypotheses
                WHERE state = 'running' AND campaign_id = %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (campaign_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    metrics_before = row[3] or {}
    snapshot = float(metrics_before.get("cost_snapshot_today", 0.0))
    return _RunningHypothesis(
        id=str(row[0]),
        budget_cap_rub=int(row[1]),
        created_at=row[2],
        cost_snapshot_rub=snapshot,
    )


# --------------------------------------------------------------- suspend path


async def _suspend_with_verify(direct: DirectAPI, campaign_id: int) -> bool:
    """Call ``pause_campaign`` and verify; best-effort, returns success bool.

    Swallows :class:`ProtectedCampaignError` — PROTECTED_CAMPAIGN_IDS cannot
    be silently muted by any automatic path, even autonomous. The Telegram
    alert still fires and owner decides.
    """
    try:
        await direct.pause_campaign(campaign_id)
    except ProtectedCampaignError:
        logger.warning(
            "budget_guard: pause_campaign(%d) blocked by protected guard — "
            "Telegram alert still sent, owner to decide",
            campaign_id,
        )
        return False
    except Exception:
        logger.exception("budget_guard: pause_campaign(%d) failed", campaign_id)
        return False
    for attempt in range(_VERIFY_RETRY_ATTEMPTS):
        try:
            if await direct.verify_campaign_paused(campaign_id):
                return True
        except Exception:
            logger.warning(
                "budget_guard: verify_campaign_paused(%d) attempt %d errored",
                campaign_id,
                attempt + 1,
                exc_info=True,
            )
        await _sleep_for_verify()
    logger.warning("budget_guard: verify_campaign_paused(%d) never True", campaign_id)
    return False


async def _sleep_for_verify() -> None:
    """Isolated for monkeypatching in tests."""
    await asyncio.sleep(_VERIFY_RETRY_SLEEP_SEC)


# ---------------------------------------------------------------- formatting


def _format_alert(
    *,
    campaign_id: int,
    name: str,
    today_cost: float,
    daily_avg: float,
    daily_limit: int | None,
    reason: str,
    trust_level: TrustLevel,
    auto_suspended: bool,
    hypothesis_context: str | None = None,
) -> str:
    title = "🛑 <b>AUTO-SUSPEND</b>" if auto_suspended else "⚠️ <b>BUDGET BREACH</b>"
    lines = [
        title,
        f"Campaign <code>{campaign_id}</code> — {name}",
        "",
        f"Today cost: <b>{today_cost:.0f}₽</b>",
        f"Avg 7d:     {daily_avg:.0f}₽",
    ]
    if daily_limit is not None:
        lines.append(f"Daily limit: {daily_limit}₽")
    lines.append("")
    lines.append(f"Reason: {reason}")
    if hypothesis_context:
        lines.append(f"Hypothesis context: {hypothesis_context}")
    lines.append(f"Trust: <code>{trust_level.value}</code>")
    if not auto_suspended:
        lines.append(f"Action: <i>no auto-suspend</i> (trust={trust_level.value})")
    return "\n".join(lines)


# ----------------------------------------------------------------------- run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Cron entry. Returns ``BudgetGuardResult.model_dump()``.

    When invoked through the default ``JOB_REGISTRY`` wrapper (only
    ``pool + dry_run``), the job runs in a degraded no-op path: it
    reads ``trust_level``, logs a warning that DirectAPI was not
    injected, and returns ``status='ok', checked_campaigns=[]``. Real
    prod runs go through the FastAPI ``/run/budget_guard`` handler
    which injects ``direct``, ``http_client``, ``settings`` from
    ``app.state``.
    """
    trust_level = await _safe_get_trust_level(pool)

    if direct is None or settings is None:
        logger.warning("budget_guard: direct/settings not injected — degraded no-op run")
        return BudgetGuardResult(status="ok", trust_level=trust_level.value).model_dump(mode="json")

    checked: list[int] = []
    breached: list[int] = []
    notified: list[int] = []
    suspended: list[int] = []
    would_suspend: list[int] = []
    concluded: list[str] = []

    for campaign_id in settings.PROTECTED_CAMPAIGN_IDS:
        try:
            budget = await _collect_campaign_budget(direct, campaign_id)
        except Exception:
            logger.exception(
                "budget_guard: stats fetch failed for campaign %d, skip tick",
                campaign_id,
            )
            continue
        if budget is None:
            continue
        checked.append(campaign_id)

        hyp = await _find_running_hypothesis_for_campaign(pool, campaign_id)
        if hyp is not None:
            delta = max(0.0, budget.today_cost - hyp.cost_snapshot_rub)
            if delta >= hyp.budget_cap_rub:
                await _conclude_hypothesis_at_cap(pool, hyp, budget, trust_level=trust_level)
                concluded.append(hyp.id)
                continue

        breach = await _detect_breach(pool, direct, settings, budget, trust_level)
        if not breach:
            continue
        breached.append(campaign_id)
        reason, hyp_context = breach

        auto_suspend_allowed = trust_level == TrustLevel.AUTONOMOUS and not dry_run
        suspended_ok = False
        if auto_suspend_allowed:
            suspended_ok = await _suspend_with_verify(direct, campaign_id)
            if suspended_ok:
                suspended.append(campaign_id)
        elif trust_level == TrustLevel.AUTONOMOUS and dry_run:
            would_suspend.append(campaign_id)

        if not dry_run and http_client is not None:
            text = _format_alert(
                campaign_id=campaign_id,
                name=budget.name,
                today_cost=budget.today_cost,
                daily_avg=budget.daily_avg_7d,
                daily_limit=budget.daily_limit_rub,
                reason=reason,
                trust_level=trust_level,
                auto_suspended=suspended_ok,
                hypothesis_context=hyp_context,
            )
            try:
                await telegram_tools.send_message(http_client, settings, text=text)
                notified.append(campaign_id)
            except Exception:
                logger.exception("budget_guard: telegram alert failed")

        try:
            await insert_audit_log(
                pool,
                hypothesis_id=None,
                trust_level=trust_level.value,
                tool_name="budget_guard",
                tool_input={
                    "campaign_id": campaign_id,
                    "today_cost": budget.today_cost,
                    "daily_avg_7d": budget.daily_avg_7d,
                    "daily_limit_rub": budget.daily_limit_rub,
                },
                tool_output={
                    "breached": True,
                    "reason": reason,
                    "suspended": suspended_ok,
                    "dry_run": dry_run,
                },
                is_mutation=suspended_ok,
                kill_switch_triggered="budget_cap",
            )
        except Exception:
            logger.exception("budget_guard: audit_log write failed")

    citation = await _kb_citation(trust_level)

    result = BudgetGuardResult(
        status="ok",
        trust_level=trust_level.value,
        checked_campaigns=checked,
        breached=breached,
        notified=notified,
        suspended=suspended,
        would_suspend=would_suspend,
        hypothesis_concluded=concluded,
        kb_citation=citation,
    )
    return result.model_dump(mode="json")


# ------------------------------------------------------------ internal helpers


async def _safe_get_trust_level(pool: AsyncConnectionPool) -> TrustLevel:
    try:
        return await get_trust_level(pool)
    except Exception:
        logger.warning("budget_guard: trust_level lookup failed, defaulting shadow", exc_info=True)
        return TrustLevel.SHADOW


async def _collect_campaign_budget(direct: DirectAPI, campaign_id: int) -> _CampaignBudget | None:
    campaigns = await direct.get_campaigns([campaign_id])
    if not campaigns:
        return None
    camp = campaigns[0]
    if camp.get("State") != "ON":
        return None
    daily_budget = camp.get("DailyBudget") or {}
    amount_micro = daily_budget.get("Amount")
    daily_limit = int(amount_micro) // _MICRO_TO_RUB if amount_micro else None
    today_cost, daily_avg = await _fetch_today_and_avg(direct, campaign_id)
    return _CampaignBudget(
        campaign_id=campaign_id,
        name=str(camp.get("Name") or ""),
        daily_limit_rub=daily_limit,
        today_cost=today_cost,
        daily_avg_7d=daily_avg,
    )


async def _detect_breach(
    pool: AsyncConnectionPool,
    direct: DirectAPI,
    settings: Settings,
    budget: _CampaignBudget,
    trust_level: TrustLevel,
) -> tuple[str, str | None] | None:
    """Run BudgetCap + absolute limit. Returns (reason, hyp_context) or None."""
    action = Action(
        type="pause_campaign",
        params={"campaign_id": budget.campaign_id, "reason": "budget_guard_breach"},
    )
    ctx = KillSwitchContext(
        pool=pool,
        direct=direct,
        metrika=None,
        bitrix=None,
        settings=settings,
        trust_level=trust_level.value,
        budget_history={
            budget.campaign_id: {
                "today_cost": budget.today_cost,
                "daily_avg_7d": budget.daily_avg_7d,
            }
        },
    )
    ks_result = await BudgetCap().check(action, ctx)
    surge_breach = not ks_result.allow
    absolute_breach = (
        budget.daily_limit_rub is not None and budget.today_cost > budget.daily_limit_rub
    )
    if not surge_breach and not absolute_breach:
        return None
    reason_parts: list[str] = []
    if surge_breach:
        reason_parts.append(ks_result.reason)
    if absolute_breach:
        reason_parts.append(
            f"today_cost={budget.today_cost:.0f} > daily_limit={budget.daily_limit_rub}"
        )
    return " | ".join(reason_parts), None


async def _conclude_hypothesis_at_cap(
    pool: AsyncConnectionPool,
    hyp: _RunningHypothesis,
    budget: _CampaignBudget,
    *,
    trust_level: TrustLevel,
) -> None:
    lesson = (
        f"budget_cap {hyp.budget_cap_rub}₽ reached "
        f"(today_cost={budget.today_cost:.0f}, snapshot={hyp.cost_snapshot_rub:.0f}) "
        "→ concluded"
    )
    try:
        await decision_journal.update_outcome(
            pool,
            hyp.id,
            outcome="neutral",
            metrics_after={"cost": budget.today_cost},
            lesson=lesson,
        )
    except Exception:
        logger.exception("budget_guard: update_outcome(%s) failed", hyp.id)
        return
    try:
        await insert_audit_log(
            pool,
            hypothesis_id=hyp.id,
            trust_level=trust_level.value,
            tool_name="budget_guard",
            tool_input={"campaign_id": budget.campaign_id, "today_cost": budget.today_cost},
            tool_output={"concluded": True, "reason": "hypothesis_budget_cap_reached"},
            is_mutation=False,
            kill_switch_triggered="budget_cap",
        )
    except Exception:
        logger.exception("budget_guard: audit_log (hyp conclude) failed")


async def _kb_citation(trust_level: TrustLevel) -> dict[str, Any] | None:
    """Best-effort KB lookup — never raises. Absence is logged, not fatal."""
    try:
        answer = await knowledge.consult(
            "budget_guard policy when today_cost exceeds daily average × 1.5",
            context={"trust_level": trust_level.value},
        )
    except Exception as exc:
        logger.info("budget_guard: kb.consult skipped (%s)", exc.__class__.__name__)
        return None
    return {
        "q": "budget_guard policy when today_cost exceeds daily average × 1.5",
        "a": answer.get("answer", ""),
        "citations": answer.get("citations", []),
    }


__all__ = [
    "BudgetGuardResult",
    "run",
]
