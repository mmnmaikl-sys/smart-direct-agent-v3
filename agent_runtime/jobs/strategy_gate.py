"""Strategy Gate — 4-signal maturity evaluator + state machine (Task 17).

Detects when the BFL campaigns are ready to switch from manual
(``WB_MAX_CLICKS``) to auto-bid (``WB_MAXIMUM_CONVERSION_RATE``), and once
on auto-bid, watches for CPA degradation and demands manual intervention.

**4 independent signals** (all must be green to move ``learning →
ready_to_switch``):

1. ``won_30d`` — count of Bitrix C45:WON deals in the last 30 days whose
   originating lead carries ``UTM_CAMPAIGN`` ∈ :data:`UTM_CAMPAIGN_WHITELIST`.
   Target ≥ :data:`SG_WON_REQUIRED` (default 10).
2. ``cpa_stability_7d`` — coefficient of variation (stdev / mean) of
   per-day CPA across the last 7 days ≤ :data:`SG_CPA_VARIANCE_MAX`.
   Fewer than 3 data points → ``stable=False, reason='too_few_days'``.
3. ``offline_conversions_ok`` — ≥ :data:`SG_OFFLINE_MIN_RUNS` successful
   runs of the ``offline_conversions`` agent (``is_error=FALSE``) in
   ``audit_log`` over :data:`SG_OFFLINE_DAYS_OK` days, zero errors.
4. ``direct_conversions_accumulated`` — sum of Direct TSV ``Conversions``
   column across :attr:`Settings.PROTECTED_CAMPAIGN_IDS` for 30 days
   ≥ :data:`SG_DIRECT_CONV_REQUIRED` (default 20).

**State machine** (persisted in ``sda_state[key='strategy_gate_state']``
as JSONB with ``SELECT FOR UPDATE`` row lock):

* ``learning`` → ``ready_to_switch`` when all 4 signals green.
* ``ready_to_switch`` → ``learning`` when ``won_30d`` regresses below
  target.
* ``auto_pilot`` → ``degraded`` when ``current_cpa > baseline_cpa ×
  SG_DEGRADED_CPA_GROWTH`` (default 1.5).
* ``degraded`` is sticky — only :func:`manual_switch` moves out.

Every transition fires a best-effort Telegram alert (HTML). Telegram
failure is logged but does NOT block the state write — the PG row is
the authority, the alert is a convenience.

TODO(integration): the module-level ``SG_*`` constants and
``UTM_CAMPAIGN_WHITELIST`` will move to :class:`Settings` during the
task-17 integration pass. They live here first so this module lands in
one commit with no schema drift.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import bitrix as bitrix_tools
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)


# --- thresholds (module constants) ------------------------------------------
# TODO(integration): move to Settings as SG_* pydantic fields; keep these as
# fallback defaults wired through a Field(default=...).
SG_WON_REQUIRED: int = 10
SG_DIRECT_CONV_REQUIRED: int = 20
SG_CPA_VARIANCE_MAX: float = 0.3
SG_OFFLINE_DAYS_OK: int = 7
SG_OFFLINE_MIN_RUNS: int = 5
SG_DEGRADED_CPA_GROWTH: float = 1.5
SG_WATCHDOG_INTERVAL_LEARNING_MIN: int = 15
SG_WATCHDOG_INTERVAL_AUTOPILOT_MIN: int = 240
UTM_CAMPAIGN_WHITELIST: tuple[str, ...] = ("bfl-rf",)

_STATE_KEY = "strategy_gate_state"
_VALID_STATUSES: frozenset[str] = frozenset(
    {"learning", "ready_to_switch", "auto_pilot", "degraded"}
)
_STATUS_EMOJI: dict[str, str] = {
    "learning": "🔨",
    "ready_to_switch": "🔔",
    "auto_pilot": "🚀",
    "degraded": "⚠️",
}

_HISTORY_CAP = 50  # truncate history so the JSONB never unbounded-grows

Status = Literal["learning", "ready_to_switch", "auto_pilot", "degraded"]


# --- signal result types ----------------------------------------------------


@dataclass(frozen=True)
class WonResult:
    count: int
    revenue_rub: float
    met: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.count,
            "revenue_rub": self.revenue_rub,
            "required": SG_WON_REQUIRED,
            "met": self.met,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CpaStabilityResult:
    stable: bool
    mean: float
    stdev: float
    variance: float
    days: int
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.variance, 4),
            "mean": round(self.mean, 2),
            "stdev": round(self.stdev, 2),
            "days": self.days,
            "threshold": SG_CPA_VARIANCE_MAX,
            "met": self.stable,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OfflineConversionsResult:
    total: int
    errors: int
    met: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.total,
            "errors": self.errors,
            "required": SG_OFFLINE_MIN_RUNS,
            "met": self.met,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DirectConversionsResult:
    count: int
    met: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.count,
            "required": SG_DIRECT_CONV_REQUIRED,
            "met": self.met,
            "reason": self.reason,
        }


# --- signal: won_30d --------------------------------------------------------


async def _count_won_30d(
    http: httpx.AsyncClient,
    settings: Settings,
    *,
    whitelist: tuple[str, ...] = UTM_CAMPAIGN_WHITELIST,
) -> WonResult:
    """Count WON deals in last 30d whose originating lead is whitelisted.

    Two-step query (spec: stagehistory → deal.list → lead.list), but Bitrix
    ``crm.deal.list`` already supports ``STAGE_ID`` + ``>=CLOSEDATE`` filters
    so we skip the stagehistory hop unless we need transition timing — we
    only need counts here. LEAD_ID on each deal points back to the
    originating lead whose UTM_CAMPAIGN we filter against.
    """
    try:
        since = (datetime.now(UTC) - timedelta(days=30)).date().isoformat()
        deals = await bitrix_tools.get_deal_list(
            http,
            settings,
            filter={"STAGE_ID": "C45:WON", ">=CLOSEDATE": since},
            select=["ID", "OPPORTUNITY", "LEAD_ID"],
            max_total=1000,
        )
        if not deals:
            return WonResult(count=0, revenue_rub=0.0, met=False, reason="no_deals")

        lead_ids = [str(d["LEAD_ID"]) for d in deals if d.get("LEAD_ID")]
        if not lead_ids:
            return WonResult(count=0, revenue_rub=0.0, met=False, reason="deals_without_lead")

        leads = await bitrix_tools.get_lead_list(
            http,
            settings,
            filter={"ID": lead_ids},
            select=["ID", "UTM_CAMPAIGN"],
            max_total=1000,
        )
        allowed = {w.lower() for w in whitelist}
        whitelisted_ids = {
            str(lead["ID"])
            for lead in leads
            if str(lead.get("UTM_CAMPAIGN") or "").lower() in allowed
        }
        matched = [d for d in deals if str(d.get("LEAD_ID")) in whitelisted_ids]
        count = len(matched)
        revenue = sum(float(d.get("OPPORTUNITY") or 0) for d in matched)
        return WonResult(
            count=count,
            revenue_rub=revenue,
            met=count >= SG_WON_REQUIRED,
        )
    except Exception as exc:  # noqa: BLE001 — signal is fail-safe by contract
        logger.warning("strategy_gate: _count_won_30d failed: %s", exc, exc_info=True)
        return WonResult(count=0, revenue_rub=0.0, met=False, reason=str(exc))


# --- signal: cpa_stability_7d ----------------------------------------------


def _parse_tsv_column(tsv: str, column: str) -> list[tuple[str, float]]:
    """Parse a Direct CAMPAIGN_PERFORMANCE_REPORT TSV column into [(date, value)].

    Mirrors ``budget_guard._parse_costs`` but returns the requested column
    as a float (handles ``--`` / empty cells → 0.0). Skips header / title
    lines and the trailing ``Total`` row.
    """
    out: list[tuple[str, float]] = []
    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    header_idx = -1
    for idx, line in enumerate(lines):
        if "Date" in line and column in line:
            header_idx = idx
            break
    if header_idx < 0:
        return out
    header_cols = lines[header_idx].split("\t")
    try:
        date_i = header_cols.index("Date")
        col_i = header_cols.index(column)
    except ValueError:
        return out
    for line in lines[header_idx + 1 :]:
        cols = line.split("\t")
        if not cols or cols[0].startswith("Total") or len(cols) <= max(date_i, col_i):
            continue
        try:
            date_str = cols[date_i]
            raw = cols[col_i].strip()
            value = 0.0 if raw in {"", "--"} else float(raw)
        except (ValueError, IndexError):
            continue
        out.append((date_str, value))
    return out


async def _cpa_stability_7d(
    direct: DirectAPI,
    settings: Settings,
) -> CpaStabilityResult:
    """Pull 7-day per-day CPA across PROTECTED_CAMPAIGN_IDS and measure spread.

    CPA per day = Σ(cost) / Σ(conversions). Days with zero conversions are
    skipped (div-by-zero + noise from off-hours crawls). <3 usable days →
    ``stable=False, reason='too_few_days'`` per spec.
    """
    try:
        now = datetime.now(UTC) + timedelta(hours=3)  # МСК (matches Direct local day)
        today = now.date().isoformat()
        week_ago = (now - timedelta(days=7)).date().isoformat()

        per_day_cost: dict[str, float] = {}
        per_day_conv: dict[str, float] = {}
        for campaign_id in settings.PROTECTED_CAMPAIGN_IDS:
            try:
                raw = await direct.get_campaign_stats(campaign_id, week_ago, today)
            except Exception:
                logger.warning(
                    "strategy_gate: get_campaign_stats(%d) failed; skipping",
                    campaign_id,
                    exc_info=True,
                )
                continue
            tsv = str(raw.get("tsv", ""))
            for date, cost_micro in _parse_tsv_column(tsv, "Cost"):
                per_day_cost[date] = per_day_cost.get(date, 0.0) + cost_micro / 1_000_000
            for date, conv in _parse_tsv_column(tsv, "Conversions"):
                per_day_conv[date] = per_day_conv.get(date, 0.0) + conv

        daily_cpa: list[float] = []
        for date, cost in per_day_cost.items():
            conv = per_day_conv.get(date, 0.0)
            if conv <= 0 or cost <= 0:
                continue
            daily_cpa.append(cost / conv)

        if len(daily_cpa) < 3:
            return CpaStabilityResult(
                stable=False,
                mean=0.0,
                stdev=0.0,
                variance=0.0,
                days=len(daily_cpa),
                reason="too_few_days",
            )

        mean = statistics.mean(daily_cpa)
        stdev = statistics.stdev(daily_cpa)
        variance = stdev / mean if mean > 0 else 0.0
        return CpaStabilityResult(
            stable=variance <= SG_CPA_VARIANCE_MAX,
            mean=mean,
            stdev=stdev,
            variance=variance,
            days=len(daily_cpa),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("strategy_gate: _cpa_stability_7d failed: %s", exc, exc_info=True)
        return CpaStabilityResult(
            stable=False, mean=0.0, stdev=0.0, variance=0.0, days=0, reason=str(exc)
        )


# --- signal: offline_conversions_ok ----------------------------------------


# NOTE: decision_journal.get_actions_today() filters to today + is_mutation=TRUE,
# which is not what we want here. We need a per-agent + days window + error
# count — so we query audit_log directly. Task 12b may later extend the
# journal with ``get_actions_by_agent(agent, days)``; until then this one
# parameterised SELECT is the sanctioned way.
async def _offline_conversions_ok(
    pool: AsyncConnectionPool,
    *,
    days: int = SG_OFFLINE_DAYS_OK,
    min_runs: int = SG_OFFLINE_MIN_RUNS,
) -> OfflineConversionsResult:
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE is_error = TRUE) AS errors
                    FROM audit_log
                    WHERE tool_name = 'offline_conversions'
                      AND ts > NOW() - make_interval(days => %s)
                    """,
                    (days,),
                )
                row = await cur.fetchone()
        total = int(row[0]) if row and row[0] is not None else 0
        errors = int(row[1]) if row and row[1] is not None else 0
        met = total >= min_runs and errors == 0
        reason = ""
        if total < min_runs:
            reason = f"too_few_runs({total}<{min_runs})"
        elif errors > 0:
            reason = f"errors={errors}"
        return OfflineConversionsResult(total=total, errors=errors, met=met, reason=reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("strategy_gate: _offline_conversions_ok failed: %s", exc, exc_info=True)
        return OfflineConversionsResult(total=0, errors=0, met=False, reason=str(exc))


# --- signal: direct_conversions_accumulated ---------------------------------


async def _direct_conversions_accumulated(
    direct: DirectAPI,
    settings: Settings,
) -> DirectConversionsResult:
    """Sum Conversions column across PROTECTED_CAMPAIGN_IDS for 30 days.

    One ``get_campaign_stats`` per campaign (Direct has no multi-campaign
    report endpoint — v2.1 did the same fan-out). Fail-safe: one campaign
    failure doesn't void the whole signal; we log and move on.
    """
    try:
        now = datetime.now(UTC) + timedelta(hours=3)
        today = now.date().isoformat()
        month_ago = (now - timedelta(days=30)).date().isoformat()
        total = 0
        for campaign_id in settings.PROTECTED_CAMPAIGN_IDS:
            try:
                raw = await direct.get_campaign_stats(campaign_id, month_ago, today)
            except Exception:
                logger.warning(
                    "strategy_gate: conversions fetch(%d) failed; skipping",
                    campaign_id,
                    exc_info=True,
                )
                continue
            tsv = str(raw.get("tsv", ""))
            for _, value in _parse_tsv_column(tsv, "Conversions"):
                total += int(value)
        return DirectConversionsResult(count=total, met=total >= SG_DIRECT_CONV_REQUIRED)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "strategy_gate: _direct_conversions_accumulated failed: %s", exc, exc_info=True
        )
        return DirectConversionsResult(count=0, met=False, reason=str(exc))


# --- state I/O --------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_state() -> dict[str, Any]:
    return {
        "status": "learning",
        "entered_at": _now_iso(),
        "history": [],
        "autopilot_baseline_cpa": None,
    }


def _coerce_state(raw: Any) -> dict[str, Any]:
    """Accept either a dict (psycopg JSONB decode) or a JSON string."""
    if isinstance(raw, dict):
        out = dict(raw)
    else:
        try:
            decoded = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            decoded = None
        out = decoded if isinstance(decoded, dict) else {}
    # Fill missing keys with defaults so downstream code never KeyErrors.
    default = _default_state()
    for key, value in default.items():
        out.setdefault(key, value)
    if out.get("status") not in _VALID_STATUSES:
        logger.warning(
            "strategy_gate: unknown status %r in state, reset to learning",
            out.get("status"),
        )
        return _default_state()
    return out


async def _load_state_for_update(cur: Any) -> dict[str, Any]:
    """Read state with ``SELECT FOR UPDATE`` — cursor must be in an open tx."""
    await cur.execute(
        "SELECT value FROM sda_state WHERE key = %s FOR UPDATE",
        (_STATE_KEY,),
    )
    row = await cur.fetchone()
    if row is None or row[0] is None:
        default = _default_state()
        await cur.execute(
            """
            INSERT INTO sda_state (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO NOTHING
            """,
            (_STATE_KEY, Jsonb(default)),
        )
        return default
    return _coerce_state(row[0])


async def _save_state(cur: Any, state: dict[str, Any]) -> None:
    """UPSERT inside the same transaction as the FOR UPDATE row lock."""
    # Cap history growth.
    history = state.get("history", [])
    if isinstance(history, list) and len(history) > _HISTORY_CAP:
        state = {**state, "history": history[-_HISTORY_CAP:]}
    await cur.execute(
        """
        INSERT INTO sda_state (key, value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value, updated_at = NOW()
        """,
        (_STATE_KEY, Jsonb(state)),
    )


async def _load_state_readonly(pool: AsyncConnectionPool) -> dict[str, Any]:
    """Plain read (no FOR UPDATE) — used by helpers outside the main tx."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT value FROM sda_state WHERE key = %s", (_STATE_KEY,))
            row = await cur.fetchone()
    if row is None or row[0] is None:
        return _default_state()
    return _coerce_state(row[0])


# --- transition logic -------------------------------------------------------


def _decide_transition(
    current: Status,
    signals: dict[str, dict[str, Any]],
    baseline_cpa: float | None,
    current_cpa: float,
) -> tuple[Status, str]:
    """Pure function: (new_status, reason). If no transition, new_status == current."""
    all_met = all(s.get("met") for s in signals.values())
    won_met = signals.get("won_30d", {}).get("met", False)

    if current == "learning" and all_met:
        return "ready_to_switch", "all 4 maturity signals green"
    if current == "ready_to_switch" and not won_met:
        won = signals.get("won_30d", {})
        return (
            "learning",
            f"WON regressed: {won.get('value', 0)} < {won.get('required', SG_WON_REQUIRED)}",
        )
    if current == "auto_pilot":
        if (
            baseline_cpa
            and baseline_cpa > 0
            and current_cpa > baseline_cpa * SG_DEGRADED_CPA_GROWTH
        ):
            return (
                "degraded",
                (
                    f"CPA {int(current_cpa)}₽ > baseline {int(baseline_cpa)}₽ "
                    f"× {SG_DEGRADED_CPA_GROWTH}"
                ),
            )
    # degraded is sticky — no auto-exit.
    return current, ""


def _watchdog_interval_for(status: Status) -> int:
    return (
        SG_WATCHDOG_INTERVAL_AUTOPILOT_MIN
        if status == "auto_pilot"
        else SG_WATCHDOG_INTERVAL_LEARNING_MIN
    )


# --- telegram alert ---------------------------------------------------------


def _format_transition_alert(
    from_status: Status,
    to_status: Status,
    reason: str,
    signals: dict[str, dict[str, Any]],
) -> str:
    def _flag(met: Any) -> str:
        return "✅" if met else "❌"

    won = signals.get("won_30d", {})
    cpa = signals.get("cpa_stability_7d", {})
    offline = signals.get("offline_conversions", {})
    direct = signals.get("direct_conversions", {})

    from_emoji = _STATUS_EMOJI.get(from_status, "•")
    to_emoji = _STATUS_EMOJI.get(to_status, "•")

    severity = "CRITICAL" if to_status == "degraded" else "INFO"
    lines = [
        f"<b>STRATEGY GATE [{severity}]</b>",
        f"{from_emoji} <code>{from_status}</code> → {to_emoji} <code>{to_status}</code>",
        "",
        f"Reason: {reason}",
        "",
        "<b>Signals:</b>",
        f"  {_flag(won.get('met'))} WON 30d: {won.get('value', 0)} / "
        f"{won.get('required', SG_WON_REQUIRED)} (revenue {int(won.get('revenue_rub', 0))}₽)",
        f"  {_flag(cpa.get('met'))} CPA variance 7d: {cpa.get('value', 0):.3f} ≤ "
        f"{SG_CPA_VARIANCE_MAX} ({cpa.get('days', 0)} days, mean={int(cpa.get('mean', 0))}₽)",
        f"  {_flag(offline.get('met'))} Offline conv 7d: {offline.get('value', 0)} runs, "
        f"{offline.get('errors', 0)} errors",
        f"  {_flag(direct.get('met'))} Direct conv 30d: {direct.get('value', 0)} / "
        f"{direct.get('required', SG_DIRECT_CONV_REQUIRED)}",
        "",
    ]
    if to_status == "ready_to_switch":
        lines.append(
            "<i>Action:</i> switch strategy to WB_MAXIMUM_CONVERSION_RATE in Direct UI, "
            "then POST /strategy-gate/switch?to=auto_pilot"
        )
    elif to_status == "auto_pilot":
        lines.append("<i>Action:</i> watchdog interval now 240 min; monitoring for degradation.")
    elif to_status == "degraded":
        lines.append(
            "<i>Action:</i> CPA deteriorated. Consider rolling back to WB_MAX_CLICKS "
            "and POST /strategy-gate/switch?to=learning after investigation."
        )
    elif to_status == "learning":
        lines.append("<i>Action:</i> WON has regressed; waiting for re-accumulation.")
    return "\n".join(lines)


async def _notify_transition(
    http: httpx.AsyncClient | None,
    settings: Settings | None,
    text: str,
) -> bool:
    """Best-effort Telegram alert. Never propagates exception.

    Returns True iff the send completed without raising. A False here does
    NOT block the state transition — the PG row is the source of truth.
    """
    if http is None or settings is None:
        logger.info("strategy_gate: telegram deps not injected; skipping alert")
        return False
    try:
        await telegram_tools.send_message(http, settings, text=text)
        return True
    except Exception:
        logger.warning("strategy_gate: telegram notify failed", exc_info=True)
        return False


# --- public helpers ---------------------------------------------------------


def format_section(state_eval: dict[str, Any]) -> str:
    """HTML block summarising the current gate status (for Reporter digest).

    Consumed by Task 20 (Reporter) to embed strategy-gate state inside the
    daily Telegram digest. Kept format identical to the v2.1 block so the
    Reporter doesn't need branching.
    """
    status = state_eval.get("status", "unknown")
    signals = state_eval.get("signals", {})
    reason = state_eval.get("reason", "") or "—"
    emoji = _STATUS_EMOJI.get(status, "•")
    lines = [
        f"<b>Strategy Gate</b>: {emoji} <code>{status}</code>",
        f"  reason: {reason}",
    ]
    won = signals.get("won_30d", {})
    cpa = signals.get("cpa_stability_7d", {})
    offline = signals.get("offline_conversions", {})
    direct = signals.get("direct_conversions", {})
    lines.append(
        f"  won_30d={won.get('value', 0)}/{won.get('required', SG_WON_REQUIRED)} "
        f"cpa_var={cpa.get('value', 0):.3f} "
        f"offline={offline.get('value', 0)}/{SG_OFFLINE_MIN_RUNS} "
        f"direct_conv={direct.get('value', 0)}/{direct.get('required', SG_DIRECT_CONV_REQUIRED)}"
    )
    baseline = state_eval.get("autopilot_baseline_cpa")
    if baseline is not None:
        lines.append(f"  baseline_cpa={int(baseline)}₽")
    return "\n".join(lines)


# --- manual_switch ----------------------------------------------------------


async def manual_switch(
    pool: AsyncConnectionPool,
    to_status: str,
    *,
    direct: DirectAPI | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Operator-initiated transition; wired to ``POST /strategy-gate/switch``.

    TODO(integration): add FastAPI endpoint ``POST /strategy-gate/switch?to={status}``
    guarded by ``SDA_INTERNAL_API_KEY`` that delegates to this function.

    When ``to_status == 'auto_pilot'`` we also snapshot the current mean
    CPA (from the 7-day stability signal) into ``autopilot_baseline_cpa``
    so the degradation check later has a fixed reference point. If direct
    is unavailable we still accept the switch but record ``None`` — the
    degradation gate is then no-op until the next successful evaluate().
    """
    if to_status not in _VALID_STATUSES:
        return {
            "ok": False,
            "error": f"unknown status: {to_status}",
            "valid": sorted(_VALID_STATUSES),
        }

    # Compute baseline BEFORE opening the transaction so the long-running
    # Direct fetch does not hold the sda_state row lock.
    baseline_cpa: float | None = None
    if to_status == "auto_pilot" and direct is not None and settings is not None:
        try:
            cpa = await _cpa_stability_7d(direct, settings)
            baseline_cpa = cpa.mean if cpa.mean > 0 else None
        except Exception:
            logger.warning("strategy_gate: manual_switch baseline fetch failed", exc_info=True)

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            state = await _load_state_for_update(cur)
            from_status: Status = state["status"]  # type: ignore[assignment]
            history = list(state.get("history") or [])
            history.append(
                {
                    "ts": _now_iso(),
                    "from": from_status,
                    "to": to_status,
                    "reason": "manual_switch",
                    "manual": True,
                }
            )
            new_state: dict[str, Any] = {
                **state,
                "status": to_status,
                "entered_at": _now_iso(),
                "history": history,
            }
            if to_status == "auto_pilot":
                new_state["autopilot_baseline_cpa"] = baseline_cpa
            elif to_status == "learning":
                # Leaving auto_pilot for any reason invalidates the baseline.
                new_state["autopilot_baseline_cpa"] = None
            await _save_state(cur, new_state)

    # Best-effort audit + notify outside the transaction.
    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="n/a",
            tool_name="strategy_gate.manual_switch",
            tool_input={"to": to_status},
            tool_output={
                "from": from_status,
                "baseline_cpa": baseline_cpa,
            },
            is_mutation=True,
        )
    except Exception:
        logger.warning("strategy_gate: audit_log write failed", exc_info=True)

    text = _format_transition_alert(
        from_status,
        to_status,  # type: ignore[arg-type]
        "manual_switch",
        state.get("signals") or {},
    )
    notified = await _notify_transition(http_client, settings, text)

    return {
        "ok": True,
        "from": from_status,
        "to": to_status,
        "autopilot_baseline_cpa": baseline_cpa,
        "notified": notified,
    }


# --- run / evaluate ---------------------------------------------------------


async def _run_impl(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool,
    direct: DirectAPI,
    http_client: httpx.AsyncClient | None,
    settings: Settings,
) -> dict[str, Any]:
    """Collect 4 signals in parallel, decide transition, persist + notify."""
    logger.info("strategy_gate start (dry_run=%s)", dry_run)

    if http_client is None:
        raise RuntimeError("strategy_gate needs http_client for Bitrix / Telegram")

    won_res, cpa_res, offline_res, direct_res = await asyncio.gather(
        _count_won_30d(http_client, settings),
        _cpa_stability_7d(direct, settings),
        _offline_conversions_ok(pool),
        _direct_conversions_accumulated(direct, settings),
        return_exceptions=True,
    )

    # ``return_exceptions=True`` hands the exception back instead of raising.
    # Normalise each into its dataclass with a descriptive reason so the
    # state machine can reason off them uniformly.
    def _normalise(res: Any, kind: str) -> Any:
        if isinstance(res, BaseException):
            logger.warning("strategy_gate: %s raised: %s", kind, res, exc_info=res)
            if kind == "won":
                return WonResult(0, 0.0, False, reason=str(res))
            if kind == "cpa":
                return CpaStabilityResult(False, 0.0, 0.0, 0.0, 0, reason=str(res))
            if kind == "offline":
                return OfflineConversionsResult(0, 0, False, reason=str(res))
            return DirectConversionsResult(0, False, reason=str(res))
        return res

    won = _normalise(won_res, "won")
    cpa = _normalise(cpa_res, "cpa")
    offline = _normalise(offline_res, "offline")
    direct_conv = _normalise(direct_res, "direct")

    signals: dict[str, dict[str, Any]] = {
        "won_30d": won.as_dict(),
        "cpa_stability_7d": cpa.as_dict(),
        "offline_conversions": offline.as_dict(),
        "direct_conversions": direct_conv.as_dict(),
    }
    logger.info(
        "strategy_gate signals: won=%d/%d cpa_stable=%s offline=%d/%d direct_conv=%d/%d",
        won.count,
        SG_WON_REQUIRED,
        cpa.stable,
        offline.total,
        SG_OFFLINE_MIN_RUNS,
        direct_conv.count,
        SG_DIRECT_CONV_REQUIRED,
    )

    # Short-circuit for dry-run: compute signals + what-would-happen but do
    # not open the write transaction or send alerts.
    if dry_run:
        state = await _load_state_readonly(pool)
        current: Status = state["status"]  # type: ignore[assignment]
        baseline_cpa = state.get("autopilot_baseline_cpa")
        proposed, reason = _decide_transition(current, signals, baseline_cpa, cpa.mean)
        return {
            "status": current,
            "proposed_status": proposed,
            "changed": False,
            "dry_run": True,
            "reason": reason,
            "signals": signals,
            "revenue_30d": won.revenue_rub,
            "watchdog_interval_min": _watchdog_interval_for(current),
            "autopilot_baseline_cpa": baseline_cpa,
        }

    # Live write — single transaction holds the state row lock.
    from_status: Status
    to_status: Status
    changed: bool
    reason: str
    new_state_snapshot: dict[str, Any]
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            state = await _load_state_for_update(cur)
            from_status = state["status"]  # type: ignore[assignment]
            baseline_cpa = state.get("autopilot_baseline_cpa")
            to_status, reason = _decide_transition(from_status, signals, baseline_cpa, cpa.mean)
            changed = to_status != from_status
            new_state: dict[str, Any] = {
                **state,
                "status": to_status,
                "signals": signals,
                "last_evaluated_at": _now_iso(),
            }
            if changed:
                history = list(state.get("history") or [])
                history.append(
                    {
                        "ts": _now_iso(),
                        "from": from_status,
                        "to": to_status,
                        "reason": reason,
                        "manual": False,
                    }
                )
                new_state["history"] = history
                new_state["entered_at"] = _now_iso()
                if to_status == "auto_pilot":
                    # Snapshot baseline when the state machine moves itself in
                    # (e.g. if a future ticket lets ready→auto_pilot happen
                    # automatically — today only manual_switch does it).
                    new_state["autopilot_baseline_cpa"] = cpa.mean if cpa.mean > 0 else None
                elif to_status == "learning":
                    new_state["autopilot_baseline_cpa"] = None
            await _save_state(cur, new_state)
            new_state_snapshot = new_state

    notified = False
    if changed:
        alert_text = _format_transition_alert(from_status, to_status, reason, signals)
        notified = await _notify_transition(http_client, settings, alert_text)

    # Audit log is separate tx — not blocking the state transition on telemetry.
    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="n/a",
            tool_name="strategy_gate",
            tool_input={"dry_run": dry_run},
            tool_output={
                "from": from_status,
                "to": to_status,
                "changed": changed,
                "reason": reason,
                "signals": signals,
                "notified": notified,
            },
            is_mutation=changed,
        )
    except Exception:
        logger.warning("strategy_gate: audit_log write failed", exc_info=True)

    logger.info(
        "strategy_gate done: status=%s changed=%s reason=%s notified=%s",
        to_status,
        changed,
        reason,
        notified,
    )

    return {
        "status": to_status,
        "changed": changed,
        "dry_run": False,
        "reason": reason,
        "signals": signals,
        "revenue_30d": won.revenue_rub,
        "watchdog_interval_min": _watchdog_interval_for(to_status),
        "autopilot_baseline_cpa": new_state_snapshot.get("autopilot_baseline_cpa"),
        "notified": notified,
    }


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY-compatible cron entry.

    TODO(integration):
      1. Add ``"strategy_gate": strategy_gate.run`` to
         ``agent_runtime/jobs/__init__.py::JOB_REGISTRY``.
      2. Add Railway Cron in ``railway.toml``: ``schedule = "0 */3 * * *"``
         HTTP-trigger to ``/run/strategy_gate`` with bearer auth.
      3. FastAPI handler ``POST /strategy-gate/switch`` calling
         :func:`manual_switch` with bearer auth (SDA_INTERNAL_API_KEY).
      4. Move the SG_* constants + UTM_CAMPAIGN_WHITELIST to
         :class:`Settings` and wire defaults through Field(default=...).

    When invoked through the default JOB_REGISTRY dispatch (``pool`` +
    ``dry_run`` only — no DI), degrade to a read-only no-op so a cron
    misconfig does not crash the endpoint. The FastAPI ``/run/strategy_gate``
    handler is expected to inject ``direct``, ``http_client``, ``settings``
    from ``app.state`` for real runs.
    """
    if direct is None or settings is None:
        logger.warning("strategy_gate: DI missing (direct/settings) — degraded no-op")
        state = await _load_state_readonly(pool)
        status: Status = state.get("status", "learning")  # type: ignore[assignment]
        return {
            "status": status,
            "changed": False,
            "dry_run": dry_run,
            "reason": "degraded_noop_di_missing",
            "signals": state.get("signals") or {},
            "revenue_30d": 0.0,
            "watchdog_interval_min": _watchdog_interval_for(status),
            "autopilot_baseline_cpa": state.get("autopilot_baseline_cpa"),
            "notified": False,
        }

    try:
        return await _run_impl(
            pool,
            dry_run=dry_run,
            direct=direct,
            http_client=http_client,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("strategy_gate crashed")
        # Best-effort crash alert, swallowed.
        if http_client is not None:
            try:
                await telegram_tools.send_message(
                    http_client,
                    settings,
                    text=f"<b>STRATEGY GATE CRASHED</b>: {type(exc).__name__}: {exc}",
                )
            except Exception:
                logger.exception("strategy_gate: could not deliver crash Telegram alert")
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "changed": False,
            "dry_run": dry_run,
        }


__all__ = [
    "SG_CPA_VARIANCE_MAX",
    "SG_DEGRADED_CPA_GROWTH",
    "SG_DIRECT_CONV_REQUIRED",
    "SG_OFFLINE_DAYS_OK",
    "SG_OFFLINE_MIN_RUNS",
    "SG_WATCHDOG_INTERVAL_AUTOPILOT_MIN",
    "SG_WATCHDOG_INTERVAL_LEARNING_MIN",
    "SG_WON_REQUIRED",
    "UTM_CAMPAIGN_WHITELIST",
    "CpaStabilityResult",
    "DirectConversionsResult",
    "OfflineConversionsResult",
    "WonResult",
    "format_section",
    "manual_switch",
    "run",
]
