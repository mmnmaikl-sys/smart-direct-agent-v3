"""Bitrix Feedback — CPA attribution via WON deals (Task 22).

Daily 11:00 МСК cron. This is the **single source of truth** for CPA in
SDA v3 (Decision 16): Metrika "conversions" (scroll, phone click, form
submit) are NOT contracts and ARE NOT ground truth. Operationally
confirmed on 24bankrotsttvo.ru — 250 Metrika conv/mo vs 1-5 real
contracts. Therefore CPA here is computed from ``crm.deal.list`` with
``STAGE_ID=C45:WON`` over the last 7 days, grouped by parsed numeric
campaign_id from ``UTM_CAMPAIGN``, divided by ``direct.get_campaign_stats``
cost for the same window.

Two downstream consumers rely on this job's output:

1. **``baseline_at_promote.cpa_real`` for confirmed hypotheses** — so
   :mod:`agent_runtime.regression_watch` (Task 27) can detect CPA regress
    7–30 days after promote.  ``impact_tracker.promote_to_prod`` sets an
   initial baseline (CTR/leads/click-based) at promote time, but contracts
   land with 3–14d lag — this job fills the ``cpa_real`` field once WON
   data is in.
2. **``sda_state[bitrix_feedback_cpa_history]``** JSONB — per-campaign CPA
   ring-buffer (last 14 snapshots) consumed by ``decision_engine`` to
   weight spend decisions against real conversion history.

Additional behaviour: on Monday 11:00-11:30 МСК only, atomically reset
``sda_state.mutations_this_week`` via ``SELECT FOR UPDATE`` (concurrent-
safe with :func:`decision_journal.record_hypothesis`'s bucket bumps and
``impact_tracker_job.release_bucket_and_start_waiting``).

PII barrier: Bitrix deal responses include PHONE / NAME. This module
never logs raw responses, never forwards deal dicts unsanitised — only
parsed numeric campaign_ids and aggregated counts. ``audit_log`` writes
(optional trace) route through :func:`db.insert_audit_log` which runs
``sanitize_audit_payload`` (Decision 13).

Trust overlay: in ``shadow`` alerts go out but no baseline UPDATE runs;
in ``autonomous`` / ``assisted`` the UPDATE runs unless ``dry_run=True``.

Degraded-noop pattern: when JOB_REGISTRY's default dispatcher forwards
only ``(pool, dry_run)`` without DI clients, the job returns
``status='ok' action='degraded_noop'`` — the ``/run/bitrix_feedback``
HTTP handler is responsible for injecting clients from ``app.state``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import bitrix as bitrix_tools
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI
from agent_runtime.trust_levels import TrustLevel, get_trust_level

logger = logging.getLogger(__name__)

_MSK = timezone(timedelta(hours=3))
_WON_STAGE_ID = "C45:WON"
_WON_WINDOW_DAYS = 7
_BASELINE_WINDOW_DAYS = 30
_CPA_HISTORY_KEY = "bitrix_feedback_cpa_history"
_MUTATIONS_KEY = "mutations_this_week"
_CPA_HISTORY_MAX_SNAPSHOTS = 14
_MICRO_TO_RUB = 1_000_000

_CAMPAIGN_ID_RE = re.compile(r"(\d{8,})")
_YANDEX_SOURCES: frozenset[str] = frozenset({"yandex", "yandex_direct", "direct"})

# Alert thresholds. Re-use the BFL-RF env names for symmetry with
# bfl_rf_watchdog (which fires on short-window thresholds); bitrix_feedback
# is the long-window (7d) sanity check for real CPA from contracts.
_CPA_LEAD_ALERT_DEFAULT = 2700.0
_CPA_WON_ALERT_DEFAULT = 55000.0


# --------------------------------------------------------------- thresholds


def _alert_threshold(name: str, default: float) -> float:
    """Read ``BITRIX_FEEDBACK_<name>`` from env with a numeric default."""
    try:
        raw = os.environ.get(f"BITRIX_FEEDBACK_{name}")
        if raw is None:
            return float(default)
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


# --------------------------------------------------------------- parsers


def _extract_campaign_id(utm_campaign: str | None) -> int | None:
    """Extract the 8+ digit numeric campaign_id from a UTM_CAMPAIGN string.

    Accepts ``"708978456"``, ``"yd-708978456-bfl-rf"``,
    ``"direct_campaign_708978456"``. Returns ``None`` for strings with no
    8+ digit run — which is the safe skip-path for organic/direct traffic.
    """
    if not utm_campaign or not isinstance(utm_campaign, str):
        return None
    m = _CAMPAIGN_ID_RE.search(utm_campaign)
    return int(m.group(1)) if m else None


def _is_yandex_source(utm_source: Any) -> bool:
    if not isinstance(utm_source, str):
        return False
    return utm_source.strip().lower() in _YANDEX_SOURCES


# --------------------------------------------------------------- time gating


def _should_reset_mutations(now_msk: datetime) -> bool:
    """True iff ``now_msk`` is Monday in the 11:00-11:30 MSK window.

    The job triggers daily at 11:00 МСК via Railway cron. Reset fires only
    when that tick lands on a Monday; missed weeks (Railway downtime) are
    acceptable — ``Watchdog`` (Task 13) alerts on stale heartbeats.
    """
    if now_msk.weekday() != 0:  # Monday == 0
        return False
    if now_msk.hour != 11:
        return False
    return now_msk.minute < 30


def _now_msk() -> datetime:
    return datetime.now(_MSK)


# --------------------------------------------------------------- Direct stats


def _parse_cost_from_tsv(tsv: str) -> float:
    """Sum ``Cost`` column in micro-rubles across all rows; divide at boundary.

    Mirrors :func:`budget_guard._parse_costs` shape: 'Cost' is the micro-
    ruble integer per the Direct CAMPAIGN_PERFORMANCE_REPORT contract.
    Any row where Cost does not parse as int is skipped (not worth
    crashing the job on a stray total-row).
    """
    total_micro = 0
    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    header_idx = -1
    for idx, line in enumerate(lines):
        if "CampaignId" in line and "Cost" in line:
            header_idx = idx
            break
    if header_idx < 0:
        return 0.0
    header_cols = lines[header_idx].split("\t")
    try:
        cost_i = header_cols.index("Cost")
    except ValueError:
        return 0.0
    for line in lines[header_idx + 1 :]:
        cols = line.split("\t")
        if not cols or cols[0].startswith("Total") or len(cols) <= cost_i:
            continue
        try:
            total_micro += int(cols[cost_i])
        except (ValueError, IndexError):
            continue
    return total_micro / _MICRO_TO_RUB


async def _fetch_spend_per_campaign(
    direct: DirectAPI,
    campaign_ids: list[int],
    *,
    date_from: str,
    date_to: str,
) -> dict[int, float]:
    """Per-campaign 7d cost in rubles. Missing campaign → 0, logged.

    DirectAPI.get_campaign_stats is per-campaign (single-id filter) — one
    call each. Sequential (not gathered) because the report endpoint is
    poll-based and concurrent report building can trip Direct's rate
    limiter.
    """
    out: dict[int, float] = {}
    for cid in campaign_ids:
        try:
            raw = await direct.get_campaign_stats(cid, date_from, date_to)
        except Exception:
            logger.exception("bitrix_feedback: get_campaign_stats(%d) failed, treating as 0₽", cid)
            out[cid] = 0.0
            continue
        cost = _parse_cost_from_tsv(str(raw.get("tsv", "")))
        out[cid] = cost
    return out


# --------------------------------------------------------------- sda_state IO


def _coerce_jsonb(value: Any) -> dict[str, Any]:
    """Normalise sda_state.value to a dict (psycopg vs raw-string shims)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("bitrix_feedback: sda_state value corrupt — using empty dict")
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def _get_sda_state(pool: AsyncConnectionPool, key: str) -> dict[str, Any]:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT value FROM sda_state WHERE key = %s", (key,))
            row = await cur.fetchone()
    if row is None:
        return {}
    return _coerce_jsonb(row[0])


async def _upsert_sda_state(pool: AsyncConnectionPool, key: str, value: dict[str, Any]) -> None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO sda_state (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (key, Jsonb(value)),
            )


async def _reset_mutations_this_week(pool: AsyncConnectionPool, *, now_iso: str) -> bool:
    """Atomic reset via ``SELECT FOR UPDATE``. Returns True iff row was touched.

    Matches the ``record_hypothesis`` concurrent-safe pattern. If the row
    does not exist (first-ever cron tick), INSERT it. If the pre-reset
    value is already ``{amount_rub: 0}`` (another worker got there first),
    we still UPDATE the ``reset_at`` timestamp so the last-reset moment
    is consistent — idempotent from the callers' perspective.
    """
    payload = {"amount_rub": 0, "reset_at": now_iso}
    async with pool.connection() as conn, conn.transaction():
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT value FROM sda_state WHERE key = %s FOR UPDATE",
                (_MUTATIONS_KEY,),
            )
            row = await cur.fetchone()
            if row is None:
                await cur.execute(
                    "INSERT INTO sda_state (key, value, updated_at) VALUES (%s, %s, NOW())",
                    (_MUTATIONS_KEY, Jsonb(payload)),
                )
            else:
                await cur.execute(
                    "UPDATE sda_state SET value = %s, updated_at = NOW() WHERE key = %s",
                    (Jsonb(payload), _MUTATIONS_KEY),
                )
    return True


# --------------------------------------------------------------- baseline update


@dataclass(frozen=True)
class _ConfirmedRow:
    id: str
    campaign_id: int
    baseline_at_promote: dict[str, Any]


async def _fetch_confirmed_hypotheses(
    pool: AsyncConnectionPool, campaign_ids: list[int]
) -> list[_ConfirmedRow]:
    """Confirmed hypotheses promoted within the last 30d for given campaigns."""
    if not campaign_ids:
        return []
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, campaign_id, baseline_at_promote
                FROM hypotheses
                WHERE state = 'confirmed'
                  AND promoted_at IS NOT NULL
                  AND promoted_at > NOW() - make_interval(days => %s)
                  AND campaign_id = ANY(%s)
                """,
                (_BASELINE_WINDOW_DAYS, campaign_ids),
            )
            rows = await cur.fetchall()
    return [
        _ConfirmedRow(
            id=str(r[0]),
            campaign_id=int(r[1]),
            baseline_at_promote=_coerce_jsonb(r[2]),
        )
        for r in rows
    ]


async def _update_baseline(
    pool: AsyncConnectionPool, hypothesis_id: str, new_baseline: dict[str, Any]
) -> bool:
    """Idempotent JSONB replacement on ``baseline_at_promote``.

    Guarded by ``state='confirmed'`` so a concurrently-rolled-back row is
    not silently annotated. Returns True iff a row was updated. This
    function duplicates the update_baseline CRUD expected in
    ``decision_journal.py`` (Task 22 also wants it added there); the
    canonical location will be decision_journal once the rest of the PR
    lands — see TODO(integration) in the module docstring.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE hypotheses
                SET baseline_at_promote = %s
                WHERE id = %s AND state = 'confirmed'
                """,
                (Jsonb(new_baseline), hypothesis_id),
            )
            affected = cur.rowcount
    if affected == 0:
        logger.warning(
            "bitrix_feedback: update_baseline(%s) no rows — state drifted, skip",
            hypothesis_id,
        )
        return False
    return True


# --------------------------------------------------------------- cpa history


def _merge_cpa_history(
    history: dict[str, Any],
    cpa_per_campaign: dict[int, float],
    *,
    captured_at: str,
    spend_per_campaign: dict[int, float],
    won_per_campaign: dict[int, int],
) -> dict[str, Any]:
    """Append a snapshot per campaign, cap ring-buffer at 14 entries.

    Layout::

        {
          "campaigns": {
            "708978456": {
              "snapshots": [
                {"ts": "...", "cpa": 1500.0, "spend_rub": 3000.0, "won": 2},
                ...
              ]
            }
          },
          "last_updated": "..."
        }
    """
    campaigns_raw = history.get("campaigns")
    campaigns: dict[str, Any] = campaigns_raw if isinstance(campaigns_raw, dict) else {}

    for cid, cpa in cpa_per_campaign.items():
        key = str(cid)
        existing = campaigns.get(key) or {}
        snapshots_raw = existing.get("snapshots")
        snapshots: list[dict[str, Any]] = (
            list(snapshots_raw) if isinstance(snapshots_raw, list) else []
        )
        snapshots.append(
            {
                "ts": captured_at,
                "cpa": float(cpa),
                "spend_rub": float(spend_per_campaign.get(cid, 0.0)),
                "won": int(won_per_campaign.get(cid, 0)),
            }
        )
        # Ring-buffer cap.
        if len(snapshots) > _CPA_HISTORY_MAX_SNAPSHOTS:
            snapshots = snapshots[-_CPA_HISTORY_MAX_SNAPSHOTS:]
        campaigns[key] = {"snapshots": snapshots}

    return {"campaigns": campaigns, "last_updated": captured_at}


# --------------------------------------------------------------- alerting


def _format_cpa_alert(
    cid: int,
    cpa: float,
    won: int,
    spend: float,
    *,
    kind: str,
    threshold: float,
    trust: TrustLevel,
) -> str:
    """Build a Telegram HTML alert. No raw deal data — only aggregates."""
    title = "CPA-won" if kind == "won" else "CPA-lead"
    return (
        f"<b>⚠️ {title} breach</b>\n"
        f"Campaign <code>{cid}</code>\n"
        f"CPA: <b>{cpa:.0f}₽</b> (threshold {threshold:.0f}₽)\n"
        f"Won: {won}, spend 7d: {spend:.0f}₽\n"
        f"Trust: <code>{trust.value}</code>"
    )


# --------------------------------------------------------------- grouping


def _group_won_deals(deals: list[dict[str, Any]]) -> tuple[dict[int, int], int, int]:
    """Return (won_by_campaign, total_won_kept, skipped_count).

    Skips deals without a parseable UTM_CAMPAIGN or with non-yandex
    UTM_SOURCE. Logs aggregates only — no raw deal fields.
    """
    won_by_campaign: dict[int, int] = {}
    kept = 0
    skipped = 0
    for deal in deals:
        if not isinstance(deal, dict):
            skipped += 1
            continue
        if not _is_yandex_source(deal.get("UTM_SOURCE")):
            skipped += 1
            continue
        cid = _extract_campaign_id(deal.get("UTM_CAMPAIGN"))
        if cid is None:
            skipped += 1
            continue
        won_by_campaign[cid] = won_by_campaign.get(cid, 0) + 1
        kept += 1
    return won_by_campaign, kept, skipped


# --------------------------------------------------------------- run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    bitrix_client: httpx.AsyncClient | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Daily 11:00 MSK entry. JOB_REGISTRY-compatible.

    Extra kwargs default to ``None`` so the minimal ``(pool, dry_run)``
    dispatch path (:mod:`agent_runtime.jobs.__init__.dispatch_job`) does
    not fail. The real /run/bitrix_feedback handler injects DI clients
    from ``app.state``.

    ``bitrix_client`` is the shared ``httpx.AsyncClient`` used with
    :mod:`agent_runtime.tools.bitrix`; ``http_client`` is a separate
    hook for Telegram alerts (same pattern as form_checker / watchdog).
    Both default to None; when either is absent we degrade to noop.
    """
    now_msk = _now_msk()
    now_iso = now_msk.isoformat()
    period_to = now_msk
    period_from = now_msk - timedelta(days=_WON_WINDOW_DAYS)

    trust = await _safe_get_trust_level(pool)

    if direct is None or bitrix_client is None or settings is None:
        logger.warning(
            "bitrix_feedback: DI missing (direct=%s bitrix=%s settings=%s) — degraded no-op",
            direct is not None,
            bitrix_client is not None,
            settings is not None,
        )
        return {
            "status": "ok",
            "action": "degraded_noop",
            "trust_level": trust.value,
            "cpa_per_campaign": {},
            "hypotheses_updated": [],
            "mutations_reset": False,
            "won_deals_total": 0,
            "spend_total": 0.0,
            "checked_at": now_iso,
        }

    logger.info(
        "bitrix_feedback start: period=%s..%s trust=%s dry_run=%s",
        period_from.date().isoformat(),
        period_to.date().isoformat(),
        trust.value,
        dry_run,
    )

    # Step 1 — WON deals last 7d.
    try:
        deals = await bitrix_tools.get_deal_list(
            bitrix_client,
            settings,
            filter={
                "STAGE_ID": _WON_STAGE_ID,
                ">=DATE_MODIFY": period_from.isoformat(),
            },
            select=["ID", "UTM_CAMPAIGN", "UTM_SOURCE", "OPPORTUNITY", "DATE_MODIFY"],
        )
    except Exception as exc:
        logger.exception("bitrix_feedback: get_deal_list failed")
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "trust_level": trust.value,
            "cpa_per_campaign": {},
            "hypotheses_updated": [],
            "mutations_reset": False,
            "won_deals_total": 0,
            "spend_total": 0.0,
            "checked_at": now_iso,
        }

    # Step 2 — group by numeric campaign_id.
    won_by_campaign, kept, skipped = _group_won_deals(deals)
    logger.info(
        "bitrix_feedback: deals fetched=%d kept=%d skipped=%d campaigns=%d",
        len(deals),
        kept,
        skipped,
        len(won_by_campaign),
    )

    # Step 3 — Direct spend for same 7d window.
    campaign_ids = list(won_by_campaign.keys())
    if campaign_ids:
        try:
            spend_per_campaign = await _fetch_spend_per_campaign(
                direct,
                campaign_ids,
                date_from=period_from.date().isoformat(),
                date_to=period_to.date().isoformat(),
            )
        except Exception as exc:
            logger.exception("bitrix_feedback: direct spend fetch failed")
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "trust_level": trust.value,
                "cpa_per_campaign": {},
                "hypotheses_updated": [],
                "mutations_reset": False,
                "won_deals_total": kept,
                "spend_total": 0.0,
                "checked_at": now_iso,
            }
    else:
        spend_per_campaign = {}

    # Step 4 — CPA where won > 0 (safe-divide).
    cpa_per_campaign: dict[int, float] = {}
    for cid, won in won_by_campaign.items():
        if won <= 0:
            continue
        cost = spend_per_campaign.get(cid, 0.0)
        cpa_per_campaign[cid] = cost / won

    spend_total = float(sum(spend_per_campaign.values()))

    # Step 5 — update baseline_at_promote (only non-shadow, non-dry_run).
    hypotheses_updated: list[str] = []
    mutate_allowed = (
        not dry_run and trust != TrustLevel.SHADOW and trust != TrustLevel.FORBIDDEN_LOCK
    )
    if mutate_allowed and cpa_per_campaign:
        try:
            confirmed = await _fetch_confirmed_hypotheses(pool, list(cpa_per_campaign.keys()))
            for row in confirmed:
                cpa = cpa_per_campaign.get(row.campaign_id)
                if cpa is None:
                    continue
                merged = {
                    **row.baseline_at_promote,
                    "cpa_real": float(cpa),
                    "cpa_source": "bitrix_won_7d",
                    "cpa_captured_at": now_iso,
                }
                ok = await _update_baseline(pool, row.id, merged)
                if ok:
                    hypotheses_updated.append(row.id)
        except Exception:
            logger.exception("bitrix_feedback: baseline update pass failed (partial)")

    # Step 6 — persist CPA history for decision_engine (non-shadow).
    if mutate_allowed and cpa_per_campaign:
        try:
            history = await _get_sda_state(pool, _CPA_HISTORY_KEY)
            merged_history = _merge_cpa_history(
                history,
                cpa_per_campaign,
                captured_at=now_iso,
                spend_per_campaign=spend_per_campaign,
                won_per_campaign=won_by_campaign,
            )
            await _upsert_sda_state(pool, _CPA_HISTORY_KEY, merged_history)
        except Exception:
            logger.exception("bitrix_feedback: cpa_history upsert failed")

    # Step 7 — weekly reset if Monday 11:00-11:30 MSK.
    mutations_reset = False
    if mutate_allowed and _should_reset_mutations(now_msk):
        try:
            mutations_reset = await _reset_mutations_this_week(pool, now_iso=now_iso)
        except Exception:
            logger.exception("bitrix_feedback: mutations_this_week reset failed")

    # Step 8 — threshold alerts (CPA breaches). dry_run suppresses ALL side
    # effects including Telegram. Both CPA-lead and CPA-won thresholds trip
    # independently; CPA-won is the one that matters day-to-day (contracts),
    # CPA-lead is kept as a second gate for early warning when no contract
    # data has landed yet but leads are already disproportionately expensive.
    cpa_lead_threshold = _alert_threshold("CPA_LEAD", _CPA_LEAD_ALERT_DEFAULT)
    cpa_won_threshold = _alert_threshold("CPA_WON", _CPA_WON_ALERT_DEFAULT)
    alerts_sent: list[dict[str, Any]] = []
    if not dry_run and http_client is not None:
        for cid, cpa in cpa_per_campaign.items():
            breach_kind: str | None = None
            threshold_used: float = 0.0
            if cpa > cpa_won_threshold:
                breach_kind = "won"
                threshold_used = cpa_won_threshold
            elif cpa > cpa_lead_threshold:
                breach_kind = "lead"
                threshold_used = cpa_lead_threshold
            if breach_kind is None:
                continue
            try:
                await telegram_tools.send_message(
                    http_client,
                    settings,
                    text=_format_cpa_alert(
                        cid,
                        cpa,
                        won_by_campaign.get(cid, 0),
                        spend_per_campaign.get(cid, 0.0),
                        kind=breach_kind,
                        threshold=threshold_used,
                        trust=trust,
                    ),
                )
                alerts_sent.append({"campaign_id": cid, "kind": breach_kind, "cpa": cpa})
            except Exception:
                logger.exception("bitrix_feedback: telegram alert failed for %d", cid)

    # Step 9 — audit_log trace. Never pass raw deals — sanitiser cannot
    # scrub what it cannot see through Bitrix-shaped keys, so we log only
    # numeric aggregates.
    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level=trust.value,
            tool_name="bitrix_feedback",
            tool_input={
                "period_from": period_from.isoformat(),
                "period_to": period_to.isoformat(),
                "deals_fetched": len(deals),
                "deals_kept": kept,
                "deals_skipped": skipped,
            },
            tool_output={
                "cpa_per_campaign": {str(k): v for k, v in cpa_per_campaign.items()},
                "hypotheses_updated": hypotheses_updated,
                "mutations_reset": mutations_reset,
                "spend_total": spend_total,
                "dry_run": dry_run,
            },
            is_mutation=bool(hypotheses_updated) or mutations_reset,
        )
    except Exception:
        logger.exception("bitrix_feedback: audit_log write failed")

    logger.info(
        "bitrix_feedback done: campaigns=%d hypotheses_updated=%d spend=%.0f reset=%s",
        len(cpa_per_campaign),
        len(hypotheses_updated),
        spend_total,
        mutations_reset,
    )

    return {
        "status": "ok",
        "trust_level": trust.value,
        "cpa_per_campaign": {str(k): v for k, v in cpa_per_campaign.items()},
        "hypotheses_updated": hypotheses_updated,
        "mutations_reset": mutations_reset,
        "won_deals_total": kept,
        "spend_total": spend_total,
        "alerts_sent": alerts_sent,
        "dry_run": dry_run,
        "checked_at": now_iso,
    }


async def _safe_get_trust_level(pool: AsyncConnectionPool) -> TrustLevel:
    try:
        return await get_trust_level(pool)
    except Exception:
        logger.warning(
            "bitrix_feedback: trust_level lookup failed, defaulting shadow",
            exc_info=True,
        )
        return TrustLevel.SHADOW


__all__ = [
    "run",
]
