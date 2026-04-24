"""Smart Optimizer — Decision Engine heart of SDA v3 (Task 18).

The ONE cron (every 4 h) that closes the agent loop:

1. **Detect** — :class:`agent_runtime.signal_detector.SignalDetector` emits
   typed :class:`~agent_runtime.models.Signal` facts.
2. **Reason** — :func:`agent_runtime.brain.reason` turns signals into at
   most one :class:`~agent_runtime.models.HypothesisDraft` (ReAct loop with
   prompt-injection defence, bounded by ``max_steps=12`` + LLM budget).
3. **Decide per action** — three-stage gate for every action in the draft:
   ``decision_engine.evaluate → trust_levels.allowed_action overlay →
   kill_switches.run_all``. Any kill-switch ``allow=False`` overrides AUTO.
4. **Persist** — single writer. ``decision_journal.record_hypothesis`` is
   the **only** INSERT path into ``hypotheses`` (Decision 3). Weekly budget
   cap is enforced inside that function via ``SELECT FOR UPDATE``.
5. **Execute / notify / ask** — dispatch per final ``AutonomyLevel``:
   AUTO → DirectAPI call; NOTIFY → Telegram; ASK → ``ask_queue`` insert +
   HMAC-signed inline button.
6. **Audit** — every ``(action, decision)`` row into ``audit_log`` via the
   sanctioned :func:`agent_runtime.db.insert_audit_log` (PII sanitiser
   runs there).

Invariants (enforced, grep-auditable):

* ``hypotheses`` single writer — no direct INSERT here.
* ``shadow`` hard invariant — any trust=shadow forces NOTIFY + ``is_mutation=false``
  (belt-and-braces: overlay does the downgrade, we assert before execute).
* ``daily cap`` — per-action ``ACTION_LIMITS`` (from :mod:`decision_engine`)
  counted from today's ``audit_log`` rows with ``is_mutation=true``.
* ``weekly cap`` — handled by ``record_hypothesis`` (returns state=waiting_budget).
* ``dry_run=True`` — runs detect + reason + decide (for logging), no side
  effects. No ``record_hypothesis``, no Direct mutation, no telegram, no
  audit_log INSERT.

Graceful degradation: when the HTTP endpoint passes only ``pool`` + ``dry_run``
(JOB_REGISTRY wrapper, cron smoke), DI for ``direct`` / ``settings`` /
``http_client`` / ``llm_client`` is missing → we return a ``status='ok',
reason='degraded_noop_di_missing'`` short-circuit without touching Direct or
Claude. The FastAPI ``/run/smart_optimizer`` handler injects these from
``app.state`` for real runs.

TODO(integration):
  1. ``JOB_REGISTRY['smart_optimizer'] = smart_optimizer.run`` in
     ``agent_runtime/jobs/__init__.py``.
  2. Railway cron in ``railway.toml``: ``schedule = "0 */4 * * *"``,
     HTTP-triggered to ``/run/smart_optimizer`` with bearer auth.
  3. FastAPI ``POST /run/smart_optimizer`` handler in ``main.py`` — reuse the
     generic ``/run/{job}`` dispatcher but inject ``llm_client``, ``direct``,
     ``metrika``, ``bitrix``, ``http_client``, ``settings``, ``tool_registry``,
     ``reflection_store``, ``signer`` from ``app.state``.
  4. ``SDA_AGENT_CHAT_ID`` into :class:`Settings` if NOTIFY channel differs
     from ``TELEGRAM_CHAT_ID`` (currently we reuse the owner's DM).
  5. ``tool_registry``-based AUTO dispatch (Task 12 agents-core registry).
     Current implementation calls :class:`DirectAPI` methods directly —
     that keeps the wiring honest for Wave 1 where the registry wrapper
     is still a thin shim over the same methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime import decision_engine, decision_journal
from agent_runtime.auth.signing import HMACSigner
from agent_runtime.brain import reason as brain_reason
from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.decision_engine import ACTION_LIMITS, IRREVERSIBILITY
from agent_runtime.models import AutonomyLevel, HypothesisDraft, Signal
from agent_runtime.signal_detector import SignalDetector
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI
from agent_runtime.tools.kill_switches import Action as KSAction
from agent_runtime.tools.kill_switches import KillSwitchContext, KillSwitchResult, run_all
from agent_runtime.trust_levels import TrustLevel, allowed_action, get_trust_level

logger = logging.getLogger(__name__)


_AFFECTED_PCT_DEFAULT = 0.10  # conservative "this action touches 10% of weekly budget"
_MAX_ACTIONS_PER_DRAFT = 10  # brain should cap itself but we double-check


# --- DI contract ------------------------------------------------------------


@dataclass(frozen=True)
class _ActionDecision:
    """One action after the three-stage decider has run."""

    idx: int
    action: dict[str, Any]
    action_type: str
    decision: decision_engine.Decision
    overlay: AutonomyLevel
    kill_switch_results: list[KillSwitchResult]
    final_level: AutonomyLevel
    reject_reason: str | None  # populated when final=FORBIDDEN / cap hit


# --- helpers ----------------------------------------------------------------


def _more_restrictive(a: AutonomyLevel, b: AutonomyLevel) -> AutonomyLevel:
    """Return the more restrictive of two levels (FORBIDDEN > ASK > NOTIFY > AUTO)."""
    order = {
        AutonomyLevel.AUTO: 0,
        AutonomyLevel.NOTIFY: 1,
        AutonomyLevel.ASK: 2,
        AutonomyLevel.FORBIDDEN: 3,
    }
    return a if order[a] >= order[b] else b


def _action_type_of(action: dict[str, Any]) -> str:
    # HypothesisDraft.actions is list[dict]; brain emits {"type": ..., "params": {...}}.
    raw = action.get("type") or action.get("name") or ""
    return str(raw).strip()


async def _count_today_mutations_for_action(pool: AsyncConnectionPool, action_type: str) -> int:
    """Count today's successful mutations of this action type in audit_log."""
    query = """
    SELECT COUNT(*) FROM audit_log
    WHERE ts::date = current_date
      AND is_mutation = true
      AND is_error = false
      AND tool_name = %s
    """
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (action_type,))
                row = await cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        logger.warning(
            "smart_optimizer: daily cap lookup failed for %s; defaulting 0",
            action_type,
            exc_info=True,
        )
        return 0


async def _capture_metrics_before(draft: HypothesisDraft, direct: DirectAPI) -> dict[str, Any]:
    """Collect pre-mutation numbers for impact_tracker.measure_outcome later.

    We fetch campaign-level stats when ``draft.campaign_id`` is set, plus the
    current ``cost_snapshot_today`` consumed by budget_guard's hypothesis
    short-circuit (see Decision 4). ad_group-level stats are not available in
    DirectAPI today (see code-research.md / Task 24) — we best-effort by
    asking for the parent campaign if we can resolve it.
    """
    metrics: dict[str, Any] = {}
    today = datetime.now(UTC).date().isoformat()
    if draft.campaign_id is not None:
        try:
            stats = await direct.get_campaign_stats(
                draft.campaign_id, date_from=today, date_to=today
            )
            metrics["campaign"] = stats
            # Carry today_cost for budget_guard hypothesis-cap short-circuit.
            metrics["cost_snapshot_today"] = float(stats.get("today_cost", 0) or 0)
        except Exception:
            logger.warning(
                "smart_optimizer: metrics_before campaign stats failed id=%s",
                draft.campaign_id,
                exc_info=True,
            )
    if draft.ad_group_id is not None:
        metrics["ad_group_id"] = draft.ad_group_id
    return metrics


async def _send_notify(
    http_client: httpx.AsyncClient | None,
    settings: Settings,
    text: str,
) -> None:
    """Best-effort Telegram NOTIFY. Swallows failures (logged)."""
    if http_client is None:
        logger.info("smart_optimizer: http_client missing, skip NOTIFY")
        return
    try:
        await telegram_tools.send_message(http_client, settings, text=text)
    except Exception:
        logger.warning("smart_optimizer: telegram notify failed", exc_info=True)


async def _insert_ask_queue_row(
    pool: AsyncConnectionPool,
    hypothesis_id: str,
    question: str,
    options: list[str],
) -> int | None:
    """INSERT into ``ask_queue`` and return the row id (or None on failure)."""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO ask_queue (hypothesis_id, question, options)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (hypothesis_id, question, Jsonb(options)),
                )
                row = await cur.fetchone()
        return int(row[0]) if row else None
    except Exception:
        logger.exception("smart_optimizer: ask_queue INSERT failed")
        return None


async def _update_ask_message_id(pool: AsyncConnectionPool, ask_id: int, message_id: int) -> None:
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE ask_queue SET telegram_message_id = %s WHERE id = %s",
                    (message_id, ask_id),
                )
    except Exception:
        logger.warning("smart_optimizer: ask_queue msg_id update failed", exc_info=True)


def _format_notify(hypothesis_id: str, action: dict[str, Any], reason: str) -> str:
    action_type = _action_type_of(action) or "(unknown)"
    return (
        "<b>NOTIFY</b>\n"
        f"hypothesis: <code>{hypothesis_id}</code>\n"
        f"action: <code>{action_type}</code>\n"
        f"reason: {reason}"
    )


def _format_ask(hypothesis_id: str, action: dict[str, Any], reason: str) -> str:
    action_type = _action_type_of(action) or "(unknown)"
    return (
        "<b>ASK</b>\n"
        f"hypothesis: <code>{hypothesis_id}</code>\n"
        f"action: <code>{action_type}</code>\n"
        f"reason: {reason}\n"
        "approve / reject?"
    )


def _format_forbidden(hypothesis_id: str, action: dict[str, Any], reason: str) -> str:
    action_type = _action_type_of(action) or "(unknown)"
    return (
        "<b>FORBIDDEN</b>\n"
        f"hypothesis: <code>{hypothesis_id}</code>\n"
        f"action: <code>{action_type}</code>\n"
        f"reason: {reason}"
    )


# --- core decision pipeline -------------------------------------------------


async def _decide_one_action(
    *,
    idx: int,
    action: dict[str, Any],
    trust_level: TrustLevel,
    signals_count: int,
    pool: AsyncConnectionPool,
    direct: DirectAPI,
    settings: Settings,
) -> _ActionDecision:
    """Three-stage gate: evaluate → trust overlay → kill-switches.

    Returns an :class:`_ActionDecision` describing the final level plus the
    reasons/evidence, so the caller can audit + execute without re-running
    the pipeline.
    """
    action_type = _action_type_of(action)

    # Stage 1 — base decision_engine.
    decision = decision_engine.evaluate(
        action_type,
        affected_budget_pct=_AFFECTED_PCT_DEFAULT,
        data_points=signals_count,
    )

    # Stage 2 — trust overlay (pure function, no I/O).
    overlay = allowed_action(action_type, trust_level, decision.level)

    # Stage 3 — kill switches (fail-closed on any raise).
    ks_action = KSAction.from_dict(action)
    ks_context = KillSwitchContext(
        pool=pool,
        direct=direct,
        metrika=None,
        bitrix=None,
        settings=settings,
        trust_level=trust_level.value,
    )
    ks_results = await run_all(ks_action, ks_context)
    blocking = [r for r in ks_results if not r.allow]

    if blocking:
        reasons = "; ".join(f"{r.switch_name}: {r.reason}" for r in blocking)
        final_level = AutonomyLevel.FORBIDDEN
        reject_reason = f"kill_switch: {reasons}"
    else:
        final_level = overlay
        reject_reason = None

    # Belt-and-braces: if somehow we landed on AUTO under shadow (overlay bug),
    # downgrade to NOTIFY and let shadow_monitor catch the anomaly.
    if trust_level == TrustLevel.SHADOW and final_level == AutonomyLevel.AUTO:
        logger.error(
            "smart_optimizer: shadow_invariant_violation action=%s; forcing NOTIFY",
            action_type,
        )
        final_level = AutonomyLevel.NOTIFY
        reject_reason = "shadow_invariant_violation → NOTIFY"

    # Daily cap from ACTION_LIMITS — applies only to mutating decisions.
    if final_level == AutonomyLevel.AUTO and action_type in ACTION_LIMITS:
        today_count = await _count_today_mutations_for_action(pool, action_type)
        cap = ACTION_LIMITS[action_type]
        if today_count >= cap:
            final_level = AutonomyLevel.NOTIFY
            reject_reason = (
                f"daily_cap_reached: action_type={action_type} today={today_count} cap={cap}"
            )

    logger.info(
        "smart_optimizer: action=%s decision=%s overlay=%s final=%s blockers=%d",
        action_type or "(unknown)",
        decision.level.value,
        overlay.value,
        final_level.value,
        len(blocking),
    )

    return _ActionDecision(
        idx=idx,
        action=action,
        action_type=action_type,
        decision=decision,
        overlay=overlay,
        kill_switch_results=ks_results,
        final_level=final_level,
        reject_reason=reject_reason,
    )


# --- action execution -------------------------------------------------------


async def _execute_auto_action(
    *,
    direct: DirectAPI,
    action: dict[str, Any],
) -> tuple[bool, Any, str | None]:
    """Best-effort dispatch to :class:`DirectAPI` methods.

    Wave 1 shortcut — we bypass the full ``tool_registry`` wrapper from
    agents-core (Task 12) and call a narrow handful of DirectAPI methods
    directly. The registry integration lives behind TODO(integration); the
    three methods covered here (``add_negatives``, ``pause_group``,
    ``set_bid``) are the only ones whitelisted for assisted AUTO anyway.

    Returns ``(success, tool_output, error)``.
    """
    action_type = _action_type_of(action)
    params = dict(action.get("params") or {})
    try:
        if action_type == "add_negatives":
            campaign_id = int(params["campaign_id"])
            phrases = list(params["phrases"])
            out = await direct.add_negatives(campaign_id, phrases)
            # GET-after-SET invariant — verify inside DirectAPI wrapper (Task 7).
            ok = await direct.verify_negatives_added(campaign_id, phrases)
            if not ok:
                return False, out, "verify_negatives_added returned False"
            return True, out, None
        if action_type == "pause_group":
            ad_group_id = int(params["ad_group_id"])
            out = await direct.pause_group(ad_group_id)
            ok = await direct.verify_group_paused(ad_group_id)
            if not ok:
                return False, out, "verify_group_paused returned False"
            return True, out, None
        if action_type == "set_bid":
            keyword_id = int(params["keyword_id"])
            bid_rub = int(params["bid_rub"])
            out = await direct.set_bid(keyword_id, bid_rub)
            ok = await direct.verify_bid(keyword_id, bid_rub)
            if not ok:
                return False, out, "verify_bid returned False"
            return True, out, None
        return False, None, f"unsupported_auto_action_type: {action_type}"
    except Exception as exc:
        logger.exception("smart_optimizer: direct dispatch failed for %s", action_type)
        return False, None, f"{type(exc).__name__}: {exc}"


async def _audit_action(
    *,
    pool: AsyncConnectionPool,
    hypothesis_id: str,
    trust_level: TrustLevel,
    decided: _ActionDecision,
    is_mutation: bool,
    tool_output: Any = None,
    error_detail: str | None = None,
    kill_switch_triggered: str | None = None,
) -> None:
    """One audit_log row per (action, decision). Never raises."""
    try:
        await insert_audit_log(
            pool,
            hypothesis_id=hypothesis_id,
            trust_level=trust_level.value,
            tool_name=decided.action_type or "unknown",
            tool_input=decided.action,
            tool_output=tool_output,
            is_mutation=is_mutation,
            is_error=error_detail is not None,
            error_detail=error_detail,
            kill_switch_triggered=kill_switch_triggered,
        )
    except Exception:
        logger.exception(
            "smart_optimizer: audit_log write failed action=%s final=%s",
            decided.action_type,
            decided.final_level.value,
        )


async def _dispatch_decisions(
    *,
    pool: AsyncConnectionPool,
    hypothesis_id: str,
    decisions: list[_ActionDecision],
    trust_level: TrustLevel,
    direct: DirectAPI,
    http_client: httpx.AsyncClient | None,
    settings: Settings,
    signer: HMACSigner | None,
) -> dict[str, int]:
    """Execute / notify / ask per final_level + audit each one.

    Returns counters for the result dict.
    """
    executed = 0
    notified = 0
    asked = 0
    forbidden = 0
    failed = 0

    for decided in decisions:
        # First blocking kill switch (if any) — propagated to audit_log for
        # reviewers + shadow_monitor.
        ks_triggered = next(
            (r.switch_name for r in decided.kill_switch_results if not r.allow),
            None,
        )
        level = decided.final_level
        if level == AutonomyLevel.AUTO:
            ok, output, error = await _execute_auto_action(direct=direct, action=decided.action)
            if ok:
                executed += 1
                await _audit_action(
                    pool=pool,
                    hypothesis_id=hypothesis_id,
                    trust_level=trust_level,
                    decided=decided,
                    is_mutation=True,
                    tool_output=output,
                )
            else:
                failed += 1
                await _audit_action(
                    pool=pool,
                    hypothesis_id=hypothesis_id,
                    trust_level=trust_level,
                    decided=decided,
                    is_mutation=True,
                    tool_output=output,
                    error_detail=error,
                )
                await _send_notify(
                    http_client,
                    settings,
                    text=(
                        f"<b>CRITICAL</b> action <code>{decided.action_type}</code> failed: {error}"
                    ),
                )
        elif level == AutonomyLevel.NOTIFY:
            reason = decided.reject_reason or decided.decision.reason
            await _send_notify(
                http_client,
                settings,
                text=_format_notify(hypothesis_id, decided.action, reason),
            )
            notified += 1
            await _audit_action(
                pool=pool,
                hypothesis_id=hypothesis_id,
                trust_level=trust_level,
                decided=decided,
                is_mutation=False,
            )
        elif level == AutonomyLevel.ASK:
            reason = decided.reject_reason or decided.decision.reason
            ask_id = await _insert_ask_queue_row(
                pool,
                hypothesis_id,
                question=_format_ask(hypothesis_id, decided.action, reason),
                options=["approve", "reject", "defer_24h"],
            )
            if ask_id is not None and http_client is not None and signer is not None:
                try:
                    # Single button row: approve / reject. Telegram's 64-byte
                    # callback budget is handled by signer.sign_callback.
                    from agent_runtime.tools.telegram import InlineButton

                    msg_id = await telegram_tools.send_with_inline(
                        http_client,
                        settings,
                        text=_format_ask(hypothesis_id, decided.action, reason),
                        buttons=[
                            [
                                InlineButton(text="Approve", action="approve"),
                                InlineButton(text="Reject", action="reject"),
                            ]
                        ],
                        hypothesis_id=hypothesis_id,
                    )
                    await _update_ask_message_id(pool, ask_id, msg_id)
                except Exception:
                    logger.warning("smart_optimizer: ASK inline send failed", exc_info=True)
            asked += 1
            await _audit_action(
                pool=pool,
                hypothesis_id=hypothesis_id,
                trust_level=trust_level,
                decided=decided,
                is_mutation=False,
            )
        else:  # FORBIDDEN
            reason = decided.reject_reason or "forbidden"
            await _send_notify(
                http_client,
                settings,
                text=_format_forbidden(hypothesis_id, decided.action, reason),
            )
            forbidden += 1
            await _audit_action(
                pool=pool,
                hypothesis_id=hypothesis_id,
                trust_level=trust_level,
                decided=decided,
                is_mutation=False,
                kill_switch_triggered=ks_triggered,
            )

    return {
        "executed": executed,
        "notified": notified,
        "asked": asked,
        "forbidden": forbidden,
        "failed": failed,
    }


# --- run --------------------------------------------------------------------


async def _build_signer(settings: Settings) -> HMACSigner | None:
    """HMACSigner for ASK callback_data. Falls back to None in degraded mode."""
    try:
        return HMACSigner(settings.HYPOTHESIS_HMAC_SECRET)
    except Exception:
        logger.warning("smart_optimizer: HMACSigner init failed", exc_info=True)
        return None


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    llm_client: Any = None,
    reflection_store: Any = None,
    tool_registry: Any = None,
    metrika: Any = None,
    bitrix: Any = None,
) -> dict[str, Any]:
    """JOB_REGISTRY-compatible cron entry for the smart optimiser.

    Signature is a superset of ``(pool, *, dry_run=False)`` so the generic
    ``/run/{job}`` dispatcher keeps working; real runs inject the rest.
    """
    # Degraded no-op — used by JOB_REGISTRY wrappers and CI smoke.
    if direct is None or settings is None or llm_client is None or tool_registry is None:
        logger.warning(
            "smart_optimizer: DI missing — degraded no-op "
            "(direct=%s settings=%s llm_client=%s tool_registry=%s)",
            direct is not None,
            settings is not None,
            llm_client is not None,
            tool_registry is not None,
        )
        return {
            "status": "ok",
            "reason": "degraded_noop_di_missing",
            "dry_run": dry_run,
            "signals_count": 0,
            "draft": None,
            "executed_actions": 0,
        }

    try:
        return await _run_impl(
            pool,
            dry_run=dry_run,
            direct=direct,
            http_client=http_client,
            settings=settings,
            llm_client=llm_client,
            reflection_store=reflection_store,
            tool_registry=tool_registry,
            metrika=metrika,
            bitrix=bitrix,
        )
    except Exception as exc:
        logger.exception("smart_optimizer crashed")
        # Best-effort crash alert, swallowed.
        if http_client is not None:
            try:
                await telegram_tools.send_message(
                    http_client,
                    settings,
                    text=(f"<b>SMART OPTIMIZER CRASHED</b>: {type(exc).__name__}: {exc}"),
                )
            except Exception:
                logger.exception("smart_optimizer: could not deliver crash alert")
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "dry_run": dry_run,
            "signals_count": 0,
            "draft": None,
            "executed_actions": 0,
        }


async def _run_impl(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool,
    direct: DirectAPI,
    http_client: httpx.AsyncClient | None,
    settings: Settings,
    llm_client: Any,
    reflection_store: Any,
    tool_registry: Any,
    metrika: Any,
    bitrix: Any,
) -> dict[str, Any]:
    logger.info("smart_optimizer start (dry_run=%s)", dry_run)

    # --- 1. trust level (read first — shadow_monitor could have locked us) ---
    try:
        trust_level = await get_trust_level(pool)
    except Exception:
        logger.warning("smart_optimizer: trust_level lookup failed; default SHADOW", exc_info=True)
        trust_level = TrustLevel.SHADOW
    if trust_level == TrustLevel.FORBIDDEN_LOCK:
        logger.warning("smart_optimizer: trust=FORBIDDEN_LOCK, halting")
        return {
            "status": "halted",
            "reason": "trust_forbidden_lock",
            "trust_level": trust_level.value,
            "dry_run": dry_run,
            "signals_count": 0,
            "draft": None,
            "executed_actions": 0,
        }

    # --- 2. signal detection ---
    if http_client is None:
        logger.warning("smart_optimizer: http_client missing; degraded no-op")
        return {
            "status": "ok",
            "reason": "degraded_noop_http_missing",
            "trust_level": trust_level.value,
            "dry_run": dry_run,
            "signals_count": 0,
            "draft": None,
            "executed_actions": 0,
        }

    detector = SignalDetector(
        pool=pool,
        direct=direct,
        metrika=metrika,
        bitrix=bitrix,
        http=http_client,
        settings=settings,
    )
    try:
        signals: list[Signal] = await detector.detect_all()
    except Exception:
        logger.exception("smart_optimizer: detect_all raised")
        signals = []

    if not signals:
        logger.info("smart_optimizer: no signals, done")
        return {
            "status": "ok",
            "reason": "no_signals",
            "trust_level": trust_level.value,
            "dry_run": dry_run,
            "signals_count": 0,
            "draft": None,
            "executed_actions": 0,
        }

    # --- 3. brain.reason (may return None) ---
    try:
        draft: HypothesisDraft | None = await brain_reason(
            signals,
            context={},
            trust_level=trust_level.value,
            mutations_left=0,  # brain currently uses this only for prompt context
            client=llm_client,
            registry=tool_registry,
            config=settings,
            db_pool=pool,
        )
    except Exception:
        logger.exception("smart_optimizer: brain.reason raised")
        return {
            "status": "error",
            "reason": "brain_raised",
            "trust_level": trust_level.value,
            "dry_run": dry_run,
            "signals_count": len(signals),
            "draft": None,
            "executed_actions": 0,
        }

    if draft is None:
        logger.info("smart_optimizer: brain returned None (no action or injection rejected)")
        return {
            "status": "ok",
            "reason": "brain_no_action",
            "trust_level": trust_level.value,
            "dry_run": dry_run,
            "signals_count": len(signals),
            "draft": None,
            "executed_actions": 0,
        }

    if not draft.actions:
        logger.warning("smart_optimizer: draft has no actions — skipping")
        return {
            "status": "ok",
            "reason": "draft_no_actions",
            "trust_level": trust_level.value,
            "dry_run": dry_run,
            "signals_count": len(signals),
            "draft": draft.model_dump(mode="json"),
            "executed_actions": 0,
        }

    actions = draft.actions[:_MAX_ACTIONS_PER_DRAFT]

    # --- 4. per-action three-stage decider ---
    decisions: list[_ActionDecision] = []
    for idx, action in enumerate(actions):
        decided = await _decide_one_action(
            idx=idx,
            action=action,
            trust_level=trust_level,
            signals_count=len(signals),
            pool=pool,
            direct=direct,
            settings=settings,
        )
        decisions.append(decided)

    # --- 5. dry-run short-circuit ---
    if dry_run:
        logger.info("smart_optimizer: dry_run, skip persist + dispatch")
        return {
            "status": "ok",
            "reason": "dry_run",
            "trust_level": trust_level.value,
            "dry_run": True,
            "signals_count": len(signals),
            "draft": draft.model_dump(mode="json"),
            "decisions": [
                {
                    "action_type": d.action_type,
                    "decision_level": d.decision.level.value,
                    "overlay": d.overlay.value,
                    "final_level": d.final_level.value,
                    "reject_reason": d.reject_reason,
                }
                for d in decisions
            ],
            "executed_actions": 0,
        }

    # --- 6. metrics_before snapshot (read-only) ---
    metrics_before = await _capture_metrics_before(draft, direct)

    # --- 7. record_hypothesis (SINGLE WRITER) ---
    try:
        hypothesis_id = await decision_journal.record_hypothesis(
            pool,
            draft,
            signals,
            metrics_before,
        )
    except Exception as exc:
        logger.exception("smart_optimizer: record_hypothesis failed")
        return {
            "status": "error",
            "reason": f"record_hypothesis_failed: {type(exc).__name__}: {exc}",
            "trust_level": trust_level.value,
            "dry_run": dry_run,
            "signals_count": len(signals),
            "draft": draft.model_dump(mode="json"),
            "executed_actions": 0,
        }

    # --- 8. dispatch (execute / notify / ask) + audit ---
    signer = await _build_signer(settings)
    counters = await _dispatch_decisions(
        pool=pool,
        hypothesis_id=hypothesis_id,
        decisions=decisions,
        trust_level=trust_level,
        direct=direct,
        http_client=http_client,
        settings=settings,
        signer=signer,
    )

    # --- 9. all forbidden → mark hypothesis neutral so it doesn't linger running ---
    if (
        counters["forbidden"] == len(decisions)
        and counters["executed"] == 0
        and counters["notified"] == 0
        and counters["asked"] == 0
    ):
        try:
            await decision_journal.update_outcome(
                pool,
                hypothesis_id,
                outcome="neutral",
                metrics_after={},
                lesson="all actions forbidden by trust+kill_switches",
            )
        except Exception:
            logger.warning("smart_optimizer: update_outcome(neutral) failed", exc_info=True)

    result = {
        "status": "ok",
        "reason": "dispatched",
        "trust_level": trust_level.value,
        "dry_run": False,
        "signals_count": len(signals),
        "draft": draft.model_dump(mode="json"),
        "hypothesis_id": hypothesis_id,
        "decisions": [
            {
                "action_type": d.action_type,
                "decision_level": d.decision.level.value,
                "overlay": d.overlay.value,
                "final_level": d.final_level.value,
                "reject_reason": d.reject_reason,
                "kill_switch_triggered": next(
                    (r.switch_name for r in d.kill_switch_results if not r.allow),
                    None,
                ),
            }
            for d in decisions
        ],
        "executed_actions": counters["executed"],
        "notified_actions": counters["notified"],
        "asked_actions": counters["asked"],
        "forbidden_actions": counters["forbidden"],
        "failed_actions": counters["failed"],
    }
    logger.info(
        "smart_optimizer done signals=%d hypothesis=%s "
        "executed=%d notified=%d asked=%d forbidden=%d failed=%d",
        len(signals),
        hypothesis_id,
        counters["executed"],
        counters["notified"],
        counters["asked"],
        counters["forbidden"],
        counters["failed"],
    )
    return result


__all__ = [
    "IRREVERSIBILITY",  # re-export for test convenience
    "run",
]
