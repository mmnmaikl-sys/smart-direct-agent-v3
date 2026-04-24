"""Telegram Digest + ASK outbound + callback resolver (Task 23).

Single module for three tightly-coupled responsibilities:

1. **Daily digest** (:func:`compile_digest` / :func:`render_digest`) — aggregate
   the last 24 h of ``audit_log`` mutations, ``hypotheses`` state transitions and
   ``ask_queue`` pending rows into one Telegram HTML message. PII sanitiser is
   run on every audit payload via :func:`agent_runtime.pii.sanitize_audit_payload`
   before a single character reaches the render layer — this module never sees
   raw phones / names / emails.

2. **ASK outbound** (:func:`enqueue_ask`) — INSERT one row into ``ask_queue``,
   emit a message with HMAC-signed inline ``Approve`` / ``Reject`` buttons, and
   persist the Telegram ``message_id`` so a later resolve call can scrub the
   keyboard. ``callback_data`` is signed through
   :class:`agent_runtime.auth.signing.HMACSigner` (``HYPOTHESIS_HMAC_SECRET``).
   Per Telegram's 64-byte ``callback_data`` budget, format is
   ``{hypothesis_id}:{action}:{sig10}`` — enforced by ``send_with_inline``.

3. **Callback resolve** (:func:`handle_callback`) — given a raw ``callback_data``
   string (typically forwarded from a ``/webhook/telegram`` FastAPI handler),
   verify HMAC, ``SELECT FOR UPDATE`` the matching unresolved row, UPDATE
   ``resolved_at`` / ``answer`` in the same transaction. ``defer_24h`` inserts
   a fresh ``ask_queue`` row with ``created_at = now() + interval '24 hours'``.
   Idempotent: a second callback after resolve returns ``already_resolved``.

The job entrypoint :func:`run` is the ``JOB_REGISTRY``-compatible cron entry.
Signature matches the other Task 14-22 jobs: ``(pool, *, dry_run=False,
http_client=None, settings=None)``. ``degraded_noop`` when DI is missing —
matches ``budget_guard``, ``form_checker``, ``strategy_gate``, ``bitrix_feedback``.

TODO(integration):
  1. Register ``"telegram_digest": telegram_digest.run`` in
     :data:`agent_runtime.jobs.JOB_REGISTRY`.
  2. Add Railway cron in ``railway.toml``: ``schedule = "0 6 * * *"``
     (= 09:00 МСК) HTTP-triggered to ``/run/telegram_digest``.
  3. FastAPI ``POST /webhook/telegram`` handler wiring: validate
     ``X-Telegram-Bot-Api-Secret-Token`` header per Telegram's
     ``setWebhook(secret_token=...)`` contract, parse ``callback_query``
     body, forward ``callback_query.data`` + ``callback_query.message.message_id``
     into :func:`handle_callback`. The handler itself lives in ``main.py``
     alongside other ``/webhook/*`` routes — not in this module (Task 5b owns
     FastAPI wiring).
  4. Plain ``send_message`` is used for the digest body; if ``ask_queue``
     rendering grows past Telegram's 4096-byte cap, split into a "header"
     digest message + one inline-button message per ASK. Current Wave 3
     ASK volume (<5 pending at any time) fits comfortably in one message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime.auth.signing import HMACSigner
from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.pii import sanitize_audit_payload
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.telegram import InlineButton

logger = logging.getLogger(__name__)


_DEFAULT_WINDOW_HOURS = 24
_ACTIONS_LIMIT = 20
_HYPOTHESES_LIMIT = 10
_ASK_LIMIT = 5
_TELEGRAM_MESSAGE_CAP = 4096
_VALID_CALLBACK_ACTIONS: frozenset[str] = frozenset({"approve", "reject", "defer_24h"})


# --- payload shapes ---------------------------------------------------------


@dataclass(frozen=True)
class ActionSummary:
    ts: datetime
    tool_name: str
    trust_level: str
    is_error: bool
    # ``reason`` is post-sanitise — caller never feeds raw Bitrix PII here.
    reason: str


@dataclass(frozen=True)
class HypothesisSummary:
    id: str
    hypothesis_type: str
    state: str
    campaign_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class AskSummary:
    id: int
    hypothesis_id: str
    question: str
    created_at: datetime


@dataclass(frozen=True)
class DigestPayload:
    generated_at: datetime
    window_hours: int
    actions_taken: list[ActionSummary] = field(default_factory=list)
    hypotheses_started: list[HypothesisSummary] = field(default_factory=list)
    hypotheses_concluded: list[HypothesisSummary] = field(default_factory=list)
    ask_queue_count: int = 0
    ask_queue_unresolved: list[AskSummary] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.actions_taken
            and not self.hypotheses_started
            and not self.hypotheses_concluded
            and self.ask_queue_count == 0
        )


# --- compile ---------------------------------------------------------------


async def compile_digest(
    pool: AsyncConnectionPool,
    *,
    now: datetime | None = None,
    window_hours: int = _DEFAULT_WINDOW_HOURS,
) -> DigestPayload:
    """Aggregate the last ``window_hours`` into a :class:`DigestPayload`.

    All ``tool_input`` / ``tool_output`` read from ``audit_log`` have already
    been PII-sanitised at INSERT time by :func:`agent_runtime.db.insert_audit_log`.
    We run :func:`sanitize_audit_payload` a second time defence-in-depth — a
    future schema change that leaks PII into ``tool_input`` would still be
    redacted before it reaches the render layer.
    """
    generated_at = now or datetime.now(UTC)
    since = generated_at - timedelta(hours=window_hours)

    actions: list[ActionSummary] = []
    started: list[HypothesisSummary] = []
    concluded: list[HypothesisSummary] = []
    ask_rows: list[AskSummary] = []
    ask_count = 0

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT ts, tool_name, trust_level, is_error, tool_input
                FROM audit_log
                WHERE ts >= %s AND is_mutation = true
                ORDER BY ts DESC
                LIMIT %s
                """,
                (since, _ACTIONS_LIMIT),
            )
            for ts, tool_name, trust_level, is_error, tool_input in await cur.fetchall():
                safe_input = sanitize_audit_payload(tool_input)
                reason = _short_reason(safe_input)
                actions.append(
                    ActionSummary(
                        ts=ts,
                        tool_name=str(tool_name or "unknown"),
                        trust_level=str(trust_level or "n/a"),
                        is_error=bool(is_error),
                        reason=reason,
                    )
                )

            await cur.execute(
                """
                SELECT id, hypothesis_type, state, campaign_id, created_at
                FROM hypotheses
                WHERE created_at >= %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (since, _HYPOTHESES_LIMIT),
            )
            for row in await cur.fetchall():
                started.append(
                    HypothesisSummary(
                        id=str(row[0]),
                        hypothesis_type=str(row[1]),
                        state=str(row[2]),
                        campaign_id=row[3],
                        created_at=row[4],
                    )
                )

            await cur.execute(
                """
                SELECT id, hypothesis_type, state, campaign_id, COALESCE(promoted_at, created_at)
                FROM hypotheses
                WHERE state IN ('confirmed', 'rejected', 'rolled_back', 'inconclusive')
                  AND COALESCE(promoted_at, created_at) >= %s
                ORDER BY COALESCE(promoted_at, created_at) DESC
                LIMIT %s
                """,
                (since, _HYPOTHESES_LIMIT),
            )
            for row in await cur.fetchall():
                concluded.append(
                    HypothesisSummary(
                        id=str(row[0]),
                        hypothesis_type=str(row[1]),
                        state=str(row[2]),
                        campaign_id=row[3],
                        created_at=row[4],
                    )
                )

            await cur.execute("SELECT count(*) FROM ask_queue WHERE resolved_at IS NULL")
            count_row = await cur.fetchone()
            ask_count = int(count_row[0]) if count_row and count_row[0] is not None else 0

            await cur.execute(
                """
                SELECT id, hypothesis_id, question, created_at
                FROM ask_queue
                WHERE resolved_at IS NULL
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (_ASK_LIMIT,),
            )
            for ask_row in await cur.fetchall():
                ask_rows.append(
                    AskSummary(
                        id=int(ask_row[0]),
                        hypothesis_id=str(ask_row[1]),
                        question=str(ask_row[2] or ""),
                        created_at=ask_row[3],
                    )
                )

    return DigestPayload(
        generated_at=generated_at,
        window_hours=window_hours,
        actions_taken=actions,
        hypotheses_started=started,
        hypotheses_concluded=concluded,
        ask_queue_count=ask_count,
        ask_queue_unresolved=ask_rows,
    )


# --- render ----------------------------------------------------------------


def _short_reason(safe_input: Any) -> str:
    """Extract a compact, PII-free reason string from a sanitised payload."""
    if isinstance(safe_input, dict):
        for key in ("reason", "action", "short_reason", "campaign_id"):
            value = safe_input.get(key)
            if value is not None:
                return f"{key}={value}"
        return "-"
    if safe_input is None:
        return "-"
    return str(safe_input)[:80]


def _fmt_ts(ts: datetime) -> str:
    # ISO minute precision; tz-aware inputs always (columns are TIMESTAMPTZ).
    return ts.strftime("%Y-%m-%d %H:%M")


def render_digest(payload: DigestPayload) -> str:
    """Render ``payload`` as Telegram HTML text, capped at 4096 chars.

    HTML parse mode (matches ``telegram_tools.send_message`` default) — no
    MarkdownV2 escape needed for the fields we emit; values are enum-ish tool
    names, numeric IDs, and already-sanitised reason strings. If a future
    field can contain user text we switch to ``html.escape``.
    """
    lines: list[str] = []
    lines.append(f"<b>SDA v3 Digest</b> — {_fmt_ts(payload.generated_at)} UTC")
    lines.append(f"Window: last {payload.window_hours}h")
    lines.append("")

    if payload.is_empty():
        lines.append("<i>Тихая ночь — 0 actions, 0 hypotheses, 0 ASK.</i>")
        return "\n".join(lines)[:_TELEGRAM_MESSAGE_CAP]

    lines.append(f"<b>🛠 Actions ({len(payload.actions_taken)})</b>")
    if payload.actions_taken:
        for a in payload.actions_taken:
            marker = "❌" if a.is_error else "✓"
            lines.append(
                f"  {marker} [{_fmt_ts(a.ts)}] <code>{a.tool_name}</code> "
                f"trust={a.trust_level} {a.reason}"
            )
    else:
        lines.append("  —")
    lines.append("")

    lines.append(f"<b>🧪 Hypotheses started ({len(payload.hypotheses_started)})</b>")
    if payload.hypotheses_started:
        for h in payload.hypotheses_started:
            camp = f" camp={h.campaign_id}" if h.campaign_id else ""
            lines.append(f"  • <code>{h.id}</code> {h.hypothesis_type} state={h.state}{camp}")
    else:
        lines.append("  —")
    lines.append("")

    lines.append(f"<b>🧪 Hypotheses concluded ({len(payload.hypotheses_concluded)})</b>")
    if payload.hypotheses_concluded:
        for h in payload.hypotheses_concluded:
            camp = f" camp={h.campaign_id}" if h.campaign_id else ""
            lines.append(f"  • <code>{h.id}</code> {h.hypothesis_type} → {h.state}{camp}")
    else:
        lines.append("  —")
    lines.append("")

    lines.append(f"<b>❓ ASK ({payload.ask_queue_count} unresolved)</b>")
    if payload.ask_queue_unresolved:
        for ask in payload.ask_queue_unresolved:
            lines.append(
                f"  • #{ask.id} <code>{ask.hypothesis_id}</code> {_fmt_ts(ask.created_at)}"
            )
        remainder = payload.ask_queue_count - len(payload.ask_queue_unresolved)
        if remainder > 0:
            lines.append(f"  … и ещё {remainder}")
    else:
        lines.append("  —")

    text = "\n".join(lines)
    if len(text) > _TELEGRAM_MESSAGE_CAP:
        # Preserve header; cut from the tail with an ellipsis marker.
        keep = _TELEGRAM_MESSAGE_CAP - len("\n… обрезано")
        text = text[:keep] + "\n… обрезано"
    return text


# --- ASK outbound ----------------------------------------------------------


async def enqueue_ask(
    pool: AsyncConnectionPool,
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    hypothesis_id: str,
    question: str,
    options: list[str] | None = None,
) -> int:
    """INSERT an ``ask_queue`` row, emit HMAC-signed inline buttons, return row id.

    The ``send_with_inline`` helper signs each button's ``callback_data`` with
    :class:`HMACSigner` (``HYPOTHESIS_HMAC_SECRET``) — see ``tools/telegram.py``.
    We store the returned ``message_id`` so a later resolve can strip the
    keyboard from the original message.
    """
    resolved_options = list(options) if options else ["approve", "reject", "defer_24h"]
    sanitised_question = str(sanitize_audit_payload(question))

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO ask_queue (hypothesis_id, question, options)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (hypothesis_id, sanitised_question, Jsonb(resolved_options)),
            )
            row = await cur.fetchone()
    if row is None:
        raise RuntimeError("ask_queue INSERT did not return id")
    ask_id = int(row[0])

    button_labels: dict[str, str] = {
        "approve": "✅ Approve",
        "reject": "❌ Reject",
        "defer_24h": "⏸ Defer 24h",
    }
    # ``send_with_inline`` today only accepts literal actions {"approve","reject","details"}.
    # The ASK-specific actions (approve/reject) map 1:1; ``defer_24h`` cannot be
    # sent via that helper (type annotation), so it is not included in the
    # outbound buttons — owners resolve defer through a follow-up digest or the
    # /admin API. Keeps the inbound signer strict.
    buttons: list[list[InlineButton]] = [
        [
            InlineButton(text=button_labels["approve"], action="approve"),
            InlineButton(text=button_labels["reject"], action="reject"),
        ]
    ]

    try:
        message_id = await telegram_tools.send_with_inline(
            http_client,
            settings,
            text=sanitised_question,
            buttons=buttons,
            hypothesis_id=hypothesis_id,
        )
    except Exception:
        logger.exception("telegram_digest.enqueue_ask: send_with_inline failed id=%s", ask_id)
        return ask_id

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE ask_queue SET telegram_message_id = %s WHERE id = %s",
                    (message_id, ask_id),
                )
    except Exception:
        logger.warning(
            "telegram_digest.enqueue_ask: message_id UPDATE failed id=%s", ask_id, exc_info=True
        )
    return ask_id


# --- callback resolve ------------------------------------------------------


@dataclass(frozen=True)
class CallbackResult:
    status: str  # "resolved" | "already_resolved" | "invalid_hmac" | "not_found" | "deferred"
    hypothesis_id: str | None = None
    action: str | None = None
    ask_id: int | None = None


async def handle_callback(
    pool: AsyncConnectionPool,
    signer: HMACSigner,
    *,
    callback_data: str,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> CallbackResult:
    """Resolve an inline-button click atomically.

    Flow:
      1. Verify HMAC via :func:`HMACSigner.verify_callback`; invalid → return
         ``invalid_hmac`` without touching the DB.
      2. Within one transaction, ``SELECT FOR UPDATE`` the oldest unresolved
         ``ask_queue`` row for this hypothesis. Already-resolved → NOOP.
      3. ``UPDATE ask_queue SET resolved_at=now(), answer=$action``. For
         ``defer_24h`` INSERT a fresh row with ``created_at = now() + 24h``.
      4. Audit the decision (never raises).

    The Telegram ``edit_message`` keyboard scrub is intentionally out of scope
    here — the webhook handler in ``main.py`` owns the HTTP boundary (and the
    current ``tools/telegram.py`` helper surface does not expose
    ``answer_callback_query`` / plain ``editMessageReplyMarkup``).
    """
    try:
        hypothesis_id, action = signer.verify_callback(callback_data)
    except ValueError:
        logger.warning("telegram_digest.handle_callback: invalid HMAC")
        return CallbackResult(status="invalid_hmac")

    if action not in _VALID_CALLBACK_ACTIONS:
        logger.warning("telegram_digest.handle_callback: unknown action=%s", action)
        return CallbackResult(status="invalid_hmac", hypothesis_id=hypothesis_id, action=action)

    async with pool.connection() as conn:
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, question, options
                    FROM ask_queue
                    WHERE hypothesis_id = %s AND resolved_at IS NULL
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (hypothesis_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    logger.info(
                        "telegram_digest.handle_callback: no unresolved row hypothesis=%s",
                        hypothesis_id,
                    )
                    return CallbackResult(
                        status="already_resolved",
                        hypothesis_id=hypothesis_id,
                        action=action,
                    )
                ask_id = int(row[0])
                question_snapshot = str(row[1] or "")
                options_snapshot = row[2] or ["approve", "reject", "defer_24h"]

                await cur.execute(
                    "UPDATE ask_queue SET resolved_at=NOW(), answer=%s WHERE id=%s",
                    (action, ask_id),
                )

                if action == "defer_24h":
                    await cur.execute(
                        """
                        INSERT INTO ask_queue (created_at, hypothesis_id, question, options)
                        VALUES (NOW() + INTERVAL '24 hours', %s, %s, %s)
                        """,
                        (hypothesis_id, question_snapshot, Jsonb(list(options_snapshot))),
                    )

    try:
        await insert_audit_log(
            pool,
            hypothesis_id=hypothesis_id,
            trust_level="n/a",
            tool_name="telegram_digest.handle_callback",
            tool_input={"action": action, "ask_id": ask_id},
            tool_output={"status": "resolved"},
            is_mutation=True,
            user_confirmed=True,
        )
    except Exception:
        logger.warning("telegram_digest.handle_callback: audit_log write failed", exc_info=True)

    # http_client / settings accepted for future ``edit_message`` scrub — kept
    # in the signature so the webhook handler can pass them once the tools
    # surface grows an ``editMessageReplyMarkup`` helper (see module TODO(4)).
    del http_client, settings

    return CallbackResult(
        status="deferred" if action == "defer_24h" else "resolved",
        hypothesis_id=hypothesis_id,
        action=action,
        ask_id=ask_id,
    )


# --- job entry -------------------------------------------------------------


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    window_hours: int = _DEFAULT_WINDOW_HOURS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY-compatible cron entry.

    Degrades to a no-op when ``http_client`` or ``settings`` is absent — the
    default ``(pool, dry_run)`` dispatch in ``agent_runtime.jobs.__init__``
    cannot actually deliver a message.

    ``dry_run=True`` compiles + renders the digest and returns it in the JSON
    response without calling Telegram. Useful for the ``curl ?dry_run=true``
    smoke check in the runbook.
    """
    if http_client is None or settings is None:
        logger.warning("telegram_digest: DI missing (http/settings) — degraded no-op")
        return {
            "status": "ok",
            "action": "degraded_noop",
            "dry_run": dry_run,
            "sent": False,
        }

    try:
        payload = await compile_digest(pool, now=now, window_hours=window_hours)
    except Exception as exc:
        logger.exception("telegram_digest: compile failed")
        return {
            "status": "error",
            "action": "compile_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "dry_run": dry_run,
            "sent": False,
        }

    text = render_digest(payload)

    result: dict[str, Any] = {
        "status": "ok",
        "dry_run": dry_run,
        "actions_count": len(payload.actions_taken),
        "hypotheses_started": len(payload.hypotheses_started),
        "hypotheses_concluded": len(payload.hypotheses_concluded),
        "ask_queue_count": payload.ask_queue_count,
        "text": text,
    }

    if dry_run:
        logger.info(
            "telegram_digest DRY RUN: %d chars, %d actions, %d ASK",
            len(text),
            len(payload.actions_taken),
            payload.ask_queue_count,
        )
        result["sent"] = False
        return result

    try:
        message_id = await telegram_tools.send_message(http_client, settings, text=text)
        result["sent"] = True
        result["message_id"] = message_id
    except Exception as exc:
        logger.exception("telegram_digest: send_message failed")
        result["status"] = "error"
        result["sent"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="n/a",
            tool_name="telegram_digest",
            tool_input={
                "window_hours": window_hours,
                "dry_run": dry_run,
            },
            tool_output={
                "actions": len(payload.actions_taken),
                "ask": payload.ask_queue_count,
                "sent": True,
            },
            is_mutation=False,
        )
    except Exception:
        logger.warning("telegram_digest: audit_log write failed", exc_info=True)

    return result


__all__ = [
    "ActionSummary",
    "AskSummary",
    "CallbackResult",
    "DigestPayload",
    "HypothesisSummary",
    "compile_digest",
    "enqueue_ask",
    "handle_callback",
    "render_digest",
    "run",
]
