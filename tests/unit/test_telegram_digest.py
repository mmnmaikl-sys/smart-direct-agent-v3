"""Unit tests for agent_runtime.jobs.telegram_digest (Task 23).

All I/O is mocked: the pool is a scripted ``AsyncMock`` cursor matching the
pattern used in ``test_budget_guard.py`` / ``test_smart_optimizer.py``;
Telegram is an ``AsyncMock`` monkey-patched on ``telegram_tools``. The test
suite covers:

* Digest compilation (empty, populated, PII-sanitised).
* Render format: HTML markers, truncation at 4096 chars, empty-state message.
* ASK outbound: INSERT + send_with_inline + message_id UPDATE; Telegram
  failure is swallowed and row still returns.
* Callback resolve: valid HMAC resolves; invalid HMAC short-circuits; already
  resolved NOOP; defer_24h inserts a follow-up row.
* Job entry: dry_run skips send; degraded_noop without DI; Telegram send
  exception surfaces as ``status=error``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.auth.signing import HMACSigner
from agent_runtime.config import Settings
from agent_runtime.jobs import telegram_digest

# --------------------------------------------------------------------- fixtures


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )


@dataclass
class _Script:
    """Scripted DB responses. ``fetchone`` / ``fetchall`` are popped in order."""

    fetchone_queue: list[Any]
    fetchall_queue: list[Any]
    executed: list[tuple[Any, ...]]


def _mock_pool(*, fetchone: list[Any], fetchall: list[Any]) -> tuple[MagicMock, _Script]:
    script = _Script(
        fetchone_queue=list(fetchone),
        fetchall_queue=list(fetchall),
        executed=[],
    )

    async def _fetchone():
        if script.fetchone_queue:
            return script.fetchone_queue.pop(0)
        return (1,)

    async def _fetchall():
        if script.fetchall_queue:
            return script.fetchall_queue.pop(0)
        return []

    async def _exec(*args: Any, **kwargs: Any) -> None:
        script.executed.append(args)

    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=_exec)
    cursor.fetchone = AsyncMock(side_effect=_fetchone)
    cursor.fetchall = AsyncMock(side_effect=_fetchall)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    # conn.transaction() is an async context manager in psycopg.
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=tx)

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, script


# ---------------------------------------------------------------------- compile


@pytest.mark.asyncio
async def test_compile_digest_empty_returns_zeros() -> None:
    pool, _ = _mock_pool(
        fetchone=[(0,)],  # ask_queue count
        fetchall=[[], [], [], []],  # actions, started, concluded, unresolved
    )
    payload = await telegram_digest.compile_digest(pool)
    assert payload.actions_taken == []
    assert payload.hypotheses_started == []
    assert payload.hypotheses_concluded == []
    assert payload.ask_queue_count == 0
    assert payload.ask_queue_unresolved == []
    assert payload.is_empty()


@pytest.mark.asyncio
async def test_compile_digest_populated_sections() -> None:
    now = datetime.now(UTC)
    pool, _ = _mock_pool(
        fetchone=[(3,)],  # ask count
        fetchall=[
            # actions
            [
                (now, "add_negatives", "autonomous", False, {"reason": "garbage_queries"}),
                (now, "pause_group", "shadow", True, {"campaign_id": 123}),
            ],
            # started
            [("abc12345", "neg_kw", "running", 123, now)],
            # concluded
            [("def67890", "ad", "confirmed", 456, now)],
            # unresolved ASK
            [
                (1, "abc12345", "Approve add_negatives?", now),
                (2, "xyz00000", "Reject ad?", now - timedelta(hours=2)),
            ],
        ],
    )
    payload = await telegram_digest.compile_digest(pool, window_hours=24)
    assert len(payload.actions_taken) == 2
    assert payload.actions_taken[0].tool_name == "add_negatives"
    assert payload.actions_taken[1].is_error is True
    assert payload.hypotheses_started[0].id == "abc12345"
    assert payload.hypotheses_concluded[0].state == "confirmed"
    assert payload.ask_queue_count == 3
    assert len(payload.ask_queue_unresolved) == 2


@pytest.mark.asyncio
async def test_compile_digest_pii_sanitised_in_actions() -> None:
    """Phone in ``tool_input`` must be hashed before reaching the payload."""
    now = datetime.now(UTC)
    pool, _ = _mock_pool(
        fetchone=[(0,)],
        fetchall=[
            [(now, "bitrix_lead", "autonomous", False, {"phone": "+79991234567"})],
            [],
            [],
            [],
        ],
    )
    payload = await telegram_digest.compile_digest(pool)
    # The raw phone is never in payload — only a sanitised reason.
    assert "+79991234567" not in payload.actions_taken[0].reason
    assert "9991234567" not in payload.actions_taken[0].reason
    # sanitiser returns a dict with hashed "phone" → _short_reason falls back to "-".
    assert payload.actions_taken[0].reason == "-"


# ----------------------------------------------------------------------- render


def _make_payload(**overrides: Any) -> telegram_digest.DigestPayload:
    base: dict[str, Any] = {
        "generated_at": datetime(2026, 4, 24, 9, 0, tzinfo=UTC),
        "window_hours": 24,
        "actions_taken": [],
        "hypotheses_started": [],
        "hypotheses_concluded": [],
        "ask_queue_count": 0,
        "ask_queue_unresolved": [],
    }
    base.update(overrides)
    return telegram_digest.DigestPayload(**base)


def test_render_digest_empty_payload_reports_quiet_night() -> None:
    text = telegram_digest.render_digest(_make_payload())
    assert "Тихая ночь" in text
    assert "SDA v3 Digest" in text


def test_render_digest_contains_all_section_headers() -> None:
    now = datetime(2026, 4, 24, 9, 0, tzinfo=UTC)
    payload = _make_payload(
        actions_taken=[
            telegram_digest.ActionSummary(
                ts=now,
                tool_name="add_negatives",
                trust_level="autonomous",
                is_error=False,
                reason="reason=garbage_queries",
            )
        ],
        hypotheses_started=[
            telegram_digest.HypothesisSummary(
                id="abc12345",
                hypothesis_type="neg_kw",
                state="running",
                campaign_id=100,
                created_at=now,
            )
        ],
        hypotheses_concluded=[
            telegram_digest.HypothesisSummary(
                id="def67890",
                hypothesis_type="ad",
                state="confirmed",
                campaign_id=100,
                created_at=now,
            )
        ],
        ask_queue_count=2,
        ask_queue_unresolved=[
            telegram_digest.AskSummary(id=1, hypothesis_id="abc12345", question="q", created_at=now)
        ],
    )
    text = telegram_digest.render_digest(payload)
    assert "🛠 Actions" in text
    assert "Hypotheses started" in text
    assert "Hypotheses concluded" in text
    assert "❓ ASK (2 unresolved)" in text
    assert "add_negatives" in text
    assert "abc12345" in text
    assert "… и ещё 1" in text  # remainder hint


def test_render_digest_truncates_over_4096_chars() -> None:
    now = datetime(2026, 4, 24, 9, 0, tzinfo=UTC)
    many_actions = [
        telegram_digest.ActionSummary(
            ts=now,
            tool_name="tool_" + ("x" * 50),
            trust_level="autonomous",
            is_error=False,
            reason="reason=" + ("y" * 200),
        )
        for _ in range(50)
    ]
    payload = _make_payload(actions_taken=many_actions)
    text = telegram_digest.render_digest(payload)
    assert len(text) <= 4096
    assert text.endswith("… обрезано")


# ------------------------------------------------------------------ enqueue_ask


@pytest.mark.asyncio
async def test_enqueue_ask_inserts_row_and_sends_inline() -> None:
    pool, script = _mock_pool(fetchone=[(99,)], fetchall=[])
    send_with_inline = AsyncMock(return_value=777)
    with patch.object(telegram_digest.telegram_tools, "send_with_inline", send_with_inline):
        ask_id = await telegram_digest.enqueue_ask(
            pool,
            SimpleNamespace(),  # http_client unused directly — send_with_inline is mocked
            _settings(),
            hypothesis_id="hyp12345",
            question="Approve add_negatives for campaign 123?",
        )
    assert ask_id == 99
    send_with_inline.assert_awaited_once()
    # Two SQL ops: INSERT + UPDATE message_id.
    inserted_sqls = [
        args[0] for args in script.executed if isinstance(args[0], str) and "INSERT" in args[0]
    ]
    updated_sqls = [
        args[0]
        for args in script.executed
        if isinstance(args[0], str) and "telegram_message_id" in args[0]
    ]
    assert inserted_sqls
    assert updated_sqls


@pytest.mark.asyncio
async def test_enqueue_ask_swallows_telegram_failure() -> None:
    pool, _ = _mock_pool(fetchone=[(99,)], fetchall=[])
    send_with_inline = AsyncMock(side_effect=RuntimeError("network down"))
    with patch.object(telegram_digest.telegram_tools, "send_with_inline", send_with_inline):
        ask_id = await telegram_digest.enqueue_ask(
            pool,
            SimpleNamespace(),
            _settings(),
            hypothesis_id="hyp12345",
            question="Approve?",
        )
    # Row still returned — the DB write completed before Telegram failed.
    assert ask_id == 99


# --------------------------------------------------------------- handle_callback


def _make_signer() -> HMACSigner:
    return HMACSigner(_settings().HYPOTHESIS_HMAC_SECRET)


@pytest.mark.asyncio
async def test_handle_callback_invalid_hmac_rejected() -> None:
    signer = _make_signer()
    pool, script = _mock_pool(fetchone=[], fetchall=[])
    result = await telegram_digest.handle_callback(
        pool,
        signer,
        callback_data="hyp12345:approve:deadbeef00",  # wrong sig
    )
    assert result.status == "invalid_hmac"
    # No SQL was executed against the pool.
    assert script.executed == []


@pytest.mark.asyncio
async def test_handle_callback_valid_hmac_resolves_and_audits() -> None:
    signer = _make_signer()
    callback = signer.sign_callback("hyp12345", "approve")
    pool, script = _mock_pool(
        fetchone=[
            (55, "q", ["approve", "reject", "defer_24h"]),  # SELECT FOR UPDATE
            (1,),  # audit_log INSERT RETURNING id
        ],
        fetchall=[],
    )
    result = await telegram_digest.handle_callback(pool, signer, callback_data=callback)
    assert result.status == "resolved"
    assert result.ask_id == 55
    assert result.action == "approve"
    # UPDATE ask_queue + audit INSERT were issued.
    update_sqls = [
        args[0]
        for args in script.executed
        if isinstance(args[0], str) and "UPDATE ask_queue" in args[0]
    ]
    audit_sqls = [
        args[0]
        for args in script.executed
        if isinstance(args[0], str) and "INSERT INTO audit_log" in args[0]
    ]
    assert update_sqls
    assert audit_sqls


@pytest.mark.asyncio
async def test_handle_callback_already_resolved_is_noop() -> None:
    signer = _make_signer()
    callback = signer.sign_callback("hyp12345", "reject")
    pool, script = _mock_pool(
        fetchone=[None],  # SELECT FOR UPDATE returns no row
        fetchall=[],
    )
    result = await telegram_digest.handle_callback(pool, signer, callback_data=callback)
    assert result.status == "already_resolved"
    # No UPDATE was issued.
    update_sqls = [
        args[0]
        for args in script.executed
        if isinstance(args[0], str) and "UPDATE ask_queue" in args[0]
    ]
    assert update_sqls == []


@pytest.mark.asyncio
async def test_handle_callback_defer_24h_inserts_followup_row() -> None:
    """Verified by asserting the INSERT with NOW() + INTERVAL '24 hours' is issued."""

    # ``defer_24h`` cannot be signed via the standard HMACSigner pathway
    # (action is rejected by the ``send_with_inline`` Literal), but the
    # signer itself accepts any action for verify — replicate one.
    settings = _settings()
    signer = HMACSigner(settings.HYPOTHESIS_HMAC_SECRET)
    callback = signer.sign_callback("hyp12345", "defer_24h")

    pool, script = _mock_pool(
        fetchone=[
            (77, "Approve?", ["approve", "reject", "defer_24h"]),
            (1,),  # audit RETURNING id
        ],
        fetchall=[],
    )
    result = await telegram_digest.handle_callback(pool, signer, callback_data=callback)
    assert result.status == "deferred"
    # Exactly one INSERT INTO ask_queue ... + 24 hours.
    insert_sqls = [
        args[0]
        for args in script.executed
        if isinstance(args[0], str) and "INSERT INTO ask_queue" in args[0]
    ]
    assert any("24 hours" in sql for sql in insert_sqls)


@pytest.mark.asyncio
async def test_handle_callback_rejects_malformed_data() -> None:
    signer = _make_signer()
    pool, _ = _mock_pool(fetchone=[], fetchall=[])
    result = await telegram_digest.handle_callback(
        pool, signer, callback_data="not:a:valid:callback"
    )
    assert result.status == "invalid_hmac"


# -------------------------------------------------------------------------- run


@pytest.mark.asyncio
async def test_run_degraded_noop_when_di_missing() -> None:
    pool, _ = _mock_pool(fetchone=[], fetchall=[])
    result = await telegram_digest.run(pool)
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    assert result["sent"] is False


@pytest.mark.asyncio
async def test_run_dry_run_skips_send_but_returns_text() -> None:
    pool, _ = _mock_pool(
        fetchone=[(0,)],
        fetchall=[[], [], [], []],
    )
    send = AsyncMock(return_value=1)
    with patch.object(telegram_digest.telegram_tools, "send_message", send):
        result = await telegram_digest.run(
            pool,
            dry_run=True,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )
    assert result["status"] == "ok"
    assert result["sent"] is False
    assert "text" in result and "SDA v3 Digest" in result["text"]
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_live_sends_message_and_writes_audit() -> None:
    pool, script = _mock_pool(
        fetchone=[(0,), (1,)],  # ask count + audit RETURNING id
        fetchall=[[], [], [], []],
    )
    send = AsyncMock(return_value=5555)
    with patch.object(telegram_digest.telegram_tools, "send_message", send):
        result = await telegram_digest.run(
            pool,
            dry_run=False,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )
    assert result["status"] == "ok"
    assert result["sent"] is True
    assert result["message_id"] == 5555
    # audit_log was attempted.
    audit_sqls = [
        args[0]
        for args in script.executed
        if isinstance(args[0], str) and "INSERT INTO audit_log" in args[0]
    ]
    assert audit_sqls


@pytest.mark.asyncio
async def test_run_send_failure_returns_error() -> None:
    pool, _ = _mock_pool(fetchone=[(0,)], fetchall=[[], [], [], []])
    send = AsyncMock(side_effect=RuntimeError("telegram 500"))
    with patch.object(telegram_digest.telegram_tools, "send_message", send):
        result = await telegram_digest.run(
            pool,
            dry_run=False,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )
    assert result["status"] == "error"
    assert result["sent"] is False
    assert "telegram 500" in result["error"]


@pytest.mark.asyncio
async def test_run_compile_failure_returns_error() -> None:
    # Pool whose first cursor.execute raises — simulates a DB outage.
    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=RuntimeError("db dead"))
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=tx)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)

    send = AsyncMock()
    with patch.object(telegram_digest.telegram_tools, "send_message", send):
        result = await telegram_digest.run(
            pool,
            dry_run=False,
            http_client=SimpleNamespace(),
            settings=_settings(),
        )
    assert result["status"] == "error"
    assert result["action"] == "compile_failed"
    send.assert_not_awaited()
