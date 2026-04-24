"""Unit tests for ``agent_runtime.jobs.strategy_switcher`` (Task 26).

The SUT is purely a PG orchestrator — there is no Direct API call, no
Metrika call, no Telegram send. So the only stubs we need are:

* An in-memory fake for ``sda_state`` (single row keyed on ``strategy_gate_state``)
  plus an ``ask_queue`` dedupe SELECT and hypothesis / ask INSERTs.
* No DirectAPI / HTTP stubs at all — a test that the module never tries
  to reach outside is covered by ``test_no_direct_or_telegram_calls``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs.strategy_switcher import (
    CURRENT_STRATEGY,
    PROPOSED_STRATEGY,
    run,
)

_PROTECTED = [708978456, 708978457, 708978458]


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        PROTECTED_CAMPAIGN_IDS=_PROTECTED,
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
    )


# ------------------------------------------------------------------ fake pool


class _FakePool:
    """Minimal shim that replays a pre-wired sda_state row and records writes.

    SQL recognised:
      * ``SELECT value FROM sda_state WHERE key = 'strategy_gate_state'``
      * ``SELECT 1 FROM ask_queue WHERE resolved_at IS NULL AND options...``
        — returns (1,) iff :attr:`has_open_ask` is True.
      * ``INSERT INTO hypotheses (...)`` — appended to :attr:`hypotheses_rows`.
      * ``INSERT INTO ask_queue (...) RETURNING id`` — returns ``ask_id_seq``,
        appended to :attr:`ask_queue_rows`.
      * ``INSERT INTO audit_log ... RETURNING id`` — returns (1,).

    Anything else is a silent no-op; unexpected SQL raises AssertionError.
    """

    def __init__(
        self,
        *,
        state_row: dict[str, Any] | None = None,
        has_open_ask: bool = False,
        insert_fail_on: str | None = None,
    ) -> None:
        self.state_row = state_row
        self.has_open_ask = has_open_ask
        self.insert_fail_on = insert_fail_on  # "hypotheses" or "ask_queue" or None
        self.hypotheses_rows: list[dict[str, Any]] = []
        self.ask_queue_rows: list[dict[str, Any]] = []
        self.audit_log_rows: list[dict[str, Any]] = []
        self.ask_id_seq = 77  # arbitrary non-1

    def connection(self) -> _FakeConn:
        return _FakeConn(self)


class _FakeConn:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.pool)


class _FakeCursor:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool
        self._next: Any = None

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        s = " ".join(sql.split())  # normalise whitespace for easier matching
        if "FROM sda_state WHERE key" in s and "SELECT value" in s:
            self._next = (self.pool.state_row,) if self.pool.state_row else None
            return
        if "FROM ask_queue" in s and "resolved_at IS NULL" in s:
            self._next = (1,) if self.pool.has_open_ask else None
            return
        if "INSERT INTO hypotheses" in s:
            if self.pool.insert_fail_on == "hypotheses":
                raise RuntimeError("simulated hypotheses INSERT failure")
            assert params is not None
            self.pool.hypotheses_rows.append(
                {"id": params[0], "agent": params[1], "type": params[2]}
            )
            return
        if "INSERT INTO ask_queue" in s:
            if self.pool.insert_fail_on == "ask_queue":
                raise RuntimeError("simulated ask_queue INSERT failure")
            assert params is not None
            hypothesis_id, question, options = params[:3]
            options_obj = getattr(options, "obj", options)
            self.pool.ask_queue_rows.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "question": question,
                    "options": options_obj,
                }
            )
            self._next = (self.pool.ask_id_seq,)
            return
        if "INSERT INTO audit_log" in s:
            assert params is not None
            self.pool.audit_log_rows.append(
                {
                    "hypothesis_id": params[0],
                    "trust_level": params[1],
                    "tool_name": params[2],
                    "is_mutation": params[5] if len(params) >= 6 else None,
                }
            )
            self._next = (1,)
            return
        # Unknown SQL — treat as no-op (keeps test failures readable when the
        # SUT adds a harmless statement).

    async def fetchone(self) -> Any:
        out = self._next
        self._next = None
        return out


def _state(status: str = "ready_to_switch", **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": status,
        "entered_at": "2026-04-21T08:00:00+00:00",
        "signals": {"cpa_stability_7d": {"mean": 3400.0}},
    }
    base.update(extra)
    return base


# ------------------------------------------------------------------ tests


@pytest.mark.asyncio
async def test_skip_when_state_learning() -> None:
    pool = _FakePool(state_row=_state(status="learning"))
    result = await run(pool, settings=_settings())
    assert result["action"] == "skip"
    assert result["state"] == "learning"
    assert pool.hypotheses_rows == []
    assert pool.ask_queue_rows == []


@pytest.mark.asyncio
async def test_skip_when_state_auto_pilot() -> None:
    pool = _FakePool(state_row=_state(status="auto_pilot"))
    result = await run(pool, settings=_settings())
    assert result["action"] == "skip"
    assert result["state"] == "auto_pilot"
    assert pool.ask_queue_rows == []


@pytest.mark.asyncio
async def test_skip_when_state_degraded() -> None:
    pool = _FakePool(state_row=_state(status="degraded"))
    result = await run(pool, settings=_settings())
    assert result["action"] == "skip"
    assert pool.ask_queue_rows == []


@pytest.mark.asyncio
async def test_skip_when_state_missing() -> None:
    pool = _FakePool(state_row=None)
    result = await run(pool, settings=_settings())
    assert result["action"] == "skip"
    assert result["state"] == "missing"
    assert pool.ask_queue_rows == []


@pytest.mark.asyncio
async def test_ask_when_ready_to_switch() -> None:
    pool = _FakePool(state_row=_state(status="ready_to_switch"))
    result = await run(pool, settings=_settings())
    assert result["action"] == "ask_created"
    assert len(pool.hypotheses_rows) == 1
    assert len(pool.ask_queue_rows) == 1
    # Question carries the target strategy name and the gate tick marker.
    ask_row = pool.ask_queue_rows[0]
    assert PROPOSED_STRATEGY in ask_row["question"]
    assert CURRENT_STRATEGY in ask_row["question"]
    assert "2026-04-21" in ask_row["question"]
    # Campaigns default to settings.PROTECTED_CAMPAIGN_IDS when the state
    # does not carry an active_campaigns override.
    assert ask_row["options"]["campaigns"] == _PROTECTED
    assert ask_row["options"]["to_strategy"] == PROPOSED_STRATEGY
    assert ask_row["options"]["kind"] == "strategy_switcher"


@pytest.mark.asyncio
async def test_ask_uses_state_active_campaigns_when_present() -> None:
    override = [111, 222]
    pool = _FakePool(state_row=_state(active_campaigns=override))
    result = await run(pool, settings=_settings())
    assert result["action"] == "ask_created"
    assert pool.ask_queue_rows[0]["options"]["campaigns"] == override
    assert str(override[0]) in result["question"]


@pytest.mark.asyncio
async def test_dry_run_no_commit() -> None:
    pool = _FakePool(state_row=_state(status="ready_to_switch"))
    result = await run(pool, dry_run=True, settings=_settings())
    assert result["action"] == "ask_drafted"
    assert result["dry_run"] is True
    assert "hypothesis_draft_id" in result
    # Critical: NO PG writes despite the draft being produced.
    assert pool.hypotheses_rows == []
    assert pool.ask_queue_rows == []
    assert pool.audit_log_rows == []


@pytest.mark.asyncio
async def test_idempotent_same_gate_tick() -> None:
    pool = _FakePool(state_row=_state(status="ready_to_switch"), has_open_ask=True)
    result = await run(pool, settings=_settings())
    assert result["action"] == "skip_duplicate"
    assert pool.ask_queue_rows == []
    assert pool.hypotheses_rows == []


@pytest.mark.asyncio
async def test_cpa_mean_rendered_in_question() -> None:
    pool = _FakePool(
        state_row=_state(
            status="ready_to_switch",
            signals={"cpa_stability_7d": {"mean": 2750.0}},
        )
    )
    result = await run(pool, settings=_settings())
    assert result["action"] == "ask_created"
    assert "2750₽" in result["question"]


@pytest.mark.asyncio
async def test_missing_cpa_renders_nan_gracefully() -> None:
    pool = _FakePool(
        state_row=_state(status="ready_to_switch", signals={"cpa_stability_7d": {}}),
    )
    result = await run(pool, settings=_settings())
    assert result["action"] == "ask_created"
    assert "n/a" in result["question"]


@pytest.mark.asyncio
async def test_degraded_noop_when_settings_missing() -> None:
    pool = _FakePool(state_row=_state(status="ready_to_switch"))
    result = await run(pool, settings=None)
    assert result["action"] == "degraded_noop"
    assert pool.ask_queue_rows == []


@pytest.mark.asyncio
async def test_error_captured_when_persist_fails() -> None:
    pool = _FakePool(
        state_row=_state(status="ready_to_switch"),
        insert_fail_on="ask_queue",
    )
    result = await run(pool, settings=_settings())
    assert result["action"] == "error"
    assert "simulated" in result["error"]


@pytest.mark.asyncio
async def test_hypothesis_draft_has_correct_type() -> None:
    pool = _FakePool(state_row=_state(status="ready_to_switch"))
    await run(pool, settings=_settings())
    assert pool.hypotheses_rows[0]["type"] == "strategy_switch"
    assert pool.hypotheses_rows[0]["agent"] == "strategy_switcher"


@pytest.mark.asyncio
async def test_no_direct_or_telegram_calls() -> None:
    """The switcher must not touch Direct / Telegram — even if passed in."""
    pool = _FakePool(state_row=_state(status="ready_to_switch"))
    direct = MagicMock()
    # Every public mutating method is an AsyncMock we can assert_not_called on.
    direct.update_strategy = AsyncMock()
    direct.pause_campaign = AsyncMock()
    direct.resume_campaign = AsyncMock()
    direct.set_bid = AsyncMock()
    direct.get_campaigns = AsyncMock()
    http = MagicMock()
    http.post = AsyncMock()
    http.get = AsyncMock()
    await run(pool, direct=direct, http_client=http, settings=_settings())
    direct.update_strategy.assert_not_called()
    direct.pause_campaign.assert_not_called()
    direct.resume_campaign.assert_not_called()
    direct.set_bid.assert_not_called()
    http.post.assert_not_called()
    http.get.assert_not_called()


@pytest.mark.asyncio
async def test_audit_log_written_on_live_ask() -> None:
    pool = _FakePool(state_row=_state(status="ready_to_switch"))
    await run(pool, settings=_settings())
    assert len(pool.audit_log_rows) == 1
    audit = pool.audit_log_rows[0]
    assert audit["tool_name"] == "strategy_switcher"
    # draft-only; is_mutation should be False (real mutation happens on Approve).
    assert audit["is_mutation"] is False


@pytest.mark.asyncio
async def test_sda_state_read_failure_becomes_missing_skip() -> None:
    """_load_gate_state swallows PG errors internally → run() sees missing, skips."""

    class _BrokenPool(_FakePool):
        def connection(self) -> _BrokenConn:  # type: ignore[override]
            return _BrokenConn(self)

    class _BrokenConn:
        def __init__(self, pool: _FakePool) -> None:
            self.pool = pool

        async def __aenter__(self) -> _BrokenConn:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        def cursor(self) -> _BrokenCursor:
            return _BrokenCursor()

    class _BrokenCursor:
        async def __aenter__(self) -> _BrokenCursor:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def execute(self, sql: str, params: Any | None = None) -> None:
            raise RuntimeError("pg down")

        async def fetchone(self) -> Any:
            return None

    pool = _BrokenPool()
    result = await run(pool, settings=_settings())
    # _load_gate_state catches the error and returns None → state='missing' skip.
    assert result["action"] == "skip"
    assert result["state"] == "missing"
    assert pool.ask_queue_rows == []


@pytest.mark.asyncio
async def test_ask_queue_payload_has_strategy_fields() -> None:
    """The JSONB options row must carry enough fields for Task 23 handler."""
    pool = _FakePool(state_row=_state(status="ready_to_switch"))
    await run(pool, settings=_settings())
    opts = pool.ask_queue_rows[0]["options"]
    for key in ("kind", "gate_entered_at", "campaigns", "from_strategy", "to_strategy"):
        assert key in opts, f"missing key {key} in ask_queue.options JSONB"
    assert opts["from_strategy"] == CURRENT_STRATEGY
    assert opts["to_strategy"] == PROPOSED_STRATEGY
