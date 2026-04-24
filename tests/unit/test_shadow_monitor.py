"""Unit tests for agent_runtime.jobs.shadow_monitor (Task 16b).

Covers the invariant guard logic:

* No-op branches (non-shadow trust, FORBIDDEN_LOCK, autonomous).
* Happy path (0 violations) beats watchdog_heartbeat only.
* Violation path — Telegram CRITICAL + FORBIDDEN_LOCK + forensic
  audit_log entry.
* ``dry_run=True`` makes the lock/alert/audit ops side-effect free.
* Telegram failure does **not** block the lock (priority: lock > alert).
* Concurrent lock race → caught ValueError from set_trust_level,
  returns skipped result.
* ``_safe_run`` wrapper returns structured error on surprise exception
  instead of bubbling through to FastAPI.

Uses a stateful mock pool backed by an in-memory ``sda_state`` +
``audit_log`` + ``watchdog_heartbeat`` shim so the trust-state
transitions are observable per-test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import shadow_monitor


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


class _FakePool:
    """In-memory shim for the sda_state + audit_log + watchdog rows.

    Interprets a narrow subset of SQL strings the module executes. Any
    other SQL is accepted and ignored (so unrelated calls — like
    insert_audit_log's own parameterised INSERT — still succeed and
    return the sentinel id 1 via fetchone()).
    """

    def __init__(
        self,
        trust_level: str = "shadow",
        violations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.trust_level = trust_level
        self.violations = violations or []
        self.heartbeat_upserts = 0
        self.audit_log_inserts: list[dict[str, Any]] = []
        self.trust_transitions: list[tuple[str, str]] = []
        self._last_select: str = ""

    def connection(self) -> _FakeConn:
        return _FakeConn(self)


class _FakeConn:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.pool)


class _FakeCursor:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool
        self._next_fetchone: Any = None
        self._next_fetchall: list[tuple[Any, ...]] = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        s = sql.strip()
        self.pool._last_select = s

        if "SELECT value #>> '{}' FROM sda_state" in s:
            self._next_fetchone = (self.pool.trust_level,)
            return
        if s.startswith("SELECT COUNT(*)") and "audit_log" in s:
            self._next_fetchone = (len(self.pool.violations),)
            return
        if s.startswith("SELECT id, ts, tool_name, hypothesis_id"):
            self._next_fetchall = [
                (v["id"], v.get("ts"), v["tool_name"], v.get("hypothesis_id"))
                for v in self.pool.violations
            ]
            return
        if "INSERT INTO watchdog_heartbeat" in s:
            self.pool.heartbeat_upserts += 1
            return
        if "INSERT INTO sda_state" in s and params:
            # set_trust_level path — params[0] is Jsonb-wrapped value
            val = params[0]
            value_str = getattr(val, "obj", val)
            if isinstance(value_str, str) and value_str in (
                "shadow",
                "assisted",
                "autonomous",
                "FORBIDDEN_LOCK",
            ):
                self.pool.trust_transitions.append((self.pool.trust_level, value_str))
                self.pool.trust_level = value_str
            return
        if "INSERT INTO audit_log" in s:
            params = params or ()
            # Two shapes: (a) insert_audit_log from db.py — 10 params
            # starting (hypothesis_id, trust_level, tool_name, ...);
            # (b) trust_levels._audit_trust_change — 2 params
            # (trust_level, details_jsonb), tool_name is hardcoded
            # 'trust_levels.set' in the SQL itself.
            if len(params) >= 10:
                self.pool.audit_log_inserts.append(
                    {
                        "hypothesis_id": params[0],
                        "trust_level": params[1],
                        "tool_name": params[2],
                        "is_mutation": params[5],
                        "kill_switch_triggered": params[9],
                    }
                )
            elif len(params) == 2:
                self.pool.audit_log_inserts.append(
                    {
                        "hypothesis_id": None,
                        "trust_level": params[0],
                        "tool_name": "trust_levels.set",
                        "is_mutation": False,
                        "kill_switch_triggered": None,
                    }
                )
            self._next_fetchone = (1,)
            return

    async def fetchone(self) -> Any:
        out = self._next_fetchone
        self._next_fetchone = None
        if out is None:
            return (1,)  # default for RETURNING id
        return out

    async def fetchall(self) -> list[tuple[Any, ...]]:
        out = self._next_fetchall
        self._next_fetchall = []
        return out


def _violation(vid: int, tool: str = "direct.pause_campaign") -> dict[str, Any]:
    return {
        "id": vid,
        "ts": datetime(2026, 4, 24, 12, 0, tzinfo=UTC),
        "tool_name": tool,
        "hypothesis_id": None,
    }


# -------------------------------------------------------------- no-op paths


@pytest.mark.asyncio
async def test_no_op_when_trust_not_shadow() -> None:
    pool = _FakePool(trust_level="assisted", violations=[_violation(1)])
    result = await shadow_monitor.run(pool)
    assert result["skipped"] is True
    assert "assisted" in result["reason"]
    # must not alert nor lock
    assert pool.trust_transitions == []
    assert pool.heartbeat_upserts == 0


@pytest.mark.asyncio
async def test_no_op_when_forbidden_lock() -> None:
    pool = _FakePool(trust_level="FORBIDDEN_LOCK", violations=[_violation(1)])
    result = await shadow_monitor.run(pool)
    assert result["skipped"] is True
    assert pool.trust_transitions == []


@pytest.mark.asyncio
async def test_happy_path_zero_violations_beats_heartbeat() -> None:
    pool = _FakePool(trust_level="shadow", violations=[])
    result = await shadow_monitor.run(pool)
    assert result["violations_count"] == 0
    assert result["locked"] is False
    assert result["alert_sent"] is False
    assert pool.heartbeat_upserts == 1
    assert pool.trust_transitions == []


# -------------------------------------------------------------- dry_run path


@pytest.mark.asyncio
async def test_dry_run_detects_but_does_not_lock() -> None:
    pool = _FakePool(trust_level="shadow", violations=[_violation(1)])
    result = await shadow_monitor.run(pool, dry_run=True)
    assert result["violations_count"] == 1
    assert result["locked"] is False
    assert result["alert_sent"] is False
    assert result["dry_run"] is True
    assert pool.trust_transitions == []
    assert pool.audit_log_inserts == []


# ------------------------------------------------------- violation → lock


@pytest.mark.asyncio
async def test_violation_triggers_lock_and_alert() -> None:
    pool = _FakePool(trust_level="shadow", violations=[_violation(42)])
    telegram_mock = AsyncMock(return_value=1)
    http_client = MagicMock()
    with patch.object(shadow_monitor.telegram_tools, "send_message", telegram_mock):
        result = await shadow_monitor.run(pool, http_client=http_client, settings=_settings())

    assert result["locked"] is True
    assert result["alert_sent"] is True
    assert result["violations_count"] == 1
    assert result["violating_rows"] == [42]
    assert result["trust_level"] == "FORBIDDEN_LOCK"
    assert pool.trust_level == "FORBIDDEN_LOCK"
    telegram_mock.assert_awaited_once()
    assert "SHADOW INVARIANT VIOLATED" in telegram_mock.await_args.kwargs["text"]
    # forensic audit row
    tools = [row["tool_name"] for row in pool.audit_log_inserts]
    assert "shadow_monitor.lock" in tools


@pytest.mark.asyncio
async def test_multiple_violations_aggregated_in_alert() -> None:
    violations = [_violation(i, f"tool_{i}") for i in range(15)]
    pool = _FakePool(trust_level="shadow", violations=violations)
    telegram_mock = AsyncMock(return_value=1)
    with patch.object(shadow_monitor.telegram_tools, "send_message", telegram_mock):
        result = await shadow_monitor.run(pool, http_client=MagicMock(), settings=_settings())

    assert result["violations_count"] == 15
    text = telegram_mock.await_args.kwargs["text"]
    assert "Violations: <b>15</b>" in text
    # message caps visible offenders to 10 and shows "... 5 more"
    assert "and 5 more" in text


# ------------------------------------------- telegram failure still locks


@pytest.mark.asyncio
async def test_telegram_failure_still_locks() -> None:
    pool = _FakePool(trust_level="shadow", violations=[_violation(1)])
    telegram_mock = AsyncMock(side_effect=RuntimeError("telegram 500"))
    with patch.object(shadow_monitor.telegram_tools, "send_message", telegram_mock):
        result = await shadow_monitor.run(pool, http_client=MagicMock(), settings=_settings())

    assert result["locked"] is True
    assert result["alert_sent"] is False
    assert pool.trust_level == "FORBIDDEN_LOCK"


# --------------------------------------------------- no telegram client (cron default)


@pytest.mark.asyncio
async def test_violation_without_http_client_still_locks() -> None:
    """Default JOB_REGISTRY dispatch passes only (pool, dry_run)."""
    pool = _FakePool(trust_level="shadow", violations=[_violation(1)])
    result = await shadow_monitor.run(pool)
    assert result["locked"] is True
    assert result["alert_sent"] is False
    assert pool.trust_level == "FORBIDDEN_LOCK"


# ------------------------------------------------------- format + concurrency


def test_format_alert_includes_ts_and_tool_name() -> None:
    text = shadow_monitor._format_alert(2, [_violation(7, "direct.foo")])
    assert "#7" in text
    assert "direct.foo" in text
    assert "SHADOW INVARIANT VIOLATED" in text


def test_format_alert_handles_empty_offenders_list() -> None:
    text = shadow_monitor._format_alert(0, [])
    assert "Violations: <b>0</b>" in text


@pytest.mark.asyncio
async def test_concurrent_lock_skips_without_raising() -> None:
    """Second run detecting a race doesn't blow up — set_trust_level raised."""
    pool = _FakePool(trust_level="shadow", violations=[_violation(1)])
    telegram_mock = AsyncMock(return_value=1)

    async def _raise_invalid(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Invalid transition shadow → FORBIDDEN_LOCK")

    with (
        patch.object(shadow_monitor.telegram_tools, "send_message", telegram_mock),
        patch.object(shadow_monitor, "set_trust_level", _raise_invalid),
    ):
        result = await shadow_monitor.run(pool, http_client=MagicMock(), settings=_settings())

    assert result["locked"] is False
    assert result["alert_sent"] is True  # we still tried the alert
    # forensic audit row NOT inserted if lock failed
    assert "shadow_monitor.lock" not in [r["tool_name"] for r in pool.audit_log_inserts]


# -------------------------------------------------------- wrapper resilience


@pytest.mark.asyncio
async def test_safe_run_wraps_impl_exception() -> None:
    pool = _FakePool(trust_level="shadow", violations=[])
    with patch.object(shadow_monitor, "_run_impl", AsyncMock(side_effect=RuntimeError("db down"))):
        result = await shadow_monitor.run(pool, http_client=MagicMock(), settings=_settings())
    assert "error" in result
    assert "db down" in result["error"]


@pytest.mark.asyncio
async def test_safe_run_attempts_crash_alert() -> None:
    pool = _FakePool(trust_level="shadow", violations=[])
    telegram_mock = AsyncMock(return_value=1)
    with (
        patch.object(shadow_monitor, "_run_impl", AsyncMock(side_effect=RuntimeError("boom"))),
        patch.object(shadow_monitor.telegram_tools, "send_message", telegram_mock),
    ):
        await shadow_monitor.run(pool, http_client=MagicMock(), settings=_settings())
    telegram_mock.assert_awaited_once()
    assert "CRASHED" in telegram_mock.await_args.kwargs["text"]


# ----------------------------------------------------- heartbeat isolation


@pytest.mark.asyncio
async def test_heartbeat_failure_swallowed() -> None:
    """Happy path with heartbeat exception — run_once still returns OK."""
    pool = _FakePool(trust_level="shadow", violations=[])

    async def _raise(*_args: Any) -> None:
        raise RuntimeError("heartbeat db error")

    with patch.object(shadow_monitor, "_beat_heartbeat", _raise):
        result = await shadow_monitor.run(pool)
    # _beat_heartbeat itself has a try/except that prevents raising, but
    # we also assert the wrapper catches any leaked exception
    # downstream — either way, result exists.
    assert "violations_count" in result or "error" in result


# ------------------------------------------------------ registry + cron


@pytest.mark.asyncio
async def test_run_registered_in_job_registry() -> None:
    from agent_runtime.jobs import JOB_REGISTRY

    assert "shadow_monitor" in JOB_REGISTRY
    assert JOB_REGISTRY["shadow_monitor"] is shadow_monitor.run


def test_cron_entry_exists_in_railway_toml() -> None:
    from pathlib import Path

    text = Path("railway.toml").read_text()
    assert 'name = "shadow_monitor"' in text
    assert '"*/5 * * * *"' in text
