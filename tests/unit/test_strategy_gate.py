"""Unit tests for agent_runtime.jobs.strategy_gate (Task 17).

All external I/O is mocked:
* ``_FakePool`` — in-memory shim over ``sda_state`` + ``audit_log`` rows,
  with ``SELECT FOR UPDATE`` behaving the same way as ``SELECT`` (we don't
  spawn real concurrency in unit tests; there's a dedicated async test for
  the ordering property).
* DirectAPI stub via ``SimpleNamespace`` that just returns canned TSV.
* Bitrix calls patched at import site (``bitrix_tools.get_deal_list`` /
  ``bitrix_tools.get_lead_list``) — the module imports the module, not the
  function, so a single ``monkeypatch`` on the attribute covers both paths.
* ``telegram_tools.send_message`` patched the same way.

The ``_FakePool`` is intentionally dumb: it stores the JSONB value as a
dict and replays it on the next SELECT. ``_save_state`` stores whatever
payload the module passed through; tests then inspect it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import strategy_gate

# ------------------------------------------------------------ settings stub


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
        BITRIX_WEBHOOK_URL="https://example.bitrix24.ru/rest/1/TOKEN",
        BITRIX_WEBHOOK_TOKEN="d" * 64,
        YANDEX_DIRECT_TOKEN="token",
    )


# ------------------------------------------------------------ fake pool


class _FakePool:
    """In-memory sda_state + audit_log shim.

    Handles:
    * ``SELECT value FROM sda_state WHERE key = %s`` (+ FOR UPDATE).
    * ``INSERT ... ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value``.
    * ``INSERT INTO audit_log`` — just records the row.
    * The offline_conversions COUNT(*) query (uses preset ``offline_counts``).

    Anything else is silently accepted so the module can issue its own
    extra statements (e.g. audit_log INSERT ... RETURNING id).
    """

    def __init__(
        self,
        *,
        initial_state: dict[str, Any] | None = None,
        offline_counts: tuple[int, int] = (0, 0),
        offline_raises: Exception | None = None,
    ) -> None:
        self.state_row: dict[str, Any] | None = initial_state
        self.offline_counts = offline_counts
        self.offline_raises = offline_raises
        self.audit_log_inserts: list[dict[str, Any]] = []
        self.saved_states: list[dict[str, Any]] = []

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
        self._next_fetchone: Any = None

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        s = sql.strip()
        if "FROM sda_state" in s and "SELECT" in s:
            # Both FOR UPDATE and plain SELECT take the same branch.
            if self.pool.state_row is None:
                self._next_fetchone = None
            else:
                self._next_fetchone = (self.pool.state_row,)
            return
        if "INSERT INTO sda_state" in s and params:
            value = params[1] if len(params) >= 2 else None
            value_obj = getattr(value, "obj", value)
            if isinstance(value_obj, str):
                try:
                    value_obj = json.loads(value_obj)
                except (TypeError, ValueError):
                    pass
            if isinstance(value_obj, dict):
                self.pool.state_row = value_obj
                self.pool.saved_states.append(value_obj)
            return
        if "FROM audit_log" in s and "COUNT(*)" in s and "offline_conversions" in s:
            if self.pool.offline_raises is not None:
                raise self.pool.offline_raises
            self._next_fetchone = self.pool.offline_counts
            return
        if "INSERT INTO audit_log" in s and params:
            if len(params) >= 10:
                self.pool.audit_log_inserts.append(
                    {
                        "hypothesis_id": params[0],
                        "trust_level": params[1],
                        "tool_name": params[2],
                        "is_mutation": params[5],
                    }
                )
            self._next_fetchone = (1,)
            return
        # Any other statement — default no-op.

    async def fetchone(self) -> Any:
        out = self._next_fetchone
        self._next_fetchone = None
        return out


# ------------------------------------------------------------ direct stub


def _tsv(rows: list[tuple[str, float, float]]) -> str:
    """Build a minimal CAMPAIGN_PERFORMANCE_REPORT-like TSV.

    Columns mirror the real Direct output (``Date\\tCost\\tConversions``)
    plus a title / total sandwich that ``_parse_tsv_column`` skips. ``cost``
    in the tuple is expressed in RUBLES for readability — we convert to
    micro-rubles here to match Direct's wire format.
    """
    header = "ReportName\nDate\tCost\tConversions"
    body = "\n".join(f"{d}\t{int(c * 1_000_000)}\t{int(conv)}" for d, c, conv in rows)
    total = "Total\t0\t0"
    return f"{header}\n{body}\n{total}"


def _direct_stub(tsv: str) -> SimpleNamespace:
    return SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={"tsv": tsv}),
    )


# ------------------------------------------------------------ signal: won_30d


@pytest.mark.asyncio
async def test_won_30d_counts_only_whitelisted_utm() -> None:
    deals = [{"ID": f"{i}", "LEAD_ID": f"{i}", "OPPORTUNITY": 5000} for i in range(1, 21)]
    leads = [{"ID": f"{i}", "UTM_CAMPAIGN": "bfl-rf" if i <= 8 else "other"} for i in range(1, 21)]
    with (
        patch.object(
            strategy_gate.bitrix_tools,
            "get_deal_list",
            AsyncMock(return_value=deals),
        ),
        patch.object(
            strategy_gate.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=leads),
        ),
    ):
        res = await strategy_gate._count_won_30d(MagicMock(), _settings())
    assert res.count == 8
    assert res.revenue_rub == 8 * 5000
    assert res.met is False  # 8 < 10


@pytest.mark.asyncio
async def test_won_30d_green_when_over_threshold() -> None:
    deals = [{"ID": f"{i}", "LEAD_ID": f"{i}", "OPPORTUNITY": 3000} for i in range(1, 13)]
    leads = [{"ID": f"{i}", "UTM_CAMPAIGN": "bfl-rf"} for i in range(1, 13)]
    with (
        patch.object(strategy_gate.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals)),
        patch.object(strategy_gate.bitrix_tools, "get_lead_list", AsyncMock(return_value=leads)),
    ):
        res = await strategy_gate._count_won_30d(MagicMock(), _settings())
    assert res.count == 12
    assert res.met is True


@pytest.mark.asyncio
async def test_won_30d_handles_bitrix_exception() -> None:
    with patch.object(
        strategy_gate.bitrix_tools,
        "get_deal_list",
        AsyncMock(side_effect=RuntimeError("bitrix 500")),
    ):
        res = await strategy_gate._count_won_30d(MagicMock(), _settings())
    assert res.count == 0
    assert res.met is False
    assert "bitrix 500" in res.reason


# ------------------------------------------------------ signal: cpa_stability


@pytest.mark.asyncio
async def test_cpa_stability_green_with_low_variance() -> None:
    # (cost, conv) per day → CPA per day. 7 days around 1000₽, small spread.
    rows = [
        ("2026-04-18", 10000, 10),  # cpa 1000
        ("2026-04-19", 11000, 10),  # 1100
        ("2026-04-20", 9000, 10),  # 900
        ("2026-04-21", 10500, 10),  # 1050
        ("2026-04-22", 9500, 10),  # 950
        ("2026-04-23", 10000, 10),  # 1000
        ("2026-04-24", 10000, 10),  # 1000
    ]
    # Each PROTECTED campaign returns the same TSV; stability uses aggregate.
    direct = _direct_stub(_tsv(rows))
    res = await strategy_gate._cpa_stability_7d(direct, _settings())
    assert res.stable is True
    assert res.days == 7
    assert res.mean == pytest.approx(1000, rel=0.01)


@pytest.mark.asyncio
async def test_cpa_stability_too_few_days() -> None:
    rows = [("2026-04-23", 10000, 10), ("2026-04-24", 11000, 10)]
    direct = _direct_stub(_tsv(rows))
    res = await strategy_gate._cpa_stability_7d(direct, _settings())
    assert res.stable is False
    assert res.reason == "too_few_days"
    assert res.days == 2


@pytest.mark.asyncio
async def test_cpa_stability_skips_days_with_zero_conversions() -> None:
    rows = [
        ("2026-04-22", 10000, 0),  # skipped
        ("2026-04-23", 10000, 10),
        ("2026-04-24", 10000, 10),
    ]
    direct = _direct_stub(_tsv(rows))
    res = await strategy_gate._cpa_stability_7d(direct, _settings())
    # Only 2 valid days → too_few_days.
    assert res.stable is False
    assert res.days == 2


# ----------------------------------------------------- signal: offline_conv


@pytest.mark.asyncio
async def test_offline_conversions_green() -> None:
    pool = _FakePool(offline_counts=(6, 0))
    res = await strategy_gate._offline_conversions_ok(pool)
    assert res.met is True
    assert res.total == 6
    assert res.errors == 0


@pytest.mark.asyncio
async def test_offline_conversions_fails_on_errors() -> None:
    pool = _FakePool(offline_counts=(5, 1))
    res = await strategy_gate._offline_conversions_ok(pool)
    assert res.met is False
    assert "errors=1" in res.reason


@pytest.mark.asyncio
async def test_offline_conversions_fails_on_few_runs() -> None:
    pool = _FakePool(offline_counts=(3, 0))
    res = await strategy_gate._offline_conversions_ok(pool)
    assert res.met is False
    assert "too_few_runs" in res.reason


@pytest.mark.asyncio
async def test_offline_conversions_swallows_db_error() -> None:
    pool = _FakePool(offline_raises=RuntimeError("db down"))
    res = await strategy_gate._offline_conversions_ok(pool)
    assert res.met is False
    assert "db down" in res.reason


# --------------------------------------------------- signal: direct_conversions


@pytest.mark.asyncio
async def test_direct_conversions_sums_across_campaigns() -> None:
    # 3 PROTECTED campaigns × 8 conversions each = 24.
    rows = [("2026-04-22", 0, 8)]
    direct = _direct_stub(_tsv(rows))
    res = await strategy_gate._direct_conversions_accumulated(direct, _settings())
    assert res.count == 24
    assert res.met is True


@pytest.mark.asyncio
async def test_direct_conversions_below_threshold() -> None:
    # 3 × 5 = 15 < 20.
    rows = [("2026-04-22", 0, 5)]
    direct = _direct_stub(_tsv(rows))
    res = await strategy_gate._direct_conversions_accumulated(direct, _settings())
    assert res.count == 15
    assert res.met is False


@pytest.mark.asyncio
async def test_direct_conversions_campaign_error_tolerated() -> None:
    """One failing campaign shouldn't void the whole signal."""
    call_count = {"n": 0}

    async def fake_get_stats(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("direct 500")
        return {"tsv": _tsv([("2026-04-22", 0, 10)])}

    direct = SimpleNamespace(get_campaign_stats=fake_get_stats)
    res = await strategy_gate._direct_conversions_accumulated(direct, _settings())
    # Two successful campaigns × 10 = 20 — still meets threshold.
    assert res.count == 20


# --------------------------------------------------------- state machine


def test_decide_transition_learning_to_ready_all_green() -> None:
    signals = {
        "won_30d": {"met": True, "value": 15, "required": 10},
        "cpa_stability_7d": {"met": True},
        "offline_conversions": {"met": True},
        "direct_conversions": {"met": True},
    }
    new, reason = strategy_gate._decide_transition("learning", signals, None, 0.0)
    assert new == "ready_to_switch"
    assert "4 maturity" in reason


def test_decide_transition_learning_stays_when_incomplete() -> None:
    signals = {
        "won_30d": {"met": False, "value": 5, "required": 10},
        "cpa_stability_7d": {"met": True},
        "offline_conversions": {"met": True},
        "direct_conversions": {"met": True},
    }
    new, reason = strategy_gate._decide_transition("learning", signals, None, 0.0)
    assert new == "learning"
    assert reason == ""


def test_decide_transition_ready_reverts_when_won_drops() -> None:
    signals = {
        "won_30d": {"met": False, "value": 7, "required": 10},
        "cpa_stability_7d": {"met": True},
        "offline_conversions": {"met": True},
        "direct_conversions": {"met": True},
    }
    new, reason = strategy_gate._decide_transition("ready_to_switch", signals, None, 0.0)
    assert new == "learning"
    assert "WON regressed" in reason


def test_decide_transition_auto_pilot_degrades_on_cpa_growth() -> None:
    signals = {
        "won_30d": {"met": True},
        "cpa_stability_7d": {"met": True},
        "offline_conversions": {"met": True},
        "direct_conversions": {"met": True},
    }
    # baseline=1000, current=1600 → 1.6× > 1.5×
    new, reason = strategy_gate._decide_transition(
        "auto_pilot", signals, baseline_cpa=1000.0, current_cpa=1600.0
    )
    assert new == "degraded"
    assert "baseline" in reason


def test_decide_transition_auto_pilot_stable_within_threshold() -> None:
    signals = {
        "won_30d": {"met": True},
        "cpa_stability_7d": {"met": True},
        "offline_conversions": {"met": True},
        "direct_conversions": {"met": True},
    }
    new, _ = strategy_gate._decide_transition(
        "auto_pilot", signals, baseline_cpa=1000.0, current_cpa=1400.0
    )
    assert new == "auto_pilot"


def test_decide_transition_degraded_is_sticky() -> None:
    # Even if every signal turns green, degraded requires manual_switch.
    signals = {
        "won_30d": {"met": True},
        "cpa_stability_7d": {"met": True},
        "offline_conversions": {"met": True},
        "direct_conversions": {"met": True},
    }
    new, _ = strategy_gate._decide_transition("degraded", signals, 1000.0, 900.0)
    assert new == "degraded"


# --------------------------------------------------------- run() happy path


def _make_direct_for_green_run() -> SimpleNamespace:
    # Stable 7d CPA ~1000 with 10 conversions/day = also gives 7×10×3
    # campaigns = 210 direct conversions. Comfortably green.
    rows = [
        ("2026-04-18", 10000, 10),
        ("2026-04-19", 11000, 10),
        ("2026-04-20", 9000, 10),
        ("2026-04-21", 10500, 10),
        ("2026-04-22", 9500, 10),
        ("2026-04-23", 10000, 10),
        ("2026-04-24", 10000, 10),
    ]
    return _direct_stub(_tsv(rows))


@pytest.mark.asyncio
async def test_run_learning_to_ready_all_signals_green() -> None:
    pool = _FakePool(
        initial_state={
            "status": "learning",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": None,
        },
        offline_counts=(6, 0),
    )
    deals = [{"ID": f"{i}", "LEAD_ID": f"{i}", "OPPORTUNITY": 5000} for i in range(1, 13)]
    leads = [{"ID": f"{i}", "UTM_CAMPAIGN": "bfl-rf"} for i in range(1, 13)]

    send = AsyncMock(return_value=1)
    with (
        patch.object(
            strategy_gate.bitrix_tools,
            "get_deal_list",
            AsyncMock(return_value=deals),
        ),
        patch.object(
            strategy_gate.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=leads),
        ),
        patch.object(strategy_gate.telegram_tools, "send_message", send),
    ):
        result = await strategy_gate.run(
            pool,
            direct=_make_direct_for_green_run(),
            http_client=MagicMock(),
            settings=_settings(),
        )
    assert result["status"] == "ready_to_switch"
    assert result["changed"] is True
    assert result["notified"] is True
    send.assert_awaited_once()
    assert "learning" in send.await_args.kwargs["text"]
    assert "ready_to_switch" in send.await_args.kwargs["text"]
    # State row persisted with new status.
    assert pool.state_row is not None
    assert pool.state_row["status"] == "ready_to_switch"
    # History has the transition record.
    assert len(pool.state_row["history"]) == 1
    assert pool.state_row["history"][0]["to"] == "ready_to_switch"


@pytest.mark.asyncio
async def test_run_dry_run_does_not_mutate_or_notify() -> None:
    pool = _FakePool(
        initial_state={
            "status": "learning",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": None,
        },
        offline_counts=(6, 0),
    )
    deals = [{"ID": f"{i}", "LEAD_ID": f"{i}", "OPPORTUNITY": 5000} for i in range(1, 13)]
    leads = [{"ID": f"{i}", "UTM_CAMPAIGN": "bfl-rf"} for i in range(1, 13)]

    send = AsyncMock(return_value=1)
    with (
        patch.object(
            strategy_gate.bitrix_tools,
            "get_deal_list",
            AsyncMock(return_value=deals),
        ),
        patch.object(
            strategy_gate.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=leads),
        ),
        patch.object(strategy_gate.telegram_tools, "send_message", send),
    ):
        result = await strategy_gate.run(
            pool,
            dry_run=True,
            direct=_make_direct_for_green_run(),
            http_client=MagicMock(),
            settings=_settings(),
        )
    assert result["dry_run"] is True
    assert result["changed"] is False
    assert result["proposed_status"] == "ready_to_switch"
    assert result["status"] == "learning"  # unchanged
    send.assert_not_awaited()
    # State row unchanged (no save).
    assert pool.state_row is not None
    assert pool.state_row["status"] == "learning"
    assert pool.saved_states == []


@pytest.mark.asyncio
async def test_run_auto_pilot_detects_degradation() -> None:
    # baseline 1000, current spiking to ~1600 via skewed costs.
    pool = _FakePool(
        initial_state={
            "status": "auto_pilot",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": 1000.0,
        },
        offline_counts=(6, 0),
    )
    rows = [
        ("2026-04-18", 16000, 10),
        ("2026-04-19", 16200, 10),
        ("2026-04-20", 15800, 10),
        ("2026-04-21", 16000, 10),
        ("2026-04-22", 16000, 10),
        ("2026-04-23", 16000, 10),
        ("2026-04-24", 16000, 10),
    ]
    direct = _direct_stub(_tsv(rows))
    deals = [{"ID": f"{i}", "LEAD_ID": f"{i}", "OPPORTUNITY": 5000} for i in range(1, 13)]
    leads = [{"ID": f"{i}", "UTM_CAMPAIGN": "bfl-rf"} for i in range(1, 13)]
    send = AsyncMock(return_value=1)
    with (
        patch.object(
            strategy_gate.bitrix_tools,
            "get_deal_list",
            AsyncMock(return_value=deals),
        ),
        patch.object(
            strategy_gate.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=leads),
        ),
        patch.object(strategy_gate.telegram_tools, "send_message", send),
    ):
        result = await strategy_gate.run(
            pool,
            direct=direct,
            http_client=MagicMock(),
            settings=_settings(),
        )
    assert result["status"] == "degraded"
    assert result["changed"] is True
    send.assert_awaited_once()
    assert "CRITICAL" in send.await_args.kwargs["text"]
    assert pool.state_row is not None
    assert pool.state_row["status"] == "degraded"


@pytest.mark.asyncio
async def test_run_degraded_stays_degraded_even_on_green_signals() -> None:
    pool = _FakePool(
        initial_state={
            "status": "degraded",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [{"to": "degraded", "ts": "2026-04-20T00:00:00+00:00"}],
            "autopilot_baseline_cpa": 1000.0,
        },
        offline_counts=(6, 0),
    )
    deals = [{"ID": f"{i}", "LEAD_ID": f"{i}", "OPPORTUNITY": 5000} for i in range(1, 13)]
    leads = [{"ID": f"{i}", "UTM_CAMPAIGN": "bfl-rf"} for i in range(1, 13)]
    send = AsyncMock(return_value=1)
    with (
        patch.object(
            strategy_gate.bitrix_tools,
            "get_deal_list",
            AsyncMock(return_value=deals),
        ),
        patch.object(
            strategy_gate.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=leads),
        ),
        patch.object(strategy_gate.telegram_tools, "send_message", send),
    ):
        result = await strategy_gate.run(
            pool,
            direct=_make_direct_for_green_run(),
            http_client=MagicMock(),
            settings=_settings(),
        )
    assert result["status"] == "degraded"
    assert result["changed"] is False
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_one_signal_failing_others_still_evaluate() -> None:
    """asyncio.gather(return_exceptions=True): one crash, others proceed."""
    pool = _FakePool(
        initial_state={
            "status": "learning",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": None,
        },
        offline_counts=(6, 0),
    )
    send = AsyncMock(return_value=1)
    with (
        patch.object(
            strategy_gate.bitrix_tools,
            "get_deal_list",
            AsyncMock(side_effect=RuntimeError("bitrix 500")),
        ),
        patch.object(strategy_gate.telegram_tools, "send_message", send),
    ):
        result = await strategy_gate.run(
            pool,
            direct=_make_direct_for_green_run(),
            http_client=MagicMock(),
            settings=_settings(),
        )
    # won_30d failed; others still ran. Transition does NOT fire.
    assert result["status"] == "learning"
    assert result["signals"]["won_30d"]["met"] is False
    # Other signals still got evaluated.
    assert result["signals"]["cpa_stability_7d"]["met"] in (True, False)
    assert result["signals"]["offline_conversions"]["met"] is True


@pytest.mark.asyncio
async def test_run_telegram_failure_does_not_block_transition() -> None:
    pool = _FakePool(
        initial_state={
            "status": "learning",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": None,
        },
        offline_counts=(6, 0),
    )
    deals = [{"ID": f"{i}", "LEAD_ID": f"{i}", "OPPORTUNITY": 5000} for i in range(1, 13)]
    leads = [{"ID": f"{i}", "UTM_CAMPAIGN": "bfl-rf"} for i in range(1, 13)]
    send = AsyncMock(side_effect=RuntimeError("telegram 500"))
    with (
        patch.object(
            strategy_gate.bitrix_tools,
            "get_deal_list",
            AsyncMock(return_value=deals),
        ),
        patch.object(
            strategy_gate.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=leads),
        ),
        patch.object(strategy_gate.telegram_tools, "send_message", send),
    ):
        result = await strategy_gate.run(
            pool,
            direct=_make_direct_for_green_run(),
            http_client=MagicMock(),
            settings=_settings(),
        )
    # State still transitioned despite Telegram failure.
    assert result["status"] == "ready_to_switch"
    assert result["changed"] is True
    assert result["notified"] is False
    assert pool.state_row is not None
    assert pool.state_row["status"] == "ready_to_switch"


@pytest.mark.asyncio
async def test_run_degraded_noop_when_di_missing() -> None:
    pool = _FakePool(
        initial_state={
            "status": "learning",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": None,
            "signals": {},
        }
    )
    # Called with default DI (pool + dry_run only) — the JOB_REGISTRY path.
    result = await strategy_gate.run(pool)
    assert result["status"] == "learning"
    assert result["changed"] is False
    assert result["reason"] == "degraded_noop_di_missing"
    # State unchanged.
    assert pool.saved_states == []


# --------------------------------------------------------- manual_switch


@pytest.mark.asyncio
async def test_manual_switch_invalid_status_returns_error() -> None:
    pool = _FakePool(
        initial_state={
            "status": "learning",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": None,
        }
    )
    result = await strategy_gate.manual_switch(pool, "nonsense")
    assert result["ok"] is False
    assert "nonsense" in result["error"]
    assert pool.saved_states == []


@pytest.mark.asyncio
async def test_manual_switch_auto_pilot_snapshots_baseline() -> None:
    pool = _FakePool(
        initial_state={
            "status": "ready_to_switch",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": None,
        }
    )
    # Direct returns 7 days of CPA ≈ 1000.
    direct = _make_direct_for_green_run()
    send = AsyncMock(return_value=1)
    with patch.object(strategy_gate.telegram_tools, "send_message", send):
        result = await strategy_gate.manual_switch(
            pool,
            "auto_pilot",
            direct=direct,
            http_client=MagicMock(),
            settings=_settings(),
        )
    assert result["ok"] is True
    assert result["to"] == "auto_pilot"
    assert result["autopilot_baseline_cpa"] == pytest.approx(1000, rel=0.02)
    assert pool.state_row is not None
    assert pool.state_row["status"] == "auto_pilot"
    assert pool.state_row["autopilot_baseline_cpa"] == pytest.approx(1000, rel=0.02)
    assert pool.state_row["history"][-1]["reason"] == "manual_switch"


@pytest.mark.asyncio
async def test_manual_switch_to_learning_clears_baseline() -> None:
    pool = _FakePool(
        initial_state={
            "status": "degraded",
            "entered_at": "2026-04-01T00:00:00+00:00",
            "history": [],
            "autopilot_baseline_cpa": 1234.0,
        }
    )
    result = await strategy_gate.manual_switch(pool, "learning")
    assert result["ok"] is True
    assert pool.state_row is not None
    assert pool.state_row["status"] == "learning"
    assert pool.state_row["autopilot_baseline_cpa"] is None


# ------------------------------------------------------------ format_section


def test_format_section_renders_key_fields() -> None:
    state_eval = {
        "status": "ready_to_switch",
        "reason": "all 4 maturity signals green",
        "signals": {
            "won_30d": {"value": 12, "required": 10, "met": True},
            "cpa_stability_7d": {"value": 0.12, "met": True},
            "offline_conversions": {"value": 6, "met": True},
            "direct_conversions": {"value": 24, "required": 20, "met": True},
        },
        "autopilot_baseline_cpa": None,
    }
    text = strategy_gate.format_section(state_eval)
    assert "Strategy Gate" in text
    assert "ready_to_switch" in text
    assert "12/10" in text
    assert "24/20" in text


def test_format_section_includes_baseline_when_present() -> None:
    text = strategy_gate.format_section(
        {
            "status": "auto_pilot",
            "reason": "",
            "signals": {},
            "autopilot_baseline_cpa": 1234.5,
        }
    )
    assert "baseline_cpa=1234" in text


# ---------------------------------------------------------- transition alert


def test_transition_alert_critical_for_degraded() -> None:
    signals = {
        "won_30d": {"value": 15, "required": 10, "met": True, "revenue_rub": 45000},
        "cpa_stability_7d": {"value": 0.1, "met": True, "days": 7, "mean": 1000},
        "offline_conversions": {"value": 6, "met": True, "errors": 0},
        "direct_conversions": {"value": 30, "required": 20, "met": True},
    }
    text = strategy_gate._format_transition_alert("auto_pilot", "degraded", "cpa spike", signals)
    assert "CRITICAL" in text
    assert "cpa spike" in text
    assert "⚠️" in text


def test_transition_alert_info_for_ready_to_switch() -> None:
    signals = {
        "won_30d": {"value": 15, "required": 10, "met": True, "revenue_rub": 45000},
        "cpa_stability_7d": {"value": 0.1, "met": True, "days": 7, "mean": 1000},
        "offline_conversions": {"value": 6, "met": True, "errors": 0},
        "direct_conversions": {"value": 30, "required": 20, "met": True},
    }
    text = strategy_gate._format_transition_alert(
        "learning", "ready_to_switch", "all 4 green", signals
    )
    assert "INFO" in text
    assert "WB_MAXIMUM_CONVERSION_RATE" in text


# ------------------------------------------------------ job registry import


@pytest.mark.asyncio
async def test_run_importable_and_callable_as_job_entry() -> None:
    """Smoke: strategy_gate.run is the JOB_REGISTRY-compatible signature.

    We verify the function can be called with the default JOB_REGISTRY
    dispatch shape (pool + dry_run) without raising — not that the
    registry itself contains it (that's integration work).
    """
    pool = _FakePool(initial_state={"status": "learning", "history": []})
    result = await strategy_gate.run(pool, dry_run=True)
    # No DI → degraded path.
    assert result["status"] == "learning"
    assert "reason" in result


# ------------------------------------------------------ tsv parsing edge


def test_parse_tsv_column_handles_total_and_empty_cells() -> None:
    tsv = (
        "ReportName\n"
        "Date\tCost\tConversions\n"
        "2026-04-22\t10000\t--\n"
        "2026-04-23\t11000\t5\n"
        "Total\t21000\t5"
    )
    rows = strategy_gate._parse_tsv_column(tsv, "Conversions")
    # "--" coerces to 0.0, Total row skipped.
    assert rows == [("2026-04-22", 0.0), ("2026-04-23", 5.0)]


def test_parse_tsv_column_missing_column_returns_empty() -> None:
    tsv = "Date\tImpressions\n2026-04-22\t100"
    assert strategy_gate._parse_tsv_column(tsv, "Cost") == []


# ------------------------------------------------------ state coercion


def test_coerce_state_reset_on_unknown_status() -> None:
    out = strategy_gate._coerce_state({"status": "not_a_real_status"})
    assert out["status"] == "learning"
    assert out["history"] == []


def test_coerce_state_fills_missing_keys() -> None:
    out = strategy_gate._coerce_state({"status": "learning"})
    assert "entered_at" in out
    assert out["history"] == []
    assert out["autopilot_baseline_cpa"] is None


def test_coerce_state_parses_json_string() -> None:
    raw = json.dumps({"status": "auto_pilot", "autopilot_baseline_cpa": 1500})
    out = strategy_gate._coerce_state(raw)
    assert out["status"] == "auto_pilot"
    assert out["autopilot_baseline_cpa"] == 1500


# ------------------------------------------------------ watchdog interval


def test_watchdog_interval_long_for_auto_pilot() -> None:
    assert strategy_gate._watchdog_interval_for("auto_pilot") == 240


def test_watchdog_interval_short_for_learning() -> None:
    assert strategy_gate._watchdog_interval_for("learning") == 15
    assert strategy_gate._watchdog_interval_for("degraded") == 15
