"""Unit tests for agent_runtime.jobs.bfl_rf_watchdog (Task 20).

Uses a stateful ``_FakePool`` that interprets the narrow SQL subset the
watchdog issues against ``sda_state`` (SELECT / INSERT-ON-CONFLICT for the
three keys: strategy_gate_state, bfl_rf_watchdog_cooldowns,
bfl_rf_watchdog_last_run). Any unrelated SQL is accepted as a no-op so
we can extend without retooling.

Coverage targets:

* 11 threshold env overrides — one spot-check per critical threshold is enough
  since they all pass through the same ``_th`` helper.
* 7 alert types — one parametric case per active trigger, plus the
  "no alerts when everything is nominal" case.
* Cooldown semantics — ``_in_cooldown`` 6h window; second tick inside
  window skips; expired entry fires; missing entry does not block.
* Phase machine — learning always runs; auto_pilot with ``<4h`` throttles;
  auto_pilot with ``>4h`` runs.
* dry_run — no Telegram, no sda_state writes, but alerts_active reflects
  the evaluated thresholds.
* Degraded no-op when DI missing.
* Tracker exception propagates as ``status=error`` (never crashes).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import bfl_rf_watchdog

_MSK = timezone(timedelta(hours=3))


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


# --------------------------------------------------------------- _FakePool


class _FakePool:
    """In-memory sda_state shim."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        # Stored form is the JSON-string (mirrors what psycopg passes to ::jsonb).
        self.state: dict[str, str] = {}
        if state:
            for k, v in state.items():
                self.state[k] = v if isinstance(v, str) else json.dumps(v)
        self.upserts: list[tuple[str, dict[str, Any]]] = []

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
        self._fetchone_result: Any = None

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        stripped = sql.strip()
        if stripped.startswith("SELECT value FROM sda_state") and params:
            key = params[0]
            stored = self.pool.state.get(key)
            if stored is None:
                self._fetchone_result = None
            else:
                # psycopg returns jsonb as dict; we keep strings for ease of
                # equality. The production _coerce_jsonb handles either.
                self._fetchone_result = (stored,)
            return
        if stripped.startswith("INSERT INTO sda_state") and params:
            key, payload = params[0], params[1]
            self.pool.state[key] = payload
            try:
                parsed = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                parsed = payload
            self.pool.upserts.append((key, parsed))
            return
        # Any other SQL — accept silently.

    async def fetchone(self) -> Any:
        out = self._fetchone_result
        self._fetchone_result = None
        return out


# ---------------------------------------------------------------- _check_all


def _default_thresholds() -> dict[str, float]:
    return bfl_rf_watchdog.build_thresholds()


def _tracker_snapshot(**overrides: Any) -> dict[str, Any]:
    """Assemble a tracker-shaped dict with nominal healthy defaults.

    Defaults sit safely inside all 7 alert thresholds:
    CTR=10% (>3), bounce=40% (<70), CR click→lead = 50/500 = 10% (>2.5),
    CPA lead = 1000/50 = 20₽ (<2700), CPA won = 1000/1 = 1000₽ (<55000),
    avg_time=120s (>45), leads=50 (>=3), won_deals=1.
    """
    base: dict[str, Any] = {
        "direct": {"impressions": 10000, "clicks": 500, "cost": 1000.0, "ctr": 10.0},
        "metrika": {"visits": 400, "bounce_rate": 40.0, "avg_duration_s": 120.0},
        "bitrix": {"leads": 50, "deals": {"stages": {"won": 1}}},
        "economics": {
            "cost": 1000.0,
            "leads": 50,
            "won_deals": 1,
            "cpa_lead": 20.0,
            "cpa_won": 1000.0,
        },
    }
    for layer, patch_ in overrides.items():
        base[layer] = {**base[layer], **patch_} if isinstance(patch_, dict) else patch_
    return base


def test_no_alerts_when_everything_nominal() -> None:
    alerts = bfl_rf_watchdog._check_all(_tracker_snapshot(), _default_thresholds())
    assert alerts == []


def test_ctr_low_triggers_with_noise_floor() -> None:
    data = _tracker_snapshot(direct={"impressions": 5000, "clicks": 100, "ctr": 2.0, "cost": 500.0})
    alerts = bfl_rf_watchdog._check_all(data, _default_thresholds())
    assert any(a["type"] == "ctr_low" for a in alerts)


def test_ctr_low_suppressed_when_impressions_below_noise() -> None:
    data = _tracker_snapshot(direct={"impressions": 500, "clicks": 10, "ctr": 2.0, "cost": 50.0})
    alerts = bfl_rf_watchdog._check_all(data, _default_thresholds())
    assert not any(a["type"] == "ctr_low" for a in alerts)


def test_bounce_high_triggers() -> None:
    data = _tracker_snapshot(metrika={"visits": 200, "bounce_rate": 75.0, "avg_duration_s": 120})
    alerts = bfl_rf_watchdog._check_all(data, _default_thresholds())
    assert any(a["type"] == "bounce_high" for a in alerts)


def test_bounce_high_suppressed_below_visits_floor() -> None:
    data = _tracker_snapshot(metrika={"visits": 50, "bounce_rate": 90.0, "avg_duration_s": 120})
    alerts = bfl_rf_watchdog._check_all(data, _default_thresholds())
    assert not any(a["type"] == "bounce_high" for a in alerts)


def test_zero_leads_critical_triggers() -> None:
    data = _tracker_snapshot(
        direct={"impressions": 10000, "clicks": 400, "ctr": 4.0, "cost": 1500.0},
        bitrix={"leads": 0, "deals": {"stages": {"won": 0}}},
        economics={"cost": 1500.0, "leads": 0, "won_deals": 0, "cpa_lead": 0.0, "cpa_won": 0.0},
    )
    alerts = bfl_rf_watchdog._check_all(data, _default_thresholds())
    zero = next(a for a in alerts if a["type"] == "zero_leads")
    assert zero["severity"] == "🔴"


def test_cpa_lead_respects_min_leads_floor() -> None:
    th = _default_thresholds()
    # 2 leads → below min_leads=3 → no alert even with CPA over cap
    data = _tracker_snapshot(
        bitrix={"leads": 2, "deals": {"stages": {"won": 0}}},
        economics={"cost": 8000.0, "leads": 2, "won_deals": 0, "cpa_lead": 4000.0, "cpa_won": 0.0},
    )
    alerts = bfl_rf_watchdog._check_all(data, th)
    assert not any(a["type"] == "cpa_lead_high" for a in alerts)

    # 3 leads — triggers
    data = _tracker_snapshot(
        bitrix={"leads": 3, "deals": {"stages": {"won": 0}}},
        economics={"cost": 12000.0, "leads": 3, "won_deals": 0, "cpa_lead": 4000.0, "cpa_won": 0.0},
    )
    alerts = bfl_rf_watchdog._check_all(data, th)
    assert any(a["type"] == "cpa_lead_high" for a in alerts)


def test_avg_time_low_respects_visits_floor() -> None:
    th = _default_thresholds()
    # 50 visits — below 100 noise floor
    data = _tracker_snapshot(metrika={"visits": 50, "bounce_rate": 30, "avg_duration_s": 20})
    assert not any(a["type"] == "avg_time_low" for a in bfl_rf_watchdog._check_all(data, th))
    # 200 visits — triggers
    data = _tracker_snapshot(metrika={"visits": 200, "bounce_rate": 30, "avg_duration_s": 20})
    assert any(a["type"] == "avg_time_low" for a in bfl_rf_watchdog._check_all(data, th))


def test_cr_low_critical_triggers_with_enough_clicks() -> None:
    data = _tracker_snapshot(
        direct={"impressions": 10000, "clicks": 500, "ctr": 5.0, "cost": 5000.0},
        bitrix={"leads": 5, "deals": {"stages": {"won": 0}}},
        economics={"cost": 5000.0, "leads": 5, "won_deals": 0, "cpa_lead": 1000.0, "cpa_won": 0},
    )
    alerts = bfl_rf_watchdog._check_all(data, _default_thresholds())
    cr = next(a for a in alerts if a["type"] == "cr_low")
    assert cr["severity"] == "🔴"


def test_cpa_won_high_triggers_when_deals_won() -> None:
    data = _tracker_snapshot(
        bitrix={"leads": 5, "deals": {"stages": {"won": 2}}},
        economics={
            "cost": 140000.0,
            "leads": 5,
            "won_deals": 2,
            "cpa_lead": 28000.0,
            "cpa_won": 70000.0,
        },
    )
    alerts = bfl_rf_watchdog._check_all(data, _default_thresholds())
    assert any(a["type"] == "cpa_won_high" for a in alerts)


def test_cpa_won_not_triggered_without_won_deals() -> None:
    data = _tracker_snapshot(
        bitrix={"leads": 5, "deals": {"stages": {"won": 0}}},
        economics={
            "cost": 140000.0,
            "leads": 5,
            "won_deals": 0,
            "cpa_lead": 28000.0,
            "cpa_won": 999999.0,
        },
    )
    alerts = bfl_rf_watchdog._check_all(data, _default_thresholds())
    assert not any(a["type"] == "cpa_won_high" for a in alerts)


# ------------------------------------------------------------- threshold env


def test_thresholds_default_values_match_v2() -> None:
    th = bfl_rf_watchdog.build_thresholds()
    assert th["bounce_max"] == 70.0
    assert th["bounce_min_visits"] == 100.0
    assert th["ctr_min"] == 3.0
    assert th["ctr_min_impressions"] == 3000.0
    assert th["leads_zero_clicks"] == 300.0
    assert th["cpa_lead_max"] == 2700.0
    assert th["cpa_lead_min_leads"] == 3.0
    assert th["cpa_won_max"] == 55000.0
    assert th["avg_time_min"] == 45.0
    assert th["cr_click_lead_min"] == 2.5
    assert th["cr_min_clicks"] == 100.0


def test_all_eleven_thresholds_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {
        "BFL_RF_TH_BOUNCE": "55",
        "BFL_RF_TH_BOUNCE_MIN_V": "150",
        "BFL_RF_TH_CTR": "5.5",
        "BFL_RF_TH_CTR_MIN_I": "5000",
        "BFL_RF_TH_ZERO_LEADS_CLICKS": "400",
        "BFL_RF_TH_CPA_LEAD": "3000",
        "BFL_RF_TH_CPA_LEAD_MIN_LEADS": "5",
        "BFL_RF_TH_CPA_WON": "60000",
        "BFL_RF_TH_AVG_TIME": "60",
        "BFL_RF_TH_CR_CL_MIN": "4.0",
        "BFL_RF_TH_CR_MIN_CLICKS": "200",
    }
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    th = bfl_rf_watchdog.build_thresholds()
    assert th["bounce_max"] == 55.0
    assert th["bounce_min_visits"] == 150.0
    assert th["ctr_min"] == 5.5
    assert th["ctr_min_impressions"] == 5000.0
    assert th["leads_zero_clicks"] == 400.0
    assert th["cpa_lead_max"] == 3000.0
    assert th["cpa_lead_min_leads"] == 5.0
    assert th["cpa_won_max"] == 60000.0
    assert th["avg_time_min"] == 60.0
    assert th["cr_click_lead_min"] == 4.0
    assert th["cr_min_clicks"] == 200.0


def test_threshold_garbage_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BFL_RF_TH_CTR", "not-a-number")
    th = bfl_rf_watchdog.build_thresholds()
    assert th["ctr_min"] == 3.0


# --------------------------------------------------------------- cooldown


def test_in_cooldown_missing_key_returns_false() -> None:
    assert bfl_rf_watchdog._in_cooldown({}, "ctr_low") is False


def test_in_cooldown_recent_blocks() -> None:
    recent = (datetime.now(_MSK) - timedelta(hours=2)).isoformat()
    assert bfl_rf_watchdog._in_cooldown({"ctr_low": recent}, "ctr_low") is True


def test_in_cooldown_expired_allows() -> None:
    old = (datetime.now(_MSK) - timedelta(hours=7)).isoformat()
    assert bfl_rf_watchdog._in_cooldown({"ctr_low": old}, "ctr_low") is False


def test_in_cooldown_invalid_iso_returns_false() -> None:
    assert bfl_rf_watchdog._in_cooldown({"ctr_low": "not-iso"}, "ctr_low") is False


# --------------------------------------------------------- _coerce_jsonb


def test_coerce_jsonb_passes_dict() -> None:
    assert bfl_rf_watchdog._coerce_jsonb({"a": 1}) == {"a": 1}


def test_coerce_jsonb_parses_json_string() -> None:
    assert bfl_rf_watchdog._coerce_jsonb('{"a": 1}') == {"a": 1}


def test_coerce_jsonb_corrupt_returns_empty() -> None:
    assert bfl_rf_watchdog._coerce_jsonb("{bad json") == {}


def test_coerce_jsonb_none_returns_empty() -> None:
    assert bfl_rf_watchdog._coerce_jsonb(None) == {}


def test_coerce_jsonb_non_dict_json_returns_empty() -> None:
    assert bfl_rf_watchdog._coerce_jsonb("[1,2,3]") == {}


# --------------------------------------------------------- throttle logic


def test_should_throttle_learning_false() -> None:
    recent = {"ts": (datetime.now(_MSK) - timedelta(minutes=10)).isoformat()}
    assert bfl_rf_watchdog._should_throttle("learning", recent) is False


def test_should_throttle_auto_pilot_recent_true() -> None:
    recent = {"ts": (datetime.now(_MSK) - timedelta(hours=2)).isoformat()}
    assert bfl_rf_watchdog._should_throttle("auto_pilot", recent) is True


def test_should_throttle_auto_pilot_old_false() -> None:
    old = {"ts": (datetime.now(_MSK) - timedelta(hours=5)).isoformat()}
    assert bfl_rf_watchdog._should_throttle("auto_pilot", old) is False


def test_should_throttle_auto_pilot_missing_ts_false() -> None:
    assert bfl_rf_watchdog._should_throttle("auto_pilot", {}) is False


# --------------------------------------------------------- run: degraded noop


@pytest.mark.asyncio
async def test_run_degraded_noop_without_di() -> None:
    pool = _FakePool()
    result = await bfl_rf_watchdog.run(pool)
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    assert result["alerts_active"] == []
    # pool state untouched
    assert pool.state == {}


# -------------------------------------------------------- run: auto_pilot


@pytest.mark.asyncio
async def test_run_auto_pilot_throttles_when_recent() -> None:
    recent = (datetime.now(_MSK) - timedelta(hours=1)).isoformat()
    pool = _FakePool(
        {
            "strategy_gate_state": {"phase": "auto_pilot"},
            "bfl_rf_watchdog_last_run": {"ts": recent},
        }
    )
    direct = MagicMock()
    http = MagicMock()
    telegram_mock = AsyncMock(return_value=1)

    with patch.object(bfl_rf_watchdog.telegram_tools, "send_message", telegram_mock):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    assert result["status"] == "skipped"
    assert result["reason"] == "auto_pilot_throttle"
    telegram_mock.assert_not_awaited()
    # Nothing written — last_run stays as-is.
    assert pool.state["bfl_rf_watchdog_last_run"] == json.dumps({"ts": recent})


@pytest.mark.asyncio
async def test_run_auto_pilot_runs_after_4h() -> None:
    old = (datetime.now(_MSK) - timedelta(hours=5)).isoformat()
    pool = _FakePool(
        {
            "strategy_gate_state": {"phase": "auto_pilot"},
            "bfl_rf_watchdog_last_run": {"ts": old},
        }
    )
    direct = MagicMock()
    http = MagicMock()
    with (
        patch.object(
            bfl_rf_watchdog.bfl_rf_tracker,
            "collect",
            AsyncMock(return_value=_tracker_snapshot()),
        ) as collect_mock,
        patch.object(bfl_rf_watchdog.telegram_tools, "send_message", AsyncMock(return_value=1)),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    assert result["status"] == "ok"
    collect_mock.assert_awaited_once()


# ----------------------------------------------------- run: learning always


@pytest.mark.asyncio
async def test_run_learning_always_runs_even_with_recent_last_run() -> None:
    recent = (datetime.now(_MSK) - timedelta(minutes=10)).isoformat()
    pool = _FakePool(
        {
            "strategy_gate_state": {"phase": "learning"},
            "bfl_rf_watchdog_last_run": {"ts": recent},
        }
    )
    direct = MagicMock()
    http = MagicMock()
    with (
        patch.object(
            bfl_rf_watchdog.bfl_rf_tracker,
            "collect",
            AsyncMock(return_value=_tracker_snapshot()),
        ) as collect_mock,
        patch.object(bfl_rf_watchdog.telegram_tools, "send_message", AsyncMock(return_value=1)),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    assert result["status"] == "ok"
    assert result["phase"] == "learning"
    collect_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_missing_phase_defaults_to_learning() -> None:
    """No sda_state row for strategy_gate_state → default to 'learning'."""
    pool = _FakePool()
    direct = MagicMock()
    http = MagicMock()
    with (
        patch.object(
            bfl_rf_watchdog.bfl_rf_tracker,
            "collect",
            AsyncMock(return_value=_tracker_snapshot()),
        ),
        patch.object(bfl_rf_watchdog.telegram_tools, "send_message", AsyncMock(return_value=1)),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    assert result["status"] == "ok"
    assert result["phase"] == "learning"


# -------------------------------------------------------- run: send + cooldown


@pytest.mark.asyncio
async def test_run_sends_alerts_and_persists_cooldown() -> None:
    # bounce breach → 1 alert
    data = _tracker_snapshot(metrika={"visits": 200, "bounce_rate": 80.0, "avg_duration_s": 120})
    pool = _FakePool({"strategy_gate_state": {"phase": "learning"}})
    direct = MagicMock()
    http = MagicMock()
    telegram_mock = AsyncMock(return_value=1)

    with (
        patch.object(bfl_rf_watchdog.bfl_rf_tracker, "collect", AsyncMock(return_value=data)),
        patch.object(bfl_rf_watchdog.telegram_tools, "send_message", telegram_mock),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    assert result["alerts_active"] == ["bounce_high"]
    assert result["alerts_sent"] == ["bounce_high"]
    assert result["alerts_skipped_cooldown"] == []
    telegram_mock.assert_awaited_once()
    assert "Алерт BFL-RF" in telegram_mock.await_args.kwargs["text"]
    # cooldown persisted
    cooldown_upserts = [u for u in pool.upserts if u[0] == "bfl_rf_watchdog_cooldowns"]
    assert cooldown_upserts
    assert "bounce_high" in cooldown_upserts[-1][1]
    last_run_upserts = [u for u in pool.upserts if u[0] == "bfl_rf_watchdog_last_run"]
    assert last_run_upserts


@pytest.mark.asyncio
async def test_run_second_tick_within_cooldown_skips_send() -> None:
    recent = (datetime.now(_MSK) - timedelta(hours=1)).isoformat()
    data = _tracker_snapshot(metrika={"visits": 200, "bounce_rate": 80.0, "avg_duration_s": 120})
    pool = _FakePool(
        {
            "strategy_gate_state": {"phase": "learning"},
            "bfl_rf_watchdog_cooldowns": {"bounce_high": recent},
        }
    )
    direct = MagicMock()
    http = MagicMock()
    telegram_mock = AsyncMock(return_value=1)

    with (
        patch.object(bfl_rf_watchdog.bfl_rf_tracker, "collect", AsyncMock(return_value=data)),
        patch.object(bfl_rf_watchdog.telegram_tools, "send_message", telegram_mock),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    assert result["alerts_active"] == ["bounce_high"]
    assert result["alerts_sent"] == []
    assert result["alerts_skipped_cooldown"] == ["bounce_high"]
    telegram_mock.assert_not_awaited()
    # cooldown state NOT rewritten (no mutation), last_run still upserted
    cooldown_upserts = [u for u in pool.upserts if u[0] == "bfl_rf_watchdog_cooldowns"]
    assert cooldown_upserts == []
    last_run_upserts = [u for u in pool.upserts if u[0] == "bfl_rf_watchdog_last_run"]
    assert last_run_upserts


@pytest.mark.asyncio
async def test_run_cooldown_expired_allows_resend() -> None:
    old = (datetime.now(_MSK) - timedelta(hours=7)).isoformat()
    data = _tracker_snapshot(metrika={"visits": 200, "bounce_rate": 80.0, "avg_duration_s": 120})
    pool = _FakePool(
        {
            "strategy_gate_state": {"phase": "learning"},
            "bfl_rf_watchdog_cooldowns": {"bounce_high": old},
        }
    )
    direct = MagicMock()
    http = MagicMock()
    telegram_mock = AsyncMock(return_value=1)

    with (
        patch.object(bfl_rf_watchdog.bfl_rf_tracker, "collect", AsyncMock(return_value=data)),
        patch.object(bfl_rf_watchdog.telegram_tools, "send_message", telegram_mock),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    assert result["alerts_sent"] == ["bounce_high"]
    telegram_mock.assert_awaited_once()


# -------------------------------------------------------------- run: dry_run


@pytest.mark.asyncio
async def test_run_dry_run_no_telegram_no_state_write() -> None:
    data = _tracker_snapshot(metrika={"visits": 200, "bounce_rate": 80.0, "avg_duration_s": 120})
    pool = _FakePool({"strategy_gate_state": {"phase": "learning"}})
    direct = MagicMock()
    http = MagicMock()
    telegram_mock = AsyncMock(return_value=1)

    with (
        patch.object(bfl_rf_watchdog.bfl_rf_tracker, "collect", AsyncMock(return_value=data)),
        patch.object(bfl_rf_watchdog.telegram_tools, "send_message", telegram_mock),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings(), dry_run=True
        )

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["alerts_active"] == ["bounce_high"]
    # alerts_sent logged but not actually sent
    assert result["alerts_sent"] == ["bounce_high"]
    telegram_mock.assert_not_awaited()
    # sda_state untouched
    assert pool.upserts == []


# --------------------------------------------------- run: tracker exception


@pytest.mark.asyncio
async def test_run_tracker_exception_returns_error_dict() -> None:
    pool = _FakePool({"strategy_gate_state": {"phase": "learning"}})
    direct = MagicMock()
    http = MagicMock()
    with patch.object(
        bfl_rf_watchdog.bfl_rf_tracker,
        "collect",
        AsyncMock(side_effect=RuntimeError("tracker boom")),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    assert result["status"] == "error"
    assert "tracker boom" in result["error"]
    # no state rewrites
    assert pool.upserts == []


@pytest.mark.asyncio
async def test_run_telegram_failure_does_not_update_cooldown_for_that_alert() -> None:
    data = _tracker_snapshot(metrika={"visits": 200, "bounce_rate": 80.0, "avg_duration_s": 120})
    pool = _FakePool({"strategy_gate_state": {"phase": "learning"}})
    direct = MagicMock()
    http = MagicMock()

    async def _send(*_a: Any, **_k: Any) -> int:
        raise RuntimeError("telegram 500")

    with (
        patch.object(bfl_rf_watchdog.bfl_rf_tracker, "collect", AsyncMock(return_value=data)),
        patch.object(bfl_rf_watchdog.telegram_tools, "send_message", AsyncMock(side_effect=_send)),
    ):
        result = await bfl_rf_watchdog.run(
            pool, direct=direct, http_client=http, settings=_settings()
        )

    # bounce_high was active, but not counted as sent (telegram failed)
    assert result["alerts_active"] == ["bounce_high"]
    assert result["alerts_sent"] == []
    # cooldown NOT rewritten for bounce_high; only last_run upserted
    cooldown_upserts = [u for u in pool.upserts if u[0] == "bfl_rf_watchdog_cooldowns"]
    assert cooldown_upserts == []


# ------------------------------------------- alert message formatting


def test_format_alert_message_contains_severity_and_hint() -> None:
    msg = bfl_rf_watchdog._format_alert_message(
        {
            "type": "ctr_low",
            "severity": "⚠️",
            "title": "CTR 2.00% при 5000 показах",
            "hint": "Переписать заголовки",
        }
    )
    assert "⚠️" in msg
    assert "Переписать заголовки" in msg
    assert "Cooldown 6ч" in msg


# ----------------------------------------------------- sda_state read error


@pytest.mark.asyncio
async def test_run_sda_state_read_failure_returns_error() -> None:
    class _BrokenPool:
        def connection(self) -> Any:
            raise RuntimeError("pg pool dead")

    pool = _BrokenPool()
    result = await bfl_rf_watchdog.run(
        pool,  # type: ignore[arg-type]
        direct=MagicMock(),
        http_client=MagicMock(),
        settings=_settings(),
    )
    assert result["status"] == "error"
    assert "pg pool dead" in result["error"]
