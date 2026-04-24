"""Unit tests for agent_runtime.jobs.bfl_rf_lead_poller (Task 16).

Covers:

* Pure-helper logic: ``_parse_source_description``, ``_compute_fetch_from``,
  ``_filter_new``, ``_merge_notified``, ``_parse_iso``.
* ``LeadPoller.run_once`` state machine: cold start (initial lookback),
  dedup by last_seen + notified_ids, partial Telegram failure, Bitrix
  error path (no state advance), dry_run (no telegram, no state write).
* ``_lead_poller_loop`` — CancelledError propagates cleanly, exceptions in
  a tick don't kill the loop.
* Hardcoded-constant regression: no ``"bfl-rf"`` / ``"sodeystvieko"`` /
  legacy state path in the source file.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import bfl_rf_lead_poller as poller_mod
from agent_runtime.jobs.bfl_rf_lead_poller import (
    LeadPoller,
    _compute_fetch_from,
    _filter_new,
    _lead_poller_loop,
    _merge_notified,
    _parse_iso,
    _parse_source_description,
)

_MSK = timezone(timedelta(hours=3))


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        "SDA_INTERNAL_API_KEY": "a" * 64,
        "SDA_WEBHOOK_HMAC_SECRET": "b" * 64,
        "HYPOTHESIS_HMAC_SECRET": "c" * 64,
        "PII_SALT": "pii-test-salt-" + "0" * 32,
        "LEAD_POLLER_UTM_WHITELIST": ["bfl-rf"],
        "LEAD_POLLER_INTERVAL_SEC": 60,
        "LEAD_POLLER_INITIAL_LOOKBACK_MIN": 5,
        "LEAD_POLLER_MAX_PAGES": 5,
        "LEAD_POLLER_NOTIFIED_IDS_CAP": 100,
        "BITRIX_PORTAL_BASE_URL": "https://sodeystvieko.bitrix24.ru",
        "TELEGRAM_BOT_TOKEN": "1234:test",
        "TELEGRAM_CHAT_ID": 42,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _mock_pool(state_store: dict[str, Any] | None = None) -> MagicMock:
    """Pool whose cursor answers SELECT/INSERT for sda_state key=lead_poller_state."""
    store = state_store if state_store is not None else {}

    async def _exec(sql: str, params: tuple[Any, ...] | None = None) -> None:
        stripped = sql.strip()
        if stripped.startswith("SELECT") and params and params[0] == "lead_poller_state":
            cursor.fetchone = AsyncMock(return_value=(store.get("value"),))
        elif stripped.startswith("INSERT INTO sda_state") and params:
            store["value"] = params[1]  # json-string

    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=_exec)
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    pool._state_store = store
    return pool


def _lead(lead_id: str, dt: datetime, **extra: Any) -> dict[str, Any]:
    base = {
        "ID": lead_id,
        "DATE_CREATE": dt.isoformat(),
        "NAME": f"lead_{lead_id}",
        "PHONE": [{"VALUE": "+79990000000"}],
        "UTM_CAMPAIGN": "bfl-rf",
        "UTM_TERM": "банкротство",
    }
    base.update(extra)
    return base


# ------------------------------------------------------------ pure helpers


def test_parse_source_description_extracts_pairs() -> None:
    parsed = _parse_source_description("debt_amount=500000 | property=no | goal=списать долги")
    assert parsed["debt_amount"] == "500000"
    assert parsed["property"] == "no"
    assert parsed["goal"] == "списать долги"


def test_parse_source_description_empty_returns_dict() -> None:
    assert _parse_source_description("") == {}


def test_parse_source_description_skips_malformed() -> None:
    parsed = _parse_source_description("good=1 | bad_no_equals | other=2")
    assert parsed == {"good": "1", "other": "2"}


def test_compute_fetch_from_cold_start_uses_lookback() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    result = _compute_fetch_from(None, now, 5)
    assert result == now - timedelta(minutes=5)


def test_compute_fetch_from_warm_applies_safety_margin() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    last_seen = datetime(2026, 4, 24, 11, 59, tzinfo=_MSK)
    result = _compute_fetch_from(last_seen, now, 5)
    assert result == last_seen - timedelta(seconds=30)


def test_filter_new_skips_seen_and_old() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    last_seen = datetime(2026, 4, 24, 11, 55, tzinfo=_MSK)
    leads = [
        _lead("1", now - timedelta(minutes=10)),  # old
        _lead("2", now),  # new
        _lead("3", now),  # already seen
    ]
    out = _filter_new(leads, last_seen, {"3"})
    assert [lead["ID"] for lead in out] == ["2"]


def test_filter_new_cold_start_accepts_all() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    leads = [_lead("1", now), _lead("2", now)]
    out = _filter_new(leads, None, set())
    assert [lead["ID"] for lead in out] == ["1", "2"]


def test_filter_new_skips_malformed_timestamp() -> None:
    lead = {"ID": "1", "DATE_CREATE": "not-iso"}
    out = _filter_new([lead], None, set())
    assert out == []


def test_merge_notified_trims_to_cap() -> None:
    merged = _merge_notified(
        [str(i) for i in range(95)], [str(i) for i in range(100, 110)], cap=100
    )
    assert len(merged) == 100
    assert merged[-1] == "109"
    assert merged[0] == "5"


def test_merge_notified_deduplicates() -> None:
    merged = _merge_notified(["1", "2"], ["2", "3"], cap=10)
    assert merged == ["1", "2", "3"]


def test_parse_iso_handles_z_suffix() -> None:
    dt = _parse_iso("2026-04-24T12:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_iso_naive_assumes_msk() -> None:
    dt = _parse_iso("2026-04-24T12:00:00")
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_iso_rejects_garbage() -> None:
    assert _parse_iso("not a date") is None
    assert _parse_iso(None) is None


# -------------------------------------------------------- LeadPoller.run_once


@pytest.mark.asyncio
async def test_first_run_empty_state_uses_initial_lookback() -> None:
    """Cold start must not replay entire CRM history."""
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    pool = _mock_pool({})
    poller = LeadPoller(pool, MagicMock(), _settings())

    with (
        patch.object(
            poller_mod.bitrix_tools, "get_lead_list", AsyncMock(return_value=[])
        ) as bitrix_mock,
        patch.object(poller_mod.telegram_tools, "send_message", AsyncMock(return_value=1)),
        patch("agent_runtime.jobs.bfl_rf_lead_poller.datetime") as dt_mock,
    ):
        dt_mock.now.return_value = now
        dt_mock.fromisoformat = datetime.fromisoformat
        await poller.run_once()

    kwargs = bitrix_mock.await_args.kwargs
    assert kwargs["filter"][">=DATE_CREATE"] == (now - timedelta(minutes=5)).isoformat()


@pytest.mark.asyncio
async def test_new_lead_triggers_telegram_and_updates_state() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    pool = _mock_pool({})
    poller = LeadPoller(pool, MagicMock(), _settings())
    telegram_mock = AsyncMock(return_value=1)

    with (
        patch.object(
            poller_mod.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=[_lead("777", now)]),
        ),
        patch.object(poller_mod.telegram_tools, "send_message", telegram_mock),
    ):
        result = await poller.run_once()

    assert result["new_leads"] == 1
    assert result["sent_ids"] == ["777"]
    telegram_mock.assert_awaited_once()
    # state written: last parameter on the INSERT call contains ID 777
    assert "777" in pool._state_store.get("value", "")


@pytest.mark.asyncio
async def test_duplicate_lead_not_sent_twice() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    last_seen = now - timedelta(minutes=1)
    import json as _json

    pool = _mock_pool(
        {
            "value": _json.dumps(
                {
                    "last_seen": last_seen.isoformat(),
                    "notified_ids": ["777"],
                }
            )
        }
    )
    poller = LeadPoller(pool, MagicMock(), _settings())
    telegram_mock = AsyncMock(return_value=1)

    with (
        patch.object(
            poller_mod.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=[_lead("777", now)]),
        ),
        patch.object(poller_mod.telegram_tools, "send_message", telegram_mock),
    ):
        result = await poller.run_once()

    assert result["new_leads"] == 0
    assert result["sent_ids"] == []
    telegram_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_whitelist_filter_applied_to_bitrix_query() -> None:
    pool = _mock_pool({})
    settings = _settings(LEAD_POLLER_UTM_WHITELIST=["bfl-rf", "other"])
    poller = LeadPoller(pool, MagicMock(), settings)
    with (
        patch.object(
            poller_mod.bitrix_tools, "get_lead_list", AsyncMock(return_value=[])
        ) as bitrix_mock,
        patch.object(poller_mod.telegram_tools, "send_message", AsyncMock(return_value=1)),
    ):
        await poller.run_once()
    kwargs = bitrix_mock.await_args.kwargs
    assert kwargs["filter"]["UTM_CAMPAIGN"] == ["bfl-rf", "other"]


@pytest.mark.asyncio
async def test_dry_run_no_telegram_no_state_write() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    pool = _mock_pool({})
    poller = LeadPoller(pool, MagicMock(), _settings())
    telegram_mock = AsyncMock(return_value=1)

    with (
        patch.object(
            poller_mod.bitrix_tools,
            "get_lead_list",
            AsyncMock(return_value=[_lead("888", now)]),
        ),
        patch.object(poller_mod.telegram_tools, "send_message", telegram_mock),
    ):
        result = await poller.run_once(dry_run=True)

    assert result["new_leads"] == 1
    assert result["sent_ids"] == ["888"]
    telegram_mock.assert_not_awaited()
    assert "value" not in pool._state_store


@pytest.mark.asyncio
async def test_notified_ids_capped_at_limit() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    pool = _mock_pool({})
    settings = _settings(LEAD_POLLER_NOTIFIED_IDS_CAP=5)
    poller = LeadPoller(pool, MagicMock(), settings)

    # 10 new leads in one tick → notified_ids should be trimmed to 5
    leads = [_lead(str(i), now + timedelta(seconds=i)) for i in range(10)]
    with (
        patch.object(poller_mod.bitrix_tools, "get_lead_list", AsyncMock(return_value=leads)),
        patch.object(poller_mod.telegram_tools, "send_message", AsyncMock(return_value=1)),
    ):
        await poller.run_once()

    import json as _json

    stored = _json.loads(pool._state_store["value"])
    assert len(stored["notified_ids"]) == 5
    assert stored["notified_ids"] == ["5", "6", "7", "8", "9"]


@pytest.mark.asyncio
async def test_bitrix_error_does_not_update_last_seen() -> None:
    import json as _json

    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    last_seen_iso = (now - timedelta(minutes=1)).isoformat()
    pool = _mock_pool({"value": _json.dumps({"last_seen": last_seen_iso, "notified_ids": []})})
    poller = LeadPoller(pool, MagicMock(), _settings())

    with (
        patch.object(
            poller_mod.bitrix_tools,
            "get_lead_list",
            AsyncMock(side_effect=RuntimeError("503")),
        ),
        patch.object(poller_mod.telegram_tools, "send_message", AsyncMock(return_value=1)),
    ):
        result = await poller.run_once()

    assert "error" in result
    # state should NOT have been rewritten with a new last_seen
    stored = _json.loads(pool._state_store["value"])
    assert stored["last_seen"] == last_seen_iso


@pytest.mark.asyncio
async def test_telegram_error_keeps_lead_unnotified() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    pool = _mock_pool({})
    poller = LeadPoller(pool, MagicMock(), _settings())

    # Lead "1" fails Telegram, lead "2" succeeds
    async def _send(client, settings, *, text, parse_mode="HTML", **_: Any):
        if "lead_1" in text:
            raise RuntimeError("telegram 500")
        return 1

    with (
        patch.object(
            poller_mod.bitrix_tools,
            "get_lead_list",
            AsyncMock(
                return_value=[
                    _lead("1", now),
                    _lead("2", now + timedelta(seconds=1)),
                ]
            ),
        ),
        patch.object(poller_mod.telegram_tools, "send_message", AsyncMock(side_effect=_send)),
    ):
        result = await poller.run_once()

    assert result["sent_ids"] == ["2"]

    import json as _json

    stored = _json.loads(pool._state_store["value"])
    assert stored["notified_ids"] == ["2"]
    assert "1" not in stored["notified_ids"]


@pytest.mark.asyncio
async def test_format_lead_renders_quiz_fields() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    poller = LeadPoller(_mock_pool({}), MagicMock(), _settings())
    lead = _lead(
        "1",
        now,
        SOURCE_DESCRIPTION=("debt_amount=500000 | property=no | goal=списать | income=50000"),
    )
    text = poller._format_lead(lead)
    assert "debt_amount: 500000" in text
    assert "property: no" in text
    assert "goal: списать" in text
    assert "income: 50000" in text


@pytest.mark.asyncio
async def test_format_lead_uses_config_portal_url() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    settings = _settings(BITRIX_PORTAL_BASE_URL="https://test.bitrix24.ru")
    poller = LeadPoller(_mock_pool({}), MagicMock(), settings)
    text = poller._format_lead(_lead("123", now))
    assert "https://test.bitrix24.ru/crm/lead/details/123/" in text
    assert "sodeystvieko" not in text


@pytest.mark.asyncio
async def test_format_lead_missing_name_falls_back() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    poller = LeadPoller(_mock_pool({}), MagicMock(), _settings())
    lead = {
        "ID": "5",
        "DATE_CREATE": now.isoformat(),
        "NAME": None,
        "TITLE": "Лид #12345 из Яндекс Бизнес",
    }
    text = poller._format_lead(lead)
    assert "Лид" in text


@pytest.mark.asyncio
async def test_format_lead_missing_phone_shows_dash() -> None:
    now = datetime(2026, 4, 24, 12, 0, tzinfo=_MSK)
    poller = LeadPoller(_mock_pool({}), MagicMock(), _settings())
    text = poller._format_lead({"ID": "1", "DATE_CREATE": now.isoformat(), "NAME": "X"})
    assert "—" in text


# ------------------------------------------------------- loop behaviour


@pytest.mark.asyncio
async def test_loop_cancelled_propagates_cleanly() -> None:
    """asyncio.CancelledError must escape the loop on shutdown."""
    settings = _settings(LEAD_POLLER_INTERVAL_SEC=0)
    poller = LeadPoller(_mock_pool({}), MagicMock(), settings)

    with (
        patch.object(poller, "run_once", AsyncMock(return_value={"new_leads": 0, "sent_ids": []})),
    ):
        import asyncio as _a

        task = _a.create_task(_lead_poller_loop(poller))
        await _a.sleep(0.01)
        task.cancel()
        with pytest.raises(_a.CancelledError):
            await task


@pytest.mark.asyncio
async def test_loop_survives_exception_in_tick() -> None:
    """Exception in one tick must not kill the loop."""
    settings = _settings(LEAD_POLLER_INTERVAL_SEC=0)
    poller = LeadPoller(_mock_pool({}), MagicMock(), settings)

    call_count = {"n": 0}

    async def _tick(**_: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first tick boom")
        return {"new_leads": 0, "sent_ids": []}

    with patch.object(poller, "run_once", AsyncMock(side_effect=_tick)):
        import asyncio as _a

        task = _a.create_task(_lead_poller_loop(poller))
        # let at least 2 ticks run
        for _ in range(50):
            await _a.sleep(0.005)
            if call_count["n"] >= 2:
                break
        task.cancel()
        with pytest.raises(_a.CancelledError):
            await task

    assert call_count["n"] >= 2


# -------------------------------------------------------------- job registry


@pytest.mark.asyncio
async def test_run_degraded_noop_without_deps() -> None:
    pool = _mock_pool({})
    result = await poller_mod.run(pool)
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"


@pytest.mark.asyncio
async def test_run_registered_in_job_registry() -> None:
    from agent_runtime.jobs import JOB_REGISTRY

    assert "lead_poller" in JOB_REGISTRY
    assert JOB_REGISTRY["lead_poller"] is poller_mod.run


# --------------------------------------------------------- hygiene greps


def test_no_hardcoded_utm_or_portal_in_source() -> None:
    src = Path("agent_runtime/jobs/bfl_rf_lead_poller.py").read_text()
    assert '"bfl-rf"' not in src
    assert "sodeystvieko" not in src


def test_no_legacy_state_file_path() -> None:
    src = Path("agent_runtime/jobs/bfl_rf_lead_poller.py").read_text()
    assert "/data/bfl_rf_poller_state.json" not in src


def test_no_legacy_app_imports() -> None:
    src = Path("agent_runtime/jobs/bfl_rf_lead_poller.py").read_text()
    assert "from app." not in src
    assert "from app import" not in src


def test_poller_not_in_railway_cron() -> None:
    """Decision 9: poller runs in-process, not via Railway Cron."""
    src = Path("railway.toml").read_text()
    # crude grep — we just make sure there's no cron service named
    # lead_poller / bfl_rf_lead_poller.
    assert 'name = "lead_poller"' not in src
    assert 'name = "bfl_rf_lead_poller"' not in src
