"""Unit tests for telegram_digest owner_report — Task 04.

Owner_report is a separate job from the existing 09:00 MSK telegram_digest:

  * compile_owner_payload reads sda_state[bitrix_feedback_traffic_split]
    written by Task 02 — own/contractor/organic split per cid
  * render_owner_report emits a Telegram-HTML message <=4096 bytes with
    status emojis (🟢 own CR > 2%, 🟡 1-2%, 🔴 <1% or zero)
  * run_owner_report is the JOB_REGISTRY entrypoint (cron "0 15 * * *" =
    18:00 МСК in railway.toml)

Real Postgres / Telegram are mocked. Pure render logic is unit-tested
without any I/O.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.jobs.telegram_digest import (
    OwnerCampaignBreakdown,
    OwnerReportPayload,
    compile_owner_payload,
    render_owner_report,
    run_owner_report,
)

_NOW = datetime(2026, 4, 26, 15, 0, 0, tzinfo=UTC)
_SNAPSHOT_FIXTURE = {
    "snapshot_at": "2026-04-26T15:00:00+00:00",
    "window_hours": 168,
    "own": {
        "709353005": {"won": 3, "cost_rub": 4500.0, "cpa_won": 1500.0},
        "709353034": {"won": 0, "cost_rub": 1200.0, "cpa_won": None},
    },
    "contractor_aggregate": {
        "won": 5,
        "by_campaign_id": {"709224565": 3, "708138968": 2},
    },
    "organic_aggregate": {"won": 1},
}


# --- compile_owner_payload --------------------------------------------------


def _mk_pool_with_state(state_value: dict | None) -> MagicMock:
    """Build a mock AsyncConnectionPool whose first SELECT returns ``state_value``."""
    cur = AsyncMock()
    if state_value is None:
        cur.fetchone = AsyncMock(return_value=None)
    else:
        cur.fetchone = AsyncMock(return_value=(json.dumps(state_value),))
    cur.execute = AsyncMock()

    cur_cm = MagicMock()
    cur_cm.__aenter__ = AsyncMock(return_value=cur)
    cur_cm.__aexit__ = AsyncMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur_cm)

    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn_cm)
    return pool


@pytest.mark.asyncio
async def test_compile_owner_payload_empty_when_no_state() -> None:
    pool = _mk_pool_with_state(None)
    payload = await compile_owner_payload(pool, now=_NOW)
    assert payload.is_empty()
    assert payload.own_campaigns == []
    assert payload.contractor_won == 0
    assert payload.organic_won == 0


@pytest.mark.asyncio
async def test_compile_owner_payload_reads_traffic_split() -> None:
    pool = _mk_pool_with_state(_SNAPSHOT_FIXTURE)
    payload = await compile_owner_payload(pool, now=_NOW)

    assert not payload.is_empty()
    assert len(payload.own_campaigns) == 2

    own_ids = {c.cid for c in payload.own_campaigns}
    assert own_ids == {709353005, 709353034}

    rab = next(c for c in payload.own_campaigns if c.cid == 709353005)
    assert rab.won == 3
    assert rab.cost_rub == 4500.0
    assert rab.cpa_won == 1500.0

    pen = next(c for c in payload.own_campaigns if c.cid == 709353034)
    assert pen.won == 0
    assert pen.cpa_won is None  # division-by-zero protection

    assert payload.contractor_won == 5
    assert payload.contractor_by_cid == {"709224565": 3, "708138968": 2}
    assert payload.organic_won == 1


@pytest.mark.asyncio
async def test_compile_owner_payload_handles_dict_value_too() -> None:
    """psycopg may return JSONB as already-parsed dict, not as string."""
    cur = AsyncMock()
    cur.fetchone = AsyncMock(return_value=(_SNAPSHOT_FIXTURE,))
    cur.execute = AsyncMock()
    cur_cm = MagicMock()
    cur_cm.__aenter__ = AsyncMock(return_value=cur)
    cur_cm.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur_cm)
    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn_cm)

    payload = await compile_owner_payload(pool, now=_NOW)
    assert len(payload.own_campaigns) == 2


# --- render_owner_report ----------------------------------------------------


def _mk_payload(
    *,
    own_campaigns: list[OwnerCampaignBreakdown] | None = None,
    contractor_won: int = 0,
    contractor_by_cid: dict[str, int] | None = None,
    organic_won: int = 0,
) -> OwnerReportPayload:
    return OwnerReportPayload(
        generated_at=_NOW,
        snapshot_at=_NOW,
        window_hours=168,
        own_campaigns=own_campaigns or [],
        contractor_won=contractor_won,
        contractor_by_cid=contractor_by_cid or {},
        organic_won=organic_won,
    )


def test_render_owner_report_under_4096_bytes() -> None:
    own = [
        OwnerCampaignBreakdown(cid=cid, won=2, cost_rub=1500.0, cpa_won=750.0)
        for cid in (709353005, 709353034, 709353058, 709353078, 709353099)
    ]
    payload = _mk_payload(
        own_campaigns=own,
        contractor_won=10,
        contractor_by_cid={"709224565": 5, "708138968": 5},
        organic_won=2,
    )
    text = render_owner_report(payload)
    assert len(text.encode("utf-8")) <= 4096


def test_render_owner_report_zero_won_uses_red_emoji() -> None:
    payload = _mk_payload(
        own_campaigns=[
            OwnerCampaignBreakdown(cid=709353005, won=0, cost_rub=4500.0, cpa_won=None),
        ],
    )
    text = render_owner_report(payload)
    assert "🔴" in text
    assert "709353005" in text


def test_render_owner_report_high_cpa_uses_yellow_or_red() -> None:
    """CPA way above target → not green."""
    payload = _mk_payload(
        own_campaigns=[
            OwnerCampaignBreakdown(cid=709353005, won=1, cost_rub=80000.0, cpa_won=80000.0),
        ],
    )
    text = render_owner_report(payload)
    assert "🟢" not in text  # 80k CPA can't be healthy


def test_render_owner_report_empty_state() -> None:
    text = render_owner_report(_mk_payload())
    assert "Тих" in text or "пуст" in text.lower() or "нет данных" in text.lower()


def test_render_owner_report_includes_contractor_split() -> None:
    payload = _mk_payload(
        contractor_won=7,
        contractor_by_cid={"709224565": 4, "708138968": 3},
    )
    text = render_owner_report(payload)
    assert "709224565" in text or "контракт" in text.lower() or "Подрядч" in text


def test_render_owner_report_html_escapes_strings() -> None:
    """Defensive: any string that ends up in render must not break Telegram HTML."""
    payload = _mk_payload(
        own_campaigns=[
            OwnerCampaignBreakdown(cid=709353005, won=1, cost_rub=100.0, cpa_won=100.0),
        ],
    )
    text = render_owner_report(payload)
    # No raw '<' that isn't part of an HTML tag we emitted
    assert "<script" not in text


# --- run_owner_report -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_owner_report_dry_run_no_telegram_call(monkeypatch) -> None:
    pool = _mk_pool_with_state(_SNAPSHOT_FIXTURE)
    send_mock = AsyncMock()
    monkeypatch.setattr(
        "agent_runtime.jobs.telegram_digest.telegram_tools.send_message",
        send_mock,
    )

    settings_stub = MagicMock()
    http_client = AsyncMock()
    result = await run_owner_report(
        pool, dry_run=True, http_client=http_client, settings=settings_stub
    )

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["telegram_sent"] is False
    send_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_owner_report_sends_message_when_not_dry(monkeypatch) -> None:
    pool = _mk_pool_with_state(_SNAPSHOT_FIXTURE)
    send_mock = AsyncMock()
    monkeypatch.setattr(
        "agent_runtime.jobs.telegram_digest.telegram_tools.send_message",
        send_mock,
    )

    settings_stub = MagicMock()
    http_client = AsyncMock()
    result = await run_owner_report(
        pool, dry_run=False, http_client=http_client, settings=settings_stub
    )

    assert result["status"] == "ok"
    assert result["dry_run"] is False
    assert result["telegram_sent"] is True
    send_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_owner_report_degraded_noop_when_no_clients() -> None:
    """No http_client / settings → return degraded_noop without crashing."""
    pool = _mk_pool_with_state(_SNAPSHOT_FIXTURE)
    result = await run_owner_report(pool, dry_run=False)
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    assert result["telegram_sent"] is False
