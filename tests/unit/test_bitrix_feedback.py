"""Unit tests for agent_runtime.jobs.bitrix_feedback (Task 22).

Pool is a stateful ``_FakePool`` interpreting the narrow SQL surface the
job issues against ``hypotheses``, ``sda_state``, and ``audit_log``.
Bitrix and Direct clients are patched via ``monkeypatch`` / ``AsyncMock``
to keep the network stack out of the unit layer.

Coverage targets:

* UTM parsing (8+ digit regex, ``yd-<id>-bfl``, bare ``<id>``, non-digit → None).
* UTM_SOURCE filter (yandex kept, google/organic skipped).
* Safe-divide on zero won: campaigns with 0 wins never appear in result.
* Baseline update: only state='confirmed' rows within 30d, JSONB merge
  preserves existing fields (no overwrite).
* sda_state[bitrix_feedback_cpa_history] ring-buffer persists with ts/cpa/won.
* mutations_this_week reset: Monday 11:00-11:30 MSK only; uses SELECT
  FOR UPDATE pattern; other weekdays/hours → no reset.
* dry_run suppresses mutations: no UPDATE on hypotheses, no upsert on
  sda_state, no Telegram send.
* Trust-level overlay: shadow → no UPDATE on baseline/state even in
  prod-mode (Telegram noop).
* PII sanitisation: raw PHONE / NAME never reach audit_log payload —
  audit_log writes go through :func:`db.insert_audit_log` which already
  sanitises; we verify the job itself never forwards raw Bitrix dicts
  into the audit payload keys.
* Error handling: Bitrix API error → status=error, state untouched.
* Degraded-noop when DI missing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import bitrix_feedback
from agent_runtime.tools.bitrix import BitrixAPIError
from agent_runtime.trust_levels import TrustLevel

_MSK = timezone(timedelta(hours=3))


# --------------------------------------------------------------- settings


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        TELEGRAM_BOT_TOKEN="1234:test",
        TELEGRAM_CHAT_ID=42,
        BITRIX_WEBHOOK_URL="https://example.bitrix24.ru/rest/1/TOKEN",
    )


# --------------------------------------------------------------- _FakePool


class _FakePool:
    """In-memory shim for pool.connection() → cursor.execute / fetch.

    Interprets:
      * sda_state (SELECT value WHERE key=..., INSERT ON CONFLICT, FOR UPDATE).
      * hypotheses (SELECT confirmed rows within window, UPDATE baseline_at_promote).
      * audit_log (INSERT — captures the sanitised payload for assertion).
    """

    def __init__(
        self,
        *,
        sda_state: dict[str, Any] | None = None,
        confirmed_hypotheses: Iterable[dict[str, Any]] = (),
        trust_level_value: str | None = "autonomous",
    ) -> None:
        self.sda_state: dict[str, Any] = dict(sda_state or {})
        self.sda_state_history: list[tuple[str, str, Any]] = []  # (op, key, value)
        self.confirmed_hypotheses: list[dict[str, Any]] = [dict(r) for r in confirmed_hypotheses]
        self.hypothesis_updates: list[tuple[str, dict[str, Any]]] = []
        self.audit_log_inserts: list[dict[str, Any]] = []
        self.select_for_update_seen = 0
        self._trust_level_value = trust_level_value
        self.mutations_this_week_seen_under_for_update: list[Any] = []

    def connection(self) -> _FakeConn:
        return _FakeConn(self)


class _FakeConn:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool
        self._in_transaction = False

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def transaction(self) -> _FakeConn:
        self._in_transaction = True
        return self

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.pool)


class _FakeCursor:
    def __init__(self, pool: _FakePool) -> None:
        self.pool = pool
        self._fetchone_result: Any = None
        self._fetchall_result: list[Any] = []
        self.rowcount: int = 0

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        stripped = " ".join(sql.split()).strip()
        lowered = stripped.lower()

        # trust_level lookup (agent_runtime.trust_levels.get_trust_level)
        if "select value" in lowered and "sda_state" in lowered and "trust_level" in lowered:
            val = self.pool._trust_level_value
            self._fetchone_result = (val,) if val is not None else None
            return

        # mutations_this_week reset FOR UPDATE path
        if (
            "select value from sda_state where key = %s for update" in lowered
            and params is not None
        ):
            key = params[0]
            self.pool.select_for_update_seen += 1
            self.pool.mutations_this_week_seen_under_for_update.append(key)
            stored = self.pool.sda_state.get(key)
            self._fetchone_result = (stored,) if stored is not None else None
            return

        # generic SELECT value FROM sda_state WHERE key = %s
        if lowered.startswith("select value from sda_state where key = %s") and params is not None:
            key = params[0]
            stored = self.pool.sda_state.get(key)
            self._fetchone_result = (stored,) if stored is not None else None
            return

        # INSERT INTO sda_state ... ON CONFLICT
        if lowered.startswith("insert into sda_state") and params is not None:
            key, value = params[0], params[1]
            self.pool.sda_state[key] = _jsonb_value(value)
            self.pool.sda_state_history.append(("upsert", key, _jsonb_value(value)))
            return

        # UPDATE sda_state SET value = %s WHERE key = %s
        if lowered.startswith("update sda_state") and params is not None:
            value, key = params[0], params[1]
            self.pool.sda_state[key] = _jsonb_value(value)
            self.pool.sda_state_history.append(("update", key, _jsonb_value(value)))
            return

        # SELECT confirmed hypotheses in window
        if (
            "from hypotheses" in lowered
            and "state = 'confirmed'" in lowered
            and "promoted_at > now()" in lowered
        ):
            assert params is not None
            # params = (window_days, campaign_ids)
            _, campaign_ids = params[0], params[1]
            campaign_ids_set = set(campaign_ids or [])
            rows = [
                (h["id"], h["campaign_id"], h.get("baseline_at_promote") or {})
                for h in self.pool.confirmed_hypotheses
                if h["campaign_id"] in campaign_ids_set
            ]
            self._fetchall_result = rows
            return

        # UPDATE hypotheses SET baseline_at_promote = %s WHERE id = %s
        if lowered.startswith("update hypotheses") and params is not None:
            new_baseline, hypothesis_id = params[0], params[1]
            matched = 0
            for h in self.pool.confirmed_hypotheses:
                if h["id"] == hypothesis_id and h.get("state", "confirmed") == "confirmed":
                    h["baseline_at_promote"] = _jsonb_value(new_baseline)
                    matched = 1
            self.pool.hypothesis_updates.append((hypothesis_id, _jsonb_value(new_baseline)))
            self.rowcount = matched
            return

        # INSERT INTO audit_log ... RETURNING id
        if lowered.startswith("insert into audit_log") and params is not None:
            # columns order per agent_runtime.db.insert_audit_log
            self.pool.audit_log_inserts.append(
                {
                    "hypothesis_id": params[0],
                    "trust_level": params[1],
                    "tool_name": params[2],
                    "tool_input": _jsonb_value(params[3]),
                    "tool_output": _jsonb_value(params[4]) if params[4] is not None else None,
                    "is_mutation": params[5],
                    "is_error": params[6],
                    "error_detail": params[7],
                    "user_confirmed": params[8],
                    "kill_switch_triggered": params[9],
                }
            )
            self._fetchone_result = (len(self.pool.audit_log_inserts),)
            return

        # Anything else — silent accept (SELECT 1, schema probes, etc).

    async def fetchone(self) -> Any:
        out = self._fetchone_result
        self._fetchone_result = None
        return out

    async def fetchall(self) -> list[Any]:
        out = self._fetchall_result
        self._fetchall_result = []
        return out


def _jsonb_value(val: Any) -> Any:
    """Unwrap psycopg Jsonb(...) wrapper if present, else return as-is."""
    # psycopg.types.json.Jsonb is a simple wrapper carrying .obj
    obj_attr = getattr(val, "obj", None)
    if obj_attr is not None:
        return obj_attr
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (TypeError, json.JSONDecodeError):
            return val
    return val


# --------------------------------------------------------------- mocks


def _make_clients(
    *,
    deals: list[dict[str, Any]] | Exception,
    direct_costs_tsv: dict[int, str] | None = None,
    direct_raises: Exception | None = None,
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Returns (bitrix_http_client, direct_client, telegram_http_client).

    Bitrix is patched at the ``bitrix_tools.get_deal_list`` layer in the
    test (not here) so this mock only represents the httpx client itself.
    Direct is a mock with ``get_campaign_stats(cid, date_from, date_to)``.
    Telegram http client exposes .post() but is actually passed through
    to telegram_tools.send_message which the test patches.
    """
    bitrix_client = AsyncMock()
    direct = AsyncMock()
    if direct_raises is not None:
        direct.get_campaign_stats.side_effect = direct_raises
    else:

        async def _stats(cid: int, _df: str, _dt: str) -> dict[str, Any]:
            tsv = (direct_costs_tsv or {}).get(int(cid))
            if tsv is None:
                return {"tsv": ""}
            return {"tsv": tsv}

        direct.get_campaign_stats.side_effect = _stats
    telegram_http = AsyncMock()
    return bitrix_client, direct, telegram_http


def _tsv_for_cost(cost_rub: float) -> str:
    """Build a minimal Direct CAMPAIGN_PERFORMANCE_REPORT TSV.

    Cost is micro-rubles per the Direct contract.
    """
    cost_micro = int(cost_rub * 1_000_000)
    return f"Some report title\nCampaignId\tCost\n0\t{cost_micro}\nTotal rows:\t1\n"


# ----------------------------------------------------- parser unit cases


def test_extract_campaign_id_bare_numeric() -> None:
    assert bitrix_feedback._extract_campaign_id("708978456") == 708978456


def test_extract_campaign_id_yandex_prefix() -> None:
    assert bitrix_feedback._extract_campaign_id("yd-708978456-bfl-rf") == 708978456


def test_extract_campaign_id_returns_none_for_non_digits() -> None:
    assert bitrix_feedback._extract_campaign_id("organic") is None
    assert bitrix_feedback._extract_campaign_id(None) is None
    assert bitrix_feedback._extract_campaign_id("") is None
    # Short numeric (less than 8 digits) → None (not a Direct campaign id).
    assert bitrix_feedback._extract_campaign_id("1234567") is None


# ----------------------------------------------------- time gating


def test_should_reset_mutations_monday_window_true() -> None:
    # Monday 11:15 MSK — inside window.
    mon_11_15 = datetime(2026, 4, 27, 11, 15, tzinfo=_MSK)
    assert bitrix_feedback._should_reset_mutations(mon_11_15) is True


def test_should_reset_mutations_outside_window() -> None:
    tue_11_15 = datetime(2026, 4, 28, 11, 15, tzinfo=_MSK)
    mon_12_00 = datetime(2026, 4, 27, 12, 0, tzinfo=_MSK)
    mon_11_45 = datetime(2026, 4, 27, 11, 45, tzinfo=_MSK)
    assert bitrix_feedback._should_reset_mutations(tue_11_15) is False
    assert bitrix_feedback._should_reset_mutations(mon_12_00) is False
    assert bitrix_feedback._should_reset_mutations(mon_11_45) is False


# ----------------------------------------------------- grouping


def test_group_won_deals_skips_non_yandex_and_missing_utm() -> None:
    deals = [
        {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
        {"ID": "2", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
        {"ID": "3", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "709014142"},
        {"ID": "4", "UTM_SOURCE": "google", "UTM_CAMPAIGN": "708978456"},  # skip
        {"ID": "5", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": None},  # skip
        {"ID": "6", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "organic"},  # skip
    ]
    won, kept, skipped = bitrix_feedback._group_won_deals(deals)
    assert won == {708978456: 2, 709014142: 1}
    assert kept == 3
    assert skipped == 3


# ----------------------------------------------------- happy path


@pytest.mark.asyncio
async def test_cpa_calculation_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """3 WON [456, 456, 457], spend {456:3000, 457:2000} → CPA={456:1500, 457:2000}."""
    pool = _FakePool(
        confirmed_hypotheses=[
            {
                "id": "hyp-456",
                "campaign_id": 708978456,
                "state": "confirmed",
                "baseline_at_promote": {"ctr": 3.5, "leads": 10},
            }
        ]
    )
    bitrix, direct, telegram_http = _make_clients(
        deals=[
            {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
            {"ID": "2", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
            {"ID": "3", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "709014142"},
        ],
        direct_costs_tsv={708978456: _tsv_for_cost(3000.0), 709014142: _tsv_for_cost(2000.0)},
    )
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
                {"ID": "2", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
                {"ID": "3", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "709014142"},
            ]
        ),
    )
    monkeypatch.setattr(
        bitrix_feedback.telegram_tools,
        "send_message",
        AsyncMock(return_value=42),
    )

    result = await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=bitrix,
        http_client=telegram_http,
        settings=_settings(),
        dry_run=False,
    )

    assert result["status"] == "ok"
    assert result["cpa_per_campaign"] == {"708978456": 1500.0, "709014142": 2000.0}
    assert result["won_deals_total"] == 3
    assert result["spend_total"] == 5000.0


@pytest.mark.asyncio
async def test_zero_won_excluded_from_cpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Campaign appearing in Direct with cost but 0 wins is never in result.

    A campaign never lands in ``won_by_campaign`` without at least one
    WON deal — that's the structural guard against ZeroDivisionError.
    """
    pool = _FakePool()
    # Only one won deal → campaign 708978456; Direct returns cost for
    # 708978456 only. Other would-be-zero campaigns cannot leak in because
    # the job computes CPA only over campaigns seen in won_by_campaign.
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
            ]
        ),
    )
    _, direct, telegram_http = _make_clients(
        deals=[],
        direct_costs_tsv={708978456: _tsv_for_cost(500.0)},
    )
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", AsyncMock(return_value=42))

    result = await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )
    # Only the campaign with >=1 won gets a CPA.
    assert result["cpa_per_campaign"] == {"708978456": 500.0}


# ----------------------------------------------------- baseline update


@pytest.mark.asyncio
async def test_updates_baseline_preserves_existing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool(
        confirmed_hypotheses=[
            {
                "id": "hyp-456",
                "campaign_id": 708978456,
                "state": "confirmed",
                "baseline_at_promote": {"ctr": 3.5, "leads": 10},
            }
        ]
    )
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
                {"ID": "2", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
            ]
        ),
    )
    _, direct, telegram_http = _make_clients(
        deals=[], direct_costs_tsv={708978456: _tsv_for_cost(3000.0)}
    )
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", AsyncMock(return_value=42))

    result = await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )

    assert result["hypotheses_updated"] == ["hyp-456"]
    assert len(pool.hypothesis_updates) == 1
    hyp_id, new_baseline = pool.hypothesis_updates[0]
    assert hyp_id == "hyp-456"
    # Old fields preserved.
    assert new_baseline["ctr"] == 3.5
    assert new_baseline["leads"] == 10
    # New fields added.
    assert new_baseline["cpa_real"] == 1500.0
    assert new_baseline["cpa_source"] == "bitrix_won_7d"
    assert "cpa_captured_at" in new_baseline


# ----------------------------------------------------- dry_run


@pytest.mark.asyncio
async def test_dry_run_does_not_mutate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full read path runs; no UPDATE on hypotheses / sda_state / telegram."""
    pool = _FakePool(
        confirmed_hypotheses=[
            {
                "id": "hyp-456",
                "campaign_id": 708978456,
                "state": "confirmed",
                "baseline_at_promote": {"ctr": 3.5},
            }
        ]
    )
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
                {"ID": "2", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
            ]
        ),
    )
    send_mock = AsyncMock(return_value=42)
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", send_mock)
    _, direct, telegram_http = _make_clients(
        deals=[], direct_costs_tsv={708978456: _tsv_for_cost(3000.0)}
    )

    result = await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
        dry_run=True,
    )

    # CPA still reported.
    assert result["cpa_per_campaign"] == {"708978456": 1500.0}
    # But no hypothesis UPDATE, no sda_state CPA history upsert.
    assert pool.hypothesis_updates == []
    assert not any(op for op, _, _ in pool.sda_state_history)
    # And no Telegram send.
    send_mock.assert_not_awaited()


# ----------------------------------------------------- trust shadow overlay


@pytest.mark.asyncio
async def test_shadow_trust_suppresses_baseline_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool(
        trust_level_value=TrustLevel.SHADOW.value,
        confirmed_hypotheses=[
            {
                "id": "hyp-456",
                "campaign_id": 708978456,
                "state": "confirmed",
                "baseline_at_promote": {"ctr": 3.5},
            }
        ],
    )
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
            ]
        ),
    )
    _, direct, telegram_http = _make_clients(
        deals=[], direct_costs_tsv={708978456: _tsv_for_cost(2700.0)}
    )
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", AsyncMock(return_value=42))

    result = await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )

    assert result["trust_level"] == "shadow"
    # CPA computed and returned.
    assert result["cpa_per_campaign"] == {"708978456": 2700.0}
    # But no baseline UPDATE under shadow.
    assert pool.hypothesis_updates == []
    assert result["hypotheses_updated"] == []


# ----------------------------------------------------- mutations reset


@pytest.mark.asyncio
async def test_mutations_reset_only_on_monday_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tuesday 11:15 → no reset. Monday 11:15 → reset via SELECT FOR UPDATE."""
    # --- Tuesday: no reset ---
    pool_tue = _FakePool(
        sda_state={"mutations_this_week": json.dumps({"amount_rub": 5000})},
    )
    monkeypatch.setattr(
        bitrix_feedback,
        "_now_msk",
        lambda: datetime(2026, 4, 28, 11, 15, tzinfo=_MSK),  # Tuesday
    )
    monkeypatch.setattr(bitrix_feedback.bitrix_tools, "get_deal_list", AsyncMock(return_value=[]))
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", AsyncMock(return_value=42))
    _, direct, telegram_http = _make_clients(deals=[], direct_costs_tsv={})

    result_tue = await bitrix_feedback.run(
        pool_tue,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )
    assert result_tue["mutations_reset"] is False
    assert pool_tue.select_for_update_seen == 0

    # --- Monday: reset happens ---
    pool_mon = _FakePool(
        sda_state={"mutations_this_week": json.dumps({"amount_rub": 7500})},
    )
    monkeypatch.setattr(
        bitrix_feedback,
        "_now_msk",
        lambda: datetime(2026, 4, 27, 11, 15, tzinfo=_MSK),  # Monday
    )

    result_mon = await bitrix_feedback.run(
        pool_mon,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )
    assert result_mon["mutations_reset"] is True
    # SELECT FOR UPDATE must be observed exactly once (concurrent-safe guard).
    assert pool_mon.select_for_update_seen == 1
    # Post-reset value is {amount_rub: 0, reset_at: ...}.
    stored = pool_mon.sda_state["mutations_this_week"]
    parsed = json.loads(stored) if isinstance(stored, str) else stored
    assert parsed["amount_rub"] == 0
    assert "reset_at" in parsed


# ----------------------------------------------------- cpa history ring buffer


@pytest.mark.asyncio
async def test_cpa_history_ring_buffer_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    existing_snapshots = [{"ts": "old", "cpa": 1.0, "spend_rub": 1.0, "won": 1}]
    pool = _FakePool(
        sda_state={
            "bitrix_feedback_cpa_history": json.dumps(
                {"campaigns": {"708978456": {"snapshots": existing_snapshots}}}
            ),
        }
    )
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
                {"ID": "2", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
            ]
        ),
    )
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", AsyncMock(return_value=42))
    _, direct, telegram_http = _make_clients(
        deals=[], direct_costs_tsv={708978456: _tsv_for_cost(3000.0)}
    )

    await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )

    stored = pool.sda_state.get("bitrix_feedback_cpa_history")
    parsed = json.loads(stored) if isinstance(stored, str) else stored
    assert "campaigns" in parsed
    snaps = parsed["campaigns"]["708978456"]["snapshots"]
    # Old snapshot preserved + new appended.
    assert len(snaps) == 2
    assert snaps[0]["ts"] == "old"
    assert snaps[-1]["cpa"] == 1500.0
    assert snaps[-1]["won"] == 2


# ----------------------------------------------------- PII sanitisation


@pytest.mark.asyncio
async def test_audit_log_payload_contains_no_raw_phone_or_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if the Bitrix response includes PHONE/NAME fields, audit_log
    payload must never carry them verbatim — the job forwards ONLY
    aggregated numeric counters into ``insert_audit_log``.
    """
    pool = _FakePool()
    # Deals with full Bitrix shape including PHONE/NAME — our grouping
    # ignores these, but a naive refactor could leak them. Guard via
    # explicit negative assertion on audit_log_inserts.
    deals_with_pii = [
        {
            "ID": "1",
            "UTM_SOURCE": "yandex",
            "UTM_CAMPAIGN": "708978456",
            "PHONE": [{"VALUE": "+79991234567", "VALUE_TYPE": "WORK"}],
            "NAME": "Иван",
            "LAST_NAME": "Иванов",
            "SOURCE_DESCRIPTION": "Иван, тел +7 999 123 45 67",
        }
    ]
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(return_value=deals_with_pii),
    )
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", AsyncMock(return_value=42))
    _, direct, telegram_http = _make_clients(
        deals=[], direct_costs_tsv={708978456: _tsv_for_cost(2000.0)}
    )

    await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )

    assert pool.audit_log_inserts, "audit_log write should have been called"
    entry = pool.audit_log_inserts[-1]
    payload_blob = json.dumps(entry["tool_input"]) + json.dumps(entry["tool_output"])
    # Raw phone digits never reach audit_log.
    assert "+79991234567" not in payload_blob
    assert "79991234567" not in payload_blob
    # Raw name never reaches audit_log.
    assert "Иван" not in payload_blob
    assert "Иванов" not in payload_blob


# ----------------------------------------------------- error path


@pytest.mark.asyncio
async def test_bitrix_api_error_returns_status_error_without_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool(
        confirmed_hypotheses=[
            {
                "id": "hyp-456",
                "campaign_id": 708978456,
                "state": "confirmed",
                "baseline_at_promote": {"ctr": 3.5},
            }
        ]
    )
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(side_effect=BitrixAPIError("RATE_LIMIT", "too fast", 429)),
    )
    _, direct, telegram_http = _make_clients(deals=[], direct_costs_tsv={})

    result = await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )

    assert result["status"] == "error"
    assert "RATE_LIMIT" in result["error"]
    assert pool.hypothesis_updates == []  # no side effect on Bitrix failure
    assert result["hypotheses_updated"] == []


@pytest.mark.asyncio
async def test_direct_error_per_campaign_treated_as_zero_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad campaign in Direct should not sink the whole job."""
    pool = _FakePool(
        confirmed_hypotheses=[
            {
                "id": "hyp-ok",
                "campaign_id": 709014142,
                "state": "confirmed",
                "baseline_at_promote": {"ctr": 3.0},
            }
        ]
    )
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"},
                {"ID": "2", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "709014142"},
            ]
        ),
    )
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", AsyncMock(return_value=42))
    direct = AsyncMock()

    # 708978456 → raises, 709014142 → returns a TSV.
    async def _stats(cid: int, _df: str, _dt: str) -> dict[str, Any]:
        if cid == 708978456:
            raise RuntimeError("Direct flaky")
        return {"tsv": _tsv_for_cost(2000.0)}

    direct.get_campaign_stats.side_effect = _stats

    result = await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=AsyncMock(),
        settings=_settings(),
    )

    # Failed campaign lands with cost=0 → CPA=0; healthy one computes correctly.
    assert result["status"] == "ok"
    assert result["cpa_per_campaign"]["708978456"] == 0.0
    assert result["cpa_per_campaign"]["709014142"] == 2000.0


# ----------------------------------------------------- degraded noop


@pytest.mark.asyncio
async def test_degraded_noop_when_di_missing() -> None:
    pool = _FakePool()
    result = await bitrix_feedback.run(pool, dry_run=False)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    assert result["cpa_per_campaign"] == {}
    assert result["hypotheses_updated"] == []
    assert result["mutations_reset"] is False


# ----------------------------------------------------- telegram alert


@pytest.mark.asyncio
async def test_telegram_alert_on_cpa_won_breach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    # CPA = 120000 / 1 = 120000, threshold default 55000 → breach.
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(return_value=[{"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"}]),
    )
    _, direct, telegram_http = _make_clients(
        deals=[], direct_costs_tsv={708978456: _tsv_for_cost(120_000.0)}
    )
    send_mock = AsyncMock(return_value=42)
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", send_mock)

    result = await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
    )

    assert result["alerts_sent"], "Expected CPA-won breach alert"
    send_mock.assert_awaited()
    # Alert payload (text arg) carries campaign id but no raw PII.
    _args, kwargs = send_mock.await_args
    text = kwargs.get("text", "")
    assert "708978456" in text
    assert "+7" not in text  # PII sanity


@pytest.mark.asyncio
async def test_dry_run_suppresses_telegram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    monkeypatch.setattr(
        bitrix_feedback.bitrix_tools,
        "get_deal_list",
        AsyncMock(return_value=[{"ID": "1", "UTM_SOURCE": "yandex", "UTM_CAMPAIGN": "708978456"}]),
    )
    _, direct, telegram_http = _make_clients(
        deals=[], direct_costs_tsv={708978456: _tsv_for_cost(120_000.0)}
    )
    send_mock = AsyncMock(return_value=42)
    monkeypatch.setattr(bitrix_feedback.telegram_tools, "send_message", send_mock)

    await bitrix_feedback.run(
        pool,  # type: ignore[arg-type]
        direct=direct,
        bitrix_client=AsyncMock(),
        http_client=telegram_http,
        settings=_settings(),
        dry_run=True,
    )

    send_mock.assert_not_awaited()
