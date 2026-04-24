"""Unit tests for agent_runtime.jobs.offline_conversions (Task 24).

Pool is a small stateful ``_FakePool`` interpreting the narrow SQL surface
the job touches (``sda_state`` + ``audit_log``). Bitrix ``get_deal_list``
is monkeypatched on the module-level ``bitrix_tools`` import; Metrika is
an ``AsyncMock`` passed in as DI.

Coverage targets:

* stage → target mapping (C45:5/6/WON)
* identifier priority (YCLID > CLIENT_ID > skip)
* external_id format (``bitrix_deal_{ID}_{STAGE_ID}``)
* YCLID + CLIENT_ID mix → two separate Metrika upload calls
* dry_run suppresses upload
* Metrika API error → ``status='error'``
* duplicate suppression via sda_state
* degraded_noop when DI missing
* happy path + by_target counts
* no PII in audit_log payload
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import offline_conversions

# --------------------------------------------------------------- settings


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
        METRIKA_COUNTER_ID=107734488,
        BITRIX_WEBHOOK_URL="https://example.bitrix24.ru/rest/1/TOKEN",
    )


# --------------------------------------------------------------- _FakePool


def _jsonb_value(val: Any) -> Any:
    obj_attr = getattr(val, "obj", None)
    if obj_attr is not None:
        return obj_attr
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (TypeError, json.JSONDecodeError):
            return val
    return val


class _FakePool:
    def __init__(self, sda_state: dict[str, Any] | None = None) -> None:
        self.sda_state: dict[str, Any] = dict(sda_state or {})
        self.audit_log_inserts: list[dict[str, Any]] = []

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
        lowered = " ".join(sql.split()).lower()
        if lowered.startswith("select value from sda_state") and params is not None:
            key = params[0]
            stored = self.pool.sda_state.get(key)
            self._fetchone_result = (stored,) if stored is not None else None
            return
        if lowered.startswith("insert into sda_state") and params is not None:
            key, value = params[0], params[1]
            self.pool.sda_state[key] = _jsonb_value(value)
            return
        if lowered.startswith("insert into audit_log") and params is not None:
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

    async def fetchone(self) -> Any:
        out = self._fetchone_result
        self._fetchone_result = None
        return out

    async def fetchall(self) -> list[Any]:
        return []


# --------------------------------------------------------------- pure unit


def test_stage_to_target_mapping_exact() -> None:
    assert offline_conversions._STAGE_TO_TARGET == {
        "C45:5": "deal_agreed",
        "C45:6": "deal_signed",
        "C45:WON": "deal_won",
    }


def test_build_row_yclid_priority_over_client_id() -> None:
    deal = {
        "ID": "123",
        "STAGE_ID": "C45:WON",
        "UF_CRM_YCLID": "yclid-abc",
        "UF_CRM_CLIENT_ID": "cid-xyz",
        "OPPORTUNITY": "55000",
        "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
    }
    row = offline_conversions._build_row(deal)
    assert row is not None
    assert row.identifier_value == "yclid-abc"
    assert row.identifier_type == "YCLID"
    assert row.external_id == "bitrix_deal_123_C45:WON"
    assert row.target == "deal_won"
    assert row.price == 55000.0


def test_build_row_fallback_to_client_id() -> None:
    deal = {
        "ID": "7",
        "STAGE_ID": "C45:5",
        "UF_CRM_YCLID": "",
        "UF_CRM_CLIENT_ID": "cid-only",
        "OPPORTUNITY": "0",
        "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
    }
    row = offline_conversions._build_row(deal)
    assert row is not None
    assert row.identifier_type == "CLIENT_ID"
    assert row.identifier_value == "cid-only"
    assert row.external_id == "bitrix_deal_7_C45:5"
    assert row.target == "deal_agreed"


def test_build_row_none_when_no_identifier() -> None:
    deal = {
        "ID": "999",
        "STAGE_ID": "C45:6",
        "UF_CRM_YCLID": None,
        "UF_CRM_CLIENT_ID": "",
    }
    assert offline_conversions._build_row(deal) is None


def test_split_by_identifier_separates_types() -> None:
    rows = [
        offline_conversions._Row(
            external_id="bitrix_deal_1_C45:WON",
            identifier_value="y1",
            identifier_type="YCLID",
            target="deal_won",
            datetime_s="2026-04-24T10:00",
            price=100.0,
            currency="RUB",
        ),
        offline_conversions._Row(
            external_id="bitrix_deal_2_C45:WON",
            identifier_value="c1",
            identifier_type="CLIENT_ID",
            target="deal_won",
            datetime_s="2026-04-24T10:00",
            price=100.0,
            currency="RUB",
        ),
    ]
    yclid, client_id = offline_conversions._split_by_identifier(rows)
    assert [r.identifier_value for r in yclid] == ["y1"]
    assert [r.identifier_value for r in client_id] == ["c1"]


# --------------------------------------------------------------- run happy path


@pytest.mark.asyncio
async def test_happy_path_uploads_three_deals(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool()
    deals = [
        {
            "ID": "1",
            "STAGE_ID": "C45:5",
            "UF_CRM_YCLID": "y1",
            "OPPORTUNITY": "30000",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
        {
            "ID": "2",
            "STAGE_ID": "C45:6",
            "UF_CRM_YCLID": "y2",
            "OPPORTUNITY": "35000",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
        {
            "ID": "3",
            "STAGE_ID": "C45:WON",
            "UF_CRM_YCLID": "y3",
            "OPPORTUNITY": "55000",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
    ]
    monkeypatch.setattr(
        offline_conversions.bitrix_tools,
        "get_deal_list",
        AsyncMock(return_value=deals),
    )
    metrika = AsyncMock()
    metrika.upload_offline_conversions = AsyncMock(return_value={"status": "ok"})

    result = await offline_conversions.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        metrika_client=metrika,
        settings=_settings(),
    )

    assert result["status"] == "ok"
    assert result["uploaded"] == 3
    assert result["by_target"] == {"deal_agreed": 1, "deal_signed": 1, "deal_won": 1}
    assert result["skipped_no_identifier"] == []
    # Only YCLID → one Metrika upload call.
    assert metrika.upload_offline_conversions.await_count == 1
    call_args = metrika.upload_offline_conversions.await_args
    assert call_args.kwargs["client_id_type"] == "YCLID"
    rows_uploaded = call_args.args[1]
    assert len(rows_uploaded) == 3
    ext_ids = {r["external_id"] for r in rows_uploaded}
    assert ext_ids == {
        "bitrix_deal_1_C45:5",
        "bitrix_deal_2_C45:6",
        "bitrix_deal_3_C45:WON",
    }


@pytest.mark.asyncio
async def test_yclid_and_client_id_mix_two_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool()
    deals = [
        {
            "ID": "1",
            "STAGE_ID": "C45:WON",
            "UF_CRM_YCLID": "y1",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
        {
            "ID": "2",
            "STAGE_ID": "C45:WON",
            "UF_CRM_YCLID": "y2",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
        {
            "ID": "3",
            "STAGE_ID": "C45:WON",
            "UF_CRM_CLIENT_ID": "c1",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
        {
            "ID": "4",
            "STAGE_ID": "C45:WON",
            "UF_CRM_CLIENT_ID": "c2",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
    ]
    monkeypatch.setattr(
        offline_conversions.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals)
    )
    metrika = AsyncMock()
    metrika.upload_offline_conversions = AsyncMock(return_value={"status": "ok"})

    result = await offline_conversions.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        metrika_client=metrika,
        settings=_settings(),
    )

    assert result["uploaded"] == 4
    assert metrika.upload_offline_conversions.await_count == 2
    calls = metrika.upload_offline_conversions.await_args_list
    types_called = [c.kwargs["client_id_type"] for c in calls]
    assert set(types_called) == {"YCLID", "CLIENT_ID"}


@pytest.mark.asyncio
async def test_skips_deals_without_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool()
    deals = [
        {
            "ID": "1",
            "STAGE_ID": "C45:WON",
            "UF_CRM_YCLID": "y1",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
        # No identifier → skip.
        {
            "ID": "99",
            "STAGE_ID": "C45:WON",
            "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
        },
    ]
    monkeypatch.setattr(
        offline_conversions.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals)
    )
    metrika = AsyncMock()
    metrika.upload_offline_conversions = AsyncMock(return_value={"status": "ok"})

    result = await offline_conversions.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        metrika_client=metrika,
        settings=_settings(),
    )

    assert result["uploaded"] == 1
    assert result["skipped_no_identifier"] == ["99"]


@pytest.mark.asyncio
async def test_dry_run_does_not_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool()
    monkeypatch.setattr(
        offline_conversions.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {
                    "ID": "1",
                    "STAGE_ID": "C45:WON",
                    "UF_CRM_YCLID": "y1",
                    "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
                }
            ]
        ),
    )
    metrika = AsyncMock()
    metrika.upload_offline_conversions = AsyncMock(return_value={"status": "ok"})

    result = await offline_conversions.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        metrika_client=metrika,
        settings=_settings(),
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["uploaded"] == 0
    assert result["uploaded_would_be"] == 1
    metrika.upload_offline_conversions.assert_not_awaited()


@pytest.mark.asyncio
async def test_metrika_api_error_returns_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    monkeypatch.setattr(
        offline_conversions.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {
                    "ID": "1",
                    "STAGE_ID": "C45:WON",
                    "UF_CRM_YCLID": "y1",
                    "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
                }
            ]
        ),
    )

    class FakeMetrikaError(Exception):
        pass

    metrika = AsyncMock()
    metrika.upload_offline_conversions = AsyncMock(side_effect=FakeMetrikaError("boom"))

    result = await offline_conversions.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        metrika_client=metrika,
        settings=_settings(),
    )

    assert result["status"] == "error"
    assert result["uploaded"] == 0
    assert result["errors"] and "boom" in result["errors"][0]


@pytest.mark.asyncio
async def test_duplicate_external_id_suppressed_via_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool(
        sda_state={
            offline_conversions._STATE_KEY: {
                "external_ids": ["bitrix_deal_1_C45:WON"],
            }
        }
    )
    monkeypatch.setattr(
        offline_conversions.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {
                    "ID": "1",
                    "STAGE_ID": "C45:WON",
                    "UF_CRM_YCLID": "y1",
                    "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
                },
                {
                    "ID": "2",
                    "STAGE_ID": "C45:WON",
                    "UF_CRM_YCLID": "y2",
                    "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
                },
            ]
        ),
    )
    metrika = AsyncMock()
    metrika.upload_offline_conversions = AsyncMock(return_value={"status": "ok"})

    result = await offline_conversions.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        metrika_client=metrika,
        settings=_settings(),
    )

    # Deal 1 is already in state → only deal 2 is uploaded.
    assert result["uploaded"] == 1
    call_args = metrika.upload_offline_conversions.await_args
    rows = call_args.args[1]
    assert [r["external_id"] for r in rows] == ["bitrix_deal_2_C45:WON"]


@pytest.mark.asyncio
async def test_degraded_noop_when_di_missing() -> None:
    pool = _FakePool()
    result = await offline_conversions.run(pool)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    assert result["uploaded"] == 0


@pytest.mark.asyncio
async def test_bitrix_error_returns_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool()

    class FakeBitrixError(Exception):
        pass

    monkeypatch.setattr(
        offline_conversions.bitrix_tools,
        "get_deal_list",
        AsyncMock(side_effect=FakeBitrixError("bitrix blew up")),
    )
    metrika = AsyncMock()
    metrika.upload_offline_conversions = AsyncMock()

    result = await offline_conversions.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        metrika_client=metrika,
        settings=_settings(),
    )

    assert result["status"] == "error"
    assert "bitrix blew up" in result["error"]
    metrika.upload_offline_conversions.assert_not_awaited()


@pytest.mark.asyncio
async def test_audit_log_has_only_aggregates_no_raw_deal_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    # Include an innocuous-looking field that would be PII-adjacent if leaked.
    monkeypatch.setattr(
        offline_conversions.bitrix_tools,
        "get_deal_list",
        AsyncMock(
            return_value=[
                {
                    "ID": "1",
                    "STAGE_ID": "C45:WON",
                    "UF_CRM_YCLID": "y-super-secret-attribution",
                    "OPPORTUNITY": "100000",
                    "DATE_MODIFY": "2026-04-24T10:00:00+03:00",
                }
            ]
        ),
    )
    metrika = AsyncMock()
    metrika.upload_offline_conversions = AsyncMock(return_value={"status": "ok"})

    await offline_conversions.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        metrika_client=metrika,
        settings=_settings(),
    )

    assert len(pool.audit_log_inserts) == 1
    entry = pool.audit_log_inserts[0]
    payload_text = json.dumps(entry["tool_input"]) + json.dumps(entry["tool_output"])
    # Aggregates only — no yclid, no opportunity, no external_id in the payload.
    assert "super-secret-attribution" not in payload_text
    assert "bitrix_deal_" not in payload_text
