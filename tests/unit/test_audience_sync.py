"""Unit tests for agent_runtime.jobs.audience_sync (Task 24).

Critical 152-ФЗ invariants:

* Deals without ``UF_CRM_CONSENT_ADVERTISING='Y'`` must land in
  ``skipped_no_consent`` BEFORE contact lookup (saves API calls AND
  satisfies the legal gate).
* Raw phone values and MD5 hex strings must NOT appear in ``audit_log``
  payload — only counters and deal_id lists.
* MD5 hashing is deterministic + normalisation-idempotent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_runtime.config import Settings
from agent_runtime.jobs import audience_sync

# --------------------------------------------------------------- settings


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY="a" * 64,
        SDA_WEBHOOK_HMAC_SECRET="b" * 64,
        HYPOTHESIS_HMAC_SECRET="c" * 64,
        PII_SALT="pii-test-salt-" + "0" * 32,
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
    def __init__(self) -> None:
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


def test_phone_md5_deterministic_constant() -> None:
    """MD5 of +79991234567 is a well-known constant — pin it."""
    expected = hashlib.md5(b"+79991234567").hexdigest()  # noqa: S324
    assert audience_sync._md5_hex("+79991234567") == expected
    assert len(expected) == 32
    assert all(c in "0123456789abcdef" for c in expected)


def test_phone_normalization_before_md5_identity() -> None:
    """Different surface forms of the same RU phone → same MD5."""
    forms = ["+7 (999) 123-45-67", "8 999 123 45 67", "+79991234567", "8-999-123-45-67"]
    normalized = [audience_sync._normalize_phone(f) for f in forms]
    assert all(n == "+79991234567" for n in normalized)
    hashes = {audience_sync._md5_hex(n) for n in normalized if n}
    assert len(hashes) == 1


def test_normalize_phone_rejects_invalid() -> None:
    assert audience_sync._normalize_phone(None) is None
    assert audience_sync._normalize_phone("") is None
    assert audience_sync._normalize_phone("abc") is None
    assert audience_sync._normalize_phone("12345") is None  # too short


def test_extract_first_phone_from_bitrix_list_shape() -> None:
    contact = {"PHONE": [{"VALUE": "+79991234567", "VALUE_TYPE": "WORK"}]}
    assert audience_sync._extract_first_phone(contact) == "+79991234567"


def test_extract_first_phone_handles_scalar_and_empty() -> None:
    assert audience_sync._extract_first_phone({"PHONE": "+79990001122"}) == "+79990001122"
    assert audience_sync._extract_first_phone({"PHONE": []}) is None
    assert audience_sync._extract_first_phone({}) is None


# --------------------------------------------------------------- run-level


def _deal(
    deal_id: str,
    *,
    contact_id: int | None = 100,
    consent: str | None = "Y",
    closedate: str = "2026-04-10",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ID": deal_id,
        "CONTACT_ID": str(contact_id) if contact_id is not None else None,
        "CLOSEDATE": closedate,
    }
    if consent is not None:
        out["UF_CRM_CONSENT_ADVERTISING"] = consent
    return out


def _mock_get_contact(phone_by_contact: dict[int, str | None]) -> AsyncMock:
    async def _impl(contact_id: int) -> dict[str, Any]:
        phone = phone_by_contact.get(contact_id)
        if phone is None:
            return {"PHONE": []}
        return {"PHONE": [{"VALUE": phone, "VALUE_TYPE": "WORK"}]}

    return AsyncMock(side_effect=_impl)


@pytest.mark.asyncio
async def test_won_without_consent_skipped_no_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CRITICAL 152-ФЗ: all 3 deals without consent → no upload."""
    pool = _FakePool()
    deals = [
        _deal("1", consent="N"),
        _deal("2", consent=None),
        _deal("3", consent=""),
    ]
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})
    get_contact = AsyncMock()  # should NOT be called

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=get_contact,
        settings=_settings(),
        dry_run=True,
    )

    assert result["skipped_no_consent"] == ["1", "2", "3"]
    assert result["uploaded_would_be"] == 0
    audience.modify_segment_data.assert_not_awaited()
    get_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_won_with_consent_uploaded_as_md5_no_raw_phone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    deals = [
        _deal("1", contact_id=101, consent="Y"),
        _deal("2", contact_id=102, consent="Y"),
    ]
    phones = {101: "+79991234567", 102: "+79998887766"}
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=_mock_get_contact(phones),
        settings=_settings(),
    )

    assert result["status"] == "ok"
    assert result["uploaded"] == 2
    audience.modify_segment_data.assert_awaited_once()
    call_args = audience.modify_segment_data.await_args
    hashes = call_args.kwargs["hashes"]
    assert len(hashes) == 2
    assert all(len(h) == 32 and all(c in "0123456789abcdef" for c in h) for h in hashes)
    # No raw phone in the Audience call payload either.
    payload_text = json.dumps({"hashes": hashes, "id": call_args.kwargs["segment_id"]})
    assert "+7999" not in payload_text
    # And NOT in audit_log payload.
    for entry in pool.audit_log_inserts:
        combined = json.dumps(entry["tool_input"]) + json.dumps(entry["tool_output"])
        assert "+7" not in combined
        for h in hashes:
            assert h not in combined


@pytest.mark.asyncio
async def test_mixed_consent_only_y_uploaded(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool()
    deals = [
        _deal("1", contact_id=101, consent="Y"),
        _deal("2", contact_id=102, consent="Y"),
        _deal("3", contact_id=103, consent="N"),
        _deal("4", contact_id=104, consent="N"),
        _deal("5", contact_id=105, consent=None),
    ]
    phones = {
        101: "+79991111111",
        102: "+79992222222",
        103: "+79993333333",
        104: "+79994444444",
        105: "+79995555555",
    }
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=_mock_get_contact(phones),
        settings=_settings(),
    )

    assert result["uploaded"] == 2
    assert result["skipped_no_consent"] == ["3", "4", "5"]
    assert result["consent_rate"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_contact_without_phone_skipped_no_phone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    deals = [
        _deal("1", contact_id=101, consent="Y"),
        _deal("2", contact_id=102, consent="Y"),
    ]
    phones = {101: "+79991234567", 102: None}  # second contact has no phone
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=_mock_get_contact(phones),
        settings=_settings(),
    )

    assert result["uploaded"] == 1
    assert result["skipped_no_phone"] == ["2"]


@pytest.mark.asyncio
async def test_dry_run_does_not_call_audience_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    deals = [_deal("1", contact_id=101, consent="Y")]
    phones = {101: "+79991234567"}
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=_mock_get_contact(phones),
        settings=_settings(),
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["uploaded_would_be"] == 1
    assert result["uploaded"] == 0
    audience.modify_segment_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_audience_api_error_returns_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    deals = [_deal("1", contact_id=101, consent="Y")]
    phones = {101: "+79991234567"}
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))

    class FakeAudienceError(Exception):
        pass

    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(side_effect=FakeAudienceError("segment 404"))

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=_mock_get_contact(phones),
        settings=_settings(),
    )

    assert result["status"] == "error"
    assert result["uploaded"] == 0
    assert "segment 404" in (result.get("error") or "")


@pytest.mark.asyncio
async def test_empty_hash_set_short_circuits_audience_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When all consented deals lack phones the Audience API is NOT called."""
    pool = _FakePool()
    deals = [_deal("1", contact_id=101, consent="Y")]
    phones: dict[int, str | None] = {101: None}
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=_mock_get_contact(phones),
        settings=_settings(),
    )

    assert result["status"] == "ok"
    assert result["uploaded"] == 0
    audience.modify_segment_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_audit_log_no_raw_phone_no_md5(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _FakePool()
    deals = [
        _deal("1", contact_id=101, consent="Y"),
        _deal("2", contact_id=102, consent="Y"),
    ]
    phones = {101: "+79991234567", 102: "+79998887766"}
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})

    await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=_mock_get_contact(phones),
        settings=_settings(),
    )

    assert len(pool.audit_log_inserts) >= 1
    for entry in pool.audit_log_inserts:
        combined = json.dumps(entry["tool_input"]) + json.dumps(entry["tool_output"])
        assert "+7" not in combined
        assert "1234567" not in combined
        assert "8887766" not in combined
        md5_1 = hashlib.md5(b"+79991234567").hexdigest()  # noqa: S324
        md5_2 = hashlib.md5(b"+79998887766").hexdigest()  # noqa: S324
        assert md5_1 not in combined
        assert md5_2 not in combined


@pytest.mark.asyncio
async def test_degraded_noop_when_di_missing() -> None:
    pool = _FakePool()
    result = await audience_sync.run(pool)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    assert result["uploaded"] == 0


@pytest.mark.asyncio
async def test_low_consent_rate_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pool = _FakePool()
    deals = [
        _deal("1", contact_id=101, consent="Y"),
        _deal("2", contact_id=102, consent="Y"),
        _deal("3", contact_id=103, consent="N"),
        _deal("4", contact_id=104, consent="N"),
        _deal("5", contact_id=105, consent="N"),
        _deal("6", contact_id=106, consent="N"),
        _deal("7", contact_id=107, consent="N"),
        _deal("8", contact_id=108, consent="N"),
        _deal("9", contact_id=109, consent="N"),
        _deal("10", contact_id=110, consent="N"),
    ]
    phones = {101: "+79991234567", 102: "+79998887766"}
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})

    with caplog.at_level("WARNING", logger=audience_sync.__name__):
        await audience_sync.run(
            pool,  # type: ignore[arg-type]
            bitrix_client=AsyncMock(),
            audience_client=audience,
            get_contact=_mock_get_contact(phones),
            settings=_settings(),
        )

    assert any("low consent rate" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_bitrix_error_returns_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()

    class FakeBitrixError(Exception):
        pass

    monkeypatch.setattr(
        audience_sync.bitrix_tools,
        "get_deal_list",
        AsyncMock(side_effect=FakeBitrixError("bitrix 500")),
    )
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock()

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=AsyncMock(),
        settings=_settings(),
    )

    assert result["status"] == "error"
    assert "bitrix 500" in result["error"]
    audience.modify_segment_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_phones_deduplicated_within_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    deals = [
        _deal("1", contact_id=101, consent="Y"),
        _deal("2", contact_id=102, consent="Y"),
    ]
    # Both contacts carry the same phone → one hash.
    phones = {101: "+79991234567", 102: "8 999 123 45 67"}
    monkeypatch.setattr(audience_sync.bitrix_tools, "get_deal_list", AsyncMock(return_value=deals))
    audience = AsyncMock()
    audience.modify_segment_data = AsyncMock(return_value={"status": "ok"})

    result = await audience_sync.run(
        pool,  # type: ignore[arg-type]
        bitrix_client=AsyncMock(),
        audience_client=audience,
        get_contact=_mock_get_contact(phones),
        settings=_settings(),
    )

    assert result["uploaded"] == 1
    call_args = audience.modify_segment_data.await_args
    assert len(call_args.kwargs["hashes"]) == 1


def test_no_pii_salt_usage_or_pii_module_import() -> None:
    """Security-auditor grep: PII_SALT must NOT be used as a value and
    ``agent_runtime.pii`` must NOT be imported — this is the other
    hash contract (bare MD5 for Yandex Audience, not salted SHA256).

    We tolerate mentions inside comments / docstrings (the module
    *explains* why PII_SALT is absent); the test guards against any
    executable reference — ``settings.PII_SALT`` access, ``import
    agent_runtime.pii``, ``from agent_runtime import pii``.
    """
    import pathlib

    path = pathlib.Path(audience_sync.__file__)
    lines = path.read_text(encoding="utf-8").splitlines()
    # Strip docstring bodies and full-line comments from the scan.
    in_docstring = False
    code_lines: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        triple_count = stripped.count('"""')
        if triple_count == 2:
            # Single-line docstring — skip whole line.
            continue
        if triple_count == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        code_lines.append(raw)
    code_text = "\n".join(code_lines)
    # Code must not contain a PII_SALT reference nor pull in the sanitiser module.
    assert "PII_SALT" not in code_text, "PII_SALT must not be used in audience_sync code"
    assert (
        "agent_runtime.pii" not in code_text
    ), "audience_sync must not import agent_runtime.pii (different hash contract)"
    assert "from agent_runtime import pii" not in code_text
