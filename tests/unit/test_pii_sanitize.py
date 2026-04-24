"""PII sanitiser tests — Task 5c TDD anchor.

Coverage target is 100% (security-critical, Decision 13 / 152-ФЗ). Tests
assert both positive (PII IS masked) and negative (non-PII keys untouched,
nothing mutated, fail-fast on missing salt) properties.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent_runtime.pii import (
    _MASKED,
    hash_phone,
    mask_email,
    mask_name,
    sanitize_audit_payload,
)

_RU_PHONE_CANONICAL = "+79991234567"


# --- hash_phone -------------------------------------------------------------


def test_phone_hashed_deterministic_and_64hex() -> None:
    h1 = hash_phone(_RU_PHONE_CANONICAL)
    h2 = hash_phone(_RU_PHONE_CANONICAL)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_phone_different_salt_yields_different_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_runtime.config import get_settings

    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None
    h1 = hash_phone(_RU_PHONE_CANONICAL)

    monkeypatch.setenv("PII_SALT", "different-salt-" + "9" * 32)
    # Recompute — late binding reads env fresh
    h2 = hash_phone(_RU_PHONE_CANONICAL)
    assert h1 != h2


def test_phone_normalized_before_hash() -> None:
    variants = [
        "+79991234567",
        "+7 999 123 45 67",
        "+7 (999) 123-45-67",
        "8 (999) 123-45-67",
        "89991234567",
    ]
    hashes = {hash_phone(v) for v in variants}
    assert len(hashes) == 1, f"expected single canonical hash, got {hashes}"


def test_phone_international_not_mangled() -> None:
    # +44 London mobile — should be normalised to +447... not +7447...
    h_uk = hash_phone("+44 7700 900000")
    h_ru = hash_phone(_RU_PHONE_CANONICAL)
    assert h_uk != h_ru


def test_phone_empty_string_returns_hash_of_salt() -> None:
    # Not a hard requirement but documents current behaviour: empty input
    # produces a stable hash so the structure of payload is preserved.
    h = hash_phone("")
    assert len(h) == 64


# --- mask_name ---------------------------------------------------------------


def test_name_initial_masked() -> None:
    assert mask_name("Иванов Иван Иванович") == "И***"
    assert mask_name("John Smith") == "J***"


def test_name_empty_or_none_returns_masked() -> None:
    assert mask_name("") == _MASKED
    assert mask_name(None) == _MASKED
    assert mask_name("   ") == _MASKED


# --- mask_email --------------------------------------------------------------


def test_email_always_masked() -> None:
    assert mask_email("test@example.com") == _MASKED
    assert mask_email("") == _MASKED
    assert mask_email(None) == _MASKED


# --- sanitize_audit_payload --------------------------------------------------


def test_audit_payload_no_raw_phone() -> None:
    payload = {
        "result": {
            "ID": "42",
            "PHONE": [{"VALUE": _RU_PHONE_CANONICAL, "VALUE_TYPE": "WORK"}],
        }
    }
    sanitised = sanitize_audit_payload(payload)
    rendered = json.dumps(sanitised, ensure_ascii=False)
    assert _RU_PHONE_CANONICAL not in rendered
    # structure preserved
    assert sanitised["result"]["ID"] == "42"


def test_nested_dict_sanitized() -> None:
    payload = {
        "result": {
            "leads": [
                {"NAME": "Иван", "LAST_NAME": "Иванов", "PHONE": _RU_PHONE_CANONICAL},
                {"NAME": "Петр", "PHONE": "+78005553535"},
            ]
        }
    }
    sanitised = sanitize_audit_payload(payload)
    leads = sanitised["result"]["leads"]
    assert leads[0]["NAME"] == "И***"
    assert leads[0]["LAST_NAME"] == "И***"
    assert len(leads[0]["PHONE"]) == 64  # sha256 hex
    assert leads[1]["NAME"] == "П***"


def test_source_description_masked_wholesale() -> None:
    payload = {"SOURCE_DESCRIPTION": "Иван Иванов +79991234567 просит перезвонить"}
    sanitised = sanitize_audit_payload(payload)
    assert sanitised["SOURCE_DESCRIPTION"] == _MASKED


def test_case_insensitive_keys() -> None:
    payload = {
        "Phone": _RU_PHONE_CANONICAL,
        "phone": _RU_PHONE_CANONICAL,
        "PHONE": _RU_PHONE_CANONICAL,
        "MOBILE": _RU_PHONE_CANONICAL,
    }
    sanitised = sanitize_audit_payload(payload)
    for value in sanitised.values():
        assert len(value) == 64  # all hashed
        assert _RU_PHONE_CANONICAL not in value


def test_non_pii_keys_untouched() -> None:
    payload = {
        "campaign_id": 708978456,
        "click_count": 42,
        "status": "active",
        "nested": {"ratio": 0.12, "label": "healthy"},
    }
    sanitised = sanitize_audit_payload(payload)
    assert sanitised == payload


def test_does_not_mutate_input() -> None:
    original = {
        "NAME": "Иван",
        "PHONE": [{"VALUE": _RU_PHONE_CANONICAL}],
        "SOURCE_DESCRIPTION": "PII soup",
    }
    snapshot = json.dumps(original, ensure_ascii=False)
    sanitize_audit_payload(original)
    assert json.dumps(original, ensure_ascii=False) == snapshot


def test_email_key_masked_in_payload() -> None:
    payload = {"EMAIL": [{"VALUE": "lead@example.com"}], "email": "other@example.com"}
    sanitised = sanitize_audit_payload(payload)
    # EMAIL dict value — wholly masked (non-str)
    assert sanitised["EMAIL"] == _MASKED
    assert sanitised["email"] == _MASKED


def test_non_dict_non_list_passthrough() -> None:
    # Top-level str/int/bool — passthrough
    assert sanitize_audit_payload("hello") == "hello"
    assert sanitize_audit_payload(42) == 42
    assert sanitize_audit_payload(None) is None


def test_missing_salt_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PII_SALT", raising=False)
    from agent_runtime.config import Settings

    with pytest.raises(ValidationError):
        Settings(  # type: ignore[call-arg]
            DATABASE_URL="postgresql://test:test@localhost:5432/test",
            SDA_INTERNAL_API_KEY="a" * 64,
            SDA_WEBHOOK_HMAC_SECRET="b" * 64,
            HYPOTHESIS_HMAC_SECRET="c" * 64,
        )


def test_phone_in_list_without_wrapper_dict() -> None:
    """Bitrix sometimes returns PHONE as [string, string] not [{VALUE: ...}]."""
    payload = {"PHONE": [_RU_PHONE_CANONICAL, "+78005553535"]}
    sanitised = sanitize_audit_payload(payload)
    assert all(len(v) == 64 for v in sanitised["PHONE"])
    assert _RU_PHONE_CANONICAL not in json.dumps(sanitised)


def test_phone_container_preserves_non_str_leaves() -> None:
    """Non-str inside PHONE container (int, bool, None) passes through untouched."""
    payload = {"PHONE": [{"VALUE": _RU_PHONE_CANONICAL, "IS_PRIMARY": True, "ORDER": 1}]}
    sanitised = sanitize_audit_payload(payload)
    entry = sanitised["PHONE"][0]
    assert entry["IS_PRIMARY"] is True
    assert entry["ORDER"] == 1
    assert len(entry["VALUE"]) == 64


def test_name_in_list_of_dicts_sanitized() -> None:
    """Covers _sanitize_value_for_name dict+list branches."""
    payload = {"NAME": [{"FIRST": "Иван", "LAST": "Иванов"}, {"FIRST": "Петр"}]}
    sanitised = sanitize_audit_payload(payload)
    assert sanitised["NAME"][0]["FIRST"] == "И***"
    assert sanitised["NAME"][0]["LAST"] == "И***"
    assert sanitised["NAME"][1]["FIRST"] == "П***"


def test_name_container_preserves_non_str() -> None:
    payload = {"NAME": [{"TITLE": "Иван", "LEGAL": False, "YEAR": 1990}]}
    sanitised = sanitize_audit_payload(payload)
    entry = sanitised["NAME"][0]
    assert entry["TITLE"] == "И***"
    assert entry["LEGAL"] is False
    assert entry["YEAR"] == 1990


def test_bare_local_phone_without_plus_or_eight() -> None:
    """Phone that's just digits without +7 or 8 prefix → normalised with + prepended."""
    h1 = hash_phone("9991234567")
    h2 = hash_phone("+9991234567")
    assert h1 == h2  # both canonicalise to +9991234567
