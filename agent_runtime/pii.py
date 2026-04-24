"""PII sanitisation for audit_log (Decision 13, 152-ФЗ).

Original PII (phone, name, email) lives ONLY in Bitrix — that system is the
data-controller and carries the legal compliance. Anything the agent writes to
``audit_log`` must be sanitised first: phone → ``sha256(phone_normalised ||
PII_SALT)``, name → initial + ``***``, email → ``<masked>``, and freeform
fields like ``SOURCE_DESCRIPTION`` → ``<masked>`` wholesale (regex extraction
inside freeform text is unreliable — a single missed format leaks the record).

The four public functions here are pure and stateless. Salt is looked up on
each ``hash_phone`` call (not module-level) so tests can ``monkeypatch.setenv``
without re-importing the module.

Bitrix payload shapes supported:
    {"PHONE": [{"VALUE": "+7...", "VALUE_TYPE": "WORK"}]}    → list item hashed
    {"PHONE": "+7..."}                                       → str hashed
    {"NAME": "Иван", "LAST_NAME": "Иванов"}                  → both masked
    {"SOURCE_DESCRIPTION": "Иван +7..., просит..."}          → fully masked
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# Lowercase match-sets — keys are compared via .lower() so PHONE / Phone /
# phone / phoneNumber all hit. Keep these narrow — anything not listed stays
# untouched. New PII keys should be added here, NOT in calling code.
PII_PHONE_KEYS: frozenset[str] = frozenset(
    {"phone", "phones", "phone_number", "phonenumber", "mobile", "tel"}
)
PII_NAME_KEYS: frozenset[str] = frozenset(
    {"name", "last_name", "second_name", "full_name", "first_name", "lastname", "firstname"}
)
PII_EMAIL_KEYS: frozenset[str] = frozenset({"email", "emails"})
# Freeform keys get fully masked — their value is user-typed prose and will
# reliably contain PII we can't structurally extract.
PII_FREEFORM_KEYS: frozenset[str] = frozenset({"source_description", "comments", "comment"})

_NON_DIGIT_OR_PLUS = re.compile(r"[^\d+]")

_MASKED = "<masked>"


def _get_salt() -> str:
    """Late binding: read from env at call time, not import time.

    Tests use ``monkeypatch.setenv("PII_SALT", ...)`` — if we captured the
    salt at module import, those overrides would not take effect.
    """
    from agent_runtime.config import get_settings  # noqa: PLC0415

    return get_settings().PII_SALT.get_secret_value()


def _normalise_phone(phone: str) -> str:
    """Strip whitespace / punctuation; coerce RU ``8...`` → ``+7...``.

    Non-RU international numbers (``+44...``) keep their country code.
    A bare local number that arrives without leading ``8`` or ``+`` is
    returned as-is after stripping punctuation — we don't know the country
    so we shouldn't guess.
    """
    compact = _NON_DIGIT_OR_PLUS.sub("", phone)
    if not compact:
        return compact
    if compact.startswith("8") and len(compact) == 11:
        return "+7" + compact[1:]
    if not compact.startswith("+"):
        return "+" + compact.lstrip("+")
    return compact


def hash_phone(phone: str) -> str:
    """Return ``sha256_hex(normalise(phone) + PII_SALT)``."""
    normalised = _normalise_phone(phone)
    salt = _get_salt()
    return hashlib.sha256((normalised + salt).encode("utf-8")).hexdigest()


def mask_name(full_name: str | None) -> str:
    """Return ``first_char + '***'`` or ``<masked>`` for empty/None."""
    if not full_name:
        return _MASKED
    stripped = full_name.strip()
    if not stripped:
        return _MASKED
    return stripped[0] + "***"


def mask_email(email: str | None) -> str:
    """Return ``<masked>`` unconditionally — no email-hash use case.

    Parameter preserved so the signature matches the other maskers; callers
    can pass any incoming value without a branch.
    """
    del email  # intentionally unused — preserve signature
    return _MASKED


def _sanitize_value_for_phone(value: Any) -> Any:
    """Apply :func:`hash_phone` to every str reachable in ``value``."""
    if isinstance(value, str):
        return hash_phone(value) if value else value
    if isinstance(value, list):
        return [_sanitize_value_for_phone(v) for v in value]
    if isinstance(value, dict):
        # Inside a phone-typed container, every VALUE/value str is a phone.
        return {k: _sanitize_value_for_phone(v) for k, v in value.items()}
    return value


def _sanitize_value_for_name(value: Any) -> Any:
    if isinstance(value, str):
        return mask_name(value)
    if isinstance(value, list):
        return [_sanitize_value_for_name(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value_for_name(v) for k, v in value.items()}
    return value


def sanitize_audit_payload(payload: Any) -> Any:
    """Return a PII-free copy of ``payload``.

    Walks dicts/lists recursively. When a key matches one of the PII key
    sets above, the corresponding masker is applied to the value (recursing
    into nested containers so Bitrix-style ``PHONE=[{VALUE: ...}]`` works).
    Non-matching keys pass through unchanged. Input is never mutated —
    every container is rebuilt.
    """
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = key.lower() if isinstance(key, str) else key
            if isinstance(lowered, str) and lowered in PII_PHONE_KEYS:
                out[key] = _sanitize_value_for_phone(value)
            elif isinstance(lowered, str) and lowered in PII_NAME_KEYS:
                out[key] = _sanitize_value_for_name(value)
            elif isinstance(lowered, str) and lowered in PII_EMAIL_KEYS:
                out[key] = mask_email(value) if isinstance(value, str) else _MASKED
            elif isinstance(lowered, str) and lowered in PII_FREEFORM_KEYS:
                out[key] = _MASKED
            else:
                out[key] = sanitize_audit_payload(value)
        return out
    if isinstance(payload, list):
        return [sanitize_audit_payload(item) for item in payload]
    return payload


__all__ = [
    "PII_EMAIL_KEYS",
    "PII_FREEFORM_KEYS",
    "PII_NAME_KEYS",
    "PII_PHONE_KEYS",
    "hash_phone",
    "mask_email",
    "mask_name",
    "sanitize_audit_payload",
]
