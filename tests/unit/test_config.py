"""Tests for agent_runtime.config.Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_runtime.config import Settings


def _kwargs(**overrides):
    base = dict(
        DATABASE_URL="postgresql://u:p@h:5432/db",
        SDA_INTERNAL_API_KEY="k" + "0" * 59,
        SDA_WEBHOOK_HMAC_SECRET="k" + "0" * 59,
        HYPOTHESIS_HMAC_SECRET="k" + "0" * 59,
    )
    base.update(overrides)
    return base


def test_defaults_are_decision_17_compliant() -> None:
    s = Settings(**_kwargs())  # type: ignore[arg-type]
    assert s.METRIKA_COUNTER_ID == 107734488
    assert set(s.PROTECTED_CAMPAIGN_IDS) == {
        708978456,
        708978457,
        708978458,
        709014142,
        709307228,
    }


def test_db_pool_validates_bounds() -> None:
    with pytest.raises(ValidationError):
        Settings(**_kwargs(DB_POOL_MIN_SIZE=10, DB_POOL_MAX_SIZE=2))  # type: ignore[arg-type]


def test_missing_database_url_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "DATABASE_URL",
        "SDA_INTERNAL_API_KEY",
        "SDA_WEBHOOK_HMAC_SECRET",
        "HYPOTHESIS_HMAC_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]
