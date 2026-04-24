"""Environment-driven configuration for smart-direct-agent-v3.

All runtime settings flow through :class:`Settings`. Secrets are loaded at
import time; missing required values abort startup rather than fail at first
request (fail-fast per Decision 11 auth layer and Decision 14 single driver).
"""

from __future__ import annotations

import re
from importlib import metadata

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MIN_SECRET_HEX_LEN = 64  # 32 bytes of entropy as hex — Decision 11 / Task 5b AC
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

DEFAULT_PROTECTED_CAMPAIGNS: tuple[int, ...] = (
    708978456,
    708978457,
    708978458,
    709014142,
    709307228,
)


def _resolve_app_version() -> str:
    try:
        return metadata.version("smart-direct-agent-v3")
    except metadata.PackageNotFoundError:
        return "0.0.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    DATABASE_URL: str
    METRIKA_COUNTER_ID: int = 107734488
    PROTECTED_CAMPAIGN_IDS: list[int] = Field(
        default_factory=lambda: list(DEFAULT_PROTECTED_CAMPAIGNS)
    )

    SDA_INTERNAL_API_KEY: SecretStr
    SDA_WEBHOOK_HMAC_SECRET: SecretStr
    HYPOTHESIS_HMAC_SECRET: SecretStr
    # PII_SALT is deliberately NOT hex-enforced: it is rotated never (rotating
    # would invalidate all prior phone hashes, breaking audit_log-> CRM lookups).
    # Length >= 32 chars is the only strength gate.
    PII_SALT: SecretStr

    # LLM providers. Anthropic is required at runtime (kb.consult, brain);
    # the Settings default of "" lets tests and local imports load without
    # a key. Jobs that actually call the LLM surface AnthropicKeyMissing.
    ANTHROPIC_API_KEY: SecretStr = Field(default=SecretStr(""))

    LOG_LEVEL: str = "INFO"
    APP_VERSION: str = Field(default_factory=_resolve_app_version)
    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 10

    @field_validator("DATABASE_URL")
    @classmethod
    def normalize_dsn(cls, v: str) -> str:
        if v.startswith("postgres://"):
            return "postgresql://" + v[len("postgres://") :]
        return v

    @field_validator("DB_POOL_MAX_SIZE")
    @classmethod
    def max_size_ge_min(cls, v: int, info) -> int:
        min_size = info.data.get("DB_POOL_MIN_SIZE", 2)
        if v < min_size:
            raise ValueError(f"DB_POOL_MAX_SIZE ({v}) must be >= DB_POOL_MIN_SIZE ({min_size})")
        return v

    @field_validator(
        "SDA_INTERNAL_API_KEY",
        "SDA_WEBHOOK_HMAC_SECRET",
        "HYPOTHESIS_HMAC_SECRET",
    )
    @classmethod
    def validate_hex_secret_strength(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        if len(raw) < _MIN_SECRET_HEX_LEN:
            raise ValueError(
                f"secret too short: {len(raw)} chars < {_MIN_SECRET_HEX_LEN} "
                "(generate with `openssl rand -hex 32`)"
            )
        if not _HEX_RE.match(raw):
            raise ValueError("secret must be hex-encoded (0-9 a-f)")
        return v

    @field_validator("PII_SALT")
    @classmethod
    def validate_pii_salt_strength(cls, v: SecretStr) -> SecretStr:
        # Phone-space in RF is ~10^10 values — an 8-char salt leaves cheap
        # rainbow tables; 32 chars gives 256 bits of entropy. Salt is never
        # rotated (Decision 13 / reference_credentials_and_services memory).
        raw = v.get_secret_value()
        if len(raw) < 32:
            raise ValueError(
                f"PII_SALT too short: {len(raw)} chars < 32 "
                "(generate with `python -c 'import secrets; print(secrets.token_hex(32))'`)"
            )
        return v


def get_settings() -> Settings:
    """Factory to instantiate settings; also used by tests via monkeypatch."""
    return Settings()  # type: ignore[call-arg]
