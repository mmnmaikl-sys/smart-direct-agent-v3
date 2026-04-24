"""Environment-driven configuration for smart-direct-agent-v3.

All runtime settings flow through :class:`Settings`. Secrets are loaded at
import time; missing required values abort startup rather than fail at first
request (fail-fast per Decision 11 auth layer and Decision 14 single driver).
"""

from __future__ import annotations

from importlib import metadata

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


def get_settings() -> Settings:
    """Factory to instantiate settings; also used by tests via monkeypatch."""
    return Settings()  # type: ignore[call-arg]
