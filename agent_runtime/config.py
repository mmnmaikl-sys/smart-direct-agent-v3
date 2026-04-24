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

    # Yandex Direct API token. Same "optional at import, required at call
    # site" pattern as ANTHROPIC_API_KEY — DirectAPI raises if empty.
    YANDEX_DIRECT_TOKEN: SecretStr = Field(default=SecretStr(""))
    YANDEX_DIRECT_BASE_URL: str = "https://api.direct.yandex.com/json/v5"

    # Bitrix24 REST webhook. URL includes the user id (full-scope); token is
    # the trailing segment — kept separately for `validate_webhook_token`.
    BITRIX_WEBHOOK_URL: str = ""  # e.g. https://x.bitrix24.ru/rest/1/TOKEN/
    BITRIX_WEBHOOK_TOKEN: SecretStr = Field(default=SecretStr(""))

    # Yandex.Metrika stat/v1 API.
    METRIKA_OAUTH_TOKEN: SecretStr = Field(default=SecretStr(""))

    # Telegram Bot API — single bot, owner's DM.
    TELEGRAM_BOT_TOKEN: SecretStr = Field(default=SecretStr(""))
    TELEGRAM_CHAT_ID: int = 0

    # Signal Detector thresholds (Task 11). Moved out of hardcoded module
    # constants so Wave 2 tuning via Railway env is trivial.
    TARGET_CPA: int = 5000
    DAILY_BUDGET_LIMIT: int = 3000
    GOAL_ID: int = 0  # Metrika conversion goal — optional (0 == not configured)
    PROTECTED_LANDING_URLS: list[str] = Field(
        default_factory=lambda: [
            "https://24bankrotsttvo.ru/pages/ad/bankrotstvo-v4.html",
            "https://24bankrotsttvo.ru/pages/ad/price-v4.html",
            "https://24bankrotsttvo.ru/pages/ad/spisanie-dolgov-v4.html",
            "https://24bankrotsttvo.ru/pages/ad/yurist-v4.html",
            "https://24bankrotsttvo.ru/pages/ad/cherez-mfc-v4.html",
        ]
    )

    # Own public base URL used by jobs that POST back to SDA endpoints (e.g.
    # form_checker probing /lead). Prod value is set to
    # "https://{RAILWAY_PUBLIC_DOMAIN}" via env in Railway; localhost fallback
    # is safe — form_checker tests never hit the real wire.
    PUBLIC_BASE_URL: str = "http://localhost:8000"

    # Lead poller (Task 16). In-process asyncio loop in FastAPI lifespan —
    # Railway Cron minimum interval is 5 min, we want realtime-ish 60 s so
    # the owner sees the agent "alive". Guarded by a non-empty whitelist:
    # setting LEAD_POLLER_UTM_WHITELIST=[] disables the loop entirely
    # (useful for local dev / CI).
    LEAD_POLLER_UTM_WHITELIST: list[str] = Field(default_factory=lambda: ["bfl-rf"])
    LEAD_POLLER_INTERVAL_SEC: int = 60
    LEAD_POLLER_INITIAL_LOOKBACK_MIN: int = 5
    LEAD_POLLER_MAX_PAGES: int = 5
    LEAD_POLLER_NOTIFIED_IDS_CAP: int = 100
    BITRIX_PORTAL_BASE_URL: str = "https://sodeystvieko.bitrix24.ru"

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

    @field_validator("PROTECTED_CAMPAIGN_IDS")
    @classmethod
    def protected_list_non_empty(cls, v: list[int]) -> list[int]:
        # Decision 17 requires explicit guard list. Empty = defence disabled,
        # which must never happen silently on deploy. Override is possible
        # only by editing the default in code (grep-auditable).
        if not v:
            raise ValueError(
                "PROTECTED_CAMPAIGN_IDS must be non-empty "
                "(Decision 17 — live campaigns require runtime guard)"
            )
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
