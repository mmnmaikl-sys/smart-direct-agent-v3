"""Auth layer tests — Task 5b TDD anchor.

Covers the three security criticals from the audit:
1. Bearer on ``/run/*`` and ``/admin/*`` (timing-safe)
2. HMAC-SHA256 + 5-min replay window on ``/webhook/*``
3. Admin endpoints go through ask_queue, not one-shot curl

Rate-limit tests use a patched limiter so we don't need to fire 101 real
requests per test. Fast, deterministic, still exercises the slowapi wiring.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agent_runtime.auth.signing import HMACSigner
from agent_runtime.config import Settings

# --- shared test settings -----------------------------------------------------

_INTERNAL_KEY = "a" * 64
_WEBHOOK_SECRET = "b" * 64
_HYPOTHESIS_SECRET = "c" * 64


def _make_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        SDA_INTERNAL_API_KEY=_INTERNAL_KEY,
        SDA_WEBHOOK_HMAC_SECRET=_WEBHOOK_SECRET,
        HYPOTHESIS_HMAC_SECRET=_HYPOTHESIS_SECRET,
        PII_SALT="pii-test-salt-" + "0" * 32,
    )


@contextlib.contextmanager
def _app_client(
    settings: Settings | None = None,
    mock_pool: MagicMock | None = None,
) -> Iterator[TestClient]:
    """Lifespan-mocked FastAPI TestClient — isolates auth tests from DB/LLM."""
    from agent_runtime.main import create_app

    if mock_pool is None:
        mock_pool = MagicMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()

    with (
        patch("agent_runtime.main.create_pool", return_value=mock_pool),
        patch("agent_runtime.main.PGReflectionStore") as mock_reflection_cls,
        patch("agent_runtime.main.run_migrations", new=AsyncMock(return_value=[])),
        patch("agent_runtime.main.db_ping", new=AsyncMock(return_value=True)),
    ):
        mock_reflection_cls.return_value = MagicMock(ensure_schema=AsyncMock())
        app = create_app(settings or _make_settings())
        with TestClient(app) as client:
            yield client


# --- config fail-fast ---------------------------------------------------------


def test_startup_rejects_short_api_key() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(  # type: ignore[call-arg]
            DATABASE_URL="postgresql://test:test@localhost:5432/test",
            SDA_INTERNAL_API_KEY="short",
            SDA_WEBHOOK_HMAC_SECRET=_WEBHOOK_SECRET,
            HYPOTHESIS_HMAC_SECRET=_HYPOTHESIS_SECRET,
            PII_SALT="pii-test-salt-" + "0" * 32,
        )
    assert "too short" in str(exc.value).lower()


# --- Bearer -------------------------------------------------------------------


def test_bearer_required() -> None:
    with _app_client() as client:
        r = client.post("/run/budget_guard")
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]


def test_bearer_invalid_returns_401() -> None:
    with _app_client() as client:
        r = client.post("/run/budget_guard", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401
    # Neutral message — no oracle on missing-vs-wrong
    assert r.json()["detail"] == "Invalid or missing API key"


def test_bearer_valid_returns_200() -> None:
    with _app_client() as client:
        r = client.post("/run/budget_guard", headers={"Authorization": f"Bearer {_INTERNAL_KEY}"})
    assert r.status_code == 200
    body = r.json()
    assert body["job"] == "budget_guard"
    assert body["executed"] is False  # stub


def test_bearer_uses_compare_digest() -> None:
    """Regression test: ensure compare_digest, not raw ==, guards the Bearer path."""
    import hmac as stdlib_hmac

    with patch(
        "agent_runtime.auth.compare_digest",
        wraps=stdlib_hmac.compare_digest,
    ) as m:
        with _app_client() as client:
            client.post(
                "/run/budget_guard",
                headers={"Authorization": f"Bearer {_INTERNAL_KEY}"},
            )
        assert m.called


# --- Webhook HMAC -------------------------------------------------------------


def _sign(body: bytes, secret: str, ts: int | None = None) -> tuple[str, int, bytes]:
    ts = ts if ts is not None else int(time.time())
    signer = HMACSigner.__new__(HMACSigner)
    # Manually construct without pydantic SecretStr wrapping
    signer._secret = secret.encode()  # type: ignore[attr-defined]
    sig = signer.sign(body, ts)
    return sig, ts, body


def test_hmac_valid_accepted() -> None:
    body = b'{"event":"ONCRMLEADADD"}'
    sig, ts, _ = _sign(body, _WEBHOOK_SECRET)
    with _app_client() as client:
        r = client.post(
            "/webhook/bitrix",
            content=body,
            headers={
                "X-SDA-Signature": sig,
                "X-SDA-Timestamp": str(ts),
                "Content-Type": "application/json",
            },
        )
    assert r.status_code == 200
    assert r.json()["status"] == "received"


def test_hmac_replay_rejected() -> None:
    body = b'{"event":"ONCRMLEADADD"}'
    old_ts = int(time.time()) - 600  # 10 min old
    sig, _, _ = _sign(body, _WEBHOOK_SECRET, ts=old_ts)
    with _app_client() as client:
        r = client.post(
            "/webhook/bitrix",
            content=body,
            headers={"X-SDA-Signature": sig, "X-SDA-Timestamp": str(old_ts)},
        )
    assert r.status_code == 401


def test_hmac_invalid_signature_rejected() -> None:
    body = b'{"event":"ONCRMLEADADD"}'
    sig, ts, _ = _sign(body, _WEBHOOK_SECRET)
    tampered = b'{"event":"EVIL"}'
    with _app_client() as client:
        r = client.post(
            "/webhook/bitrix",
            content=tampered,
            headers={"X-SDA-Signature": sig, "X-SDA-Timestamp": str(ts)},
        )
    assert r.status_code == 401


def test_hmac_missing_headers_rejected() -> None:
    with _app_client() as client:
        r = client.post("/webhook/bitrix", content=b"{}")
    assert r.status_code == 401


def test_hmac_bad_timestamp_header_rejected() -> None:
    with _app_client() as client:
        r = client.post(
            "/webhook/bitrix",
            content=b"{}",
            headers={"X-SDA-Signature": "sha256=00", "X-SDA-Timestamp": "not-a-number"},
        )
    assert r.status_code == 401


# --- health unauth ------------------------------------------------------------


def test_health_no_auth_required() -> None:
    with _app_client() as client:
        r = client.get("/health")
    assert r.status_code == 200


# --- admin confirmation gate --------------------------------------------------


def test_admin_trust_level_triggers_ask_queue() -> None:
    """POST /admin/trust_level → 202 + ask_id; ask_queue INSERT invoked."""
    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()
    mock_pool.close = AsyncMock()

    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=(42,))
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    mock_pool.connection = MagicMock(return_value=conn)

    with _app_client(mock_pool=mock_pool) as client:
        r = client.post(
            "/admin/trust_level?new=assisted",
            headers={"Authorization": f"Bearer {_INTERNAL_KEY}"},
        )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending_confirmation"
    assert body["ask_id"] == 42
    # Two INSERTs: hypotheses upsert + ask_queue
    assert cursor.execute.await_count >= 2


def test_admin_trust_level_requires_bearer() -> None:
    with _app_client() as client:
        r = client.post("/admin/trust_level?new=assisted")
    assert r.status_code == 401


def test_admin_trust_level_503_when_pool_missing() -> None:
    """If DB pool is not on app.state, admin confirmation must 503, not crash."""
    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()
    mock_pool.close = AsyncMock()
    with _app_client(mock_pool=mock_pool) as client:
        # Sabotage: strip pool from app.state after startup
        client.app.state.pool = None  # type: ignore[attr-defined]
        r = client.post(
            "/admin/trust_level",
            headers={"Authorization": f"Bearer {_INTERNAL_KEY}"},
        )
    assert r.status_code == 503


def test_admin_trust_level_503_when_db_insert_fails() -> None:
    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()
    mock_pool.close = AsyncMock()

    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=RuntimeError("pg down"))
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    mock_pool.connection = MagicMock(return_value=conn)

    with _app_client(mock_pool=mock_pool) as client:
        r = client.post(
            "/admin/trust_level",
            headers={"Authorization": f"Bearer {_INTERNAL_KEY}"},
        )
    assert r.status_code == 503


def test_settings_from_request_falls_back_when_state_missing() -> None:
    """Covers the get_settings() fallback in _settings_from_request."""
    from unittest.mock import MagicMock as MM

    from agent_runtime.auth import _settings_from_request

    fake_request = MM()
    fake_request.app.state = MM(spec=[])  # no `settings` attr
    settings = _settings_from_request(fake_request)
    # Uses env vars from conftest.py
    assert settings.DATABASE_URL.startswith("postgresql://")


# --- callback_data signer -----------------------------------------------------


def test_callback_data_sign_verify_roundtrip() -> None:
    from pydantic import SecretStr

    signer = HMACSigner(SecretStr(_HYPOTHESIS_SECRET))
    signed = signer.sign_callback("h123", "approve")
    assert signed.startswith("h123:approve:")
    assert len(signed) <= 64  # Telegram callback_data budget

    hid, action = signer.verify_callback(signed)
    assert hid == "h123"
    assert action == "approve"


def test_callback_data_tampered_rejected() -> None:
    from pydantic import SecretStr

    signer = HMACSigner(SecretStr(_HYPOTHESIS_SECRET))
    signed = signer.sign_callback("h123", "approve")
    parts = signed.split(":")
    # Swap hypothesis_id but keep original signature — should fail
    tampered = f"h999:{parts[1]}:{parts[2]}"
    with pytest.raises(ValueError):
        signer.verify_callback(tampered)


def test_callback_data_bad_format_rejected() -> None:
    from pydantic import SecretStr

    signer = HMACSigner(SecretStr(_HYPOTHESIS_SECRET))
    with pytest.raises(ValueError):
        signer.verify_callback("only:two_parts")


def test_callback_data_rejects_colons_in_id() -> None:
    from pydantic import SecretStr

    signer = HMACSigner(SecretStr(_HYPOTHESIS_SECRET))
    with pytest.raises(ValueError):
        signer.sign_callback("h:bad", "approve")


# --- HMACSigner low-level -----------------------------------------------------


def test_hmacsigner_sign_format() -> None:
    from pydantic import SecretStr

    signer = HMACSigner(SecretStr(_WEBHOOK_SECRET))
    sig = signer.sign(b"payload", timestamp=1234567890)
    assert sig.startswith("sha256=")
    # SHA-256 hex = 64 chars
    assert len(sig) == len("sha256=") + 64


def test_hmacsigner_verify_rejects_replay_exact_edge() -> None:
    from pydantic import SecretStr

    signer = HMACSigner(SecretStr(_WEBHOOK_SECRET))
    ts = 1000
    sig = signer.sign(b"x", ts)
    # now = ts + 300 → exactly at window edge → accept
    assert signer.verify(b"x", ts, sig, now=ts + 300) is True
    # now = ts + 301 → 1s past → reject
    assert signer.verify(b"x", ts, sig, now=ts + 301) is False


# --- Rate limit ---------------------------------------------------------------


def test_rate_limit_run_endpoint() -> None:
    """TestClient shares one remote address → 31 hits should produce at least one 429."""
    with _app_client() as client:
        codes = []
        for _ in range(35):
            r = client.post(
                "/run/budget_guard",
                headers={"Authorization": f"Bearer {_INTERNAL_KEY}"},
            )
            codes.append(r.status_code)
    assert 429 in codes, f"expected at least one 429 in {codes[-5:]}"
