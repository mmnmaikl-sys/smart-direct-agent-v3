"""FastAPI auth dependencies (Decision 11 / Task 5b).

Public API consumed by ``agent_runtime.main`` and later by Task 8 (webhook
handlers) and Task 11 (Telegram ask_queue worker):

* :func:`require_internal_key` — HTTPBearer gate for ``/run/*`` and ``/admin/*``
* :func:`verify_webhook_signature` — HMAC-SHA256 + replay-window gate for
  ``/webhook/*`` endpoints. Returns raw request body (bytes) to the handler
  since the signature was computed over those bytes.
* :func:`require_admin_confirmation` — Telegram confirmation flow for any
  ``/admin/*`` action. INSERTs a row into ``ask_queue`` and forces the HTTP
  caller to poll ``/admin/confirm?ask_id=X`` after the owner approves inline.
* :func:`build_rate_limiter` — slowapi Limiter factory.
* :class:`HMACSigner` — re-exported from ``agent_runtime.auth.hmac``.
"""

from __future__ import annotations

import json
import logging
import time
from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

from agent_runtime.auth.signing import HMACSigner
from agent_runtime.config import Settings, get_settings

logger = logging.getLogger(__name__)

# auto_error=False so we can raise 401 with a neutral message instead of the
# FastAPI default (which discloses expected auth scheme).
_bearer_scheme = HTTPBearer(auto_error=False)


def _settings_from_request(request: Request) -> Settings:
    """Use the Settings instance the app was created with (app.state.settings).

    Falls back to ``get_settings()`` only if the app has not yet populated
    ``app.state.settings`` — which happens during early lifespan before yield,
    so in practice every request sees the same instance create_app() bound.
    """
    state_settings = getattr(request.app.state, "settings", None)
    if isinstance(state_settings, Settings):
        return state_settings
    return get_settings()


SettingsDep = Annotated[Settings, Depends(_settings_from_request)]


# --------------------------------------------------------------------- Bearer


async def require_internal_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    settings: SettingsDep,
) -> None:
    """Require a timing-safe-equal Bearer token for internal endpoints.

    Does NOT disclose *why* the key is invalid (missing vs wrong) — both paths
    return the same 401 detail so the endpoint does not act as an oracle.
    """
    expected = settings.SDA_INTERNAL_API_KEY.get_secret_value()
    provided = credentials.credentials if credentials else ""
    if not compare_digest(expected, provided):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------- Webhook HMAC


async def verify_webhook_signature(
    request: Request,
    settings: SettingsDep,
) -> bytes:
    """Verify X-SDA-Signature + X-SDA-Timestamp HMAC on raw request body.

    Returns the raw body bytes so the route handler can parse JSON itself
    (we can't let FastAPI decode it first — the signature was computed over
    the raw bytes and any re-serialization would change the payload).
    """
    signature = request.headers.get("X-SDA-Signature")
    ts_header = request.headers.get("X-SDA-Timestamp")
    if not signature or not ts_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing webhook signature headers",
        )
    try:
        timestamp = int(ts_header)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        ) from exc

    body = await request.body()
    signer = HMACSigner(settings.SDA_WEBHOOK_HMAC_SECRET)
    if not signer.verify(body, timestamp, signature, now=int(time.time())):
        # Single 401 for every failure mode — no timestamp/signature oracle.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )
    return body


# ------------------------------------------------------- /admin/* confirmation


async def require_admin_confirmation(
    request: Request,
) -> int:
    """Queue a Telegram confirmation for any /admin/* action.

    Does NOT execute the requested action. Instead:
      1. INSERT a row into ``ask_queue`` with ``question`` encoding the
         endpoint + sanitized query params.
      2. Return the new ``ask_id`` so the caller can poll for the resolution.

    Task 11 (Telegram ask_queue worker) picks the row up, sends an inline
    Approve/Reject button to the owner, and marks the row resolved. A
    follow-up GET ``/admin/confirm?ask_id=X`` endpoint (Task 11) actually
    performs the action — this dependency only creates the gate.

    The ``/admin/trust_level`` endpoint raises ``HTTPAdminConfirmationQueued``
    (status 202) to signal to the caller to poll. That exception is caught by
    ``agent_runtime.main`` and serialised to 202 + {ask_id}.
    """
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DB pool not initialised",
        )

    # Sanitize: store endpoint + params but not full headers/body — avoid
    # stashing secrets in ask_queue. hypothesis_id='admin' sentinel so the
    # FK to hypotheses is satisfied via a pre-seeded 'admin' stub; until Task
    # 11 lands, we tolerate a foreign-key-less schema by inserting a zero
    # hypothesis_id placeholder and letting the worker substitute.
    question = json.dumps(
        {
            "endpoint": request.url.path,
            "method": request.method,
            "query": dict(request.query_params),
        }
    )

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # We use a synthetic hypothesis_id='admin' so the existing FK
                # (Task 5 schema) accepts the row. Task 11 will swap the FK
                # for a polymorphic type column; for Task 5b we only need the
                # gate to be observable and auditable.
                await cur.execute(
                    "INSERT INTO hypotheses "
                    "  (id, agent, hypothesis_type, signals, hypothesis, "
                    "   reasoning, actions, expected_outcome, budget_cap_rub, "
                    "   autonomy_level, risk_score, metrics_before) "
                    "VALUES ('admin', 'admin', 'account_level', '[]'::jsonb, "
                    "   'admin action gate', 'admin gate', '[]'::jsonb, "
                    "   'owner approval', 1, 'ASK', 0.0, '{}'::jsonb) "
                    "ON CONFLICT (id) DO NOTHING"
                )
                await cur.execute(
                    "INSERT INTO ask_queue (hypothesis_id, question) VALUES (%s, %s) RETURNING id",
                    ("admin", question),
                )
                row = await cur.fetchone()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("require_admin_confirmation: ask_queue insert failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not queue admin confirmation",
        ) from exc

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not queue admin confirmation",
        )
    return int(row[0])


# ------------------------------------------------------------------ slowapi


def build_rate_limiter() -> Limiter:
    """Create the slowapi Limiter used by FastAPI middleware.

    Keyed on remote address. Behind Railway's ingress proxy this resolves to
    the upstream X-Forwarded-For; Railway terminates TLS and appends the real
    client IP, so per-IP limits are not trivially bypassable from a single
    attacker box.
    """
    return Limiter(key_func=get_remote_address)


__all__ = [
    "HMACSigner",
    "build_rate_limiter",
    "require_admin_confirmation",
    "require_internal_key",
    "verify_webhook_signature",
]
