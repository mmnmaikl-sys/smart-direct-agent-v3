"""HMAC helpers for webhook signatures and Telegram callback_data.

Two distinct use cases — same primitive (HMAC-SHA256), different envelopes:

* **Webhook signatures** — HTTP POST bodies from Bitrix / lead webhooks. Sender
  includes ``X-SDA-Signature`` and ``X-SDA-Timestamp`` headers; replay window is
  300 s (Decision 11). Format: ``sha256=<64 hex>``.

* **Telegram callback_data** — inline-button payload, must fit in Telegram's
  64-byte limit. Compact format: ``{hypothesis_id}:{action}:{sig10}`` where
  ``sig10`` is the first 10 hex chars of HMAC-SHA256(secret, id + ":" + action).
  10 hex chars = 40 bits of integrity — weak vs full SHA256, but acceptable for
  a single-tenant bot where the attacker would need to compromise the Telegram
  chat to deliver a forged callback in the first place.

Two different secrets are used (``SDA_WEBHOOK_HMAC_SECRET`` and
``HYPOTHESIS_HMAC_SECRET``) so that leaking one does not compromise the other.
All compares go through ``hmac.compare_digest`` to avoid timing oracles.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Final

from pydantic import SecretStr

SIGNATURE_PREFIX: Final[str] = "sha256="
DEFAULT_REPLAY_WINDOW_SEC: Final[int] = 300
CALLBACK_SIG_LEN: Final[int] = 10  # first 10 hex chars (40 bits) for Telegram 64-byte limit


class HMACSigner:
    """HMAC-SHA256 signer/verifier.

    Instantiated once per secret (``SDA_WEBHOOK_HMAC_SECRET`` for webhooks,
    ``HYPOTHESIS_HMAC_SECRET`` for Telegram callbacks) — secrets must never
    cross use cases. Accepts a ``SecretStr`` so accidental logs print ``***``.
    """

    def __init__(self, secret: SecretStr) -> None:
        self._secret = secret.get_secret_value().encode("utf-8")

    # ------------------------------------------------------------------ webhook

    def sign(self, body: bytes, timestamp: int) -> str:
        """Return ``sha256=<hex>`` signature over ``f"{timestamp}.{body}"``."""
        payload = f"{timestamp}.".encode() + body
        digest = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        return f"{SIGNATURE_PREFIX}{digest}"

    def verify(
        self,
        body: bytes,
        timestamp: int,
        signature: str,
        *,
        max_age_sec: int = DEFAULT_REPLAY_WINDOW_SEC,
        now: int | None = None,
    ) -> bool:
        """Verify signature + replay window.

        Returns True only if the timestamp is within ``max_age_sec`` AND the
        recomputed signature matches ``signature`` under a timing-safe compare.
        Never raises — callers can 401 on a False result.
        """
        now_ts = int(time.time()) if now is None else now
        if abs(now_ts - timestamp) > max_age_sec:
            return False
        expected = self.sign(body, timestamp)
        return hmac.compare_digest(expected, signature)

    # ---------------------------------------------------------------- callback

    def sign_callback(self, hypothesis_id: str, action: str) -> str:
        """Compact signed payload for Telegram inline-button callback_data.

        Format: ``{hypothesis_id}:{action}:{sig10}``. Kept ≤ 40-50 chars so it
        fits Telegram's 64-byte callback_data budget even with long IDs.
        """
        if ":" in hypothesis_id or ":" in action:
            raise ValueError("hypothesis_id and action must not contain ':'")
        payload = f"{hypothesis_id}:{action}".encode()
        digest = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        return f"{hypothesis_id}:{action}:{digest[:CALLBACK_SIG_LEN]}"

    def verify_callback(self, data: str) -> tuple[str, str]:
        """Parse and verify a signed callback_data string.

        Raises ``ValueError`` on bad format or signature mismatch.
        Returns ``(hypothesis_id, action)`` on success.
        """
        parts = data.split(":")
        if len(parts) != 3:
            raise ValueError("callback_data must be 'id:action:sig'")
        hypothesis_id, action, provided_sig = parts
        if len(provided_sig) != CALLBACK_SIG_LEN:
            raise ValueError("invalid callback signature length")
        payload = f"{hypothesis_id}:{action}".encode()
        expected = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()[:CALLBACK_SIG_LEN]
        if not hmac.compare_digest(expected, provided_sig):
            raise ValueError("invalid callback signature")
        return hypothesis_id, action
