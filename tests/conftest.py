"""Session-wide test fixtures.

Dummy required env vars are set BEFORE any ``agent_runtime`` import so the
module-level ``app = create_app()`` in ``agent_runtime.main`` does not fail at
import time (pydantic-settings raises when ``DATABASE_URL`` is missing).
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
# Dummy 64-char hex strings. Real secrets ship via Railway env (openssl rand -hex 32).
# Keep these distinct from each other so tests can detect accidental secret reuse.
os.environ.setdefault("SDA_INTERNAL_API_KEY", "a" * 64)
os.environ.setdefault("SDA_WEBHOOK_HMAC_SECRET", "b" * 64)
os.environ.setdefault("HYPOTHESIS_HMAC_SECRET", "c" * 64)
os.environ.setdefault("PII_SALT", "pii-test-salt-" + "0" * 32)
