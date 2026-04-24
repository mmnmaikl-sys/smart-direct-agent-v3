"""Session-wide test fixtures.

Dummy required env vars are set BEFORE any ``agent_runtime`` import so the
module-level ``app = create_app()`` in ``agent_runtime.main`` does not fail at
import time (pydantic-settings raises when ``DATABASE_URL`` is missing).
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("SDA_INTERNAL_API_KEY", "test-internal-" + "0" * 48)
os.environ.setdefault("SDA_WEBHOOK_HMAC_SECRET", "test-webhook-" + "0" * 48)
os.environ.setdefault("HYPOTHESIS_HMAC_SECRET", "test-hypothesis-" + "0" * 48)
