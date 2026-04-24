"""Trust-level state machine — overlay on top of decision_engine (Decision 7).

Three layers in one file:

1. Constants — :class:`TrustLevel` enum, :data:`ASSISTED_AUTO_WHITELIST`,
   :data:`DANGER_ACTIONS` (derived from ``decision_engine.IRREVERSIBILITY``).
2. Pure overlay — :func:`allowed_action` combines a ``TrustLevel`` with
   :class:`~agent_runtime.decision_engine.Decision.level` into an effective
   :class:`~agent_runtime.models.AutonomyLevel`. No I/O.
3. Async DB helpers — :func:`get_trust_level`, :func:`set_trust_level`,
   :func:`assert_allowed`. Persisted in ``sda_state`` (JSONB value,
   ``key='trust_level'``). Transitions are validated; every change is
   audit-logged atomically with the state update.

The endpoint ``/admin/trust_level`` (Task 5b) queues confirmations via
Telegram and finally calls :func:`set_trust_level`. The ``shadow_monitor``
job (Task 16b) may flip to ``FORBIDDEN_LOCK`` on invariant violation.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime.decision_engine import IRREVERSIBILITY, Decision
from agent_runtime.models import AutonomyLevel

logger = logging.getLogger(__name__)


class TrustLevel(StrEnum):
    SHADOW = "shadow"
    ASSISTED = "assisted"
    AUTONOMOUS = "autonomous"
    FORBIDDEN_LOCK = "FORBIDDEN_LOCK"


# Hypothesis types that may auto-execute in ``assisted``. Anything not in
# this set becomes ``ASK`` even if decision_engine returned AUTO.
ASSISTED_AUTO_WHITELIST: frozenset[str] = frozenset(
    {
        "budget_guard",
        "form_checker",
        "auto_resume",
        "query_analyzer:minus_kw",
    }
)

# Derived from decision_engine.IRREVERSIBILITY so the two don't drift.
# If someone raises a score to 70+, it automatically joins danger tier.
DANGER_ACTIONS: frozenset[str] = frozenset(
    action for action, score in IRREVERSIBILITY.items() if score >= 70
)


_DANGER_UNKNOWN_FALLBACK = AutonomyLevel.ASK

# Allowed shadow/assisted/autonomous transitions. FORBIDDEN_LOCK rules are
# special-cased in :func:`set_trust_level` — the unlock requires a specific
# actor string.
_ALLOWED_TRANSITIONS: frozenset[tuple[TrustLevel, TrustLevel]] = frozenset(
    {
        (TrustLevel.SHADOW, TrustLevel.ASSISTED),
        (TrustLevel.ASSISTED, TrustLevel.AUTONOMOUS),
        (TrustLevel.ASSISTED, TrustLevel.SHADOW),
        (TrustLevel.AUTONOMOUS, TrustLevel.ASSISTED),
        (TrustLevel.AUTONOMOUS, TrustLevel.SHADOW),
        (TrustLevel.SHADOW, TrustLevel.FORBIDDEN_LOCK),
        (TrustLevel.ASSISTED, TrustLevel.FORBIDDEN_LOCK),
        (TrustLevel.AUTONOMOUS, TrustLevel.FORBIDDEN_LOCK),
    }
)

_FORBIDDEN_UNLOCK_ACTOR = "owner-unlock"


def allowed_action(
    action_type: str,
    trust_level: TrustLevel,
    decision_level: AutonomyLevel,
) -> AutonomyLevel:
    """Overlay: trust-level downgrades never upgrade, never by-pass FORBIDDEN.

    Order matters — FORBIDDEN_LOCK wins first, then decision-level FORBIDDEN,
    then shadow/assisted/autonomous policies.
    """
    normalised = action_type.strip().lower() if isinstance(action_type, str) else ""

    if trust_level == TrustLevel.FORBIDDEN_LOCK:
        return AutonomyLevel.FORBIDDEN
    if decision_level == AutonomyLevel.FORBIDDEN:
        return AutonomyLevel.FORBIDDEN

    if trust_level == TrustLevel.SHADOW:
        if decision_level in (AutonomyLevel.AUTO, AutonomyLevel.NOTIFY):
            return AutonomyLevel.NOTIFY
        return decision_level  # ASK passes through

    if trust_level == TrustLevel.ASSISTED:
        if decision_level == AutonomyLevel.AUTO:
            if normalised in {w.lower() for w in ASSISTED_AUTO_WHITELIST}:
                return AutonomyLevel.AUTO
            # Unknown action types in assisted default to ASK — conservative.
            return AutonomyLevel.ASK
        return decision_level

    if trust_level == TrustLevel.AUTONOMOUS:
        if decision_level == AutonomyLevel.AUTO and normalised in DANGER_ACTIONS:
            return AutonomyLevel.ASK
        if normalised not in IRREVERSIBILITY and decision_level == AutonomyLevel.AUTO:
            # Unknown action + AUTO → conservative fallback even in autonomous.
            return _DANGER_UNKNOWN_FALLBACK
        return decision_level

    # Unknown trust level — be conservative.
    logger.warning("trust_levels: unknown trust_level=%r, defaulting to ASK", trust_level)
    return AutonomyLevel.ASK


# --- async DB helpers -------------------------------------------------------


async def get_trust_level(pool: AsyncConnectionPool) -> TrustLevel:
    """Return current trust level; cold-start inserts ``shadow`` + timestamp."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT value #>> '{}' FROM sda_state WHERE key = 'trust_level'")
            row = await cur.fetchone()
            if row and row[0]:
                return TrustLevel(row[0])

            # Cold start: seed shadow + shadow_started_at atomically.
            await cur.execute(
                """
                INSERT INTO sda_state (key, value) VALUES
                    ('trust_level', %s),
                    ('shadow_started_at', %s)
                ON CONFLICT (key) DO NOTHING
                """,
                (Jsonb("shadow"), Jsonb(_now_iso())),
            )
    return TrustLevel.SHADOW


async def set_trust_level(
    pool: AsyncConnectionPool,
    new: TrustLevel,
    *,
    actor: str,
    reason: str,
) -> None:
    """Validated atomic transition + audit_log entry.

    Raises ``ValueError`` on disallowed transition (e.g. ``shadow → autonomous``)
    or on ``FORBIDDEN_LOCK → shadow`` without ``actor == 'owner-unlock'``.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Lock the row for the duration of the transaction — prevents
            # concurrent transitions from racing each other.
            await cur.execute(
                "SELECT value #>> '{}' FROM sda_state WHERE key = 'trust_level' FOR UPDATE"
            )
            row = await cur.fetchone()
            current = TrustLevel(row[0]) if row and row[0] else TrustLevel.SHADOW

            if current == new:
                # Idempotent no-op, but still audit so the log is complete.
                await _audit_trust_change(cur, current, new, actor, reason)
                return

            if current == TrustLevel.FORBIDDEN_LOCK:
                if new != TrustLevel.SHADOW or actor != _FORBIDDEN_UNLOCK_ACTOR:
                    raise ValueError(
                        "FORBIDDEN_LOCK can only be unlocked to shadow with "
                        f"actor='{_FORBIDDEN_UNLOCK_ACTOR}' "
                        f"(got actor={actor!r}, new={new.value!r})"
                    )
            elif (current, new) not in _ALLOWED_TRANSITIONS:
                raise ValueError(f"Invalid transition {current.value} → {new.value}")

            await cur.execute(
                """
                INSERT INTO sda_state (key, value, updated_at)
                VALUES ('trust_level', %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (Jsonb(new.value),),
            )
            await cur.execute(
                """
                INSERT INTO sda_state (key, value, updated_at)
                VALUES ('last_state_change_at', %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (Jsonb(_now_iso()),),
            )
            await _audit_trust_change(cur, current, new, actor, reason)

    logger.info(
        "trust_levels: %s → %s (actor=%s, reason=%s)",
        current.value,
        new.value,
        actor,
        reason,
    )


async def assert_allowed(
    pool: AsyncConnectionPool,
    action_type: str,
    decision: Decision,
) -> AutonomyLevel:
    """Read trust level, apply overlay, log any downgrade."""
    trust = await get_trust_level(pool)
    effective = allowed_action(action_type, trust, decision.level)
    if effective != decision.level:
        logger.info(
            "trust overlay: trust=%s action=%s decision=%s → effective=%s",
            trust.value,
            action_type,
            decision.level.value,
            effective.value,
        )
    return effective


# --- internals --------------------------------------------------------------


async def _audit_trust_change(
    cur,
    old: TrustLevel,
    new: TrustLevel,
    actor: str,
    reason: str,
) -> None:
    await cur.execute(
        """
        INSERT INTO audit_log (
            hypothesis_id, trust_level, tool_name,
            tool_input, tool_output, is_mutation
        )
        VALUES (NULL, %s, 'trust_levels.set', %s, NULL, false)
        """,
        (
            new.value,
            Jsonb(
                {
                    "old": old.value,
                    "new": new.value,
                    "actor": actor,
                    "reason": reason,
                }
            ),
        ),
    )


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


__all__ = [
    "ASSISTED_AUTO_WHITELIST",
    "DANGER_ACTIONS",
    "TrustLevel",
    "allowed_action",
    "assert_allowed",
    "get_trust_level",
    "set_trust_level",
]
