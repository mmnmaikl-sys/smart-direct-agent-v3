"""Unit tests for agent_runtime.trust_levels.

Pure ``allowed_action`` cases are tested without any DB. The async helpers
(``get_trust_level`` / ``set_trust_level`` / ``assert_allowed``) run against
real PG via ``pytest-postgresql`` and skip locally when ``pg_config`` is
missing — same pattern as test_db.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.decision_engine import IRREVERSIBILITY, Decision
from agent_runtime.models import AutonomyLevel
from agent_runtime.trust_levels import (
    ASSISTED_AUTO_WHITELIST,
    DANGER_ACTIONS,
    TrustLevel,
    allowed_action,
)

# ---- pure allowed_action matrix -------------------------------------------


@pytest.mark.parametrize(
    "trust,decision,action,expected",
    [
        # shadow — everything AUTO/NOTIFY → NOTIFY; ASK passes through
        (TrustLevel.SHADOW, AutonomyLevel.AUTO, "add_negative", AutonomyLevel.NOTIFY),
        (TrustLevel.SHADOW, AutonomyLevel.NOTIFY, "add_negative", AutonomyLevel.NOTIFY),
        (TrustLevel.SHADOW, AutonomyLevel.ASK, "switch_strategy", AutonomyLevel.ASK),
        # assisted — only whitelist stays AUTO, rest downgrades to ASK
        (TrustLevel.ASSISTED, AutonomyLevel.AUTO, "budget_guard", AutonomyLevel.AUTO),
        (TrustLevel.ASSISTED, AutonomyLevel.AUTO, "form_checker", AutonomyLevel.AUTO),
        (TrustLevel.ASSISTED, AutonomyLevel.AUTO, "auto_resume", AutonomyLevel.AUTO),
        (
            TrustLevel.ASSISTED,
            AutonomyLevel.AUTO,
            "query_analyzer:minus_kw",
            AutonomyLevel.AUTO,
        ),
        (TrustLevel.ASSISTED, AutonomyLevel.AUTO, "change_budget", AutonomyLevel.ASK),
        (TrustLevel.ASSISTED, AutonomyLevel.AUTO, "add_negative", AutonomyLevel.ASK),
        (TrustLevel.ASSISTED, AutonomyLevel.NOTIFY, "add_negative", AutonomyLevel.NOTIFY),
        (TrustLevel.ASSISTED, AutonomyLevel.ASK, "switch_strategy", AutonomyLevel.ASK),
        # autonomous — danger → ASK, safe → AUTO
        (
            TrustLevel.AUTONOMOUS,
            AutonomyLevel.AUTO,
            "switch_strategy",
            AutonomyLevel.ASK,
        ),
        (TrustLevel.AUTONOMOUS, AutonomyLevel.AUTO, "delete_keyword", AutonomyLevel.ASK),
        (TrustLevel.AUTONOMOUS, AutonomyLevel.AUTO, "add_negative", AutonomyLevel.AUTO),
        (
            TrustLevel.AUTONOMOUS,
            AutonomyLevel.AUTO,
            "some_unknown_action",
            AutonomyLevel.ASK,
        ),
        # FORBIDDEN_LOCK — everything FORBIDDEN
        (TrustLevel.FORBIDDEN_LOCK, AutonomyLevel.AUTO, "add_negative", AutonomyLevel.FORBIDDEN),
        (
            TrustLevel.FORBIDDEN_LOCK,
            AutonomyLevel.NOTIFY,
            "add_negative",
            AutonomyLevel.FORBIDDEN,
        ),
        (
            TrustLevel.FORBIDDEN_LOCK,
            AutonomyLevel.ASK,
            "switch_strategy",
            AutonomyLevel.FORBIDDEN,
        ),
        # decision-level FORBIDDEN wins
        (
            TrustLevel.AUTONOMOUS,
            AutonomyLevel.FORBIDDEN,
            "enable_autotargeting",
            AutonomyLevel.FORBIDDEN,
        ),
    ],
)
def test_allowed_action_matrix(trust, decision, action, expected) -> None:
    assert allowed_action(action, trust, decision) == expected


def test_danger_actions_derived_from_irreversibility() -> None:
    expected = {a for a, s in IRREVERSIBILITY.items() if s >= 70}
    assert DANGER_ACTIONS == frozenset(expected)
    assert "switch_strategy" in DANGER_ACTIONS
    assert "delete_keyword" in DANGER_ACTIONS
    assert "enable_autotargeting" in DANGER_ACTIONS


def test_whitelist_has_expected_members() -> None:
    assert ASSISTED_AUTO_WHITELIST == frozenset(
        {"budget_guard", "form_checker", "auto_resume", "query_analyzer:minus_kw"}
    )


def test_assisted_case_insensitive_whitelist_match() -> None:
    # Accept mis-cased input; conservative path if totally unknown.
    assert (
        allowed_action("Budget_Guard", TrustLevel.ASSISTED, AutonomyLevel.AUTO)
        == AutonomyLevel.AUTO
    )


def test_allowed_action_unknown_trust_level_defaults_to_ask() -> None:
    # Passing something that is not a TrustLevel enum value should not crash.
    class _Fake:
        pass

    result = allowed_action("add_negative", _Fake(), AutonomyLevel.AUTO)  # type: ignore[arg-type]
    assert result == AutonomyLevel.ASK


# ---- mocked-pool tests for DB helpers (run everywhere) ---------------------


def _mock_pool(fetchone_sequence):
    """Build a pool whose cursor.fetchone returns values in order."""
    call_counter = {"n": 0}

    async def _fetchone():
        i = call_counter["n"]
        call_counter["n"] += 1
        return fetchone_sequence[i] if i < len(fetchone_sequence) else None

    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(side_effect=_fetchone)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, cursor


@pytest.mark.asyncio
async def test_get_trust_level_reads_existing_value_mocked() -> None:
    from agent_runtime.trust_levels import get_trust_level

    pool, _cursor = _mock_pool(fetchone_sequence=[("assisted",)])
    result = await get_trust_level(pool)
    assert result == TrustLevel.ASSISTED


@pytest.mark.asyncio
async def test_get_trust_level_cold_start_mocked() -> None:
    from agent_runtime.trust_levels import get_trust_level

    pool, cursor = _mock_pool(fetchone_sequence=[None])
    result = await get_trust_level(pool)
    assert result == TrustLevel.SHADOW
    # Verify we issued the cold-start INSERT statement
    executed_sqls = [call.args[0] for call in cursor.execute.await_args_list]
    assert any("INSERT INTO sda_state" in sql for sql in executed_sqls)


@pytest.mark.asyncio
async def test_set_trust_level_valid_transition_mocked() -> None:
    from agent_runtime.trust_levels import set_trust_level

    pool, cursor = _mock_pool(fetchone_sequence=[("shadow",)])
    await set_trust_level(pool, TrustLevel.ASSISTED, actor="owner-via-telegram", reason="promote")
    # Expected calls: SELECT ... FOR UPDATE, UPSERT sda_state, UPSERT
    # last_state_change_at, INSERT audit_log
    assert cursor.execute.await_count >= 4


@pytest.mark.asyncio
async def test_set_trust_level_invalid_transition_raises_mocked() -> None:
    from agent_runtime.trust_levels import set_trust_level

    pool, _cursor = _mock_pool(fetchone_sequence=[("shadow",)])
    with pytest.raises(ValueError, match="shadow → autonomous"):
        await set_trust_level(
            pool, TrustLevel.AUTONOMOUS, actor="owner-via-telegram", reason="skip"
        )


@pytest.mark.asyncio
async def test_set_trust_level_forbidden_lock_wrong_actor_raises_mocked() -> None:
    from agent_runtime.trust_levels import set_trust_level

    pool, _cursor = _mock_pool(fetchone_sequence=[("FORBIDDEN_LOCK",)])
    with pytest.raises(ValueError, match="FORBIDDEN_LOCK"):
        await set_trust_level(pool, TrustLevel.SHADOW, actor="auto", reason="unlock try")


@pytest.mark.asyncio
async def test_set_trust_level_forbidden_lock_owner_unlock_ok_mocked() -> None:
    from agent_runtime.trust_levels import set_trust_level

    pool, cursor = _mock_pool(fetchone_sequence=[("FORBIDDEN_LOCK",)])
    await set_trust_level(pool, TrustLevel.SHADOW, actor="owner-unlock", reason="manual")
    assert cursor.execute.await_count >= 4  # lock + 2 upserts + audit


@pytest.mark.asyncio
async def test_set_trust_level_same_state_noop_audits_mocked() -> None:
    from agent_runtime.trust_levels import set_trust_level

    pool, cursor = _mock_pool(fetchone_sequence=[("shadow",)])
    await set_trust_level(pool, TrustLevel.SHADOW, actor="owner-via-telegram", reason="noop")
    # SELECT + audit INSERT (no sda_state upserts because state unchanged)
    assert cursor.execute.await_count == 2


@pytest.mark.asyncio
async def test_assert_allowed_logs_downgrade_mocked(caplog: pytest.LogCaptureFixture) -> None:
    from agent_runtime.trust_levels import assert_allowed

    pool, _cursor = _mock_pool(fetchone_sequence=[("shadow",)])
    decision = Decision(
        action="add_negative",
        level=AutonomyLevel.AUTO,
        risk_score=10.0,
        reason="",
        can_execute=True,
        affected_budget_pct=0.0,
        data_points_effective=30.0,
    )
    caplog.set_level("INFO", logger="agent_runtime.trust_levels")
    result = await assert_allowed(pool, "add_negative", decision)
    assert result == AutonomyLevel.NOTIFY  # shadow downgrades AUTO → NOTIFY
    assert any("trust overlay" in r.message for r in caplog.records)


# ---- async DB helpers (pytest-postgresql) ---------------------------------


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"
_HAS_PG = shutil.which("pg_config") is not None
_SKIP_NO_PG = pytest.mark.skipif(not _HAS_PG, reason="pg_config not available locally")


if _HAS_PG:
    from pytest_postgresql import factories as pg_factories

    postgresql_proc = pg_factories.postgresql_proc(port=None)
    postgresql = pg_factories.postgresql("postgresql_proc")


def _dsn(pg) -> str:  # noqa: ANN001
    i = pg.info
    return f"postgresql://{i.user}:{i.password or ''}@{i.host}:{i.port}/{i.dbname}"


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_get_trust_level_cold_start(postgresql) -> None:  # noqa: ANN001
    from agent_runtime.db import create_pool, run_migrations
    from agent_runtime.trust_levels import get_trust_level

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        level = await get_trust_level(pool)
        assert level == TrustLevel.SHADOW

        # After cold start, sda_state has both trust_level and shadow_started_at.
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT key FROM sda_state ORDER BY key")
                keys = [r[0] for r in await cur.fetchall()]
        assert "trust_level" in keys
        assert "shadow_started_at" in keys
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_set_trust_level_shadow_to_assisted(postgresql) -> None:  # noqa: ANN001
    from agent_runtime.db import create_pool, run_migrations
    from agent_runtime.trust_levels import get_trust_level, set_trust_level

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        await get_trust_level(pool)  # seed shadow
        await set_trust_level(pool, TrustLevel.ASSISTED, actor="owner-via-telegram", reason="test")
        assert await get_trust_level(pool) == TrustLevel.ASSISTED

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT tool_name, trust_level FROM audit_log "
                    "WHERE tool_name='trust_levels.set' ORDER BY id DESC LIMIT 1"
                )
                row = await cur.fetchone()
        assert row is not None
        assert row[1] == "assisted"
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_set_trust_level_shadow_to_autonomous_raises(postgresql) -> None:  # noqa: ANN001
    from agent_runtime.db import create_pool, run_migrations
    from agent_runtime.trust_levels import get_trust_level, set_trust_level

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        await get_trust_level(pool)
        with pytest.raises(ValueError, match="shadow → autonomous"):
            await set_trust_level(
                pool,
                TrustLevel.AUTONOMOUS,
                actor="owner-via-telegram",
                reason="skip the gate",
            )
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
@pytest.mark.parametrize(
    "from_state",
    [TrustLevel.SHADOW, TrustLevel.ASSISTED, TrustLevel.AUTONOMOUS],
)
async def test_set_trust_level_to_forbidden_lock_always_allowed(
    postgresql, from_state: TrustLevel
) -> None:  # noqa: ANN001
    from agent_runtime.db import create_pool, run_migrations
    from agent_runtime.trust_levels import get_trust_level, set_trust_level

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        await get_trust_level(pool)
        # Ramp up to from_state via allowed transitions first
        if from_state != TrustLevel.SHADOW:
            await set_trust_level(
                pool, TrustLevel.ASSISTED, actor="owner-via-telegram", reason="ramp"
            )
        if from_state == TrustLevel.AUTONOMOUS:
            await set_trust_level(
                pool, TrustLevel.AUTONOMOUS, actor="owner-via-telegram", reason="ramp"
            )

        await set_trust_level(
            pool, TrustLevel.FORBIDDEN_LOCK, actor="shadow-monitor", reason="invariant"
        )
        assert await get_trust_level(pool) == TrustLevel.FORBIDDEN_LOCK
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_forbidden_lock_unlock_requires_owner_actor(postgresql) -> None:  # noqa: ANN001
    from agent_runtime.db import create_pool, run_migrations
    from agent_runtime.trust_levels import get_trust_level, set_trust_level

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        await get_trust_level(pool)
        await set_trust_level(
            pool, TrustLevel.FORBIDDEN_LOCK, actor="shadow-monitor", reason="lock"
        )
        # Wrong actor
        with pytest.raises(ValueError, match="FORBIDDEN_LOCK"):
            await set_trust_level(pool, TrustLevel.SHADOW, actor="auto", reason="try unlock")
        # Correct actor
        await set_trust_level(pool, TrustLevel.SHADOW, actor="owner-unlock", reason="manual")
        assert await get_trust_level(pool) == TrustLevel.SHADOW
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_assert_allowed_integration(postgresql) -> None:  # noqa: ANN001
    from agent_runtime.db import create_pool, run_migrations
    from agent_runtime.trust_levels import assert_allowed, get_trust_level, set_trust_level

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        await get_trust_level(pool)
        await set_trust_level(pool, TrustLevel.ASSISTED, actor="owner-via-telegram", reason="test")
        whitelist_decision = Decision(
            action="form_checker",
            level=AutonomyLevel.AUTO,
            risk_score=20.0,
            reason="",
            can_execute=True,
            affected_budget_pct=0.0,
            data_points_effective=30.0,
        )
        assert await assert_allowed(pool, "form_checker", whitelist_decision) == AutonomyLevel.AUTO
        non_whitelist_decision = Decision(
            action="change_budget",
            level=AutonomyLevel.AUTO,
            risk_score=20.0,
            reason="",
            can_execute=True,
            affected_budget_pct=0.0,
            data_points_effective=30.0,
        )
        assert (
            await assert_allowed(pool, "change_budget", non_whitelist_decision) == AutonomyLevel.ASK
        )
    finally:
        await pool.close()


@pytest.mark.asyncio
@_SKIP_NO_PG
async def test_set_trust_level_idempotent_same_state(postgresql) -> None:  # noqa: ANN001
    from agent_runtime.db import create_pool, run_migrations
    from agent_runtime.trust_levels import get_trust_level, set_trust_level

    pool = create_pool(_dsn(postgresql), min_size=1, max_size=2)
    await pool.open()
    try:
        await run_migrations(pool, MIGRATIONS_DIR)
        await get_trust_level(pool)  # shadow
        # Same state — audit but don't raise
        await set_trust_level(pool, TrustLevel.SHADOW, actor="owner-via-telegram", reason="noop")
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) FROM audit_log WHERE tool_name='trust_levels.set'"
                )
                row = await cur.fetchone()
        assert row is not None and row[0] >= 1
    finally:
        await pool.close()
