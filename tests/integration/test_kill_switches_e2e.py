"""E2E tests for kill-switches + brain wrapper integration.

These verify the full path: HypothesisDraft.action → run_all → brain wrapper
writes ``audit_log.kill_switch_triggered`` + sends Telegram NOTIFY + leaves
hypothesis state=running.

The brain wrapper itself lands in Task 12; until then this file is a skeleton
that skips the scenarios which require brain code. Unit coverage in
``tests/unit/test_kill_switches.py`` proves every guard in isolation — the
integration tier just proves wiring once Task 12 stubs the wrapper.
"""

from __future__ import annotations

import shutil

import pytest

_HAS_PG = shutil.which("pg_config") is not None
_BRAIN_READY = False  # flipped once Task 12 lands agent_runtime.brain

_SKIP_NO_PG = pytest.mark.skipif(not _HAS_PG, reason="local PostgreSQL (pg_config) not available")
_SKIP_NO_BRAIN = pytest.mark.skipif(
    not _BRAIN_READY, reason="brain wrapper (Task 12) not yet implemented"
)


@pytest.mark.asyncio
@_SKIP_NO_PG
@_SKIP_NO_BRAIN
@pytest.mark.parametrize(
    "switch_name",
    [
        "budget_cap",
        "cpc_ceiling",
        "neg_kw_floor",
        "qs_guard",
        "budget_balance",
        "conversion_integrity",
        "query_drift",
    ],
)
async def test_each_kill_switch_triggers_rejects_and_logs(switch_name: str) -> None:  # noqa: ARG001
    """Scaffolded — implement in Task 12 once brain wrapper exists.

    For each switch_name:
        1. Seed trigger condition (budget history / productivity / ...).
        2. Call brain wrapper with a HypothesisDraft whose action is
           crafted to fire exactly that guard.
        3. Assert audit_log row with kill_switch_triggered=<switch_name>,
           is_mutation=false. Hypothesis row state unchanged ('running').
        4. Assert Telegram mock received NOTIFY with the reason string.
    """
    pytest.skip("brain wrapper (Task 12) pending")


@pytest.mark.asyncio
@_SKIP_NO_PG
@_SKIP_NO_BRAIN
async def test_multiple_kill_switches_can_fire_simultaneously() -> None:
    """Two rejects at once — BudgetCap + QSGuard — must both log.

    Scaffolded for Task 12. Unit-level counterpart lives in
    ``tests/unit/test_kill_switches.py::test_run_all_multiple_guards_can_fire_simultaneously``
    and already proves the underlying ``run_all`` behaviour.
    """
    pytest.skip("brain wrapper (Task 12) pending")
