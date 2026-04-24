"""Unit tests for agent_runtime.decision_journal.

Mocked pool — DB logic is exercised by assertions on the SQL statements
issued and the args passed. Integration tests (real PG) live in
``tests/integration/test_impact_tracker_e2e.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.decision_journal import (
    REVERSE_ACTION_MAP,
    FlipFlopEvent,
    detect_flip_flop,
    get_actions_today,
    get_pending_checks,
    record_hypothesis,
    update_outcome,
)
from agent_runtime.models import HypothesisDraft, HypothesisType, Signal, SignalType


def _fake_signal() -> Signal:
    return Signal(
        type=SignalType.HIGH_CPA,
        severity="warning",
        data={"campaign_id": 708978456},
        ts=datetime.now(UTC),
    )


def _draft(
    hypothesis_type: HypothesisType = HypothesisType.AD,
    ad_group_id: int | None = 123,
    campaign_id: int | None = None,
) -> HypothesisDraft:
    return HypothesisDraft(
        hypothesis_type=hypothesis_type,
        hypothesis="test hypothesis",
        reasoning="test reasoning",
        actions=[{"type": "add_keyword", "params": {"keyword": "kw"}}],
        expected_outcome="ctr > 2%",
        ad_group_id=ad_group_id,
        campaign_id=campaign_id,
    )


def _pool_with_rows(fetchone_sequence: list[Any], fetchall_sequence: list[Any] | None = None):
    """Build a mock pool that walks through a fixed sequence of fetch* results."""
    one_iter = iter(fetchone_sequence)
    all_iter = iter(fetchall_sequence or [])

    async def _fetchone():
        try:
            return next(one_iter)
        except StopIteration:
            return None

    async def _fetchall():
        try:
            return next(all_iter)
        except StopIteration:
            return []

    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(side_effect=_fetchone)
    cursor.fetchall = AsyncMock(side_effect=_fetchall)
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, cursor


# ---- record_hypothesis ------------------------------------------------------


@pytest.mark.asyncio
async def test_record_hypothesis_inserts_running_when_bucket_free() -> None:
    pool, cursor = _pool_with_rows([(0,)])  # mutations_this_week = 0
    hid = await record_hypothesis(pool, _draft(), [_fake_signal()], {"ctr": 2.0})
    assert len(hid) == 8  # uuid4 hex prefix
    sqls = [call.args[0] for call in cursor.execute.await_args_list]
    insert_sql = next(s for s in sqls if "INSERT INTO hypotheses" in s)
    assert "state" in insert_sql.lower()
    # verify state param is 'running'
    insert_call = next(
        call for call in cursor.execute.await_args_list if "INSERT INTO hypotheses" in call.args[0]
    )
    assert "running" in insert_call.args[1]


@pytest.mark.asyncio
async def test_record_hypothesis_queues_when_bucket_full() -> None:
    # 11_500 already spent; ad cap=500 → fits in 12_000; new_camp cap=3500 → waiting.
    pool, _ = _pool_with_rows([(11_500,)])
    draft = _draft(hypothesis_type=HypothesisType.NEW_CAMP, campaign_id=999, ad_group_id=None)
    await record_hypothesis(pool, draft, [], {"leads": 0})
    # The only way to see the final state is to check the INSERT payload.
    # _pool_with_rows patched execute — grab the last INSERT call.
    conn = pool.connection.return_value.__aenter__.return_value
    all_calls = conn.cursor.return_value.execute.await_args_list
    insert_call = next(c for c in all_calls if "INSERT INTO hypotheses" in c.args[0])
    assert "waiting_budget" in insert_call.args[1]


@pytest.mark.asyncio
async def test_record_hypothesis_cap_ceiling_exact_fit() -> None:
    # 11_500 spent, ad cap=500 → exact 12_000 → still RUNNING (≤ free).
    pool, _ = _pool_with_rows([(11_500,)])
    await record_hypothesis(pool, _draft(), [], {})
    conn = pool.connection.return_value.__aenter__.return_value
    all_calls = conn.cursor.return_value.execute.await_args_list
    insert_call = next(c for c in all_calls if "INSERT INTO hypotheses" in c.args[0])
    assert "running" in insert_call.args[1]


# ---- update_outcome --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,expected_state",
    [
        ("positive", "confirmed"),
        ("negative", "rejected"),
        ("neutral", "inconclusive"),
    ],
)
async def test_update_outcome_maps_correctly(outcome: str, expected_state: str) -> None:
    pool, cursor = _pool_with_rows([])
    await update_outcome(pool, "abc123", outcome, {"ctr": 2.5}, "test lesson")  # type: ignore[arg-type]
    update_call = cursor.execute.await_args_list[0]
    assert "UPDATE hypotheses" in update_call.args[0]
    params = update_call.args[1]
    assert outcome in params
    assert expected_state in params


# ---- get_pending_checks ----------------------------------------------------


@pytest.mark.asyncio
async def test_get_pending_checks_returns_rows_older_than_72h() -> None:
    sample_row = (
        "abc12345",
        "ad",
        "brain",
        None,
        123,
        "running",
        [{"type": "add_keyword"}],
        {"ctr": 2.0},
        500,
        datetime.now(UTC) - timedelta(hours=100),
        14,
    )
    pool, _ = _pool_with_rows(fetchone_sequence=[], fetchall_sequence=[[sample_row]])
    rows = await get_pending_checks(pool)
    assert len(rows) == 1
    assert rows[0].id == "abc12345"
    assert rows[0].hypothesis_type == "ad"
    assert rows[0].ad_group_id == 123
    assert rows[0].maximum_running_days == 14


@pytest.mark.asyncio
async def test_get_pending_checks_empty() -> None:
    pool, _ = _pool_with_rows(fetchone_sequence=[], fetchall_sequence=[[]])
    rows = await get_pending_checks(pool)
    assert rows == []


# ---- get_actions_today -----------------------------------------------------


@pytest.mark.asyncio
async def test_get_actions_today_returns_structured_rows() -> None:
    row = (
        1,
        datetime.now(UTC),
        "h1",
        "autonomous",
        "direct.set_bid",
        {"keyword_id": 42, "bid_rub": 25},
        {"success": True},
        True,
        False,
    )
    pool, _ = _pool_with_rows(fetchone_sequence=[], fetchall_sequence=[[row]])
    actions = await get_actions_today(pool)
    assert len(actions) == 1
    assert actions[0]["tool_name"] == "direct.set_bid"
    assert actions[0]["is_mutation"] is True


# ---- detect_flip_flop ------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_flip_flop_finds_pause_resume_pair() -> None:
    t1 = datetime.now(UTC) - timedelta(days=5)
    t2 = datetime.now(UTC) - timedelta(days=3)
    rows = [
        ("h1", [{"type": "pause_group"}], t1),
        ("h2", [{"type": "resume_group"}], t2),
    ]
    pool, _ = _pool_with_rows(fetchone_sequence=[], fetchall_sequence=[rows])
    events = await detect_flip_flop(pool, ad_group_id=123, window_days=30)
    assert len(events) == 1
    assert events[0].action_pair == ("pause_group", "resume_group")
    assert events[0].pair_ids == ("h1", "h2")


@pytest.mark.asyncio
async def test_detect_flip_flop_classifies_severity() -> None:
    # 2-day gap → critical (< 7d)
    t1 = datetime.now(UTC) - timedelta(days=5)
    t2 = datetime.now(UTC) - timedelta(days=3)
    rows = [
        ("h1", [{"type": "pause_campaign"}], t1),
        ("h2", [{"type": "resume_campaign"}], t2),
    ]
    pool, _ = _pool_with_rows(fetchone_sequence=[], fetchall_sequence=[rows])
    events = await detect_flip_flop(pool, campaign_id=708978456, window_days=30)
    assert events and events[0].severity == "critical"


@pytest.mark.asyncio
async def test_detect_flip_flop_no_match_on_unrelated_actions() -> None:
    rows = [
        ("h1", [{"type": "set_bid"}], datetime.now(UTC) - timedelta(days=2)),
        ("h2", [{"type": "add_keyword"}], datetime.now(UTC)),
    ]
    pool, _ = _pool_with_rows(fetchone_sequence=[], fetchall_sequence=[rows])
    events = await detect_flip_flop(pool, ad_group_id=42)
    assert events == []


@pytest.mark.asyncio
async def test_detect_flip_flop_requires_target_id() -> None:
    # When both campaign_id and ad_group_id are None the helper short-circuits.
    pool, _ = _pool_with_rows([])
    events = await detect_flip_flop(pool)
    assert events == []


def test_reverse_action_map_covers_canonical_pairs() -> None:
    for name in (
        "pause_group",
        "resume_group",
        "pause_campaign",
        "resume_campaign",
        "set_bid",
        "add_negatives",
    ):
        assert name in REVERSE_ACTION_MAP


def test_reverse_set_bid_requires_original_bid() -> None:
    action = {"type": "set_bid", "params": {"keyword_id": 1, "ad_group_id": 2, "bid_rub": 50}}
    # metrics_before without original_bid → reverser raises.
    with pytest.raises(ValueError, match="original_bid"):
        REVERSE_ACTION_MAP["set_bid"](action, {})


def test_reverse_set_bid_uses_original_bid_from_metrics() -> None:
    action = {"type": "set_bid", "params": {"keyword_id": 1, "ad_group_id": 2, "bid_rub": 50}}
    reverse = REVERSE_ACTION_MAP["set_bid"](action, {"original_bid": 30})
    assert reverse["type"] == "set_bid"
    assert reverse["params"]["bid_rub"] == 30


def test_flip_flop_event_is_frozen_dataclass() -> None:
    from dataclasses import FrozenInstanceError

    event = FlipFlopEvent(
        pair_ids=("a", "b"),
        action_pair=("pause_group", "resume_group"),
        interval_days=1.5,
        severity="critical",
    )
    with pytest.raises(FrozenInstanceError):
        event.severity = "warn"  # type: ignore[misc]
