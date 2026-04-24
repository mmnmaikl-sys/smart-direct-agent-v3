"""Unit tests for agent_runtime.impact_tracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.impact_tracker import (
    MIN_CLICKS_FOR_CONFIDENCE,
    Outcome,
    _HypothesisSnapshot,
    mark_expired,
    measure_outcome,
    promote_to_prod,
    release_bucket_and_start_waiting,
    rollback,
)


def _snapshot(**kw) -> _HypothesisSnapshot:
    base: dict[str, Any] = dict(
        id="hyp1",
        hypothesis_type="ad",
        campaign_id=708978456,
        ad_group_id=42,
        state="running",
        actions=[{"type": "pause_group", "params": {"ad_group_id": 42}}],
        metrics_before={"ctr": 2.0, "clicks": 10},
        created_at=datetime.now(UTC) - timedelta(hours=80),
        budget_cap_rub=500,
    )
    base.update(kw)
    return _HypothesisSnapshot(**base)


def _mock_pool(fetchone_sequence=None, fetchall_sequence=None, rowcount: int = 0):
    one_iter = iter(fetchone_sequence or [])
    all_iter = iter(fetchall_sequence or [])

    async def _fetchone():
        try:
            return next(one_iter)
        except StopIteration:
            # Default so audit_log INSERT ... RETURNING id (via
            # insert_audit_log) always resolves to a plausible row.
            return (1,)

    async def _fetchall():
        try:
            return next(all_iter)
        except StopIteration:
            return []

    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(side_effect=_fetchone)
    cursor.fetchall = AsyncMock(side_effect=_fetchall)
    cursor.rowcount = rowcount
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, cursor


def _snapshot_row(snapshot: _HypothesisSnapshot) -> tuple:
    return (
        snapshot.id,
        snapshot.hypothesis_type,
        snapshot.campaign_id,
        snapshot.ad_group_id,
        snapshot.state,
        snapshot.actions,
        snapshot.metrics_before,
        snapshot.created_at,
        snapshot.budget_cap_rub,
    )


def _direct_stub() -> SimpleNamespace:
    return SimpleNamespace(
        get_campaign_stats=AsyncMock(return_value={}),
        pause_group=AsyncMock(return_value={}),
        resume_group=AsyncMock(return_value={}),
        pause_campaign=AsyncMock(return_value={}),
        resume_campaign=AsyncMock(return_value={}),
        set_bid=AsyncMock(return_value={}),
        add_negatives=AsyncMock(return_value={}),
    )


# ---- measure_outcome -------------------------------------------------------


@pytest.mark.asyncio
async def test_measure_outcome_classifies_positive() -> None:
    snapshot = _snapshot(metrics_before={"ctr": 2.0, "clicks": 10})
    pool, _ = _mock_pool(fetchone_sequence=[_snapshot_row(snapshot)])
    direct = _direct_stub()
    reflection = SimpleNamespace(save=AsyncMock())
    outcome = await measure_outcome(
        pool,
        "hyp1",
        direct=direct,
        reflection_store=reflection,
        current_metrics={"ctr": 2.5, "clicks": 100},  # +25% → positive
    )
    assert outcome.classification == "positive"
    assert outcome.confidence == "high"
    assert outcome.delta_pct == pytest.approx(0.25, abs=0.01)
    # update_outcome → positive → state=confirmed → promote_to_prod UPDATE
    reflection.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_measure_outcome_classifies_negative_triggers_rollback() -> None:
    snapshot = _snapshot(metrics_before={"ctr": 3.0, "clicks": 10})
    # _load_snapshot → measure_outcome → rollback → _load_snapshot again (in rollback)
    pool, _ = _mock_pool(fetchone_sequence=[_snapshot_row(snapshot)])
    direct = _direct_stub()
    reflection = SimpleNamespace(save=AsyncMock())
    outcome = await measure_outcome(
        pool,
        "hyp1",
        direct=direct,
        reflection_store=reflection,
        current_metrics={"ctr": 2.0, "clicks": 100},  # -33% → negative
    )
    assert outcome.classification == "negative"
    # rollback replayed the inverse: pause_group → resume_group on ad_group_id=42
    direct.resume_group.assert_awaited_with(42)


@pytest.mark.asyncio
async def test_measure_outcome_neutral_on_low_confidence() -> None:
    snapshot = _snapshot()
    pool, _ = _mock_pool(fetchone_sequence=[_snapshot_row(snapshot)])
    direct = _direct_stub()
    outcome = await measure_outcome(
        pool,
        "hyp1",
        direct=direct,
        current_metrics={"ctr": 4.0, "clicks": MIN_CLICKS_FOR_CONFIDENCE - 1},
    )
    assert outcome.classification == "neutral"
    assert outcome.confidence == "low"
    assert "insufficient signal" in outcome.lesson


@pytest.mark.asyncio
async def test_measure_outcome_neutral_on_small_delta() -> None:
    snapshot = _snapshot(metrics_before={"ctr": 3.0, "clicks": 10})
    pool, _ = _mock_pool(fetchone_sequence=[_snapshot_row(snapshot)])
    direct = _direct_stub()
    outcome = await measure_outcome(
        pool,
        "hyp1",
        direct=direct,
        current_metrics={"ctr": 3.1, "clicks": 100},  # +3% → neutral
    )
    assert outcome.classification == "neutral"
    assert outcome.confidence == "high"


@pytest.mark.asyncio
async def test_measure_outcome_skips_already_terminal() -> None:
    snapshot = _snapshot(state="confirmed")
    pool, _ = _mock_pool(fetchone_sequence=[_snapshot_row(snapshot)])
    direct = _direct_stub()
    outcome = await measure_outcome(
        pool, "hyp1", direct=direct, current_metrics={"ctr": 4.0, "clicks": 100}
    )
    assert outcome.classification == "neutral"
    assert "terminal state" in outcome.lesson
    direct.resume_group.assert_not_awaited()


@pytest.mark.asyncio
async def test_measure_outcome_missing_hypothesis_raises() -> None:
    pool, _ = _mock_pool(fetchone_sequence=[None])
    with pytest.raises(LookupError, match="not found"):
        await measure_outcome(pool, "nope", direct=_direct_stub(), current_metrics={})


@pytest.mark.asyncio
async def test_measure_outcome_handles_reflection_failure() -> None:
    snapshot = _snapshot()
    pool, _ = _mock_pool(fetchone_sequence=[_snapshot_row(snapshot)])
    direct = _direct_stub()
    bad_reflection = SimpleNamespace(save=AsyncMock(side_effect=RuntimeError("pg down")))
    # Should not raise — outcome still returned.
    outcome = await measure_outcome(
        pool,
        "hyp1",
        direct=direct,
        reflection_store=bad_reflection,
        current_metrics={"ctr": 2.5, "clicks": 100},
    )
    assert outcome.classification == "positive"


# ---- promote_to_prod / rollback -------------------------------------------


@pytest.mark.asyncio
async def test_promote_to_prod_sets_baseline_at_promote() -> None:
    pool, cursor = _mock_pool()
    await promote_to_prod(pool, "hyp1", metrics_after={"ctr": 2.5, "leads": 3})
    call = cursor.execute.await_args_list[0]
    assert "state = 'confirmed'" in call.args[0]
    assert "baseline_at_promote" in call.args[0]
    assert "NOW()" in call.args[0]


@pytest.mark.asyncio
async def test_rollback_inverts_pause_group() -> None:
    snapshot = _snapshot(actions=[{"type": "pause_group", "params": {"ad_group_id": 42}}])
    # snapshot is passed explicitly → _load_snapshot is NOT called; fetchone
    # only serves audit_log RETURNING id (default (1,)).
    pool, _ = _mock_pool()
    direct = _direct_stub()
    await rollback(pool, "hyp1", direct=direct, snapshot=snapshot)
    direct.resume_group.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_rollback_inverts_set_bid_using_original() -> None:
    snapshot = _snapshot(
        actions=[
            {
                "type": "set_bid",
                "params": {"keyword_id": 7, "ad_group_id": 42, "bid_rub": 50},
            }
        ],
        metrics_before={"ctr": 2.0, "clicks": 10, "original_bid": 30},
    )
    pool, _ = _mock_pool(fetchone_sequence=[])
    direct = _direct_stub()
    await rollback(pool, "hyp1", direct=direct, snapshot=snapshot)
    direct.set_bid.assert_awaited_once_with(7, bid_rub=30, context_bid_rub=None)


@pytest.mark.asyncio
async def test_rollback_skips_unknown_action_types() -> None:
    snapshot = _snapshot(actions=[{"type": "future_action_xyz", "params": {"foo": 1}}])
    pool, _ = _mock_pool(fetchone_sequence=[])
    direct = _direct_stub()
    await rollback(pool, "hyp1", direct=direct, snapshot=snapshot)
    # No direct mutation was performed.
    direct.pause_group.assert_not_awaited()
    direct.resume_group.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollback_skips_set_bid_without_original() -> None:
    snapshot = _snapshot(
        actions=[
            {
                "type": "set_bid",
                "params": {"keyword_id": 7, "ad_group_id": 42, "bid_rub": 50},
            }
        ],
        metrics_before={"clicks": 10},  # no original_bid
    )
    pool, _ = _mock_pool()
    direct = _direct_stub()
    # reverser raises; rollback should audit and continue.
    await rollback(pool, "hyp1", direct=direct, snapshot=snapshot)
    direct.set_bid.assert_not_awaited()


# ---- release_bucket_and_start_waiting -------------------------------------


@pytest.mark.asyncio
async def test_release_bucket_promotes_oldest_waiting_first() -> None:
    pool, cursor = _mock_pool(
        fetchone_sequence=[(5000,), (5000,)],  # mutations_this_week, running_sum
        fetchall_sequence=[[("wait1", 500), ("wait2", 3000)]],
    )
    released = await release_bucket_and_start_waiting(pool)
    assert released == ["wait1", "wait2"]


@pytest.mark.asyncio
async def test_release_bucket_skips_overbudget_rows() -> None:
    # 11_000 already spent, free=1000. First waiting fits (500), second (3000) does not.
    pool, cursor = _mock_pool(
        fetchone_sequence=[(11_000,), (11_000,)],
        fetchall_sequence=[[("wait1", 500), ("wait2", 3000)]],
    )
    released = await release_bucket_and_start_waiting(pool)
    assert released == ["wait1"]


@pytest.mark.asyncio
async def test_release_bucket_updates_sda_state_total() -> None:
    pool, cursor = _mock_pool(
        fetchone_sequence=[(5000,), (5000,)],
        fetchall_sequence=[[("wait1", 500)]],
    )
    await release_bucket_and_start_waiting(pool)
    calls = cursor.execute.await_args_list
    upsert = next(c for c in calls if "INSERT INTO sda_state" in c.args[0])
    # Jsonb stores its payload in `.obj`; new total = 5000 + 500 = 5500.
    jsonb_arg = upsert.args[1][0]
    assert jsonb_arg.obj == 5500


@pytest.mark.asyncio
async def test_release_bucket_empty_queue() -> None:
    pool, _ = _mock_pool(
        fetchone_sequence=[(0,), (0,)],
        fetchall_sequence=[[]],
    )
    released = await release_bucket_and_start_waiting(pool)
    assert released == []


# ---- mark_expired ----------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_expired_returns_ids() -> None:
    pool, _ = _mock_pool(fetchall_sequence=[[("h1",), ("h2",)]])
    ids = await mark_expired(pool)
    assert ids == ["h1", "h2"]


@pytest.mark.asyncio
async def test_mark_expired_no_rows() -> None:
    pool, _ = _mock_pool(fetchall_sequence=[[]])
    ids = await mark_expired(pool)
    assert ids == []


# ---- Outcome model --------------------------------------------------------


def test_outcome_model_rejects_unknown_classification() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Outcome(
            hypothesis_id="hyp1",
            classification="unknown",  # type: ignore[arg-type]
            confidence="high",
            delta_pct=0.0,
            lesson="",
        )


def test_outcome_model_extra_forbidden() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Outcome(
            hypothesis_id="hyp1",
            classification="neutral",
            confidence="high",
            delta_pct=0.0,
            lesson="",
            extra_field="boom",  # type: ignore[call-arg]
        )
