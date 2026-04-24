"""Unit tests for the impact_tracker_job orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.decision_journal import HypothesisRow
from agent_runtime.impact_tracker import Outcome
from agent_runtime.jobs.impact_tracker_job import run


def _row(id_: str = "hyp1") -> HypothesisRow:
    return HypothesisRow(
        id=id_,
        hypothesis_type="ad",
        agent="brain",
        campaign_id=708978456,
        ad_group_id=42,
        state="running",
        actions=[{"type": "pause_group", "params": {"ad_group_id": 42}}],
        metrics_before={"ctr": 2.0},
        budget_cap_rub=500,
        created_at=datetime.now(UTC) - timedelta(hours=80),
        maximum_running_days=14,
    )


def _pool() -> MagicMock:
    return MagicMock()


@pytest.mark.asyncio
async def test_run_dry_run_skips_measurement() -> None:
    with (
        patch(
            "agent_runtime.jobs.impact_tracker_job.decision_journal.get_pending_checks",
            new=AsyncMock(return_value=[_row(), _row("hyp2")]),
        ),
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.mark_expired",
            new=AsyncMock(return_value=[]),
        ) as mock_expire,
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.release_bucket_and_start_waiting",
            new=AsyncMock(return_value=[]),
        ) as mock_release,
    ):
        result = await run(_pool(), dry_run=True)

    assert result["dry_run"] is True
    assert result["pending_measured"] == 2
    assert all(o.get("skipped") is True for o in result["outcomes"])
    # dry_run → no expire / release pass
    mock_expire.assert_not_awaited()
    mock_release.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_measures_when_direct_client_provided() -> None:
    outcome = Outcome(
        hypothesis_id="hyp1",
        classification="positive",
        confidence="high",
        delta_pct=0.25,
        metrics_after={"ctr": 2.5},
        lesson="ad → positive",
    )
    direct = SimpleNamespace()
    with (
        patch(
            "agent_runtime.jobs.impact_tracker_job.decision_journal.get_pending_checks",
            new=AsyncMock(return_value=[_row()]),
        ),
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.measure_outcome",
            new=AsyncMock(return_value=outcome),
        ) as mock_measure,
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.mark_expired",
            new=AsyncMock(return_value=["expired1"]),
        ),
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.release_bucket_and_start_waiting",
            new=AsyncMock(return_value=["released1", "released2"]),
        ),
    ):
        result = await run(_pool(), direct=direct)

    mock_measure.assert_awaited_once()
    assert result["pending_measured"] == 1
    assert result["expired"] == ["expired1"]
    assert result["released"] == ["released1", "released2"]
    assert result["outcomes"][0]["classification"] == "positive"


@pytest.mark.asyncio
async def test_run_skips_without_direct_client() -> None:
    with (
        patch(
            "agent_runtime.jobs.impact_tracker_job.decision_journal.get_pending_checks",
            new=AsyncMock(return_value=[_row()]),
        ),
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.measure_outcome",
            new=AsyncMock(),
        ) as mock_measure,
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.mark_expired",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.release_bucket_and_start_waiting",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await run(_pool(), direct=None)

    mock_measure.assert_not_awaited()
    assert result["outcomes"][0]["skipped"] is True
    assert result["outcomes"][0]["reason"] == "no_direct_client"


@pytest.mark.asyncio
async def test_run_measure_exception_does_not_abort_cycle() -> None:
    direct = SimpleNamespace()
    with (
        patch(
            "agent_runtime.jobs.impact_tracker_job.decision_journal.get_pending_checks",
            new=AsyncMock(return_value=[_row("hyp1"), _row("hyp2")]),
        ),
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.measure_outcome",
            new=AsyncMock(
                side_effect=[
                    RuntimeError("boom"),
                    Outcome(
                        hypothesis_id="hyp2",
                        classification="neutral",
                        confidence="low",
                        delta_pct=0.0,
                        metrics_after={},
                        lesson="insufficient signal",
                    ),
                ]
            ),
        ),
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.mark_expired",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "agent_runtime.jobs.impact_tracker_job.impact_tracker.release_bucket_and_start_waiting",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await run(_pool(), direct=direct)

    assert result["pending_measured"] == 2
    # First row → error record; second row → real outcome
    assert result["outcomes"][0]["error"] is True
    assert result["outcomes"][1]["classification"] == "neutral"


@pytest.mark.asyncio
async def test_run_registered_in_job_registry() -> None:
    from agent_runtime.jobs import JOB_REGISTRY

    assert "impact_tracker" in JOB_REGISTRY
