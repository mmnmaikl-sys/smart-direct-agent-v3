"""Unit tests for agent_runtime.jobs.regression_watch (Task 27).

Coverage targets (Decision 6):

* Drawdown math — higher_is_better vs lower_is_better (CPA inversion).
* Threshold tiers (budget_cap → warn/hard pair) boundaries.
* ``ok`` verdict → no side effects.
* ``warn`` verdict → Telegram warning, no rollback, no reflection.
* ``hard`` verdict → impact_tracker.rollback + reflection.save + critical
  Telegram.
* ``dry_run=True`` suppresses rollback, reflection, Telegram.
* Trust-level shadow → hard downgraded to critical-notify (no DB mutation).
* Degraded no-op when the Direct client is missing.
* Idempotency — a hypothesis already ``rolled_back`` does not appear in
  the SELECT window (verified by feeding only ``state='confirmed'`` rows
  through the fake pool).
* Rollback failure still emits reflection + alert.
* Empty baseline → skip_no_baseline.
* Low-signal (<50 clicks) → skip_low_signal.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.jobs import regression_watch
from agent_runtime.trust_levels import TrustLevel

# --------------------------------------------------------------- fake pool


class _FakePool:
    """Async-context-manager mimic that returns a canned fetchall list.

    The job issues exactly one SELECT (``_select_confirmed_in_window``) and
    no writes — the fake does not need to model ``UPDATE`` or ``audit_log``
    because ``impact_tracker.rollback`` itself is mocked.
    """

    def __init__(self, rows: list[tuple[Any, ...]]):
        self._rows = rows
        self.executed: list[str] = []

    def connection(self) -> _FakeConn:
        return _FakeConn(self)


class _FakeConn:
    def __init__(self, pool: _FakePool):
        self.pool = pool

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.pool)


class _FakeCursor:
    def __init__(self, pool: _FakePool):
        self.pool = pool

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self.pool.executed.append(sql)

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.pool._rows


# --------------------------------------------------------------- helpers


def _row(
    *,
    id_: str = "h1",
    agent: str = "brain",
    hypothesis_type: str = "ad",
    budget_cap_rub: int = 500,
    campaign_id: int | None = 708978456,
    ad_group_id: int | None = 42,
    actions: list[dict[str, Any]] | None = None,
    baseline: dict[str, Any] | None = None,
    promoted_at: datetime | None = None,
) -> tuple[Any, ...]:
    if baseline is None:
        baseline = {"ctr": 3.0, "cpa": 1000.0, "leads": 10, "clicks": 500}
    if actions is None:
        actions = [{"type": "pause_group", "params": {"ad_group_id": 42}}]
    if promoted_at is None:
        promoted_at = datetime(2026, 4, 17, 12, 0)
    return (
        id_,
        agent,
        hypothesis_type,
        budget_cap_rub,
        campaign_id,
        ad_group_id,
        actions,
        baseline,
        promoted_at,
    )


def _direct_with_current(current: dict[str, Any]) -> AsyncMock:
    direct = AsyncMock()
    direct.get_campaign_stats = AsyncMock(return_value=current)
    return direct


def _telegram() -> AsyncMock:
    tg = AsyncMock()
    tg.send_message = AsyncMock(return_value=None)
    return tg


def _reflection_store() -> AsyncMock:
    rs = AsyncMock()
    rs.save = AsyncMock(return_value=None)
    return rs


@pytest.fixture
def autonomous_trust():
    with patch(
        "agent_runtime.jobs.regression_watch.get_trust_level",
        new=AsyncMock(return_value=TrustLevel.AUTONOMOUS),
    ):
        yield


@pytest.fixture
def shadow_trust():
    with patch(
        "agent_runtime.jobs.regression_watch.get_trust_level",
        new=AsyncMock(return_value=TrustLevel.SHADOW),
    ):
        yield


_NOW = datetime(2026, 4, 24, 12, 0)


# --------------------------------------------------------------- pure helpers


def test_thresholds_for_budget_cap_500() -> None:
    assert regression_watch._regression_thresholds_for(500) == (25, 40)


def test_thresholds_for_budget_cap_501_boundary() -> None:
    assert regression_watch._regression_thresholds_for(501) == (20, 30)


def test_thresholds_for_budget_cap_2500_upper_boundary() -> None:
    assert regression_watch._regression_thresholds_for(2500) == (20, 30)


def test_thresholds_for_budget_cap_2501_boundary() -> None:
    assert regression_watch._regression_thresholds_for(2501) == (15, 25)


def test_thresholds_for_budget_cap_5000_upper_boundary() -> None:
    assert regression_watch._regression_thresholds_for(5000) == (15, 25)


def test_thresholds_for_budget_cap_5001_top_tier() -> None:
    assert regression_watch._regression_thresholds_for(5001) == (10, 20)


def test_thresholds_for_budget_cap_large_value_top_tier() -> None:
    assert regression_watch._regression_thresholds_for(100_000) == (10, 20)


def test_compute_drawdown_higher_is_better_drop() -> None:
    # CTR 3.0 → 2.25 → 25% drawdown.
    assert regression_watch._compute_drawdown_pct(
        baseline_value=3.0, current_value=2.25, direction="higher_is_better"
    ) == pytest.approx(25.0)


def test_compute_drawdown_higher_is_better_improvement() -> None:
    # CTR 3.0 → 3.3 → -10% drawdown (i.e. improvement).
    assert regression_watch._compute_drawdown_pct(
        baseline_value=3.0, current_value=3.3, direction="higher_is_better"
    ) == pytest.approx(-10.0)


def test_compute_drawdown_lower_is_better_cpa_growth() -> None:
    # CPA 1000 → 1400 → +40% drawdown (bad).
    assert regression_watch._compute_drawdown_pct(
        baseline_value=1000.0, current_value=1400.0, direction="lower_is_better"
    ) == pytest.approx(40.0)


def test_compute_drawdown_zero_baseline_returns_zero() -> None:
    assert (
        regression_watch._compute_drawdown_pct(
            baseline_value=0.0, current_value=5.0, direction="higher_is_better"
        )
        == 0.0
    )


# --------------------------------------------------------------- run() scenarios


@pytest.mark.asyncio
async def test_no_regression_no_op(autonomous_trust) -> None:
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500})])
    direct = _direct_with_current({"ctr": 3.1, "clicks": 600})
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    assert result["status"] == "ok"
    assert result["checked"] == 1
    assert result["warnings"] == []
    assert result["rollbacks"] == []
    assert mock_rollback.await_count == 0
    assert tg.send_message.await_count == 0
    assert rs.save.await_count == 0


@pytest.mark.asyncio
async def test_warning_threshold_notify_only(autonomous_trust) -> None:
    # CTR 3.0 → 2.25 → 25% drawdown. budget_cap=500 → warn=25%, hard=40%.
    # 25% ≥ warn (25%) and < hard (40%) → warn path.
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500})])
    direct = _direct_with_current({"ctr": 2.25, "clicks": 600})
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    assert result["warnings"] == ["h1"]
    assert result["rollbacks"] == []
    assert mock_rollback.await_count == 0
    # Telegram warning — one call, priority 'warning'.
    assert tg.send_message.await_count == 1
    kwargs = tg.send_message.await_args.kwargs
    assert kwargs["priority"] == "warning"
    text_lc = kwargs["text"].lower()
    assert "warning" in text_lc or "регрессия" in text_lc or "drawdown" in text_lc
    # No reflection on warn.
    assert rs.save.await_count == 0


@pytest.mark.asyncio
async def test_hard_threshold_triggers_rollback(autonomous_trust) -> None:
    # CTR 3.0 → 1.7 → ~43% drawdown; budget_cap=500 → hard=40%.
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500})])
    direct = _direct_with_current({"ctr": 1.7, "clicks": 600})
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(return_value=None),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    assert result["warnings"] == []
    assert len(result["rollbacks"]) == 1
    report = result["rollbacks"][0]
    assert report["hypothesis_id"] == "h1"
    assert report["rollback_ok"] is True
    assert report["reflection_ok"] is True
    assert report["alert_ok"] is True

    # rollback was called with the positional (pool, hypothesis_id, direct=)
    mock_rollback.assert_awaited_once()
    call = mock_rollback.await_args
    assert call is not None
    assert call.args[0] is pool
    assert call.args[1] == "h1"
    assert call.kwargs["direct"] is direct

    # Reflection payload has the right metadata.
    rs.save.assert_awaited_once()
    reflection_kwargs = rs.save.await_args.kwargs
    md = reflection_kwargs["metadata"]
    assert md["hypothesis_id"] == "h1"
    assert md["outcome"] == "regressed"
    assert md["agent"] == "brain"
    assert md["budget_cap_rub"] == 500
    assert md["trigger"] == "regression_watch"
    assert md["rollback_status"] == "ok"
    assert md["drawdown_pct"] > 40.0

    # Critical Telegram alert.
    tg.send_message.assert_awaited_once()
    assert tg.send_message.await_args.kwargs["priority"] == "critical"


@pytest.mark.asyncio
async def test_dry_run_does_not_mutate(autonomous_trust) -> None:
    # Hard drawdown but dry_run=True → no rollback, no reflection, no Telegram.
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500})])
    direct = _direct_with_current({"ctr": 1.5, "clicks": 600})  # -50% drop, very hard
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            dry_run=True,
            now=_NOW,
        )

    assert result["dry_run"] is True
    # Hard without mutate_allowed → hypothesis listed in warnings (surfaced
    # but no side effects fire).
    assert result["warnings"] == ["h1"]
    assert result["rollbacks"] == []
    assert mock_rollback.await_count == 0
    assert tg.send_message.await_count == 0
    assert rs.save.await_count == 0


@pytest.mark.asyncio
async def test_shadow_trust_hard_notifies_only(shadow_trust) -> None:
    # Hard drawdown, trust=SHADOW → no rollback, no reflection; Telegram
    # still fires CRITICAL so owner sees it.
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500})])
    direct = _direct_with_current({"ctr": 1.5, "clicks": 600})
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    assert result["trust_level"] == "shadow"
    assert result["rollbacks"] == []
    assert result["warnings"] == ["h1"]
    mock_rollback.assert_not_awaited()
    rs.save.assert_not_awaited()
    tg.send_message.assert_awaited_once()
    assert tg.send_message.await_args.kwargs["priority"] == "critical"


@pytest.mark.asyncio
async def test_budget_cap_calibration_tiers(autonomous_trust) -> None:
    """4 parametrised hypotheses — each just above its own warn threshold.

    Budget 500 (warn 25%) → dd 26% → warn.
    Budget 2000 (warn 20%) → dd 21% → warn.
    Budget 4000 (warn 15%) → dd 16% → warn.
    Budget 7000 (warn 10%) → dd 11% → warn.

    All four must land in ``warnings`` — proves thresholds_for() is
    consulted per-row, not a global default.
    """
    # baseline ctr=100 makes drawdown math trivial: current=100-dd gives dd%.
    rows = [
        _row(id_="h500", budget_cap_rub=500, baseline={"ctr": 100.0, "clicks": 500}),
        _row(id_="h2000", budget_cap_rub=2000, baseline={"ctr": 100.0, "clicks": 500}),
        _row(id_="h4000", budget_cap_rub=4000, baseline={"ctr": 100.0, "clicks": 500}),
        _row(id_="h7000", budget_cap_rub=7000, baseline={"ctr": 100.0, "clicks": 500}),
    ]
    pool = _FakePool(rows)

    current_by_id = {
        "h500": {"ctr": 74.0, "clicks": 600},  # dd = 26%
        "h2000": {"ctr": 79.0, "clicks": 600},  # dd = 21%
        "h4000": {"ctr": 84.0, "clicks": 600},  # dd = 16%
        "h7000": {"ctr": 89.0, "clicks": 600},  # dd = 11%
    }

    # Direct is keyed by (cid, date_from, date_to). We look the id up via
    # order-of-call tracking since every row shares the same campaign_id.
    call_order: list[str] = ["h500", "h2000", "h4000", "h7000"]
    call_counter = {"i": 0}

    async def _stats(_cid: int, **_kw: Any) -> dict[str, Any]:
        idx = call_counter["i"]
        call_counter["i"] += 1
        return current_by_id[call_order[idx]]

    direct = AsyncMock()
    direct.get_campaign_stats.side_effect = _stats
    tg = _telegram()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            now=_NOW,
        )

    assert set(result["warnings"]) == {"h500", "h2000", "h4000", "h7000"}
    assert result["rollbacks"] == []
    mock_rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_cpa_inversion_triggers_rollback(autonomous_trust) -> None:
    # hypothesis_type=neg_kw → target=cpa, direction=lower_is_better.
    # baseline cpa=1000, current cpa=1400 → +40% drawdown.
    # budget_cap=300 (≤500 tier) → hard=40% → rollback.
    pool = _FakePool(
        [
            _row(
                id_="neg1",
                hypothesis_type="neg_kw",
                budget_cap_rub=300,
                baseline={"cpa": 1000.0, "clicks": 500},
                actions=[
                    {
                        "type": "add_negatives",
                        "params": {"campaign_id": 100, "phrases": ["x"]},
                    }
                ],
            )
        ]
    )
    direct = _direct_with_current({"cpa": 1400.0, "clicks": 600})
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(return_value=None),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    assert len(result["rollbacks"]) == 1
    mock_rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_idempotency_rolled_back_absent_from_select(autonomous_trust) -> None:
    """A ``state='rolled_back'`` hypothesis must not enter the job.

    The SELECT filters ``state='confirmed'``; we model that by feeding an
    empty fetchall. Second-run safety is therefore a consequence of the
    SQL, not of imperative bookkeeping in this module.
    """
    pool = _FakePool([])  # no confirmed rows
    direct = _direct_with_current({"ctr": 1.0})
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    assert result["checked"] == 0
    assert result["warnings"] == []
    assert result["rollbacks"] == []
    mock_rollback.assert_not_awaited()
    rs.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_degraded_noop_when_direct_missing(autonomous_trust) -> None:
    pool = _FakePool([_row()])

    result = await regression_watch.run(pool, now=_NOW)  # type: ignore[arg-type]

    assert result["status"] == "ok"
    assert result["action"] == "degraded_noop"
    assert result["checked"] == 0
    assert result["rollbacks"] == []


@pytest.mark.asyncio
async def test_skip_no_baseline(autonomous_trust) -> None:
    pool = _FakePool([_row(baseline={})])  # pre-Task-27 hypothesis
    direct = _direct_with_current({"ctr": 0.5})
    tg = _telegram()

    result = await regression_watch.run(pool, direct=direct, telegram=tg, now=_NOW)  # type: ignore[arg-type]

    assert result["warnings"] == []
    assert result["rollbacks"] == []
    assert any(s["reason"] == "no_baseline" for s in result["skipped"])
    tg.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_skip_low_signal_under_50_clicks(autonomous_trust) -> None:
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500})])
    # Huge drawdown BUT clicks = 10 → below signal floor → skip.
    direct = _direct_with_current({"ctr": 0.5, "clicks": 10})
    tg = _telegram()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(),
    ) as mock_rollback:
        result = await regression_watch.run(pool, direct=direct, telegram=tg, now=_NOW)  # type: ignore[arg-type]

    assert result["rollbacks"] == []
    assert result["warnings"] == []
    assert any(s["reason"] == "low_signal" for s in result["skipped"])
    mock_rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollback_failure_still_emits_reflection_and_alert(autonomous_trust) -> None:
    """impact_tracker.rollback raises → reflection + rollback_failed alert."""
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500})])
    direct = _direct_with_current({"ctr": 1.0, "clicks": 600})  # ~66% drop → hard
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(side_effect=RuntimeError("direct 500")),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    assert len(result["rollbacks"]) == 1
    report = result["rollbacks"][0]
    assert report["rollback_ok"] is False
    assert "direct 500" in (report["rollback_error"] or "")
    assert report["reflection_ok"] is True
    assert report["alert_ok"] is True

    mock_rollback.assert_awaited_once()
    # Reflection metadata carries rollback_status=failed.
    md = rs.save.await_args.kwargs["metadata"]
    assert md["rollback_status"] == "failed"
    # Critical Telegram fired with the "FAILED" variant.
    tg.send_message.assert_awaited_once()
    txt = tg.send_message.await_args.kwargs["text"]
    assert "FAILED" in txt or "failed" in txt.lower()


@pytest.mark.asyncio
async def test_bitrix_lead_count_enters_current_metrics(autonomous_trust) -> None:
    # hypothesis_type=new_camp → target=leads. baseline leads=10,
    # bitrix returns 6 → drawdown = 40% → budget_cap=3500 (tier 2501-5000
    # → warn=15, hard=25) → hard → rollback.
    pool = _FakePool(
        [
            _row(
                id_="c1",
                hypothesis_type="new_camp",
                budget_cap_rub=3500,
                baseline={"leads": 10, "clicks": 500},
            )
        ]
    )
    direct = _direct_with_current({"clicks": 600})  # no 'leads' from Direct
    bitrix = AsyncMock()
    bitrix.get_leads_count_by_utm = AsyncMock(return_value=6)
    tg = _telegram()
    rs = _reflection_store()

    with patch(
        "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
        new=AsyncMock(return_value=None),
    ) as mock_rollback:
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            bitrix=bitrix,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    bitrix.get_leads_count_by_utm.assert_awaited_once()
    assert len(result["rollbacks"]) == 1
    mock_rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_error_does_not_sink_loop(autonomous_trust) -> None:
    """First row crashes mid-check; second row still gets processed."""
    pool = _FakePool(
        [
            _row(id_="bad"),
            _row(id_="good", baseline={"ctr": 3.0, "clicks": 500}),
        ]
    )
    # get_campaign_stats raises on the first call, returns good data on the
    # second — matches order-of-iteration in _select_confirmed_in_window.
    seq = [RuntimeError("boom"), {"ctr": 3.1, "clicks": 600}]

    async def _side(_cid: int, **_kw: Any) -> dict[str, Any]:
        val = seq.pop(0)
        if isinstance(val, Exception):
            raise val
        return val

    direct = AsyncMock()
    direct.get_campaign_stats.side_effect = _side
    tg = _telegram()

    result = await regression_watch.run(pool, direct=direct, telegram=tg, now=_NOW)  # type: ignore[arg-type]

    # Both rows reach the classifier: the raising one degrades to
    # empty-current (→ skip_low_signal or ok depending on branch) and the
    # healthy one ends up ok. Either way the loop must not raise.
    assert result["status"] == "ok"
    assert result["checked"] == 2
    # No rollback in either branch.
    assert result["rollbacks"] == []


@pytest.mark.asyncio
async def test_trust_lookup_failure_defaults_shadow() -> None:
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500})])
    direct = _direct_with_current({"ctr": 1.0, "clicks": 600})
    tg = _telegram()
    rs = _reflection_store()

    with (
        patch(
            "agent_runtime.jobs.regression_watch.get_trust_level",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ),
        patch(
            "agent_runtime.jobs.regression_watch.impact_tracker.rollback",
            new=AsyncMock(),
        ) as mock_rollback,
    ):
        result = await regression_watch.run(
            pool,  # type: ignore[arg-type]
            direct=direct,
            telegram=tg,
            reflection_store=rs,
            now=_NOW,
        )

    # Fallback trust = shadow → hard verdict downgraded to notify.
    assert result["trust_level"] == "shadow"
    assert result["rollbacks"] == []
    mock_rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_promoted_at_window_is_passed_through_as_date(autonomous_trust) -> None:
    """get_campaign_stats receives promoted_at date, not 'now' minus 30d."""
    promoted = datetime(2026, 4, 20, 14, 0)
    pool = _FakePool([_row(baseline={"ctr": 3.0, "clicks": 500}, promoted_at=promoted)])
    direct = _direct_with_current({"ctr": 3.05, "clicks": 600})
    tg = _telegram()

    await regression_watch.run(pool, direct=direct, telegram=tg, now=_NOW)  # type: ignore[arg-type]

    assert direct.get_campaign_stats.await_count == 1
    args, kwargs = direct.get_campaign_stats.await_args
    # (campaign_id,) positional, date_from/date_to keyword
    assert args == (708978456,)
    assert kwargs["date_from"] == "2026-04-20"
    assert kwargs["date_to"] == "2026-04-24"


# --------------------------------------------------------------- parametric


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget_cap", "dd_pct", "expected_verdict"),
    [
        (500, 24.9, "ok"),
        (500, 25.0, "warn"),
        (500, 39.9, "warn"),
        (500, 40.0, "hard"),
        (2000, 19.9, "ok"),
        (2000, 20.0, "warn"),
        (2000, 29.9, "warn"),
        (2000, 30.0, "hard"),
        (4000, 14.9, "ok"),
        (4000, 15.0, "warn"),
        (4000, 25.0, "hard"),
        (7000, 9.9, "ok"),
        (7000, 10.0, "warn"),
        (7000, 20.0, "hard"),
    ],
)
async def test_verdict_boundaries_per_budget_tier(
    autonomous_trust, budget_cap: int, dd_pct: float, expected_verdict: str
) -> None:
    """Verdict boundaries per tier — parametrised probe."""
    baseline_ctr = 100.0
    current_ctr = baseline_ctr * (1 - dd_pct / 100.0)

    row = regression_watch._ConfirmedRow(
        id="probe",
        agent="brain",
        hypothesis_type="ad",
        budget_cap_rub=budget_cap,
        campaign_id=1,
        ad_group_id=2,
        actions=[],
        baseline_at_promote={"ctr": baseline_ctr, "clicks": 500},
        promoted_at=_NOW - timedelta(days=2),
    )
    direct = _direct_with_current({"ctr": current_ctr, "clicks": 500})
    result = await regression_watch._check_one(row, direct=direct, bitrix=None, now=_NOW)
    assert result.verdict == expected_verdict
