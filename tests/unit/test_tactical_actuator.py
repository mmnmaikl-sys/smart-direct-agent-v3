"""Unit tests for tactical_actuator decision rules — pure function, no I/O.

The actuator runs every 4h via cron. Its single job: adjust DailyBudget
on the 5 own campaigns based on lifetime CTR — scale winners, starve
losers, with hard min/max guardrails.

This is the FIRST autonomous actuator in SDA v3 — strict pure-rule
(no LLM) so behaviour is fully testable and reversible.
"""

from __future__ import annotations

from agent_runtime.jobs.tactical_actuator import (
    Decision,
    decide_action,
    plan_total_within_cap,
)


def _camp(cid: int, name: str, daily_rub: int, impr: int, clicks: int) -> dict:
    return {
        "Id": cid,
        "Name": name,
        "DailyBudget": {"Amount": daily_rub * 1_000_000, "Mode": "STANDARD"},
        "Statistics": {"Impressions": impr, "Clicks": clicks},
    }


# --- decide_action ----------------------------------------------------------


def test_scale_winner_high_ctr_low_budget() -> None:
    """CTR > 5% AND DailyBudget < 2000 → scale +500."""
    d = decide_action(_camp(709353034, "pensioner", 1500, 222, 19))  # CTR 8.56%
    assert d.kind == "scale"
    assert d.new_daily_rub == 2000
    assert "CTR" in d.reason


def test_starve_loser_low_ctr_high_budget() -> None:
    """CTR < 2% AND DailyBudget > 200 → starve -200."""
    d = decide_action(_camp(709353005, "rabotyaga", 300, 1074, 17))  # CTR 1.58%
    assert d.kind == "starve"
    assert d.new_daily_rub == 100
    assert "CTR" in d.reason


def test_noop_moderate_ctr() -> None:
    """CTR 2-5% → no action (control band)."""
    d = decide_action(_camp(709353058, "mother", 500, 236, 8))  # CTR 3.39%
    assert d.kind == "noop"


def test_noop_winner_already_at_cap() -> None:
    """CTR > 5% but DailyBudget already at max (2500) → noop."""
    d = decide_action(_camp(709353034, "pensioner", 2500, 222, 19))
    assert d.kind == "noop"


def test_noop_loser_already_at_floor() -> None:
    """CTR < 2% but DailyBudget already at min (100) → noop."""
    d = decide_action(_camp(709353005, "rabotyaga", 100, 1074, 17))
    assert d.kind == "noop"


def test_noop_zero_impressions_insufficient_data() -> None:
    """Zero impressions ever → no signal, no action."""
    d = decide_action(_camp(709353099, "property", 500, 0, 0))
    assert d.kind == "noop"
    assert "insufficient" in d.reason.lower() or "no data" in d.reason.lower()


def test_noop_low_impressions_below_threshold() -> None:
    """Lifetime impressions < 100 → too few for CTR signal."""
    d = decide_action(_camp(709353099, "property", 500, 50, 5))  # CTR 10% but only 50 impr
    assert d.kind == "noop"


def test_scale_does_not_exceed_max_cap() -> None:
    """If current is 1800 and rule wants +500, clamp to 2500 (max)."""
    d = decide_action(_camp(709353034, "pensioner", 1800, 222, 19))
    assert d.kind == "scale"
    assert d.new_daily_rub == 2300  # 1800 + 500, still below 2500
    # Edge: from 2100, +500 would be 2600 → clamp to 2500
    d2 = decide_action(_camp(709353034, "pensioner", 2100, 222, 19))
    assert d2.kind == "scale"
    assert d2.new_daily_rub == 2500


def test_starve_does_not_undershoot_min() -> None:
    """If current is 200 and rule wants -200, clamp to 100 (min)."""
    d = decide_action(_camp(709353005, "rabotyaga", 200, 1074, 17))
    assert d.kind == "starve"
    assert d.new_daily_rub == 100


# --- plan_total_within_cap --------------------------------------------------


def test_plan_within_total_cap_passes() -> None:
    decisions = [
        Decision(
            cid=1,
            name="a",
            kind="scale",
            current_daily_rub=500,
            new_daily_rub=1000,
            reason="",
        ),
        Decision(
            cid=2,
            name="b",
            kind="scale",
            current_daily_rub=500,
            new_daily_rub=1500,
            reason="",
        ),
    ]
    others_current_total = 500 * 3  # 3 control campaigns at 500 each
    ok, total = plan_total_within_cap(decisions, others_current_total, cap_rub=5000)
    assert ok is True
    assert total == 1000 + 1500 + 1500


def test_plan_exceeds_total_cap_fails() -> None:
    decisions = [
        Decision(
            cid=1,
            name="a",
            kind="scale",
            current_daily_rub=500,
            new_daily_rub=2500,
            reason="",
        ),
        Decision(
            cid=2,
            name="b",
            kind="scale",
            current_daily_rub=500,
            new_daily_rub=2500,
            reason="",
        ),
    ]
    others_current_total = 500 * 3
    ok, total = plan_total_within_cap(decisions, others_current_total, cap_rub=5000)
    assert ok is False
    assert total == 2500 + 2500 + 1500
