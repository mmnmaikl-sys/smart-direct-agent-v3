"""Unit tests for agent_runtime.decision_engine (Task 10)."""

from __future__ import annotations

import pytest

from agent_runtime.decision_engine import (
    ACTION_LIMITS,
    ENTITY_COOLDOWN_HOURS,
    FORBIDDEN_ACTIONS,
    IRREVERSIBILITY,
    MAX_MUTATIONS_PER_WEEK,
    Decision,
    _apply_time_decay,
    calculate_risk,
    check_cooldown,
    check_daily_limit,
    check_weekly_mutations,
    evaluate,
)
from agent_runtime.models import AutonomyLevel

# ---- forbidden ------------------------------------------------------------


def test_forbidden_action_returns_forbidden() -> None:
    d = evaluate("enable_autotargeting", affected_budget_pct=0.0, data_points=100)
    assert d.level == AutonomyLevel.FORBIDDEN
    assert d.risk_score == 100
    assert d.can_execute is False


@pytest.mark.parametrize("action", sorted(FORBIDDEN_ACTIONS))
def test_forbidden_actions_complete(action: str) -> None:
    d = evaluate(action, affected_budget_pct=0.01, data_points=1000)
    assert d.level == AutonomyLevel.FORBIDDEN


# ---- risk band thresholds -------------------------------------------------


def test_low_risk_returns_auto() -> None:
    # add_negative: irreversibility 10; lots of data → uncertainty 10
    d = evaluate("add_negative", affected_budget_pct=0.01, data_points=50)
    assert d.risk_score <= 25
    assert d.level == AutonomyLevel.AUTO
    assert d.can_execute is True


def test_medium_risk_returns_notify() -> None:
    # resume_campaign: irreversibility 40; uncertainty 50 (medium data)
    d = evaluate("resume_campaign", affected_budget_pct=0.05, data_points=15)
    assert 25 < d.risk_score <= 50
    assert d.level == AutonomyLevel.NOTIFY


def test_high_risk_returns_ask() -> None:
    # switch_strategy: irreversibility 70; low data → uncertainty 90
    d = evaluate("switch_strategy", affected_budget_pct=0.3, data_points=5)
    assert d.risk_score > 50
    assert d.level == AutonomyLevel.ASK


def test_risk_formula_weights() -> None:
    # Pick inputs that land on money=50, irrev=60, unc=90
    # money: affected_budget_pct * 100 clamped at 100 → 0.5 → 50
    # irreversibility: change_ad_text → 45; need 60 → use pause_campaign (60)
    # uncertainty: low data → 90
    risk, _ = calculate_risk(
        "pause_campaign",
        affected_budget_pct=0.5,
        data_points=1,
        min_confident_samples=30,
    )
    expected = 50 * 0.3 + 60 * 0.4 + 90 * 0.3
    assert abs(risk - expected) < 0.1


def test_irreversibility_mapping_complete() -> None:
    for action, score in IRREVERSIBILITY.items():
        assert 0 <= score <= 100, f"{action}: {score}"


def test_unknown_action_uses_default_irreversibility() -> None:
    risk_unknown, _ = calculate_risk("some_unknown_action", 0.1, 30)
    # default irrev=50 → irrev_term = 50*0.4 = 20
    # money 10*0.3 = 3; uncertainty 10*0.3 = 3; total 26
    assert 25 < risk_unknown < 27


def test_decision_dataclass_fields() -> None:
    d = evaluate("add_negative", affected_budget_pct=0.1, data_points=30)
    assert d.affected_budget_pct == 0.1
    assert d.data_points_effective == 30.0


# ---- parameterization (v2 regression) -------------------------------------


def test_affected_budget_pct_parameterized() -> None:
    d1 = evaluate("add_negative", affected_budget_pct=0.1, data_points=50)
    d2 = evaluate("add_negative", affected_budget_pct=0.5, data_points=50)
    assert d1.risk_score < d2.risk_score


def test_min_confident_samples_parameterized() -> None:
    risk_low, _ = calculate_risk("add_negative", 0.0, data_points=10, min_confident_samples=10)
    risk_high, _ = calculate_risk("add_negative", 0.0, data_points=10, min_confident_samples=100)
    # Same data, higher threshold → more uncertainty → higher risk
    assert risk_low < risk_high


# ---- time decay -----------------------------------------------------------


def test_time_decay_fresh_full_weight() -> None:
    assert _apply_time_decay([1, 2, 3]) == 3.0
    assert _apply_time_decay([7, 7]) == 2.0


def test_time_decay_old_reduced_weight() -> None:
    # Three 30-day points should sum to ~0.9
    assert _apply_time_decay([30, 30, 30]) == pytest.approx(0.9, abs=0.01)
    # And beyond 30d stays at plateau
    assert _apply_time_decay([45, 60]) == pytest.approx(0.6, abs=0.01)


def test_time_decay_linear_interpolation() -> None:
    # At 18.5 days → weight ~ 1.0 - (11.5/23)*0.7 = 0.65
    assert _apply_time_decay([18.5]) == pytest.approx(0.65, abs=0.02)


def test_time_decay_empty_list() -> None:
    assert _apply_time_decay([]) == 0.0
    risk, effective = calculate_risk("add_negative", 0.0, data_points_ages_days=[], data_points=0)
    assert effective == 0.0
    # zero effective → uncertainty 90


def test_time_decay_negative_age_clamped() -> None:
    # age -5 clamped to 0 → fresh weight 1.0
    assert _apply_time_decay([-5, -1]) == 2.0


def test_time_decay_overrides_raw_count() -> None:
    # 30 raw data_points but all 40 days old → effective ~9
    risk, effective = calculate_risk(
        "add_negative",
        affected_budget_pct=0.0,
        data_points=30,
        min_confident_samples=30,
        data_points_ages_days=[40.0] * 30,
    )
    assert effective == pytest.approx(9.0, abs=0.1)
    # With effective 9 < 30 but > 15 → uncertainty=50 (partial)


# ---- helpers --------------------------------------------------------------


def test_check_daily_limit_default() -> None:
    # Unknown action → default 10
    assert check_daily_limit("some_other", today_count=9) is True
    assert check_daily_limit("some_other", today_count=10) is False


def test_check_daily_limit_per_action() -> None:
    assert ACTION_LIMITS["add_negative"] == 20
    assert check_daily_limit("add_negative", today_count=19) is True
    assert check_daily_limit("add_negative", today_count=20) is False


def test_check_weekly_mutations() -> None:
    assert check_weekly_mutations(MAX_MUTATIONS_PER_WEEK - 1) is True
    assert check_weekly_mutations(MAX_MUTATIONS_PER_WEEK) is False


def test_check_cooldown() -> None:
    assert check_cooldown(ENTITY_COOLDOWN_HOURS - 1) is False
    assert check_cooldown(ENTITY_COOLDOWN_HOURS) is True


def test_evaluate_returns_decision_instance() -> None:
    d = evaluate("add_negative", 0.0, 30)
    assert isinstance(d, Decision)
