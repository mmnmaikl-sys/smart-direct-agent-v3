"""Pydantic models contract tests — mirrors SQL CHECK constraints and Decision 4/5."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_runtime.models import (
    HYPOTHESIS_BUDGET_CAP,
    AutonomyLevel,
    HypothesisDraft,
    HypothesisOutcome,
    HypothesisState,
    HypothesisType,
    Signal,
    SignalType,
)


def test_HypothesisDraft_valid() -> None:
    draft = HypothesisDraft(
        hypothesis_type=HypothesisType.AD,
        hypothesis="Replace headline with pain-point variant",
        reasoning="CTR dropped 40% over 7d on current headline",
        actions=[{"tool": "direct.ads.add", "params": {"title": "x"}}],
        expected_outcome="CTR > 0.8% over 100+ impressions",
        ad_group_id=123456,
    )
    assert draft.hypothesis_type is HypothesisType.AD
    assert draft.ad_group_id == 123456
    assert draft.campaign_id is None


def test_HypothesisDraft_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HypothesisDraft(
            hypothesis_type=HypothesisType.AD,
            hypothesis="x",
            reasoning="y",
            actions=[{"a": 1}],
            expected_outcome="z",
            ad_group_id=1,
            unknown_field="boom",  # type: ignore[call-arg]
        )


def test_attribution_single_requires_attribution_for_non_account_level() -> None:
    with pytest.raises(ValidationError) as exc:
        HypothesisDraft(
            hypothesis_type=HypothesisType.AD,
            hypothesis="x",
            reasoning="y",
            actions=[{"a": 1}],
            expected_outcome="z",
        )
    assert "attribution_single" in str(exc.value)


def test_attribution_single_allows_account_level_without_ids() -> None:
    draft = HypothesisDraft(
        hypothesis_type=HypothesisType.ACCOUNT_LEVEL,
        hypothesis="Raise weekly hypothesis cap",
        reasoning="4 confirmed wins last 14d; headroom available",
        actions=[{"tool": "sda_state.set", "params": {"key": "weekly_cap", "value": 15000}}],
        expected_outcome="No regression trigger within 14d",
    )
    assert draft.ad_group_id is None
    assert draft.campaign_id is None


def test_attribution_single_accepts_campaign_id_without_ad_group() -> None:
    draft = HypothesisDraft(
        hypothesis_type=HypothesisType.NEW_CAMP,
        hypothesis="Launch РСЯ retargeting variant",
        reasoning="MK-test 709307228 shows ≥3 leads/week",
        actions=[{"tool": "direct.campaigns.add", "params": {}}],
        expected_outcome="CPA < 1500 within 21d",
        campaign_id=709307228,
    )
    assert draft.campaign_id == 709307228


def test_hypothesis_budget_cap_complete_and_matches_decision_4() -> None:
    assert set(HYPOTHESIS_BUDGET_CAP.keys()) == set(HypothesisType)
    assert HYPOTHESIS_BUDGET_CAP[HypothesisType.AD] == 500
    assert HYPOTHESIS_BUDGET_CAP[HypothesisType.NEG_KW] == 300
    assert HYPOTHESIS_BUDGET_CAP[HypothesisType.IMAGE] == 500
    assert HYPOTHESIS_BUDGET_CAP[HypothesisType.LANDING] == 2500
    assert HYPOTHESIS_BUDGET_CAP[HypothesisType.NEW_CAMP] == 3500
    assert HYPOTHESIS_BUDGET_CAP[HypothesisType.FORMAT_CHANGE] == 7000
    assert HYPOTHESIS_BUDGET_CAP[HypothesisType.STRATEGY_SWITCH] == 5000
    assert HYPOTHESIS_BUDGET_CAP[HypothesisType.ACCOUNT_LEVEL] == 500


def test_signal_type_values() -> None:
    expected = {
        "budget_threshold",
        "spend_no_clicks",
        "high_bounce",
        "zero_leads",
        "high_cpa",
        "landing_broken",
        "landing_slow",
        "garbage_queries",
        "campaign_stopped",
        "api_error",
    }
    assert {s.value for s in SignalType} == expected
    assert SignalType.BUDGET_THRESHOLD == "budget_threshold"


def test_signal_model_accepts_valid_severity() -> None:
    sig = Signal(
        type=SignalType.HIGH_BOUNCE,
        severity="warning",
        data={"campaign_id": 123, "bounce_rate": 0.72},
        ts=datetime.now(UTC),
    )
    assert sig.severity == "warning"


def test_signal_rejects_invalid_severity() -> None:
    with pytest.raises(ValidationError):
        Signal(
            type=SignalType.HIGH_BOUNCE,
            severity="URGENT",  # type: ignore[arg-type]
            data={},
            ts=datetime.now(UTC),
        )


def test_autonomy_level_matches_sql_check() -> None:
    assert {a.value for a in AutonomyLevel} == {"AUTO", "NOTIFY", "ASK", "FORBIDDEN"}


def test_hypothesis_state_matches_sql_check() -> None:
    assert {s.value for s in HypothesisState} == {
        "running",
        "confirmed",
        "rejected",
        "inconclusive",
        "rolled_back",
        "waiting_budget",
    }


def test_hypothesis_outcome_matches_sql_check() -> None:
    assert {o.value for o in HypothesisOutcome} == {"positive", "negative", "neutral"}
