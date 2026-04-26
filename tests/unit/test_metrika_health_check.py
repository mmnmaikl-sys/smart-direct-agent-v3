"""Pure-rule tests for metrika_health_check.assess_health."""

from __future__ import annotations

from agent_runtime.jobs.metrika_health_check import _MIN_VISITS_FOR_SIGNAL, assess_health


def test_broken_when_visits_above_threshold_and_zero_goals() -> None:
    rows = {709353034: {"visits": 50, "bounce": 30.0, "goals": 0}}
    out = assess_health(rows)
    pensioner = next(h for h in out if h.cid == 709353034)
    assert pensioner.is_broken is True
    assert pensioner.visits == 50
    assert pensioner.goals == 0


def test_healthy_when_goals_present() -> None:
    rows = {709353034: {"visits": 50, "bounce": 30.0, "goals": 3}}
    out = assess_health(rows)
    pensioner = next(h for h in out if h.cid == 709353034)
    assert pensioner.is_broken is False
    assert pensioner.goals == 3


def test_cold_when_visits_below_threshold_no_alert() -> None:
    rows = {709353034: {"visits": 5, "bounce": 30.0, "goals": 0}}
    out = assess_health(rows)
    pensioner = next(h for h in out if h.cid == 709353034)
    assert pensioner.is_broken is False  # too cold to be a real signal


def test_missing_cid_zero_visits_not_broken() -> None:
    """A cid not in the Metrika response gets defaults (0 visits) — not broken."""
    out = assess_health({})
    assert all(h.visits == 0 and h.is_broken is False for h in out)


def test_returns_all_known_cids() -> None:
    """Always covers the full list of own campaigns, not just Metrika rows."""
    out = assess_health({})
    assert len(out) == 5
    assert {h.cid for h in out} == {709353005, 709353034, 709353058, 709353078, 709353099}


def test_threshold_constant_documented() -> None:
    """Sanity: threshold is a small but meaningful sample (>=10)."""
    assert _MIN_VISITS_FOR_SIGNAL >= 10
