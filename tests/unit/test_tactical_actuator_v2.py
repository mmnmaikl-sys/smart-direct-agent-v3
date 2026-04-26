"""Tests for tactical_actuator v2 — multi-signal composite rules.

v1 used lifetime CTR only (one signal). After Deep Research 2026-04
on what experienced директологи track on launch (eLama, K50, ppc.world,
vc.ru БФЛ benchmarks: CPL 625-684₽, CPO 22 567₽), v2 uses 4 signals
per cid plus sample-size confidence:

  * 7d clicks (Direct stats)
  * 7d cost (Direct stats)
  * 7d leads (Bitrix via bitrix_feedback_traffic_split)
  * 7d bounce_rate (Metrika lastDirectClickOrder)

Composite rules (in priority order — first match wins):
  1. EARLY_KILL_SWITCH — clicks>=100 AND leads==0 AND bounce>60% → starve to floor
  2. STARVE_RED — CPL>1500 AND bounce>55% AND sample OK → -50% budget
  3. SCALE_GREEN — CPL<800 AND bounce<40% AND sample OK → +30% budget
  4. NOOP — control band or insufficient sample
"""

from __future__ import annotations

from agent_runtime.jobs.tactical_actuator import (
    CampaignSignals,
    decide_v2,
)


def _sig(
    cid: int,
    name: str,
    *,
    daily_rub: int,
    clicks_7d: int,
    cost_7d: float,
    leads_7d: int,
    bounce_pct: float | None,
) -> CampaignSignals:
    return CampaignSignals(
        cid=cid,
        name=name,
        daily_rub=daily_rub,
        clicks_7d=clicks_7d,
        cost_7d=cost_7d,
        leads_7d=leads_7d,
        bounce_pct=bounce_pct,
    )


# --- EARLY_KILL_SWITCH ------------------------------------------------------


def test_early_kill_when_100_clicks_zero_leads_high_bounce() -> None:
    """The owner's 03.04 incident pattern: junk traffic burning budget."""
    s = _sig(
        709353078, "mfo", daily_rub=300, clicks_7d=120, cost_7d=8000.0, leads_7d=0, bounce_pct=72.0
    )
    d = decide_v2(s)
    assert d.kind == "early_kill"
    assert d.new_daily_rub == 100  # floor
    assert "0 leads" in d.reason or "EARLY_KILL" in d.reason


def test_no_early_kill_when_bounce_acceptable() -> None:
    """100+ clicks 0 leads but bounce ok → not early-kill, may be cold but salvageable."""
    s = _sig(
        709353034,
        "pensioner",
        daily_rub=1500,
        clicks_7d=120,
        cost_7d=8000.0,
        leads_7d=0,
        bounce_pct=35.0,
    )
    d = decide_v2(s)
    assert d.kind != "early_kill"  # bounce acceptable, not the obvious junk pattern


def test_no_early_kill_below_clicks_threshold() -> None:
    """50 clicks too few — wait for more sample even if 0 leads."""
    s = _sig(
        709353058,
        "mother",
        daily_rub=500,
        clicks_7d=50,
        cost_7d=2000.0,
        leads_7d=0,
        bounce_pct=80.0,
    )
    d = decide_v2(s)
    assert d.kind == "noop"  # insufficient sample


# --- STARVE_RED ------------------------------------------------------------


def test_starve_red_high_cpl_high_bounce() -> None:
    """CPL=2000, bounce=70%, ≥50 clicks AND ≥1 lead — clear red signal."""
    s = _sig(
        709353005,
        "rabotyaga",
        daily_rub=300,
        clicks_7d=80,
        cost_7d=4000.0,
        leads_7d=2,
        bounce_pct=70.0,
    )
    # CPL = 4000 / 2 = 2000₽
    d = decide_v2(s)
    assert d.kind == "starve"
    assert d.new_daily_rub == 150  # 300 * 0.5
    assert "CPL" in d.reason


def test_no_starve_when_cpl_acceptable() -> None:
    """CPL=400 even with high bounce — leads coming in cheap, don't starve."""
    s = _sig(
        709353034,
        "pensioner",
        daily_rub=1500,
        clicks_7d=80,
        cost_7d=4000.0,
        leads_7d=10,
        bounce_pct=70.0,
    )
    # CPL = 400₽
    d = decide_v2(s)
    assert d.kind != "starve"


# --- SCALE_GREEN -----------------------------------------------------------


def test_scale_green_low_cpl_low_bounce_sample_ok() -> None:
    """CPL=600, bounce=30%, 100 clicks 5 leads — clear winner."""
    s = _sig(
        709353034,
        "pensioner",
        daily_rub=1500,
        clicks_7d=100,
        cost_7d=3000.0,
        leads_7d=5,
        bounce_pct=30.0,
    )
    # CPL = 600₽
    d = decide_v2(s)
    assert d.kind == "scale"
    assert d.new_daily_rub == 1950  # 1500 * 1.3
    assert "CPL" in d.reason


def test_no_scale_when_bounce_too_high() -> None:
    """CPL=600 but bounce=55% — quality concern, don't pour fuel."""
    s = _sig(
        709353034,
        "pensioner",
        daily_rub=1500,
        clicks_7d=100,
        cost_7d=3000.0,
        leads_7d=5,
        bounce_pct=55.0,
    )
    d = decide_v2(s)
    assert d.kind != "scale"


def test_scale_clamped_to_max() -> None:
    """Scale +30% but result above 2500 → clamp."""
    s = _sig(
        709353034,
        "pensioner",
        daily_rub=2200,
        clicks_7d=100,
        cost_7d=3000.0,
        leads_7d=5,
        bounce_pct=30.0,
    )
    d = decide_v2(s)
    assert d.kind == "scale"
    assert d.new_daily_rub == 2500


# --- sample-size guard -----------------------------------------------------


def test_noop_when_clicks_below_sample_floor_and_no_kill() -> None:
    """20 clicks too few for any decision (and no kill condition)."""
    s = _sig(
        709353099,
        "property",
        daily_rub=300,
        clicks_7d=20,
        cost_7d=600.0,
        leads_7d=0,
        bounce_pct=20.0,
    )
    d = decide_v2(s)
    assert d.kind == "noop"
    assert "sample" in d.reason.lower() or "insufficient" in d.reason.lower()


def test_noop_when_no_metrika_data() -> None:
    """bounce_pct=None → can't apply rules that depend on bounce → noop."""
    s = _sig(
        709353034,
        "pensioner",
        daily_rub=1500,
        clicks_7d=100,
        cost_7d=3000.0,
        leads_7d=5,
        bounce_pct=None,
    )
    d = decide_v2(s)
    assert d.kind == "noop"
    assert "bounce" in d.reason.lower() or "metrika" in d.reason.lower()


def test_noop_in_control_band() -> None:
    """CPL=900 (between 800 and 1500), bounce=45% — neither winner nor loser."""
    s = _sig(
        709353058,
        "mother",
        daily_rub=500,
        clicks_7d=80,
        cost_7d=4500.0,
        leads_7d=5,
        bounce_pct=45.0,
    )
    d = decide_v2(s)
    assert d.kind == "noop"
