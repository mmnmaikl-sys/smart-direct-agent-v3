"""Tests for funnel_cr_tracker pure aggregator.

Computes per-cid funnel stats:
  visits → leads → quals (ЕПК UF_CRM_1740791420) → deals (C45:WON)
plus CPL, CPQ (cost per qualified), CPO (cost per won).
"""

from __future__ import annotations

from agent_runtime.jobs.funnel_cr_tracker import FunnelStats, compute_funnel


def test_full_funnel_with_all_stages() -> None:
    stats = compute_funnel(
        cid=709353034,
        name="pensioner",
        cost_rub=4500.0,
        leads=10,
        quals=4,
        wons=1,
    )
    assert stats.cid == 709353034
    assert stats.cpl == 450.0  # 4500 / 10
    assert stats.cpq == 1125.0  # 4500 / 4
    assert stats.cpo == 4500.0  # 4500 / 1
    assert stats.cr_lead_qual == 0.4  # 4/10
    assert stats.cr_qual_won == 0.25  # 1/4
    assert stats.cr_lead_won == 0.1  # 1/10


def test_zero_leads_no_division_errors() -> None:
    stats = compute_funnel(
        cid=709353058,
        name="mother",
        cost_rub=200.0,
        leads=0,
        quals=0,
        wons=0,
    )
    assert stats.cpl is None
    assert stats.cpq is None
    assert stats.cpo is None
    assert stats.cr_lead_qual == 0.0
    assert stats.cr_qual_won == 0.0


def test_leads_but_no_quals() -> None:
    """Lead pipeline empty after qualification — typical 'мусорный трафик' signal."""
    stats = compute_funnel(
        cid=709353005,
        name="rabotyaga",
        cost_rub=3000.0,
        leads=8,
        quals=0,
        wons=0,
    )
    assert stats.cpl == 375.0
    assert stats.cpq is None
    assert stats.cr_lead_qual == 0.0
    assert stats.cr_qual_won == 0.0


def test_funnel_stats_is_dataclass() -> None:
    stats = FunnelStats(
        cid=1,
        name="x",
        cost_rub=100.0,
        leads=2,
        quals=1,
        wons=0,
        cpl=50.0,
        cpq=100.0,
        cpo=None,
        cr_lead_qual=0.5,
        cr_qual_won=0.0,
        cr_lead_won=0.0,
    )
    assert stats.cid == 1


def test_compute_handles_floats_safely() -> None:
    stats = compute_funnel(cid=1, name="x", cost_rub=100.5, leads=3, quals=2, wons=1)
    assert abs(stats.cpl - 33.5) < 0.01
    assert abs(stats.cr_lead_qual - 2 / 3) < 0.01
