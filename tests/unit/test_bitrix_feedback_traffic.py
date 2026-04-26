"""Unit tests for bitrix_feedback._compute_traffic_split — Task 02.

Tests the pure aggregator that produces sda_state[bitrix_feedback_traffic_split]
JSONB. The asymmetry (own gets per-cid breakdown, contractor + organic stay
aggregate) reflects what we actually know:

  * own        — we own the Direct cabinet, so per-cid spend is available
  * contractor — UTM_CAMPAIGN exists but cid is in someone else's cabinet,
                 so we only see leads through Bitrix; spend stays unknown
  * organic    — no UTM_CAMPAIGN at all, no attribution, only Bitrix count

The pure function is decoupled from DB / API so tests stay fast.
"""

from __future__ import annotations

from agent_runtime.jobs.bitrix_feedback import _compute_traffic_split

_TS = "2026-04-26T15:00:00+03:00"


def test_compute_split_own_only() -> None:
    payload = _compute_traffic_split(
        own_won_by_cid={709353005: 3},
        own_spend_by_cid={709353005: 4500.0},
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
    )

    assert payload["snapshot_at"] == _TS
    # leads/cpl fields added 26.04.2026; traffic_split now also stores
    # Bitrix lead counts so tactical_actuator decision rules from Deep
    # Research can use real CPL (not just CPA-on-won).
    assert payload["own"] == {
        "709353005": {
            "won": 3,
            "leads": 0,
            "cost_rub": 4500.0,
            "cpa_won": 1500.0,
            "cpl": None,
        },
    }
    assert payload["contractor_aggregate"] == {"won": 0, "by_campaign_id": {}}
    assert payload["organic_aggregate"] == {"won": 0}


def test_compute_split_mixed() -> None:
    payload = _compute_traffic_split(
        own_won_by_cid={709353005: 3, 709353034: 1},
        own_spend_by_cid={709353005: 4500.0, 709353034: 800.0},
        contractor_won_by_cid={709224565: 2, 708138968: 5},
        organic_won=1,
        captured_at=_TS,
    )

    assert payload["own"]["709353005"]["won"] == 3
    assert payload["own"]["709353005"]["cpa_won"] == 1500.0
    assert payload["own"]["709353034"]["cpa_won"] == 800.0
    assert payload["contractor_aggregate"]["won"] == 7
    assert payload["contractor_aggregate"]["by_campaign_id"] == {
        "709224565": 2,
        "708138968": 5,
    }
    assert payload["organic_aggregate"]["won"] == 1


def test_compute_split_zero_won_handles_safely() -> None:
    """own cid with won=0 → cpa_won is None (not division-by-zero)."""
    payload = _compute_traffic_split(
        own_won_by_cid={709353005: 0},
        own_spend_by_cid={709353005: 1200.0},
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
    )

    assert payload["own"]["709353005"]["won"] == 0
    assert payload["own"]["709353005"]["cost_rub"] == 1200.0
    assert payload["own"]["709353005"]["cpa_won"] is None


def test_compute_split_missing_spend_treated_as_zero() -> None:
    """own cid present in won but absent in spend → cost_rub=0, cpa_won=0."""
    payload = _compute_traffic_split(
        own_won_by_cid={709353005: 2},
        own_spend_by_cid={},  # spend missing entirely
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
    )

    assert payload["own"]["709353005"]["cost_rub"] == 0.0
    assert payload["own"]["709353005"]["cpa_won"] == 0.0


def test_compute_split_empty_state() -> None:
    """No data at all → snapshot still recorded with zeroed buckets."""
    payload = _compute_traffic_split(
        own_won_by_cid={},
        own_spend_by_cid={},
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
    )

    assert payload["own"] == {}
    assert payload["contractor_aggregate"]["won"] == 0
    assert payload["contractor_aggregate"]["by_campaign_id"] == {}
    assert payload["organic_aggregate"]["won"] == 0


def test_compute_split_serialises_int_keys_as_str() -> None:
    """Postgres JSONB requires string keys — int cid must serialise to str."""
    payload = _compute_traffic_split(
        own_won_by_cid={709353005: 1},
        own_spend_by_cid={709353005: 100.0},
        contractor_won_by_cid={709224565: 1},
        organic_won=0,
        captured_at=_TS,
    )

    for key in payload["own"]:
        assert isinstance(key, str), f"own key must be str, got {type(key)}"
    for key in payload["contractor_aggregate"]["by_campaign_id"]:
        assert isinstance(key, str), f"by_campaign_id key must be str, got {type(key)}"


def test_compute_split_window_hours_default() -> None:
    payload = _compute_traffic_split(
        own_won_by_cid={},
        own_spend_by_cid={},
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
    )
    assert payload["window_hours"] == 7 * 24  # 7-day window like the rest of bitrix_feedback


def test_compute_split_leads_alongside_won_with_cpl() -> None:
    """26.04.2026: leads + cpl land in own[cid] alongside won + cpa_won.

    Decision rules from director_metrics_2026-04 expect CPL (cost / leads),
    not just CPA-on-won — leads land within hours of click whereas won lags
    3-14d.
    """
    payload = _compute_traffic_split(
        own_won_by_cid={709353005: 1},
        own_spend_by_cid={709353005: 1500.0},
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
        own_leads_by_cid={709353005: 5},
    )
    own = payload["own"]["709353005"]
    assert own["won"] == 1
    assert own["leads"] == 5
    assert own["cost_rub"] == 1500.0
    assert own["cpa_won"] == 1500.0  # 1500 / 1 won
    assert own["cpl"] == 300.0  # 1500 / 5 leads


def test_compute_split_leads_only_no_won_yet() -> None:
    """Cold campaign: leads=2, won=0 (won lag) — CPL valid, CPA None."""
    payload = _compute_traffic_split(
        own_won_by_cid={},
        own_spend_by_cid={709353005: 1000.0},
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
        own_leads_by_cid={709353005: 2},
    )
    own = payload["own"]["709353005"]
    assert own["won"] == 0
    assert own["leads"] == 2
    assert own["cpa_won"] is None  # no won yet
    assert own["cpl"] == 500.0  # 1000 / 2


def test_compute_split_leads_optional_param_default() -> None:
    """own_leads_by_cid omitted → leads=0, cpl=0 when spend=0 / None when spend>0."""
    payload = _compute_traffic_split(
        own_won_by_cid={709353005: 1},
        own_spend_by_cid={709353005: 1500.0},
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
    )
    own = payload["own"]["709353005"]
    assert own["leads"] == 0
    assert own["cpl"] is None  # spend>0 + leads=0 → unknown CPL


def test_compute_split_window_hours_override() -> None:
    payload = _compute_traffic_split(
        own_won_by_cid={},
        own_spend_by_cid={},
        contractor_won_by_cid={},
        organic_won=0,
        captured_at=_TS,
        window_hours=24,
    )
    assert payload["window_hours"] == 24
