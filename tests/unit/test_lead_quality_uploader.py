"""Tests for lead_quality_uploader.classify_deal pure decision."""

from __future__ import annotations

from agent_runtime.jobs.lead_quality_uploader import (
    _JUNK_STAGE_IDS,
    classify_deal,
)

_DEAL_BASE = {
    "ID": "12345",
    "UF_CRM_YCLID": "yclid_abc",
    "DATE_MODIFY": "2026-04-26T10:00:00+03:00",
}


def test_qualified_via_uf_crm_field() -> None:
    deal = {**_DEAL_BASE, "STAGE_ID": "C49:NEW", "UF_CRM_1740791420": "Y"}
    row = classify_deal(deal)
    assert row is not None
    assert row.target == "bfl_lead_qualified"
    assert row.external_id == "bitrix_deal_12345_quality_qualified"
    assert row.identifier_type == "YCLID"


def test_qualified_via_c49_won_stage() -> None:
    deal = {**_DEAL_BASE, "STAGE_ID": "C49:WON"}
    row = classify_deal(deal)
    assert row is not None
    assert row.target == "bfl_lead_qualified"


def test_qualified_via_c45_promotion() -> None:
    deal = {**_DEAL_BASE, "STAGE_ID": "C45:5"}
    row = classify_deal(deal)
    assert row is not None
    assert row.target == "bfl_lead_qualified"


def test_junk_via_stage() -> None:
    for stage in _JUNK_STAGE_IDS:
        deal = {**_DEAL_BASE, "STAGE_ID": stage}
        row = classify_deal(deal)
        assert row is not None, f"stage {stage} expected junk"
        assert row.target == "bfl_lead_junk"
        assert row.external_id == "bitrix_deal_12345_quality_junk"


def test_in_flight_returns_none() -> None:
    deal = {**_DEAL_BASE, "STAGE_ID": "C49:NEW"}  # not qualified, not junk
    assert classify_deal(deal) is None


def test_qualified_priority_over_junk_when_both_present() -> None:
    # Edge case: deal moved to C45:5 (in OP) but old STAGE_ID was junk.
    # Current stage wins; if stage is C45:5 we treat as qualified.
    deal = {**_DEAL_BASE, "STAGE_ID": "C45:5", "UF_CRM_1740791420": "Y"}
    row = classify_deal(deal)
    assert row is not None
    assert row.target == "bfl_lead_qualified"


def test_skip_when_no_identifier() -> None:
    deal = {"ID": "12345", "STAGE_ID": "C49:WON"}  # no yclid, no client_id
    assert classify_deal(deal) is None


def test_client_id_fallback_when_no_yclid() -> None:
    deal = {
        "ID": "12345",
        "STAGE_ID": "C49:WON",
        "UF_CRM_CLIENT_ID": "metrika_visitor_999",
        "DATE_MODIFY": "2026-04-26T10:00:00+03:00",
    }
    row = classify_deal(deal)
    assert row is not None
    assert row.identifier_type == "CLIENT_ID"
    assert row.identifier_value == "metrika_visitor_999"


def test_custom_target_names_propagate() -> None:
    deal = {**_DEAL_BASE, "STAGE_ID": "C49:WON"}
    row = classify_deal(deal, target_qualified="custom_q", target_junk="custom_j")
    assert row is not None
    assert row.target == "custom_q"
    deal_junk = {**_DEAL_BASE, "STAGE_ID": "C49:5"}
    row_j = classify_deal(deal_junk, target_qualified="custom_q", target_junk="custom_j")
    assert row_j is not None
    assert row_j.target == "custom_j"


def test_upload_dict_uses_yclid_or_client_id_key() -> None:
    deal_y = {**_DEAL_BASE, "STAGE_ID": "C49:WON"}
    row_y = classify_deal(deal_y)
    assert row_y is not None
    d = row_y.as_upload_dict()
    assert d["yclid"] == "yclid_abc"
    assert "client_id" not in d
    assert d["target"] == "bfl_lead_qualified"
    assert d["external_id"] == "bitrix_deal_12345_quality_qualified"
