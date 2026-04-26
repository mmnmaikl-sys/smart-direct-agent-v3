"""Tests for settings_reconciler pure diff function."""

from __future__ import annotations

from agent_runtime.jobs.settings_reconciler import (
    CampaignDiff,
    _build_diffs,
    diff_campaign,
    format_telegram,
)
from agent_runtime.knowledge.own_campaigns_reference import (
    OWN_CAMPAIGNS_REFERENCE,
    RETG_BOUNCE_30S,
    RETG_LOOK_ALIKE,
)

_REF_RABOTYAGA = OWN_CAMPAIGNS_REFERENCE[709353005]


def _matching_live_demo(ref) -> list[dict]:
    return [
        {"Age": d.age, "Gender": d.gender, "BidModifier": d.bid_modifier} for d in ref.demographics
    ]


def _matching_live_retg(ref) -> list[dict]:
    return [
        {"RetargetingConditionId": r.retargeting_condition_id, "BidModifier": r.bid_modifier}
        for r in ref.retargeting
    ]


def test_clean_state_when_live_matches_reference() -> None:
    diff = diff_campaign(
        ref=_REF_RABOTYAGA,
        live_demo=_matching_live_demo(_REF_RABOTYAGA),
        live_retg=_matching_live_retg(_REF_RABOTYAGA),
        live_schedule=_REF_RABOTYAGA.schedule_items,
        live_draft_count=0,
    )
    assert diff.is_clean
    assert diff.total_issues == 0


def test_missing_demo_modifier_detected() -> None:
    diff = diff_campaign(
        ref=_REF_RABOTYAGA,
        live_demo=[],  # no demo at all
        live_retg=_matching_live_retg(_REF_RABOTYAGA),
        live_schedule=_REF_RABOTYAGA.schedule_items,
        live_draft_count=0,
    )
    assert not diff.is_clean
    # Both AGE_0_17 and AGE_18_24 should be reported missing
    assert len(diff.missing_demo) == 2
    ages = {m.age for m in diff.missing_demo}
    assert ages == {"AGE_0_17", "AGE_18_24"}


def test_wrong_bid_modifier_value_treated_as_missing() -> None:
    diff = diff_campaign(
        ref=_REF_RABOTYAGA,
        live_demo=[
            {"Age": "AGE_0_17", "Gender": None, "BidModifier": 50},  # wrong: should be 0
            {"Age": "AGE_18_24", "Gender": None, "BidModifier": 0},
        ],
        live_retg=_matching_live_retg(_REF_RABOTYAGA),
        live_schedule=_REF_RABOTYAGA.schedule_items,
        live_draft_count=0,
    )
    # AGE_0_17 has wrong bid → counted as missing
    assert len(diff.missing_demo) == 1
    assert diff.missing_demo[0].age == "AGE_0_17"
    assert diff.missing_demo[0].bid_modifier == 0


def test_extra_retargeting_modifier_flagged() -> None:
    diff = diff_campaign(
        ref=_REF_RABOTYAGA,
        live_demo=_matching_live_demo(_REF_RABOTYAGA),
        live_retg=[
            {"RetargetingConditionId": RETG_BOUNCE_30S, "BidModifier": 0},
            {"RetargetingConditionId": RETG_LOOK_ALIKE, "BidModifier": 150},
            {"RetargetingConditionId": 999999, "BidModifier": 200},  # rogue
        ],
        live_schedule=_REF_RABOTYAGA.schedule_items,
        live_draft_count=0,
    )
    assert not diff.is_clean
    assert len(diff.extra_retg) == 1
    assert "999999" in diff.extra_retg[0]


def test_draft_ads_flagged() -> None:
    diff = diff_campaign(
        ref=_REF_RABOTYAGA,
        live_demo=_matching_live_demo(_REF_RABOTYAGA),
        live_retg=_matching_live_retg(_REF_RABOTYAGA),
        live_schedule=_REF_RABOTYAGA.schedule_items,
        live_draft_count=3,
    )
    assert not diff.is_clean
    assert diff.draft_ads_count == 3


def test_no_schedule_treated_as_drift() -> None:
    diff = diff_campaign(
        ref=_REF_RABOTYAGA,
        live_demo=_matching_live_demo(_REF_RABOTYAGA),
        live_retg=_matching_live_retg(_REF_RABOTYAGA),
        live_schedule=None,
        live_draft_count=0,
    )
    assert diff.schedule_drift


def test_build_diffs_groups_by_campaign() -> None:
    live = {
        "bidmodifiers": [
            {
                "CampaignId": 709353005,
                "Type": "DEMOGRAPHICS_ADJUSTMENT",
                "DemographicsAdjustment": {"Age": "AGE_0_17", "Gender": None, "BidModifier": 0},
            },
            {
                "CampaignId": 709353005,
                "Type": "DEMOGRAPHICS_ADJUSTMENT",
                "DemographicsAdjustment": {"Age": "AGE_18_24", "Gender": None, "BidModifier": 0},
            },
            {
                "CampaignId": 709353005,
                "Type": "RETARGETING_ADJUSTMENT",
                "RetargetingAdjustment": {
                    "RetargetingConditionId": RETG_BOUNCE_30S,
                    "BidModifier": 0,
                },
            },
            {
                "CampaignId": 709353005,
                "Type": "RETARGETING_ADJUSTMENT",
                "RetargetingAdjustment": {
                    "RetargetingConditionId": RETG_LOOK_ALIKE,
                    "BidModifier": 150,
                },
            },
        ],
        "campaigns": [
            {
                "Id": 709353005,
                "TimeTargeting": {"Schedule": {"Items": list(_REF_RABOTYAGA.schedule_items)}},
            }
        ],
        "draft_ads": [],
    }
    diffs = _build_diffs(live)
    assert len(diffs) == len(OWN_CAMPAIGNS_REFERENCE)
    by_cid = {d.cid: d for d in diffs}
    # rabotyaga is fully matched
    assert by_cid[709353005].is_clean
    # other 4 cids have nothing in live → all dirty
    for cid in OWN_CAMPAIGNS_REFERENCE:
        if cid != 709353005:
            assert not by_cid[cid].is_clean


def test_format_telegram_clean_short_when_all_clean() -> None:
    diffs = [
        CampaignDiff(cid=709353005, persona="rabotyaga"),
        CampaignDiff(cid=709353034, persona="pensioner"),
    ]
    msg = format_telegram(diffs)
    assert "соответствуют reference" in msg
    assert "Drift" not in msg


def test_format_telegram_lists_dirty_campaigns() -> None:
    from agent_runtime.knowledge.own_campaigns_reference import DemoMod

    diffs = [
        CampaignDiff(
            cid=709353005,
            persona="rabotyaga",
            missing_demo=(DemoMod(age="AGE_0_17", gender=None, bid_modifier=0),),
        ),
        CampaignDiff(cid=709353034, persona="pensioner"),
    ]
    msg = format_telegram(diffs)
    assert "Drift" in msg
    assert "rabotyaga" in msg
    assert "AGE_0_17" in msg
    assert "Чисто (1)" in msg
