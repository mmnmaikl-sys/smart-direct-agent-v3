"""Reference configuration for 5 own [24bfl] persona campaigns.

This is the **expected state** described in
``memory/24bfl_campaigns_setup.md`` (26.04.2026 + Presnyakov checklist).
``settings_reconciler`` reads live state from Direct API and diffs it
against this reference. Drift = either hand-edit or platform regression.

Source of truth for owner-approved settings; if owner changes intent,
update this file (one PR), not the live cabinet directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Retargeting condition IDs in 9061212 cabinet (verified 26.04.2026 via
# retargetinglists.get).
RETG_BOUNCE_30S = 39729064  # "Отказы до 30 сек — исключить"
RETG_LOOK_ALIKE = 4725087  # "Похожий сегмент"


@dataclass(frozen=True)
class DemoMod:
    """One demographic adjustment."""

    age: str  # AGE_0_17, AGE_18_24, AGE_25_34, AGE_35_44, AGE_45_54, AGE_55
    gender: str | None  # GENDER_MALE / GENDER_FEMALE / None (any)
    bid_modifier: int  # 0 = -100% (exclude), 100 = base, 150 = +50%


@dataclass(frozen=True)
class RetgMod:
    """One retargeting adjustment."""

    retargeting_condition_id: int
    bid_modifier: int


@dataclass(frozen=True)
class CampaignReference:
    """Reference for a single own campaign."""

    cid: int
    persona: str
    landing_path: str  # /lp/{persona}/
    demographics: tuple[DemoMod, ...]
    retargeting: tuple[RetgMod, ...]
    # TimeTargeting schedule: 7 strings, one per day-of-week (1=Mon..7=Sun).
    # Format: "{day},{h0_pct},{h1_pct},...,{h23_pct}" — 100=base, 50=half, 0=off.
    schedule_items: tuple[str, ...] = field(
        default=(
            "1,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100,100,50,50,50,50",
            "2,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100,100,50,50,50,50",
            "3,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100,100,50,50,50,50",
            "4,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100,100,50,50,50,50",
            "5,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100,100,50,50,50,50",
            "6,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100,100,50,50,50,50",
            "7,50,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100,100,100,100,50,50,50,50",
        )
    )
    holiday_bid_percent: int = 100
    # DRAFT ads policy: any ad in Status=DRAFT is a stuck creative — must be
    # moderated. ACCEPTED ones are allowed to be OFF (paused) but not DRAFT.
    max_draft_ads: int = 0


_COMMON_DEMO: tuple[DemoMod, ...] = (
    DemoMod(age="AGE_0_17", gender=None, bid_modifier=0),
    DemoMod(age="AGE_18_24", gender=None, bid_modifier=0),
)

_COMMON_RETG: tuple[RetgMod, ...] = (
    RetgMod(retargeting_condition_id=RETG_BOUNCE_30S, bid_modifier=0),
    RetgMod(retargeting_condition_id=RETG_LOOK_ALIKE, bid_modifier=150),
)


OWN_CAMPAIGNS_REFERENCE: dict[int, CampaignReference] = {
    709353005: CampaignReference(
        cid=709353005,
        persona="rabotyaga",
        landing_path="/lp/rabotyaga/",
        demographics=_COMMON_DEMO,
        retargeting=_COMMON_RETG,
    ),
    709353034: CampaignReference(
        cid=709353034,
        persona="pensioner",
        landing_path="/lp/pensioner/",
        demographics=_COMMON_DEMO + (DemoMod(age="AGE_55", gender=None, bid_modifier=150),),
        retargeting=_COMMON_RETG,
    ),
    709353058: CampaignReference(
        cid=709353058,
        persona="mother",
        landing_path="/lp/mother/",
        demographics=_COMMON_DEMO
        + (
            DemoMod(age="AGE_25_34", gender="GENDER_FEMALE", bid_modifier=130),
            DemoMod(age="AGE_35_44", gender="GENDER_FEMALE", bid_modifier=130),
        ),
        retargeting=_COMMON_RETG,
    ),
    709353078: CampaignReference(
        cid=709353078,
        persona="mfo",
        landing_path="/lp/mfo/",
        demographics=_COMMON_DEMO,
        retargeting=_COMMON_RETG,
    ),
    709353099: CampaignReference(
        cid=709353099,
        persona="property",
        landing_path="/lp/property/",
        demographics=_COMMON_DEMO,
        retargeting=_COMMON_RETG,
    ),
}


__all__ = [
    "CampaignReference",
    "DemoMod",
    "OWN_CAMPAIGNS_REFERENCE",
    "RETG_BOUNCE_30S",
    "RETG_LOOK_ALIKE",
    "RetgMod",
]
