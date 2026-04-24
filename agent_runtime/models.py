"""Shared data types for smart-direct-agent-v3 (Wave 1).

This module is the single source of truth for Pydantic models and enums used
across Wave 1 tasks (signal_detector, brain, impact_tracker, decision_engine).
It MUST NOT import from other ``agent_runtime`` modules to keep the dependency
DAG acyclic — everything else imports from here.

Invariants encoded here must mirror the SQL schema (``migrations/001_initial.sql``):

* ``HypothesisType`` values  ↔  ``hypotheses.hypothesis_type`` CHECK values
* ``HypothesisState`` values ↔  ``hypotheses.state`` CHECK values
* ``AutonomyLevel`` values   ↔  ``hypotheses.autonomy_level`` CHECK values
* ``HypothesisDraft.validate_attribution`` ↔ ``attribution_single`` SQL CHECK
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SignalType(StrEnum):
    BUDGET_THRESHOLD = "budget_threshold"
    SPEND_NO_CLICKS = "spend_no_clicks"
    HIGH_BOUNCE = "high_bounce"
    ZERO_LEADS = "zero_leads"
    HIGH_CPA = "high_cpa"
    LANDING_BROKEN = "landing_broken"
    LANDING_SLOW = "landing_slow"
    GARBAGE_QUERIES = "garbage_queries"
    CAMPAIGN_STOPPED = "campaign_stopped"
    API_ERROR = "api_error"


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: SignalType
    severity: Literal["info", "warning", "critical"]
    data: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


class HypothesisType(StrEnum):
    AD = "ad"
    NEG_KW = "neg_kw"
    IMAGE = "image"
    LANDING = "landing"
    NEW_CAMP = "new_camp"
    FORMAT_CHANGE = "format_change"
    STRATEGY_SWITCH = "strategy_switch"
    ACCOUNT_LEVEL = "account_level"


HYPOTHESIS_BUDGET_CAP: dict[HypothesisType, int] = {
    HypothesisType.AD: 500,
    HypothesisType.NEG_KW: 300,
    HypothesisType.IMAGE: 500,
    HypothesisType.LANDING: 2500,
    HypothesisType.NEW_CAMP: 3500,
    HypothesisType.FORMAT_CHANGE: 7000,
    HypothesisType.STRATEGY_SWITCH: 5000,
    HypothesisType.ACCOUNT_LEVEL: 500,
}


class AutonomyLevel(StrEnum):
    AUTO = "AUTO"
    NOTIFY = "NOTIFY"
    ASK = "ASK"
    FORBIDDEN = "FORBIDDEN"


class HypothesisState(StrEnum):
    RUNNING = "running"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    ROLLED_BACK = "rolled_back"
    WAITING_BUDGET = "waiting_budget"


class HypothesisOutcome(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class HypothesisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_type: HypothesisType
    hypothesis: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    actions: list[dict[str, Any]] = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)
    ad_group_id: int | None = None
    campaign_id: int | None = None

    @model_validator(mode="after")
    def validate_attribution(self) -> HypothesisDraft:
        """Mirror SQL ``attribution_single`` CHECK (see tech-spec Decision 5).

        Runs after all fields are bound so ``ad_group_id`` / ``campaign_id``
        are already populated (unlike a field validator on ``actions``).
        """
        if self.hypothesis_type == HypothesisType.ACCOUNT_LEVEL:
            return self
        if self.ad_group_id is None and self.campaign_id is None:
            raise ValueError(
                "attribution_single: hypothesis with type "
                f"{self.hypothesis_type.value!r} must set "
                "ad_group_id or campaign_id"
            )
        return self


__all__ = [
    "AutonomyLevel",
    "HypothesisDraft",
    "HypothesisOutcome",
    "HypothesisState",
    "HypothesisType",
    "HYPOTHESIS_BUDGET_CAP",
    "Signal",
    "SignalType",
]
