"""Decision Engine — agent autonomy matrix (Decision 7 in tech-spec).

Ported from v2 ``app/decision_engine.py`` with two targeted changes:

1. ``affected_budget_pct`` is a callable parameter at every layer — the v2
   code had ``0.2`` hardcoded in a handful of places which masked which
   call site really knew the share. Callers in Wave 1 compute this from
   ``HYPOTHESIS_BUDGET_CAP`` or ``sda_state.weekly_budget_total_rub``.
2. ``min_confident_samples`` is per-call, because different hypothesis
   types need different statistical depth (10 observations suffice for a
   minus-keyword rule; 100 are needed before swapping a bid strategy).

Additionally, :func:`_apply_time_decay` introduces a time-weighted effective
sample count so a gang of stale "confirmed" data points stops masquerading
as fresh evidence.

``AutonomyLevel`` lives in :mod:`agent_runtime.models` (Wave 1 single source
of truth) — this module only imports it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent_runtime.models import AutonomyLevel

logger = logging.getLogger(__name__)


# Irreversibility scores per action type (0-100). Copied as-is from v2;
# ``DANGER_ACTIONS`` in trust_levels.py derives from this map (filter >=70).
IRREVERSIBILITY: dict[str, int] = {
    "add_negative": 10,
    "pause_keyword": 10,
    "resume_keyword": 10,
    "bid_modifier": 15,
    "pause_group": 25,
    "resume_group": 20,
    "pause_campaign": 60,
    "resume_campaign": 40,
    "switch_strategy": 70,
    "delete_keyword": 85,
    "create_campaign": 40,
    "enable_autotargeting": 100,  # FORBIDDEN — v1 trauma
    "change_budget": 50,
    "change_ad_text": 45,
    "add_keywords": 30,
    "switch_landing": 35,
}

# Hard-block list. No trust level can override.
FORBIDDEN_ACTIONS: frozenset[str] = frozenset(
    {
        "enable_autotargeting",
        "create_campaign",
        "delete_campaign",
        "increase_budget",  # only decrease is allowed automatically
    }
)

# Per-day soft cap per action type.
ACTION_LIMITS: dict[str, int] = {
    "add_negative": 20,
    "pause_keyword": 5,
    "bid_modifier": 10,
    "pause_group": 2,
    "pause_campaign": 1,
    "switch_landing": 2,
}

# Aggregate mutation cap per week (all action types combined).
MAX_MUTATIONS_PER_WEEK = 5

# Cooldown between two actions on the same entity (hours).
ENTITY_COOLDOWN_HOURS = 72

_DEFAULT_IRREVERSIBILITY = 50


@dataclass(frozen=True)
class Decision:
    """Result of decision engine evaluation.

    ``data_points_effective`` captures the time-weighted count so audit_log
    can distinguish "30 samples all from last month" from "30 samples all
    from last week". ``affected_budget_pct`` is echoed back so the caller
    doesn't have to correlate its inputs separately.
    """

    action: str
    level: AutonomyLevel
    risk_score: float
    reason: str
    can_execute: bool
    affected_budget_pct: float
    data_points_effective: float


def _apply_time_decay(data_points_ages_days: list[float]) -> float:
    """Effective sample count with time-weighted decay.

    Weight schedule:
        age <= 7d  → 1.0
        7d < age < 30d → linear from 1.0 to 0.3
        age >= 30d → 0.3
        age < 0 → clamped to 0 (treat "future" as fresh; not expected in prod)
    """
    total = 0.0
    for raw_age in data_points_ages_days:
        age = max(0.0, float(raw_age))
        if age <= 7.0:
            total += 1.0
        elif age >= 30.0:
            total += 0.3
        else:
            # Linear interpolation between (7, 1.0) and (30, 0.3)
            total += 1.0 - (age - 7.0) / 23.0 * 0.7
    return total


def calculate_risk(
    action: str,
    affected_budget_pct: float = 0.0,
    data_points: int = 0,
    min_confident_samples: int = 30,
    data_points_ages_days: list[float] | None = None,
) -> tuple[float, float]:
    """Return ``(risk_score, effective_samples)``.

    Formula: ``money_impact * 0.3 + irreversibility * 0.4 + uncertainty * 0.3``

    If ``data_points_ages_days`` is given it overrides ``data_points`` via
    :func:`_apply_time_decay`. Otherwise the legacy v2 behaviour is preserved
    (raw count comparison against ``min_confident_samples``).
    """
    money_impact = min(abs(affected_budget_pct) * 100, 100)
    irreversibility = IRREVERSIBILITY.get(action, _DEFAULT_IRREVERSIBILITY)

    if data_points_ages_days is not None:
        effective = _apply_time_decay(data_points_ages_days)
    else:
        effective = float(data_points)

    if effective >= min_confident_samples:
        uncertainty = 10.0
    elif effective >= min_confident_samples * 0.5:
        uncertainty = 50.0
    else:
        uncertainty = 90.0

    risk = money_impact * 0.3 + irreversibility * 0.4 + uncertainty * 0.3
    return risk, effective


def evaluate(
    action: str,
    affected_budget_pct: float = 0.0,
    data_points: int = 0,
    min_confident_samples: int = 30,
    data_points_ages_days: list[float] | None = None,
) -> Decision:
    """Classify an action into AUTO / NOTIFY / ASK / FORBIDDEN.

    Hard block wins first: any action in :data:`FORBIDDEN_ACTIONS` is
    reported as FORBIDDEN regardless of the risk numbers.

    Risk thresholds (from v2, Decision 7):
        <=25 → AUTO
        <=50 → NOTIFY
        <=75 → ASK
        >75  → ASK (flagged "very high" in reason)
    """
    if action in FORBIDDEN_ACTIONS:
        return Decision(
            action=action,
            level=AutonomyLevel.FORBIDDEN,
            risk_score=100.0,
            reason=f"Action '{action}' is hardcoded FORBIDDEN",
            can_execute=False,
            affected_budget_pct=affected_budget_pct,
            data_points_effective=float(data_points),
        )

    if action not in IRREVERSIBILITY:
        logger.warning(
            "decision_engine: unknown action '%s', using irreversibility default %d",
            action,
            _DEFAULT_IRREVERSIBILITY,
        )

    risk, effective = calculate_risk(
        action,
        affected_budget_pct=affected_budget_pct,
        data_points=data_points,
        min_confident_samples=min_confident_samples,
        data_points_ages_days=data_points_ages_days,
    )

    if risk <= 25:
        level = AutonomyLevel.AUTO
        reason = f"Low risk ({risk:.0f}): auto-execute"
    elif risk <= 50:
        level = AutonomyLevel.NOTIFY
        reason = f"Medium risk ({risk:.0f}): execute + notify"
    elif risk <= 75:
        level = AutonomyLevel.ASK
        reason = f"High risk ({risk:.0f}): confirmation required"
    else:
        level = AutonomyLevel.ASK
        reason = f"Very high risk ({risk:.0f}): block until explicit approve"

    decision = Decision(
        action=action,
        level=level,
        risk_score=risk,
        reason=reason,
        can_execute=level in (AutonomyLevel.AUTO, AutonomyLevel.NOTIFY),
        affected_budget_pct=affected_budget_pct,
        data_points_effective=effective,
    )
    logger.info(
        "decision_engine: action=%s level=%s risk=%.1f effective=%.1f",
        action,
        level.value,
        risk,
        effective,
    )
    return decision


# --- helpers (copied from v2, unchanged) ------------------------------------


def check_daily_limit(action: str, today_count: int) -> bool:
    limit = ACTION_LIMITS.get(action, 10)
    return today_count < limit


def check_weekly_mutations(mutations_this_week: int) -> bool:
    return mutations_this_week < MAX_MUTATIONS_PER_WEEK


def check_cooldown(last_action_hours_ago: float) -> bool:
    return last_action_hours_ago >= ENTITY_COOLDOWN_HOURS


__all__ = [
    "ACTION_LIMITS",
    "ENTITY_COOLDOWN_HOURS",
    "FORBIDDEN_ACTIONS",
    "IRREVERSIBILITY",
    "MAX_MUTATIONS_PER_WEEK",
    "Decision",
    "calculate_risk",
    "check_cooldown",
    "check_daily_limit",
    "check_weekly_mutations",
    "evaluate",
]
