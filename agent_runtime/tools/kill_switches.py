"""Seven kill-switch guards — last line of defence before a mutation ships.

Each guard implements the :class:`BaseKillSwitch` Protocol: one async ``check``
that returns :class:`KillSwitchResult` (``allow`` + ``reason`` + ``switch_name``).
:func:`run_all` fires every guard in parallel via ``asyncio.gather`` and wraps
exceptions into ``allow=False`` entries — ``fail-closed`` is the invariant here,
because the cost of a missed block (budget drain, PROTECTED campaign paused) is
higher than the cost of a false reject in shadow/assisted.

The brain wrapper (Task 12) owns:

- aggregating ``run_all`` results (``any(not r.allow)`` → reject),
- writing ``audit_log.kill_switch_triggered``,
- sending Telegram NOTIFY,
- ``trust_level`` overlay.

Guards themselves stay **pure**: input (action, context) → output
(KillSwitchResult). They do not mutate state, do not send alerts, and do not
look at ``trust_level`` — that keeps their tests trivial and their role
auditable.

The narrow :class:`_DirectLike` / :class:`_MetrikaLike` / :class:`_BitrixLike`
Protocols decouple guards from concrete client classes (Task 7 DirectAPI is
already available; Task 8 Bitrix/Metrika clients will satisfy these protocols
when they land). Tests pass mocks that implement the protocol.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Protocol, runtime_checkable

from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- models


@dataclass(frozen=True)
class Action:
    """Lightweight runtime representation of one HypothesisDraft action.

    HypothesisDraft.actions ships as ``list[dict[str, Any]]`` (Pydantic-friendly
    escape hatch); this dataclass gives guard code a typed view for the subset
    of fields they actually inspect.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(type=str(data.get("type", "")), params=dict(data.get("params") or {}))


@dataclass(frozen=True)
class KillSwitchResult:
    allow: bool
    reason: str
    switch_name: str


@runtime_checkable
class _DirectLike(Protocol):
    async def get_campaigns(self, ids: list[int]) -> list[dict[str, Any]]: ...
    async def get_adgroups(
        self,
        *,
        campaign_id: int | None = None,
        ids: list[int] | None = None,
    ) -> list[dict[str, Any]]: ...
    async def get_keywords(self, ad_group_ids: list[int]) -> list[dict[str, Any]]: ...
    async def get_campaign_stats(
        self, campaign_id: int, date_from: str, date_to: str
    ) -> dict[str, Any]: ...


@runtime_checkable
class _MetrikaLike(Protocol):
    async def recent_visits(self, hours: int) -> list[dict[str, Any]]: ...


@runtime_checkable
class _BitrixLike(Protocol):
    async def recent_leads(self, hours: int) -> list[dict[str, Any]]: ...


@dataclass
class KillSwitchContext:
    """Shared context handed to every guard by the brain wrapper (Task 12)."""

    pool: AsyncConnectionPool
    direct: _DirectLike
    metrika: _MetrikaLike | None
    bitrix: _BitrixLike | None
    settings: Settings
    trust_level: str  # "shadow" | "assisted" | "autonomous"
    hypothesis_id: str | None = None
    # Optional pre-fetched helpers so guards don't re-hit APIs when brain
    # already has the data. All optional: guards fall back to fetching.
    budget_history: dict[int, dict[str, float]] | None = None
    adgroup_productivity: dict[int, int] | None = None
    baseline_queries: list[str] | None = None
    recent_queries: list[str] | None = None
    bid_history_by_adgroup: dict[int, list[int]] | None = None  # bids in RUB
    weekly_budget_total_rub: int | None = None


class BaseKillSwitch(Protocol):
    name: ClassVar[str]

    async def check(self, action: Action, context: KillSwitchContext) -> KillSwitchResult: ...


# ----------------------------------------------------------- protected registry


# Hardcoded fallback. The authoritative list lives in sda_state.protected_
# keywords_registry (populated by a separate migration); the seed guarantees
# defence even if the DB row is missing on a cold deploy.
PROTECTED_KEYWORDS_SEED: frozenset[str] = frozenset(
    {
        "банкротство физ лиц",
        "банкротство физических лиц",
        "списание долгов физ лиц",
        "списание долгов физических лиц",
        "пройти процедуру банкротства",
        "стать банкротом физлицу",
        "банкротство через мфц",
    }
)

# Substring patterns on PROTECTED_CAMPAIGN_IDS names — guards against a prompt-
# injected rename like "BFL Bashkortostan RSYA (renamed)". Patterns kept narrow
# enough to not match arbitrary campaigns.
PROTECTED_CAMPAIGN_NAME_PATTERNS: frozenset[str] = frozenset(
    {
        "бфл башкортостан",
        "бфл татарстан",
        "бфл удмуртия",
        "24bankrotstvo",
        "24банкротство",
    }
)

# Action types that imply a specific guard family.
_ACTIONS_BUDGET_CAP = frozenset({"raise_budget", "enable_campaign", "set_bid_strategy_with_limit"})
_ACTIONS_CPC_CEILING = frozenset({"set_bid", "raise_bid", "keyword_add_with_bid"})
_ACTIONS_NEG_KW_FLOOR = frozenset(
    {"remove_keyword", "pause_keyword", "pause_campaign", "pause_adgroup"}
)
_ACTIONS_QS_GUARD = frozenset({"raise_budget", "increase_bid"})
_ACTIONS_BUDGET_BALANCE = frozenset({"raise_budget"})
_ACTIONS_CONVERSION_INTEGRITY = frozenset(
    {
        "pause_keyword_on_conversions",
        "enable_campaign_after_conversions",
        "raise_budget_based_on_cpa",
    }
)
_ACTIONS_QUERY_DRIFT = frozenset({"add_keyword", "expand_match_type"})


# --------------------------------------------------------------------- helpers


def _current_msk_time() -> datetime:
    """Current datetime in МСК (UTC+3). Module-level indirection so tests
    can monkeypatch `kill_switches._current_msk_time` and deterministically
    pass wall-clock-sensitive checks like ``MORNING_QUIET_HOURS`` (fixes
    Task 30 audit finding on 3 flaky tests between 00:00-02:00 МСК).
    """
    return datetime.now(UTC) + timedelta(hours=3)


def _today_utc_date(now: datetime | None = None) -> str:
    dt = now if now is not None else datetime.now(UTC)
    return dt.date().isoformat()


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


async def _load_protected_keywords(pool: AsyncConnectionPool) -> frozenset[str]:
    """Read ``sda_state.protected_keywords_registry`` with seed fallback."""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM sda_state WHERE key = 'protected_keywords_registry'"
                )
                row = await cur.fetchone()
    except Exception:
        logger.warning("protected keywords load failed, using seed", exc_info=True)
        return PROTECTED_KEYWORDS_SEED
    if not row or not row[0]:
        return PROTECTED_KEYWORDS_SEED
    value = row[0]
    items = value if isinstance(value, list) else []
    extracted = {_normalise(str(i.get("keyword", ""))) for i in items if i.get("keyword")}
    return frozenset(extracted | {_normalise(k) for k in PROTECTED_KEYWORDS_SEED})


# ----------------------------------------------------------------- guard: budget


class BudgetCap:
    name: ClassVar[str] = "budget_cap"
    PROTECTED_DAILY_FLOOR_RUB: ClassVar[int] = 1500
    SURGE_MULTIPLIER: ClassVar[float] = 1.5
    MORNING_QUIET_HOURS: ClassVar[tuple[int, int]] = (0, 2)  # МСК

    async def check(self, action: Action, context: KillSwitchContext) -> KillSwitchResult:
        if action.type not in _ACTIONS_BUDGET_CAP and "campaign_id" not in action.params:
            return KillSwitchResult(True, "n/a for this action", self.name)

        now = _current_msk_time()
        if self.MORNING_QUIET_HOURS[0] <= now.hour < self.MORNING_QUIET_HOURS[1]:
            return KillSwitchResult(True, "morning quiet hours; skipping", self.name)

        campaign_id = int(action.params.get("campaign_id", 0))
        if not campaign_id:
            return KillSwitchResult(True, "no campaign_id in action; nothing to gate", self.name)

        history = (context.budget_history or {}).get(campaign_id)
        if history is None:
            history = await self._fetch_budget_history(campaign_id, context)

        today_cost = float(history.get("today_cost", 0))
        avg_7d = float(history.get("daily_avg_7d", 0))
        threshold = avg_7d * self.SURGE_MULTIPLIER

        if today_cost < self.PROTECTED_DAILY_FLOOR_RUB:
            return KillSwitchResult(
                True,
                f"today_cost={today_cost:.0f} below floor {self.PROTECTED_DAILY_FLOOR_RUB}",
                self.name,
            )
        if today_cost > threshold:
            return KillSwitchResult(
                False,
                f"today_cost={today_cost:.0f} > avg_7d({avg_7d:.0f}) × {self.SURGE_MULTIPLIER}",
                self.name,
            )
        return KillSwitchResult(True, "budget surge check passed", self.name)

    async def _fetch_budget_history(
        self, campaign_id: int, context: KillSwitchContext
    ) -> dict[str, float]:
        # Reports API is slow — deliberate fall-back when context doesn't pre-
        # populate. Guards avoid hitting Direct for every call in the hot path.
        today = _today_utc_date()
        week_ago = (datetime.now(UTC) - timedelta(days=7)).date().isoformat()
        try:
            stats = await context.direct.get_campaign_stats(campaign_id, week_ago, today)
        except Exception:
            logger.exception("budget_cap: stats fetch failed; fail-closed")
            raise
        # TSV parsing is brain wrapper's job; this guard only expects a
        # pre-summarised dict with 'today_cost' and 'daily_avg_7d'. If brain
        # did not pre-populate, we signal unknown by returning zeros → surge
        # check can't fire and the floor exemption returns allow=True.
        return {
            "today_cost": float(stats.get("today_cost", 0)),
            "daily_avg_7d": float(stats.get("daily_avg_7d", 0)),
        }


# ----------------------------------------------------------------- guard: CPC


class CPCCeiling:
    name: ClassVar[str] = "cpc_ceiling"
    P90_MULTIPLIER: ClassVar[float] = 1.3
    NEW_ADGROUP_MULTIPLIER: ClassVar[float] = 2.0

    async def check(self, action: Action, context: KillSwitchContext) -> KillSwitchResult:
        if action.type not in _ACTIONS_CPC_CEILING:
            return KillSwitchResult(True, "n/a for this action", self.name)

        new_bid = action.params.get("bid")
        if new_bid is None:
            return KillSwitchResult(True, "no bid in action", self.name)
        new_bid_rub = int(new_bid)

        ad_group_id = action.params.get("ad_group_id")
        if ad_group_id is None:
            return KillSwitchResult(True, "no ad_group_id; cannot compute ceiling", self.name)

        history = (context.bid_history_by_adgroup or {}).get(int(ad_group_id)) or []
        if len(history) < 5:
            fallback = action.params.get("effective_bid")
            if fallback is None:
                return KillSwitchResult(
                    True, "no bid history and no effective_bid fallback", self.name
                )
            ceiling = int(int(fallback) * self.NEW_ADGROUP_MULTIPLIER)
            if new_bid_rub > ceiling:
                return KillSwitchResult(
                    False,
                    (
                        f"new_bid={new_bid_rub} > effective_bid({int(fallback)}) "
                        f"× {self.NEW_ADGROUP_MULTIPLIER}"
                    ),
                    self.name,
                )
            return KillSwitchResult(True, "within fallback ceiling", self.name)

        p90 = _percentile(history, 0.9)
        ceiling = int(p90 * self.P90_MULTIPLIER)
        if new_bid_rub > ceiling:
            return KillSwitchResult(
                False,
                f"new_bid={new_bid_rub} > p90_30d({int(p90)}) × {self.P90_MULTIPLIER}",
                self.name,
            )
        return KillSwitchResult(True, "bid below ceiling", self.name)


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    try:
        # 10 buckets → index 8 = p90. statistics.quantiles gives 9 cut-points
        # between 10 buckets for n=10.
        cuts = statistics.quantiles(values, n=10)
        if q >= 0.9:
            return float(cuts[8])
        if q <= 0.1:
            return float(cuts[0])
        # round to nearest decile for other q
        idx = max(0, min(8, round(q * 10) - 1))
        return float(cuts[idx])
    except statistics.StatisticsError:
        return float(sorted(values)[-1])


# --------------------------------------------------------------- guard: NegKWFloor


class NegKWFloor:
    name: ClassVar[str] = "neg_kw_floor"

    async def check(self, action: Action, context: KillSwitchContext) -> KillSwitchResult:
        if action.type not in _ACTIONS_NEG_KW_FLOOR and action.type != "add_neg_keyword":
            return KillSwitchResult(True, "n/a for this action", self.name)

        # PROTECTED campaigns: block by id OR by name substring.
        protected_ids = set(context.settings.PROTECTED_CAMPAIGN_IDS)
        campaign_id = action.params.get("campaign_id")
        if campaign_id is not None and int(campaign_id) in protected_ids:
            return KillSwitchResult(
                False, f"campaign {campaign_id} is PROTECTED (by id)", self.name
            )

        name_raw = action.params.get("campaign_name")
        if isinstance(name_raw, str):
            normalised = _normalise(name_raw)
            for pattern in PROTECTED_CAMPAIGN_NAME_PATTERNS:
                if pattern in normalised:
                    return KillSwitchResult(
                        False,
                        f"campaign_name='{name_raw}' matches PROTECTED pattern '{pattern}'",
                        self.name,
                    )

        if action.type not in {"remove_keyword", "pause_keyword", "add_neg_keyword"}:
            return KillSwitchResult(True, "not a keyword mutation", self.name)

        # protected keyword registry check — apply to remove/pause/add_neg.
        keyword = action.params.get("keyword") or action.params.get("phrase")
        if not isinstance(keyword, str):
            return KillSwitchResult(True, "no keyword in action", self.name)

        registry = await _load_protected_keywords(context.pool)
        normalised_kw = _normalise(keyword)
        for protected in registry:
            if protected == normalised_kw or protected in normalised_kw:
                return KillSwitchResult(
                    False,
                    f"keyword='{keyword}' matches protected registry entry '{protected}'",
                    self.name,
                )
        return KillSwitchResult(True, "keyword not in protected registry", self.name)


# ------------------------------------------------------------------- guard: QS


class QSGuard:
    name: ClassVar[str] = "qs_guard"
    PRODUCTIVITY_FLOOR: ClassVar[int] = 6

    async def check(self, action: Action, context: KillSwitchContext) -> KillSwitchResult:
        if action.type not in _ACTIONS_QS_GUARD:
            return KillSwitchResult(True, "n/a for this action", self.name)
        ad_group_id = action.params.get("ad_group_id")
        if ad_group_id is None:
            return KillSwitchResult(True, "no ad_group_id; nothing to gate", self.name)

        productivity = (context.adgroup_productivity or {}).get(int(ad_group_id))
        if productivity is None:
            try:
                groups = await context.direct.get_adgroups(ids=[int(ad_group_id)])
            except Exception:
                logger.exception("qs_guard: adgroup lookup failed; fail-closed")
                raise
            if not groups:
                return KillSwitchResult(True, "adgroup not found (new?)", self.name)
            productivity = int(groups[0].get("Productivity", 10))
        if productivity < self.PRODUCTIVITY_FLOOR:
            return KillSwitchResult(
                False,
                f"QS={productivity} below floor {self.PRODUCTIVITY_FLOOR}; "
                "optimize ads/kws before scaling",
                self.name,
            )
        return KillSwitchResult(True, f"QS={productivity} above floor", self.name)


# ---------------------------------------------------------- guard: BudgetBalance


class BudgetBalance:
    name: ClassVar[str] = "budget_balance"
    WEEKLY_CAP_FRACTION: ClassVar[float] = 0.20

    async def check(self, action: Action, context: KillSwitchContext) -> KillSwitchResult:
        if action.type not in _ACTIONS_BUDGET_BALANCE:
            return KillSwitchResult(True, "n/a for this action", self.name)

        delta = action.params.get("delta_rub", 0)
        weekly_total = context.weekly_budget_total_rub
        if weekly_total is None:
            weekly_total = await self._load_weekly_total(context.pool, context.settings)
        if weekly_total <= 0:
            return KillSwitchResult(True, "weekly total unknown; skipping", self.name)

        sum_delta = await self._sum_weekly_raise_budget(context.pool)
        projected = abs(sum_delta) + abs(int(delta))
        fraction = projected / weekly_total
        if fraction > self.WEEKLY_CAP_FRACTION:
            return KillSwitchResult(
                False,
                f"weekly budget redistribution {fraction:.0%} exceeds "
                f"{self.WEEKLY_CAP_FRACTION:.0%} cap "
                f"(sum_abs_delta={abs(sum_delta)} + new={int(delta)} vs {weekly_total})",
                self.name,
            )
        return KillSwitchResult(True, f"weekly redistribution {fraction:.0%} within cap", self.name)

    async def _load_weekly_total(self, pool: AsyncConnectionPool, settings: Settings) -> int:
        # Heuristic: total active-campaigns weekly budget = sum of caps from
        # sda_state. Brain pre-populates for real runs; this fallback returns
        # 0 so guard no-ops when data is absent.
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT value FROM sda_state WHERE key = 'weekly_budget_total_rub'"
                    )
                    row = await cur.fetchone()
        except Exception:
            logger.exception("budget_balance: weekly_total fetch failed; fail-closed")
            raise
        if row and row[0]:
            try:
                return int(row[0])
            except (TypeError, ValueError):
                return 0
        return 0

    async def _sum_weekly_raise_budget(self, pool: AsyncConnectionPool) -> int:
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT COALESCE(SUM((action->>'delta_rub')::int), 0) "
                        "FROM hypotheses, jsonb_array_elements(actions) AS action "
                        "WHERE created_at >= now() - interval '7 days' "
                        "AND state IN ('running', 'confirmed') "
                        "AND action->>'type' = 'raise_budget'"
                    )
                    row = await cur.fetchone()
        except Exception:
            logger.exception("budget_balance: weekly sum fetch failed; fail-closed")
            raise
        return int(row[0]) if row and row[0] else 0


# ------------------------------------------------------- guard: ConversionIntegrity


class ConversionIntegrity:
    name: ClassVar[str] = "conversion_integrity"
    DUPLICATE_WINDOW_MIN: ClassVar[int] = 15
    BOT_SHARE_CEILING: ClassVar[float] = 0.7

    async def check(self, action: Action, context: KillSwitchContext) -> KillSwitchResult:
        if action.type not in _ACTIONS_CONVERSION_INTEGRITY:
            return KillSwitchResult(True, "n/a for this action", self.name)

        if context.bitrix is None or context.metrika is None:
            return KillSwitchResult(
                True, "bitrix/metrika clients not wired yet; skipping", self.name
            )

        try:
            recent_leads = await context.bitrix.recent_leads(hours=24)
            visits = await context.metrika.recent_visits(hours=24)
        except Exception:
            logger.exception("conversion_integrity: client call failed; fail-closed")
            raise

        dup = self._find_duplicate_visitor(recent_leads)
        if dup is not None:
            visitor_id, minutes = dup
            return KillSwitchResult(
                False,
                f"duplicate visitor_id={visitor_id} within {minutes}min window",
                self.name,
            )

        total = len(visits) or 1
        bots = sum(1 for v in visits if v.get("is_robot"))
        share = bots / total
        if share > self.BOT_SHARE_CEILING:
            return KillSwitchResult(
                False, f"bot_share={share:.0%} over ceiling {self.BOT_SHARE_CEILING:.0%}", self.name
            )
        return KillSwitchResult(True, f"no duplicate visitors; bot_share={share:.0%}", self.name)

    def _find_duplicate_visitor(self, leads: list[dict[str, Any]]) -> tuple[str, int] | None:
        by_visitor: dict[str, list[datetime]] = {}
        for lead in leads:
            visitor = lead.get("visitor_id") or lead.get("ym_visitor_id")
            ts = lead.get("created_at")
            if not isinstance(visitor, str) or ts is None:
                continue
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
            by_visitor.setdefault(visitor, []).append(ts)
        window = timedelta(minutes=self.DUPLICATE_WINDOW_MIN)
        for visitor, stamps in by_visitor.items():
            if len(stamps) < 2:
                continue
            stamps.sort()
            for prev, nxt in zip(stamps, stamps[1:], strict=False):
                diff = nxt - prev
                if diff <= window:
                    return visitor, int(diff.total_seconds() // 60) or 1
        return None


# -------------------------------------------------------------- guard: QueryDrift


class QueryDrift:
    name: ClassVar[str] = "query_drift"
    JACCARD_FLOOR: ClassVar[float] = 0.5
    MIN_SET_SIZE: ClassVar[int] = 5

    async def check(self, action: Action, context: KillSwitchContext) -> KillSwitchResult:
        if action.type not in _ACTIONS_QUERY_DRIFT:
            return KillSwitchResult(True, "n/a for this action", self.name)

        baseline = context.baseline_queries
        if baseline is None:
            return KillSwitchResult(True, "no baseline queries; first-time adgroup", self.name)
        current = context.recent_queries or []

        baseline_norm = {_normalise(q) for q in baseline if q}
        current_norm = {_normalise(q) for q in current if q}
        if len(baseline_norm) < self.MIN_SET_SIZE or len(current_norm) < self.MIN_SET_SIZE:
            return KillSwitchResult(
                True,
                f"small sets (baseline={len(baseline_norm)}, current={len(current_norm)})",
                self.name,
            )

        score = _jaccard(baseline_norm, current_norm)
        if score < self.JACCARD_FLOOR:
            return KillSwitchResult(
                False,
                f"query drift: jaccard={score:.2f} < {self.JACCARD_FLOOR}; "
                "analyze trajectories before expansion",
                self.name,
            )
        return KillSwitchResult(True, f"jaccard={score:.2f} above floor", self.name)


# -------------------------------------------------------------------- run_all


ALL_GUARDS: tuple[type[BaseKillSwitch], ...] = (
    BudgetCap,
    CPCCeiling,
    NegKWFloor,
    QSGuard,
    BudgetBalance,
    ConversionIntegrity,
    QueryDrift,
)


async def run_all(action: Action, context: KillSwitchContext) -> list[KillSwitchResult]:
    """Fire every guard in parallel; never raise — fail-closed on any error.

    The brain wrapper (Task 12) receives the full list and decides on the
    aggregate: ``any(not r.allow)`` → reject the mutation.
    """
    guards = [cls() for cls in ALL_GUARDS]
    coros = [_safe_check(guard, action, context) for guard in guards]
    return list(await asyncio.gather(*coros))


async def _safe_check(
    guard: BaseKillSwitch, action: Action, context: KillSwitchContext
) -> KillSwitchResult:
    try:
        return await guard.check(action, context)
    except Exception as exc:  # noqa: BLE001 - fail-closed on ANY failure
        logger.exception("kill-switch '%s' raised; failing closed", guard.name)
        return KillSwitchResult(
            allow=False,
            reason=f"guard {guard.name} exception: {type(exc).__name__}: {exc}",
            switch_name=guard.name,
        )


__all__ = [
    "ALL_GUARDS",
    "Action",
    "BaseKillSwitch",
    "BudgetBalance",
    "BudgetCap",
    "CPCCeiling",
    "ConversionIntegrity",
    "KillSwitchContext",
    "KillSwitchResult",
    "NegKWFloor",
    "PROTECTED_CAMPAIGN_NAME_PATTERNS",
    "PROTECTED_KEYWORDS_SEED",
    "QSGuard",
    "QueryDrift",
    "run_all",
]
