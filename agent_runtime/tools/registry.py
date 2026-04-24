"""ToolRegistry builder — one immutable registry per process (Task 12).

Wraps the pure-function tools from Tasks 7/8 and the kill-switches from
Task 9 into ``agents_core.tools.registry.Tool`` instances with correct
``tier`` values. The brain wrapper in :mod:`agent_runtime.brain` consumes
the result via ``registry.filter(tiers=...)`` — the shadow/assisted trust
level gets only ``read`` + ``write`` tools; ``danger`` (kill-switches) is
reserved for the autonomous path (where in practice guards are applied
via ``run_all`` rather than called directly by Claude).

The registry is built once in ``agent_runtime.main`` lifespan and stored in
``app.state.registry``. ``build_registry`` deliberately takes its
dependencies as parameters (DirectAPI instance, httpx client) instead of
reading global state — keeps it testable and lifecycle-explicit.
"""

from __future__ import annotations

from typing import Any

from agents_core.tools.registry import Tool, ToolRegistry

from agent_runtime.config import Settings
from agent_runtime.tools import bitrix as bitrix_tools
from agent_runtime.tools import metrika as metrika_tools
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI
from agent_runtime.tools.kill_switches import ALL_GUARDS

_INT_ARRAY = {"type": "array", "items": {"type": "integer"}}
_STR_ARRAY = {"type": "array", "items": {"type": "string"}}


def build_registry(
    settings: Settings,
    *,
    direct: DirectAPI,
    http_client: Any,
) -> ToolRegistry:
    """Construct the single-process tool registry.

    ``direct`` — active :class:`~agent_runtime.tools.direct_api.DirectAPI`
    instance (created in ``main.py`` lifespan, used across requests).
    ``http_client`` — the shared ``httpx.AsyncClient`` used by bitrix/
    metrika/telegram helpers.
    """
    registry = ToolRegistry()

    # -------------------------------- Direct API read
    async def direct_get_campaigns(ids: list[int]) -> list[dict[str, Any]]:
        return await direct.get_campaigns(ids)

    registry.register(
        Tool(
            name="direct.get_campaigns",
            description="Read Yandex Direct campaigns by ids.",
            input_schema={
                "type": "object",
                "properties": {"ids": _INT_ARRAY},
                "required": ["ids"],
            },
            handler=direct_get_campaigns,
            tier="read",
            tags=("direct",),
        )
    )

    async def direct_get_adgroups(
        campaign_id: int | None = None,
        ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        return await direct.get_adgroups(campaign_id=campaign_id, ids=ids)

    registry.register(
        Tool(
            name="direct.get_adgroups",
            description="Read ad groups by campaign or by explicit ids.",
            input_schema={
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "ids": _INT_ARRAY,
                },
            },
            handler=direct_get_adgroups,
            tier="read",
            tags=("direct",),
        )
    )

    async def direct_get_keywords(ad_group_ids: list[int]) -> list[dict[str, Any]]:
        return await direct.get_keywords(ad_group_ids)

    registry.register(
        Tool(
            name="direct.get_keywords",
            description="Read keywords for the given ad group ids.",
            input_schema={
                "type": "object",
                "properties": {"ad_group_ids": _INT_ARRAY},
                "required": ["ad_group_ids"],
            },
            handler=direct_get_keywords,
            tier="read",
            tags=("direct",),
        )
    )

    # -------------------------------- Direct API write (requires_verify)
    async def direct_set_bid(
        keyword_id: int,
        bid_rub: int,
        context_bid_rub: int | None = None,
    ) -> dict[str, Any]:
        return await direct.set_bid(keyword_id, bid_rub=bid_rub, context_bid_rub=context_bid_rub)

    registry.register(
        Tool(
            name="direct.set_bid",
            description=(
                "Set search (and optional network) bid on a keyword. PROTECTED "
                "campaigns are runtime-blocked. Caller must invoke verify_bid."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "keyword_id": {"type": "integer"},
                    "bid_rub": {"type": "integer", "minimum": 0},
                    "context_bid_rub": {"type": "integer", "minimum": 0},
                },
                "required": ["keyword_id", "bid_rub"],
            },
            handler=direct_set_bid,
            tier="write",
            requires_verify=True,
            idempotent=False,
            tags=("direct",),
        )
    )

    async def direct_add_negatives(campaign_id: int, phrases: list[str]) -> dict[str, Any]:
        return await direct.add_negatives(campaign_id, phrases)

    registry.register(
        Tool(
            name="direct.add_negatives",
            description="Append negative keywords to a campaign (set-union).",
            input_schema={
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "phrases": _STR_ARRAY,
                },
                "required": ["campaign_id", "phrases"],
            },
            handler=direct_add_negatives,
            tier="write",
            requires_verify=True,
            tags=("direct",),
        )
    )

    async def direct_pause_group(ad_group_id: int) -> dict[str, Any]:
        return await direct.pause_group(ad_group_id)

    registry.register(
        Tool(
            name="direct.pause_group",
            description="Suspend an ad group. PROTECTED campaigns are runtime-blocked.",
            input_schema={
                "type": "object",
                "properties": {"ad_group_id": {"type": "integer"}},
                "required": ["ad_group_id"],
            },
            handler=direct_pause_group,
            tier="write",
            requires_verify=True,
            idempotent=False,
            tags=("direct",),
        )
    )

    async def direct_resume_group(ad_group_id: int) -> dict[str, Any]:
        return await direct.resume_group(ad_group_id)

    registry.register(
        Tool(
            name="direct.resume_group",
            description="Resume a paused ad group. PROTECTED campaigns blocked.",
            input_schema={
                "type": "object",
                "properties": {"ad_group_id": {"type": "integer"}},
                "required": ["ad_group_id"],
            },
            handler=direct_resume_group,
            tier="write",
            requires_verify=True,
            idempotent=False,
            tags=("direct",),
        )
    )

    async def direct_pause_campaign(campaign_id: int) -> dict[str, Any]:
        return await direct.pause_campaign(campaign_id)

    registry.register(
        Tool(
            name="direct.pause_campaign",
            description="Suspend a whole campaign. PROTECTED campaigns blocked.",
            input_schema={
                "type": "object",
                "properties": {"campaign_id": {"type": "integer"}},
                "required": ["campaign_id"],
            },
            handler=direct_pause_campaign,
            tier="write",
            requires_verify=True,
            idempotent=False,
            tags=("direct",),
        )
    )

    async def direct_resume_campaign(campaign_id: int) -> dict[str, Any]:
        return await direct.resume_campaign(campaign_id)

    registry.register(
        Tool(
            name="direct.resume_campaign",
            description="Resume a paused campaign. Rejects DRAFT campaigns.",
            input_schema={
                "type": "object",
                "properties": {"campaign_id": {"type": "integer"}},
                "required": ["campaign_id"],
            },
            handler=direct_resume_campaign,
            tier="write",
            requires_verify=True,
            idempotent=False,
            tags=("direct",),
        )
    )

    # -------------------------------- Bitrix read
    async def bitrix_get_lead_list(
        filter: dict[str, Any] | None = None,
        select: list[str] | None = None,
        max_total: int = 100,
    ) -> list[dict[str, Any]]:
        return await bitrix_tools.get_lead_list(
            http_client, settings, filter=filter, select=select, max_total=max_total
        )

    registry.register(
        Tool(
            name="bitrix.get_lead_list",
            description="Paginated crm.lead.list. Results are raw — contains PII until sanitised.",
            input_schema={
                "type": "object",
                "properties": {
                    "filter": {"type": "object"},
                    "select": _STR_ARRAY,
                    "max_total": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
            },
            handler=bitrix_get_lead_list,
            tier="read",
            tags=("bitrix",),
        )
    )

    async def bitrix_get_deal_list(
        filter: dict[str, Any] | None = None,
        max_total: int = 100,
    ) -> list[dict[str, Any]]:
        return await bitrix_tools.get_deal_list(
            http_client, settings, filter=filter, max_total=max_total
        )

    registry.register(
        Tool(
            name="bitrix.get_deal_list",
            description="Paginated crm.deal.list. Common filter: STAGE_ID='C45:WON'.",
            input_schema={
                "type": "object",
                "properties": {
                    "filter": {"type": "object"},
                    "max_total": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
            },
            handler=bitrix_get_deal_list,
            tier="read",
            tags=("bitrix",),
        )
    )

    # -------------------------------- Metrika read
    async def metrika_get_bounce(date1: str, date2: str) -> dict[int, float]:
        return await metrika_tools.get_bounce_by_campaign(
            http_client, settings, date1=date1, date2=date2
        )

    registry.register(
        Tool(
            name="metrika.get_bounce_by_campaign",
            description="Bounce rate by Direct campaign id over [date1, date2].",
            input_schema={
                "type": "object",
                "properties": {
                    "date1": {"type": "string"},
                    "date2": {"type": "string"},
                },
                "required": ["date1", "date2"],
            },
            handler=metrika_get_bounce,
            tier="read",
            tags=("metrika",),
        )
    )

    async def metrika_get_conversions(
        goal_ids: list[int], date1: str, date2: str
    ) -> dict[int, int]:
        return await metrika_tools.get_conversions(
            http_client, settings, goal_ids=goal_ids, date1=date1, date2=date2
        )

    registry.register(
        Tool(
            name="metrika.get_conversions",
            description="Goal reach counts per goal_id over [date1, date2].",
            input_schema={
                "type": "object",
                "properties": {
                    "goal_ids": _INT_ARRAY,
                    "date1": {"type": "string"},
                    "date2": {"type": "string"},
                },
                "required": ["goal_ids", "date1", "date2"],
            },
            handler=metrika_get_conversions,
            tier="read",
            tags=("metrika",),
        )
    )

    # -------------------------------- Telegram write (notify tag)
    async def telegram_send_message(text: str) -> int:
        return await telegram_tools.send_message(http_client, settings, text=text)

    registry.register(
        Tool(
            name="telegram.send_message",
            description="Post a plain message to the owner chat. Text must not contain PII.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 4000}},
                "required": ["text"],
            },
            handler=telegram_send_message,
            tier="write",
            tags=("telegram", "notify"),
        )
    )

    # -------------------------------- Kill-switches (danger tier, introspective)
    # Registering guards here is intentionally symbolic: the LLM never calls
    # them directly — the brain runs ``run_all`` after proposing an action.
    # We put them in the registry so filter(tiers=["danger"]) surfaces them
    # for observability / documentation.
    for guard_cls in ALL_GUARDS:

        def _make_handler(name: str):
            async def _not_directly_callable(**_: Any) -> dict[str, str]:
                return {
                    "guard": name,
                    "message": (
                        "kill-switches are applied via brain.run_all, not via "
                        "direct tool_use; this handler is metadata-only"
                    ),
                }

            return _not_directly_callable

        registry.register(
            Tool(
                name=f"killswitch.{guard_cls.name}",
                description=f"{guard_cls.__name__} guard (metadata-only; applied via run_all).",
                input_schema={"type": "object", "properties": {}},
                handler=_make_handler(guard_cls.name),
                tier="danger",
                tags=("killswitch",),
            )
        )

    return registry


__all__ = ["build_registry"]
