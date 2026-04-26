"""Settings reconciler — diff live Direct cabinet vs ``OWN_CAMPAIGNS_REFERENCE``.

Daily 12:00 UTC (15:00 МСК) cron. Reads:
  * ``bidmodifiers.get`` with ``Levels=["CAMPAIGN","AD_GROUP"]`` —
    without the explicit ``Levels`` parameter the API returns an empty
    array (see ``feedback_bidmodifiers_get_levels.md``).
  * ``campaigns.get`` for ``TimeTargeting``.
  * ``ads.get`` filtered by ``Status=DRAFT``.

For each ``OWN_CAMPAIGNS_REFERENCE`` entry produces a per-campaign diff:
  * ``missing_demo``  — demographic mods that should exist but don't
  * ``extra_demo``    — demographic mods that exist but shouldn't (or
                         have wrong bid_modifier)
  * ``missing_retg``  — retargeting mods that should exist but don't
  * ``extra_retg``    — retargeting mods that exist but shouldn't
  * ``schedule_drift`` — TimeTargeting items differ from reference
  * ``draft_ads``     — any ad with Status=DRAFT (per reference, max=0)

Telegram digest summarises drift. Read-only — Direct mutations are NOT
applied here. Auto-fix lives in ``settings_reconciler_actuator`` (out of
scope for v0; v0 just observes and alerts).

Degraded-noop pattern: when called without DI clients (legacy dispatch),
returns ``{"status":"ok","action":"degraded_noop"}`` like other jobs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from agent_runtime.knowledge.own_campaigns_reference import (
    OWN_CAMPAIGNS_REFERENCE,
    CampaignReference,
    DemoMod,
    RetgMod,
)
from agent_runtime.tools import telegram as telegram_tools

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from agent_runtime.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CampaignDiff:
    """Per-campaign drift report."""

    cid: int
    persona: str
    missing_demo: tuple[DemoMod, ...] = field(default=())
    extra_demo: tuple[str, ...] = field(default=())
    missing_retg: tuple[RetgMod, ...] = field(default=())
    extra_retg: tuple[str, ...] = field(default=())
    schedule_drift: bool = False
    draft_ads_count: int = 0

    @property
    def is_clean(self) -> bool:
        return (
            not self.missing_demo
            and not self.extra_demo
            and not self.missing_retg
            and not self.extra_retg
            and not self.schedule_drift
            and self.draft_ads_count == 0
        )

    @property
    def total_issues(self) -> int:
        return (
            len(self.missing_demo)
            + len(self.extra_demo)
            + len(self.missing_retg)
            + len(self.extra_retg)
            + (1 if self.schedule_drift else 0)
            + (1 if self.draft_ads_count > 0 else 0)
        )


def _demo_key(age: str, gender: str | None) -> str:
    """Stable key for matching DemoMod between live and reference."""
    return f"{age}|{gender or '-'}"


def _retg_key(rid: int) -> str:
    return str(rid)


def diff_campaign(
    *,
    ref: CampaignReference,
    live_demo: list[dict[str, Any]],
    live_retg: list[dict[str, Any]],
    live_schedule: tuple[str, ...] | None,
    live_draft_count: int,
) -> CampaignDiff:
    """Pure diff — no I/O. Easy to unit-test.

    ``live_demo`` items: ``{"Age": str, "Gender": str|None, "BidModifier": int}``.
    ``live_retg`` items: ``{"RetargetingConditionId": int, "BidModifier": int}``.
    ``live_schedule``: tuple of 7 day-strings or None (no schedule set).
    """
    ref_demo_by_key: dict[str, DemoMod] = {_demo_key(d.age, d.gender): d for d in ref.demographics}
    ref_retg_by_key: dict[str, RetgMod] = {
        _retg_key(r.retargeting_condition_id): r for r in ref.retargeting
    }

    live_demo_by_key: dict[str, dict[str, Any]] = {
        _demo_key(d.get("Age", ""), d.get("Gender")): d for d in live_demo
    }
    live_retg_by_key: dict[str, dict[str, Any]] = {
        _retg_key(int(d.get("RetargetingConditionId", 0))): d for d in live_retg
    }

    missing_demo: list[DemoMod] = []
    for k, expected in ref_demo_by_key.items():
        live = live_demo_by_key.get(k)
        if live is None or int(live.get("BidModifier", -1)) != expected.bid_modifier:
            missing_demo.append(expected)

    extra_demo: list[str] = []
    for k, live in live_demo_by_key.items():
        if k not in ref_demo_by_key:
            extra_demo.append(f"{k}={live.get('BidModifier')}%")

    missing_retg: list[RetgMod] = []
    for k, expected in ref_retg_by_key.items():
        live = live_retg_by_key.get(k)
        if live is None or int(live.get("BidModifier", -1)) != expected.bid_modifier:
            missing_retg.append(expected)

    extra_retg: list[str] = []
    for k, live in live_retg_by_key.items():
        if k not in ref_retg_by_key:
            extra_retg.append(f"{k}={live.get('BidModifier')}%")

    schedule_drift = False
    if live_schedule is not None:
        schedule_drift = tuple(live_schedule) != tuple(ref.schedule_items)
    else:
        schedule_drift = True  # no schedule at all = drift

    return CampaignDiff(
        cid=ref.cid,
        persona=ref.persona,
        missing_demo=tuple(missing_demo),
        extra_demo=tuple(extra_demo),
        missing_retg=tuple(missing_retg),
        extra_retg=tuple(extra_retg),
        schedule_drift=schedule_drift,
        draft_ads_count=live_draft_count,
    )


def format_telegram(diffs: list[CampaignDiff]) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    clean = [d for d in diffs if d.is_clean]
    dirty = [d for d in diffs if not d.is_clean]
    head = f"<b>🪞 Settings Reconciler</b> — {when}"
    if not dirty:
        return f"{head}\n\nВсе {len(diffs)} кампаний соответствуют reference."

    lines = [head, ""]
    if clean:
        names = ", ".join(d.persona for d in clean)
        lines.append(f"✅ Чисто ({len(clean)}): {names}")
        lines.append("")
    lines.append(f"⚠️ Drift в {len(dirty)} кампаниях ({sum(d.total_issues for d in dirty)} issues):")
    for d in dirty:
        lines.append("")
        lines.append(f"<b>{d.persona}</b> (cid={d.cid}): {d.total_issues} issues")
        if d.missing_demo:
            for m in d.missing_demo:
                gender = f"+{m.gender}" if m.gender else ""
                lines.append(f"  • missing demo: {m.age}{gender} = {m.bid_modifier}%")
        if d.extra_demo:
            for e in d.extra_demo:
                lines.append(f"  • extra demo: {e}")
        if d.missing_retg:
            for m in d.missing_retg:
                lines.append(f"  • missing retg: {m.retargeting_condition_id} = {m.bid_modifier}%")
        if d.extra_retg:
            for e in d.extra_retg:
                lines.append(f"  • extra retg: {e}")
        if d.schedule_drift:
            lines.append("  • schedule drift (TimeTargeting != reference)")
        if d.draft_ads_count > 0:
            lines.append(f"  • {d.draft_ads_count} ads in DRAFT (must be moderated)")
    return "\n".join(lines)


async def _fetch_live_state(
    *, http_client: httpx.AsyncClient, settings: Settings, cids: list[int]
) -> dict[str, Any]:
    """Fetch bidmodifiers + TimeTargeting + DRAFT ads in one batch.

    Uses the shared httpx client + settings.YANDEX_DIRECT_TOKEN directly —
    no DirectAPI public method covers bidmodifiers listing or DRAFT-ad
    filtering yet, so we hit the v5 endpoints inline. This keeps
    settings_reconciler decoupled from DirectAPI internals.
    """
    base = settings.YANDEX_DIRECT_BASE_URL.rstrip("/")
    token = settings.YANDEX_DIRECT_TOKEN.get_secret_value()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Language": "ru",
        "Content-Type": "application/json",
    }

    # bidmodifiers — Levels parameter is REQUIRED to see Campaign-level mods
    bm_resp = await http_client.post(
        f"{base}/bidmodifiers",
        headers=headers,
        json={
            "method": "get",
            "params": {
                "SelectionCriteria": {"CampaignIds": cids, "Levels": ["CAMPAIGN", "AD_GROUP"]},
                "FieldNames": ["Id", "CampaignId", "Type"],
                "DemographicsAdjustmentFieldNames": ["Age", "Gender", "BidModifier"],
                "RetargetingAdjustmentFieldNames": ["RetargetingConditionId", "BidModifier"],
            },
        },
    )
    bm = bm_resp.json().get("result", {}).get("BidModifiers", []) or []

    # TimeTargeting via campaigns.get
    camps_resp = await http_client.post(
        f"{base}/campaigns",
        headers=headers,
        json={
            "method": "get",
            "params": {
                "SelectionCriteria": {"Ids": cids},
                "FieldNames": ["Id", "TimeTargeting"],
            },
        },
    )
    camps = camps_resp.json().get("result", {}).get("Campaigns", []) or []

    # DRAFT ads
    ads_resp = await http_client.post(
        f"{base}/ads",
        headers=headers,
        json={
            "method": "get",
            "params": {
                "SelectionCriteria": {"CampaignIds": cids, "Statuses": ["DRAFT"]},
                "FieldNames": ["Id", "CampaignId", "Status"],
            },
        },
    )
    ads = ads_resp.json().get("result", {}).get("Ads", []) or []
    return {"bidmodifiers": bm, "campaigns": camps, "draft_ads": ads}


def _build_diffs(live: dict[str, Any]) -> list[CampaignDiff]:
    """Group live data by cid and diff against each reference."""
    bm_by_cid: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for m in live.get("bidmodifiers", []):
        cid = int(m.get("CampaignId", 0))
        bucket = bm_by_cid.setdefault(cid, {"demo": [], "retg": []})
        t = m.get("Type")
        if t == "DEMOGRAPHICS_ADJUSTMENT" and m.get("DemographicsAdjustment"):
            bucket["demo"].append(m["DemographicsAdjustment"])
        elif t == "RETARGETING_ADJUSTMENT" and m.get("RetargetingAdjustment"):
            bucket["retg"].append(m["RetargetingAdjustment"])

    sched_by_cid: dict[int, tuple[str, ...] | None] = {}
    for c in live.get("campaigns", []):
        cid = int(c.get("Id", 0))
        tt = c.get("TimeTargeting") or {}
        sched = tt.get("Schedule") or {}
        items = tuple(sched.get("Items") or ()) if isinstance(sched, dict) else None
        sched_by_cid[cid] = items if items else None

    drafts_by_cid: dict[int, int] = {}
    for a in live.get("draft_ads", []):
        cid = int(a.get("CampaignId", 0))
        drafts_by_cid[cid] = drafts_by_cid.get(cid, 0) + 1

    diffs: list[CampaignDiff] = []
    for cid, ref in OWN_CAMPAIGNS_REFERENCE.items():
        bucket = bm_by_cid.get(cid, {"demo": [], "retg": []})
        diffs.append(
            diff_campaign(
                ref=ref,
                live_demo=bucket["demo"],
                live_retg=bucket["retg"],
                live_schedule=sched_by_cid.get(cid),
                live_draft_count=drafts_by_cid.get(cid, 0),
            )
        )
    return diffs


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron daily 12:00 UTC."""
    if http_client is None or settings is None:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "campaigns_checked": 0,
            "issues_total": 0,
            "dry_run": dry_run,
        }

    cids = list(OWN_CAMPAIGNS_REFERENCE.keys())
    try:
        live = await _fetch_live_state(http_client=http_client, settings=settings, cids=cids)
    except Exception:
        logger.exception("settings_reconciler: fetch_live_state failed")
        return {
            "status": "error",
            "action": "fetch_failed",
            "campaigns_checked": 0,
            "issues_total": 0,
            "dry_run": dry_run,
        }

    diffs = _build_diffs(live)
    total_issues = sum(d.total_issues for d in diffs)
    dirty_count = sum(1 for d in diffs if not d.is_clean)
    msg = format_telegram(diffs)

    if not dry_run and dirty_count > 0:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("settings_reconciler: telegram send failed")

    return {
        "status": "ok",
        "action": "observed",
        "campaigns_checked": len(diffs),
        "dirty_count": dirty_count,
        "issues_total": total_issues,
        "dry_run": dry_run,
    }


__all__ = [
    "CampaignDiff",
    "_build_diffs",
    "diff_campaign",
    "format_telegram",
    "run",
]
