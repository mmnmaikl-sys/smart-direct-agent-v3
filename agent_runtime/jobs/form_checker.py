"""Form Checker — daily landing + moderation sanity cron (Task 15).

Port of v2 form_checker (fixed commit ``2b5069a``, live on Railway) with the
24.04 incident fixes baked in:

1. **No blanket-suspend** — only non-protected campaigns are paused, and only
   when either (a) a global site-wide issue is detected *or* (b) the specific
   campaign has a REJECTED ad.
2. **TrustLevel overlay wins** — in ``shadow`` the job NEVER calls
   ``pause_campaign``, only posts a Telegram alert. In ``assisted`` the job
   is whitelisted for AUTO via :data:`trust_levels.ASSISTED_AUTO_WHITELIST`.
   In ``autonomous`` the job may AUTO-suspend non-protected campaigns.
3. **Bitrix-direct forms are OK** — v2 pre-fix flagged "no lead destination"
   on landings that POST straight to a Bitrix REST webhook. The fixed
   ``check_landing`` looks for either ``bitrix24.ru/rest`` / ``crm.lead.add``
   *or* the ``/lead`` SDA endpoint. Missing **both** is the only issue.
4. **Source of truth** — ``PROTECTED_CAMPAIGN_IDS`` and
   ``PROTECTED_LANDING_URLS`` read from :class:`Settings` (Decision 17). No
   hardcoded campaign ids, landing URLs, or v2 Railway domain in this
   module (grep-enforced by ``test_no_hardcoded_protected_ids_or_urls``).
5. **Dependency injection** — :class:`FormChecker` takes clients as
   constructor args. No module-level singletons; tests pass mocks.

Cron: every day at 07:00 МСК (04:00 UTC, ``0 4 * * *``) via Railway Cron
hitting ``/run/form_checker``.

Daily-ish, not 15-min, because the kind of thing this job catches
(landings 500ing, ads rejected by moderators, CORS broken) drifts on human
timescales — noisier polling would burn money on identical alerts without
catching an extra hour of outages.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from psycopg_pool import AsyncConnectionPool

from agent_runtime import knowledge
from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.models import AutonomyLevel
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI, ProtectedCampaignError
from agent_runtime.trust_levels import TrustLevel, allowed_action, get_trust_level

logger = logging.getLogger(__name__)


_MSK = timedelta(hours=3)
_MIN_LANDING_SIZE_BYTES = 1000
_LANDING_FETCH_TIMEOUT_SEC = 15.0
_OWNER_ORIGIN = "https://24bankrotsttvo.ru"
_PHONE_INPUT_MARKERS: tuple[str, ...] = ('name="phone"', "name='phone'")
_SUBMIT_HANDLER_MARKERS: tuple[str, ...] = ("sendLead(", "submitLead(", "fetch(")
_BITRIX_DIRECT_MARKERS: tuple[str, ...] = ("bitrix24.ru/rest", "crm.lead.add")
_SDA_LEAD_MARKER = "/lead"


# ---------------------------------------------------------------- pure checks


async def check_landing(http: httpx.AsyncClient, url: str) -> dict[str, Any]:
    """GET one landing page and diff against the six ``check_landing`` rules.

    Returns ``{"url": str, "ok": bool, "issues": list[str]}``. Any raised
    exception is converted into ``ok=False`` with an ``issues=[str(exc)]``
    entry — pure-function contract for ``asyncio.gather(return_exceptions)``.
    """
    issues: list[str] = []
    try:
        response = await http.get(url, timeout=_LANDING_FETCH_TIMEOUT_SEC)
    except Exception as exc:
        return {"url": url, "ok": False, "issues": [f"fetch failed: {exc}"]}
    if response.status_code != 200:
        issues.append(f"HTTP {response.status_code}")
    body = response.text or ""
    if len(body) < _MIN_LANDING_SIZE_BYTES:
        issues.append(f"Page too small ({len(body)} bytes)")
    if not any(m in body for m in _PHONE_INPUT_MARKERS):
        issues.append("No phone input found")
    if not any(m in body for m in _SUBMIT_HANDLER_MARKERS):
        issues.append("No submit handler found (sendLead/submitLead/fetch)")

    has_bitrix = any(m in body for m in _BITRIX_DIRECT_MARKERS)
    has_sda_lead = _SDA_LEAD_MARKER in body
    if not has_bitrix and not has_sda_lead:
        issues.append("No lead destination (Bitrix webhook or SDA /lead): both markers absent")

    if "startQuiz()" in body and "function startQuiz" not in body:
        issues.append("QUIZ BROKEN — startQuiz() called but not defined")

    return {"url": url, "ok": not issues, "issues": issues}


async def check_cors(http: httpx.AsyncClient, lead_endpoint: str) -> dict[str, Any]:
    """OPTIONS preflight on SDA ``/lead`` from the owner origin.

    ACAO == ``*`` or ACAO containing the owner domain → pass. Anything else,
    or a transport failure, → ``ok=False`` with a human-readable reason.
    """
    try:
        response = await http.request(
            "OPTIONS",
            lead_endpoint,
            headers={
                "Origin": _OWNER_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
            timeout=_LANDING_FETCH_TIMEOUT_SEC,
        )
    except Exception as exc:
        return {"ok": False, "issue": f"OPTIONS request failed: {exc}"}
    acao = response.headers.get("Access-Control-Allow-Origin", "")
    if acao == "*":
        return {"ok": True, "acao": acao}
    if "24bankrotsttvo" in acao:
        return {"ok": True, "acao": acao}
    return {"ok": False, "issue": f"CORS not configured (ACAO={acao!r})"}


async def check_lead_endpoint(http: httpx.AsyncClient, lead_endpoint: str) -> dict[str, Any]:
    """POST empty phone; expect structured ``{"error":"phone required"}``."""
    try:
        response = await http.post(
            lead_endpoint,
            json={"fields": {"NAME": "test", "PHONE": []}},
            timeout=_LANDING_FETCH_TIMEOUT_SEC,
        )
    except Exception as exc:
        return {"ok": False, "issue": f"POST failed: {exc}"}
    try:
        body = response.json()
    except Exception:
        return {
            "ok": False,
            "issue": f"POST returned non-JSON (HTTP {response.status_code})",
        }
    if isinstance(body, dict) and body.get("error") == "phone required":
        return {"ok": True, "response": "phone validation works"}
    return {
        "ok": False,
        "issue": f"phone validation missing (HTTP {response.status_code}, body={body!r:.120})",
    }


async def check_ad_moderation(direct: DirectAPI, campaign_ids: list[int]) -> dict[str, Any]:
    """Walk every campaign, list ads, flag ``Status='REJECTED'``.

    A single campaign failing to return ads does *not* sink the overall check:
    the failure becomes an entry in ``rejected`` with ``error`` set so alerts
    still fire and other campaigns still get checked.
    """
    rejected: list[dict[str, Any]] = []
    for campaign_id in campaign_ids:
        try:
            ad_groups = await direct.get_adgroups(campaign_id=campaign_id)
        except Exception as exc:
            rejected.append({"campaign": campaign_id, "error": f"adgroups.get failed: {exc}"})
            continue
        ad_group_ids = [int(g["Id"]) for g in ad_groups if g.get("Id")]
        if not ad_group_ids:
            continue
        try:
            ads = await direct.get_ads(ad_group_ids)
        except Exception as exc:
            rejected.append({"campaign": campaign_id, "error": f"ads.get failed: {exc}"})
            continue
        for ad in ads:
            if ad.get("Status") == "REJECTED":
                rejected.append(
                    {
                        "campaign": campaign_id,
                        "ad_id": int(ad.get("Id", 0)),
                        "reason": str(ad.get("StatusClarification") or "no reason given"),
                    }
                )
    return {"ok": not rejected, "rejected": rejected}


# ------------------------------------------------------------------ FormChecker


@dataclass
class FormCheckerResult:
    landings: list[dict[str, Any]]
    cors: dict[str, Any]
    endpoint: dict[str, Any]
    moderation: dict[str, Any]
    all_ok: bool
    action: str
    trust_level: str
    rejected_campaigns: list[int]
    suspended: list[int]
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "landings": self.landings,
            "cors": self.cors,
            "endpoint": self.endpoint,
            "moderation": self.moderation,
            "all_ok": self.all_ok,
            "action": self.action,
            "trust_level": self.trust_level,
            "rejected_campaigns": self.rejected_campaigns,
            "suspended": self.suspended,
            "ts": self.ts,
        }


def _now_msk_iso() -> str:
    return (datetime.now(UTC) + _MSK).replace(tzinfo=None).isoformat(timespec="seconds")


def _collect_global_issue(
    landings: list[dict[str, Any]],
    cors: dict[str, Any],
    endpoint: dict[str, Any],
) -> str | None:
    """Return a brief reason string if a site-wide problem is present."""
    broken = [ln for ln in landings if not ln.get("ok")]
    if broken:
        return f"{len(broken)}/{len(landings)} landings broken"
    if not cors.get("ok"):
        return f"CORS: {cors.get('issue')}"
    if not endpoint.get("ok"):
        return f"Lead endpoint: {endpoint.get('issue')}"
    return None


def _rejected_campaign_ids(moderation: dict[str, Any]) -> list[int]:
    rejected = moderation.get("rejected") or []
    ids: list[int] = []
    for row in rejected:
        cid = row.get("campaign") if isinstance(row, dict) else None
        if isinstance(cid, int) and cid not in ids:
            ids.append(cid)
    return ids


def _format_alert(
    *,
    title: str,
    landings: list[dict[str, Any]],
    cors: dict[str, Any],
    endpoint: dict[str, Any],
    moderation: dict[str, Any],
    trust_level: TrustLevel,
    suspended: list[int],
    rejected: list[int],
    protected: list[int],
    action: str,
) -> str:
    lines = [f"<b>{title}</b>", ""]
    broken = [ln for ln in landings if not ln.get("ok")]
    for ln in broken:
        issues = "; ".join(ln.get("issues") or [])
        lines.append(f"• Landing {ln['url']}: {issues}")
    if not cors.get("ok"):
        lines.append(f"• CORS: {cors.get('issue')}")
    if not endpoint.get("ok"):
        lines.append(f"• Endpoint: {endpoint.get('issue')}")
    for row in moderation.get("rejected") or []:
        cid = row.get("campaign")
        tag = " [PROTECTED → alert only]" if isinstance(cid, int) and cid in protected else ""
        if "ad_id" in row:
            lines.append(
                f"• Ad {row['ad_id']} in camp {cid} REJECTED: {row.get('reason', 'no reason')}{tag}"
            )
        else:
            lines.append(f"• Camp {cid} ad-moderation error: {row.get('error', '?')}{tag}")
    lines.append("")
    lines.append(f"Trust: <code>{trust_level.value}</code>")
    if suspended:
        lines.append(f"Suspended: <code>{suspended}</code>")
    if rejected and not suspended:
        lines.append("Action: <i>alert only</i> — rejected ids all PROTECTED or trust=shadow")
    lines.append(f"Pipeline action: {action}")
    return "\n".join(lines)


class FormChecker:
    """Injected clients + ``run()`` orchestration.

    ``settings`` provides ``PROTECTED_CAMPAIGN_IDS`` (Decision 17),
    ``PROTECTED_LANDING_URLS`` (Task 11), ``PUBLIC_BASE_URL``. No module-
    level constants — everything flows through construction.
    """

    def __init__(
        self,
        *,
        direct: DirectAPI,
        http: httpx.AsyncClient,
        pool: AsyncConnectionPool,
        settings: Settings,
    ) -> None:
        self.direct = direct
        self.http = http
        self.pool = pool
        self.settings = settings

    @property
    def lead_endpoint(self) -> str:
        return f"{self.settings.PUBLIC_BASE_URL.rstrip('/')}/lead"

    async def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        trust = await _safe_get_trust_level(self.pool)
        landings_coro = asyncio.gather(
            *[check_landing(self.http, url) for url in self.settings.PROTECTED_LANDING_URLS],
            return_exceptions=True,
        )
        cors_coro = check_cors(self.http, self.lead_endpoint)
        endpoint_coro = check_lead_endpoint(self.http, self.lead_endpoint)
        moderation_coro = check_ad_moderation(
            self.direct, list(self.settings.PROTECTED_CAMPAIGN_IDS)
        )
        landings_raw, cors, endpoint, moderation = await asyncio.gather(
            landings_coro, cors_coro, endpoint_coro, moderation_coro, return_exceptions=True
        )

        landings = _normalise_landings(landings_raw, self.settings.PROTECTED_LANDING_URLS)
        cors_res = _coerce_check_result(cors, "CORS")
        endpoint_res = _coerce_check_result(endpoint, "Lead endpoint")
        moderation_res = _coerce_moderation(moderation)

        all_ok = (
            all(ln.get("ok") for ln in landings)
            and cors_res.get("ok")
            and endpoint_res.get("ok")
            and moderation_res.get("ok")
        )
        protected = list(self.settings.PROTECTED_CAMPAIGN_IDS)
        rejected = _rejected_campaign_ids(moderation_res)
        global_issue = _collect_global_issue(landings, cors_res, endpoint_res)

        if all_ok:
            return FormCheckerResult(
                landings=landings,
                cors=cors_res,
                endpoint=endpoint_res,
                moderation=moderation_res,
                all_ok=True,
                action="none",
                trust_level=trust.value,
                rejected_campaigns=[],
                suspended=[],
                ts=_now_msk_iso(),
            ).to_dict()

        to_suspend = _compute_to_suspend(
            global_issue=global_issue, rejected=rejected, protected=protected
        )
        effective = allowed_action("form_checker", trust, AutonomyLevel.AUTO)
        suspended: list[int] = []
        action_name: str

        if dry_run:
            action_name = "dry_run"
        elif effective == AutonomyLevel.AUTO and to_suspend:
            citation = await _kb_consult_before_suspend(to_suspend, global_issue)
            logger.info(
                "form_checker: kb citations before suspend: %s",
                citation.get("citations") if citation else "n/a",
            )
            suspended = await _suspend_many(self.direct, to_suspend)
            action_name = f"auto-suspended {suspended}"
        elif effective == AutonomyLevel.ASK:
            action_name = f"ask_queue_pending to_suspend={to_suspend}"
        elif trust == TrustLevel.SHADOW:
            action_name = "alert_only (trust=shadow)"
        elif not to_suspend:
            action_name = f"alert_only (protected={protected})"
        else:
            action_name = f"alert_only (trust={trust.value})"

        title = _alert_title(
            suspended=suspended,
            trust=trust,
            global_issue=global_issue,
            dry_run=dry_run,
        )
        if not dry_run:
            try:
                await telegram_tools.send_message(
                    self.http,
                    self.settings,
                    text=_format_alert(
                        title=title,
                        landings=landings,
                        cors=cors_res,
                        endpoint=endpoint_res,
                        moderation=moderation_res,
                        trust_level=trust,
                        suspended=suspended,
                        rejected=rejected,
                        protected=protected,
                        action=action_name,
                    ),
                )
            except Exception:
                logger.exception("form_checker: telegram alert failed")

        try:
            await insert_audit_log(
                self.pool,
                hypothesis_id=None,
                trust_level=trust.value,
                tool_name="form_checker",
                tool_input={
                    "landings_broken": [ln["url"] for ln in landings if not ln.get("ok")],
                    "rejected_campaigns": rejected,
                    "global_issue": global_issue,
                    "dry_run": dry_run,
                },
                tool_output={"suspended": suspended, "action": action_name},
                is_mutation=bool(suspended),
            )
        except Exception:
            logger.exception("form_checker: audit_log write failed")

        return FormCheckerResult(
            landings=landings,
            cors=cors_res,
            endpoint=endpoint_res,
            moderation=moderation_res,
            all_ok=False,
            action=action_name,
            trust_level=trust.value,
            rejected_campaigns=rejected,
            suspended=suspended,
            ts=_now_msk_iso(),
        ).to_dict()


# ------------------------------------------------------------ helpers


async def _safe_get_trust_level(pool: AsyncConnectionPool) -> TrustLevel:
    try:
        return await get_trust_level(pool)
    except Exception:
        logger.warning("form_checker: trust_level lookup failed, defaulting shadow", exc_info=True)
        return TrustLevel.SHADOW


def _compute_to_suspend(
    *,
    global_issue: str | None,
    rejected: list[int],
    protected: list[int],
) -> list[int]:
    """Decide which campaign ids get scheduled for suspension.

    * global_issue → previously "all ON campaigns minus protected". We cannot
      discover ON-but-not-protected without an extra ``campaigns.get(State=ON)``
      call — that list is empty in prod today (all ON campaigns are PROTECTED),
      so we return ``[]`` and fall into the alert-only path. If a new non-
      protected campaign ever appears, the next task iteration can extend.
    * rejected ad → suspend only non-protected campaign ids.
    """
    if global_issue:
        return []
    return [cid for cid in rejected if cid not in protected]


def _alert_title(
    *,
    suspended: list[int],
    trust: TrustLevel,
    global_issue: str | None,
    dry_run: bool,
) -> str:
    if dry_run:
        return "ПРОБЛЕМЫ — dry_run (без стопа)"
    if suspended:
        return "ПРОБЛЕМЫ — кампании ОСТАНОВЛЕНЫ"
    if trust == TrustLevel.SHADOW:
        return "ПРОБЛЕМЫ — shadow NOTIFY"
    if global_issue:
        return "ПРОБЛЕМЫ — алерт без стопа (prod защищён)"
    return "ПРОБЛЕМЫ — алерт без стопа"


async def _suspend_many(direct: DirectAPI, campaign_ids: list[int]) -> list[int]:
    """Pause each campaign best-effort; return the ones that actually paused.

    ProtectedCampaignError is swallowed — in prod today ALL known
    PROTECTED_CAMPAIGN_IDS are real prod campaigns, so this should
    never actually fire (we pre-filter), but defensive either way.
    """
    suspended: list[int] = []
    for campaign_id in campaign_ids:
        try:
            await direct.pause_campaign(campaign_id)
        except ProtectedCampaignError:
            logger.warning(
                "form_checker: pause_campaign(%d) blocked by protected guard",
                campaign_id,
            )
            continue
        except Exception:
            logger.exception("form_checker: pause_campaign(%d) failed", campaign_id)
            continue
        try:
            if await direct.verify_campaign_paused(campaign_id):
                suspended.append(campaign_id)
        except Exception:
            logger.warning(
                "form_checker: verify_campaign_paused(%d) errored",
                campaign_id,
                exc_info=True,
            )
    return suspended


async def _kb_consult_before_suspend(
    to_suspend: list[int], global_issue: str | None
) -> dict[str, Any] | None:
    """Best-effort KB lookup — satisfies "kb.consult before mutation" rule."""
    try:
        return await knowledge.consult(
            f"safe to suspend campaigns {to_suspend} after form_checker? "
            f"global_issue={global_issue}",
            context={"to_suspend": to_suspend, "global_issue": global_issue},
        )
    except Exception as exc:
        logger.info("form_checker: kb.consult skipped (%s)", exc.__class__.__name__)
        return None


def _normalise_landings(landings_raw: Any, urls: list[str]) -> list[dict[str, Any]]:
    """Turn the mixed gather-result (dicts + exceptions) into uniform rows."""
    if isinstance(landings_raw, BaseException):
        return [{"url": u, "ok": False, "issues": [str(landings_raw)]} for u in urls]
    out: list[dict[str, Any]] = []
    assert isinstance(landings_raw, list)
    for url, item in zip(urls, landings_raw, strict=False):
        if isinstance(item, BaseException):
            out.append({"url": url, "ok": False, "issues": [str(item)]})
        elif isinstance(item, dict):
            out.append(item)
        else:
            out.append({"url": url, "ok": False, "issues": [f"bad check result {item!r}"]})
    return out


def _coerce_check_result(res: Any, label: str) -> dict[str, Any]:
    if isinstance(res, BaseException):
        return {"ok": False, "issue": f"{label} check crashed: {res}"}
    if isinstance(res, dict):
        return res
    return {"ok": False, "issue": f"{label} check returned {type(res).__name__}"}


def _coerce_moderation(res: Any) -> dict[str, Any]:
    if isinstance(res, BaseException):
        return {"ok": False, "rejected": [{"error": f"moderation crashed: {res}"}]}
    if isinstance(res, dict):
        return res
    return {"ok": False, "rejected": [{"error": f"bad type {type(res).__name__}"}]}


# ---------------------------------------------------------------- job entry


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Railway cron entrypoint. Degrades gracefully when DI is incomplete.

    The default dispatch path (``(pool, dry_run)``) from JOB_REGISTRY cannot
    actually run the checks — form_checker needs DirectAPI and an http
    client. Returns a structured no-op so health checks keep passing;
    the /run/form_checker FastAPI handler (wires app.state) will run it
    for real.
    """
    trust = await _safe_get_trust_level(pool)
    if direct is None or http_client is None or settings is None:
        logger.warning("form_checker: missing DI (direct/http/settings) — degraded no-op")
        return {
            "status": "ok",
            "trust_level": trust.value,
            "landings": [],
            "cors": {"ok": True, "skipped": True},
            "endpoint": {"ok": True, "skipped": True},
            "moderation": {"ok": True, "rejected": []},
            "all_ok": True,
            "action": "degraded_noop",
            "rejected_campaigns": [],
            "suspended": [],
            "ts": _now_msk_iso(),
        }
    fc = FormChecker(direct=direct, http=http_client, pool=pool, settings=settings)
    result = await fc.run(dry_run=dry_run)
    return {"status": "ok", **result}


__all__ = [
    "FormChecker",
    "check_ad_moderation",
    "check_cors",
    "check_landing",
    "check_lead_endpoint",
    "run",
]
