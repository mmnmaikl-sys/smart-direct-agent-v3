"""Auto-Resume — resume auto-suspended PROTECTED campaigns (Task 21, part 1).

Every day at 08:00 МСК (05:00 UTC — ``0 5 * * *`` Railway Cron HTTP-trigger)
this job walks :attr:`Settings.PROTECTED_CAMPAIGN_IDS`, picks the ones that
ended up in ``State='SUSPENDED'`` (budget_guard auto-suspend, form_checker
auto-suspend, watchdog dead-man stop, Direct daily-budget exhaustion,
external operator pause on moderation error) and resumes them — provided the
campaign is not archived (``StatusArchive='YES'`` means operator intent to
keep it down, e.g. 458 Удмуртия and 709014142 РСЯ-ретаргет).

Trust overlay (Decision 7, tech-spec §160 — ``auto_resume`` is in
:data:`agent_runtime.trust_levels.ASSISTED_AUTO_WHITELIST`):

* ``shadow``           → NOTIFY only (Telegram alert, ``would_resume`` marker).
* ``assisted``         → AUTO (resume + verify), then notify.
* ``autonomous``       → AUTO (resume + verify), then notify.
* ``FORBIDDEN_LOCK``   → NOTIFY only.

PROTECTED campaigns carry a DirectAPI-layer guard (Decision 12,
:class:`agent_runtime.tools.direct_api.ProtectedCampaignError`): the
``_check_not_protected`` pre-flight raises on mutations. ``resume_campaign``
is a mutation on a PROTECTED id — the guard WILL raise. We swallow it and
fall through to NOTIFY so the alert still fires and the owner decides.
Ostensibly this makes resume impossible in prod; the expected resolution
(tracked as ``TODO(integration)``) is to add ``resume_campaign`` to the
DirectAPI allowlist specifically for the auto-resume path, since
resume-of-whitelisted-protected is by-definition safe (the guard protects
against silent MUTES, not silent UN-MUTES).

``dry_run=True`` runs read-only (``get_campaigns``) and populates
``would_resume``; no mutations, no Telegram, no audit_log writes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.models import AutonomyLevel
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI, ProtectedCampaignError
from agent_runtime.trust_levels import TrustLevel, allowed_action, get_trust_level

logger = logging.getLogger(__name__)


_VERIFY_RETRY_ATTEMPTS = 3
_VERIFY_RETRY_SLEEP_SEC = 2.0
# TODO(integration): expose as Settings.AUTO_RESUME_ACTION_TYPE if we ever
# need per-tenant overrides. Stable enough to live inline today.
_ACTION_TYPE = "auto_resume"


# ----------------------------------------------------------------- result model


@dataclass(frozen=True)
class _CampaignView:
    campaign_id: int
    name: str
    state: str
    status_archive: str


class AutoResumeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str  # "ok" | "error"
    trust_level: str
    checked: int = 0
    suspended_found: list[int] = Field(default_factory=list)
    resumed: list[int] = Field(default_factory=list)
    would_resume: list[int] = Field(default_factory=list)
    notified: list[int] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    dry_run: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ------------------------------------------------------------- helper coroutines


async def _sleep_for_verify() -> None:
    """Isolated for monkeypatching in tests."""
    await asyncio.sleep(_VERIFY_RETRY_SLEEP_SEC)


async def _safe_get_trust_level(pool: AsyncConnectionPool) -> TrustLevel:
    try:
        return await get_trust_level(pool)
    except Exception:
        logger.warning("auto_resume: trust_level lookup failed, defaulting shadow", exc_info=True)
        return TrustLevel.SHADOW


async def _fetch_campaign_views(
    direct: DirectAPI,
    campaign_ids: list[int],
) -> list[_CampaignView]:
    """Fetch ``{Id, Name, State, StatusArchive}`` for every id, fail-safe.

    One failing fetch does not void the whole pass — other ids still get
    checked. Missing ``StatusArchive`` (older campaigns without the flag)
    is coerced to ``'NO'`` so the downstream filter logic is branchless.
    """
    views: list[_CampaignView] = []
    # DirectAPI.get_campaigns already batches on a single SelectionCriteria
    # request; one call covers all ids.
    try:
        raw = await direct.get_campaigns(list(campaign_ids))
    except Exception:
        logger.exception("auto_resume: get_campaigns(%s) failed", campaign_ids)
        return views
    for row in raw:
        try:
            views.append(
                _CampaignView(
                    campaign_id=int(row["Id"]),
                    name=str(row.get("Name") or ""),
                    state=str(row.get("State") or ""),
                    status_archive=str(row.get("StatusArchive") or "NO"),
                )
            )
        except (KeyError, ValueError, TypeError):
            logger.warning("auto_resume: skipping malformed campaign row: %r", row)
    return views


def _select_resume_targets(views: list[_CampaignView]) -> list[_CampaignView]:
    """Filter to SUSPENDED & not-archived. Explicit single-purpose helper so
    the test suite can tabulate edge cases (archived ON, archived SUSPENDED,
    etc.) without mocking DirectAPI."""
    targets: list[_CampaignView] = []
    for v in views:
        if v.state != "SUSPENDED":
            continue
        if v.status_archive == "YES":
            continue
        targets.append(v)
    return targets


async def _resume_with_verify(direct: DirectAPI, campaign_id: int) -> tuple[bool, str | None]:
    """Call ``resume_campaign`` + ``verify_campaign_resumed`` with retry.

    Returns ``(ok, error_reason)``.

    * :class:`ProtectedCampaignError` is swallowed — every id we touch here
      is BY DEFINITION in :data:`Settings.PROTECTED_CAMPAIGN_IDS`, which the
      DirectAPI guard blocks. Today that means resume is a no-op in prod;
      the ``TODO(integration)`` in the module docstring tracks the fix.
    * Verify uses 3 attempts with :func:`_sleep_for_verify` to absorb the
      eventual-consistency window on the Direct control plane.
    """
    try:
        await direct.resume_campaign(campaign_id)
    except ProtectedCampaignError:
        logger.warning(
            "auto_resume: resume_campaign(%d) blocked by protected guard — "
            "Telegram NOTIFY still fires; owner to unblock (TODO integration)",
            campaign_id,
        )
        return False, "protected_guard"
    except Exception as exc:
        logger.exception("auto_resume: resume_campaign(%d) failed", campaign_id)
        return False, f"{type(exc).__name__}: {exc}"
    for attempt in range(_VERIFY_RETRY_ATTEMPTS):
        try:
            if await direct.verify_campaign_resumed(campaign_id):
                return True, None
        except Exception:
            logger.warning(
                "auto_resume: verify_campaign_resumed(%d) attempt %d errored",
                campaign_id,
                attempt + 1,
                exc_info=True,
            )
        await _sleep_for_verify()
    logger.warning("auto_resume: verify_campaign_resumed(%d) never True", campaign_id)
    return False, "verify_mismatch"


def _format_alert(
    *,
    trust_level: TrustLevel,
    action_taken: str,
    resumed: list[int],
    would_resume: list[int],
    suspended_views: list[_CampaignView],
    errors: list[dict[str, Any]],
) -> str:
    lines = ["<b>AUTO-RESUME</b>", ""]
    if resumed:
        lines.append(f"Resumed: <code>{resumed}</code>")
    if would_resume:
        lines.append(f"Would resume (dry-run / NOTIFY): <code>{would_resume}</code>")
    if not resumed and not would_resume:
        lines.append("Nothing to resume — all PROTECTED campaigns ON or archived.")
    lines.append("")
    if suspended_views:
        lines.append("<b>Suspended (non-archive) detected:</b>")
        for v in suspended_views:
            lines.append(f"• <code>{v.campaign_id}</code> {v.name} (state={v.state})")
        lines.append("")
    if errors:
        lines.append("<b>Errors:</b>")
        for err in errors:
            lines.append(f"• <code>{err.get('campaign_id')}</code>: {err.get('error', 'unknown')}")
        lines.append("")
    lines.append(f"Trust: <code>{trust_level.value}</code>")
    lines.append(f"Action: {action_taken}")
    return "\n".join(lines)


# --------------------------------------------------------------------- run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Railway cron entrypoint. Returns :class:`AutoResumeResult` dict.

    When invoked through the default JOB_REGISTRY wrapper (only
    ``pool + dry_run``) the job returns a degraded no-op so a cron misfire
    does not 500. The FastAPI handler at ``/run/auto_resume`` is expected
    to inject ``direct``, ``http_client``, ``settings`` from ``app.state``
    for real runs.

    TODO(integration): register in
    ``agent_runtime/jobs/__init__.py::JOB_REGISTRY`` under slug
    ``"auto_resume"`` and add Railway cron ``0 5 * * *`` + FastAPI handler
    ``POST /run/auto_resume`` (HTTPBearer SDA_INTERNAL_API_KEY,
    ``@limiter.limit("2/hour")``).
    """
    trust_level = await _safe_get_trust_level(pool)
    if direct is None or settings is None:
        logger.warning("auto_resume: direct/settings not injected — degraded no-op")
        return AutoResumeResult(
            status="ok", trust_level=trust_level.value, dry_run=dry_run
        ).model_dump(mode="json")

    try:
        views = await _fetch_campaign_views(direct, list(settings.PROTECTED_CAMPAIGN_IDS))
    except Exception as exc:
        logger.exception("auto_resume: fatal fetch error")
        return AutoResumeResult(
            status="error",
            trust_level=trust_level.value,
            dry_run=dry_run,
            errors=[{"campaign_id": None, "error": f"{type(exc).__name__}: {exc}"}],
        ).model_dump(mode="json")

    targets = _select_resume_targets(views)
    effective = allowed_action(_ACTION_TYPE, trust_level, AutonomyLevel.AUTO)

    resumed: list[int] = []
    would_resume: list[int] = []
    notified: list[int] = []
    errors: list[dict[str, Any]] = []

    for target in targets:
        # dry_run is the short-circuit: never touch the wire beyond the
        # read-only get_campaigns we already did.
        if dry_run:
            would_resume.append(target.campaign_id)
            continue
        if effective == AutonomyLevel.AUTO:
            ok, err = await _resume_with_verify(direct, target.campaign_id)
            if ok:
                resumed.append(target.campaign_id)
            else:
                errors.append(
                    {
                        "campaign_id": target.campaign_id,
                        "error": err or "unknown",
                    }
                )
        else:
            # NOTIFY / ASK / FORBIDDEN — the alert path handles reporting;
            # would_resume captures "we would have resumed under AUTO".
            would_resume.append(target.campaign_id)

    # --- side effects: telegram + audit -------------------------------------

    action_taken = _describe_action(
        dry_run=dry_run,
        effective=effective,
        trust_level=trust_level,
        targets=targets,
    )

    if not dry_run and targets and http_client is not None:
        try:
            await telegram_tools.send_message(
                http_client,
                settings,
                text=_format_alert(
                    trust_level=trust_level,
                    action_taken=action_taken,
                    resumed=resumed,
                    would_resume=would_resume,
                    suspended_views=targets,
                    errors=errors,
                ),
            )
            notified = [t.campaign_id for t in targets]
        except Exception:
            logger.exception("auto_resume: telegram alert failed")

    if not dry_run and targets:
        try:
            await insert_audit_log(
                pool,
                hypothesis_id=None,
                trust_level=trust_level.value,
                tool_name="auto_resume",
                tool_input={
                    "protected_ids": list(settings.PROTECTED_CAMPAIGN_IDS),
                    "suspended_found": [t.campaign_id for t in targets],
                },
                tool_output={
                    "resumed": resumed,
                    "would_resume": would_resume,
                    "errors": errors,
                    "action": action_taken,
                },
                is_mutation=bool(resumed),
                is_error=bool(errors),
                error_detail=(errors[0].get("error") if errors else None),
            )
        except Exception:
            logger.exception("auto_resume: audit_log write failed")

    result = AutoResumeResult(
        status="ok",
        trust_level=trust_level.value,
        checked=len(views),
        suspended_found=[t.campaign_id for t in targets],
        resumed=resumed,
        would_resume=would_resume,
        notified=notified,
        errors=errors,
        dry_run=dry_run,
    )
    logger.info(
        "auto_resume done trust=%s checked=%d suspended=%d resumed=%d errors=%d dry_run=%s",
        trust_level.value,
        len(views),
        len(targets),
        len(resumed),
        len(errors),
        dry_run,
    )
    return result.model_dump(mode="json")


def _describe_action(
    *,
    dry_run: bool,
    effective: AutonomyLevel,
    trust_level: TrustLevel,
    targets: list[_CampaignView],
) -> str:
    if not targets:
        return "noop_all_active"
    if dry_run:
        return "dry_run_would_resume"
    if effective == AutonomyLevel.AUTO:
        return "auto_resumed"
    if effective == AutonomyLevel.NOTIFY:
        return f"notify_only (trust={trust_level.value})"
    if effective == AutonomyLevel.FORBIDDEN:
        return "forbidden_lock_notify_only"
    return f"notify_only (effective={effective.value})"


__all__ = [
    "AutoResumeResult",
    "run",
]
