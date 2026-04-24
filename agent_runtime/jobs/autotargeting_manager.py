"""Autotargeting Manager — enforce Decision 15 on ЕПК-Поиск (Task 21, part 2).

Every 3 hours at :50 past (07:50 МСК start — ``50 4-22/3 * * *`` Railway Cron
HTTP-trigger) this job walks the three ЕПК-Поиск campaigns
(:data:`EPK_SEARCH_CAMPAIGNS` = 708978456/457/458 — Башкортостан, Татарстан,
Удмуртия) and verifies that every AdGroup's ``AutotargetingSettings`` is
``Category=EXACT, Brands=WithoutBrands``. Any drift
(``Category`` in {``WIDER``, ``COMPETITORS``, ``ALTERNATIVE``,
``ACCESSORIES``}) or any non-``WithoutBrands`` ``Brands`` triggers a
rollback to the Decision 15 target.

Trust overlay (Decision 7, tech-spec §160):

* ``shadow``            → NOTIFY only (Telegram, ``would_rollback`` marker).
* ``assisted``          → NOTIFY only today — ``set_autotargeting`` is NOT
  in :data:`agent_runtime.trust_levels.ASSISTED_AUTO_WHITELIST`. Resolution
  (``TODO(integration)``) is to add ``autotargeting_set`` to the whitelist
  once the DirectAPI wrapper lands.
* ``autonomous``        → AUTO: rollback + GET-after-SET verify. A verify
  mismatch emits a CRITICAL Telegram + audit ``error_detail=
  'autotargeting_set_verify_fail'`` and moves on — next cron tick retries.
* ``FORBIDDEN_LOCK``    → NOTIFY only.

The read path uses :meth:`DirectAPI.get_adgroups` (existing) to enumerate
AdGroup ids; the read of autotargeting settings and the SET path both go
through ``getattr(direct, 'get_autotargeting', None)`` /
``getattr(direct, 'set_autotargeting', None)`` so this module LANDS without
editing ``agent_runtime/tools/direct_api.py``. When those wrappers are
absent (today), the run degrades to a structured no-op with
``status='ok', action='degraded_wrapper_missing'`` — smoke/health keeps
passing while the Task 7 extension rolls in behind.

``dry_run=True`` runs read-only and populates ``would_rollback``; 0
mutations, 0 Telegram, 0 audit_log writes.
"""

from __future__ import annotations

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


# --- Decision 15 enforcement targets ----------------------------------------
# TODO(integration): these constants will move to Settings in the integration
# pass so Wave 3 tuning does not require a code deploy. Kept inline here so
# Task 21 lands in one self-contained module (no schema drift).
EPK_SEARCH_CAMPAIGNS: tuple[int, ...] = (708978456, 708978457, 708978458)
REQUIRED_CATEGORY: str = "EXACT"
REQUIRED_BRANDS: str = "WithoutBrands"
_ACTION_TYPE = "autotargeting_set"


# --- result model -----------------------------------------------------------


@dataclass(frozen=True)
class _Drift:
    campaign_id: int
    ad_group_id: int
    from_category: str
    from_brands: str


class AutotargetingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str  # "ok" | "error" | "degraded_wrapper_missing"
    trust_level: str
    checked_campaigns: list[int] = Field(default_factory=list)
    checked_ad_groups: int = 0
    drift_detected: list[dict[str, Any]] = Field(default_factory=list)
    rolled_back: list[dict[str, Any]] = Field(default_factory=list)
    would_rollback: list[dict[str, Any]] = Field(default_factory=list)
    notified: bool = False
    errors: list[dict[str, Any]] = Field(default_factory=list)
    dry_run: bool = False
    action: str = "noop"
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- helpers ----------------------------------------------------------------


async def _safe_get_trust_level(pool: AsyncConnectionPool) -> TrustLevel:
    try:
        return await get_trust_level(pool)
    except Exception:
        logger.warning(
            "autotargeting_manager: trust_level lookup failed, defaulting shadow",
            exc_info=True,
        )
        return TrustLevel.SHADOW


def _detect_drift(settings_blob: Any) -> tuple[str, str] | None:
    """Normalise the ``AutotargetingSettings`` payload and return (category,
    brands) when either is NOT at target. Returns ``None`` on compliance.

    Direct API returns different shapes across API versions — accept a dict
    with ``Category``/``Brands`` (v5 current), a dict wrapped under
    ``AutotargetingSettings``, or a flat adgroup row. Anything unrecognised
    is reported as drift (``Category='<unknown>'``) so the owner gets
    visibility instead of silent skip.
    """
    if settings_blob is None:
        return ("<missing>", "<missing>")
    payload = settings_blob
    if isinstance(payload, dict) and "AutotargetingSettings" in payload:
        payload = payload["AutotargetingSettings"]
    if not isinstance(payload, dict):
        return ("<unparsable>", "<unparsable>")
    category = str(payload.get("Category") or "<missing>")
    brands = str(payload.get("Brands") or "<missing>")
    if category == REQUIRED_CATEGORY and brands == REQUIRED_BRANDS:
        return None
    return (category, brands)


async def _read_autotargeting(direct: DirectAPI, ad_group_id: int) -> Any:
    """Best-effort read through an optional ``get_autotargeting`` method.

    Returns ``None`` if the wrapper is missing on this Task 7 revision —
    the caller treats that as "wrapper not yet shipped", emits the
    degraded-noop result, and does not call ``set_autotargeting`` either.
    """
    getter = getattr(direct, "get_autotargeting", None)
    if getter is None:
        return None
    try:
        return await getter(ad_group_id)
    except Exception:
        logger.exception(
            "autotargeting_manager: get_autotargeting(%d) failed — treat as drift",
            ad_group_id,
        )
        return {"Category": "<error>", "Brands": "<error>"}


async def _set_autotargeting_with_verify(
    direct: DirectAPI,
    ad_group_id: int,
) -> tuple[bool, str | None]:
    """Call ``set_autotargeting`` + GET-after-SET verify. Returns ``(ok, err)``."""
    setter = getattr(direct, "set_autotargeting", None)
    if setter is None:
        return False, "wrapper_missing"
    try:
        await setter(
            ad_group_id,
            Category=REQUIRED_CATEGORY,
            Brands=REQUIRED_BRANDS,
        )
    except ProtectedCampaignError:
        logger.warning(
            "autotargeting_manager: set_autotargeting(%d) blocked by protected guard",
            ad_group_id,
        )
        return False, "protected_guard"
    except Exception as exc:
        logger.exception("autotargeting_manager: set_autotargeting(%d) failed", ad_group_id)
        return False, f"{type(exc).__name__}: {exc}"
    # GET-after-SET verify — mandatory per Decision 3.
    after = await _read_autotargeting(direct, ad_group_id)
    drift_after = _detect_drift(after)
    if drift_after is None:
        return True, None
    return False, "autotargeting_set_verify_fail"


async def _enumerate_ad_groups(
    direct: DirectAPI,
    campaign_ids: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Return ``[(campaign_id, ad_group_id)]`` for each AdGroup in the scope.

    One campaign failing does not sink the whole tick — we log and move on.
    """
    pairs: list[tuple[int, int]] = []
    for campaign_id in campaign_ids:
        try:
            rows = await direct.get_adgroups(campaign_id=campaign_id)
        except Exception:
            logger.exception("autotargeting_manager: get_adgroups(%d) failed", campaign_id)
            continue
        for row in rows:
            try:
                pairs.append((campaign_id, int(row["Id"])))
            except (KeyError, ValueError, TypeError):
                logger.warning("autotargeting_manager: skipping malformed adgroup row: %r", row)
    return pairs


def _format_alert(
    *,
    trust_level: TrustLevel,
    action: str,
    drift_detected: list[_Drift],
    rolled_back: list[int],
    would_rollback: list[int],
    errors: list[dict[str, Any]],
) -> str:
    lines = ["<b>AUTOTARGETING MANAGER</b>", ""]
    if drift_detected:
        lines.append("<b>Drift detected (from Decision 15 target):</b>")
        for d in drift_detected:
            lines.append(
                f"• camp=<code>{d.campaign_id}</code> "
                f"ag=<code>{d.ad_group_id}</code>: "
                f"Category={d.from_category} Brands={d.from_brands} "
                f"→ {REQUIRED_CATEGORY}/{REQUIRED_BRANDS}"
            )
        lines.append("")
    if rolled_back:
        lines.append(f"Rolled back: <code>{rolled_back}</code>")
    if would_rollback:
        lines.append(f"Would rollback: <code>{would_rollback}</code>")
    if errors:
        lines.append("<b>Errors:</b>")
        for err in errors:
            lines.append(
                f"• ag=<code>{err.get('ad_group_id')}</code>: {err.get('error', 'unknown')}"
            )
        lines.append("")
    lines.append(f"Trust: <code>{trust_level.value}</code>")
    lines.append(f"Action: {action}")
    return "\n".join(lines)


# --- run --------------------------------------------------------------------


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Railway cron entrypoint. Returns :class:`AutotargetingResult` dict.

    Degraded no-op when DI is missing (no direct / no settings). Also
    degrades to ``status='degraded_wrapper_missing'`` when
    ``direct.get_autotargeting`` is not yet shipped — keeps the cron path
    alive until the Task 7 wrapper extension.

    TODO(integration):
      1. Add ``"autotargeting_manager": autotargeting_manager.run`` to
         :data:`agent_runtime.jobs.JOB_REGISTRY`.
      2. Ship ``DirectAPI.get_autotargeting(ad_group_id)`` +
         ``DirectAPI.set_autotargeting(ad_group_id, Category, Brands)`` in
         ``agent_runtime/tools/direct_api.py`` with GET-after-SET invariant.
      3. Add ``"autotargeting_set"`` to
         :data:`agent_runtime.trust_levels.ASSISTED_AUTO_WHITELIST` so
         assisted can AUTO-rollback without owner poke for every tick.
      4. Railway Cron ``50 4-22/3 * * *`` on ``/run/autotargeting_manager``
         (HTTPBearer + ``@limiter.limit("10/hour")``).
    """
    trust_level = await _safe_get_trust_level(pool)
    if direct is None or settings is None:
        logger.warning("autotargeting_manager: direct/settings not injected — degraded no-op")
        return AutotargetingResult(
            status="ok",
            trust_level=trust_level.value,
            action="degraded_noop",
            dry_run=dry_run,
        ).model_dump(mode="json")

    # Fast fail if the Task 7 wrapper extension has not landed yet — keeps
    # cron green and structured.
    if getattr(direct, "get_autotargeting", None) is None:
        logger.warning(
            "autotargeting_manager: DirectAPI.get_autotargeting missing — "
            "waiting for Task 7 integration"
        )
        return AutotargetingResult(
            status="degraded_wrapper_missing",
            trust_level=trust_level.value,
            action="wrapper_missing",
            dry_run=dry_run,
        ).model_dump(mode="json")

    effective = allowed_action(_ACTION_TYPE, trust_level, AutonomyLevel.AUTO)
    pairs = await _enumerate_ad_groups(direct, EPK_SEARCH_CAMPAIGNS)

    drift_rows: list[_Drift] = []
    rolled_back: list[int] = []
    would_rollback: list[int] = []
    errors: list[dict[str, Any]] = []

    for campaign_id, ad_group_id in pairs:
        settings_blob = await _read_autotargeting(direct, ad_group_id)
        drift = _detect_drift(settings_blob)
        if drift is None:
            continue
        from_category, from_brands = drift
        drift_rows.append(
            _Drift(
                campaign_id=campaign_id,
                ad_group_id=ad_group_id,
                from_category=from_category,
                from_brands=from_brands,
            )
        )

        if dry_run:
            would_rollback.append(ad_group_id)
            continue
        if effective == AutonomyLevel.AUTO:
            ok, err = await _set_autotargeting_with_verify(direct, ad_group_id)
            if ok:
                rolled_back.append(ad_group_id)
            else:
                errors.append(
                    {
                        "ad_group_id": ad_group_id,
                        "campaign_id": campaign_id,
                        "error": err or "unknown",
                    }
                )
        else:
            would_rollback.append(ad_group_id)

    action = _describe_action(
        dry_run=dry_run,
        effective=effective,
        trust_level=trust_level,
        drift_count=len(drift_rows),
    )

    if not dry_run and drift_rows and http_client is not None:
        try:
            await telegram_tools.send_message(
                http_client,
                settings,
                text=_format_alert(
                    trust_level=trust_level,
                    action=action,
                    drift_detected=drift_rows,
                    rolled_back=rolled_back,
                    would_rollback=would_rollback,
                    errors=errors,
                ),
            )
            notified = True
        except Exception:
            logger.exception("autotargeting_manager: telegram alert failed")
            notified = False
    else:
        notified = False

    if not dry_run and drift_rows:
        try:
            await insert_audit_log(
                pool,
                hypothesis_id=None,
                trust_level=trust_level.value,
                tool_name="autotargeting_manager",
                tool_input={
                    "epk_search_campaigns": list(EPK_SEARCH_CAMPAIGNS),
                    "drift_detected": [
                        {
                            "campaign_id": d.campaign_id,
                            "ad_group_id": d.ad_group_id,
                            "from": {
                                "Category": d.from_category,
                                "Brands": d.from_brands,
                            },
                        }
                        for d in drift_rows
                    ],
                },
                tool_output={
                    "rolled_back": rolled_back,
                    "would_rollback": would_rollback,
                    "errors": errors,
                    "action": action,
                },
                is_mutation=bool(rolled_back),
                is_error=bool(errors),
                error_detail=(errors[0].get("error") if errors else None),
            )
        except Exception:
            logger.exception("autotargeting_manager: audit_log write failed")

    drift_payload = [
        {
            "campaign_id": d.campaign_id,
            "ad_group_id": d.ad_group_id,
            "from": {"Category": d.from_category, "Brands": d.from_brands},
            "to": {"Category": REQUIRED_CATEGORY, "Brands": REQUIRED_BRANDS},
        }
        for d in drift_rows
    ]
    rolled_back_payload = [
        {
            "ad_group_id": ag,
            "to": {"Category": REQUIRED_CATEGORY, "Brands": REQUIRED_BRANDS},
        }
        for ag in rolled_back
    ]
    would_rollback_payload = [
        {
            "ad_group_id": ag,
            "to": {"Category": REQUIRED_CATEGORY, "Brands": REQUIRED_BRANDS},
        }
        for ag in would_rollback
    ]

    result = AutotargetingResult(
        status="ok",
        trust_level=trust_level.value,
        checked_campaigns=list(EPK_SEARCH_CAMPAIGNS),
        checked_ad_groups=len(pairs),
        drift_detected=drift_payload,
        rolled_back=rolled_back_payload,
        would_rollback=would_rollback_payload,
        notified=notified,
        errors=errors,
        dry_run=dry_run,
        action=action,
    )
    logger.info(
        "autotargeting_manager done trust=%s ad_groups=%d drift=%d rolled_back=%d "
        "errors=%d dry_run=%s",
        trust_level.value,
        len(pairs),
        len(drift_rows),
        len(rolled_back),
        len(errors),
        dry_run,
    )
    return result.model_dump(mode="json")


def _describe_action(
    *,
    dry_run: bool,
    effective: AutonomyLevel,
    trust_level: TrustLevel,
    drift_count: int,
) -> str:
    if drift_count == 0:
        return "noop_no_drift"
    if dry_run:
        return "dry_run_would_rollback"
    if effective == AutonomyLevel.AUTO:
        return "auto_rolled_back"
    if effective == AutonomyLevel.FORBIDDEN:
        return "forbidden_lock_notify_only"
    return f"notify_only (trust={trust_level.value})"


__all__ = [
    "EPK_SEARCH_CAMPAIGNS",
    "REQUIRED_BRANDS",
    "REQUIRED_CATEGORY",
    "AutotargetingResult",
    "run",
]
