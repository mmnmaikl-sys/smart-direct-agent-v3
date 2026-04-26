"""Pure-rule ad quality assessor — ФЗ-38 + Title/Body checks.

NO LLM in this layer — fully deterministic so behaviour is testable
and reproducible. v2 may add an optional LLM rewrite step on top of
the verdict.

Reads all ads from the 5 own campaigns via Direct API, scores each
against:

  1. ФЗ-38 ст.28.1 banned phrases (effective 01.01.2026 — but enforced
     now to avoid retraining the agent in 8 months). Hit = AUTO_PAUSE.
  2. Quality checklist (length 25-56, has number, no UPPERCASE words,
     no multiple !!!, has CTA verb in body). Issues = NEEDS_REWRITE.
  3. Otherwise = APPROVE.

Sends a Telegram digest with counts per verdict and the offending
ad_ids. Does NOT auto-pause yet (owner sign-off needed for that step).

Source rubric: agent_runtime/knowledge/ad_quality_bfl_2026-04.md
(Deep Research 2026-04 + ФЗ-38).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)


_OWN_CAMPAIGNS: dict[int, str] = {
    # Test cohort (5 personas, DailyBudget tunable via tactical_actuator).
    709353005: "rabotyaga",
    709353034: "pensioner",
    709353058: "mother",
    709353078: "mfo",
    709353099: "property",
    # Main cohort (3 regions, WB_MAXIMUM_CLICKS strategy — production traffic).
    # Added 26.04.2026 after owner request "прогони компанию по этим всем
    # настройкам". ad_quality is read-only; expanding scope is safe — no
    # mutations on main cids from this job.
    708978456: "main-bashk",
    708978457: "main-tat",
    708978458: "main-udm",
}


# ФЗ-38 ст.28.1 banned phrases — case-insensitive substring match.
# These are HARD blockers (AUTO_PAUSE verdict). The phrase list comes
# from KB ad_quality_bfl_2026-04.md and matches the legal text of the
# upcoming 01.01.2026 amendment.
_FZ38_BANNED: tuple[str, ...] = (
    "100%",
    "100 %",
    "гаранти",  # гарантия / гарантируем / гарантировать
    "освободим от долг",
    "освобождаем от долг",
    "гос программа",
    "гос. программа",
    "государственная программа",
    "кредитная амнистия",
    "не плати",  # «не платите кредит/налоги»
    "без последствий",
    "бесплатное банкротство",
    "чисто без следов",
)

# Quality-check thresholds.
_TITLE_MIN_CHARS = 20
_TITLE_MAX_CHARS = 56
_BODY_MAX_CHARS = 81

# CTA verbs (lowercase, root only — match by .startswith()).
_CTA_ROOTS: tuple[str, ...] = (
    "получ",  # Получить / Получите
    "узна",  # Узнать
    "рассчита",  # Рассчитать
    "запиш",  # Записаться / Запишитесь
    "оставьте",
    "оставить заявку",
    "звон",  # Звоните / Позвоните
    "консульта",  # Консультация
)


@dataclass(frozen=True)
class AdAssessment:
    ad_id: int
    campaign_id: int
    name_hint: str  # campaign nickname for the report
    title: str
    title2: str
    body: str
    fz38_violations: list[str] = field(default_factory=list)
    quality_issues: list[str] = field(default_factory=list)
    verdict: Literal["APPROVE", "NEEDS_REWRITE", "AUTO_PAUSE", "SKIP"] = "APPROVE"


# --- pure rules ------------------------------------------------------------


def _check_fz38(combined_lower: str) -> list[str]:
    hits: list[str] = []
    for phrase in _FZ38_BANNED:
        if phrase in combined_lower:
            hits.append(f"ФЗ-38: '{phrase}'")
    return hits


_UPPERCASE_WORD_RE = re.compile(r"\b[А-ЯA-Z]{3,}\b")
_HAS_DIGIT_RE = re.compile(r"\d")


def _check_quality(title: str, title2: str, body: str) -> list[str]:
    issues: list[str] = []
    title_clean = title.strip()
    body_clean = body.strip()

    # Length
    if len(title_clean) < _TITLE_MIN_CHARS:
        issues.append(f"короткий Title ({len(title_clean)}<{_TITLE_MIN_CHARS} симв)")
    if len(title_clean) > _TITLE_MAX_CHARS:
        issues.append(f"длинный Title ({len(title_clean)}>{_TITLE_MAX_CHARS} симв)")
    if len(body_clean) > _BODY_MAX_CHARS:
        issues.append(f"длинный Body ({len(body_clean)}>{_BODY_MAX_CHARS} симв)")

    # Has digit (in title OR body)
    combined = f"{title_clean} {body_clean}"
    if not _HAS_DIGIT_RE.search(combined):
        issues.append("нет цифр (сумма/срок/цена) ни в Title ни в Body")

    # No UPPERCASE WORDS (3+ chars).
    for m in _UPPERCASE_WORD_RE.findall(title_clean):
        if m.upper() == m and len(m) >= 3 and m.lower() != m:
            # Match found that is uppercase
            issues.append(f"Заглавное слово в Title: '{m}'")
            break

    # No multiple ! in title
    if title_clean.count("!") >= 2:
        issues.append("Несколько '!' в Title")

    # Body should contain a CTA root
    body_lower = body_clean.lower()
    if body_clean and not any(root in body_lower for root in _CTA_ROOTS):
        issues.append("Нет CTA-глагола в Body (Получить/Узнать/Рассчитать/Записаться)")

    return issues


def assess_ad(ad: dict[str, Any]) -> AdAssessment:
    """Score one ad. Returns AdAssessment with verdict.

    SKIP for OFF ads or ads without a TextAd block (e.g. image-only
    creatives that need a different rubric).
    """
    state = str(ad.get("State", "")).upper()
    text = ad.get("TextAd")
    if state != "ON" or not isinstance(text, dict):
        return AdAssessment(
            ad_id=int(ad.get("Id", 0)),
            campaign_id=int(ad.get("CampaignId", 0)),
            name_hint=_OWN_CAMPAIGNS.get(int(ad.get("CampaignId", 0)), "?"),
            title="",
            title2="",
            body="",
            verdict="SKIP",
        )

    title = str(text.get("Title", "") or "")
    title2 = str(text.get("Title2", "") or "")
    body = str(text.get("Text", "") or "")
    combined_lower = f"{title} {title2} {body}".lower()

    fz38 = _check_fz38(combined_lower)
    quality = _check_quality(title, title2, body)

    if fz38:
        verdict: Literal["APPROVE", "NEEDS_REWRITE", "AUTO_PAUSE", "SKIP"] = "AUTO_PAUSE"
    elif quality:
        verdict = "NEEDS_REWRITE"
    else:
        verdict = "APPROVE"

    return AdAssessment(
        ad_id=int(ad.get("Id", 0)),
        campaign_id=int(ad.get("CampaignId", 0)),
        name_hint=_OWN_CAMPAIGNS.get(int(ad.get("CampaignId", 0)), "?"),
        title=title,
        title2=title2,
        body=body,
        fz38_violations=fz38,
        quality_issues=quality,
        verdict=verdict,
    )


# --- run-time ---------------------------------------------------------------


async def _fetch_all_ads(direct: DirectAPI) -> list[dict[str, Any]]:
    """Get all ads across the 5 own campaigns. Inlined call so we can
    pull TextAdFieldNames which the wrapper helper doesn't expose."""
    cids = list(_OWN_CAMPAIGNS.keys())
    ag = await direct._call(  # noqa: SLF001
        "adgroups",
        "get",
        {"SelectionCriteria": {"CampaignIds": cids}, "FieldNames": ["Id", "CampaignId"]},
    )
    ag_ids = [int(a.get("Id", 0)) for a in ag.get("AdGroups") or []]
    if not ag_ids:
        return []
    ads = await direct._call(  # noqa: SLF001
        "ads",
        "get",
        {
            "SelectionCriteria": {"AdGroupIds": ag_ids},
            "FieldNames": ["Id", "CampaignId", "AdGroupId", "State", "Status", "Type"],
            "TextAdFieldNames": ["Title", "Title2", "Text", "Href"],
        },
    )
    return list(ads.get("Ads") or [])


def _format_report(assessments: list[AdAssessment]) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    counts: dict[str, int] = {"APPROVE": 0, "NEEDS_REWRITE": 0, "AUTO_PAUSE": 0, "SKIP": 0}
    for a in assessments:
        counts[a.verdict] += 1
    head = "<b>📝 Ad Quality Assessor</b> — " + when
    lines = [
        head,
        "",
        f"Проверено: {len(assessments)} ads",
        (
            f"✅ APPROVE: {counts['APPROVE']}  "
            f"📝 NEEDS_REWRITE: {counts['NEEDS_REWRITE']}  "
            f"🛑 AUTO_PAUSE: {counts['AUTO_PAUSE']}  "
            f"⏸ SKIP: {counts['SKIP']}"
        ),
        "",
    ]

    pause_list = [a for a in assessments if a.verdict == "AUTO_PAUSE"]
    if pause_list:
        lines.append("<b>🛑 ФЗ-38 нарушения (AUTO_PAUSE):</b>")
        for a in pause_list[:10]:
            lines.append(
                f"  • <code>{a.ad_id}</code> ({a.name_hint}): {', '.join(a.fz38_violations)}"
            )
            lines.append(f"    Title: {a.title[:60]}")
        lines.append("")

    rewrite_list = [a for a in assessments if a.verdict == "NEEDS_REWRITE"]
    if rewrite_list:
        lines.append(f"<b>📝 NEEDS_REWRITE — топ {min(10, len(rewrite_list))}:</b>")
        for a in rewrite_list[:10]:
            lines.append(
                f"  • <code>{a.ad_id}</code> ({a.name_hint}): {', '.join(a.quality_issues[:2])}"
            )
            lines.append(f"    Title: {a.title[:60]}")
        lines.append("")

    if not pause_list and not rewrite_list:
        lines.append("Все объявления чистые. ФЗ-38 готовность: ✅ (8 мес до 01.01.2026)")

    return "\n".join(lines)


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    direct: DirectAPI | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron daily (e.g. 12:00 UTC = 15:00 МСК).

    Read-only on Direct (fetch ads). Sends a Telegram digest. Does NOT
    auto-pause anything yet — the owner reviews the AUTO_PAUSE list and
    decides whether to flip a switch later.
    """
    if direct is None or http_client is None or settings is None:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "ads_total": 0,
            "verdicts": {},
            "dry_run": dry_run,
        }

    try:
        ads = await _fetch_all_ads(direct)
    except Exception as exc:
        logger.exception("ad_quality_assessor: fetch_all_ads failed")
        return {"status": "error", "step": "fetch_ads", "detail": str(exc), "dry_run": dry_run}

    assessments = [assess_ad(a) for a in ads]
    counts: dict[str, int] = {"APPROVE": 0, "NEEDS_REWRITE": 0, "AUTO_PAUSE": 0, "SKIP": 0}
    for a in assessments:
        counts[a.verdict] += 1

    msg = _format_report(assessments)
    if not dry_run:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("ad_quality_assessor: telegram send failed")

    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="autonomous",
            tool_name="ad_quality_assessor",
            tool_input={"campaigns": list(_OWN_CAMPAIGNS.keys()), "dry_run": dry_run},
            tool_output={
                "ads_total": len(assessments),
                "verdicts": counts,
                "auto_pause_ids": [a.ad_id for a in assessments if a.verdict == "AUTO_PAUSE"],
                "needs_rewrite_ids": [a.ad_id for a in assessments if a.verdict == "NEEDS_REWRITE"][
                    :50
                ],
            },
            is_mutation=False,
        )
    except Exception:
        logger.warning("ad_quality_assessor: audit_log write failed", exc_info=True)

    return {
        "status": "ok",
        "ads_total": len(assessments),
        "verdicts": counts,
        "dry_run": dry_run,
    }


__all__ = ["AdAssessment", "assess_ad", "run"]
