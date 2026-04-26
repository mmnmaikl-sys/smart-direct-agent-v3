"""Landing critic — pure-rule HTML check + ФЗ-38 disclaimer detector.

v0 (this version): text-only via httpx fetch. Checks the page HTML for:
  * H1 with number/region/city
  * Lead-form (input/form tags) in first 5000 chars (≈ above the fold)
  * Trust markers (СРО, ИНН, ОГРН)
  * ФЗ-38 disclaimer (mandatory from 01.01.2026)
  * ФЗ-38 banned phrases in body (same list as ad_quality_assessor)
  * Page weight + meta viewport (mobile-first hint)

v1 (future): Playwright + Sonnet Vision for visual rubric (color, lice
ЦА, text % of area). v0 covers the legal/structural priorities now.

Reads PROTECTED_LANDING_URLS from settings, scores each, sends Telegram
digest. Read-only — no mutations to Direct.
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

logger = logging.getLogger(__name__)


# Same banned-phrase list as ad_quality_assessor — text on landing must
# also comply (and even more strictly, since it's the destination).
_FZ38_BANNED: tuple[str, ...] = (
    "100%",
    "100 %",
    "гаранти",
    "освободим от долг",
    "освобождаем от долг",
    "гос программа",
    "гос. программа",
    "государственная программа",
    "кредитная амнистия",
    "не плати",
    "без последствий",
    "бесплатное банкротство",
    "чисто без следов",
)

# Mandatory disclaimer text (ФЗ-38 ст.28.1, effective 01.01.2026).
# Match by partial substrings — exact wording can vary, but core
# anchors must be there.
_DISCLAIMER_ANCHORS: tuple[str, ...] = (
    "негативные последствия",
    "ограничения на получение кредита",
    "повторное банкротство",
)

# Trust markers — at least 1 from the СРО group + 1 ИНН/ОГРН
# expected on a legitimate БФЛ landing.
_SRO_HINTS: tuple[str, ...] = (
    "сро",
    "арбитражн",
    "лиценз",
)
_REGISTRY_HINTS: tuple[str, ...] = (
    "инн",
    "огрн",
    "огрнип",
)


_INN_OGRN_RE = re.compile(r"\b(?:ИНН|ОГРН|ОГРНИП)\s*\d{10,15}\b", re.IGNORECASE)
_FORM_TAG_RE = re.compile(r"<(?:form|input|button)\b", re.IGNORECASE)
_VIEWPORT_RE = re.compile(
    r'<meta\s+name=["\']viewport["\']\s+content=["\'][^"\']*width=device-width', re.IGNORECASE
)
_H1_RE = re.compile(r"<h1[^>]*>([\s\S]*?)</h1>", re.IGNORECASE)
_HAS_DIGIT_RE = re.compile(r"\d")
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class LandingAssessment:
    url: str
    http_status: int
    fz38_violations: list[str] = field(default_factory=list)
    structural_issues: list[str] = field(default_factory=list)
    verdict: Literal["APPROVE", "NEEDS_REWRITE", "AUTO_FLAG", "FETCH_ERROR"] = "APPROVE"


def _strip_tags(html: str) -> str:
    return _TAG_STRIP_RE.sub(" ", html)


def assess_landing(url: str, status: int, html: str) -> LandingAssessment:
    if status >= 400 or not html:
        return LandingAssessment(
            url=url,
            http_status=status,
            structural_issues=[f"HTTP {status} or empty body"],
            verdict="FETCH_ERROR",
        )

    head_only = html[:5000]  # above the fold heuristic
    visible_lower = _strip_tags(html).lower()

    fz38: list[str] = []
    for phrase in _FZ38_BANNED:
        if phrase in visible_lower:
            fz38.append(f"ФЗ-38: '{phrase}'")

    issues: list[str] = []

    # H1 with number / region
    h1_match = _H1_RE.search(html)
    if not h1_match:
        issues.append("нет <h1> на странице")
    else:
        h1_text = _strip_tags(h1_match.group(1)).strip()
        if not _HAS_DIGIT_RE.search(h1_text):
            issues.append("в <h1> нет цифры (сумма/срок/город)")

    # Lead form above the fold
    if not _FORM_TAG_RE.search(head_only):
        issues.append("нет <form>/<input>/<button> в первых 5000 байт (выше viewport)")

    # Mobile viewport
    if not _VIEWPORT_RE.search(html):
        issues.append("нет <meta viewport=device-width> — mobile сломан")

    # Trust: СРО + ИНН/ОГРН
    has_sro = any(h in visible_lower for h in _SRO_HINTS)
    has_registry = bool(_INN_OGRN_RE.search(html)) or any(
        h in visible_lower for h in _REGISTRY_HINTS
    )
    if not has_sro:
        issues.append("нет упоминания СРО / арбитражного управляющего")
    if not has_registry:
        issues.append("нет ИНН / ОГРН / ОГРНИП в видимом тексте")

    # ФЗ-38 disclaimer
    disclaimer_hits = sum(1 for anchor in _DISCLAIMER_ANCHORS if anchor in visible_lower)
    if disclaimer_hits < 2:
        issues.append(
            f"нет дисклеймера ФЗ-38 ст.28.1 (anchors {disclaimer_hits}/3) — обязателен с 01.01.2026"
        )

    if fz38:
        verdict: Literal["APPROVE", "NEEDS_REWRITE", "AUTO_FLAG", "FETCH_ERROR"] = "AUTO_FLAG"
    elif issues:
        verdict = "NEEDS_REWRITE"
    else:
        verdict = "APPROVE"

    return LandingAssessment(
        url=url,
        http_status=status,
        fz38_violations=fz38,
        structural_issues=issues,
        verdict=verdict,
    )


async def _fetch_landing(http: httpx.AsyncClient, url: str) -> tuple[int, str]:
    try:
        # Mobile UA — БФЛ ниша 70-85% mobile, must check mobile rendering.
        resp = await http.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
                    " AppleWebKit/605.1.15 (KHTML, like Gecko)"
                    " Version/17.0 Mobile/15E148 Safari/604.1"
                )
            },
        )
        return (resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("landing_critic: fetch %s failed: %s", url, exc)
        return (0, "")


def _format_report(assessments: list[LandingAssessment]) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    counts: dict[str, int] = {
        "APPROVE": 0,
        "NEEDS_REWRITE": 0,
        "AUTO_FLAG": 0,
        "FETCH_ERROR": 0,
    }
    for a in assessments:
        counts[a.verdict] += 1
    lines = [
        f"<b>🌐 Landing Critic</b> — {when}",
        "",
        f"Проверено: {len(assessments)} лендингов",
        (
            f"✅ APPROVE: {counts['APPROVE']}  "
            f"📝 NEEDS_REWRITE: {counts['NEEDS_REWRITE']}  "
            f"🛑 AUTO_FLAG: {counts['AUTO_FLAG']}  "
            f"⚠️ FETCH_ERROR: {counts['FETCH_ERROR']}"
        ),
        "",
    ]
    for a in assessments:
        url_short = a.url.replace("https://", "").replace("http://", "")[:60]
        if a.verdict == "AUTO_FLAG":
            lines.append(f"🛑 <code>{url_short}</code>:")
            for v in a.fz38_violations:
                lines.append(f"    {v}")
        elif a.verdict == "NEEDS_REWRITE":
            lines.append(f"📝 <code>{url_short}</code>:")
            for issue in a.structural_issues[:3]:
                lines.append(f"    • {issue}")
        elif a.verdict == "FETCH_ERROR":
            lines.append(f"⚠️ <code>{url_short}</code>: HTTP {a.http_status}")
        else:  # APPROVE
            lines.append(f"✅ <code>{url_short}</code>")
    return "\n".join(lines)


def _resolve_landing_urls(settings: Settings) -> list[str]:
    raw: Any = getattr(settings, "PROTECTED_LANDING_URLS", None) or []
    if isinstance(raw, str):
        # Comma- or JSON-parseable string.
        if raw.strip().startswith("["):
            import json

            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []
        else:
            raw = [u.strip() for u in raw.split(",") if u.strip()]
    if not isinstance(raw, list):
        return []
    return [str(u) for u in raw if u]


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron weekly (Mon 13:00 UTC = 16:00 МСК)."""
    if http_client is None or settings is None:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "landings": 0,
            "verdicts": {},
            "dry_run": dry_run,
        }

    urls = _resolve_landing_urls(settings)
    if not urls:
        return {
            "status": "ok",
            "action": "no_landings",
            "landings": 0,
            "verdicts": {},
            "dry_run": dry_run,
        }

    assessments: list[LandingAssessment] = []
    for url in urls:
        status, html = await _fetch_landing(http_client, url)
        assessments.append(assess_landing(url, status, html))

    counts: dict[str, int] = {"APPROVE": 0, "NEEDS_REWRITE": 0, "AUTO_FLAG": 0, "FETCH_ERROR": 0}
    for a in assessments:
        counts[a.verdict] += 1

    msg = _format_report(assessments)
    if not dry_run:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("landing_critic: telegram send failed")

    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="autonomous",
            tool_name="landing_critic",
            tool_input={"urls": urls, "dry_run": dry_run},
            tool_output={
                "landings": len(assessments),
                "verdicts": counts,
                "fz38_flags": [a.url for a in assessments if a.verdict == "AUTO_FLAG"],
            },
            is_mutation=False,
        )
    except Exception:
        logger.warning("landing_critic: audit_log write failed", exc_info=True)

    return {
        "status": "ok",
        "landings": len(assessments),
        "verdicts": counts,
        "dry_run": dry_run,
    }


__all__ = ["LandingAssessment", "assess_landing", "run"]
