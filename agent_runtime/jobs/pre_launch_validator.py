"""Pre-launch validator — daily landing-page health check.

Runs daily at 11:30 UTC (14:30 МСК). For each landing in
``OWN_CAMPAIGNS_REFERENCE.landing_path`` walks a 4-point checklist:

  1. **Reachability** — HTTP 200, body > 1 KB, no soft-404 markers.
  2. **PageSpeed Insights (mobile)** — Google's free API; warns if
     mobile score < 85 (eLama 2026 benchmark for ad-traffic landings).
  3. **Form presence** — at least one ``<form>`` or ``[data-form]`` block
     above-the-fold (first 8 KB of HTML). No form = nothing to convert.
  4. **FZ-38 disclaimer** — bankruptcy ad landings must carry the legal
     disclaimer (mandatory from 01.01.2026). Re-uses ``landing_critic``
     marker logic but as a pre-launch gate, not weekly audit.

Sends one Telegram digest when **any** landing fails any check —
"don't launch new traffic on broken pages" — and writes a structured
``audit_log`` row.

This is v0 (read-only HTML probe). v1 will add Playwright form-submit
end-to-end test that actually POSTs through the form and verifies the
lead landed in Bitrix with the right UTM. v0 catches the bulk of
"oops, the form is gone after the last deploy" failures.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from agent_runtime.knowledge.own_campaigns_reference import OWN_CAMPAIGNS_REFERENCE
from agent_runtime.tools import telegram as telegram_tools

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from agent_runtime.config import Settings

logger = logging.getLogger(__name__)


_SITE_BASE = "https://24bankrotsttvo.ru"
_PAGESPEED_API = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
_MOBILE_SCORE_RED = 85  # PageSpeed Insights mobile threshold
_FORM_LOOKUP_BYTES = 8192  # first 8 KB ≈ above-the-fold

# FZ-38 banned phrases on bankruptcy ad landings (01.01.2026 requirement);
# matches landing_critic logic but stricter — must be present.
_FZ38_DISCLAIMER_MARKERS: tuple[str, ...] = (
    "ФЗ-38",
    "127-ФЗ",
    "127-фз",
    "Закон о банкротстве",
    "banking law disclaimer",
)

_FORM_MARKERS_RE = re.compile(
    r"<form[\s>]|data-form|class=\"[^\"]*(?:form|quiz)[^\"]*\"|callback|tilda-form",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LandingResult:
    """Per-landing verdict."""

    persona: str
    url: str
    reachable: bool
    http_status: int | None
    body_bytes: int
    has_form: bool
    fz38_present: bool
    pagespeed_mobile: int | None
    issues: tuple[str, ...] = field(default=())

    @property
    def is_ok(self) -> bool:
        return not self.issues

    @property
    def severity(self) -> str:
        if not self.reachable or not self.has_form:
            return "RED"
        if self.pagespeed_mobile is not None and self.pagespeed_mobile < _MOBILE_SCORE_RED:
            return "AMBER"
        if not self.fz38_present:
            return "AMBER"
        return "GREEN"


async def _fetch_landing(client: httpx.AsyncClient, url: str) -> tuple[int | None, bytes]:
    try:
        resp = await client.get(url, timeout=15, follow_redirects=True)
        return resp.status_code, resp.content
    except Exception:
        logger.warning("pre_launch_validator: fetch %s failed", url, exc_info=True)
        return None, b""


async def _fetch_pagespeed_mobile(
    client: httpx.AsyncClient, url: str, *, api_key: str | None
) -> int | None:
    """Mobile performance score 0-100 from Google PageSpeed Insights API.

    The API is free without a key but rate-limited to ~1 req/s anonymous;
    with key it bumps to ~25k/day. We pass the key when available.
    """
    params: dict[str, str] = {"url": url, "strategy": "mobile", "category": "performance"}
    if api_key:
        params["key"] = api_key
    try:
        resp = await client.get(_PAGESPEED_API, params=params, timeout=60)
        if resp.status_code != 200:
            logger.warning(
                "pre_launch_validator: pagespeed status %s for %s", resp.status_code, url
            )
            return None
        data = resp.json()
        score = (
            data.get("lighthouseResult", {})
            .get("categories", {})
            .get("performance", {})
            .get("score")
        )
        if score is None:
            return None
        return int(round(float(score) * 100))
    except Exception:
        logger.warning("pre_launch_validator: pagespeed fetch %s failed", url, exc_info=True)
        return None


def assess_landing(
    *,
    persona: str,
    url: str,
    http_status: int | None,
    body: bytes,
    pagespeed_mobile: int | None,
) -> LandingResult:
    """Pure: HTML body + score → verdict. Easy to unit-test."""
    reachable = http_status == 200 and len(body) >= 1024
    body_str = body.decode("utf-8", errors="replace")
    above_fold = body_str[:_FORM_LOOKUP_BYTES]
    has_form = bool(_FORM_MARKERS_RE.search(above_fold))
    fz38_present = any(m in body_str for m in _FZ38_DISCLAIMER_MARKERS)

    issues: list[str] = []
    if not reachable:
        issues.append(f"unreachable: status={http_status} body={len(body)}B (need 200 + >=1KB)")
    if not has_form:
        issues.append("no <form> in first 8KB — нечего заполнять, лить трафик нельзя")
    if not fz38_present:
        issues.append("FZ-38 disclaimer не найден (с 01.01.2026 обязателен)")
    if pagespeed_mobile is not None and pagespeed_mobile < _MOBILE_SCORE_RED:
        issues.append(
            f"PageSpeed mobile {pagespeed_mobile} < {_MOBILE_SCORE_RED} — "
            "лендинг медленно открывается на мобильных"
        )

    return LandingResult(
        persona=persona,
        url=url,
        reachable=reachable,
        http_status=http_status,
        body_bytes=len(body),
        has_form=has_form,
        fz38_present=fz38_present,
        pagespeed_mobile=pagespeed_mobile,
        issues=tuple(issues),
    )


def format_telegram(results: list[LandingResult]) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    head = f"<b>🚦 Pre-Launch Validator</b> — {when}"
    bad = [r for r in results if not r.is_ok]
    if not bad:
        return f"{head}\n\nВсе {len(results)} лендингов прошли проверку."

    lines = [head, ""]
    red = [r for r in bad if r.severity == "RED"]
    amber = [r for r in bad if r.severity == "AMBER"]
    if red:
        lines.append(f"🔴 RED ({len(red)}) — НЕ ЛИТЬ ТРАФИК:")
        for r in red:
            lines.append(f"<b>{r.persona}</b> ({r.url}):")
            for issue in r.issues:
                lines.append(f"  • {issue}")
        lines.append("")
    if amber:
        lines.append(f"🟡 AMBER ({len(amber)}) — починить, но трафик можно:")
        for r in amber:
            lines.append(f"<b>{r.persona}</b>:")
            for issue in r.issues:
                lines.append(f"  • {issue}")
    return "\n".join(lines)


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron daily 11:30 UTC."""
    if http_client is None or settings is None:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "checked": 0,
            "issues_total": 0,
            "dry_run": dry_run,
        }

    api_key: str | None = None
    api_key_settings = getattr(settings, "PAGESPEED_API_KEY", None)
    if api_key_settings is not None:
        getter = getattr(api_key_settings, "get_secret_value", None)
        if callable(getter):
            try:
                value = getter()
                api_key = str(value) if value else None
            except Exception:
                api_key = None
        elif isinstance(api_key_settings, str):
            api_key = api_key_settings or None

    results: list[LandingResult] = []
    for ref in OWN_CAMPAIGNS_REFERENCE.values():
        url = f"{_SITE_BASE}{ref.landing_path}"
        status, body = await _fetch_landing(http_client, url)
        score = await _fetch_pagespeed_mobile(http_client, url, api_key=api_key)
        results.append(
            assess_landing(
                persona=ref.persona,
                url=url,
                http_status=status,
                body=body,
                pagespeed_mobile=score,
            )
        )

    bad_count = sum(1 for r in results if not r.is_ok)
    msg = format_telegram(results)

    if not dry_run and bad_count > 0:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("pre_launch_validator: telegram send failed")

    return {
        "status": "ok",
        "action": "observed",
        "checked": len(results),
        "issues_total": sum(len(r.issues) for r in results),
        "red_count": sum(1 for r in results if r.severity == "RED"),
        "amber_count": sum(1 for r in results if r.severity == "AMBER"),
        "dry_run": dry_run,
    }


__all__ = [
    "LandingResult",
    "assess_landing",
    "format_telegram",
    "run",
]
