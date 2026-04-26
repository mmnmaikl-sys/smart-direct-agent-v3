"""LLM ad rewriter — Sonnet переписывает top-N проблемных ads по KB rubric.

Использует pure-rule `ad_quality_assessor.assess_ad` для классификации,
затем для top-3 (по приоритету AUTO_PAUSE > NEEDS_REWRITE) дёргает
Claude Sonnet с system prompt из KB ad_quality_bfl_2026-04.md.

Возвращает rewrite-предложения в Telegram (original | issue | rewritten
side-by-side). НЕ применяет в Direct автоматически — owner копирует
вручную или одобряет следующим итерационным actuator-ом (отдельный
будущий job).

Cost cap: top-3 ads × ~600 tokens output × Sonnet ≈ $0.05/cycle.
Cron: weekly Mon 13:30 UTC (после landing_critic) — overall ad
rewrite — медленный процесс, не нужен ежедневно.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agent_runtime.jobs.ad_quality_assessor import (
    AdAssessment,
    _fetch_all_ads,
    assess_ad,
)
from agent_runtime.tools import telegram as telegram_tools

if TYPE_CHECKING:
    import httpx
    from agents_core.llm.client import LLMClient
    from psycopg_pool import AsyncConnectionPool

    from agent_runtime.config import Settings
    from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)


_TOP_N_TO_REWRITE = 3  # cost cap

SYSTEM_PROMPT = """Ты — директолог-эксперт в нише банкротство физических лиц (БФЛ) с 9 годами опыта.

Твоя задача — переписать рекламное объявление Яндекс.Директа чтобы оно:

1. **НЕ содержало запрещённых ФЗ-38 ст.28.1 фраз** (вступает в силу 01.01.2026):
   - "100%", "100 %", "гарантия", "гарантируем", "освободим от долгов"
   - "гос. программа", "государственная программа", "кредитная амнистия"
   - "не плати", "не платите", "без последствий"
   - "бесплатное банкротство", "чисто без следов"

2. **Title (25-56 символов)** — содержит:
   - Число (сумма долга, срок процедуры, цена услуги)
   - Регион/город если возможно
   - НЕ ЗАГЛАВНЫЕ слова, НЕ "!!!"
   - НЕТ превосходной степени без обоснования

3. **Body (макс 81 символ)** — содержит:
   - CTA-глагол: "Получить", "Узнать", "Рассчитать", "Записаться", "Звоните"
   - Минимум 1 цифру (сумма / срок / цена)
   - Структура: проблема → решение → CTA

4. **Title2 (макс 30 символов)** — дополняет Title, не дублирует.

КОНТЕКСТ:
- ЦА: физлица 35-55, доход 30-80К/мес, долг 250К-3млн
- Целевая страница: 24bankrotsttvo.ru (двойная "т" — это правильно)
- Бенчмарк CPL: 625-684₽ (vc.ru кейс)

ВЕРНИ строго JSON (никакого markdown):
{
  "title": "переписанный Title 25-56 симв",
  "title2": "переписанный Title2 ≤30 симв",
  "body": "переписанный Body ≤81 симв",
  "rationale": "Кратко (1-2 предложения): что изменил и почему."
}

НИКОГДА не возвращай rewritten с запрещёнными фразами. Лучше переписать спорный
вариант, чем получить штраф ФАС."""


@dataclass(frozen=True)
class RewriteSuggestion:
    ad_id: int
    name_hint: str
    verdict: str
    original_title: str
    original_title2: str
    original_body: str
    issues: list[str]
    new_title: str = ""
    new_title2: str = ""
    new_body: str = ""
    rationale: str = ""
    error: str = ""


def _build_user_prompt(a: AdAssessment) -> str:
    issues_str = "\n".join(f"  • {i}" for i in (a.fz38_violations + a.quality_issues))
    return (
        f"Кампания: {a.name_hint} (cid={a.campaign_id})\n\n"
        f"ОРИГИНАЛЬНОЕ ОБЪЯВЛЕНИЕ:\n"
        f"  Title:  {a.title}\n"
        f"  Title2: {a.title2}\n"
        f"  Body:   {a.body}\n\n"
        f"НАЙДЕННЫЕ ПРОБЛЕМЫ (verdict={a.verdict}):\n{issues_str}\n\n"
        f"Перепиши строго в JSON формате (см. system prompt)."
    )


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _parse_llm_json(text: str) -> dict[str, str]:
    """Tolerantly parse JSON from LLM response — strip markdown fences if present."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    # Fallback: find any {...} block
    if not cleaned.startswith("{"):
        m = _JSON_BLOCK_RE.search(cleaned)
        if m:
            cleaned = m.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "title": str(data.get("title", "")).strip(),
        "title2": str(data.get("title2", "")).strip(),
        "body": str(data.get("body", "")).strip(),
        "rationale": str(data.get("rationale", "")).strip(),
    }


def _select_top_n(assessments: list[AdAssessment], n: int) -> list[AdAssessment]:
    """AUTO_PAUSE first (legal risk), then NEEDS_REWRITE; cap at N."""
    pause = [a for a in assessments if a.verdict == "AUTO_PAUSE"]
    rewrite = [a for a in assessments if a.verdict == "NEEDS_REWRITE"]
    return (pause + rewrite)[:n]


async def _rewrite_one(client: LLMClient, a: AdAssessment) -> RewriteSuggestion:
    prompt = _build_user_prompt(a)
    try:
        resp = await client.chat(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            model="sonnet",
            max_tokens=600,
            name="ad_rewriter",
        )
        text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as exc:
        return RewriteSuggestion(
            ad_id=a.ad_id,
            name_hint=a.name_hint,
            verdict=a.verdict,
            original_title=a.title,
            original_title2=a.title2,
            original_body=a.body,
            issues=list(a.fz38_violations + a.quality_issues),
            error=f"LLM error: {type(exc).__name__}",
        )
    parsed = _parse_llm_json(text)
    return RewriteSuggestion(
        ad_id=a.ad_id,
        name_hint=a.name_hint,
        verdict=a.verdict,
        original_title=a.title,
        original_title2=a.title2,
        original_body=a.body,
        issues=list(a.fz38_violations + a.quality_issues),
        new_title=parsed.get("title", ""),
        new_title2=parsed.get("title2", ""),
        new_body=parsed.get("body", ""),
        rationale=parsed.get("rationale", ""),
        error="" if parsed else "JSON parse failed",
    )


def _format_report(suggestions: list[RewriteSuggestion], total_problems: int) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"<b>✍️ Ad Rewriter (Sonnet)</b> — {when}",
        "",
        f"Проблемных ads: {total_problems}. Переписано (top-{len(suggestions)}):",
        "",
    ]
    if not suggestions:
        lines.append("Все ads чистые — нечего переписывать. ✅")
        return "\n".join(lines)
    for i, s in enumerate(suggestions, 1):
        emoji = "🛑" if s.verdict == "AUTO_PAUSE" else "📝"
        lines.append(f"<b>{i}. {emoji} ad_id={s.ad_id} ({s.name_hint}) — {s.verdict}</b>")
        if s.error:
            lines.append(f"  ⚠️ {s.error}")
            lines.append("")
            continue
        lines.append(f"  <i>Проблемы:</i> {', '.join(s.issues[:2])}")
        lines.append("")
        lines.append("  <b>БЫЛО:</b>")
        lines.append(f"    Title:  <code>{s.original_title[:80]}</code>")
        if s.original_title2:
            lines.append(f"    Title2: <code>{s.original_title2[:60]}</code>")
        lines.append(f"    Body:   <code>{s.original_body[:120]}</code>")
        lines.append("  <b>СТАЛО:</b>")
        lines.append(f"    Title:  <code>{s.new_title[:80]}</code>")
        if s.new_title2:
            lines.append(f"    Title2: <code>{s.new_title2[:60]}</code>")
        lines.append(f"    Body:   <code>{s.new_body[:120]}</code>")
        if s.rationale:
            lines.append(f"  <i>{s.rationale[:200]}</i>")
        lines.append("")
    lines.append(
        "Скопируй текст в Direct UI вручную — авто-применение пока выключено "
        "(будет в следующем actuator)."
    )
    return "\n".join(lines)


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    direct: DirectAPI | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron weekly Mon 13:30 UTC (after landing_critic)."""
    if direct is None or http_client is None or settings is None or llm_client is None:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "rewritten": 0,
            "total_problems": 0,
            "dry_run": dry_run,
        }

    try:
        ads = await _fetch_all_ads(direct)
    except Exception as exc:
        logger.exception("ad_rewriter: fetch_all_ads failed")
        return {"status": "error", "step": "fetch_ads", "detail": str(exc), "dry_run": dry_run}

    assessments = [assess_ad(a) for a in ads]
    problems = [a for a in assessments if a.verdict in ("AUTO_PAUSE", "NEEDS_REWRITE")]
    selected = _select_top_n(problems, _TOP_N_TO_REWRITE)

    suggestions: list[RewriteSuggestion] = []
    for a in selected:
        suggestions.append(await _rewrite_one(llm_client, a))

    msg = _format_report(suggestions, total_problems=len(problems))
    if not dry_run and suggestions:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("ad_rewriter: telegram send failed")

    return {
        "status": "ok",
        "rewritten": len([s for s in suggestions if s.new_title]),
        "total_problems": len(problems),
        "dry_run": dry_run,
    }


__all__ = ["RewriteSuggestion", "run"]
