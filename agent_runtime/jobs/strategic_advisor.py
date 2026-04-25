"""Strategic Advisor — «думающий» аналитик для 5 кампаний 24bankrotsttvo.

Каждые 4 часа собирает stats + читает knowledge base + просит Claude Sonnet
дать 3-5 конкретных рекомендаций. Отправляет в Telegram владельцу.

Это NOTIFY-only job: НЕ мутирует кампании. Уровень автономии ниже даже
form_checker — только аналитика, без действий.

Сигнатура соответствует другим Wave 3 jobs:
``(pool, *, dry_run=False, http_client, settings, direct, llm_client)``.
``degraded_noop`` если DI не построен.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agent_runtime.knowledge import consult as kb_consult
from agent_runtime.tools import telegram as telegram_tools

if TYPE_CHECKING:
    import httpx
    from agents_core.llm.client import LLMClient
    from psycopg_pool import AsyncConnectionPool

    from agent_runtime.config import Settings
    from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)


CAMPAIGN_NAMES: dict[int, str] = {
    709353005: "rabotyaga",
    709353034: "pensioner",
    709353058: "mother",
    709353078: "mfo",
    709353099: "property",
}

SYSTEM_PROMPT = (
    "Ты — директолог-эксперт с 9 годами опыта в нише банкротство физических лиц (БФЛ). "
    "Знаешь наизусть курсы Идалина (21 урок), Цымбалиста (9 глав), Абрамовой (12 уроков), "
    "приёмы Presnyakov (30 шт). Цель: дать владельцу 24bankrotsttvo КОНКРЕТНЫЕ рекомендации "
    "по 5 кампаниям Я.Директа, основанные на ДАННЫХ.\n\n"
    "Правила:\n"
    "1. Только русский язык, по-человечески (без терминов CTR/CPA без расшифровки).\n"
    "2. Каждая рекомендация цитирует источник (Идалин §6, Цымбалист 11.5, Presnyakov #11).\n"
    "3. Конкретика: «увеличить bid на rabotyaga на 20%», не «оптимизировать ставки».\n"
    "4. Приоритет: что критично сейчас → что желательно потом.\n"
    "5. Не более 5 рекомендаций. Каждая ≤ 2 предложений.\n"
    "6. Учитывай минимальные sample sizes (нельзя судить по 14 кликам — нужно 100+).\n"
    "7. Формат: «Кампания: что делать. Почему. Источник.»"
)


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
    direct: DirectAPI | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Анализирует 5 кампаний + KB → Telegram-рекомендации."""
    if http_client is None or settings is None or direct is None or llm_client is None:
        return {
            "status": "degraded_noop",
            "reason": "missing http_client / settings / direct / llm_client",
        }

    started = datetime.now(UTC)

    # 1) Stats всех 5 PROTECTED кампаний
    cids = [cid for cid in CAMPAIGN_NAMES.keys()]
    try:
        camps = await direct.get_campaigns(cids)
    except Exception as exc:
        logger.exception("strategic_advisor: get_campaigns failed")
        return {"status": "error", "step": "get_campaigns", "detail": str(exc)}

    stats_lines = ["=== STATS КАМПАНИЙ (Direct API, накопленные) ==="]
    for c in camps:
        cid_raw = c.get("Id")
        if cid_raw is None:
            continue
        cid = int(cid_raw)
        name = CAMPAIGN_NAMES.get(cid, str(cid))
        s = c.get("Statistics", {}) or {}
        impr = int(s.get("Impressions", 0) or 0)
        clicks = int(s.get("Clicks", 0) or 0)
        ctr_pct = round(100 * clicks / impr, 2) if impr else 0
        stats_lines.append(
            f"  {name:9s} (cid={cid}) state={c.get('State')} "
            f"impr={impr} clicks={clicks} CTR={ctr_pct}%"
        )

    stats_block = "\n".join(stats_lines)

    # 2) KB excerpt (короткий, 4 ключевых принципа)
    try:
        kb = await kb_consult(
            "общий обзор: правила оптимизации БФЛ-кампаний по Идалину/Цымбалисту/Presnyakov",
            context={"task": "strategic_advisor"},
        )
        kb_text = kb.get("answer", "")[:4000] if kb else ""
    except Exception:
        logger.info("strategic_advisor: kb.consult skipped")
        kb_text = ""

    # 3) Спрашиваем Claude Sonnet
    user_prompt = (
        f"{stats_block}\n\n"
        f"=== ВЫЖИМКА БАЗЫ ЗНАНИЙ ===\n{kb_text}\n\n"
        "Дай 3-5 рекомендаций по формату: «Кампания: что делать. Почему. Источник.»"
    )

    try:
        # agents_core.LLMClient.chat — prompt + system + model + max_tokens
        # Возвращает LLMResponse с полем .text
        response = await llm_client.chat(
            prompt=user_prompt,
            system=SYSTEM_PROMPT,
            model="sonnet",
            max_tokens=2000,
            name="strategic_advisor.chat",
        )
        advice = response.text if hasattr(response, "text") else str(response)
    except Exception as exc:
        logger.exception("strategic_advisor: LLM call failed")
        return {"status": "error", "step": "llm", "detail": str(exc)}

    # 4) Telegram digest
    msg = (
        f"🧠 <b>Strategic Advisor</b> — {started.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"<pre>{stats_block}</pre>\n\n"
        f"<b>Рекомендации:</b>\n{advice}"
    )

    if not dry_run:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception as exc:
            logger.exception("strategic_advisor: telegram send failed")
            return {
                "status": "partial",
                "stats": stats_block,
                "advice_len": len(advice),
                "telegram_error": str(exc),
            }

    return {
        "status": "ok",
        "campaigns_analysed": len(camps),
        "advice_chars": len(advice),
        "kb_chars": len(kb_text),
        "telegram_sent": not dry_run,
    }
