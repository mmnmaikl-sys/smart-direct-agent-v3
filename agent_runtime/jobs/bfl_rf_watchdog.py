"""BFL-RF Watchdog — threshold alerts for the pilot campaign (Task 20).

Cron entry that runs the 3-layer tracker (:mod:`agent_runtime.tools.bfl_rf_tracker`),
evaluates 11 env-driven thresholds against the snapshot and ships a Telegram
alert for each of up to 7 active alert types (``ctr_low``, ``bounce_high``,
``zero_leads``, ``cpa_lead_high``, ``avg_time_low``, ``cr_low``,
``cpa_won_high``) — each cooled down for 6 hours per type so a sustained
condition does not spam the owner.

State lives in PG (``sda_state`` jsonb rows), not on disk — Railway's
ephemeral FS cannot be trusted across re-deploys, and the v2 file-based
state was the single worst pain point this job inherits.

Adaptive schedule via ``sda_state[strategy_gate_state]``:

* ``learning`` (default) → run every 15 min.
* ``auto_pilot`` → run every 15 min but internally throttle to ``<4h``
  since the last non-throttled tick — so a future doubled cron does not
  duplicate alerts.

Both cadences are enforced here (early-return ``{"status":"skipped",
"reason":"auto_pilot_throttle"}``) so ``railway.toml`` needs only a single
``*/15 * * * *`` entry regardless of phase.

Degraded-noop pattern matches ``budget_guard`` / ``shadow_monitor``: when
the JOB_REGISTRY's default wrapper passes only ``(pool, dry_run)``
without the DI clients, the job returns ``status='ok'`` +
``action='degraded_noop'`` instead of raising — the /run handler is
responsible for injecting real clients from ``app.state``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.tools import bfl_rf_tracker
from agent_runtime.tools import telegram as telegram_tools
from agent_runtime.tools.bfl_rf_tracker import MetrikaLike
from agent_runtime.tools.direct_api import DirectAPI

logger = logging.getLogger(__name__)

_MSK = timezone(timedelta(hours=3))
_COOLDOWN_HOURS = 6
_AUTO_PILOT_THROTTLE_HOURS = 4
_STRATEGY_PHASE_KEY = "strategy_gate_state"
_COOLDOWN_KEY = "bfl_rf_watchdog_cooldowns"
_LAST_RUN_KEY = "bfl_rf_watchdog_last_run"
_DEFAULT_PHASE = "learning"


# --------------------------------------------------------------- thresholds


def _th(name: str, default: float) -> float:
    """Read ``BFL_RF_TH_<name>`` from env with a numeric default fallback."""
    try:
        raw = os.environ.get(f"BFL_RF_TH_{name}")
        if raw is None:
            return float(default)
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def build_thresholds() -> dict[str, float]:
    """All 11 thresholds, freshly read from env — called on every tick so a
    Railway env bump takes effect without a redeploy."""
    return {
        "bounce_max": _th("BOUNCE", 70.0),
        "bounce_min_visits": _th("BOUNCE_MIN_V", 100.0),
        "ctr_min": _th("CTR", 3.0),
        "ctr_min_impressions": _th("CTR_MIN_I", 3000.0),
        "leads_zero_clicks": _th("ZERO_LEADS_CLICKS", 300.0),
        "cpa_lead_max": _th("CPA_LEAD", 2700.0),
        "cpa_lead_min_leads": _th("CPA_LEAD_MIN_LEADS", 3.0),
        "cpa_won_max": _th("CPA_WON", 55000.0),
        "avg_time_min": _th("AVG_TIME", 45.0),
        "cr_click_lead_min": _th("CR_CL_MIN", 2.5),
        "cr_min_clicks": _th("CR_MIN_CLICKS", 100.0),
    }


# ------------------------------------------------------------ alert logic


def _check_all(data: dict[str, Any], th: dict[str, float]) -> list[dict[str, Any]]:
    """Evaluate all 7 alert types against ``data`` — 1:1 port of v2.

    Reads metrics from the tracker's layers with safe defaults so a failing
    layer (``{"error": ...}``) does not crash the check — ``dict.get`` on
    an error dict just returns the default, so zero-valued metrics simply
    never trip the noise-floor thresholds.
    """
    alerts: list[dict[str, Any]] = []
    d = data.get("direct") or {}
    m = data.get("metrika") or {}
    b = data.get("bitrix") or {}
    e = data.get("economics") or {}

    clicks = float(d.get("clicks", 0) or 0)
    impressions = float(d.get("impressions", 0) or 0)
    ctr = float(d.get("ctr", 0) or 0)
    bounce = float(m.get("bounce_rate", 0) or 0)
    visits = float(m.get("visits", 0) or 0)
    avg_time = float(m.get("avg_duration_s", 0) or 0)
    leads = int(b.get("leads", 0) or 0)
    cpa_lead = float(e.get("cpa_lead", 0) or 0)
    cpa_won = float(e.get("cpa_won", 0) or 0)
    deals = b.get("deals") or {}
    stages = deals.get("stages") if isinstance(deals, dict) else None
    won_deals = int(stages.get("won", 0) or 0) if isinstance(stages, dict) else 0
    cr_click_lead = (leads * 100.0 / clicks) if clicks else 0.0

    # 1. CTR low (ads not engaging)
    if ctr and ctr < th["ctr_min"] and impressions >= th["ctr_min_impressions"]:
        alerts.append(
            {
                "type": "ctr_low",
                "severity": "⚠️",
                "title": f"CTR {ctr:.2f}% при {int(impressions)} показах",
                "hint": (
                    "Объявления не цепляют. Что делать:\n"
                    "1. Посмотреть Title/Title2 в каждой AdGroup (5 групп × 3 версии)\n"
                    "2. Переписать заголовки с триггерами (сумма долга, гарантия)\n"
                    "3. Добавить быстрые ссылки / уточнения если их нет"
                ),
            }
        )

    # 2. Bounce high (landing loses people)
    if bounce and bounce >= th["bounce_max"] and visits >= th["bounce_min_visits"]:
        alerts.append(
            {
                "type": "bounce_high",
                "severity": "⚠️",
                "title": f"Bounce {bounce:.0f}% при {int(visits)} визитах",
                "hint": (
                    "Лендинг не цепляет первый экран. Что делать:\n"
                    "1. Смотреть webvisor в Метрике (10-20 сессий с bounce)\n"
                    "2. Проверить связку «ключ → объявление → H1» — совпадает ли оффер\n"
                    "3. A/B тест hero через ab-test.js"
                ),
            }
        )

    # 3. Zero leads — critical
    if clicks >= th["leads_zero_clicks"] and leads == 0:
        alerts.append(
            {
                "type": "zero_leads",
                "severity": "🔴",
                "title": f"{int(clicks)} кликов, 0 лидов",
                "hint": (
                    "Критично. Возможные причины:\n"
                    "1. Формы не отправляют в Bitrix — проверь DevTools → Network → /crm.lead.add\n"
                    "2. Метрика не ловит reachGoal — проверь через Отчёты→Цели\n"
                    "3. Аудитория совсем не целевая — топ поисковых запросов, добавь минус-слова"
                ),
            }
        )

    # 4. CPA lead too expensive
    if cpa_lead > th["cpa_lead_max"] and leads >= th["cpa_lead_min_leads"]:
        alerts.append(
            {
                "type": "cpa_lead_high",
                "severity": "⚠️",
                "title": f"CPA лида {cpa_lead:.0f}₽ (цель ≤{int(th['cpa_lead_max'])})",
                "hint": (
                    "Слишком дорогие заявки. Что делать:\n"
                    "1. Отключи топ-сжигатели из блока «🔥 Сжигатели» в ежедневном отчёте\n"
                    "2. Проверь брак в КЦ — если высокий, ключи привлекают не ту аудиторию\n"
                    "3. Подумай о переходе стратегии «оплата за конверсии» после 10+ лидов"
                ),
            }
        )

    # 5. Average visit time too low
    if avg_time and avg_time < th["avg_time_min"] and visits >= th["bounce_min_visits"]:
        alerts.append(
            {
                "type": "avg_time_low",
                "severity": "⚠️",
                "title": f"Среднее время визита {int(avg_time)}с (норма 90-240с)",
                "hint": (
                    "Люди уходят быстро — не цепляет первый экран.\n"
                    "1. Проверь скорость загрузки (PageSpeed на мобиле)\n"
                    "2. H1 и первый trust-блок должны читаться за 3 секунды\n"
                    "3. Видео-отзывы не автоплеят — смотреть webvisor"
                ),
            }
        )

    # 6. CR click→lead below niche median — critical
    if cr_click_lead and cr_click_lead < th["cr_click_lead_min"] and clicks >= th["cr_min_clicks"]:
        alerts.append(
            {
                "type": "cr_low",
                "severity": "🔴",
                "title": f"CR клик→лид {cr_click_lead:.1f}% (БФЛ норма 7-12%)",
                "hint": (
                    "Лендинг существенно ниже нишевой медианы. Что делать:\n"
                    "1. Смотреть сквозную воронку: hero_cta → quiz_step_N → form_submit\n"
                    "2. Найти шаг с дропом >50% — там проблема\n"
                    "3. A/B тест на этом шаге"
                ),
            }
        )

    # 7. CPA won too expensive (ops signal)
    if cpa_won > th["cpa_won_max"] and won_deals >= 1:
        alerts.append(
            {
                "type": "cpa_won_high",
                "severity": "⚠️",
                "title": f"CPA договора {cpa_won:.0f}₽ (цель ≤{int(th['cpa_won_max'])})",
                "hint": (
                    "Эффективность платного трафика ниже рынка. Что делать:\n"
                    "1. Убедись что офлайн-конверсии Bitrix→Метрика грузятся (SDA job 13:00)\n"
                    "2. Подожди 10-14 дней — автостратегия должна переобучиться на WON-сигнал"
                ),
            }
        )

    return alerts


def _in_cooldown(state: dict[str, str], alert_type: str, hours: int = _COOLDOWN_HOURS) -> bool:
    ts_str = state.get(alert_type)
    if not ts_str:
        return False
    try:
        ts = datetime.fromisoformat(ts_str)
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_MSK)
    return datetime.now(_MSK) - ts < timedelta(hours=hours)


def _format_alert_message(alert: dict[str, Any]) -> str:
    return (
        f"{alert['severity']} <b>Алерт BFL-RF</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        f"<b>{alert['title']}</b>\n\n"
        f"{alert['hint']}\n\n"
        f"Данные за 48 часов. Cooldown {_COOLDOWN_HOURS}ч на повторы."
    )


# --------------------------------------------------------------- sda_state IO


def _coerce_jsonb(value: Any) -> dict[str, Any]:
    """Normalise sda_state.value to a dict. Psycopg returns jsonb as dict by
    default, but our _FakePool-style shims pass it through as a JSON string —
    handle both shapes without fighting the driver."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("bfl_rf_watchdog: sda_state value corrupt — using empty")
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def _sda_state_get(pool: AsyncConnectionPool, key: str) -> dict[str, Any]:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT value FROM sda_state WHERE key = %s", (key,))
            row = await cur.fetchone()
    if row is None:
        return {}
    return _coerce_jsonb(row[0])


async def _sda_state_upsert(pool: AsyncConnectionPool, key: str, value: dict[str, Any]) -> None:
    payload = json.dumps(value)
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO sda_state (key, value, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (key, payload),
            )


# --------------------------------------------------------- throttle decision


def _phase_from_state(state: dict[str, Any]) -> str:
    phase = state.get("phase") if isinstance(state, dict) else None
    if isinstance(phase, str):
        return phase
    # Also accept plain-string shapes ("learning") for forward compat.
    if isinstance(state, str):  # pragma: no cover - defensive
        return state
    return _DEFAULT_PHASE


def _should_throttle(phase: str, last_run_state: dict[str, Any]) -> bool:
    """True iff phase is auto_pilot AND last run was <4h ago."""
    if phase != "auto_pilot":
        return False
    ts_str = last_run_state.get("ts") if isinstance(last_run_state, dict) else None
    if not isinstance(ts_str, str):
        return False
    try:
        ts = datetime.fromisoformat(ts_str)
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_MSK)
    return datetime.now(_MSK) - ts < timedelta(hours=_AUTO_PILOT_THROTTLE_HOURS)


# --------------------------------------------------------------------- run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: DirectAPI | None = None,
    metrika: MetrikaLike | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Cron entry. JOB_REGISTRY-compatible — extra kwargs default to ``None``.

    The minimal signature ``(pool, dry_run=False)`` used by the default
    registry dispatcher returns a ``degraded_noop`` result; the real
    /run/bfl_rf_watchdog HTTP handler (pending Task 20 integration) injects
    ``direct`` / ``http_client`` / ``settings`` from ``app.state``.
    """
    thresholds = build_thresholds()
    checked_at = datetime.now(_MSK).isoformat()

    if direct is None or http_client is None or settings is None:
        logger.warning(
            "bfl_rf_watchdog: DI missing (direct=%s http=%s settings=%s) — degraded no-op",
            direct is not None,
            http_client is not None,
            settings is not None,
        )
        return {
            "status": "ok",
            "action": "degraded_noop",
            "phase": _DEFAULT_PHASE,
            "alerts_active": [],
            "alerts_sent": [],
            "alerts_skipped_cooldown": [],
            "thresholds": thresholds,
            "checked_at": checked_at,
        }

    try:
        phase_state = await _sda_state_get(pool, _STRATEGY_PHASE_KEY)
        last_run_state = await _sda_state_get(pool, _LAST_RUN_KEY)
    except Exception as exc:
        logger.exception("bfl_rf_watchdog: sda_state read failed")
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "phase": _DEFAULT_PHASE,
            "thresholds": thresholds,
            "checked_at": checked_at,
        }

    phase = _phase_from_state(phase_state)
    if _should_throttle(phase, last_run_state):
        logger.info("bfl_rf_watchdog: auto_pilot throttle — skipping tick")
        return {
            "status": "skipped",
            "reason": "auto_pilot_throttle",
            "phase": phase,
            "thresholds": thresholds,
            "checked_at": checked_at,
        }

    try:
        data = await bfl_rf_tracker.collect(http_client, direct, settings, metrika=metrika, days=2)
    except Exception as exc:
        logger.exception("bfl_rf_watchdog: tracker.collect raised")
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "phase": phase,
            "thresholds": thresholds,
            "checked_at": checked_at,
        }

    alerts = _check_all(data, thresholds)
    cooldown_state = await _sda_state_get(pool, _COOLDOWN_KEY)
    # Keep only string values for known alert types (defensive against
    # someone writing garbage into the sda_state row).
    cooldown_state = {k: v for k, v in cooldown_state.items() if isinstance(v, str)}

    sent: list[str] = []
    skipped: list[str] = []
    cooldown_mutated = False

    for alert in alerts:
        a_type = alert["type"]
        if _in_cooldown(cooldown_state, a_type):
            skipped.append(a_type)
            continue
        if not dry_run:
            try:
                await telegram_tools.send_message(
                    http_client, settings, text=_format_alert_message(alert)
                )
            except Exception:
                logger.exception("bfl_rf_watchdog: telegram send failed for %s", a_type)
                # Do not update cooldown — next tick retries.
                continue
            cooldown_state[a_type] = datetime.now(_MSK).isoformat()
            cooldown_mutated = True
        sent.append(a_type)

    if not dry_run:
        try:
            if cooldown_mutated:
                await _sda_state_upsert(pool, _COOLDOWN_KEY, cooldown_state)
            await _sda_state_upsert(pool, _LAST_RUN_KEY, {"ts": datetime.now(_MSK).isoformat()})
        except Exception:
            logger.exception("bfl_rf_watchdog: sda_state upsert failed")

    logger.info(
        "bfl_rf_watchdog done phase=%s active=%d sent=%d skipped=%d dry_run=%s",
        phase,
        len(alerts),
        len(sent),
        len(skipped),
        dry_run,
    )

    return {
        "status": "ok",
        "phase": phase,
        "alerts_active": [a["type"] for a in alerts],
        "alerts_sent": sent,
        "alerts_skipped_cooldown": skipped,
        "thresholds": thresholds,
        "checked_at": checked_at,
        "dry_run": dry_run,
    }


__all__ = [
    "build_thresholds",
    "run",
]
