"""Metrika tracking health — alerts when goals don't fire despite visits.

Daily probe: pulls per-Direct-cid `visits` + `goal{GOAL_ID}reaches` last
7d. For each cid where `visits >= MIN_VISITS_FOR_SIGNAL` AND `goals == 0`,
emits a Telegram warning — this is almost always a broken tracking setup
(missing ym snippet on landing, wrong goal_id, or autotargeting traffic
hitting a different page).

Diagnostic-only — no Direct mutations. Cron daily 13:30 UTC (16:30 МСК).

Discovery context (26.04.2026): probe revealed 21 visits / 0 goals on
all 5 own campaigns over the last 7d, blocking tactical_actuator's CR
signal. Owner needs this surfaced as an actionable alert, not as a
silent gap in the dashboards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from agent_runtime.tools import telegram as telegram_tools

if TYPE_CHECKING:
    import httpx
    from psycopg_pool import AsyncConnectionPool

    from agent_runtime.config import Settings

logger = logging.getLogger(__name__)


_OWN_CAMPAIGNS: dict[int, str] = {
    709353005: "rabotyaga",
    709353034: "pensioner",
    709353058: "mother",
    709353078: "mfo",
    709353099: "property",
}

_MIN_VISITS_FOR_SIGNAL = 10  # below this — too cold to alert
_WINDOW_DAYS = 7


@dataclass(frozen=True)
class CidHealth:
    cid: int
    name: str
    visits: int
    goals: int
    bounce_pct: float
    is_broken: bool  # visits>=threshold AND goals==0


def assess_health(rows_by_cid: dict[int, dict[str, float]]) -> list[CidHealth]:
    """Pure function: classify each known cid."""
    out: list[CidHealth] = []
    for cid, name in _OWN_CAMPAIGNS.items():
        row = rows_by_cid.get(cid) or {}
        visits = int(row.get("visits", 0))
        goals = int(row.get("goals", 0))
        bounce = float(row.get("bounce", 0.0))
        broken = visits >= _MIN_VISITS_FOR_SIGNAL and goals == 0
        out.append(
            CidHealth(
                cid=cid, name=name, visits=visits, goals=goals, bounce_pct=bounce, is_broken=broken
            )
        )
    return out


def _format_report(health: list[CidHealth], goal_id: int, window_days: int) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    broken = [h for h in health if h.is_broken]
    cold = [h for h in health if not h.is_broken and h.visits < _MIN_VISITS_FOR_SIGNAL]
    healthy = [h for h in health if not h.is_broken and h.visits >= _MIN_VISITS_FOR_SIGNAL]

    lines = [
        f"<b>📡 Metrika Health Check</b> — {when}",
        "",
        f"Окно: last {window_days}d. Goal_id: {goal_id}.",
        "",
    ]

    if broken:
        lines.append(
            f"<b>🔴 Tracking сломан ({len(broken)} cid):</b> visits есть, "
            "но goals=0 — либо нет Метрики на лендинге, либо неверный goal_id."
        )
        for h in broken:
            lines.append(
                f"  • <code>{h.cid}</code> ({h.name}): visits={h.visits}, "
                f"goals=0, bounce={h.bounce_pct:.0f}%"
            )
        lines.append("")
        lines.append(
            "<i>Действие: открыть лендинг, F12 → Network → проверить ym(...).reachGoal."
            " Если Метрика не загружена — добавить counter код."
            " Если goal_id другой — обновить env GOAL_ID.</i>"
        )
        lines.append("")

    if healthy:
        lines.append(f"<b>🟢 Tracking ok ({len(healthy)} cid):</b>")
        for h in healthy:
            cr = (100 * h.goals / h.visits) if h.visits else 0
            lines.append(
                f"  • <code>{h.cid}</code> ({h.name}): "
                f"visits={h.visits}, goals={h.goals}, CR={cr:.1f}%"
            )
        lines.append("")

    if cold:
        cold_str = ", ".join(f"{h.name}({h.visits}v)" for h in cold)
        lines.append(f"<i>⏸ Холодные ({len(cold)}): {cold_str}</i>")

    return "\n".join(lines)


async def _fetch_per_cid(
    http_client: httpx.AsyncClient,
    *,
    counter: str,
    token: str,
    goal_ids: list[int],
    date_from: str,
    date_to: str,
) -> dict[int, dict[str, float]]:
    """Per-cid {visits, bounce, goals} where goals = sum across all goal_ids.

    Composite goal 549152162 catches quiz + form + phone_click but NOT
    Novofon Call (549163547). Summing both gives full conversions across
    all 4 channels (квиз + form + phone-click + real call).
    """
    if not goal_ids:
        return {}
    goal_metrics = ",".join(f"ym:s:goal{gid}reaches" for gid in goal_ids)
    params = {
        "ids": str(counter),
        "metrics": f"ym:s:visits,ym:s:bounceRate,{goal_metrics}",
        "dimensions": "ym:s:lastDirectClickOrder",
        "date1": date_from,
        "date2": date_to,
        "limit": "500",
        "accuracy": "full",
    }
    url = "https://api-metrika.yandex.net/stat/v1/data?" + urlencode(params)
    try:
        resp = await http_client.get(url, headers={"Authorization": f"OAuth {token}"}, timeout=30)
        if resp.status_code != 200:
            logger.warning("metrika_health: %d %s", resp.status_code, resp.text[:200])
            return {}
        data = resp.json()
    except Exception:
        logger.exception("metrika_health: fetch failed")
        return {}

    out: dict[int, dict[str, float]] = {}
    for row in data.get("data") or []:
        dims = row.get("dimensions") or []
        m = row.get("metrics") or []
        if not dims or len(m) < 2 + len(goal_ids):
            continue
        try:
            cid = int(dims[0].get("id"))
        except (TypeError, ValueError):
            continue
        # m[0]=visits, m[1]=bounce, m[2..]=each goal's reaches
        goals_sum = sum(float(m[2 + i]) for i in range(len(goal_ids)))
        out[cid] = {"visits": float(m[0]), "bounce": float(m[1]), "goals": goals_sum}
    return out


def _resolve_secret(token_obj: Any) -> str:
    if token_obj is None:
        return ""
    if hasattr(token_obj, "get_secret_value"):
        return token_obj.get_secret_value()
    return str(token_obj)


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """JOB_REGISTRY entrypoint. Cron daily 13:30 UTC (16:30 МСК)."""
    if http_client is None or settings is None:
        return {"status": "ok", "action": "degraded_noop", "broken": 0, "dry_run": dry_run}

    counter = str(getattr(settings, "METRIKA_COUNTER_ID", "") or "")
    token = _resolve_secret(getattr(settings, "METRIKA_OAUTH_TOKEN", None))

    # GOAL_ID + optional CALL_GOAL_ID — sum of conversions across both.
    # Composite (549152162) covers reachGoal events (quiz, form, phone_click);
    # CALL_GOAL_ID (549163547) covers actual calls registered by Novofon
    # call-tracking integration. Together they give full attribution.
    raw_ids: list[int] = []
    for var in ("GOAL_ID", "CALL_GOAL_ID"):
        v = getattr(settings, var, None)
        if v is None:
            continue
        try:
            n = int(v)
            if n > 0:
                raw_ids.append(n)
        except (TypeError, ValueError):
            continue
    if not counter or not token or not raw_ids:
        return {
            "status": "ok",
            "action": "degraded_noop",
            "reason": "missing METRIKA_COUNTER_ID / METRIKA_OAUTH_TOKEN / GOAL_ID",
            "dry_run": dry_run,
        }

    today = datetime.now(UTC).date().isoformat()
    date_from = (datetime.now(UTC).date() - timedelta(days=_WINDOW_DAYS)).isoformat()

    rows = await _fetch_per_cid(
        http_client,
        counter=counter,
        token=token,
        goal_ids=raw_ids,
        date_from=date_from,
        date_to=today,
    )

    health = assess_health(rows)
    broken_count = sum(1 for h in health if h.is_broken)

    msg = _format_report(
        health,
        goal_id=raw_ids[0] if raw_ids else 0,
        window_days=_WINDOW_DAYS,
    )
    if not dry_run:
        try:
            await telegram_tools.send_message(http_client, settings, text=msg, parse_mode="HTML")
        except Exception:
            logger.exception("metrika_health: telegram send failed")

    return {
        "status": "ok",
        "broken": broken_count,
        "total_cids": len(health),
        "dry_run": dry_run,
    }


__all__ = ["CidHealth", "assess_health", "run"]
