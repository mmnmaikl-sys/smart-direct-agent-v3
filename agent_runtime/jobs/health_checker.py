"""Health Checker — daily read-only digest of yesterday's Metrika health (Task 26).

Runs every day at 13:00 МСК (``0 10 * * *`` UTC). Pulls three parallel
Metrika aggregates over *yesterday* (MSK calendar day):

1. **Bounce by campaign** — ``ym:s:lastSignDirectOrderID`` attribution
   (NOT ``directCampaignID``; see ``memory/feedback_metrika_api_attributes.md``).
2. **Device breakdown** — ``ym:s:deviceCategory`` with share-of-visits.
3. **Top landings** — ``ym:s:startURL`` restricted to ``yandex`` source so
   non-Direct traffic does not skew the health picture.

Parallel isolation: one check raising does NOT sink the other two —
``asyncio.gather(return_exceptions=True)`` + per-section normalisation.
Each section surfaces ``error=`` in the rendered summary so the owner
knows *what* is stale, not just that the digest is incomplete.

This job is a **reporter**. Explicitly no mutations:

* No Direct API calls (``direct`` is reserved for a future "active campaign
  discovery" extension — currently we resolve the ID set from gate state +
  settings fallback).
* No ``INSERT INTO hypotheses`` / ``INSERT INTO ask_queue``.
* No Kill-switch evaluation.

``dry_run=True`` builds and returns the Markdown summary in
``result['summary']`` without calling Telegram. Live runs post the summary
to :data:`Settings.TELEGRAM_CHAT_ID` through
:func:`agent_runtime.tools.telegram.send_message` (single-chat, no inline
buttons, throttled by the shared semaphore).

Trust overlay applies at the *notify* layer only — in ``shadow`` the send
still happens, because reading-only reports are explicitly whitelisted as
NOTIFY-safe. In ``FORBIDDEN_LOCK`` we still post the text (it's read-only)
but tag the header so the owner sees the trust state.

TODO(integration):
  1. Register ``"health_checker": health_checker.run`` in
     ``agent_runtime/jobs/__init__.py::JOB_REGISTRY`` (out-of-scope here).
  2. Add Railway Cron ``0 10 * * *`` for ``/run/health_checker``.
  3. If active-campaign discovery moves to a dedicated cache row, swap the
     ``_resolve_active_campaigns`` fallback chain for a cache lookup.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.tools import metrika as metrika_tools
from agent_runtime.tools import telegram as telegram_tools

logger = logging.getLogger(__name__)


# --- thresholds & knobs ------------------------------------------------------

BOUNCE_RED_THRESHOLD: float = 60.0  # >=60 % → 🔴
BOUNCE_YELLOW_THRESHOLD: float = 50.0  # 50..60 % → 🟡
TOP_LANDINGS_LIMIT: int = 5
TOP_DEVICES_LIMIT: int = 3
TELEGRAM_MAX_LEN: int = 3000  # below the 4096 hard limit with generous slack
_MSK_OFFSET = timedelta(hours=3)
_STATE_KEY = "strategy_gate_state"


# --- date window helpers -----------------------------------------------------


def _yesterday_msk_bounds() -> tuple[str, str]:
    """Return ISO-YMD strings for yesterday's MSK calendar day.

    Metrika ``stat/v1`` accepts equal ``date1``/``date2`` meaning 'that one day'.
    We pin both to yesterday MSK so the caller does not wander into a UTC/MSK
    day-boundary off-by-one.
    """
    now_msk = datetime.now(UTC) + _MSK_OFFSET
    yesterday = (now_msk - timedelta(days=1)).date().isoformat()
    return yesterday, yesterday


# --- per-section result objects ---------------------------------------------


@dataclass(frozen=True)
class BounceSection:
    ok: bool
    rows: list[tuple[int, float]]  # [(campaign_id, bounce_pct)]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": [{"campaign_id": cid, "bounce_pct": round(pct, 2)} for cid, pct in self.rows],
            "error": self.error,
        }


@dataclass(frozen=True)
class DeviceSection:
    ok: bool
    rows: list[tuple[str, float]]  # [(device, share_pct)]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": [{"device": d, "share_pct": round(s, 2)} for d, s in self.rows],
            "error": self.error,
        }


@dataclass(frozen=True)
class LandingSection:
    ok: bool
    rows: list[tuple[str, int, float]]  # [(url, visits, bounce_pct)]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": [{"url": u, "visits": v, "bounce_pct": round(p, 2)} for u, v, p in self.rows],
            "error": self.error,
        }


# --- parallel checks ---------------------------------------------------------


async def _check_bounce_by_campaign(
    http: httpx.AsyncClient,
    settings: Settings,
    *,
    campaign_ids: list[int],
    date1: str,
    date2: str,
) -> BounceSection:
    try:
        raw = await metrika_tools.get_bounce_by_campaign(http, settings, date1=date1, date2=date2)
    except Exception as exc:
        logger.warning("health_checker: bounce-by-campaign failed: %s", exc, exc_info=True)
        return BounceSection(ok=False, rows=[], error=f"{type(exc).__name__}: {exc}")

    campaign_set = set(campaign_ids)
    rows: list[tuple[int, float]] = []
    for campaign_id, bounce in raw.items():
        if campaign_set and campaign_id not in campaign_set:
            continue
        try:
            pct = float(bounce)
        except (TypeError, ValueError):
            continue
        rows.append((int(campaign_id), pct))
    rows.sort(key=lambda r: (-r[1], r[0]))  # worst bounce first
    return BounceSection(ok=True, rows=rows)


async def _check_device_breakdown(
    http: httpx.AsyncClient,
    settings: Settings,
    *,
    date1: str,
    date2: str,
) -> DeviceSection:
    try:
        data = await metrika_tools.get_stats(
            http,
            settings,
            metrics=["ym:s:visits"],
            dimensions=["ym:s:deviceCategory"],
            filters="ym:s:lastSignUTMSource=='yandex'",
            date1=date1,
            date2=date2,
        )
    except Exception as exc:
        logger.warning("health_checker: device breakdown failed: %s", exc, exc_info=True)
        return DeviceSection(ok=False, rows=[], error=f"{type(exc).__name__}: {exc}")

    entries: list[tuple[str, int]] = []
    for row in data.get("data") or []:
        dims = row.get("dimensions") or []
        metrics = row.get("metrics") or []
        if not dims or not metrics:
            continue
        raw_name = dims[0].get("name") or dims[0].get("id") or "unknown"
        try:
            visits = int(metrics[0] or 0)
        except (TypeError, ValueError):
            continue
        entries.append((str(raw_name), visits))

    total = sum(v for _, v in entries)
    if total <= 0:
        return DeviceSection(ok=True, rows=[])
    shares = [(name, (visits / total) * 100.0) for name, visits in entries]
    shares.sort(key=lambda r: (-r[1], r[0]))
    return DeviceSection(ok=True, rows=shares[:TOP_DEVICES_LIMIT])


async def _check_top_landings(
    http: httpx.AsyncClient,
    settings: Settings,
    *,
    date1: str,
    date2: str,
) -> LandingSection:
    try:
        data = await metrika_tools.get_stats(
            http,
            settings,
            metrics=["ym:s:visits", "ym:s:bounceRate"],
            dimensions=["ym:s:startURL"],
            filters=(
                "ym:s:lastSignUTMSource=='yandex' "
                "AND ym:s:startURL=~'.*24bankrotsttvo.*/pages/ad/.*'"
            ),
            date1=date1,
            date2=date2,
            limit=TOP_LANDINGS_LIMIT * 4,
        )
    except Exception as exc:
        logger.warning("health_checker: top-landings failed: %s", exc, exc_info=True)
        return LandingSection(ok=False, rows=[], error=f"{type(exc).__name__}: {exc}")

    entries: list[tuple[str, int, float]] = []
    for row in data.get("data") or []:
        dims = row.get("dimensions") or []
        metrics = row.get("metrics") or []
        if not dims or len(metrics) < 2:
            continue
        url = str(dims[0].get("name") or dims[0].get("id") or "")
        try:
            visits = int(metrics[0] or 0)
            bounce = float(metrics[1] or 0)
        except (TypeError, ValueError):
            continue
        entries.append((url, visits, bounce))
    entries.sort(key=lambda r: (-r[1], r[0]))  # highest traffic first
    return LandingSection(ok=True, rows=entries[:TOP_LANDINGS_LIMIT])


# --- active campaign resolver ------------------------------------------------


async def _resolve_active_campaigns(
    pool: AsyncConnectionPool,
    settings: Settings,
) -> list[int]:
    """Active campaign IDs from strategy_gate cache, else PROTECTED_CAMPAIGN_IDS.

    Kept read-only on purpose — this job never mutates PG or Direct. If gate
    cache is missing/empty the fallback is the Settings whitelist; if BOTH
    are empty we still return ``[]`` and the bounce-by-campaign section
    degrades to "no data" without raising.
    """
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT value FROM sda_state WHERE key = %s",
                    (_STATE_KEY,),
                )
                row = await cur.fetchone()
    except Exception:
        logger.warning("health_checker: sda_state read failed", exc_info=True)
        row = None

    if row and row[0]:
        raw = row[0]
        if not isinstance(raw, dict):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                raw = {}
        if isinstance(raw, dict):
            active = raw.get("active_campaigns")
            if isinstance(active, list) and active:
                out: list[int] = []
                for item in active:
                    try:
                        out.append(int(item))
                    except (TypeError, ValueError):
                        continue
                if out:
                    return out

    return list(settings.PROTECTED_CAMPAIGN_IDS)


# --- markdown rendering ------------------------------------------------------


def _bounce_flag(pct: float) -> str:
    if pct >= BOUNCE_RED_THRESHOLD:
        return "🔴"
    if pct >= BOUNCE_YELLOW_THRESHOLD:
        return "🟡"
    return "🟢"


def _render_summary(
    *,
    date1: str,
    bounce: BounceSection,
    device: DeviceSection,
    landing: LandingSection,
) -> str:
    """Human-readable Markdown summary for Telegram.

    Strictly capped at :data:`TELEGRAM_MAX_LEN` — if rendering overshoots
    we truncate with an explicit note so the owner knows to page into logs.
    """
    lines: list[str] = [f"*Health check* — {date1} МСК", ""]

    # Bounce section.
    lines.append("*Bounce by campaign:*")
    if not bounce.ok:
        lines.append(f"  error: {bounce.error}")
    elif not bounce.rows:
        lines.append("  нет данных за вчера")
    else:
        for cid, pct in bounce.rows:
            flag = _bounce_flag(pct)
            lines.append(f"  {flag} camp `{cid}`: {pct:.0f}%")
    lines.append("")

    # Device section.
    lines.append("*Devices (top-3):*")
    if not device.ok:
        lines.append(f"  error: {device.error}")
    elif not device.rows:
        lines.append("  нет данных за вчера")
    else:
        for name, share in device.rows:
            lines.append(f"  • {name}: {share:.0f}%")
    lines.append("")

    # Landings section.
    lines.append("*Top landings:*")
    if not landing.ok:
        lines.append(f"  error: {landing.error}")
    elif not landing.rows:
        lines.append("  нет данных за вчера")
    else:
        for url, visits, pct in landing.rows:
            flag = _bounce_flag(pct)
            lines.append(f"  {flag} {url} — {visits} visits, bounce {pct:.0f}%")
    out = "\n".join(lines)
    if len(out) > TELEGRAM_MAX_LEN:
        out = out[: TELEGRAM_MAX_LEN - 30] + "\n…(truncated)"
    return out


def _all_ok(bounce: BounceSection, device: DeviceSection, landing: LandingSection) -> bool:
    return bounce.ok and device.ok and landing.ok


# --- run entry ---------------------------------------------------------------


async def _run_impl(
    pool: AsyncConnectionPool,
    settings: Settings,
    *,
    http_client: httpx.AsyncClient,
    dry_run: bool,
) -> dict[str, Any]:
    date1, date2 = _yesterday_msk_bounds()
    campaigns = await _resolve_active_campaigns(pool, settings)

    logger.info(
        "health_checker: fetching yesterday=%s campaigns=%d dry_run=%s",
        date1,
        len(campaigns),
        dry_run,
    )

    bounce_res, device_res, landing_res = await asyncio.gather(
        _check_bounce_by_campaign(
            http_client, settings, campaign_ids=campaigns, date1=date1, date2=date2
        ),
        _check_device_breakdown(http_client, settings, date1=date1, date2=date2),
        _check_top_landings(http_client, settings, date1=date1, date2=date2),
        return_exceptions=True,
    )

    def _coerce_bounce(res: Any) -> BounceSection:
        if isinstance(res, BounceSection):
            return res
        if isinstance(res, BaseException):
            return BounceSection(ok=False, rows=[], error=f"{type(res).__name__}: {res}")
        return BounceSection(ok=False, rows=[], error=f"bad type {type(res).__name__}")

    def _coerce_device(res: Any) -> DeviceSection:
        if isinstance(res, DeviceSection):
            return res
        if isinstance(res, BaseException):
            return DeviceSection(ok=False, rows=[], error=f"{type(res).__name__}: {res}")
        return DeviceSection(ok=False, rows=[], error=f"bad type {type(res).__name__}")

    def _coerce_landing(res: Any) -> LandingSection:
        if isinstance(res, LandingSection):
            return res
        if isinstance(res, BaseException):
            return LandingSection(ok=False, rows=[], error=f"{type(res).__name__}: {res}")
        return LandingSection(ok=False, rows=[], error=f"bad type {type(res).__name__}")

    bounce = _coerce_bounce(bounce_res)
    device = _coerce_device(device_res)
    landing = _coerce_landing(landing_res)

    summary = _render_summary(date1=date1, bounce=bounce, device=device, landing=landing)

    sent_message_id: int | None = None
    send_error: str | None = None
    if not dry_run:
        try:
            sent_message_id = await telegram_tools.send_message(
                http_client, settings, text=summary, parse_mode="Markdown"
            )
        except Exception as exc:
            logger.warning("health_checker: telegram send failed: %s", exc, exc_info=True)
            send_error = f"{type(exc).__name__}: {exc}"

    return {
        "action": "sent" if (not dry_run and sent_message_id) else "drafted",
        "dry_run": dry_run,
        "all_ok": _all_ok(bounce, device, landing),
        "date": date1,
        "campaigns": campaigns,
        "bounce": bounce.to_dict(),
        "devices": device.to_dict(),
        "landings": landing.to_dict(),
        "summary": summary,
        "telegram_message_id": sent_message_id,
        "telegram_error": send_error,
    }


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    direct: Any | None = None,  # retained for JOB_REGISTRY symmetry; unused.
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Cron entry. Daily read-only health digest.

    Degraded paths:
      * ``http_client`` or ``settings`` missing → ``degraded_noop`` return
        so a cron misconfig does not crash the job; the /run/health_checker
        FastAPI handler injects both from ``app.state`` for real runs.
      * Any one of the three parallel checks raising → that section surfaces
        ``error=<message>`` in the summary; the other two still render.
      * Telegram failing on live run → summary is still returned, with
        ``telegram_error`` populated. The job never raises from ``run()``.
    """
    _ = direct  # read-only reporter; kept in signature for registry symmetry

    if http_client is None or settings is None:
        logger.warning(
            "health_checker: missing DI (http=%s settings=%s) — degraded no-op",
            http_client is not None,
            settings is not None,
        )
        return {
            "action": "degraded_noop",
            "dry_run": dry_run,
            "reason": "http_or_settings_missing",
        }

    try:
        return await _run_impl(pool, settings, http_client=http_client, dry_run=dry_run)
    except Exception as exc:
        logger.exception("health_checker crashed")
        return {
            "action": "error",
            "dry_run": dry_run,
            "error": f"{type(exc).__name__}: {exc}",
        }


__all__ = [
    "BOUNCE_RED_THRESHOLD",
    "BOUNCE_YELLOW_THRESHOLD",
    "BounceSection",
    "DeviceSection",
    "LandingSection",
    "run",
]
