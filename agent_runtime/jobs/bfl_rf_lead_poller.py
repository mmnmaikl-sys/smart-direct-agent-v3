"""BFL-RF Lead Poller — in-process realtime Telegram alerts (Task 16).

Port of v2 ``bfl_rf_lead_poller`` onto the async stack with three
structural changes:

1. **State in PG**, not a JSON file under ``/data/``. Row lives in
   ``sda_state`` under key ``lead_poller_state`` as a JSONB blob
   ``{"last_seen": "ISO", "notified_ids": [100 last]}``. UPSERT via
   ``INSERT ... ON CONFLICT DO UPDATE`` — durable across container
   restarts; survives Railway re-deploys without relying on a volume.
2. **Async clients from Task 8** — :func:`bitrix.get_lead_list` and
   :func:`telegram.send_message` over a shared ``httpx.AsyncClient``.
   No legacy ``app.bitrix_client`` / ``app.telegram_client`` imports.
3. **In-process asyncio loop**, registered in the FastAPI ``lifespan``
   (Decision 9 exception — the rest of the SDA jobs run via Railway
   Cron, but Railway's 5-minute floor is too slow for the lead pulse).
   Graceful cancel + await on shutdown; exceptions in one tick do not
   sink the loop.

Crash-safety invariants:

* ``last_seen`` is **only** advanced after a successful Bitrix fetch.
  A Bitrix 5xx → return ``{"error": ...}``, state untouched, next tick
  retries the same window.
* ``notified_ids`` is appended only on successful Telegram send. One
  failing ``send_message`` does not block the others in the same tick.
* A 30-second safety margin on ``fetch_from = last_seen - 30s`` absorbs
  Bitrix's second-level timestamp granularity without duplicating
  alerts (dedup via ``notified_ids``).

No config literals (UTM whitelist, portal URL) live in this module —
all through :class:`Settings`. Grep test in the unit file enforces it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.tools import bitrix as bitrix_tools
from agent_runtime.tools import telegram as telegram_tools

logger = logging.getLogger(__name__)


_MSK = timezone(timedelta(hours=3))
_FETCH_SAFETY_MARGIN = timedelta(seconds=30)
_STATE_KEY = "lead_poller_state"
_QUIZ_FIELDS: tuple[str, ...] = (
    "debt_amount",
    "property",
    "goal",
    "debt_type",
    "income",
    "deals",
)


# ----------------------------------------------------------- SOURCE_DESCRIPTION


def _parse_source_description(sd: str) -> dict[str, str]:
    """Parse Bitrix ``SOURCE_DESCRIPTION`` ``"k1=v1 | k2=v2"`` into a dict."""
    if not sd:
        return {}
    result: dict[str, str] = {}
    for part in (p.strip() for p in sd.split("|")):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------- LeadPoller


class LeadPoller:
    """Stateless class; all durable state lives in PG ``sda_state``."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> None:
        self.pool = pool
        self.http = http_client
        self.settings = settings

    @property
    def utm_label(self) -> str:
        return ",".join(self.settings.LEAD_POLLER_UTM_WHITELIST) or "—"

    async def run_once(self, *, dry_run: bool = False) -> dict[str, Any]:
        """One poll cycle. Returns a structured summary dict.

        Notes on error semantics:

        * Bitrix error → ``{"error": ...}`` with ``last_seen`` untouched.
          Upstream loop logs and waits for the next tick.
        * Per-lead Telegram error → lead skipped, not retried — prevents
          alert spam. ``last_seen`` still advances so the tick completes.
        """
        state = await self._load_state()
        now = datetime.now(_MSK)
        fetch_from = _compute_fetch_from(
            state["last_seen"], now, self.settings.LEAD_POLLER_INITIAL_LOOKBACK_MIN
        )

        try:
            leads = await self._fetch_leads(fetch_from)
        except Exception as exc:
            logger.warning("lead_poller: Bitrix fetch failed: %s", exc)
            return {
                "checked_at": now.isoformat(),
                "error": str(exc),
                "fetched_total": 0,
                "new_leads": 0,
                "sent_ids": [],
            }

        new_leads = _filter_new(leads, state["last_seen"], set(state["notified_ids"]))
        sent_ids: list[str] = []
        for lead in new_leads:
            if dry_run:
                sent_ids.append(str(lead.get("ID")))
                continue
            text = self._format_lead(lead)
            try:
                await telegram_tools.send_message(
                    self.http,
                    self.settings,
                    text=text,
                    parse_mode="HTML",
                )
            except Exception:
                logger.warning(
                    "lead_poller: telegram send failed for ID=%s",
                    lead.get("ID"),
                    exc_info=True,
                )
                continue
            sent_ids.append(str(lead.get("ID")))

        if not dry_run:
            merged = _merge_notified(
                state["notified_ids"], sent_ids, self.settings.LEAD_POLLER_NOTIFIED_IDS_CAP
            )
            await self._save_state({"last_seen": now.isoformat(), "notified_ids": merged})

        logger.info(
            "lead_poller tick: fetched=%d new=%d sent=%d dry=%s",
            len(leads),
            len(new_leads),
            len(sent_ids),
            dry_run,
        )
        return {
            "checked_at": now.isoformat(),
            "fetched_total": len(leads),
            "new_leads": len(new_leads),
            "sent_ids": sent_ids,
        }

    # ---------------------------------------------------- data access

    async def _fetch_leads(self, fetch_from: datetime) -> list[dict[str, Any]]:
        return await bitrix_tools.get_lead_list(
            self.http,
            self.settings,
            filter={
                ">=DATE_CREATE": fetch_from.isoformat(),
                "UTM_CAMPAIGN": list(self.settings.LEAD_POLLER_UTM_WHITELIST),
            },
            select=[
                "ID",
                "DATE_CREATE",
                "NAME",
                "TITLE",
                "PHONE",
                "UTM_SOURCE",
                "UTM_CAMPAIGN",
                "UTM_CONTENT",
                "UTM_TERM",
                "SOURCE_DESCRIPTION",
            ],
            max_total=self.settings.LEAD_POLLER_MAX_PAGES * 50,
        )

    async def _load_state(self) -> dict[str, Any]:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT value FROM sda_state WHERE key = %s", (_STATE_KEY,))
                row = await cur.fetchone()
        if row is None or row[0] is None:
            return {"last_seen": None, "notified_ids": []}
        raw = row[0]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("lead_poller: state decode failed, reset")
                return {"last_seen": None, "notified_ids": []}
        last_seen_iso = raw.get("last_seen") if isinstance(raw, dict) else None
        last_seen = _parse_iso(last_seen_iso) if last_seen_iso else None
        notified = raw.get("notified_ids", []) if isinstance(raw, dict) else []
        return {
            "last_seen": last_seen,
            "notified_ids": [str(i) for i in notified],
        }

    async def _save_state(self, state: dict[str, Any]) -> None:
        payload = json.dumps(state)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO sda_state (key, value, updated_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (key) DO UPDATE
                        SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    (_STATE_KEY, payload),
                )

    # ------------------------------------------------- message formatting

    def _format_lead(self, lead: dict[str, Any]) -> str:
        # SECURITY (Task 29 audit A03): every CRM-provided field is HTML-
        # escaped before composing a `parse_mode=HTML` message. A lead
        # created with NAME="<a href='evil'>..." would otherwise render
        # as a live link in the owner's Telegram.
        import html as _html

        def _h(v: Any) -> str:
            return _html.escape(str(v), quote=False)

        raw_name = (
            lead.get("NAME") or str(lead.get("TITLE") or "").split("#")[0].strip() or "без имени"
        )
        phones = lead.get("PHONE") or []
        raw_phone = phones[0].get("VALUE", "?") if isinstance(phones, list) and phones else "—"
        raw_utm_content = lead.get("UTM_CONTENT") or ""
        raw_utm_term = lead.get("UTM_TERM") or ""
        raw_utm_campaign = lead.get("UTM_CAMPAIGN") or self.utm_label
        lead_id = lead.get("ID")
        quiz = _parse_source_description(lead.get("SOURCE_DESCRIPTION") or "")

        lines = [
            f"🆕 <b>ЛИД {_h(raw_utm_campaign)}</b>",
            "━━━━━━━━━━━━━━━━━",
            f"👤 {_h(raw_name)}",
            f"📞 {_h(raw_phone)}",
        ]
        if raw_utm_term:
            lines.append(f"🔑 «{_h(raw_utm_term)}»")
        if raw_utm_content:
            lines.append(f"📢 ad_id: {_h(raw_utm_content)}")
        if quiz:
            lines.append("")
            lines.append("<b>Квиз:</b>")
            for key in _QUIZ_FIELDS:
                if key in quiz:
                    lines.append(f"  • {key}: {_h(quiz[key])}")
        lines.append("")
        base = self.settings.BITRIX_PORTAL_BASE_URL.rstrip("/")
        lines.append(f'🔗 <a href="{base}/crm/lead/details/{lead_id}/">Открыть в CRM</a>')
        return "\n".join(lines)


# --------------------------------------------------- pure filter helpers


def _compute_fetch_from(last_seen: datetime | None, now: datetime, lookback_min: int) -> datetime:
    """Return the Bitrix ``>=DATE_CREATE`` lower bound.

    Cold start → ``now - lookback_min``. Subsequent ticks →
    ``last_seen - 30s`` (safety margin for Bitrix's second-level
    ISO granularity).
    """
    if last_seen is None:
        return now - timedelta(minutes=lookback_min)
    return last_seen - _FETCH_SAFETY_MARGIN


def _filter_new(
    leads: list[dict[str, Any]],
    last_seen: datetime | None,
    notified_ids: set[str],
) -> list[dict[str, Any]]:
    """Keep only leads strictly newer than ``last_seen`` AND unseen by ID.

    Bitrix's ``DATE_CREATE`` round-trips as a string; we parse once per
    row and silently skip rows with malformed timestamps (should never
    happen in practice — Bitrix is consistent — but keeps the poller
    from sinking on a single bad row).
    """
    out: list[dict[str, Any]] = []
    for lead in leads:
        lead_id = str(lead.get("ID") or "")
        if not lead_id or lead_id in notified_ids:
            continue
        dc = _parse_iso(lead.get("DATE_CREATE"))
        if dc is None:
            continue
        if last_seen is not None and dc <= last_seen:
            continue
        out.append(lead)
    return out


def _merge_notified(old: list[str], sent: list[str], cap: int) -> list[str]:
    """Append new ids, dedup preserving order, trim to ``cap`` most recent."""
    merged: list[str] = []
    seen: set[str] = set()
    for i in [*old, *sent]:
        if i in seen:
            continue
        seen.add(i)
        merged.append(i)
    return merged[-cap:]


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_MSK)
    return dt.astimezone(_MSK)


# ------------------------------------------------------------ in-process loop


async def _lead_poller_loop(poller: LeadPoller) -> None:
    """Infinite loop registered in FastAPI ``lifespan``.

    Exception isolation — one failing tick logs + sleeps + retries. Only
    ``asyncio.CancelledError`` escapes (re-raised for graceful shutdown).
    Sleep is **in the finally block of the inner try** so a crashing tick
    still backs off before the next attempt.
    """
    interval = poller.settings.LEAD_POLLER_INTERVAL_SEC
    logger.info(
        "Lead poller started (interval=%ds, whitelist=%s)",
        interval,
        poller.settings.LEAD_POLLER_UTM_WHITELIST,
    )
    try:
        while True:
            try:
                await poller.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("lead_poller: tick failed")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Lead poller cancelled")
        raise


# ----------------------------------------------------------- job registry hook


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Single-shot entry for the /run/{job} dispatcher (used in smoke only).

    The poller's natural home is the lifespan loop, not the HTTP cron
    path — but having a ``run()`` lets ops hit ``POST /run/lead_poller?
    dry_run=true`` to get a one-shot snapshot without touching the
    async loop. Returns a no-op payload if DI is missing.
    """
    if http_client is None or settings is None:
        logger.warning("lead_poller.run: missing http_client/settings — degraded no-op")
        return {
            "status": "ok",
            "action": "degraded_noop",
            "checked_at": datetime.now(_MSK).isoformat(),
            "new_leads": 0,
            "sent_ids": [],
        }
    poller = LeadPoller(pool, http_client, settings)
    result = await poller.run_once(dry_run=dry_run)
    return {"status": "ok", **result}


__all__ = [
    "LeadPoller",
    "_lead_poller_loop",
    "run",
]
