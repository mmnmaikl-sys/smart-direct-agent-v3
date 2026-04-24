"""Offline Conversions — daily Bitrix WON → Metrika upload (Task 24 part 1).

Railway Cron: ``0 7 * * *`` (07:00 UTC = 10:00 МСК).

Reads Bitrix deals in stages ``C45:5`` ("договор согласован"), ``C45:6``
("договор подписан"), ``C45:WON`` ("договор заключён") whose ``DATE_MODIFY``
is within the last 24 hours, converts each into a Metrika Offline Conversions
row and uploads via ``metrika_client.upload_offline_conversions``. This is the
**ground truth** CPA signal (Decision 16): Direct's auto-bid strategy
"максимум конверсий" learns on these uploads rather than on Metrika web
conversions (which diverge from actual contracts by 50-250× on 24bankrotsttvo
— 250 Metrika conv / mo vs 1-5 real contracts).

Identifier priority per deal:

1. ``UF_CRM_YCLID`` (Yandex Click Id — preferred, full-funnel attribution)
2. ``UF_CRM_CLIENT_ID`` (Metrika client id — fallback)
3. skip → ``skipped_no_identifier``

Idempotency: every row carries ``external_id = bitrix_deal_{ID}_{STAGE_ID}``
— Metrika dedupes on it, so re-running the cron (or picking up a deal that
moved back into the same stage) does not create duplicates.

``dry_run=True`` collects rows + returns the shape, without calling the
Metrika upload. Useful for the first smoke after deploy before the Audience
OAuth token + segment id land in Railway env.

Degraded-noop pattern: when JOB_REGISTRY's default dispatcher forwards only
``(pool, dry_run)`` without DI clients, the job returns
``status='ok', action='degraded_noop'`` — the FastAPI ``/run/offline_conversions``
handler is responsible for injecting clients from ``app.state``.

PII barrier: deal responses contain NAME / PHONE / SOURCE_DESCRIPTION. This
module never forwards those fields — only the stripped row (yclid, target,
datetime, price, currency, external_id) makes it past this boundary. The
``audit_log`` write routes through :func:`agent_runtime.db.insert_audit_log`
(Decision 13 sanitiser).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Literal, Protocol

import httpx
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import bitrix as bitrix_tools

logger = logging.getLogger(__name__)

_MSK = timezone(timedelta(hours=3))
_WINDOW_HOURS = 24
_STATE_KEY = "offline_conversions_uploaded"
_UPLOADED_HISTORY_CAP = 500  # bounded growth of already-uploaded external_ids.

# Stage → Metrika target mapping. Module constant by spec; exposed in __all__
# so smoke tests / reviewer greps confirm the mapping.
_STAGE_TO_TARGET: dict[str, str] = {
    "C45:5": "deal_agreed",
    "C45:6": "deal_signed",
    "C45:WON": "deal_won",
}


class MetrikaOfflineClient(Protocol):
    """Narrow protocol for the Metrika Offline Conversions upload endpoint.

    TODO(integration): a concrete implementation lives in
    ``agent_runtime/tools/metrika.py::MetrikaClient.upload_offline_conversions``
    (to be added alongside this job per Task 24 step 3). Until then, the
    FastAPI ``/run/offline_conversions`` handler can inject any object
    honouring this protocol; tests pass :class:`unittest.mock.AsyncMock`.
    """

    async def upload_offline_conversions(
        self,
        counter_id: int,
        rows: list[dict[str, Any]],
        client_id_type: Literal["YCLID", "CLIENT_ID", "USER_ID"],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _Row:
    """One CSV row ready for Metrika upload. No PII reachable from here."""

    external_id: str
    identifier_value: str
    identifier_type: Literal["YCLID", "CLIENT_ID"]
    target: str
    datetime_s: str
    price: float
    currency: str

    def as_upload_dict(self) -> dict[str, Any]:
        """Metrika API accepts either CSV multipart or JSON; return JSON shape."""
        key = "yclid" if self.identifier_type == "YCLID" else "client_id"
        return {
            key: self.identifier_value,
            "target": self.target,
            "datetime": self.datetime_s,
            "price": self.price,
            "currency": self.currency,
            "external_id": self.external_id,
        }


# --------------------------------------------------------------- helpers


def _now_msk() -> datetime:
    return datetime.now(_MSK)


def _safe_float(value: Any) -> float:
    try:
        return float(value) if value is not None and value != "" else 0.0
    except (TypeError, ValueError):
        return 0.0


def _resolve_identifier(
    deal: dict[str, Any],
) -> tuple[str | None, Literal["YCLID", "CLIENT_ID"] | None]:
    """Return ``(value, type)`` or ``(None, None)`` if neither field is set.

    ``UF_CRM_YCLID`` wins over ``UF_CRM_CLIENT_ID`` — yclid carries the
    full Direct funnel (campaign/adgroup/keyword) while client_id is the
    fallback Metrika visitor id.
    """
    yclid = deal.get("UF_CRM_YCLID")
    if isinstance(yclid, str) and yclid.strip():
        return yclid.strip(), "YCLID"
    client_id = deal.get("UF_CRM_CLIENT_ID")
    if isinstance(client_id, str) and client_id.strip():
        return client_id.strip(), "CLIENT_ID"
    return None, None


def _build_row(deal: dict[str, Any]) -> _Row | None:
    """Collapse a Bitrix deal dict into a Metrika upload row.

    Returns ``None`` for:
      * unknown stage id (defence — should not happen given filter)
      * missing identifier (caller bumps ``skipped_no_identifier``)
    """
    stage_id = str(deal.get("STAGE_ID") or "")
    target = _STAGE_TO_TARGET.get(stage_id)
    if target is None:
        return None
    identifier, identifier_type = _resolve_identifier(deal)
    if identifier is None or identifier_type is None:
        return None
    deal_id = str(deal.get("ID") or "").strip()
    if not deal_id:
        return None
    datetime_s = str(deal.get("DATE_MODIFY") or deal.get("DATE_CREATE") or _now_msk().isoformat())
    price = _safe_float(deal.get("OPPORTUNITY"))
    return _Row(
        external_id=f"bitrix_deal_{deal_id}_{stage_id}",
        identifier_value=identifier,
        identifier_type=identifier_type,
        target=target,
        datetime_s=datetime_s,
        price=price,
        currency="RUB",
    )


def _split_by_identifier(rows: list[_Row]) -> tuple[list[_Row], list[_Row]]:
    """Return (yclid_rows, client_id_rows) — Metrika needs two uploads."""
    yclid: list[_Row] = []
    client_id: list[_Row] = []
    for row in rows:
        if row.identifier_type == "YCLID":
            yclid.append(row)
        else:
            client_id.append(row)
    return yclid, client_id


# --------------------------------------------------------------- state IO


async def _load_uploaded_set(pool: AsyncConnectionPool) -> set[str]:
    """Read the ring-buffer of already-uploaded ``external_id`` values.

    This is defense-in-depth against Metrika's own dedup: if the API ever
    accepts a row silently (happened with legacy external_id format on
    24bankrotsttvo.ru in March 2026), we still skip client-side so the
    ``uploaded`` count is truthful.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT value FROM sda_state WHERE key = %s", (_STATE_KEY,))
            row = await cur.fetchone()
    if row is None or row[0] is None:
        return set()
    raw = row[0]
    ids: Any = raw.get("external_ids") if isinstance(raw, dict) else None
    if not isinstance(ids, list):
        return set()
    return {str(item) for item in ids}


async def _save_uploaded_set(pool: AsyncConnectionPool, ids: set[str]) -> None:
    capped = sorted(ids)[-_UPLOADED_HISTORY_CAP:]
    payload = {"external_ids": capped, "updated_at": datetime.now(UTC).isoformat()}
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO sda_state (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (_STATE_KEY, Jsonb(payload)),
            )


# --------------------------------------------------------------- run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    bitrix_client: httpx.AsyncClient | None = None,
    metrika_client: MetrikaOfflineClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Daily cron entry. JOB_REGISTRY-compatible.

    Extra kwargs default to ``None`` so the minimal ``(pool, dry_run)``
    dispatch path (:mod:`agent_runtime.jobs.__init__.dispatch_job`) does
    not fail. The real /run/offline_conversions handler injects DI clients
    from ``app.state``.

    Returns a structured dict with ``status``, ``uploaded``, per-target
    counts, and the list of deal ids skipped because no yclid/client_id
    was set.
    """
    now = _now_msk()
    now_iso = now.isoformat()

    if bitrix_client is None or metrika_client is None or settings is None:
        logger.warning(
            "offline_conversions: DI missing (bitrix=%s metrika=%s settings=%s) — degraded no-op",
            bitrix_client is not None,
            metrika_client is not None,
            settings is not None,
        )
        return {
            "status": "ok",
            "action": "degraded_noop",
            "uploaded": 0,
            "by_target": {},
            "skipped_no_identifier": [],
            "errors": [],
            "dry_run": dry_run,
            "checked_at": now_iso,
        }

    since = (now - timedelta(hours=_WINDOW_HOURS)).isoformat()

    # Step 1 — Bitrix deals in last 24h across the three paying stages.
    try:
        deals = await bitrix_tools.get_deal_list(
            bitrix_client,
            settings,
            filter={
                "STAGE_ID": list(_STAGE_TO_TARGET.keys()),
                ">=DATE_MODIFY": since,
            },
            select=[
                "ID",
                "STAGE_ID",
                "UTM_CAMPAIGN",
                "UTM_CONTENT",
                "UTM_TERM",
                "UF_CRM_YCLID",
                "UF_CRM_CLIENT_ID",
                "OPPORTUNITY",
                "DATE_MODIFY",
                "DATE_CREATE",
            ],
        )
    except Exception as exc:
        logger.exception("offline_conversions: get_deal_list failed")
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "uploaded": 0,
            "by_target": {},
            "skipped_no_identifier": [],
            "errors": [str(exc)],
            "dry_run": dry_run,
            "checked_at": now_iso,
        }

    # Step 2-3 — build rows; track skip reasons.
    rows: list[_Row] = []
    skipped_no_identifier: list[str] = []
    for deal in deals:
        if not isinstance(deal, dict):
            continue
        row = _build_row(deal)
        if row is None:
            deal_id = str(deal.get("ID") or "").strip()
            if deal_id and str(deal.get("STAGE_ID") or "") in _STAGE_TO_TARGET:
                # unknown identifier — the only shaped skip we expose.
                skipped_no_identifier.append(deal_id)
            continue
        rows.append(row)

    # Step 5 — client-side dedup against previous runs.
    uploaded_set = await _load_uploaded_set(pool)
    fresh_rows = [r for r in rows if r.external_id not in uploaded_set]
    duplicate_count = len(rows) - len(fresh_rows)
    logger.info(
        "offline_conversions: fetched=%d rows=%d fresh=%d duplicates=%d skipped_no_id=%d",
        len(deals),
        len(rows),
        len(fresh_rows),
        duplicate_count,
        len(skipped_no_identifier),
    )

    by_target: dict[str, int] = {}
    for row in fresh_rows:
        by_target[row.target] = by_target.get(row.target, 0) + 1

    # Step 4 — upload.
    uploaded = 0
    errors: list[str] = []

    if dry_run:
        logger.info("offline_conversions: dry_run=True, skipping Metrika upload")
        return {
            "status": "ok",
            "dry_run": True,
            "uploaded_would_be": len(fresh_rows),
            "uploaded": 0,
            "by_target": by_target,
            "skipped_no_identifier": skipped_no_identifier,
            "duplicate_count": duplicate_count,
            "errors": [],
            "checked_at": now_iso,
        }

    yclid_rows, client_id_rows = _split_by_identifier(fresh_rows)
    newly_uploaded: set[str] = set()

    for batch, id_type in (
        (yclid_rows, "YCLID"),
        (client_id_rows, "CLIENT_ID"),
    ):
        if not batch:
            continue
        try:
            await metrika_client.upload_offline_conversions(
                settings.METRIKA_COUNTER_ID,
                [r.as_upload_dict() for r in batch],
                client_id_type=id_type,  # type: ignore[arg-type]
            )
            uploaded += len(batch)
            for row in batch:
                newly_uploaded.add(row.external_id)
        except Exception as exc:
            logger.exception("offline_conversions: upload(%s, %d rows) failed", id_type, len(batch))
            errors.append(f"{id_type}: {type(exc).__name__}: {exc}")

    # Step 5 cont. — persist newly-uploaded external_ids so the next cron
    # skips them even if Metrika silently accepts the duplicate.
    if newly_uploaded:
        try:
            await _save_uploaded_set(pool, uploaded_set | newly_uploaded)
        except Exception:
            logger.exception("offline_conversions: state persist failed")

    status = "error" if errors and uploaded == 0 else "ok"

    # Step 6 — audit_log trace (aggregate only, no raw deal fields).
    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="n/a",
            tool_name="offline_conversions",
            tool_input={
                "period_hours": _WINDOW_HOURS,
                "stages": list(_STAGE_TO_TARGET.keys()),
                "deals_fetched": len(deals),
            },
            tool_output={
                "uploaded": uploaded,
                "by_target": by_target,
                "skipped_no_identifier_count": len(skipped_no_identifier),
                "duplicate_count": duplicate_count,
                "errors": errors,
                "dry_run": False,
            },
            is_mutation=uploaded > 0,
            is_error=bool(errors) and uploaded == 0,
            error_detail="; ".join(errors) if errors else None,
        )
    except Exception:
        logger.exception("offline_conversions: audit_log write failed")

    return {
        "status": status,
        "uploaded": uploaded,
        "by_target": by_target,
        "skipped_no_identifier": skipped_no_identifier,
        "duplicate_count": duplicate_count,
        "errors": errors,
        "dry_run": False,
        "checked_at": now_iso,
    }


__all__ = [
    "MetrikaOfflineClient",
    "run",
]
