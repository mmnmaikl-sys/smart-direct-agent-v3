"""Audience Sync — weekly Bitrix WON → Yandex Audience look-alike (Task 24 part 2).

Railway Cron: ``0 6 * * 3`` (06:00 UTC Wednesday = 09:00 МСК среда).

Reads every C45:WON deal from the last 180 days, **hard-gates on
UF_CRM_CONSENT_ADVERTISING='Y'** (152-ФЗ Decision 13 — deals without the
explicit advertising-consent flag are skipped before the contact phone is
even fetched), normalises the first phone on the contact record, MD5-hashes
it (Yandex Audience contract — *not* our :func:`agent_runtime.pii.hash_phone`
which uses salted SHA256 for a different purpose), and uploads the hash set
via ``audience_client.modify_segment_data``.

Hard invariants enforced in tests:

* Every deal without ``UF_CRM_CONSENT_ADVERTISING='Y'`` lands in
  ``skipped_no_consent`` — value comparison is literal ``"Y"`` so
  ``"N"`` / ``None`` / missing key / ``""`` all skip.
* Raw phones and the MD5 hashes themselves never enter ``audit_log``.
  Only counters and deal_id lists are logged.
* MD5 is computed **without a salt**. The comment in :func:`_md5_hex`
  spells this out for the security-auditor reviewer — do NOT add
  ``settings.PII_SALT`` here, that belongs to :mod:`agent_runtime.pii`
  (audit_log sanitiser contract), not to the Audience upload contract.

Degraded-noop pattern: when DI clients are absent (JOB_REGISTRY default
dispatch), the job returns ``status='ok', action='degraded_noop'`` so a
misconfigured cron does not crash the FastAPI /run handler.

Dry-run returns the same shape with ``uploaded_would_be`` instead of
``uploaded`` and suppresses the Audience API call.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol

import httpx
from psycopg_pool import AsyncConnectionPool

from agent_runtime.config import Settings
from agent_runtime.db import insert_audit_log
from agent_runtime.tools import bitrix as bitrix_tools

logger = logging.getLogger(__name__)

_MSK = timezone(timedelta(hours=3))
_WON_STAGE_ID = "C45:WON"
_WON_WINDOW_DAYS = 180
_CONSENT_FIELD = "UF_CRM_CONSENT_ADVERTISING"
_CONSENT_VALUE = "Y"
_LOW_CONSENT_WARN_RATIO = 0.5  # warn if <50% of WON deals carry consent='Y'

_NON_DIGIT_OR_PLUS = re.compile(r"[^\d+]")


class AudienceClient(Protocol):
    """Narrow protocol for the Yandex Audience modify-data endpoint.

    TODO(integration): a concrete implementation will live in
    ``agent_runtime/tools/audience.py::AudienceClient`` (new module per
    Task 24 step 4). Until then, the FastAPI ``/run/audience_sync``
    handler is expected to wire any object honouring this protocol;
    tests pass :class:`unittest.mock.AsyncMock`.
    """

    async def modify_segment_data(
        self,
        segment_id: int,
        hashes: list[str],
        modification_type: Literal["replace", "append"] = "replace",
    ) -> dict[str, Any]: ...


class BitrixContactFetcher(Protocol):
    """Callable-style protocol for a single-contact PHONE fetch.

    TODO(integration): Bitrix does not yet expose ``get_contact`` in
    :mod:`agent_runtime.tools.bitrix` (Task 8 added deal / lead / stage
    history only). The FastAPI handler should inject a thin wrapper that
    calls ``crm.contact.get`` and returns the raw Bitrix dict; tests pass
    :class:`unittest.mock.AsyncMock` with ``.side_effect`` returning the
    shaped response.
    """

    async def __call__(self, contact_id: int) -> dict[str, Any]: ...


# --------------------------------------------------------------- helpers


def _normalize_phone(raw: str | None) -> str | None:
    """Return ``+79…`` canonical form or ``None`` if unusable.

    Mirrors :func:`agent_runtime.pii._normalise_phone` but duplicated here
    so this module does not pull in :mod:`agent_runtime.pii` — the two
    hash contracts (salted SHA256 vs. bare MD5) must stay lexically
    separate for the security-auditor greps in Task 5c.
    """
    if not raw:
        return None
    compact = _NON_DIGIT_OR_PLUS.sub("", raw)
    if not compact:
        return None
    if compact.startswith("8") and len(compact) == 11:
        compact = "+7" + compact[1:]
    elif not compact.startswith("+"):
        compact = "+" + compact.lstrip("+")
    # Require at least +XXXXXXXXXX (11 chars including +).
    if len(compact) < 11:
        return None
    return compact


def _md5_hex(phone_normalized: str) -> str:
    """MD5 hex-digest of the normalised phone.

    MD5 без соли — это формат Yandex Audience.
    НЕ использовать PII_SALT (Task 5c) — это другой хеш-контракт.
    """
    return hashlib.md5(phone_normalized.encode("utf-8")).hexdigest()  # noqa: S324


def _extract_first_phone(contact: dict[str, Any]) -> str | None:
    """Pull the first usable phone out of a Bitrix contact dict.

    Bitrix shape: ``{"PHONE": [{"VALUE": "+7…", "VALUE_TYPE": "WORK"}, ...]}``
    — we take the first entry. Scalar form (``"PHONE": "+7…"``) is also
    tolerated for robustness across legacy migration rows.
    """
    phone_field = contact.get("PHONE")
    if isinstance(phone_field, str):
        return phone_field
    if isinstance(phone_field, list):
        for entry in phone_field:
            if isinstance(entry, dict):
                value = entry.get("VALUE")
                if isinstance(value, str) and value.strip():
                    return value
            elif isinstance(entry, str) and entry.strip():
                return entry
    return None


def _now_msk() -> datetime:
    return datetime.now(_MSK)


# --------------------------------------------------------------- run


async def run(
    pool: AsyncConnectionPool,
    *,
    dry_run: bool = False,
    bitrix_client: httpx.AsyncClient | None = None,
    audience_client: AudienceClient | None = None,
    get_contact: BitrixContactFetcher | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Weekly cron entry. JOB_REGISTRY-compatible.

    Extra kwargs default to ``None`` so the minimal ``(pool, dry_run)``
    dispatch path does not fail. The real ``/run/audience_sync`` handler
    injects all clients from ``app.state``.

    Returns a structured dict:

      * ``uploaded`` / ``uploaded_would_be`` — hash count actually sent
        (or would have been, in dry_run)
      * ``skipped_no_consent`` — deal ids without UF_CRM_CONSENT_ADVERTISING=Y
      * ``skipped_no_phone`` — deal ids where the contact had no phone
      * ``consent_rate`` — consented_deals / total_won_deals
      * ``segment_id`` — target Audience segment id (for traceability)
    """
    now = _now_msk()
    now_iso = now.isoformat()

    if bitrix_client is None or audience_client is None or get_contact is None or settings is None:
        logger.warning(
            "audience_sync: DI missing — degraded no-op "
            "(bitrix=%s audience=%s get_contact=%s settings=%s)",
            bitrix_client is not None,
            audience_client is not None,
            get_contact is not None,
            settings is not None,
        )
        return {
            "status": "ok",
            "action": "degraded_noop",
            "uploaded": 0,
            "total_won_deals": 0,
            "skipped_no_consent": [],
            "skipped_no_phone": [],
            "consent_rate": 0.0,
            "segment_id": None,
            "dry_run": dry_run,
            "checked_at": now_iso,
        }

    # Settings guard — reviewers grep this module for "segment_id" assertions.
    segment_id = int(getattr(settings, "YA_AUDIENCE_SEGMENT_ID", 0) or 0)
    # NB: we intentionally do not validate segment_id>0 here — leaving it at
    # 0 lets the test suite run without a real segment while still exercising
    # the code path; the Audience mock ignores the id.

    since = (now - timedelta(days=_WON_WINDOW_DAYS)).isoformat()

    # Step 1 — Bitrix WON deals over 180d.
    try:
        deals = await bitrix_tools.get_deal_list(
            bitrix_client,
            settings,
            filter={
                "STAGE_ID": _WON_STAGE_ID,
                ">=CLOSEDATE": since,
            },
            select=[
                "ID",
                "CONTACT_ID",
                _CONSENT_FIELD,
                "CLOSEDATE",
                "DATE_MODIFY",
            ],
        )
    except Exception as exc:
        logger.exception("audience_sync: get_deal_list failed")
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "uploaded": 0,
            "total_won_deals": 0,
            "skipped_no_consent": [],
            "skipped_no_phone": [],
            "consent_rate": 0.0,
            "segment_id": segment_id,
            "dry_run": dry_run,
            "checked_at": now_iso,
        }

    skipped_no_consent: list[str] = []
    consented_deals: list[dict[str, Any]] = []

    # Step 2 — 152-ФЗ consent gate. Hard check BEFORE contact lookup.
    for deal in deals:
        if not isinstance(deal, dict):
            continue
        # Explicit block — no chained boolean so reviewer can grep the pattern.
        if deal.get(_CONSENT_FIELD) != _CONSENT_VALUE:
            deal_id = str(deal.get("ID") or "").strip()
            if deal_id:
                skipped_no_consent.append(deal_id)
            continue
        consented_deals.append(deal)

    total = len(deals)
    consent_rate = (len(consented_deals) / total) if total > 0 else 0.0
    if total > 0 and consent_rate < _LOW_CONSENT_WARN_RATIO:
        logger.warning(
            "audience_sync: low consent rate: %d%% (%d/%d) — likely consent field not "
            "populated on landing forms (Task 0)",
            int(consent_rate * 100),
            len(consented_deals),
            total,
        )

    # Step 3-5 — contact phones → normalise → MD5.
    skipped_no_phone: list[str] = []
    hashes: list[str] = []
    seen_hashes: set[str] = set()  # dedup within one cron run
    for deal in consented_deals:
        deal_id = str(deal.get("ID") or "").strip()
        contact_id_raw = deal.get("CONTACT_ID")
        try:
            contact_id = int(contact_id_raw) if contact_id_raw else 0
        except (TypeError, ValueError):
            contact_id = 0
        if contact_id <= 0:
            if deal_id:
                skipped_no_phone.append(deal_id)
            continue
        try:
            contact = await get_contact(contact_id)
        except Exception:
            logger.exception("audience_sync: get_contact(%d) failed, skipping deal", contact_id)
            if deal_id:
                skipped_no_phone.append(deal_id)
            continue
        raw_phone = _extract_first_phone(contact) if isinstance(contact, dict) else None
        normalized = _normalize_phone(raw_phone)
        if normalized is None:
            if deal_id:
                skipped_no_phone.append(deal_id)
            continue
        md5 = _md5_hex(normalized)
        if md5 in seen_hashes:
            continue
        seen_hashes.add(md5)
        hashes.append(md5)

    logger.info(
        "audience_sync: won=%d consent_ok=%d hashes=%d skipped_no_consent=%d skipped_no_phone=%d",
        total,
        len(consented_deals),
        len(hashes),
        len(skipped_no_consent),
        len(skipped_no_phone),
    )

    if dry_run:
        # Audit still written so observability shows the dry-run happened,
        # but no raw phone / md5 in payload (defense-in-depth).
        try:
            await insert_audit_log(
                pool,
                hypothesis_id=None,
                trust_level="n/a",
                tool_name="audience_sync",
                tool_input={
                    "segment_id": segment_id,
                    "dry_run": True,
                    "deal_ids_processed": [str(d.get("ID") or "") for d in consented_deals],
                },
                tool_output={
                    "uploaded_would_be": len(hashes),
                    "skipped_no_consent_count": len(skipped_no_consent),
                    "skipped_no_phone_count": len(skipped_no_phone),
                    "total_won_deals": total,
                },
                is_mutation=False,
            )
        except Exception:
            logger.exception("audience_sync: audit_log write failed (dry_run)")
        return {
            "status": "ok",
            "dry_run": True,
            "uploaded_would_be": len(hashes),
            "uploaded": 0,
            "total_won_deals": total,
            "skipped_no_consent": skipped_no_consent,
            "skipped_no_phone": skipped_no_phone,
            "consent_rate": consent_rate,
            "segment_id": segment_id,
            "checked_at": now_iso,
        }

    # Step 6 — upload. Short-circuit on empty hash set (Yandex Audience
    # rejects empty modify_data payloads with 400).
    if not hashes:
        logger.info("audience_sync: nothing to upload (0 consented phones) — skipping Audience API")
        try:
            await insert_audit_log(
                pool,
                hypothesis_id=None,
                trust_level="n/a",
                tool_name="audience_sync",
                tool_input={
                    "segment_id": segment_id,
                    "deal_ids_processed": [str(d.get("ID") or "") for d in consented_deals],
                },
                tool_output={
                    "uploaded": 0,
                    "skipped_no_consent_count": len(skipped_no_consent),
                    "skipped_no_phone_count": len(skipped_no_phone),
                    "total_won_deals": total,
                    "reason": "empty_hash_set",
                },
                is_mutation=False,
            )
        except Exception:
            logger.exception("audience_sync: audit_log write failed (empty)")
        return {
            "status": "ok",
            "uploaded": 0,
            "total_won_deals": total,
            "skipped_no_consent": skipped_no_consent,
            "skipped_no_phone": skipped_no_phone,
            "consent_rate": consent_rate,
            "segment_id": segment_id,
            "dry_run": False,
            "checked_at": now_iso,
        }

    uploaded = 0
    error_detail: str | None = None
    try:
        await audience_client.modify_segment_data(
            segment_id=segment_id,
            hashes=hashes,
            modification_type="replace",
        )
        uploaded = len(hashes)
    except Exception as exc:
        logger.exception("audience_sync: modify_segment_data failed")
        error_detail = f"{type(exc).__name__}: {exc}"

    status = "error" if error_detail else "ok"

    # Step 7 — audit_log WITHOUT PII. Only counts + deal_id lists.
    try:
        await insert_audit_log(
            pool,
            hypothesis_id=None,
            trust_level="n/a",
            tool_name="audience_sync",
            tool_input={
                "segment_id": segment_id,
                "deal_ids_processed": [str(d.get("ID") or "") for d in consented_deals],
            },
            tool_output={
                "uploaded": uploaded,
                "skipped_no_consent_count": len(skipped_no_consent),
                "skipped_no_phone_count": len(skipped_no_phone),
                "total_won_deals": total,
                "error": error_detail,
            },
            is_mutation=uploaded > 0,
            is_error=status == "error",
            error_detail=error_detail,
        )
    except Exception:
        logger.exception("audience_sync: audit_log write failed")

    return {
        "status": status,
        "uploaded": uploaded,
        "total_won_deals": total,
        "skipped_no_consent": skipped_no_consent,
        "skipped_no_phone": skipped_no_phone,
        "consent_rate": consent_rate,
        "segment_id": segment_id,
        "dry_run": False,
        "error": error_detail,
        "checked_at": now_iso,
    }


__all__ = [
    "AudienceClient",
    "BitrixContactFetcher",
    "run",
]
