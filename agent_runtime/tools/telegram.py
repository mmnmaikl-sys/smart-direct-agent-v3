"""Telegram Bot API client (Task 8).

``send_with_inline`` signs each button's ``callback_data`` via
:class:`agent_runtime.auth.signing.HMACSigner` (``HYPOTHESIS_HMAC_SECRET``)
so a replay of an Approve click by a third party fails verification.
``callback_data`` format: ``{action}:{hypothesis_id}:{sig10}`` — fits
Telegram's 64-byte budget (``action``<=8, UUID=36, sig=10 → total < 58).

Text payloads are never logged — they can carry PII from Bitrix alerts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from agent_runtime.auth.signing import HMACSigner
from agent_runtime.config import Settings
from agent_runtime.tools._http import retry_with_backoff

logger = logging.getLogger(__name__)

_SEMAPHORE = asyncio.Semaphore(1)  # one message at a time to owner chat

ButtonAction = Literal["approve", "reject", "details"]


@dataclass(frozen=True)
class InlineButton:
    text: str
    action: ButtonAction


class TelegramAPIError(Exception):
    def __init__(self, code: int | str, description: str):
        self.code = code
        self.description = description
        super().__init__(f"[{code}] {description}")


def _bot_base(settings: Settings) -> str:
    token = settings.TELEGRAM_BOT_TOKEN.get_secret_value()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN empty — set in Railway env before calling telegram tools"
        )
    return f"https://api.telegram.org/bot{token}"


async def _call(
    client: httpx.AsyncClient,
    settings: Settings,
    method: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    url = f"{_bot_base(settings)}/{method}"
    async with _SEMAPHORE:
        response = await retry_with_backoff(
            lambda: client.post(url, json=payload, timeout=30.0),
            name=f"telegram.{method}",
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise TelegramAPIError("INVALID_JSON", str(exc)) from exc
    if not data.get("ok", False):
        raise TelegramAPIError(
            data.get("error_code") or response.status_code,
            str(data.get("description") or response.reason_phrase or ""),
        )
    return dict(data.get("result") or {})


async def send_message(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    text: str,
    parse_mode: str = "HTML",
    disable_web_page_preview: bool = True,
) -> int:
    """Send a plain message to the owner chat; return ``message_id``."""
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_page_preview,
    }
    logger.info("telegram.send_message: len=%d", len(text))
    result = await _call(client, settings, "sendMessage", payload)
    return int(result["message_id"])


async def send_with_inline(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    text: str,
    buttons: list[list[InlineButton]],
    hypothesis_id: str,
    parse_mode: str = "HTML",
) -> int:
    """Send a message with HMAC-signed inline buttons; return ``message_id``.

    Each button's ``callback_data`` = ``{action}:{hypothesis_id}:{sig10}`` so
    the inbound callback handler (Task 11) can verify integrity before
    executing any mutation.
    """
    signer = HMACSigner(settings.HYPOTHESIS_HMAC_SECRET)
    keyboard = [
        [
            {
                "text": btn.text,
                "callback_data": signer.sign_callback(hypothesis_id, btn.action),
            }
            for btn in row
        ]
        for row in buttons
    ]
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "reply_markup": {"inline_keyboard": keyboard},
    }
    logger.info(
        "telegram.send_with_inline: len=%d buttons=%d hypothesis=%s",
        len(text),
        sum(len(row) for row in buttons),
        hypothesis_id,
    )
    result = await _call(client, settings, "sendMessage", payload)
    return int(result["message_id"])


async def edit_message(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    message_id: int,
    text: str,
    parse_mode: str = "HTML",
    reply_markup: dict[str, Any] | None = None,
) -> None:
    """Edit an existing message (typically to ACK an alert)."""
    payload: dict[str, Any] = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "message_id": message_id,
        "text": text,
        "parse_mode": parse_mode,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    logger.info("telegram.edit_message: message_id=%d len=%d", message_id, len(text))
    await _call(client, settings, "editMessageText", payload)


__all__ = [
    "InlineButton",
    "TelegramAPIError",
    "edit_message",
    "send_message",
    "send_with_inline",
]
