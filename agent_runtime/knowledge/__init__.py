"""Knowledge base consultation layer — ``consult(question, context)``.

All KB markdown files in this package are concatenated once at import time
and shipped as the Anthropic ``system`` parameter with ``cache_control`` so
that each subsequent call pays ~10% of the cached tokens' cost. Typical
system prompt is 40K-70K characters; without caching, 100 consults/day would
burn ~$3/day just on re-ingest.

Each caller (jobs, brain) does:

    from agent_runtime.knowledge import consult

    result = await consult(
        "CPA=1500 on Bashkortostan campaign, what to do?",
        context={"campaign_id": 708978456, "spend_7d": 42000},
    )
    # result["answer"]    -> "Рассмотри… (2-4 абзаца)"
    # result["citations"] -> ["direct-knowledge-base.md#autostrategies", ...]

A thread-local-free LRU+TTL cache absorbs duplicate questions for 1 hour;
different ``context`` values yield different cache keys so scenario-specific
answers are not cross-contaminated.

SECURITY: ``context`` MUST NOT contain raw PII (phones, names, emails). The
caller is responsible for running ``agent_runtime.pii.sanitize_audit_payload``
first. Anything passed here is shipped to Anthropic and audited under
Decision 13.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from agents_core.llm.client import LLMClient, LLMResponse

from agent_runtime.config import get_settings

logger = logging.getLogger(__name__)

_KB_DIR = Path(__file__).resolve().parent
_SEPARATOR = "\n\n===== FILE: {name} =====\n\n"
_CACHE_MAX = 100
_CACHE_TTL_SEC = 3600
_MAX_CONTEXT_BYTES = 50_000
_CITATION_RE = re.compile(r"\[([a-z0-9._\-]+\.md(?:#[\w\-]+)?)\]", re.IGNORECASE)


def _load_kb_system_prompt() -> str:
    """Concatenate every ``*.md`` in this package into one deterministic string."""
    files = sorted(_KB_DIR.glob("*.md"))
    if len(files) < 5:
        logger.warning(
            "KB has only %d files, expected >=5 (direct-knowledge-base, presnyakov-257, "
            "api-gotchas, minus-words-bfl, legal-compliance-wave1)",
            len(files),
        )
    parts: list[str] = []
    for path in files:
        parts.append(_SEPARATOR.format(name=path.name))
        parts.append(path.read_text(encoding="utf-8"))
    return "".join(parts)


# Loaded once per process — the deterministic order keeps Anthropic's
# prompt-cache key stable across requests.
_KB_SYSTEM_PROMPT: str = _load_kb_system_prompt()
KB_FILENAMES: tuple[str, ...] = tuple(sorted(p.name for p in _KB_DIR.glob("*.md")))


# --- cache (LRU + TTL) -------------------------------------------------------


class _TTLCache:
    """Minimal OrderedDict-backed LRU with per-entry expiry.

    Not thread-safe; asyncio is single-threaded per loop so that is fine in
    the FastAPI process. Two concurrent coroutines hitting the same cold key
    both miss and both hit the LLM — acceptable race (the cost is one extra
    call, not correctness).
    """

    def __init__(self, maxsize: int, ttl_sec: int) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_sec
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: str, *, now: float | None = None) -> dict[str, Any] | None:
        current = time.monotonic() if now is None else now
        entry = self._data.get(key)
        if entry is None:
            return None
        inserted_at, value = entry
        if current - inserted_at > self._ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: dict[str, Any], *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (current, value)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


_cache: _TTLCache = _TTLCache(maxsize=_CACHE_MAX, ttl_sec=_CACHE_TTL_SEC)


# --- lazy LLM client ---------------------------------------------------------


_client: LLMClient | None = None


def _get_client() -> LLMClient:
    global _client  # noqa: PLW0603
    if _client is None:
        key = get_settings().ANTHROPIC_API_KEY.get_secret_value()
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is empty — set it in Railway env before calling kb.consult()"
            )
        _client = LLMClient(anthropic_api_key=key)
    return _client


# --- consult -----------------------------------------------------------------


_INSTRUCTION = (
    "Ответь на вопрос ниже, опираясь ТОЛЬКО на знания из system prompt "
    "(KB-файлы с маркером `===== FILE: ...`). Ответ краток (2-4 абзаца). "
    "В конце добавь строку `CITATIONS:` и перечисли файлы/секции, "
    "на которые ты опирался, в формате `[filename.md#section]`. "
    "Если ответа нет в KB — честно ответь «нет в базе» и дай best-effort "
    "интерпретацию с пометкой «без ссылки на KB»."
)


def _cache_key(question: str, context: dict[str, Any] | None) -> str:
    ctx_json = json.dumps(context, sort_keys=True, ensure_ascii=False) if context else "null"
    return f"{question}␟{ctx_json}"


def _build_prompt(question: str, context: dict[str, Any] | None) -> str:
    if context is not None:
        ctx_json = json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True)
        if len(ctx_json.encode("utf-8")) > _MAX_CONTEXT_BYTES:
            logger.warning(
                "kb.consult: context oversize (%d bytes), truncating",
                len(ctx_json.encode("utf-8")),
            )
            ctx_json = ctx_json[:_MAX_CONTEXT_BYTES] + "\n... (truncated)"
    else:
        ctx_json = "нет"
    return f"**Контекст:**\n{ctx_json}\n\n**Вопрос:** {question}\n\n{_INSTRUCTION}"


def _parse_response(text: str) -> dict[str, Any]:
    """Split the model output into ``answer`` + ``citations``.

    Citation format: ``[filename.md]`` or ``[filename.md#section]`` — we
    extract every match inside the trailing ``CITATIONS:`` block; if that
    block is missing, we still scan the whole text (models sometimes put
    inline citations).
    """
    if "CITATIONS:" in text:
        head, tail = text.rsplit("CITATIONS:", 1)
        answer = head.strip()
        citations_region = tail
    else:
        answer = text.strip()
        citations_region = text
    citations = [m.group(1) for m in _CITATION_RE.finditer(citations_region)]
    # deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return {"answer": answer, "citations": deduped}


async def consult(
    question: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer ``question`` against the in-package KB with citations.

    Returns a dict ``{"answer": str, "citations": list[str]}``.

    * Uses Claude Sonnet (``model="sonnet"``) — Haiku hallucinates on 50K+
      system prompts per Anthropic's own eval guidance.
    * ``system_cache=True`` so the KB ships once per 5-min TTL, not per call.
    * LRU-caches (question, context) for 1 hour to absorb duplicate asks
      across agent runs.

    Raises ``ValueError`` on empty question; ``RuntimeError`` if
    ``ANTHROPIC_API_KEY`` is unset.
    """
    if not question or not question.strip():
        raise ValueError("empty question")

    key = _cache_key(question, context)
    cached = _cache.get(key)
    if cached is not None:
        logger.debug("kb.consult cache hit: %s", question[:60])
        return cached

    client = _get_client()
    prompt = _build_prompt(question, context)
    response: LLMResponse = await client.chat(
        prompt=prompt,
        model="sonnet",
        system=_KB_SYSTEM_PROMPT,
        system_cache=True,
        max_tokens=1024,
        name="kb.consult",
    )
    result = _parse_response(response.text)
    _cache.set(key, result)
    return result


__all__ = [
    "KB_FILENAMES",
    "_KB_SYSTEM_PROMPT",
    "consult",
]
