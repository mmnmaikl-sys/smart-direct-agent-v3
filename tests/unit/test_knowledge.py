"""Unit tests for agent_runtime.knowledge.consult() — Task 6 TDD anchor.

Real Anthropic is never hit here — ``LLMClient.chat`` is mocked per test via
monkeypatch. Deterministic inputs / deterministic outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_runtime import knowledge


@dataclass
class _FakeUsage:
    input: int = 0
    output: int = 0
    cache_creation: int = 0
    cache_read: int = 0


@dataclass
class _FakeLLMResponse:
    text: str
    model: str = "claude-sonnet-4"
    usage: _FakeUsage | None = None
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    attempt: int = 1
    raw: object = None


def _mk_response(text: str) -> _FakeLLMResponse:
    return _FakeLLMResponse(text=text, usage=_FakeUsage())


@pytest.fixture
def clear_cache():
    """Reset the module-level LRU between tests so ordering is irrelevant."""
    knowledge._cache.clear()
    # Also reset the lazy client so monkeypatched env takes effect
    knowledge._client = None
    yield
    knowledge._cache.clear()


# --- KB loading --------------------------------------------------------------


def test_kb_files_loaded() -> None:
    for name in (
        "direct-knowledge-base.md",
        "presnyakov-257.md",
        "api-gotchas.md",
        "minus-words-bfl.md",
        "legal-compliance-wave1.md",
    ):
        marker = f"===== FILE: {name} ====="
        assert marker in knowledge._KB_SYSTEM_PROMPT, f"missing {name} in KB prompt"
    assert len(knowledge._KB_SYSTEM_PROMPT) > 40_000, (
        f"KB prompt too short: {len(knowledge._KB_SYSTEM_PROMPT)} chars "
        "(a single file is ~5k; 5 files should be 40k+)"
    )


def test_kb_filenames_exported() -> None:
    assert set(knowledge.KB_FILENAMES) >= {
        "direct-knowledge-base.md",
        "presnyakov-257.md",
        "api-gotchas.md",
        "minus-words-bfl.md",
        "legal-compliance-wave1.md",
    }


# --- consult ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_consult_returns_citation(monkeypatch: pytest.MonkeyPatch, clear_cache) -> None:
    calls: list[dict] = []

    async def fake_chat(**kwargs):
        calls.append(kwargs)
        return _mk_response(
            "Grace period — это 30-дневный буфер… (короткий ответ)\n\n"
            "CITATIONS: [direct-knowledge-base.md#autostrategies]"
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-abc")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    result = await knowledge.consult("что такое grace period?")
    assert result["citations"] == ["direct-knowledge-base.md#autostrategies"]
    assert "Grace period" in result["answer"]


@pytest.mark.asyncio
async def test_consult_without_citations_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch, clear_cache
) -> None:
    async def fake_chat(**kw):
        return _mk_response("Нет в базе — best-effort интерпретация, без ссылки на KB")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    r = await knowledge.consult("экзотика")
    assert r["citations"] == []
    assert "Нет в базе" in r["answer"]


@pytest.mark.asyncio
async def test_consult_lru_cache(monkeypatch: pytest.MonkeyPatch, clear_cache) -> None:
    call_count = {"n": 0}

    async def fake_chat(**kw):
        call_count["n"] += 1
        return _mk_response(f"ответ #{call_count['n']}\n\nCITATIONS: [presnyakov-257.md]")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    r1 = await knowledge.consult("одинаковый вопрос")
    r2 = await knowledge.consult("одинаковый вопрос")
    assert call_count["n"] == 1, "second identical call must hit cache"
    assert r1 == r2
    await knowledge.consult("другой вопрос")
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_consult_cache_key_includes_context(
    monkeypatch: pytest.MonkeyPatch, clear_cache
) -> None:
    count = {"n": 0}

    async def fake_chat(**kw):
        count["n"] += 1
        return _mk_response("ok\n\nCITATIONS: [api-gotchas.md]")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    await knowledge.consult("q", context={"a": 1})
    await knowledge.consult("q", context={"a": 2})
    assert count["n"] == 2, "different context must not share cache entry"


def test_cache_ttl_expires() -> None:
    c = knowledge._TTLCache(maxsize=10, ttl_sec=3600)
    payload = {"answer": "x", "citations": []}
    c.set("k", payload, now=1000.0)
    assert c.get("k", now=1000.0) == payload
    # 3700s later = 100s past TTL → expired and evicted on read
    assert c.get("k", now=4700.0) is None
    assert c.get("k", now=4700.0) is None  # still gone


def test_cache_lru_eviction() -> None:
    c = knowledge._TTLCache(maxsize=2, ttl_sec=3600)
    c.set("a", {"answer": "1", "citations": []}, now=1.0)
    c.set("b", {"answer": "2", "citations": []}, now=2.0)
    c.set("c", {"answer": "3", "citations": []}, now=3.0)
    assert c.get("a", now=4.0) is None  # evicted
    assert c.get("b", now=4.0) is not None
    assert c.get("c", now=4.0) is not None


@pytest.mark.asyncio
async def test_system_cache_flag_true(monkeypatch: pytest.MonkeyPatch, clear_cache) -> None:
    captured: dict = {}

    async def fake_chat(**kw):
        captured.update(kw)
        return _mk_response("ok\n\nCITATIONS: [direct-knowledge-base.md]")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    await knowledge.consult("flag check")
    assert captured["system_cache"] is True
    assert captured["model"] == "sonnet"
    assert captured["name"] == "kb.consult"


@pytest.mark.asyncio
async def test_context_serialized_in_prompt(monkeypatch: pytest.MonkeyPatch, clear_cache) -> None:
    captured: dict = {}

    async def fake_chat(**kw):
        captured.update(kw)
        return _mk_response("ok\n\nCITATIONS: [direct-knowledge-base.md]")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    await knowledge.consult("q", context={"campaign_id": 708978456, "cpa": 1500})
    prompt = captured["prompt"]
    assert '"campaign_id": 708978456' in prompt
    assert '"cpa": 1500' in prompt
    assert "**Вопрос:** q" in prompt


@pytest.mark.asyncio
async def test_consult_returns_dict_shape(monkeypatch: pytest.MonkeyPatch, clear_cache) -> None:
    async def fake_chat(**kw):
        return _mk_response("text\n\nCITATIONS: [api-gotchas.md]")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    r = await knowledge.consult("shape")
    assert set(r.keys()) == {"answer", "citations"}
    assert isinstance(r["answer"], str)
    assert isinstance(r["citations"], list)


@pytest.mark.asyncio
async def test_empty_question_rejected() -> None:
    with pytest.raises(ValueError, match="empty question"):
        await knowledge.consult("   ")


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch, clear_cache) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await knowledge.consult("q")


@pytest.mark.asyncio
async def test_citations_deduped(monkeypatch: pytest.MonkeyPatch, clear_cache) -> None:
    async def fake_chat(**kw):
        return _mk_response(
            "answer text\n\nCITATIONS: [api-gotchas.md] [api-gotchas.md] [presnyakov-257.md]"
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    r = await knowledge.consult("dedupe test")
    assert r["citations"] == ["api-gotchas.md", "presnyakov-257.md"]


@pytest.mark.asyncio
async def test_citations_with_section_anchors(monkeypatch: pytest.MonkeyPatch, clear_cache) -> None:
    async def fake_chat(**kw):
        return _mk_response(
            "text\n\nCITATIONS: [direct-knowledge-base.md#autostrategies] "
            "[presnyakov-257.md#sezonnost]"
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    r = await knowledge.consult("anchors")
    assert "direct-knowledge-base.md#autostrategies" in r["citations"]
    assert "presnyakov-257.md#sezonnost" in r["citations"]


@pytest.mark.asyncio
async def test_context_oversize_truncated(
    monkeypatch: pytest.MonkeyPatch, clear_cache, caplog: pytest.LogCaptureFixture
) -> None:
    captured: dict = {}

    async def fake_chat(**kw):
        captured.update(kw)
        return _mk_response("ok\n\nCITATIONS: [api-gotchas.md]")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent_runtime.knowledge.LLMClient",
        lambda **_: type("C", (), {"chat": lambda self, **kw: fake_chat(**kw)})(),
    )
    big_ctx = {"huge": "x" * 60_000}
    with caplog.at_level("WARNING", logger="agent_runtime.knowledge"):
        await knowledge.consult("huge context", context=big_ctx)
    assert any("oversize" in r.message for r in caplog.records)
    assert "(truncated)" in captured["prompt"]
