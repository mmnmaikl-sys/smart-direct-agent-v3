"""Tests for ad_rewriter pure helpers — JSON parser, top-N picker, prompt builder."""

from __future__ import annotations

from agent_runtime.jobs.ad_quality_assessor import AdAssessment
from agent_runtime.jobs.ad_rewriter import (
    _build_user_prompt,
    _parse_llm_json,
    _select_top_n,
)


def _mk(verdict: str, ad_id: int = 1) -> AdAssessment:
    return AdAssessment(
        ad_id=ad_id,
        campaign_id=709353005,
        name_hint="rabotyaga",
        title="Title",
        title2="T2",
        body="Body",
        fz38_violations=[],
        quality_issues=["короткий Title"],
        verdict=verdict,
    )


def test_parse_clean_json() -> None:
    text = '{"title":"A","title2":"B","body":"C","rationale":"R"}'
    out = _parse_llm_json(text)
    assert out == {"title": "A", "title2": "B", "body": "C", "rationale": "R"}


def test_parse_strips_markdown_fences() -> None:
    text = '```json\n{"title":"A","title2":"","body":"X","rationale":""}\n```'
    out = _parse_llm_json(text)
    assert out["title"] == "A"
    assert out["body"] == "X"


def test_parse_handles_text_before_json() -> None:
    text = 'Вот переписанное:\n{"title":"A","title2":"","body":"X","rationale":""}'
    out = _parse_llm_json(text)
    assert out["title"] == "A"


def test_parse_returns_empty_on_garbage() -> None:
    out = _parse_llm_json("not json at all")
    assert out == {}


def test_parse_returns_empty_on_empty_string() -> None:
    assert _parse_llm_json("") == {}


def test_parse_handles_missing_fields() -> None:
    text = '{"title":"A"}'
    out = _parse_llm_json(text)
    assert out["title"] == "A"
    assert out["title2"] == ""
    assert out["body"] == ""
    assert out["rationale"] == ""


def test_select_top_n_prioritises_auto_pause() -> None:
    items = [
        _mk("NEEDS_REWRITE", 1),
        _mk("AUTO_PAUSE", 2),
        _mk("NEEDS_REWRITE", 3),
        _mk("AUTO_PAUSE", 4),
        _mk("NEEDS_REWRITE", 5),
    ]
    out = _select_top_n(items, 3)
    assert [a.ad_id for a in out] == [2, 4, 1]  # AUTO_PAUSE first, then NEEDS_REWRITE


def test_select_top_n_returns_all_when_fewer_than_n() -> None:
    items = [_mk("NEEDS_REWRITE", 1), _mk("AUTO_PAUSE", 2)]
    out = _select_top_n(items, 5)
    assert len(out) == 2


def test_build_user_prompt_includes_original_and_issues() -> None:
    a = AdAssessment(
        ad_id=42,
        campaign_id=709353005,
        name_hint="rabotyaga",
        title="100% списание долгов",
        title2="Гарантия",
        body="Без последствий и проблем",
        fz38_violations=["ФЗ-38: '100%'", "ФЗ-38: 'гаранти'"],
        quality_issues=["Заглавное слово"],
        verdict="AUTO_PAUSE",
    )
    p = _build_user_prompt(a)
    assert "100%" in p
    assert "ФЗ-38" in p
    assert "AUTO_PAUSE" in p
    assert "rabotyaga" in p
