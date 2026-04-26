"""Pure-rule ad quality assessor — ФЗ-38 + basic Title/Body checks.

This is the deterministic layer. LLM rewrite is a v2 add-on. Rules
derived from KB ad_quality_bfl_2026-04.md (Deep Research 2026-04).

Verdict ordering:
  AUTO_PAUSE  — fz38_violation OR title/body has guarantee/100%
  NEEDS_REWRITE — quality issues (no number, no CTA, too short, etc.)
  APPROVE — clean
"""

from __future__ import annotations

from agent_runtime.jobs.ad_quality_assessor import (
    assess_ad,
)


def _ad(
    *, ad_id: int = 1, cid: int = 709353005, title: str = "", title2: str = "", body: str = ""
) -> dict:
    return {
        "Id": ad_id,
        "CampaignId": cid,
        "AdGroupId": 1,
        "State": "ON",
        "Status": "ACCEPTED",
        "TextAd": {"Title": title, "Title2": title2, "Text": body},
    }


# --- ФЗ-38 (priority 1, AUTO_PAUSE) ----------------------------------------


def test_fz38_100_percent_in_title_auto_pause() -> None:
    a = assess_ad(_ad(title="100% списание долгов — Казань", body="За 6 мес"))
    assert a.verdict == "AUTO_PAUSE"
    assert any("100%" in v.lower() or "fz" in v.lower() for v in a.fz38_violations)


def test_fz38_garantiruem_in_body_auto_pause() -> None:
    a = assess_ad(_ad(title="Спишем долги от 250К — Уфа", body="Гарантируем результат за 6 мес"))
    assert a.verdict == "AUTO_PAUSE"
    assert any("гаранти" in v.lower() for v in a.fz38_violations)


def test_fz38_gosprogramma_auto_pause() -> None:
    a = assess_ad(_ad(title="Государственная программа списания", body="Подайте заявку"))
    assert a.verdict == "AUTO_PAUSE"


def test_fz38_kreditnaya_amnistiya_auto_pause() -> None:
    a = assess_ad(_ad(title="Кредитная амнистия 2025", body="Спишем все долги"))
    assert a.verdict == "AUTO_PAUSE"


def test_fz38_ne_platite_auto_pause() -> None:
    a = assess_ad(_ad(title="Не платите кредиты", body="Поможем оформить банкротство"))
    assert a.verdict == "AUTO_PAUSE"


def test_fz38_bez_posledstviy_auto_pause() -> None:
    a = (
        assess_ad(title="Списание долгов без последствий", body="За 6 мес")
        if False
        else assess_ad(_ad(title="Списание долгов без последствий", body="За 6 мес"))
    )
    assert a.verdict == "AUTO_PAUSE"


def test_fz38_case_insensitive() -> None:
    """Banned phrases must match regardless of case."""
    a = assess_ad(_ad(title="ГАРАНТИЯ списания", body=""))
    assert a.verdict == "AUTO_PAUSE"


# --- quality checks (NEEDS_REWRITE) ----------------------------------------


def test_clean_title_with_number_cta_approve() -> None:
    a = assess_ad(
        _ad(
            title="Спишем долг от 250 000 ₽ — Казань",
            title2="Бесплатная консультация юриста",
            body="5 кредитов + 8 микрозаймов = 1 процедура. Получить расчёт.",
        )
    )
    assert a.verdict == "APPROVE"


def test_no_number_in_title_or_body_needs_rewrite() -> None:
    a = assess_ad(
        _ad(
            title="Услуги юриста по банкротству",
            body="Бесплатная консультация юриста. Записаться легко.",
        )
    )
    assert a.verdict == "NEEDS_REWRITE"
    assert any("цифр" in i.lower() or "число" in i.lower() for i in a.quality_issues)


def test_title_too_short_needs_rewrite() -> None:
    a = assess_ad(_ad(title="Долги", body="Спишем за 6 мес от 79 000 ₽"))
    assert a.verdict == "NEEDS_REWRITE"
    assert any("длин" in i.lower() or "коротк" in i.lower() for i in a.quality_issues)


def test_title_too_long_needs_rewrite() -> None:
    long_title = "Спишем абсолютно все ваши проблемные долги " * 3  # > 56 chars
    a = assess_ad(_ad(title=long_title, body="За 6 мес от 79 000"))
    assert a.verdict == "NEEDS_REWRITE"


def test_uppercase_word_in_title_needs_rewrite() -> None:
    a = assess_ad(_ad(title="СПИШЕМ долги от 250 000 ₽", body="Бесплатно за 6 мес"))
    assert a.verdict == "NEEDS_REWRITE"
    assert any("заглавн" in i.lower() or "uppercase" in i.lower() for i in a.quality_issues)


def test_multiple_exclamation_needs_rewrite() -> None:
    a = assess_ad(_ad(title="Спишем долги!!! от 250К", body="За 6 мес"))
    assert a.verdict == "NEEDS_REWRITE"


# --- skip non-ON ads --------------------------------------------------------


def test_off_ad_returns_skip_verdict() -> None:
    ad = _ad(title="empty", body="empty")
    ad["State"] = "OFF"
    a = assess_ad(ad)
    assert a.verdict == "SKIP"


def test_no_textad_block_returns_skip() -> None:
    ad = _ad()
    del ad["TextAd"]
    a = assess_ad(ad)
    assert a.verdict == "SKIP"
