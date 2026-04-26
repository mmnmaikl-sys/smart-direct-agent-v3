"""Pure-rule landing critic — ФЗ-38 disclaimer + structure check."""

from __future__ import annotations

from agent_runtime.jobs.landing_critic import assess_landing

_GOOD_HTML = """
<!doctype html>
<html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Списать долги</title>
</head><body>
<h1>Спишем долг от 250 000 ₽ — Казань</h1>
<form><input name="phone"><button>Получить расчёт</button></form>
<p>Работаем с СРО арбитражных управляющих. ИНН 7700000000, ОГРН 1234567890123.</p>
<p>Банкротство влечёт негативные последствия, в том числе ограничения на получение
кредита и повторное банкротство в течение пяти лет. Предварительно обратитесь к
своему кредитору и в МФЦ.</p>
</body></html>
"""


def test_clean_landing_approves() -> None:
    a = assess_landing("https://example.com/lp", 200, _GOOD_HTML)
    assert a.verdict == "APPROVE"
    assert a.fz38_violations == []
    assert a.structural_issues == []


def test_no_h1_needs_rewrite() -> None:
    html = _GOOD_HTML.replace("<h1>Спишем долг от 250 000 ₽ — Казань</h1>", "")
    a = assess_landing("https://example.com/lp", 200, html)
    assert a.verdict == "NEEDS_REWRITE"
    assert any("h1" in i.lower() for i in a.structural_issues)


def test_h1_without_number_needs_rewrite() -> None:
    html = _GOOD_HTML.replace("Спишем долг от 250 000 ₽ — Казань", "Списание долгов в Казани")
    a = assess_landing("https://example.com/lp", 200, html)
    assert a.verdict == "NEEDS_REWRITE"


def test_no_form_needs_rewrite() -> None:
    html = _GOOD_HTML.replace(
        '<form><input name="phone"><button>Получить расчёт</button></form>', ""
    )
    a = assess_landing("https://example.com/lp", 200, html)
    assert a.verdict == "NEEDS_REWRITE"


def test_no_disclaimer_needs_rewrite() -> None:
    html = _GOOD_HTML.replace(
        "Банкротство влечёт негативные последствия, в том числе ограничения на получение\n"
        "кредита и повторное банкротство в течение пяти лет. Предварительно обратитесь к\n"
        "своему кредитору и в МФЦ.",
        "",
    )
    a = assess_landing("https://example.com/lp", 200, html)
    assert a.verdict == "NEEDS_REWRITE"
    assert any("дисклеймер" in i.lower() or "ФЗ-38" in i for i in a.structural_issues)


def test_fz38_banned_phrase_auto_flag() -> None:
    html = _GOOD_HTML.replace("Спишем долг от 250 000 ₽", "100% гарантия списания долгов")
    a = assess_landing("https://example.com/lp", 200, html)
    assert a.verdict == "AUTO_FLAG"
    assert len(a.fz38_violations) >= 1


def test_no_viewport_needs_rewrite() -> None:
    html = _GOOD_HTML.replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1">', ""
    )
    a = assess_landing("https://example.com/lp", 200, html)
    assert a.verdict == "NEEDS_REWRITE"
    assert any("viewport" in i.lower() or "mobile" in i.lower() for i in a.structural_issues)


def test_http_error_returns_fetch_error() -> None:
    a = assess_landing("https://example.com/lp", 500, "")
    assert a.verdict == "FETCH_ERROR"


def test_zero_status_treated_as_fetch_error() -> None:
    a = assess_landing("https://example.com/lp", 0, "")
    assert a.verdict == "FETCH_ERROR"
