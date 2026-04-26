"""Tests for pre_launch_validator pure assess_landing function."""

from __future__ import annotations

from agent_runtime.jobs.pre_launch_validator import (
    LandingResult,
    assess_landing,
    format_telegram,
)

_GOOD_HTML = (
    "<!DOCTYPE html><html><head><title>BFL</title></head><body>"
    "<h1>Banking 127-FZ disclaimer ФЗ-38</h1>"
    "<form action='/submit'><input type='text'/></form>" + "x" * 1500 + "</body></html>"
).encode("utf-8")


def test_clean_landing_no_issues() -> None:
    r = assess_landing(
        persona="rabotyaga",
        url="https://24bankrotsttvo.ru/lp/rabotyaga/",
        http_status=200,
        body=_GOOD_HTML,
        pagespeed_mobile=92,
    )
    assert r.is_ok
    assert r.severity == "GREEN"


def test_unreachable_marks_red() -> None:
    r = assess_landing(
        persona="mother",
        url="https://24bankrotsttvo.ru/lp/mother/",
        http_status=404,
        body=b"",
        pagespeed_mobile=None,
    )
    assert not r.is_ok
    assert r.severity == "RED"
    assert any("unreachable" in i for i in r.issues)


def test_missing_form_marks_red() -> None:
    body = b"<html><body><h1>FZ-38</h1>" + b"y" * 2000 + b"</body></html>"  # no form
    r = assess_landing(
        persona="pensioner",
        url="https://24bankrotsttvo.ru/lp/pensioner/",
        http_status=200,
        body=body,
        pagespeed_mobile=90,
    )
    assert not r.is_ok
    assert r.severity == "RED"  # no form is a HARD gate
    assert any("form" in i.lower() for i in r.issues)


def test_pagespeed_low_marks_amber() -> None:
    r = assess_landing(
        persona="property",
        url="https://24bankrotsttvo.ru/lp/property/",
        http_status=200,
        body=_GOOD_HTML,
        pagespeed_mobile=70,  # < 85
    )
    assert not r.is_ok
    assert r.severity == "AMBER"
    assert any("PageSpeed" in i for i in r.issues)


def test_missing_fz38_marks_amber() -> None:
    body = (
        b"<html><body><h1>BFL</h1><form><input/></form>"
        + b"y" * 2000
        + b"</body></html>"  # no FZ-38
    )
    r = assess_landing(
        persona="mfo",
        url="https://24bankrotsttvo.ru/lp/mfo/",
        http_status=200,
        body=body,
        pagespeed_mobile=92,
    )
    assert not r.is_ok
    assert r.severity == "AMBER"
    assert any("FZ-38" in i or "ФЗ" in i for i in r.issues)


def test_pagespeed_none_does_not_block() -> None:
    """If PageSpeed API was unreachable (rate limit), don't fail the gate."""
    r = assess_landing(
        persona="rabotyaga",
        url="https://24bankrotsttvo.ru/lp/rabotyaga/",
        http_status=200,
        body=_GOOD_HTML,
        pagespeed_mobile=None,
    )
    assert r.is_ok


def test_format_telegram_clean() -> None:
    results = [
        LandingResult(
            persona="rabotyaga",
            url="x",
            reachable=True,
            http_status=200,
            body_bytes=2000,
            has_form=True,
            fz38_present=True,
            pagespeed_mobile=92,
        )
    ]
    msg = format_telegram(results)
    assert "прошли проверку" in msg
    assert "RED" not in msg


def test_format_telegram_split_red_amber() -> None:
    red = LandingResult(
        persona="mother",
        url="x",
        reachable=False,
        http_status=500,
        body_bytes=0,
        has_form=False,
        fz38_present=False,
        pagespeed_mobile=None,
        issues=("unreachable: status=500", "no <form>"),
    )
    amber = LandingResult(
        persona="property",
        url="y",
        reachable=True,
        http_status=200,
        body_bytes=2000,
        has_form=True,
        fz38_present=True,
        pagespeed_mobile=70,
        issues=("PageSpeed mobile 70 < 85",),
    )
    msg = format_telegram([red, amber])
    assert "RED (1)" in msg
    assert "AMBER (1)" in msg
    assert "mother" in msg
    assert "property" in msg
