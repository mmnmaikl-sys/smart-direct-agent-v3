"""Tests for main_campaigns_watchdog.assess_main pure rule."""

from __future__ import annotations

from agent_runtime.jobs.main_campaigns_watchdog import assess_main


def _camp(
    state: str, status: str = "ACCEPTED", payment: str = "ALLOWED", cid: int = 708978456
) -> dict:
    return {"Id": cid, "Name": "test", "State": state, "Status": status, "StatusPayment": payment}


def test_suspended_with_payment_ok_triggers_resume() -> None:
    s = assess_main(_camp("SUSPENDED"))
    assert s.needs_resume is True
    assert s.needs_alert is False


def test_on_state_does_nothing() -> None:
    s = assess_main(_camp("ON"))
    assert s.needs_resume is False
    assert s.needs_alert is False


def test_off_state_does_nothing_manual_pause() -> None:
    """OFF = owner-paused. Don't override owner's decision."""
    s = assess_main(_camp("OFF"))
    assert s.needs_resume is False
    assert s.needs_alert is False


def test_suspended_no_payment_alerts_only() -> None:
    s = assess_main(_camp("SUSPENDED", payment="DISALLOWED"))
    assert s.needs_resume is False
    assert s.needs_alert is True
    assert "пополнить" in s.alert_reason.lower() or "баланс" in s.alert_reason.lower()


def test_suspended_moderation_alerts_only() -> None:
    s = assess_main(_camp("SUSPENDED", status="REJECTED"))
    assert s.needs_resume is False
    assert s.needs_alert is True
    assert "модер" in s.alert_reason.lower()


def test_known_main_cid_gets_friendly_name() -> None:
    s = assess_main(_camp("ON", cid=708978456))
    assert "Башкортостан" in s.name


def test_unknown_cid_gets_str_name() -> None:
    s = assess_main(_camp("ON", cid=999999999))
    assert s.name == "999999999"
