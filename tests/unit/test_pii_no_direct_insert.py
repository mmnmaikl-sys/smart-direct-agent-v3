"""Enforce: ``audit_log`` writes only go through sanctioned entry points.

The primary rule is Decision 13: ``agent_runtime/db.py::insert_audit_log``
(which runs ``sanitize_audit_payload``) is the only path that ships Bitrix
/ Direct / Metrika payloads into audit_log.

Narrow exemption: ``agent_runtime/trust_levels.py`` writes trust-transition
metadata (``old``/``new``/``actor``/``reason`` — strings known to be PII-free)
inside the same transaction as the ``sda_state`` update, so going through
``insert_audit_log`` (which opens its own connection) would break atomicity.
The grep test allow-lists that module with documented rationale.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = ROOT / "agent_runtime"

_ALLOWED_FILES: frozenset[str] = frozenset(
    {
        "db.py",  # insert_audit_log wrapper — always sanitises
        "trust_levels.py",  # internal trust-transition audit inside same-tx
    }
)


def test_no_direct_insert_into_audit_log_outside_db_module() -> None:
    violations: list[tuple[Path, int, str]] = []
    for py_file in SOURCE.rglob("*.py"):
        if py_file.name in _ALLOWED_FILES:
            continue
        text = py_file.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if "insert into audit_log" in lowered:
                violations.append((py_file, i, line.strip()))
    assert not violations, (
        "Found direct INSERT INTO audit_log outside allowed modules "
        f"(bypasses PII sanitiser — Decision 13): {violations}"
    )
