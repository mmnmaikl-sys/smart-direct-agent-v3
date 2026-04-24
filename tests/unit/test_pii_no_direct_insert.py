"""Enforce: only ``agent_runtime.db.insert_audit_log`` writes to ``audit_log``.

Any other ``INSERT INTO audit_log`` in the codebase would bypass the PII
sanitiser (Decision 13). The check is a simple grep over ``agent_runtime/``;
it runs in CI so regressions are caught on PR.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = ROOT / "agent_runtime"


def test_no_direct_insert_into_audit_log_outside_db_module() -> None:
    violations: list[tuple[Path, int, str]] = []
    for py_file in SOURCE.rglob("*.py"):
        # The only legitimate callsite is agent_runtime/db.py::insert_audit_log
        if py_file.name == "db.py":
            continue
        text = py_file.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if "insert into audit_log" in lowered:
                violations.append((py_file, i, line.strip()))
    assert not violations, (
        "Found direct INSERT INTO audit_log outside agent_runtime/db.py "
        f"(bypasses PII sanitiser — Decision 13): {violations}"
    )
