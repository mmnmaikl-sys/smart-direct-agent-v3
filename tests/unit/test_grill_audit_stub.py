"""Task 3 TDD: grill_audit.sh stub exits zero on Wave 0."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_grill_audit_exits_zero() -> None:
    result = subprocess.run(
        ["bash", "scripts/grill_audit.sh"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"grill_audit.sh must exit 0 on Wave 0 stub. stderr={result.stderr}"
    assert (
        "stub" in result.stdout.lower()
    ), f"grill_audit.sh output should mention 'stub'. stdout={result.stdout}"
