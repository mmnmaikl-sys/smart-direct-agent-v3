"""Task 3 TDD: smoke_test.sh passes shellcheck."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_smoke_test_shellcheck_clean() -> None:
    result = subprocess.run(
        ["shellcheck", "scripts/smoke_test.sh"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"shellcheck errors in smoke_test.sh:\n{result.stdout}\n{result.stderr}"


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_grill_audit_shellcheck_clean() -> None:
    result = subprocess.run(
        ["shellcheck", "scripts/grill_audit.sh"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"shellcheck errors in grill_audit.sh:\n{result.stdout}\n{result.stderr}"


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_check_tmp_sda_v2_imports_shellcheck_clean() -> None:
    result = subprocess.run(
        ["shellcheck", "scripts/check_tmp_sda_v2_imports.sh"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"shellcheck errors in check_tmp_sda_v2_imports.sh:\n{result.stdout}\n{result.stderr}"
