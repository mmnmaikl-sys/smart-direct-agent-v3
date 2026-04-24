"""Smoke test: ruff passes on the current agent_runtime skeleton."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ruff_check_passes_on_agent_runtime() -> None:
    result = subprocess.run(
        ["ruff", "check", "agent_runtime"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"ruff check failed:\nstdout={result.stdout}\nstderr={result.stderr}"


def test_ruff_format_check_passes_on_agent_runtime() -> None:
    result = subprocess.run(
        ["ruff", "format", "--check", "agent_runtime"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"ruff format --check failed:\nstdout={result.stdout}\nstderr={result.stderr}"
