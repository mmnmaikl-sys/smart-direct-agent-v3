"""Smoke test: pyright passes on the current agent_runtime skeleton."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pyright_passes_on_agent_runtime() -> None:
    result = subprocess.run(
        ["pyright", "agent_runtime"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"pyright failed:\nstdout={result.stdout}\nstderr={result.stderr}"
