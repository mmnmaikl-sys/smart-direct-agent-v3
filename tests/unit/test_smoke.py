"""Smoke tests for Task 1 bootstrap: editable install + pyproject correctness."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_package_importable() -> None:
    """Editable install works, package importable, version is 0.1.0."""
    import agent_runtime

    assert agent_runtime.__version__ == "0.1.0"


def test_pyproject_parseable() -> None:
    """pyproject.toml is valid TOML and contains required fields.

    Validates:
      - project.name = smart-direct-agent-v3
      - project.dependencies contains agents-core pinned by git+ (Decision 2: SHA pinning)
    """
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    assert data["project"]["name"] == "smart-direct-agent-v3"
    deps = data["project"]["dependencies"]
    assert any(
        d.startswith("agents-core") and "git+" in d for d in deps
    ), "agents-core must be pinned via git+https:// (Decision 2: SHA pinning, not tag)"
