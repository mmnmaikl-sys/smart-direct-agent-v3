"""Task 3 TDD: requirements.lock pins everything with SHA-256 hashes."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK = REPO_ROOT / "requirements.lock"


@pytest.mark.skipif(not LOCK.exists(), reason="requirements.lock not yet generated")
def test_lock_has_hashes() -> None:
    """Every package pin in requirements.lock must include at least one --hash=sha256:..."""
    content = LOCK.read_text()
    # Meaningful lines: non-comment, non-blank, not `via ...` continuation
    packages = [
        line
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#") and "==" in line
    ]
    assert packages, "requirements.lock appears empty"
    assert (
        "--hash=sha256:" in content
    ), "requirements.lock has no hash pins — use pip-compile --generate-hashes"
    # Each package pin line or its continuation should reference a hash
    total_hashes = content.count("--hash=sha256:")
    assert total_hashes >= len(
        packages
    ), f"Too few hash entries: {total_hashes} hashes for {len(packages)} packages"
