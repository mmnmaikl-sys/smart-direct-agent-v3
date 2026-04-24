"""Tests for scripts/check_tmp_sda_v2_imports.sh (Task 2 TDD Anchor).

The hook must block imports originating from /tmp/sda-v2* paths (the 24.04 incident
root cause) while allowing the same text inside comments/docstrings and legitimate
`agent_runtime` imports.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_tmp_sda_v2_imports.sh"


def _run_hook(tmp_path: Path, content: str) -> subprocess.CompletedProcess[str]:
    """Write content to a .py file in tmp_path and run the hook against it."""
    target = tmp_path / "candidate.py"
    target.write_text(content)
    return subprocess.run(
        [str(SCRIPT), str(target)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def test_blocks_from_tmp_sda_v2(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, "from /tmp/sda-v2.brain import reason\n")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "BLOCKED" in combined
    assert "24.04" in combined or "Decision 10" in combined


def test_blocks_import_tmp_sda_v2_variant(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, "import tmp.sda_v2.tools as t\n")
    assert result.returncode != 0
    assert "BLOCKED" in result.stdout + result.stderr


def test_blocks_sys_path_insert(tmp_path: Path) -> None:
    content = "import sys\n" 'sys.path.insert(0, "/tmp/sda-v2")\n' "from sda_v2 import tools\n"
    result = _run_hook(tmp_path, content)
    assert result.returncode != 0
    assert "BLOCKED" in result.stdout + result.stderr


def test_allows_clean_code(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, "from agent_runtime.brain import reason\n")
    assert (
        result.returncode == 0
    ), f"Clean code should not be blocked. stdout={result.stdout} stderr={result.stderr}"


def test_allows_comment_mentioning_tmp_sda_v2(tmp_path: Path) -> None:
    """Regex must not match when /tmp/sda-v2 appears only in a comment or docstring.

    Tasks 7/10/11/12 reference v2 source files in their Context Files as historical
    pointers; those comments must survive.
    """
    content = (
        '"""Test file — ported logic from /tmp/sda-v2-inspect/app/brain.py."""\n'
        "# see /tmp/sda-v2/app/agents/strategy_gate.py for original\n"
        "from agent_runtime.brain import reason\n"
    )
    result = _run_hook(tmp_path, content)
    assert (
        result.returncode == 0
    ), f"Comment/docstring mentions must pass. stdout={result.stdout} stderr={result.stderr}"
