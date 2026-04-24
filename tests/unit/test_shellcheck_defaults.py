"""Task 3 TDD: enforce shell script conventions (shebang + strict mode)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

SHELL_SCRIPTS = sorted(SCRIPTS_DIR.glob("*.sh"))


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_all_shell_scripts_have_shebang(script: Path) -> None:
    """Each script starts with `#!/usr/bin/env bash` (portable, not hardcoded path)."""
    first_line = script.read_text().splitlines()[0]
    assert (
        first_line == "#!/usr/bin/env bash"
    ), f"{script.name}: shebang must be `#!/usr/bin/env bash`, got {first_line!r}"


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_all_shell_scripts_have_strict_mode(script: Path) -> None:
    """Each script contains `set -euo pipefail` in the first 20 lines."""
    head = "\n".join(script.read_text().splitlines()[:20])
    assert "set -euo pipefail" in head, f"{script.name}: missing `set -euo pipefail` in header"
