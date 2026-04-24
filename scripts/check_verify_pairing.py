#!/usr/bin/env python3
"""AST gate: every mutating ``async def`` in ``DirectAPI`` has a verify pair.

Run by CI job ``verify-pairing``. The mapping below encodes the rename from
mutation verb → verify name (``pause_group`` → ``verify_group_paused``); any
new mutation the team adds must be registered here or the build fails, which
forces the reviewer to justify why a verify pair is unnecessary.

Exit codes:
    0 — every mutation has its registered verify counterpart
    1 — at least one mutation is unpaired OR an unknown mutation appeared
    2 — usage / parse error
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Mutation prefix → expected verify name. New entries must be added here
# deliberately (reviewer gate). Unknown mutation prefixes trip exit 1.
EXPECTED_VERIFY: dict[str, str] = {
    "set_bid": "verify_bid",
    "add_negatives": "verify_negatives_added",
    "pause_group": "verify_group_paused",
    "resume_group": "verify_group_resumed",
    "pause_campaign": "verify_campaign_paused",
    "resume_campaign": "verify_campaign_resumed",
    "update_ad_href": "verify_ad_href",
    "update_strategy": "verify_strategy",
}

MUTATION_PREFIXES: tuple[str, ...] = ("set_", "add_", "pause_", "resume_", "update_")


def _collect_async_methods(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            if node.name.startswith("_"):
                continue
            names.add(node.name)
    return names


def check_module(path: Path) -> int:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        print(f"cannot parse {path}: {exc}", file=sys.stderr)
        return 2

    methods = _collect_async_methods(tree)
    mutations = sorted(m for m in methods if any(m.startswith(p) for p in MUTATION_PREFIXES))

    bad: list[tuple[str, str]] = []
    paired = 0
    for name in mutations:
        expected = EXPECTED_VERIFY.get(name)
        if expected is None:
            bad.append(
                (
                    name,
                    f"MUTATION WITHOUT MAPPING: {name} is not in EXPECTED_VERIFY. "
                    "Add it deliberately.",
                )
            )
            continue
        if expected not in methods:
            bad.append(
                (
                    name,
                    f"MUTATION WITHOUT VERIFY: {name} expects {expected}, but "
                    f"{expected} is not defined in the module.",
                )
            )
            continue
        paired += 1

    if bad:
        for _, msg in bad:
            print(msg, file=sys.stderr)
        return 1

    print(f"OK: {paired} mutations paired with verify methods.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--module", required=True, type=Path)
    args = parser.parse_args(argv)
    return check_module(args.module)


if __name__ == "__main__":
    sys.exit(main())
