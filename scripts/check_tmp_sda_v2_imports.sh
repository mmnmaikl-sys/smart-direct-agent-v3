#!/usr/bin/env bash
# check_tmp_sda_v2_imports.sh — block imports from /tmp/sda-v2* (24.04 incident, Decision 10).
#
# Used by:
#   - .pre-commit-config.yaml (local hook)
#   - .github/workflows/ci.yml (CI gate — Task 3)
#
# Input:  staged Python files via $@ (from pre-commit) or all python files if none passed.
# Output: exit 0 if clean, exit 1 + explanation if any forbidden pattern found.
#
# KNOWN LIMITATIONS (defense-in-depth: CI AST check + agents-core pinning are parallel layers):
#   - Dynamic imports (__import__, importlib, exec) are NOT caught — by design, regex
#     cannot detect runtime-constructed module names. These are caught by supply chain
#     pinning: agents-core is git+SHA, forbidden modules simply won't resolve.
#   - Non-.py files (Dockerfile, shell, CI yaml) are out of scope — Task 3 CI adds
#     a repo-wide grep pass for literal "/tmp/sda-v2" in any tracked file.
#   - String concat / variable indirection (p = "/tmp/sda-v2"; sys.path.insert(0, p))
#     is NOT caught — same reason as dynamic imports.

set -euo pipefail

# Patterns to block (each a single ERE, grep -E):
#   1. from /tmp/sda-v2... or import /tmp/sda-v2...
#   2. from tmp.sda_v2... / import tmp.sda_v2...  (dotted path variant)
#   3. sys.path.insert/append with "/tmp/sda-v2" literal (single or double quotes)
PATTERNS=(
  '^[[:space:]]*(from|import)[[:space:]]+[^[:space:]#]*(/tmp/sda-?v2|\btmp\.sda[._-]?v2)'
  "sys\.path\.(insert|append)\([^)]*['\"]/tmp/sda-?v2"
)

FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
  # No args → scan all tracked python files (CI mode).
  # Use `while read` loop instead of `mapfile` (bash 3.2 compat on macOS default shell).
  FILES=()
  while IFS= read -r line; do
    FILES+=("$line")
  done < <(git ls-files '*.py')
fi

FOUND=0
for file in "${FILES[@]}"; do
  [[ -f "$file" ]] || continue
  [[ "$file" == *.py ]] || continue
  for pattern in "${PATTERNS[@]}"; do
    if grep -E -n -H "$pattern" "$file" 2>/dev/null; then
      FOUND=1
    fi
  done
done

if [[ $FOUND -eq 1 ]]; then
  cat >&2 <<'MSG'

BLOCKED: import из /tmp/sda-v2* запрещён.
Причина: инцидент 24.04 (deploy из непроверенной /tmp папки снёс 16 jobs на prod).
См. Decision 10 в tech-spec.md.
Используй packaged copy из agent_runtime/* или agents-core.

MSG
  exit 1
fi

exit 0
