#!/usr/bin/env bash
# check_tmp_sda_v2_imports.sh — block imports from /tmp/sda-v2* (24.04 incident, Decision 10).
#
# Used by:
#   - .pre-commit-config.yaml (local hook)
#   - .github/workflows/ci.yml (CI gate — Task 3)
#
# Input:  staged Python files via $@ (from pre-commit) or all python files if none passed.
# Output: exit 0 if clean, exit 1 + explanation if any forbidden pattern found.

set -euo pipefail

# Patterns to block (each a single ERE, grep -E):
#   1. from /tmp/sda-v2...
#   2. import /tmp/sda-v2...
#   3. from tmp.sda_v2... / import tmp.sda_v2...
#   4. sys.path.insert/append with "/tmp/sda-v2" literal
PATTERNS=(
  '^[[:space:]]*(from|import)[[:space:]]+[^[:space:]#]*(/tmp/sda-?v2|\btmp\.sda[._-]?v2)'
  'sys\.path\.(insert|append)\([^)]*["\x27]/tmp/sda-?v2'
)

FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
  # No args → scan all tracked python files (CI mode)
  mapfile -t FILES < <(git ls-files '*.py')
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
