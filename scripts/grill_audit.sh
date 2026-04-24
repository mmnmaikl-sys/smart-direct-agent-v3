#!/usr/bin/env bash
# grill_audit.sh — STUB for Wave 0/1/2.
#
# Real 12-case audit (from smart-agent-methodology SKILL.md) lands in Wave 3 Task 29.
# On Wave 0/1/2 this script prints the scope and exits 0 so CI stays green.
#
# 12 cases to implement in Wave 3:
#   Knowledge layer: (1) KB files on prod, (2) KB used in code (grep kb.consult),
#     (3) Rules YAML == code.
#   API layer:        (4) All tokens valid, (5) Each SET verified by GET.
#   Decision Engine:  (6) full-cycle dry-run OK, (7) full-cycle prod OK,
#     (8) result contains course_advice (KB citation).
#   Deploy:           (9) All files on prod (ls -la), (10) Cron installed,
#     (11) Last cron log non-empty + no errors.
#   Result:           (12) Agent actually did something useful (not just "checked & did nothing").

set -euo pipefail

echo "grill_audit: stub (Wave 0-2). 12 cases to be implemented in Wave 3 Task 29."
echo "See smart-agent-methodology SKILL.md for the canonical 12-point grill."
exit 0
