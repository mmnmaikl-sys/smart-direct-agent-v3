#!/usr/bin/env bash
# smoke_test.sh — post-deploy smoke check for smart-direct-agent-v3.
#
# Called from .github/workflows/deploy.yml after Railway deploy completes.
# Fails (exit non-zero) if /health is unreachable or does not return status=ok.

set -euo pipefail

URL="${SMOKE_URL:-https://smart-direct-agent-v3-production.up.railway.app}"
MAX_ATTEMPTS="${SMOKE_MAX_ATTEMPTS:-10}"
SLEEP_BETWEEN="${SMOKE_SLEEP:-5}"

echo "Smoke test: ${URL}/health"

attempt=0
while [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; do
  attempt=$((attempt + 1))
  echo "Attempt ${attempt}/${MAX_ATTEMPTS}..."

  # -f: fail on HTTP >=400, -s: silent, -S: show errors, -L: follow redirects
  if response=$(curl -fsSL --max-time 15 "${URL}/health" 2>&1); then
    echo "Response: ${response}"
    if echo "${response}" | grep -q '"status":\s*"ok"'; then
      echo "Smoke test PASSED"
      exit 0
    else
      echo "Smoke test FAILED: /health did not return status=ok"
      exit 1
    fi
  fi

  echo "Health check failed, retrying in ${SLEEP_BETWEEN}s..."
  sleep "${SLEEP_BETWEEN}"
done

echo "Smoke test FAILED: /health unreachable after ${MAX_ATTEMPTS} attempts"
exit 1
