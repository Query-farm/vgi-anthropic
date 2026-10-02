#!/usr/bin/env bash
# vgi-lint's behavioural tier — every shipped example executed, and every
# declared expected_result compared — against the local fake Admin API, so it
# needs no key and no network.
#
#   scripts/lint-behavioural.sh [extra vgi-lint args...]
#
# The anthropic_admin secret type only exists once the worker is attached, and
# vgi-lint's --setup-sql runs before its own ATTACH. So setup attaches the
# worker once, creates the secret, and detaches; the type and the secret
# outlive the DETACH, and vgi-lint's ATTACH then finds both.
set -euo pipefail
cd "$(dirname "$0")/.."

VGI_LINT="${VGI_LINT:-uvx --from vgi-lint-check==0.82.0 vgi-lint}"
WORKER="${WORKER:-uv run --no-sources anthropic_worker.py}"
url_file="$(mktemp)"

uv run --no-sources python -m tests.fake_admin_api > "$url_file" &
fake_pid=$!
trap 'kill "$fake_pid" 2>/dev/null || true; rm -f "$url_file"' EXIT
for _ in $(seq 100); do [[ -s "$url_file" ]] && break; sleep 0.1; done
[[ -s "$url_file" ]] || { echo "fake Admin API did not start" >&2; exit 1; }

VGI_ANTHROPIC_BASE_URL="$(head -1 "$url_file")" $VGI_LINT lint "$WORKER" \
  --execute-mode run --no-check-links --audit-waivers --fail-on warning \
  --setup-sql "SET TimeZone = 'UTC'" \
  --setup-sql "ATTACH 'anthropic' (TYPE vgi, LOCATION '$WORKER')" \
  --setup-sql "CREATE SECRET (TYPE anthropic_admin, api_key 'sk-ant-admin01-test-key')" \
  --setup-sql "DETACH anthropic" \
  "$@"
