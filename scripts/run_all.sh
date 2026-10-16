#!/usr/bin/env bash
# Start the whole stack locally: mock ticket UI, mock user service, agent runtime, Streamlit dashboard.
# Usage: scripts/run_all.sh        (Ctrl-C stops everything)
# First-time setup:
#   uv venv --python 3.12 && uv pip install -e '.[dev]' && .venv/bin/playwright install chromium
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=.venv/bin
mkdir -p data logs artifacts

pids=()
cleanup() { echo "stopping..."; for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait; }
trap cleanup EXIT INT TERM

wait_http() {  # url, name
  for _ in $(seq 1 50); do curl -sf -o /dev/null "$1" && return 0; sleep 0.2; done
  echo "$2 did not come up at $1" >&2; exit 1
}

$BIN/support-agent ticket-ui --port 8001 & pids+=($!)
$BIN/support-agent user-service --port 8002 & pids+=($!)
wait_http http://127.0.0.1:8001/login "ticket UI"
wait_http http://127.0.0.1:8002/health "user service"

$BIN/support-agent run & pids+=($!)
$BIN/support-agent dashboard --port 8501 & pids+=($!)

echo
echo "  Ticket UI      http://localhost:8001   (login agent / agent)"
echo "  User service   http://localhost:8002/docs"
echo "  Ops API        http://localhost:8003/metrics"
echo "  Dashboard      http://localhost:8501"
echo
wait
