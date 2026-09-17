#!/usr/bin/env bash
# End-to-end demo: stream a degrading link and wait for one reroute.
#
#   scripts/demo.sh            # docker compose (falls back to --local if unavailable)
#   scripts/demo.sh --local    # run the five processes locally with background PIDs
#
# Success = a REROUTE_EMITTED / "accepted": true line within TIMEOUT_S seconds.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIMEOUT_S="${TIMEOUT_S:-120}"
PYTHON="${PYTHON:-python3}"
[ -x "$ROOT/.venv/bin/python" ] && PYTHON="$ROOT/.venv/bin/python"
MODE="auto"
SUCCESS_PATTERN='REROUTE_EMITTED|"accepted": *true'

for arg in "$@"; do
  case "$arg" in
    --local) MODE="local" ;;
    --docker) MODE="docker" ;;
    *) echo "usage: demo.sh [--local|--docker]" >&2; exit 2 ;;
  esac
done

if [ "$MODE" = "auto" ]; then
  if docker compose version >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    MODE="docker"
  else
    echo "docker compose unavailable -> running local demo" >&2
    MODE="local"
  fi
fi

LOG_DIR="$ROOT/.cache/demo"
mkdir -p "$LOG_DIR"
COMBINED="$LOG_DIR/demo.log"
: > "$COMBINED"

report_success() {
  echo
  echo "=== DEMO SUCCESS: reroute emitted ==="
  grep -E "$SUCCESS_PATTERN" "$COMBINED" | head -3
  echo "====================================="
}

# --------------------------------------------------------------------------- #
# docker mode
# --------------------------------------------------------------------------- #
run_docker() {
  trap 'docker compose down -v --remove-orphans >/dev/null 2>&1 || true' EXIT
  docker compose up --build -d || return 1
  docker compose logs -f --no-color >"$COMBINED" 2>&1 &
  local logs_pid=$!

  local waited=0
  while [ "$waited" -lt "$TIMEOUT_S" ]; do
    if grep -Eq "$SUCCESS_PATTERN" "$COMBINED"; then
      kill "$logs_pid" 2>/dev/null || true
      report_success
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  kill "$logs_pid" 2>/dev/null || true
  echo "TIMEOUT after ${TIMEOUT_S}s without a reroute" >&2
  tail -40 "$COMBINED" >&2
  return 1
}

# --------------------------------------------------------------------------- #
# local mode
# --------------------------------------------------------------------------- #
PIDS=()

cleanup_local() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}

start_local() {
  local name="$1"
  shift
  ( "$@" >>"$COMBINED" 2>&1 ) &
  local pid=$!
  PIDS+=("$pid")
  echo "started $name (pid $pid)"
}

wait_for_http() {
  local url="$1"
  local tries="${2:-40}"
  for _ in $(seq 1 "$tries"); do
    if "$PYTHON" -c "import sys,urllib.request;urllib.request.urlopen('$url',timeout=2).read()" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

wait_for_tcp() {
  local host="$1" port="$2" tries="${3:-40}"
  for _ in $(seq 1 "$tries"); do
    if "$PYTHON" -c "import socket,sys;s=socket.create_connection(('$host',$port),2);s.close()" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

run_local() {
  local ingest_bin="$ROOT/apps/ingest_cpp/build/optics_ingest"
  if [ ! -x "$ingest_bin" ]; then
    echo "missing $ingest_bin -- run 'make ingest' first" >&2
    return 1
  fi
  if [ ! -f "$ROOT/models/horizon.json" ]; then
    echo "missing model artifacts -- run 'make train' first" >&2
    return 1
  fi

  trap cleanup_local EXIT
  export PYTHONPATH="$ROOT"
  export REROUTE_LOG="$LOG_DIR/reroutes.jsonl"
  export PATH_MANAGER_HOST=127.0.0.1
  export HEALER_HOST=127.0.0.1
  export INFER_LISTEN_HOST=127.0.0.1
  export PATH_MANAGER_REST=http://127.0.0.1:8080
  export PATH_MANAGER_GRPC=127.0.0.1:8081
  export HEALER_URL=http://127.0.0.1:8091

  start_local path_manager "$PYTHON" -m apps.path_manager.main
  wait_for_http "http://127.0.0.1:8080/health" || { echo "path_manager did not start" >&2; return 1; }

  start_local healer "$PYTHON" -m apps.healer.main
  wait_for_http "http://127.0.0.1:8091/health" || { echo "healer did not start" >&2; return 1; }

  start_local infer "$PYTHON" -m apps.infer.main --host 127.0.0.1
  wait_for_http "http://127.0.0.1:8090/health" || { echo "infer did not start" >&2; return 1; }

  start_local ingest "$ingest_bin" --listen 127.0.0.1:9000 --downstream 127.0.0.1:9001
  wait_for_tcp 127.0.0.1 9000 || { echo "ingest did not start" >&2; return 1; }

  start_local producer "$PYTHON" -m apps.producer.main stream --host 127.0.0.1 --port 9000 --demo

  local waited=0
  while [ "$waited" -lt "$TIMEOUT_S" ]; do
    if grep -Eq "$SUCCESS_PATTERN" "$COMBINED"; then
      report_success
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "TIMEOUT after ${TIMEOUT_S}s without a reroute" >&2
  tail -40 "$COMBINED" >&2
  return 1
}

echo "optics-softfail demo: mode=$MODE timeout=${TIMEOUT_S}s log=$COMBINED"
if [ "$MODE" = "docker" ]; then
  run_docker
else
  run_local
fi
