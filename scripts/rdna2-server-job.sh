#!/usr/bin/env bash
# Run a server plus benchmark client inside an rdna2-job process group.
set -euo pipefail

PORT=8080
READY_TIMEOUT=600
SERVER=()
CLIENT=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --port)
            PORT=${2:?missing port}
            shift 2
            ;;
        --ready-timeout)
            READY_TIMEOUT=${2:?missing ready timeout}
            shift 2
            ;;
        --server)
            shift
            while [ "$#" -gt 0 ] && [ "$1" != "--client" ]; do
                SERVER+=("$1")
                shift
            done
            ;;
        --client)
            shift
            CLIENT=("$@")
            break
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

[ "${#SERVER[@]}" -gt 0 ] || { echo "ERROR: missing --server command" >&2; exit 2; }
[ "${#CLIENT[@]}" -gt 0 ] || { echo "ERROR: missing --client command" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid port" >&2; exit 2; }
[[ "$READY_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid ready timeout" >&2; exit 2; }

JOB_DIR="${LLAMA_JOB_DIR:-$(mktemp -d)}"
mkdir -p "$JOB_DIR"
SERVER_LOG="$JOB_DIR/server.log"
SERVER_PID_FILE="$JOB_DIR/server.pid"

set_status() {
    printf '%s\n' "$1" > "$JOB_DIR/status.tmp"
    mv "$JOB_DIR/status.tmp" "$JOB_DIR/status"
}

server_pid=""
cleanup() {
    local rc=$?
    trap - EXIT TERM INT HUP
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$server_pid" 2>/dev/null || true
    fi
    [ -n "$server_pid" ] && wait "$server_pid" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT TERM INT HUP

set_status loading-model
"${SERVER[@]}" > "$SERVER_LOG" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "$SERVER_PID_FILE"

ready=0
for _ in $(seq 1 "$READY_TIMEOUT"); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: server exited before becoming ready" >&2
        tail -80 "$SERVER_LOG" >&2 || true
        exit 3
    fi
    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo "ERROR: server did not become ready within ${READY_TIMEOUT}s" >&2
    tail -80 "$SERVER_LOG" >&2 || true
    exit 4
fi

set_status ready
export LLAMA_SERVER_URL="http://127.0.0.1:$PORT"
set_status benchmarking
"${CLIENT[@]}" | tee "$JOB_DIR/result.jsonl"
set_status benchmark-complete