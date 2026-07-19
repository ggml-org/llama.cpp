#!/usr/bin/env bash
# Non-blocking GPU job runner for RDNA2 llama.cpp development.
set -euo pipefail

JOB_ROOT="${LLAMA_JOB_ROOT:-$HOME/llama-jobs}"
DEFAULT_TIMEOUT="${LLAMA_JOB_TIMEOUT:-3600}"
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
mkdir -p "$JOB_ROOT"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

valid_name() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

job_dir() {
    echo "$JOB_ROOT/$1"
}

write_status() {
    local dir=$1 state=$2
    printf '%s\n' "$state" > "$dir/status.tmp"
    mv "$dir/status.tmp" "$dir/status"
}

show_status() {
    local id=$1 dir
    dir="$(job_dir "$id")"
    [ -d "$dir" ] || fail "unknown job: $id"
    local state="unknown" pid="" pgid="" exit_code=""
    [ -f "$dir/status" ] && state="$(<"$dir/status")"
    [ -f "$dir/pid" ] && pid="$(<"$dir/pid")"
    [ -f "$dir/pgid" ] && pgid="$(<"$dir/pgid")"
    [ -f "$dir/exit_code" ] && exit_code="$(<"$dir/exit_code")"
    if [[ "$state" =~ ^(queued|running|loading-model|ready|benchmarking)$ ]] &&
       [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        state="stale"
    fi
    printf 'id=%s status=%s' "$id" "$state"
    [ -n "$pid" ] && printf ' pid=%s' "$pid"
    [ -n "$pgid" ] && printf ' pgid=%s' "$pgid"
    [ -n "$exit_code" ] && printf ' exit=%s' "$exit_code"
    printf '\n'
    [ -f "$dir/started_at" ] && printf 'started=%s\n' "$(<"$dir/started_at")"
    [ -f "$dir/finished_at" ] && printf 'finished=%s\n' "$(<"$dir/finished_at")"
    [ -f "$dir/command.sh" ] && printf 'command=%s\n' "$(<"$dir/command.sh")"
}

run_job() {
    local id=$1 timeout_s=$2
    shift 2
    [ "${1:-}" = "--" ] || fail "internal runner invocation missing --"
    shift
    local dir
    dir="$(job_dir "$id")"
    printf '%s\n' "$$" > "$dir/pid"
    ps -o pgid= -p $$ | tr -d ' ' > "$dir/pgid"
    date --iso-8601=seconds > "$dir/started_at"

    local terminal=0
    on_term() {
        terminal=1
        write_status "$dir" stopped
        printf '143\n' > "$dir/exit_code"
        date --iso-8601=seconds > "$dir/finished_at"
        exit 143
    }
    trap on_term TERM INT HUP

    exec 9>"$JOB_ROOT/gpu.lock"
    write_status "$dir" queued
    flock 9
    write_status "$dir" running

    set +e
    LLAMA_JOB_DIR="$dir" timeout --signal=TERM --kill-after=30s "${timeout_s}s" \
        "$@" > "$dir/stdout.log" 2> "$dir/stderr.log"
    local rc=$?
    set -e

    printf '%s\n' "$rc" > "$dir/exit_code"
    date --iso-8601=seconds > "$dir/finished_at"

    if grep -Eqi 'Memory Fault|illegal memory access|HSA_STATUS_ERROR|GPU reset required' \
        "$dir/stdout.log" "$dir/stderr.log" 2>/dev/null; then
        write_status "$dir" reset-required
    elif [ "$rc" -eq 0 ]; then
        write_status "$dir" done
    elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        write_status "$dir" timeout
    else
        write_status "$dir" failed
    fi
    exit "$rc"
}

start_job() {
    local label=${1:-}
    shift || true
    valid_name "$label" || fail "job label must contain only letters, numbers, dot, underscore, or dash"
    local timeout_s="$DEFAULT_TIMEOUT"
    if [ "${1:-}" = "--timeout" ]; then
        timeout_s=${2:-}
        [[ "$timeout_s" =~ ^[0-9]+$ ]] || fail "timeout must be an integer number of seconds"
        shift 2
    fi
    [ "${1:-}" = "--" ] || fail "usage: $0 start LABEL [--timeout SEC] -- COMMAND [ARGS...]"
    shift
    [ "$#" -gt 0 ] || fail "missing command"

    local id dir
    id="$(date +%Y%m%d-%H%M%S)-$label"
    dir="$(job_dir "$id")"
    mkdir -p "$dir/inputs"
    write_status "$dir" queued

    # Queue-safe snapshots for small ad-hoc /tmp dependencies. Model files and
    # installed binaries outside /tmp are referenced in place and never copied.
    local -a command=("$@")
    local i src dst size
    for i in "${!command[@]}"; do
        src=${command[$i]}
        if [[ "$src" == /tmp/* && -f "$src" ]]; then
            size=$(stat -c %s "$src")
            if [ "$size" -le 1048576 ]; then
                dst="$dir/inputs/${i}-$(basename "$src")"
                cp -p "$src" "$dst"
                command[$i]=$dst
            fi
        fi
    done

    printf '%q ' "${command[@]}" > "$dir/command.sh"
    printf '\n' >> "$dir/command.sh"

    setsid nohup "$SELF" _run "$id" "$timeout_s" -- "${command[@]}" \
        > "$dir/supervisor.log" 2>&1 < /dev/null &
    local launcher_pid=$!
    printf '%s\n' "$launcher_pid" > "$dir/pid"
    echo "$id"
}

case "${1:-}" in
    start)
        shift
        start_job "$@"
        ;;
    status)
        [ "$#" -eq 2 ] || fail "usage: $0 status JOB_ID"
        show_status "$2"
        ;;
    logs)
        [ "$#" -ge 2 ] || fail "usage: $0 logs JOB_ID [LINES]"
        dir="$(job_dir "$2")"
        [ -d "$dir" ] || fail "unknown job: $2"
        lines=${3:-80}
        echo "=== stdout ==="
        tail -n "$lines" "$dir/stdout.log" 2>/dev/null || true
        echo "=== stderr ==="
        tail -n "$lines" "$dir/stderr.log" 2>/dev/null || true
        ;;
    result)
        [ "$#" -eq 2 ] || fail "usage: $0 result JOB_ID"
        dir="$(job_dir "$2")"
        [ -d "$dir" ] || fail "unknown job: $2"
        show_status "$2"
        for f in summary.json result.json result.jsonl; do
            [ -f "$dir/$f" ] && { echo "=== $f ==="; cat "$dir/$f"; }
        done
        ;;
    stop)
        [ "$#" -eq 2 ] || fail "usage: $0 stop JOB_ID"
        dir="$(job_dir "$2")"
        [ -d "$dir" ] || fail "unknown job: $2"
        [ -f "$dir/pgid" ] || fail "job has no process group yet"
        pgid="$(<"$dir/pgid")"
        kill -TERM -- "-$pgid" 2>/dev/null || true
        echo "stop requested for $2"
        ;;
    list)
        for dir in "$JOB_ROOT"/*; do
            [ -d "$dir" ] || continue
            show_status "$(basename "$dir")" | head -1
        done
        ;;
    _run)
        shift
        run_job "$@"
        ;;
    *)
        cat <<EOF
Usage:
  $0 start LABEL [--timeout SEC] -- COMMAND [ARGS...]
  $0 status JOB_ID
  $0 logs JOB_ID [LINES]
  $0 result JOB_ID
  $0 stop JOB_ID
  $0 list
EOF
        ;;
esac
