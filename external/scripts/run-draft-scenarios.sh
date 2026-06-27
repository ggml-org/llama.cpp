#!/usr/bin/env bash
# Run the deterministic-draft scenarios end to end.
#
# For each drafter type this starts llama-server with the reference plugin,
# issues the completion request, and prints the output plus the
# "draft acceptance" / "statistics draft-deterministic" lines.
#
# This script keeps its own copy of the scenarios and request, mirroring the
# examples in README.md. Keep the two in sync by hand.
#
# Usage:
#   external/scripts/run-draft-scenarios.sh [name ...]
#
# Examples:
#   # all six scenarios (3 default-mode + 3 intercept-all), models at the default root
#   external/scripts/run-draft-scenarios.sh
#
#   # only the default-mode (no --det-draft-intercept-all) runs
#   external/scripts/run-draft-scenarios.sh default-mtp default-dspark default-eagle3
#
#   # a single default-mode run
#   external/scripts/run-draft-scenarios.sh default-eagle3
#
#   # models elsewhere
#   DET_MODELS=/mnt/models external/scripts/run-draft-scenarios.sh
#
#   # only the intercept-all runs
#   external/scripts/run-draft-scenarios.sh ia-mtp ia-dspark ia-eagle3
#
#   # single default dspark run, keep logs under /tmp/mylogs, ports from 19000
#   DET_WORK=/tmp/mylogs DET_PORT_BASE=19000 external/scripts/run-draft-scenarios.sh default-dspark
#
# Model dir layout, rooted at $DET_MODELS:
#   MTP/Qwen3.5-4B.Q4_K_M.gguf
#   dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf       (target)
#   dspark/Qwen3-4B-DSpark-Q8_0.gguf             (draft)
#   eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf           (target)
#   eagle3/Qwen3-8B-speculator.eagle3-F16.gguf   (draft)
#
# Without arguments all scenarios run. Names:
#   default-mtp default-dspark default-eagle3 ia-mtp ia-dspark ia-eagle3
#
# Requires: a built llama-server and plugin .so (see README.md "Build").

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODELS="${DET_MODELS:-/home/samueldoyle/AI_LOCAL/Models}"
SERVER="build/bin/llama-server"
PLUGIN="external/plugins/lib/libdeterministic_draft_spec_plugin.so"
PROMPT='{"prompt":"# Write a short story\nOnce upon a time","max_tokens":256}'
WORK="${DET_WORK:-/tmp/det-draft-scenarios}"

PORT_BASE="${DET_PORT_BASE:-18080}"
TIMEOUT_S="${DET_TIMEOUT:-600}"

mkdir -p "$WORK"

if [ ! -x "$SERVER" ]; then
    echo "error: $SERVER not built (see README.md Build)" >&2
    exit 1
fi
if [ ! -f "$PLUGIN" ]; then
    echo "error: $PLUGIN not built - run: cmake --build build --target deterministic_draft_spec_plugin" >&2
    exit 1
fi

FMT=(--reasoning off --temp 0 --top-k 1 --top-p 1.0 -c 8192 -ngl 99 -fa on --jinja)

# scenario: name|args
scenarios=(
    "default-mtp|-m $MODELS/MTP/Qwen3.5-4B.Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 16 --det-draft-model $PLUGIN"
    "default-dspark|-m $MODELS/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf -md $MODELS/dspark/Qwen3-4B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 16 --det-draft-model $PLUGIN"
    "default-eagle3|-m $MODELS/eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf -md $MODELS/eagle3/Qwen3-8B-speculator.eagle3-F16.gguf --spec-type draft-eagle3 --spec-draft-n-max 16 --det-draft-model $PLUGIN"
    "ia-mtp|-m $MODELS/MTP/Qwen3.5-4B.Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 16 --det-draft-model $PLUGIN --det-draft-intercept-all"
    "ia-dspark|-m $MODELS/dspark/Qwen3-4B-DSpark-Model-Q8_0.gguf -md $MODELS/dspark/Qwen3-4B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 16 --det-draft-model $PLUGIN --det-draft-intercept-all"
    "ia-eagle3|-m $MODELS/eagle3/Qwen3-8B-eagle3-Q4_K_M.gguf -md $MODELS/eagle3/Qwen3-8B-speculator.eagle3-F16.gguf --spec-type draft-eagle3 --spec-draft-n-max 16 --det-draft-model $PLUGIN --det-draft-intercept-all"
)

if [ "$#" -gt 0 ]; then
    wanted=" $* "
else
    wanted=""
fi

run_one() {
    local name="$1" args="$2" port="$3"
    local log="$WORK/$name.log" out="$WORK/$name.out"
    local pid

    printf '\n===== %s =====\n' "$name"
    # shellcheck disable=SC2086
    "$SERVER" $args "${FMT[@]}" --port "$port" >"$log" 2>&1 &
    pid=$!

    local ready=0
    for _ in $(seq 1 "$TIMEOUT_S"); do
        if grep -q "llama_server: listening on" "$log" 2>/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "server exited during startup:"
            tail -20 "$log"
            return 1
        fi
        sleep 1
    done

    if [ "$ready" -ne 1 ]; then
        echo "timed out waiting for server"
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        return 1
    fi

    curl -s "http://127.0.0.1:$port/v1/completions" \
        -H "Content-Type: application/json" -d "$PROMPT" >"$out"

    sleep 1
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null

    if command -v python3 >/dev/null 2>&1; then
        python3 - "$out" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d["choices"][0]["text"])
except Exception as e:
    print("parse failed:", e)
    print(open(sys.argv[1]).read()[:500])
PY
    else
        cat "$out"
    fi

    grep -E "draft acceptance|statistics draft-deterministic|intercept-all is enabled|requires a compatible|exceeds the trained block size" "$log" | tail -8
    echo "(log: $log)"
}

port="$PORT_BASE"
status=0
for entry in "${scenarios[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"
    if [ -n "$wanted" ] && [[ "$wanted" != *" $name "* ]]; then
        continue
    fi
    run_one "$name" "$args" "$port" || status=1
    port=$((port + 1))
done

exit "$status"
