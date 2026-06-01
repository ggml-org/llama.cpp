#!/usr/bin/env bash
# run_experiment.sh — Build, benchmark, and validate a CUDA kernel experiment.
# Outputs structured JSON to stdout. All logging goes to stderr.
#
# Usage: run_experiment.sh <track-name> [--quick]
#   --quick: skip PPL validation (speed-only iteration)

set -euo pipefail

TRACK="${1:?Usage: run_experiment.sh <track-name> [--quick]}"
QUICK=false
[[ "${2:-}" == "--quick" ]] && QUICK=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRACK_DIR="$SCRIPT_DIR/$TRACK"

if [[ ! -f "$TRACK_DIR/baseline.json" ]]; then
    echo '{"status": "error", "error": "No baseline.json found for track: '"$TRACK"'"}'
    exit 1
fi

# Load track config from baseline.json
BENCH_ARGS=$(jq -r '.bench_args // ""' "$TRACK_DIR/baseline.json")
MODEL=$(jq -r '.model' "$TRACK_DIR/baseline.json")
NO_CONVERT=$(jq -r '.no_convert // false' "$TRACK_DIR/baseline.json")

# For tracks that benchmark the TQ4_1S runtime kernel, disable load-time conversion
BENCH_ENV=""
if [[ "$NO_CONVERT" == "true" ]]; then
    BENCH_ENV="GGML_TQ_NO_CONVERT=1"
fi
PPL_BASELINE=$(jq -r '.ppl // 0' "$TRACK_DIR/baseline.json")
PPL_THRESHOLD=$(jq -r '.ppl_threshold // 0.1' "$TRACK_DIR/baseline.json")
PPL_FILE=$(jq -r '.ppl_file // ""' "$TRACK_DIR/baseline.json")

# ---- Phase 1: Build ----
echo ">>> Building..." >&2
BUILD_START=$(date +%s)

BUILD_OUTPUT=$("$REPO_DIR/scripts/autoresearch/build.sh" 2>&1) || {
    BUILD_END=$(date +%s)
    # Extract last 20 lines of error for the agent
    ERROR=$(echo "$BUILD_OUTPUT" | grep "error:" | head -10)
    jq -n \
        --arg status "build_failed" \
        --arg error "$ERROR" \
        --argjson build_time_s $((BUILD_END - BUILD_START)) \
        '{status: $status, error: $error, build_time_s: $build_time_s}'
    exit 0
}
BUILD_END=$(date +%s)
echo ">>> Build OK ($(( BUILD_END - BUILD_START ))s)" >&2

# ---- Phase 2: Benchmark ----
echo ">>> Benchmarking..." >&2
BENCH_START=$(date +%s)

BENCH_OUTPUT=$(timeout 120 env $BENCH_ENV "$REPO_DIR/build-cuda/bin/llama-bench" \
    -m "$MODEL" -ngl 99 $BENCH_ARGS -r 3 2>&1) || {
    BENCH_END=$(date +%s)
    ERROR=$(echo "$BENCH_OUTPUT" | grep -iE "error|crash|abort|signal" | head -5)
    jq -n \
        --arg status "runtime_crash" \
        --arg error "${ERROR:-unknown crash}" \
        --argjson build_time_s $((BUILD_END - BUILD_START)) \
        --argjson bench_time_s $(($(date +%s) - BENCH_START)) \
        '{status: $status, error: $error, build_time_s: $build_time_s, bench_time_s: $bench_time_s}'
    exit 0
}
BENCH_END=$(date +%s)

# Parse t/s from llama-bench output
# Format: "| model | size | params | backend | ngl | test | t/s ± err |"
# The t/s value is before the "±" in the last data column
parse_ts() {
    grep -E "^\|" | grep -v "model" | grep -v "\-\-\-" | \
        grep -oP '\d+\.\d+\s*±' | grep -oP '[\d.]+' | head -1
}

TG128=$(echo "$BENCH_OUTPUT" | parse_ts)
# For tracks with both pp and tg, there will be two data rows
BENCH_LINES=$(echo "$BENCH_OUTPUT" | grep -E "^\|" | grep -v "model" | grep -v "\-\-\-" | wc -l)
if (( BENCH_LINES > 1 )); then
    PP512=$(echo "$BENCH_OUTPUT" | grep -E "^\|" | grep -v "model" | grep -v "\-\-\-" | head -1 | grep -oP '\d+\.\d+\s*±' | grep -oP '[\d.]+')
    TG128=$(echo "$BENCH_OUTPUT" | grep -E "^\|" | grep -v "model" | grep -v "\-\-\-" | tail -1 | grep -oP '\d+\.\d+\s*±' | grep -oP '[\d.]+')
else
    PP512=""
fi

echo ">>> Benchmark: tg=${TG128:-?} pp=${PP512:-n/a}" >&2

# ---- Phase 3: Correctness (unless --quick) ----
PPL=""
if [[ "$QUICK" == "false" && -n "$PPL_FILE" && -n "$PPL_BASELINE" && "$PPL_BASELINE" != "0" ]]; then
    echo ">>> PPL check..." >&2
    PPL_OUTPUT=$(timeout 180 env $BENCH_ENV "$REPO_DIR/build-cuda/bin/llama-perplexity" \
        -m "$MODEL" -f "$PPL_FILE" -ngl 99 --chunks 10 2>&1) || true
    PPL=$(echo "$PPL_OUTPUT" | grep "Final estimate" | grep -oP 'PPL = \K[0-9.]+')

    if [[ -n "$PPL" ]]; then
        PPL_DELTA=$(echo "$PPL - $PPL_BASELINE" | bc -l 2>/dev/null || echo "999")
        if (( $(echo "$PPL_DELTA > $PPL_THRESHOLD" | bc -l 2>/dev/null || echo 1) )); then
            echo ">>> PPL regression: $PPL (baseline: $PPL_BASELINE, delta: $PPL_DELTA)" >&2
            jq -n \
                --arg status "ppl_regression" \
                --argjson ppl "${PPL}" \
                --argjson ppl_baseline "${PPL_BASELINE}" \
                --argjson build_time_s $((BUILD_END - BUILD_START)) \
                --argjson bench_time_s $((BENCH_END - BENCH_START)) \
                --arg tg128 "${TG128:-0}" \
                '{status: $status, tg128: ($tg128|tonumber), ppl: $ppl, ppl_baseline: $ppl_baseline, build_time_s: $build_time_s, bench_time_s: $bench_time_s}'
            exit 0
        fi
        echo ">>> PPL OK: $PPL (baseline: $PPL_BASELINE)" >&2
    fi
fi

# ---- Phase 4: Compute delta and output ----
BASELINE_TG=$(jq -r '.tg128 // 0' "$TRACK_DIR/baseline.json")
if [[ -n "$TG128" && "$BASELINE_TG" != "0" ]]; then
    DELTA_PCT=$(echo "scale=1; ($TG128 - $BASELINE_TG) / $BASELINE_TG * 100" | bc -l 2>/dev/null || echo "0")
else
    DELTA_PCT="0"
fi

# GPU temperature
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")

jq -n \
    --arg status "success" \
    --arg tg128 "${TG128:-0}" \
    --arg pp512 "${PP512:-}" \
    --arg ppl "${PPL:-}" \
    --argjson baseline_tg128 "${BASELINE_TG}" \
    --arg delta_pct "${DELTA_PCT}%" \
    --argjson build_time_s $((BUILD_END - BUILD_START)) \
    --argjson bench_time_s $((BENCH_END - BENCH_START)) \
    --arg gpu_temp_c "${GPU_TEMP}" \
    '{status: $status, tg128: ($tg128|tonumber), baseline_tg128: $baseline_tg128, delta_pct: $delta_pct, build_time_s: $build_time_s, bench_time_s: $bench_time_s, gpu_temp_c: $gpu_temp_c} + (if $pp512 != "" then {pp512: ($pp512|tonumber)} else {} end) + (if $ppl != "" then {ppl: ($ppl|tonumber)} else {} end)'
