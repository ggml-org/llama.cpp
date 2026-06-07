#!/usr/bin/env bash
#
# Blue-noise dithered KV cache benchmark runner
# Tests Q2_K, Q3_K, Q4_0, Q4_1 with and without blue-noise dithering
#
set -euo pipefail

MODEL="${MODEL:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
NGL="${NGL:-999}"
CTX="${CTX:-4096}"
SEED="${SEED:-42}"
N_OUTPUT="${N_OUTPUT:-256}"
PROMPT_DIR="tests/prompts"
RESULTS_DIR="reports/bench_results"

mkdir -p "$RESULTS_DIR"

CLI="$BUILD_DIR/bin/llama-cli"
PERP="$BUILD_DIR/bin/llama-perplexity"

if [ ! -f "$CLI" ]; then
    echo "ERROR: $CLI not found. Build first."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "WARNING: $MODEL not found. Set MODEL env var or download the model."
fi

# KV cache type combinations to test
declare -a TESTS=(
    # Baseline
    "f16:f16"
    # Q4_0 variants
    "q4_0:q4_0"
    "q4_0_blue:q4_0"
    "q4_0:q4_0_blue"
    "q4_0_blue:q4_0_blue"
    # Q4_1 variants
    "q4_1:q4_1"
    "q4_1_blue:q4_1"
    "q4_1:q4_1_blue"
    "q4_1_blue:q4_1_blue"
    # Q3_K variants
    "q3_K:q3_K"
    "q3_K_blue:q3_K"
    "q3_K:q3_K_blue"
    "q3_K_blue:q3_K_blue"
    # Q2_K variants
    "q2_K:q2_K"
    "q2_K_blue:q2_K"
    "q2_K:q2_K_blue"
    "q2_K_blue:q2_K_blue"
    # Cross-bit mixed
    "q2_K_blue:q3_K_blue"
    "q3_K_blue:q2_K_blue"
)

# Prompt files
PROMPTS=(
    "needle_4k.txt"
    "needle_8k.txt"
    "summarise_long.txt"
    "math_reasoning.txt"
    "repetition_stress.txt"
)

run_test() {
    local k_type="$1"
    local v_type="$2"
    local prompt="$3"
    local label="${k_type}_${v_type}_$(basename $prompt .txt)"
    local result_file="$RESULTS_DIR/${label}.txt"

    echo "=== TEST: K=$k_type V=$v_type prompt=$prompt ==="

    if [ ! -f "$MODEL" ]; then
        echo "SKIP (no model)" > "$result_file"
        echo "SKIP: model not found"
        return
    fi

    # Run with timeout (5 minutes per test)
    timeout 300 "$CLI" \
        -m "$MODEL" \
        -ngl "$NGL" \
        -c "$CTX" \
        --cache-type-k "$k_type" \
        --cache-type-v "$v_type" \
        --seed "$SEED" \
        --temp 0 \
        -f "$PROMPT_DIR/$prompt" \
        -n "$N_OUTPUT" \
        2>"${result_file}.stderr" \
        > "$result_file" || echo "FAILED (exit code $?)" >> "$result_file"

    echo "  -> Saved to $result_file"
    echo ""
}

run_perplexity() {
    local k_type="$1"
    local v_type="$2"
    local label="${k_type}_${v_type}"
    local result_file="$RESULTS_DIR/perplexity_${label}.txt"

    if [ ! -f "$MODEL" ] || [ ! -f "tests/data/wiki_small.txt" ]; then
        echo "SKIP: model or wiki_small.txt not found"
        return
    fi

    echo "=== PERPLEXITY: K=$k_type V=$v_type ==="

    timeout 600 "$PERP" \
        -m "$MODEL" \
        -f tests/data/wiki_small.txt \
        -c "$CTX" \
        --cache-type-k "$k_type" \
        --cache-type-v "$v_type" \
        2>"${result_file}.stderr" \
        > "$result_file" || echo "FAILED" >> "$result_file"

    echo "  -> Saved to $result_file"
    echo ""
}

echo "============================================"
echo "  Blue-Noise KV Cache Benchmark Suite"
echo "  Model: $MODEL"
echo "  Build: $BUILD_DIR"
echo "  Seed:  $SEED"
echo "============================================"
echo ""

# Run inference tests
for test_combo in "${TESTS[@]}"; do
    IFS=':' read -r k v <<< "$test_combo"
    for prompt in "${PROMPTS[@]}"; do
        if [ ! -f "$PROMPT_DIR/$prompt" ]; then
            echo "WARNING: prompt file $PROMPT_DIR/$prompt not found, skipping"
            continue
        fi
        run_test "$k" "$v" "$prompt"
    done
done

# Run perplexity tests (subset - fewer combos to save time)
echo "=== Perplexity tests (subset) ==="
PERP_TESTS=(
    "f16:f16"
    "q4_0:q4_0"
    "q4_0_blue:q4_0"
    "q4_0_blue:q4_0_blue"
    "q4_1_blue:q4_1"
    "q3_K:q3_K"
    "q3_K_blue:q3_K"
    "q3_K_blue:q3_K_blue"
    "q2_K:q2_K"
    "q2_K_blue:q2_K"
    "q2_K_blue:q2_K_blue"
)
for test_combo in "${PERP_TESTS[@]}"; do
    IFS=':' read -r k v <<< "$test_combo"
    run_perplexity "$k" "$v"
done

echo "============================================"
echo "  All tests completed!"
echo "  Results in: $RESULTS_DIR"
echo "============================================"
