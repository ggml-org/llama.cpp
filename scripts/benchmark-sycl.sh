#!/bin/bash
# SYCL XMX Flash Attention Benchmark Script
#
# This script benchmarks various models with different configurations to track
# performance over time. Results are saved to timestamped files for comparison.
#
# Usage:
#   ./scripts/benchmark-sycl.sh              # Run all benchmarks
#   ./scripts/benchmark-sycl.sh --quick      # Quick test with small models only
#   ./scripts/benchmark-sycl.sh --compare    # Compare latest vs previous results
#
# Environment:
#   - Requires Intel oneAPI (setvars.sh)
#   - Uses level_zero:1 for Intel Arc GPU

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_DIR}/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.txt"
RESULTS_JSON="${RESULTS_DIR}/benchmark_${TIMESTAMP}.json"

# Models to benchmark
MODEL_MISTRAL_Q4="/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf"
MODEL_MISTRAL_Q8="/Storage/GenAI/models/mistral-7b-v0.1.Q8_0.gguf"
MODEL_GPT_OSS_20B="/Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf"
MODEL_GPT2="/Storage/GenAI/models/gpt2.Q8_0.gguf"

# GPU selector for Intel Arc
export ONEAPI_DEVICE_SELECTOR=level_zero:1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
QUICK_MODE=false
COMPARE_MODE=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --compare) COMPARE_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick    Run quick benchmarks with small models only"
            echo "  --compare  Compare latest results with previous run"
            echo "  --help     Show this help message"
            exit 0
            ;;
    esac
done

# Initialize environment
init_environment() {
    echo -e "${BLUE}=== Initializing Intel oneAPI Environment ===${NC}"

    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
    else
        echo -e "${RED}Warning: Intel oneAPI setvars.sh not found${NC}"
    fi

    mkdir -p "$RESULTS_DIR"

    # Check llama-bench exists
    if [ ! -f "${PROJECT_DIR}/build/bin/llama-bench" ]; then
        echo -e "${RED}Error: llama-bench not found. Please build the project first.${NC}"
        exit 1
    fi

    echo "Results will be saved to: $RESULTS_FILE"
    echo ""
}

# Log to both console and file
log() {
    echo -e "$1" | tee -a "$RESULTS_FILE"
}

# Run a single benchmark
run_benchmark() {
    local model="$1"
    local model_name="$2"
    local prompt_tokens="$3"
    local gen_tokens="$4"
    local flash_attn="$5"
    local extra_args="${6:-}"

    local fa_str="FA_OFF"
    [ "$flash_attn" = "1" ] && fa_str="FA_ON"

    log "  [${model_name}] p=${prompt_tokens} n=${gen_tokens} ${fa_str} ${extra_args}"

    local cmd="${PROJECT_DIR}/build/bin/llama-bench \
        -m \"$model\" \
        -p $prompt_tokens \
        -n $gen_tokens \
        -ngl 99 \
        -fa $flash_attn \
        $extra_args \
        -o json 2>/dev/null"

    local result
    result=$(eval "$cmd" 2>&1) || true

    if [ -n "$result" ]; then
        # Extract key metrics from JSON output
        local pp_avg=$(echo "$result" | jq -r '.[0].avg_ts // "N/A"' 2>/dev/null || echo "N/A")
        local tg_avg=$(echo "$result" | jq -r '.[1].avg_ts // "N/A"' 2>/dev/null || echo "N/A")

        log "    Prompt: ${pp_avg} t/s | Generate: ${tg_avg} t/s"

        # Append to JSON results
        echo "$result" >> "$RESULTS_JSON"
    else
        log "    ${RED}FAILED${NC}"
    fi
}

# Benchmark a model with multiple configurations
benchmark_model() {
    local model="$1"
    local model_name="$2"
    local prompt_sizes="$3"
    local gen_tokens="${4:-128}"

    if [ ! -f "$model" ]; then
        log "  ${YELLOW}Skipping ${model_name}: model file not found${NC}"
        return
    fi

    log ""
    log "${GREEN}--- ${model_name} ---${NC}"

    # Test Flash Attention OFF vs ON
    for p in $prompt_sizes; do
        run_benchmark "$model" "$model_name" "$p" "$gen_tokens" "0"
        run_benchmark "$model" "$model_name" "$p" "$gen_tokens" "1"
    done
}

# Benchmark prompt processing (pp) speed
benchmark_pp() {
    local model="$1"
    local model_name="$2"

    if [ ! -f "$model" ]; then
        return
    fi

    log ""
    log "${GREEN}--- ${model_name} Prompt Processing ---${NC}"

    for p in 128 256 512 1024; do
        run_benchmark "$model" "$model_name" "$p" "1" "1"
    done
}

# Benchmark token generation (tg) speed
benchmark_tg() {
    local model="$1"
    local model_name="$2"

    if [ ! -f "$model" ]; then
        return
    fi

    log ""
    log "${GREEN}--- ${model_name} Token Generation ---${NC}"

    for n in 32 64 128 256; do
        run_benchmark "$model" "$model_name" "128" "$n" "1"
    done
}

# Benchmark batch sizes
benchmark_batch() {
    local model="$1"
    local model_name="$2"

    if [ ! -f "$model" ]; then
        return
    fi

    log ""
    log "${GREEN}--- ${model_name} Batch Size Scaling ---${NC}"

    for b in 128 256 512; do
        run_benchmark "$model" "$model_name" "512" "128" "1" "-b $b"
    done
}

# Quick benchmarks (small models only)
run_quick_benchmarks() {
    log "=== Quick Benchmark Mode ==="
    log "Timestamp: $(date)"
    log "GPU: Intel Arc (level_zero:1)"
    log ""

    # GPT-2 (tiny, fast)
    benchmark_model "$MODEL_GPT2" "GPT-2 Q8" "128 256" "64"

    # Mistral Q4 (medium, common)
    benchmark_model "$MODEL_MISTRAL_Q4" "Mistral-7B Q4" "256 512" "128"
}

# Full benchmarks (all models)
run_full_benchmarks() {
    log "=== Full Benchmark Suite ==="
    log "Timestamp: $(date)"
    log "GPU: Intel Arc (level_zero:1)"
    log "Build: $(cd "$PROJECT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    log ""

    # Initialize JSON output
    echo "[" > "$RESULTS_JSON"

    log "${BLUE}=== 1. Flash Attention Comparison ===${NC}"
    benchmark_model "$MODEL_MISTRAL_Q4" "Mistral-7B Q4" "128 256 512 1024" "128"
    benchmark_model "$MODEL_MISTRAL_Q8" "Mistral-7B Q8" "256 512" "128"
    benchmark_model "$MODEL_GPT_OSS_20B" "GPT-OSS-20B Q8" "256 512" "128"

    log ""
    log "${BLUE}=== 2. Prompt Processing Scaling ===${NC}"
    benchmark_pp "$MODEL_MISTRAL_Q4" "Mistral-7B Q4"
    benchmark_pp "$MODEL_GPT_OSS_20B" "GPT-OSS-20B Q8"

    log ""
    log "${BLUE}=== 3. Token Generation Scaling ===${NC}"
    benchmark_tg "$MODEL_MISTRAL_Q4" "Mistral-7B Q4"
    benchmark_tg "$MODEL_GPT_OSS_20B" "GPT-OSS-20B Q8"

    log ""
    log "${BLUE}=== 4. Batch Size Scaling ===${NC}"
    benchmark_batch "$MODEL_MISTRAL_Q4" "Mistral-7B Q4"

    log ""
    log "${BLUE}=== 5. Unified KV Mode Correctness (FA Fix Validation) ===${NC}"
    test_unified_kv_correctness "$MODEL_MISTRAL_Q4" "Mistral-7B Q4"
    test_unified_kv_correctness "$MODEL_GPT_OSS_20B" "GPT-OSS-20B Q8"

    # Close JSON array
    echo "]" >> "$RESULTS_JSON"
}

# Compare results
compare_results() {
    echo -e "${BLUE}=== Comparing Benchmark Results ===${NC}"

    # Find the two most recent benchmark files
    local latest=$(ls -t "$RESULTS_DIR"/benchmark_*.txt 2>/dev/null | head -1)
    local previous=$(ls -t "$RESULTS_DIR"/benchmark_*.txt 2>/dev/null | head -2 | tail -1)

    if [ -z "$latest" ]; then
        echo -e "${RED}No benchmark results found${NC}"
        exit 1
    fi

    if [ "$latest" = "$previous" ]; then
        echo -e "${YELLOW}Only one benchmark result found. Run benchmarks again to compare.${NC}"
        echo "Latest: $latest"
        cat "$latest"
        exit 0
    fi

    echo "Latest:   $latest"
    echo "Previous: $previous"
    echo ""
    echo "=== Diff ==="
    diff -u "$previous" "$latest" || true
}

# Server throughput benchmark (tests unified KV mode)
benchmark_server_throughput() {
    local model="$1"
    local model_name="$2"
    local port=8099

    if [ ! -f "$model" ]; then
        log "  ${YELLOW}Skipping server benchmark: model not found${NC}"
        return
    fi

    log ""
    log "${GREEN}--- ${model_name} Server Throughput (Unified KV) ---${NC}"

    # Kill any existing server
    pkill -9 llama-server 2>/dev/null || true
    sleep 1

    # Start server with unified KV
    "${PROJECT_DIR}/build/bin/llama-server" \
        -m "$model" \
        --flash-attn on -kvu --parallel 4 -c 4096 --port $port -ngl 99 \
        2>&1 > /tmp/bench_server.log &

    local server_pid=$!
    sleep 12

    # Check server health
    if ! curl -s "http://localhost:$port/health" > /dev/null; then
        log "  ${RED}Server failed to start${NC}"
        kill $server_pid 2>/dev/null || true
        return
    fi

    # Run 10 concurrent requests and measure throughput
    local start_time=$(date +%s.%N)
    for i in {1..10}; do
        curl -s "http://localhost:$port/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"test","prompt":"Hello world","max_tokens":32,"temperature":0}' \
            > /tmp/bench_$i.json &
    done
    wait
    local end_time=$(date +%s.%N)

    # Calculate throughput
    local total_tokens=0
    for i in {1..10}; do
        local tokens=$(cat /tmp/bench_$i.json 2>/dev/null | jq -r '.usage.total_tokens // 0')
        total_tokens=$((total_tokens + tokens))
    done

    local elapsed=$(echo "$end_time - $start_time" | bc)
    local throughput=$(echo "scale=2; $total_tokens / $elapsed" | bc)

    log "  10 concurrent requests: ${elapsed}s, ${total_tokens} tokens, ${throughput} t/s"

    # Cleanup
    kill $server_pid 2>/dev/null || true
}

# Test unified KV mode correctness (tests our flash attention fix)
test_unified_kv_correctness() {
    local model="$1"
    local model_name="$2"
    local port=8099

    if [ ! -f "$model" ]; then
        log "  ${YELLOW}Skipping unified KV test: model not found${NC}"
        return
    fi

    log ""
    log "${GREEN}--- ${model_name} Unified KV Correctness Test ---${NC}"

    # Kill any existing server
    pkill -9 llama-server 2>/dev/null || true
    sleep 1

    # Start server with unified KV + flash attention
    "${PROJECT_DIR}/build/bin/llama-server" \
        -m "$model" \
        --flash-attn on -kvu --parallel 4 -c 4096 --port $port -ngl 99 \
        2>&1 > /tmp/bench_unified_kv.log &

    local server_pid=$!
    sleep 15

    # Check server health
    if ! curl -s "http://localhost:$port/health" > /dev/null; then
        log "  ${RED}Server failed to start${NC}"
        kill $server_pid 2>/dev/null || true
        return
    fi

    # Test 1: 4 identical prompts should produce identical outputs (temp=0)
    log "  Test 1: Determinism (4 identical requests, temp=0)"
    for i in 1 2 3 4; do
        curl -s "http://localhost:$port/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"test","prompt":"1+1=","max_tokens":5,"temperature":0}' \
            > /tmp/bench_det_$i.json &
    done
    wait

    local r1=$(cat /tmp/bench_det_1.json 2>/dev/null | jq -r '.choices[0].text // "ERROR"')
    local r2=$(cat /tmp/bench_det_2.json 2>/dev/null | jq -r '.choices[0].text // "ERROR"')
    local r3=$(cat /tmp/bench_det_3.json 2>/dev/null | jq -r '.choices[0].text // "ERROR"')
    local r4=$(cat /tmp/bench_det_4.json 2>/dev/null | jq -r '.choices[0].text // "ERROR"')

    if [ "$r1" = "$r2" ] && [ "$r2" = "$r3" ] && [ "$r3" = "$r4" ]; then
        log "    ${GREEN}PASS${NC}: All 4 responses identical: '$r1'"
    else
        log "    ${RED}FAIL${NC}: Responses differ: '$r1' vs '$r2' vs '$r3' vs '$r4'"
    fi

    # Test 2: Different prompts should remain isolated
    log "  Test 2: Isolation (4 different prompts)"
    curl -s "http://localhost:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","prompt":"The capital of France is","max_tokens":5,"temperature":0}' \
        > /tmp/bench_iso_1.json &
    curl -s "http://localhost:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","prompt":"The largest planet is","max_tokens":5,"temperature":0}' \
        > /tmp/bench_iso_2.json &
    curl -s "http://localhost:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","prompt":"2+2=","max_tokens":3,"temperature":0}' \
        > /tmp/bench_iso_3.json &
    curl -s "http://localhost:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","prompt":"Hello world","max_tokens":5,"temperature":0}' \
        > /tmp/bench_iso_4.json &
    wait

    local france=$(cat /tmp/bench_iso_1.json 2>/dev/null | jq -r '.choices[0].text // "ERROR"')
    local planet=$(cat /tmp/bench_iso_2.json 2>/dev/null | jq -r '.choices[0].text // "ERROR"')
    local math=$(cat /tmp/bench_iso_3.json 2>/dev/null | jq -r '.choices[0].text // "ERROR"')
    local hello=$(cat /tmp/bench_iso_4.json 2>/dev/null | jq -r '.choices[0].text // "ERROR"')

    log "    France: '$france'"
    log "    Planet: '$planet'"
    log "    Math: '$math'"
    log "    Hello: '$hello'"

    # Check for obvious cross-contamination
    if echo "$planet" | grep -qi "paris\|france"; then
        log "    ${RED}FAIL${NC}: Cross-contamination detected"
    else
        log "    ${GREEN}PASS${NC}: Responses appear properly isolated"
    fi

    # Cleanup
    kill $server_pid 2>/dev/null || true
}

# Benchmark llama-cli with flash attention (direct inference test)
benchmark_cli() {
    local model="$1"
    local model_name="$2"
    local prompt_tokens="$3"
    local gen_tokens="$4"
    local flash_attn="$5"

    if [ ! -f "$model" ]; then
        return
    fi

    local fa_str="FA_OFF"
    local fa_flag=""
    [ "$flash_attn" = "1" ] && fa_str="FA_ON" && fa_flag="--flash-attn"

    log "  [${model_name}] CLI p=${prompt_tokens} n=${gen_tokens} ${fa_str}"

    # Generate a prompt of approximately the right length
    local prompt="The quick brown fox jumps over the lazy dog. "
    local full_prompt=""
    for i in $(seq 1 $((prompt_tokens / 10))); do
        full_prompt="${full_prompt}${prompt}"
    done

    local start_time=$(date +%s.%N)
    "${PROJECT_DIR}/build/bin/llama-cli" \
        -m "$model" \
        -p "$full_prompt" \
        -n "$gen_tokens" \
        -ngl 99 \
        $fa_flag \
        --no-display-prompt \
        2>&1 > /tmp/bench_cli.txt || true
    local end_time=$(date +%s.%N)

    local elapsed=$(echo "$end_time - $start_time" | bc)
    log "    Total time: ${elapsed}s"
}

# Main
main() {
    if [ "$COMPARE_MODE" = true ]; then
        compare_results
        exit 0
    fi

    init_environment

    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     SYCL XMX Flash Attention Benchmark Suite                  ║${NC}"
    echo -e "${BLUE}║     Intel Arc GPU - llama.cpp                                 ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    if [ "$QUICK_MODE" = true ]; then
        run_quick_benchmarks
    else
        run_full_benchmarks

        # Server throughput test
        log ""
        log "${BLUE}=== 6. Server Throughput (Unified KV Mode) ===${NC}"
        benchmark_server_throughput "$MODEL_MISTRAL_Q4" "Mistral-7B Q4"
        benchmark_server_throughput "$MODEL_GPT_OSS_20B" "GPT-OSS-20B Q8"
    fi

    log ""
    log "=== Benchmark Complete ==="
    log "Results saved to: $RESULTS_FILE"

    echo ""
    echo -e "${GREEN}Done! Results saved to:${NC}"
    echo "  Text: $RESULTS_FILE"
    echo "  JSON: $RESULTS_JSON"
    echo ""
    echo "To compare with previous runs: $0 --compare"
}

main "$@"
