#!/bin/bash
#
# Runtime determinism test for Flash Attention on SYCL backend
#
# This script tests that FA produces deterministic results across multiple runs.
# It's designed to catch regressions in the barrier fix for tile_S race condition.
#
# Usage:
#   ./tests/test-fattn-determinism.sh [--quick]
#
# Options:
#   --quick   Run fewer iterations (5 instead of 10)
#
# Requirements:
#   - Built llama-cli binary
#   - Model files in /Storage/GenAI/models/
#   - Intel GPU with SYCL support
#

set -e

# Configuration
LLAMA_CLI="./build/bin/llama-cli"
MISTRAL_MODEL="/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf"
GPTOSS_MODEL="/Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf"

# Parse arguments
ITERATIONS=10
if [[ "$1" == "--quick" ]]; then
    ITERATIONS=5
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Source oneAPI environment
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
fi

echo "=============================================="
echo "Flash Attention Determinism Test"
echo "=============================================="
echo "Iterations per test: $ITERATIONS"
echo ""

# Check if binaries exist
if [ ! -f "$LLAMA_CLI" ]; then
    echo -e "${RED}ERROR: $LLAMA_CLI not found. Build first with: cmake --build build${NC}"
    exit 1
fi

# Test function
run_determinism_test() {
    local model="$1"
    local model_name="$2"
    local prompt="$3"
    local expected_pattern="$4"
    local iterations="$5"

    echo "Testing: $model_name"
    echo "  Prompt: '$prompt'"
    echo "  Expected pattern: '$expected_pattern'"

    local first_output=""
    local success=0
    local fail=0

    for i in $(seq 1 $iterations); do
        output=$(GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
            timeout 120 "$LLAMA_CLI" \
            -m "$model" \
            -ngl 99 --flash-attn on --no-conversation \
            -p "$prompt" -n 15 --seed 42 --temp 0 2>&1 \
            | grep -oE "$expected_pattern" | head -1 || echo "NO_MATCH")

        if [ -z "$first_output" ]; then
            first_output="$output"
        fi

        if [ "$output" == "$first_output" ] && [ "$output" != "NO_MATCH" ]; then
            echo -n "."
            ((success++))
        else
            echo -n "X"
            ((fail++))
            if [ "$output" != "$first_output" ]; then
                echo ""
                echo -e "${RED}  Run $i differs: '$output' vs first: '$first_output'${NC}"
            fi
        fi
    done

    echo ""

    if [ $fail -eq 0 ] && [ "$first_output" != "NO_MATCH" ]; then
        echo -e "${GREEN}  PASSED: $success/$iterations identical, output: '$first_output'${NC}"
        return 0
    else
        echo -e "${RED}  FAILED: $success/$iterations identical, $fail different${NC}"
        return 1
    fi
}

# Track overall results
TESTS_PASSED=0
TESTS_FAILED=0

echo "----------------------------------------------"
echo "Test 1: Mistral 7B (D=128) - Number continuation"
echo "----------------------------------------------"
if [ -f "$MISTRAL_MODEL" ]; then
    if run_determinism_test "$MISTRAL_MODEL" "Mistral 7B" "1, 2, 3, 4, 5," "6, 7, 8, 9, 10" $ITERATIONS; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
else
    echo -e "${YELLOW}SKIPPED: Model not found${NC}"
fi
echo ""

echo "----------------------------------------------"
echo "Test 2: GPT-OSS 20B (D=64) - Number counting"
echo "----------------------------------------------"
if [ -f "$GPTOSS_MODEL" ]; then
    if run_determinism_test "$GPTOSS_MODEL" "GPT-OSS 20B" "Count from 1 to 5:" "1,2,3,4,5" $ITERATIONS; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
else
    echo -e "${YELLOW}SKIPPED: Model not found${NC}"
fi
echo ""

echo "----------------------------------------------"
echo "Test 3: GPT-OSS 20B (D=64) - Hello world"
echo "----------------------------------------------"
if [ -f "$GPTOSS_MODEL" ]; then
    if run_determinism_test "$GPTOSS_MODEL" "GPT-OSS 20B" "Hello world" "Hello world.*test" $ITERATIONS; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
else
    echo -e "${YELLOW}SKIPPED: Model not found${NC}"
fi
echo ""

echo "----------------------------------------------"
echo "Test 4: FA ON vs FA OFF parity"
echo "----------------------------------------------"
if [ -f "$GPTOSS_MODEL" ]; then
    echo "Testing FA ON vs FA OFF produce same output..."

    fa_off=$(GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
        timeout 120 "$LLAMA_CLI" \
        -m "$GPTOSS_MODEL" \
        -ngl 99 --flash-attn off --no-conversation \
        -p "Count from 1 to 5:" -n 15 --seed 42 --temp 0 2>&1 \
        | grep -oE "1,2,3,4,5" | head -1 || echo "NO_MATCH")

    fa_on=$(GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
        timeout 120 "$LLAMA_CLI" \
        -m "$GPTOSS_MODEL" \
        -ngl 99 --flash-attn on --no-conversation \
        -p "Count from 1 to 5:" -n 15 --seed 42 --temp 0 2>&1 \
        | grep -oE "1,2,3,4,5" | head -1 || echo "NO_MATCH")

    echo "  FA OFF: '$fa_off'"
    echo "  FA ON:  '$fa_on'"

    if [ "$fa_off" == "$fa_on" ] && [ "$fa_off" != "NO_MATCH" ]; then
        echo -e "${GREEN}  PASSED: FA ON matches FA OFF${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}  FAILED: FA ON differs from FA OFF${NC}"
        ((TESTS_FAILED++))
    fi
else
    echo -e "${YELLOW}SKIPPED: Model not found${NC}"
fi
echo ""

# Summary
echo "=============================================="
echo "Summary"
echo "=============================================="
TOTAL=$((TESTS_PASSED + TESTS_FAILED))
echo "Passed: $TESTS_PASSED / $TOTAL"
echo "Failed: $TESTS_FAILED / $TOTAL"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
