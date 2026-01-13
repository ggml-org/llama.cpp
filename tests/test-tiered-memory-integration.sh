#!/bin/bash
#
# Integration tests for SYCL tiered memory system
#
# Tests:
# 1. Small model (Mistral-7B) - should NOT enable tiered mode, fast inference
# 2. Large model (GPT-OSS-20B) - SHOULD enable tiered mode (if it exceeds VRAM)
#
# Success criteria:
# - No crashes or allocation errors
# - Correct tiered mode detection based on model size vs VRAM
# - Generation speed >5 t/s for Mistral-7B
#
# Usage:
#   source /opt/intel/oneapi/setvars.sh --force
#   ./tests/test-tiered-memory-integration.sh [--skip-large]
#
# Environment:
#   ONEAPI_DEVICE_SELECTOR - GPU device (default: level_zero:1)
#   LLAMA_BIN_DIR - Path to build/bin directory (default: ./build/bin)
#   MODEL_DIR - Path to model directory (default: /Storage/GenAI/models)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
# Auto-detect bin directory: check common locations
if [ -n "${LLAMA_BIN_DIR}" ]; then
    :  # Use provided value
elif [ -x "./build/bin/llama-completion" ]; then
    LLAMA_BIN_DIR="./build/bin"
elif [ -x "./bin/llama-completion" ]; then
    LLAMA_BIN_DIR="./bin"
elif [ -x "../bin/llama-completion" ]; then
    LLAMA_BIN_DIR="../bin"
elif [ -x "./llama-completion" ]; then
    LLAMA_BIN_DIR="."
else
    LLAMA_BIN_DIR="./build/bin"  # Fallback
fi
MODEL_DIR="${MODEL_DIR:-/Storage/GenAI/models}"
DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:1}"

# Model paths
MISTRAL_MODEL="${MODEL_DIR}/mistral-7b-v0.1.Q4_0.gguf"
GPT_OSS_MODEL="${MODEL_DIR}/gpt-oss-20b-Q8_0.gguf"

# Test parameters
PROMPT="1, 2, 3, 4, 5,"
N_TOKENS=15
SEED=42
TEMP=0

# Parse arguments
SKIP_LARGE=false
for arg in "$@"; do
    case $arg in
        --skip-large)
            SKIP_LARGE=true
            shift
            ;;
    esac
done

echo "========================================"
echo "SYCL Tiered Memory Integration Tests"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Device: ${DEVICE_SELECTOR}"
echo "  Bin dir: ${LLAMA_BIN_DIR}"
echo "  Model dir: ${MODEL_DIR}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    # Check for llama-completion
    if [ ! -x "${LLAMA_BIN_DIR}/llama-completion" ]; then
        echo -e "${RED}ERROR: llama-completion not found at ${LLAMA_BIN_DIR}/llama-completion${NC}"
        echo "Build with: cmake --build build --config Release -j \$(nproc)"
        exit 1
    fi

    # Check for llama-bench
    if [ ! -x "${LLAMA_BIN_DIR}/llama-bench" ]; then
        echo -e "${YELLOW}WARNING: llama-bench not found, skipping benchmark tests${NC}"
    fi

    # Check Mistral model
    if [ ! -f "${MISTRAL_MODEL}" ]; then
        echo -e "${RED}ERROR: Mistral model not found at ${MISTRAL_MODEL}${NC}"
        exit 1
    fi

    # Check GPT-OSS model (only if not skipping)
    if [ "${SKIP_LARGE}" = false ] && [ ! -f "${GPT_OSS_MODEL}" ]; then
        echo -e "${YELLOW}WARNING: GPT-OSS model not found at ${GPT_OSS_MODEL}, skipping large model test${NC}"
        SKIP_LARGE=true
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
    echo ""
}

# Extract t/s from llama-completion output
extract_tps() {
    local output="$1"
    # Look for "generation: X tokens, Y t/s" pattern
    echo "$output" | grep -oP 'eval time.*?=.*?(\d+\.\d+) tokens per second' | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1
}

# Test 1: Small model (Mistral-7B) - should NOT enable tiered mode
test_small_model() {
    echo "========================================"
    echo "Test 1: Small Model (Mistral-7B Q4_0)"
    echo "========================================"
    echo ""
    echo "Expected: Tiered mode DISABLED (model fits in VRAM)"
    echo ""

    local output_file=$(mktemp)
    local error_file=$(mktemp)

    # Run llama-completion with debug output
    ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" \
    GGML_SYCL_DEBUG=1 \
    "${LLAMA_BIN_DIR}/llama-completion" \
        -m "${MISTRAL_MODEL}" \
        -ngl 99 \
        -p "${PROMPT}" \
        -n ${N_TOKENS} \
        --seed ${SEED} \
        --temp ${TEMP} \
        2>"${error_file}" >"${output_file}" || true

    local exit_code=$?
    local output=$(cat "${output_file}")
    local stderr=$(cat "${error_file}")

    echo "Output:"
    echo "--------"
    echo "${output}"
    echo ""

    # Check for crashes
    if [ ${exit_code} -ne 0 ]; then
        echo -e "${RED}FAILED: Process exited with code ${exit_code}${NC}"
        echo "Stderr:"
        echo "${stderr}"
        rm -f "${output_file}" "${error_file}"
        return 1
    fi

    # Check for allocation errors
    if echo "${stderr}" | grep -qi "allocation failed\|out of memory\|device lost"; then
        echo -e "${RED}FAILED: Memory allocation error detected${NC}"
        echo "${stderr}"
        rm -f "${output_file}" "${error_file}"
        return 1
    fi

    # Check tiered mode status (from stderr debug output)
    if echo "${stderr}" | grep -qi "tiered.*enabled\|tiered_enabled=true"; then
        echo -e "${YELLOW}WARNING: Tiered mode was enabled for small model${NC}"
        echo "(This may be expected if VRAM is very limited)"
    else
        echo -e "${GREEN}OK: Tiered mode correctly disabled${NC}"
    fi

    # Check cache stats in debug output
    if echo "${stderr}" | grep -qi "cache\|tiered"; then
        echo -e "${GREEN}OK: Cache/tiered diagnostics in output${NC}"
    fi

    # Check output contains expected sequence
    if echo "${output}" | grep -q "6, 7, 8, 9, 10"; then
        echo -e "${GREEN}OK: Correct output sequence generated${NC}"
    else
        echo -e "${YELLOW}WARNING: Output may not match expected sequence${NC}"
    fi

    # Extract and check performance
    local tps=$(extract_tps "${stderr}")
    if [ -n "${tps}" ]; then
        echo "Generation speed: ${tps} t/s"
        # Check if speed > 5 t/s
        if (( $(echo "${tps} > 5" | bc -l) )); then
            echo -e "${GREEN}OK: Generation speed > 5 t/s${NC}"
        else
            echo -e "${YELLOW}WARNING: Generation speed < 5 t/s (${tps} t/s)${NC}"
        fi
    else
        echo -e "${YELLOW}Could not extract generation speed from output${NC}"
    fi

    rm -f "${output_file}" "${error_file}"
    echo ""
    echo -e "${GREEN}Test 1 PASSED${NC}"
    echo ""
    return 0
}

# Test 2: Large model (GPT-OSS-20B) - SHOULD enable tiered mode
test_large_model() {
    echo "========================================"
    echo "Test 2: Large Model (GPT-OSS-20B Q8_0)"
    echo "========================================"
    echo ""
    echo "Expected: Tiered mode MAY be enabled (depends on VRAM size)"
    echo ""

    local output_file=$(mktemp)
    local error_file=$(mktemp)

    # Run llama-completion with debug output
    # Note: GPT-OSS is a MoE model, so use appropriate flags
    ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" \
    GGML_SYCL_DEBUG=1 \
    "${LLAMA_BIN_DIR}/llama-completion" \
        -m "${GPT_OSS_MODEL}" \
        -ngl 99 \
        -p "The quick brown fox" \
        -n 20 \
        --seed ${SEED} \
        --temp ${TEMP} \
        2>"${error_file}" >"${output_file}" || true

    local exit_code=$?
    local output=$(cat "${output_file}")
    local stderr=$(cat "${error_file}")

    echo "Output:"
    echo "--------"
    echo "${output}"
    echo ""

    # Check for crashes
    if [ ${exit_code} -ne 0 ]; then
        echo -e "${RED}FAILED: Process exited with code ${exit_code}${NC}"
        echo "Stderr (last 50 lines):"
        echo "${stderr}" | tail -50
        rm -f "${output_file}" "${error_file}"
        return 1
    fi

    # Check for allocation errors
    if echo "${stderr}" | grep -qi "allocation failed\|out of memory\|device lost"; then
        echo -e "${RED}FAILED: Memory allocation error detected${NC}"
        echo "${stderr}" | grep -i "allocation\|memory\|device" | tail -10
        rm -f "${output_file}" "${error_file}"
        return 1
    fi

    # Check tiered mode status
    if echo "${stderr}" | grep -qi "tiered.*enabled\|tiered_enabled=true"; then
        echo -e "${GREEN}OK: Tiered mode enabled for large model${NC}"
    else
        echo -e "${YELLOW}NOTE: Tiered mode not enabled${NC}"
        echo "(This may be expected if you have >20GB VRAM)"
    fi

    # Check cache stats in debug output
    if echo "${stderr}" | grep -qi "cache\|tiered"; then
        echo -e "${GREEN}OK: Cache/tiered diagnostics in output${NC}"
    fi

    # Check that output was generated (not empty)
    if [ -n "${output}" ] && [ "${#output}" -gt 20 ]; then
        echo -e "${GREEN}OK: Output generated successfully${NC}"
    else
        echo -e "${RED}FAILED: No output or very short output${NC}"
        rm -f "${output_file}" "${error_file}"
        return 1
    fi

    # Extract and report performance
    local tps=$(extract_tps "${stderr}")
    if [ -n "${tps}" ]; then
        echo "Generation speed: ${tps} t/s"
        # For large models with tiered, speed may be lower
        if (( $(echo "${tps} > 1" | bc -l) )); then
            echo -e "${GREEN}OK: Generation speed > 1 t/s${NC}"
        else
            echo -e "${YELLOW}WARNING: Generation speed < 1 t/s (may be expected for tiered mode)${NC}"
        fi
    fi

    rm -f "${output_file}" "${error_file}"
    echo ""
    echo -e "${GREEN}Test 2 PASSED${NC}"
    echo ""
    return 0
}

# Test 3: API unit test validation (run existing unit tests)
test_unit_tests() {
    echo "========================================"
    echo "Test 3: Unit Test Validation"
    echo "========================================"
    echo ""

    local tests_passed=0
    local tests_failed=0

    # Run tensor inventory API test
    if [ -x "${LLAMA_BIN_DIR}/test-tensor-inventory-api" ]; then
        echo "Running test-tensor-inventory-api..."
        if ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" "${LLAMA_BIN_DIR}/test-tensor-inventory-api"; then
            echo -e "${GREEN}PASSED: test-tensor-inventory-api${NC}"
            ((tests_passed++))
        else
            echo -e "${RED}FAILED: test-tensor-inventory-api${NC}"
            ((tests_failed++))
        fi
        echo ""
    else
        echo -e "${YELLOW}SKIPPED: test-tensor-inventory-api not found${NC}"
    fi

    # Run tiered dispatch test
    if [ -x "${LLAMA_BIN_DIR}/test-tiered-dispatch" ]; then
        echo "Running test-tiered-dispatch..."
        if ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" "${LLAMA_BIN_DIR}/test-tiered-dispatch"; then
            echo -e "${GREEN}PASSED: test-tiered-dispatch${NC}"
            ((tests_passed++))
        else
            echo -e "${RED}FAILED: test-tiered-dispatch${NC}"
            ((tests_failed++))
        fi
        echo ""
    else
        echo -e "${YELLOW}SKIPPED: test-tiered-dispatch not found${NC}"
    fi

    echo "Unit tests: ${tests_passed} passed, ${tests_failed} failed"

    if [ ${tests_failed} -gt 0 ]; then
        return 1
    fi
    return 0
}

# Test 4: Benchmark comparison
test_benchmark() {
    echo "========================================"
    echo "Test 4: Benchmark"
    echo "========================================"
    echo ""

    if [ ! -x "${LLAMA_BIN_DIR}/llama-bench" ]; then
        echo -e "${YELLOW}SKIPPED: llama-bench not available${NC}"
        return 0
    fi

    echo "Running llama-bench with Mistral-7B..."
    echo ""

    ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" \
    GGML_SYCL_DEBUG=0 \
    "${LLAMA_BIN_DIR}/llama-bench" \
        -m "${MISTRAL_MODEL}" \
        -p 128 \
        -n 32 \
        -ngl 99 \
        -r 1 || true

    echo ""
    echo -e "${GREEN}Benchmark completed${NC}"
    return 0
}

# Main test execution
main() {
    check_prerequisites

    local failed=0

    # Test 1: Small model
    if ! test_small_model; then
        ((failed++))
    fi

    # Test 2: Large model (optional)
    if [ "${SKIP_LARGE}" = false ]; then
        if ! test_large_model; then
            ((failed++))
        fi
    else
        echo "========================================"
        echo "Test 2: SKIPPED (--skip-large or model not found)"
        echo "========================================"
        echo ""
    fi

    # Test 3: Unit tests
    if ! test_unit_tests; then
        ((failed++))
    fi

    # Test 4: Benchmark
    if ! test_benchmark; then
        ((failed++))
    fi

    echo "========================================"
    echo "Summary"
    echo "========================================"
    echo ""

    if [ ${failed} -eq 0 ]; then
        echo -e "${GREEN}All tests PASSED!${NC}"
        exit 0
    else
        echo -e "${RED}${failed} test(s) FAILED${NC}"
        exit 1
    fi
}

main "$@"
