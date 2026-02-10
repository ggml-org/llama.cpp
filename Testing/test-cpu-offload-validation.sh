#!/bin/bash
# test-cpu-offload-validation.sh — Validate data-local CPU compute end-to-end
#
# Runs three configurations and verifies correct output:
# 1. Default (all GPU) — baseline correctness
# 2. CPU offload (30% VRAM budget) — data-local compute path
# 3. Auto-streaming (30% VRAM budget, GPU-only) — comparison path
#
# Usage: bash Testing/test-cpu-offload-validation.sh [build_dir]
#
# Requires: oneAPI environment sourced, model file at $MODEL path
#
# Device selector notes:
#   Test 1 & 3: level_zero:0  — GPU only (Arc B580)
#   Test 2:     level_zero:0;opencl:0  — GPU + CPU (for sycl::cpu_selector_v)
#   The CPU offload path needs OpenCL CPU device exposed alongside the
#   Level Zero GPU device. Without opencl:0, cpu_selector_v fails.
set -euo pipefail

BUILD_DIR="${1:-build}"
MODEL="/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf"
PROMPT='1, 2, 3, 4, 5,'
EXPECTED_PATTERN='6, 7, 8, 9, 10'
LLAMA_COMPLETION="${BUILD_DIR}/bin/llama-completion"

# Ensure binary exists
if [ ! -x "$LLAMA_COMPLETION" ]; then
    echo "FAIL: $LLAMA_COMPLETION not found or not executable"
    exit 1
fi

# Ensure model exists
if [ ! -f "$MODEL" ]; then
    echo "FAIL: Model not found at $MODEL"
    exit 1
fi

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local env_prefix="$2"
    local extra_args="${3:-}"
    local output

    echo "=== $name ==="
    output=$(eval "$env_prefix $LLAMA_COMPLETION -m $MODEL -p '$PROMPT' -n 15 --seed 42 --temp 0 $extra_args 2>/dev/null" || true)

    echo "Output: $output"

    if echo "$output" | grep -q "$EXPECTED_PATTERN"; then
        echo "PASS: Found expected pattern '$EXPECTED_PATTERN'"
        PASS=$((PASS + 1))
    else
        echo "FAIL: Expected pattern '$EXPECTED_PATTERN' not found"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

echo "=========================================="
echo "  CPU Offload Validation Test Suite"
echo "=========================================="
echo ""

# Test 1: Default (all GPU, no budget pressure)
# All weights stay in VRAM, standard GPU dispatch
run_test "Test 1: Default (all GPU)" \
    "ONEAPI_DEVICE_SELECTOR=level_zero:0"

# Wait for GPU thermal cooldown
sleep 15

# Test 2: CPU offload with 30% VRAM budget
# Weights that don't fit in VRAM are evicted to host pinned memory.
# CPU offload dispatches those layers' computation to CPU device.
# Requires opencl:0 in device selector so sycl::cpu_selector_v finds CPU.
# -fit off bypasses fit_params (known separate issue llama.cpp-io0q).
run_test "Test 2: CPU offload (30% VRAM budget)" \
    "GGML_SYCL_VRAM_BUDGET_PCT=30 GGML_SYCL_CPU_OFFLOAD=1 ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:0'" \
    "-fit off"

sleep 15

# Test 3: Auto-streaming with 30% VRAM budget (GPU-only comparison)
# Same budget constraint but GPU handles everything via weight streaming.
# Auto-streaming activates when model exceeds VRAM budget.
run_test "Test 3: Auto-streaming (30% VRAM budget, GPU-only)" \
    "GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0" \
    "-fit off"

# Summary
echo "=========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=========================================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "All tests passed."
