#!/bin/bash
# Test that MoE prompt processing is not slower than baseline
#
# This test validates that the fused ESIMD MoE kernel correctly falls back
# to oneDNN batching for large batch sizes (prefill). The fused kernel is
# optimized for decode (single-token) but is slower than oneDNN for large batches.

set -e

source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

MODEL="/Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf"
BASELINE=250  # t/s - conservative threshold (master achieves ~280 t/s)

if [ ! -f "$MODEL" ]; then
    echo "SKIP: Model not found: $MODEL"
    exit 0
fi

if [ ! -f "./build/bin/llama-bench" ]; then
    echo "SKIP: llama-bench not found - build first"
    exit 0
fi

# Run benchmark and extract pp512 performance
# Output format: | model | size | params | backend | ngl | fa | test | t/s |
# Extract field 9 (t/s column), then get just the first number (before ±)
RESULT=$(ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m "$MODEL" \
  -p 512 -n 0 -ngl 99 -fa 1 2>&1 | grep "pp512" | awk -F'|' '{print $9}' | awk '{print $1}')

if [ -z "$RESULT" ]; then
    echo "FAIL: Could not extract pp512 result from benchmark"
    exit 1
fi

# Validate that result is a valid number (digits with optional decimal point)
if ! echo "$RESULT" | grep -qE '^[0-9]+\.?[0-9]*$'; then
    echo "FAIL: Invalid result format: '$RESULT' (expected numeric value)"
    exit 1
fi

echo "MoE pp512: ${RESULT} t/s (target >= ${BASELINE})"

# Compare as integers (multiply by 100 to handle decimals)
RESULT_INT=$(echo "$RESULT * 100" | bc | cut -d'.' -f1)
BASELINE_INT=$((BASELINE * 100))

if [ "$RESULT_INT" -lt "$BASELINE_INT" ]; then
    echo "FAIL: Performance ${RESULT} t/s below threshold ${BASELINE} t/s"
    exit 1
fi

echo "PASS"
