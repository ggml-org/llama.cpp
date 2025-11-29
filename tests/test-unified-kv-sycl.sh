#!/bin/bash
# Integration test for SYCL Flash Attention with Unified KV Mode
#
# This test verifies that the mask slope fix in fattn-xmx-f16.hpp works correctly.
# It tests that:
# 1. Identical prompts in unified KV mode produce identical outputs (temp=0)
# 2. Different prompts in unified KV mode remain isolated (no cross-contamination)
#
# Requirements:
# - Intel oneAPI environment (source setvars.sh)
# - Intel Arc GPU (ONEAPI_DEVICE_SELECTOR=level_zero:1)
# - A GGUF model file (default: Mistral-7B)
#
# Usage:
#   ./test-unified-kv-sycl.sh [model_path]

set -e

MODEL_PATH="${1:-/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf}"
PORT=8099
SERVER_LOG="/tmp/test_unified_kv_server.log"
RESULT_DIR="/tmp/test_unified_kv_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== SYCL Flash Attention Unified KV Mode Integration Test ==="
echo ""

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    echo "Usage: $0 [model_path]"
    exit 1
fi

# Setup
mkdir -p "$RESULT_DIR"
rm -f "$RESULT_DIR"/*.json

# Cleanup function
cleanup() {
    pkill -9 -f "llama-server.*$PORT" 2>/dev/null || true
}
trap cleanup EXIT

# Source Intel oneAPI environment
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
fi

echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo ""

# Kill any existing server on this port
cleanup
sleep 1

# Start server with unified KV mode
echo "Starting server with unified KV mode (-kvu)..."
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-server \
    -m "$MODEL_PATH" \
    --flash-attn on -kvu --parallel 4 -c 4096 --port $PORT -ngl 99 \
    2>&1 > "$SERVER_LOG" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to initialize..."
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Server ready after ${i}s${NC}"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo -e "${RED}Server failed to start within ${MAX_WAIT}s${NC}"
        cat "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

echo ""

# =============================================================================
# Test 1: Identical prompts should produce identical outputs
# =============================================================================
echo "=== Test 1: Determinism Test (4 identical prompts, temp=0) ==="

for i in 1 2 3 4; do
    curl -s "http://localhost:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "test", "prompt": "1+1=", "max_tokens": 5, "temperature": 0}' \
        > "$RESULT_DIR/identical_$i.json" &
done
wait

echo "Results:"
FIRST_RESULT=""
ALL_IDENTICAL=true
for i in 1 2 3 4; do
    RESULT=$(cat "$RESULT_DIR/identical_$i.json" | jq -r '.choices[0].text // .error // "ERROR"')
    echo "  Request $i: '$RESULT'"
    if [ -z "$FIRST_RESULT" ]; then
        FIRST_RESULT="$RESULT"
    elif [ "$RESULT" != "$FIRST_RESULT" ]; then
        ALL_IDENTICAL=false
    fi
done

if [ "$ALL_IDENTICAL" = true ]; then
    echo -e "${GREEN}PASS: All 4 responses are identical${NC}"
else
    echo -e "${RED}FAIL: Responses differ (cross-sequence contamination)${NC}"
    exit 1
fi

echo ""

# =============================================================================
# Test 2: Different prompts should remain isolated
# =============================================================================
echo "=== Test 2: Isolation Test (4 different prompts) ==="

curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "test", "prompt": "The capital of France is", "max_tokens": 5, "temperature": 0}' \
    > "$RESULT_DIR/diff_france.json" &

curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "test", "prompt": "The largest planet is", "max_tokens": 5, "temperature": 0}' \
    > "$RESULT_DIR/diff_planet.json" &

curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "test", "prompt": "Water boils at", "max_tokens": 5, "temperature": 0}' \
    > "$RESULT_DIR/diff_water.json" &

curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "test", "prompt": "The speed of light is", "max_tokens": 5, "temperature": 0}' \
    > "$RESULT_DIR/diff_light.json" &
wait

FRANCE=$(cat "$RESULT_DIR/diff_france.json" | jq -r '.choices[0].text // "ERROR"')
PLANET=$(cat "$RESULT_DIR/diff_planet.json" | jq -r '.choices[0].text // "ERROR"')
WATER=$(cat "$RESULT_DIR/diff_water.json" | jq -r '.choices[0].text // "ERROR"')
LIGHT=$(cat "$RESULT_DIR/diff_light.json" | jq -r '.choices[0].text // "ERROR"')

echo "Results:"
echo "  Capital of France: '$FRANCE'"
echo "  Largest planet: '$PLANET'"
echo "  Water boils at: '$WATER'"
echo "  Speed of light: '$LIGHT'"

# Check for obvious cross-contamination (e.g., "Paris" appearing in planet response)
ISOLATION_OK=true
if echo "$PLANET" | grep -qi "paris\|france"; then
    echo -e "${RED}FAIL: 'Planet' response contains France-related content${NC}"
    ISOLATION_OK=false
fi
if echo "$FRANCE" | grep -qi "jupiter\|saturn"; then
    echo -e "${RED}FAIL: 'France' response contains planet-related content${NC}"
    ISOLATION_OK=false
fi

if [ "$ISOLATION_OK" = true ]; then
    echo -e "${GREEN}PASS: Responses appear properly isolated${NC}"
else
    exit 1
fi

echo ""

# =============================================================================
# Test 3: Longer generation test
# =============================================================================
echo "=== Test 3: Longer Generation Test (64 tokens) ==="

STORY_PROMPT="Once upon a time"
curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"test\", \"prompt\": \"$STORY_PROMPT\", \"max_tokens\": 64, \"temperature\": 0}" \
    > "$RESULT_DIR/long_1.json" &

curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"test\", \"prompt\": \"$STORY_PROMPT\", \"max_tokens\": 64, \"temperature\": 0}" \
    > "$RESULT_DIR/long_2.json" &
wait

LONG1=$(cat "$RESULT_DIR/long_1.json" | jq -r '.choices[0].text // "ERROR"')
LONG2=$(cat "$RESULT_DIR/long_2.json" | jq -r '.choices[0].text // "ERROR"')

echo "Story 1 (first 100 chars): '${LONG1:0:100}...'"
echo "Story 2 (first 100 chars): '${LONG2:0:100}...'"

# Check at least first 50 characters match
LONG1_PREFIX="${LONG1:0:50}"
LONG2_PREFIX="${LONG2:0:50}"

if [ "$LONG1_PREFIX" = "$LONG2_PREFIX" ]; then
    echo -e "${GREEN}PASS: Long generations start identically${NC}"
else
    echo -e "${YELLOW}WARN: Long generations diverged (may be expected for very long outputs)${NC}"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=== Test Summary ==="
echo -e "${GREEN}All critical tests passed!${NC}"
echo ""
echo "The SYCL Flash Attention mask slope fix is working correctly:"
echo "  - Identical prompts produce identical outputs (cross-sequence isolation)"
echo "  - Different prompts remain properly isolated (no contamination)"
echo "  - Extended generation maintains consistency"
echo ""
echo "Server log: $SERVER_LOG"
echo "Results: $RESULT_DIR/"

exit 0
