#!/bin/bash
# Bisect test script - returns 0 for good, 1 for bad, 125 for skip

set -e

# Source oneAPI
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

# Build
echo "Building..."
if ! ninja -C build -j $(nproc) llama-completion 2>&1 | tail -20; then
    echo "Build failed, skip"
    exit 125
fi

# Run test
echo "Running inference test..."
OUTPUT=$(ONEAPI_DEVICE_SELECTOR=level_zero:1 timeout 120 ./build/bin/llama-completion \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
    -ngl 99 -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 2>&1 || true)

echo "Output: $OUTPUT"

# Check for expected output
if echo "$OUTPUT" | grep -q "6, 7, 8, 9, 10"; then
    echo "GOOD - correct output"
    exit 0
else
    echo "BAD - incorrect output"
    exit 1
fi
