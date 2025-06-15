#!/bin/bash
# KQV Tensor Reader Test Script - Simple Version

set -e

# Clean up any existing GGUF files in current directory
echo "Cleaning up existing GGUF files..."
rm -f *.gguf
echo "✓ GGUF files cleaned"

MODEL="/datasets/gguf/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
PROMPT=""
STEPS=2
TRACE_LAYER=0
OUTPUT_FILE="reference_f32.gguf"
THREADS=1

echo "=== KQV Tensor Reader Test ==="

# Step 1: Generate tensor data using kqv-trace-monitor
CMD="./build-arm64/bin/kqv-trace-monitor \
    -m \"$MODEL\" \
    -p \"$PROMPT\" \
    --layer $TRACE_LAYER \
    -t $THREADS \
    -fa \
    -n $STEPS \
    -ngl 0 \
    --seed 1024 \
    -ctk f16 \
    -ctv f16 \
    --mixed-kv-cache \
    --save-gguf $OUTPUT_FILE"
echo "Executing: $CMD"
eval $CMD > /dev/null 2>&1 && echo "✓ KQV tensor GGUF generated"

# Step 2: Read tensor data using kqv-tensor-reader
CMD2="./build-arm64/bin/kqv-tensor-reader -i $OUTPUT_FILE"
echo "Executing: $CMD2"
eval $CMD2

echo
echo "=== Test Completed Successfully ==="
echo "✓ KQV tensor generation completed"
echo "✓ KQV tensor reading completed"