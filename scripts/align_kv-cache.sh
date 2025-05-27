#!/bin/bash
# KV Cache Alignment Testing Script - Simplified Version

set -e

# Clean up any existing GGUF files in current directory
echo "Cleaning up existing GGUF files..."
rm -f *.gguf
echo "✓ GGUF files cleaned"

MODEL="/datasets/gguf/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
PROMPT="Write a quick sort: "
STEPS=1

echo "=== KV Cache Alignment Test ==="
# Create F16 reference
CMD="./build-arm64/bin/llama-kqv-trace-monitor \
    -m \"$MODEL\" \
    -p \"$PROMPT\" \
    --layer 0 \
    -t 12 \
    -fa \
    -n $STEPS \
    -ngl 0 \
    --seed 1024 \
    -ctk f16 \
    -ctv f16 \
    --save-gguf reference_f16.gguf"
echo "Executing: $CMD"
eval $CMD > /dev/null 2>&1 && echo "✓ F16 reference created"

# Test Q4_0 alignment and compare with reference
CMD="./build-arm64/bin/llama-tensor-diff-analyzer \
    -m \"$MODEL\" \
    -p \"$PROMPT\" \
    --layer 0 \
    -t 12 \
    -fa \
    -n $STEPS \
    -ngl 0 \
    --seed 1024 \
    -ctk f16 \
    -ctv f16 \
    --mixed-kv-cache \
   --reference reference_f16.gguf \
    --tolerance-abs 1e-3"
echo "Executing: $CMD"
eval $CMD && echo "✓ Q4_0 alignment test completed"
