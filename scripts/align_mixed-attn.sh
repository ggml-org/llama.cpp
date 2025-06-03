#!/bin/bash
# Flash Attention Mixed KV Cache Debug Script - Simplified Version

set -e

# Configuration
MODEL_PATH="/datasets/gguf/Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
PROMPT="Hello, world Zijie Tian"
TARGET_LAYER=0
MAX_STEPS=3
BUILD_DIR="build-arm64"

# Clean up existing files
echo "Cleaning up existing GGUF files..."
rm -f flash_attn_trace.gguf debug_report.txt
echo "GGUF files cleaned"

# Check model file
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model file not found: $MODEL_PATH"
    exit 1
fi

# Build if needed
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Build directory not found: $BUILD_DIR"
    echo "Run: cmake -B $BUILD_DIR && cmake --build $BUILD_DIR --config Release -j12"
    exit 1
fi

# Ensure binaries exist
echo "Building required binaries..."
cmake --build "$BUILD_DIR" --config Release -j12
echo "Build completed"

# Step 1: Create reference trace
echo "=== Flash Attention Mixed KV Cache Test ==="
CMD="$BUILD_DIR/bin/llama-kqv-trace-monitor \
    -m \"$MODEL_PATH\" \
    -p \"$PROMPT\" \
    --layer $TARGET_LAYER \
    -n $MAX_STEPS \
    --save-gguf flash_attn_trace.gguf \
    -ngl 0 \
    -ctk f16 \
    -ctv f16 \
    -fa \
    -t 12 \
    --seed 1024"
echo "Executing: $CMD"
eval $CMD > /dev/null 2>&1 && echo "Reference trace created"

# Step 2: Verify implementation
CMD="$BUILD_DIR/bin/llama-flash-attn-mixed-verify \
    --input flash_attn_trace.gguf \
    --step 2 \
    --seed 1024"
echo "Executing: $CMD"

if eval $CMD; then
    VERIFY_SUCCESS=true
    echo "Verification completed successfully"
else
    VERIFY_SUCCESS=false
    echo "Verification found differences"
fi