#!/bin/bash

set -e

CONVERTED_MODEL="${1:-"$CONVERTED_MODEL"}"
QUANTIZED_TYPE="${2:-"$QUANTIZED_TYPE"}"
QUANTIZED_MODEL=$CONVERTED_MODEL

# Final check if we have a model path
if [ -z "$CONVERTED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. CONVERTED_MODEL environment variable" >&2
    exit 1
fi

echo $CONVERTED_MODEL

# Process the quantized model filename
if [[ "$QUANTIZED_MODEL" == *.gguf ]]; then
    # Remove .gguf suffix, add quantized type, then add .gguf back
    BASE_NAME="${QUANTIZED_MODEL%.gguf}"
    QUANTIZED_MODEL="${BASE_NAME}-${QUANTIZED_TYPE}.gguf"
else
    echo "Error: QUANTIZED_MODEL must end with .gguf extension" >&2
    exit 1
fi


cmake --build ../../build --target llama-quantize -j8

../../build/bin/llama-quantize $CONVERTED_MODEL $QUANTIZED_MODEL $QUANTIZED_TYPE

echo "Quantized model saved to: $QUANTIZED_MODEL"
