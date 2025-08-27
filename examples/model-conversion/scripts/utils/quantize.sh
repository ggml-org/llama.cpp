#!/bin/bash

set -e

CONVERTED_MODEL="${1:-"$CONVERTED_MODEL"}"
QUANTIZED_TYPE="${2:-"$QUANTIZED_TYPE"}"
TOKEN_EMBD_TYPE="${3:-"${TOKEN_EMBD_TYPE}"}"
OUTPUT_TYPE="${4:-"${OUTPUT_TYPE}"}"
QUANTIZED_MODEL=$CONVERTED_MODEL

# Final check if we have a model path
if [ -z "$CONVERTED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. CONVERTED_MODEL environment variable" >&2
    exit 1
fi

if [ -z "$QUANTIZED_TYPE" ]; then
    echo "Error: QUANTIZED_TYPE is required" >&2
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

echo $TOKEN_EMBD_TYPE
echo $OUTPUT_TYPE

CMD_ARGS=("../../build/bin/llama-quantize")
[[ -n "$TOKEN_EMBD_TYPE" ]] && CMD_ARGS+=("--token-embedding-type" "$TOKEN_EMBD_TYPE")
[[ -n "$OUTPUT_TYPE" ]]     && CMD_ARGS+=("--output-tensor-type" "$OUTPUT_TYPE")
CMD_ARGS+=("$CONVERTED_MODEL" "$QUANTIZED_MODEL" "$QUANTIZED_TYPE")

"${CMD_ARGS[@]}"

echo "Quantized model saved to: $QUANTIZED_MODEL"
