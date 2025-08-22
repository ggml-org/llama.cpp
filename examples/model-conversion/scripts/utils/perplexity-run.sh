#!/bin/bash

set -e

QUANTIZED_MODEL="${1:-"$QUANTIZED_MODEL"}"
LOGITS_FILE="${1:-"$LOGITS_FILE"}"

if [ -z "$QUANTIZED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. QUANTIZED_MODEL environment variable" >&2
    exit 1
fi

if [ ! -f ${LOGITS_FILE} ]; then
    echo "Error: logits file '${LOGITS_FILE} was not found"
    echo "Did you run the perplexity-gen.sh script?"
    exit 1
fi

echo "Model: $QUANTIZED_MODEL"
echo "Data file: $LOGITS_FILE"

cmake --build ../../build --target llama-perplexity -j8

../.././build/bin/llama-perplexity -m $QUANTIZED_MODEL \
    --kl-divergence-base $LOGITS_FILE \
    --kl-divergence
