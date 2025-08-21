#!/bin/bash

set -e

# First try command line argument, then environment variable, then file
CONVERTED_MODEL="${1:-"$CONVERTED_MODEL"}"

# Final check if we have a model path
if [ -z "$CONVERTED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. CONVERTED_MODEL environment variable" >&2
    exit 1
fi

echo $CONVERTED_MODEL

cmake --build ../../build --target llama-logits -j8

../../build/bin/llama-logits -m "$CONVERTED_MODEL" "Hello, my name is"
