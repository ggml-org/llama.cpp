#!/usr/bin/env bash

set -e

# First try command line argument, then environment variable, then file
CONVERTED_MODEL="${1:-"$CONVERTED_MODEL"}"
BUILD_DIR="${2:-"$BUILD_DIR"}"

# Final check if we have a model path
if [ -z "$CONVERTED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. CONVERTED_MODEL environment variable" >&2
    exit 1
fi

if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR="../../build"
fi

cmake --build ${BUILD_DIR} --target llama-debug -j8

${BUILD_DIR}/bin/llama-debug -m $CONVERTED_MODEL --embedding -p "Hello world today" --save-logits
