#!/usr/bin/env bash

set -e

# First try command line argument, then environment variable, then file
CONVERTED_MODEL="${1:-"$CONVERTED_MODEL"}"
MODEL_TESTING_PROMPT="${2:-"$MODEL_TESTING_PROMPT"}"
BUILD_DIR="${3:-"$BUILD_DIR"}"

if [ -z "$MODEL_TESTING_PROMPT" ]; then
    MODEL_TESTING_PROMPT="Hello, my name is"
fi

if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR="../../build"
fi

# Final check if we have a model path
if [ -z "$CONVERTED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. CONVERTED_MODEL environment variable" >&2
    exit 1
fi

echo $CONVERTED_MODEL
echo $MODEL_TESTING_PROMPT

cmake --build ${BUILD_DIR} --target llama-debug -j8

${BUILD_DIR}/bin/llama-debug -m "$CONVERTED_MODEL" -p "$MODEL_TESTING_PROMPT" --save-logits
