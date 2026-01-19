#!/usr/bin/env bash

set -e

# Parse command line arguments
CONVERTED_MODEL=""
PROMPTS_FILE=""
EMBD_NORMALIZE="2"

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--prompts-file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --embd-normalize)
            EMBD_NORMALIZE="$2"
            shift 2
            ;;
        *)
            if [ -z "$CONVERTED_MODEL" ]; then
                CONVERTED_MODEL="$1"
            fi
            shift
            ;;
    esac
done

# First try command line argument, then environment variable
CONVERTED_MODEL="${CONVERTED_MODEL:-"$CONVERTED_EMBEDDING_MODEL"}"
BUILD_DIR="${BUILD_DIR:-"../../build"}"

# Final check if we have a model path
if [ -z "$CONVERTED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. CONVERTED_EMBEDDING_MODEL environment variable" >&2
    exit 1
fi

# Read prompt from file or use default
if [ -n "$PROMPTS_FILE" ]; then
    if [ ! -f "$PROMPTS_FILE" ]; then
        echo "Error: Prompts file '$PROMPTS_FILE' not found" >&2
        exit 1
    fi
    PROMPT=$(cat "$PROMPTS_FILE")
else
    PROMPT="Hello world today"
fi

echo $CONVERTED_MODEL

cmake --build ${BUILD_DIR} --target llama-debug -j8
${BUILD_DIR}/bin/llama-debug -m "$CONVERTED_MODEL" --embedding -p "$PROMPT" --save-logits --embd-normalize $EMBD_NORMALIZE
