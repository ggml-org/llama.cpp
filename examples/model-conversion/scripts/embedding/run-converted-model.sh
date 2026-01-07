#!/usr/bin/env bash

set -e

# Parse command line arguments
CONVERTED_MODEL=""
PROMPTS_FILE=""
USE_POOLING=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--prompts-file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --pooling)
            USE_POOLING="1"
            shift
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

cmake --build ../../build --target llama-debug -j8
if [ -n "$USE_POOLING" ]; then
    ../../build/bin/llama-debug -m "$CONVERTED_MODEL" --embedding --pooling mean -p "$PROMPT" --save-logits
else
    ../../build/bin/llama-debug -m "$CONVERTED_MODEL" --embedding --pooling none -p "$PROMPT" --save-logits
fi
