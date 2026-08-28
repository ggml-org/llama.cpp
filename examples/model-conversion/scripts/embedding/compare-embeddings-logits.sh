#!/usr/bin/env bash

set -e

# Parse command line arguments
MODEL_PATH=""
MODEL_NAME=""
PROMPTS_FILE=""

# First argument is always model path
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    MODEL_PATH="$1"
    shift
fi

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompts-file|-pf)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        *)
            # If MODEL_NAME not set and this isn't a flag, use as model name
            if [ -z "$MODEL_NAME" ] && [[ "$1" != --* ]]; then
                MODEL_NAME="$1"
            fi
            shift
            ;;
    esac
done

# Set defaults
MODEL_PATH="${MODEL_PATH:-"$EMBEDDING_MODEL_PATH"}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

CONVERTED_MODEL_PATH="${CONVERTED_EMBEDDING_PATH:-"$CONVERTED_EMBEDDING_MODEL"}"
CONVERTED_MODEL_NAME="${CONVERTED_MODEL_NAME:-$(basename "$CONVERTED_MODEL_PATH" .gguf)}"

if [ -t 0 ]; then
    CPP_EMBEDDINGS="data/llamacpp-${CONVERTED_MODEL_NAME}-embeddings.bin"
else
    # Process piped JSON data and convert to binary (matching logits.cpp format)
    TEMP_FILE=$(mktemp /tmp/tmp.XXXXXX.binn)
    python3 -c "
import json
import sys
import struct

data = json.load(sys.stdin)

# Flatten all embeddings completely
flattened = []
for item in data:
    embedding = item['embedding']
    for token_embedding in embedding:
        flattened.extend(token_embedding)

print(f'Total embedding values: {len(flattened)}', file=sys.stderr)

# Write as binary floats - matches logitc.cpp fwrite format
with open('$TEMP_FILE', 'wb') as f:
    for value in flattened:
        f.write(struct.pack('f', value))
"
    CPP_EMBEDDINGS="$TEMP_FILE"
    trap "rm -f $TEMP_FILE" EXIT
fi

# Build the semantic_check.py command
SEMANTIC_CMD="python scripts/utils/semantic_check.py --model-path $MODEL_PATH \
    --python-embeddings data/pytorch-${MODEL_NAME}-embeddings.bin \
    --cpp-embeddings $CPP_EMBEDDINGS"

# Add prompts file if specified, otherwise use default prompt
if [ -n "$PROMPTS_FILE" ]; then
    SEMANTIC_CMD="$SEMANTIC_CMD --prompts-file \"$PROMPTS_FILE\""
else
    SEMANTIC_CMD="$SEMANTIC_CMD --prompt \"Hello world today\""
fi

# Execute the command
eval $SEMANTIC_CMD

