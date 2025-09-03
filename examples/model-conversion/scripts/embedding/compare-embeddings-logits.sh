#!/usr/bin/env bash

set -e

MODEL_PATH="${1:-"$EMBEDDING_MODEL_PATH"}"
MODEL_NAME="${2:-$(basename "$MODEL_PATH")}"

if [ -t 0 ]; then
    CPP_EMBEDDINGS="data/llamacpp-${MODEL_NAME}-embeddings.bin"
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

python scripts/utils/semantic_check.py --model-path $MODEL_PATH \
    --python-embeddings data/pytorch-${MODEL_NAME}-embeddings.bin \
    --cpp-embeddings $CPP_EMBEDDINGS \
    --prompt "Hello world today"

