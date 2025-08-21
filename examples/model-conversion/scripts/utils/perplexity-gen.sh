#!/bin/bash

set -e

CONVERTED_MODEL="${1:-"$CONVERTED_MODEL"}"

# Final check if we have a model path
if [ -z "$CONVERTED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. CONVERTED_MODEL environment variable" >&2
    exit 1
fi

# Check if data/wikitext-2-raw directory exists
if [ ! -d "ppl/wikitext-2-raw" ]; then
    echo "ppl/wikitext-2-raw directory does not exist. Downloading..." >&2
    mkdir -p ppl
    pushd ppl
    ./../../../scripts/get-wikitext-2.sh
    popd
fi

mkdir -p ppl
OUTPUTFILE="ppl/$(basename $CONVERTED_MODEL).kld"
echo "Model: $CONVERTED_MODEL"

cmake --build ../../build --target llama-perplexity -j8

../.././build/bin/llama-perplexity -m $CONVERTED_MODEL \
    -f ppl/wikitext-2-raw/wiki.test.raw \
    --kl-divergence-base $OUTPUTFILE

echo "Generated logits in $OUTPUTFILE"

