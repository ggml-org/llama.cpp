#!/usr/bin/env bash

set -e

QUANTIZED_MODEL="${1:-"$QUANTIZED_MODEL"}"
BUILD_DIR="${2:-"$BUILD_DIR"}"

if [ -z "$QUANTIZED_MODEL" ]; then
    echo "Error: Model path must be provided either as:" >&2
    echo "  1. Command line argument" >&2
    echo "  2. QUANTIZED_MODEL environment variable" >&2
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

if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR="../../build"
fi

cmake --build $BUILD_DIR --target llama-perplexity -j8

${BUILD_DIR}/bin/llama-perplexity -m $QUANTIZED_MODEL -f ppl/wikitext-2-raw/wiki.test.raw


