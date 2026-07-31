#!/bin/bash
# Build and run the interleaved kernel benchmark.
# Usage: bash tools/quantize/tessera/bench_interleaved.sh
# Run from the repository root.

set -e

echo "Building bench_interleaved..."
clang -x objective-c -std=c17 -O2 \
    -framework Metal -framework Foundation \
    -I ggml/src -I ggml/include \
    tools/quantize/tessera/bench_interleaved.m \
    -o /tmp/bench_interleaved \
    -fobjc-arc

echo "Running benchmark..."
cd "$(git rev-parse --show-toplevel)"
/tmp/bench_interleaved
