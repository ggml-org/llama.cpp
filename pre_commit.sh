#!/bin/bash
set -e
echo "Running pre-commit checks..."
# Check tests
./tests/run_all.sh
# Build
cd llama.cpp-PoC && cmake --build build -j$(nproc)
cd build-win && cmake --build . -j$(nproc)
echo "Pre-commit passed."
