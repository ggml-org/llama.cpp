#!/usr/bin/env bash
# build.sh — Build CUDA target. Exits non-zero on failure.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

# Configure if build dir doesn't exist
if [[ ! -d build-cuda ]]; then
    cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;120" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF 2>&1
fi

cmake --build build-cuda -j$(nproc) 2>&1
