#!/usr/bin/env bash
# build_expert_profile.sh
# Builds llama.cpp with the expert-profile tool in WSL2 with CUDA.
# Run this from the tools/moe-pruning/ directory: bash build_expert_profile.sh

set -e

LLAMA_SRC="../.."
BUILD_DIR="$LLAMA_SRC/build_expert"

echo "=== Building llama.cpp + expert-profile tool ==="
echo "  Source : $LLAMA_SRC"
echo "  Build  : $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CUDA
cmake "$LLAMA_SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    2>&1 | tail -20

# Build only the expert-profile target (fast)
cmake --build . --target llama-expert-profile --config Release -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "  Binary: $BUILD_DIR/tools/expert-profile/llama-expert-profile"
echo ""
echo "=== Usage ==="
echo "  $BUILD_DIR/tools/expert-profile/llama-expert-profile \\"
echo "    -m ~/nemotron-3-nano-30b-Q4_K_M.gguf \\"
echo "    --jsonl ./sample_calibration.jsonl \\"
echo "    --output ./expert_stats_reap.json \\"
echo "    --n-experts 128 \\"
echo "    --ctx-size 16384 \\"
echo "    -ngl 99"
