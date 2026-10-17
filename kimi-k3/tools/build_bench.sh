#!/usr/bin/env bash
# build matmul_bench and bw_probe against the existing build (no CMake target, no ggml change)
#   kimi-k3/tools/build_bench.sh [OUT_DIR]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
LIB="$ROOT/build/bin"
OUT=${1:-/tmp/claude-1000/-home-aaronzhong-Projects-llama-cpp/9bbbffdd-0d58-401e-a700-32b343899505/scratchpad/bench/matmul}
mkdir -p "$OUT"

# libggml-cuda links the system CUDA 12.0 runtime (/usr/lib/x86_64-linux-gnu): use the same nvcc / cudart
NVCC=${NVCC:-/usr/bin/nvcc}

g++ -O2 -std=c++17 -Wall -Wextra -I"$ROOT/ggml/include" "$HERE/matmul_bench.cpp" -o "$OUT/matmul_bench" -ldl \
    -L"$LIB" -lggml -lggml-base -lggml-cpu -lggml-cuda -Wl,-rpath,"$LIB" -lcudart -lpthread

"$NVCC" -O3 -std=c++17 -arch=sm_86 -Xcompiler -O3,-march=native,-Wall "$HERE/bw_probe.cu" -o "$OUT/bw_probe"

echo "built $OUT/matmul_bench $OUT/bw_probe"
