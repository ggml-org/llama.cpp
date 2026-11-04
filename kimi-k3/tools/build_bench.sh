#!/usr/bin/env bash
# build matmul_bench and bw_probe against the existing build (no CMake target, no ggml change)
#   kimi-k3/tools/build_bench.sh [OUT_DIR]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BUILD=${BUILD:-build}
LIB="$ROOT/$BUILD/bin"
OUT=${1:-/tmp/claude-1000/-home-aaronzhong-Projects-llama-cpp/9bbbffdd-0d58-401e-a700-32b343899505/scratchpad/bench/matmul}
mkdir -p "$OUT"

# use the CUDA toolkit that libggml-cuda was built with (its cudart directory is in the library's RUNPATH)
CUDA=${CUDA:-/usr/local/cuda}
NVCC=${NVCC:-$CUDA/bin/nvcc}
CUDA_LIB="$CUDA/lib64"
[[ -d "$CUDA_LIB" ]] || CUDA_LIB="$CUDA/lib"

g++ -O2 -std=c++17 -Wall -Wextra -I"$ROOT/ggml/include" "$HERE/matmul_bench.cpp" -o "$OUT/matmul_bench" -ldl \
    -L"$LIB" -lggml -lggml-base -lggml-cpu -lggml-cuda -Wl,-rpath,"$LIB" -I"$CUDA/include" -L"$CUDA_LIB" -Wl,-rpath,"$CUDA_LIB" -lcudart -lpthread

"$NVCC" -O3 -std=c++17 -arch=${CUDA_ARCH:-sm_120a} -Xcompiler -O3,-march=native,-Wall "$HERE/bw_probe.cu" -o "$OUT/bw_probe"

echo "built $OUT/matmul_bench $OUT/bw_probe"
