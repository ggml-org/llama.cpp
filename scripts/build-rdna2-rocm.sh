#!/usr/bin/env bash
# Build this llama.cpp checkout for RDNA2/gfx1030 with HIP graphs and RCCL.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-7.14}"
TARGET_ARCH="${TARGET_ARCH:-gfx1030}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v cmake >/dev/null 2>&1 || fail "cmake is not installed"
[ -x "$ROCM_PATH/llvm/bin/clang" ] || fail "ROCm HIP compiler not found under $ROCM_PATH"
[ -f "$ROCM_PATH/lib/librccl.so" ] || fail "RCCL library not found: $ROCM_PATH/lib/librccl.so"

if pgrep -f "^$BUILD_DIR/bin/llama-server" >/dev/null 2>&1; then
    fail "llama-server from this build is running; stop it before rebuilding"
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export HIP_PATH="$ROCM_PATH"
export HIPCXX="$ROCM_PATH/llvm/bin/clang"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CMAKE_FLAGS=(
    -DGGML_HIP=ON
    -DGGML_HIP_RCCL=ON
    -DGGML_HIP_GRAPHS=ON
    -DGGML_HIP_NO_VMM=ON
    -DGGML_NATIVE=ON
    -DAMDGPU_TARGETS="$TARGET_ARCH"
    -DCMAKE_HIP_ARCHITECTURES="$TARGET_ARCH"
    -DCMAKE_HIP_COMPILER="$HIPCXX"
    -DCMAKE_PREFIX_PATH="$ROCM_PATH"
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DCMAKE_BUILD_TYPE=Release
)

echo "Building RDNA2 ROCm/RCCL server"
echo "  source: $ROOT_DIR"
echo "  output: $BUILD_DIR/bin/llama-server"
echo "  target: $TARGET_ARCH"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" "${CMAKE_FLAGS[@]}"
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

SERVER="$BUILD_DIR/bin/llama-server"
HIP_LIB="$BUILD_DIR/bin/libggml-hip.so.0"

[ -x "$SERVER" ] || fail "server binary was not produced"
[ -f "$HIP_LIB" ] || fail "HIP backend library was not produced"
grep -q '^GGML_HIP_RCCL:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt" || fail "RCCL is not enabled in CMake cache"
ldd "$HIP_LIB" | grep -q 'librccl\.so' || fail "HIP backend is not linked to librccl"
if ldd "$SERVER" "$HIP_LIB" | grep -q 'not found'; then
    ldd "$SERVER" "$HIP_LIB" | grep 'not found' >&2
    fail "one or more shared libraries are unresolved"
fi

echo
echo "Build complete: $SERVER"
echo "Run tensor split with GGML_CUDA_ALLREDUCE=nccl"