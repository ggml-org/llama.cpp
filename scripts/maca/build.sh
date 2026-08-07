#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)

MACA_PATH=${MACA_PATH:-/opt/maca}
MACA_CU_BRIDGE=${MACA_CU_BRIDGE:-$MACA_PATH/tools/cu-bridge}
MACA_VIRTUAL_ROOT=${MACA_VIRTUAL_ROOT:-${HOME:?HOME must be set}/cu-bridge}
MACA_VIRTUAL_CUDA=${MACA_VIRTUAL_CUDA:-$MACA_VIRTUAL_ROOT/CUDA_DIR}
MACA_BUILD_DIR=${MACA_BUILD_DIR:-$REPO_ROOT/build-maca}
MACA_BUILD_JOBS=${MACA_BUILD_JOBS:-8}
MACA_GRAPHS=${MACA_GRAPHS:-OFF}

for tool in "$MACA_CU_BRIDGE/tools/cmake_mock" "$MACA_CU_BRIDGE/tools/cmake_maca"; do
    if [[ ! -x "$tool" ]]; then
        echo "Required MACA cu-bridge tool is missing: $tool" >&2
        exit 1
    fi
done

if [[ -z "${CUCC_TARGETS:-}" && -z "${CUCC_TARGETS_FROM_DEVICE:-}" ]]; then
    MACA_MULTI_TARGET="$MACA_CU_BRIDGE/tools/multi_target.sh"
    if [[ ! -f "$MACA_MULTI_TARGET" ]]; then
        echo "Set CUCC_TARGETS or CUCC_TARGETS_FROM_DEVICE for the target GPU" >&2
        exit 1
    fi

    # Use the SDK detector so one source package can build for C500 or C600.
    # Run it in a child shell because older SDK revisions reference an unset
    # local variable and are not compatible with this script's nounset mode.
    CUCC_TARGETS_FROM_DEVICE=$(bash -c 'source "$1"; get_arch_from_device' _ "$MACA_MULTI_TARGET")
    if [[ -z "$CUCC_TARGETS_FROM_DEVICE" || "$CUCC_TARGETS_FROM_DEVICE" == "NOT_FOUND" ]]; then
        echo "Unable to detect the MACA target; set CUCC_TARGETS explicitly" >&2
        exit 1
    fi
fi

export MACA_PATH
export MACA_ROOT="$MACA_PATH"
export CUCC_PATH="$MACA_CU_BRIDGE"
export CUCC_TARGETS_FROM_DEVICE="${CUCC_TARGETS_FROM_DEVICE:-}"
export PATH="$MACA_PATH/bin:$MACA_CU_BRIDGE/bin:$MACA_CU_BRIDGE/tools:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_CU_BRIDGE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "MACA target: ${CUCC_TARGETS:-${CUCC_TARGETS_FROM_DEVICE}}"

cd "$REPO_ROOT"

"$MACA_CU_BRIDGE/tools/cmake_mock" \
    -S . \
    -B "$MACA_BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=52 \
    -DCUDAToolkit_ROOT="$MACA_VIRTUAL_CUDA" \
    -DCMAKE_CUDA_COMPILER="$MACA_VIRTUAL_CUDA/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$MACA_VIRTUAL_ROOT/bin/c++" \
    -DCMAKE_CUDA_FLAGS="-I$MACA_CU_BRIDGE/include" \
    -DGGML_CUDA=OFF \
    -DGGML_MACA=ON \
    -DGGML_MACA_GRAPHS="$MACA_GRAPHS" \
    -DGGML_NATIVE=OFF \
    -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_TESTS=ON

# cu-bridge can consume substantial memory while instantiating ggml-cuda
# templates. Build the backend conservatively, then finish host targets in
# parallel.
"$MACA_CU_BRIDGE/tools/cmake_maca" \
    --build "$MACA_BUILD_DIR" \
    --target ggml-maca \
    -j1

"$MACA_CU_BRIDGE/tools/cmake_maca" \
    --build "$MACA_BUILD_DIR" \
    --target test-backend-ops llama-bench llama-cli llama-server \
    -j"$MACA_BUILD_JOBS"

echo "MACA build completed: $MACA_BUILD_DIR"
