#!/usr/bin/env bash

set -euo pipefail

build_dir="${BUILD_DIR:-build-rocm}"
jobs="${JOBS:-$(nproc)}"
rocm_path="${ROCM_PATH:-/opt/rocm}"

if [[ ! -d "${rocm_path}" ]]; then
    echo "ROCm was not found at ${rocm_path}. Set ROCM_PATH to the ROCm SDK directory." >&2
    exit 1
fi

export ROCM_PATH="${rocm_path}"
export HIP_PATH="${HIP_PATH:-${rocm_path}}"

cmake_args=(
    -S .
    -B "${build_dir}"
    -DGGML_HIP=ON
    -DGGML_CUDA=OFF
    -DGGML_VULKAN=OFF
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_CURL=OFF
    -DCMAKE_BUILD_TYPE=Release
    "-DROCM_PATH=${rocm_path}"
    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${rocm_path}"
)

if [[ -n "${CMAKE_HIP_COMPILER:-}" ]]; then
    cmake_args+=("-DCMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER}")
elif [[ -x "${rocm_path}/llvm/bin/clang" ]]; then
    cmake_args+=("-DCMAKE_HIP_COMPILER=${rocm_path}/llvm/bin/clang")
fi

if [[ -n "${GPU_TARGETS:-}" ]]; then
    cmake_args+=("-DGPU_TARGETS=${GPU_TARGETS}")
fi

cmake "${cmake_args[@]}" "$@"
cmake --build "${build_dir}" --config Release --parallel "${jobs}"
