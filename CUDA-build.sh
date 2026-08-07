#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
build_dir="${root}/build"

rm -rf "${build_dir}"
cmake -S "${root}" -B "${build_dir}" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DGGML_VULKAN=ON
cmake --build "${build_dir}" --config Release -j16
