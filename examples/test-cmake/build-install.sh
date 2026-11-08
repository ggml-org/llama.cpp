#!/bin/bash

set -e

rm -rf llama-build-install install

cmake --fresh -S ../../. -B llama-build-install -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_TESTS_INSTALL=OFF \
  -DCMAKE_INSTALL_PREFIX="${PWD}/install" \
  -DGGML_BACKEND_DIR="${PWD}/install/lib/llama.cpp" \
  -DGGML_LIB_INSTALL_DIR="${PWD}/install/lib/llama.cpp" \
  -DLLAMA_LIB_INSTALL_DIR="${PWD}/install/lib/llama.cpp" \
  -DLLAMA_TOOLS_INSTALL=OFF

cmake --build llama-build-install --parallel 12
cmake --install llama-build-install
