#!/bin/bash

set -e

if [ "${USE_SUBDIR:-OFF}" = "ON" ]; then
    BUILD_DIR="build-subdir"
    CMAKE_ARGS="-DLLAMA_TEST_USE_SUBDIR=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON"
    LIB_PATH="${PWD}/${BUILD_DIR}/bin"
else
    BUILD_DIR="build"
    CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${PWD}/install"
    LIB_PATH="${PWD}/install/lib/llama.cpp"
fi

cmake --fresh -S . -B "${BUILD_DIR}" ${CMAKE_ARGS}
cmake --build "${BUILD_DIR}" -j 8

LD_LIBRARY_PATH="${LIB_PATH}:${PWD}/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "./${BUILD_DIR}/test-cmake"
