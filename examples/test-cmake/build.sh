#!/bin/bash

set -e

cmake -S . -B build -DCMAKE_PREFIX_PATH="${PWD}/install"
cmake --build build
LD_LIBRARY_PATH="${PWD}/install/lib/llama.cpp:${PWD}/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./build/test-cmake
