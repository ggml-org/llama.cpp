#!/usr/bin/env bash
#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

print_usage() {
    echo "Usage: ./build.sh [fp32|fp16] [ssl|nossl] [--help]"
    echo ""
    echo "Options:"
    echo "  fp32    Build with FP32 precision (default)"
    echo "  fp16    Build with FP16 precision (faster for long-prompt inference)"
    echo "  ssl     Build with OpenSSL (default)"
    echo "  nossl   Build without OpenSSL"
    echo "  --help  Print this help message"
}

PRECISION=fp32
SSL=ON

for arg in "$@"; do
    case "$arg" in
        --help)
            print_usage
            exit 0
            ;;
        fp32|fp16)
            PRECISION="$arg"
            ;;
        ssl)
            SSL=ON
            ;;
        nossl)
            SSL=OFF
            ;;
        *)
            echo "Error: unknown option '$arg'"
            print_usage
            exit 1
            ;;
    esac
done

mkdir -p build
cd build
source /opt/intel/oneapi/setvars.sh

if [ "$PRECISION" = "fp16" ]; then
    #for FP16
    echo "cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DLLAMA_OPENSSL=$SSL # faster for long-prompt inference
"
    cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DLLAMA_OPENSSL="$SSL" # faster for long-prompt inference
else
    #for FP32
    echo "cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_OPENSSL=$SSL"
    cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_OPENSSL="$SSL"
fi

#build example/main
#cmake --build . --config Release --target main

#build example/llama-bench
#cmake --build . --config Release --target llama-bench

#build all binary
echo "cmake --build . --config Release -j$((($(nproc)+1)/2)) -v"
cmake --build . --config Release -j$((($(nproc)+1)/2)) -v
