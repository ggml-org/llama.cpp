#!/usr/bin/env bash
#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

print_usage() {
    echo "Usage: ./build.sh [fp32|fp16] [ssl|no-ssl] [no-xmx] [--aot aot_flags] [--help]"
    echo ""
    echo "Options:"
    echo "  fp32    Build with FP32 precision (default)"
    echo "  fp16    Build with FP16 precision (faster for long-prompt inference)"
    echo "  ssl     Build with OpenSSL (default)"
    echo "  no-ssl  Build without OpenSSL"
    echo "  no-xmx  Disable XMX (DPAS) support"
    echo "  --aot   Set SYCL AOT device architecture, e.g. arl-h,acm-g10,bmg-g21 or bmg-g21"
    echo "  --help  Print this help message"
}

PRECISION=fp32
SSL=ON
AOT_FLAGS=""
SYCL_SUPPORT_XMX=ON

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            print_usage
            exit 0
            ;;
        fp32|fp16)
            PRECISION="$1"
            shift
            ;;
        ssl)
            SSL=ON
            shift
            ;;
        no-ssl)
            SSL=OFF
            shift
            ;;
        no-xmx)
            SYCL_SUPPORT_XMX=OFF
            shift
            ;;
        --aot)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --aot requires aot_flags"
                print_usage
                exit 1
            fi
            AOT_FLAGS="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown option '$1'"
            print_usage
            exit 1
            ;;
    esac
done

AOT_OPTION=()
if [[ -n "$AOT_FLAGS" ]]; then
    AOT_OPTION=("-DGGML_SYCL_DEVICE_ARCH=$AOT_FLAGS")
fi

mkdir -p build
cd build
source /opt/intel/oneapi/setvars.sh

if [ "$PRECISION" = "fp16" ]; then
    #for FP16
    printf 'cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DGGML_SYCL_SUPPORT_XMX=%s -DLLAMA_OPENSSL=%s' "$SYCL_SUPPORT_XMX" "$SSL"
    if [[ ${#AOT_OPTION[@]} -gt 0 ]]; then
        printf ' %s' "${AOT_OPTION[@]}"
    fi
    echo " # faster for long-prompt inference"
    cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DGGML_SYCL_SUPPORT_XMX="$SYCL_SUPPORT_XMX" -DLLAMA_OPENSSL="$SSL" "${AOT_OPTION[@]}" # faster for long-prompt inference
else
    #for FP32
    printf 'cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_SUPPORT_XMX=%s -DLLAMA_OPENSSL=%s' "$SYCL_SUPPORT_XMX" "$SSL"
    if [[ ${#AOT_OPTION[@]} -gt 0 ]]; then
        printf ' %s' "${AOT_OPTION[@]}"
    fi
    echo
    cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_SUPPORT_XMX="$SYCL_SUPPORT_XMX" -DLLAMA_OPENSSL="$SSL" "${AOT_OPTION[@]}"
fi

#build example/main
#cmake --build . --config Release --target main

#build example/llama-bench
#cmake --build . --config Release --target llama-bench

#build all binary
BUILD_JOBS=$((($(nproc) + 1) / 2))
echo "cmake --build . --config Release -j$BUILD_JOBS -v"
cmake --build . --config Release -j"$BUILD_JOBS" -v
