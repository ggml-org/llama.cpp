#!/bin/bash

#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

INPUT2="Building a website can be done in 10 simple steps:\nStep 1:"
source /opt/intel/oneapi/setvars.sh

if [ $# -gt 0 ]; then
    GGML_SYCL_DEVICE=$1
    GGML_SYCL_SINGLE_GPU=1
else
    GGML_SYCL_DEVICE=0
    GGML_SYCL_SINGLE_GPU=0
fi

#export GGML_SYCL_DEBUG=1


#ZES_ENABLE_SYSMAN=1, Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory. Recommended to use when --split-mode = layer.

if [ $GGML_SYCL_SINGLE_GPU -eq 1 ]; then
    echo "use $GGML_SYCL_DEVICE as main GPU"
    #use signle GPU only
    ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "${INPUT2}" -n 400 -e -ngl 33 -s 0 -mg $GGML_SYCL_DEVICE -sm none
else
    #use multiple GPUs with same max compute units
    ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "${INPUT2}" -n 400 -e -ngl 33 -s 0
fi

#use main GPU only
#ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "${INPUT2}" -n 400 -e -ngl 33 -s 0 -mg $GGML_SYCL_DEVICE -sm none

#use multiple GPUs with same max compute units
#ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "${INPUT2}" -n 400 -e -ngl 33 -s 0
