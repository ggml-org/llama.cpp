#!/usr/bin/env bash

#  MIT license
#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: MIT

# If you want more control, DPC++ Allows selecting a specific device through the
# following environment variable
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
source /opt/intel/oneapi/setvars.sh

#export GGML_SYCL_DEBUG=1

#ZES_ENABLE_SYSMAN=1, Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory. Recommended to use when --split-mode = layer.

INPUT_PROMPT="Building a website can be done in 10 simple steps:\nStep 1:"
MODEL_FILE=models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
NGL=99 # Layers offloaded to the GPU. If the device runs out of memory, reduce this value according to the model you are using.
CONTEXT=4096

#support malloc device memory more than 4GB.
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

if [ $# -gt 0 ]; then
    GGML_SYCL_DEVICE=$1
    echo "Using $GGML_SYCL_DEVICE as the main GPU"
    ZES_ENABLE_SYSMAN=1 ./build/bin/llama-completion -m ${MODEL_FILE} -no-cnv -p "${INPUT_PROMPT}" -n 400 -e -ngl ${NGL} -s 0 -c ${CONTEXT} -mg $GGML_SYCL_DEVICE -sm none
else
    #use multiple GPUs with same max compute units
    ZES_ENABLE_SYSMAN=1 ./build/bin/llama-completion -m ${MODEL_FILE} -no-cnv -p "${INPUT_PROMPT}" -n 400 -e -ngl ${NGL} -s 0 -c ${CONTEXT}
fi
