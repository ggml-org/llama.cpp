#!/bin/bash

#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

Help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

This script processes files with specified options.

Options:
  -h, --help    Display this help message and exit.
  -c, --context <value>    Set context length. Bigger need more memory.
  -p, --promote <value>    Prompt to start generation with.
  -m, --model   <value>    Full model file path.
  -mg,--main-gpu <value>   Set main GPU ID (0 - n) for single GPU mode.
  -sm,--split-mode <value> How to split the model across multiple GPUs, one of:
                            - none: use one GPU only
                            - layer (default): split layers and KV across GPUs
                            - row: split rows across GPUs
  -ngl,--n-gpu-layers <value>  Max. number of layers to store in VRAM (default: -1)
  -lv,--log-verbosity <value>  Set the verbosity threshold. Messages with a higher verbosity will be
                               ignored. Values:
                                - 0: generic output
                                - 1: error
                                - 2: warning
                                - 3: info
                                - 4: debug


EOF
}

BIN_FILE=./build/bin/llama-completion
SEED=0
GPUS_SETTING=""

INPUT_PROMPT="Building a website can be done in 10 simple steps:\nStep 1:"
MODEL_FILE=models/llama-2-7b.Q4_0.gguf
NGL=99
CONTEXT=4096
GGML_SYCL_DEVICE=-1
SPLIT_MODE=layer
LOG_VERBOSE=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--context)
            CONTEXT=$2
            # Shift twice to consume both the option flag and its value
            shift
            shift
            ;;
        -p|--promote)
            # Option that is a simple flag (boolean)
            INPUT_PROMPT="$2"
            # Shift once to consume the option flag
            shift
            shift
            ;;
        -m|--model)
            MODEL_FILE="$2"
            # Shift twice to consume both the option flag and its value
            shift
            shift
            ;;
        -mg|--main-gpu)
            GGML_SYCL_DEVICE=$2
            SPLIT_MODE=none
            # Shift twice to consume both the option flag and its value
            shift
            shift
            ;;
        -sm|--split-mode)
            SPLIT_MODE=$2
            # Shift twice to consume both the option flag and its value
            shift
            shift
            ;;
        -ngl|--n-gpu-layers)
            NGL=$2
            # Shift twice to consume both the option flag and its value
            shift
            shift
            ;;
        -lv|--log-verbosity)
            LOG_VERBOSE=$2
            # Shift twice to consume both the option flag and its value
            shift
            shift
            ;;
        -h|--help)
            Help
            exit 0
            ;;
        *)
            # Handle unknown options or stop processing options
            echo "Invalid option: $1"
            # Optional: exit script or shift to treat remaining as positional args
            exit 1
            ;;
    esac
done



source /opt/intel/oneapi/setvars.sh

#export GGML_SYCL_DEBUG=1

#ZES_ENABLE_SYSMAN=1, Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory. Recommended to use when --split-mode = layer.

#support malloc device memory more than 4GB.
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
echo "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=${UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS}"

if [ $GGML_SYCL_DEVICE -ne -1 ]; then
    echo "Use $GGML_SYCL_DEVICE as main GPU"
    #use signle GPU only
    GPUS_SETTING="-mg $GGML_SYCL_DEVICE -sm ${SPLIT_MODE}"
    export ONEAPI_DEVICE_SELECTOR="level_zero:${$GGML_SYCL_DEVICE}"
    echo "ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR}"
else
   echo "Use all Intel GPUs, including iGPU & dGPU"
 fi

echo "run cmd: ZES_ENABLE_SYSMAN=1 ${BIN_FILE} -m ${MODEL_FILE} -no-cnv -p "${INPUT_PROMPT}" -n 400 -e -ngl ${NGL} -s ${SEED} -c ${CONTEXT} ${GPUS_SETTING} -lv ${LOG_VERBOSE}  --mmap "
ZES_ENABLE_SYSMAN=1 ${BIN_FILE} -m ${MODEL_FILE} -no-cnv -p "${INPUT_PROMPT}" -n 400 -e -ngl ${NGL} -s ${SEED} -c ${CONTEXT} ${GPUS_SETTING} -lv ${LOG_VERBOSE} --mmap

