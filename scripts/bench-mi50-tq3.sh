#!/usr/bin/env bash

set -euo pipefail

export PATH=/opt/rocm/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCBLAS_LAYER=0
export ROCBLAS_LOG_LEVEL=0
export HIPBLASLT_LOG_LEVEL=0
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=8
export HSA_ENABLE_SDMA=0
export GPU_SINGLE_ALLOC_PERCENT=100
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library
export LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=0

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build-mi50}"
model_path="${MODEL_PATH:-/home/stefan/.lmstudio/models/mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF/Huihui-Qwen3.5-27B-abliterated.Q4_K_S.gguf}"

"$build_dir/bin/llama-bench" \
  -m "$model_path" \
  -ngl 999 \
  -fa 0 \
  -ctk f16,tq3_0 \
  -ctv f16,q8_0 \
  -p 512,4096,16384 \
  -n 128,512
