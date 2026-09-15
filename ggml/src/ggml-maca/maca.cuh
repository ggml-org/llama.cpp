#pragma once

#if defined(__has_include)
#  if !__has_include(<bridge/runtime/cuda_to_maca_mcr_adaptor.h>)
#    error "MACA cu-bridge headers were not found"
#  endif
#  if !__has_include(<bridge/blas/cublas_v2_wrapper.h>)
#    error "MACA cu-bridge mcBLAS wrapper was not found"
#  endif
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <common/__clang_maca_mckl_lib_header.h>

#include "maca-policy.cuh"

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED \
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif
