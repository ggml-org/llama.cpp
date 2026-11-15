#pragma once

#include "ggml-cuda/common.cuh"

// rocBLAS' Tensile selection heuristic does not reliably return the fastest
// solution for a given shape. Measure the alternatives once per shape, cache the
// winner across runs, and replay it. cc is the compute capability of the device
// the handle belongs to, solution indices are not portable across architectures.
//
// Returns true if the GEMM was issued here, false if the caller still has to
// issue its own cublasGemmEx.
bool ggml_hip_gemm_autotune(
        cublasHandle_t handle, int cc,
        int64_t m, int64_t n, int64_t k,
        const void * alpha, const void * a, int64_t lda,
        const void * b, int64_t ldb,
        const void * beta, void * d, int64_t ldd,
        cudaDataType_t type_a, cudaDataType_t type_b, cudaDataType_t type_d,
        cublasComputeType_t compute_type);
