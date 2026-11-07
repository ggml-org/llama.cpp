#pragma once

#include "repack-cutlass-blockscaled.cuh"

bool ggml_cuda_cutlass_compiled();

bool ggml_cuda_cutlass_mul_mat_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

bool ggml_cuda_cutlass_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
