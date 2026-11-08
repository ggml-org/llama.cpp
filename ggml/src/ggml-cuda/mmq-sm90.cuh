#pragma once

#include "common.cuh"

#ifdef GGML_CUDA_MMQ_SM90
bool ggml_cuda_should_use_mmq_q6_k_sm90(
        int device_id,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst);

bool ggml_cuda_mul_mat_q6_k_sm90(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
#endif
