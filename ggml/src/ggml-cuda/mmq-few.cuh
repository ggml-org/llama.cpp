#pragma once

#include "common.cuh"

// ids != nullptr: MUL_MAT_ID. dst is the MUL_MAT or MUL_MAT_ID node, it holds the src1 precision requested by the graph.
bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc);

// fusion: only the scale of an NVFP4 src0 (x_scale) and the requested src1 precision of the MUL_MAT, dst is the output of the MUL by the scale.
void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion = nullptr);
