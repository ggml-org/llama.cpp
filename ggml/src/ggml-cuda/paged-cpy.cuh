#pragma once

#include "common.cuh"

// CUDA backend function for GGML_OP_PAGED_CPY
void ggml_cuda_op_paged_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
