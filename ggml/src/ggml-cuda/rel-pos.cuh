#pragma once
#include "common.cuh"

#define CUDA_GET_REL_POS_BLOCK_SIZE   256

void ggml_cuda_op_get_rel_pos(ggml_backend_cuda_context & ctx, ggml_tensor * dst);