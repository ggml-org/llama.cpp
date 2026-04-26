#include "common.cuh"

#define CUDA_HC_WEIGHTED_SUM_BLOCK_SIZE 256

void ggml_cuda_op_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
