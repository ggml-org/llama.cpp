#include "common.cuh"

#define CUDA_POOL1D_BLOCK_SIZE 256

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);