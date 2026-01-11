#include "common.cuh"

#define CUDA_LERP_BLOCK_SIZE 256

void ggml_cuda_op_lerp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
