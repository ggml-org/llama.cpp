#include "common.cuh"

#define CUDA_WKV_BLOCK_SIZE 64

void ggml_cuda_op_rwkv_wkv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
