#include "common.cuh"

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx, ggml_tensor * logits, ggml_tensor * weights, ggml_tensor * top_k);
