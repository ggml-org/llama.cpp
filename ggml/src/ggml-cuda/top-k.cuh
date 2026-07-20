#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_vocab_top_k_device(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                  int32_t k, int32_t global_offset, uint64_t * packed);
