#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_regular_hadamard(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, int group_size);
