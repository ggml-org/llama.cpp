#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// MUL by a vector that is broadcast over the rows, then the hinted Hadamard MUL_MAT mm of the reshaped product
bool ggml_cuda_should_fuse_mul_fwht(const ggml_tensor * mul, const ggml_tensor * mm);

// Returns whether the fused transform could be used. If dst_q8_1 is not null, the result is also written there as q8_1 blocks.
bool ggml_cuda_op_mul_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, ggml_tensor * mm, block_q8_1 * dst_q8_1);
