#include "common.cuh"

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions.
// supports_op and the dispatch must ask the same question: an op admitted by one
// but rejected by the other reaches the unconditional same-shape assert in
// ggml_cuda_op_fwht.
bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
