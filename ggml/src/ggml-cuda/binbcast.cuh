#include "common.cuh"


__device__ __forceinline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

__device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

__device__ __forceinline__ float op_sub(const float a, const float b) {
    return a - b;
}

__device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

__device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template<float (*op)(const float, const float)>
void ggml_cuda_op_fused_binbcast(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
