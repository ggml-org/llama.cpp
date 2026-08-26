#include "rwkv-fusion.cuh"

#include <cstdint>
#include <limits>

static __global__ void k_key_adjust_fused_f32(
        const float * __restrict__ k,
        const float * __restrict__ a,
        const float * __restrict__ k_a,
        float * __restrict__ dst,
        const int64_t ne0,
        const int64_t total) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const float ka = k[idx] * k_a[idx % ne0];
    dst[idx] = k[idx] + (a[idx] * ka - ka);
}

void ggml_cuda_op_key_adjust_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * k,
        const ggml_tensor * a,
        const ggml_tensor * k_a,
        ggml_tensor * dst) {
    GGML_ASSERT(k->type   == GGML_TYPE_F32);
    GGML_ASSERT(a->type   == GGML_TYPE_F32);
    GGML_ASSERT(k_a->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(k, a));
    GGML_ASSERT(ggml_are_same_shape(k, dst));
    GGML_ASSERT(k_a->ne[0] == dst->ne[0]);
    GGML_ASSERT(ggml_nelements(k_a) == dst->ne[0]);
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(k_a));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t total = ggml_nelements(dst);
    const int block = 256;
    const int64_t grid = (total + block - 1) / block;
    GGML_ASSERT(grid <= std::numeric_limits<uint32_t>::max());

    const ggml_cuda_kernel_launch_params launch_params(dim3((uint32_t) grid), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_key_adjust_fused_f32, launch_params,
            (const float *) k->data, (const float *) a->data, (const float *) k_a->data,
            (float *) dst->data, dst->ne[0], total);
}
