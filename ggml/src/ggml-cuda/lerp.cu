#include "lerp.cuh"

#include <cstdint>
#include <limits>

static __device__ __forceinline__ size_t ggml_cuda_index_4d(
        const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t s0,  const int64_t s1,  const int64_t s2,  const int64_t s3) {
    return size_t(i0 % ne0) * s0 + size_t(i1 % ne1) * s1 + size_t(i2 % ne2) * s2 + size_t(i3 % ne3) * s3;
}

static __global__ void k_lerp_fused_f32(
        const float * __restrict__ x_prev,
        const float * __restrict__ cur,
        const float * __restrict__ weight,
        float * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t xp_ne0, const int64_t xp_ne1, const int64_t xp_ne2, const int64_t xp_ne3,
        const int64_t xp_s0,  const int64_t xp_s1,  const int64_t xp_s2,  const int64_t xp_s3,
        const int64_t c_ne0,  const int64_t c_ne1,  const int64_t c_ne2,  const int64_t c_ne3,
        const int64_t c_s0,   const int64_t c_s1,   const int64_t c_s2,   const int64_t c_s3,
        const int64_t w_ne0,  const int64_t w_ne1,  const int64_t w_ne2,  const int64_t w_ne3,
        const int64_t w_s0,   const int64_t w_s1,   const int64_t w_s2,   const int64_t w_s3) {
    const int64_t idx   = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = ne0 * ne1 * ne2 * ne3;

    if (idx >= total) {
        return;
    }

    const int64_t i0 = idx % ne0;
    const int64_t t1 = idx / ne0;
    const int64_t i1 = t1 % ne1;
    const int64_t t2 = t1 / ne1;
    const int64_t i2 = t2 % ne2;
    const int64_t i3 = t2 / ne2;

    const size_t ixp = ggml_cuda_index_4d(i0, i1, i2, i3, xp_ne0, xp_ne1, xp_ne2, xp_ne3, xp_s0, xp_s1, xp_s2, xp_s3);
    const size_t ic  = ggml_cuda_index_4d(i0, i1, i2, i3, c_ne0,  c_ne1,  c_ne2,  c_ne3,  c_s0,  c_s1,  c_s2,  c_s3);
    const size_t iw  = ggml_cuda_index_4d(i0, i1, i2, i3, w_ne0,  w_ne1,  w_ne2,  w_ne3,  w_s0,  w_s1,  w_s2,  w_s3);

    const float c = cur[ic];
    dst[idx] = c + (x_prev[ixp] - c) * weight[iw];
}

static __global__ void k_lerp_fused_contig_f32(
        const float * __restrict__ x_prev,
        const float * __restrict__ cur,
        const float * __restrict__ weight,
        float * __restrict__ dst,
        const int64_t ne0,
        const int64_t base_total) {
    const int64_t ibase = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ibase >= base_total) {
        return;
    }

    const int64_t imix = blockIdx.y;
    const int64_t i0   = ibase % ne0;
    const float c = cur[ibase];
    dst[imix * base_total + ibase] = c + (x_prev[ibase] - c) * weight[imix * ne0 + i0];
}

void ggml_cuda_op_lerp_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x_prev,
        const ggml_tensor * cur,
        const ggml_tensor * weight,
        ggml_tensor * dst) {
    GGML_ASSERT(x_prev->type == GGML_TYPE_F32);
    GGML_ASSERT(cur->type    == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_can_repeat(x_prev, dst));
    GGML_ASSERT(ggml_can_repeat(cur, dst));
    GGML_ASSERT(ggml_can_repeat(weight, dst));
    GGML_ASSERT(ggml_is_contiguous(x_prev));
    GGML_ASSERT(ggml_is_contiguous(cur));
    GGML_ASSERT(ggml_is_contiguous(weight));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t total      = ggml_nelements(dst);
    const int64_t base_total = ggml_nelements(x_prev);
    const int64_t n_mix      = dst->ne[3];
    const int block = 256;

    if (ggml_are_same_shape(x_prev, cur) &&
            dst->ne[0] == x_prev->ne[0] && dst->ne[1] == x_prev->ne[1] && dst->ne[2] == x_prev->ne[2] &&
            total == base_total * n_mix &&
            weight->ne[0] == dst->ne[0] && weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == n_mix) {
        const int64_t grid = (base_total + block - 1) / block;
        GGML_ASSERT(grid <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(n_mix <= std::numeric_limits<uint32_t>::max());

        const ggml_cuda_kernel_launch_params launch_params(dim3((uint32_t) grid, (uint32_t) n_mix), block, 0, ctx.stream());
        ggml_cuda_kernel_launch(k_lerp_fused_contig_f32, launch_params,
                (const float *) x_prev->data, (const float *) cur->data, (const float *) weight->data,
                (float *) dst->data, dst->ne[0], base_total);
        return;
    }

    const int64_t grid = (total + block - 1) / block;
    GGML_ASSERT(grid <= std::numeric_limits<uint32_t>::max());

    auto stride = [](const ggml_tensor * t, int dim) {
        return int64_t(t->nb[dim] / ggml_element_size(t));
    };

    const ggml_cuda_kernel_launch_params launch_params(dim3((uint32_t) grid), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_lerp_fused_f32, launch_params,
            (const float *) x_prev->data, (const float *) cur->data, (const float *) weight->data, (float *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            x_prev->ne[0], x_prev->ne[1], x_prev->ne[2], x_prev->ne[3],
            stride(x_prev, 0), stride(x_prev, 1), stride(x_prev, 2), stride(x_prev, 3),
            cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3],
            stride(cur, 0), stride(cur, 1), stride(cur, 2), stride(cur, 3),
            weight->ne[0], weight->ne[1], weight->ne[2], weight->ne[3],
            stride(weight, 0), stride(weight, 1), stride(weight, 2), stride(weight, 3));
}
