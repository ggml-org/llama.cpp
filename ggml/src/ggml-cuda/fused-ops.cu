#include "fused-ops.cuh"

#include <cstdint>
#include <limits>

static __device__ __forceinline__ size_t ggml_cuda_index_4d(
        const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t s0,  const int64_t s1,  const int64_t s2,  const int64_t s3) {
    const int64_t j0 = i0 % ne0;
    const int64_t j1 = i1 % ne1;
    const int64_t j2 = i2 % ne2;
    const int64_t j3 = i3 % ne3;

    return size_t(j0)*s0 + size_t(j1)*s1 + size_t(j2)*s2 + size_t(j3)*s3;
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

static __global__ void k_lerp_fused_rwv_contig_f32(
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

static __global__ void k_mul_sub_add_fused_f32(
        const float * __restrict__ base,
        const float * __restrict__ scale,
        const float * __restrict__ value,
        float * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t b_ne0, const int64_t b_ne1, const int64_t b_ne2, const int64_t b_ne3,
        const int64_t b_s0,  const int64_t b_s1,  const int64_t b_s2,  const int64_t b_s3,
        const int64_t s_ne0, const int64_t s_ne1, const int64_t s_ne2, const int64_t s_ne3,
        const int64_t s_s0,  const int64_t s_s1,  const int64_t s_s2,  const int64_t s_s3,
        const int64_t v_ne0, const int64_t v_ne1, const int64_t v_ne2, const int64_t v_ne3,
        const int64_t v_s0,  const int64_t v_s1,  const int64_t v_s2,  const int64_t v_s3) {
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

    const size_t ib = ggml_cuda_index_4d(i0, i1, i2, i3, b_ne0, b_ne1, b_ne2, b_ne3, b_s0, b_s1, b_s2, b_s3);
    const size_t is = ggml_cuda_index_4d(i0, i1, i2, i3, s_ne0, s_ne1, s_ne2, s_ne3, s_s0, s_s1, s_s2, s_s3);
    const size_t iv = ggml_cuda_index_4d(i0, i1, i2, i3, v_ne0, v_ne1, v_ne2, v_ne3, v_s0, v_s1, v_s2, v_s3);

    const float v = value[iv];
    dst[idx] = base[ib] + (scale[is] * v - v);
}

static __global__ void k_add_mul_fused_f32(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
        const float * __restrict__ scale,
        float * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t s0_ne0, const int64_t s0_ne1, const int64_t s0_ne2, const int64_t s0_ne3,
        const int64_t s0_s0,  const int64_t s0_s1,  const int64_t s0_s2,  const int64_t s0_s3,
        const int64_t s1_ne0, const int64_t s1_ne1, const int64_t s1_ne2, const int64_t s1_ne3,
        const int64_t s1_s0,  const int64_t s1_s1,  const int64_t s1_s2,  const int64_t s1_s3,
        const int64_t sc_ne0, const int64_t sc_ne1, const int64_t sc_ne2, const int64_t sc_ne3,
        const int64_t sc_s0,  const int64_t sc_s1,  const int64_t sc_s2,  const int64_t sc_s3) {
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

    const size_t is0 = ggml_cuda_index_4d(i0, i1, i2, i3, s0_ne0, s0_ne1, s0_ne2, s0_ne3, s0_s0, s0_s1, s0_s2, s0_s3);
    const size_t is1 = ggml_cuda_index_4d(i0, i1, i2, i3, s1_ne0, s1_ne1, s1_ne2, s1_ne3, s1_s0, s1_s1, s1_s2, s1_s3);
    const size_t isc = ggml_cuda_index_4d(i0, i1, i2, i3, sc_ne0, sc_ne1, sc_ne2, sc_ne3, sc_s0, sc_s1, sc_s2, sc_s3);

    dst[idx] = (src0[is0] + src1[is1]) * scale[isc];
}

template <int head_size>
static __global__ void k_rwkv_rk_fused_f32(
        const float * __restrict__ cur,
        const float * __restrict__ k,
        const float * __restrict__ r,
        const float * __restrict__ v,
        const float * __restrict__ r_k,
        float * __restrict__ dst,
        const int64_t C,
        const int64_t H) {
    const int64_t tid = threadIdx.x;
    const int64_t h   = blockIdx.x % H;
    const int64_t t   = blockIdx.x / H;
    const int64_t off = t*C + h*head_size;

    __shared__ float rk[head_size];
    rk[tid] = k[off + tid] * r[off + tid] * r_k[h*head_size + tid];
    __syncthreads();

    for (int stride = head_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rk[tid] += rk[tid + stride];
        }
        __syncthreads();
    }

    dst[off + tid] = cur[off + tid] + v[off + tid] * rk[0];
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

    const int64_t total = ggml_nelements(dst);
    const int     block = 256;
    const int64_t grid  = (total + block - 1) / block;

    GGML_ASSERT(grid <= std::numeric_limits<uint32_t>::max());

    const int64_t base_total = ggml_nelements(x_prev);
    const int64_t n_mix      = dst->ne[3];

    if (ggml_are_same_shape(x_prev, cur) &&
            dst->ne[0] == x_prev->ne[0] &&
            dst->ne[1] == x_prev->ne[1] &&
            dst->ne[2] == x_prev->ne[2] &&
            total == base_total * n_mix &&
            weight->ne[0] == dst->ne[0] &&
            weight->ne[1] == 1 &&
            weight->ne[2] == 1 &&
            weight->ne[3] == n_mix &&
            ggml_nelements(weight) == dst->ne[0] * n_mix) {
        const int64_t base_grid = (base_total + block - 1) / block;
        GGML_ASSERT(base_grid <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(n_mix      <= std::numeric_limits<uint32_t>::max());

        const ggml_cuda_kernel_launch_params launch_params(dim3((uint32_t) base_grid, (uint32_t) n_mix), block, 0, ctx.stream());
        ggml_cuda_kernel_launch(k_lerp_fused_rwv_contig_f32, launch_params,
                (const float *) x_prev->data,
                (const float *) cur->data,
                (const float *) weight->data,
                (float *) dst->data,
                dst->ne[0],
                base_total);
        return;
    }

    auto stride = [](const ggml_tensor * t, int dim) {
        return int64_t(t->nb[dim] / ggml_element_size(t));
    };

    const ggml_cuda_kernel_launch_params launch_params(dim3((uint32_t) grid), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_lerp_fused_f32, launch_params,
            (const float *) x_prev->data,
            (const float *) cur->data,
            (const float *) weight->data,
            (float *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            x_prev->ne[0], x_prev->ne[1], x_prev->ne[2], x_prev->ne[3],
            stride(x_prev, 0), stride(x_prev, 1), stride(x_prev, 2), stride(x_prev, 3),
            cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3],
            stride(cur, 0), stride(cur, 1), stride(cur, 2), stride(cur, 3),
            weight->ne[0], weight->ne[1], weight->ne[2], weight->ne[3],
            stride(weight, 0), stride(weight, 1), stride(weight, 2), stride(weight, 3));
}

void ggml_cuda_op_mul_sub_add_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * base,
        const ggml_tensor * scale,
        const ggml_tensor * value,
        ggml_tensor * dst) {
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(value->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    GGML_ASSERT(ggml_can_repeat(base, dst));
    GGML_ASSERT(ggml_can_repeat(scale, dst));
    GGML_ASSERT(ggml_can_repeat(value, dst));
    GGML_ASSERT(ggml_is_contiguous(base));
    GGML_ASSERT(ggml_is_contiguous(scale));
    GGML_ASSERT(ggml_is_contiguous(value));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t total = ggml_nelements(dst);
    const int     block = 256;
    const int64_t grid  = (total + block - 1) / block;

    GGML_ASSERT(grid <= std::numeric_limits<uint32_t>::max());

    auto stride = [](const ggml_tensor * t, int dim) {
        return int64_t(t->nb[dim] / ggml_element_size(t));
    };

    const ggml_cuda_kernel_launch_params launch_params(dim3((uint32_t) grid), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_mul_sub_add_fused_f32, launch_params,
            (const float *) base->data,
            (const float *) scale->data,
            (const float *) value->data,
            (float *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            base->ne[0], base->ne[1], base->ne[2], base->ne[3],
            stride(base, 0), stride(base, 1), stride(base, 2), stride(base, 3),
            scale->ne[0], scale->ne[1], scale->ne[2], scale->ne[3],
            stride(scale, 0), stride(scale, 1), stride(scale, 2), stride(scale, 3),
            value->ne[0], value->ne[1], value->ne[2], value->ne[3],
            stride(value, 0), stride(value, 1), stride(value, 2), stride(value, 3));
}

void ggml_cuda_op_add_mul_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * scale,
        ggml_tensor * dst) {
    GGML_ASSERT(src0->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->type  == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    GGML_ASSERT(ggml_can_repeat(src0, dst));
    GGML_ASSERT(ggml_can_repeat(src1, dst));
    GGML_ASSERT(ggml_can_repeat(scale, dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t total = ggml_nelements(dst);
    const int     block = 256;
    const int64_t grid  = (total + block - 1) / block;

    GGML_ASSERT(grid <= std::numeric_limits<uint32_t>::max());

    auto stride = [](const ggml_tensor * t, int dim) {
        return int64_t(t->nb[dim] / ggml_element_size(t));
    };

    const ggml_cuda_kernel_launch_params launch_params(dim3((uint32_t) grid), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_add_mul_fused_f32, launch_params,
            (const float *) src0->data,
            (const float *) src1->data,
            (const float *) scale->data,
            (float *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            stride(src0, 0), stride(src0, 1), stride(src0, 2), stride(src0, 3),
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            stride(src1, 0), stride(src1, 1), stride(src1, 2), stride(src1, 3),
            scale->ne[0], scale->ne[1], scale->ne[2], scale->ne[3],
            stride(scale, 0), stride(scale, 1), stride(scale, 2), stride(scale, 3));
}

void ggml_cuda_op_rwkv_rk_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur,
        const ggml_tensor * k,
        const ggml_tensor * r,
        const ggml_tensor * v,
        const ggml_tensor * r_k,
        ggml_tensor * dst) {
    GGML_ASSERT(cur->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type   == GGML_TYPE_F32);
    GGML_ASSERT(r->type   == GGML_TYPE_F32);
    GGML_ASSERT(v->type   == GGML_TYPE_F32);
    GGML_ASSERT(r_k->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(cur));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r_k));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t head_size = k->ne[0];
    const int64_t H         = k->ne[1];
    const int64_t T         = k->ne[2];
    const int64_t C         = head_size * H;

    GGML_ASSERT(head_size == 64 || head_size == 128);
    GGML_ASSERT(dst->ne[0] == C && dst->ne[1] == T);
    GGML_ASSERT(cur->ne[0] == C && cur->ne[1] == T);
    GGML_ASSERT(r_k->ne[0] == head_size && r_k->ne[1] == H);
    GGML_ASSERT(ggml_are_same_shape(k, r));
    GGML_ASSERT(ggml_are_same_shape(k, v));

    const int64_t grid = H * T;
    GGML_ASSERT(grid <= std::numeric_limits<uint32_t>::max());

    const cudaStream_t stream = ctx.stream();
    if (head_size == 64) {
        const ggml_cuda_kernel_launch_params launch_params((uint32_t) grid, 64, 0, stream);
        ggml_cuda_kernel_launch(k_rwkv_rk_fused_f32<64>, launch_params,
                (const float *) cur->data,
                (const float *) k->data,
                (const float *) r->data,
                (const float *) v->data,
                (const float *) r_k->data,
                (float *) dst->data,
                C, H);
    } else {
        const ggml_cuda_kernel_launch_params launch_params((uint32_t) grid, 128, 0, stream);
        ggml_cuda_kernel_launch(k_rwkv_rk_fused_f32<128>, launch_params,
                (const float *) cur->data,
                (const float *) k->data,
                (const float *) r->data,
                (const float *) v->data,
                (const float *) r_k->data,
                (float *) dst->data,
                C, H);
    }
}
