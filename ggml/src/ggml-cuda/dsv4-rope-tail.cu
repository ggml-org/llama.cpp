#include "dsv4-rope-tail.cuh"

#include "ggml.h"      // ggml_rope_yarn_corr_dims, ggml_get_op_params_i32

#include <algorithm>   // std::min / std::max in dispatch
#include <cstring>     // memcpy

// YaRN helper. Direct port of ggml/src/ggml-cuda/rope.cu:22-41
// (template<bool forward> rope_yarn). Duplicated here to keep this
// translation unit self-contained — rope.cuh does not currently expose
// the function as a reusable device helper. The math is identical.

struct dsv4_rope_corr_dims {
    float v[2];
};

static __device__ __forceinline__ float dsv4_rope_yarn_ramp(
        const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// forward=true: standard rotation; forward=false: inverse (sin flipped).
template<bool forward>
static __device__ __forceinline__ void dsv4_rope_yarn(
        const float theta_extrap, const float freq_scale,
        const dsv4_rope_corr_dims corr_dims, const int i0,
        const float ext_factor, float mscale,
        float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = dsv4_rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta = -sin_theta;
    }
}

// Main kernel. Launch shape matches Metal:
//   grid  = (ne01, ne02, ne03)
//   block = (min(256, ne00), 1, 1)
// Each thread walks the ne00 dim with stride ntg (== blockDim.x).
// Translation of kernel_dsv4_rope_tail_f32 at
// ggml/src/ggml-metal/ggml-metal.metal:4906-4997.
static __global__ void dsv4_rope_tail_f32_kernel(
        const float * __restrict__ src0,
        const int   * __restrict__ pos,
        const float * __restrict__ freq_factors,
        float       * __restrict__ dst,
        const int    ne00,
        const int    nb00, const int nb01, const int nb02, const int nb03,
        const int    nb0,  const int nb1,  const int nb2,  const int nb3,
        const int    n_dims,
        const float  freq_base, const float freq_scale,
        const float  ext_factor, const float attn_factor,
        const dsv4_rope_corr_dims corr_dims,
        const bool   is_neox, const bool inverse) {
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;
    const int tid = threadIdx.x;
    const int ntg = blockDim.x;

    const int n_nope = ne00 - n_dims;
    if (n_nope < 0) {
        return;
    }

    const float theta_base_pos = (float) pos[i2];
    const float inv_ndims = -1.0f / (float) n_dims;

    const char * src_base = (const char *) src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    char       * dst_base = (char *)       dst  + i3 * nb3  + i2 * nb2  + i1 * nb1;

    for (int i0 = tid; i0 < ne00; i0 += ntg) {
        // Pass-through prefix: non-RoPE portion of the row.
        if (i0 < n_nope) {
            *((float *) (dst_base + i0 * nb0)) = *((const float *) (src_base + i0 * nb00));
            continue;
        }

        const int r = i0 - n_nope;

        if (is_neox) {
            const int n_half = n_dims / 2;
            if (r >= n_half) {
                continue;
            }

            const int ic = r;
            const int rel_i0 = 2 * ic;
            const float theta = theta_base_pos * powf(freq_base, inv_ndims * (float) rel_i0);
            const float freq_factor = freq_factors ? freq_factors[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            // Use forward=true; inverse handled as a sign flip below to match
            // Metal's "if (args.inverse) sin_theta = -sin_theta" pattern.
            dsv4_rope_yarn<true>(theta / freq_factor, freq_scale, corr_dims,
                                 rel_i0, ext_factor, attn_factor,
                                 cos_theta, sin_theta);
            if (inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *((const float *) (src_base + j0 * nb00));
            const float x1 = *((const float *) (src_base + j1 * nb00));
            *((float *) (dst_base + j0 * nb0)) = x0 * cos_theta - x1 * sin_theta;
            *((float *) (dst_base + j1 * nb0)) = x0 * sin_theta + x1 * cos_theta;
        } else {
            // NORMAL mode: rotate adjacent pair (j0, j0+1).
            if ((r & 1) != 0) {
                continue;
            }

            const int ic = r / 2;
            const float theta = theta_base_pos * powf(freq_base, inv_ndims * (float) r);
            const float freq_factor = freq_factors ? freq_factors[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            dsv4_rope_yarn<true>(theta / freq_factor, freq_scale, corr_dims,
                                 r, ext_factor, attn_factor,
                                 cos_theta, sin_theta);
            if (inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *((const float *) (src_base + j0 * nb00));
            const float x1 = *((const float *) (src_base + j1 * nb00));
            *((float *) (dst_base + j0 * nb0)) = x0 * cos_theta - x1 * sin_theta;
            *((float *) (dst_base + j1 * nb0)) = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * pos  = dst->src[1];
    const ggml_tensor * ff   = dst->src[2];  // optional; may be NULL

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(pos->type  == GGML_TYPE_I32);

    // op_params layout — matches Metal dispatch at
    // ggml/src/ggml-metal/ggml-metal-ops.cpp:1606-1623 verbatim:
    //   [0] = n_dims      (i32)
    //   [1] = mode        (i32)
    //   [2] = n_ctx_orig  (i32)
    //   [3] = inverse     (i32, treated as bool)
    //   [4] = freq_base   (f32)
    //   [5] = freq_scale  (f32)
    //   [6] = ext_factor  (f32)
    //   [7] = attn_factor (f32)
    //   [8] = beta_fast   (f32)
    //   [9] = beta_slow   (f32)
    const int32_t n_dims     = ggml_get_op_params_i32(dst, 0);
    const int32_t mode       = ggml_get_op_params_i32(dst, 1);
    const int32_t n_ctx_orig = ggml_get_op_params_i32(dst, 2);
    const int32_t inverse_i  = ggml_get_op_params_i32(dst, 3);
    const bool    inverse    = inverse_i != 0;

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base,   (const int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) dst->op_params + 9, sizeof(float));

    const bool is_neox = (mode == GGML_ROPE_TYPE_NEOX);

    // Precompute YaRN corr_dims host-side (matches Metal call at
    // ggml/src/ggml-metal/ggml-metal.metal:4927).
    dsv4_rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const int ne00 = (int) src0->ne[0];
    const int ne01 = (int) src0->ne[1];
    const int ne02 = (int) src0->ne[2];
    const int ne03 = (int) src0->ne[3];

    GGML_ASSERT(ne01 > 0 && ne02 > 0 && ne03 > 0);

    const int nth = std::min(256, std::max(1, ne00));
    const dim3 grid(ne01, ne02, ne03);
    const dim3 block(nth, 1, 1);

    cudaStream_t stream = ctx.stream();
    dsv4_rope_tail_f32_kernel<<<grid, block, 0, stream>>>(
        (const float *) src0->data,
        (const int *)   pos->data,
        ff ? (const float *) ff->data : nullptr,
        (float *)       dst->data,
        ne00,
        (int) src0->nb[0], (int) src0->nb[1], (int) src0->nb[2], (int) src0->nb[3],
        (int) dst->nb[0],  (int) dst->nb[1],  (int) dst->nb[2],  (int) dst->nb[3],
        n_dims,
        freq_base, freq_scale, ext_factor, attn_factor,
        corr_dims,
        is_neox, inverse);

    CUDA_CHECK(cudaGetLastError());
}
