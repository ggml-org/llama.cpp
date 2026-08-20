#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-ops.h"
#include "htp-tensor.h"

// Redefined the rope type constants as we can't include ggml.h
#define HTP_ROPE_TYPE_NORMAL 0
#define HTP_ROPE_TYPE_NEOX   2
#define HTP_ROPE_TYPE_MROPE  8
#define HTP_ROPE_TYPE_VISION 24
#define HTP_ROPE_TYPE_IMROPE 40

#define HTP_ROPE_SPAD_NROWS  16
#define HTP_ROPE_SPAD_BLOCK  (HTP_ROPE_SPAD_NROWS/2)

#define htp_rope_preamble              \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

struct htp_rope_context {
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t sections[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    float theta_scale;
    float corr_dims[2];

    uint32_t src0_nrows_per_thread;
    size_t spad_stride;

    struct htp_ops_context * octx;

    size_t src0_row_size;
    size_t src0_row_stride;
    size_t dst_row_size;
    size_t dst_row_stride;
    size_t src0_row_size_aligned;
    size_t dst_row_size_aligned;
    size_t theta_cache_offset;
    uint32_t src0_nrows;

    struct fastdiv_values div_ne2_ne1;
    struct fastdiv_values div_ne1;

    uint64_t t_start;
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);

    return (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
static inline void rope_yarn_one(float theta, float freq_scale, float * corr_dims,
                                 uint32_t i0, float ext_factor, float mscale,
                                 float * cache) {
    float theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta_final  = theta_interp;
    float mscale_final = mscale;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    cache[i0 + 0] = cosf(theta_final) * mscale_final;
    cache[i0 + 1] = sinf(theta_final) * mscale_final;
}

static __attribute__((noinline)) void rope_cache_init(const float    theta_base,
                            const float    freq_scale,
                            const float *  freq_factors,
                            float *        corr_dims,
                            const uint32_t ne0,
                            const float    ext_factor,
                            const float    mscale,
                            float *        cache,
                            const float    theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
#if __HVX_ARCH__ >= 79
    const bool is_v79_or_newer = true;
#else
    const bool is_v79_or_newer = false;
#endif

    if (is_v79_or_newer && ext_factor == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        const uint32_t n_blocks = ne0 / 64;

        // Initialize theta scale powers: [1.0f, theta_scale, theta_scale^2, ..., theta_scale^31]
        float __attribute__((aligned(128))) theta_powers[32];
        theta_powers[0] = 1.0f;
        for (int j = 1; j < 32; j++) {
            theta_powers[j] = theta_powers[j - 1] * theta_scale;
        }
        HVX_Vector v_theta_powers = hvx_vmem(theta_powers);

        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale = hvx_vec_splat_f32(mscale);

        // Base theta starts at theta_base
        float theta_block = theta_base;
        // The scale factor for the next block is theta_scale^32
        float theta_scale_32 = 1.0f;
        for (int j = 0; j < 32; j++) {
            theta_scale_32 *= theta_scale;
        }

        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t i0 = b * 64;
            HVX_Vector v_theta_base = hvx_vec_splat_f32(theta_block);
            HVX_Vector v_theta = hvx_vec_mul_f32_f32(v_theta_base, v_theta_powers);

            if (freq_factors) {
                // Load 32 elements of freq_factors
                HVX_Vector v_ff = hvx_vmemu(freq_factors + i0 / 2);
                HVX_Vector v_inv_ff = hvx_vec_inverse_f32(v_ff);
                v_theta = hvx_vec_mul_f32_f32(v_theta, v_inv_ff);
            }

            HVX_Vector v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);

            HVX_Vector vcos = hvx_vec_cos_f32(v_theta_final);
            HVX_Vector vsin = hvx_vec_sin_f32(v_theta_final);

            vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
            vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

            HVX_VectorPair vstore = Q6_W_vshuff_VVR(vsin, vcos, -4);

            if (((uintptr_t)cache) % 128 == 0) {
                hvx_vmem(cache + i0 + 0)  = Q6_V_lo_W(vstore);
                hvx_vmem(cache + i0 + 32) = Q6_V_hi_W(vstore);
            } else {
                hvx_vec_store_u(cache + i0 + 0,  32 * sizeof(float), Q6_V_lo_W(vstore));
                hvx_vec_store_u(cache + i0 + 32, 32 * sizeof(float), Q6_V_hi_W(vstore));
            }

            theta_block *= theta_scale_32;
        }

        // Leftovers
        float theta = theta_block;
        for (uint32_t i0 = n_blocks * 64; i0 < ne0; i0 += 2) {
            const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    } else {
        // Fallback to original scalar loop
        float theta = theta_base;
        for (uint32_t i0 = 0; i0 < ne0; i0 += 2) {
            const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    }
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
static __attribute__((noinline)) void mrope_cache_init(const float    pos_t,
                             const float    pos_h,
                             const float    pos_w,
                             const float    pos_e,
                             const int32_t  sections[4],
                             const bool     is_imrope,
                             const bool     indep_sects,
                             const float    freq_scale,
                             const float *  freq_factors,
                             float *        corr_dims,
                             const uint32_t ne0,
                             const float    ext_factor,
                             const float    mscale,
                             float *        cache,
                             const float    theta_scale) {
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const int sec_w     = sections[0] + sections[1];
    const int sec_e     = sec_w + sections[2];

    float theta_t = pos_t;
    float theta_h = pos_h;
    float theta_w = pos_w;
    float theta_e = pos_e;

    for (uint32_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff     = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        const int   sector = (i0 / 2) % sect_dims;

        if (indep_sects) {
            // Reset theta when crossing into a new section.
            if      (sector == 0)           { theta_t = pos_t; }
            else if (sector == sections[0]) { theta_h = pos_h; }
            else if (sector == sec_w)       { theta_w = pos_w; }
            else if (sector == sec_e)       { theta_e = pos_e; }
        }

        float theta;
        if (is_imrope) {
            // Interleaved: sector mod 3 selects component
            if      (sector % 3 == 0 && sector < 3 * sections[0]) { theta = theta_t; }
            else if (sector % 3 == 1 && sector < 3 * sections[1]) { theta = theta_h; }
            else if (sector % 3 == 2 && sector < 3 * sections[2]) { theta = theta_w; }
            else                                                   { theta = theta_e; }
        } else {
            // Contiguous sections
            if      (sector < sections[0]) { theta = theta_t; }
            else if (sector < sec_w)       { theta = theta_h; }
            else if (sector < sec_e)       { theta = theta_w; }
            else                           { theta = theta_e; }
        }

        rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }
}

#define M_PI 3.1415926535897932384626433

static void rope_corr_dims(int     n_dims,
                           int     n_ctx_orig,
                           float   freq_base,
                           float   beta_fast,
                           float   beta_slow,
                           float * dims) {
    float start = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (float) M_PI)) / (2 * logf(freq_base)));
    float end   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (float) M_PI)) / (2 * logf(freq_base)));
    dims[0]     = MAX(0, start);
    dims[1]     = MIN(n_dims - 1, end);
}

static inline void hvx_rope_neox_f32_aa(float * restrict dst, const float * restrict src0, uint32_t ne, const float * restrict theta_cache) {
    const uint32_t he = ne / 2;
    const uint32_t nvec = he / 32;
    const uint32_t nloe = he % 32;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v0 = ((const HVX_Vector *) src0)[i];
        HVX_Vector v1 = hvx_vmemu(src0 + he + i * 32);

        HVX_Vector v2 = ((const HVX_Vector *) theta_cache)[i * 2 + 0];
        HVX_Vector v3 = ((const HVX_Vector *) theta_cache)[i * 2 + 1];

        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        ((HVX_Vector *) dst)[i] = Q6_Vsf_equals_Vqf32(v4);
        hvx_vmemu(dst + he + i * 32) = Q6_Vsf_equals_Vqf32(v5);
    }

    if (nloe > 0) {
        HVX_Vector v0 = hvx_vmemu(src0 + nvec * 32);
        HVX_Vector v1 = hvx_vmemu(src0 + he + nvec * 32);

        HVX_Vector v2 = ((const HVX_Vector *) theta_cache)[nvec * 2 + 0];
        HVX_Vector v3 = ((const HVX_Vector *) theta_cache)[nvec * 2 + 1];

        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        hvx_vec_store_u(dst + nvec * 32, nloe * sizeof(float), Q6_Vsf_equals_Vqf32(v4));
        hvx_vec_store_u(dst + he + nvec * 32, nloe * sizeof(float), Q6_Vsf_equals_Vqf32(v5));
    }
}

static inline void hvx_rope_f32_aa(float * restrict dst, const float * restrict src0, uint32_t ne, const float * restrict theta_cache) {
    const uint32_t nvec = ne / 64;
    const uint32_t nloe = ne % 64;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v0 = ((const HVX_Vector *) src0)[i * 2 + 0];
        HVX_Vector v1 = ((const HVX_Vector *) src0)[i * 2 + 1];

        HVX_Vector v2 = ((const HVX_Vector *) theta_cache)[i * 2 + 0];
        HVX_Vector v3 = ((const HVX_Vector *) theta_cache)[i * 2 + 1];

        HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_Vector *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_Vector *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    if (nloe > 0) {
        if (nloe <= 32) {
            HVX_Vector v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_Vector v2 = hvx_vmemu(theta_cache + nvec * 64);

            HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);
            HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(Q6_V_vzero(), v2, -4);

            HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
            HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
            HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
            HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));

            HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * sizeof(float), Q6_V_lo_W(vstore));
        } else {
            HVX_Vector v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_Vector v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_Vector v2 = hvx_vmemu(theta_cache + nvec * 64);
            HVX_Vector v3 = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(v1, v0, -4);
            HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);

            HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
            HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
            HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
            HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));

            HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_Vector *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * sizeof(float), Q6_V_hi_W(vstore));
        }
    }
}

static void inline rope_basic_f32(struct htp_rope_context * rctx, uint8_t * restrict dst, uint8_t * restrict src,
                   uint32_t nr, uint32_t ne0, const float * restrict theta_cache) {
    #pragma unroll(4)
    for (uint32_t i = 0; i < nr; i++) {
        float * d = (float *) (dst + i * rctx->dst_row_size_aligned);
        float * s = (float *) (src + i * rctx->src0_row_size_aligned);

        hvx_rope_f32_aa(d, s, rctx->n_dims, theta_cache);

        // fill the remain channels with data from src tensor
        if (rctx->n_dims < ne0) {
            hvx_copy_f32_uu((uint8_t *)(d + rctx->n_dims), (uint8_t *)(s + rctx->n_dims), ne0 - rctx->n_dims);
        }
    }
}

static void inline rope_neox_f32(struct htp_rope_context * rctx, uint8_t * restrict dst, uint8_t * restrict src,
                   uint32_t nr, uint32_t ne0, const float * restrict theta_cache) {
    #pragma unroll(4)
    for (uint32_t i = 0; i < nr; i++) {
        float * d = (float *) (dst + i * rctx->dst_row_size_aligned);
        float * s = (float *) (src + i * rctx->src0_row_size_aligned);

        hvx_rope_neox_f32_aa(d, s, rctx->n_dims, theta_cache);

        // fill the remain channels with data from src tensor
        if (rctx->n_dims < ne0) {
            hvx_copy_f32_uu((uint8_t *)(d + rctx->n_dims), (uint8_t *)(s + rctx->n_dims), ne0 - rctx->n_dims);
        }
    }
}

static void inline rope_vision_f32(struct htp_rope_context * rctx, uint8_t * restrict dst, uint8_t * restrict src,
                   uint32_t nr, uint32_t ne0, const float * restrict theta_cache) {
    #pragma unroll(4)
    for (uint32_t i = 0; i < nr; i++) {
        float * d = (float *) (dst + i * rctx->dst_row_size_aligned);
        float * s = (float *) (src + i * rctx->src0_row_size_aligned);

        hvx_rope_neox_f32_aa(d, s, ne0, theta_cache);
    }
}

static void rope_job_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_rope_context * rctx = (struct htp_rope_context *) data;
    struct htp_ops_context * octx = rctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * src2 = octx->src[2];
    const struct htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    const uint32_t src0_nrows = rctx->src0_nrows;
    const uint32_t src0_nrows_per_thread = rctx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t tt = HAP_perf_get_qtimer_count();

    const int32_t mode    = rctx->mode;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    const bool    is_neox = (mode & HTP_ROPE_TYPE_NEOX) || (mode & HTP_ROPE_TYPE_MROPE);
    const bool    is_vision = (mode == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    float *   theta_cache    = (float *) (src0_spad_base);
              src0_spad_base = src0_spad_base + rctx->theta_cache_offset;
    uint8_t * dst_spad_base  = octx->dst_spad.data + (ith * octx->dst_spad.size_per_thread);

    dma_queue * dma_queue = octx->ctx->dma[ith];
    const int32_t * pos = (const int32_t *) src1->data;
    const float * freq_factors = src2 ? (const float *) src2->data : NULL;

    const uint32_t i3_start = fastdiv(src0_start_row, &rctx->div_ne2_ne1);
    const uint32_t rem      = fastmodulo(src0_start_row, ne2 * ne1, &rctx->div_ne2_ne1);
    const uint32_t i2_start = fastdiv(rem, &rctx->div_ne1);
    const uint32_t i1_start = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_start_row;
    uint32_t prev_i2 = (uint32_t) -1;

    for (uint32_t i3 = i3_start; i3 < ne3; i3++) { // batch
        const uint32_t i2_init = (i3 == i3_start) ? i2_start : 0;
        for (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            const uint32_t i1_init = (i3 == i3_start && i2 == i2_start) ? i1_start : 0;
            for (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                if (ir >= src0_end_row) goto done;

                // Rows in this block
                const uint32_t nrows = MIN(src0_end_row - ir, ne1 - i1);

                // Depth before prefetch
                uint32_t dma_depth = dma_queue_depth(dma_queue);

                // FARF(HIGH, "rope-block %u: ir %u n-rows %u dma-depth %u : usec %u", ith, ir, nrows, dma_depth,
                //             (unsigned) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - rctx->t_start));

                // Prefetch loop
                for (uint32_t pnr = 0, pr = 0; pr < nrows && pr < HTP_ROPE_SPAD_NROWS; pr += pnr) {
                    pnr = MIN(nrows - pr, HTP_ROPE_SPAD_BLOCK);

                    uint32_t pi1 = i1 + pr;
                    uint32_t pir = ir + pr;

                    // Dummy DMA transaction for sequencing (interleaving dst,src,dst,...)
                    dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr((void *) dst->data, dst_spad_base + pr * rctx->dst_row_size_aligned), 0, 0, 0);

                    const uint8_t * src_addr = (const uint8_t *) src0->data + i3 * nb03 + i2 * nb02 + pi1 * nb01;
                          uint8_t * src_spad = src0_spad_base + pr * rctx->src0_row_size_aligned;

                    // Copy only the row payload while striding the DDR source
                    dma_queue_push(dma_queue, dma_make_ptr(src_spad, src_addr),
                        rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);

                    // FARF(HIGH, "rope-prefetch %u: pr %u i1 %u i2 %u i3 %u src-spad %p src-addr %p pnr %u", ith, pir, pi1, i2, i3, src_spad, src_addr, pnr);
                }

                // Update theta cache
                if (i2 != prev_i2) {
                    prev_i2 = i2;

                    const bool is_mrope = (rctx->mode & HTP_ROPE_TYPE_MROPE) != 0;
                    if (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        const bool is_imrope = (rctx->mode == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (float) pos[i2],
                            (float) pos[i2 + ne2],
                            (float) pos[i2 + ne2 * 2],
                            (float) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            ne0, rctx->ext_factor, rctx->attn_factor,
                            theta_cache, rctx->theta_scale);
                    } else {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        ne0, rctx->ext_factor, rctx->attn_factor,
                                        theta_cache, rctx->theta_scale);
                    }
                }

                // Skip output DMA transactions from prev block (if any)
                // No need to wait for those here since we're explicitly waiting for the latest prefecthes below.
                for (uint32_t d=0; d < dma_depth; d++) { dma_queue_pop_nowait(dma_queue); }

                // Compute loop
                for (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr, ir += cnr, i1 += cnr) {
                    // Number of rows to compute
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_BLOCK);

                    uint8_t * dst_spad = (uint8_t *) dma_queue_pop(dma_queue).src;
                    uint8_t * src_spad = (uint8_t *) dma_queue_pop(dma_queue).dst;

                    // FARF(HIGH, "rope-compute %u: ir %u i1 %u i2 %u i3 %u src-spad %p cnr %u : usec %u", ith, ir, i1, i2, i3, src_spad, cnr,
                    //         (unsigned) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - rctx->t_start));

                    if (is_vision) {
                        rope_vision_f32(rctx, dst_spad, src_spad, cnr, ne0, theta_cache);
                    } else if (is_neox) {
                        rope_neox_f32(rctx, dst_spad, src_spad, cnr, ne0, theta_cache);
                    } else {
                        rope_basic_f32(rctx, dst_spad, src_spad, cnr, ne0, theta_cache);
                    }

                    uint8_t * dst_addr = (uint8_t *) dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1;

                    // Write only the row payload while striding the DDR dst
                    dma_queue_push(dma_queue, dma_make_ptr(dst_addr, dst_spad),
                        rctx->dst_row_stride, rctx->dst_row_size_aligned, rctx->dst_row_size, cnr);

                    // Prefetch more rows (if any)
                    if ((cr + HTP_ROPE_SPAD_NROWS) < nrows) {
                        uint32_t pnr = MIN(nrows - (cr + HTP_ROPE_SPAD_NROWS), HTP_ROPE_SPAD_BLOCK);
                        uint32_t pi1 = i1 + HTP_ROPE_SPAD_NROWS;
                        uint32_t pir = ir + HTP_ROPE_SPAD_NROWS;

                        const uint8_t * src_addr = (const uint8_t *) src0->data + i3 * nb03 + i2 * nb02 + pi1 * nb01;
                        dma_queue_push(dma_queue, dma_make_ptr(src_spad, src_addr),
                            rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);

                        // FARF(HIGH, "rope-prefetch %u: pr %u i1 %u i2 %u i3 %u src-spad %p src-addr %p pnr %u", ith, pir, pi1, i2, i3, src_spad, src_addr, pnr);
                    }
                }
            }
        }
    }

done:
    dma_queue_flush(dma_queue);
    tt = HAP_perf_get_qtimer_count() - tt;

    FARF(HIGH, "rope-f32: %d/%d: (%u:%u) usec %u\n", ith, nth, src0_start_row, src0_end_row, (unsigned) HAP_perf_qtimer_count_to_us(tt));
}

static int execute_op_rope_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * src2 = octx->src[2];
    const struct htp_tensor * dst  = octx->dst;

    const char * op_type = "rope-f32";

    switch (octx->op) {
        case HTP_OP_ROPE:
            break;

        default:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t ne0 = dst->ne[0];
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads = MIN(octx->n_threads, src0_nrows);

    const size_t src0_row_size   = src0->ne[0] * sizeof(float);
    const size_t src0_row_stride = src0->nb[1];
    const size_t dst_row_size    = dst->ne[0] * sizeof(float);
    const size_t dst_row_stride  = dst->nb[1];

    // Aligned row sizes for VTCM
    const size_t src0_row_size_aligned    = hex_round_up(src0_row_size, VLEN);
    const size_t dst_row_size_aligned     = hex_round_up(dst_row_stride, VLEN);
    const size_t theta_cache_size_aligned = hex_round_up(src0->ne[0] * sizeof(float), 256);

    // Calculate spad sizes per thread
    size_t src0_spad_per_thread = theta_cache_size_aligned + HTP_ROPE_SPAD_NROWS * src0_row_size_aligned;
    size_t dst_spad_per_thread  = HTP_ROPE_SPAD_NROWS * dst_row_size_aligned;
    size_t spad_per_thread = src0_spad_per_thread + dst_spad_per_thread;

    // Check if we fit in VTCM
    size_t total_vtcm_needed = spad_per_thread * n_threads;
    if (octx->ctx->vtcm_size < total_vtcm_needed) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, total_vtcm_needed);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.size_per_thread = src0_spad_per_thread;
    octx->dst_spad.size_per_thread  = dst_spad_per_thread;
    octx->src0_spad.size = n_threads * src0_spad_per_thread;
    octx->dst_spad.size  = n_threads * dst_spad_per_thread;
    octx->src1_spad.size = 0;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = NULL;
    octx->src1_spad.data = NULL;                                        octx->src1_spad.src = NULL;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = NULL;

    struct htp_rope_context rctx;
    memset(&rctx, 0, sizeof(struct htp_rope_context));

    rctx.t_start = HAP_perf_get_qtimer_count();

    rctx.octx = octx;

    const int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((const int32_t *) op_params)[1];
    rctx.mode       = ((const int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((const int32_t *) op_params)[4];

    memcpy(&rctx.freq_base,   (int32_t *) op_params + 5,  sizeof(float));
    memcpy(&rctx.freq_scale,  (int32_t *) op_params + 6,  sizeof(float));
    memcpy(&rctx.ext_factor,  (int32_t *) op_params + 7,  sizeof(float));
    memcpy(&rctx.attn_factor, (int32_t *) op_params + 8,  sizeof(float));
    memcpy(&rctx.beta_fast,   (int32_t *) op_params + 9,  sizeof(float));
    memcpy(&rctx.beta_slow,   (int32_t *) op_params + 10, sizeof(float));
    memcpy(&rctx.sections,    (int32_t *) op_params + 11, sizeof(int) * 4);

    rctx.theta_scale = powf(rctx.freq_base, -2.0f / rctx.n_dims);

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_base, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_row_size   = src0_row_size;
    rctx.src0_row_stride = src0_row_stride;
    rctx.dst_row_size    = dst_row_size;
    rctx.dst_row_stride  = dst_row_stride;
    rctx.src0_row_size_aligned = src0_row_size_aligned;
    rctx.dst_row_size_aligned  = dst_row_size_aligned;
    rctx.theta_cache_offset    = theta_cache_size_aligned;

    rctx.src0_nrows = src0_nrows;
    rctx.src0_nrows_per_thread = (src0_nrows + n_threads - 1) / n_threads;

    if (src0_nrows > 0) {
        rctx.div_ne2_ne1 = init_fastdiv_values(dst->ne[2] * dst->ne[1]);
        rctx.div_ne1     = init_fastdiv_values(dst->ne[1]);
    }

    FARF(HIGH, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_factor, rctx.theta_scale, rctx.attn_factor);

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        worker_pool_run_func(octx->ctx->worker_pool, rope_job_f32, &rctx, n_threads);
    }

    return err;
}

int op_rope(struct htp_ops_context * octx) {
    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            return execute_op_rope_f32(octx);

        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
