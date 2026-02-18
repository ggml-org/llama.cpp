#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

// Redefined the types GGML_ROPE_TYPE_NORMAL & GGML_ROPE_TYPE_NEOX as we cant include ggml.h
#define HTP_ROPE_TYPE_NORMAL 0
#define HTP_ROPE_TYPE_NEOX   2

#define HTP_ROPE_SPAD_NROWS  8

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
    struct fastdiv_values fastdiv_ne01;
    struct fastdiv_values fastdiv_ne02;
    size_t spad_stride;

    struct htp_ops_context * octx;

    size_t src0_row_size;
    size_t dst_row_size;
    size_t src0_row_size_aligned;
    size_t dst_row_size_aligned;
    size_t theta_cache_offset;
    uint32_t src0_nrows;
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);

    return (1 - MIN(1, MAX(0, y)));
}

static void rope_cache_init(const float    theta_base,
                            const float    freq_scale,
                            const float *  freq_factors,
                            float *        corr_dims,
                            const uint32_t ne0,
                            const float    ext_factor,
                            const float    mscale,
                            float *        cache,
                            const float    theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;

    for (uint32_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;

        float theta_extrap = theta / ff;

        // Get n-d rotational scaling corrected for extrapolation
        float theta_interp = freq_scale * theta_extrap;
        float theta_final  = theta_interp;
        float mscale_final = mscale;

        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
            theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

            // Get n-d magnitude scaling corrected for interpolation
            mscale_final *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }

        cache[i0 + 0] = cosf(theta_final) * mscale_final;
        cache[i0 + 1] = sinf(theta_final) * mscale_final;

        theta *= theta_scale;
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

static void init_rope_ctx(struct htp_rope_context * rctx, struct htp_ops_context * octx) {
    memset(rctx, 0, sizeof(struct htp_rope_context));

    const int32_t * op_params = &octx->op_params[0];

    rctx->n_dims     = ((const int32_t *) op_params)[1];
    rctx->mode       = ((const int32_t *) op_params)[2];
    rctx->n_ctx_orig = ((const int32_t *) op_params)[4];

    memcpy(&rctx->freq_base,   (int32_t *) op_params + 5,  sizeof(float));
    memcpy(&rctx->freq_scale,  (int32_t *) op_params + 6,  sizeof(float));
    memcpy(&rctx->ext_factor,  (int32_t *) op_params + 7,  sizeof(float));
    memcpy(&rctx->attn_factor, (int32_t *) op_params + 8,  sizeof(float));
    memcpy(&rctx->beta_fast,   (int32_t *) op_params + 9,  sizeof(float));
    memcpy(&rctx->beta_slow,   (int32_t *) op_params + 10, sizeof(float));
    memcpy(&rctx->sections,    (int32_t *) op_params + 11, sizeof(int) * 4);

    rctx->theta_scale = powf(rctx->freq_base, -2.0f / rctx->n_dims);

    rope_corr_dims(rctx->n_dims, rctx->n_ctx_orig, rctx->freq_base, rctx->beta_fast, rctx->beta_slow, rctx->corr_dims);

    rctx->octx = octx;

    // Initialize fastdiv values
    const uint32_t ne01 = octx->src0.ne[1];
    const uint32_t ne02 = octx->src0.ne[2];

    if (ne01 > 0) rctx->fastdiv_ne01 = init_fastdiv_values(ne01);
    if (ne02 > 0) rctx->fastdiv_ne02 = init_fastdiv_values(ne02);

    const uint32_t ne0 = octx->dst.ne[0];

    FARF(HIGH, "rope-f32 n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx->n_dims, ne0,
         rctx->ext_factor, rctx->theta_scale, rctx->attn_factor);
}

static void hvx_rope_neox_f32_aa(float * restrict dst, const float * restrict src0, uint32_t ne, const float * restrict theta_cache) {
    const uint8_t * restrict src0_curr  = (const uint8_t *) src0;
    const uint8_t * restrict theta_curr = (const uint8_t *) theta_cache;
    uint8_t * restrict dst_curr         = (uint8_t *) dst;

    int step_of_1 = ne >> 6;  // 6 because we process two vectors at once
    int half_size = (sizeof(float) * (ne / 2));
    const int32_t half_dims = ne / 2;

    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v0 = *(HVX_Vector *) src0_curr;
        HVX_Vector v1 = *(HVX_Vector *) (src0_curr + half_size);

        HVX_Vector v2 = *(HVX_Vector *) theta_curr;
        HVX_Vector v3 = *(HVX_Vector *) (theta_curr + VLEN);

        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);  // vcos_sin[0] = cos_theta, vcos_sin[1] = sin_theta

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        *(HVX_Vector *) dst_curr               = Q6_Vsf_equals_Vqf32(v4);
        *(HVX_Vector *) (dst_curr + half_size) = Q6_Vsf_equals_Vqf32(v5);

        src0_curr  += VLEN;
        theta_curr += 2 * VLEN;
        dst_curr   += VLEN;
    }

    int processed = (ne >> 6) << 6; // multiples of 64
    for (int i = processed; i < ne; i += 2) {
        const float cos_theta = theta_cache[i+0];
        const float sin_theta = theta_cache[i+1];
        float x0 = src0[i/2];
        float x1 = src0[i/2 + half_dims];
        dst[i/2]             = x0 * cos_theta - x1 * sin_theta;
        dst[i/2 + half_dims] = x0 * sin_theta + x1 * cos_theta;
    }
}

static void hvx_rope_f32_aa(float * restrict dst, const float * restrict src0, uint32_t ne, const float * restrict theta_cache) {
    const uint8_t * restrict src0_curr  = (const uint8_t *) src0;
    const uint8_t * restrict theta_curr = (const uint8_t *) theta_cache;
    uint8_t * restrict dst_curr         = (uint8_t *) dst;

    int step_of_1 = ne >> 6;  // 6 because we process two vectors at once

    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v0 = *(HVX_Vector *) src0_curr;
        HVX_Vector v1 = *(HVX_Vector *) (src0_curr + VLEN);

        HVX_Vector v2 = *(HVX_Vector *) theta_curr;
        HVX_Vector v3 = *(HVX_Vector *) (theta_curr + VLEN);

        HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(v1, v0, -4);  // vx0_x1[0] = x0, vx0_x1[1] = x1
        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);  // vcos_sin[0] = cos_theta, vcos_sin[1] = sin_theta

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        *(HVX_Vector *) dst_curr          = Q6_V_lo_W(vstore);
        *(HVX_Vector *) (dst_curr + VLEN) = Q6_V_hi_W(vstore);

        src0_curr  += 2 * VLEN;
        theta_curr += 2 * VLEN;
        dst_curr   += 2 * VLEN;
    }

    int processed = (ne >> 6) << 6;
    for (int i = processed; i < ne; i += 2) {
        const float cos_theta = theta_cache[i+0];
        const float sin_theta = theta_cache[i+1];
        float x0 = src0[i+0];
        float x1 = src0[i+1];
        dst[i+0] = x0 * cos_theta - x1 * sin_theta;
        dst[i+1] = x0 * sin_theta + x1 * cos_theta;
    }
}

struct rope_rowidx_cache {
    uint32_t i1;
    uint32_t i2;
    uint32_t i3;
    uint32_t pad;
};

static void rope_job_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_rope_context * rctx = (struct htp_rope_context *) data;
    struct htp_ops_context * octx = rctx->octx;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    const struct htp_tensor * src2 = &octx->src2;
    struct htp_tensor *       dst  = &octx->dst;

    htp_rope_preamble;

    const uint32_t src0_nrows = rctx->src0_nrows;
    const uint32_t src0_nrows_per_thread = rctx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const int32_t mode    = rctx->mode;
    const bool    is_neox = mode & HTP_ROPE_TYPE_NEOX;

    // VTCM setup
    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    float *   theta_cache    = (float *) (src0_spad_base);
              src0_spad_base = src0_spad_base + rctx->theta_cache_offset;
    uint8_t * dst_spad_base  = octx->dst_spad.data + (ith * octx->dst_spad.size_per_thread);

    dma_queue * dma_queue = octx->ctx->dma[ith];
    const int32_t * pos = (const int32_t *) src1->data;
    const float * freq_factors = src2->data ? (const float *) src2->data : NULL;

    struct rope_rowidx_cache rowidx_cache[HTP_ROPE_SPAD_NROWS];

    for (uint32_t ir = src0_start_row, is = 0; ir < src0_end_row && is < HTP_ROPE_SPAD_NROWS; ir++, is++) {
        // Dummy DMA transaction for sequencing (interleaving dst,src,dst,...)
        dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr((void *) dst->data, dst_spad_base + is * rctx->dst_row_size_aligned), 0, 0, 0);

        uint32_t i1 = fastmodulo(ir, ne01, &rctx->fastdiv_ne01);
        uint32_t r_div_ne01 = fastdiv(ir, &rctx->fastdiv_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, &rctx->fastdiv_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, &rctx->fastdiv_ne02);
        const uint8_t * src_addr = (const uint8_t *) src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01;

        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src0_spad_base + is * rctx->src0_row_size_aligned, src_addr),
            rctx->src0_row_size_aligned, rctx->src0_row_size, 1);

        rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i1 = i1;
        rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i2 = i2;
        rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i3 = i3;
    }

    uint32_t prev_i2 = (uint32_t) -1;
    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir++) {
        uint32_t i1 = rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i1;
        uint32_t i2 = rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i2;
        uint32_t i3 = rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i3;

        if (i2 != prev_i2) {
            const int32_t p = pos[i2];
            rope_cache_init(p, rctx->freq_scale, freq_factors, rctx->corr_dims, ne0, rctx->ext_factor, rctx->attn_factor, theta_cache, rctx->theta_scale);
            prev_i2 = i2;
        }

        float * dst_spad  = (float *) dma_queue_pop(dma_queue).src;
        float * src0_spad = (float *) dma_queue_pop(dma_queue).dst;

        if (is_neox) {
            hvx_rope_neox_f32_aa(dst_spad, src0_spad, rctx->n_dims, theta_cache);
        } else {
            hvx_rope_f32_aa(dst_spad, src0_spad, rctx->n_dims, theta_cache);
        }

        // fill the remain channels with data from src tensor
        if (rctx->n_dims < ne0) {
            hvx_copy_f32_uu((uint8_t *)(dst_spad + rctx->n_dims), (uint8_t *)(src0_spad + rctx->n_dims), ne0 - rctx->n_dims);
        }

        uint8_t * dst_addr = (uint8_t *) dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1;
        dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(dst_addr, dst_spad), rctx->dst_row_size, rctx->dst_row_size_aligned, 1);

        // prefetch next row
        const uint32_t pr = (ir + HTP_ROPE_SPAD_NROWS);
        if (pr < src0_end_row) {
            // Re-calculate src ptr for prefetch
            uint32_t pi1 = fastmodulo(pr, ne01, &rctx->fastdiv_ne01);
            uint32_t pr_div_ne01 = fastdiv(pr, &rctx->fastdiv_ne01);
            uint32_t pi2 = fastmodulo(pr_div_ne01, ne02, &rctx->fastdiv_ne02);
            uint32_t pi3 = fastdiv(pr_div_ne01, &rctx->fastdiv_ne02);
            const uint8_t * psrc_addr = (const uint8_t *) src0->data + pi3 * nb03 + pi2 * nb02 + pi1 * nb01;

            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src0_spad, psrc_addr), // reusing src0_spad buffer
                rctx->src0_row_size_aligned, rctx->src0_row_size, 1);

            rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i1 = pi1;
            rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i2 = pi2;
            rowidx_cache[ir % HTP_ROPE_SPAD_NROWS].i3 = pi3;
        }
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "rope-f32: %d/%d: (%u:%u) usec %u\n", ith, nth, src0_start_row, src0_end_row,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static int execute_op_rope_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    const struct htp_tensor * src2 = &octx->src2;
    struct htp_tensor *       dst  = &octx->dst;

    struct htp_rope_context rctx;
    const char * op_type = "rope-f32";

    switch (octx->op) {
        case HTP_OP_ROPE:
            init_rope_ctx(&rctx, octx);
            break;

        default:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_threads = octx->n_threads;

    const size_t src0_row_size = src0->nb[1];
    const size_t dst_row_size  = dst->nb[1];

    // Aligned row sizes for VTCM
    const size_t src0_row_size_aligned    = hex_round_up(src0_row_size, VLEN);
    const size_t dst_row_size_aligned     = hex_round_up(dst_row_size, VLEN);
    const size_t theta_cache_size_aligned = hex_round_up(src0->ne[0] * sizeof(float), 128);

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

    // Assign sizes
    octx->src0_spad.size_per_thread = src0_spad_per_thread;
    octx->dst_spad.size_per_thread  = dst_spad_per_thread;
    octx->src0_spad.size = n_threads * src0_spad_per_thread;
    octx->dst_spad.size  = n_threads * dst_spad_per_thread;
    octx->src1_spad.size = 0;

    // Assign pointers
    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = NULL;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size;

    // Fill context
    rctx.src0_row_size = src0_row_size;
    rctx.dst_row_size  = dst_row_size;
    rctx.src0_row_size_aligned = src0_row_size_aligned;
    rctx.dst_row_size_aligned  = dst_row_size_aligned;
    rctx.theta_cache_offset    = theta_cache_size_aligned;

    uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    rctx.src0_nrows = src0_nrows;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs             = MIN(n_threads, src0_nrows);
        rctx.src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;
        worker_pool_run_func(octx->ctx->worker_pool, rope_job_f32, &rctx, n_jobs);
    }

    return err;
}

int op_rope(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_rope_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
