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
    size_t dst_row_size;
    size_t src0_row_size_aligned;
    size_t dst_row_size_aligned;
    size_t theta_cache_offset;
    uint32_t src0_nrows;

    uint64_t t_start;
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

static inline void hvx_rope_neox_f32_aa(float * restrict dst, const float * restrict src0, uint32_t ne, const float * restrict theta_cache) {
    const HVX_Vector * restrict vsrc   = (const HVX_Vector *) src0;
    const HVX_Vector * restrict vtheta = (const HVX_Vector *) theta_cache;
    HVX_Vector       * restrict vdst   = (HVX_Vector *) dst;

    uint32_t nvec = (ne / (VLEN_FP32 * 2) * 2); // 2 vecs per loop, step of 2

    uint32_t he = ne / 2;         // half_dims offset in elements
    uint32_t hv = he / VLEN_FP32; // half_dims offset in vectors

    #pragma unroll(2)
    for (uint32_t i = 0; i < nvec; i += 2) {
        HVX_Vector v0 = vsrc[i/2+0];
        HVX_Vector v1 = vsrc[i/2+hv];

        HVX_Vector v2 = vtheta[i+0];
        HVX_Vector v3 = vtheta[i+1];

        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);  // vcos_sin[0] = cos_theta, vcos_sin[1] = sin_theta

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        vdst[i/2+0]  = Q6_Vsf_equals_Vqf32(v4);
        vdst[i/2+hv] = Q6_Vsf_equals_Vqf32(v5);
    }

    for (uint32_t i = nvec * VLEN_FP32; i < ne; i += 2) {
        const float cos_theta = theta_cache[i+0];
        const float sin_theta = theta_cache[i+1];
        float x0 = src0[i/2];
        float x1 = src0[i/2 + he];
        dst[i/2]      = x0 * cos_theta - x1 * sin_theta;
        dst[i/2 + he] = x0 * sin_theta + x1 * cos_theta;
    }
}

static inline void hvx_rope_f32_aa(float * restrict dst, const float * restrict src0, uint32_t ne, const float * restrict theta_cache) {
    const HVX_Vector * restrict vsrc   = (const HVX_Vector *) src0;
    const HVX_Vector * restrict vtheta = (const HVX_Vector *) theta_cache;
    HVX_Vector       * restrict vdst   = (HVX_Vector *) dst;

    uint32_t nvec = (ne / (VLEN_FP32 * 2)) * 2; // 2 vecs per loop, step of two

    #pragma unroll(2)
    for (uint32_t i = 0; i < nvec; i+=2) {
        HVX_Vector v0 = vsrc[i+0];
        HVX_Vector v1 = vsrc[i+1];

        HVX_Vector v2 = vtheta[i+0];
        HVX_Vector v3 = vtheta[i+1];

        HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(v1, v0, -4);  // vx0_x1[0] = x0, vx0_x1[1] = x1
        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);  // vcos_sin[0] = cos_theta, vcos_sin[1] = sin_theta

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        vdst[i+0] = Q6_V_lo_W(vstore);
        vdst[i+1] = Q6_V_hi_W(vstore);
    }

    for (uint32_t i = nvec * VLEN_FP32; i < ne; i += 2) {
        const float cos_theta = theta_cache[i+0];
        const float sin_theta = theta_cache[i+1];
        float x0 = src0[i+0];
        float x1 = src0[i+1];
        dst[i+0] = x0 * cos_theta - x1 * sin_theta;
        dst[i+1] = x0 * sin_theta + x1 * cos_theta;
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

    uint64_t tt = HAP_perf_get_qtimer_count();

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

    uint32_t ir = 0;
    uint32_t prev_i2 = (uint32_t) -1;

    for (uint32_t i3 = 0; i3 < ne3; i3++) { // batch
        for (uint32_t i2 = 0; i2 < ne2; i2++) { // seq-len
            for (uint32_t i1 = 0; i1 < ne1; ) { // attn-heads
                if (ir < src0_start_row) { ir++; i1++; continue; }
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
                    dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src_spad, src_addr),
                        rctx->src0_row_size_aligned, rctx->src0_row_size, pnr);

                    // FARF(HIGH, "rope-prefetch %u: pr %u i1 %u i2 %u i3 %u src-spad %p src-addr %p pnr %u", ith, pir, pi1, i2, i3, src_spad, src_addr, pnr);
                }

                // Update theta cache
                if (i2 != prev_i2) {
                    prev_i2 = i2;

                    const int32_t p = pos[i2];
                    rope_cache_init(p, rctx->freq_scale, freq_factors, rctx->corr_dims, ne0, rctx->ext_factor, rctx->attn_factor, theta_cache, rctx->theta_scale);

                    // FARF(HIGH, "rope-theta %u: ir %u i1 %u i2 %u i3 %u cache %p : usec %u", ith, ir, i1, i2, i3, theta_cache,
                    //         (unsigned) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - rctx->t_start));
                }

                // Skip DMA transactions from prev block (if any)
                // No need to wait for these since the DMA is setup for in-order processing
                for (uint32_t d=0; d < dma_depth; d++) { dma_queue_pop_nowait(dma_queue); }

                // Compute loop
                for (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr, ir += cnr, i1 += cnr) {
                    // Number of rows to compute
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_BLOCK);

                    uint8_t * dst_spad = (uint8_t *) dma_queue_pop(dma_queue).src;
                    uint8_t * src_spad = (uint8_t *) dma_queue_pop(dma_queue).dst;

                    // FARF(HIGH, "rope-compute %u: ir %u i1 %u i2 %u i3 %u src-spad %p cnr %u : usec %u", ith, ir, i1, i2, i3, src_spad, cnr,
                    //         (unsigned) HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - rctx->t_start));

                    if (is_neox) {
                        rope_neox_f32(rctx, dst_spad, src_spad, cnr, ne0, theta_cache);
                    } else {
                        rope_basic_f32(rctx, dst_spad, src_spad, cnr, ne0, theta_cache);
                    }

                    uint8_t * dst_addr = (uint8_t *) dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1;
                    dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(dst_addr, dst_spad), rctx->dst_row_size, rctx->dst_row_size_aligned, cnr);

                    // Prefetch more rows (if any)
                    if ((cr + HTP_ROPE_SPAD_NROWS) < nrows) {
                        uint32_t pnr = MIN(nrows - (cr + HTP_ROPE_SPAD_NROWS), HTP_ROPE_SPAD_BLOCK);
                        uint32_t pi1 = i1 + HTP_ROPE_SPAD_NROWS;
                        uint32_t pir = ir + HTP_ROPE_SPAD_NROWS;

                        const uint8_t * src_addr = (const uint8_t *) src0->data + i3 * nb03 + i2 * nb02 + pi1 * nb01;
                        dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src_spad, src_addr),
                            rctx->src0_row_size_aligned, rctx->src0_row_size, pnr);

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

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    const struct htp_tensor * src2 = &octx->src2;
    struct htp_tensor *       dst  = &octx->dst;

    const char * op_type = "rope-f32";

    switch (octx->op) {
        case HTP_OP_ROPE:
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

    rctx.src0_row_size = src0_row_size;
    rctx.dst_row_size  = dst_row_size;
    rctx.src0_row_size_aligned = src0_row_size_aligned;
    rctx.dst_row_size_aligned  = dst_row_size_aligned;
    rctx.theta_cache_offset    = theta_cache_size_aligned;

    uint32_t ne0 = dst->ne[0];
    uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    rctx.src0_nrows = src0_nrows;

    FARF(HIGH, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_factor, rctx.theta_scale, rctx.attn_factor);

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        uint32_t n_jobs = MIN(n_threads, src0_nrows);
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
