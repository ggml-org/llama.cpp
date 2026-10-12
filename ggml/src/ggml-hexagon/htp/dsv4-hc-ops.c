#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "dsv4-hc-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct dsv4_hc_pre_context {
    struct htp_ops_context * octx;
    const struct htp_dsv4_hc_pre_kernel_params * kparams;
    uint32_t token_start;
    uint32_t n_chunks;
    uint32_t total_jobs;
    uint32_t jobs_per_thread;
    float scale;
    uint8_t * vtcm_x;
    uint8_t * vtcm_g;
    uint32_t dma_block_chunks;
    uint32_t dma_row_stride;
    uint32_t dma_buffer_size;
};

static inline HVX_Vector dsv4_hc_fast_sigmoid_f32_2it(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector two_sf = hvx_vec_splat_f32(2.0f);
    HVX_Vector i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_Vector r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));

    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, Q6_Vsf_equals_Vqf32(r_qf)));
}

static inline HVX_Vector dsv4_hc_fast_sigmoid_f32_guard_2it(
        HVX_Vector v,
        HVX_Vector one,
        HVX_Vector max_exp,
        HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = dsv4_hc_fast_sigmoid_f32_2it(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

static inline void dsv4_hc_l2fetch_rows(const uint8_t * p, size_t row_stride, uint32_t i0, uint32_t n_cols, uint32_t n_rows) {
    if (row_stride > 0xFFFF) {
        return;
    }
    hex_l2fetch(p + i0*sizeof(float), n_cols*sizeof(float), row_stride, n_rows);
}

#define DSV4_HC_PF_CHUNKS 16

// prefetch x and gate for the aligned block after the one with chunk c, block does not cross tokens
static inline void dsv4_hc_prefetch_gated(
        const struct htp_tensor * x,
        const struct htp_tensor * g,
        uint32_t it,
        uint32_t c,
        uint32_t n_chunks,
        uint32_t n_hc) {
    uint32_t c0 = (c / DSV4_HC_PF_CHUNKS + 1) * DSV4_HC_PF_CHUNKS;
    if (c0 >= n_chunks) {
        c0 = 0;
        it++;
    }
    const uint32_t i0     = c0 * VLEN_FP32;
    const uint32_t n_cols = MIN(DSV4_HC_PF_CHUNKS * VLEN_FP32, x->ne[0] - i0);
    dsv4_hc_l2fetch_rows((const uint8_t *) x->data + it*x->nb[2], x->nb[1], i0, n_cols, n_hc);
    dsv4_hc_l2fetch_rows((const uint8_t *) g->data + it*g->nb[2], g->nb[1], i0, n_cols, n_hc);
}

static inline void dsv4_hc_pre_tail_f32(
        const struct htp_tensor * x,
        const struct htp_tensor * weights,
        const struct htp_tensor * dst,
        const uint8_t * x_data,
        const uint8_t * weights_data,
        uint8_t * dst_data,
        uint32_t i0,
        uint32_t it,
        uint32_t n,
        uint32_t n_hc,
        float scale,
        int gated) {
    for (uint32_t c = 0; c < n; ++c) {
        float sum = 0.0f;
        for (uint32_t ih = 0; ih < n_hc; ++ih) {
            const float xv = *(const float *) (x_data + (i0 + c)*x->nb[0] + ih*x->nb[1] + it*x->nb[2]);
            float wv;
            if (gated) {
                const float gv = *(const float *) (weights_data + (i0 + c)*weights->nb[0] + ih*weights->nb[1] + it*weights->nb[2]);
                wv = 1.0f / (1.0f + expf(-gv));
            } else {
                wv = *(const float *) (weights_data + ih*weights->nb[0] + it*weights->nb[1]);
            }
            sum += xv * wv;
        }
        *(float *) (dst_data + (i0 + c)*dst->nb[0] + it*dst->nb[1]) = scale * sum;
    }
}

static void dsv4_hc_pre_thread_plain_f32(unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    const uint8_t * x_data       = (const uint8_t *) x->data;
    const uint8_t * weights_data = (const uint8_t *) weights->data;
    uint8_t *       dst_data     = (uint8_t *) dst->data;

    const uint32_t n_embd = x->ne[0];
    const uint32_t start  = dctx->jobs_per_thread * ith;
    const uint32_t end    = MIN(start + dctx->jobs_per_thread, dctx->total_jobs);

    if (start >= end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    const HVX_Vector scale_v = hvx_vec_splat_f32(dctx->scale);

    const uint32_t n_chunks = dctx->n_chunks;
    uint32_t       it       = fastdiv(start, &dctx->kparams->div_n_chunks);
    uint32_t       chunk    = start - it*n_chunks;
    it += dctx->token_start;

    for (uint32_t job = start; job < end; ++it, chunk = 0) {
        const uint32_t chunk_end = MIN(n_chunks, chunk + (end - job));
        job += chunk_end - chunk;

        // plain path is cheap per chunk, so fetch the whole next token
        if (job < end) {
            dsv4_hc_l2fetch_rows(x_data + (it + 1)*x->nb[2], x->nb[1], 0, MIN(n_embd, (end - job)*VLEN_FP32), 4);
        }

        const uint8_t * wb = weights_data + it*weights->nb[1];

        const HVX_Vector w0 = hvx_vec_splat_f32(*(const float *) wb);
        const HVX_Vector w1 = hvx_vec_splat_f32(*(const float *) (wb + weights->nb[0]));
        const HVX_Vector w2 = hvx_vec_splat_f32(*(const float *) (wb + 2*weights->nb[0]));
        const HVX_Vector w3 = hvx_vec_splat_f32(*(const float *) (wb + 3*weights->nb[0]));

        for (; chunk < chunk_end; ++chunk) {
            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = MIN(VLEN_FP32, n_embd - i0);

            if (n != VLEN_FP32) {
                dsv4_hc_pre_tail_f32(x, weights, dst, x_data, weights_data, dst_data, i0, it, n, 4, dctx->scale, 0);
                continue;
            }

            const uint8_t * xb = x_data + i0*x->nb[0] + it*x->nb[2];
            uint8_t *       db = dst_data + i0*dst->nb[0] + it*dst->nb[1];

            const HVX_Vector x0 = *(const HVX_UVector *) xb;
            const HVX_Vector x1 = *(const HVX_UVector *) (xb + x->nb[1]);
            const HVX_Vector x2 = *(const HVX_UVector *) (xb + 2*x->nb[1]);
            const HVX_Vector x3 = *(const HVX_UVector *) (xb + 3*x->nb[1]);

            HVX_Vector acc = hvx_vec_mul_f32_f32(x0, w0);
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x1, w1));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x2, w2));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x3, w3));

            acc = hvx_vec_mul_f32_f32(acc, scale_v);
            hvx_vec_store_u(db, n * sizeof(float), acc);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    (void) nth;
}

static void dsv4_hc_pre_thread_gated_f32(unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    const uint8_t * x_data       = (const uint8_t *) x->data;
    const uint8_t * weights_data = (const uint8_t *) weights->data;
    uint8_t *       dst_data     = (uint8_t *) dst->data;

    const uint32_t n_embd = x->ne[0];
    const uint32_t start  = dctx->jobs_per_thread * ith;
    const uint32_t end    = MIN(start + dctx->jobs_per_thread, dctx->total_jobs);

    if (start >= end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    const HVX_Vector scale_v = hvx_vec_splat_f32(dctx->scale);
    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);

    const uint32_t n_chunks = dctx->n_chunks;
    uint32_t       it       = fastdiv(start, &dctx->kparams->div_n_chunks);
    uint32_t       chunk    = start - it*n_chunks;
    it += dctx->token_start;

    for (uint32_t job = start; job < end; ++it, chunk = 0) {
        const uint32_t chunk_end = MIN(n_chunks, chunk + (end - job));

        for (; chunk < chunk_end; ++chunk, ++job) {
            const uint32_t c_pf = chunk % DSV4_HC_PF_CHUNKS;
            if ((c_pf == 0 || job == start) && job + MIN(DSV4_HC_PF_CHUNKS - c_pf, n_chunks - chunk) < end) {
                dsv4_hc_prefetch_gated(x, weights, it, chunk, n_chunks, 4);
            }

            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = MIN(VLEN_FP32, n_embd - i0);

            if (n != VLEN_FP32) {
                dsv4_hc_pre_tail_f32(x, weights, dst, x_data, weights_data, dst_data, i0, it, n, 4, dctx->scale, 1);
                continue;
            }

            const uint8_t * xb = x_data + i0*x->nb[0] + it*x->nb[2];
            const uint8_t * gb = weights_data + i0*weights->nb[0] + it*weights->nb[2];
            uint8_t *       db = dst_data + i0*dst->nb[0] + it*dst->nb[1];

            const HVX_Vector x0 = *(const HVX_UVector *) xb;
            const HVX_Vector x1 = *(const HVX_UVector *) (xb + x->nb[1]);
            const HVX_Vector x2 = *(const HVX_UVector *) (xb + 2*x->nb[1]);
            const HVX_Vector x3 = *(const HVX_UVector *) (xb + 3*x->nb[1]);

            const HVX_Vector g0 = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_UVector *) gb, one, max_exp, min_exp);
            const HVX_Vector g1 = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_UVector *) (gb + weights->nb[1]), one, max_exp, min_exp);
            const HVX_Vector g2 = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_UVector *) (gb + 2*weights->nb[1]), one, max_exp, min_exp);
            const HVX_Vector g3 = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_UVector *) (gb + 3*weights->nb[1]), one, max_exp, min_exp);

            HVX_Vector acc = hvx_vec_mul_f32_f32(x0, g0);
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x1, g1));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x2, g2));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x3, g3));

            acc = hvx_vec_mul_f32_f32(acc, scale_v);
            hvx_vec_store_u(db, n * sizeof(float), acc);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    (void) nth;
}

static void dsv4_hc_pre_thread_plain_generic_f32(unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    const uint8_t * x_data       = (const uint8_t *) x->data;
    const uint8_t * weights_data = (const uint8_t *) weights->data;
    uint8_t *       dst_data     = (uint8_t *) dst->data;

    const uint32_t n_embd = x->ne[0];
    const uint32_t n_hc   = x->ne[1];
    const uint32_t start  = dctx->jobs_per_thread * ith;
    const uint32_t end    = MIN(start + dctx->jobs_per_thread, dctx->total_jobs);

    if (start >= end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    const HVX_Vector scale_v = hvx_vec_splat_f32(dctx->scale);

    const uint32_t n_chunks = dctx->n_chunks;
    uint32_t       it       = fastdiv(start, &dctx->kparams->div_n_chunks);
    uint32_t       chunk    = start - it*n_chunks;
    it += dctx->token_start;

    for (uint32_t job = start; job < end; ++it, chunk = 0) {
        const uint32_t chunk_end = MIN(n_chunks, chunk + (end - job));
        job += chunk_end - chunk;

        if (job < end) {
            dsv4_hc_l2fetch_rows(x_data + (it + 1)*x->nb[2], x->nb[1], 0, MIN(n_embd, (end - job)*VLEN_FP32), n_hc);
        }

        const uint8_t * wb = weights_data + it*weights->nb[1];

        for (; chunk < chunk_end; ++chunk) {
            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = MIN(VLEN_FP32, n_embd - i0);

            if (n != VLEN_FP32) {
                dsv4_hc_pre_tail_f32(x, weights, dst, x_data, weights_data, dst_data, i0, it, n, n_hc, dctx->scale, 0);
                continue;
            }

            const uint8_t * xb = x_data + i0*x->nb[0] + it*x->nb[2];
            uint8_t *       db = dst_data + i0*dst->nb[0] + it*dst->nb[1];

            HVX_Vector acc = hvx_vec_mul_f32_f32(*(const HVX_UVector *) xb, hvx_vec_splat_f32(*(const float *) wb));
            for (uint32_t ih = 1; ih < n_hc; ++ih) {
                const HVX_Vector xv = *(const HVX_UVector *) (xb + ih*x->nb[1]);
                const HVX_Vector wv = hvx_vec_splat_f32(*(const float *) (wb + ih*weights->nb[0]));
                acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(xv, wv));
            }

            acc = hvx_vec_mul_f32_f32(acc, scale_v);
            hvx_vec_store_u(db, n * sizeof(float), acc);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    (void) nth;
}

static __attribute__((noinline)) void dsv4_hc_pre_gated_generic_chunk_f32(
        const uint8_t * xb,
        const uint8_t * gb,
        uint8_t * db,
        size_t x_nb1,
        size_t g_nb1,
        uint32_t n_hc,
        float scale) {
    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);

    HVX_Vector gv  = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_UVector *) gb, one, max_exp, min_exp);
    HVX_Vector acc = hvx_vec_mul_f32_f32(*(const HVX_UVector *) xb, gv);
    // pipelined loop spills a sigmoid constant on v81
    #pragma clang loop pipeline(disable)
    for (uint32_t ih = 1; ih < n_hc; ++ih) {
        gv  = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_UVector *) (gb + ih*g_nb1), one, max_exp, min_exp);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_UVector *) (xb + ih*x_nb1), gv));
    }

    acc = hvx_vec_mul_f32_f32(acc, hvx_vec_splat_f32(scale));
    hvx_vec_store_u(db, VLEN, acc);
}

static void dsv4_hc_pre_thread_gated_generic_f32(unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    const uint8_t * x_data       = (const uint8_t *) x->data;
    const uint8_t * weights_data = (const uint8_t *) weights->data;
    uint8_t *       dst_data     = (uint8_t *) dst->data;

    const uint32_t n_embd = x->ne[0];
    const uint32_t n_hc   = x->ne[1];
    const uint32_t start  = dctx->jobs_per_thread * ith;
    const uint32_t end    = MIN(start + dctx->jobs_per_thread, dctx->total_jobs);

    if (start >= end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    const uint32_t n_chunks = dctx->n_chunks;
    uint32_t       it       = fastdiv(start, &dctx->kparams->div_n_chunks);
    uint32_t       chunk    = start - it*n_chunks;
    it += dctx->token_start;

    for (uint32_t job = start; job < end; ++it, chunk = 0) {
        const uint32_t chunk_end = MIN(n_chunks, chunk + (end - job));

        for (; chunk < chunk_end; ++chunk, ++job) {
            const uint32_t c_pf = chunk % DSV4_HC_PF_CHUNKS;
            if ((c_pf == 0 || job == start) && job + MIN(DSV4_HC_PF_CHUNKS - c_pf, n_chunks - chunk) < end) {
                dsv4_hc_prefetch_gated(x, weights, it, chunk, n_chunks, n_hc);
            }

            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = MIN(VLEN_FP32, n_embd - i0);

            if (n != VLEN_FP32) {
                dsv4_hc_pre_tail_f32(x, weights, dst, x_data, weights_data, dst_data, i0, it, n, n_hc, dctx->scale, 1);
                continue;
            }

            const uint8_t * xb = x_data + i0*x->nb[0] + it*x->nb[2];
            const uint8_t * gb = weights_data + i0*weights->nb[0] + it*weights->nb[2];
            uint8_t *       db = dst_data + i0*dst->nb[0] + it*dst->nb[1];

            dsv4_hc_pre_gated_generic_chunk_f32(xb, gb, db, x->nb[1], weights->nb[1], n_hc, dctx->scale);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    (void) nth;
}

struct dsv4_hc_dma_block {
    uint32_t it;
    uint32_t chunk;
    uint32_t count;
};

static inline struct dsv4_hc_dma_block dsv4_hc_dma_block_make(
        const struct dsv4_hc_pre_context * dctx,
        uint32_t job,
        uint32_t end) {
    const uint32_t local_token = fastdiv(job, &dctx->kparams->div_n_chunks);
    const uint32_t chunk       = job - local_token*dctx->n_chunks;
    const uint32_t count       = MIN(dctx->dma_block_chunks, MIN(dctx->n_chunks - chunk, end - job));
    return (struct dsv4_hc_dma_block) {
        .it    = dctx->token_start + local_token,
        .chunk = chunk,
        .count = count,
    };
}

static inline void dsv4_hc_dma_push_gated(
        dma_queue * dma_q,
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_dma_block * block,
        uint8_t * x_spad,
        uint8_t * g_spad) {
    const struct htp_tensor * x = dctx->octx->src[0];
    const struct htp_tensor * g = dctx->octx->src[1];
    const uint32_t row_size = block->count * VLEN;
    const dma_addr_t x_src = x->data + (uint64_t) block->it*x->nb[2] + (uint64_t) block->chunk*VLEN;
    const dma_addr_t g_src = g->data + (uint64_t) block->it*g->nb[2] + (uint64_t) block->chunk*VLEN;

    dma_queue_push(dma_q, dma_make_data(x_spad, x_src), dctx->dma_row_stride, x->nb[1], row_size, 4);
    dma_queue_push(dma_q, dma_make_data(g_spad, g_src), dctx->dma_row_stride, g->nb[1], row_size, 4);
}

static __attribute__((noinline)) void dsv4_hc_pre_gated_dma_block_f32(
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_dma_block * block,
        const uint8_t * x_spad,
        const uint8_t * g_spad) {
    const struct htp_tensor * dst = dctx->octx->dst;
    uint8_t * dst_data = (uint8_t *) dst->data;

    const HVX_Vector scale_v = hvx_vec_splat_f32(dctx->scale);
    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);

    for (uint32_t c = 0; c < block->count; ++c) {
        const uint32_t i0 = (block->chunk + c) * VLEN_FP32;
        const uint8_t * xb = x_spad + c*VLEN;
        const uint8_t * gb = g_spad + c*VLEN;
        uint8_t * db = dst_data + i0*dst->nb[0] + block->it*dst->nb[1];

        HVX_Vector xv = *(const HVX_Vector *) xb;
        HVX_Vector gv = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_Vector *) gb, one, max_exp, min_exp);
        HVX_Vector acc = hvx_vec_mul_f32_f32(xv, gv);

        xv = *(const HVX_Vector *) (xb + dctx->dma_row_stride);
        gv = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_Vector *) (gb + dctx->dma_row_stride), one, max_exp, min_exp);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(xv, gv));

        xv = *(const HVX_Vector *) (xb + 2*dctx->dma_row_stride);
        gv = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_Vector *) (gb + 2*dctx->dma_row_stride), one, max_exp, min_exp);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(xv, gv));

        xv = *(const HVX_Vector *) (xb + 3*dctx->dma_row_stride);
        gv = dsv4_hc_fast_sigmoid_f32_guard_2it(*(const HVX_Vector *) (gb + 3*dctx->dma_row_stride), one, max_exp, min_exp);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(xv, gv));

        acc = hvx_vec_mul_f32_f32(acc, scale_v);
        hvx_vec_store_u(db, VLEN, acc);
    }
}

static void dsv4_hc_pre_thread_gated_dma_f32(unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const uint32_t start = dctx->jobs_per_thread * ith;
    const uint32_t end   = MIN(start + dctx->jobs_per_thread, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    const uint32_t thread_offset = 2*dctx->dma_buffer_size*ith;
    uint8_t * x_spad[2] = {
        dctx->vtcm_x + thread_offset,
        dctx->vtcm_x + thread_offset + dctx->dma_buffer_size,
    };
    uint8_t * g_spad[2] = {
        dctx->vtcm_g + thread_offset,
        dctx->vtcm_g + thread_offset + dctx->dma_buffer_size,
    };

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_dma_block block = dsv4_hc_dma_block_make(dctx, prefetch_job, end);
        dsv4_hc_dma_push_gated(dma_q, dctx, &block, x_spad[i], g_spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    uint32_t job = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_dma_block block = dsv4_hc_dma_block_make(dctx, job, end);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        dsv4_hc_pre_gated_dma_block_f32(dctx, &block, x_spad[curr], g_spad[curr]);

        if (prefetch_job < end) {
            const struct dsv4_hc_dma_block next = dsv4_hc_dma_block_make(dctx, prefetch_job, end);
            dsv4_hc_dma_push_gated(dma_q, dctx, &next, x_spad[curr], g_spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    (void) nth;
}

int op_dsv4_hc_pre(struct htp_ops_context * octx) {
    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    if (x->type != HTP_TYPE_F32 || weights->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (htp_tensor_is_extended(x) || htp_tensor_is_extended(weights) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const uint32_t n_hc = x->ne[1];
    if (n_hc == 0 || x->ne[3] != 1 || dst->ne[0] != x->ne[0] || dst->ne[1] != x->ne[2]) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (x->nb[0] != sizeof(float) || weights->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (htp_tensor_is_permuted(x) || htp_tensor_is_permuted(weights) || htp_tensor_is_permuted(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    float scale;
    memcpy(&scale, &octx->op_params[0], sizeof(float));
    const int gated = octx->op_params[1] != 0;

    if (gated) {
        if (weights->ne[0] != x->ne[0] || weights->ne[1] != n_hc || weights->ne[2] != x->ne[2] || weights->ne[3] != 1) {
            return HTP_STATUS_NO_SUPPORT;
        }
    } else {
        if (weights->ne[0] != n_hc || weights->ne[1] != x->ne[2] || weights->ne[2] != 1 || weights->ne[3] != 1) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    uint32_t token_start = 0;
    uint32_t n_tokens    = x->ne[2];
    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_tokens, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        token_start = range.start;
        n_tokens    = range.count;
    }

    if (n_tokens == 0) {
        return HTP_STATUS_OK;
    }

    const struct htp_dsv4_hc_pre_kernel_params * kparams =
        (const struct htp_dsv4_hc_pre_kernel_params *) octx->kernel_params;
    if (kparams->n_chunks == 0 || kparams->n_threads == 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_chunks   = kparams->n_chunks;
    const uint32_t total_jobs = n_tokens * n_chunks;
    const uint32_t n_threads  = kparams->n_threads;
    if (!htp_ops_context_set_n_threads_div(octx, n_threads, kparams->div_n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t dma_block_chunks = kparams->dma_block_chunks;
    const uint32_t dma_row_stride   = kparams->dma_row_stride;
    const uint32_t dma_buffer_size  = kparams->dma_buffer_size;
    const uint32_t jobs_per_thread  = octx->ctx->mdev.count > 1
        ? fastdiv(total_jobs + n_threads - 1, &kparams->div_n_threads)
        : kparams->jobs_per_thread;
    const size_t   dma_input_size   = kparams->dma_input_size;
    uint8_t *      vtcm_base        = (uint8_t *) octx->ctx->vtcm_base;
    const bool     use_dma          = kparams->dma_eligible &&
                                     gated &&
                                     n_hc == 4 &&
                                     n_tokens > 1 &&
                                     jobs_per_thread >= 2*dma_block_chunks &&
                                     vtcm_base != NULL &&
                                     kparams->dma_vtcm_size <= octx->ctx->vtcm_size;

    struct dsv4_hc_pre_context dctx = {
        .octx             = octx,
        .kparams          = kparams,
        .token_start      = token_start,
        .n_chunks         = n_chunks,
        .total_jobs       = total_jobs,
        .jobs_per_thread  = jobs_per_thread,
        .scale            = scale,
        .vtcm_x           = vtcm_base,
        .vtcm_g           = vtcm_base ? vtcm_base + dma_input_size : NULL,
        .dma_block_chunks = dma_block_chunks,
        .dma_row_stride   = dma_row_stride,
        .dma_buffer_size  = dma_buffer_size,
    };

    FARF(HIGH, "dsv4-hc-pre: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : gated %d dma %d scale %.6f tokens %u:%u n-threads %u\n",
         x->ne[0], x->ne[1], x->ne[2], x->ne[3],
         weights->ne[0], weights->ne[1], weights->ne[2], weights->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gated, use_dma, scale, token_start, token_start + n_tokens, n_threads);

    if (use_dma) {
        work_queue_run(octx->ctx->work_queue, dsv4_hc_pre_thread_gated_dma_f32, &dctx, n_threads);
    } else if (n_hc != 4) {
        work_queue_run(octx->ctx->work_queue, gated ? dsv4_hc_pre_thread_gated_generic_f32 : dsv4_hc_pre_thread_plain_generic_f32, &dctx, n_threads);
    } else if (gated) {
        work_queue_run(octx->ctx->work_queue, dsv4_hc_pre_thread_gated_f32, &dctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, dsv4_hc_pre_thread_plain_f32, &dctx, n_threads);
    }

    return HTP_STATUS_OK;
}
