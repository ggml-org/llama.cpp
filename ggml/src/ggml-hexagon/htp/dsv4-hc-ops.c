#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dma-queue.h"
#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "dsv4-hc-ops.h"

// ---------------------------------------------------------------------------
// DSV4_HC_PRE
// ---------------------------------------------------------------------------

// hvx_vec_fast_sigmoid_f32_guard_2it with fewer live constants and predicates:
// log2(e)/2 folded into one multiply, input clamped to [min_exp, max_exp] instead of the 0/1 mux,
// hw float to int conversion instead of hvx_vec_truncate_f32
static inline HVX_Vector dsv4_hc_sigmoid_f32(HVX_Vector v, HVX_Vector max_exp, HVX_Vector min_exp) {
    v = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(v, min_exp), max_exp);
    v = Q6_Vqf32_vmpy_VsfVsf(v, hvx_vec_splat_f32(0.5f * 1.442695022f));

    HVX_Vector in_int = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(v));
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

    // Newton-Raphson with 2 iterations, r' = (r + r) - r*(r*v5) needs no 2.0 constant
    HVX_Vector r = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    for (int i = 0; i < 2; ++i) {
        const HVX_Vector rv = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r, v5));
        r = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vadd_VsfVsf(r, r), Q6_Vqf32_vmpy_VsfVsf(r, rv)));
    }

    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v3, r));
}

struct dsv4_hc_pre_context {
    struct htp_ops_context * octx;
    const struct htp_dsv4_hc_pre_kernel_params * kparams;
    uint32_t token_start;
    uint32_t n_chunks;
    uint32_t total_jobs;
    uint32_t jobs_per_thread;
    float scale;
    uint8_t * vtcm;
    uint32_t dma_block_chunks;
    uint32_t dma_row_stride;
    uint32_t dma_buffer_size;
};

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
    const uint32_t n_cols = hex_smin(DSV4_HC_PF_CHUNKS * VLEN_FP32, x->ne[0] - i0);
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
    const uint32_t end    = hex_smin(start + dctx->jobs_per_thread, dctx->total_jobs);

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
        const uint32_t chunk_end = hex_smin(n_chunks, chunk + (end - job));
        job += chunk_end - chunk;

        // plain path is cheap per chunk, so fetch the whole next token
        if (job < end) {
            dsv4_hc_l2fetch_rows(x_data + (it + 1)*x->nb[2], x->nb[1], 0, hex_smin(n_embd, (end - job)*VLEN_FP32), 4);
        }

        const uint8_t * wb = weights_data + it*weights->nb[1];

        const HVX_Vector w0 = hvx_vec_splat_f32(*(const float *) wb);
        const HVX_Vector w1 = hvx_vec_splat_f32(*(const float *) (wb + weights->nb[0]));
        const HVX_Vector w2 = hvx_vec_splat_f32(*(const float *) (wb + 2*weights->nb[0]));
        const HVX_Vector w3 = hvx_vec_splat_f32(*(const float *) (wb + 3*weights->nb[0]));

        for (; chunk < chunk_end; ++chunk) {
            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = hex_smin(VLEN_FP32, n_embd - i0);

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
    const uint32_t end    = hex_smin(start + dctx->jobs_per_thread, dctx->total_jobs);

    if (start >= end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    const HVX_Vector scale_v = hvx_vec_splat_f32(dctx->scale);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);

    const uint32_t n_chunks = dctx->n_chunks;
    uint32_t       it       = fastdiv(start, &dctx->kparams->div_n_chunks);
    uint32_t       chunk    = start - it*n_chunks;
    it += dctx->token_start;

    for (uint32_t job = start; job < end; ++it, chunk = 0) {
        const uint32_t chunk_end = hex_smin(n_chunks, chunk + (end - job));

        for (; chunk < chunk_end; ++chunk, ++job) {
            const uint32_t c_pf = chunk % DSV4_HC_PF_CHUNKS;
            if ((c_pf == 0 || job == start) && job + hex_smin(DSV4_HC_PF_CHUNKS - c_pf, n_chunks - chunk) < end) {
                dsv4_hc_prefetch_gated(x, weights, it, chunk, n_chunks, 4);
            }

            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = hex_smin(VLEN_FP32, n_embd - i0);

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

            const HVX_Vector g0 = dsv4_hc_sigmoid_f32(*(const HVX_UVector *) gb, max_exp, min_exp);
            const HVX_Vector g1 = dsv4_hc_sigmoid_f32(*(const HVX_UVector *) (gb + weights->nb[1]), max_exp, min_exp);
            const HVX_Vector g2 = dsv4_hc_sigmoid_f32(*(const HVX_UVector *) (gb + 2*weights->nb[1]), max_exp, min_exp);
            const HVX_Vector g3 = dsv4_hc_sigmoid_f32(*(const HVX_UVector *) (gb + 3*weights->nb[1]), max_exp, min_exp);

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
    const uint32_t end    = hex_smin(start + dctx->jobs_per_thread, dctx->total_jobs);

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
        const uint32_t chunk_end = hex_smin(n_chunks, chunk + (end - job));
        job += chunk_end - chunk;

        if (job < end) {
            dsv4_hc_l2fetch_rows(x_data + (it + 1)*x->nb[2], x->nb[1], 0, hex_smin(n_embd, (end - job)*VLEN_FP32), n_hc);
        }

        const uint8_t * wb = weights_data + it*weights->nb[1];

        for (; chunk < chunk_end; ++chunk) {
            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = hex_smin(VLEN_FP32, n_embd - i0);

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
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);

    HVX_Vector gv  = dsv4_hc_sigmoid_f32(*(const HVX_UVector *) gb, max_exp, min_exp);
    HVX_Vector acc = hvx_vec_mul_f32_f32(*(const HVX_UVector *) xb, gv);
    for (uint32_t ih = 1; ih < n_hc; ++ih) {
        gv  = dsv4_hc_sigmoid_f32(*(const HVX_UVector *) (gb + ih*g_nb1), max_exp, min_exp);
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
    const uint32_t end    = hex_smin(start + dctx->jobs_per_thread, dctx->total_jobs);

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
        const uint32_t chunk_end = hex_smin(n_chunks, chunk + (end - job));

        for (; chunk < chunk_end; ++chunk, ++job) {
            const uint32_t c_pf = chunk % DSV4_HC_PF_CHUNKS;
            if ((c_pf == 0 || job == start) && job + hex_smin(DSV4_HC_PF_CHUNKS - c_pf, n_chunks - chunk) < end) {
                dsv4_hc_prefetch_gated(x, weights, it, chunk, n_chunks, n_hc);
            }

            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = hex_smin(VLEN_FP32, n_embd - i0);

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
    const uint32_t count       = hex_smin(dctx->dma_block_chunks, hex_smin(dctx->n_chunks - chunk, end - job));
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
        uint8_t * spad) {
    const struct htp_tensor * x = dctx->octx->src[0];
    const struct htp_tensor * g = dctx->octx->src[1];
    const uint32_t row_stride = dctx->dma_row_stride;
    const uint32_t row_size   = block->count * VLEN;
    const dma_addr_t x_src = x->data + (block->it*x->nb[2] + block->chunk*VLEN);
    const dma_addr_t g_src = g->data + (block->it*g->nb[2] + block->chunk*VLEN);

    dma_queue_push(dma_q, dma_make_data(spad, x_src), row_stride, x->nb[1], row_size, 4);
    dma_queue_push(dma_q, dma_make_data(spad + 4*row_stride, g_src), row_stride, g->nb[1], row_size, 4);
}

static __attribute__((noinline)) void dsv4_hc_pre_gated_dma_block_f32(
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_dma_block * block,
        const uint8_t * x_spad,
        const uint8_t * g_spad,
        uint8_t * d_spad) {
    const uint32_t row_stride = dctx->dma_row_stride;
    const HVX_Vector scale_v = hvx_vec_splat_f32(dctx->scale);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);

    for (uint32_t c = 0; c < block->count; ++c) {
        const uint8_t * xb = x_spad + c*VLEN;
        const uint8_t * gb = g_spad + c*VLEN;

        HVX_Vector xv = *(const HVX_Vector *) xb;
        HVX_Vector gv = dsv4_hc_sigmoid_f32(*(const HVX_Vector *) gb, max_exp, min_exp);
        HVX_Vector acc = hvx_vec_mul_f32_f32(xv, gv);

        xv = *(const HVX_Vector *) (xb + row_stride);
        gv = dsv4_hc_sigmoid_f32(*(const HVX_Vector *) (gb + row_stride), max_exp, min_exp);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(xv, gv));

        xv = *(const HVX_Vector *) (xb + 2*row_stride);
        gv = dsv4_hc_sigmoid_f32(*(const HVX_Vector *) (gb + 2*row_stride), max_exp, min_exp);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(xv, gv));

        xv = *(const HVX_Vector *) (xb + 3*row_stride);
        gv = dsv4_hc_sigmoid_f32(*(const HVX_Vector *) (gb + 3*row_stride), max_exp, min_exp);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(xv, gv));

        *(HVX_Vector *) (d_spad + c*VLEN) = hvx_vec_mul_f32_f32(acc, scale_v);
    }
}

static void dsv4_hc_pre_thread_gated_dma_f32(unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * dst = octx->dst;
    const dma_addr_t d_data = dst->data;

    const uint32_t row_stride  = dctx->dma_row_stride;
    const uint32_t buffer_size = dctx->dma_buffer_size;
    const uint32_t start       = dctx->jobs_per_thread * ith;
    const uint32_t end         = hex_smin(start + dctx->jobs_per_thread, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_dma_block block = dsv4_hc_dma_block_make(dctx, prefetch_job, end);
        // empty dst writeback keeps pop order dst, x, g in the loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 8*row_stride), block.count*VLEN, block.count*VLEN, 0, 0);
        dsv4_hc_dma_push_gated(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    uint32_t job      = start;
    uint32_t curr     = 0;
    uint32_t n_blocks = 0;
    while (job < end) {
        const struct dsv4_hc_dma_block block = dsv4_hc_dma_block_make(dctx, job, end);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        uint8_t * d_spad = spad[curr] + 8*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        dsv4_hc_pre_gated_dma_block_f32(dctx, &block, spad[curr], spad[curr] + 4*row_stride, d_spad);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const dma_addr_t d_dst = d_data + (block.it*dst->nb[1] + block.chunk*VLEN);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), block.count*VLEN, block.count*VLEN, block.count*VLEN, 1);

        if (prefetch_job < end) {
            const struct dsv4_hc_dma_block next = dsv4_hc_dma_block_make(dctx, prefetch_job, end);
            dsv4_hc_dma_push_gated(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
        n_blocks++;
    }

    for (uint32_t i = 0; i < hex_smin(n_blocks, 2); ++i) {
        dma_queue_pop(dma_q);
    }

    (void) nth;
}

int op_dsv4_hc_pre(struct htp_ops_context * octx) {
    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    float scale;
    memcpy(&scale, &octx->op_params[0], sizeof(float));

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

    const uint32_t n_threads = kparams->n_threads;
    if (!htp_ops_context_set_n_threads_div(octx, n_threads, kparams->div_n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t kernel_type     = kparams->kernel_type;
    uint32_t total_jobs      = kparams->total_jobs;
    uint32_t jobs_per_thread = kparams->jobs_per_thread;
    if (octx->ctx->mdev.count > 1) {
        total_jobs      = n_tokens * kparams->n_chunks;
        jobs_per_thread = fastdiv(total_jobs + n_threads - 1, &kparams->div_n_threads);

        // mdev split can leave too few jobs per thread for the dma pipeline
        if (kernel_type == HTP_DSV4_HC_PRE_KERNEL_GATED_DMA && jobs_per_thread < 2*kparams->dma_block_chunks) {
            kernel_type = HTP_DSV4_HC_PRE_KERNEL_GATED;
        }
    }

    // only the dma kernel accesses tensors through 64-bit dma addresses
    if (kernel_type != HTP_DSV4_HC_PRE_KERNEL_GATED_DMA &&
        (htp_tensor_is_extended(x) || htp_tensor_is_extended(weights) || htp_tensor_is_extended(dst))) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct dsv4_hc_pre_context dctx = {
        .octx             = octx,
        .kparams          = kparams,
        .token_start      = token_start,
        .n_chunks         = kparams->n_chunks,
        .total_jobs       = total_jobs,
        .jobs_per_thread  = jobs_per_thread,
        .scale            = scale,
        .vtcm             = (uint8_t *) octx->ctx->vtcm_base,
        .dma_block_chunks = kparams->dma_block_chunks,
        .dma_row_stride   = kparams->dma_row_stride,
        .dma_buffer_size  = kparams->dma_buffer_size,
    };

    FARF(HIGH, "dsv4-hc-pre: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : kernel %u scale %.6f tokens %u:%u n-threads %u\n",
         x->ne[0], x->ne[1], x->ne[2], x->ne[3],
         weights->ne[0], weights->ne[1], weights->ne[2], weights->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kernel_type, scale, token_start, token_start + n_tokens, n_threads);

    work_queue_func_t worker;
    switch (kernel_type) {
        case HTP_DSV4_HC_PRE_KERNEL_PLAIN:         worker = dsv4_hc_pre_thread_plain_f32;         break;
        case HTP_DSV4_HC_PRE_KERNEL_GATED:         worker = dsv4_hc_pre_thread_gated_f32;         break;
        case HTP_DSV4_HC_PRE_KERNEL_PLAIN_GENERIC: worker = dsv4_hc_pre_thread_plain_generic_f32; break;
        case HTP_DSV4_HC_PRE_KERNEL_GATED_GENERIC: worker = dsv4_hc_pre_thread_gated_generic_f32; break;
        case HTP_DSV4_HC_PRE_KERNEL_GATED_DMA:     worker = dsv4_hc_pre_thread_gated_dma_f32;     break;
        default:                                   return HTP_STATUS_INVAL_PARAMS;
    }
    work_queue_run(octx->ctx->work_queue, worker, &dctx, n_threads);

    return HTP_STATUS_OK;
}

// ---------------------------------------------------------------------------
// DSV4_HC_POST
// ---------------------------------------------------------------------------

struct dsv4_hc_post_context {
    struct htp_ops_context * octx;
    const struct htp_dsv4_hc_post_kernel_params * kparams;
    uint32_t token_start;
    uint32_t total_jobs;
    uint32_t jobs_per_thread;
    uint8_t * vtcm;
};

static inline struct dsv4_hc_dma_block dsv4_hc_post_block_make(
        const struct dsv4_hc_post_context * dctx,
        uint32_t job,
        uint32_t end,
        uint32_t max_chunks) {
    const uint32_t n_chunks    = dctx->kparams->n_chunks;
    const uint32_t local_token = fastdiv(job, &dctx->kparams->div_n_chunks);
    const uint32_t chunk       = job - local_token*n_chunks;
    const uint32_t count       = hex_smin(max_chunks, hex_smin(n_chunks - chunk, end - job));

    return (struct dsv4_hc_dma_block) {
        .it    = dctx->token_start + local_token,
        .chunk = chunk,
        .count = count,
    };
}

static inline void dsv4_hc_post_load_coefs(const struct htp_ops_context * octx, uint32_t it, float * p, float * c) {
    const struct htp_tensor * post = octx->src[2];
    const struct htp_tensor * comb = octx->src[3];

    for (uint32_t id = 0; id < 4; ++id) {
        p[id] = *(const float *) ((const uint8_t *) post->data + id*post->nb[0] + it*post->nb[1]);
    }
    if (comb) {
        for (uint32_t id = 0; id < 4; ++id) {
            for (uint32_t is = 0; is < 4; ++is) {
                c[4*id + is] = *(const float *) ((const uint8_t *) comb->data + id*comb->nb[0] + is*comb->nb[1] + it*comb->nb[2]);
            }
        }
    }
}

static inline HVX_Vector dsv4_hc_post_load(const uint8_t * p, const int aligned) {
    return aligned ? *(const HVX_Vector *) p : *(const HVX_UVector *) p;
}

static inline void dsv4_hc_post_store(uint8_t * p, HVX_Vector v, const int aligned) {
    if (aligned) {
        *(HVX_Vector *) p = v;
    } else {
        hvx_vec_store_u(p, VLEN, v);
    }
}

// one dst stream at a time keeps few splats live, inputs are reloaded from L1 or VTCM
static inline __attribute__((always_inline)) void dsv4_hc_post_comb_impl(
        const uint8_t * xb,
        const uint8_t * rb,
        size_t r_stride,
        uint8_t * db,
        size_t d_stride,
        uint32_t count,
        const float * p,
        const float * c,
        const int aligned) {
    for (uint32_t id = 0; id < 4; ++id) {
        const HVX_Vector pv = hvx_vec_splat_f32(p[id]);
        const HVX_Vector c0 = hvx_vec_splat_f32(c[4*id + 0]);
        const HVX_Vector c1 = hvx_vec_splat_f32(c[4*id + 1]);
        const HVX_Vector c2 = hvx_vec_splat_f32(c[4*id + 2]);
        const HVX_Vector c3 = hvx_vec_splat_f32(c[4*id + 3]);
        uint8_t * dd = db + id*d_stride;

        for (uint32_t i = 0; i < count; ++i) {
            const uint8_t * r = rb + i*VLEN;
            HVX_Vector acc = hvx_vec_mul_f32_f32(dsv4_hc_post_load(xb + i*VLEN, aligned), pv);
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(dsv4_hc_post_load(r, aligned), c0));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(dsv4_hc_post_load(r + r_stride, aligned), c1));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(dsv4_hc_post_load(r + 2*r_stride, aligned), c2));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(dsv4_hc_post_load(r + 3*r_stride, aligned), c3));
            dsv4_hc_post_store(dd + i*VLEN, acc, aligned);
        }
    }
}

static inline __attribute__((always_inline)) void dsv4_hc_post_identity_impl(
        const uint8_t * xb,
        const uint8_t * rb,
        size_t r_stride,
        uint8_t * db,
        size_t d_stride,
        uint32_t count,
        const float * p,
        const int aligned) {
    for (uint32_t id = 0; id < 4; ++id) {
        const HVX_Vector pv = hvx_vec_splat_f32(p[id]);
        const uint8_t *  rr = rb + id*r_stride;
        uint8_t *        dd = db + id*d_stride;

        for (uint32_t i = 0; i < count; ++i) {
            const HVX_Vector xv = dsv4_hc_post_load(xb + i*VLEN, aligned);
            dsv4_hc_post_store(dd + i*VLEN, hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(xv, pv), dsv4_hc_post_load(rr + i*VLEN, aligned)), aligned);
        }
    }
}

static __attribute__((noinline)) void dsv4_hc_post_block_f32(
        const uint8_t * xb, const uint8_t * rb, size_t r_stride, uint8_t * db, size_t d_stride,
        uint32_t count, const float * p, const float * c) {
    if (c) {
        dsv4_hc_post_comb_impl(xb, rb, r_stride, db, d_stride, count, p, c, 0);
    } else {
        dsv4_hc_post_identity_impl(xb, rb, r_stride, db, d_stride, count, p, 0);
    }
}

static __attribute__((noinline)) void dsv4_hc_post_block_vtcm_f32(
        const uint8_t * xb, const uint8_t * rb, size_t r_stride, uint8_t * db, size_t d_stride,
        uint32_t count, const float * p, const float * c) {
    if (c) {
        dsv4_hc_post_comb_impl(xb, rb, r_stride, db, d_stride, count, p, c, 1);
    } else {
        dsv4_hc_post_identity_impl(xb, rb, r_stride, db, d_stride, count, p, 1);
    }
}

static void dsv4_hc_post_tail_f32(const struct htp_ops_context * octx, uint32_t it, uint32_t i0, const float * p, const float * c) {
    const struct htp_tensor * x        = octx->src[0];
    const struct htp_tensor * residual = octx->src[1];
    const struct htp_tensor * dst      = octx->dst;

    for (uint32_t i = i0; i < x->ne[0]; ++i) {
        const float xv = *(const float *) ((const uint8_t *) x->data + i*x->nb[0] + it*x->nb[1]);
        const uint8_t * rb = (const uint8_t *) residual->data + i*residual->nb[0] + it*residual->nb[2];
        for (uint32_t id = 0; id < 4; ++id) {
            float sum = xv * p[id];
            if (c) {
                for (uint32_t is = 0; is < 4; ++is) {
                    sum += *(const float *) (rb + is*residual->nb[1]) * c[4*id + is];
                }
            } else {
                sum += *(const float *) (rb + id*residual->nb[1]);
            }
            *(float *) ((uint8_t *) dst->data + i*dst->nb[0] + id*dst->nb[1] + it*dst->nb[2]) = sum;
        }
    }
}

static void dsv4_hc_post_thread_l2_f32(unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_post_context * dctx = (struct dsv4_hc_post_context *) data;
    struct htp_ops_context *      octx = dctx->octx;

    const struct htp_tensor * x        = octx->src[0];
    const struct htp_tensor * residual = octx->src[1];
    const struct htp_tensor * dst      = octx->dst;

    const uint8_t * x_data = (const uint8_t *) x->data;
    const uint8_t * r_data = (const uint8_t *) residual->data;
    uint8_t *       d_data = (uint8_t *) dst->data;

    const uint32_t n_embd   = x->ne[0];
    const uint32_t n_chunks = dctx->kparams->n_chunks;
    const uint32_t start    = dctx->jobs_per_thread * ith;
    const uint32_t end      = hex_smin(start + dctx->jobs_per_thread, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    float p[4];
    float c[16];
    const float * cp = dctx->kparams->has_comb ? c : NULL;
    uint32_t it_coefs = UINT32_MAX;

    for (uint32_t job = start; job < end;) {
        const struct dsv4_hc_dma_block block = dsv4_hc_post_block_make(dctx, job, end, DSV4_HC_PF_CHUNKS);
        job += block.count;

        if (job < end) {
            const struct dsv4_hc_dma_block next = dsv4_hc_post_block_make(dctx, job, end, DSV4_HC_PF_CHUNKS);
            const uint32_t n_cols = hex_smin(next.count*VLEN_FP32, n_embd - next.chunk*VLEN_FP32);
            dsv4_hc_l2fetch_rows(r_data + next.it*residual->nb[2], residual->nb[1], next.chunk*VLEN_FP32, n_cols, 4);
            dsv4_hc_l2fetch_rows(x_data + next.it*x->nb[1], n_cols*sizeof(float), next.chunk*VLEN_FP32, n_cols, 1);
        }

        if (block.it != it_coefs) {
            dsv4_hc_post_load_coefs(octx, block.it, p, c);
            it_coefs = block.it;
        }

        uint32_t n_full = block.count;
        if (block.chunk + block.count == n_chunks && n_embd % VLEN_FP32 != 0) {
            n_full--;
            dsv4_hc_post_tail_f32(octx, block.it, (n_chunks - 1)*VLEN_FP32, p, cp);
        }
        if (n_full == 0) {
            continue;
        }

        const size_t off = block.chunk*VLEN;
        dsv4_hc_post_block_f32(x_data + block.it*x->nb[1] + off,
                               r_data + block.it*residual->nb[2] + off, residual->nb[1],
                               d_data + block.it*dst->nb[2] + off, dst->nb[1],
                               n_full, p, cp);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    (void) nth;
}

static inline void dsv4_hc_post_dma_push(
        dma_queue * dma_q,
        const struct dsv4_hc_post_context * dctx,
        const struct dsv4_hc_dma_block * block,
        uint8_t * spad) {
    const struct htp_tensor * x        = dctx->octx->src[0];
    const struct htp_tensor * residual = dctx->octx->src[1];
    const uint32_t row_stride = dctx->kparams->dma_row_stride;
    const uint32_t row_size   = block->count * VLEN;
    const dma_addr_t r_src = residual->data + (block->it*residual->nb[2] + block->chunk*VLEN);
    const dma_addr_t x_src = x->data + (block->it*x->nb[1] + block->chunk*VLEN);

    dma_queue_push(dma_q, dma_make_data(spad, r_src), row_stride, residual->nb[1], row_size, 4);
    dma_queue_push(dma_q, dma_make_data(spad + 4*row_stride, x_src), row_stride, row_size, row_size, 1);
}

static void dsv4_hc_post_thread_dma_f32(unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_post_context * dctx = (struct dsv4_hc_post_context *) data;
    struct htp_ops_context *      octx = dctx->octx;

    const struct htp_tensor * dst = octx->dst;
    const dma_addr_t d_data = dst->data;

    const uint32_t block_chunks = dctx->kparams->dma_block_chunks;
    const uint32_t row_stride   = dctx->kparams->dma_row_stride;
    const uint32_t buffer_size  = dctx->kparams->dma_buffer_size;
    const uint32_t start        = dctx->jobs_per_thread * ith;
    const uint32_t end          = hex_smin(start + dctx->jobs_per_thread, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_dma_block block = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
        // empty dst writeback keeps the pop order dst, residual, x in the loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 5*row_stride), dst->nb[1], row_stride, 0, 0);
        dsv4_hc_post_dma_push(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    float p[4];
    float c[16];
    const float * cp = dctx->kparams->has_comb ? c : NULL;
    uint32_t it_coefs = UINT32_MAX;

    uint32_t job      = start;
    uint32_t curr     = 0;
    uint32_t n_blocks = 0;
    while (job < end) {
        const struct dsv4_hc_dma_block block = dsv4_hc_post_block_make(dctx, job, end, block_chunks);
        if (block.it != it_coefs) {
            dsv4_hc_post_load_coefs(octx, block.it, p, c);
            it_coefs = block.it;
        }

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        uint8_t * d_spad = spad[curr] + 5*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        dsv4_hc_post_block_vtcm_f32(spad[curr] + 4*row_stride, spad[curr], row_stride, d_spad, row_stride, block.count, p, cp);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const dma_addr_t d_dst = d_data + (block.it*dst->nb[2] + block.chunk*VLEN);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), dst->nb[1], row_stride, block.count*VLEN, 4);

        if (prefetch_job < end) {
            const struct dsv4_hc_dma_block next = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
            dsv4_hc_post_dma_push(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
        n_blocks++;
    }

    // drain the dst writebacks of the last two blocks
    for (uint32_t i = 0; i < hex_smin(n_blocks, 2); ++i) {
        dma_queue_pop(dma_q);
    }

    (void) nth;
}

int op_dsv4_hc_post(struct htp_ops_context * octx) {
    const struct htp_tensor * x        = octx->src[0];
    const struct htp_tensor * residual = octx->src[1];
    const struct htp_tensor * post     = octx->src[2];
    const struct htp_tensor * comb     = octx->src[3];
    const struct htp_tensor * dst      = octx->dst;

    // post and comb coefficients are always read through pointers
    if (htp_tensor_is_extended(post) || htp_tensor_is_extended(comb)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    uint32_t token_start = 0;
    uint32_t n_tokens    = x->ne[1];
    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_tokens, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        token_start = range.start;
        n_tokens    = range.count;
    }

    if (n_tokens == 0) {
        return HTP_STATUS_OK;
    }

    const struct htp_dsv4_hc_post_kernel_params * kparams =
        (const struct htp_dsv4_hc_post_kernel_params *) octx->kernel_params;

    const uint32_t n_threads = kparams->n_threads;
    if (!htp_ops_context_set_n_threads_div(octx, n_threads, kparams->div_n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t kernel_type     = kparams->kernel_type;
    uint32_t total_jobs      = kparams->total_jobs;
    uint32_t jobs_per_thread = kparams->jobs_per_thread;
    if (octx->ctx->mdev.count > 1) {
        total_jobs      = n_tokens * kparams->n_chunks;
        jobs_per_thread = fastdiv(total_jobs + n_threads - 1, &kparams->div_n_threads);

        // mdev split can leave too few jobs per thread for the dma pipeline
        if (kernel_type == HTP_DSV4_HC_POST_KERNEL_DMA && jobs_per_thread < 2*kparams->dma_block_chunks) {
            kernel_type = HTP_DSV4_HC_POST_KERNEL_L2;
        }
    }

    const bool use_dma = kernel_type == HTP_DSV4_HC_POST_KERNEL_DMA;

    // the dma kernel moves x, residual and dst through 64-bit dma addresses
    if (!use_dma && (htp_tensor_is_extended(x) || htp_tensor_is_extended(residual) || htp_tensor_is_extended(dst))) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct dsv4_hc_post_context dctx = {
        .octx            = octx,
        .kparams         = kparams,
        .token_start     = token_start,
        .total_jobs      = total_jobs,
        .jobs_per_thread = jobs_per_thread,
        .vtcm            = (uint8_t *) octx->ctx->vtcm_base,
    };

    FARF(HIGH, "dsv4-hc-post: (%ux%u) (%ux%ux%u) -> (%ux%ux%u) : comb %u dma %d tokens %u:%u n-threads %u\n",
         x->ne[0], x->ne[1], residual->ne[0], residual->ne[1], residual->ne[2],
         dst->ne[0], dst->ne[1], dst->ne[2], kparams->has_comb, use_dma, token_start, token_start + n_tokens, n_threads);

    work_queue_run(octx->ctx->work_queue,
                   use_dma ? dsv4_hc_post_thread_dma_f32 : dsv4_hc_post_thread_l2_f32, &dctx,
                   n_threads);

    return HTP_STATUS_OK;
}
