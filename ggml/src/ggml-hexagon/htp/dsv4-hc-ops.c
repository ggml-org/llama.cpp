#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <stdint.h>
#include <string.h>
#include <math.h>

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
    uint32_t token_start;
    uint32_t n_chunks;
    uint32_t total_jobs;
    uint32_t jobs_per_thread;
    float scale;
    int gated;
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

static inline void dsv4_hc_l2fetch_4rows(const uint8_t * p, size_t row_stride, uint32_t i0, uint32_t n_cols) {
    if (row_stride > 0xFFFF) {
        return;
    }
    hex_l2fetch(p + i0*sizeof(float), n_cols*sizeof(float), row_stride, 4);
}

#define DSV4_HC_PF_CHUNKS 16

// prefetch x and gate for the aligned block after the one with chunk c, block does not cross tokens
static inline void dsv4_hc_prefetch_gated(
        const struct htp_tensor * x,
        const struct htp_tensor * g,
        uint32_t it,
        uint32_t c,
        uint32_t n_chunks) {
    uint32_t c0 = (c / DSV4_HC_PF_CHUNKS + 1) * DSV4_HC_PF_CHUNKS;
    if (c0 >= n_chunks) {
        c0 = 0;
        it++;
    }
    const uint32_t i0     = c0 * VLEN_FP32;
    const uint32_t n_cols = MIN(DSV4_HC_PF_CHUNKS * VLEN_FP32, x->ne[0] - i0);
    dsv4_hc_l2fetch_4rows((const uint8_t *) x->data + it*x->nb[2], x->nb[1], i0, n_cols);
    dsv4_hc_l2fetch_4rows((const uint8_t *) g->data + it*g->nb[2], g->nb[1], i0, n_cols);
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
        float scale,
        int gated) {
    for (uint32_t c = 0; c < n; ++c) {
        float sum = 0.0f;
        for (uint32_t ih = 0; ih < 4; ++ih) {
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
    uint32_t       it       = start / n_chunks;
    uint32_t       chunk    = start - it*n_chunks;
    it += dctx->token_start;

    for (uint32_t job = start; job < end; ++it, chunk = 0) {
        const uint32_t chunk_end = MIN(n_chunks, chunk + (end - job));
        job += chunk_end - chunk;

        // plain path is cheap per chunk, so fetch the whole next token
        if (job < end) {
            dsv4_hc_l2fetch_4rows(x_data + (it + 1)*x->nb[2], x->nb[1], 0, MIN(n_embd, (end - job)*VLEN_FP32));
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
                dsv4_hc_pre_tail_f32(x, weights, dst, x_data, weights_data, dst_data, i0, it, n, dctx->scale, 0);
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
    uint32_t       it       = start / n_chunks;
    uint32_t       chunk    = start - it*n_chunks;
    it += dctx->token_start;

    for (uint32_t job = start; job < end; ++it, chunk = 0) {
        const uint32_t chunk_end = MIN(n_chunks, chunk + (end - job));

        for (; chunk < chunk_end; ++chunk, ++job) {
            const uint32_t c_pf = chunk % DSV4_HC_PF_CHUNKS;
            if ((c_pf == 0 || job == start) && job + MIN(DSV4_HC_PF_CHUNKS - c_pf, n_chunks - chunk) < end) {
                dsv4_hc_prefetch_gated(x, weights, it, chunk, n_chunks);
            }

            const uint32_t i0 = chunk * VLEN_FP32;
            const uint32_t n  = MIN(VLEN_FP32, n_embd - i0);

            if (n != VLEN_FP32) {
                dsv4_hc_pre_tail_f32(x, weights, dst, x_data, weights_data, dst_data, i0, it, n, dctx->scale, 1);
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
    if (x->ne[1] != 4 || x->ne[3] != 1 || dst->ne[0] != x->ne[0] || dst->ne[1] != x->ne[2]) {
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
        if (weights->ne[0] != x->ne[0] || weights->ne[1] != 4 || weights->ne[2] != x->ne[2] || weights->ne[3] != 1) {
            return HTP_STATUS_NO_SUPPORT;
        }
    } else {
        if (weights->ne[0] != 4 || weights->ne[1] != x->ne[2] || weights->ne[2] != 1 || weights->ne[3] != 1) {
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

    const uint32_t n_chunks   = (x->ne[0] + VLEN_FP32 - 1) / VLEN_FP32;
    const uint32_t total_jobs = n_tokens * n_chunks;
    const uint32_t n_threads  = MIN(octx->ctx->n_threads, total_jobs);
    if (!htp_ops_context_set_n_threads(octx, n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct dsv4_hc_pre_context dctx = {
        .octx            = octx,
        .token_start     = token_start,
        .n_chunks        = n_chunks,
        .total_jobs      = total_jobs,
        .jobs_per_thread = (total_jobs + n_threads - 1) / n_threads,
        .scale           = scale,
        .gated           = gated,
    };

    if (gated) {
        work_queue_run(octx->ctx->work_queue, dsv4_hc_pre_thread_gated_f32, &dctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, dsv4_hc_pre_thread_plain_f32, &dctx, n_threads);
    }

    return HTP_STATUS_OK;
}
