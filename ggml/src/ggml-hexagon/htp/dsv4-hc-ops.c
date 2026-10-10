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

static inline HVX_Vector dsv4_hc_weight_f32(
        const struct htp_tensor * weights,
        const uint8_t * weights_data,
        uint32_t i0,
        uint32_t ih,
        uint32_t it,
        int gated) {
    if (gated) {
        const uint8_t * gate = weights_data + i0*weights->nb[0] + ih*weights->nb[1] + it*weights->nb[2];
        const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
        const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
        const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);
        return hvx_vec_fast_sigmoid_f32_guard(*(const HVX_UVector *) gate, one, max_exp, min_exp);
    }

    const float w = *(const float *) (weights_data + ih*weights->nb[0] + it*weights->nb[1]);
    return hvx_vec_splat_f32(w);
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

static void dsv4_hc_pre_thread_f32(unsigned int nth, unsigned int ith, void * data) {
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

    for (uint32_t job = start; job < end; ++job) {
        const uint32_t local_token = job / dctx->n_chunks;
        const uint32_t chunk       = job - local_token*dctx->n_chunks;
        const uint32_t it          = dctx->token_start + local_token;
        const uint32_t i0          = chunk * VLEN_FP32;
        const uint32_t n           = MIN(VLEN_FP32, n_embd - i0);

        if (n != VLEN_FP32) {
            dsv4_hc_pre_tail_f32(x, weights, dst, x_data, weights_data, dst_data, i0, it, n, dctx->scale, dctx->gated);
            continue;
        }

        const uint8_t * xb = x_data + i0*x->nb[0] + it*x->nb[2];
        uint8_t *       db = dst_data + i0*dst->nb[0] + it*dst->nb[1];

        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t ih = 0; ih < 4; ++ih) {
            HVX_Vector xv = *(const HVX_UVector *) (xb + ih*x->nb[1]);
            HVX_Vector wv = dsv4_hc_weight_f32(weights, weights_data, i0, ih, it, dctx->gated);
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(xv, wv));
        }

        acc = hvx_vec_mul_f32_f32(acc, scale_v);
        hvx_vec_store_u(db, n * sizeof(float), acc);
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

    work_queue_run(octx->ctx->work_queue, dsv4_hc_pre_thread_f32, &dctx, n_threads);

    return HTP_STATUS_OK;
}
