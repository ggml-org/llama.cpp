#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dma-queue.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "dsv4-hc-ops.h"

// ---------------------------------------------------------------------------
// DSV4_HC_PRE
// ---------------------------------------------------------------------------

// hvx_vec_fast_sigmoid_f32_guard_2it with fewer live constants and predicates
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

    // Newton-Raphson with 2 iterations
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
    uint32_t jobs_per_hvx;
    float scale;
    uint8_t * vtcm;
    uint32_t dma_block_chunks;
    uint32_t dma_row_stride;
    uint32_t dma_buffer_size;
};

struct dsv4_hc_block {
    uint32_t it;
    uint32_t chunk;
    uint32_t count;
};

static inline struct dsv4_hc_block dsv4_hc_pre_block_make(
        const struct dsv4_hc_pre_context * dctx,
        uint32_t job,
        uint32_t end) {
    const uint32_t local_token = fastdiv(job, &dctx->kparams->div_n_chunks);
    const uint32_t chunk       = job - local_token*dctx->n_chunks;
    const uint32_t count       = hex_smin(dctx->dma_block_chunks, hex_smin(dctx->n_chunks - chunk, end - job));
    return (struct dsv4_hc_block) {
        .it    = dctx->token_start + local_token,
        .chunk = chunk,
        .count = count,
    };
}

static inline void dsv4_hc_pre_push_plain(
        dma_queue * dma_q,
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_block * block,
        uint8_t * spad) {
    const struct htp_tensor * x       = dctx->octx->src[0];
    const struct htp_tensor * weights = dctx->octx->src[1];
    const uint32_t row_stride         = dctx->dma_row_stride;
    const uint32_t row_size           = block->count * VLEN;

    dma_queue_push(dma_q, dma_make_data(spad, x->data + (block->it*x->nb[2] + block->chunk*VLEN)), row_stride, x->nb[1], row_size, 4);
    dma_queue_push(dma_q, dma_make_data(spad + 5*row_stride, weights->data + (block->it*weights->nb[1])), 16, 16, 16, 1);
}

static inline void dsv4_hc_pre_push_gated(
        dma_queue * dma_q,
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_block * block,
        uint8_t * spad) {
    const struct htp_tensor * x       = dctx->octx->src[0];
    const struct htp_tensor * g       = dctx->octx->src[1];
    const uint32_t row_stride         = dctx->dma_row_stride;
    const uint32_t row_size           = block->count * VLEN;

    dma_queue_push(dma_q, dma_make_data(spad, x->data + (block->it*x->nb[2] + block->chunk*VLEN)), row_stride, x->nb[1], row_size, 4);
    dma_queue_push(dma_q, dma_make_data(spad + 4*row_stride, g->data + (block->it*g->nb[2] + block->chunk*VLEN)), row_stride, g->nb[1], row_size, 4);
}

static __attribute__((noinline)) void dsv4_hc_pre_plain_block_f32(
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_block * block,
        const uint8_t * x_spad,
        uint8_t * d_spad,
        const float * wb) {
    const uint32_t row_stride = dctx->dma_row_stride;
    const HVX_Vector scale_v  = hvx_vec_splat_f32(dctx->scale);
    const HVX_Vector w0       = hvx_vec_splat_f32(wb[0]);
    const HVX_Vector w1       = hvx_vec_splat_f32(wb[1]);
    const HVX_Vector w2       = hvx_vec_splat_f32(wb[2]);
    const HVX_Vector w3       = hvx_vec_splat_f32(wb[3]);

    for (uint32_t c = 0; c < block->count; ++c) {
        const uint8_t * xb = x_spad + c*VLEN;

        HVX_Vector x0 = *(const HVX_Vector *) xb;
        HVX_Vector x1 = *(const HVX_Vector *) (xb + row_stride);
        HVX_Vector x2 = *(const HVX_Vector *) (xb + 2*row_stride);
        HVX_Vector x3 = *(const HVX_Vector *) (xb + 3*row_stride);

        HVX_Vector acc = hvx_vec_mul_f32_f32(x0, w0);
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x1, w1));
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x2, w2));
        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(x3, w3));

        *(HVX_Vector *) (d_spad + c*VLEN) = hvx_vec_mul_f32_f32(acc, scale_v);
    }
}

static __attribute__((noinline)) void dsv4_hc_pre_gated_block_f32(
        const struct dsv4_hc_pre_context * dctx,
        const struct dsv4_hc_block * block,
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

static void dsv4_hc_pre_thread_plain_f32(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * dst    = octx->dst;
    const uint64_t            d_data = dst->data;

    const uint32_t row_stride  = dctx->dma_row_stride;
    const uint32_t buffer_size = dctx->dma_buffer_size;
    const uint32_t start       = dctx->jobs_per_hvx * ith;
    const uint32_t end         = hex_smin(start + dctx->jobs_per_hvx, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = th->dma_q;
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_block block = dsv4_hc_pre_block_make(dctx, prefetch_job, end);
        // empty dst writeback keeps pop order dst, x, weights in loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 4*row_stride), block.count*VLEN, block.count*VLEN, 0, 0);
        dsv4_hc_pre_push_plain(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = th->trace;

    uint32_t job  = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_block block = dsv4_hc_pre_block_make(dctx, job, end);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        const float * wb = (const float *) (spad[curr] + 5*row_stride);
        uint8_t * d_spad = spad[curr] + 4*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        dsv4_hc_pre_plain_block_f32(dctx, &block, spad[curr], d_spad, wb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const uint64_t d_dst = d_data + (block.it*dst->nb[1] + block.chunk*VLEN);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), block.count*VLEN, block.count*VLEN, block.count*VLEN, 1);

        if (prefetch_job < end) {
            const struct dsv4_hc_block next = dsv4_hc_pre_block_make(dctx, prefetch_job, end);
            dsv4_hc_pre_push_plain(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

static void dsv4_hc_pre_thread_gated_f32(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_pre_context * dctx = (struct dsv4_hc_pre_context *) data;
    struct htp_ops_context *     octx = dctx->octx;

    const struct htp_tensor * dst    = octx->dst;
    const uint64_t            d_data = dst->data;

    const uint32_t row_stride  = dctx->dma_row_stride;
    const uint32_t buffer_size = dctx->dma_buffer_size;
    const uint32_t start       = dctx->jobs_per_hvx * ith;
    const uint32_t end         = hex_smin(start + dctx->jobs_per_hvx, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = th->dma_q;
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_block block = dsv4_hc_pre_block_make(dctx, prefetch_job, end);
        // empty dst writeback keeps pop order dst, x, g in loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 8*row_stride), block.count*VLEN, block.count*VLEN, 0, 0);
        dsv4_hc_pre_push_gated(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = th->trace;

    uint32_t job  = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_block block = dsv4_hc_pre_block_make(dctx, job, end);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        uint8_t * d_spad = spad[curr] + 8*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        dsv4_hc_pre_gated_block_f32(dctx, &block, spad[curr], spad[curr] + 4*row_stride, d_spad);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const uint64_t d_dst = d_data + (block.it*dst->nb[1] + block.chunk*VLEN);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), block.count*VLEN, block.count*VLEN, block.count*VLEN, 1);

        if (prefetch_job < end) {
            const struct dsv4_hc_block next = dsv4_hc_pre_block_make(dctx, prefetch_job, end);
            dsv4_hc_pre_push_gated(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

int op_dsv4_hc_pre(struct htp_ops_context * octx) {
    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    const struct htp_dsv4_hc_pre_kernel_params * kparams =
        (const struct htp_dsv4_hc_pre_kernel_params *) octx->kernel_params;

    const uint32_t n_hvx = kparams->n_hvx;
    if (!htp_ops_context_set_n_hvx_div(octx, n_hvx, kparams->div_n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t token_start     = 0;
    uint32_t n_tokens        = x->ne[2];
    uint32_t total_jobs      = kparams->total_jobs;
    uint32_t jobs_per_hvx    = kparams->jobs_per_hvx;
    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_tokens, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        token_start     = range.start;
        n_tokens        = range.count;
        total_jobs      = n_tokens * kparams->n_chunks;
        jobs_per_hvx    = fastdiv(total_jobs + n_hvx - 1, &kparams->div_n_hvx);
    }

    if (n_tokens == 0) {
        return HTP_STATUS_OK;
    }

    float scale;
    memcpy(&scale, octx->op_params, sizeof(scale));

    struct dsv4_hc_pre_context dctx = {
        .octx             = octx,
        .kparams          = kparams,
        .token_start      = token_start,
        .n_chunks         = kparams->n_chunks,
        .total_jobs       = total_jobs,
        .jobs_per_hvx     = jobs_per_hvx,
        .scale            = scale,
        .vtcm             = (uint8_t *) octx->ctx->vtcm_base,
        .dma_block_chunks = kparams->dma_block_chunks,
        .dma_row_stride   = kparams->dma_row_stride,
        .dma_buffer_size  = kparams->dma_buffer_size,
    };

    work_queue_func_t worker = NULL;
    switch (kparams->kernel_type) {
        case HTP_DSV4_HC_PRE_KERNEL_PLAIN: worker = dsv4_hc_pre_thread_plain_f32; break;
        case HTP_DSV4_HC_PRE_KERNEL_GATED: worker = dsv4_hc_pre_thread_gated_f32; break;
        default: return HTP_STATUS_INVAL_PARAMS;
    }

    FARF(HIGH, "dsv4-hc-pre: (%ux%ux%u) -> (%ux%u) : ktype %u tokens %u:%u nhvx %u\n",
         x->ne[0], x->ne[1], x->ne[2], dst->ne[0], dst->ne[1], kparams->kernel_type,
         token_start, token_start + n_tokens, n_hvx);

    work_queue_run(octx->ctx->hvx_queue, worker, &dctx, n_hvx);
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
    uint32_t jobs_per_hvx;
    uint8_t * vtcm;
};

static inline struct dsv4_hc_block dsv4_hc_post_block_make(
        const struct dsv4_hc_post_context * dctx,
        uint32_t job,
        uint32_t end,
        uint32_t block_chunks) {
    const uint32_t local_token = fastdiv(job, &dctx->kparams->div_n_chunks);
    const uint32_t chunk       = job - local_token*dctx->kparams->n_chunks;
    const uint32_t count       = hex_smin(block_chunks, hex_smin(dctx->kparams->n_chunks - chunk, end - job));
    return (struct dsv4_hc_block) {
        .it    = dctx->token_start + local_token,
        .chunk = chunk,
        .count = count,
    };
}

static __attribute__((noinline)) void dsv4_hc_post_identity_block_f32(
        const uint8_t * xb,
        const uint8_t * rb,
        size_t r_stride,
        uint8_t * db,
        size_t d_stride,
        uint32_t count,
        const float * p) {
    for (uint32_t id = 0; id < 4; ++id) {
        const HVX_Vector pv = hvx_vec_splat_f32(p[id]);
        const uint8_t *  rr = rb + id*r_stride;
        uint8_t *        dd = db + id*d_stride;

        for (uint32_t i = 0; i < count; ++i) {
            const HVX_Vector xv = *(const HVX_Vector *) (xb + i*VLEN);
            const HVX_Vector rv = *(const HVX_Vector *) (rr + i*VLEN);
            *(HVX_Vector *) (dd + i*VLEN) = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(xv, pv), rv);
        }
    }
}

static __attribute__((noinline)) void dsv4_hc_post_comb_block_f32(
        const uint8_t * xb,
        const uint8_t * rb,
        size_t r_stride,
        uint8_t * db,
        size_t d_stride,
        uint32_t count,
        const float * p,
        const float * c) {
    for (uint32_t id = 0; id < 4; ++id) {
        const HVX_Vector pv = hvx_vec_splat_f32(p[id]);
        const HVX_Vector c0 = hvx_vec_splat_f32(c[id]);
        const HVX_Vector c1 = hvx_vec_splat_f32(c[4 + id]);
        const HVX_Vector c2 = hvx_vec_splat_f32(c[8 + id]);
        const HVX_Vector c3 = hvx_vec_splat_f32(c[12 + id]);
        uint8_t * dd = db + id*d_stride;

        for (uint32_t i = 0; i < count; ++i) {
            const uint8_t * r = rb + i*VLEN;
            HVX_Vector acc = hvx_vec_mul_f32_f32(*(const HVX_Vector *) (xb + i*VLEN), pv);
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) r, c0));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (r + r_stride), c1));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (r + 2*r_stride), c2));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (r + 3*r_stride), c3));
            *(HVX_Vector *) (dd + i*VLEN) = acc;
        }
    }
}

static inline void dsv4_hc_post_push_identity(
        dma_queue * dma_q,
        const struct dsv4_hc_post_context * dctx,
        const struct dsv4_hc_block * block,
        uint8_t * spad) {
    const struct htp_tensor * x        = dctx->octx->src[0];
    const struct htp_tensor * residual = dctx->octx->src[1];
    const struct htp_tensor * post     = dctx->octx->src[2];
    const uint32_t row_stride          = dctx->kparams->dma_row_stride;
    const uint32_t row_size            = block->count * VLEN;

    dma_queue_push(dma_q, dma_make_data(spad, residual->data + (block->it*residual->nb[2] + block->chunk*VLEN)),
                   row_stride, residual->nb[1], row_size, 4);
    dma_queue_push(dma_q, dma_make_data(spad + 4*row_stride, x->data + (block->it*x->nb[1] + block->chunk*VLEN)),
                   row_stride, row_size, row_size, 1);
    dma_queue_push(dma_q, dma_make_data(spad + 9*row_stride, post->data + (block->it*post->nb[1])),
                   16, 16, 16, 1);
}

static inline void dsv4_hc_post_push_comb(
        dma_queue * dma_q,
        const struct dsv4_hc_post_context * dctx,
        const struct dsv4_hc_block * block,
        uint8_t * spad) {
    const struct htp_tensor * comb = dctx->octx->src[3];
    const uint32_t row_stride      = dctx->kparams->dma_row_stride;

    dsv4_hc_post_push_identity(dma_q, dctx, block, spad);
    dma_queue_push(dma_q, dma_make_data(spad + 9*row_stride + 64, comb->data + (block->it*comb->nb[2])),
                   16, comb->nb[1], 16, 4);
}

static void dsv4_hc_post_thread_identity_f32(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_post_context * dctx = (struct dsv4_hc_post_context *) data;
    struct htp_ops_context *      octx = dctx->octx;

    const struct htp_tensor * dst    = octx->dst;
    const uint64_t            d_data = dst->data;

    const uint32_t block_chunks = dctx->kparams->dma_block_chunks;
    const uint32_t row_stride   = dctx->kparams->dma_row_stride;
    const uint32_t buffer_size  = dctx->kparams->dma_buffer_size;
    const uint32_t start        = dctx->jobs_per_hvx * ith;
    const uint32_t end          = hex_smin(start + dctx->jobs_per_hvx, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = th->dma_q;
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_block block = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
        // empty dst writeback keeps pop order dst, residual, x, post in loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 5*row_stride), dst->nb[1], row_stride, 0, 0);
        dsv4_hc_post_push_identity(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = th->trace;

    uint32_t job  = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_block block = dsv4_hc_post_block_make(dctx, job, end, block_chunks);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        const float * p      = (const float *) (spad[curr] + 9*row_stride);
        uint8_t *     d_spad = spad[curr] + 5*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        dsv4_hc_post_identity_block_f32(spad[curr] + 4*row_stride, spad[curr], row_stride, d_spad, row_stride, block.count, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const uint64_t d_dst = d_data + (block.it*dst->nb[2] + block.chunk*VLEN);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), dst->nb[1], row_stride, block.count*VLEN, 4);

        if (prefetch_job < end) {
            const struct dsv4_hc_block next = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
            dsv4_hc_post_push_identity(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

static void dsv4_hc_post_thread_comb_f32(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct dsv4_hc_post_context * dctx = (struct dsv4_hc_post_context *) data;
    struct htp_ops_context *      octx = dctx->octx;

    const struct htp_tensor * dst    = octx->dst;
    const uint64_t            d_data = dst->data;

    const uint32_t block_chunks = dctx->kparams->dma_block_chunks;
    const uint32_t row_stride   = dctx->kparams->dma_row_stride;
    const uint32_t buffer_size  = dctx->kparams->dma_buffer_size;
    const uint32_t start        = dctx->jobs_per_hvx * ith;
    const uint32_t end          = hex_smin(start + dctx->jobs_per_hvx, dctx->total_jobs);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*buffer_size*ith,
        dctx->vtcm + 2*buffer_size*ith + buffer_size,
    };

    dma_queue * dma_q = th->dma_q;
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_block block = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
        // empty dst writeback keeps pop order dst, residual, x, post, comb in loop
        dma_queue_push(dma_q, dma_make_data(d_data, spad[i] + 5*row_stride), dst->nb[1], row_stride, 0, 0);
        dsv4_hc_post_push_comb(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = th->trace;

    uint32_t job  = start;
    uint32_t curr = 0;
    while (job < end) {
        const struct dsv4_hc_block block = dsv4_hc_post_block_make(dctx, job, end, block_chunks);

        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        const float * p      = (const float *) (spad[curr] + 9*row_stride);
        const float * c      = (const float *) (spad[curr] + 9*row_stride + 64);
        uint8_t *     d_spad = spad[curr] + 5*row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        dsv4_hc_post_comb_block_f32(spad[curr] + 4*row_stride, spad[curr], row_stride, d_spad, row_stride, block.count, p, c);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        const uint64_t d_dst = d_data + (block.it*dst->nb[2] + block.chunk*VLEN);
        dma_queue_push(dma_q, dma_make_data(d_dst, d_spad), dst->nb[1], row_stride, block.count*VLEN, 4);

        if (prefetch_job < end) {
            const struct dsv4_hc_block next = dsv4_hc_post_block_make(dctx, prefetch_job, end, block_chunks);
            dsv4_hc_post_push_comb(dma_q, dctx, &next, spad[curr]);
            prefetch_job += next.count;
        }

        job += block.count;
        curr ^= 1;
    }

    dma_queue_flush(dma_q);

    (void) nth;
}

int op_dsv4_hc_post(struct htp_ops_context * octx) {
    const struct htp_tensor * x        = octx->src[0];
    const struct htp_tensor * residual = octx->src[1];
    const struct htp_tensor * dst      = octx->dst;

    const struct htp_dsv4_hc_post_kernel_params * kparams =
        (const struct htp_dsv4_hc_post_kernel_params *) octx->kernel_params;

    const uint32_t n_hvx = kparams->n_hvx;
    if (!htp_ops_context_set_n_hvx_div(octx, n_hvx, kparams->div_n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t token_start     = 0;
    uint32_t n_tokens        = x->ne[1];
    uint32_t total_jobs      = kparams->total_jobs;
    uint32_t jobs_per_hvx    = kparams->jobs_per_hvx;
    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_tokens, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        token_start     = range.start;
        n_tokens        = range.count;
        total_jobs      = n_tokens * kparams->n_chunks;
        jobs_per_hvx    = fastdiv(total_jobs + n_hvx - 1, &kparams->div_n_hvx);
    }

    if (n_tokens == 0) {
        return HTP_STATUS_OK;
    }

    struct dsv4_hc_post_context dctx = {
        .octx            = octx,
        .kparams         = kparams,
        .token_start     = token_start,
        .total_jobs      = total_jobs,
        .jobs_per_hvx    = jobs_per_hvx,
        .vtcm            = (uint8_t *) octx->ctx->vtcm_base,
    };

    work_queue_func_t worker = NULL;
    switch (kparams->kernel_type) {
        case HTP_DSV4_HC_POST_KERNEL_IDENTITY: worker = dsv4_hc_post_thread_identity_f32; break;
        case HTP_DSV4_HC_POST_KERNEL_COMB:     worker = dsv4_hc_post_thread_comb_f32;     break;
        default: return HTP_STATUS_INVAL_PARAMS;
    }

    FARF(HIGH, "dsv4-hc-post: (%ux%u) (%ux%ux%u) -> (%ux%ux%u) : ktype %u tokens %u:%u nhvx %u\n",
         x->ne[0], x->ne[1], residual->ne[0], residual->ne[1], residual->ne[2],
         dst->ne[0], dst->ne[1], dst->ne[2], kparams->kernel_type,
         token_start, token_start + n_tokens, n_hvx);

    work_queue_run(octx->ctx->hvx_queue, worker, &dctx, n_hvx);
    return HTP_STATUS_OK;
}
