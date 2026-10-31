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

// Both ops stream token rows through the same double-buffered DMA pipeline.
// A job is one 32-element chunk of a token, a block is up to dma_block_chunks jobs of one token.
// Each block moves its input rows DDR -> VTCM, is computed in VTCM, and its dst rows go VTCM -> DDR.

struct dsv4_hc_dma_block {
    uint32_t it;    // token
    uint32_t chunk; // first chunk in the token
    uint32_t count; // number of chunks
    uint32_t size;  // bytes per row, the last chunk of a token can be partial
};

struct dsv4_hc_dma_stream {
    dma_addr_t data;
    uint32_t   nb_row;
    uint32_t   nb_token;
    uint32_t   n_rows;
    uint32_t   spad_offset;
};

struct dsv4_hc_context;

typedef void (*dsv4_hc_compute_t)(const struct dsv4_hc_context * dctx, const struct dsv4_hc_dma_block * block, uint8_t * spad);

struct dsv4_hc_context {
    struct htp_ops_context * octx;
    const void *             kparams;
    dsv4_hc_compute_t        compute;
    float                    scale;

    struct dsv4_hc_dma_stream src[2];
    uint32_t                  n_src;
    struct dsv4_hc_dma_stream dst;

    uint32_t job_start;
    uint32_t job_end;
    uint32_t jobs_per_thread;
    uint32_t n_chunks;
    uint32_t row_size;
    uint32_t block_chunks;
    uint32_t row_stride;
    uint32_t buffer_size;
    const struct fastdiv_values * div_n_chunks;
    uint8_t * vtcm;
};

// blocks never cross tokens
static inline struct dsv4_hc_dma_block dsv4_hc_dma_block_make(const struct dsv4_hc_context * dctx, uint32_t job, uint32_t end) {
    const uint32_t it    = fastdiv(job, dctx->div_n_chunks);
    const uint32_t chunk = job - it*dctx->n_chunks;
    const uint32_t count = hex_smin(dctx->block_chunks, hex_smin(dctx->n_chunks - chunk, end - job));
    return (struct dsv4_hc_dma_block) {
        .it    = it,
        .chunk = chunk,
        .count = count,
        .size  = hex_smin(count*VLEN, dctx->row_size - chunk*VLEN),
    };
}

static inline void dsv4_hc_dma_push_src(dma_queue * dma_q, const struct dsv4_hc_context * dctx, const struct dsv4_hc_dma_block * block, uint8_t * spad) {
    const uint32_t off = block->chunk*VLEN;
    for (uint32_t i = 0; i < dctx->n_src; ++i) {
        const struct dsv4_hc_dma_stream * s = &dctx->src[i];
        dma_queue_push(dma_q, dma_make_data(spad + s->spad_offset, s->data + (block->it*s->nb_token + off)),
                       dctx->row_stride, s->nb_row, block->size, s->n_rows);
    }
}

static inline void dsv4_hc_dma_push_dst(dma_queue * dma_q, const struct dsv4_hc_context * dctx, const struct dsv4_hc_dma_block * block, uint8_t * spad) {
    const struct dsv4_hc_dma_stream * d = &dctx->dst;
    dma_queue_push(dma_q, dma_make_data(d->data + (block->it*d->nb_token + block->chunk*VLEN), spad + d->spad_offset),
                   d->nb_row, dctx->row_stride, block->size, d->n_rows);
}

static void dsv4_hc_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct dsv4_hc_context * dctx = (const struct dsv4_hc_context *) data;
    struct htp_ops_context *       octx = dctx->octx;

    const uint32_t start = dctx->job_start + dctx->jobs_per_thread * ith;
    const uint32_t end   = hex_smin(start + dctx->jobs_per_thread, dctx->job_end);
    if (start >= end) {
        return;
    }

    uint8_t * spad[2] = {
        dctx->vtcm + 2*dctx->buffer_size*ith,
        dctx->vtcm + 2*dctx->buffer_size*ith + dctx->buffer_size,
    };

    dma_queue * dma_q = octx->ctx->dma[ith];

    // an empty dst writeback ahead of the first two blocks keeps the loop pop order dst(N-2), src(N)
    uint32_t prefetch_job = start;
    for (uint32_t i = 0; i < 2 && prefetch_job < end; ++i) {
        const struct dsv4_hc_dma_block block = dsv4_hc_dma_block_make(dctx, prefetch_job, end);
        dma_queue_push(dma_q, dma_make_data(dctx->dst.data, spad[i] + dctx->dst.spad_offset), 0, 0, 0, 0);
        dsv4_hc_dma_push_src(dma_q, dctx, &block, spad[i]);
        prefetch_job += block.count;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t n_pops = 1 + dctx->n_src;

    uint32_t job      = start;
    uint32_t curr     = 0;
    uint32_t n_blocks = 0;
    while (job < end) {
        const struct dsv4_hc_dma_block block = dsv4_hc_dma_block_make(dctx, job, end);

        for (uint32_t i = 0; i < n_pops; ++i) {
            dma_queue_pop(dma_q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);
        dctx->compute(dctx, &block, spad[curr]);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) block.it);

        dsv4_hc_dma_push_dst(dma_q, dctx, &block, spad[curr]);

        if (prefetch_job < end) {
            const struct dsv4_hc_dma_block next = dsv4_hc_dma_block_make(dctx, prefetch_job, end);
            dsv4_hc_dma_push_src(dma_q, dctx, &next, spad[curr]);
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

// dst is only written by dma, so devices can split jobs without cache line alignment
static int dsv4_hc_run(struct dsv4_hc_context * dctx, uint32_t total_jobs, uint32_t n_threads, struct fastdiv_values div_n_threads) {
    struct htp_ops_context * octx = dctx->octx;

    uint32_t job_start = 0;
    uint32_t n_jobs    = total_jobs;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        job_start = range.start;
        n_jobs    = range.count;
    }

    if (n_jobs == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_threads_div(octx, n_threads, div_n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    dctx->job_start = job_start;
    dctx->job_end   = job_start + n_jobs;
    dctx->vtcm      = (uint8_t *) octx->ctx->vtcm_base;

    work_queue_run(octx->ctx->work_queue, dsv4_hc_thread, dctx, n_threads);

    return HTP_STATUS_OK;
}

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

// one x row per pass, partial sums stay in the vtcm dst row
static void dsv4_hc_pre_plain_f32(const struct dsv4_hc_context * dctx, const struct dsv4_hc_dma_block * block, uint8_t * spad) {
    const struct htp_dsv4_hc_pre_kernel_params * kparams = (const struct htp_dsv4_hc_pre_kernel_params *) dctx->kparams;

    const struct htp_tensor * w  = dctx->octx->src[1];
    const uint8_t *           wb = (const uint8_t *) w->data + block->it*w->nb[1];

    const uint32_t n_hc       = kparams->n_hc;
    const uint32_t row_stride = dctx->row_stride;
    const uint32_t count      = block->count;

    HVX_Vector * restrict vd = (HVX_Vector *) (spad + kparams->dma_d_offset);

    for (uint32_t ih = 0; ih < n_hc; ++ih) {
        const HVX_Vector * restrict vx = (const HVX_Vector *) (spad + ih*row_stride);
        const HVX_Vector            wv = hvx_vec_splat_f32(*(const float *) (wb + ih*w->nb[0]));

        if (ih == 0) {
            for (uint32_t c = 0; c < count; ++c) {
                vd[c] = hvx_vec_mul_f32_f32(vx[c], wv);
            }
        } else {
            for (uint32_t c = 0; c < count; ++c) {
                vd[c] = hvx_vec_add_f32_f32(vd[c], hvx_vec_mul_f32_f32(vx[c], wv));
            }
        }
    }

    const HVX_Vector scale_v = hvx_vec_splat_f32(dctx->scale);
    for (uint32_t c = 0; c < count; ++c) {
        vd[c] = hvx_vec_mul_f32_f32(vd[c], scale_v);
    }
}

// four x and gate rows per pass keep four independent sigmoid chains in flight,
// the remaining rows go one per pass, partial sums stay in the vtcm dst row
static void dsv4_hc_pre_gated_f32(const struct dsv4_hc_context * dctx, const struct dsv4_hc_dma_block * block, uint8_t * spad) {
    const struct htp_dsv4_hc_pre_kernel_params * kparams = (const struct htp_dsv4_hc_pre_kernel_params *) dctx->kparams;

    const uint32_t   n_hc       = kparams->n_hc;
    const uint32_t   row_stride = dctx->row_stride;
    const uint8_t *  g_spad     = spad + kparams->dma_g_offset;
    const uint32_t   count      = block->count;
    const HVX_Vector max_exp    = hvx_vec_splat_f32(87.0f);
    const HVX_Vector min_exp    = hvx_vec_splat_f32(-87.0f);

    HVX_Vector * restrict vd = (HVX_Vector *) (spad + kparams->dma_d_offset);

    uint32_t ih = 0;
    for (; ih + 4 <= n_hc; ih += 4) {
        const uint8_t * xb = spad + ih*row_stride;
        const uint8_t * gb = g_spad + ih*row_stride;

        for (uint32_t c = 0; c < count; ++c) {
            const uint8_t * xc = xb + c*VLEN;
            const uint8_t * gc = gb + c*VLEN;

            HVX_Vector acc = hvx_vec_mul_f32_f32(*(const HVX_Vector *) xc, dsv4_hc_sigmoid_f32(*(const HVX_Vector *) gc, max_exp, min_exp));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (xc + row_stride),
                                                               dsv4_hc_sigmoid_f32(*(const HVX_Vector *) (gc + row_stride), max_exp, min_exp)));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (xc + 2*row_stride),
                                                               dsv4_hc_sigmoid_f32(*(const HVX_Vector *) (gc + 2*row_stride), max_exp, min_exp)));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (xc + 3*row_stride),
                                                               dsv4_hc_sigmoid_f32(*(const HVX_Vector *) (gc + 3*row_stride), max_exp, min_exp)));

            vd[c] = ih == 0 ? acc : hvx_vec_add_f32_f32(vd[c], acc);
        }
    }

    for (; ih < n_hc; ++ih) {
        const HVX_Vector * restrict vx = (const HVX_Vector *) (spad + ih*row_stride);
        const HVX_Vector * restrict vg = (const HVX_Vector *) (g_spad + ih*row_stride);

        if (ih == 0) {
            for (uint32_t c = 0; c < count; ++c) {
                vd[c] = hvx_vec_mul_f32_f32(vx[c], dsv4_hc_sigmoid_f32(vg[c], max_exp, min_exp));
            }
        } else {
            for (uint32_t c = 0; c < count; ++c) {
                vd[c] = hvx_vec_add_f32_f32(vd[c], hvx_vec_mul_f32_f32(vx[c], dsv4_hc_sigmoid_f32(vg[c], max_exp, min_exp)));
            }
        }
    }

    const HVX_Vector scale_v = hvx_vec_splat_f32(dctx->scale);
    for (uint32_t c = 0; c < count; ++c) {
        vd[c] = hvx_vec_mul_f32_f32(vd[c], scale_v);
    }
}

int op_dsv4_hc_pre(struct htp_ops_context * octx) {
    const struct htp_tensor * x       = octx->src[0];
    const struct htp_tensor * weights = octx->src[1];
    const struct htp_tensor * dst     = octx->dst;

    const struct htp_dsv4_hc_pre_kernel_params * kparams =
        (const struct htp_dsv4_hc_pre_kernel_params *) octx->kernel_params;

    const bool gated = kparams->kernel_type == HTP_DSV4_HC_PRE_KERNEL_GATED;

    // plain weights are read through pointers, x, gate and dst move through 64-bit dma addresses
    if (!gated && htp_tensor_is_extended(weights)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    float scale;
    memcpy(&scale, &octx->op_params[0], sizeof(float));

    struct dsv4_hc_context dctx = {
        .octx            = octx,
        .kparams         = kparams,
        .compute         = gated ? dsv4_hc_pre_gated_f32 : dsv4_hc_pre_plain_f32,
        .scale           = scale,
        .src             = {
            { x->data,       x->nb[1],       x->nb[2],       kparams->n_hc, 0                     },
            { weights->data, weights->nb[1], weights->nb[2], kparams->n_hc, kparams->dma_g_offset },
        },
        .n_src           = gated ? 2 : 1,
        .dst             = { dst->data, 0, dst->nb[1], 1, kparams->dma_d_offset },
        .jobs_per_thread = kparams->jobs_per_thread,
        .n_chunks        = kparams->n_chunks,
        .row_size        = kparams->row_size,
        .block_chunks    = kparams->dma_block_chunks,
        .row_stride      = kparams->dma_row_stride,
        .buffer_size     = kparams->dma_buffer_size,
        .div_n_chunks    = &kparams->div_n_chunks,
    };

    FARF(HIGH, "dsv4-hc-pre: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : gated %d scale %.6f n-threads %u\n",
         x->ne[0], x->ne[1], x->ne[2], x->ne[3],
         weights->ne[0], weights->ne[1], weights->ne[2], weights->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         gated, scale, kparams->n_threads);

    return dsv4_hc_run(&dctx, kparams->total_jobs, kparams->n_threads, kparams->div_n_threads);
}

// ---------------------------------------------------------------------------
// DSV4_HC_POST
// ---------------------------------------------------------------------------

static inline float dsv4_hc_post_coef(const struct htp_tensor * comb, const uint8_t * cb, uint32_t id, uint32_t is) {
    return cb ? *(const float *) (cb + id*comb->nb[0] + is*comb->nb[1]) : (float) (id == is);
}

// dst[id] = x*post[id] + sum_is residual[is]*comb[id][is], without comb the identity matrix
// one dst row per pass keeps few splats live, inputs are reloaded from vtcm
static void dsv4_hc_post_f32(const struct dsv4_hc_context * dctx, const struct dsv4_hc_dma_block * block, uint8_t * spad) {
    const struct htp_dsv4_hc_post_kernel_params * kparams = (const struct htp_dsv4_hc_post_kernel_params *) dctx->kparams;

    const struct htp_tensor * post = dctx->octx->src[2];
    const struct htp_tensor * comb = dctx->octx->src[3];

    const uint32_t  row_stride = dctx->row_stride;
    const uint32_t  count      = block->count;
    const uint8_t * rb         = spad;
    const uint8_t * xb         = spad + 4*row_stride;
    uint8_t *       db         = spad + 5*row_stride;

    const uint8_t * pb = (const uint8_t *) post->data + block->it*post->nb[1];
    const uint8_t * cb = kparams->has_comb ? (const uint8_t *) comb->data + block->it*comb->nb[2] : NULL;

    for (uint32_t id = 0; id < 4; ++id) {
        const HVX_Vector pv = hvx_vec_splat_f32(*(const float *) (pb + id*post->nb[0]));
        const HVX_Vector c0 = hvx_vec_splat_f32(dsv4_hc_post_coef(comb, cb, id, 0));
        const HVX_Vector c1 = hvx_vec_splat_f32(dsv4_hc_post_coef(comb, cb, id, 1));
        const HVX_Vector c2 = hvx_vec_splat_f32(dsv4_hc_post_coef(comb, cb, id, 2));
        const HVX_Vector c3 = hvx_vec_splat_f32(dsv4_hc_post_coef(comb, cb, id, 3));
        uint8_t * dd = db + id*row_stride;

        for (uint32_t i = 0; i < count; ++i) {
            const uint8_t * r = rb + i*VLEN;
            HVX_Vector acc = hvx_vec_mul_f32_f32(*(const HVX_Vector *) (xb + i*VLEN), pv);
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) r, c0));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (r + row_stride), c1));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (r + 2*row_stride), c2));
            acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(*(const HVX_Vector *) (r + 3*row_stride), c3));
            *(HVX_Vector *) (dd + i*VLEN) = acc;
        }
    }
}

int op_dsv4_hc_post(struct htp_ops_context * octx) {
    const struct htp_tensor * x        = octx->src[0];
    const struct htp_tensor * residual = octx->src[1];
    const struct htp_tensor * post     = octx->src[2];
    const struct htp_tensor * comb     = octx->src[3];
    const struct htp_tensor * dst      = octx->dst;

    const struct htp_dsv4_hc_post_kernel_params * kparams =
        (const struct htp_dsv4_hc_post_kernel_params *) octx->kernel_params;

    // post and comb coefficients are read through pointers, x, residual and dst move through 64-bit dma addresses
    if (htp_tensor_is_extended(post) || htp_tensor_is_extended(comb)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t row_stride = kparams->dma_row_stride;

    struct dsv4_hc_context dctx = {
        .octx            = octx,
        .kparams         = kparams,
        .compute         = dsv4_hc_post_f32,
        .src             = {
            { residual->data, residual->nb[1], residual->nb[2], 4, 0            },
            { x->data,        0,               x->nb[1],        1, 4*row_stride },
        },
        .n_src           = 2,
        .dst             = { dst->data, dst->nb[1], dst->nb[2], 4, 5*row_stride },
        .jobs_per_thread = kparams->jobs_per_thread,
        .n_chunks        = kparams->n_chunks,
        .row_size        = kparams->row_size,
        .block_chunks    = kparams->dma_block_chunks,
        .row_stride      = row_stride,
        .buffer_size     = kparams->dma_buffer_size,
        .div_n_chunks    = &kparams->div_n_chunks,
    };

    FARF(HIGH, "dsv4-hc-post: (%ux%u) (%ux%ux%u) -> (%ux%ux%u) : comb %u n-threads %u\n",
         x->ne[0], x->ne[1], residual->ne[0], residual->ne[1], residual->ne[2],
         dst->ne[0], dst->ne[1], dst->ne[2], kparams->has_comb, kparams->n_threads);

    return dsv4_hc_run(&dctx, kparams->total_jobs, kparams->n_threads, kparams->div_n_threads);
}
