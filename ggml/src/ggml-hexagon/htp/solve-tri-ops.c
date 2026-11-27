#include <HAP_farf.h>
#include <string.h>

#include "dma-queue.h"
#include "hex-common.h"
#include "htp-trace.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hvx-types.h"
#include "hvx-utils.h"
#include "solve-tri-ops.h"

struct htp_solve_tri_context {
    struct htp_ops_context *                   octx;
    const struct htp_solve_tri_kernel_params * kparams;
    uint8_t *                                  vtcm_a;
    uint8_t *                                  vtcm_bx;
    size_t                                     a_row_aligned;
    size_t                                     a_bytes_per_hvx;
    size_t                                     bx_bytes_per_hvx;
    uint32_t                                   jobs_per_hvx;
    uint32_t                                   total_jobs;
    uint32_t                                   job_start;
    uint32_t                                   k_chunks;
    uint32_t                                   col_block;
    uint32_t                                   a_full;
    uint8_t                                    a_f16;
    uint8_t                                    b_f16;
    uint8_t                                    x_f16;
};

static inline HVX_Vector hvx_solve_tri_clamp_f16(HVX_Vector v) {
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    return Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v));
}

static void __attribute__((noinline)) solve_tri_widen_bx_f16(HVX_Vector * restrict bx, uint32_t n) {
    for (uint32_t row = 0; row < n; ++row) {
        bx[row] = Q6_V_lo_W(hvx_vec_f16_to_f32(bx[row]));
    }
}

static void __attribute__((noinline)) solve_tri_pack_bx_f16(HVX_Vector * restrict bx, uint32_t n) {
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t row = 0; row < n; ++row) {
        bx[row] = hvx_vec_f32_to_f16(hvx_solve_tri_clamp_f16(bx[row]), zero);
    }
}

static void __attribute__((noinline)) solve_tri_widen_a_row_f16(uint8_t * restrict row_buf, uint32_t len) {
    HVX_Vector * restrict vecs = (HVX_Vector *) row_buf;
    const uint32_t nvec_f32    = (len + VLEN_FP32 - 1) >> 5;
    const uint32_t nvec_f16    = (len + VLEN_FP16 - 1) >> 6;
    for (int32_t vi = (int32_t) nvec_f16 - 1; vi >= 0; --vi) {
        const HVX_VectorPair p = hvx_vec_f16_to_f32(vecs[vi]);
        if ((uint32_t) (2 * vi + 1) < nvec_f32) {
            vecs[2 * vi + 1] = Q6_V_hi_W(p);
        }
        vecs[2 * vi] = Q6_V_lo_W(p);
    }
}

static void __attribute__((noinline)) solve_tri_widen_a_full_f16(uint8_t * restrict vtcm_a, size_t a_row_aligned, uint32_t n) {
    for (uint32_t row = 0; row < n; ++row) {
        solve_tri_widen_a_row_f16(vtcm_a + (size_t) row * a_row_aligned, row + 1);
    }
}

static inline void solve_tri_row_vtcm_hvx(
    const HVX_Vector * restrict a_row_vecs,
    HVX_Vector * restrict       bx,
    uint32_t                    row
) {
    HVX_Vector sum_v = Q6_V_vzero();
    uint32_t   t     = 0;

    for (; t + VLEN_FP32 <= row; t += VLEN_FP32) {
        const HVX_Vector a_vec = a_row_vecs[t >> 5];
        for (uint32_t j = 0; j < VLEN_FP32; ++j) {
            const HVX_Vector a_v = Q6_V_vsplat_R(Q6_R_vextract_VR(a_vec, j * 4));
            sum_v = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(bx[t + j], a_v));
        }
    }

    if (t < row) {
        const HVX_Vector a_vec = a_row_vecs[t >> 5];
        const uint32_t   rem   = row - t;
        for (uint32_t j = 0; j < rem; ++j) {
            const HVX_Vector a_v = Q6_V_vsplat_R(Q6_R_vextract_VR(a_vec, j * 4));
            sum_v = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(bx[t + j], a_v));
        }
    }

    const int32_t diag_bits = Q6_R_vextract_VR(a_row_vecs[row >> 5], (row & 31u) * 4);
    float         diag;
    memcpy(&diag, &diag_bits, sizeof(float));

    const HVX_Vector inv_diag_v = hvx_vec_splat_f32(1.0f / diag);
    bx[row] = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(bx[row], sum_v), inv_diag_v);
}

static void __attribute__((noinline)) solve_tri_solve_full_f32(
    const uint8_t * restrict vtcm_a,
    HVX_Vector * restrict    vtcm_bx,
    size_t                   a_row_aligned,
    uint32_t                 n
) {
    for (uint32_t row = 0; row < n; ++row) {
        const HVX_Vector * a_row_vecs = (const HVX_Vector *) (vtcm_a + (size_t) row * a_row_aligned);
        solve_tri_row_vtcm_hvx(a_row_vecs, vtcm_bx, row);
    }
}

static void __attribute__((noinline)) solve_tri_solve_row_f32(
    const HVX_Vector * restrict a_row_vecs,
    HVX_Vector * restrict       vtcm_bx,
    uint32_t                    row
) {
    solve_tri_row_vtcm_hvx(a_row_vecs, vtcm_bx, row);
}

static void solve_tri_col_chunk_a_full(
    dma_queue *               dma_q,
    struct htp_thread_trace * tr,
    const uint8_t *           vtcm_a,
    HVX_Vector *              vtcm_bx,
    dma_addr_t                B_batch,
    dma_addr_t                X_batch,
    uint32_t                  nb11,
    uint32_t                  dst_nb1,
    size_t                    a_row_aligned,
    uint32_t                  n,
    uint32_t                  col0,
    uint32_t                  coln,
    uint8_t                   b_f16,
    uint8_t                   x_f16,
    uint16_t                  trace_id
) {
    const size_t b_elem_size = b_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t x_elem_size = x_f16 ? sizeof(__fp16) : sizeof(float);

    dma_queue_push(dma_q, dma_make_data(vtcm_bx, B_batch + (size_t) col0 * b_elem_size), VLEN, nb11, (size_t) coln * b_elem_size, n);
    dma_queue_pop(dma_q);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, trace_id);
    if (b_f16) {
        solve_tri_widen_bx_f16(vtcm_bx, n);
    }
    solve_tri_solve_full_f32(vtcm_a, vtcm_bx, a_row_aligned, n);
    if (x_f16) {
        solve_tri_pack_bx_f16(vtcm_bx, n);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, trace_id);

    dma_queue_push(dma_q, dma_make_data(X_batch + (size_t) col0 * x_elem_size, vtcm_bx), dst_nb1, VLEN, (size_t) coln * x_elem_size, n);
    dma_queue_pop(dma_q);
}

static void solve_tri_col_chunk_a_stream(
    dma_queue *               dma_q,
    struct htp_thread_trace * tr,
    uint8_t *                 vtcm_a_row,
    HVX_Vector *              vtcm_bx,
    dma_addr_t                A_batch,
    dma_addr_t                B_batch,
    dma_addr_t                X_batch,
    uint32_t                  nb01,
    uint32_t                  nb11,
    uint32_t                  dst_nb1,
    uint32_t                  n,
    uint32_t                  col0,
    uint32_t                  coln,
    uint8_t                   a_f16,
    uint8_t                   b_f16,
    uint8_t                   x_f16,
    uint16_t                  trace_id
) {
    const size_t a_elem_size = a_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t b_elem_size = b_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t x_elem_size = x_f16 ? sizeof(__fp16) : sizeof(float);

    dma_queue_push(dma_q, dma_make_data(vtcm_bx, B_batch + (size_t) col0 * b_elem_size), VLEN, nb11, (size_t) coln * b_elem_size, n);
    dma_queue_pop(dma_q);

    if (b_f16) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, trace_id);
        solve_tri_widen_bx_f16(vtcm_bx, n);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, trace_id);
    }

    for (uint32_t row = 0; row < n; ++row) {
        const size_t a_bytes = (size_t) (row + 1) * a_elem_size;
        dma_sync_read(dma_q, vtcm_a_row, A_batch + (size_t) row * nb01, a_bytes);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, trace_id);
        if (a_f16) {
            solve_tri_widen_a_row_f16(vtcm_a_row, row + 1);
        }
        solve_tri_solve_row_f32((const HVX_Vector *) vtcm_a_row, vtcm_bx, row);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, trace_id);
    }

    if (x_f16) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, trace_id);
        solve_tri_pack_bx_f16(vtcm_bx, n);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, trace_id);
    }

    dma_queue_push(dma_q, dma_make_data(X_batch + (size_t) col0 * x_elem_size, vtcm_bx), dst_nb1, VLEN, (size_t) coln * x_elem_size, n);
    dma_queue_pop(dma_q);
}

// Batch-level thread: each job is one full batch.
static void solve_tri_batch_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_solve_tri_context *             sctx    = (struct htp_solve_tri_context *) data;
    struct htp_ops_context *                   octx    = sctx->octx;
    const struct htp_solve_tri_kernel_params * kparams = sctx->kparams;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t n    = src0->ne[0];
    const uint32_t k    = src1->ne[0];
    const uint32_t ne02 = src0->ne[2];

    const uint32_t start_batch = sctx->job_start + sctx->jobs_per_hvx * ith;
    const uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_hvx, sctx->job_start + sctx->total_jobs);
    if (start_batch >= end_batch) {
        return;
    }

    uint8_t *     vtcm_a        = sctx->vtcm_a + ith * sctx->a_bytes_per_hvx;
    HVX_Vector *  vtcm_bx       = (HVX_Vector *) (sctx->vtcm_bx + ith * sctx->bx_bytes_per_hvx);
    const size_t  a_row_aligned = sctx->a_row_aligned;
    const uint8_t a_f16         = sctx->a_f16;
    const uint8_t b_f16         = sctx->b_f16;
    const uint8_t x_f16         = sctx->x_f16;
    const size_t  a_row_bytes   = (size_t) n * (a_f16 ? sizeof(__fp16) : sizeof(float));

    dma_queue *               dma_q = th->dma_q;
    struct htp_thread_trace * tr    = th->trace;

    for (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        const uint32_t i03 = fastdiv(batch, &kparams->div_ne02);
        const uint32_t i02 = batch - i03 * ne02;

        const dma_addr_t A_batch = src0->data + (size_t) i02 * src0->nb[2] + (size_t) i03 * src0->nb[3];
        const dma_addr_t B_batch = src1->data + (size_t) i02 * src1->nb[2] + (size_t) i03 * src1->nb[3];
        const dma_addr_t X_batch = dst->data  + (size_t) i02 * dst->nb[2]  + (size_t) i03 * dst->nb[3];

        if (sctx->a_full) {
            dma_queue_push(dma_q, dma_make_data(vtcm_a, A_batch), a_row_aligned, src0->nb[1], a_row_bytes, n);
            dma_queue_pop(dma_q);
            if (a_f16) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) batch);
                solve_tri_widen_a_full_f16(vtcm_a, a_row_aligned, n);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) batch);
            }

            for (uint32_t col0 = 0; col0 < k; col0 += VLEN_FP32) {
                const uint32_t coln = MIN(VLEN_FP32, k - col0);
                solve_tri_col_chunk_a_full(dma_q, tr, vtcm_a, vtcm_bx, B_batch, X_batch,
                                           src1->nb[1], dst->nb[1], a_row_aligned, n, col0, coln, b_f16, x_f16, (uint16_t) batch);
            }
        } else {
            for (uint32_t col0 = 0; col0 < k; col0 += VLEN_FP32) {
                const uint32_t coln = MIN(VLEN_FP32, k - col0);
                solve_tri_col_chunk_a_stream(dma_q, tr, vtcm_a, vtcm_bx, A_batch, B_batch, X_batch,
                                             src0->nb[1], src1->nb[1], dst->nb[1], n, col0, coln, a_f16, b_f16, x_f16, (uint16_t) batch);
            }
        }
    }

    FARF(HIGH, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
static void solve_tri_chunk_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_solve_tri_context *             sctx    = (struct htp_solve_tri_context *) data;
    struct htp_ops_context *                   octx    = sctx->octx;
    const struct htp_solve_tri_kernel_params * kparams = sctx->kparams;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t n    = src0->ne[0];
    const uint32_t k    = src1->ne[0];
    const uint32_t ne02 = src0->ne[2];

    const uint32_t start_job = sctx->job_start + sctx->jobs_per_hvx * ith;
    const uint32_t end_job   = MIN(start_job + sctx->jobs_per_hvx, sctx->job_start + sctx->total_jobs);
    if (start_job >= end_job) {
        return;
    }

    uint8_t *     vtcm_a        = sctx->vtcm_a + ith * sctx->a_bytes_per_hvx;
    HVX_Vector *  vtcm_bx       = (HVX_Vector *) (sctx->vtcm_bx + ith * sctx->bx_bytes_per_hvx);
    const size_t  a_row_aligned = sctx->a_row_aligned;
    const uint8_t a_f16         = sctx->a_f16;
    const uint8_t b_f16         = sctx->b_f16;
    const uint8_t x_f16         = sctx->x_f16;
    const size_t  a_row_bytes   = (size_t) n * (a_f16 ? sizeof(__fp16) : sizeof(float));

    dma_queue *               dma_q      = th->dma_q;
    struct htp_thread_trace * tr         = th->trace;
    uint32_t                  last_batch = UINT32_MAX;

    for (uint32_t job = start_job; job < end_job; ++job) {
        const uint32_t batch = fastdiv(job, &kparams->div_k_chunks);
        const uint32_t chunk = job - batch * sctx->k_chunks;

        const uint32_t i03 = fastdiv(batch, &kparams->div_ne02);
        const uint32_t i02 = batch - i03 * ne02;

        const dma_addr_t A_batch = src0->data + (size_t) i02 * src0->nb[2] + (size_t) i03 * src0->nb[3];
        const dma_addr_t B_batch = src1->data + (size_t) i02 * src1->nb[2] + (size_t) i03 * src1->nb[3];
        const dma_addr_t X_batch = dst->data  + (size_t) i02 * dst->nb[2]  + (size_t) i03 * dst->nb[3];

        const uint32_t col0 = chunk * sctx->col_block;
        const uint32_t coln = MIN(sctx->col_block, k - col0);

        if (sctx->a_full) {
            if (batch != last_batch) {
                dma_queue_push(dma_q, dma_make_data(vtcm_a, A_batch), a_row_aligned, src0->nb[1], a_row_bytes, n);
                dma_queue_pop(dma_q);
                if (a_f16) {
                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) job);
                    solve_tri_widen_a_full_f16(vtcm_a, a_row_aligned, n);
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) job);
                }
                last_batch = batch;
            }
            solve_tri_col_chunk_a_full(dma_q, tr, vtcm_a, vtcm_bx, B_batch, X_batch,
                                       src1->nb[1], dst->nb[1], a_row_aligned, n, col0, coln, b_f16, x_f16, (uint16_t) job);
        } else {
            solve_tri_col_chunk_a_stream(dma_q, tr, vtcm_a, vtcm_bx, A_batch, B_batch, X_batch,
                                         src0->nb[1], src1->nb[1], dst->nb[1], n, col0, coln, a_f16, b_f16, x_f16, (uint16_t) job);
        }
    }

    FARF(HIGH, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

int op_solve_tri(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if ((src0->type != HTP_TYPE_F32 && src0->type != HTP_TYPE_F16) ||
        (src1->type != HTP_TYPE_F32 && src1->type != HTP_TYPE_F16) ||
        (dst->type  != HTP_TYPE_F32 && dst->type  != HTP_TYPE_F16)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_solve_tri_kernel_params * kparams =
        (const struct htp_solve_tri_kernel_params *) octx->kernel_params;

    const uint32_t col_block     = kparams->col_block;
    const uint32_t k_chunks      = kparams->k_chunks;
    const uint32_t total_batches = kparams->total_batches;
    const bool     batched       = (kparams->batched != 0);
    const uint8_t  a_f16         = (src0->type == HTP_TYPE_F16) ? 1 : 0;
    const uint8_t  b_f16         = (src1->type == HTP_TYPE_F16) ? 1 : 0;
    const uint8_t  x_f16         = (dst->type  == HTP_TYPE_F16) ? 1 : 0;

    FARF(HIGH, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    uint32_t job_start = 0;
    uint32_t njobs     = batched ? total_batches : kparams->total_jobs;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            njobs, 1,
            octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        job_start = range.start;
        njobs     = range.count;
    }

    if (njobs == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, njobs))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx = octx->n_hvx;

    struct htp_solve_tri_vtcm_layout layout;
    htp_solve_tri_vtcm_layout_build(&layout, src0->ne[0], kparams->a_full, n_hvx);
    if (octx->ctx->vtcm_size < layout.total_bytes) {
        FARF(ERROR, "solve-tri: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = octx->ctx->vtcm_base;

    struct htp_solve_tri_context sctx = {
        .octx             = octx,
        .kparams          = kparams,
        .vtcm_a           = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_a),
        .vtcm_bx          = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_bx),
        .a_row_aligned    = layout.a_row_aligned,
        .a_bytes_per_hvx  = layout.a_bytes_per_hvx,
        .bx_bytes_per_hvx = layout.bx_bytes_per_hvx,
        .jobs_per_hvx     = fastdiv(njobs + n_hvx - 1, &octx->n_hvx_div),
        .total_jobs       = njobs,
        .job_start        = job_start,
        .k_chunks         = k_chunks,
        .col_block        = col_block,
        .a_full           = kparams->a_full,
        .a_f16            = a_f16,
        .b_f16            = b_f16,
        .x_f16            = x_f16,
    };

    work_queue_func_t worker = batched ? solve_tri_batch_thread : solve_tri_chunk_thread;
    work_queue_run(octx->ctx->hvx_queue, worker, &sctx, n_hvx);

    return HTP_STATUS_OK;
}
