#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <string.h>

#include "hex-common.h"
#include "hex-profile.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-types.h"
#include "hvx-utils.h"

struct htp_solve_tri_context {
    struct htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

static inline void solve_tri_row_scalar(const float * A_row,
                                        const float * B_row,
                                        float *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        float         inv_diag) {
    for (uint32_t col = col0; col < col0 + coln; ++col) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

static inline HVX_Vector hvx_load_partial_f32(const float * src, uint32_t n) {
    HVX_Vector v = *((const HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * sizeof(float));
    return Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

static inline void solve_tri_row_hvx(const float * A_row,
                                     const float * B_row,
                                     float *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     float         inv_diag) {
    const bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    for (uint32_t t = 0; t < row; ++t) {
        const float   a         = A_row[t];
        const float * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((const HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    const float * b_row_col = B_row + col0;
    float *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((const HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((void *) x_out_col, coln * sizeof(float), out_v);
}

// Batch-level thread: each job is one full batch.
static void solve_tri_batch_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_solve_tri_context * sctx = (struct htp_solve_tri_context *) data;
    struct htp_ops_context *       octx = sctx->octx;

    const struct htp_tensor * src0 = octx->src[0];  // A
    const struct htp_tensor * src1 = octx->src[1];  // B
    const struct htp_tensor * dst  = octx->dst;     // X

    const uint32_t n = src0->ne[0];
    const uint32_t k = src1->ne[0];

    const uint32_t ne02 = src0->ne[2];

    const uint32_t col_block = VLEN_FP32;
    const uint32_t k_full    = (k / col_block) * col_block;

    const uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    const uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    for (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        const uint32_t i03 = batch / ne02;
        const uint32_t i02 = batch - i03 * ne02;

        const float * A_batch =
            (const float *) ((const uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        const float * B_batch =
            (const float *) ((const uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        float * X_batch = (float *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        for (uint32_t row = 0; row < n; ++row) {
            const float   diag     = A_batch[row * n + row];
            const float   inv_diag = 1.0f / diag;
            const float * A_row    = A_batch + row * n;
            const float * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            for (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            if (col0 < k) {
                const uint32_t coln = k - col0;
                if (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } else {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(HIGH, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
static void solve_tri_chunk_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_solve_tri_context * sctx = (struct htp_solve_tri_context *) data;
    struct htp_ops_context *       octx = sctx->octx;

    const struct htp_tensor * src0 = octx->src[0];  // A
    const struct htp_tensor * src1 = octx->src[1];  // B
    const struct htp_tensor * dst  = octx->dst;     // X

    const uint32_t n = src0->ne[0];
    const uint32_t k = src1->ne[0];

    const uint32_t ne02 = src0->ne[2];

    const uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    const uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    for (uint32_t job = start_job; job < end_job; ++job) {
        const uint32_t batch = job / sctx->k_chunks;
        const uint32_t chunk = job - batch * sctx->k_chunks;

        const uint32_t i03 = batch / ne02;
        const uint32_t i02 = batch - i03 * ne02;

        const float * A_batch =
            (const float *) ((const uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        const float * B_batch =
            (const float *) ((const uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        float * X_batch = (float *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        const uint32_t col0 = chunk * sctx->col_block;
        const uint32_t coln = MIN(sctx->col_block, k - col0);

        for (uint32_t row = 0; row < n; ++row) {
            const float diag     = A_batch[row * n + row];
            const float inv_diag = 1.0f / diag;

            const float * A_row = A_batch + row * n;
            const float * B_row = B_batch + row * k;

            if (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } else {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(HIGH, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

int op_solve_tri(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];  // A
    const struct htp_tensor * src1 = octx->src[1];  // B
    const struct htp_tensor * dst  = octx->dst;     // X

    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    if (src0->ne[0] != src0->ne[1]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (src0->ne[1] != src1->ne[1]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t k = src1->ne[0];

    const uint32_t col_block     = VLEN_FP32;
    const uint32_t k_chunks      = (k + col_block - 1) / col_block;
    const uint32_t total_batches = src0->ne[2] * src0->ne[3];
    const bool     batched       = total_batches >= (uint32_t) octx->n_threads;

    FARF(HIGH, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    if (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = total_batches;

        if (octx->ctx->mdev.count > 1) {
            const uint32_t batch_size = dst->nb[2];
            const uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(batch_size, HEX_L2_LINE_SIZE)) : 1;
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, htp_tensor_mdev_data_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            job_start = range.start;
            njobs     = range.count;
        }

        if (njobs == 0) {
            return HTP_STATUS_OK;
        }

        // Batch-level parallelism
        const uint32_t n_threads = octx->n_threads;

        struct htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } else {
        // Chunk-level parallelism
        const uint32_t total_jobs = total_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = total_jobs;

        if (octx->ctx->mdev.count > 1) {
            const bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0);
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            job_start = range.start;
            njobs     = range.count;
        }

        if (njobs == 0) {
            return HTP_STATUS_OK;
        }

        const uint32_t n_threads = octx->n_threads;

        struct htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    return HTP_STATUS_OK;
}
