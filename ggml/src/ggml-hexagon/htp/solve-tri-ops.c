#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-types.h"
#include "hvx-utils.h"

struct htp_solve_tri_context {
    struct htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

#if __HVX_ARCH__ > 75
static inline HVX_Vector hvx_add_f32_vec(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_vadd_VsfVsf(a, b);
}

static inline HVX_Vector hvx_sub_f32_vec(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_vsub_VsfVsf(a, b);
}

static inline HVX_Vector hvx_mul_f32_vec(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_vmpy_VsfVsf(a, b);
}
#else
static inline HVX_Vector hvx_add_f32_vec(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

static inline HVX_Vector hvx_sub_f32_vec(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

static inline HVX_Vector hvx_mul_f32_vec(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
#endif

static inline HVX_Vector hvx_load_partial_f32(const float * src, uint32_t n) {
    HVX_VectorAlias tmp;
    memset(&tmp, 0, sizeof(tmp));
    memcpy(tmp.fp32, src, n * sizeof(float));
    return tmp.v;
}

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
        sum_v          = hvx_add_f32_vec(sum_v, hvx_mul_f32_vec(x_v, a_v));
    }

    const float * b_row_col = B_row + col0;
    float *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((const HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_mul_f32_vec(hvx_sub_f32_vec(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((void *) x_out_col, coln * sizeof(float), out_v);
}

static void solve_tri_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_solve_tri_context * sctx = (struct htp_solve_tri_context *) data;
    struct htp_ops_context *       octx = sctx->octx;

    const struct htp_tensor * src0 = octx->src[0]; // A
    const struct htp_tensor * src1 = octx->src[1]; // B
    const struct htp_tensor * dst  = octx->dst;    // X

    const uint32_t n = src0->ne[0];
    const uint32_t k = src1->ne[0];

    const uint32_t ne02 = src0->ne[2];

    const uint32_t start_job = sctx->jobs_per_thread * ith;
    const uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->total_jobs);

    for (uint32_t job = start_job; job < end_job; ++job) {
        const uint32_t batch = job / sctx->k_chunks;
        const uint32_t chunk = job - batch * sctx->k_chunks;

        const uint32_t i03 = batch / ne02;
        const uint32_t i02 = batch - i03 * ne02;

        const uint32_t col0 = chunk * sctx->col_block;
        const uint32_t coln = MIN(sctx->col_block, k - col0);

        const float * A_batch =
            (const float *) ((const uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        const float * B_batch =
            (const float *) ((const uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        float * X_batch = (float *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        const bool use_hvx = (coln >= 8);

        for (uint32_t row = 0; row < n; ++row) {
            const float diag     = A_batch[row * n + row];
            const float inv_diag = 1.0f / diag;

            const float * A_row = A_batch + row * n;
            const float * B_row = B_batch + row * k;

            if (use_hvx) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } else {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    (void) nth;
}

int op_solve_tri(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0]; // A
    const struct htp_tensor * src1 = octx->src[1]; // B
    const struct htp_tensor * dst  = octx->dst;    // X

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

    const uint32_t n = src0->ne[0];
    const uint32_t k = src1->ne[0];

    // Keep behavior aligned with CPU implementation contract.
    for (uint32_t i03 = 0; i03 < src0->ne[3]; ++i03) {
        for (uint32_t i02 = 0; i02 < src0->ne[2]; ++i02) {
            const float * A_batch =
                (const float *) ((const uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
            for (uint32_t i = 0; i < n; ++i) {
                if (A_batch[i * n + i] == 0.0f) {
                    FARF(ERROR, "solve-tri: zero diagonal at batch (%u,%u), row %u", i02, i03, i);
                    return HTP_STATUS_INVAL_PARAMS;
                }
            }
        }
    }

    if ((uintptr_t) dst->data != (uintptr_t) src1->data) {
        const size_t dst_nbytes = dst->nb[3] * dst->ne[3];
        memcpy((void *) (uintptr_t) dst->data, (const void *) (uintptr_t) src1->data, dst_nbytes);
    }

    const uint32_t col_block     = VLEN_FP32;
    const uint32_t k_chunks      = (k + col_block - 1) / col_block;
    const uint32_t total_batches = src0->ne[2] * src0->ne[3];
    const uint32_t total_jobs    = total_batches * k_chunks;

    const uint32_t n_threads = MIN(octx->n_threads, MAX(total_jobs, 1));

    struct htp_solve_tri_context sctx = {
        .octx            = octx,
        .jobs_per_thread = (total_jobs + n_threads - 1) / n_threads,
        .total_jobs      = total_jobs,
        .k_chunks        = k_chunks,
        .col_block       = col_block,
    };

    worker_pool_run_func(octx->ctx->worker_pool, solve_tri_thread_f32, &sctx, n_threads);

    return HTP_STATUS_OK;
}
