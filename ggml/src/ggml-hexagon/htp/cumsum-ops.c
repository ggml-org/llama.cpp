#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
//#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-types.h"
#include "hvx-utils.h"

// Inclusive prefix-sum (cumulative sum) along dim 0 (ne[0]) for each row.
// dst[i] = src[0] + src[1] + ... + src[i]

struct cumsum_context {
    const float * src_data;
    float       * dst_data;
    uint32_t      ne00;
    size_t        src_row_stride;
    size_t        dst_row_stride;
    uint32_t      rows_per_thread;
    uint32_t      total_rows;
    bool          optimal;
};

#if __HVX_ARCH__ > 75
static inline HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_vadd_VsfVsf(a, b);
}
#else
static inline HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#endif  // __HVX_ARCH__ > 75

static inline HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    const HVX_Vector zero = Q6_V_vsplat_R(0);

    // Scan the raw tile first (without carry)
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));

    // Now add carry to all
    v = hvx_cumsum_vadd(v, carry_in);

    return v;
}

static inline HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    return hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

static inline void hvx_cumsum_row_f32_a(const float * restrict src, float * restrict dst, uint32_t n) {
    const uint32_t nvec  = n / VLEN_FP32;
    HVX_Vector     carry = Q6_V_vsplat_R(0);

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v            = ((const HVX_Vector *) src)[i];
        v                       = hvx_prefix_scan_f32(v, carry);
        ((HVX_Vector *) dst)[i] = v;
        carry                   = hvx_splat_last_f32(v);
    }
}

// General case: handles any n, aligned or unaligned src/dst.
static inline void hvx_cumsum_row_f32(const float * restrict src, float * restrict dst, uint32_t n) {
    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    HVX_Vector carry = Q6_V_vsplat_R(0);

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((const HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_last_f32(v);
    }

    if (nloe) {
        // Scalar carry-out from the last full tile (or 0.0f if no full tiles).
        float acc = hvx_vec_get_f32(carry);
        const float * src_nloe = src + nvec * VLEN_FP32;
        float       * dst_nloe = dst + nvec * VLEN_FP32;
        for (uint32_t i = 0; i < nloe; i++) {
            acc       += src_nloe[i];
            dst_nloe[i] = acc;
        }
    }
}

static void cumsum_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    const struct cumsum_context * cctx = (const struct cumsum_context *) data;

    const uint32_t start_row = cctx->rows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + cctx->rows_per_thread, cctx->total_rows);

    if (start_row >= end_row) {
        return;
    }

    for (uint32_t ir = start_row; ir < end_row; ir++) {
        const float * restrict src_row = cctx->src_data + ir * cctx->src_row_stride;
        float       * restrict dst_row = cctx->dst_data + ir * cctx->dst_row_stride;

        if (cctx->optimal) {
            hvx_cumsum_row_f32_a(src_row, dst_row, cctx->ne00);
        } else {
            hvx_cumsum_row_f32(src_row, dst_row, cctx->ne00);
        }
    }
}

int op_cumsum_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * dst  = &octx->dst;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        const uint32_t ne00       = src0->ne[0];
        const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
        const uint32_t n_threads  = MIN(octx->n_threads, total_rows);

        // opt_path: both src and dst row starts are 128-byte aligned and row
        // length is a multiple of VLEN_FP32 (32 floats = 128 bytes).
        const bool optimal = hex_is_aligned((void *) src0->data, VLEN) &&
                            hex_is_aligned((void *) dst->data,  VLEN) &&
                            !(src0->nb[1] & (VLEN - 1))               &&
                            !(dst->nb[1]  & (VLEN - 1))               &&
                            !(ne00 % VLEN_FP32);

        struct cumsum_context cctx = {
            .src_data        = (const float *) src0->data,
            .dst_data        = (float *)       dst->data,
            .ne00            = ne00,
            .src_row_stride  = src0->nb[1] / sizeof(float),
            .dst_row_stride  = dst->nb[1]  / sizeof(float),
            .rows_per_thread = (total_rows + n_threads - 1) / n_threads,
            .total_rows      = total_rows,
            .optimal         = optimal,
        };

        worker_pool_run_func(octx->ctx->worker_pool, cumsum_thread_f32, &cctx, n_threads);
    }

    return HTP_STATUS_OK;
}

int op_cumsum(struct htp_ops_context * octx) {
    int                 err = HTP_STATUS_OK;
    struct htp_tensor * dst = &octx->dst;

    switch (dst->type) {
        case HTP_TYPE_F32:
            err = op_cumsum_f32(octx);
            break;
        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
