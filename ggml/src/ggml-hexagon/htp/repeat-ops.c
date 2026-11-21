#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>

#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "repeat-ops.h"

struct htp_repeat_context {
    struct htp_ops_context *               octx;
    const struct htp_repeat_kernel_params * kparams;

    uint32_t nrows_per_hvx;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_size;
};

static void repeat_job_per_hvx(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_repeat_context * rctx = (const struct htp_repeat_context *) data;
    struct htp_ops_context * octx = rctx->octx;
    const struct htp_repeat_kernel_params * kparams = rctx->kparams;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t ne00 = src->ne[0];
    const uint32_t ne01 = src->ne[1];
    const uint32_t ne02 = src->ne[2];
    const uint32_t ne03 = src->ne[3];

    const uint32_t nb00 = src->nb[0];
    const uint32_t nb01 = src->nb[1];
    const uint32_t nb02 = src->nb[2];
    const uint32_t nb03 = src->nb[3];

    const uint32_t ne0 = dst->ne[0];
    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb0 = dst->nb[0];
    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t nr0 = kparams->nr0;

    const size_t row_bytes = ne00 * rctx->type_size;

    const uint32_t row_start = rctx->row_start + rctx->nrows_per_hvx * ith;
    const uint32_t row_end   = MIN(row_start + rctx->nrows_per_hvx, rctx->row_start + rctx->total_dst_rows);
    if (row_start >= row_end) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    for (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        const uint32_t dst_row_div_ne1 = fastdiv(dst_row, &kparams->div_ne1);
        const uint32_t i1              = dst_row - dst_row_div_ne1 * ne1;
        const uint32_t i3              = fastdiv(dst_row, &kparams->div_ne12);
        const uint32_t i2              = dst_row_div_ne1 - i3 * ne2;

        // Map to source indices (tiling)
        const uint32_t k1 = fastmodulo(i1, ne01, &kparams->div_ne01);
        const uint32_t k2 = fastmodulo(i2, ne02, &kparams->div_ne02);
        const uint32_t k3 = fastmodulo(i3, ne03, &kparams->div_ne03);

        const uint8_t * src_row = (const uint8_t *) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        for (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(HIGH, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

static inline HVX_Vector repeat_clamp_f16_finite(HVX_Vector v) {
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    const HVX_Vector inf     = hvx_vec_splat_f32(INFINITY);
    const HVX_VectorPred finite = Q6_Q_vcmp_gt_VsfVsf(inf, hvx_vec_abs_f32(v));
    const HVX_VectorPred high = Q6_Q_and_QQ(finite, Q6_Q_vcmp_gt_VsfVsf(v, max_f16));
    const HVX_VectorPred low  = Q6_Q_and_QQ(finite, Q6_Q_vcmp_gt_VsfVsf(min_f16, v));
    v = Q6_V_vmux_QVV(high, max_f16, v);
    return Q6_V_vmux_QVV(low, min_f16, v);
}

static inline void repeat_row_f32_to_f16(uint8_t * restrict dst_ptr, const uint8_t * restrict src_ptr, uint32_t n) {
    const HVX_UVector * restrict vsrc = (const HVX_UVector *) src_ptr;
    HVX_UVector       * restrict vdst = (HVX_UVector *) dst_ptr;

    const uint32_t n_vec = n / 64;
    const uint32_t n_rem = n % 64;

    for (uint32_t i = 0; i < n_vec; i++) {
        const HVX_Vector v0 = repeat_clamp_f16_finite(vsrc[2 * i + 0]);
        const HVX_Vector v1 = repeat_clamp_f16_finite(vsrc[2 * i + 1]);
        vdst[i] = hvx_vec_f32_to_f16(v0, v1);
    }

    if (n_rem > 0) {
        const HVX_Vector v0 = repeat_clamp_f16_finite(vsrc[2 * n_vec + 0]);
        const HVX_Vector v1 = (n_rem > 32) ? repeat_clamp_f16_finite(vsrc[2 * n_vec + 1]) : Q6_V_vzero();
        const HVX_Vector v  = hvx_vec_f32_to_f16(v0, v1);
        hvx_vec_store_u(&vdst[n_vec], n_rem * sizeof(__fp16), v);
    }
}

static void repeat_job_f32_to_f16(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_repeat_context * rctx = (const struct htp_repeat_context *) data;
    struct htp_ops_context * octx = rctx->octx;
    const struct htp_repeat_kernel_params * kparams = rctx->kparams;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t ne00 = src->ne[0];
    const uint32_t ne01 = src->ne[1];
    const uint32_t ne02 = src->ne[2];
    const uint32_t ne03 = src->ne[3];

    const uint32_t nb01 = src->nb[1];
    const uint32_t nb02 = src->nb[2];
    const uint32_t nb03 = src->nb[3];

    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];

    const uint32_t nb0 = dst->nb[0];
    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t nr0 = kparams->nr0;

    const uint32_t row_start = rctx->row_start + rctx->nrows_per_hvx * ith;
    const uint32_t row_end   = MIN(row_start + rctx->nrows_per_hvx, rctx->row_start + rctx->total_dst_rows);
    if (row_start >= row_end) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    for (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        const uint32_t dst_row_div_ne1 = fastdiv(dst_row, &kparams->div_ne1);
        const uint32_t i1              = dst_row - dst_row_div_ne1 * ne1;
        const uint32_t i3              = fastdiv(dst_row, &kparams->div_ne12);
        const uint32_t i2              = dst_row_div_ne1 - i3 * ne2;

        const uint32_t k1 = fastmodulo(i1, ne01, &kparams->div_ne01);
        const uint32_t k2 = fastmodulo(i2, ne02, &kparams->div_ne02);
        const uint32_t k3 = fastmodulo(i3, ne03, &kparams->div_ne03);

        const uint8_t * src_row = (const uint8_t *) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        for (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            repeat_row_f32_to_f16(dst_ptr, src_row, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(HIGH, "repeat-f32-f16 %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

int op_repeat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if ((src0->type != HTP_TYPE_F32 && src0->type != HTP_TYPE_F16) ||
        (dst->type != src0->type && !(src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16))) {
        FARF(ERROR, "repeat: unsupported types %u -> %u\n", src0->type, dst->type);
        return HTP_STATUS_NO_SUPPORT;
    }

    const size_t type_size = (dst->type == HTP_TYPE_F16) ? 2 : 4;

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_repeat_kernel_params * kparams =
        (const struct htp_repeat_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t total_dst_rows  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const size_t dst_row_size = dst->ne[0] * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx = octx->n_hvx;

    struct htp_repeat_context rctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_hvx    = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div),
        .total_dst_rows   = nrows,
        .row_start        = row_start,
        .type_size        = type_size,
    };

    FARF(HIGH, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->nr0, kparams->nr1, kparams->nr2, kparams->nr3);

    work_queue_func_t worker = (src0->type == dst->type) ? repeat_job_per_hvx : repeat_job_f32_to_f16;
    work_queue_run(octx->ctx->hvx_queue, worker, &rctx, n_hvx);

    return HTP_STATUS_OK;
}
