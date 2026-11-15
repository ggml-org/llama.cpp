#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>
#include <math.h>

#include "dma-queue.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "sum-rows-ops.h"

#define sum_rows_preamble                         \
    const struct htp_tensor *src0 = octx->src[0]; \
    const struct htp_tensor *dst  = octx->dst;    \
                                                  \
    const uint32_t ne00 = src0->ne[0];            \
    const uint32_t ne01 = src0->ne[1];            \
    const uint32_t ne02 = src0->ne[2];            \
    const uint32_t ne03 = src0->ne[3];            \
                                                  \
    const uint32_t nb00 = src0->nb[0];            \
    const uint32_t nb01 = src0->nb[1];            \
    const uint32_t nb02 = src0->nb[2];            \
    const uint32_t nb03 = src0->nb[3];            \
                                                  \
    const uint32_t  ne0 = dst->ne[0];             \
    const uint32_t  ne1 = dst->ne[1];             \
    const uint32_t  ne2 = dst->ne[2];             \
    const uint32_t  ne3 = dst->ne[3];             \
                                                  \
    const uint32_t  nb0 = dst->nb[0];             \
    const uint32_t  nb1 = dst->nb[1];             \
    const uint32_t  nb2 = dst->nb[2];             \
    const uint32_t  nb3 = dst->nb[3];             \

typedef void (*sum_rows_compute_fn_t)(
    const uint8_t * restrict src,
    uint8_t * restrict       dst,
    uint32_t                 ne00,
    uint32_t                 nrows,
    size_t                   src_row_stride
);

struct sum_rows_context {
    struct htp_ops_context                  * octx;
    const struct htp_sum_rows_kernel_params * kparams;
    sum_rows_compute_fn_t                     compute;
    dma_addr_t                                data_src0;
    dma_addr_t                                data_dst;
    uint8_t                                 * vtcm_src0;
    uint8_t                                 * vtcm_dst;
    size_t                                    src0_data_row_size;
    size_t                                    src0_row_size_aligned;
    size_t                                    src0_vtcm_half_size;
    size_t                                    dst_vtcm_half_size;
    size_t                                    vtcm_src0_size_per_hvx;
    size_t                                    vtcm_dst_size_per_hvx;
    size_t                                    dst_elem_size;
    uint32_t                                  row_start;
    uint32_t                                  src0_nrows;
    uint32_t                                  src0_nrows_per_hvx;
    uint32_t                                  block;
};

static inline size_t sum_rows_row_offset(
    uint32_t ir,
    uint32_t ne1,
    uint32_t ne2,
    const struct fastdiv_values * div_ne1,
    const struct fastdiv_values * div_ne2,
    const struct fastdiv_values * div_ne12,
    size_t nb1,
    size_t nb2,
    size_t nb3
) {
    const uint32_t i1         = fastmodulo(ir, ne1, div_ne1);
    const uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    const uint32_t i2         = fastmodulo(ir_div_ne1, ne2, div_ne2);
    const uint32_t i3         = fastdiv(ir, div_ne12);
    return (size_t) i1 * nb1 + (size_t) i2 * nb2 + (size_t) i3 * nb3;
}

static inline uint32_t sum_rows_block_size(
    uint32_t ir,
    uint32_t end_row,
    uint32_t block,
    bool     contig,
    uint32_t ne1,
    const struct fastdiv_values * div_ne1
) {
    uint32_t limit = MIN(block, end_row - ir);
    if (!contig) {
        const uint32_t slice_end = (fastdiv(ir, div_ne1) + 1) * ne1;
        limit = MIN(limit, slice_end - ir);
    }
    return limit;
}

static inline HVX_Vector hvx_vec_clamp_f16_f32(HVX_Vector v_sf) {
    const HVX_Vector max_f16 = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector min_f16 = hvx_vec_splat_f32(-65504.0f);
    return Q6_Vsf_vmax_VsfVsf(min_f16, Q6_Vsf_vmin_VsfVsf(max_f16, v_sf));
}

static inline HVX_Vector hvx_row_reduce_sum_f32(const uint8_t * restrict src, uint32_t num_elems) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const uint32_t nvec = num_elems / VLEN_FP32;
    const uint32_t nloe = num_elems % VLEN_FP32;

    HVX_Vector acc = Q6_V_vzero();
    uint32_t   i   = 0;

    #pragma unroll(4)
    for (; i < nvec; i++) {
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, vsrc[i]);
    }
    if (nloe) {
        const HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * sizeof(float));
        const HVX_Vector     v     = Q6_V_vand_QV(bmask, vsrc[i]);
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, v);
    }

    return Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(acc));
}

static inline HVX_Vector hvx_row_reduce_sum_f16(const uint8_t * restrict src, uint32_t num_elems) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const uint32_t nvec = num_elems / VLEN_FP16;
    const uint32_t nloe = num_elems % VLEN_FP16;

    HVX_Vector acc = Q6_V_vzero();
    uint32_t   i   = 0;

    #pragma unroll(2)
    for (; i < nvec; i++) {
        const HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(vsrc[i]);
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, Q6_V_lo_W(p));
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, Q6_V_hi_W(p));
    }
    if (nloe) {
        const HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * sizeof(__fp16));
        const HVX_Vector     v     = Q6_V_vand_QV(bmask, vsrc[i]);
        const HVX_VectorPair p     = hvx_vec_f16_to_f32_shuff(v);
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, Q6_V_lo_W(p));
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, Q6_V_hi_W(p));
    }

    return Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(acc));
}

static void __attribute__((noinline)) sum_rows_block_f32_f32(
    const uint8_t * restrict src,
    uint8_t * restrict       dst,
    const uint32_t           ne00,
    const uint32_t           nrows,
    const size_t             src_row_stride
) {
    HVX_Vector * restrict v_dst = (HVX_Vector *) dst;
    for (uint32_t b = 0; b < nrows; ++b) {
        const HVX_Vector     sum_sf = hvx_row_reduce_sum_f32(src + b * src_row_stride, ne00);
        const uint32_t       b_off  = (b & 31u) * sizeof(float);
        const HVX_VectorPred b_mask = Q6_Q_vsetq_R(b_off);
        const HVX_VectorPred e_mask = Q6_Q_vsetq2_R(b_off + sizeof(float));
        Q6_vmem_QnRIV(Q6_Q_or_QQn(b_mask, e_mask), &v_dst[b >> 5], sum_sf);
    }
}

static void __attribute__((noinline)) sum_rows_block_f32_f16(
    const uint8_t * restrict src,
    uint8_t * restrict       dst,
    const uint32_t           ne00,
    const uint32_t           nrows,
    const size_t             src_row_stride
) {
    HVX_Vector * restrict v_dst = (HVX_Vector *) dst;
    for (uint32_t b = 0; b < nrows; ++b) {
        const HVX_Vector     sum_sf = hvx_vec_clamp_f16_f32(hvx_row_reduce_sum_f32(src + b * src_row_stride, ne00));
        const HVX_Vector     sum_hf = hvx_vec_f32_to_f16_shuff(sum_sf, sum_sf);
        const uint32_t       b_off  = (b & 63u) * sizeof(__fp16);
        const HVX_VectorPred b_mask = Q6_Q_vsetq_R(b_off);
        const HVX_VectorPred e_mask = Q6_Q_vsetq2_R(b_off + sizeof(__fp16));
        Q6_vmem_QnRIV(Q6_Q_or_QQn(b_mask, e_mask), &v_dst[b >> 6], sum_hf);
    }
}

static void __attribute__((noinline)) sum_rows_block_f16_f32(
    const uint8_t * restrict src,
    uint8_t * restrict       dst,
    const uint32_t           ne00,
    const uint32_t           nrows,
    const size_t             src_row_stride
) {
    HVX_Vector * restrict v_dst = (HVX_Vector *) dst;
    for (uint32_t b = 0; b < nrows; ++b) {
        const HVX_Vector     sum_sf = hvx_row_reduce_sum_f16(src + b * src_row_stride, ne00);
        const uint32_t       b_off  = (b & 31u) * sizeof(float);
        const HVX_VectorPred b_mask = Q6_Q_vsetq_R(b_off);
        const HVX_VectorPred e_mask = Q6_Q_vsetq2_R(b_off + sizeof(float));
        Q6_vmem_QnRIV(Q6_Q_or_QQn(b_mask, e_mask), &v_dst[b >> 5], sum_sf);
    }
}

static void __attribute__((noinline)) sum_rows_block_f16_f16(
    const uint8_t * restrict src,
    uint8_t * restrict       dst,
    const uint32_t           ne00,
    const uint32_t           nrows,
    const size_t             src_row_stride
) {
    HVX_Vector * restrict v_dst = (HVX_Vector *) dst;
    for (uint32_t b = 0; b < nrows; ++b) {
        const HVX_Vector     sum_sf = hvx_vec_clamp_f16_f32(hvx_row_reduce_sum_f16(src + b * src_row_stride, ne00));
        const HVX_Vector     sum_hf = hvx_vec_f32_to_f16_shuff(sum_sf, sum_sf);
        const uint32_t       b_off  = (b & 63u) * sizeof(__fp16);
        const HVX_VectorPred b_mask = Q6_Q_vsetq_R(b_off);
        const HVX_VectorPred e_mask = Q6_Q_vsetq2_R(b_off + sizeof(__fp16));
        Q6_vmem_QnRIV(Q6_Q_or_QQn(b_mask, e_mask), &v_dst[b >> 6], sum_hf);
    }
}

static void sum_rows_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct sum_rows_context * smctx = (const struct sum_rows_context *) data;
    struct htp_ops_context        * octx  = smctx->octx;
    struct htp_thread_trace       * tr    = th->trace;

    sum_rows_preamble;

    const uint32_t src0_nrows_per_hvx = smctx->src0_nrows_per_hvx;
    const uint32_t src0_nrows         = smctx->src0_nrows;
    const uint32_t src0_start_row     = smctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row       = MIN(src0_start_row + src0_nrows_per_hvx, smctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    const dma_addr_t data_src              = smctx->data_src0;
    const dma_addr_t data_dst              = smctx->data_dst;
    uint8_t        * src0_vtcm_data        = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_hvx);
    uint8_t        * dst_vtcm_data         = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_hvx);
    const size_t     src0_data_row_size    = smctx->src0_data_row_size;
    const size_t     src0_row_size_aligned = smctx->src0_row_size_aligned;
    const size_t     src0_vtcm_half_size   = smctx->src0_vtcm_half_size;
    const size_t     dst_vtcm_half_size    = smctx->dst_vtcm_half_size;
    const size_t     dst_elem_size         = smctx->dst_elem_size;
    const uint32_t   BLOCK                 = smctx->block;
    const bool       src0_contig           = smctx->kparams->src0_contig;

    const struct fastdiv_values * div_ne01  = &smctx->kparams->div_ne01;
    const struct fastdiv_values * div_ne02  = &smctx->kparams->div_ne02;
    const struct fastdiv_values * div_ne012 = &smctx->kparams->div_ne012;

    dma_queue * dma_q = th->dma_q;

    for (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        const uint32_t block_size = sum_rows_block_size(ir, src0_end_row, BLOCK, src0_contig, ne01, div_ne01);

        dma_queue_push(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            0, 0, 0, 0);

        const size_t src0_off = src0_contig ? ((size_t) ir * nb01) :
            sum_rows_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_push(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    const sum_rows_compute_fn_t compute = smctx->compute;

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = sum_rows_block_size(ir, src0_end_row, BLOCK, src0_contig, ne01, div_ne01);

        uint8_t * dst_vtcm  = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;
        uint8_t * src0_vtcm = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(src0_vtcm, dst_vtcm, ne00, block_size, src0_row_size_aligned);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        const size_t dst_off   = (size_t) ir * dst_elem_size;
        const size_t dst_bytes = (size_t) block_size * dst_elem_size;
        dma_queue_push(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            dst_bytes, dst_bytes, dst_bytes, 1);

        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = sum_rows_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                                 ne01, div_ne01);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = sum_rows_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                     ne01, div_ne01);
                const size_t src0_pref_off = src0_contig ? ((size_t) pref_ir * nb01) :
                    sum_rows_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_push(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

int op_sum_rows(struct htp_ops_context * octx) {
    sum_rows_preamble;

    const bool src0_f16 = (src0->type == HTP_TYPE_F16);
    const bool dst_f16  = (dst->type  == HTP_TYPE_F16);

    if ((src0->type != HTP_TYPE_F32 && src0->type != HTP_TYPE_F16) ||
        (dst->type  != HTP_TYPE_F32 && dst->type  != HTP_TYPE_F16)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_sum_rows_kernel_params * kparams =
        (const struct htp_sum_rows_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const size_t   src0_elem_size     = src0_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t   dst_elem_size      = dst_f16  ? sizeof(__fp16) : sizeof(float);
    const size_t   src0_data_row_size = (size_t) ne00 * src0_elem_size;
    const size_t   dst_data_row_size  = (size_t) ne0  * dst_elem_size;
    const uint32_t src0_nrows         = ne01 * ne02 * ne03;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) dst_elem_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx        = octx->n_hvx;
    const uint32_t rows_per_hvx = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);

    struct htp_sum_rows_vtcm_layout layout;
    htp_sum_rows_vtcm_layout_build(&layout, src0_data_row_size, dst_elem_size, kparams->block, n_hvx);

    if (octx->ctx->vtcm_size < layout.total_bytes) {
        FARF(ERROR, "sum_rows: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    sum_rows_compute_fn_t compute = NULL;
    if (!src0_f16 && !dst_f16) {
        compute = sum_rows_block_f32_f32;
    } else if (!src0_f16 && dst_f16) {
        compute = sum_rows_block_f32_f16;
    } else if (src0_f16 && !dst_f16) {
        compute = sum_rows_block_f16_f32;
    } else {
        compute = sum_rows_block_f16_f16;
    }

    uint8_t * vtcm_base = octx->ctx->vtcm_base;

    struct sum_rows_context smctx = {
        .octx                   = octx,
        .kparams                = kparams,
        .compute                = compute,
        .data_src0              = (dma_addr_t) src0->data,
        .data_dst               = (dma_addr_t) dst->data,
        .vtcm_src0              = vtcm_base + layout.src0_off,
        .vtcm_dst               = vtcm_base + layout.dst_off,
        .src0_data_row_size     = src0_data_row_size,
        .src0_row_size_aligned  = layout.src0_row_size_aligned,
        .src0_vtcm_half_size    = layout.src0_half_bytes,
        .dst_vtcm_half_size     = layout.dst_half_bytes,
        .vtcm_src0_size_per_hvx = layout.src0_bytes_per_hvx,
        .vtcm_dst_size_per_hvx  = layout.dst_bytes_per_hvx,
        .dst_elem_size          = dst_elem_size,
        .row_start              = row_start,
        .src0_nrows             = nrows,
        .src0_nrows_per_hvx     = rows_per_hvx,
        .block                  = kparams->block,
    };

    work_queue_run(octx->ctx->hvx_queue, sum_rows_thread, &smctx, n_hvx);

    return HTP_STATUS_OK;
}

struct sum_context {
    struct htp_ops_context * octx;
    const float            * src_data;
    float                    partial_sums[HTP_MAX_NHVX];
    uint32_t                 total_elems;
    uint32_t                 elems_per_hvx;
};

static void sum_thread_f32(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct sum_context * sctx = (struct sum_context *) data;
    const uint32_t start = sctx->elems_per_hvx * ith;
    const uint32_t end   = MIN(start + sctx->elems_per_hvx, sctx->total_elems);

    if (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        return;
    }

    const uint32_t n = end - start;
    const float * src = sctx->src_data + start;

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((const void *) src, n * sizeof(float));

    sctx->partial_sums[ith] = hvx_reduce_sum_f32((const uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

int op_sum(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_sum_rows_kernel_params * kparams =
        (const struct htp_sum_rows_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    if (total_elems == 0) {
        ((float *) dst->data)[0] = 0.0f;
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx         = octx->n_hvx;
    const uint32_t elems_per_hvx = kparams->elems_per_hvx;

    struct sum_context sctx = {
        .octx             = octx,
        .src_data         = (const float *) src0->data,
        .total_elems      = total_elems,
        .elems_per_hvx    = elems_per_hvx,
    };

    work_queue_run(octx->ctx->hvx_queue, sum_thread_f32, &sctx, n_hvx);

    float sum = 0.0f;
    for (uint32_t i = 0; i < n_hvx; i++) {
        sum += sctx.partial_sums[i];
    }
    ((float *) dst->data)[0] = sum;

    return HTP_STATUS_OK;
}

static inline void argmax_slice_f32(
    const float * restrict src,
    uint32_t n,
    uint32_t offset,
    float * out_val,
    int32_t * out_idx
) {
    hvx_argmax_f32(src, n, offset, out_val, out_idx);
}

struct argmax_context {
    struct htp_ops_context * octx;
    const float            * src_data;
    int32_t                * dst_data;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_hvx;
    uint32_t                 elems_per_hvx;

    float                    partial_max[HTP_MAX_NHVX];
    int32_t                  partial_idx[HTP_MAX_NHVX];
};

static void argmax_thread_single_row(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct argmax_context * actx = (struct argmax_context *) data;
    const uint32_t start = actx->elems_per_hvx * ith;
    const uint32_t end   = MIN(start + actx->elems_per_hvx, actx->ne00);

    if (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        return;
    }

    const uint32_t n = end - start;
    const float * src = actx->src_data + start;

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);

    hex_l2fetch_block((const void *) src, n * sizeof(float));

    float max_val;
    int32_t max_idx;
    argmax_slice_f32(src, n, start, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start);
}

static void argmax_thread_multi_row(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct argmax_context * actx = (struct argmax_context *) data;
    const uint32_t r0 = actx->row_start + actx->rows_per_hvx * ith;
    const uint32_t r1 = MIN(r0 + actx->rows_per_hvx, actx->row_start + actx->nrows);

    if (r0 >= r1) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    for (uint32_t r = r0; r < r1; r++) {
        const float * src_row = (const float *) ((const uint8_t *) actx->src_data + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_data + r * actx->dst_stride);

        hex_l2fetch_block((const void *) src_row, actx->ne00 * sizeof(float));

        float max_val;
        int32_t max_idx;
        argmax_slice_f32(src_row, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

int op_argmax(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_sum_rows_kernel_params * kparams =
        (const struct htp_sum_rows_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t ne00 = src0->ne[0];
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    if (ne00 == 0 || src0_nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (src0_nrows == 1) {
        if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
            return HTP_STATUS_OK;
        }

        if (ne00 == 1) {
            ((int32_t *) dst->data)[0] = 0;
            return HTP_STATUS_OK;
        }

        const uint32_t n_hvx         = octx->n_hvx;
        const uint32_t elems_per_hvx = kparams->elems_per_hvx;

        struct argmax_context actx = {
            .octx             = octx,
            .src_data         = (const float *) src0->data,
            .dst_data         = (int32_t *) dst->data,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * sizeof(float)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) sizeof(int32_t),
            .row_start        = 0,
            .nrows            = 1,
            .rows_per_hvx     = 1,
            .elems_per_hvx    = elems_per_hvx,
        };

        work_queue_run(octx->ctx->hvx_queue, argmax_thread_single_row, &actx, n_hvx);

        float best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        for (uint32_t i = 1; i < n_hvx; i++) {
            if (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->data)[0] = best_idx;
        return HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, sizeof(int32_t));
        const uint32_t elems_per_chunk = can_split ? 32 : 0;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, elems_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx        = octx->n_hvx;
    const uint32_t rows_per_hvx = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);

    struct argmax_context actx = {
        .octx             = octx,
        .src_data         = (const float *) src0->data,
        .dst_data         = (int32_t *) dst->data,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * sizeof(float)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) sizeof(int32_t),
        .row_start        = row_start,
        .nrows            = nrows,
        .rows_per_hvx     = rows_per_hvx,
        .elems_per_hvx    = 0,
    };

    work_queue_run(octx->ctx->hvx_queue, argmax_thread_multi_row, &actx, n_hvx);
    return HTP_STATUS_OK;
}
