#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>
#include <math.h>

#include "dma-queue.h"
#include "dma-pipeline.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "htp-trace.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "sum-rows-ops.h"

#define sum_rows_preamble                                          \
    const struct htp_tensor *src0 = octx->src[0];                  \
    const struct htp_tensor *dst  = octx->dst;                     \
                                                                   \
    const uint32_t __attribute__((unused)) ne00 = src0->ne[0];     \
    const uint32_t __attribute__((unused)) ne01 = src0->ne[1];     \
    const uint32_t __attribute__((unused)) ne02 = src0->ne[2];     \
    const uint32_t __attribute__((unused)) ne03 = src0->ne[3];     \
                                                                   \
    const uint32_t __attribute__((unused)) nb00 = src0->nb[0];     \
    const uint32_t __attribute__((unused)) nb01 = src0->nb[1];     \
    const uint32_t __attribute__((unused)) nb02 = src0->nb[2];     \
    const uint32_t __attribute__((unused)) nb03 = src0->nb[3];     \
                                                                   \
    const uint32_t __attribute__((unused)) ne0 = dst->ne[0];       \
    const uint32_t __attribute__((unused)) ne1 = dst->ne[1];       \
    const uint32_t __attribute__((unused)) ne2 = dst->ne[2];       \
    const uint32_t __attribute__((unused)) ne3 = dst->ne[3];       \
                                                                   \
    const uint32_t __attribute__((unused)) nb0 = dst->nb[0];       \
    const uint32_t __attribute__((unused)) nb1 = dst->nb[1];       \
    const uint32_t __attribute__((unused)) nb2 = dst->nb[2];       \
    const uint32_t __attribute__((unused)) nb3 = dst->nb[3];       \

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

struct sum_rows_pipe_state {
    dma_addr_t                    data_src;
    dma_addr_t                    data_dst;
    sum_rows_compute_fn_t         compute;
    size_t                        src0_data_row_size;
    size_t                        src0_row_size_aligned;
    size_t                        dst_elem_size;
    uint32_t                      ne00;
    uint32_t                      ne01;
    uint32_t                      ne02;
    uint32_t                      nb01;
    uint32_t                      nb02;
    uint32_t                      nb03;
    uint32_t                      block;
    bool                          src0_contig;
    const struct fastdiv_values * div_ne01;
    const struct fastdiv_values * div_ne02;
    const struct fastdiv_values * div_ne012;
    uint32_t                      ir;
    uint32_t                      ir_end;
};

static bool sum_rows_next_task(void * ctx, struct dma_pipe_task * task) {
    struct sum_rows_pipe_state * s = (struct sum_rows_pipe_state *) ctx;
    if (s->ir >= s->ir_end) {
        return false;
    }
    const uint32_t block_size = sum_rows_block_size(s->ir, s->ir_end, s->block, s->src0_contig, s->ne01, s->div_ne01);
    task->row = s->ir;
    task->off = 0;
    task->len = block_size;
    task->aux = 0;
    s->ir += block_size;
    return true;
}

static void sum_rows_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct sum_rows_pipe_state * s = (struct sum_rows_pipe_state *) ctx;
    const size_t src0_off = s->src0_contig ? ((size_t) task->row * s->nb01) :
        sum_rows_row_offset(task->row, s->ne01, s->ne02, s->div_ne01, s->div_ne02, s->div_ne012, s->nb01, s->nb02, s->nb03);
    dma_queue_push(dma_q, dma_make_data(in_spad, s->data_src + src0_off),
                   s->src0_row_size_aligned, s->nb01, s->src0_data_row_size, task->len);
}

static void sum_rows_pipe_compute(void * ctx, const struct dma_pipe_task * task, void * out_spad, void * in_spad) {
    struct sum_rows_pipe_state * s = (struct sum_rows_pipe_state *) ctx;
    s->compute((const uint8_t *) in_spad, (uint8_t *) out_spad, s->ne00, task->len, s->src0_row_size_aligned);
}

static void sum_rows_push_out(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * out_spad) {
    struct sum_rows_pipe_state * s = (struct sum_rows_pipe_state *) ctx;
    const size_t dst_off   = (size_t) task->row * s->dst_elem_size;
    const size_t dst_bytes = (size_t) task->len * s->dst_elem_size;
    dma_queue_push(dma_q, dma_make_data(s->data_dst + dst_off, out_spad),
                   dst_bytes, dst_bytes, dst_bytes, 1);
}

static void sum_rows_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct sum_rows_context * smctx = (const struct sum_rows_context *) data;
    struct htp_ops_context        * octx  = smctx->octx;

    sum_rows_preamble;

    const uint32_t src0_nrows_per_hvx = smctx->src0_nrows_per_hvx;
    const uint32_t src0_nrows         = smctx->src0_nrows;
    const uint32_t src0_start_row     = smctx->row_start + src0_nrows_per_hvx * ith;
    const uint32_t src0_end_row       = MIN(src0_start_row + src0_nrows_per_hvx, smctx->row_start + src0_nrows);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint8_t * src0_vtcm_data = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_hvx);
    uint8_t * dst_vtcm_data  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_hvx);
    void *    in_buf[2]      = { src0_vtcm_data, src0_vtcm_data + smctx->src0_vtcm_half_size };
    void *    out_buf[2]     = { dst_vtcm_data,  dst_vtcm_data  + smctx->dst_vtcm_half_size };

    struct sum_rows_pipe_state s = {
        .data_src              = smctx->data_src0,
        .data_dst              = smctx->data_dst,
        .compute               = smctx->compute,
        .src0_data_row_size    = smctx->src0_data_row_size,
        .src0_row_size_aligned = smctx->src0_row_size_aligned,
        .dst_elem_size         = smctx->dst_elem_size,
        .ne00                  = ne00,
        .ne01                  = ne01,
        .ne02                  = ne02,
        .nb01                  = nb01,
        .nb02                  = nb02,
        .nb03                  = nb03,
        .block                 = smctx->block,
        .src0_contig           = smctx->kparams->src0_contig,
        .div_ne01              = &smctx->kparams->div_ne01,
        .div_ne02              = &smctx->kparams->div_ne02,
        .div_ne012             = &smctx->kparams->div_ne012,
        .ir                    = src0_start_row,
        .ir_end                = src0_end_row,
    };

    dma_pipe_io_run(th->dma_q, th->trace, &s, in_buf, out_buf,
                    sum_rows_next_task, sum_rows_push_in, sum_rows_pipe_compute, sum_rows_push_out);
}

int op_sum_rows(struct htp_ops_context * octx) {
    sum_rows_preamble;

    const bool src0_f16 = (src0->type == HTP_TYPE_F16);
    const bool dst_f16  = (dst->type  == HTP_TYPE_F16);

    if ((src0->type != HTP_TYPE_F32 && src0->type != HTP_TYPE_F16) ||
        (dst->type  != HTP_TYPE_F32 && dst->type  != HTP_TYPE_F16)) {
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
    const uint32_t src0_nrows         = ne01 * ne02 * ne03;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
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

static float __attribute__((noinline)) hvx_sum_chunk_f32(const uint8_t * restrict src, uint32_t num_elems) {
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

    return hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(acc)));
}

static float __attribute__((noinline)) hvx_sum_chunk_f16(const uint8_t * restrict src, uint32_t num_elems) {
    return hvx_vec_get_f32(hvx_row_reduce_sum_f16(src, num_elems));
}

typedef float (*sum_chunk_fn_t)(const uint8_t * restrict src, uint32_t num_elems);

struct sum_context {
    struct htp_ops_context * octx;
    sum_chunk_fn_t           sum_chunk;
    dma_addr_t               data_src0;
    uint8_t                * vtcm_src0;
    size_t                   src0_elem_size;
    size_t                   chunk_bytes;
    size_t                   src0_bytes_per_hvx;
    float                    partial_sums[HTP_MAX_NHVX];
    uint32_t                 total_elems;
    uint32_t                 elems_per_hvx;
    uint32_t                 chunk_elems;
};

struct sum_pipe_state {
    dma_addr_t     data_src;
    sum_chunk_fn_t sum_chunk;
    size_t         src0_elem_size;
    size_t         chunk_bytes;
    uint32_t       chunk_elems;
    uint32_t       pos;
    uint32_t       end;
    float          sum;
};

static bool sum_next_task(void * ctx, struct dma_pipe_task * task) {
    struct sum_pipe_state * s = (struct sum_pipe_state *) ctx;
    if (s->pos >= s->end) {
        return false;
    }
    const uint32_t cnt = MIN(s->chunk_elems, s->end - s->pos);
    task->row = s->pos;
    task->off = 0;
    task->len = cnt;
    task->aux = 0;
    s->pos += cnt;
    return true;
}

static void sum_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct sum_pipe_state * s = (struct sum_pipe_state *) ctx;
    const size_t nb = (size_t) task->len * s->src0_elem_size;
    dma_queue_push(dma_q, dma_make_data(in_spad, s->data_src + (size_t) task->row * s->src0_elem_size),
                   s->chunk_bytes, nb, nb, 1);
}

static void sum_pipe_compute(void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct sum_pipe_state * s = (struct sum_pipe_state *) ctx;
    s->sum += s->sum_chunk((const uint8_t *) in_spad, task->len);
}

static void sum_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct sum_context * sctx = (struct sum_context *) data;
    const uint32_t start = sctx->elems_per_hvx * ith;
    const uint32_t end   = MIN(start + sctx->elems_per_hvx, sctx->total_elems);

    if (start >= end) {
        sctx->partial_sums[ith] = 0.0f;
        return;
    }

    uint8_t * vtcm_src  = sctx->vtcm_src0 + ith * sctx->src0_bytes_per_hvx;
    void *    in_buf[2] = { vtcm_src, vtcm_src + sctx->chunk_bytes };

    struct sum_pipe_state s = {
        .data_src       = sctx->data_src0,
        .sum_chunk      = sctx->sum_chunk,
        .src0_elem_size = sctx->src0_elem_size,
        .chunk_bytes    = sctx->chunk_bytes,
        .chunk_elems    = sctx->chunk_elems,
        .pos            = start,
        .end            = end,
        .sum            = 0.0f,
    };

    dma_pipe_in_run(th->dma_q, th->trace, &s, in_buf, sum_next_task, sum_push_in, sum_pipe_compute);

    sctx->partial_sums[ith] = s.sum;
}

int op_sum(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if ((src0->type != HTP_TYPE_F32 && src0->type != HTP_TYPE_F16) ||
        (dst->type  != HTP_TYPE_F32 && dst->type  != HTP_TYPE_F16)) {
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

    const bool     src0_f16       = (src0->type == HTP_TYPE_F16);
    const bool     dst_f16        = (dst->type  == HTP_TYPE_F16);
    const size_t   src0_elem_size = src0_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t   dst_elem_size  = dst_f16  ? sizeof(__fp16) : sizeof(float);
    const uint32_t n_hvx          = octx->n_hvx;

    struct htp_sum_vtcm_layout layout;
    htp_sum_vtcm_layout_build(&layout, kparams->chunk_elems, src0_elem_size, n_hvx);
    if (octx->ctx->vtcm_size < layout.total_bytes) {
        FARF(ERROR, "sum: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = octx->ctx->vtcm_base;
    uint8_t * vtcm_dst  = vtcm_base + layout.dst_off;

    const uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    if (total_elems == 0) {
        hvx_vmem(vtcm_dst) = Q6_V_vzero();
        dma_sync_write(octx->ctx->dma[0], dst->data, vtcm_dst, dst_elem_size);
        return HTP_STATUS_OK;
    }

    struct sum_context sctx = {
        .octx               = octx,
        .sum_chunk          = src0_f16 ? hvx_sum_chunk_f16 : hvx_sum_chunk_f32,
        .data_src0          = (dma_addr_t) src0->data,
        .vtcm_src0          = vtcm_base + layout.src0_off,
        .src0_elem_size     = src0_elem_size,
        .chunk_bytes        = layout.chunk_bytes,
        .src0_bytes_per_hvx = layout.src0_bytes_per_hvx,
        .total_elems        = total_elems,
        .elems_per_hvx      = kparams->elems_per_hvx,
        .chunk_elems        = kparams->chunk_elems,
    };

    work_queue_run(octx->ctx->hvx_queue, sum_thread, &sctx, n_hvx);

    float sum = 0.0f;
    for (uint32_t i = 0; i < n_hvx; i++) {
        sum += sctx.partial_sums[i];
    }

    if (dst_f16) {
        const HVX_Vector sum_sf = hvx_vec_clamp_f16_f32(hvx_vec_splat_f32(sum));
        hvx_vmem(vtcm_dst) = hvx_vec_f32_to_f16_shuff(sum_sf, sum_sf);
    } else {
        hvx_vmem(vtcm_dst) = hvx_vec_splat_f32(sum);
    }
    dma_sync_write(octx->ctx->dma[0], dst->data, vtcm_dst, dst_elem_size);

    return HTP_STATUS_OK;
}

static void __attribute__((noinline)) hvx_argmax_chunk_f32(
    const uint8_t * restrict src,
    uint32_t                 cnt,
    uint32_t                 base_idx,
    float *                  best_val,
    int32_t *                best_idx
) {
    static const int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    const HVX_Vector v_step          = Q6_V_vsplat_R(32);
    HVX_Vector       v_cur_idx       = Q6_Vw_vadd_VwVw(*(const HVX_Vector *) c_lane_idx, Q6_V_vsplat_R((int32_t) base_idx));
    HVX_Vector       vmax            = hvx_vec_splat_f32(-INFINITY);
    HVX_Vector       vidx            = Q6_V_vsplat_R((int32_t) base_idx);

    const uint32_t nvec = cnt / VLEN_FP32;
    const uint32_t nloe = cnt % VLEN_FP32;

    for (uint32_t vi = 0; vi < nvec; vi++) {
        const HVX_Vector     v    = vsrc[vi];
        const HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(v, vmax);
        vmax      = Q6_V_vmux_QVV(pred, v, vmax);
        vidx      = Q6_V_vmux_QVV(pred, v_cur_idx, vidx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_step);
    }

    if (nloe) {
        const HVX_Vector     v     = vsrc[nvec];
        const HVX_VectorPred valid = Q6_Q_vsetq_R(nloe * sizeof(float));
        const HVX_VectorPred pred  = Q6_Q_and_QQ(valid, Q6_Q_vcmp_gt_VsfVsf(v, vmax));
        vmax = Q6_V_vmux_QVV(pred, v, vmax);
        vidx = Q6_V_vmux_QVV(pred, v_cur_idx, vidx);
    }

    for (uint32_t shift = 64; shift >= 4; shift >>= 1) {
        const HVX_Vector     v_val_r = Q6_V_vror_VR(vmax, shift);
        const HVX_Vector     v_idx_r = Q6_V_vror_VR(vidx, shift);
        const HVX_VectorPred gt      = Q6_Q_vcmp_gt_VsfVsf(v_val_r, vmax);
        const HVX_VectorPred eq      = Q6_Q_vcmp_eq_VwVw(v_val_r, vmax);
        const HVX_VectorPred lt_idx  = Q6_Q_vcmp_gt_VwVw(vidx, v_idx_r);
        const HVX_VectorPred take    = Q6_Q_or_QQ(gt, Q6_Q_and_QQ(eq, lt_idx));
        vmax = Q6_V_vmux_QVV(take, v_val_r, vmax);
        vidx = Q6_V_vmux_QVV(take, v_idx_r, vidx);
    }

    const int32_t val_bits = Q6_R_vextract_VR(vmax, 0);
    float         c_val;
    memcpy(&c_val, &val_bits, sizeof(float));
    const int32_t c_idx = Q6_R_vextract_VR(vidx, 0);

    if (c_val > *best_val) {
        *best_val = c_val;
        *best_idx = c_idx;
    }
}

static void __attribute__((noinline)) hvx_argmax_chunk_f16(
    const uint8_t * restrict src,
    uint32_t                 cnt,
    uint32_t                 base_idx,
    float *                  best_val,
    int32_t *                best_idx
) {
    static const int32_t c_lane_idx[32] __attribute__((aligned(128))) = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };

    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    const HVX_Vector v_step          = Q6_V_vsplat_R(32);
    HVX_Vector       v_cur_idx       = Q6_Vw_vadd_VwVw(*(const HVX_Vector *) c_lane_idx, Q6_V_vsplat_R((int32_t) base_idx));
    HVX_Vector       vmax            = hvx_vec_splat_f32(-INFINITY);
    HVX_Vector       vidx            = Q6_V_vsplat_R((int32_t) base_idx);

    const uint32_t nvec = cnt / VLEN_FP16;
    const uint32_t nloe = cnt % VLEN_FP16;

    for (uint32_t vi = 0; vi < nvec; vi++) {
        const HVX_VectorPair p  = hvx_vec_f16_to_f32(vsrc[vi]);
        const HVX_Vector     v0 = Q6_V_lo_W(p);
        const HVX_VectorPred p0 = Q6_Q_vcmp_gt_VsfVsf(v0, vmax);
        vmax      = Q6_V_vmux_QVV(p0, v0, vmax);
        vidx      = Q6_V_vmux_QVV(p0, v_cur_idx, vidx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_step);

        const HVX_Vector     v1 = Q6_V_hi_W(p);
        const HVX_VectorPred p1 = Q6_Q_vcmp_gt_VsfVsf(v1, vmax);
        vmax      = Q6_V_vmux_QVV(p1, v1, vmax);
        vidx      = Q6_V_vmux_QVV(p1, v_cur_idx, vidx);
        v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_step);
    }

    if (nloe) {
        const HVX_VectorPair p  = hvx_vec_f16_to_f32(vsrc[nvec]);
        const HVX_Vector     v0 = Q6_V_lo_W(p);
        if (nloe >= VLEN_FP32) {
            const HVX_VectorPred p0 = Q6_Q_vcmp_gt_VsfVsf(v0, vmax);
            vmax      = Q6_V_vmux_QVV(p0, v0, vmax);
            vidx      = Q6_V_vmux_QVV(p0, v_cur_idx, vidx);
            v_cur_idx = Q6_Vw_vadd_VwVw(v_cur_idx, v_step);

            const uint32_t rem = nloe - VLEN_FP32;
            if (rem) {
                const HVX_Vector     v1    = Q6_V_hi_W(p);
                const HVX_VectorPred valid = Q6_Q_vsetq_R(rem * sizeof(float));
                const HVX_VectorPred p1    = Q6_Q_and_QQ(valid, Q6_Q_vcmp_gt_VsfVsf(v1, vmax));
                vmax = Q6_V_vmux_QVV(p1, v1, vmax);
                vidx = Q6_V_vmux_QVV(p1, v_cur_idx, vidx);
            }
        } else {
            const HVX_VectorPred valid = Q6_Q_vsetq_R(nloe * sizeof(float));
            const HVX_VectorPred p0    = Q6_Q_and_QQ(valid, Q6_Q_vcmp_gt_VsfVsf(v0, vmax));
            vmax = Q6_V_vmux_QVV(p0, v0, vmax);
            vidx = Q6_V_vmux_QVV(p0, v_cur_idx, vidx);
        }
    }

    for (uint32_t shift = 64; shift >= 4; shift >>= 1) {
        const HVX_Vector     v_val_r = Q6_V_vror_VR(vmax, shift);
        const HVX_Vector     v_idx_r = Q6_V_vror_VR(vidx, shift);
        const HVX_VectorPred gt      = Q6_Q_vcmp_gt_VsfVsf(v_val_r, vmax);
        const HVX_VectorPred eq      = Q6_Q_vcmp_eq_VwVw(v_val_r, vmax);
        const HVX_VectorPred lt_idx  = Q6_Q_vcmp_gt_VwVw(vidx, v_idx_r);
        const HVX_VectorPred take    = Q6_Q_or_QQ(gt, Q6_Q_and_QQ(eq, lt_idx));
        vmax = Q6_V_vmux_QVV(take, v_val_r, vmax);
        vidx = Q6_V_vmux_QVV(take, v_idx_r, vidx);
    }

    const int32_t val_bits = Q6_R_vextract_VR(vmax, 0);
    float         c_val;
    memcpy(&c_val, &val_bits, sizeof(float));
    const int32_t c_idx = Q6_R_vextract_VR(vidx, 0);

    if (c_val > *best_val) {
        *best_val = c_val;
        *best_idx = c_idx;
    }
}

typedef void (*argmax_chunk_fn_t)(
    const uint8_t * restrict src,
    uint32_t                 cnt,
    uint32_t                 base_idx,
    float *                  best_val,
    int32_t *                best_idx
);

static void __attribute__((noinline)) hvx_vtcm_store_lane_i32(uint8_t * vtcm_dst, uint32_t lane, int32_t val) {
    const uint32_t       b_off  = lane * sizeof(int32_t);
    const HVX_VectorPred b_mask = Q6_Q_vsetq_R(b_off);
    const HVX_VectorPred e_mask = Q6_Q_vsetq2_R(b_off + sizeof(int32_t));
    Q6_vmem_QnRIV(Q6_Q_or_QQn(b_mask, e_mask), (HVX_Vector *) vtcm_dst, Q6_V_vsplat_R(val));
}

struct argmax_context {
    struct htp_ops_context * octx;
    argmax_chunk_fn_t        argmax_chunk;
    dma_addr_t               data_src0;
    dma_addr_t               data_dst;
    uint8_t                * vtcm_src0;
    uint8_t                * vtcm_dst;
    size_t                   src0_elem_size;
    size_t                   chunk_bytes;
    size_t                   src0_bytes_per_hvx;
    size_t                   dst_bytes_per_hvx;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 row_start;
    uint32_t                 nrows;
    uint32_t                 rows_per_hvx;
    uint32_t                 elems_per_hvx;
    uint32_t                 chunk_elems;

    float                    partial_max[HTP_MAX_NHVX];
    int32_t                  partial_idx[HTP_MAX_NHVX];
};

struct argmax_single_pipe_state {
    dma_addr_t        data_src;
    argmax_chunk_fn_t argmax_chunk;
    size_t            src0_elem_size;
    size_t            chunk_bytes;
    uint32_t          chunk_elems;
    uint32_t          pos;
    uint32_t          end;
    float             best_val;
    int32_t           best_idx;
};

static bool argmax_single_next_task(void * ctx, struct dma_pipe_task * task) {
    struct argmax_single_pipe_state * s = (struct argmax_single_pipe_state *) ctx;
    if (s->pos >= s->end) {
        return false;
    }
    const uint32_t cnt = MIN(s->chunk_elems, s->end - s->pos);
    task->row = s->pos;
    task->off = 0;
    task->len = cnt;
    task->aux = 0;
    s->pos += cnt;
    return true;
}

static void argmax_single_push_in(dma_queue * dma_q, void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct argmax_single_pipe_state * s = (struct argmax_single_pipe_state *) ctx;
    const size_t nb = (size_t) task->len * s->src0_elem_size;
    dma_queue_push(dma_q, dma_make_data(in_spad, s->data_src + (size_t) task->row * s->src0_elem_size),
                   s->chunk_bytes, nb, nb, 1);
}

static void argmax_single_compute(void * ctx, const struct dma_pipe_task * task, void * in_spad) {
    struct argmax_single_pipe_state * s = (struct argmax_single_pipe_state *) ctx;
    s->argmax_chunk((const uint8_t *) in_spad, task->len, task->row, &s->best_val, &s->best_idx);
}

static void argmax_thread_single_row(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct argmax_context * actx = (struct argmax_context *) data;
    const uint32_t start = actx->elems_per_hvx * ith;
    const uint32_t end   = MIN(start + actx->elems_per_hvx, actx->ne00);

    if (start >= end) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        return;
    }

    uint8_t * vtcm_src  = actx->vtcm_src0 + ith * actx->src0_bytes_per_hvx;
    void *    in_buf[2] = { vtcm_src, vtcm_src + actx->chunk_bytes };

    struct argmax_single_pipe_state s = {
        .data_src       = actx->data_src0,
        .argmax_chunk   = actx->argmax_chunk,
        .src0_elem_size = actx->src0_elem_size,
        .chunk_bytes    = actx->chunk_bytes,
        .chunk_elems    = actx->chunk_elems,
        .pos            = start,
        .end            = end,
        .best_val       = -INFINITY,
        .best_idx       = (int32_t) start,
    };

    dma_pipe_in_run(th->dma_q, th->trace, &s, in_buf,
                    argmax_single_next_task, argmax_single_push_in, argmax_single_compute);

    actx->partial_max[ith] = s.best_val;
    actx->partial_idx[ith] = s.best_idx;
}

static void argmax_thread_multi_row(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    struct argmax_context * actx = (struct argmax_context *) data;
    const uint32_t r0 = actx->row_start + actx->rows_per_hvx * ith;
    const uint32_t r1 = MIN(r0 + actx->rows_per_hvx, actx->row_start + actx->nrows);

    if (r0 >= r1) {
        return;
    }

    const dma_addr_t        data_src       = actx->data_src0;
    const dma_addr_t        data_dst       = actx->data_dst;
    uint8_t               * vtcm_src       = actx->vtcm_src0 + ith * actx->src0_bytes_per_hvx;
    uint8_t               * vtcm_dst       = actx->vtcm_dst  + ith * actx->dst_bytes_per_hvx;
    const size_t            src0_elem_size = actx->src0_elem_size;
    const uint32_t          ne00           = actx->ne00;
    const uint32_t          chunk_elems    = actx->chunk_elems;
    const size_t            chunk_bytes    = actx->chunk_bytes;
    const uint32_t          src_stride     = actx->src_stride;
    const uint32_t          dst_stride     = actx->dst_stride;
    const argmax_chunk_fn_t argmax_chunk   = actx->argmax_chunk;

    dma_queue               * dma_q = th->dma_q;
    struct htp_thread_trace * tr    = th->trace;

    uint32_t dst_row0 = r0;
    uint32_t dst_cnt  = 0;

    for (uint32_t r = r0; r < r1; r++) {
        const dma_addr_t src_row = data_src + (size_t) r * src_stride;

        uint32_t push_pos = 0;
        for (uint32_t bi = 0; bi < 2 && push_pos < ne00; bi++) {
            const uint32_t cnt = MIN(chunk_elems, ne00 - push_pos);
            const size_t   nb  = (size_t) cnt * src0_elem_size;
            dma_queue_push(dma_q,
                dma_make_data(vtcm_src + bi * chunk_bytes, src_row + (size_t) push_pos * src0_elem_size),
                chunk_bytes, nb, nb, 1);
            push_pos += cnt;
        }

        float   max_val = -INFINITY;
        int32_t max_idx = 0;

        for (uint32_t pos = 0; pos < ne00; ) {
            const uint32_t cnt = MIN(chunk_elems, ne00 - pos);
            uint8_t * buf = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            argmax_chunk(buf, cnt, pos, &max_val, &max_idx);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            if (push_pos < ne00) {
                const uint32_t pcnt = MIN(chunk_elems, ne00 - push_pos);
                const size_t   pnb  = (size_t) pcnt * src0_elem_size;
                dma_queue_push(dma_q,
                    dma_make_data(buf, src_row + (size_t) push_pos * src0_elem_size),
                    chunk_bytes, pnb, pnb, 1);
                push_pos += pcnt;
            }
            pos += cnt;
        }

        hvx_vtcm_store_lane_i32(vtcm_dst, dst_cnt, max_idx);
        dst_cnt++;

        if (dst_cnt == VLEN_FP32 || r + 1 == r1) {
            dma_sync_write(dma_q, data_dst + (size_t) dst_row0 * dst_stride, vtcm_dst, (size_t) dst_cnt * sizeof(int32_t));
            dst_row0 = r + 1;
            dst_cnt  = 0;
        }
    }
}

int op_argmax(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if ((src0->type != HTP_TYPE_F32 && src0->type != HTP_TYPE_F16) || dst->type != HTP_TYPE_I32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_sum_rows_kernel_params * kparams =
        (const struct htp_sum_rows_kernel_params *) octx->kernel_params;

    const bool              src0_f16       = (src0->type == HTP_TYPE_F16);
    const size_t            src0_elem_size = src0_f16 ? sizeof(__fp16) : sizeof(float);
    const argmax_chunk_fn_t argmax_chunk   = src0_f16 ? hvx_argmax_chunk_f16 : hvx_argmax_chunk_f32;
    const uint32_t          ne00           = src0->ne[0];
    const uint32_t          src0_nrows     = src0->ne[1] * src0->ne[2] * src0->ne[3];

    if (ne00 == 0 || src0_nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (src0_nrows == 1) {
        if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
            return HTP_STATUS_OK;
        }

        if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
            return HTP_STATUS_INVAL_PARAMS;
        }

        const uint32_t n_hvx = octx->n_hvx;

        struct htp_sum_vtcm_layout layout;
        htp_sum_vtcm_layout_build(&layout, kparams->chunk_elems, src0_elem_size, n_hvx);
        if (octx->ctx->vtcm_size < layout.total_bytes) {
            FARF(ERROR, "argmax: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, layout.total_bytes);
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        uint8_t * vtcm_base = octx->ctx->vtcm_base;
        uint8_t * vtcm_dst  = vtcm_base + layout.dst_off;

        if (ne00 == 1) {
            hvx_vmem(vtcm_dst) = Q6_V_vzero();
            dma_sync_write(octx->ctx->dma[0], dst->data, vtcm_dst, sizeof(int32_t));
            return HTP_STATUS_OK;
        }

        struct argmax_context actx = {
            .octx               = octx,
            .argmax_chunk       = argmax_chunk,
            .data_src0          = (dma_addr_t) src0->data,
            .data_dst           = (dma_addr_t) dst->data,
            .vtcm_src0          = vtcm_base + layout.src0_off,
            .vtcm_dst           = vtcm_dst,
            .src0_elem_size     = src0_elem_size,
            .chunk_bytes        = layout.chunk_bytes,
            .src0_bytes_per_hvx = layout.src0_bytes_per_hvx,
            .dst_bytes_per_hvx  = layout.dst_bytes_per_hvx,
            .ne00               = ne00,
            .src_stride         = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * src0_elem_size),
            .dst_stride         = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) sizeof(int32_t),
            .row_start          = 0,
            .nrows              = 1,
            .rows_per_hvx       = 1,
            .elems_per_hvx      = kparams->elems_per_hvx,
            .chunk_elems        = kparams->chunk_elems,
        };

        work_queue_run(octx->ctx->hvx_queue, argmax_thread_single_row, &actx, n_hvx);

        float   best_val = actx.partial_max[0];
        int32_t best_idx = actx.partial_idx[0];
        for (uint32_t i = 1; i < n_hvx; i++) {
            if (actx.partial_max[i] > best_val) {
                best_val = actx.partial_max[i];
                best_idx = actx.partial_idx[i];
            }
        }

        hvx_vmem(vtcm_dst) = Q6_V_vsplat_R(best_idx);
        dma_sync_write(octx->ctx->dma[0], dst->data, vtcm_dst, sizeof(int32_t));
        return HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_hvx(octx, MIN(kparams->n_hvx, nrows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_hvx        = octx->n_hvx;
    const uint32_t rows_per_hvx = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);

    struct htp_sum_vtcm_layout layout;
    htp_sum_vtcm_layout_build(&layout, kparams->chunk_elems, src0_elem_size, n_hvx);
    if (octx->ctx->vtcm_size < layout.total_bytes) {
        FARF(ERROR, "argmax: VTCM size %zu is too small, needed %zu\n", octx->ctx->vtcm_size, layout.total_bytes);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = octx->ctx->vtcm_base;

    struct argmax_context actx = {
        .octx               = octx,
        .argmax_chunk       = argmax_chunk,
        .data_src0          = (dma_addr_t) src0->data,
        .data_dst           = (dma_addr_t) dst->data,
        .vtcm_src0          = vtcm_base + layout.src0_off,
        .vtcm_dst           = vtcm_base + layout.dst_off,
        .src0_elem_size     = src0_elem_size,
        .chunk_bytes        = layout.chunk_bytes,
        .src0_bytes_per_hvx = layout.src0_bytes_per_hvx,
        .dst_bytes_per_hvx  = layout.dst_bytes_per_hvx,
        .ne00               = ne00,
        .src_stride         = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * src0_elem_size),
        .dst_stride         = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) sizeof(int32_t),
        .row_start          = row_start,
        .nrows              = nrows,
        .rows_per_hvx       = rows_per_hvx,
        .elems_per_hvx      = 0,
        .chunk_elems        = kparams->chunk_elems,
    };

    work_queue_run(octx->ctx->hvx_queue, argmax_thread_multi_row, &actx, n_hvx);
    return HTP_STATUS_OK;
}
