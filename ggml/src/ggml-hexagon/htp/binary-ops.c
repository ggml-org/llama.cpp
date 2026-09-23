#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "hex-profile.h"
#include "binary-ops.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Context for binary operations
struct htp_binary_context {
    struct htp_ops_context * octx;
    struct htp_binary_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;

    struct fastdiv_values src0_dim1_div; // ne01
    struct fastdiv_values src0_dim2_div; // ne02
    struct fastdiv_values src0_dim12_div;// ne03

    struct fastdiv_values src1_dim1_div; // ne11
    struct fastdiv_values src1_dim2_div; // ne12
    struct fastdiv_values src1_dim3_div; // ne13

    uint32_t block_max;
    uint32_t nrows_per_thread;
    uint32_t total_rows;
    uint32_t row_start;
    size_t   src0_row_size_aligned;
    size_t   src1_row_size_aligned;
    size_t   dst_row_size_aligned;
    size_t   row_size_bytes;

    bool split_at_ne01;
    bool split_at_ne02;

    void * compute;
};

#define htp_binary_preamble                        \
    const struct htp_tensor * src0 = octx->src[0]; \
    const struct htp_tensor * src1 = octx->src[1]; \
    const struct htp_tensor * dst  = octx->dst;    \
                                                   \
    const uint32_t ne00 = src0->ne[0];             \
    const uint32_t ne01 = src0->ne[1];             \
    const uint32_t ne02 = src0->ne[2];             \
    const uint32_t ne03 = src0->ne[3];             \
                                                   \
    const uint32_t ne10 = src1->ne[0];             \
    const uint32_t ne11 = src1->ne[1];             \
    const uint32_t ne12 = src1->ne[2];             \
    const uint32_t ne13 = src1->ne[3];             \
                                                   \
    const uint32_t nb01 = src0->nb[1];             \
    const uint32_t nb02 = src0->nb[2];             \
    const uint32_t nb03 = src0->nb[3];             \
                                                   \
    const uint32_t nb11 = src1->nb[1];             \
    const uint32_t nb12 = src1->nb[2];             \
    const uint32_t nb13 = src1->nb[3];             \
                                                   \
    const uint32_t nb1 = dst->nb[1];               \
    const uint32_t nb2 = dst->nb[2];               \
    const uint32_t nb3 = dst->nb[3];

static inline uint32_t calc_block_size(struct htp_binary_context * bctx, uint32_t ir, uint32_t end_row, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t rows_left = end_row - ir;
    uint32_t block_limit = rows_left;

    if (bctx->split_at_ne01) {
        block_limit = MIN(block_limit, ne01 - i01);
    }
    if (bctx->split_at_ne02) {
         uint32_t rows_in_plane = (ne02 * ne01) - rem;
         block_limit = MIN(block_limit, rows_in_plane);
    }

    return MIN(bctx->block_max, block_limit);
}

// Out-of-line compute micro-kernels

typedef void (*compute_scalar_dma_t)(
    uint8_t * dst, const uint8_t * src0, const void * s1_table,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SCALAR_DMA(NAME, TYPE, HVX_STMT)         \
static void compute_scalar_dma_##NAME(                          \
    uint8_t * dst, const uint8_t * src0, const void * s1_table, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    const TYPE * table = (const TYPE *) s1_table;               \
    for (uint32_t r = 0; r < n_rows; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        const uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = table[cur_i11];                              \
        HVX_STMT;                                               \
        if (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR_DMA(add_f32, float,    hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR_DMA(add_f16, _Float16, hvx_add_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR_DMA(sub_f32, float,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR_DMA(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR_DMA(mul_f32, float,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR_DMA(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR_DMA(div_f32, float,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR_DMA(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

typedef void (*compute_scalar_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_ptr, uint32_t s1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)                                          \
static void compute_scalar_##NAME(                                                           \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_ptr, uint32_t s1_stride,       \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    for (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        const uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        TYPE val = *(const TYPE *)(src1_ptr + r * s1_stride);                                \
        HVX_STMT;                                                                            \
    }                                                                                        \
}

DEFINE_COMPUTE_SCALAR(add_f32, float,    hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(add_f16, _Float16, hvx_add_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, float,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, float,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, float,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

typedef void (*compute_same_shape_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
static void compute_same_shape_##NAME(                                                           \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1,                                   \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    for (uint32_t r = 0; r < n_rows; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(add_f32, hvx_add_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(add_f16, hvx_add_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

typedef void (*compute_row_bcast_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_ROW_BCAST(NAME, HVX_FN)                                               \
static void compute_row_bcast_##NAME(                                                        \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1,                               \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    for (uint32_t r = 0; r < n_rows; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_ROW_BCAST(add_f32, hvx_add_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(add_f16, hvx_add_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(div_f16, hvx_div_f16_aaa)

typedef void (*compute_complex_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t nb11,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
static void compute_complex_##NAME(                                                          \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t nb11,         \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    for (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        const uint8_t * r_src1 = src1_plane + i11 * nb11;                                    \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(add_f32, hvx_add_f32_aau)
DEFINE_COMPUTE_COMPLEX(add_f16, hvx_add_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

typedef void (*compute_repeat_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t nb11,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#define DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
static void compute_repeat_##NAME(                                                           \
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, const struct fastdiv_values * div11, uint32_t nb11,         \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    for (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        const uint8_t * r_src1_row = src1_plane + i11 * nb11;                                \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        const uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        for (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * sizeof(TYPE), r_src0 + c * sizeof(TYPE), r_src1_row, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(add_f32, float,    hvx_add_f32_uuu)
DEFINE_COMPUTE_REPEAT(add_f16, _Float16, hvx_add_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, float,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, float,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, float,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

typedef void (*compute_add_id_t)(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_data, const char * src2_data,
    uint32_t i01, uint32_t i02, uint32_t nb20, uint32_t nb21, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

static void compute_add_id_f32(
    uint8_t * dst, const uint8_t * src0, const uint8_t * src1_data, const char * src2_data,
    uint32_t i01, uint32_t i02, uint32_t nb20, uint32_t nb21, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    for (uint32_t r = 0; r < n_rows; r++) {
        uint32_t r_i01 = i01 + r;
        const int32_t idx = *(const int32_t *)(src2_data + r_i01 * nb20 + i02 * nb21);
        if (idx < 0) {
            memcpy(dst + r * dst_stride, src0 + r * src0_stride, ne00 * sizeof(float));
            continue;
        }
        const uint8_t * r_src1 = src1_data + idx * src1_stride;
        const uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_add_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1a. Scalar src1 in VTCM via DMA (ne10 == 1, ne12 == 1, ne13 == 1)
static void binary_thread_scalar_dma(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t row_size_bytes = bctx->row_size_bytes;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-scalar-dma: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    const void * s1_table    = VTCM_LAYOUT_PTR(const void, bctx->vtcm_base, layout->off_src1);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes,
                                                             current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_dma_t compute = (compute_scalar_dma_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t cur_i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_table, cur_i11, ne11, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes,
                                                                current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes,
                                                               next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 1b. Scalar src1 dynamic / pointer (ne10 == 1)
static void binary_thread_scalar(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t row_size_bytes = bctx->row_size_bytes;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);

        const uint8_t * src1_ptr = (const uint8_t *)(uintptr_t) src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        uint32_t s1_stride = (ne11 == 1) ? 0 : nb11;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_ptr, s1_stride, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
static void binary_thread_vector_same_shape(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t row_size_bytes = bctx->row_size_bytes;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    size_t src0_spad_half = layout->src0_spad_half_size;
    size_t src1_spad_half = layout->src1_spad_half_size;
    size_t dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        dma_queue_push(dma_q, dma_make_data(s1_spad, src1_curr), bctx->src1_row_size_aligned, nb11, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, bctx->src1_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_next = src1->data + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            dma_queue_push(dma_q, dma_make_data(s1_spad, s1_next), bctx->src1_row_size_aligned, nb11, row_size_bytes, next_block_size);

            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
static void binary_thread_vector_row_broadcast(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t row_size_bytes = bctx->row_size_bytes;
    const uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    const uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    size_t src0_spad_half = layout->src0_spad_half_size;
    size_t dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    void * s1_ptr = VTCM_LAYOUT_PTR(void, bctx->vtcm_base, layout->off_src1);

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_row_bcast_t compute = (compute_row_bcast_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (const uint8_t *)s1_ptr, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
static void binary_thread_vector_complex(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t row_size_bytes = bctx->row_size_bytes;
    const uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    const uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    FARF(HIGH, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        const uint8_t * src1_plane = (const uint8_t *)(uintptr_t) src1->data + i13 * nb13 + i12 * nb12;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, nb11,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
static void binary_thread_element_repeat(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t row_size_bytes = bctx->row_size_bytes;
    const uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    const uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;

    FARF(HIGH, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        const uint8_t * src1_plane = (const uint8_t *)(uintptr_t) src1->data + i13 * nb13 + i12 * nb12;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, nb11,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        if (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
static void binary_thread_add_id_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * src2 = octx->src[2];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t nb01 = src0->nb[1];
    const uint32_t nb02 = src0->nb[2];
    const uint32_t nb03 = src0->nb[3];
    const uint32_t src1_stride = bctx->src1_row_size_aligned;

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t row_size_bytes = bctx->row_size_bytes;
    const uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    if (start_row >= end_row) return;

    const struct htp_binary_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    const uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(const uint8_t, bctx->vtcm_base, layout->off_src1);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_push(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_add_id_t compute = (compute_add_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, (const char *)(uintptr_t)src2->data,
                i01, i02, src2->nb[0], src2->nb[1], src1_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

static int execute_op_binary(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;
    const struct htp_binary_kernel_params * kparams = (const struct htp_binary_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    const uint32_t src0_type = octx->src[0]->type;
    const size_t elem_size = (src0_type == HTP_TYPE_F32) ? sizeof(float) : sizeof(_Float16);
    const size_t src0_row_size = src0->ne[0] * elem_size;
    const size_t src1_row_size = src1->ne[0] * elem_size;
    const size_t dst_row_size  = dst->ne[0]  * elem_size;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_threads = octx->n_threads;
    const size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    const size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    const size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    if (htp_tensor_is_extended(src1)) {
        if (kparams->kernel_type != HTP_BINARY_KERNEL_SAME_SHAPE &&
            kparams->kernel_type != HTP_BINARY_KERNEL_ROW_BCAST &&
            kparams->kernel_type != HTP_BINARY_KERNEL_SCALAR_DMA &&
            kparams->kernel_type != HTP_BINARY_KERNEL_ADD_ID) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    if (octx->op == HTP_OP_ADD_ID && htp_tensor_is_extended(octx->src[2])) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_binary_context bctx;
    bctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    htp_binary_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_size);

    if (bctx.vtcm_layout.rows_per_buffer == 0 || bctx.vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src1);
    if (kparams->kernel_type == HTP_BINARY_KERNEL_ROW_BCAST) {
        dma_queue_push(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1->ne[0] * elem_size, 1);
    } else if (kparams->kernel_type == HTP_BINARY_KERNEL_SCALAR_DMA) {
        dma_queue_push(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1->ne[1] * elem_size, 1);
    } else if (kparams->kernel_type == HTP_BINARY_KERNEL_ADD_ID) {
        dma_queue_push(dma_q, dma_make_data(vtcm_src1, src1->data),
                       kparams->src1_row_size_aligned, src1->nb[1],
                       src1->ne[0] * elem_size, src1->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    bctx.total_rows            = nrows;
    bctx.row_start             = row_start;
    bctx.block_max             = bctx.vtcm_layout.rows_per_buffer;
    bctx.src0_row_size_aligned = src0_row_size_aligned;
    bctx.src1_row_size_aligned = src1_row_size_aligned;
    bctx.dst_row_size_aligned  = dst_row_size_aligned;
    bctx.row_size_bytes        = src0_row_size;

    bctx.src0_dim1_div  = init_fastdiv_values(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_values(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_values(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_values(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_values(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_values(src1->ne[3]);

    bool src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    bool dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    bool src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    bool dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    bctx.split_at_ne01 = (octx->op == HTP_OP_ADD_ID) ||
        ((src0->ne[2] > 1) && ((src1->ne[1] > 1) || (src1->ne[2] > 1) || !src0_contig_dim1 || !dst_contig_dim1));
    bctx.split_at_ne02 = (src0->ne[3] > 1) && ((src1->ne[2] > 1) || (src1->ne[3] > 1) || !src0_contig_dim2 || !dst_contig_dim2);

    worker_callback_t worker_func = NULL;
    void * compute_func = NULL;

    switch (kparams->kernel_type) {
        case HTP_BINARY_KERNEL_SAME_SHAPE:
            worker_func = binary_thread_vector_same_shape;
            if (src0_type == HTP_TYPE_F32) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_same_shape_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_same_shape_div_f32; break;
                    default: break;
                }
            } else if (src0_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_same_shape_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_same_shape_div_f16; break;
                    default: break;
                }
            }
            break;
        case HTP_BINARY_KERNEL_ROW_BCAST:
            worker_func = binary_thread_vector_row_broadcast;
            if (src0_type == HTP_TYPE_F32) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_row_bcast_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_row_bcast_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_row_bcast_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_row_bcast_div_f32; break;
                    default: break;
                }
            } else if (src0_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_row_bcast_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_row_bcast_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_row_bcast_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_row_bcast_div_f16; break;
                    default: break;
                }
            }
            break;
        case HTP_BINARY_KERNEL_SCALAR_DMA:
            worker_func = binary_thread_scalar_dma;
            if (src0_type == HTP_TYPE_F32) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_scalar_dma_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_scalar_dma_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_scalar_dma_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_scalar_dma_div_f32; break;
                    default: break;
                }
            } else if (src0_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_scalar_dma_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_scalar_dma_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_scalar_dma_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_scalar_dma_div_f16; break;
                    default: break;
                }
            }
            break;
        case HTP_BINARY_KERNEL_SCALAR:
            worker_func = binary_thread_scalar;
            if (src0_type == HTP_TYPE_F32) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_scalar_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_scalar_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_scalar_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_scalar_div_f32; break;
                    default: break;
                }
            } else if (src0_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_scalar_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_scalar_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_scalar_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_scalar_div_f16; break;
                    default: break;
                }
            }
            break;
        case HTP_BINARY_KERNEL_COMPLEX:
            worker_func = binary_thread_vector_complex;
            if (src0_type == HTP_TYPE_F32) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_complex_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_complex_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_complex_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_complex_div_f32; break;
                    default: break;
                }
            } else if (src0_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_complex_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_complex_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_complex_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_complex_div_f16; break;
                    default: break;
                }
            }
            break;
        case HTP_BINARY_KERNEL_REPEAT:
            worker_func = binary_thread_element_repeat;
            if (src0_type == HTP_TYPE_F32) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_repeat_add_f32; break;
                    case HTP_OP_SUB: compute_func = compute_repeat_sub_f32; break;
                    case HTP_OP_MUL: compute_func = compute_repeat_mul_f32; break;
                    case HTP_OP_DIV: compute_func = compute_repeat_div_f32; break;
                    default: break;
                }
            } else if (src0_type == HTP_TYPE_F16) {
                switch (octx->op) {
                    case HTP_OP_ADD: compute_func = compute_repeat_add_f16; break;
                    case HTP_OP_SUB: compute_func = compute_repeat_sub_f16; break;
                    case HTP_OP_MUL: compute_func = compute_repeat_mul_f16; break;
                    case HTP_OP_DIV: compute_func = compute_repeat_div_f16; break;
                    default: break;
                }
            }
            break;
        case HTP_BINARY_KERNEL_ADD_ID:
            if (octx->op == HTP_OP_ADD_ID && src0_type == HTP_TYPE_F32) {
                worker_func = binary_thread_add_id_f32;
                compute_func = (void *) compute_add_id_f32;
            }
            break;
        default: break;
    }

    if (!worker_func || !compute_func) {
        return HTP_STATUS_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    if (kparams->kernel_type == HTP_BINARY_KERNEL_ROW_BCAST ||
        kparams->kernel_type == HTP_BINARY_KERNEL_SCALAR_DMA ||
        kparams->kernel_type == HTP_BINARY_KERNEL_ADD_ID) {
        dma_queue_pop(dma_q);
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &bctx, n_threads);

    return HTP_STATUS_OK;
}

int op_binary(struct htp_ops_context * octx) {

    // Does not support permutations of src1
    const struct htp_tensor * src1 = octx->src[1];
    if (src1->nb[1] < src1->nb[0]) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t src0_type = octx->src[0]->type;
    if ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        return execute_op_binary(octx);
    }

    return HTP_STATUS_NO_SUPPORT;
}
