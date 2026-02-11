#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Context for binary operations
struct htp_binary_context {
    struct htp_ops_context * octx;
    struct fastdiv_values dim1_div;
    struct fastdiv_values dim2_div;
    struct fastdiv_values dim12_div;

    struct fastdiv_values src1_dim1_div; // ne11
    struct fastdiv_values src1_dim2_div; // ne12
    struct fastdiv_values src1_dim3_div; // ne13

    uint32_t nrows_per_thread;
    bool split_at_ne01;
    bool split_at_ne02;

    // Precomputed values
    uint32_t block_max;
    size_t   src0_row_size_aligned;
    size_t   src1_row_size_aligned;
    size_t   dst_row_size_aligned;
    uint32_t src1_fetch_rows; // 1 or block_max
    uint32_t src1_dma_stride; // 0 or stride
};

#define htp_binary_preamble            \
    const struct htp_tensor * src0 = &octx->src0; \
    const struct htp_tensor * src1 = &octx->src1; \
    struct htp_tensor *       dst  = &octx->dst;  \
                                       \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne10 = src1->ne[0]; \
    const uint32_t ne11 = src1->ne[1]; \
    const uint32_t ne12 = src1->ne[2]; \
    const uint32_t ne13 = src1->ne[3]; \
                                       \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

static inline uint32_t calc_block_size(struct htp_binary_context * bctx, uint32_t ir, uint32_t end_row,
                                uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->dim1_div);
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

// Macro for scalar op switch
#define COMPUTE_SCALAR_OP(DST, SRC, VAL, N) \
    switch (octx->op) { \
        case HTP_OP_ADD: hvx_add_scalar_f32_aa(DST, SRC, VAL, N); break; \
        case HTP_OP_SUB: hvx_sub_scalar_f32_aa(DST, SRC, VAL, N); break; \
        case HTP_OP_MUL: hvx_mul_scalar_f32_aa(DST, SRC, VAL, N); break; \
        case HTP_OP_DIV: hvx_mul_scalar_f32_aa(DST, SRC, 1.0f / (VAL), N); break; \
        default: break; \
    }

// Macro for vector op switch (All Aligned)
#define COMPUTE_VECTOR_OP_AAA(DST, SRC0, SRC1, N) \
    switch (octx->op) { \
        case HTP_OP_ADD: hvx_add_f32_aaa(DST, SRC0, SRC1, N); break; \
        case HTP_OP_SUB: hvx_sub_f32_aaa(DST, SRC0, SRC1, N); break; \
        case HTP_OP_MUL: hvx_mul_f32_aaa(DST, SRC0, SRC1, N); break; \
        case HTP_OP_DIV: hvx_div_f32_aaa(DST, SRC0, SRC1, N); break; \
        default: break; \
    }

// Macro for vector op switch (Dst Aligned, Src0 Aligned, Src1 Unaligned)
#define COMPUTE_VECTOR_OP_AAU(DST, SRC0, SRC1, N) \
    switch (octx->op) { \
        case HTP_OP_ADD: hvx_add_f32_aau(DST, SRC0, SRC1, N); break; \
        case HTP_OP_SUB: hvx_sub_f32_aau(DST, SRC0, SRC1, N); break; \
        case HTP_OP_MUL: hvx_mul_f32_aau(DST, SRC0, SRC1, N); break; \
        case HTP_OP_DIV: hvx_div_f32_aau(DST, SRC0, SRC1, N); break; \
        default: break; \
    }

// Macro for vector op switch (All Unaligned - generic loop used in element repeat)
#define COMPUTE_VECTOR_OP_UUU(DST, SRC0, SRC1, N) \
    switch (octx->op) { \
        case HTP_OP_ADD: hvx_add_f32_uuu(DST, SRC0, SRC1, N); break; \
        case HTP_OP_SUB: hvx_sub_f32_uuu(DST, SRC0, SRC1, N); break; \
        case HTP_OP_MUL: hvx_mul_f32_uuu(DST, SRC0, SRC1, N); break; \
        case HTP_OP_DIV: hvx_div_f32_uuu(DST, SRC0, SRC1, N); break; \
        default: break; \
    }

// 1. Scalar src1 (ne10 == 1)
static void binary_job_scalar(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);
    size_t src0_spad_half    = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half     = octx->dst_spad.size_per_thread  / 2;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    // Preamble
    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    // Main loop
    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        // src1 indices (broadcast/repeat)
        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);

        uint8_t * src1_ptr = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        uint32_t s1_stride = (ne11 == 1) ? 0 : nb11;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = s0_spad + r * bctx->src0_row_size_aligned;
            uint8_t * r_dst  = d_spad + r * bctx->dst_row_size_aligned;
            float val = *(float *)src1_ptr;
            src1_ptr += s1_stride;
            COMPUTE_SCALAR_OP(r_dst, r_src0, val, ne00);
        }

        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;

             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
static void binary_job_vector_same_shape(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * src1_spad_base = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t src1_spad_half = octx->src1_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * src1_base = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        dma_queue_push(q, dma_make_ptr(s1_spad, src1_base), bctx->src1_row_size_aligned, bctx->src1_dma_stride, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_pop(q).dst;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = s0_spad + r * bctx->src0_row_size_aligned;
            uint8_t * r_src1 = s1_spad + r * bctx->src1_row_size_aligned;
            uint8_t * r_dst  = d_spad  + r * bctx->dst_row_size_aligned;
            COMPUTE_VECTOR_OP_AAA(r_dst, r_src0, r_src1, ne00);
        }

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;
        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;

             uint32_t p13 = (ne13 == 1) ? 0 : p03;
             uint32_t p12 = (ne12 == 1) ? 0 : p02;
             uint32_t p11 = (ne11 == 1) ? 0 : p01;

             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             uint8_t * s1_next = (uint8_t *)src1->data + p13 * nb13 + p12 * nb12 + p11 * nb11;

             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             dma_queue_push(q, dma_make_ptr(s1_spad, s1_next), bctx->src1_row_size_aligned, bctx->src1_dma_stride, ne00 * sizeof(float), next_block_size);

             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
static void binary_job_vector_row_broadcast(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * src1_spad = octx->src1_spad.data + (ith * octx->src1_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    void * s1_ptr = (void *) src1_spad;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint8_t * r_src0 = s0_spad + r * bctx->src0_row_size_aligned;
            uint8_t * r_src1 = (uint8_t *)s1_ptr; // Constant
            uint8_t * r_dst  = d_spad + r * bctx->dst_row_size_aligned;
            COMPUTE_VECTOR_OP_AAA(r_dst, r_src0, r_src1, ne00);
        }

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;
        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
static void binary_job_vector_complex(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);
    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint32_t r_i01 = i01 + r;
            uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
            uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
            uint32_t i11 = fastmodulo(r_i01, ne11, &bctx->src1_dim1_div);

            uint8_t * r_src0 = s0_spad + r * bctx->src0_row_size_aligned;
            uint8_t * r_src1 = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
            uint8_t * r_dst  = d_spad + r * bctx->dst_row_size_aligned;

            // Read src1 from DDR (unaligned)
            COMPUTE_VECTOR_OP_AAU(r_dst, r_src0, r_src1, ne00);
        }

        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 5. Element Repeat (ne10 != ne00)
static void binary_job_element_repeat(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;
    htp_binary_preamble;

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);
    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint32_t r_i01 = i01 + r;
            uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
            uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
            uint32_t i11 = fastmodulo(r_i01, ne11, &bctx->src1_dim1_div);

            uint8_t * r_src0 = s0_spad + r * bctx->src0_row_size_aligned;
            uint8_t * r_src1_row = (uint8_t *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
            uint8_t * r_dst  = d_spad + r * bctx->dst_row_size_aligned;

            // Repeat src1 row
            for (uint32_t c = 0; c < ne00; c += ne10) {
                uint32_t len = MIN(ne10, ne00 - c);
                // Use UUU for speed and simplicity
                COMPUTE_VECTOR_OP_UUU(r_dst + c * sizeof(float), r_src0 + c * sizeof(float), r_src1_row, len);
            }
        }

        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
static void binary_job_add_id(unsigned int nth, unsigned int ith, void * data) {
    struct htp_binary_context * bctx = (struct htp_binary_context *) data;
    struct htp_ops_context * octx = bctx->octx;

    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    const struct htp_tensor * src2 = &octx->src2;
    struct htp_tensor *       dst  = &octx->dst;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];
    const uint32_t ne11 = src1->ne[1]; // for bounds check

    const uint32_t nb01 = src0->nb[1];
    const uint32_t nb02 = src0->nb[2];
    const uint32_t nb03 = src0->nb[3];
    const uint32_t nb11 = src1->nb[1]; // src1 row stride
    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    const uint32_t total_rows = ne01 * ne02 * ne03;
    const uint32_t start_row = bctx->nrows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, total_rows);
    if (start_row >= end_row) return;

    uint8_t * src0_spad_base = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_base  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);
    size_t src0_spad_half = octx->src0_spad.size_per_thread / 2;
    size_t dst_spad_half  = octx->dst_spad.size_per_thread  / 2;

    dma_queue * q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    int spad_idx = 0;

    for (int k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        uint8_t * src0_curr = (uint8_t *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        uint8_t * dst_curr  = (uint8_t *)dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, 0);
        dma_queue_push(q, dma_make_ptr(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    for (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_pop(q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_pop(q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->dim1_div);
        i01 = rem - i02 * ne01;

        for (uint32_t r = 0; r < current_block_size; r++) {
            uint32_t r_i01 = i01 + r; // linear within block since we split at ne01

            const int32_t idx = *(int32_t *)((char *)src2->data + r_i01 * src2->nb[0] + i02 * src2->nb[1]);

            uint8_t * r_src1 = (uint8_t *)src1->data + idx * nb11;
            uint8_t * r_src0 = s0_spad + r * bctx->src0_row_size_aligned;
            uint8_t * r_dst  = d_spad + r * bctx->dst_row_size_aligned;

            hvx_add_f32_aau(r_dst, r_src0, r_src1, ne00);
        }

        uint8_t * dst_curr = (uint8_t *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_push(q, dma_make_ptr(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, ne00 * sizeof(float), current_block_size);

        if (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03 = fastdiv(ir_prefetch, &bctx->dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02 = fastdiv(prem, &bctx->dim1_div);
             p01 = prem - p02 * ne01;
             uint8_t * s0_next = (uint8_t *)src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_push(q, dma_make_ptr(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, ne00 * sizeof(float), next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }
    dma_queue_flush(q);
}

static int execute_op_binary_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    const uint32_t n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation
    const size_t src0_row_size = src0->ne[0] * sizeof(float);
    const size_t src1_row_size = src1->ne[0] * sizeof(float);
    const size_t dst_row_size  = dst->ne[0] * sizeof(float);

    // Align to VLEN
    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);
    size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);

    bool is_add_id = (octx->op == HTP_OP_ADD_ID);
    bool is_scalar = !is_add_id && (src1->ne[0] == 1);

    // Determine which kernel we will use to alloc memory and dispatch
    bool use_vector_same = !is_add_id && !is_scalar && src1->ne[0] == src0->ne[0] &&
               (src1->ne[1] == src0->ne[1] || src1->ne[1] == 1) &&
               (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
               (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);

    bool is_row_bcast = use_vector_same && (src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);
    bool use_complex = !is_add_id && !is_scalar && !use_vector_same && (src1->ne[0] == src0->ne[0]);
    bool use_repeat  = !is_add_id && !is_scalar && !use_vector_same && (src1->ne[0] != src0->ne[0]);

    size_t spad_row_total;
    if (is_scalar) {
        spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    } else if (is_row_bcast) {
        spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    } else if (use_vector_same) {
        spad_row_total = 2 * (src0_row_size_aligned + src1_row_size_aligned + dst_row_size_aligned);
    } else if (is_add_id) {
        spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned); // src1 read directly
    } else {
        spad_row_total = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    }

    size_t rows_per_buffer = octx->ctx->vtcm_size / (n_threads * spad_row_total);
    // Adjust for static src1 in row_bcast case
    if (is_row_bcast) {
        size_t needed_static = src1_row_size_aligned;
        if (octx->ctx->vtcm_size < needed_static) return HTP_STATUS_VTCM_TOO_SMALL;
        size_t avail = octx->ctx->vtcm_size - needed_static;
        rows_per_buffer = avail / (n_threads * spad_row_total);
    }

    if (rows_per_buffer < 1) {
         FARF(ERROR, "binary-f32: VTCM too small\n");
         return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.size_per_thread = rows_per_buffer * 2 * src0_row_size_aligned;
    octx->dst_spad.size_per_thread  = rows_per_buffer * 2 * dst_row_size_aligned;

    if (is_scalar || use_complex || use_repeat || is_add_id) {
        octx->src1_spad.size_per_thread = 0;
    } else if (is_row_bcast) {
        octx->src1_spad.size_per_thread = 0;
    } else {
        octx->src1_spad.size_per_thread = rows_per_buffer * 2 * src1_row_size_aligned;
    }

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    if (is_row_bcast) {
        octx->src1_spad.size = src1_row_size_aligned;
    } else {
        octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;
    }
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    if (octx->ctx->vtcm_size < (octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size)) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        return HTP_STATUS_OK;
    }

    uint32_t n_jobs = MIN(n_threads, src0_nrows);

    dma_queue * q = octx->ctx->dma[0];
    if (is_row_bcast) {
        dma_queue_push(q, dma_make_ptr(octx->src1_spad.data, (const void *) src1->data), src1_row_size_aligned, 0, src1->ne[0] * sizeof(float), 1);
    }

    struct htp_binary_context bctx;
    bctx.octx = octx;
    bctx.nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;
    bctx.block_max = rows_per_buffer;
    bctx.src0_row_size_aligned = src0_row_size_aligned;
    bctx.src1_row_size_aligned = src1_row_size_aligned;
    bctx.dst_row_size_aligned  = dst_row_size_aligned;

    bctx.dim1_div = init_fastdiv_values(src0->ne[1]);
    bctx.dim2_div = init_fastdiv_values(src0->ne[2]);
    bctx.dim12_div = init_fastdiv_values(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div = init_fastdiv_values(src1->ne[1]);
    bctx.src1_dim2_div = init_fastdiv_values(src1->ne[2]);
    bctx.src1_dim3_div = init_fastdiv_values(src1->ne[3]);

    bool src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    bool dst_contig_dim1  = (dst->nb[2] == src0->ne[1] * dst->nb[1]);

    bool src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    bool dst_contig_dim2  = (dst->nb[3] == src0->ne[2] * dst->nb[2]);

    bctx.split_at_ne01 = (src0->ne[2] > 1) &&
                         ((src1->ne[1] > 1) || (src1->ne[2] > 1) || !src0_contig_dim1 || !dst_contig_dim1);

    bctx.split_at_ne02 = (src0->ne[3] > 1) &&
                         ((src1->ne[2] > 1) || (src1->ne[3] > 1) || !src0_contig_dim2 || !dst_contig_dim2);

    // Precompute specific kernel parameters
    if (use_vector_same) {
        bctx.src1_dma_stride = (src1->ne[1] == 1) ? 0 : src1->nb[1];
        bctx.src1_fetch_rows = (src1->ne[1] == 1) ? 1 : rows_per_buffer;
    }

    worker_callback_t worker_func;
    if (is_add_id) worker_func = binary_job_add_id;
    else if (is_scalar) worker_func = binary_job_scalar;
    else if (is_row_bcast) worker_func = binary_job_vector_row_broadcast;
    else if (use_vector_same) worker_func = binary_job_vector_same_shape;
    else if (use_complex) worker_func = binary_job_vector_complex;
    else worker_func = binary_job_element_repeat;

    if (is_row_bcast) {
        dma_queue_pop(q);
    }

    worker_pool_run_func(octx->ctx->worker_pool, worker_func, &bctx, n_jobs);

    return HTP_STATUS_OK;
}

int op_binary(struct htp_ops_context * octx) {
    if (octx->src0.type == HTP_TYPE_F32) {
        return execute_op_binary_f32(octx);
    }
    return HTP_STATUS_NO_SUPPORT;
}
