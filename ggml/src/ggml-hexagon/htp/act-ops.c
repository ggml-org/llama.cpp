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

#define htp_act_preamble3              \
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
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb10 = src1->nb[0]; \
    const uint32_t nb11 = src1->nb[1]; \
    const uint32_t nb12 = src1->nb[2]; \
    const uint32_t nb13 = src1->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

#define htp_act_preamble2              \
    const uint32_t ne00 = src0->ne[0]; \
    const uint32_t ne01 = src0->ne[1]; \
    const uint32_t ne02 = src0->ne[2]; \
    const uint32_t ne03 = src0->ne[3]; \
                                       \
    const uint32_t ne0 = dst->ne[0];   \
    const uint32_t ne1 = dst->ne[1];   \
    const uint32_t ne2 = dst->ne[2];   \
    const uint32_t ne3 = dst->ne[3];   \
                                       \
    const uint32_t nb00 = src0->nb[0]; \
    const uint32_t nb01 = src0->nb[1]; \
    const uint32_t nb02 = src0->nb[2]; \
    const uint32_t nb03 = src0->nb[3]; \
                                       \
    const uint32_t nb0 = dst->nb[0];   \
    const uint32_t nb1 = dst->nb[1];   \
    const uint32_t nb2 = dst->nb[2];   \
    const uint32_t nb3 = dst->nb[3];

struct htp_act_context {
    struct htp_ops_context *  octx;

    // Precomputed values
    const uint8_t *           data_src0;
    const uint8_t *           data_src1;
    uint8_t *                 data_dst;

    size_t                    src0_row_size;
    size_t                    src1_row_size;
    size_t                    dst_row_size;

    size_t                    src0_row_size_aligned;
    size_t                    src1_row_size_aligned;
    size_t                    dst_row_size_aligned;

    size_t                    src0_spad_half_size;
    size_t                    src1_spad_half_size;
    size_t                    dst_spad_half_size;

    uint32_t                  block;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_per_thread;
    int                       nc;
};

static void glu_swiglu_f32_per_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_act_context * actx = (struct htp_act_context *) data;
    const struct htp_tensor * src0 = &actx->octx->src0;
    const struct htp_tensor * src1 = &actx->octx->src1;
    const struct htp_tensor * dst  = &actx->octx->dst;
    htp_act_preamble3;

    size_t src0_row_size = actx->src0_row_size;
    size_t src1_row_size = actx->src1_row_size;
    size_t dst_row_size  = actx->dst_row_size;

    const uint32_t src0_nrows = actx->src0_nrows;
    const uint32_t src0_nrows_per_thread = actx->src0_nrows_per_thread;
    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict data_src0 = actx->data_src0;
    const uint8_t * restrict data_src1 = actx->data_src1;
    uint8_t * restrict data_dst        = actx->data_dst;

    const int  nc = actx->nc;

    const size_t src0_row_size_aligned = actx->src0_row_size_aligned;
    const size_t src1_row_size_aligned = actx->src1_row_size_aligned;
    const size_t dst_row_size_aligned  = actx->dst_row_size_aligned;

    uint8_t * restrict src0_spad_data = actx->octx->src0_spad.data + (ith * actx->octx->src0_spad.size_per_thread);
    uint8_t * restrict src1_spad_data = actx->octx->src1_spad.data + (ith * actx->octx->src1_spad.size_per_thread);
    uint8_t * restrict dst_spad_data  = actx->octx->dst_spad.data + (ith * actx->octx->dst_spad.size_per_thread);

    size_t src0_spad_half_size = actx->src0_spad_half_size;
    size_t src1_spad_half_size = actx->src1_spad_half_size;
    size_t dst_spad_half_size  = actx->dst_spad_half_size;

    const int BLOCK = actx->block;
    if (BLOCK == 0) {
        FARF(ERROR,
             "swiglu-f32 : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n",
             actx->octx->src0_spad.size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_queue = actx->octx->ctx->dma[ith];

    // See discussion: https://github.com/ggml-org/llama.cpp/pull/18151#issuecomment-3678235379
    for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        // Dummy DMA transation for sequencing (interleaving dst,src,dst,...)
        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(data_dst, dst_spad_data + (spad_idx * dst_spad_half_size)),
            dst_row_size, dst_row_size_aligned, 0);

        dma_queue_push_ddr_to_vtcm(dma_queue,
            dma_make_ptr(src0_spad_data + (spad_idx * src0_spad_half_size), data_src0 + (ir * src0_row_size)),
            src0_row_size_aligned, src0_row_size, block_size);
        dma_queue_push_ddr_to_vtcm(dma_queue,
            dma_make_ptr(src1_spad_data + (spad_idx * src1_spad_half_size), data_src1 + (ir * src1_row_size)),
            src1_row_size_aligned, src1_row_size, block_size);
    }

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        float * dst_spad  = (float *) dma_queue_pop(dma_queue).src;
        float * src0_spad = (float *) dma_queue_pop(dma_queue).dst;
        float * src1_spad = (float *) dma_queue_pop(dma_queue).dst;

        for (uint32_t ib = 0; ib < block_size; ib++) {
            const float * src0_spad_ptr = src0_spad + ib * (src0_row_size_aligned / sizeof(float));
            const float * src1_spad_ptr = src1_spad + ib * (src1_row_size_aligned / sizeof(float));
            float *       dst_spad_ptr  = dst_spad + ib * (dst_row_size_aligned / sizeof(float));

            //swiglu(x) = x1 * sigmoid(x0)
            hvx_sigmoid_f32_aa((uint8_t *) dst_spad_ptr, (const uint8_t *) src0_spad_ptr, nc);
            hvx_mul_mul_f32_aa((uint8_t *) dst_spad_ptr, (const uint8_t *) src0_spad_ptr, (const uint8_t *) dst_spad_ptr,
                                (const uint8_t *) src1_spad_ptr, nc);
        }

        dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(data_dst + (ir * dst_row_size), dst_spad), dst_row_size,
                                   dst_row_size_aligned, block_size);

        // prefetch N+2 loop iteration if any
        const uint32_t pref_block = (ir + BLOCK * 2);
        if (pref_block < src0_end_row) {
            const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_block);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src0_spad, data_src0 + (pref_block * src0_row_size)),
                                       src0_row_size_aligned, src0_row_size, pref_block_size);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src1_spad, data_src1 + (pref_block * src1_row_size)),
                                       src1_row_size_aligned, src1_row_size, pref_block_size);
        }
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "swiglu-f32 %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, src0_start_row, src0_end_row, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static void glu_swiglu_oai_f32_per_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_act_context * actx = (struct htp_act_context *) data;
    const struct htp_tensor * src0 = &actx->octx->src0;
    const struct htp_tensor * src1 = &actx->octx->src1;
    const struct htp_tensor * dst  = &actx->octx->dst;
    htp_act_preamble3;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    size_t src0_row_size = actx->src0_row_size;
    size_t src1_row_size = actx->src1_row_size;
    size_t dst_row_size  = actx->dst_row_size;

    const uint32_t src0_nrows = actx->src0_nrows;
    const uint32_t src0_nrows_per_thread = actx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const uint8_t * restrict data_src0 = actx->data_src0;
    const uint8_t * restrict data_src1 = actx->data_src1;
    uint8_t * restrict data_dst        = actx->data_dst;

    const int nc = actx->nc;

    const size_t src0_row_size_aligned = actx->src0_row_size_aligned;
    const size_t src1_row_size_aligned = actx->src1_row_size_aligned;
    const size_t dst_row_size_aligned  = actx->dst_row_size_aligned;

    uint8_t * restrict src0_spad_data = actx->octx->src0_spad.data + (ith * actx->octx->src0_spad.size_per_thread);
    uint8_t * restrict src1_spad_data = actx->octx->src1_spad.data + (ith * actx->octx->src1_spad.size_per_thread);
    uint8_t * restrict dst_spad_data  = actx->octx->dst_spad.data + (ith * actx->octx->dst_spad.size_per_thread);

    size_t src0_spad_half_size = actx->src0_spad_half_size;
    size_t src1_spad_half_size = actx->src1_spad_half_size;
    size_t dst_spad_half_size  = actx->dst_spad_half_size;

    const int BLOCK = actx->block;
    if (BLOCK == 0) {
        FARF(ERROR,
             "swiglu-oai-f32 : current VTCM reservation %zu is too small for even 1 row per thread, needed at least "
             "%zu\n",
             actx->octx->src0_spad.size_per_thread, src0_row_size_aligned);
        return;
    }
    const float alpha = ((const float *) (actx->octx->op_params))[2];
    const float limit = ((const float *) (actx->octx->op_params))[3];

    dma_queue * dma_queue = actx->octx->ctx->dma[ith];

    // See discussion: https://github.com/ggml-org/llama.cpp/pull/18151#issuecomment-3678235379
    for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        // Dummy DMA transation for sequencing (interleaving dst,src,dst,...)
        dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(data_dst, dst_spad_data + (spad_idx * dst_spad_half_size)),
                                   dst_row_size, dst_row_size_aligned, 0);

        dma_queue_push_ddr_to_vtcm(
            dma_queue,
            dma_make_ptr(src0_spad_data + (spad_idx * src0_spad_half_size), data_src0 + (ir * src0_row_size)),
            src0_row_size_aligned, src0_row_size, block_size);
        dma_queue_push_ddr_to_vtcm(
            dma_queue,
            dma_make_ptr(src1_spad_data + (spad_idx * src1_spad_half_size), data_src1 + (ir * src1_row_size)),
            src1_row_size_aligned, src1_row_size, block_size);
    }

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        float * dst_spad  = (float *) dma_queue_pop(dma_queue).src;
        float * src0_spad = (float *) dma_queue_pop(dma_queue).dst;
        float * src1_spad = (float *) dma_queue_pop(dma_queue).dst;

        for (uint32_t ib = 0; ib < block_size; ib++) {
            const float * src0_spad_ptr = src0_spad + ib * (src0_row_size_aligned / sizeof(float));
            const float * src1_spad_ptr = src1_spad + ib * (src1_row_size_aligned / sizeof(float));
            float *       dst_spad_ptr  = dst_spad + ib * (dst_row_size_aligned / sizeof(float));

            // x (src0_spad_data) = std::min(src0_p[k], limit);
            hvx_min_scalar_f32((uint8_t *) src0_spad_ptr, (const uint8_t *) src0_spad_ptr, limit, nc);
            // y1 (src1_spad_data) = std::clamp(src1_p[k], -limit, limit);
            hvx_clamp_scalar_f32((uint8_t *) src1_spad_ptr, (const uint8_t *) src1_spad_ptr, -limit, limit, nc);
            // y (src1_spad_data)  = y1 + 1.f
            hvx_add_scalar_f32((uint8_t *) src1_spad_ptr, (const uint8_t *) src1_spad_ptr, 1.0, nc);
            // x1 (dst_spad_data) = alpha * (x)
            hvx_mul_scalar_f32((uint8_t *) dst_spad_ptr, (const uint8_t *) src0_spad_ptr, alpha, nc);
            // x2 (dst_spad_data) = sigmoid(x1) = 1/(1+exp(-x1))
            hvx_sigmoid_f32_aa((uint8_t *) dst_spad_ptr, (const uint8_t *) dst_spad_ptr, nc);
            // out = x * sigmoid(alpha * x) * (y + 1.f)
            hvx_mul_mul_f32_aa((uint8_t *) dst_spad_ptr, (const uint8_t *) src0_spad_ptr, (const uint8_t *) dst_spad_ptr,
                                (const uint8_t *) src1_spad_ptr, nc);
        }

        dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(data_dst + (ir * dst_row_size), dst_spad), dst_row_size,
                                   dst_row_size_aligned, block_size);

        // prefetch N+2 loop iteration if any
        const uint32_t pref_block = (ir + BLOCK * 2);
        if (pref_block < src0_end_row) {
            const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_block);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src0_spad, data_src0 + (pref_block * src0_row_size)),
                                       src0_row_size_aligned, src0_row_size, pref_block_size);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src1_spad, data_src1 + (pref_block * src1_row_size)),
                                       src1_row_size_aligned, src1_row_size, pref_block_size);
        }
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "swiglu-oai-f32 %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src0_start_row, src0_end_row, src1->ne[0], src1->ne[1], src1->ne[2],
         src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}


static void unary_gelu_f32_per_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_act_context * actx = (struct htp_act_context *) data;
    const struct htp_tensor * src0 = &actx->octx->src0;
    const struct htp_tensor * dst  = &actx->octx->dst;
    htp_act_preamble2;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const size_t src0_row_size = actx->src0_row_size;
    const size_t dst_row_size  = actx->dst_row_size;
    const size_t src0_row_size_aligned = actx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = actx->dst_row_size_aligned;

    const uint32_t src0_nrows = actx->src0_nrows;
    const uint32_t src0_nrows_per_thread = actx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const uint8_t * data_src0 = actx->data_src0;
    uint8_t * data_dst        = actx->data_dst;

    // nc/ne0 matches.
    const int ne0_val = actx->nc; // == dst->ne[0]

    uint8_t * src0_spad_data = actx->octx->src0_spad.data + (ith * actx->octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_data  = actx->octx->dst_spad.data  + (ith * actx->octx->dst_spad.size_per_thread);

    size_t src0_spad_half_size = actx->src0_spad_half_size;
    size_t dst_spad_half_size  = actx->dst_spad_half_size;

    // In gelu = x*sigmoid(x*1.702)
    const int BLOCK = actx->block;

    if (BLOCK == 0) {
        FARF(ERROR, "gelu-f32 : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n",
                actx->octx->src0_spad.size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_queue = actx->octx->ctx->dma[ith];

    // See discussion: https://github.com/ggml-org/llama.cpp/pull/18151#issuecomment-3678235379
    for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        // Dummy DMA transation for sequencing (interleaving dst,src,dst,...)
        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(data_dst, dst_spad_data + (spad_idx * dst_spad_half_size)),
            dst_row_size, dst_row_size_aligned, 0);

        dma_queue_push_ddr_to_vtcm(dma_queue,
            dma_make_ptr(src0_spad_data + (spad_idx * src0_spad_half_size), data_src0 + (ir * src0_row_size)),
            src0_row_size_aligned, src0_row_size, block_size);
    }

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        float* dst_spad  = (float *) dma_queue_pop(dma_queue).src;
        float* src0_spad = (float *) dma_queue_pop(dma_queue).dst;

        for (uint32_t ib = 0; ib < block_size; ib++) {
            const float* src0_spad_ptr = src0_spad + ib * (src0_row_size_aligned / sizeof(float));
            float* dst_spad_ptr        = dst_spad  + ib * (dst_row_size_aligned  / sizeof(float));

            // gelu = x * sigmoid(1.702 * x) // current implementation
            hvx_mul_scalar_f32((uint8_t *) dst_spad_ptr, (const uint8_t *) src0_spad_ptr, (float) 1.702, ne0_val);
            hvx_sigmoid_f32_aa((uint8_t *) dst_spad_ptr, (const uint8_t *) dst_spad_ptr, ne0_val);
            hvx_mul_f32_aaa((uint8_t *) dst_spad_ptr, (const uint8_t *) src0_spad_ptr, (const uint8_t *) dst_spad_ptr, ne0_val);
        }

        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(data_dst + (ir * dst_row_size), dst_spad),
            dst_row_size, dst_row_size_aligned, block_size);

        // prefetch N+2 loop iteration if any
        const uint32_t pref_block = (ir + BLOCK * 2);
        if (pref_block < src0_end_row) {
            const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_block);
            dma_queue_push_ddr_to_vtcm(dma_queue,
                dma_make_ptr(src0_spad, data_src0 + (pref_block * src0_row_size)),
                src0_row_size_aligned, src0_row_size, pref_block_size);
        }
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "gelu-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n", ith, nth, ne00, ne01, ne02,
         ne03, src0_start_row, src0_end_row, ne0, ne1, ne2, ne3, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}


static void unary_silu_f32_per_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_act_context * actx = (struct htp_act_context *) data;
    const struct htp_tensor * src0 = &actx->octx->src0;
    const struct htp_tensor * dst  = &actx->octx->dst;
    htp_act_preamble2;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const size_t src0_row_size = actx->src0_row_size;
    const size_t dst_row_size  = actx->dst_row_size;
    const size_t src0_row_size_aligned = actx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = actx->dst_row_size_aligned;

    const uint32_t src0_nrows = actx->src0_nrows;
    const uint32_t src0_nrows_per_thread = actx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const uint8_t * data_src0 = actx->data_src0;
    uint8_t * data_dst        = actx->data_dst;

    const int ne0_val = actx->nc; // == dst->ne[0]

    uint8_t * src0_spad_data = actx->octx->src0_spad.data + (ith * actx->octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_data  = actx->octx->dst_spad.data  + (ith * actx->octx->dst_spad.size_per_thread);

    size_t src0_spad_half_size = actx->src0_spad_half_size;
    size_t dst_spad_half_size  = actx->dst_spad_half_size;

    const int BLOCK = actx->block;

    if (BLOCK == 0) {
        FARF(ERROR, "silu-f32 : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n",
                actx->octx->src0_spad.size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_queue = actx->octx->ctx->dma[ith];

    // See discussion: https://github.com/ggml-org/llama.cpp/pull/18151#issuecomment-3678235379
    for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        // Dummy DMA transation for sequencing (interleaving dst,src,dst,...)
        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(data_dst, dst_spad_data + (spad_idx * dst_spad_half_size)),
            dst_row_size, dst_row_size_aligned, 0);

        dma_queue_push_ddr_to_vtcm(dma_queue,
            dma_make_ptr(src0_spad_data + (spad_idx * src0_spad_half_size), data_src0 + (ir * src0_row_size)),
            src0_row_size_aligned, src0_row_size, block_size);
    }

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        float* dst_spad  = (float *) dma_queue_pop(dma_queue).src;
        float* src0_spad = (float *) dma_queue_pop(dma_queue).dst;

        for (uint32_t ib = 0; ib < block_size; ib++) {
            const float* src0_spad_ptr = src0_spad + ib * (src0_row_size_aligned / sizeof(float));
            float* dst_spad_ptr        = dst_spad  + ib * (dst_row_size_aligned  / sizeof(float));

            // silu = x * sigmoid(x)
            hvx_sigmoid_f32_aa((uint8_t *) dst_spad_ptr, (const uint8_t *) src0_spad_ptr, ne0_val);
            hvx_mul_f32_aaa((uint8_t *) dst_spad_ptr, (const uint8_t *) src0_spad_ptr, (const uint8_t *) dst_spad_ptr, ne0_val);
        }

        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(data_dst + (ir * dst_row_size), dst_spad),
            dst_row_size, dst_row_size_aligned, block_size);

        // prefetch N+2 loop iteration if any
        const uint32_t pref_block = (ir + BLOCK * 2);
        if (pref_block < src0_end_row) {
            const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_block);
            dma_queue_push_ddr_to_vtcm(dma_queue,
                dma_make_ptr(src0_spad, data_src0 + (pref_block * src0_row_size)),
                src0_row_size_aligned, src0_row_size, pref_block_size);
        }
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "silu-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n", ith, nth, ne00, ne01, ne02,
         ne03, src0_start_row, src0_end_row, ne0, ne1, ne2, ne3, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static const float GELU_COEF_A     = 0.044715f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

static void glu_geglu_f32_per_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_act_context * actx = (struct htp_act_context *) data;
    const struct htp_tensor * src0 = &actx->octx->src0;
    const struct htp_tensor * src1 = &actx->octx->src1;
    const struct htp_tensor * dst  = &actx->octx->dst;
    htp_act_preamble3;

    size_t src0_row_size = actx->src0_row_size;
    size_t src1_row_size = actx->src1_row_size;
    size_t dst_row_size  = actx->dst_row_size;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows = actx->src0_nrows;
    const uint32_t src0_nrows_per_thread = actx->src0_nrows_per_thread;

    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    const uint8_t * restrict data_src0 = actx->data_src0;
    const uint8_t * restrict data_src1 = actx->data_src1;
    uint8_t * restrict data_dst        = actx->data_dst;

    const int nc = actx->nc;

    const size_t src0_row_size_aligned = actx->src0_row_size_aligned;
    const size_t src1_row_size_aligned = actx->src1_row_size_aligned;
    const size_t dst_row_size_aligned  = actx->dst_row_size_aligned;

    uint8_t * restrict src0_spad_data = actx->octx->src0_spad.data + (ith * actx->octx->src0_spad.size_per_thread);
    uint8_t * restrict src1_spad_data = actx->octx->src1_spad.data + (ith * actx->octx->src1_spad.size_per_thread);
    uint8_t * restrict dst_spad_data  = actx->octx->dst_spad.data + (ith * actx->octx->dst_spad.size_per_thread);

    size_t src0_spad_half_size = actx->src0_spad_half_size;
    size_t src1_spad_half_size = actx->src1_spad_half_size;
    size_t dst_spad_half_size  = actx->dst_spad_half_size;

    const int BLOCK = actx->block;
    if (BLOCK == 0) {
        FARF(ERROR,
             "geglu-f32 : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n",
             actx->octx->src0_spad.size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_queue = actx->octx->ctx->dma[ith];

    // See discussion: https://github.com/ggml-org/llama.cpp/pull/18151#issuecomment-3678235379
    for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        // Dummy DMA transation for sequencing (interleaving dst,src,dst,...)
        dma_queue_push_vtcm_to_ddr(dma_queue,
            dma_make_ptr(data_dst, dst_spad_data + (spad_idx * dst_spad_half_size)),
            dst_row_size, dst_row_size_aligned, 0);

        dma_queue_push_ddr_to_vtcm(dma_queue,
            dma_make_ptr(src0_spad_data + (spad_idx * src0_spad_half_size), data_src0 + (ir * src0_row_size)),
            src0_row_size_aligned, src0_row_size, block_size);
        dma_queue_push_ddr_to_vtcm(dma_queue,
            dma_make_ptr(src1_spad_data + (spad_idx * src1_spad_half_size), data_src1 + (ir * src1_row_size)),
            src1_row_size_aligned, src1_row_size, block_size);
    }

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {
        const uint32_t block_size = MIN(BLOCK, src0_end_row - ir);

        float * dst_spad  = (float *) dma_queue_pop(dma_queue).src;
        float * src0_spad = (float *) dma_queue_pop(dma_queue).dst;
        float * src1_spad = (float *) dma_queue_pop(dma_queue).dst;

        for (uint32_t ib = 0; ib < block_size; ib++) {
            const uint8_t * src0_spad_ptr = (const uint8_t *)(src0_spad + ib * (src0_row_size_aligned / sizeof(float)));
            const uint8_t * src1_spad_ptr = (const uint8_t *)(src1_spad + ib * (src1_row_size_aligned / sizeof(float)));
            uint8_t *       dst_spad_ptr  = (uint8_t *)(dst_spad + ib * (dst_row_size_aligned / sizeof(float)));

            // geglu tanh implementation
            // geglu(x, g) = gelu(x) * g
            // gelu(x) = 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)))
            hvx_mul_f32_aaa(dst_spad_ptr, src0_spad_ptr, src0_spad_ptr, nc);                       // res = x*x
            hvx_mul_scalar_f32_aa(dst_spad_ptr, (const uint8_t *)dst_spad_ptr, GELU_COEF_A, nc);   // res = res * GELU_COEF_A
            hvx_add_scalar_f32_aa(dst_spad_ptr, (const uint8_t *)dst_spad_ptr, 1.0f, nc);          // res = res + 1.0f
            hvx_mul_f32_aaa(dst_spad_ptr, src0_spad_ptr, (const uint8_t *)dst_spad_ptr, nc);       // res = res * x
            hvx_mul_scalar_f32_aa(dst_spad_ptr, (const uint8_t*)dst_spad_ptr, SQRT_2_OVER_PI, nc); // res = result * SQRT_2_OVER_PI
            hvx_tanh_f32_aa((uint8_t *) dst_spad_ptr, (const uint8_t *) dst_spad_ptr, nc);         // res = tanh(res)
            hvx_add_scalar_f32_aa(dst_spad_ptr, (const uint8_t*)dst_spad_ptr, 1.0f, nc);           // res = res + 1.0f
            hvx_mul_f32_aaa(dst_spad_ptr, src0_spad_ptr, (const uint8_t *)dst_spad_ptr, nc);       // res = res * x
            hvx_mul_scalar_f32_aa(dst_spad_ptr, (const uint8_t *)dst_spad_ptr, 0.5f, nc);          // res = res + 0.5f
            hvx_mul_f32_aaa(dst_spad_ptr, (const uint8_t *)dst_spad_ptr, src1_spad_ptr, nc);       // res = res * g
        }

        dma_queue_push_vtcm_to_ddr(dma_queue, dma_make_ptr(data_dst + (ir * dst_row_size), dst_spad), dst_row_size,
                                   dst_row_size_aligned, block_size);

        // prefetch N+2 loop iteration if any
        const uint32_t pref_block = (ir + BLOCK * 2);
        if (pref_block < src0_end_row) {
            const uint32_t pref_block_size = MIN(BLOCK, src0_end_row - pref_block);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src0_spad, data_src0 + (pref_block * src0_row_size)),
                                       src0_row_size_aligned, src0_row_size, pref_block_size);
            dma_queue_push_ddr_to_vtcm(dma_queue, dma_make_ptr(src1_spad, data_src1 + (pref_block * src1_row_size)),
                                       src1_row_size_aligned, src1_row_size, pref_block_size);
        }
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "geglu-f32 %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, src0_start_row, src0_end_row, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static int execute_op_activations_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * src1 = &octx->src1;
    struct htp_tensor *       dst  = &octx->dst;

    if (((src0->ne[0] * SIZEOF_FP32) != src0->nb[1]) || ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1])) {
        FARF(ERROR, "Non-contiguous tensors are not supported at this time \n");
        return HTP_STATUS_NO_SUPPORT;
    }

    worker_callback_t act_op_func;
    const char *      op_type = NULL;

    switch (octx->op) {
        case HTP_OP_UNARY_SILU:
            act_op_func = (worker_callback_t)unary_silu_f32_per_thread;
            op_type     = "silu-f32";
            break;

        case HTP_OP_GLU_SWIGLU:
            act_op_func = (worker_callback_t)glu_swiglu_f32_per_thread;
            op_type     = "swiglu-f32";
            break;

        case HTP_OP_GLU_SWIGLU_OAI:
            act_op_func = (worker_callback_t)glu_swiglu_oai_f32_per_thread;
            op_type     = "swiglu-oai-f32";
            break;
        case HTP_OP_UNARY_GELU:
            act_op_func = (worker_callback_t)unary_gelu_f32_per_thread;
            op_type     = "gelu-f32";
            break;

        case HTP_OP_GLU_GEGLU:
            act_op_func = (worker_callback_t)glu_geglu_f32_per_thread;
            op_type     = "geglu-f32";
            break;
        default:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t n_threads  = octx->n_threads;
    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    size_t src0_row_size = src0->nb[1];
    size_t src1_row_size = src1->nb[1]; // zero bytes if src1 is not used
    size_t dst_row_size  = dst->nb[1];

    const bool src1_valid = src1->ne[0];
    if (!src1_valid) {
        src1_row_size = src0_row_size;
    }

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, VLEN);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size, VLEN);
    // VTCM scratchpads for all tensors
    // N rows per thread, padded to HVX vector size

    size_t spad_size_per_row   = (src0_row_size_aligned + src1_row_size_aligned) + dst_row_size_aligned;
    size_t vtcm_row_per_thread = (octx->ctx->vtcm_size)/ (n_threads* spad_size_per_row);

    // Make sure the reserved vtcm size is sufficient
    if(vtcm_row_per_thread ==0){
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_size,
             spad_size_per_row * n_threads);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.size_per_thread = src0_row_size_aligned * vtcm_row_per_thread;
    octx->src1_spad.size_per_thread = src1_row_size_aligned * vtcm_row_per_thread;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * vtcm_row_per_thread;

    octx->dst_spad.size  = n_threads* octx->dst_spad.size_per_thread;
    octx->src0_spad.size = n_threads* octx->src0_spad.size_per_thread;
    octx->src1_spad.size = n_threads* octx->src1_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;

    if (src1->ne[0]) {
        FARF(HIGH, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], octx->src0_spad.size, octx->src1_spad.size,
             octx->dst_spad.size);
    } else {
        FARF(HIGH, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size);
    }

    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        return HTP_STATUS_OK;
    }

    uint32_t n_jobs = MIN(n_threads, src0_nrows);

    // Prepare context
    struct htp_act_context actx;
    actx.octx = octx;

    actx.src0_nrows_per_thread = (src0_nrows + n_jobs - 1) / n_jobs;

    actx.src0_row_size = src0_row_size;
    actx.src1_row_size = src1_row_size;
    actx.dst_row_size  = dst_row_size;

    actx.src0_row_size_aligned = src0_row_size_aligned;
    actx.src1_row_size_aligned = src1_row_size_aligned;
    actx.dst_row_size_aligned  = dst_row_size_aligned;

    actx.src0_spad_half_size = octx->src0_spad.size_per_thread / 2;
    actx.src1_spad_half_size = octx->src1_spad.size_per_thread / 2;
    actx.dst_spad_half_size  = octx->dst_spad.size_per_thread / 2;

    actx.block = actx.src0_spad_half_size / actx.src0_row_size_aligned;
    actx.src0_nrows = src0_nrows;

    actx.nc = dst->ne[0];

    // Pointers and GLU logic
    const uint8_t * data_src0 = (const uint8_t *) src0->data;
    const uint8_t * data_src1 = (const uint8_t *) src1->data;

    if (!src1_valid && (octx->op == HTP_OP_GLU_SWIGLU || octx->op == HTP_OP_GLU_SWIGLU_OAI || octx->op == HTP_OP_GLU_GEGLU)) {
         const int32_t swapped = octx->op_params[1];
         data_src1 = data_src0;
         actx.src1_row_size = actx.src0_row_size;

         size_t nc_in_bytes = actx.nc * SIZEOF_FP32;
         if (swapped) {
             data_src0 += nc_in_bytes;
         } else {
             data_src1 += nc_in_bytes;
         }
    }

    actx.data_src0 = data_src0;
    actx.data_src1 = data_src1;
    actx.data_dst  = (uint8_t *) dst->data;

    worker_pool_run_func(octx->ctx->worker_pool, act_op_func, &actx, n_jobs);
    return HTP_STATUS_OK;
}

int op_activations(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src0.type) {
        case HTP_TYPE_F32:
            err = execute_op_activations_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
