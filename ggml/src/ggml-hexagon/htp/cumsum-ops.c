#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-types.h"
#include "hvx-utils.h"
#include "hex-dma.h"

#define htp_cumsum_tensors_preamble                  \
    struct htp_tensor * restrict src0 = &octx->src0; \
    struct htp_tensor * restrict dst  = &octx->dst;  \
                                                     \
    const uint32_t ne00 = src0->ne[0];               \
    const uint32_t ne01 = src0->ne[1];               \
    const uint32_t ne02 = src0->ne[2];               \
    const uint32_t ne03 = src0->ne[3];               \
                                                     \
    const uint32_t ne0 = dst->ne[0];                 \
    const uint32_t ne1 = dst->ne[1];                 \
    const uint32_t ne2 = dst->ne[2];                 \
    const uint32_t ne3 = dst->ne[3];                 \
                                                     \
    const uint32_t nb00 = src0->nb[0];               \
    const uint32_t nb01 = src0->nb[1];               \
    const uint32_t nb02 = src0->nb[2];               \
    const uint32_t nb03 = src0->nb[3];               \
                                                     \
    const uint32_t nb0 = dst->nb[0];                 \
    const uint32_t nb1 = dst->nb[1];                 \
    const uint32_t nb2 = dst->nb[2];                 \
    const uint32_t nb3 = dst->nb[3];

struct htp_cumsum_context {
    struct htp_ops_context * octx;
    size_t          src_row_size;
    size_t          dst_row_size;
    size_t          src_row_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
};

#define htp_cumsum_preamble                                                \
    struct htp_cumsum_context * cctx = (struct htp_cumsum_context *) data; \
    struct htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_queue = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

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

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    return v;
}

static inline HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    return hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

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
        float acc = hvx_vec_get_f32(carry);
        const float * src_tail = src + nvec * VLEN_FP32;
        float       * dst_tail = dst + nvec * VLEN_FP32;
        for (uint32_t i = 0; i < nloe; i++) {
            acc        += src_tail[i];
            dst_tail[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

static void cumsum_thread_f32_dma(unsigned int nth, unsigned int ith, void * data) {
    htp_cumsum_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t ir0 = cctx->rows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->total_rows);

    if (ir0 >= ir1) {
        return;
    }

    const size_t src_row_size         = cctx->src_row_size;
    const size_t dst_row_size         = cctx->dst_row_size;
    const size_t src_row_size_aligned = cctx->src_row_size_aligned;
    const size_t dst_row_size_aligned = cctx->dst_row_size_aligned;

    const uint8_t * src_data = (const uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    uint8_t * src_spad = octx->src0_spad.data + (ith * src_row_size_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned * 2);

    for (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_push_vtcm_to_ddr(dma_queue,
                                   dma_make_ptr(dst_data, dst_spad + (spad_idx * dst_row_size_aligned)),
                                   dst_row_size, dst_row_size_aligned, 0);

        dma_queue_push_ddr_to_vtcm(dma_queue,
                                   dma_make_ptr(src_spad + (spad_idx * src_row_size_aligned),
                                                src_data + (ir * src_row_size)),
                                   src_row_size_aligned, src_row_size, 1);
    }

    for (uint32_t ir = ir0; ir < ir1; ir++) {
        float * dst_spad_row = (float *) dma_queue_pop(dma_queue).src;
        float * src_spad_row = (float *) dma_queue_pop(dma_queue).dst;

        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);

        dma_queue_push_vtcm_to_ddr(dma_queue,
                                   dma_make_ptr(dst_data + (ir * dst_row_size), (uint8_t *) dst_spad_row),
                                   dst_row_size, dst_row_size_aligned, 1);

        const uint32_t next_row = ir + 2;
        if (next_row < ir1) {
            dma_queue_push_ddr_to_vtcm(dma_queue,
                                       dma_make_ptr((uint8_t *) src_spad_row, src_data + (next_row * src_row_size)),
                                       src_row_size_aligned, src_row_size, 1);
        }
    }

    dma_queue_flush(dma_queue);
    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

static void cumsum_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    htp_cumsum_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * src_data = (const uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    const uint32_t ir0 = cctx->rows_per_thread * ith;
    const uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->total_rows);

    for (uint32_t ir = ir0; ir < ir1; ir++) {
        const float * restrict src_row = (const float *) (src_data + ir * cctx->src_row_size);
        float       * restrict dst_row = (float *)       (dst_data + ir * cctx->dst_row_size);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

int op_cumsum_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * dst  = &octx->dst;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = MIN(octx->n_threads, total_rows);

    const size_t src_row_size         = src0->nb[1];
    const size_t dst_row_size         = dst->nb[1];
    const size_t src_row_size_aligned = hex_round_up(src_row_size, VLEN);
    const size_t dst_row_size_aligned = hex_round_up(dst_row_size, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    const size_t spad_per_thread = 2 * (src_row_size_aligned + dst_row_size_aligned);

    octx->src0_spad.size_per_thread = src_row_size_aligned * 2;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * 2;
    octx->src0_spad.size            = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size             = n_threads * octx->dst_spad.size_per_thread;
    octx->src0_spad.data            = octx->ctx->vtcm_base;
    octx->dst_spad.data             = octx->src0_spad.data + octx->src0_spad.size;

    struct htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_size         = src_row_size,
        .dst_row_size         = dst_row_size,
        .src_row_size_aligned = src_row_size_aligned,
        .dst_row_size_aligned = dst_row_size_aligned,
        .rows_per_thread      = (total_rows + n_threads - 1) / n_threads,
        .total_rows           = total_rows,
    };

    if (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        worker_pool_run_func(octx->ctx->worker_pool, cumsum_thread_f32, &cctx, n_threads);
    } else {
        worker_pool_run_func(octx->ctx->worker_pool, cumsum_thread_f32_dma, &cctx, n_threads);
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
