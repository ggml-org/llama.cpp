#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>

#include "hex-dma.h"
#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"

/* Circular wrap: maps any integer x into [0, n) */
static inline uint32_t wrap_around(int32_t x, uint32_t n) {
    return (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
static inline void pad_decompose_row(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
static inline int pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    return ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
static inline const uint8_t * pad_src_row_ptr(const struct htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    return (const uint8_t *) src->data
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
static inline const uint8_t * pad_circ_src_row_ptr(const struct htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    return (const uint8_t *) src->data
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

struct htp_pad_context {
    struct htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;

    size_t   type_size;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_row_size;
    size_t   src_row_size_aligned;
    size_t   dst_row_size;
    size_t   dst_row_size_aligned;
};

#define htp_pad_preamble                            \
    const struct htp_tensor * src = octx->src[0];   \
    const struct htp_tensor * dst = octx->dst;      \
                                                    \
    const uint32_t ne00 = src->ne[0];               \
    const uint32_t nb00 = src->nb[0];               \
                                                    \
    const uint32_t ne0 = dst->ne[0];                \
    const uint32_t ne1 = dst->ne[1];                \
    const uint32_t ne2 = dst->ne[2];                \
    const uint32_t ne3 = dst->ne[3];                \
                                                    \
    const uint32_t nb1 = dst->nb[1];                \
    const uint32_t nb2 = dst->nb[2];                \
    const uint32_t nb3 = dst->nb[3];                \
                                                    \
    const int32_t lp0 = pctx->lp0, rp0 = pctx->rp0; \
    const int32_t lp1 = pctx->lp1, rp1 = pctx->rp1; \
    const int32_t lp2 = pctx->lp2, rp2 = pctx->rp2; \
    const int32_t lp3 = pctx->lp3, rp3 = pctx->rp3; \
                                                    \
    const size_t type_size = pctx->type_size;       \
                                                    \
    const uint32_t row_start = pctx->nrows_per_thread * ith;                                 \
    const uint32_t row_end   = MIN(row_start + pctx->nrows_per_thread, pctx->total_dst_rows);


#define htp_pad_dma_preamble                                        \
    const size_t src_row_size         = pctx->src_row_size;         \
    const size_t src_row_size_aligned = pctx->src_row_size_aligned; \
    const size_t dst_row_size         = pctx->dst_row_size;         \
    const size_t dst_row_size_aligned = pctx->dst_row_size_aligned; \
                                                                    \
    uint8_t * src_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread; \
    uint8_t * dst_spad_base = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;  \
                                                                                            \
    dma_queue * dma = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

static void pad_job_per_thread_hvx(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_pad_context * pctx = (const struct htp_pad_context *) data;
    struct htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    for (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        uint32_t i1, i2, i3;
        pad_decompose_row(dst_row, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        const int interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        if (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } else {
            const uint8_t * src_ptr = pad_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            if (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_row_start = dst_ptr + (size_t)lp0 * type_size;
            if (nb00 == type_size) {
                hvx_copy_f32_uu(dst_row_start, src_ptr, ne00);
            } else {
                for (uint32_t i = 0; i < ne00; i++) {
                    memcpy(dst_row_start + i * type_size,
                           src_ptr + (size_t)i * nb00,
                           type_size);
                }
            }

            if (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_size, 0.0f, (uint32_t)rp0);
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u usec %u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel — aligned, double-buffered
// ---------------------------------------------------------------------------

static void pad_job_per_thread_hvx_dma(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_pad_context * pctx = (const struct htp_pad_context *) data;
    struct htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    for (uint32_t ir = row_start, spad_idx = 0; ir < row_end && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_base + spad_idx * src_row_size_aligned;
        uint8_t * dst_spad_cur = dst_spad_base + spad_idx * dst_row_size_aligned;

        dma_queue_push_vtcm_to_ddr(dma,
            dma_make_ptr((uint8_t *)dst->data, dst_spad_cur),
            dst_row_size, dst_row_size_aligned, 0);

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);
        const int interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        const uint8_t * src_ptr = interior
            ? pad_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3) : NULL;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_queue_push_ddr_to_vtcm(dma,
            dma_make_ptr(src_spad_cur,
                         src_ptr ? src_ptr : (const uint8_t *)src_spad_cur),
            src_row_size_aligned, src_row_size, src_ptr ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    for (uint32_t ir = row_start; ir < row_end; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_queue_pop(dma).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_queue_pop(dma).dst;

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        const int interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        if (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } else {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_size;

            if ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } else {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }

        dma_queue_push_vtcm_to_ddr(dma,
            dma_make_ptr(dst_ptr, dst_spad_cur),
            dst_row_size, dst_row_size_aligned, 1);

        const uint32_t next_row = ir + 2;
        if (next_row < row_end) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_row(next_row, ne1, ne2, &ni1, &ni2, &ni3);
            const int next_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            const uint8_t * next_src_ptr = next_interior
                ? pad_src_row_ptr(src, ni1, ni2, ni3, lp1, lp2, lp3) : NULL;

            dma_queue_push_ddr_to_vtcm(dma,
                dma_make_ptr(src_spad_cur,
                             next_src_ptr ? next_src_ptr : (const uint8_t *)src_spad_cur),
                src_row_size_aligned, src_row_size, next_src_ptr ? 1 : 0);
        }
    }

    dma_queue_flush(dma);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u usec %u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

static void pad_job_per_thread_hvx_circular(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_pad_context * pctx = (const struct htp_pad_context *) data;
    struct htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    for (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        uint32_t i1, i2, i3;
        pad_decompose_row(dst_row, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;
        const uint8_t * src_row = pad_circ_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        if (nb00 == type_size) {

            if (lp0 > 0) {
                if ((uint32_t)lp0 < 32) {
                    memcpy(dst_ptr,
                           src_row + (size_t)(ne00 - (uint32_t)lp0) * type_size,
                           (size_t)lp0 * type_size);
                } else {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_row + (size_t)(ne00 - (uint32_t)lp0) * type_size,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_size, src_row, ne00);
            if (rp0 > 0) {
                if ((uint32_t)rp0 < 32) {
                    memcpy(dst_ptr + ((size_t)lp0 + ne00) * type_size,
                           src_row,
                           (size_t)rp0 * type_size);
                } else {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_size,
                                    src_row,
                                    (uint32_t)rp0);
                }
            }
        } else {
            for (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(float *)(dst_ptr + i * type_size) =
                    *(const float *)(src_row + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            for (uint32_t i = 0; i < ne00; i++) {
                *(float *)(dst_ptr + ((size_t)lp0 + i) * type_size) =
                    *(const float *)(src_row + (size_t)i * nb00);
            }
            for (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(float *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_size) =
                    *(const float *)(src_row + (size_t)i * nb00);
            }
        }
    }

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u usec %u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel — aligned, double-buffered
// ---------------------------------------------------------------------------

static void pad_job_per_thread_hvx_circular_dma(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_pad_context * pctx = (const struct htp_pad_context *) data;
    struct htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    for (uint32_t ir = row_start, spad_idx = 0; ir < row_end && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_base + spad_idx * src_row_size_aligned;
        uint8_t * dst_spad_cur = dst_spad_base + spad_idx * dst_row_size_aligned;

        dma_queue_push_vtcm_to_ddr(dma,
            dma_make_ptr((uint8_t *)dst->data, dst_spad_cur),
            dst_row_size, dst_row_size_aligned, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_row(ir, ne1, ne2, &pi1, &pi2, &pi3);
        dma_queue_push_ddr_to_vtcm(dma,
            dma_make_ptr(src_spad_cur, pad_circ_src_row_ptr(src, pi1, pi2, pi3, lp1, lp2, lp3)),
            src_row_size_aligned, src_row_size, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    for (uint32_t ir = row_start; ir < row_end; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_queue_pop(dma).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_queue_pop(dma).dst;

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);
        uint8_t * dst_ptr = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;


        if (lp0 > 0) {
            uint8_t * dst_left       = dst_spad_cur;
            const uint8_t * src_left = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_size;
            if ((uint32_t)lp0 < 32) {
                memcpy(dst_left, src_left, (size_t)lp0 * type_size);
            } else {
                hvx_copy_f32_uu(dst_left, src_left, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_size;
            if ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } else {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        if (rp0 > 0) {
            uint8_t * dst_right = dst_spad_cur + ((size_t)lp0 + ne00) * type_size;
            if ((uint32_t)rp0 < 32) {
                memcpy(dst_right, src_spad_cur, (size_t)rp0 * type_size);
            } else {
                if ((uintptr_t)dst_right % VLEN == 0) {
                    hvx_copy_f32_aa(dst_right, src_spad_cur, (uint32_t)rp0);
                } else {
                    hvx_copy_f32_ua(dst_right, src_spad_cur, (uint32_t)rp0);
                }
            }
        }

        dma_queue_push_vtcm_to_ddr(dma,
            dma_make_ptr(dst_ptr, dst_spad_cur),
            dst_row_size, dst_row_size_aligned, 1);

        const uint32_t next_row = ir + 2;
        if (next_row < row_end) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_row(next_row, ne1, ne2, &nri1, &nri2, &nri3);
            dma_queue_push_ddr_to_vtcm(dma,
                dma_make_ptr(src_spad_cur,
                             pad_circ_src_row_ptr(src, nri1, nri2, nri3, lp1, lp2, lp3)),
                src_row_size_aligned, src_row_size, 1);
        }
    }

    dma_queue_flush(dma);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u usec %u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end,
         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

int op_pad(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_size;
    switch (src0->type) {
        case HTP_TYPE_F32: type_size = 4; break;
        default:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const int32_t lp0 = octx->op_params[0];
    const int32_t rp0 = octx->op_params[1];
    const int32_t lp1 = octx->op_params[2];
    const int32_t rp1 = octx->op_params[3];
    const int32_t lp2 = octx->op_params[4];
    const int32_t rp2 = octx->op_params[5];
    const int32_t lp3 = octx->op_params[6];
    const int32_t rp3 = octx->op_params[7];
    const int32_t circular = octx->op_params[8];

    const uint32_t ne0  = dst->ne[0];
    const uint32_t ne00 = src0->ne[0];

    const uint32_t total_dst_rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const uint32_t n_threads = MIN(octx->n_threads, total_dst_rows > 0 ? total_dst_rows : 1);

    const size_t src_row_size         = (size_t)ne00 * type_size;
    const size_t dst_row_size         = (size_t)ne0  * type_size;
    const size_t src_row_size_aligned = hex_round_up(src_row_size, VLEN);
    const size_t dst_row_size_aligned = hex_round_up(dst_row_size, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    const size_t vtcm_needed = (size_t)n_threads * 2 * (src_row_size_aligned + dst_row_size_aligned);

    const int use_dma = (src0->nb[0] == (uint32_t)type_size) &&
                        (ne00 >= 512) &&
                        (octx->ctx->vtcm_base != NULL) &&
                        (octx->ctx->vtcm_size >= vtcm_needed);

    if (use_dma) {
        octx->src0_spad.size_per_thread = 2 * src_row_size_aligned;
        octx->dst_spad.size_per_thread  = 2 * dst_row_size_aligned;
        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;
        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size;
    }

    struct htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_per_thread = (total_dst_rows + n_threads - 1) / n_threads,
        .total_dst_rows   = total_dst_rows,
        .type_size        = type_size,
        .src_row_size         = src_row_size,
        .src_row_size_aligned = src_row_size_aligned,
        .dst_row_size         = dst_row_size,
        .dst_row_size_aligned = dst_row_size_aligned,
    };

    FARF(HIGH, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    if      (circular && use_dma) { worker_pool_run_func(octx->ctx->worker_pool, pad_job_per_thread_hvx_circular_dma, &pctx, n_threads); }
    else if (circular)            { worker_pool_run_func(octx->ctx->worker_pool, pad_job_per_thread_hvx_circular,     &pctx, n_threads); }
    else if (use_dma)             { worker_pool_run_func(octx->ctx->worker_pool, pad_job_per_thread_hvx_dma,          &pctx, n_threads); }
    else                          { worker_pool_run_func(octx->ctx->worker_pool, pad_job_per_thread_hvx,              &pctx, n_threads); }

    return HTP_STATUS_OK;
}

