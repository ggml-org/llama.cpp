#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#include "hex-common.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-vtcm.h"
#include "htp-tensor.h"

struct htp_im2col_context {
    struct htp_ops_context * octx;
    uint32_t                 patch_base;           // first patch index assigned to this dev
    uint32_t                 npatches;             // number of patches assigned to this dev
    uint32_t                 npatches_per_thread;  // patches = N*OH*OW (pure-DDR kernel)

    uint32_t pe_row_base;                          // first N*OH row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of N*OH rows assigned to this dev (DMA path)
    uint32_t pe_rows_per_thread;                   // N*OH rows per worker
    uint32_t pe_src_row_bytes;                     // one output row's source: IC*KH*IW*4, rounded 256
    uint32_t pe_dst_row_bytes;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_size_per_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_size_per_thread;              // 2 * pe_dst_row_bytes

    uint32_t pe_owb;                               // output-col block size
    uint32_t pe_wb;                                // staged source window width
};

// Per-op VTCM layout for the patch-embed DMA path
struct htp_im2col_vtcm_layout {
    size_t off_src;
    size_t off_dst;
    size_t src_bytes_per_thread;
    size_t dst_bytes_per_thread;
    size_t total_bytes;
};

static inline void htp_im2col_vtcm_layout_build(struct htp_im2col_vtcm_layout * L,
                                                size_t                          src_row_bytes,
                                                size_t                          dst_row_bytes,
                                                uint32_t                        n_threads) {
    L->src_bytes_per_thread = 2 * src_row_bytes;
    L->dst_bytes_per_thread = 2 * dst_row_bytes;

    L->off_src     = 0;
    L->off_dst     = L->off_src + L->src_bytes_per_thread * n_threads;
    L->total_bytes = L->off_dst + L->dst_bytes_per_thread * n_threads;
}

#define IM2COL_PATCHEMBED_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN, DST_ELEM, TAG)                         \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                   \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                      \
        struct htp_ops_context *    octx        = ictx->octx;                                              \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                                  \
        const struct htp_tensor * restrict src0 = octx->src[0];                                            \
        const struct htp_tensor * restrict src1 = octx->src[1];                                            \
        const struct htp_tensor * restrict dst  = octx->dst;                                               \
        const int32_t  s0                       = octx->op_params[0];                                      \
        const int32_t  s1                       = octx->op_params[1];                                      \
        const int32_t  p0                       = octx->op_params[2];                                      \
        const int32_t  p1                       = octx->op_params[3];                                      \
        const int32_t  d0                       = octx->op_params[4];                                      \
        const int32_t  d1                       = octx->op_params[5];                                      \
        const int32_t  is_2D                    = octx->op_params[6] == 1;                                 \
        const uint32_t N                        = is_2D ? src1->ne[3] : src1->ne[2];                       \
        const uint32_t IC                       = is_2D ? src1->ne[2] : src1->ne[1];                       \
        const uint32_t IH                       = is_2D ? src1->ne[1] : 1;                                 \
        const uint32_t IW                       = src1->ne[0];                                             \
        const uint32_t KH                       = is_2D ? src0->ne[1] : 1;                                 \
        const uint32_t KW                       = src0->ne[0];                                             \
        const uint32_t OH                       = is_2D ? dst->ne[2] : 1;                                  \
        const uint32_t OW                       = dst->ne[1];                                              \
        const uint32_t patch_stride             = IC * KH * KW;                                            \
        const float * restrict src_data         = (const float *) (uintptr_t) src1->data;                  \
        DST_CTYPE * restrict dst_data           = (DST_CTYPE *) (uintptr_t) dst->data;                     \
        const uint32_t patch_end                = ictx->patch_base + ictx->npatches;                       \
        const uint32_t patch_start              = ictx->patch_base + ictx->npatches_per_thread * ith;      \
        const uint32_t patch_stop               = MIN(patch_start + ictx->npatches_per_thread, patch_end); \
        if (patch_start >= patch_stop) {                                                                   \
            return;                                                                                        \
        }                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, patch_start);                                    \
        for (uint32_t p = patch_start; p < patch_stop; p++) {                                              \
            const uint32_t iow             = p % OW;                                                       \
            const uint32_t ioh             = (p / OW) % OH;                                                \
            const uint32_t in              = p / (OW * OH);                                                \
            DST_CTYPE * restrict dst_patch = dst_data + (uint64_t) p * patch_stride;                       \
            for (uint32_t iic = 0; iic < IC; iic++) {                                                      \
                const float * restrict src_plane = src_data + ((uint64_t) in * IC + iic) * IH * IW;        \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                  \
                    const int32_t iih            = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;           \
                    DST_CTYPE * restrict out_run = dst_patch + iic * (KH * KW) + ikh * KW;                 \
                    if (iih < 0 || iih >= (int32_t) IH) {                                                  \
                        SPLAT_FN(out_run, 0.0f, KW);                                                       \
                        continue;                                                                          \
                    }                                                                                      \
                    const int32_t iiw0             = (int32_t) iow * s0 - p0;                              \
                    const float * restrict src_run = src_plane + (uint64_t) iih * IW + iiw0;               \
                    if (d0 == 1) {                                                                         \
                        /* contiguous source run: [lo,hi) is in-bounds, tails are zero pad */              \
                        const int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                           \
                        int32_t       hi = (int32_t) IW - iiw0;                                            \
                        if (hi > (int32_t) KW) {                                                           \
                            hi = (int32_t) KW;                                                             \
                        }                                                                                  \
                        if (hi <= lo) {                                                                    \
                            SPLAT_FN(out_run, 0.0f, KW);                                                   \
                        } else {                                                                           \
                            if (lo > 0) {                                                                  \
                                SPLAT_FN(out_run, 0.0f, (uint32_t) lo);                                    \
                            }                                                                              \
                            COPY_FN((uint8_t *) (out_run + lo), (const uint8_t *) (src_run + lo),          \
                                    (uint32_t) (hi - lo));                                                 \
                            if (hi < (int32_t) KW) {                                                       \
                                SPLAT_FN(out_run + hi, 0.0f, (KW - (uint32_t) hi));                        \
                            }                                                                              \
                        }                                                                                  \
                        continue;                                                                          \
                    }                                                                                      \
                    for (uint32_t ikw = 0; ikw < KW; ikw++) {                                              \
                        const int32_t iiw = (int32_t) iow * s0 + (int32_t) ikw * d0 - p0;                  \
                        out_run[ikw]      = (iiw < 0 || iiw >= (int32_t) IW) ?                             \
                                                (DST_CTYPE) 0.0f :                                         \
                                                (DST_CTYPE) src_plane[(uint64_t) iih * IW + iiw];          \
                    }                                                                                      \
                }                                                                                          \
            }                                                                                              \
        }                                                                                                  \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, patch_start);                                     \
    }

IM2COL_PATCHEMBED_BODY(im2col_patchembed_thread, __fp16, hvx_copy_f16_f32_uu, hvx_splat_f16_u, sizeof(__fp16), "f32-f16")
IM2COL_PATCHEMBED_BODY(im2col_patchembed_f32_thread, float, hvx_copy_f32_uu, hvx_splat_f32_u, sizeof(float), "f32-f32")

// Software-pipelined 2-deep: while HVX computes block bi from buffer slot
// (bi&1), the DMA engine stages block bi+1 into the other slot concurrently.
// A single dma_queue_flush per iteration (after issuing the next stage-in and
// this block's store-out) waits for both - safe because the ring is strict
// FIFO and each buffer slot is only reused after its prior consumer (compute
// or store-out) already finished in program order.
#define IM2COL_BLOCKED_DMA_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN, DST_ELEM, TAG)                                            \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                                       \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                                          \
        struct htp_ops_context *    octx        = ictx->octx;                                                                  \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                                                      \
        const struct htp_tensor * restrict src1 = octx->src[1];                                                                \
        const struct htp_tensor * restrict dst  = octx->dst;                                                                   \
        const int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                                                       \
        const int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                                                       \
        const int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                                                       \
        const int32_t  is_2D = octx->op_params[6] == 1;                                                                        \
        const uint32_t N     = is_2D ? src1->ne[3] : src1->ne[2];                                                              \
        const uint32_t IC    = is_2D ? src1->ne[2] : src1->ne[1];                                                              \
        const uint32_t IH    = is_2D ? src1->ne[1] : 1;                                                                        \
        const uint32_t IW    = src1->ne[0];                                                                                    \
        const uint32_t KH    = is_2D ? octx->src[0]->ne[1] : 1;                                                                \
        const uint32_t KW    = octx->src[0]->ne[0];                                                                            \
        const uint32_t OH    = is_2D ? dst->ne[2] : 1;                                                                         \
        const uint32_t OW    = dst->ne[1];                                                                                     \
        const uint32_t owb = ictx->pe_owb, Wb = ictx->pe_wb;                                                                   \
        const uint32_t patch_stride     = IC * KH * KW;                                                                        \
        const dma_addr_t src_data       = src1->data;                                                                          \
        const dma_addr_t dst_data       = dst->data;                                                                           \
        dma_queue *    dmaq             = octx->ctx->dma[ith];                                                                 \
        uint8_t *      srcb_base        = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;                              \
        uint8_t *      dstb_base        = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;                              \
        float *        srcb2[2]         = { (float *) srcb_base, (float *) (srcb_base + ictx->pe_src_row_bytes) };             \
        DST_CTYPE *    dstb2[2]   = { (DST_CTYPE *) dstb_base, (DST_CTYPE *) (dstb_base + ictx->pe_dst_row_bytes) };           \
        const uint32_t nrows      = N * OH;                                                                                    \
        const uint32_t per_thread = ictx->pe_rows_per_thread;                                                                  \
        const uint32_t row_start  = per_thread * ith;                                                                          \
        const uint32_t row_end    = MIN(row_start + per_thread, nrows);                                                        \
        if (row_start >= row_end)                                                                                              \
            return;                                                                                                            \
        const uint32_t nbpr         = (OW + owb - 1) / owb;                                                                    \
        const uint32_t nrows_local  = row_end - row_start;                                                                     \
        const uint32_t total_blocks = nrows_local * nbpr;                                                                      \
        for (uint32_t bi = 0; bi < total_blocks; bi++) {                                                                       \
            const uint32_t buf  = bi & 1u;                                                                                     \
            float *        srcb = srcb2[buf];                                                                                  \
            DST_CTYPE *    dstb = dstb2[buf];                                                                                  \
            const uint32_t r    = row_start + bi / nbpr;                                                                       \
            const uint32_t in   = r / OH;                                                                                      \
            const uint32_t ioh  = r % OH;                                                                                      \
            const uint32_t c0   = (bi % nbpr) * owb;                                                                           \
            const uint32_t nb   = MIN(owb, OW - c0);                                                                           \
            const int32_t  win0 = (int32_t) c0 * s0 - p0;                                                                      \
            if (bi == 0) {                                                                                                     \
                /* prologue: stage block 0 and wait - nothing to overlap with yet */                                           \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                                      \
                    const int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                                          \
                    if (iih < 0 || iih >= (int32_t) IH)                                                                        \
                        continue;                                                                                              \
                    const int32_t lo = win0 < 0 ? -win0 : 0;                                                                   \
                    int32_t       hi = (int32_t) IW - win0;                                                                    \
                    if (hi > (int32_t) Wb)                                                                                     \
                        hi = (int32_t) Wb;                                                                                     \
                    if (hi <= lo)                                                                                              \
                        continue;                                                                                              \
                    const uint32_t cpw  = (uint32_t) (hi - lo);                                                                \
                    float *        vdst = srcb + (size_t) ikh * Wb + lo;                                                       \
                    const dma_addr_t vsrc = src_data + (size_t) (((in * IC) * IH + iih) * IW + (win0 + lo)) * sizeof(float);   \
                    while (!dma_queue_push(dmaq, dma_make_data(vdst, vsrc),                                                    \
                                           (size_t) KH * Wb * sizeof(float), (size_t) IH * IW * sizeof(float),                 \
                                           cpw * sizeof(float), IC)) {                                                         \
                        dma_queue_pop(dmaq);                                                                                   \
                    }                                                                                                          \
                }                                                                                                              \
                dma_queue_flush(dmaq);                                                                                         \
            }                                                                                                                  \
            if (bi + 1 < total_blocks) {                                                                                       \
                /* prefetch: stage block bi+1 into the other slot; overlaps with this block's compute below */                 \
                const uint32_t nbuf  = 1u - buf;                                                                               \
                float *        nsrcb = srcb2[nbuf];                                                                            \
                const uint32_t nr    = row_start + (bi + 1) / nbpr;                                                            \
                const uint32_t nin   = nr / OH;                                                                                \
                const uint32_t nioh  = nr % OH;                                                                                \
                const uint32_t nc0   = ((bi + 1) % nbpr) * owb;                                                                \
                const int32_t  nwin0 = (int32_t) nc0 * s0 - p0;                                                                \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                                      \
                    const int32_t iih = (int32_t) nioh * s1 + (int32_t) ikh * d1 - p1;                                         \
                    if (iih < 0 || iih >= (int32_t) IH)                                                                        \
                        continue;                                                                                              \
                    const int32_t lo = nwin0 < 0 ? -nwin0 : 0;                                                                 \
                    int32_t       hi = (int32_t) IW - nwin0;                                                                   \
                    if (hi > (int32_t) Wb)                                                                                     \
                        hi = (int32_t) Wb;                                                                                     \
                    if (hi <= lo)                                                                                              \
                        continue;                                                                                              \
                    const uint32_t cpw  = (uint32_t) (hi - lo);                                                                \
                    float *        vdst = nsrcb + (size_t) ikh * Wb + lo;                                                      \
                    const dma_addr_t vsrc = src_data + (size_t) (((nin * IC) * IH + iih) * IW + (nwin0 + lo)) * sizeof(float); \
                    while (!dma_queue_push(dmaq, dma_make_data(vdst, vsrc),                                                    \
                                           (size_t) KH * Wb * sizeof(float), (size_t) IH * IW * sizeof(float),                 \
                                           cpw * sizeof(float), IC)) {                                                         \
                        dma_queue_pop(dmaq);                                                                                   \
                    }                                                                                                          \
                }                                                                                                              \
            }                                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);                                                              \
            for (uint32_t j = 0; j < nb; j++) {                                                                                \
                const uint32_t iow       = c0 + j;                                                                             \
                DST_CTYPE *    dst_patch = dstb + (uint64_t) j * patch_stride;                                                 \
                const int32_t  iiw0      = (int32_t) iow * s0 - p0;                                                            \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                                      \
                    const int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                                          \
                    const int     okh = (iih >= 0 && iih < (int32_t) IH);                                                      \
                    for (uint32_t iic = 0; iic < IC; iic++) {                                                                  \
                        DST_CTYPE * out_run = dst_patch + iic * (KH * KW) + ikh * KW;                                          \
                        if (!okh) {                                                                                            \
                            SPLAT_FN(out_run, 0.0f, KW);                                                                       \
                            continue;                                                                                          \
                        }                                                                                                      \
                        const float * vrow = srcb + ((uint64_t) (iic * KH + ikh)) * Wb; /* col win0 at idx 0*/                 \
                        if (d0 == 1) {                                                                                         \
                            /* contiguous run within the staged window: [lo,hi) in-bounds, tails zero pad */                   \
                            const int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                                           \
                            int32_t       hi = (int32_t) IW - iiw0;                                                            \
                            if (hi > (int32_t) KW) {                                                                           \
                                hi = (int32_t) KW;                                                                             \
                            }                                                                                                  \
                            if (hi <= lo) {                                                                                    \
                                SPLAT_FN(out_run, 0.0f, KW);                                                                   \
                            } else {                                                                                           \
                                if (lo > 0) {                                                                                  \
                                    SPLAT_FN(out_run, 0.0f, (uint32_t) lo);                                                    \
                                }                                                                                              \
                                COPY_FN((uint8_t *) (out_run + lo), (const uint8_t *) (vrow + (iiw0 + lo - win0)),             \
                                        (uint32_t) (hi - lo));                                                                 \
                                if (hi < (int32_t) KW) {                                                                       \
                                    SPLAT_FN(out_run + hi, 0.0f, (KW - (uint32_t) hi));                                        \
                                }                                                                                              \
                            }                                                                                                  \
                            continue;                                                                                          \
                        }                                                                                                      \
                        for (uint32_t ikw = 0; ikw < KW; ikw++) {                                                              \
                            const int32_t iiw = iiw0 + (int32_t) ikw * d0;                                                     \
                            out_run[ikw] =                                                                                     \
                                (iiw < 0 || iiw >= (int32_t) IW) ? (DST_CTYPE) 0.0f : (DST_CTYPE) vrow[iiw - win0];            \
                        }                                                                                                      \
                    }                                                                                                          \
                }                                                                                                              \
            }                                                                                                                  \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);                                                               \
            const dma_addr_t ddr = dst_data + (size_t) ((in * OH + ioh) * OW + c0) * patch_stride * (DST_ELEM);                \
            dma_queue_push(dmaq, dma_make_data(ddr, dstb),                                                                     \
                           nb * patch_stride * (DST_ELEM), nb * patch_stride * (DST_ELEM),                                     \
                           nb * patch_stride * (DST_ELEM), 1);                                                                 \
            dma_queue_flush(dmaq);                                                                                             \
        }                                                                                                                      \
    }
IM2COL_BLOCKED_DMA_BODY(im2col_blocked_dma_thread,     __fp16, hvx_copy_f16_f32_uu, hvx_splat_f16_u, sizeof(__fp16), "blk-dma-f16")
IM2COL_BLOCKED_DMA_BODY(im2col_blocked_dma_f32_thread, float,  hvx_copy_f32_uu,     hvx_splat_f32_u, sizeof(float),  "blk-dma-f32")

// Exact-tiling patch-embed DMA fast path (s0==KW, p0=0, d0=1; and 2D s1==KH,
// p1=0, d1=1). Intentionally reads no stride/pad/dilation params so the inner
// copy stays tight and fully hoisted - do NOT graft the general gather in here.
#define IM2COL_PATCHEMBED_DMA_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN, DST_ELEM, TAG)                           \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                         \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                            \
        struct htp_ops_context *    octx        = ictx->octx;                                                    \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                                        \
        const struct htp_tensor * restrict src1 = octx->src[1];                                                  \
        const struct htp_tensor * restrict dst  = octx->dst;                                                     \
        const int32_t  is_2D                    = octx->op_params[6] == 1;                                       \
        const uint32_t N                        = is_2D ? src1->ne[3] : src1->ne[2];                             \
        const uint32_t IC                       = is_2D ? src1->ne[2] : src1->ne[1];                             \
        const uint32_t IH                       = is_2D ? src1->ne[1] : 1;                                       \
        const uint32_t IW                       = src1->ne[0];                                                   \
        const uint32_t KH                       = is_2D ? octx->src[0]->ne[1] : 1;                               \
        const uint32_t KW                       = octx->src[0]->ne[0];                                           \
        const uint32_t OH                       = is_2D ? dst->ne[2] : 1;                                        \
        const uint32_t OW                       = dst->ne[1];                                                    \
        const uint32_t patch_stride             = IC * KH * KW;                                                  \
        const dma_addr_t src_data       = src1->data;                                                            \
        const dma_addr_t dst_data       = dst->data;                                                             \
        dma_queue *    dma_q            = octx->ctx->dma[ith];                                                   \
        uint8_t *      src_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;                \
        uint8_t *      dst_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;                \
        float *        srcb             = (float *) src_base;                                                    \
        DST_CTYPE *    dstb             = (DST_CTYPE *) dst_base;                                                \
        const uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;                            \
        const uint32_t per_thread               = ictx->pe_rows_per_thread;                                      \
        const uint32_t row_start                = ictx->pe_row_base + per_thread * ith;                          \
        const uint32_t row_end                  = MIN(row_start + per_thread, row_end_max);                      \
        if (row_start >= row_end)                                                                                \
            return;                                                                                              \
        for (uint32_t r = row_start; r < row_end; r++) {                                                         \
            const uint32_t in  = r / OH;                                                                         \
            const uint32_t ioh = r % OH;                                                                         \
            for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                            \
                int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;                                      \
                int     ok  = (iih >= 0 && iih < (int32_t) IH);                                                  \
                for (uint32_t iic = 0; iic < IC; iic++) {                                                        \
                    float *       vdst = srcb + (size_t) (iic * KH + ikh) * IW;                                  \
                    const dma_addr_t vsrc = ok                                                                   \
                        ? (src_data + (size_t) ((in * IC + iic) * IH + iih) * IW * sizeof(float))                \
                        : src_data;                                                                              \
                    dma_queue_push(dma_q, dma_make_data(vdst, vsrc),                                             \
                                   IW * sizeof(float), IW * sizeof(float), IW * sizeof(float), ok ? 1 : 0);      \
                }                                                                                                \
            }                                                                                                    \
            for (uint32_t i = 0; i < IC * KH; i++)                                                               \
                dma_queue_pop(dma_q);                                                                            \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);                                                \
            for (uint32_t iow = 0; iow < OW; iow++) {                                                            \
                DST_CTYPE * dst_patch = dstb + (uint64_t) iow * patch_stride;                                    \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                        \
                    int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;                                  \
                    for (uint32_t iic = 0; iic < IC; iic++) {                                                    \
                        DST_CTYPE * out_run = dst_patch + iic * (KH * KW) + ikh * KW;                            \
                        if (iih < 0 || iih >= (int32_t) IH) {                                                    \
                            SPLAT_FN(out_run, 0.0f, KW);                                                         \
                            continue;                                                                            \
                        }                                                                                        \
                        const float * src_run = srcb + ((uint64_t) (iic * KH + ikh)) * IW + (uint64_t) iow * KW; \
                        COPY_FN((uint8_t *) out_run, (const uint8_t *) src_run, KW);                             \
                    }                                                                                            \
                }                                                                                                \
            }                                                                                                    \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);                                                 \
            const dma_addr_t ddr_row = dst_data + (size_t) (in * OH + ioh) * OW * patch_stride * (DST_ELEM);     \
            dma_queue_push(dma_q, dma_make_data(ddr_row, dstb),                                                  \
                           OW * patch_stride * (DST_ELEM), OW * patch_stride * (DST_ELEM),                       \
                           OW * patch_stride * (DST_ELEM), 1);                                                   \
            dma_queue_flush(dma_q);                                                                              \
        }                                                                                                        \
    }

IM2COL_PATCHEMBED_DMA_BODY(im2col_patchembed_dma_thread,     __fp16, hvx_copy_f16_f32_uu, hvx_splat_f16_u, sizeof(__fp16), "pe-dma-f16")
IM2COL_PATCHEMBED_DMA_BODY(im2col_patchembed_dma_f32_thread, float,  hvx_copy_f32_uu,     hvx_splat_f32_u, sizeof(float),  "pe-dma-f32")

static bool im2col_use_patchembed_dma(const struct htp_ops_context * octx) {
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    const int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    const int32_t d0 = octx->op_params[4], d1 = octx->op_params[5];
    const int     is_2D = octx->op_params[6] == 1;
    if (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        return false;
    }
    const uint32_t KH = is_2D ? octx->src[0]->ne[1] : 1;
    const uint32_t KW = octx->src[0]->ne[0];
    if (s0 != (int32_t) KW) {
        return false;  // non-overlapping (width)
    }
    if (p0 != 0) {
        return false;  // no padding (width)
    }
    if (d0 != 1) {
        return false;  // no dilation (width)
    }
    if (is_2D) {
        if (s1 != (int32_t) KH) {
            return false;  // non-overlapping (height)
        }
        if (p1 != 0) {
            return false;  // no padding (height)
        }
        if (d1 != 1) {
            return false;  // no dilation (height)
        }
    }
    return true;
}

// Sizes the per-thread 2x(src,dst) VTCM ping-pong for the patch-embed DMA path.
// Returns false if it doesn't fit the VTCM budget (caller falls back).
static bool im2col_patchembed_dma_fits(struct htp_ops_context *    octx,
                                       struct htp_im2col_context * ictx,
                                       uint32_t                    n_threads) {
    const int32_t  is_2D = octx->op_params[6] == 1;
    const uint32_t IC = is_2D ? octx->src[1]->ne[2] : octx->src[1]->ne[1];
    const uint32_t IW = octx->src[1]->ne[0];
    const uint32_t KH = is_2D ? octx->src[0]->ne[1] : 1;
    const uint32_t KW = octx->src[0]->ne[0];
    const uint32_t OW           = octx->dst->ne[1];
    const uint32_t patch_stride = IC * KH * KW;

    ictx->pe_src_row_bytes  = hex_round_up(IC * KH * IW * sizeof(float), 256);
    const uint32_t dst_elem = (octx->dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    ictx->pe_dst_row_bytes  = hex_round_up(OW * patch_stride * dst_elem, 256);

    // 2 src + 2 dst buffers per thread (ping-pong), src region first then dst.
    struct htp_im2col_vtcm_layout L;
    htp_im2col_vtcm_layout_build(&L, ictx->pe_src_row_bytes, ictx->pe_dst_row_bytes, n_threads);
    if (L.total_bytes > octx->ctx->vtcm_size) {
        return false;
    }

    uint8_t * const base        = octx->ctx->vtcm_base;
    ictx->pe_vtcm_src           = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src);
    ictx->pe_vtcm_dst           = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    ictx->pe_src_size_per_thread = (uint32_t) L.src_bytes_per_thread;
    ictx->pe_dst_size_per_thread = (uint32_t) L.dst_bytes_per_thread;
    return true;
}

// Sizes a per-thread 2x(src,dst) VTCM ping-pong for the blocked general kernel.
// Stages Wb=(owb-1)*s0+(KW-1)*d0+1 source cols per (iic,ikh) row and owb patches
// of dst. Picks the largest owb that fits; returns false if even owb=1 does not.
static bool im2col_blocked_dma_fits(struct htp_ops_context *    octx,
                                    struct htp_im2col_context * ictx,
                                    uint32_t                    n_threads) {
    const int32_t  is_2D = octx->op_params[6] == 1;
    const int32_t  s0 = octx->op_params[0];
    const int32_t  d0 = octx->op_params[4];
    const uint32_t IC = is_2D ? octx->src[1]->ne[2] : octx->src[1]->ne[1];
    const uint32_t KH = is_2D ? octx->src[0]->ne[1] : 1;
    const uint32_t KW = octx->src[0]->ne[0];
    const uint32_t OW = octx->dst->ne[1];
    const uint32_t patch_stride = IC * KH * KW;
    const uint32_t dst_elem = (octx->dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    for (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        const uint32_t Wb = (owb - 1) * (uint32_t) s0 + (KW - 1) * (uint32_t) d0 + 1;
        const uint32_t src_row_bytes = hex_round_up(IC * KH * Wb * sizeof(float), 256);
        const uint32_t dst_row_bytes = hex_round_up(owb * patch_stride * dst_elem, 256);
        struct htp_im2col_vtcm_layout L;
        htp_im2col_vtcm_layout_build(&L, src_row_bytes, dst_row_bytes, n_threads);
        if (L.total_bytes <= octx->ctx->vtcm_size) {
            uint8_t * const base = octx->ctx->vtcm_base;
            ictx->pe_owb                 = owb;
            ictx->pe_wb                  = Wb;
            ictx->pe_src_row_bytes       = src_row_bytes;
            ictx->pe_dst_row_bytes       = dst_row_bytes;
            ictx->pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src);
            ictx->pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
            ictx->pe_src_size_per_thread = (uint32_t) L.src_bytes_per_thread;
            ictx->pe_dst_size_per_thread = (uint32_t) L.dst_bytes_per_thread;
            return true;
        }
        if (owb == 1) break;  // avoid unsigned underflow
    }
    return false;
}

int op_im2col(struct htp_ops_context * octx) {
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src1->type != HTP_TYPE_F32 || (dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_F32)) {
        FARF(ERROR, "im2col: only (F32 image -> F16/F32 columns) supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t  is_2D         = octx->op_params[6] == 1;
    const uint32_t N             = is_2D ? src1->ne[3] : src1->ne[2];
    const uint32_t OH            = is_2D ? dst->ne[2] : 1;
    const uint32_t OW            = dst->ne[1];
    const uint32_t total_patches = N * OH * OW;
    const uint32_t total_rows    = N * OH;

    uint32_t patch_base = 0;
    uint32_t npatches   = total_patches;
    if (octx->ctx->mdev.count > 1) {
        const uint32_t patch_size = dst->nb[1];
        const uint32_t patches_per_chunk =
            (patch_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(patch_size, HEX_L2_LINE_SIZE)) : 1;
        const struct htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(total_patches, htp_tensor_mdev_data_aligned(dst) ? patches_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        patch_base = range.start;
        npatches   = range.count;
    }

    uint32_t row_base = 0;
    uint32_t nrows    = total_rows;
    if (octx->ctx->mdev.count > 1) {
        const uint32_t row_size = dst->nb[2];
        const uint32_t rows_per_chunk =
            (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
        const struct htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_base = range.start;
        nrows    = range.count;
    }

    if (npatches == 0 && nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = MIN(octx->n_threads, MAX(npatches, 1));

    struct htp_im2col_context ictx = { 0 };
    ictx.octx                = octx;
    ictx.patch_base          = patch_base;
    ictx.npatches            = npatches;
    ictx.npatches_per_thread = (npatches + n_threads - 1) / n_threads;

    // Clean non-overlapping patch-embed -> DMA kernel (if it fits VTCM);
    // everything else (padding/dilation/stride edges) -> blocked-staging DMA
    // kernel; if neither fits VTCM -> pure-DDR kernel.
    if (nrows > 0) {
        const uint32_t pth = MIN(octx->n_threads, nrows);
        if (pth > 0) {
            ictx.pe_row_base = row_base;
            ictx.pe_nrows    = nrows;
            const bool exact = im2col_use_patchembed_dma(octx);
            if (exact && im2col_patchembed_dma_fits(octx, &ictx, pth)) {
                ictx.pe_rows_per_thread = (nrows + pth - 1) / pth;
                work_queue_run(octx->ctx->work_queue,
                    dst->type == HTP_TYPE_F16 ? im2col_patchembed_dma_thread
                                              : im2col_patchembed_dma_f32_thread, &ictx, pth);
                return HTP_STATUS_OK;
            }
            if (!exact && im2col_blocked_dma_fits(octx, &ictx, pth)) {
                ictx.pe_rows_per_thread = (nrows + pth - 1) / pth;
                work_queue_run(octx->ctx->work_queue,
                    dst->type == HTP_TYPE_F16 ? im2col_blocked_dma_thread
                                              : im2col_blocked_dma_f32_thread, &ictx, pth);
                return HTP_STATUS_OK;
            }
        }
    }
    // Fall through to pure-DDR.
    if (npatches == 0) {
        return HTP_STATUS_OK;
    }

    if (htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (dst->type == HTP_TYPE_F16) {
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_thread, &ictx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_f32_thread, &ictx, n_threads);
    }
    return HTP_STATUS_OK;
}
