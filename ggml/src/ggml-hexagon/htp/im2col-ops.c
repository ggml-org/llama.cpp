#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "hex-dma.h"

struct htp_im2col_context {
    struct htp_ops_context * octx;
    uint32_t                 npatches_per_thread;  // patches = N*OH*OW (pure-DDR kernel)

    uint32_t pe_rows_per_thread;                   // N*OH rows per worker
    uint32_t pe_src_row_bytes;                     // one output row's source: IC*KH*IW*4, rounded 256
    uint32_t pe_dst_row_bytes;                     // one output row's dst: OW*patch_stride*2, rounded 256
};

#define IM2COL_PATCHEMBED_BODY(FNAME, DST_CTYPE, COPY_FN, DST_ELEM, TAG)                                             \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                             \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                                \
        struct htp_ops_context *    octx        = ictx->octx;                                                        \
        const struct htp_tensor * restrict src1 = octx->src[1];                                                      \
        const struct htp_tensor * restrict dst  = octx->dst;                                                         \
        const int32_t  s0                       = octx->op_params[0];                                                \
        const int32_t  s1                       = octx->op_params[1];                                                \
        const int32_t  p0                       = octx->op_params[2];                                                \
        const int32_t  p1                       = octx->op_params[3];                                                \
        const int32_t  d0                       = octx->op_params[4];                                                \
        const int32_t  d1                       = octx->op_params[5];                                                \
        const uint32_t N                        = src1->ne[3];                                                       \
        const uint32_t IC                       = src1->ne[2];                                                       \
        const uint32_t IH                       = src1->ne[1];                                                       \
        const uint32_t IW                       = src1->ne[0];                                                       \
        const uint32_t KH                       = octx->src[0]->ne[1];                                               \
        const uint32_t KW                       = octx->src[0]->ne[0];                                               \
        const uint32_t OH                       = dst->ne[2];                                                        \
        const uint32_t OW                       = dst->ne[1];                                                        \
        const uint32_t patch_stride             = IC * KH * KW;                                                      \
        const float * restrict src_data         = (const float *) src1->data;                                        \
        DST_CTYPE * restrict dst_data           = (DST_CTYPE *) dst->data;                                           \
        const uint32_t npatches                 = N * OH * OW;                                                       \
        const uint32_t patch_start              = ictx->npatches_per_thread * ith;                                   \
        const uint32_t patch_end                = MIN(patch_start + ictx->npatches_per_thread, npatches);            \
        if (patch_start >= patch_end) {                                                                              \
            return;                                                                                                  \
        }                                                                                                            \
        for (uint32_t p = patch_start; p < patch_end; p++) {                                                         \
            const uint32_t iow             = p % OW;                                                                 \
            const uint32_t ioh             = (p / OW) % OH;                                                          \
            const uint32_t in              = p / (OW * OH);                                                          \
            DST_CTYPE * restrict dst_patch = dst_data + (uint64_t) p * patch_stride;                                 \
            for (uint32_t iic = 0; iic < IC; iic++) {                                                                \
                const float * restrict src_plane = src_data + ((uint64_t) in * IC + iic) * IH * IW;                  \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                            \
                    const int32_t iih            = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                     \
                    DST_CTYPE * restrict out_run = dst_patch + iic * (KH * KW) + ikh * KW;                           \
                    if (iih < 0 || iih >= (int32_t) IH) {                                                            \
                        memset(out_run, 0, KW * (DST_ELEM));                                                         \
                        continue;                                                                                    \
                    }                                                                                                \
                    const int32_t iiw0             = (int32_t) iow * s0 - p0;                                        \
                    const float * restrict src_run = src_plane + (uint64_t) iih * IW + iiw0;                         \
                    if (d0 == 1 && iiw0 >= 0 && iiw0 + (int32_t) KW <= (int32_t) IW) {                               \
                        COPY_FN((uint8_t *) out_run, (const uint8_t *) src_run, KW);                                 \
                        continue;                                                                                    \
                    }                                                                                                \
                    for (uint32_t ikw = 0; ikw < KW; ikw++) {                                                        \
                        const int32_t iiw = (int32_t) iow * s0 + (int32_t) ikw * d0 - p0;                            \
                        out_run[ikw]      = (iiw < 0 || iiw >= (int32_t) IW) ?                                       \
                                                (DST_CTYPE) 0.0f :                                                   \
                                                (DST_CTYPE) src_plane[(uint64_t) iih * IW + iiw];                    \
                    }                                                                                                \
                }                                                                                                    \
            }                                                                                                        \
        }                                                                                                            \
    }

IM2COL_PATCHEMBED_BODY(im2col_patchembed_thread, __fp16, hvx_copy_f16_f32_uu, sizeof(__fp16), "f32-f16")
IM2COL_PATCHEMBED_BODY(im2col_patchembed_f32_thread, float, hvx_copy_f32_uu, sizeof(float), "f32-f32")

#define IM2COL_PATCHEMBED_DMA_BODY(FNAME, DST_CTYPE, COPY_FN, DST_ELEM, TAG)                                         \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                             \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                                \
        struct htp_ops_context *    octx        = ictx->octx;                                                        \
        const struct htp_tensor * restrict src1 = octx->src[1];                                                      \
        const struct htp_tensor * restrict dst  = octx->dst;                                                         \
        const uint32_t N = src1->ne[3], IC = src1->ne[2], IH = src1->ne[1], IW = src1->ne[0];                        \
        const uint32_t KH = octx->src[0]->ne[1], KW = octx->src[0]->ne[0];                                           \
        const uint32_t OH = dst->ne[2], OW = dst->ne[1];                                                             \
        const uint32_t patch_stride     = IC * KH * KW;                                                              \
        const float * restrict src_data = (const float *) src1->data;                                                \
        DST_CTYPE * restrict dst_data   = (DST_CTYPE *) dst->data;                                                   \
        dma_queue *    dmaq             = octx->ctx->dma[ith];                                                       \
        uint8_t *      src_base         = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;              \
        uint8_t *      dst_base         = octx->dst_spad.data + ith * octx->dst_spad.size_per_thread;                \
        float *        srcb             = (float *) src_base;                                                        \
        DST_CTYPE *    dstb             = (DST_CTYPE *) dst_base;                                                    \
        const uint32_t nrows            = N * OH;                                                                    \
        const uint32_t per_thread       = ictx->pe_rows_per_thread;                                                  \
        const uint32_t row_start        = per_thread * ith;                                                          \
        const uint32_t row_end          = MIN(row_start + per_thread, nrows);                                        \
        if (row_start >= row_end)                                                                                    \
            return;                                                                                                  \
        for (uint32_t r = row_start; r < row_end; r++) {                                                             \
            const uint32_t in  = r / OH;                                                                             \
            const uint32_t ioh = r % OH;                                                                             \
            for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                                \
                int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;                                          \
                int     ok  = (iih >= 0 && iih < (int32_t) IH);                                                      \
                for (uint32_t iic = 0; iic < IC; iic++) {                                                            \
                    float *       vdst = srcb + ((uint64_t) (iic * KH + ikh)) * IW;                                  \
                    const float * _vsrc =                                                                            \
                        ok ? (src_data + ((uint64_t) (in * IC + iic) * IH + iih) * IW) : (const float *) vdst;       \
                    dma_queue_push_ddr_to_vtcm(                                                                      \
                        dmaq, dma_make_ptr((uint8_t *) vdst, ok ? (const uint8_t *) _vsrc : (const uint8_t *) vdst), \
                        IW * sizeof(float), IW * sizeof(float), ok ? 1 : 0);                                         \
                }                                                                                                    \
            }                                                                                                        \
            for (uint32_t i = 0; i < IC * KH; i++)                                                                   \
                dma_queue_pop(dmaq);                                                                                 \
            for (uint32_t iow = 0; iow < OW; iow++) {                                                                \
                DST_CTYPE * dst_patch = dstb + (uint64_t) iow * patch_stride;                                        \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                            \
                    int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;                                      \
                    for (uint32_t iic = 0; iic < IC; iic++) {                                                        \
                        DST_CTYPE * out_run = dst_patch + iic * (KH * KW) + ikh * KW;                                \
                        if (iih < 0 || iih >= (int32_t) IH) {                                                        \
                            memset(out_run, 0, KW * (DST_ELEM));                                                     \
                            continue;                                                                                \
                        }                                                                                            \
                        const float * src_run = srcb + ((uint64_t) (iic * KH + ikh)) * IW + (uint64_t) iow * KW;     \
                        COPY_FN((uint8_t *) out_run, (const uint8_t *) src_run, KW);                                 \
                    }                                                                                                \
                }                                                                                                    \
            }                                                                                                        \
            DST_CTYPE * ddr_row = dst_data + ((uint64_t) (in * OH + ioh) * OW) * patch_stride;                       \
            dma_queue_push_vtcm_to_ddr(dmaq, dma_make_ptr((uint8_t *) ddr_row, (uint8_t *) dstb),                    \
                                       OW * patch_stride * (DST_ELEM), OW * patch_stride * (DST_ELEM), 1);           \
            dma_queue_flush(dmaq);                                                                                   \
        }                                                                                                            \
    }

IM2COL_PATCHEMBED_DMA_BODY(im2col_patchembed_dma_thread,     __fp16, hvx_copy_f16_f32_uu, sizeof(__fp16), "pe-dma-f16")
IM2COL_PATCHEMBED_DMA_BODY(im2col_patchembed_dma_f32_thread, float,  hvx_copy_f32_uu,     sizeof(float),  "pe-dma-f32")

static bool im2col_use_patchembed_dma(const struct htp_ops_context * octx) {
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    const int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    const int32_t d0 = octx->op_params[4], d1 = octx->op_params[5];
    const int     is_2D = octx->op_params[6] == 1;
    if (!is_2D) {
        return false;
    }
    if (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        return false;
    }
    const uint32_t KH = octx->src[0]->ne[1], KW = octx->src[0]->ne[0];
    if (s0 != (int32_t) KW || s1 != (int32_t) KH) {
        return false;  // non-overlapping
    }
    if (p0 != 0 || p1 != 0) {
        return false;  // no padding
    }
    if (d0 != 1 || d1 != 1) {
        return false;  // no dilation
    }
    return true;
}

// Sizes the per-thread 2x(src,dst) VTCM ping-pong for the patch-embed DMA path.
// Returns false if it doesn't fit the VTCM budget (caller falls back).
static bool im2col_patchembed_dma_fits(struct htp_ops_context *    octx,
                                       struct htp_im2col_context * ictx,
                                       uint32_t                    n_threads) {
    const uint32_t IC = octx->src[1]->ne[2], IW = octx->src[1]->ne[0];
    const uint32_t KH = octx->src[0]->ne[1], KW = octx->src[0]->ne[0];
    const uint32_t OW           = octx->dst->ne[1];
    const uint32_t patch_stride = IC * KH * KW;

    ictx->pe_src_row_bytes  = hex_round_up(IC * KH * IW * sizeof(float), 256);
    const uint32_t dst_elem = (octx->dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    ictx->pe_dst_row_bytes  = hex_round_up(OW * patch_stride * dst_elem, 256);

    // 2 src + 2 dst buffers per thread (ping-pong).
    const uint64_t per_thread = 2ull * ictx->pe_src_row_bytes + 2ull * ictx->pe_dst_row_bytes;
    const uint64_t total      = per_thread * n_threads;
    if (total > octx->ctx->vtcm_size) {
        return false;
    }

    // src buffers first, then dst buffers, in vtcm.
    octx->src1_spad.size_per_thread = 2 * ictx->pe_src_row_bytes;
    octx->src1_spad.size            = octx->src1_spad.size_per_thread * n_threads;
    octx->src1_spad.data            = octx->ctx->vtcm_base;
    octx->dst_spad.size_per_thread  = 2 * ictx->pe_dst_row_bytes;
    octx->dst_spad.size             = octx->dst_spad.size_per_thread * n_threads;
    octx->dst_spad.data             = octx->src1_spad.data + octx->src1_spad.size;
    return true;
}

int op_im2col(struct htp_ops_context * octx) {
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src1->type != HTP_TYPE_F32 || (dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_F32)) {
        FARF(ERROR, "im2col: only (F32 image -> F16/F32 columns) supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t N         = src1->ne[3];
    const uint32_t OH        = dst->ne[2];
    const uint32_t OW        = dst->ne[1];
    const uint32_t npatches  = N * OH * OW;
    const uint32_t n_threads = MIN(octx->n_threads, npatches);

    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) || n_threads == 0) {
        return HTP_STATUS_OK;
    }

    struct htp_im2col_context ictx = { 0 };
    ictx.octx                      = octx;
    ictx.npatches_per_thread       = (npatches + n_threads - 1) / n_threads;

    // Clean non-overlapping patch-embed -> DMA kernel (if it fits VTCM);
    // everything else (padding/dilation/stride edges) -> pure-DDR kernel.
    if (im2col_use_patchembed_dma(octx)) {
        const uint32_t nrows = N * OH;
        const uint32_t pth   = MIN(octx->n_threads, nrows);
        if (pth > 0 && im2col_patchembed_dma_fits(octx, &ictx, pth)) {
            ictx.pe_rows_per_thread = (nrows + pth - 1) / pth;
            if (dst->type == HTP_TYPE_F16) {
                worker_pool_run_func(octx->ctx->worker_pool, im2col_patchembed_dma_thread, &ictx, pth);
            } else {
                worker_pool_run_func(octx->ctx->worker_pool, im2col_patchembed_dma_f32_thread, &ictx, pth);
            }
            return HTP_STATUS_OK;
        }
        // else: doesn't fit -> fall through to the pure-DDR kernel below.
    }

    if (dst->type == HTP_TYPE_F16) {
        worker_pool_run_func(octx->ctx->worker_pool, im2col_patchembed_thread, &ictx, n_threads);
    } else {
        worker_pool_run_func(octx->ctx->worker_pool, im2col_patchembed_f32_thread, &ictx, n_threads);
    }
    return HTP_STATUS_OK;
}
