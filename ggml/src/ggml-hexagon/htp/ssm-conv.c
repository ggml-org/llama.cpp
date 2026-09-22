#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_ps.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <qurt_thread.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "ssm-conv.h"

struct htp_ssm_conv_context {
    struct htp_ops_context *                  octx;
    const struct htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_per_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
};

// In-register 32x32 fp32 transpose using std 5-stage HVX vshuff butterfly.
static inline void hvx_transpose_32x32_f32(HVX_Vector m[32]) {
    HVX_Vector tmp[32];

    // Stage 0 (R = -4): pair (2i, 2i+1) for i = 0..15. m -> tmp.
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }

    // Stage 1 (R = -8): per block of 4, pair (b+0, b+2) and (b+1, b+3). tmp -> m.
    for (int b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }

    // Stage 2 (R = -16): per block of 8, pair (b+i, b+i+4) for i = 0..3. m -> tmp.
    for (int b = 0; b < 32; b += 8) {
        for (int i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 3 (R = -32): per block of 16, pair (b+i, b+i+8) for i = 0..7. tmp -> m.
    for (int b = 0; b < 32; b += 16) {
        for (int i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }

    // Stage 4 (R = -64): pair (i, i+16) for i = 0..15. m -> tmp -> m.
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }

    for (int i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
static inline void hvx_ssm_conv_unpack_to_T_4(const float * raw, float * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride) {
    for (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        HVX_Vector v0 = *(const HVX_Vector *)(raw + (cb + 0) * 4);
        HVX_Vector v1 = *(const HVX_Vector *)(raw + (cb + 8) * 4);
        HVX_Vector v2 = *(const HVX_Vector *)(raw + (cb + 16) * 4);
        HVX_Vector v3 = *(const HVX_Vector *)(raw + (cb + 24) * 4);

        HVX_VectorPair p01 = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_VectorPair p23 = Q6_W_vdeal_VVR(v3, v2, -4);

        HVX_VectorPair p_w02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_VectorPair p_w13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        *(HVX_Vector *)(T + 0 * d_inner_stride + cb) = Q6_V_lo_W(p_w02);
        *(HVX_Vector *)(T + 1 * d_inner_stride + cb) = Q6_V_lo_W(p_w13);
        *(HVX_Vector *)(T + 2 * d_inner_stride + cb) = Q6_V_hi_W(p_w02);
        *(HVX_Vector *)(T + 3 * d_inner_stride + cb) = Q6_V_hi_W(p_w13);
    }
}

// HVX transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
static inline void hvx_ssm_conv_unpack_to_T_gen(const float * raw, float * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
    for (uint32_t j = 0; j < d_conv; ++j) {
        mask_buf[j] = 0xFFFFFFFF;
    }
    const HVX_Vector mask = *(const HVX_Vector *) mask_buf;

    for (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
        const uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
        HVX_Vector sub[32];
        for (uint32_t r = 0; r < cb_n; ++r) {
            const float * ch_ptr = raw + (cb + r) * d_conv;
            sub[r] = Q6_V_vand_VV(*(const HVX_UVector *) ch_ptr, mask);
        }
        for (uint32_t r = cb_n; r < 32; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        for (uint32_t j = 0; j < d_conv; ++j) {
            *(HVX_Vector *)(T + j * d_inner_stride + cb) = sub[j];
        }
    }
}

static inline void hvx_ssm_conv_unpack_to_T(const float * raw, float * T, uint32_t d_inner_per_thread, uint32_t d_inner_stride, uint32_t d_conv) {
    if (d_conv == 4 && (d_inner_per_thread % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_to_T_4(raw, T, d_inner_per_thread, d_inner_stride);
    } else {
        hvx_ssm_conv_unpack_to_T_gen(raw, T, d_inner_per_thread, d_inner_stride, d_conv);
    }
}

// HVX 32x32 src0 transpose for prefill: src0 {tile_n, ncs} (VTCM) -> src0_T {ncs, d_inner_tile} (VTCM)
static inline void transpose_src0_block(const float * src0_block,
                                        uint32_t      ncs,
                                        uint32_t      cb_n,
                                        uint32_t      d_inner_tile,
                                        float *       src0_T_block_dst,
                                        uint32_t      cb) {
    const uint32_t T_TILE = VLEN_FP32;

    HVX_Vector __attribute__((aligned(VLEN))) sub[32];

    for (uint32_t t0 = 0; t0 < ncs; t0 += T_TILE) {
        const uint32_t t_n = MIN(T_TILE, ncs - t0);

        uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
        for (uint32_t k = 0; k < t_n; ++k) {
            mask_buf[k] = 0xFFFFFFFF;
        }
        const HVX_Vector mask = *(const HVX_Vector *) mask_buf;

        for (uint32_t r = 0; r < cb_n; ++r) {
            const float * src_row = src0_block + r * ncs + t0;
            sub[r] = (t_n == T_TILE) ? *(const HVX_UVector *) src_row : Q6_V_vand_VV(*(const HVX_UVector *) src_row, mask);
        }
        for (uint32_t r = cb_n; r < T_TILE; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        for (uint32_t r = 0; r < t_n; ++r) {
            float * dst = src0_T_block_dst + (t0 + r) * d_inner_tile + cb;
            if (cb_n == T_TILE) {
                *(HVX_UVector *) dst = sub[r];
            } else {
                hvx_vec_store_u(dst, cb_n * sizeof(float), sub[r]);
            }
        }
    }
}

// Single-row decode worker (n_t == 1)
static void ssm_conv_thread_f32_decode(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const uint32_t d_conv  = kparams->d_conv;
    const uint32_t d_inner = kparams->d_inner;
    const uint32_t n_s     = kparams->n_s;

    const uint32_t dr  = scctx->nrows_per_thread;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_thread = ir1 - ir0;
    const uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);

    const size_t src0_stride_seq_bytes = src0->nb[2];
    const size_t dst_stride_seq_bytes  = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    const size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * sizeof(float);
    const size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    float * src0_raw = (float *) src0_spad_base;
    float * src0_T   = (float *) (src0_spad_base + weight_raw_size);

    float * dst_spad = (float *) dst_spad_base;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    // 1. Fetch weights src1 from DDR into VTCM via DMA (DMA64-safe)
    const dma_addr_t src1_ddr = src1->data + ir0 * d_conv * sizeof(float);
    dma_queue_push(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);
    dma_queue_pop(dma_q);

    // 2. Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const size_t input_bytes  = (size_t) d_inner_per_thread * d_conv * sizeof(float);
    const size_t output_bytes = (size_t) d_inner_per_thread * sizeof(float);

    // 3. Process each sequence
    for (uint32_t s = 0; s < n_s; ++s) {
        const dma_addr_t src0_ddr = src0->data + s * src0_stride_seq_bytes + ir0 * d_conv * sizeof(float);
        dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_raw, src0_ddr), input_bytes, input_bytes, input_bytes, 1);
        dma_queue_pop(dma_q);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) s);
        hvx_ssm_conv_unpack_to_T(src0_raw, src0_T, d_inner_per_thread, d_inner_stride, d_conv);

        for (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
            const uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
            HVX_Vector acc = hvx_vec_splat_f32(0.0f);
            for (uint32_t j = 0; j < d_conv; ++j) {
                HVX_Vector x = *(const HVX_Vector *)(src0_T + j * d_inner_stride + cb);
                HVX_Vector w = *(const HVX_Vector *)(src1_T + j * d_inner_stride + cb);
                acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
            }
            HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
            if (cb_n == VLEN_FP32) {
                *(HVX_Vector *)(dst_spad + cb) = y;
            } else {
                hvx_vec_store_u(dst_spad + cb, cb_n * sizeof(float), y);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) s);

        const dma_addr_t dst_ddr = dst->data + s * dst_stride_seq_bytes + ir0 * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(dst_ddr, (uint8_t *) dst_spad), output_bytes, output_bytes, output_bytes, 1);
        dma_queue_pop(dma_q);
    }

    FARF(HIGH, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
static void ssm_conv_thread_f32_prefill(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const uint32_t d_conv  = kparams->d_conv;
    const uint32_t d_inner = kparams->d_inner;
    const uint32_t n_t     = kparams->n_t;
    const uint32_t n_s     = kparams->n_s;
    const uint32_t ncs     = src0->ne[0];

    const uint32_t dr  = scctx->nrows_per_thread;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_thread = ir1 - ir0;
    const uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
    const uint32_t d_inner_tile       = scctx->d_inner_tile;

    const size_t src0_stride_inner_bytes = src0->nb[1];
    const size_t src0_stride_seq_bytes   = src0->nb[2];
    const size_t dst_stride_token_bytes  = dst->nb[1];
    const size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    const size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * sizeof(float);
    const size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    const size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * ncs * sizeof(float), 128);
    float * src0_tile_raw = (float *) src0_spad_base;
    float * src0_T        = (float *) (src0_spad_base + src0_tile_raw_bytes);

    float * dst_tile = (float *) dst_spad_base;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    // 1. Fetch weights src1 from DDR into VTCM via DMA (DMA64-safe)
    const dma_addr_t src1_ddr = src1->data + ir0 * d_conv * sizeof(float);
    dma_queue_push(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);
    dma_queue_pop(dma_q);

    // 2. Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const uint32_t C_TILE = VLEN_FP32;

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        for (uint32_t tile_off = 0; tile_off < d_inner_per_thread; tile_off += d_inner_tile) {
            const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);

            // Fetch src0 chunk from DDR to VTCM via 2D DMA
            const dma_addr_t src0_tile_ddr = src0->data +
                i3 * src0_stride_seq_bytes +
                (ir0 + tile_off) * src0_stride_inner_bytes;
            const size_t row_bytes = ncs * sizeof(float);

            dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_tile_raw, src0_tile_ddr),
                           row_bytes, src0_stride_inner_bytes, row_bytes, tile_n);
            dma_queue_pop(dma_q);

            // Transpose src0 chunk in VTCM into {d_inner_tile, ncs} layout
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
                transpose_src0_block(src0_tile_raw + cb * ncs, ncs, cb_n, d_inner_tile, src0_T, cb);
            }

            // Compute convolution
            for (uint32_t t = 0; t < n_t; ++t) {
                for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                    const uint32_t cb_n = MIN(C_TILE, tile_n - cb);

                    HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                    for (uint32_t j = 0; j < d_conv; ++j) {
                        HVX_Vector x = *(const HVX_Vector *) (src0_T + (t + j) * d_inner_tile + cb);
                        HVX_Vector w = *(const HVX_Vector *) (src1_T + j * d_inner_stride + tile_off + cb);
                        acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }

                    HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
                    float * dst_tile_ptr = dst_tile + t * tile_n + cb;
                    if (cb_n == C_TILE) {
                        *(HVX_Vector *) dst_tile_ptr = y;
                    } else {
                        hvx_vec_store_u(dst_tile_ptr, cb_n * sizeof(float), y);
                    }
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);

            // Writeback dst_tile from VTCM to DDR via 2D DMA
            const dma_addr_t dst_tile_ddr = dst->data +
                i3 * dst_stride_seq_bytes +
                (ir0 + tile_off) * sizeof(float);
            const size_t dst_row_bytes = tile_n * sizeof(float);

            dma_queue_push(dma_q, dma_make_data(dst_tile_ddr, (uint8_t *) dst_tile),
                           dst_stride_token_bytes, dst_row_bytes, dst_row_bytes, n_t);
            dma_queue_pop(dma_q);
        }
    }

    FARF(HIGH, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

int op_ssm_conv_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_ssm_conv_kernel_params * kparams = (const struct htp_ssm_conv_kernel_params *) octx->kernel_params;


    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->d_inner;

    if (octx->ctx->mdev.count > 1) {
        const uint32_t elems_per_chunk = VLEN_FP32;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->d_inner,
            htp_tensor_mdev_data_aligned(dst) ? elems_per_chunk : 0,
            octx->ctx->mdev.idx,
            octx->ctx->mdev.count,
            &octx->ctx->mdev.count_div
        );
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t n_threads = octx->n_threads;

    octx->src0_spad.size_per_thread = kparams->vtcm_src0_size_per_thread;
    octx->src1_spad.size_per_thread = kparams->vtcm_src1_size_per_thread;
    octx->dst_spad.size_per_thread  = kparams->vtcm_dst_size_per_thread;

    octx->src0_spad.size = kparams->vtcm_src0_size;
    octx->src1_spad.size = kparams->vtcm_src1_size;
    octx->dst_spad.size  = kparams->vtcm_dst_size;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size;
    octx->src0_spad.src  = NULL;
    octx->src1_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    const uint32_t raw_rpt            = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    const uint32_t d_inner_per_thread = hex_round_up(raw_rpt, VLEN_FP32);

    struct htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = d_inner_per_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .row_start        = row_start,
        .nrows            = nrows,
    };

    FARF(HIGH, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill");

    if (kparams->n_t == 1) {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_decode, &scctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_prefill, &scctx, n_threads);
    }

    return HTP_STATUS_OK;
}

int op_ssm_conv(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;

    switch (dst->type) {
        case HTP_TYPE_F32:
            return op_ssm_conv_f32(octx);
        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
