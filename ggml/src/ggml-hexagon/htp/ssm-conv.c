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
#include "hex-dma.h"
#include "hex-profile.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"

#define htp_ssm_conv_tensors_preamble                           \
    const struct htp_tensor * restrict src0 = octx->src[0];     \
    const struct htp_tensor * restrict src1 = octx->src[1];     \
    const struct htp_tensor * restrict dst  = octx->dst;        \
    struct htp_spad * restrict src0_spad    = &octx->src0_spad; \
    struct htp_spad * restrict src1_spad    = &octx->src1_spad; \
    struct htp_spad * restrict dst_spad     = &octx->dst_spad;  \
                                                                \
    const uint32_t ne00 = src0->ne[0];                          \
    const uint32_t ne01 = src0->ne[1];                          \
    const uint32_t ne02 = src0->ne[2];                          \
    const uint32_t ne03 = src0->ne[3];                          \
                                                                \
    const uint32_t ne10 = src1->ne[0];                          \
    const uint32_t ne11 = src1->ne[1];                          \
    const uint32_t ne12 = src1->ne[2];                          \
    const uint32_t ne13 = src1->ne[3];                          \
                                                                \
    const uint32_t ne0 = dst->ne[0];                            \
    const uint32_t ne1 = dst->ne[1];                            \
    const uint32_t ne2 = dst->ne[2];                            \
    const uint32_t ne3 = dst->ne[3];                            \
                                                                \
    const uint32_t nb00 = src0->nb[0];                          \
    const uint32_t nb01 = src0->nb[1];                          \
    const uint32_t nb02 = src0->nb[2];                          \
    const uint32_t nb03 = src0->nb[3];                          \
                                                                \
    const uint32_t nb10 = src1->nb[0];                          \
    const uint32_t nb11 = src1->nb[1];                          \
    const uint32_t nb12 = src1->nb[2];                          \
    const uint32_t nb13 = src1->nb[3];                          \
                                                                \
    const uint32_t nb0 = dst->nb[0];                            \
    const uint32_t nb1 = dst->nb[1];                            \
    const uint32_t nb2 = dst->nb[2];                            \
    const uint32_t nb3 = dst->nb[3];

struct htp_ssm_conv_context {
    struct htp_ops_context * octx;
    uint32_t nrows_per_thread;
    uint32_t d_inner_tile;
    uint64_t t_start;
    uint32_t row_start;
    uint32_t nrows;
    struct htp_ops_context *                  octx;
    const struct htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_per_thread;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
    // CONCAT+SSM_CONV(+SILU) fusion (try_fuse_concat_ssm_conv, ggml-hexagon.cpp): when
    // fused is set, octx->src[0] is only the small conv_states tail
    // ([n_state_rows, d_inner, n_s]) and octx->src[2] is qkv_mixed_orig (token-major,
    // [d_inner, n_qkv_tokens, n_s]) -- CONCAT's own DDR materialization of conv_input is
    // skipped entirely; the two disjoint sources are read directly in the prefill worker.
    bool                                       fused;
    uint32_t                                   n_state_rows; // conv_kernel_size - 1, i.e. src0->ne[0] in fused mode
    bool                                       apply_silu;   // fuse silu(x) = x*sigmoid(x) into the output-store (prefill-only)
};

#define htp_ssm_conv_preamble                                                   \
    struct htp_ssm_conv_context * scctx = (struct htp_ssm_conv_context *) data; \
    struct htp_ops_context *      octx  = scctx->octx;                          \
    htp_ssm_conv_tensors_preamble;                                              \
    dma_queue * dma_queue = octx->ctx->dma[ith];

// Scalar FP32 SSM_CONV implementation
static void ssm_conv_thread_f32_f32(unsigned int nth, unsigned int ith, void *data) {
    htp_ssm_conv_preamble;

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];
    const uint32_t n_s     = dst->ne[2];

    const uint32_t src0_stride_inner = src0->nb[1] / sizeof(float); // stride for inner dimension
    const uint32_t src0_stride_seq   = src0->nb[2] / sizeof(float); // stride for sequence dimension
    const uint32_t src1_stride_inner = src1->nb[1] / sizeof(float); // stride for inner dimension
    const uint32_t dst_stride_token  = dst->nb[1]  / sizeof(float); // stride for token dimension
    const uint32_t dst_stride_seq    = dst->nb[2]  / sizeof(float); // stride for sequence dimension

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float *       dst_data  = (float *) dst->data;

    // Calculate row range for this thread
    const uint32_t d_inner_per_thread = scctx->nrows_per_thread;
    const uint32_t d_inner_start = scctx->row_start + d_inner_per_thread * ith;
    const uint32_t d_inner_end   = MIN(d_inner_start + d_inner_per_thread, scctx->row_start + scctx->nrows);

    // No work for this thread
    if (d_inner_start >= d_inner_end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) d_inner_start);

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        for (uint32_t i2 = 0; i2 < n_t; ++i2) {
            for (uint32_t i1 = d_inner_start; i1 < d_inner_end; ++i1) {
                float sumf = 0.0f;

                for (uint32_t i0 = 0; i0 < d_conv; ++i0) {
                    const uint32_t src0_idx = (i2 + i0) + i1 * src0_stride_inner + i3 * src0_stride_seq;
                    const uint32_t src1_idx = i0 + i1 * src1_stride_inner;

                    sumf += src0_data[src0_idx] * src1_data[src1_idx];
                }

                const uint32_t dst_idx = i1 + i2 * dst_stride_token + i3 * dst_stride_seq;
                dst_data[dst_idx] = sumf;
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) d_inner_end);

    FARF(HIGH, "ssm-conv-f32 %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], d_inner_start, d_inner_end,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}


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

// HVX FP32 SSM_CONV implementation - channel-vectorized HVX kernel with src0/src1
// transposed into VTCM.
//
// VTCM layouts (per thread):
//   src1_T : {d_inner_stride, d_conv}       - staged once per launch (small).
//   src0_T : {d_inner_tile,     ncs}        - staged per d_inner-tile.
//
// d_inner_tile is chosen so that per-thread VTCM stays under the budget.
// Each thread iterates ceil(d_inner_per_thread d_inner_tile) tiles serially.
#define HTP_SSM_CONV_VTCM_BUDGET (1u << 20) // 1 MiB per thread

// Scalar transpose: src1 {d_conv, d_inner} (DDR) -> {d_inner_stride, d_conv} (VTCM)
static inline void transpose_src1(const float * src1_data,
                                  uint32_t      src1_stride_inner,
                                  uint32_t      i1_off,
                                  uint32_t      d_inner_per_thread,
                                  uint32_t      d_inner_stride,
                                  uint32_t      d_conv,
                                  float *       src1_T) {
    for (uint32_t i = 0; i < d_inner_per_thread; ++i) {
        const float * src_row = src1_data + (i1_off + i) * src1_stride_inner;
        for (uint32_t j = 0; j < d_conv; ++j) {
            src1_T[j * d_inner_stride + i] = src_row[j];
        }
    }
}

// HVX 32x32 src0 transpose: src0 {ncs, d_inner} (DDR) -> src0_T {d_inner_tile, ncs} (VTCM)
static inline void transpose_src0_block(const float * src0_block,
                                        uint32_t      ncs,
                                        uint32_t      cb_n,
                                        uint32_t      d_inner_tile,
                                        float *       src0_T_block_dst,
                                        uint32_t      cb /* dst column offset */) {
    const uint32_t T_TILE = VLEN_FP32;

    HVX_Vector __attribute__((aligned(VLEN))) sub[32];

    for (uint32_t t0 = 0; t0 < ncs; t0 += T_TILE) {
        const uint32_t t_n = MIN(T_TILE, ncs - t0);

        // Load 32 rows (channels) of T_TILE samples; pad missing channels with zeros.
        for (uint32_t r = 0; r < cb_n; ++r) {
            const float * src_row = src0_block + r * ncs + t0;
            if (t_n == T_TILE) {
                sub[r] = *(const HVX_UVector *) src_row;
            } else {
                HVX_Vector v = hvx_vec_splat_f32(0.0f);
                hvx_vec_store_u(&v, t_n * sizeof(float), hvx_vec_splat_f32(0.0f));

                float __attribute__((aligned(VLEN))) tmp[VLEN_FP32] = { 0 };
                for (uint32_t k = 0; k < t_n; ++k) tmp[k] = src_row[k];
                v = *(const HVX_Vector *) tmp;
                sub[r] = v;
            }
        }
        for (uint32_t r = cb_n; r < T_TILE; ++r) {
            sub[r] = hvx_vec_splat_f32(0.0f);
        }

        hvx_transpose_32x32_f32(sub);

        // Store transposed sub-tile to src0_T at offsets (t0 + j) * d_inner_tile + cb.
        // Only write the valid t_n rows of the transposed result.
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

static void ssm_conv_thread_f32_f32_hvx(unsigned int nth, unsigned int ith, void *data) {
    htp_ssm_conv_preamble;

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];
    const uint32_t n_s     = dst->ne[2];
    const uint32_t ncs     = src0->ne[0];
// CONCAT+SSM_CONV(+SILU) fusion (try_fuse_concat_ssm_conv, ggml-hexagon.cpp): builds the
// same {d_inner_tile, ncs} VTCM destination layout as the unfused transpose_src0_block
// above, but for the qkv_mixed_orig portion (time >= n_state_rows) it is a straight
// strided DDR->VTCM DMA copy -- no transpose at all, since qkv_mixed_orig is already
// token-major/channel-contiguous, exactly the layout src0_T's channel axis wants. Only the
// tiny conv_states tail (n_state_rows rows, e.g. 3) needs an actual (scalar) gather-
// transpose, done by the sibling function below.
//
// DMA-pipelined: destinations are disjoint per column block, so this pushes the transfer
// to the DMA queue and returns immediately (caller must dma_queue_pop() once per push,
// matching this file's existing push/pop discipline elsewhere) instead of blocking on a
// synchronous unaligned-HVX-load-per-row copy -- letting the DMA engine's own transfer run
// concurrently with the state-rows scalar gather below.
static inline void transpose_src0_block_fused_dma_push(dma_queue *   dma_q,
                                                        const float * qkv_block,
                                                        uint32_t      qkv_stride_token,
                                                        uint32_t      n_state_rows,
                                                        uint32_t      ncs,
                                                        uint32_t      cb_n,
                                                        uint32_t      d_inner_tile,
                                                        float *       src0_T_block_dst,
                                                        uint32_t      cb) {
    const uint32_t n_qkv_rows = ncs - n_state_rows;
    if (n_qkv_rows == 0) {
        return;
    }
    float * dst_row0 = src0_T_block_dst + n_state_rows * d_inner_tile + cb;
    dma_queue_push(dma_q, dma_make_data(dst_row0, qkv_block),
                   d_inner_tile * sizeof(float), qkv_stride_token * sizeof(float),
                   cb_n * sizeof(float), n_qkv_rows);
}

// Scalar gather-transpose for the boundary (state-rows) part only -- paired with
// transpose_src0_block_fused_dma_push above, which handles the bulk qkv rows via DMA.
// n_state_rows is always < VLEN_FP32 (a handful of taps vs a 32-lane tile), so this only
// ever touches the first output tile and is at most a few dozen scalar stores total.
static inline void transpose_src0_state_rows_fused(const float * state_block,
                                                    uint32_t      state_stride_inner,
                                                    uint32_t      n_state_rows,
                                                    uint32_t      cb_n,
                                                    uint32_t      d_inner_tile,
                                                    float *       src0_T_block_dst,
                                                    uint32_t      cb) {
    for (uint32_t tt = 0; tt < n_state_rows; ++tt) {
        float * dst_row = src0_T_block_dst + tt * d_inner_tile + cb;
        for (uint32_t r = 0; r < cb_n; ++r) {
            dst_row[r] = state_block[r * state_stride_inner + tt];
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
static uint32_t ssm_conv_prefill_enqueue_input(
        dma_queue * dma_q,
        bool fused,
        const struct htp_tensor * src0,
        const struct htp_tensor * src2,
        uint8_t * src0_spad_base,
        size_t src0_slot_size,
        size_t src0_tile_raw_bytes,
        uint32_t d_inner_tile,
        uint32_t tile_idx,
        uint32_t d_inner_per_thread,
        uint32_t i3,
        uint32_t ncs,
        uint32_t n_state_rows,
        size_t src0_stride_inner_bytes,
        size_t src0_stride_seq_bytes,
        size_t qkv_stride_token_bytes,
        size_t qkv_stride_seq_bytes) {
    const uint32_t tile_off = tile_idx * d_inner_tile;
    const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);
    uint8_t * slot_base = src0_spad_base + (size_t) (fused ? (tile_idx & 1) : 0) * src0_slot_size;
    float * src0_tile_raw = (float *) slot_base;
    float * src0_T = (float *) (slot_base + src0_tile_raw_bytes);
    uint32_t count = 0;

    if (fused) {
        const uint32_t n_qkv_rows = ncs - n_state_rows;
        if (n_state_rows > 0) {
            const dma_addr_t state_tile_ddr = src0->data +
                i3 * src0_stride_seq_bytes + tile_off * src0_stride_inner_bytes;
            const size_t state_row_bytes = n_state_rows * sizeof(float);

            dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_tile_raw, state_tile_ddr),
                           state_row_bytes, src0_stride_inner_bytes, state_row_bytes, tile_n);
            count++;
        }

        if (n_qkv_rows > 0) {
            const dma_addr_t qkv_tile_ddr = src2->data +
                i3 * qkv_stride_seq_bytes + tile_off * sizeof(float);
            dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_T, qkv_tile_ddr),
                           d_inner_tile * sizeof(float), qkv_stride_token_bytes,
                           tile_n * sizeof(float), n_qkv_rows);
            count++;
        }
    } else {
        const dma_addr_t src0_tile_ddr = src0->data +
            i3 * src0_stride_seq_bytes + tile_off * src0_stride_inner_bytes;
        const size_t row_bytes = ncs * sizeof(float);

        dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_tile_raw, src0_tile_ddr),
                       row_bytes, src0_stride_inner_bytes, row_bytes, tile_n);
        count++;
    }

    return count;
}

static inline void ssm_conv_prefill_wait_input(dma_queue * dma_q, uint32_t count) {
    while (count-- > 0) {
        dma_queue_pop(dma_q);
    }
}

static void ssm_conv_thread_f32_prefill(unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict dst  = octx->dst;

    const bool fused = kparams->fused_concat != 0;
    const uint32_t n_state_rows = fused ? kparams->n_state_rows : 0;
    const struct htp_tensor * restrict src2 = fused ? octx->src[2] : NULL;

    dma_queue * dma_q = octx->ctx->dma[ith];

    const uint32_t d_conv  = kparams->d_conv;
    const uint32_t d_inner = kparams->d_inner;
    const uint32_t n_t     = kparams->n_t;
    const uint32_t n_s     = kparams->n_s;
    const uint32_t ncs     = fused ? (n_state_rows + src2->ne[1]) : src0->ne[0];

    const uint32_t src0_stride_inner = src0->nb[1] / sizeof(float);
    const uint32_t src0_stride_seq   = src0->nb[2] / sizeof(float);
    const uint32_t src1_stride_inner = src1->nb[1] / sizeof(float);
    const uint32_t dst_stride_token  = dst->nb[1]  / sizeof(float);
    const uint32_t dst_stride_seq    = dst->nb[2]  / sizeof(float);

    const uint32_t dr  = scctx->nrows_per_thread;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const uint32_t d_inner_per_thread = ir1 - ir0;
    const uint32_t d_inner_stride     = scctx->nrows_per_thread;
    const uint32_t d_inner_tile       = scctx->d_inner_tile;

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float       * dst_data  = (float       *) dst->data;

    // Per-thread VTCM regions.
    float * src0_T = (float *)(octx->src0_spad.data + ith * octx->src0_spad.size_per_thread);
    float * src1_T = (float *)(octx->src1_spad.data + ith * octx->src1_spad.size_per_thread);

    // Stage src1 weights once into VTCM in {d_inner_stride, d_conv} layout.
    transpose_src1(src1_data, src1_stride_inner, ir0, d_inner_per_thread, d_inner_stride, d_conv, src1_T);
    const size_t qkv_stride_token_bytes = fused ? src2->nb[1] : 0;
    const size_t qkv_stride_seq_bytes   = fused ? src2->nb[2] : 0;

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;

    const size_t weight_bytes    = (size_t) d_inner_per_thread * d_conv * sizeof(float);
    const size_t weight_raw_size = hex_round_up(weight_bytes, 128);

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    const uint32_t src0_raw_rows = fused ? n_state_rows : ncs;
    const size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * src0_raw_rows * sizeof(float), 128) + 128;
    const size_t src0_slot_size = fused ? octx->src0_spad.size_per_thread / 2 : octx->src0_spad.size_per_thread;
    float * dst_tile = (float *) dst_spad_base;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const HVX_Vector silu_one     = hvx_vec_splat_f32(1.f);
    const HVX_Vector silu_max_exp = hvx_vec_splat_f32(87.f);
    const HVX_Vector silu_min_exp = hvx_vec_splat_f32(-87.f);

    const dma_addr_t src1_ddr = src1->data + ir0 * d_conv * sizeof(float);
    dma_queue_push(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);
    dma_queue_pop(dma_q);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    const uint32_t n_tiles = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
    const uint32_t C_TILE = VLEN_FP32;

    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        if (fused) {
            const uint32_t count = ssm_conv_prefill_enqueue_input(
                dma_q, true, src0, src2, src0_spad_base, src0_slot_size,
                src0_tile_raw_bytes, d_inner_tile, 0, d_inner_per_thread, i3, ncs,
                n_state_rows, src0_stride_inner_bytes, src0_stride_seq_bytes,
                qkv_stride_token_bytes, qkv_stride_seq_bytes);
            ssm_conv_prefill_wait_input(dma_q, count);
        }

        for (uint32_t tile_idx = 0; tile_idx < n_tiles; ++tile_idx) {
            const uint32_t tile_off = tile_idx * d_inner_tile;
            const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);
            const uint32_t slot = fused ? (tile_idx & 1) : 0;
            uint8_t * slot_base = src0_spad_base + (size_t) slot * src0_slot_size;
            float * src0_tile_raw = (float *) slot_base;
            float * src0_T = (float *) (slot_base + src0_tile_raw_bytes);
            uint32_t next_input_count = 0;

            // Place src0 chunk into VTCM in {d_inner_tile, ncs} layout.
            const float * src0_block = src0_data + i3 * src0_stride_seq + (ir0 + tile_off) * src0_stride_inner;

            for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
                transpose_src0_block(src0_block + cb * src0_stride_inner, ncs, cb_n, d_inner_tile, src0_T, cb);
            if (!fused) {
                const uint32_t count = ssm_conv_prefill_enqueue_input(
                    dma_q, false, src0, NULL, src0_spad_base, src0_slot_size,
                    src0_tile_raw_bytes, d_inner_tile, tile_idx, d_inner_per_thread, i3,
                    ncs, 0, src0_stride_inner_bytes, src0_stride_seq_bytes, 0, 0);
                ssm_conv_prefill_wait_input(dma_q, count);
            } else if (tile_idx + 1 < n_tiles) {
                next_input_count = ssm_conv_prefill_enqueue_input(
                    dma_q, true, src0, src2, src0_spad_base, src0_slot_size,
                    src0_tile_raw_bytes, d_inner_tile, tile_idx + 1, d_inner_per_thread, i3,
                    ncs, n_state_rows, src0_stride_inner_bytes, src0_stride_seq_bytes,
                    qkv_stride_token_bytes, qkv_stride_seq_bytes);
            }

            if (fused) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
                if (n_state_rows > 0) {
                    for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                        const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
                        transpose_src0_state_rows_fused(src0_tile_raw + cb * n_state_rows, n_state_rows,
                                n_state_rows, cb_n, d_inner_tile, src0_T, cb);
                    }
                }
            } else {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
                for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
                    const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
                    transpose_src0_block(src0_tile_raw + cb * ncs, ncs, cb_n, d_inner_tile, src0_T, cb);
                }
            }

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

                    float * dst_ptr = dst_data + (ir0 + tile_off + cb) + t * dst_stride_token + i3 * dst_stride_seq;
                    if (cb_n == C_TILE) {
                        *(HVX_UVector *) dst_ptr = y;
                    } else {
                        hvx_vec_store_u(dst_ptr, cb_n * sizeof(float), y);
                    }
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);

    FARF(HIGH, "ssm-conv-f32-hvx %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,

            if (scctx->apply_silu) {
                HVX_Vector * restrict vdst_tile = (HVX_Vector *) dst_tile;
                const uint32_t total_vecs = (n_t * tile_n) / VLEN_FP32;
                _Pragma("unroll(4)")
                for (uint32_t i = 0; i < total_vecs; i++) {
                    HVX_Vector yv  = vdst_tile[i];
                    HVX_Vector sig = hvx_vec_fast_sigmoid_f32_guard(yv, silu_one, silu_max_exp, silu_min_exp);
                    vdst_tile[i]   = hvx_vec_mul_f32_f32(yv, sig);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);

            if (next_input_count > 0) {
                ssm_conv_prefill_wait_input(dma_q, next_input_count);
            }

            const dma_addr_t dst_tile_ddr = dst->data +
                i3 * dst_stride_seq_bytes + tile_off * sizeof(float);
            const size_t dst_row_bytes = tile_n * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(dst_tile_ddr, (uint8_t *) dst_tile),
                           dst_stride_token_bytes, dst_row_bytes, dst_row_bytes, n_t);
            dma_queue_pop(dma_q);
        }
    }

    FARF(HIGH, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u fused %d silu %d\n",
         ith, nth, ncs, d_inner, n_s, 1u, ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3], fused, scctx->apply_silu);
}
int op_ssm_conv_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t d_conv  = src1->ne[0];
    const uint32_t d_inner = src0->ne[1];
    const uint32_t n_t     = dst->ne[1];  // tokens per sequence
    const uint32_t n_s     = dst->ne[2];  // number of sequences in the batch

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = d_inner;

    if (octx->ctx->mdev.count > 1) {
        const uint32_t elems_per_chunk = VLEN_FP32;
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(d_inner, htp_tensor_mdev_data_aligned(dst) ? elems_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

    struct htp_ssm_conv_context scctx = { 0 };
    scctx.octx      = octx;
    scctx.row_start = row_start;
    scctx.nrows     = nrows;

    uint32_t use_hvx = 0;
    if (nrows >= VLEN_FP32 && n_t >= VLEN_FP32) {
        use_hvx = 1;
    }

    const uint32_t raw_rpt = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    scctx.nrows_per_thread = hex_round_up(raw_rpt, VLEN_FP32);

    const uint32_t d_inner_per_thread = scctx.nrows_per_thread;
    const uint32_t ncs                = src0->ne[0];

    const uint32_t src1_T_size = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 256);
    const uint32_t src0_T_max = HTP_SSM_CONV_VTCM_BUDGET > src1_T_size ? HTP_SSM_CONV_VTCM_BUDGET - src1_T_size : 0;

    uint32_t d_inner_tile = (src0_T_max / sizeof(float)) / ncs;
    d_inner_tile -= (d_inner_tile % VLEN_FP32);
    if (d_inner_tile == 0) {
        FARF(HIGH, "ssm_conv-f32: inner tile rounds to 0 (ncs=%u), falling back to scalar\n", ncs);
        use_hvx = 0;
    struct htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_thread = d_inner_per_thread,
        .d_inner_tile     = kparams->d_inner_tile,
        .row_start        = row_start,
        .nrows            = nrows,
        .fused            = (kparams->fused_concat != 0),
        .n_state_rows     = kparams->n_state_rows,
        .apply_silu       = (octx->op == HTP_OP_SSM_CONV_SILU),
    };

    FARF(HIGH, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : mode %s fused %d silu %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->n_t == 1 ? "decode" : "prefill", scctx.fused, scctx.apply_silu);

    if (kparams->n_t == 1) {
        // Neither the CONCAT-fusion nor the SILU-fusion decode worker path is
        // implemented (try_fuse_concat_ssm_conv and try_fuse_ssm_conv_silu both gate on
        // n_t > 1 host-side) -- if either ever reaches here, that is a host-side gating
        // bug, not something to silently miscompute.
        if (scctx.fused || scctx.apply_silu) {
            FARF(ERROR, "ssm_conv: fused mode (concat=%d silu=%d) reached the decode "
                 "(n_t==1) worker, which implements neither; host-side fusion should "
                 "never trigger this\n", scctx.fused, scctx.apply_silu);
            return HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_decode, &scctx, n_threads);
    } else {
        scctx.d_inner_tile = d_inner_tile;

        octx->src0_spad.size_per_thread = hex_round_up(d_inner_tile * ncs * sizeof(float), 256);
        octx->src1_spad.size_per_thread = src1_T_size;
        octx->dst_spad.size_per_thread  = 0;

        octx->src0_spad.size = octx->src0_spad.size_per_thread * n_threads;
        octx->src1_spad.size = octx->src1_spad.size_per_thread * n_threads;
        octx->dst_spad.size  = 0;

        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
        octx->src0_spad.src  = NULL;
        octx->src1_spad.src  = NULL;

        const size_t total_spad = octx->src0_spad.size + octx->src1_spad.size;
        if (total_spad > octx->ctx->vtcm_size) {
            FARF(HIGH, "ssm_conv-f32: scratchpad %zu exceeds VTCM %zu, falling back to scalar\n",
                 total_spad, octx->ctx->vtcm_size);
            use_hvx = 0;
        }
    }

    FARF(HIGH, "ssm-conv-f32: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : use_hvx %d\n", src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], use_hvx);

    if (use_hvx) {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_f32_hvx, &scctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, ssm_conv_thread_f32_f32, &scctx, n_threads);
    }

    return HTP_STATUS_OK;
}

int op_ssm_conv(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;

    int err = HTP_STATUS_OK;

    switch (dst->type) {
        case HTP_TYPE_F32:
            err = op_ssm_conv_f32(octx);
            break;
        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
