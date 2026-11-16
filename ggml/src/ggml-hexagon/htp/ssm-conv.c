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
#include "htp-trace.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "ssm-conv.h"

struct htp_ssm_conv_context {
    struct htp_ops_context *                  octx;
    const struct htp_ssm_conv_kernel_params * kparams;
    uint32_t                                  nrows_per_hvx;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
};

static __attribute__((noinline)) void hvx_ssm_conv_widen_f16_vtcm(const __fp16 * restrict src,
                                                                  float * restrict        dst,
                                                                  uint32_t                n_elems) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;
    HVX_Vector * restrict       vdst = (HVX_Vector *) dst;
    const uint32_t              n_v  = (n_elems + VLEN_FP16 - 1) / VLEN_FP16;
    for (uint32_t i = 0; i < n_v; ++i) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[i]);
        vdst[2 * i + 0]  = Q6_V_lo_W(p);
        vdst[2 * i + 1]  = Q6_V_hi_W(p);
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_pack_f16_rows_vtcm(float * restrict buf,
                                                                      uint32_t         n_rows,
                                                                      uint32_t         n_elems_per_row) {
    const HVX_Vector            f16_min = hvx_vec_splat_f32(-65504.0f);
    const HVX_Vector            f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector            zero    = Q6_V_vzero();
    const uint32_t              n_v32   = (n_elems_per_row + VLEN_FP32 - 1) / VLEN_FP32;
    const uint32_t              n_v16   = (n_v32 + 1) / 2;
    const HVX_Vector * restrict vsrc    = (const HVX_Vector *) buf;
    HVX_Vector * restrict       vdst    = (HVX_Vector *) buf;

    for (uint32_t r = 0; r < n_rows; ++r) {
        const HVX_Vector * restrict sr = vsrc + (size_t) r * n_v32;
        HVX_Vector * restrict       dr = vdst + (size_t) r * n_v16;
        for (uint32_t i = 0; i < n_v32; i += 2) {
            HVX_Vector v0 = sr[i];
            HVX_Vector v1 = (i + 1 < n_v32) ? sr[i + 1] : zero;
            v0            = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(v0, f16_max), f16_min);
            v1            = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(v1, f16_max), f16_min);
            dr[i / 2]     = hvx_vec_f32_to_f16(v0, v1);
        }
    }
}

// HVX deinterleave for d_conv == 4: channel-major raw VTCM -> tap-major T VTCM
static __attribute__((noinline)) void hvx_ssm_conv_unpack_to_T_4(const float * restrict raw,
                                                                 float * restrict       T,
                                                                 uint32_t               d_inner_per_hvx,
                                                                 uint32_t               d_inner_stride) {
    for (uint32_t cb = 0; cb < d_inner_per_hvx; cb += VLEN_FP32) {
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

// HVX gather transpose for general d_conv <= 32: channel-major raw VTCM -> tap-major T VTCM
static __attribute__((noinline)) void hvx_ssm_conv_unpack_to_T_gen(const float * restrict raw,
                                                                   float * restrict       T,
                                                                   uint32_t               d_inner_per_hvx,
                                                                   uint32_t               d_inner_stride,
                                                                   uint32_t               d_conv) {
    const uint32_t   raw_stride_bytes = d_conv * sizeof(float);
    const HVX_Vector vv               = hvx_vec_gather_offsets_w(raw_stride_bytes);

    for (uint32_t cb = 0; cb < d_inner_per_hvx; cb += VLEN_FP32) {
        const uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_hvx - cb);
        const size_t   base = (size_t) (raw + (size_t) cb * d_conv);
        const uint32_t mu   = cb_n * raw_stride_bytes - 1;

        for (uint32_t j = 0; j < d_conv; ++j) {
            Q6_vgather_ARMVw((HVX_Vector *) (T + (size_t) j * d_inner_stride + cb),
                             base + j * sizeof(float), mu, vv);
        }
    }
}

static inline void hvx_ssm_conv_unpack_to_T(const float * raw, float * T, uint32_t d_inner_per_hvx, uint32_t d_inner_stride, uint32_t d_conv) {
    if (d_conv == 4 && (d_inner_per_hvx % VLEN_FP32 == 0)) {
        hvx_ssm_conv_unpack_to_T_4(raw, T, d_inner_per_hvx, d_inner_stride);
    } else {
        hvx_ssm_conv_unpack_to_T_gen(raw, T, d_inner_per_hvx, d_inner_stride, d_conv);
    }
}

// HVX gather transpose + F16->F32 widen for decode when d_conv != 4
static __attribute__((noinline)) void hvx_ssm_conv_unpack_to_T_f16(const __fp16 * restrict raw,
                                                                   float * restrict        T,
                                                                   uint32_t                d_inner_per_hvx,
                                                                   uint32_t                d_inner_stride,
                                                                   uint32_t                d_conv) {
    const uint32_t   raw_stride_bytes = hex_round_up(d_conv, 2) * sizeof(__fp16);
    const HVX_Vector vv               = hvx_vec_gather_offsets_w(raw_stride_bytes);

    for (uint32_t cb = 0; cb < d_inner_per_hvx; cb += VLEN_FP32) {
        const uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_hvx - cb);
        const size_t   base = (size_t) ((const uint8_t *) raw + (size_t) cb * raw_stride_bytes);
        const uint32_t mu   = cb_n * raw_stride_bytes - 1;

        for (uint32_t j = 0; j < d_conv; j += 2) {
            Q6_vgather_ARMVw((HVX_Vector *) (T + (size_t) j * d_inner_stride + cb),
                             base + j * sizeof(__fp16), mu, vv);
        }
    }

    for (uint32_t cb = 0; cb < d_inner_per_hvx; cb += VLEN_FP32) {
        for (uint32_t j = 0; j < d_conv; j += 2) {
            HVX_Vector     v_pair = *(const HVX_Vector *) (T + (size_t) j * d_inner_stride + cb);
            HVX_VectorPair p      = hvx_vec_f16_to_f32_shuff(v_pair);
            *(HVX_Vector *) (T + (size_t) j * d_inner_stride + cb) = Q6_V_lo_W(p);
            if (j + 1 < d_conv) {
                *(HVX_Vector *) (T + (size_t) (j + 1) * d_inner_stride + cb) = Q6_V_hi_W(p);
            }
        }
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_silu_vtcm(float * restrict buf, uint32_t n_vecs) {
    HVX_Vector * restrict vbuf = (HVX_Vector *) buf;
    for (uint32_t i = 0; i < n_vecs; ++i) {
        vbuf[i] = hvx_vec_silu_f32(vbuf[i]);
    }
}

// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
// Keeps both operands in DMA layout - no transpose, no scratch.
static __attribute__((noinline)) void hvx_ssm_conv_decode_4(const float * restrict x,
                                                            const float * restrict w,
                                                            float * restrict       out,
                                                            uint32_t               n_ch) {
    for (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        const float * xp = x + cb * 4;
        const float * wp = w + cb * 4;

        HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(xp +  0), *(const HVX_Vector *)(wp +  0));
        HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(xp + 32), *(const HVX_Vector *)(wp + 32));
        HVX_Vector p2 = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(xp + 64), *(const HVX_Vector *)(wp + 64));
        HVX_Vector p3 = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(xp + 96), *(const HVX_Vector *)(wp + 96));

        HVX_VectorPair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
        HVX_VectorPair p23 = Q6_W_vdeal_VVR(p3, p2, -4);

        HVX_VectorPair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_VectorPair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
        HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));

        *(HVX_Vector *)(out + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_decode_4_f16(const __fp16 * restrict x,
                                                                const float * restrict  w,
                                                                float * restrict        out,
                                                                uint32_t                n_ch) {
    for (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
        const __fp16 * xp = x + cb * 4;
        const float *  wp = w + cb * 4;

        HVX_VectorPair x01 = hvx_vec_f16_to_f32(*(const HVX_Vector *)(xp +  0));
        HVX_VectorPair x23 = hvx_vec_f16_to_f32(*(const HVX_Vector *)(xp + 64));

        HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(x01), *(const HVX_Vector *)(wp +  0));
        HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(x01), *(const HVX_Vector *)(wp + 32));
        HVX_Vector p2 = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(x23), *(const HVX_Vector *)(wp + 64));
        HVX_Vector p3 = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(x23), *(const HVX_Vector *)(wp + 96));

        HVX_VectorPair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
        HVX_VectorPair p23 = Q6_W_vdeal_VVR(p3, p2, -4);

        HVX_VectorPair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
        HVX_VectorPair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);

        HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
        HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));

        *(HVX_Vector *)(out + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_decode_dot_gen(const float * restrict src0_T,
                                                                  const float * restrict src1_T,
                                                                  float * restrict       out,
                                                                  uint32_t               tile_n,
                                                                  uint32_t               tile_stride,
                                                                  uint32_t               d_inner_stride,
                                                                  uint32_t               tile_off,
                                                                  uint32_t               d_conv) {
    for (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
        HVX_Vector acc = hvx_vec_splat_f32(0.0f);
        for (uint32_t j = 0; j < d_conv; ++j) {
            HVX_Vector x = *(const HVX_Vector *)(src0_T + j * tile_stride + cb);
            HVX_Vector w = *(const HVX_Vector *)(src1_T + j * d_inner_stride + tile_off + cb);
            acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
        }
        *(HVX_Vector *)(out + cb) = Q6_Vsf_equals_Vqf32(acc);
    }
}

// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
// the lanes past the channel tail.
static __attribute__((noinline)) void hvx_ssm_conv_transpose_block(const float * restrict raw_block,
                                                                   float * restrict       T,
                                                                   uint32_t               raw_stride_bytes,
                                                                   uint32_t               ncs,
                                                                   uint32_t               cb_n) {
    const size_t     base = (size_t) raw_block;
    const uint32_t   mu   = cb_n * raw_stride_bytes - 1;
    const HVX_Vector vv   = hvx_vec_gather_offsets_w(raw_stride_bytes);

    for (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_Vector *) (T + (size_t) t * VLEN_FP32), base + t * sizeof(float), mu, vv);
    }
}

// Transpose + widen F16 src0 for prefill: each 32-bit word gather fetches 2 consecutive F16 time steps
// across 32 channels into even rows of T, then hvx_vec_f16_to_f32_shuff widens them into rows t and t+1.
static __attribute__((noinline)) void hvx_ssm_conv_transpose_block_f16(const __fp16 * restrict raw_block,
                                                                       float * restrict        T,
                                                                       uint32_t                raw_stride_bytes,
                                                                       uint32_t                ncs,
                                                                       uint32_t                cb_n) {
    const size_t     base  = (size_t) raw_block;
    const uint32_t   mu    = cb_n * raw_stride_bytes - 1;
    const HVX_Vector vv    = hvx_vec_gather_offsets_w(raw_stride_bytes);
    HVX_Vector *     T_vec = (HVX_Vector *) T;

    for (uint32_t t = 0; t < ncs; t += 2) {
        Q6_vgather_ARMVw(&T_vec[t], base + t * sizeof(__fp16), mu, vv);
    }

    for (uint32_t t = 0; t < ncs; t += 2) {
        HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(T_vec[t]);
        T_vec[t]         = Q6_V_lo_W(p);
        if (t + 1 < ncs) {
            T_vec[t + 1] = Q6_V_hi_W(p);
        }
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_prefill_dot_block(const float * restrict src0_T,
                                                                     const float * restrict wp,
                                                                     float * restrict       op,
                                                                     uint32_t               n_t,
                                                                     uint32_t               d_conv,
                                                                     uint32_t               d_inner_stride,
                                                                     uint32_t               dst_tile_stride) {
    const uint32_t C_TILE = VLEN_FP32;
    if (d_conv == 4) {
        const HVX_Vector w0 = *(const HVX_Vector *) (wp);
        const HVX_Vector w1 = *(const HVX_Vector *) (wp + d_inner_stride);
        const HVX_Vector w2 = *(const HVX_Vector *) (wp + 2 * d_inner_stride);
        const HVX_Vector w3 = *(const HVX_Vector *) (wp + 3 * d_inner_stride);

        HVX_Vector x0 = *(const HVX_Vector *) (src0_T);
        HVX_Vector x1 = *(const HVX_Vector *) (src0_T + C_TILE);
        HVX_Vector x2 = *(const HVX_Vector *) (src0_T + 2 * C_TILE);

        for (uint32_t t = 0; t < n_t; ++t) {
            const HVX_Vector x3 = *(const HVX_Vector *) (src0_T + (t + 3) * C_TILE);

            HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
                                                    Q6_Vqf32_vmpy_VsfVsf(x1, w1));
            HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
                                                    Q6_Vqf32_vmpy_VsfVsf(x3, w3));

            *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));

            x0 = x1;
            x1 = x2;
            x2 = x3;
        }
    } else {
        for (uint32_t t = 0; t < n_t; ++t) {
            HVX_Vector acc = hvx_vec_splat_f32(0.0f);
            for (uint32_t j = 0; j < d_conv; ++j) {
                HVX_Vector x = *(const HVX_Vector *) (src0_T + (t + j) * C_TILE);
                HVX_Vector w = *(const HVX_Vector *) (wp + j * d_inner_stride);
                acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
            }
            *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
        }
    }
}

// Single-token decode worker (n_t == 1)
static void ssm_conv_thread_f32_decode(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = th->dma_q;

    const uint32_t d_conv     = kparams->d_conv;
    const uint32_t d_inner    = kparams->d_inner;
    const uint32_t n_s        = kparams->n_s;
    const bool     apply_silu = kparams->silu != 0;
    const bool     src0_f16   = (src0->type == HTP_TYPE_F16);
    const bool     src1_f16   = (src1->type == HTP_TYPE_F16);
    const bool     dst_f16    = (dst->type  == HTP_TYPE_F16);

    const size_t src0_elem_size = src0_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t src1_elem_size = src1_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t dst_elem_size  = dst_f16  ? sizeof(__fp16) : sizeof(float);

    const uint32_t dr  = scctx->nrows_per_hvx;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_hvx = ir1 - ir0;
    const uint32_t d_inner_stride  = hex_round_up(d_inner_per_hvx, VLEN_FP32);
    const uint32_t d_inner_tile    = scctx->d_inner_tile;

    const size_t src0_stride_inner_bytes = src0->nb[1];
    const size_t src0_stride_seq_bytes   = src0->nb[2];
    const size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_hvx;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_hvx;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_hvx;

    const size_t weight_f32_bytes = (size_t) d_inner_per_hvx * d_conv * sizeof(float);
    const size_t weight_dma_bytes = (size_t) d_inner_per_hvx * d_conv * src1_elem_size;
    const size_t weight_raw_size  = hex_round_up(weight_f32_bytes, 128) + 128;

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    const size_t raw_stride_bytes    = src0_f16 ? ((size_t) hex_round_up(d_conv, 2) * sizeof(__fp16)) : ((size_t) d_conv * sizeof(float));
    const size_t src0_tile_raw_bytes = hex_round_up((size_t) d_inner_tile * raw_stride_bytes, 128);
    const size_t dst_tile_bytes      = hex_round_up((size_t) d_inner_tile * sizeof(float), 128);

    float * src0_tile_raw[2] = { (float *) src0_spad_base, (float *) (src0_spad_base + src0_tile_raw_bytes) };
    float * src0_T           = (float *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    float * dst_tile[2] = { (float *) dst_spad_base, (float *) (dst_spad_base + dst_tile_bytes) };

    struct htp_thread_trace * tr = th->trace;

    // raw_dot keeps both operands in the DMA layout, so src1 needs no transpose prep pass
    const bool raw_dot = (d_conv == 4) && (d_inner_per_hvx % VLEN_FP32 == 0);

    const uint32_t n_tiles   = fastdiv(d_inner_per_hvx + d_inner_tile - 1, &kparams->div_d_inner_tile);
    const uint32_t n_chunks  = n_s * n_tiles;
    const size_t   row_bytes = (size_t) d_conv * src0_elem_size;

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    #define SSM_CONV_DECODE_PUSH_FETCH(c)                                                         \
        do {                                                                                      \
            const uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_hvx - tile_off_fetch);    \
            const dma_addr_t fetch_ddr  = src0->data + (s_fetch * src0_stride_seq_bytes +         \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes);      \
            dma_queue_push(dma_q,                                                                 \
                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           raw_stride_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);     \
            tile_off_fetch += d_inner_tile;                                                       \
            if (tile_off_fetch >= d_inner_per_hvx) {                                              \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } while (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    const dma_addr_t src1_ddr      = src1->data + (size_t) ir0 * d_conv * src1_elem_size;
    uint8_t *        src1_dma_dest = src1_f16 ? (uint8_t *) src1_T : (uint8_t *) src1_raw;
    dma_queue_push(dma_q, dma_make_data(src1_dma_dest, src1_ddr), weight_dma_bytes, weight_dma_bytes, weight_dma_bytes, 1);

    dma_queue_push(dma_q, dma_make_data(dst->data, (uint8_t *) dst_tile[0]), 0, 0, 0, 0);
    SSM_CONV_DECODE_PUSH_FETCH(0);
    if (n_chunks > 1) {
        dma_queue_push(dma_q, dma_make_data(dst->data, (uint8_t *) dst_tile[1]), 0, 0, 0, 0);
        SSM_CONV_DECODE_PUSH_FETCH(1);
    }

    dma_queue_pop(dma_q);  // weights

    if (src1_f16 || !raw_dot) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
        if (src1_f16) {
            hvx_ssm_conv_widen_f16_vtcm((const __fp16 *) src1_T, src1_raw, d_inner_per_hvx * d_conv);
        }
        if (!raw_dot) {
            hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_hvx, d_inner_stride, d_conv);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    }

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    for (uint32_t c = 0; c < n_chunks; ++c) {
        const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_hvx - tile_off);

        dma_queue_pop(dma_q);      // writeback of chunk c-2, frees dst_tile[c & 1]
        dma_queue_pop(dma_q);      // fetch chunk c

        float * restrict out = dst_tile[c & 1];

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
        if (raw_dot) {
            const float * wp = src1_raw + tile_off * 4;
            if (src0_f16) {
                hvx_ssm_conv_decode_4_f16((const __fp16 *) src0_tile_raw[c & 1], wp, out, tile_n);
            } else {
                hvx_ssm_conv_decode_4(src0_tile_raw[c & 1], wp, out, tile_n);
            }
        } else {
            const uint32_t tile_stride = hex_round_up(tile_n, VLEN_FP32);
            if (src0_f16) {
                hvx_ssm_conv_unpack_to_T_f16((const __fp16 *) src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);
            } else {
                hvx_ssm_conv_unpack_to_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);
            }
            hvx_ssm_conv_decode_dot_gen(src0_T, src1_T, out, tile_n, tile_stride, d_inner_stride, tile_off, d_conv);
        }
        if (apply_silu) {
            hvx_ssm_conv_silu_vtcm(out, hex_round_up(tile_n, VLEN_FP32) / VLEN_FP32);
        }
        if (dst_f16) {
            hvx_ssm_conv_pack_f16_rows_vtcm(out, 1, tile_n);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);

        const dma_addr_t dst_ddr        = dst->data + (i3 * dst_stride_seq_bytes + (size_t) (ir0 + tile_off) * dst_elem_size);
        const size_t     tile_out_bytes = (size_t) tile_n * dst_elem_size;
        dma_queue_push(dma_q, dma_make_data(dst_ddr, (uint8_t *) out),
                       tile_out_bytes, tile_out_bytes, tile_out_bytes, 1);

        if (c + 2 < n_chunks) {
            SSM_CONV_DECODE_PUSH_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        if (tile_off >= d_inner_per_hvx) {
            tile_off = 0;
            i3++;
        }
    }

    dma_queue_flush(dma_q);

    #undef SSM_CONV_DECODE_PUSH_FETCH

    FARF(HIGH, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

// Multi-token prefill worker (n_t > 1)
static void ssm_conv_thread_f32_prefill(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = th->dma_q;

    const uint32_t d_conv     = kparams->d_conv;
    const uint32_t d_inner    = kparams->d_inner;
    const uint32_t n_t        = kparams->n_t;
    const uint32_t n_s        = kparams->n_s;
    const uint32_t ncs        = src0->ne[0];
    const bool     apply_silu = kparams->silu != 0;
    const bool     src0_f16   = (src0->type == HTP_TYPE_F16);
    const bool     src1_f16   = (src1->type == HTP_TYPE_F16);
    const bool     dst_f16    = (dst->type  == HTP_TYPE_F16);

    const size_t src0_elem_size = src0_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t src1_elem_size = src1_f16 ? sizeof(__fp16) : sizeof(float);
    const size_t dst_elem_size  = dst_f16  ? sizeof(__fp16) : sizeof(float);

    const uint32_t dr  = scctx->nrows_per_hvx;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_hvx = ir1 - ir0;
    const uint32_t d_inner_stride  = hex_round_up(d_inner_per_hvx, VLEN_FP32);
    const uint32_t d_inner_tile    = scctx->d_inner_tile;

    const size_t src0_stride_inner_bytes = src0->nb[1];
    const size_t src0_stride_seq_bytes   = src0->nb[2];
    const size_t dst_stride_token_bytes  = dst->nb[1];
    const size_t dst_stride_seq_bytes    = dst->nb[2];

    uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_hvx;
    uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_hvx;
    uint8_t * dst_spad_base  = octx->dst_spad.data  + ith * octx->dst_spad.size_per_hvx;

    const size_t weight_f32_bytes = (size_t) d_inner_per_hvx * d_conv * sizeof(float);
    const size_t weight_dma_bytes = (size_t) d_inner_per_hvx * d_conv * src1_elem_size;
    const size_t weight_raw_size  = hex_round_up(weight_f32_bytes, 128) + 128;

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
    const size_t raw_stride_bytes    = htp_ssm_conv_raw_stride_bytes(ncs, src0_elem_size);
    const size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * raw_stride_bytes, 128);
    const size_t dst_tile_bytes      = hex_round_up(d_inner_tile * n_t * sizeof(float), 128);

    float * src0_tile_raw[2] = { (float *) src0_spad_base, (float *) (src0_spad_base + src0_tile_raw_bytes) };
    float * src0_T           = (float *) (src0_spad_base + 2 * src0_tile_raw_bytes);

    float * dst_tile[2] = { (float *) dst_spad_base, (float *) (dst_spad_base + dst_tile_bytes) };

    struct htp_thread_trace * tr = th->trace;

    const uint32_t n_tiles   = fastdiv(d_inner_per_hvx + d_inner_tile - 1, &kparams->div_d_inner_tile);
    const uint32_t n_chunks  = n_s * n_tiles;
    const size_t   row_bytes = (size_t) ncs * src0_elem_size;

    uint32_t s_fetch        = 0;
    uint32_t tile_off_fetch = 0;

    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
    // Two fetches run ahead with dummy writeback pushes in prologue: W-2 F0 W-1 F1 W0 F2 ...
    #define SSM_CONV_PUSH_FETCH(c)                                                                \
        do {                                                                                      \
            const uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_hvx - tile_off_fetch);    \
            const dma_addr_t fetch_ddr  = src0->data + (s_fetch * src0_stride_seq_bytes +         \
                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes);      \
            dma_queue_push(dma_q,                                                                 \
                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
                           raw_stride_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);     \
            tile_off_fetch += d_inner_tile;                                                       \
            if (tile_off_fetch >= d_inner_per_hvx) {                                              \
                tile_off_fetch = 0;                                                               \
                s_fetch++;                                                                        \
            }                                                                                     \
        } while (0)

    // Queue weights and initial input tiles together so DDR reads overlap
    const dma_addr_t src1_ddr      = src1->data + (size_t) ir0 * d_conv * src1_elem_size;
    uint8_t *        src1_dma_dest = src1_f16 ? (uint8_t *) src1_T : (uint8_t *) src1_raw;
    dma_queue_push(dma_q, dma_make_data(src1_dma_dest, src1_ddr), weight_dma_bytes, weight_dma_bytes, weight_dma_bytes, 1);

    dma_queue_push(dma_q, dma_make_data(dst->data, (uint8_t *) dst_tile[0]), 0, 0, 0, 0);
    SSM_CONV_PUSH_FETCH(0);
    if (n_chunks > 1) {
        dma_queue_push(dma_q, dma_make_data(dst->data, (uint8_t *) dst_tile[1]), 0, 0, 0, 0);
        SSM_CONV_PUSH_FETCH(1);
    }

    dma_queue_pop(dma_q);  // weights

    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    if (src1_f16) {
        hvx_ssm_conv_widen_f16_vtcm((const __fp16 *) src1_T, src1_raw, d_inner_per_hvx * d_conv);
    }
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_hvx, d_inner_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    const uint32_t C_TILE = VLEN_FP32;

    uint32_t i3       = 0;
    uint32_t tile_off = 0;

    for (uint32_t c = 0; c < n_chunks; ++c) {
        const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_hvx - tile_off);

        const float * restrict raw = src0_tile_raw[c & 1];
        float * restrict       out = dst_tile[c & 1];

        dma_queue_pop(dma_q);      // writeback of chunk c-2, frees dst_tile[c & 1]
        dma_queue_pop(dma_q);      // fetch chunk c

        // Channel block outer, token inner: the taps and the sliding window stay in
        // registers, so a new output row costs one src0_T load.
        const uint32_t dst_tile_stride = hex_round_up(tile_n, C_TILE);

        for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
            const uint32_t cb_n = MIN(C_TILE, tile_n - cb);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            const uint8_t * raw_ch = (const uint8_t *) raw + (size_t) cb * raw_stride_bytes;
            if (src0_f16) {
                hvx_ssm_conv_transpose_block_f16((const __fp16 *) raw_ch, src0_T, raw_stride_bytes, ncs, cb_n);
            } else {
                hvx_ssm_conv_transpose_block((const float *) raw_ch, src0_T, raw_stride_bytes, ncs, cb_n);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            const float * restrict wp = src1_T + tile_off + cb;
            float * restrict       op = out + cb;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            hvx_ssm_conv_prefill_dot_block(src0_T, wp, op, n_t, d_conv, d_inner_stride, dst_tile_stride);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        if (apply_silu) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            hvx_ssm_conv_silu_vtcm(out, n_t * (dst_tile_stride / C_TILE));
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        size_t dst_vtcm_stride_bytes = (size_t) dst_tile_stride * sizeof(float);
        if (dst_f16) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            hvx_ssm_conv_pack_f16_rows_vtcm(out, n_t, tile_n);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            dst_vtcm_stride_bytes = (size_t) hex_round_up(tile_n, VLEN_FP16) * sizeof(__fp16);
        }

        // Writeback dst_tile VTCM -> DDR via 2D DMA
        dma_queue_push(dma_q, dma_make_data(dst->data + (i3 * dst_stride_seq_bytes + (size_t) (ir0 + tile_off) * dst_elem_size), (uint8_t *) out),
                       dst_stride_token_bytes, dst_vtcm_stride_bytes, (size_t) tile_n * dst_elem_size, n_t);

        if (c + 2 < n_chunks) {
            SSM_CONV_PUSH_FETCH(c + 2);
        }

        tile_off += d_inner_tile;
        if (tile_off >= d_inner_per_hvx) {
            tile_off = 0;
            i3++;
        }
    }

    dma_queue_flush(dma_q);

    #undef SSM_CONV_PUSH_FETCH

    FARF(HIGH, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
         dst->ne[2], dst->ne[3]);
}

int op_ssm_conv_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const struct htp_ssm_conv_kernel_params * kparams = (const struct htp_ssm_conv_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->d_inner;

    if (octx->ctx->mdev.count > 1) {
        const uint32_t elems_per_chunk = (dst->type == HTP_TYPE_F16) ? VLEN_FP16 : VLEN_FP32;
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

    const uint32_t n_hvx           = octx->n_hvx;
    const uint32_t raw_rpt         = fastdiv(nrows + n_hvx - 1, &octx->n_hvx_div);
    const uint32_t d_inner_per_hvx = hex_round_up(raw_rpt, VLEN_FP32);
    const size_t   src0_elem_size  = (src0->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    struct htp_ssm_conv_vtcm_layout L;
    htp_ssm_conv_vtcm_layout_build(&L, d_inner_per_hvx, kparams->d_conv, kparams->n_t, kparams->d_inner_tile, n_hvx, src0_elem_size);

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;

    octx->src0_spad.size_per_hvx = L.src0_bytes_per_hvx;
    octx->src1_spad.size_per_hvx = L.src1_bytes_per_hvx;
    octx->dst_spad.size_per_hvx  = L.dst_bytes_per_hvx;

    octx->src0_spad.size = L.src0_bytes;
    octx->src1_spad.size = L.src1_bytes;
    octx->dst_spad.size  = L.dst_bytes;

    octx->src0_spad.data = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    octx->src1_spad.data = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    octx->dst_spad.data  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    octx->src0_spad.src  = NULL;
    octx->src1_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    struct htp_ssm_conv_context scctx = {
        .octx             = octx,
        .kparams          = kparams,
        .nrows_per_hvx    = d_inner_per_hvx,
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
        work_queue_run(octx->ctx->hvx_queue, ssm_conv_thread_f32_decode, &scctx, n_hvx);
    } else {
        work_queue_run(octx->ctx->hvx_queue, ssm_conv_thread_f32_prefill, &scctx, n_hvx);
    }

    return HTP_STATUS_OK;
}

int op_ssm_conv(struct htp_ops_context * octx) {
    const struct htp_tensor * dst = octx->dst;

    switch (dst->type) {
        case HTP_TYPE_F32:
        case HTP_TYPE_F16:
            return op_ssm_conv_f32(octx);
        default:
            return HTP_STATUS_NO_SUPPORT;
    }
}
