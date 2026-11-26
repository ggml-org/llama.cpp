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

typedef void (*htp_ssm_conv_decode_dot_fn_t)(
    const void * restrict  raw_in,
    float * restrict       src0_T,
    const float * restrict src1_raw,
    const float * restrict src1_T,
    float * restrict       out,
    uint32_t               tile_n,
    uint32_t               d_inner_stride,
    uint32_t               tile_off,
    uint32_t               d_conv);

typedef void (*htp_ssm_conv_transpose_block_fn_t)(
    const void * restrict raw_block,
    float * restrict      T,
    uint32_t              raw_stride_bytes,
    uint32_t              ncs,
    uint32_t              cb_n);

typedef void (*htp_ssm_conv_prefill_dot_fn_t)(
    const float * restrict src0_T,
    const float * restrict wp,
    float * restrict       op,
    uint32_t               n_t,
    uint32_t               d_conv,
    uint32_t               d_inner_stride,
    uint32_t               dst_tile_stride);

typedef size_t (*htp_ssm_conv_epilogue_fn_t)(
    float * restrict out,
    uint32_t         n_t,
    uint32_t         tile_n);

typedef void (*htp_ssm_conv_token_step_fn_t)(
    float * restrict       hist,
    const void * restrict  x_tile,
    const float * restrict wp,
    void * restrict        dst_tile,
    float * restrict       strip_f32,
    uint32_t               cur_c,
    uint32_t               cur_t,
    uint32_t               d_conv,
    uint32_t               c_stride,
    uint32_t               w_stride,
    size_t                 x_row_stride,
    size_t                 d_row_stride);

typedef void (*htp_ssm_conv_state_unpack_fn_t)(
    float * restrict state_raw,
    float * restrict hist,
    uint32_t         cur_c,
    uint32_t         c_stride,
    uint32_t         s_taps);

typedef void (*htp_ssm_conv_state_pack_fn_t)(
    float * restrict hist,
    float * restrict state_raw,
    uint32_t         cur_c,
    uint32_t         c_stride,
    uint32_t         s_taps);

struct htp_ssm_conv_context {
    struct htp_ops_context *                  octx;
    const struct htp_ssm_conv_kernel_params * kparams;
    struct htp_ssm_conv_vtcm_layout           layout;
    uint8_t *                                 vtcm_base;
    uint32_t                                  nrows_per_hvx;
    uint32_t                                  d_inner_tile;
    uint32_t                                  row_start;
    uint32_t                                  nrows;
    bool                                      raw_dot;
    htp_ssm_conv_decode_dot_fn_t              decode_dot_fn;
    htp_ssm_conv_transpose_block_fn_t         transpose_block_fn;
    htp_ssm_conv_prefill_dot_fn_t             prefill_dot_fn;
    htp_ssm_conv_epilogue_fn_t                epilogue_fn;
    htp_ssm_conv_token_step_fn_t              token_step_fn;
    htp_ssm_conv_state_unpack_fn_t            state_unpack_fn;
    htp_ssm_conv_state_pack_fn_t              state_pack_fn;
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

static inline HVX_Vector hvx_ssm_conv_clamp_f16_f32(HVX_Vector x) {
    const HVX_Vector f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector f16_min = hvx_vec_splat_f32(-65504.0f);
    return Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, f16_min), f16_max);
}

static inline HVX_Vector hvx_ssm_conv_silu_pack_f16(HVX_Vector u0, HVX_Vector u1) {
    HVX_Vector x = hvx_vec_f32_to_f16(hvx_ssm_conv_clamp_f16_f32(u0),
                                      hvx_ssm_conv_clamp_f16_f32(u1));
    return hvx_vec_mul_f16_f16(x, hvx_vec_fast_sigmoid_f16(x));
}

static __attribute__((noinline)) void hvx_ssm_conv_pack_f16_rows_vtcm(float * restrict buf,
                                                                      uint32_t         n_rows,
                                                                      uint32_t         n_elems_per_row) {
    const HVX_Vector            f16_min = hvx_vec_splat_f32(-65504.0f);
    const HVX_Vector            f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector            zero    = Q6_V_vzero();
    const uint32_t              n_v32   = (n_elems_per_row + VLEN_FP32 - 1) / VLEN_FP32;
    const uint32_t              n_v16   = (n_v32 + 1) / 2;
    HVX_Vector * restrict vbuf = (HVX_Vector *) buf;

    for (uint32_t r = 0; r < n_rows; ++r) {
        HVX_Vector * sr = vbuf + (size_t) r * n_v32;
        HVX_Vector * dr = vbuf + (size_t) r * n_v16;
        for (uint32_t i = 0; i < n_v32; i += 2) {
            HVX_Vector v0 = sr[i];
            HVX_Vector v1 = (i + 1 < n_v32) ? sr[i + 1] : zero;
            v0            = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(v0, f16_max), f16_min);
            v1            = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(v1, f16_max), f16_min);
            dr[i / 2]     = hvx_vec_f32_to_f16(v0, v1);
        }
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_silu_pack_f16_rows_vtcm(float * restrict buf,
                                                                           uint32_t         n_rows,
                                                                           uint32_t         n_elems_per_row) {
    const HVX_Vector      zero  = Q6_V_vzero();
    const uint32_t        n_v32 = (n_elems_per_row + VLEN_FP32 - 1) / VLEN_FP32;
    const uint32_t        n_v16 = (n_v32 + 1) / 2;
    HVX_Vector * restrict vbuf  = (HVX_Vector *) buf;

    for (uint32_t r = 0; r < n_rows; ++r) {
        HVX_Vector * sr = vbuf + (size_t) r * n_v32;
        HVX_Vector * dr = vbuf + (size_t) r * n_v16;
        for (uint32_t i = 0; i < n_v32; i += 2) {
            HVX_Vector v0 = sr[i];
            HVX_Vector v1 = (i + 1 < n_v32) ? sr[i + 1] : zero;
            dr[i / 2]     = hvx_ssm_conv_silu_pack_f16(v0, v1);
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

    hvx_gather_sync(T);
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

    hvx_gather_sync(T);

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

static size_t ssm_conv_epilogue_f32(float * restrict out, uint32_t n_t, uint32_t tile_n) {
    (void) out;
    (void) n_t;
    return (size_t) hex_round_up(tile_n, VLEN_FP32) * sizeof(float);
}

static size_t ssm_conv_epilogue_silu_f32(float * restrict out, uint32_t n_t, uint32_t tile_n) {
    const uint32_t dst_tile_stride = hex_round_up(tile_n, VLEN_FP32);
    hvx_ssm_conv_silu_vtcm(out, n_t * (dst_tile_stride / VLEN_FP32));
    return (size_t) dst_tile_stride * sizeof(float);
}

static size_t ssm_conv_epilogue_f16(float * restrict out, uint32_t n_t, uint32_t tile_n) {
    hvx_ssm_conv_pack_f16_rows_vtcm(out, n_t, tile_n);
    return (size_t) hex_round_up(tile_n, VLEN_FP16) * sizeof(__fp16);
}

static size_t ssm_conv_epilogue_silu_f16(float * restrict out, uint32_t n_t, uint32_t tile_n) {
    hvx_ssm_conv_silu_pack_f16_rows_vtcm(out, n_t, tile_n);
    return (size_t) hex_round_up(tile_n, VLEN_FP16) * sizeof(__fp16);
}

static void ssm_conv_decode_dot_4_f32(const void * restrict  raw_in,
                                      float * restrict       src0_T,
                                      const float * restrict src1_raw,
                                      const float * restrict src1_T,
                                      float * restrict       out,
                                      uint32_t               tile_n,
                                      uint32_t               d_inner_stride,
                                      uint32_t               tile_off,
                                      uint32_t               d_conv) {
    (void) src0_T;
    (void) src1_T;
    (void) d_inner_stride;
    (void) d_conv;
    hvx_ssm_conv_decode_4((const float *) raw_in, src1_raw + tile_off * 4, out, tile_n);
}

static void ssm_conv_decode_dot_4_f16(const void * restrict  raw_in,
                                      float * restrict       src0_T,
                                      const float * restrict src1_raw,
                                      const float * restrict src1_T,
                                      float * restrict       out,
                                      uint32_t               tile_n,
                                      uint32_t               d_inner_stride,
                                      uint32_t               tile_off,
                                      uint32_t               d_conv) {
    (void) src0_T;
    (void) src1_T;
    (void) d_inner_stride;
    (void) d_conv;
    hvx_ssm_conv_decode_4_f16((const __fp16 *) raw_in, src1_raw + tile_off * 4, out, tile_n);
}

static void ssm_conv_decode_dot_gen_f32(const void * restrict  raw_in,
                                        float * restrict       src0_T,
                                        const float * restrict src1_raw,
                                        const float * restrict src1_T,
                                        float * restrict       out,
                                        uint32_t               tile_n,
                                        uint32_t               d_inner_stride,
                                        uint32_t               tile_off,
                                        uint32_t               d_conv) {
    (void) src1_raw;
    const uint32_t tile_stride = hex_round_up(tile_n, VLEN_FP32);
    hvx_ssm_conv_unpack_to_T((const float *) raw_in, src0_T, tile_n, tile_stride, d_conv);
    hvx_ssm_conv_decode_dot_gen(src0_T, src1_T, out, tile_n, tile_stride, d_inner_stride, tile_off, d_conv);
}

static void ssm_conv_decode_dot_gen_f16(const void * restrict  raw_in,
                                        float * restrict       src0_T,
                                        const float * restrict src1_raw,
                                        const float * restrict src1_T,
                                        float * restrict       out,
                                        uint32_t               tile_n,
                                        uint32_t               d_inner_stride,
                                        uint32_t               tile_off,
                                        uint32_t               d_conv) {
    (void) src1_raw;
    const uint32_t tile_stride = hex_round_up(tile_n, VLEN_FP32);
    hvx_ssm_conv_unpack_to_T_f16((const __fp16 *) raw_in, src0_T, tile_n, tile_stride, d_conv);
    hvx_ssm_conv_decode_dot_gen(src0_T, src1_T, out, tile_n, tile_stride, d_inner_stride, tile_off, d_conv);
}

// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
// the lanes past the channel tail.
static __attribute__((noinline)) void hvx_ssm_conv_transpose_block(const void * restrict raw_block,
                                                                   float * restrict      T,
                                                                   uint32_t              raw_stride_bytes,
                                                                   uint32_t              ncs,
                                                                   uint32_t              cb_n) {
    const size_t     base = (size_t) raw_block;
    const uint32_t   mu   = cb_n * raw_stride_bytes - 1;
    const HVX_Vector vv   = hvx_vec_gather_offsets_w(raw_stride_bytes);

    for (uint32_t t = 0; t < ncs; ++t) {
        Q6_vgather_ARMVw((HVX_Vector *) (T + (size_t) t * VLEN_FP32), base + t * sizeof(float), mu, vv);
    }

    hvx_gather_sync(T);
}

// Transpose + widen F16 src0 for prefill: each 32-bit word gather fetches 2 consecutive F16 time steps
// across 32 channels into even rows of T, then hvx_vec_f16_to_f32_shuff widens them into rows t and t+1.
static __attribute__((noinline)) void hvx_ssm_conv_transpose_block_f16(const void * restrict raw_block,
                                                                       float * restrict      T,
                                                                       uint32_t              raw_stride_bytes,
                                                                       uint32_t              ncs,
                                                                       uint32_t              cb_n) {
    const size_t     base  = (size_t) raw_block;
    const uint32_t   mu    = cb_n * raw_stride_bytes - 1;
    const HVX_Vector vv    = hvx_vec_gather_offsets_w(raw_stride_bytes);
    HVX_Vector *     T_vec = (HVX_Vector *) T;

    for (uint32_t t = 0; t < ncs; t += 2) {
        Q6_vgather_ARMVw(&T_vec[t], base + t * sizeof(__fp16), mu, vv);
    }

    hvx_gather_sync(T);

    for (uint32_t t = 0; t < ncs; t += 2) {
        HVX_VectorPair p = hvx_vec_f16_to_f32_shuff(T_vec[t]);
        T_vec[t]         = Q6_V_lo_W(p);
        if (t + 1 < ncs) {
            T_vec[t + 1] = Q6_V_hi_W(p);
        }
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_prefill_dot_block_4(const float * restrict src0_T,
                                                                       const float * restrict wp,
                                                                       float * restrict       op,
                                                                       uint32_t               n_t,
                                                                       uint32_t               d_conv,
                                                                       uint32_t               d_inner_stride,
                                                                       uint32_t               dst_tile_stride) {
    (void) d_conv;
    const uint32_t   C_TILE = VLEN_FP32;
    const HVX_Vector w0     = *(const HVX_Vector *) (wp);
    const HVX_Vector w1     = *(const HVX_Vector *) (wp + d_inner_stride);
    const HVX_Vector w2     = *(const HVX_Vector *) (wp + 2 * d_inner_stride);
    const HVX_Vector w3     = *(const HVX_Vector *) (wp + 3 * d_inner_stride);

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
}

static __attribute__((noinline)) void hvx_ssm_conv_prefill_dot_block_gen(const float * restrict src0_T,
                                                                         const float * restrict wp,
                                                                         float * restrict       op,
                                                                         uint32_t               n_t,
                                                                         uint32_t               d_conv,
                                                                         uint32_t               d_inner_stride,
                                                                         uint32_t               dst_tile_stride) {
    const uint32_t C_TILE = VLEN_FP32;
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

// HVX scatter transpose for tap-major hist [s_taps, c_stride] -> channel-major raw [cur_c, s_taps]
static __attribute__((noinline)) void hvx_ssm_conv_pack_from_T_gen(const float * restrict hist,
                                                                   float * restrict       raw,
                                                                   uint32_t               cur_c,
                                                                   uint32_t               c_stride,
                                                                   uint32_t               s_taps) {
    const uint32_t   raw_stride_bytes = s_taps * sizeof(float);
    const HVX_Vector vv               = hvx_vec_gather_offsets_w(raw_stride_bytes);

    for (uint32_t cb = 0; cb < cur_c; cb += VLEN_FP32) {
        const uint32_t cb_n = MIN(VLEN_FP32, cur_c - cb);
        const size_t   base = (size_t) (raw + (size_t) cb * s_taps);
        const uint32_t mu   = cb_n * raw_stride_bytes - 1;

        for (uint32_t j = 0; j < s_taps; ++j) {
            const HVX_Vector vj = *(const HVX_Vector *) (hist + (size_t) j * c_stride + cb);
            Q6_vscatter_RMVwV(base + j * sizeof(float), mu, vv, vj);
        }
    }

    hvx_gather_sync(raw);
}

static __attribute__((noinline)) void hvx_ssm_conv_pack_f16_flat_vtcm(const float * restrict src,
                                                                      __fp16 * restrict      dst,
                                                                      uint32_t               n_elems) {
    const HVX_Vector            f16_min = hvx_vec_splat_f32(-65504.0f);
    const HVX_Vector            f16_max = hvx_vec_splat_f32(65504.0f);
    const HVX_Vector            zero    = Q6_V_vzero();
    const uint32_t              n_v32   = (n_elems + VLEN_FP32 - 1) / VLEN_FP32;
    const HVX_Vector * restrict vsrc    = (const HVX_Vector *) src;
    HVX_Vector * restrict       vdst    = (HVX_Vector *) dst;

    for (uint32_t i = 0; i < n_v32; i += 2) {
        HVX_Vector v0 = vsrc[i];
        HVX_Vector v1 = (i + 1 < n_v32) ? vsrc[i + 1] : zero;
        v0            = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(v0, f16_max), f16_min);
        v1            = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(v1, f16_max), f16_min);
        vdst[i / 2]   = hvx_vec_f32_to_f16(v0, v1);
    }
}

static void ssm_conv_state_unpack_f32_nat(float * restrict state_raw,
                                          float * restrict hist,
                                          uint32_t         cur_c,
                                          uint32_t         c_stride,
                                          uint32_t         s_taps) {
    hvx_ssm_conv_unpack_to_T_gen(state_raw, hist, cur_c, c_stride, s_taps);
}

static void ssm_conv_state_unpack_f16_nat(float * restrict state_raw,
                                          float * restrict hist,
                                          uint32_t         cur_c,
                                          uint32_t         c_stride,
                                          uint32_t         s_taps) {
    hvx_ssm_conv_widen_f16_vtcm((const __fp16 *) hist, state_raw, cur_c * s_taps);
    hvx_ssm_conv_unpack_to_T_gen(state_raw, hist, cur_c, c_stride, s_taps);
}

static void ssm_conv_state_pack_f32_nat(float * restrict hist,
                                        float * restrict state_raw,
                                        uint32_t         cur_c,
                                        uint32_t         c_stride,
                                        uint32_t         s_taps) {
    hvx_ssm_conv_pack_from_T_gen(hist, state_raw, cur_c, c_stride, s_taps);
}

static void ssm_conv_state_pack_f16_nat(float * restrict hist,
                                        float * restrict state_raw,
                                        uint32_t         cur_c,
                                        uint32_t         c_stride,
                                        uint32_t         s_taps) {
    hvx_ssm_conv_pack_from_T_gen(hist, state_raw, cur_c, c_stride, s_taps);
    hvx_ssm_conv_pack_f16_flat_vtcm(state_raw, (__fp16 *) hist, cur_c * s_taps);
}

static __attribute__((noinline)) void hvx_ssm_conv_pack_f16_strip_1d(const float * restrict strip_f32,
                                                                     uint8_t * restrict     dp,
                                                                     uint32_t               cur_t,
                                                                     size_t                 d_row_stride,
                                                                     bool                   has_hi) {
    const HVX_Vector * restrict sp     = (const HVX_Vector *) strip_f32;
    const uint32_t              hi_off = has_hi ? 1 : 0;
    _Pragma("clang loop unroll(disable)")
    for (uint32_t t = 0; t < cur_t; ++t) {
        HVX_Vector u0      = sp[0];
        HVX_Vector u1      = sp[hi_off];
        *(HVX_Vector *) dp = hvx_vec_f32_to_f16(hvx_ssm_conv_clamp_f16_f32(u0),
                                                hvx_ssm_conv_clamp_f16_f32(u1));
        sp += 2;
        dp += d_row_stride;
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_silu_pack_f16_strip_1d(const float * restrict strip_f32,
                                                                          uint8_t * restrict     dp,
                                                                          uint32_t               cur_t,
                                                                          size_t                 d_row_stride,
                                                                          bool                   has_hi) {
    const HVX_Vector * restrict sp     = (const HVX_Vector *) strip_f32;
    const uint32_t              hi_off = has_hi ? 1 : 0;

#if __HVX_ARCH__ >= 81
    _Pragma("clang loop unroll(disable)")
#else
    _Pragma("clang loop unroll(disable) pipeline(disable)")
#endif
    for (uint32_t t = 0; t < cur_t; ++t) {
        *(HVX_Vector *) dp = hvx_ssm_conv_silu_pack_f16(sp[0], sp[hi_off]);
        sp += 2;
        dp += d_row_stride;
    }
}

#define DEFINE_SSM_CONV_TOKEN_4_DOT_STRIP(NAME, X_F16)                                                \
    static __attribute__((noinline)) void NAME(float * restrict       hist,                           \
                                               const void * restrict  x_tile,                         \
                                               const float * restrict wp,                             \
                                               void * restrict        out_ptr,                        \
                                               uint32_t               cb,                             \
                                               uint32_t               cur_t,                          \
                                               uint32_t               c_stride,                       \
                                               uint32_t               w_stride,                       \
                                               size_t                 x_row_stride,                   \
                                               size_t                 out_row_stride) {               \
        const HVX_Vector w0_0 = *(const HVX_Vector *) (wp + 0 * w_stride + cb + 0);                   \
        const HVX_Vector w0_1 = *(const HVX_Vector *) (wp + 0 * w_stride + cb + 32);                  \
        const HVX_Vector w1_0 = *(const HVX_Vector *) (wp + 1 * w_stride + cb + 0);                   \
        const HVX_Vector w1_1 = *(const HVX_Vector *) (wp + 1 * w_stride + cb + 32);                  \
        const HVX_Vector w2_0 = *(const HVX_Vector *) (wp + 2 * w_stride + cb + 0);                   \
        const HVX_Vector w2_1 = *(const HVX_Vector *) (wp + 2 * w_stride + cb + 32);                  \
        const HVX_Vector w3_0 = *(const HVX_Vector *) (wp + 3 * w_stride + cb + 0);                   \
        const HVX_Vector w3_1 = *(const HVX_Vector *) (wp + 3 * w_stride + cb + 32);                  \
                                                                                                      \
        HVX_Vector xm3_0 = *(const HVX_Vector *) (hist + 0 * c_stride + cb + 0);                      \
        HVX_Vector xm3_1 = *(const HVX_Vector *) (hist + 0 * c_stride + cb + 32);                     \
        HVX_Vector xm2_0 = *(const HVX_Vector *) (hist + 1 * c_stride + cb + 0);                      \
        HVX_Vector xm2_1 = *(const HVX_Vector *) (hist + 1 * c_stride + cb + 32);                     \
        HVX_Vector xm1_0 = *(const HVX_Vector *) (hist + 2 * c_stride + cb + 0);                      \
        HVX_Vector xm1_1 = *(const HVX_Vector *) (hist + 2 * c_stride + cb + 32);                     \
                                                                                                      \
        const uint8_t * restrict xp = (const uint8_t *) x_tile + cb * (X_F16 ? 2 : 4);                \
        uint8_t * restrict       dp = (uint8_t *) out_ptr;                                            \
                                                                                                      \
        _Pragma("clang loop unroll(disable)")                                                         \
        for (uint32_t t = 0; t < cur_t; ++t) {                                                        \
            HVX_Vector x0_0;                                                                          \
            HVX_Vector x0_1;                                                                          \
            if (X_F16) {                                                                              \
                HVX_VectorPair x0_p = hvx_vec_f16_to_f32(*(const HVX_Vector *) xp);                   \
                x0_0                = Q6_V_lo_W(x0_p);                                                \
                x0_1                = Q6_V_hi_W(x0_p);                                                \
            } else {                                                                                  \
                x0_0 = *((const HVX_Vector *) xp + 0);                                                \
                x0_1 = *((const HVX_Vector *) xp + 1);                                                \
            }                                                                                         \
                                                                                                      \
            HVX_Vector a0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(xm3_0, w0_0),               \
                                                     Q6_Vqf32_vmpy_VsfVsf(xm2_0, w1_0));              \
            HVX_Vector b0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(xm1_0, w2_0),               \
                                                     Q6_Vqf32_vmpy_VsfVsf(x0_0,  w3_0));              \
            HVX_Vector a1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(xm3_1, w0_1),               \
                                                     Q6_Vqf32_vmpy_VsfVsf(xm2_1, w1_1));              \
            HVX_Vector b1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(xm1_1, w2_1),               \
                                                     Q6_Vqf32_vmpy_VsfVsf(x0_1,  w3_1));              \
            *((HVX_Vector *) dp + 0) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a0, b0));         \
            *((HVX_Vector *) dp + 1) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a1, b1));         \
                                                                                                      \
            xm3_0 = xm2_0;                                                                            \
            xm3_1 = xm2_1;                                                                            \
            xm2_0 = xm1_0;                                                                            \
            xm2_1 = xm1_1;                                                                            \
            xm1_0 = x0_0;                                                                             \
            xm1_1 = x0_1;                                                                             \
                                                                                                      \
            xp += x_row_stride;                                                                       \
            dp += out_row_stride;                                                                     \
        }                                                                                             \
                                                                                                      \
        *(HVX_Vector *) (hist + 0 * c_stride + cb + 0)  = xm3_0;                                      \
        *(HVX_Vector *) (hist + 0 * c_stride + cb + 32) = xm3_1;                                      \
        *(HVX_Vector *) (hist + 1 * c_stride + cb + 0)  = xm2_0;                                      \
        *(HVX_Vector *) (hist + 1 * c_stride + cb + 32) = xm2_1;                                      \
        *(HVX_Vector *) (hist + 2 * c_stride + cb + 0)  = xm1_0;                                      \
        *(HVX_Vector *) (hist + 2 * c_stride + cb + 32) = xm1_1;                                      \
    }

DEFINE_SSM_CONV_TOKEN_4_DOT_STRIP(hvx_ssm_conv_token_4_dot_f32_strip, false)
DEFINE_SSM_CONV_TOKEN_4_DOT_STRIP(hvx_ssm_conv_token_4_dot_f16_strip, true)

#undef DEFINE_SSM_CONV_TOKEN_4_DOT_STRIP

#define DEFINE_SSM_CONV_TOKEN_4_DOT_SILU_F32(NAME, X_F16)                                           \
    static __attribute__((noinline)) void NAME(float * restrict       hist,                         \
                                               const void * restrict  x_tile,                       \
                                               const float * restrict wp,                           \
                                               void * restrict        dst_tile,                     \
                                               uint32_t               cur_c,                        \
                                               uint32_t               cur_t,                        \
                                               uint32_t               c_stride,                     \
                                               uint32_t               w_stride,                     \
                                               size_t                 x_row_stride,                 \
                                               size_t                 f32_row_stride) {             \
        for (uint32_t cb = 0; cb < cur_c; cb += 64) {                                               \
            for (uint32_t sub = 0; sub < 64 && cb + sub < cur_c; sub += 32) {                       \
                const uint32_t   ch = cb + sub;                                                     \
                const HVX_Vector w0 = *(const HVX_Vector *) (wp + 0 * w_stride + ch);               \
                const HVX_Vector w1 = *(const HVX_Vector *) (wp + 1 * w_stride + ch);               \
                const HVX_Vector w2 = *(const HVX_Vector *) (wp + 2 * w_stride + ch);               \
                const HVX_Vector w3 = *(const HVX_Vector *) (wp + 3 * w_stride + ch);               \
                                                                                                    \
                HVX_Vector xm3 = *(const HVX_Vector *) (hist + 0 * c_stride + ch);                  \
                HVX_Vector xm2 = *(const HVX_Vector *) (hist + 1 * c_stride + ch);                  \
                HVX_Vector xm1 = *(const HVX_Vector *) (hist + 2 * c_stride + ch);                  \
                                                                                                    \
                const uint8_t * restrict xp = (const uint8_t *) x_tile +                            \
                                              (X_F16 ? (cb * 2) : (ch * 4));                        \
                uint8_t * restrict       dp = (uint8_t *) dst_tile + ch * sizeof(float);            \
                                                                                                    \
                _Pragma("clang loop unroll(disable) pipeline(disable)")                             \
                for (uint32_t t = 0; t < cur_t; ++t) {                                              \
                    HVX_Vector x0;                                                                  \
                    if (X_F16) {                                                                    \
                        HVX_VectorPair x0_p = hvx_vec_f16_to_f32(*(const HVX_Vector *) xp);         \
                        x0                  = (sub == 0) ? Q6_V_lo_W(x0_p) : Q6_V_hi_W(x0_p);       \
                    } else {                                                                        \
                        x0 = *(const HVX_Vector *) xp;                                              \
                    }                                                                               \
                                                                                                    \
                    HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(xm3, w0),          \
                                                            Q6_Vqf32_vmpy_VsfVsf(xm2, w1));         \
                    HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(xm1, w2),          \
                                                            Q6_Vqf32_vmpy_VsfVsf(x0,  w3));         \
                    HVX_Vector u = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));             \
                    *(HVX_Vector *) dp = hvx_vec_silu_f32(u);                                       \
                                                                                                    \
                    xm3 = xm2;                                                                      \
                    xm2 = xm1;                                                                      \
                    xm1 = x0;                                                                       \
                                                                                                    \
                    xp += x_row_stride;                                                             \
                    dp += f32_row_stride;                                                           \
                }                                                                                   \
                                                                                                    \
                *(HVX_Vector *) (hist + 0 * c_stride + ch) = xm3;                                   \
                *(HVX_Vector *) (hist + 1 * c_stride + ch) = xm2;                                   \
                *(HVX_Vector *) (hist + 2 * c_stride + ch) = xm1;                                   \
            }                                                                                       \
        }                                                                                           \
    }

DEFINE_SSM_CONV_TOKEN_4_DOT_SILU_F32(hvx_ssm_conv_token_4_dot_f32_silu, false)
DEFINE_SSM_CONV_TOKEN_4_DOT_SILU_F32(hvx_ssm_conv_token_4_dot_f16_silu, true)

#undef DEFINE_SSM_CONV_TOKEN_4_DOT_SILU_F32

static __attribute__((noinline)) void hvx_ssm_conv_token_4_f16_f16(float * restrict       hist,
                                                                   const void * restrict  x_tile,
                                                                   const float * restrict wp,
                                                                   void * restrict        dst_tile,
                                                                   float * restrict       strip_f32,
                                                                   uint32_t               cur_c,
                                                                   uint32_t               cur_t,
                                                                   uint32_t               d_conv,
                                                                   uint32_t               c_stride,
                                                                   uint32_t               w_stride,
                                                                   size_t                 x_row_stride,
                                                                   size_t                 d_row_stride) {
    (void) d_conv;
    for (uint32_t cb = 0; cb < cur_c; cb += 64) {
        hvx_ssm_conv_token_4_dot_f16_strip(hist, x_tile, wp, strip_f32, cb, cur_t, c_stride, w_stride, x_row_stride, 256);
        hvx_ssm_conv_pack_f16_strip_1d(strip_f32, (uint8_t *) dst_tile + cb * sizeof(__fp16), cur_t, d_row_stride, cb + 32 < cur_c);
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_token_4_f16_f16_silu(float * restrict       hist,
                                                                        const void * restrict  x_tile,
                                                                        const float * restrict wp,
                                                                        void * restrict        dst_tile,
                                                                        float * restrict       strip_f32,
                                                                        uint32_t               cur_c,
                                                                        uint32_t               cur_t,
                                                                        uint32_t               d_conv,
                                                                        uint32_t               c_stride,
                                                                        uint32_t               w_stride,
                                                                        size_t                 x_row_stride,
                                                                        size_t                 d_row_stride) {
    (void) d_conv;
    for (uint32_t cb = 0; cb < cur_c; cb += 64) {
        hvx_ssm_conv_token_4_dot_f16_strip(hist, x_tile, wp, strip_f32, cb, cur_t, c_stride, w_stride, x_row_stride, 256);
        hvx_ssm_conv_silu_pack_f16_strip_1d(strip_f32, (uint8_t *) dst_tile + cb * sizeof(__fp16), cur_t, d_row_stride, cb + 32 < cur_c);
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_token_4_f32_f32(float * restrict       hist,
                                                                   const void * restrict  x_tile,
                                                                   const float * restrict wp,
                                                                   void * restrict        dst_tile,
                                                                   float * restrict       strip_f32,
                                                                   uint32_t               cur_c,
                                                                   uint32_t               cur_t,
                                                                   uint32_t               d_conv,
                                                                   uint32_t               c_stride,
                                                                   uint32_t               w_stride,
                                                                   size_t                 x_row_stride,
                                                                   size_t                 d_row_stride) {
    (void) d_conv;
    (void) strip_f32;
    for (uint32_t cb = 0; cb < cur_c; cb += 64) {
        hvx_ssm_conv_token_4_dot_f32_strip(hist, x_tile, wp, (uint8_t *) dst_tile + cb * sizeof(float), cb, cur_t, c_stride, w_stride, x_row_stride, d_row_stride);
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_token_4_f32_f32_silu(float * restrict       hist,
                                                                        const void * restrict  x_tile,
                                                                        const float * restrict wp,
                                                                        void * restrict        dst_tile,
                                                                        float * restrict       strip_f32,
                                                                        uint32_t               cur_c,
                                                                        uint32_t               cur_t,
                                                                        uint32_t               d_conv,
                                                                        uint32_t               c_stride,
                                                                        uint32_t               w_stride,
                                                                        size_t                 x_row_stride,
                                                                        size_t                 d_row_stride) {
    (void) d_conv;
    (void) strip_f32;
    hvx_ssm_conv_token_4_dot_f32_silu(hist, x_tile, wp, dst_tile, cur_c, cur_t, c_stride, w_stride, x_row_stride, d_row_stride);
}

static __attribute__((noinline)) void hvx_ssm_conv_token_4_f32_f16(float * restrict       hist,
                                                                   const void * restrict  x_tile,
                                                                   const float * restrict wp,
                                                                   void * restrict        dst_tile,
                                                                   float * restrict       strip_f32,
                                                                   uint32_t               cur_c,
                                                                   uint32_t               cur_t,
                                                                   uint32_t               d_conv,
                                                                   uint32_t               c_stride,
                                                                   uint32_t               w_stride,
                                                                   size_t                 x_row_stride,
                                                                   size_t                 d_row_stride) {
    (void) d_conv;
    for (uint32_t cb = 0; cb < cur_c; cb += 64) {
        hvx_ssm_conv_token_4_dot_f32_strip(hist, x_tile, wp, strip_f32, cb, cur_t, c_stride, w_stride, x_row_stride, 256);
        hvx_ssm_conv_pack_f16_strip_1d(strip_f32, (uint8_t *) dst_tile + cb * sizeof(__fp16), cur_t, d_row_stride, cb + 32 < cur_c);
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_token_4_f32_f16_silu(float * restrict       hist,
                                                                        const void * restrict  x_tile,
                                                                        const float * restrict wp,
                                                                        void * restrict        dst_tile,
                                                                        float * restrict       strip_f32,
                                                                        uint32_t               cur_c,
                                                                        uint32_t               cur_t,
                                                                        uint32_t               d_conv,
                                                                        uint32_t               c_stride,
                                                                        uint32_t               w_stride,
                                                                        size_t                 x_row_stride,
                                                                        size_t                 d_row_stride) {
    (void) d_conv;
    for (uint32_t cb = 0; cb < cur_c; cb += 64) {
        hvx_ssm_conv_token_4_dot_f32_strip(hist, x_tile, wp, strip_f32, cb, cur_t, c_stride, w_stride, x_row_stride, 256);
        hvx_ssm_conv_silu_pack_f16_strip_1d(strip_f32, (uint8_t *) dst_tile + cb * sizeof(__fp16), cur_t, d_row_stride, cb + 32 < cur_c);
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_token_4_f16_f32(float * restrict       hist,
                                                                   const void * restrict  x_tile,
                                                                   const float * restrict wp,
                                                                   void * restrict        dst_tile,
                                                                   float * restrict       strip_f32,
                                                                   uint32_t               cur_c,
                                                                   uint32_t               cur_t,
                                                                   uint32_t               d_conv,
                                                                   uint32_t               c_stride,
                                                                   uint32_t               w_stride,
                                                                   size_t                 x_row_stride,
                                                                   size_t                 d_row_stride) {
    (void) d_conv;
    (void) strip_f32;
    for (uint32_t cb = 0; cb < cur_c; cb += 64) {
        hvx_ssm_conv_token_4_dot_f16_strip(hist, x_tile, wp, (uint8_t *) dst_tile + cb * sizeof(float), cb, cur_t, c_stride, w_stride, x_row_stride, d_row_stride);
    }
}

static __attribute__((noinline)) void hvx_ssm_conv_token_4_f16_f32_silu(float * restrict       hist,
                                                                        const void * restrict  x_tile,
                                                                        const float * restrict wp,
                                                                        void * restrict        dst_tile,
                                                                        float * restrict       strip_f32,
                                                                        uint32_t               cur_c,
                                                                        uint32_t               cur_t,
                                                                        uint32_t               d_conv,
                                                                        uint32_t               c_stride,
                                                                        uint32_t               w_stride,
                                                                        size_t                 x_row_stride,
                                                                        size_t                 d_row_stride) {
    (void) d_conv;
    (void) strip_f32;
    hvx_ssm_conv_token_4_dot_f16_silu(hist, x_tile, wp, dst_tile, cur_c, cur_t, c_stride, w_stride, x_row_stride, d_row_stride);
}

#define DEFINE_SSM_CONV_TOKEN_GEN(NAME, X_F16, DST_F16, APPLY_SILU)                                     \
    static __attribute__((noinline)) void NAME(float * restrict       hist,                             \
                                               const void * restrict  x_tile,                           \
                                               const float * restrict wp,                               \
                                               void * restrict        dst_tile,                         \
                                               float * restrict       strip_f32,                        \
                                               uint32_t               cur_c,                            \
                                               uint32_t               cur_t,                            \
                                               uint32_t               d_conv,                           \
                                               uint32_t               c_stride,                         \
                                               uint32_t               w_stride,                         \
                                               size_t                 x_row_stride,                     \
                                               size_t                 d_row_stride) {                   \
        (void) strip_f32;                                                                               \
        const uint32_t s_taps = d_conv - 1;                                                             \
        for (uint32_t cb = 0; cb < cur_c; cb += 64) {                                                   \
            const uint8_t * restrict xp = (const uint8_t *) x_tile + cb * (X_F16 ? 2 : 4);              \
            uint8_t * restrict       dp = (uint8_t *) dst_tile + cb * (DST_F16 ? 2 : 4);                \
                                                                                                        \
            _Pragma("clang loop unroll(disable) pipeline(disable)")                                     \
            for (uint32_t t = 0; t < cur_t; ++t) {                                                      \
                HVX_Vector x0_0;                                                                        \
                HVX_Vector x0_1;                                                                        \
                if (X_F16) {                                                                            \
                    HVX_VectorPair x0_p = hvx_vec_f16_to_f32(*(const HVX_Vector *) xp);                 \
                    x0_0                = Q6_V_lo_W(x0_p);                                              \
                    x0_1                = Q6_V_hi_W(x0_p);                                              \
                } else {                                                                                \
                    x0_0 = *((const HVX_Vector *) xp + 0);                                              \
                    x0_1 = *((const HVX_Vector *) xp + 1);                                              \
                }                                                                                       \
                                                                                                        \
                HVX_Vector acc0 = Q6_Vqf32_vmpy_VsfVsf(                                                 \
                    x0_0, *(const HVX_Vector *) (wp + (size_t) s_taps * w_stride + cb + 0));            \
                HVX_Vector acc1 = Q6_Vqf32_vmpy_VsfVsf(                                                 \
                    x0_1, *(const HVX_Vector *) (wp + (size_t) s_taps * w_stride + cb + 32));           \
                                                                                                        \
                _Pragma("clang loop unroll(disable)")                                                   \
                for (uint32_t j = 0; j < s_taps; ++j) {                                                 \
                    HVX_Vector hj_0 = *(const HVX_Vector *) (hist + (size_t) j * c_stride + cb + 0);    \
                    HVX_Vector hj_1 = *(const HVX_Vector *) (hist + (size_t) j * c_stride + cb + 32);   \
                    HVX_Vector wj_0 = *(const HVX_Vector *) (wp + (size_t) j * w_stride + cb + 0);      \
                    HVX_Vector wj_1 = *(const HVX_Vector *) (wp + (size_t) j * w_stride + cb + 32);     \
                    acc0            = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(hj_0, wj_0)); \
                    acc1            = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(hj_1, wj_1)); \
                    if (j > 0) {                                                                        \
                        *(HVX_Vector *) (hist + (size_t) (j - 1) * c_stride + cb + 0)  = hj_0;          \
                        *(HVX_Vector *) (hist + (size_t) (j - 1) * c_stride + cb + 32) = hj_1;          \
                    }                                                                                   \
                }                                                                                       \
                *(HVX_Vector *) (hist + (size_t) (s_taps - 1) * c_stride + cb + 0)  = x0_0;             \
                *(HVX_Vector *) (hist + (size_t) (s_taps - 1) * c_stride + cb + 32) = x0_1;             \
                                                                                                        \
                HVX_Vector u0 = Q6_Vsf_equals_Vqf32(acc0);                                              \
                HVX_Vector u1 = Q6_Vsf_equals_Vqf32(acc1);                                              \
                                                                                                        \
                if (DST_F16) {                                                                          \
                    if (APPLY_SILU) {                                                                   \
                        *(HVX_Vector *) dp = hvx_ssm_conv_silu_pack_f16(u0, u1);                        \
                    } else {                                                                            \
                        *(HVX_Vector *) dp = hvx_vec_f32_to_f16(hvx_ssm_conv_clamp_f16_f32(u0),         \
                                                                hvx_ssm_conv_clamp_f16_f32(u1));        \
                    }                                                                                   \
                } else {                                                                                \
                    if (APPLY_SILU) {                                                                   \
                        u0 = hvx_vec_silu_f32(u0);                                                      \
                        u1 = hvx_vec_silu_f32(u1);                                                      \
                    }                                                                                   \
                    *((HVX_Vector *) dp + 0) = u0;                                                      \
                    *((HVX_Vector *) dp + 1) = u1;                                                      \
                }                                                                                       \
                                                                                                        \
                xp += x_row_stride;                                                                     \
                dp += d_row_stride;                                                                     \
            }                                                                                           \
        }                                                                                               \
    }

DEFINE_SSM_CONV_TOKEN_GEN(hvx_ssm_conv_token_gen_f32_f32,      false, false, false)
DEFINE_SSM_CONV_TOKEN_GEN(hvx_ssm_conv_token_gen_f32_f32_silu, false, false, true)
DEFINE_SSM_CONV_TOKEN_GEN(hvx_ssm_conv_token_gen_f32_f16,      false, true,  false)
DEFINE_SSM_CONV_TOKEN_GEN(hvx_ssm_conv_token_gen_f32_f16_silu, false, true,  true)
DEFINE_SSM_CONV_TOKEN_GEN(hvx_ssm_conv_token_gen_f16_f32,      true,  false, false)
DEFINE_SSM_CONV_TOKEN_GEN(hvx_ssm_conv_token_gen_f16_f32_silu, true,  false, true)
DEFINE_SSM_CONV_TOKEN_GEN(hvx_ssm_conv_token_gen_f16_f16,      true,  true,  false)
DEFINE_SSM_CONV_TOKEN_GEN(hvx_ssm_conv_token_gen_f16_f16_silu, true,  true,  true)

#undef DEFINE_SSM_CONV_TOKEN_GEN

// Single-token decode worker (n_t == 1)
static void ssm_conv_thread_f32_decode(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict src1 = octx->src[1];
    const struct htp_tensor * restrict dst  = octx->dst;

    dma_queue * dma_q = th->dma_q;

    const uint32_t d_conv   = kparams->d_conv;
    const uint32_t n_s      = kparams->n_s;
    const bool     src0_f16 = (src0->type == HTP_TYPE_F16);
    const bool     src1_f16 = (src1->type == HTP_TYPE_F16);
    const bool     dst_f16  = (dst->type  == HTP_TYPE_F16);
    const bool     raw_dot  = scctx->raw_dot;

    const htp_ssm_conv_decode_dot_fn_t decode_dot_fn = scctx->decode_dot_fn;
    const htp_ssm_conv_epilogue_fn_t   epilogue_fn   = scctx->epilogue_fn;

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

    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_src1) + ith * scctx->layout.src1_bytes_per_hvx;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_src0) + ith * scctx->layout.src0_bytes_per_hvx;
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_dst)  + ith * scctx->layout.dst_bytes_per_hvx;

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
        decode_dot_fn(src0_tile_raw[c & 1], src0_T, src1_raw, src1_T, out, tile_n, d_inner_stride, tile_off, d_conv);
        epilogue_fn(out, 1, tile_n);
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

    const uint32_t d_conv   = kparams->d_conv;
    const uint32_t n_t      = kparams->n_t;
    const uint32_t n_s      = kparams->n_s;
    const uint32_t ncs      = src0->ne[0];
    const bool     src0_f16 = (src0->type == HTP_TYPE_F16);
    const bool     src1_f16 = (src1->type == HTP_TYPE_F16);
    const bool     dst_f16  = (dst->type  == HTP_TYPE_F16);

    const htp_ssm_conv_transpose_block_fn_t transpose_block_fn = scctx->transpose_block_fn;
    const htp_ssm_conv_prefill_dot_fn_t     prefill_dot_fn     = scctx->prefill_dot_fn;
    const htp_ssm_conv_epilogue_fn_t        epilogue_fn        = scctx->epilogue_fn;

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

    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_src1) + ith * scctx->layout.src1_bytes_per_hvx;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_src0) + ith * scctx->layout.src0_bytes_per_hvx;
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_dst)  + ith * scctx->layout.dst_bytes_per_hvx;

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
            transpose_block_fn(raw_ch, src0_T, raw_stride_bytes, ncs, cb_n);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            const float * restrict wp = src1_T + tile_off + cb;
            float * restrict       op = out + cb;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
            prefill_dot_fn(src0_T, wp, op, n_t, d_conv, d_inner_stride, dst_tile_stride);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
        const size_t dst_vtcm_stride_bytes = epilogue_fn(out, n_t, tile_n);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);

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

// Fused token-major worker (n_t == 1 decode and n_t > 1 prefill)
static void ssm_conv_thread_token_major(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
    struct htp_ops_context *                  octx    = scctx->octx;
    const struct htp_ssm_conv_kernel_params * kparams = scctx->kparams;

    const struct htp_tensor * restrict state     = octx->src[0];
    const struct htp_tensor * restrict weight    = octx->src[1];
    const struct htp_tensor * restrict x         = octx->src[2];
    const struct htp_tensor * restrict dst       = octx->dsts[0];
    const struct htp_tensor * restrict dst_state = octx->dsts[1];

    dma_queue * dma_q = th->dma_q;

    const uint32_t d_conv     = kparams->d_conv;
    const uint32_t s_taps     = d_conv - 1;
    const uint32_t n_t        = kparams->n_t;
    const uint32_t n_s        = kparams->n_s;
    const uint32_t n_t_tile   = kparams->n_t_tile;
    const bool     state_f16     = (state->type     == HTP_TYPE_F16);
    const bool     weight_f16    = (weight->type    == HTP_TYPE_F16);
    const bool     x_f16         = (x->type         == HTP_TYPE_F16);
    const bool     dst_f16       = (dst->type       == HTP_TYPE_F16);
    const bool     dst_state_f16 = (dst_state && dst_state->type == HTP_TYPE_F16);

    const htp_ssm_conv_token_step_fn_t   token_step_fn   = scctx->token_step_fn;
    const htp_ssm_conv_state_unpack_fn_t state_unpack_fn = scctx->state_unpack_fn;
    const htp_ssm_conv_state_pack_fn_t   state_pack_fn   = scctx->state_pack_fn;

    const size_t state_elem_size     = state_f16     ? sizeof(__fp16) : sizeof(float);
    const size_t weight_elem_size    = weight_f16    ? sizeof(__fp16) : sizeof(float);
    const size_t x_elem_size         = x_f16         ? sizeof(__fp16) : sizeof(float);
    const size_t dst_elem_size       = dst_f16       ? sizeof(__fp16) : sizeof(float);
    const size_t dst_state_elem_size = dst_state_f16 ? sizeof(__fp16) : sizeof(float);

    const uint32_t dr  = scctx->nrows_per_hvx;
    const uint32_t ir0 = scctx->row_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, scctx->row_start + scctx->nrows);

    if (ir0 >= ir1) {
        return;
    }

    const uint32_t d_inner_per_hvx = ir1 - ir0;
    const uint32_t w_stride        = hex_round_up(d_inner_per_hvx, 64);
    const uint32_t d_inner_tile    = scctx->d_inner_tile;
    const uint32_t c_stride        = hex_round_up(d_inner_tile, 64);

    const size_t state_stride_seq_bytes     = state->nb[2];
    const size_t x_stride_token_bytes       = x->nb[1];
    const size_t x_stride_seq_bytes         = x->nb[2];
    const size_t dst_stride_token_bytes     = dst->nb[1];
    const size_t dst_stride_seq_bytes       = dst->nb[2];
    const size_t dst_state_stride_seq_bytes = dst_state ? dst_state->nb[1] : 0;

    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_src1) + ith * scctx->layout.src1_bytes_per_hvx;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_src0) + ith * scctx->layout.src0_bytes_per_hvx;
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, scctx->vtcm_base, scctx->layout.off_dst)  + ith * scctx->layout.dst_bytes_per_hvx;

    const size_t weight_dma_bytes = (size_t) d_inner_per_hvx * d_conv * weight_elem_size;
    const size_t weight_raw_size  = hex_round_up((size_t) w_stride * d_conv * sizeof(float), 128) + 128;

    float * src1_raw = (float *) src1_spad_base;
    float * src1_T   = (float *) (src1_spad_base + weight_raw_size);

    const size_t x_row_stride = hex_round_up((size_t) c_stride * x_elem_size, 128);
    const size_t x_tile_bytes = hex_round_up((size_t) n_t_tile * x_row_stride, 128);
    const size_t state_raw_sz = hex_round_up((size_t) c_stride * s_taps * sizeof(float), 128) + 128;
    const size_t d_row_stride = hex_round_up((size_t) c_stride * dst_elem_size, 128);
    const size_t d_tile_bytes = hex_round_up((size_t) n_t_tile * d_row_stride, 128);

    uint8_t * x_tile[2] = { src0_spad_base, src0_spad_base + x_tile_bytes };
    float *   state_raw = (float *) (src0_spad_base + 2 * x_tile_bytes);
    float *   hist      = (float *) ((uint8_t *) state_raw + state_raw_sz);
    uint8_t * d_tile[2] = { dst_spad_base, dst_spad_base + d_tile_bytes };
    float *   strip_f32 = (float *) (dst_spad_base + 2 * d_tile_bytes);

    struct htp_thread_trace * tr = th->trace;

    const dma_addr_t src1_ddr      = weight->data + (size_t) ir0 * d_conv * weight_elem_size;
    uint8_t *        src1_dma_dest = weight_f16 ? (uint8_t *) src1_T : (uint8_t *) src1_raw;
    dma_queue_push(dma_q, dma_make_data(src1_dma_dest, src1_ddr), weight_dma_bytes, weight_dma_bytes, weight_dma_bytes, 1);
    dma_queue_pop(dma_q);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
    if (weight_f16) {
        hvx_ssm_conv_widen_f16_vtcm((const __fp16 *) src1_T, src1_raw, d_inner_per_hvx * d_conv);
    }
    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_hvx, w_stride, d_conv);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);

    const uint32_t n_t_blocks = kparams->n_t_blocks;

    for (uint32_t seq = 0; seq < n_s; ++seq) {
        for (uint32_t tile_off = 0; tile_off < d_inner_per_hvx; tile_off += d_inner_tile) {
            const uint32_t cur_c      = MIN(d_inner_tile, d_inner_per_hvx - tile_off);
            const size_t   st_in_sz   = (size_t) cur_c * s_taps * state_elem_size;
            const dma_addr_t st_in_ddr = state->data + seq * state_stride_seq_bytes + (size_t) (ir0 + tile_off) * s_taps * state_elem_size;
            const size_t   x_width    = (size_t) cur_c * x_elem_size;
            const size_t   d_width    = (size_t) cur_c * dst_elem_size;
            const dma_addr_t x_base   = x->data + seq * x_stride_seq_bytes + (size_t) (ir0 + tile_off) * x_elem_size;
            const dma_addr_t d_base   = dst->data + seq * dst_stride_seq_bytes + (size_t) (ir0 + tile_off) * dst_elem_size;

            uint8_t * st_in_vtcm = state_f16 ? (uint8_t *) hist : (uint8_t *) state_raw;
            dma_queue_push(dma_q, dma_make_data(st_in_vtcm, st_in_ddr), st_in_sz, st_in_sz, st_in_sz, 1);

            const uint32_t cur_t0 = MIN(n_t_tile, n_t);
            dma_queue_push(dma_q, dma_make_data(dst->data, d_tile[0]), 0, 0, 0, 0);
            dma_queue_push(dma_q, dma_make_data(x_tile[0], x_base), x_row_stride, x_stride_token_bytes, x_width, cur_t0);
            if (n_t_blocks > 1) {
                const uint32_t cur_t1 = MIN(n_t_tile, n_t - n_t_tile);
                dma_queue_push(dma_q, dma_make_data(dst->data, d_tile[1]), 0, 0, 0, 0);
                dma_queue_push(dma_q, dma_make_data(x_tile[1], x_base + (size_t) n_t_tile * x_stride_token_bytes),
                               x_row_stride, x_stride_token_bytes, x_width, cur_t1);
            }

            dma_queue_pop(dma_q);  // state
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
            state_unpack_fn(state_raw, hist, cur_c, c_stride, s_taps);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

            const float * restrict wp = src1_T + tile_off;

            for (uint32_t tb = 0; tb < n_t_blocks; ++tb) {
                const uint32_t t0    = tb * n_t_tile;
                const uint32_t cur_t = MIN(n_t_tile, n_t - t0);

                dma_queue_pop(dma_q);  // writeback of block tb-2
                dma_queue_pop(dma_q);  // fetch of block tb

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) t0);
                token_step_fn(hist, x_tile[tb & 1], wp, d_tile[tb & 1], strip_f32,
                              cur_c, cur_t, d_conv, c_stride, w_stride, x_row_stride, d_row_stride);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) t0);

                dma_queue_push(dma_q, dma_make_data(d_base + (size_t) t0 * dst_stride_token_bytes, d_tile[tb & 1]),
                               dst_stride_token_bytes, d_row_stride, d_width, cur_t);

                if (tb + 2 < n_t_blocks) {
                    const uint32_t next_t0 = (tb + 2) * n_t_tile;
                    const uint32_t next_t  = MIN(n_t_tile, n_t - next_t0);
                    dma_queue_push(dma_q, dma_make_data(x_tile[tb & 1], x_base + (size_t) next_t0 * x_stride_token_bytes),
                                   x_row_stride, x_stride_token_bytes, x_width, next_t);
                }
            }

            if (dst_state) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
                state_pack_fn(hist, state_raw, cur_c, c_stride, s_taps);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);

                const size_t     st_out_sz   = (size_t) cur_c * s_taps * dst_state_elem_size;
                const dma_addr_t st_out_ddr  = dst_state->data + seq * dst_state_stride_seq_bytes + (size_t) (ir0 + tile_off) * s_taps * dst_state_elem_size;
                uint8_t *        st_out_vtcm = dst_state_f16 ? (uint8_t *) hist : (uint8_t *) state_raw;
                dma_queue_push(dma_q, dma_make_data(st_out_ddr, st_out_vtcm), st_out_sz, st_out_sz, st_out_sz, 1);
            }

            dma_queue_flush(dma_q);
        }
    }

    FARF(HIGH, "ssm-conv-token-major %d/%d: (%u:%u) d_conv=%u n_t=%u n_s=%u\n",
         ith, nth, ir0, ir1, d_conv, n_t, n_s);
}

int op_ssm_conv_f32(struct htp_ops_context * octx) {
    const struct htp_tensor * src0      = octx->src[0];
    const struct htp_tensor * src2      = octx->src[2];
    const struct htp_tensor * dst       = octx->dsts[0];
    const struct htp_tensor * dst_state = octx->dsts[1];

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
            elems_per_chunk,
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
    const size_t   dst_elem_size   = (dst->type  == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    struct htp_ssm_conv_vtcm_layout L;
    if (kparams->token_major) {
        const size_t x_elem_size = (src2->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
        htp_ssm_conv_token_vtcm_layout_build(&L, d_inner_per_hvx, kparams->d_conv, kparams->n_t_tile,
                                             kparams->d_inner_tile, n_hvx, x_elem_size, dst_elem_size);
    } else {
        htp_ssm_conv_vtcm_layout_build(&L, d_inner_per_hvx, kparams->d_conv, kparams->n_t,
                                       kparams->d_inner_tile, n_hvx, src0_elem_size);
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;

    const bool src0_f16   = (src0->type == HTP_TYPE_F16);
    const bool dst_f16    = (dst->type  == HTP_TYPE_F16);
    const bool apply_silu = kparams->silu != 0;

    if (kparams->token_major) {
        const bool x_f16      = (src2->type == HTP_TYPE_F16);
        const bool dst_st_f16 = (dst_state && dst_state->type == HTP_TYPE_F16);
        const bool is_d4      = (kparams->d_conv == 4);

        htp_ssm_conv_token_step_fn_t token_step_fn;
        if (is_d4) {
            if (x_f16 && dst_f16) {
                token_step_fn = apply_silu ? hvx_ssm_conv_token_4_f16_f16_silu : hvx_ssm_conv_token_4_f16_f16;
            } else if (x_f16 && !dst_f16) {
                token_step_fn = apply_silu ? hvx_ssm_conv_token_4_f16_f32_silu : hvx_ssm_conv_token_4_f16_f32;
            } else if (!x_f16 && dst_f16) {
                token_step_fn = apply_silu ? hvx_ssm_conv_token_4_f32_f16_silu : hvx_ssm_conv_token_4_f32_f16;
            } else {
                token_step_fn = apply_silu ? hvx_ssm_conv_token_4_f32_f32_silu : hvx_ssm_conv_token_4_f32_f32;
            }
        } else {
            if (x_f16 && dst_f16) {
                token_step_fn = apply_silu ? hvx_ssm_conv_token_gen_f16_f16_silu : hvx_ssm_conv_token_gen_f16_f16;
            } else if (x_f16 && !dst_f16) {
                token_step_fn = apply_silu ? hvx_ssm_conv_token_gen_f16_f32_silu : hvx_ssm_conv_token_gen_f16_f32;
            } else if (!x_f16 && dst_f16) {
                token_step_fn = apply_silu ? hvx_ssm_conv_token_gen_f32_f16_silu : hvx_ssm_conv_token_gen_f32_f16;
            } else {
                token_step_fn = apply_silu ? hvx_ssm_conv_token_gen_f32_f32_silu : hvx_ssm_conv_token_gen_f32_f32;
            }
        }

        const htp_ssm_conv_state_unpack_fn_t state_unpack_fn =
            src0_f16 ? ssm_conv_state_unpack_f16_nat : ssm_conv_state_unpack_f32_nat;

        const htp_ssm_conv_state_pack_fn_t state_pack_fn =
            dst_st_f16 ? ssm_conv_state_pack_f16_nat : ssm_conv_state_pack_f32_nat;

        struct htp_ssm_conv_context scctx = {
            .octx            = octx,
            .kparams         = kparams,
            .layout          = L,
            .vtcm_base       = base,
            .nrows_per_hvx   = d_inner_per_hvx,
            .d_inner_tile    = kparams->d_inner_tile,
            .row_start       = row_start,
            .nrows           = nrows,
            .token_step_fn   = token_step_fn,
            .state_unpack_fn = state_unpack_fn,
            .state_pack_fn   = state_pack_fn,
        };

        work_queue_run(octx->ctx->hvx_queue, ssm_conv_thread_token_major, &scctx, n_hvx);
        return HTP_STATUS_OK;
    }

    const bool raw_dot = (kparams->d_conv == 4) && (nrows % VLEN_FP32 == 0);

    htp_ssm_conv_decode_dot_fn_t decode_dot_fn;
    if (raw_dot) {
        decode_dot_fn = src0_f16 ? ssm_conv_decode_dot_4_f16 : ssm_conv_decode_dot_4_f32;
    } else {
        decode_dot_fn = src0_f16 ? ssm_conv_decode_dot_gen_f16 : ssm_conv_decode_dot_gen_f32;
    }

    htp_ssm_conv_epilogue_fn_t epilogue_fn;
    if (apply_silu) {
        epilogue_fn = dst_f16 ? ssm_conv_epilogue_silu_f16 : ssm_conv_epilogue_silu_f32;
    } else {
        epilogue_fn = dst_f16 ? ssm_conv_epilogue_f16 : ssm_conv_epilogue_f32;
    }

    struct htp_ssm_conv_context scctx = {
        .octx               = octx,
        .kparams            = kparams,
        .layout             = L,
        .vtcm_base          = base,
        .nrows_per_hvx      = d_inner_per_hvx,
        .d_inner_tile       = kparams->d_inner_tile,
        .row_start          = row_start,
        .nrows              = nrows,
        .raw_dot            = raw_dot,
        .decode_dot_fn      = decode_dot_fn,
        .transpose_block_fn = src0_f16 ? hvx_ssm_conv_transpose_block_f16 : hvx_ssm_conv_transpose_block,
        .prefill_dot_fn     = (kparams->d_conv == 4) ? hvx_ssm_conv_prefill_dot_block_4 : hvx_ssm_conv_prefill_dot_block_gen,
        .epilogue_fn        = epilogue_fn,
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
