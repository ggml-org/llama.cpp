#define GGML_COMMON_DECL_CPP
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-ifairy-lut-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif
#ifndef GGML_FP32_TO_BF16
#    define GGML_FP32_TO_BF16 ggml_fp32_to_bf16
#endif
#ifndef GGML_BF16_TO_FP32
#    define GGML_BF16_TO_FP32 ggml_bf16_to_fp32
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif

// iFairy LUT V2: lut_c-style 16-entry LUT + packed 16-row weight tiles.

static inline size_t ggml_ifairy_checked_mul_size(size_t a, size_t b) {
    GGML_ASSERT(a == 0 || b <= SIZE_MAX / a);
    return a * b;
}

static inline size_t ggml_ifairy_checked_add_size(size_t a, size_t b) {
    GGML_ASSERT(a <= SIZE_MAX - b);
    return a + b;
}

#if defined(__ARM_NEON) && defined(__aarch64__)
static inline int8x16_t ggml_ifairy_vbsl_s8(uint8x16_t mask, int8x16_t t, int8x16_t f) {
    return vreinterpretq_s8_u8(vbslq_u8(mask, vreinterpretq_u8_s8(t), vreinterpretq_u8_s8(f)));
}

static inline int8x16_t ggml_ifairy_neg_if(uint8x16_t mask, int8x16_t v) {
    return ggml_ifairy_vbsl_s8(mask, vnegq_s8(v), v);
}

// Decode per-lane 2-flag code (idx16+flags) into 4 int8 vectors using a 16-entry LUT (4x16B).
// This is a branchless formulation of the scalar decode in ggml_ifairy_lut_decode_lane_scalar().
static inline int8x16x4_t ggml_ifairy_lut_decode_16x3_3x1_with_lut(uint8x16_t  iweight_16x3,
                                                                   int8x16x4_t ilut,
                                                                   uint8x16_t  mask_idx,
                                                                   uint8x16_t  mask_b6,
                                                                   uint8x16_t  mask_b7) {
    const uint8x16_t index = vandq_u8(iweight_16x3, mask_idx);
    const uint8x16_t fl0   = vtstq_u8(iweight_16x3, mask_b6);
    const uint8x16_t fl1   = vtstq_u8(iweight_16x3, mask_b7);

    const uint8x16_t fl0_xor_fl1 = veorq_u8(fl0, fl1);
    const uint8x16_t not_fl0     = vmvnq_u8(fl0);

    const int8x16_t ac_00 = vqtbl1q_s8(ilut.val[0], index);
    const int8x16_t bd_01 = vqtbl1q_s8(ilut.val[1], index);
    const int8x16_t ac_11 = vqtbl1q_s8(ilut.val[2], index);
    const int8x16_t bd_11 = vqtbl1q_s8(ilut.val[3], index);

    int8x16x4_t out;
    // out0 (ac): (fl1 ? ac_11 : ac_00) * sign(fl0 xor fl1)
    out.val[0] = ggml_ifairy_neg_if(fl0_xor_fl1, ggml_ifairy_vbsl_s8(fl1, ac_11, ac_00));
    // out1 (bc): (fl1 ? ac_00 : ac_11) * sign(fl0)
    out.val[1] = ggml_ifairy_neg_if(fl0, ggml_ifairy_vbsl_s8(fl1, ac_00, ac_11));
    // out2 (ad, conj-folded): (fl1 ? bd_01 : bd_11) * sign(fl0 xor fl1)
    out.val[2] = ggml_ifairy_neg_if(fl0_xor_fl1, ggml_ifairy_vbsl_s8(fl1, bd_01, bd_11));
    // out3 (bd): (fl1 ? bd_11 : bd_01) * sign(!fl0)
    out.val[3] = ggml_ifairy_neg_if(not_fl0, ggml_ifairy_vbsl_s8(fl1, bd_11, bd_01));

    return out;
}

static inline float32x4_t ggml_ifairy_s16x4_to_f32(int16x4_t v) {
    return vcvtq_f32_s32(vmovl_s16(v));
}
#endif

static inline void ggml_ifairy_lut_decode_lane_scalar(const uint8_t  code,
                                                      const int8_t * tbl,
                                                      int8_t &       out0,
                                                      int8_t &       out1,
                                                      int8_t &       out2,
                                                      int8_t &       out3) {
    const uint8_t idx = code & 0x3f;
    const bool    fl0 = (code & 0x40u) != 0;
    const bool    fl1 = (code & 0x80u) != 0;

    const int8_t ac_00 = tbl[0 * 16 + idx];
    const int8_t bd_01 = tbl[1 * 16 + idx];
    const int8_t ac_11 = tbl[2 * 16 + idx];
    const int8_t bd_11 = tbl[3 * 16 + idx];

    const int8_t ac_01 = (int8_t) -ac_00;
    const int8_t ac_10 = (int8_t) -ac_11;
    const int8_t bd_00 = (int8_t) -bd_01;
    const int8_t bd_10 = (int8_t) -bd_11;

    const int8_t ac_l = fl0 ? ac_01 : ac_00;
    const int8_t ad_l = fl0 ? ac_10 : ac_11;
    const int8_t bc_l = fl0 ? bd_10 : bd_11;

    const int8_t ac_h = fl0 ? ac_11 : ac_10;
    const int8_t bc_h = fl0 ? bd_01 : bd_00;
    const int8_t bd_h = fl0 ? bd_11 : bd_10;

    out0 = fl1 ? ac_h : ac_l;
    out1 = fl1 ? ac_l : ad_l;
    out2 = fl1 ? bc_h : bc_l;
    out3 = fl1 ? bd_h : bc_h;
}

void ggml_ifairy_lut_qgemm_lut16(int          m,
                                 int          k,
                                 int          n,
                                 const void * packed_wtiles,
                                 const void * lut,
                                 const void * lut_scales,
                                 float *      dst,
                                 size_t       dst_col_stride,
                                 size_t       dst_row_stride,
                                 bool         pack_bf16,
                                 bool         add) {
    if (!packed_wtiles || !dst || !lut || !lut_scales) {
        return;
    }

    if (m <= 0 || k <= 0 || n <= 0) {
        return;
    }

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    const struct ifairy_lut_wtile_16 * wtiles = (const struct ifairy_lut_wtile_16 *) packed_wtiles;
    const int                          tiles  = (m + 15) / 16;

    // `add` is not used by the current ggml-cpu LUT route. Keep it correct but out of the hot path.
    if (add) {
        for (int col = 0; col < n; ++col) {
            const int8_t * lut_col =
                (const int8_t *) lut + (size_t) col * (size_t) groups * (size_t) k_ifairy_lut_group_bytes;
            const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;

            for (int row = 0; row < m; ++row) {
                const int       tile  = row >> 4;
                const int       lane  = row & 15;
                float           out_r = 0.0f;
                float           out_i = 0.0f;
                const uint8_t * old_base =
                    (const uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;

                if (pack_bf16) {
                    const ggml_bf16_t br = ((const ggml_bf16_t *) old_base)[0];
                    const ggml_bf16_t bi = ((const ggml_bf16_t *) old_base)[1];
                    out_r                = GGML_BF16_TO_FP32(br);
                    out_i                = GGML_BF16_TO_FP32(bi);
                } else {
                    const float * old_f = (const float *) old_base;
                    out_r               = old_f[0];
                    out_i               = old_f[1];
                }

                for (int64_t blk = 0; blk < blocks; ++blk) {
                    const struct ifairy_lut_wtile_16 * wt = wtiles + (size_t) tile * (size_t) blocks + (size_t) blk;

                    const float lr = scales[blk * 2 + 0];
                    const float li = scales[blk * 2 + 1];
                    const float wr = wt->d_real[lane];
                    const float wi = wt->d_imag[lane];

                    int sum_ac = 0;
                    int sum_bc = 0;
                    int sum_ad = 0;
                    int sum_bd = 0;

                    const int8_t * lut_blk =
                        lut_col + (size_t) blk * (size_t) groups_per_block * (size_t) k_ifairy_lut_group_bytes;
                    for (int gi = 0; gi < groups_per_block; ++gi) {
                        const uint8_t  code = wt->qs[gi][lane];
                        const int8_t * tbl  = lut_blk + (size_t) gi * (size_t) k_ifairy_lut_group_bytes;

                        int8_t v0 = 0;
                        int8_t v1 = 0;
                        int8_t v2 = 0;
                        int8_t v3 = 0;
                        ggml_ifairy_lut_decode_lane_scalar(code, tbl, v0, v1, v2, v3);
                        sum_ac += (int) v0;
                        sum_bc += (int) v1;
                        sum_ad += (int) v2;
                        sum_bd += (int) v3;
                    }

                    out_r += (float) sum_ac * (lr * wr) + (float) sum_bd * (li * wi);
                    out_i += (float) sum_bc * (lr * wi) + (float) sum_ad * (li * wr);
                }

                uint8_t * out_base = (uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
                if (pack_bf16) {
                    ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r);
                    ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i);
                } else {
                    ((float *) out_base)[0] = out_r;
                    ((float *) out_base)[1] = out_i;
                }
            }
        }
        return;
    }

#if defined(__ARM_NEON) && defined(__aarch64__)
    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col =
            (const int8_t *) lut + (size_t) col * (size_t) groups * (size_t) k_ifairy_lut_group_bytes;
        const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;

        const uint8x16_t mask_idx = vdupq_n_u8(0x3f);
        const uint8x16_t mask_b6  = vdupq_n_u8(0x40);
        const uint8x16_t mask_b7  = vdupq_n_u8(0x80);

        for (int t = 0; t < tiles; ++t) {
            const int rows_left = m - (t << 4);
            if (rows_left <= 0) {
                break;
            }
            const int rows_in_tile = rows_left >= 16 ? 16 : rows_left;

            float32x4_t acc_r0 = vdupq_n_f32(0.0f);
            float32x4_t acc_r1 = vdupq_n_f32(0.0f);
            float32x4_t acc_r2 = vdupq_n_f32(0.0f);
            float32x4_t acc_r3 = vdupq_n_f32(0.0f);

            float32x4_t acc_i0 = vdupq_n_f32(0.0f);
            float32x4_t acc_i1 = vdupq_n_f32(0.0f);
            float32x4_t acc_i2 = vdupq_n_f32(0.0f);
            float32x4_t acc_i3 = vdupq_n_f32(0.0f);

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const struct ifairy_lut_wtile_16 * wt = wtiles + (size_t) t * (size_t) blocks + (size_t) blk;

                int16x8_t sum_ac_0 = vdupq_n_s16(0);
                int16x8_t sum_ac_1 = vdupq_n_s16(0);
                int16x8_t sum_bc_0 = vdupq_n_s16(0);
                int16x8_t sum_bc_1 = vdupq_n_s16(0);
                int16x8_t sum_ad_0 = vdupq_n_s16(0);
                int16x8_t sum_ad_1 = vdupq_n_s16(0);
                int16x8_t sum_bd_0 = vdupq_n_s16(0);
                int16x8_t sum_bd_1 = vdupq_n_s16(0);

                const int8_t * lut_blk =
                    lut_col + (size_t) blk * (size_t) groups_per_block * (size_t) k_ifairy_lut_group_bytes;

                for (int gi = 0; gi < groups_per_block; ++gi) {
                    const uint8x16_t code = vld1q_u8(wt->qs[gi]);
                    const int8_t *   tbl  = lut_blk + (size_t) gi * (size_t) k_ifairy_lut_group_bytes;

                    // One 64B load for the 4x16B LUT vectors (AArch64: ld1 {v0-v3.16b}, [x]).
                    const int8x16x4_t ilut = vld1q_s8_x4(tbl);

                    const int8x16x4_t r =
                        ggml_ifairy_lut_decode_16x3_3x1_with_lut(code, ilut, mask_idx, mask_b6, mask_b7);

                    sum_ac_0 = vaddw_s8(sum_ac_0, vget_low_s8(r.val[0]));
                    sum_ac_1 = vaddw_s8(sum_ac_1, vget_high_s8(r.val[0]));

                    sum_bc_0 = vaddw_s8(sum_bc_0, vget_low_s8(r.val[1]));
                    sum_bc_1 = vaddw_s8(sum_bc_1, vget_high_s8(r.val[1]));

                    sum_ad_0 = vaddw_s8(sum_ad_0, vget_low_s8(r.val[2]));
                    sum_ad_1 = vaddw_s8(sum_ad_1, vget_high_s8(r.val[2]));

                    sum_bd_0 = vaddw_s8(sum_bd_0, vget_low_s8(r.val[3]));
                    sum_bd_1 = vaddw_s8(sum_bd_1, vget_high_s8(r.val[3]));
                }

                const float lr = scales[blk * 2 + 0];
                const float li = scales[blk * 2 + 1];

                const float32x4_t v_lr = vdupq_n_f32(lr);
                const float32x4_t v_li = vdupq_n_f32(li);

                // lanes 0..3
                {
                    const float32x4_t wr = vld1q_f32(wt->d_real + 0);
                    const float32x4_t wi = vld1q_f32(wt->d_imag + 0);

                    const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
                    const float32x4_t li_wi = vmulq_f32(v_li, wi);
                    const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
                    const float32x4_t li_wr = vmulq_f32(v_li, wr);

                    const float32x4_t v_ac = ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_ac_0));
                    const float32x4_t v_bc = ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_bc_0));
                    const float32x4_t v_ad = ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_ad_0));
                    const float32x4_t v_bd = ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_bd_0));

                    acc_r0 = vmlaq_f32(acc_r0, v_ac, lr_wr);
                    acc_r0 = vmlaq_f32(acc_r0, v_bd, li_wi);

                    acc_i0 = vmlaq_f32(acc_i0, v_bc, lr_wi);
                    acc_i0 = vmlaq_f32(acc_i0, v_ad, li_wr);
                }

                // lanes 4..7
                {
                    const float32x4_t wr = vld1q_f32(wt->d_real + 4);
                    const float32x4_t wi = vld1q_f32(wt->d_imag + 4);

                    const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
                    const float32x4_t li_wi = vmulq_f32(v_li, wi);
                    const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
                    const float32x4_t li_wr = vmulq_f32(v_li, wr);

                    const float32x4_t v_ac = ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_ac_0));
                    const float32x4_t v_bc = ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_bc_0));
                    const float32x4_t v_ad = ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_ad_0));
                    const float32x4_t v_bd = ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_bd_0));

                    acc_r1 = vmlaq_f32(acc_r1, v_ac, lr_wr);
                    acc_r1 = vmlaq_f32(acc_r1, v_bd, li_wi);

                    acc_i1 = vmlaq_f32(acc_i1, v_bc, lr_wi);
                    acc_i1 = vmlaq_f32(acc_i1, v_ad, li_wr);
                }

                // lanes 8..11
                {
                    const float32x4_t wr = vld1q_f32(wt->d_real + 8);
                    const float32x4_t wi = vld1q_f32(wt->d_imag + 8);

                    const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
                    const float32x4_t li_wi = vmulq_f32(v_li, wi);
                    const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
                    const float32x4_t li_wr = vmulq_f32(v_li, wr);

                    const float32x4_t v_ac = ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_ac_1));
                    const float32x4_t v_bc = ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_bc_1));
                    const float32x4_t v_ad = ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_ad_1));
                    const float32x4_t v_bd = ggml_ifairy_s16x4_to_f32(vget_low_s16(sum_bd_1));

                    acc_r2 = vmlaq_f32(acc_r2, v_ac, lr_wr);
                    acc_r2 = vmlaq_f32(acc_r2, v_bd, li_wi);

                    acc_i2 = vmlaq_f32(acc_i2, v_bc, lr_wi);
                    acc_i2 = vmlaq_f32(acc_i2, v_ad, li_wr);
                }

                // lanes 12..15
                {
                    const float32x4_t wr = vld1q_f32(wt->d_real + 12);
                    const float32x4_t wi = vld1q_f32(wt->d_imag + 12);

                    const float32x4_t lr_wr = vmulq_f32(v_lr, wr);
                    const float32x4_t li_wi = vmulq_f32(v_li, wi);
                    const float32x4_t lr_wi = vmulq_f32(v_lr, wi);
                    const float32x4_t li_wr = vmulq_f32(v_li, wr);

                    const float32x4_t v_ac = ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_ac_1));
                    const float32x4_t v_bc = ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_bc_1));
                    const float32x4_t v_ad = ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_ad_1));
                    const float32x4_t v_bd = ggml_ifairy_s16x4_to_f32(vget_high_s16(sum_bd_1));

                    acc_r3 = vmlaq_f32(acc_r3, v_ac, lr_wr);
                    acc_r3 = vmlaq_f32(acc_r3, v_bd, li_wi);

                    acc_i3 = vmlaq_f32(acc_i3, v_bc, lr_wi);
                    acc_i3 = vmlaq_f32(acc_i3, v_ad, li_wr);
                }
            }

            alignas(16) float out_r[16];
            alignas(16) float out_i[16];
            vst1q_f32(out_r + 0, acc_r0);
            vst1q_f32(out_r + 4, acc_r1);
            vst1q_f32(out_r + 8, acc_r2);
            vst1q_f32(out_r + 12, acc_r3);

            vst1q_f32(out_i + 0, acc_i0);
            vst1q_f32(out_i + 4, acc_i1);
            vst1q_f32(out_i + 8, acc_i2);
            vst1q_f32(out_i + 12, acc_i3);

            uint8_t * dst_col = (uint8_t *) dst + (size_t) col * dst_col_stride;
            for (int lane = 0; lane < rows_in_tile; ++lane) {
                uint8_t * out_base = dst_col + (size_t) ((t << 4) + lane) * dst_row_stride;
                if (pack_bf16) {
                    ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r[lane]);
                    ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i[lane]);
                } else {
                    ((float *) out_base)[0] = out_r[lane];
                    ((float *) out_base)[1] = out_i[lane];
                }
            }
        }
    }
    return;
#endif

    // Non-NEON (or non-aarch64): keep a scalar reference for portability and unit tests.
    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col =
            (const int8_t *) lut + (size_t) col * (size_t) groups * (size_t) k_ifairy_lut_group_bytes;
        const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2u;

        for (int row = 0; row < m; ++row) {
            const int tile = row >> 4;
            const int lane = row & 15;

            float out_r = 0.0f;
            float out_i = 0.0f;

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const struct ifairy_lut_wtile_16 * wt = wtiles + (size_t) tile * (size_t) blocks + (size_t) blk;

                const float lr = scales[blk * 2 + 0];
                const float li = scales[blk * 2 + 1];
                const float wr = wt->d_real[lane];
                const float wi = wt->d_imag[lane];

                int sum_ac = 0;
                int sum_bc = 0;
                int sum_ad = 0;
                int sum_bd = 0;

                const int8_t * lut_blk =
                    lut_col + (size_t) blk * (size_t) groups_per_block * (size_t) k_ifairy_lut_group_bytes;
                for (int gi = 0; gi < groups_per_block; ++gi) {
                    const uint8_t  code = wt->qs[gi][lane];
                    const int8_t * tbl  = lut_blk + (size_t) gi * (size_t) k_ifairy_lut_group_bytes;

                    int8_t v0 = 0;
                    int8_t v1 = 0;
                    int8_t v2 = 0;
                    int8_t v3 = 0;
                    ggml_ifairy_lut_decode_lane_scalar(code, tbl, v0, v1, v2, v3);
                    sum_ac += (int) v0;
                    sum_bc += (int) v1;
                    sum_ad += (int) v2;
                    sum_bd += (int) v3;
                }

                out_r += (float) sum_ac * (lr * wr) + (float) sum_bd * (li * wi);
                out_i += (float) sum_bc * (lr * wi) + (float) sum_ad * (li * wr);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ((ggml_bf16_t *) out_base)[0] = GGML_FP32_TO_BF16(out_r);
                ((ggml_bf16_t *) out_base)[1] = GGML_FP32_TO_BF16(out_i);
            } else {
                ((float *) out_base)[0] = out_r;
                ((float *) out_base)[1] = out_i;
            }
        }
    }
}

void ggml_ifairy_lut_mul_mat_scalar(int          m,
                                    int          k,
                                    int          n,
                                    const void * qweights,
                                    const void * act,
                                    size_t       act_stride,
                                    float *      dst) {
    if (!qweights || !act || !dst) {
        return;
    }

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    const size_t index_bytes_raw = (size_t) m * (size_t) groups;
    const size_t index_bytes     = GGML_PAD(index_bytes_raw, 64);

    const int64_t tiles        = (m + 15) / 16;
    const size_t  packed_bytes = (size_t) tiles * (size_t) blocks * sizeof(struct ifairy_lut_wtile_16);

    const size_t lut_bytes   = (size_t) n * (size_t) groups * (size_t) k_ifairy_lut_group_bytes;
    const size_t scale_bytes = (size_t) n * (size_t) blocks * 2u * sizeof(float);

    const size_t tmp0        = ggml_ifairy_checked_add_size(index_bytes, packed_bytes);
    const size_t tmp1        = ggml_ifairy_checked_add_size(tmp0, lut_bytes);
    const size_t total_bytes = ggml_ifairy_checked_add_size(tmp1, scale_bytes);

    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, total_bytes) != 0) {
        return;
    }

    uint8_t * buf      = (uint8_t *) ptr;
    uint8_t * indexes  = buf;
    uint8_t * packed_p = buf + index_bytes;
    uint8_t * lut_p    = packed_p + packed_bytes;
    float *   scales   = (float *) (lut_p + lut_bytes);

    memset(buf, 0, total_bytes);

    ggml_ifairy_3w_encode((const block_ifairy *) qweights, K, m, indexes, index_bytes_raw);

    // pat (6-bit) -> packed code byte (idx16 + flags for lut_c-style decode).
    // This mapping matches ggml's direct triplet encoding:
    //   pat = c0 | (c1<<2) | (c2<<4)
    static const uint8_t k_ifairy_pat_to_code_u8[64] = {
        0x00, 0x45, 0x8f, 0xca, 0x04, 0x41, 0x8b, 0xce, 0x08, 0x4d, 0x83, 0xc6, 0x0c, 0x49, 0x87, 0xc2,
        0x01, 0x44, 0x8e, 0xcb, 0x05, 0x40, 0x8a, 0xcf, 0x09, 0x4c, 0x82, 0xc7, 0x0d, 0x48, 0x86, 0xc3,
        0x02, 0x47, 0x8c, 0xc9, 0x06, 0x43, 0x88, 0xcd, 0x0a, 0x4f, 0x80, 0xc5, 0x0e, 0x4b, 0x84, 0xc1,
        0x03, 0x46, 0x8d, 0xc8, 0x07, 0x42, 0x89, 0xcc, 0x0b, 0x4e, 0x81, 0xc4, 0x0f, 0x4a, 0x85, 0xc0,
    };

    // Build packed 16-lane weights from per-row 6-bit patterns (lut_c-style).
    struct ifairy_lut_wtile_16 * packed_w = (struct ifairy_lut_wtile_16 *) packed_p;
    const block_ifairy *         w_blocks = (const block_ifairy *) qweights;

    for (int row = 0; row < m; ++row) {
        const int64_t tile = (int64_t) row >> 4;
        const int64_t lane = (int64_t) row & 15;

        const uint8_t * row_indexes = indexes + (size_t) row * (size_t) groups;

        for (int64_t blk = 0; blk < blocks; ++blk) {
            struct ifairy_lut_wtile_16 * t = packed_w + (size_t) tile * (size_t) blocks + (size_t) blk;

            const block_ifairy * wb = w_blocks + (size_t) row * (size_t) blocks + (size_t) blk;
            t->d_real[lane]         = GGML_FP16_TO_FP32(wb->d_real);
            t->d_imag[lane]         = GGML_FP16_TO_FP32(wb->d_imag);

            const uint8_t * blk_idx = row_indexes + (size_t) blk * (size_t) groups_per_block;
            for (int gi = 0; gi < groups_per_block; ++gi) {
                const uint8_t pat = blk_idx[gi] & 0x3fu;
                t->qs[gi][lane]   = k_ifairy_pat_to_code_u8[pat];
            }
        }
    }

    ggml_ifairy_lut_preprocess_ex_lut16(m, k, n, act, act_stride, scales, lut_p, 0, 1);

    const size_t dst_col_stride = (size_t) m * 2u * sizeof(float);
    const size_t dst_row_stride = 2u * sizeof(float);
    ggml_ifairy_lut_qgemm_lut16(m, k, n, packed_w, lut_p, scales, dst, dst_col_stride, dst_row_stride,
                                /*pack_bf16*/ false, /*add*/ false);

    free(ptr);
}
