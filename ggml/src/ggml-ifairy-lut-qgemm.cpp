#define GGML_COMMON_DECL_CPP
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-ifairy-lut-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif

// Prefetch is enabled by default; set GGML_IFAIRY_LUT_PREFETCH=0 to disable for tuning.
// Note: env is read once per process (cached) to avoid per-op getenv overhead.
static inline bool ggml_ifairy_lut_prefetch_enabled(void) {
    static std::atomic<int> cached(-1);  // -1=unset, 0=disabled, 1=enabled
    int                     v = cached.load(std::memory_order_relaxed);
    if (v >= 0) {
        return v != 0;
    }
    const char * env = getenv("GGML_IFAIRY_LUT_PREFETCH");
    v                = (env && strcmp(env, "0") == 0) ? 0 : 1;
    cached.store(v, std::memory_order_relaxed);
    return v != 0;
}

// Prefetch distance in groups; defaults to 2. Set GGML_IFAIRY_LUT_PREFETCH_DIST=0 to disable distance-based prefetch.
// Note: env is read once per process (cached) to avoid per-op getenv overhead.
static inline int ggml_ifairy_lut_prefetch_dist(void) {
    static std::atomic<int> cached(-1);  // -1=unset, else the prefetch distance
    int                     v = cached.load(std::memory_order_relaxed);
    if (v >= 0) {
        return v;
    }

    const char * env = getenv("GGML_IFAIRY_LUT_PREFETCH_DIST");
    if (!env || env[0] == '\0') {
        v = 2;
    } else {
        char *     end = NULL;
        const long val = strtol(env, &end, 10);
        if (end == env) {
            v = 2;
        } else if (val <= 0) {
            v = 0;
        } else if (val > 16) {
            v = 16;
        } else {
            v = (int) val;
        }
    }

    cached.store(v, std::memory_order_relaxed);
    return v;
}

// N==1 fast-path is enabled by default; set GGML_IFAIRY_LUT_N1_FASTPATH=0 to force the generic path (for tuning/regression checks).
// Note: env is read once per process (cached) to avoid per-token getenv overhead.
static inline bool ggml_ifairy_lut_n1_fastpath_enabled(void) {
    static std::atomic<int> cached(-1);  // -1=unset, 0=disabled, 1=enabled
    int                     v = cached.load(std::memory_order_relaxed);
    if (v >= 0) {
        return v != 0;
    }
    const char * env = getenv("GGML_IFAIRY_LUT_N1_FASTPATH");
    v                = (env && strcmp(env, "0") == 0) ? 0 : 1;
    cached.store(v, std::memory_order_relaxed);
    return v != 0;
}

// Unroll factor for the compact N==1 fast-path (decode tuning). Default is 4; set to 2 for A/B experiments.
// Note: env is read once per process (cached) to avoid per-token getenv overhead.
static inline int ggml_ifairy_lut_compact_n1_unroll(void) {
    static std::atomic<int> cached(-1);  // -1=unset, else the unroll factor
    int                     v = cached.load(std::memory_order_relaxed);
    if (v >= 0) {
        return v;
    }
    const char * env = getenv("GGML_IFAIRY_LUT_COMPACT_N1_UNROLL");
    v                = (env && strcmp(env, "2") == 0) ? 2 : 4;
    cached.store(v, std::memory_order_relaxed);
    return v;
}

static inline int ggml_ifairy_lut_u8_to_s8(uint8_t v) {
    // Sign-extend stored 8-bit value without relying on signed-char promotion.
    return (int) (v ^ 0x80u) - 0x80;
}

#ifndef __ARM_FEATURE_DOTPROD
static std::atomic<bool> g_ifairy_lut_warned_kernel_unavailable(false);
#endif
static std::atomic<bool> g_ifairy_lut_warned_kernel_unsupported(false);

#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
static const int8_t k_ifairy_lut_dot_mask_bytes[16] = {
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
};

static inline int32x4_t ggml_ifairy_lut_accum_dot(int32x4_t       acc,
                                                  const int32_t * t0,
                                                  uint8_t         c0,
                                                  const int32_t * t1,
                                                  uint8_t         c1,
                                                  const int32_t * t2,
                                                  uint8_t         c2,
                                                  const int8x16_t dot_mask) {
    const int32x4_t p0 = vdupq_n_s32(t0[c0]);
    const int32x4_t p1 = vdupq_n_s32(t1[c1]);
    const int32x4_t p2 = vdupq_n_s32(t2[c2]);

    acc = vdotq_s32(acc, vreinterpretq_s8_s32(p0), dot_mask);
    acc = vdotq_s32(acc, vreinterpretq_s8_s32(p1), dot_mask);
    acc = vdotq_s32(acc, vreinterpretq_s8_s32(p2), dot_mask);
    return acc;
}
#endif
static void ggml_ifairy_lut_qgemm_ex_legacy(int             m,
                                            int             k,
                                            int             n,
                                            const void *    qweights,
                                            const uint8_t * indexes,
                                            const void *    lut,
                                            const void *    lut_scales,
                                            const void *    act,
                                            size_t          act_stride,
                                            float *         dst,
                                            size_t          dst_col_stride,
                                            size_t          dst_row_stride,
                                            bool            pack_bf16,
                                            bool            strict,
                                            bool            add) {
    if (!indexes || !dst || !qweights || !lut || !lut_scales) {
        return;
    }
    if (strict) {
        GGML_ASSERT(add == false);
    }

    const int    prefetch_dist   = ggml_ifairy_lut_prefetch_dist();
    const bool   prefetch        = ggml_ifairy_lut_prefetch_enabled() && prefetch_dist > 0;
    const size_t prefetch_groups = prefetch ? (size_t) prefetch_dist : 0;

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    const block_ifairy * w_blocks = (const block_ifairy *) qweights;

#if 0
    // Fast-path for decode: N == 1 avoids the col loop and some pointer arithmetic.
    if (n == 1) {
        const int8_t * lut_base = (const int8_t *) lut;
        const float * scales = (const float *) lut_scales;
        const block_ifairy_q16 * act_blocks = act ? (const block_ifairy_q16 *) act : NULL;

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#    if defined(__ARM_NEON) && defined(__aarch64__)
            const bool want_dotprod = kernel == GGML_IFAIRY_LUT_KERNEL_SDOT;
#        if defined(__ARM_FEATURE_DOTPROD)
            const bool use_dotprod = want_dotprod;
            const int8x16_t dot_mask = vld1q_s8(k_ifairy_lut_dot_mask_bytes);
#        else
            const bool use_dotprod = false;
            if (want_dotprod && ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG") &&
                !g_ifairy_lut_warned_kernel_unavailable.exchange(true)) {
                GGML_LOG_WARN("ifairy_lut: GGML_IFAIRY_LUT_KERNEL=sdot requires __ARM_FEATURE_DOTPROD, using default kernel\n");
            }
#        endif
            float32x4_t accv = vdupq_n_f32(0.0f); // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                int64_t gi = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                    const uint8_t c00 = (uint8_t) (pat0 & 3);
                    const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                    const uint8_t c02 = (uint8_t) (pat0 >> 4);

                    const uint8_t c10 = (uint8_t) (pat1 & 3);
                    const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                    const uint8_t c12 = (uint8_t) (pat1 >> 4);

                    const uint8_t c20 = (uint8_t) (pat2 & 3);
                    const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                    const uint8_t c22 = (uint8_t) (pat2 >> 4);

                    const uint8_t c30 = (uint8_t) (pat3 & 3);
                    const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                    const uint8_t c32 = (uint8_t) (pat3 >> 4);

                    const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                    const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                    const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                    const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                    if (prefetch) {
                        __builtin_prefetch(grp0 + 4 * k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

#        if defined(__ARM_FEATURE_DOTPROD)
                    if (use_dotprod) {
                        isum0 = ggml_ifairy_lut_accum_dot(isum0, t00, c00, t01, c01, t02, c02, dot_mask);
                    } else
#        endif
                    {
                        const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                        const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                        const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                        int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                        s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                        s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                        isum0 = vaddw_s16(isum0, vget_low_s16(s160));
                    }

                    const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

#        if defined(__ARM_FEATURE_DOTPROD)
                    if (use_dotprod) {
                        isum1 = ggml_ifairy_lut_accum_dot(isum1, t10, c10, t11, c11, t12, c12, dot_mask);
                    } else
#        endif
                    {
                        const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                        const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                        const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                        int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                        s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                        s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                        isum1 = vaddw_s16(isum1, vget_low_s16(s161));
                    }

                    const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

#        if defined(__ARM_FEATURE_DOTPROD)
                    if (use_dotprod) {
                        isum0 = ggml_ifairy_lut_accum_dot(isum0, t20, c20, t21, c21, t22, c22, dot_mask);
                    } else
#        endif
                    {
                        const int32x2_t p20 = vld1_dup_s32(t20 + c20);
                        const int32x2_t p21 = vld1_dup_s32(t21 + c21);
                        const int32x2_t p22 = vld1_dup_s32(t22 + c22);

                        int16x8_t s162 = vmovl_s8(vreinterpret_s8_s32(p20));
                        s162 = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p21)));
                        s162 = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p22)));
                        isum0 = vaddw_s16(isum0, vget_low_s16(s162));
                    }

                    const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

#        if defined(__ARM_FEATURE_DOTPROD)
                    if (use_dotprod) {
                        isum1 = ggml_ifairy_lut_accum_dot(isum1, t30, c30, t31, c31, t32, c32, dot_mask);
                    } else
#        endif
                    {
                        const int32x2_t p30 = vld1_dup_s32(t30 + c30);
                        const int32x2_t p31 = vld1_dup_s32(t31 + c31);
                        const int32x2_t p32 = vld1_dup_s32(t32 + c32);

                        int16x8_t s163 = vmovl_s8(vreinterpret_s8_s32(p30));
                        s163 = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p31)));
                        s163 = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p32)));
                        isum1 = vaddw_s16(isum1, vget_low_s16(s163));
                    }

                    idx_g += 4;
                    grp   += 4 * k_ifairy_lut_group_bytes;
                }
                for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0 = (uint8_t) (pat & 3);
                    const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2 = (uint8_t) (pat >> 4);

                    if (prefetch) {
                        __builtin_prefetch(grp + k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

#        if defined(__ARM_FEATURE_DOTPROD)
                    if (use_dotprod) {
                        isum0 = ggml_ifairy_lut_accum_dot(isum0, t0, c0, t1, c1, t2, c2, dot_mask);
                    } else
#        endif
                    {
                        const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                        const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                        const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                        int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                        s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                        s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                        isum0 = vaddw_s16(isum0, vget_low_s16(s16));
                    }
                }

                const float32x2_t srsi = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv = vcombine_f32(srsi, srsi); // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#    else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                for (int64_t gi = 0; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0 = (uint8_t) (pat & 3);
                    const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2 = (uint8_t) (pat >> 4);

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int8_t * e0 = (const int8_t *) &t0[c0];
                    const int8_t * e1 = (const int8_t *) &t1[c1];
                    const int8_t * e2 = (const int8_t *) &t2[c2];

                    sum_ac += (int32_t) e0[0] + (int32_t) e1[0] + (int32_t) e2[0];
                    sum_ad += (int32_t) e0[1] + (int32_t) e1[1] + (int32_t) e2[1];
                    sum_bc += (int32_t) e0[2] + (int32_t) e1[2] + (int32_t) e2[2];
                    sum_bd += (int32_t) e0[3] + (int32_t) e1[3] + (int32_t) e2[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#    endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, 0, out_r, out_i);
            }

            if (strict) {
                GGML_ASSERT(act_blocks != NULL);
                double ref_ac_xr = 0.0;
                double ref_ad_xi = 0.0;
                double ref_bc_xr = 0.0;
                double ref_bd_xi = 0.0;

                for (int blk = 0; blk < (int) blocks; ++blk) {
                    const uint8_t * GGML_RESTRICT w_ptr   = w_row[blk].qs;
                    const uint8_t * GGML_RESTRICT x_r_ptr = act_blocks[blk].x_real;
                    const uint8_t * GGML_RESTRICT x_i_ptr = act_blocks[blk].x_imag;

                    int32_t sum_ac = 0;
                    int32_t sum_ad = 0;
                    int32_t sum_bc = 0;
                    int32_t sum_bd = 0;

                    for (int j = 0; j < QK_K; ++j) {
                        const int chunk    = j >> 6;
                        const int lane     = j & 0xF;
                        const int part     = (j >> 4) & 0x3;
                        const int byte_idx = (chunk << 4) + lane;
                        const int bit_off  = part * 2;

                        const uint8_t packed = w_ptr[byte_idx];
                        const uint8_t code   = (packed >> bit_off) & 0x3;

                        int wr = 0;
                        int wi = 0;
                        switch (code) {
                            case 0: wr = -1; wi =  0; break;
                            case 1: wr =  1; wi =  0; break;
                            case 2: wr =  0; wi = -1; break;
                            case 3: wr =  0; wi =  1; break;
                            default:
                                GGML_ASSERT(false);
                                break;
                        }

                        const int xr = ggml_ifairy_lut_u8_to_s8(x_r_ptr[j]);
                        const int xi = ggml_ifairy_lut_u8_to_s8(x_i_ptr[j]);

                        sum_ac += xr * wr;
                        sum_ad += xi * wr;
                        sum_bc += xr * wi;
                        sum_bd += xi * wi;
                    }

                    const double x_real = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                    const double x_imag = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_imag);

                    ref_ac_xr += x_real * (double) sum_ac;
                    ref_ad_xi += x_imag * (double) sum_ad;
                    ref_bc_xr += x_real * (double) sum_bc;
                    ref_bd_xi += x_imag * (double) sum_bd;
                }

                const double ref_r = (double) coeff_w_real * ref_ac_xr + (double) coeff_w_imag * ref_bd_xi;
                const double ref_i = (double) coeff_w_imag * ref_bc_xr - (double) coeff_w_real * ref_ad_xi;

                const float dr = out_r - (float) ref_r;
                const float di = out_i - (float) ref_i;
                GGML_ASSERT(fabsf(dr) <= 1e-3f && fabsf(di) <= 1e-3f);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
        return;
    }
#endif

    // Fast-path for decode: N == 1 avoids the col loop and some pointer arithmetic.
    // Keep strict mode on the generic path (strict validation assumes the generic structure).
    if (n == 1 && !strict) {
        const int16_t *          lut_base   = (const int16_t *) lut;
        const float *            scales     = (const float *) lut_scales;
        const block_ifairy_q16 * act_blocks = act ? (const block_ifairy_q16 *) act : NULL;

        const size_t group_stride = (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row   = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t *      idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t accv = vdupq_n_f32(0.0f);  // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block * group_stride;

                int64_t gi = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                    const int16_t * grp0 = lut_blk + (size_t) (gi + 0) * group_stride;
                    const int16_t * grp1 = lut_blk + (size_t) (gi + 1) * group_stride;
                    const int16_t * grp2 = lut_blk + (size_t) (gi + 2) * group_stride;
                    const int16_t * grp3 = lut_blk + (size_t) (gi + 3) * group_stride;

                    if (prefetch) {
                        const size_t prefetch_stride = prefetch_groups * group_stride;
                        __builtin_prefetch(idx_blk + gi + prefetch_groups, 0, 1);
                        __builtin_prefetch(grp0 + prefetch_stride, 0, 1);
                        __builtin_prefetch(grp1 + prefetch_stride, 0, 1);
                    }

                    const int16_t * tbl0 = grp0 + (size_t) pat0 * k_ifairy_lut_channels;
                    const int16_t * tbl1 = grp1 + (size_t) pat1 * k_ifairy_lut_channels;
                    const int16_t * tbl2 = grp2 + (size_t) pat2 * k_ifairy_lut_channels;
                    const int16_t * tbl3 = grp3 + (size_t) pat3 * k_ifairy_lut_channels;

                    const int16x4_t s0 = vld1_s16(tbl0);
                    const int16x4_t s1 = vld1_s16(tbl1);
                    const int16x4_t s2 = vld1_s16(tbl2);
                    const int16x4_t s3 = vld1_s16(tbl3);

                    isum0 = vaddw_s16(isum0, s0);
                    isum1 = vaddw_s16(isum1, s1);
                    isum0 = vaddw_s16(isum0, s2);
                    isum1 = vaddw_s16(isum1, s3);
                }
                for (; gi < groups_per_block; ++gi) {
                    const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int16_t * tbl = lut_blk + (size_t) gi * group_stride + (size_t) pat * k_ifairy_lut_channels;
                    const int16x4_t sums16 = vld1_s16(tbl);
                    isum0                  = vaddw_s16(isum0, sums16);
                }

                const float32x2_t srsi  = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv   = vcombine_f32(srsi, srsi);  // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv                    = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block * group_stride;
                for (int64_t gi = 0; gi < groups_per_block; ++gi) {
                    const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int16_t * tbl = lut_blk + (size_t) gi * group_stride + (size_t) pat * k_ifairy_lut_channels;
                    sum_ac += (int32_t) tbl[0];
                    sum_ad += (int32_t) tbl[1];
                    sum_bc += (int32_t) tbl[2];
                    sum_bd += (int32_t) tbl[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, 0, out_r, out_i);
            }

            (void) act_blocks;

            uint8_t * out_base = (uint8_t *) dst + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br                = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi                = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
        return;
    }

    for (int row = 0; row < m; ++row) {
        const block_ifairy * w_row   = w_blocks + (size_t) row * (size_t) blocks;
        const uint8_t *      idx_row = indexes + (size_t) row * (size_t) groups;

        const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
        const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

        for (int col = 0; col < n; ++col) {
            const int16_t * lut_base =
                (const int16_t *) ((const uint8_t *) lut +
                                   (size_t) col * (size_t) groups *
                                       (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) * sizeof(int16_t));
            const float *            scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;
            const block_ifairy_q16 * act_blocks =
                act ? (const block_ifairy_q16 *) ((const uint8_t *) act + (size_t) col * act_stride) : NULL;

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t accv = vdupq_n_f32(0.0f);  // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block *
                                                         (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

                const size_t group_stride = (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);
                int64_t      gi           = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                    const int16_t * grp0 = lut_blk + (size_t) (gi + 0) * group_stride;
                    const int16_t * grp1 = lut_blk + (size_t) (gi + 1) * group_stride;
                    const int16_t * grp2 = lut_blk + (size_t) (gi + 2) * group_stride;
                    const int16_t * grp3 = lut_blk + (size_t) (gi + 3) * group_stride;

                    if (prefetch) {
                        const size_t prefetch_stride = prefetch_groups * group_stride;
                        __builtin_prefetch(idx_blk + gi + prefetch_groups, 0, 1);
                        __builtin_prefetch(grp0 + prefetch_stride, 0, 1);
                        __builtin_prefetch(grp1 + prefetch_stride, 0, 1);
                    }

                    const int16_t * tbl0 = grp0 + (size_t) pat0 * k_ifairy_lut_channels;
                    const int16_t * tbl1 = grp1 + (size_t) pat1 * k_ifairy_lut_channels;
                    const int16_t * tbl2 = grp2 + (size_t) pat2 * k_ifairy_lut_channels;
                    const int16_t * tbl3 = grp3 + (size_t) pat3 * k_ifairy_lut_channels;

                    const int16x4_t s0 = vld1_s16(tbl0);
                    const int16x4_t s1 = vld1_s16(tbl1);
                    const int16x4_t s2 = vld1_s16(tbl2);
                    const int16x4_t s3 = vld1_s16(tbl3);

                    isum0 = vaddw_s16(isum0, s0);
                    isum1 = vaddw_s16(isum1, s1);
                    isum0 = vaddw_s16(isum0, s2);
                    isum1 = vaddw_s16(isum1, s3);
                }
                for (; gi < groups_per_block; ++gi) {
                    const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int16_t * tbl = lut_blk + (size_t) gi * group_stride + (size_t) pat * k_ifairy_lut_channels;
                    const int16x4_t sums16 = vld1_s16(tbl);
                    isum0                  = vaddw_s16(isum0, sums16);
                }

                const float32x2_t srsi  = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv   = vcombine_f32(srsi, srsi);  // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv                    = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block *
                                                         (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

                for (int64_t gi = 0; gi < groups_per_block; ++gi) {
                    const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int16_t * tbl = lut_blk +
                                          (size_t) gi * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels) +
                                          (size_t) pat * k_ifairy_lut_channels;
                    sum_ac += (int32_t) tbl[0];
                    sum_ad += (int32_t) tbl[1];
                    sum_bc += (int32_t) tbl[2];
                    sum_bd += (int32_t) tbl[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, col, out_r, out_i);
            }

            if (strict) {
                GGML_ASSERT(act_blocks != NULL);
                double ref_ac_xr = 0.0;
                double ref_ad_xi = 0.0;
                double ref_bc_xr = 0.0;
                double ref_bd_xi = 0.0;

                for (int64_t blk = 0; blk < blocks; ++blk) {
                    int32_t sum_ac = 0;
                    int32_t sum_ad = 0;
                    int32_t sum_bc = 0;
                    int32_t sum_bd = 0;

                    const uint8_t * w_ptr   = w_row[blk].qs;
                    const uint8_t * x_r_ptr = act_blocks[blk].x_real;
                    const uint8_t * x_i_ptr = act_blocks[blk].x_imag;

                    for (int j = 0; j < QK_K; ++j) {
                        const int chunk    = j >> 6;
                        const int lane     = j & 0xF;
                        const int part     = (j >> 4) & 0x3;
                        const int byte_idx = (chunk << 4) + lane;
                        const int bit_off  = part * 2;

                        const uint8_t packed = w_ptr[byte_idx];
                        const uint8_t code   = (packed >> bit_off) & 0x3;

                        int wr = 0;
                        int wi = 0;
                        switch (code) {
                            case 0:
                                wr = -1;
                                wi = 0;
                                break;
                            case 1:
                                wr = 1;
                                wi = 0;
                                break;
                            case 2:
                                wr = 0;
                                wi = -1;
                                break;
                            case 3:
                                wr = 0;
                                wi = 1;
                                break;
                            default:
                                GGML_UNREACHABLE();
                        }

                        const int xr = ggml_ifairy_lut_u8_to_s8(x_r_ptr[j]);
                        const int xi = ggml_ifairy_lut_u8_to_s8(x_i_ptr[j]);

                        sum_ac += xr * wr;
                        sum_ad += xi * wr;
                        sum_bc += xr * wi;
                        sum_bd += xi * wi;
                    }

                    const double x_real = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                    const double x_imag = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_imag);

                    ref_ac_xr += x_real * (double) sum_ac;
                    ref_ad_xi += x_imag * (double) sum_ad;
                    ref_bc_xr += x_real * (double) sum_bc;
                    ref_bd_xi += x_imag * (double) sum_bd;
                }

                const double ref_r = (double) coeff_w_real * ref_ac_xr + (double) coeff_w_imag * ref_bd_xi;
                const double ref_i = (double) coeff_w_imag * ref_bc_xr - (double) coeff_w_real * ref_ad_xi;

                const float dr = out_r - (float) ref_r;
                const float di = out_i - (float) ref_i;
                GGML_ASSERT(fabsf(dr) <= 1e-3f && fabsf(di) <= 1e-3f);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br                = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi                = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
    }
}

// NOLINTNEXTLINE(readability-function-size)
void ggml_ifairy_lut_qgemm_ex(int             m,
                              int             k,
                              int             n,
                              const void *    qweights,
                              const uint8_t * indexes,
                              const void *    lut,
                              const void *    lut_scales,
                              const void *    act,
                              size_t          act_stride,
                              float *         dst,
                              size_t          dst_col_stride,
                              size_t          dst_row_stride,
                              bool            pack_bf16,
                              bool            strict,
                              bool            add) {
    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env(n);
    if (layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY) {
        ggml_ifairy_lut_qgemm_ex_legacy(m, k, n, qweights, indexes, lut, lut_scales, act, act_stride, dst,
                                        dst_col_stride, dst_row_stride, pack_bf16, strict, add);
        return;
    }

    if (!indexes || !dst || !qweights || !lut || !lut_scales) {
        return;
    }
    if (strict) {
        GGML_ASSERT(add == false);
    }

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    const block_ifairy *         w_blocks      = (const block_ifairy *) qweights;
    const int                    prefetch_dist = ggml_ifairy_lut_prefetch_dist();
    const bool                   prefetch      = ggml_ifairy_lut_prefetch_enabled() && prefetch_dist > 0;
    const ggml_ifairy_lut_kernel kernel        = ggml_ifairy_lut_kernel_from_env();

    if ((kernel == GGML_IFAIRY_LUT_KERNEL_TBL || kernel == GGML_IFAIRY_LUT_KERNEL_MERGED64) &&
        ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG") && !g_ifairy_lut_warned_kernel_unsupported.exchange(true)) {
        const char * kernel_name = kernel == GGML_IFAIRY_LUT_KERNEL_TBL ? "tbl" : "merged64";
        GGML_LOG_WARN("ifairy_lut: GGML_IFAIRY_LUT_KERNEL=%s not implemented yet, using default kernel\n", kernel_name);
    }

    // Fast-path for decode: N == 1 avoids the col loop and some pointer arithmetic.
    // Keep strict mode on the generic path (strict validation assumes the generic structure).
    if (n == 1 && !strict && ggml_ifairy_lut_n1_fastpath_enabled()) {
        const int8_t * lut_base = (const int8_t *) lut;
        const float *  scales   = (const float *) lut_scales;
        (void) act;

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row   = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t *      idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            const bool want_dotprod = kernel == GGML_IFAIRY_LUT_KERNEL_SDOT;
#    ifdef __ARM_FEATURE_DOTPROD
            const bool      use_dotprod = want_dotprod;
            const int8x16_t dot_mask    = vld1q_s8(k_ifairy_lut_dot_mask_bytes);
#    else
            if (want_dotprod && ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG") &&
                !g_ifairy_lut_warned_kernel_unavailable.exchange(true)) {
                GGML_LOG_WARN(
                    "ifairy_lut: GGML_IFAIRY_LUT_KERNEL=sdot requires __ARM_FEATURE_DOTPROD, using default kernel\n");
            }
#    endif
            float32x4_t accv = vdupq_n_f32(0.0f);  // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t *  grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                int64_t   gi     = 0;
                const int unroll = ggml_ifairy_lut_compact_n1_unroll();
#    ifdef __ARM_FEATURE_DOTPROD
                if (use_dotprod) {
                    if (prefetch) {
                        const size_t prefetch_groups4 = (size_t) prefetch_dist * 4u;
                        const size_t prefetch_groups2 = (size_t) prefetch_dist * 2u;
                        const size_t prefetch_groups1 = (size_t) prefetch_dist;
                        const size_t prefetch_bytes4  = prefetch_groups4 * k_ifairy_lut_group_bytes;
                        const size_t prefetch_bytes2  = prefetch_groups2 * k_ifairy_lut_group_bytes;
                        const size_t prefetch_bytes1  = prefetch_groups1 * k_ifairy_lut_group_bytes;

                        for (; unroll >= 4 && gi + 3 < groups_per_block; gi += 4) {
                            const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                            const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                            const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                            const uint8_t c00 = (uint8_t) (pat0 & 3);
                            const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                            const uint8_t c02 = (uint8_t) (pat0 >> 4);

                            const uint8_t c10 = (uint8_t) (pat1 & 3);
                            const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                            const uint8_t c12 = (uint8_t) (pat1 >> 4);

                            const uint8_t c20 = (uint8_t) (pat2 & 3);
                            const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                            const uint8_t c22 = (uint8_t) (pat2 >> 4);

                            const uint8_t c30 = (uint8_t) (pat3 & 3);
                            const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                            const uint8_t c32 = (uint8_t) (pat3 >> 4);

                            const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                            const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                            const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                            const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                            __builtin_prefetch(idx_g + prefetch_groups4, 0, 1);
                            __builtin_prefetch(grp0 + prefetch_bytes4, 0, 1);

                            const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                            isum0 = ggml_ifairy_lut_accum_dot(isum0, t00, c00, t01, c01, t02, c02, dot_mask);

                            const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                            isum1 = ggml_ifairy_lut_accum_dot(isum1, t10, c10, t11, c11, t12, c12, dot_mask);

                            const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

                            isum0 = ggml_ifairy_lut_accum_dot(isum0, t20, c20, t21, c21, t22, c22, dot_mask);

                            const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

                            isum1 = ggml_ifairy_lut_accum_dot(isum1, t30, c30, t31, c31, t32, c32, dot_mask);

                            idx_g += 4;
                            grp += 4 * k_ifairy_lut_group_bytes;
                        }
                        for (; gi + 1 < groups_per_block; gi += 2) {
                            const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                            const uint8_t c00 = (uint8_t) (pat0 & 3);
                            const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                            const uint8_t c02 = (uint8_t) (pat0 >> 4);

                            const uint8_t c10 = (uint8_t) (pat1 & 3);
                            const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                            const uint8_t c12 = (uint8_t) (pat1 >> 4);

                            const int8_t * grp0 = grp;
                            const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                            __builtin_prefetch(idx_g + prefetch_groups2, 0, 1);
                            __builtin_prefetch(grp0 + prefetch_bytes2, 0, 1);

                            const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                            isum0 = ggml_ifairy_lut_accum_dot(isum0, t00, c00, t01, c01, t02, c02, dot_mask);

                            const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                            isum1 = ggml_ifairy_lut_accum_dot(isum1, t10, c10, t11, c11, t12, c12, dot_mask);

                            idx_g += 2;
                            grp += 2 * k_ifairy_lut_group_bytes;
                        }
                        for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                            const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                            const uint8_t c0  = (uint8_t) (pat & 3);
                            const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                            const uint8_t c2  = (uint8_t) (pat >> 4);

                            __builtin_prefetch(idx_g + prefetch_groups1, 0, 1);
                            __builtin_prefetch(grp + prefetch_bytes1, 0, 1);

                            const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                            isum0 = ggml_ifairy_lut_accum_dot(isum0, t0, c0, t1, c1, t2, c2, dot_mask);
                        }
                    } else {
                        for (; unroll >= 4 && gi + 3 < groups_per_block; gi += 4) {
                            const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                            const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                            const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                            const uint8_t c00 = (uint8_t) (pat0 & 3);
                            const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                            const uint8_t c02 = (uint8_t) (pat0 >> 4);

                            const uint8_t c10 = (uint8_t) (pat1 & 3);
                            const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                            const uint8_t c12 = (uint8_t) (pat1 >> 4);

                            const uint8_t c20 = (uint8_t) (pat2 & 3);
                            const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                            const uint8_t c22 = (uint8_t) (pat2 >> 4);

                            const uint8_t c30 = (uint8_t) (pat3 & 3);
                            const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                            const uint8_t c32 = (uint8_t) (pat3 >> 4);

                            const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                            const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                            const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                            const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                            const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                            isum0 = ggml_ifairy_lut_accum_dot(isum0, t00, c00, t01, c01, t02, c02, dot_mask);

                            const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                            isum1 = ggml_ifairy_lut_accum_dot(isum1, t10, c10, t11, c11, t12, c12, dot_mask);

                            const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

                            isum0 = ggml_ifairy_lut_accum_dot(isum0, t20, c20, t21, c21, t22, c22, dot_mask);

                            const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

                            isum1 = ggml_ifairy_lut_accum_dot(isum1, t30, c30, t31, c31, t32, c32, dot_mask);

                            idx_g += 4;
                            grp += 4 * k_ifairy_lut_group_bytes;
                        }
                        for (; gi + 1 < groups_per_block; gi += 2) {
                            const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                            const uint8_t c00 = (uint8_t) (pat0 & 3);
                            const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                            const uint8_t c02 = (uint8_t) (pat0 >> 4);

                            const uint8_t c10 = (uint8_t) (pat1 & 3);
                            const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                            const uint8_t c12 = (uint8_t) (pat1 >> 4);

                            const int8_t * grp0 = grp;
                            const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                            const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                            isum0 = ggml_ifairy_lut_accum_dot(isum0, t00, c00, t01, c01, t02, c02, dot_mask);

                            const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                            isum1 = ggml_ifairy_lut_accum_dot(isum1, t10, c10, t11, c11, t12, c12, dot_mask);

                            idx_g += 2;
                            grp += 2 * k_ifairy_lut_group_bytes;
                        }
                        for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                            const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                            const uint8_t c0  = (uint8_t) (pat & 3);
                            const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                            const uint8_t c2  = (uint8_t) (pat >> 4);

                            const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                            isum0 = ggml_ifairy_lut_accum_dot(isum0, t0, c0, t1, c1, t2, c2, dot_mask);
                        }
                    }
                } else
#    endif
                {
                    if (prefetch) {
                        const size_t prefetch_groups4 = (size_t) prefetch_dist * 4u;
                        const size_t prefetch_groups2 = (size_t) prefetch_dist * 2u;
                        const size_t prefetch_groups1 = (size_t) prefetch_dist;
                        const size_t prefetch_bytes4  = prefetch_groups4 * k_ifairy_lut_group_bytes;
                        const size_t prefetch_bytes2  = prefetch_groups2 * k_ifairy_lut_group_bytes;
                        const size_t prefetch_bytes1  = prefetch_groups1 * k_ifairy_lut_group_bytes;

                        for (; unroll >= 4 && gi + 3 < groups_per_block; gi += 4) {
                            const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                            const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                            const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                            const uint8_t c00 = (uint8_t) (pat0 & 3);
                            const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                            const uint8_t c02 = (uint8_t) (pat0 >> 4);

                            const uint8_t c10 = (uint8_t) (pat1 & 3);
                            const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                            const uint8_t c12 = (uint8_t) (pat1 >> 4);

                            const uint8_t c20 = (uint8_t) (pat2 & 3);
                            const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                            const uint8_t c22 = (uint8_t) (pat2 >> 4);

                            const uint8_t c30 = (uint8_t) (pat3 & 3);
                            const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                            const uint8_t c32 = (uint8_t) (pat3 >> 4);

                            const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                            const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                            const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                            const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                            __builtin_prefetch(idx_g + prefetch_groups4, 0, 1);
                            __builtin_prefetch(grp0 + prefetch_bytes4, 0, 1);

                            const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                            const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                            const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                            int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                            s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                            s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                            isum0          = vaddw_s16(isum0, vget_low_s16(s160));

                            const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                            const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                            const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                            int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                            s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                            s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                            isum1          = vaddw_s16(isum1, vget_low_s16(s161));

                            const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p20 = vld1_dup_s32(t20 + c20);
                            const int32x2_t p21 = vld1_dup_s32(t21 + c21);
                            const int32x2_t p22 = vld1_dup_s32(t22 + c22);

                            int16x8_t s162 = vmovl_s8(vreinterpret_s8_s32(p20));
                            s162           = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p21)));
                            s162           = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p22)));
                            isum0          = vaddw_s16(isum0, vget_low_s16(s162));

                            const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p30 = vld1_dup_s32(t30 + c30);
                            const int32x2_t p31 = vld1_dup_s32(t31 + c31);
                            const int32x2_t p32 = vld1_dup_s32(t32 + c32);

                            int16x8_t s163 = vmovl_s8(vreinterpret_s8_s32(p30));
                            s163           = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p31)));
                            s163           = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p32)));
                            isum1          = vaddw_s16(isum1, vget_low_s16(s163));

                            idx_g += 4;
                            grp += 4 * k_ifairy_lut_group_bytes;
                        }
                        for (; gi + 1 < groups_per_block; gi += 2) {
                            const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                            const uint8_t c00 = (uint8_t) (pat0 & 3);
                            const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                            const uint8_t c02 = (uint8_t) (pat0 >> 4);

                            const uint8_t c10 = (uint8_t) (pat1 & 3);
                            const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                            const uint8_t c12 = (uint8_t) (pat1 >> 4);

                            const int8_t * grp0 = grp;
                            const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                            __builtin_prefetch(idx_g + prefetch_groups2, 0, 1);
                            __builtin_prefetch(grp0 + prefetch_bytes2, 0, 1);

                            const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                            const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                            const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                            int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                            s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                            s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                            isum0          = vaddw_s16(isum0, vget_low_s16(s160));

                            const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                            const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                            const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                            int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                            s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                            s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                            isum1          = vaddw_s16(isum1, vget_low_s16(s161));

                            idx_g += 2;
                            grp += 2 * k_ifairy_lut_group_bytes;
                        }
                        for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                            const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                            const uint8_t c0  = (uint8_t) (pat & 3);
                            const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                            const uint8_t c2  = (uint8_t) (pat >> 4);

                            __builtin_prefetch(idx_g + prefetch_groups1, 0, 1);
                            __builtin_prefetch(grp + prefetch_bytes1, 0, 1);

                            const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                            const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                            const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                            int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                            s16           = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                            s16           = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                            isum0         = vaddw_s16(isum0, vget_low_s16(s16));
                        }
                    } else {
                        for (; unroll >= 4 && gi + 3 < groups_per_block; gi += 4) {
                            const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                            const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                            const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                            const uint8_t c00 = (uint8_t) (pat0 & 3);
                            const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                            const uint8_t c02 = (uint8_t) (pat0 >> 4);

                            const uint8_t c10 = (uint8_t) (pat1 & 3);
                            const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                            const uint8_t c12 = (uint8_t) (pat1 >> 4);

                            const uint8_t c20 = (uint8_t) (pat2 & 3);
                            const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                            const uint8_t c22 = (uint8_t) (pat2 >> 4);

                            const uint8_t c30 = (uint8_t) (pat3 & 3);
                            const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                            const uint8_t c32 = (uint8_t) (pat3 >> 4);

                            const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                            const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                            const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                            const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                            const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                            const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                            const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                            int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                            s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                            s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                            isum0          = vaddw_s16(isum0, vget_low_s16(s160));

                            const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                            const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                            const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                            int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                            s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                            s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                            isum1          = vaddw_s16(isum1, vget_low_s16(s161));

                            const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p20 = vld1_dup_s32(t20 + c20);
                            const int32x2_t p21 = vld1_dup_s32(t21 + c21);
                            const int32x2_t p22 = vld1_dup_s32(t22 + c22);

                            int16x8_t s162 = vmovl_s8(vreinterpret_s8_s32(p20));
                            s162           = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p21)));
                            s162           = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p22)));
                            isum0          = vaddw_s16(isum0, vget_low_s16(s162));

                            const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p30 = vld1_dup_s32(t30 + c30);
                            const int32x2_t p31 = vld1_dup_s32(t31 + c31);
                            const int32x2_t p32 = vld1_dup_s32(t32 + c32);

                            int16x8_t s163 = vmovl_s8(vreinterpret_s8_s32(p30));
                            s163           = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p31)));
                            s163           = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p32)));
                            isum1          = vaddw_s16(isum1, vget_low_s16(s163));

                            idx_g += 4;
                            grp += 4 * k_ifairy_lut_group_bytes;
                        }
                        for (; gi + 1 < groups_per_block; gi += 2) {
                            const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                            const uint8_t c00 = (uint8_t) (pat0 & 3);
                            const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                            const uint8_t c02 = (uint8_t) (pat0 >> 4);

                            const uint8_t c10 = (uint8_t) (pat1 & 3);
                            const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                            const uint8_t c12 = (uint8_t) (pat1 >> 4);

                            const int8_t * grp0 = grp;
                            const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                            const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                            const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                            const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                            int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                            s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                            s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                            isum0          = vaddw_s16(isum0, vget_low_s16(s160));

                            const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                            const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                            const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                            int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                            s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                            s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                            isum1          = vaddw_s16(isum1, vget_low_s16(s161));

                            idx_g += 2;
                            grp += 2 * k_ifairy_lut_group_bytes;
                        }
                        for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                            const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                            const uint8_t c0  = (uint8_t) (pat & 3);
                            const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                            const uint8_t c2  = (uint8_t) (pat >> 4);

                            const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                            const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                            const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                            const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                            const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                            const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                            int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                            s16           = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                            s16           = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                            isum0         = vaddw_s16(isum0, vget_low_s16(s16));
                        }
                    }
                }

                const float32x2_t srsi  = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv   = vcombine_f32(srsi, srsi);  // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv                    = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t *  grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                for (int64_t gi = 0; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0  = (uint8_t) (pat & 3);
                    const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2  = (uint8_t) (pat >> 4);

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int8_t * e0 = (const int8_t *) &t0[c0];
                    const int8_t * e1 = (const int8_t *) &t1[c1];
                    const int8_t * e2 = (const int8_t *) &t2[c2];

                    sum_ac += (int32_t) e0[0] + (int32_t) e1[0] + (int32_t) e2[0];
                    sum_ad += (int32_t) e0[1] + (int32_t) e1[1] + (int32_t) e2[1];
                    sum_bc += (int32_t) e0[2] + (int32_t) e1[2] + (int32_t) e2[2];
                    sum_bd += (int32_t) e0[3] + (int32_t) e1[3] + (int32_t) e2[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, 0, out_r, out_i);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br                = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi                = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }

        return;
    }

    for (int row = 0; row < m; ++row) {
        const block_ifairy * w_row   = w_blocks + (size_t) row * (size_t) blocks;
        const uint8_t *      idx_row = indexes + (size_t) row * (size_t) groups;

        const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
        const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

        for (int col = 0; col < n; ++col) {
            const int8_t * lut_base =
                (const int8_t *) ((const uint8_t *) lut + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);
            const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t accv = vdupq_n_f32(0.0f);  // {ac, ad, bc, bd}
            const size_t prefetch_groups4 = prefetch ? (size_t) prefetch_dist * 4u : 0;
            const size_t prefetch_groups2 = prefetch ? (size_t) prefetch_dist * 2u : 0;
            const size_t prefetch_groups1 = prefetch ? (size_t) prefetch_dist : 0;
            const size_t prefetch_bytes4  = prefetch_groups4 * k_ifairy_lut_group_bytes;
            const size_t prefetch_bytes2  = prefetch_groups2 * k_ifairy_lut_group_bytes;
            const size_t prefetch_bytes1  = prefetch_groups1 * k_ifairy_lut_group_bytes;
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t *  grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                int64_t gi = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                    const uint8_t c00 = (uint8_t) (pat0 & 3);
                    const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                    const uint8_t c02 = (uint8_t) (pat0 >> 4);

                    const uint8_t c10 = (uint8_t) (pat1 & 3);
                    const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                    const uint8_t c12 = (uint8_t) (pat1 >> 4);

                    const uint8_t c20 = (uint8_t) (pat2 & 3);
                    const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                    const uint8_t c22 = (uint8_t) (pat2 >> 4);

                    const uint8_t c30 = (uint8_t) (pat3 & 3);
                    const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                    const uint8_t c32 = (uint8_t) (pat3 >> 4);

                    const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                    const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                    const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                    const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                    if (prefetch) {
                        __builtin_prefetch(idx_g + prefetch_groups4, 0, 1);
                        __builtin_prefetch(grp0 + prefetch_bytes4, 0, 1);
                    }

                    const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                    const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                    const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                    int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                    s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                    s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                    isum0          = vaddw_s16(isum0, vget_low_s16(s160));

                    const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                    const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                    const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                    int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                    s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                    s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                    isum1          = vaddw_s16(isum1, vget_low_s16(s161));

                    const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p20 = vld1_dup_s32(t20 + c20);
                    const int32x2_t p21 = vld1_dup_s32(t21 + c21);
                    const int32x2_t p22 = vld1_dup_s32(t22 + c22);

                    int16x8_t s162 = vmovl_s8(vreinterpret_s8_s32(p20));
                    s162           = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p21)));
                    s162           = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p22)));
                    isum0          = vaddw_s16(isum0, vget_low_s16(s162));

                    const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p30 = vld1_dup_s32(t30 + c30);
                    const int32x2_t p31 = vld1_dup_s32(t31 + c31);
                    const int32x2_t p32 = vld1_dup_s32(t32 + c32);

                    int16x8_t s163 = vmovl_s8(vreinterpret_s8_s32(p30));
                    s163           = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p31)));
                    s163           = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p32)));
                    isum1          = vaddw_s16(isum1, vget_low_s16(s163));

                    idx_g += 4;
                    grp += 4 * k_ifairy_lut_group_bytes;
                }
                for (; gi + 1 < groups_per_block; gi += 2) {
                    const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                    const uint8_t c00 = (uint8_t) (pat0 & 3);
                    const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                    const uint8_t c02 = (uint8_t) (pat0 >> 4);

                    const uint8_t c10 = (uint8_t) (pat1 & 3);
                    const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                    const uint8_t c12 = (uint8_t) (pat1 >> 4);

                    const int8_t * grp0 = grp;
                    const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                    if (prefetch) {
                        __builtin_prefetch(idx_g + prefetch_groups2, 0, 1);
                        __builtin_prefetch(grp0 + prefetch_bytes2, 0, 1);
                    }

                    const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                    const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                    const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                    int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                    s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                    s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                    isum0          = vaddw_s16(isum0, vget_low_s16(s160));

                    const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                    const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                    const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                    int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                    s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                    s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                    isum1          = vaddw_s16(isum1, vget_low_s16(s161));

                    idx_g += 2;
                    grp += 2 * k_ifairy_lut_group_bytes;
                }
                for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0  = (uint8_t) (pat & 3);
                    const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2  = (uint8_t) (pat >> 4);

                    if (prefetch) {
                        __builtin_prefetch(idx_g + prefetch_groups1, 0, 1);
                        __builtin_prefetch(grp + prefetch_bytes1, 0, 1);
                    }

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                    const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                    const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                    int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                    s16           = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                    s16           = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                    isum0         = vaddw_s16(isum0, vget_low_s16(s16));
                }

                const float32x2_t srsi  = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv   = vcombine_f32(srsi, srsi);  // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv                    = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t *  grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                for (int64_t gi = 0; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0  = (uint8_t) (pat & 3);
                    const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2  = (uint8_t) (pat >> 4);

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int8_t * e0 = (const int8_t *) &t0[c0];
                    const int8_t * e1 = (const int8_t *) &t1[c1];
                    const int8_t * e2 = (const int8_t *) &t2[c2];

                    sum_ac += (int32_t) e0[0] + (int32_t) e1[0] + (int32_t) e2[0];
                    sum_ad += (int32_t) e0[1] + (int32_t) e1[1] + (int32_t) e2[1];
                    sum_bc += (int32_t) e0[2] + (int32_t) e1[2] + (int32_t) e2[2];
                    sum_bd += (int32_t) e0[3] + (int32_t) e1[3] + (int32_t) e2[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, col, out_r, out_i);
            }

            if (strict) {
                GGML_ASSERT(act != NULL);
                const block_ifairy_q16 * act_blocks =
                    (const block_ifairy_q16 *) ((const uint8_t *) act + (size_t) col * act_stride);
                double ref_ac_xr = 0.0;
                double ref_ad_xi = 0.0;
                double ref_bc_xr = 0.0;
                double ref_bd_xi = 0.0;

                for (int blk = 0; blk < (int) blocks; ++blk) {
                    const uint8_t * GGML_RESTRICT w_ptr   = w_row[blk].qs;
                    const uint8_t * GGML_RESTRICT x_r_ptr = act_blocks[blk].x_real;
                    const uint8_t * GGML_RESTRICT x_i_ptr = act_blocks[blk].x_imag;

                    int32_t sum_ac = 0;
                    int32_t sum_ad = 0;
                    int32_t sum_bc = 0;
                    int32_t sum_bd = 0;

                    for (int j = 0; j < QK_K; ++j) {
                        const int chunk    = j >> 6;
                        const int lane     = j & 0xF;
                        const int part     = (j >> 4) & 0x3;
                        const int byte_idx = (chunk << 4) + lane;
                        const int bit_off  = part * 2;

                        const uint8_t packed = w_ptr[byte_idx];
                        const uint8_t code   = (packed >> bit_off) & 0x3;

                        int wr = 0;
                        int wi = 0;
                        switch (code) {
                            case 0:
                                wr = -1;
                                wi = 0;
                                break;
                            case 1:
                                wr = 1;
                                wi = 0;
                                break;
                            case 2:
                                wr = 0;
                                wi = -1;
                                break;
                            case 3:
                                wr = 0;
                                wi = 1;
                                break;
                            default:
                                GGML_UNREACHABLE();
                        }

                        const int xr = ggml_ifairy_lut_u8_to_s8(x_r_ptr[j]);
                        const int xi = ggml_ifairy_lut_u8_to_s8(x_i_ptr[j]);

                        sum_ac += xr * wr;
                        sum_ad += xi * wr;
                        sum_bc += xr * wi;
                        sum_bd += xi * wi;
                    }

                    const double x_real = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                    const double x_imag = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_imag);

                    ref_ac_xr += x_real * (double) sum_ac;
                    ref_ad_xi += x_imag * (double) sum_ad;
                    ref_bc_xr += x_real * (double) sum_bc;
                    ref_bd_xi += x_imag * (double) sum_bd;
                }

                const double ref_r = (double) coeff_w_real * ref_ac_xr + (double) coeff_w_imag * ref_bd_xi;
                const double ref_i = (double) coeff_w_imag * ref_bc_xr - (double) coeff_w_real * ref_ad_xi;

                const float dr = out_r - (float) ref_r;
                const float di = out_i - (float) ref_i;
                GGML_ASSERT(fabsf(dr) <= 1e-3f && fabsf(di) <= 1e-3f);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br                = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi                = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
    }
}

void ggml_ifairy_lut_qgemm(int             m,
                           int             k,
                           int             n,
                           const void *    qweights,
                           const uint8_t * indexes,
                           const void *    lut,
                           const void *    lut_scales,
                           const void *    act,
                           size_t          act_stride,
                           float *         dst,
                           size_t          dst_col_stride,
                           size_t          dst_row_stride,
                           bool            pack_bf16,
                           bool            strict) {
    ggml_ifairy_lut_qgemm_ex(m, k, n, qweights, indexes, lut, lut_scales, act, act_stride, dst, dst_col_stride,
                             dst_row_stride, pack_bf16, strict, false);
}

// Accumulate the 4 basis sums for the iFairy complex dot product:
//   dst[c*dst_col_stride + 0..3] += { acc_ac_xr, acc_ad_xi, acc_bc_xr, acc_bd_xi }
// where each term already includes the per-block activation scales (real/imag).
static void ggml_ifairy_lut_accum4_ex_legacy(int             k,
                                             int             n,
                                             const uint8_t * indexes,
                                             const void *    lut,
                                             const void *    lut_scales,
                                             float *         dst,
                                             size_t          dst_col_stride,
                                             bool            add) {
    if (!indexes || !dst || !lut || !lut_scales) {
        return;
    }

    const int    prefetch_dist   = ggml_ifairy_lut_prefetch_dist();
    const bool   prefetch        = ggml_ifairy_lut_prefetch_enabled() && prefetch_dist > 0;
    const size_t prefetch_groups = prefetch ? (size_t) prefetch_dist : 0;

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    for (int col = 0; col < n; ++col) {
        const int16_t * lut_base =
            (const int16_t *) ((const uint8_t *) lut + (size_t) col * (size_t) groups *
                                                           (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) *
                                                           sizeof(int16_t));
        const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;
        float *       out    = (float *) ((uint8_t *) dst + (size_t) col * dst_col_stride);

#if defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t accv = add ? vld1q_f32(out) : vdupq_n_f32(0.0f);
        for (int64_t blk = 0; blk < blocks; ++blk) {
            int32x4_t       isum0   = vdupq_n_s32(0);
            int32x4_t       isum1   = vdupq_n_s32(0);
            const uint8_t * idx_blk = indexes + (size_t) blk * (size_t) groups_per_block;
            const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block *
                                                     (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

            const size_t group_stride = (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);
            int64_t      gi           = 0;
            for (; gi + 3 < groups_per_block; gi += 4) {
                const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                const int16_t * grp0 = lut_blk + (size_t) (gi + 0) * group_stride;
                const int16_t * grp1 = lut_blk + (size_t) (gi + 1) * group_stride;
                const int16_t * grp2 = lut_blk + (size_t) (gi + 2) * group_stride;
                const int16_t * grp3 = lut_blk + (size_t) (gi + 3) * group_stride;

                if (prefetch) {
                    const size_t prefetch_stride = prefetch_groups * group_stride;
                    __builtin_prefetch(idx_blk + gi + prefetch_groups, 0, 1);
                    __builtin_prefetch(grp0 + prefetch_stride, 0, 1);
                    __builtin_prefetch(grp1 + prefetch_stride, 0, 1);
                }

                const int16_t * tbl0 = grp0 + (size_t) pat0 * k_ifairy_lut_channels;
                const int16_t * tbl1 = grp1 + (size_t) pat1 * k_ifairy_lut_channels;
                const int16_t * tbl2 = grp2 + (size_t) pat2 * k_ifairy_lut_channels;
                const int16_t * tbl3 = grp3 + (size_t) pat3 * k_ifairy_lut_channels;

                const int16x4_t s0 = vld1_s16(tbl0);
                const int16x4_t s1 = vld1_s16(tbl1);
                const int16x4_t s2 = vld1_s16(tbl2);
                const int16x4_t s3 = vld1_s16(tbl3);

                isum0 = vaddw_s16(isum0, s0);
                isum1 = vaddw_s16(isum1, s1);
                isum0 = vaddw_s16(isum0, s2);
                isum1 = vaddw_s16(isum1, s3);
            }
            for (; gi < groups_per_block; ++gi) {
                const uint8_t   pat    = (uint8_t) (idx_blk[gi] & 0x3f);
                const int16_t * tbl    = lut_blk + (size_t) gi * group_stride + (size_t) pat * k_ifairy_lut_channels;
                const int16x4_t sums16 = vld1_s16(tbl);  // {ac, ad, bc, bd}
                isum0                  = vaddw_s16(isum0, sums16);
            }

            const float32x2_t srsi  = vld1_f32(scales + (size_t) blk * 2);
            const float32x4_t scv   = vcombine_f32(srsi, srsi);  // {sr, si, sr, si}
            const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
            accv                    = vmlaq_f32(accv, sumsf, scv);
        }
        vst1q_f32(out, accv);
#else
        float acc_ac_xr = add ? out[0] : 0.0f;
        float acc_ad_xi = add ? out[1] : 0.0f;
        float acc_bc_xr = add ? out[2] : 0.0f;
        float acc_bd_xi = add ? out[3] : 0.0f;

        for (int64_t blk = 0; blk < blocks; ++blk) {
            int32_t sum_ac = 0;
            int32_t sum_ad = 0;
            int32_t sum_bc = 0;
            int32_t sum_bd = 0;

            const uint8_t * idx_blk = indexes + (size_t) blk * (size_t) groups_per_block;
            const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block *
                                                     (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

            for (int64_t gi = 0; gi < groups_per_block; ++gi) {
                const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                const int16_t * tbl = lut_blk + (size_t) gi * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels) +
                                      (size_t) pat * k_ifairy_lut_channels;
                sum_ac += (int32_t) tbl[0];
                sum_ad += (int32_t) tbl[1];
                sum_bc += (int32_t) tbl[2];
                sum_bd += (int32_t) tbl[3];
            }

            const float act_scale_r = scales[blk * 2 + 0];
            const float act_scale_i = scales[blk * 2 + 1];
            acc_ac_xr += act_scale_r * (float) sum_ac;
            acc_ad_xi += act_scale_i * (float) sum_ad;
            acc_bc_xr += act_scale_r * (float) sum_bc;
            acc_bd_xi += act_scale_i * (float) sum_bd;
        }

        out[0] = acc_ac_xr;
        out[1] = acc_ad_xi;
        out[2] = acc_bc_xr;
        out[3] = acc_bd_xi;
#endif
    }
}

void ggml_ifairy_lut_accum4_ex(int             k,
                               int             n,
                               const uint8_t * indexes,
                               const void *    lut,
                               const void *    lut_scales,
                               float *         dst,
                               size_t          dst_col_stride,
                               bool            add) {
    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env(n);
    if (layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY) {
        ggml_ifairy_lut_accum4_ex_legacy(k, n, indexes, lut, lut_scales, dst, dst_col_stride, add);
        return;
    }

    if (!indexes || !dst || !lut || !lut_scales) {
        return;
    }

    const int    prefetch_dist   = ggml_ifairy_lut_prefetch_dist();
    const bool   prefetch        = ggml_ifairy_lut_prefetch_enabled() && prefetch_dist > 0;
    const size_t prefetch_groups = prefetch ? (size_t) prefetch_dist : 0;

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_base =
            (const int8_t *) ((const uint8_t *) lut + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);
        const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;
        float *       out    = (float *) ((uint8_t *) dst + (size_t) col * dst_col_stride);

#if defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t  accv             = add ? vld1q_f32(out) : vdupq_n_f32(0.0f);
        const size_t prefetch_groups2 = prefetch ? prefetch_groups * 2u : 0;
        const size_t prefetch_groups1 = prefetch ? prefetch_groups : 0;
        const size_t prefetch_bytes2  = prefetch_groups2 * k_ifairy_lut_group_bytes;
        const size_t prefetch_bytes1  = prefetch_groups1 * k_ifairy_lut_group_bytes;
        for (int64_t blk = 0; blk < blocks; ++blk) {
            int32x4_t       isum0 = vdupq_n_s32(0);
            int32x4_t       isum1 = vdupq_n_s32(0);
            const uint8_t * idx_g = indexes + (size_t) blk * (size_t) groups_per_block;
            const int8_t *  grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

            int64_t gi = 0;
            for (; gi + 1 < groups_per_block; gi += 2) {
                const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                const uint8_t c00 = (uint8_t) (pat0 & 3);
                const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                const uint8_t c02 = (uint8_t) (pat0 >> 4);

                const uint8_t c10 = (uint8_t) (pat1 & 3);
                const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                const uint8_t c12 = (uint8_t) (pat1 >> 4);

                const int8_t * grp0 = grp;
                const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                if (prefetch) {
                    __builtin_prefetch(idx_g + prefetch_groups2, 0, 1);
                    __builtin_prefetch(grp0 + prefetch_bytes2, 0, 1);
                }

                const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                s160           = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                isum0          = vaddw_s16(isum0, vget_low_s16(s160));

                const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                s161           = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                isum1          = vaddw_s16(isum1, vget_low_s16(s161));

                idx_g += 2;
                grp += 2 * k_ifairy_lut_group_bytes;
            }
            for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                const uint8_t c0  = (uint8_t) (pat & 3);
                const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                const uint8_t c2  = (uint8_t) (pat >> 4);

                if (prefetch) {
                    __builtin_prefetch(idx_g + prefetch_groups1, 0, 1);
                    __builtin_prefetch(grp + prefetch_bytes1, 0, 1);
                }

                const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                s16           = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                s16           = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                isum0         = vaddw_s16(isum0, vget_low_s16(s16));
            }

            const float32x2_t srsi  = vld1_f32(scales + (size_t) blk * 2);
            const float32x4_t scv   = vcombine_f32(srsi, srsi);  // {sr, si, sr, si}
            const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
            accv                    = vmlaq_f32(accv, sumsf, scv);
        }
        vst1q_f32(out, accv);
#else
        float acc_ac_xr = add ? out[0] : 0.0f;
        float acc_ad_xi = add ? out[1] : 0.0f;
        float acc_bc_xr = add ? out[2] : 0.0f;
        float acc_bd_xi = add ? out[3] : 0.0f;

        for (int64_t blk = 0; blk < blocks; ++blk) {
            int32_t sum_ac = 0;
            int32_t sum_ad = 0;
            int32_t sum_bc = 0;
            int32_t sum_bd = 0;

            const uint8_t * idx_g = indexes + (size_t) blk * (size_t) groups_per_block;
            const int8_t *  grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

            for (int64_t gi = 0; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                const uint8_t c0  = (uint8_t) (pat & 3);
                const uint8_t c1  = (uint8_t) ((pat >> 2) & 3);
                const uint8_t c2  = (uint8_t) (pat >> 4);

                const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                const int8_t * e0 = (const int8_t *) &t0[c0];
                const int8_t * e1 = (const int8_t *) &t1[c1];
                const int8_t * e2 = (const int8_t *) &t2[c2];

                sum_ac += (int32_t) e0[0] + (int32_t) e1[0] + (int32_t) e2[0];
                sum_ad += (int32_t) e0[1] + (int32_t) e1[1] + (int32_t) e2[1];
                sum_bc += (int32_t) e0[2] + (int32_t) e1[2] + (int32_t) e2[2];
                sum_bd += (int32_t) e0[3] + (int32_t) e1[3] + (int32_t) e2[3];
            }

            const float act_scale_r = scales[blk * 2 + 0];
            const float act_scale_i = scales[blk * 2 + 1];
            acc_ac_xr += act_scale_r * (float) sum_ac;
            acc_ad_xi += act_scale_i * (float) sum_ad;
            acc_bc_xr += act_scale_r * (float) sum_bc;
            acc_bd_xi += act_scale_i * (float) sum_bd;
        }

        out[0] = acc_ac_xr;
        out[1] = acc_ad_xi;
        out[2] = acc_bc_xr;
        out[3] = acc_bd_xi;
#endif
    }
}

static void ggml_ifairy_lut_mul_mat_scalar_internal(int             m,
                                                    int             k,
                                                    int             n,
                                                    const void *    qweights,
                                                    const void *    act,
                                                    size_t          act_stride,
                                                    const uint8_t * indexes,
                                                    int8_t *        lut,
                                                    float *         scales,
                                                    float *         dst,
                                                    size_t          dst_col_stride) {
    if (!qweights || !act || !dst || !indexes || !lut || !scales) {
        return;
    }

    const bool strict = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_VALIDATE_STRICT");

    // preprocess activations -> LUT per column
    ggml_ifairy_lut_preprocess(m, k, n, act, act_stride, scales, lut);
    const size_t dst_row_stride = 2 * sizeof(float);
    ggml_ifairy_lut_qgemm(m, k, n, qweights, indexes, lut, scales, act, act_stride, dst, dst_col_stride, dst_row_stride,
                          false, strict);
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

    // workspace: indexes + LUT + scales
    const size_t                 index_bytes_raw = (size_t) m * (size_t) groups;
    const size_t                 index_bytes     = GGML_PAD(index_bytes_raw, 64);
    const ggml_ifairy_lut_layout layout          = ggml_ifairy_lut_layout_from_env(n);
    const size_t                 lut_bytes =
        layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY ?
                            (size_t) n * (size_t) groups * (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) * sizeof(int16_t) :
                            (size_t) n * (size_t) groups * (size_t) k_ifairy_lut_group_bytes;
    const size_t scale_bytes = (size_t) n * (size_t) blocks * 2 * sizeof(float);
    const size_t total_bytes = index_bytes + lut_bytes + scale_bytes;

    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, total_bytes) != 0) {
        return;
    }
    uint8_t * buf = (uint8_t *) ptr;
    memset(buf, 0, total_bytes);
    uint8_t * indexes = buf;
    int8_t *  lut     = (int8_t *) (buf + index_bytes);
    float *   scales  = (float *) (buf + index_bytes + lut_bytes);

    // build indexes per row
    ggml_ifairy_3w_encode((const block_ifairy *) qweights, K, m, indexes, index_bytes_raw);
    ggml_ifairy_lut_mul_mat_scalar_internal(m, k, n, qweights, act, act_stride, indexes, lut, scales, dst,
                                            (size_t) m * 2 * sizeof(float));

    free(buf);
}
