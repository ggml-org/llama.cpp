#define GGML_COMMON_DECL_CPP
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-ifairy-lut-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif

// iFairy LUT V2: keep a single production layout (lut16), matching lut_c's 16-entry compressed table.

static inline int8_t ggml_ifairy_lut_sat_s8(int v) {
    if (v > INT8_MAX) {
        return INT8_MAX;
    }
    if (v < INT8_MIN) {
        return INT8_MIN;
    }
    return (int8_t) v;
}

static inline int ggml_ifairy_u8_to_s8_int(uint8_t v) {
    return v < 128 ? (int) v : (int) v - 256;
}

static void ggml_ifairy_lut_preprocess_lut16(int          m,
                                             int          k,
                                             int          n,
                                             const void * act,
                                             size_t       act_stride,
                                             void *       lut_scales,
                                             void *       lut_buf,
                                             int          ith,
                                             int          nth) {
    (void) m;  // rows unused in preprocess (per-column)
    if (!act || !lut_scales || !lut_buf) {
        return;
    }

    nth = std::max(nth, 1);
    if (ith < 0 || ith >= nth) {
        return;
    }

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    const bool shard_by_col = n >= nth;

    const int col_start = shard_by_col ? ith : 0;
    const int col_step  = shard_by_col ? nth : 1;
    const int col_end   = n;

    for (int col = col_start; col < col_end; col += col_step) {
        const uint8_t *          act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks    = (const block_ifairy_q16 *) act_col_bytes;
        float *                  scales_out    = (float *) lut_scales + (size_t) col * (size_t) blocks * 2;

        if (shard_by_col || ith == 0) {
            for (int64_t blk = 0; blk < blocks; ++blk) {
                scales_out[blk * 2 + 0] = GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                scales_out[blk * 2 + 1] = GGML_FP16_TO_FP32(act_blocks[blk].d_imag);
            }
        }

        // Layout (per group, int8): tbl[(entry*4*16) + lane] laid out as 4x int8x16 vectors (64B/group).
        // Stored order matches lut_c: {ac, bd, ad, bc}.
        int8_t * lut_out = (int8_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);

        const int64_t g0    = shard_by_col ? 0 : ith;
        const int64_t gstep = shard_by_col ? 1 : (int64_t) nth;

        for (int64_t g = g0; g < groups; g += gstep) {
            const int64_t blk   = g / groups_per_block;
            const int64_t intra = g - blk * groups_per_block;

            const bool    tail     = intra == groups_per_block - 1;
            const int64_t base_off = tail ? (QK_K - 1) : intra * 3;
            const int64_t idx0     = blk * QK_K + base_off + 0;

            const int blk0 = (int) blk;
            const int off0 = (int) base_off;
            const int blk1 = (int) blk;
            const int blk2 = (int) blk;
            const int off1 = (int) (base_off + 1);
            const int off2 = (int) (base_off + 2);

            int xr0 = 0;
            int xi0 = 0;
            int xr1 = 0;
            int xi1 = 0;
            int xr2 = 0;
            int xi2 = 0;

            if (idx0 < K) {
                xr0 = ggml_ifairy_u8_to_s8_int(act_blocks[blk0].x_real[off0]);
                xi0 = ggml_ifairy_u8_to_s8_int(act_blocks[blk0].x_imag[off0]);
            }
            if (!tail) {
                xr1 = ggml_ifairy_u8_to_s8_int(act_blocks[blk1].x_real[off1]);
                xi1 = ggml_ifairy_u8_to_s8_int(act_blocks[blk1].x_imag[off1]);
                xr2 = ggml_ifairy_u8_to_s8_int(act_blocks[blk2].x_real[off2]);
                xi2 = ggml_ifairy_u8_to_s8_int(act_blocks[blk2].x_imag[off2]);
            }

            // lut_c-style: pre-negate imag to fold conj(x) into LUT.
            const int r0 = xr0;
            const int r1 = xr1;
            const int r2 = xr2;
            const int i0 = -xi0;
            const int i1 = -xi1;
            const int i2 = -xi2;

            alignas(16) int8_t ac_tbl[16] = {
                ggml_ifairy_lut_sat_s8(-r0 - r1 - r2), ggml_ifairy_lut_sat_s8(-r0 - r1 + r2),
                ggml_ifairy_lut_sat_s8(-r0 - r1),      ggml_ifairy_lut_sat_s8(-r0 - r1),
                ggml_ifairy_lut_sat_s8(-r0 + r1 - r2), ggml_ifairy_lut_sat_s8(-r0 + r1 + r2),
                ggml_ifairy_lut_sat_s8(-r0 + r1),      ggml_ifairy_lut_sat_s8(-r0 + r1),
                ggml_ifairy_lut_sat_s8(-r0 - r2),      ggml_ifairy_lut_sat_s8(-r0 + r2),
                ggml_ifairy_lut_sat_s8(-r0),           ggml_ifairy_lut_sat_s8(-r0),
                ggml_ifairy_lut_sat_s8(-r0 - r2),      ggml_ifairy_lut_sat_s8(-r0 + r2),
                ggml_ifairy_lut_sat_s8(-r0),           ggml_ifairy_lut_sat_s8(-r0),
            };

            alignas(16) int8_t bd_tbl[16] = {
                0,
                0,
                ggml_ifairy_lut_sat_s8(-i2),
                ggml_ifairy_lut_sat_s8(i2),
                0,
                0,
                ggml_ifairy_lut_sat_s8(-i2),
                ggml_ifairy_lut_sat_s8(i2),
                ggml_ifairy_lut_sat_s8(-i1),
                ggml_ifairy_lut_sat_s8(-i1),
                ggml_ifairy_lut_sat_s8(-i1 - i2),
                ggml_ifairy_lut_sat_s8(-i1 + i2),
                ggml_ifairy_lut_sat_s8(i1),
                ggml_ifairy_lut_sat_s8(i1),
                ggml_ifairy_lut_sat_s8(i1 - i2),
                ggml_ifairy_lut_sat_s8(i1 + i2),
            };

            alignas(16) int8_t ad_tbl[16] = {
                0,
                0,
                ggml_ifairy_lut_sat_s8(-r2),
                ggml_ifairy_lut_sat_s8(r2),
                0,
                0,
                ggml_ifairy_lut_sat_s8(-r2),
                ggml_ifairy_lut_sat_s8(r2),
                ggml_ifairy_lut_sat_s8(-r1),
                ggml_ifairy_lut_sat_s8(-r1),
                ggml_ifairy_lut_sat_s8(-r1 - r2),
                ggml_ifairy_lut_sat_s8(-r1 + r2),
                ggml_ifairy_lut_sat_s8(r1),
                ggml_ifairy_lut_sat_s8(r1),
                ggml_ifairy_lut_sat_s8(r1 - r2),
                ggml_ifairy_lut_sat_s8(r1 + r2),
            };

            alignas(16) int8_t bc_tbl[16] = {
                ggml_ifairy_lut_sat_s8(-i0 - i1 - i2), ggml_ifairy_lut_sat_s8(-i0 - i1 + i2),
                ggml_ifairy_lut_sat_s8(-i0 - i1),      ggml_ifairy_lut_sat_s8(-i0 - i1),
                ggml_ifairy_lut_sat_s8(-i0 + i1 - i2), ggml_ifairy_lut_sat_s8(-i0 + i1 + i2),
                ggml_ifairy_lut_sat_s8(-i0 + i1),      ggml_ifairy_lut_sat_s8(-i0 + i1),
                ggml_ifairy_lut_sat_s8(-i0 - i2),      ggml_ifairy_lut_sat_s8(-i0 + i2),
                ggml_ifairy_lut_sat_s8(-i0),           ggml_ifairy_lut_sat_s8(-i0),
                ggml_ifairy_lut_sat_s8(-i0 - i2),      ggml_ifairy_lut_sat_s8(-i0 + i2),
                ggml_ifairy_lut_sat_s8(-i0),           ggml_ifairy_lut_sat_s8(-i0),
            };

            int8_t * tbl = lut_out + (size_t) g * k_ifairy_lut_group_bytes;

#if defined(__ARM_NEON) && defined(__aarch64__)
            vst1q_s8(tbl + 0 * 16, vld1q_s8(ac_tbl));
            vst1q_s8(tbl + 1 * 16, vld1q_s8(bd_tbl));
            vst1q_s8(tbl + 2 * 16, vld1q_s8(ad_tbl));
            vst1q_s8(tbl + 3 * 16, vld1q_s8(bc_tbl));
#else
            memcpy(tbl + 0 * 16, ac_tbl, 16);
            memcpy(tbl + 1 * 16, bd_tbl, 16);
            memcpy(tbl + 2 * 16, ad_tbl, 16);
            memcpy(tbl + 3 * 16, bc_tbl, 16);
#endif
        }
    }
}

void ggml_ifairy_lut_preprocess_ex_lut16(int          m,
                                         int          k,
                                         int          n,
                                         const void * act,
                                         size_t       act_stride,
                                         void *       lut_scales,
                                         void *       lut_buf,
                                         int          ith,
                                         int          nth) {
    ggml_ifairy_lut_preprocess_lut16(m, k, n, act, act_stride, lut_scales, lut_buf, ith, nth);
}

void ggml_ifairy_lut_preprocess_ex_lut_c(int          m,
                                         int          k,
                                         int          n,
                                         const void * act,
                                         size_t       act_stride,
                                         void *       lut_scales,
                                         void *       lut_buf,
                                         int          ith,
                                         int          nth) {
    // lut_c backend uses the same 16-entry LUT layout; scaling differences are handled at activation quantization.
    ggml_ifairy_lut_preprocess_lut16(m, k, n, act, act_stride, lut_scales, lut_buf, ith, nth);
}
