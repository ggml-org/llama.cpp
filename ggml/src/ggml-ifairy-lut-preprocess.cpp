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

// iFairy LUT V2: 2-weight direct 4-bit index, 16-entry LUT.

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
    const int64_t blocks           = K / QK_IFAIRY;
    const int64_t groups_per_block = QK_IFAIRY_GROUPS_PER_BLOCK;
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

        // Layout (per group, int8): 4x int8x16 vectors (64B/group).
        // Stored order: {ac, bd, bc, ad} to match AVX2 lane extraction.
        int8_t * lut_out = (int8_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);

        const int64_t g0    = shard_by_col ? 0 : ith;
        const int64_t gstep = shard_by_col ? 1 : (int64_t) nth;

        for (int64_t g = g0; g < groups; g += gstep) {
            const int64_t blk      = g / groups_per_block;
            const int64_t intra    = g - blk * groups_per_block;
            const int64_t base_off = intra * 2;

            const int blk0 = (int) blk;

            int xr0 = ggml_ifairy_u8_to_s8_int(act_blocks[blk0].x_real[base_off + 0]);
            int xi0 = ggml_ifairy_u8_to_s8_int(act_blocks[blk0].x_imag[base_off + 0]);
            int xr1 = ggml_ifairy_u8_to_s8_int(act_blocks[blk0].x_real[base_off + 1]);
            int xi1 = ggml_ifairy_u8_to_s8_int(act_blocks[blk0].x_imag[base_off + 1]);

            // Pre-negate imag to fold conj(x) into LUT.
            const int r0 = xr0;
            const int r1 = xr1;
            const int i0 = -xi0;
            const int i1 = -xi1;

            // 2-weight LUT: pat = c0 | (c1 << 2), 16 entries.
            // sign_r(c) = {-1, +1, 0, 0}[c], sign_i(c) = {0, 0, -1, +1}[c]
            // ac[pat] = sign_r(c0)*r0 + sign_r(c1)*r1
            alignas(16) int8_t ac_tbl[16] = {
                ggml_ifairy_lut_sat_s8(-r0 - r1), ggml_ifairy_lut_sat_s8(+r0 - r1),
                ggml_ifairy_lut_sat_s8(-r1),      ggml_ifairy_lut_sat_s8(-r1),
                ggml_ifairy_lut_sat_s8(-r0 + r1), ggml_ifairy_lut_sat_s8(+r0 + r1),
                ggml_ifairy_lut_sat_s8(+r1),      ggml_ifairy_lut_sat_s8(+r1),
                ggml_ifairy_lut_sat_s8(-r0),      ggml_ifairy_lut_sat_s8(+r0),
                0,                                 0,
                ggml_ifairy_lut_sat_s8(-r0),      ggml_ifairy_lut_sat_s8(+r0),
                0,                                 0,
            };

            // bd[pat] = sign_i(c0)*i0 + sign_i(c1)*i1
            alignas(16) int8_t bd_tbl[16] = {
                0,                                 0,
                ggml_ifairy_lut_sat_s8(-i0),      ggml_ifairy_lut_sat_s8(+i0),
                0,                                 0,
                ggml_ifairy_lut_sat_s8(-i0),      ggml_ifairy_lut_sat_s8(+i0),
                ggml_ifairy_lut_sat_s8(-i1),      ggml_ifairy_lut_sat_s8(-i1),
                ggml_ifairy_lut_sat_s8(-i0 - i1), ggml_ifairy_lut_sat_s8(+i0 - i1),
                ggml_ifairy_lut_sat_s8(+i1),      ggml_ifairy_lut_sat_s8(+i1),
                ggml_ifairy_lut_sat_s8(-i0 + i1), ggml_ifairy_lut_sat_s8(+i0 + i1),
            };

            // bc[pat] = sign_r(c0)*i0 + sign_r(c1)*i1
            alignas(16) int8_t bc_tbl[16] = {
                ggml_ifairy_lut_sat_s8(-i0 - i1), ggml_ifairy_lut_sat_s8(+i0 - i1),
                ggml_ifairy_lut_sat_s8(-i1),      ggml_ifairy_lut_sat_s8(-i1),
                ggml_ifairy_lut_sat_s8(-i0 + i1), ggml_ifairy_lut_sat_s8(+i0 + i1),
                ggml_ifairy_lut_sat_s8(+i1),      ggml_ifairy_lut_sat_s8(+i1),
                ggml_ifairy_lut_sat_s8(-i0),      ggml_ifairy_lut_sat_s8(+i0),
                0,                                 0,
                ggml_ifairy_lut_sat_s8(-i0),      ggml_ifairy_lut_sat_s8(+i0),
                0,                                 0,
            };

            // ad[pat] = sign_i(c0)*r0 + sign_i(c1)*r1
            alignas(16) int8_t ad_tbl[16] = {
                0,                                 0,
                ggml_ifairy_lut_sat_s8(-r0),      ggml_ifairy_lut_sat_s8(+r0),
                0,                                 0,
                ggml_ifairy_lut_sat_s8(-r0),      ggml_ifairy_lut_sat_s8(+r0),
                ggml_ifairy_lut_sat_s8(-r1),      ggml_ifairy_lut_sat_s8(-r1),
                ggml_ifairy_lut_sat_s8(-r0 - r1), ggml_ifairy_lut_sat_s8(+r0 - r1),
                ggml_ifairy_lut_sat_s8(+r1),      ggml_ifairy_lut_sat_s8(+r1),
                ggml_ifairy_lut_sat_s8(-r0 + r1), ggml_ifairy_lut_sat_s8(+r0 + r1),
            };

            // Store order: {ac, bd, bc, ad} to match AVX2 lane extraction convention.
            int8_t * tbl = lut_out + (size_t) g * k_ifairy_lut_group_bytes;

#if defined(__ARM_NEON) && defined(__aarch64__)
            vst1q_s8(tbl + 0 * 16, vld1q_s8(ac_tbl));
            vst1q_s8(tbl + 1 * 16, vld1q_s8(bd_tbl));
            vst1q_s8(tbl + 2 * 16, vld1q_s8(bc_tbl));
            vst1q_s8(tbl + 3 * 16, vld1q_s8(ad_tbl));
#else
            memcpy(tbl + 0 * 16, ac_tbl, 16);
            memcpy(tbl + 1 * 16, bd_tbl, 16);
            memcpy(tbl + 2 * 16, bc_tbl, 16);
            memcpy(tbl + 3 * 16, ad_tbl, 16);
#endif
        }
    }
}

// ggml_ifairy_lut_preprocess_ex_lut16 and ggml_ifairy_lut_preprocess_ex_lut_c
// are now defined in ggml-ifairy-lut-qgemm.cpp (with AVX2 vectorized preprocess).
