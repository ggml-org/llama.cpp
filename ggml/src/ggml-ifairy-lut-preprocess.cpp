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

// iFairy LUT V2: keep a single production layout (merged64).

#if defined(__ARM_NEON) && defined(__aarch64__)
// wr(code) / wi(code) coefficients for all 64 3-weight patterns (direct 6-bit encoding).
// code -> (wr, wi): 0 -> (-1,0), 1 -> (1,0), 2 -> (0,-1), 3 -> (0,1)
static const int8_t k_ifairy_wr0[64] = { -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1,
                                         0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0,
                                         -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0 };
static const int8_t k_ifairy_wr1[64] = { -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                         -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                         -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                         -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
static const int8_t k_ifairy_wr2[64] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                                         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 };
static const int8_t k_ifairy_wi0[64] = { 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0,
                                         -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1,
                                         0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1, 0,  0, -1, 1 };
static const int8_t k_ifairy_wi1[64] = { 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1 };
static const int8_t k_ifairy_wi2[64] = { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 };
#endif

static inline int8_t ggml_ifairy_lut_sat_s8(int v) {
    if (v > INT8_MAX) {
        return INT8_MAX;
    }
    if (v < INT8_MIN) {
        return INT8_MIN;
    }
    return (int8_t) v;
}

static void ggml_ifairy_lut_preprocess_merged64(int          m,
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

        // Layout (per group, int8): tbl[(pat*4) + 0..3] = {ac, ad, bc, bd} (256B/group).
        int8_t * lut_out =
            (int8_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups * k_ifairy_lut_merged64_group_bytes);

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

            int8_t xr0 = 0;
            int8_t xi0 = 0;
            int8_t xr1 = 0;
            int8_t xi1 = 0;
            int8_t xr2 = 0;
            int8_t xi2 = 0;

            if (idx0 < K) {
                xr0 = (int8_t) act_blocks[blk0].x_real[off0];
                xi0 = (int8_t) act_blocks[blk0].x_imag[off0];
            }
            if (!tail) {
                xr1 = (int8_t) act_blocks[blk1].x_real[off1];
                xi1 = (int8_t) act_blocks[blk1].x_imag[off1];
                xr2 = (int8_t) act_blocks[blk2].x_real[off2];
                xi2 = (int8_t) act_blocks[blk2].x_imag[off2];
            }

            int8_t * tbl = lut_out + (size_t) g * k_ifairy_lut_merged64_group_bytes;

#if defined(__ARM_NEON) && defined(__aarch64__)
            const int8_t xr0_s8 = xr0;
            const int8_t xr1_s8 = xr1;
            const int8_t xr2_s8 = xr2;
            const int8_t xi0_s8 = xi0;
            const int8_t xi1_s8 = xi1;
            const int8_t xi2_s8 = xi2;

            for (int pat = 0; pat < 64; pat += 16) {
                const int8x16_t wr0 = vld1q_s8(k_ifairy_wr0 + pat);
                const int8x16_t wr1 = vld1q_s8(k_ifairy_wr1 + pat);
                const int8x16_t wr2 = vld1q_s8(k_ifairy_wr2 + pat);
                const int8x16_t wi0 = vld1q_s8(k_ifairy_wi0 + pat);
                const int8x16_t wi1 = vld1q_s8(k_ifairy_wi1 + pat);
                const int8x16_t wi2 = vld1q_s8(k_ifairy_wi2 + pat);

                int16x8_t ac0 = vmull_s8(vget_low_s8(wr0), vdup_n_s8(xr0_s8));
                ac0           = vmlal_s8(ac0, vget_low_s8(wr1), vdup_n_s8(xr1_s8));
                ac0           = vmlal_s8(ac0, vget_low_s8(wr2), vdup_n_s8(xr2_s8));

                int16x8_t ad0 = vmull_s8(vget_low_s8(wr0), vdup_n_s8(xi0_s8));
                ad0           = vmlal_s8(ad0, vget_low_s8(wr1), vdup_n_s8(xi1_s8));
                ad0           = vmlal_s8(ad0, vget_low_s8(wr2), vdup_n_s8(xi2_s8));

                int16x8_t bc0 = vmull_s8(vget_low_s8(wi0), vdup_n_s8(xr0_s8));
                bc0           = vmlal_s8(bc0, vget_low_s8(wi1), vdup_n_s8(xr1_s8));
                bc0           = vmlal_s8(bc0, vget_low_s8(wi2), vdup_n_s8(xr2_s8));

                int16x8_t bd0 = vmull_s8(vget_low_s8(wi0), vdup_n_s8(xi0_s8));
                bd0           = vmlal_s8(bd0, vget_low_s8(wi1), vdup_n_s8(xi1_s8));
                bd0           = vmlal_s8(bd0, vget_low_s8(wi2), vdup_n_s8(xi2_s8));

                int16x8_t ac1 = vmull_s8(vget_high_s8(wr0), vdup_n_s8(xr0_s8));
                ac1           = vmlal_s8(ac1, vget_high_s8(wr1), vdup_n_s8(xr1_s8));
                ac1           = vmlal_s8(ac1, vget_high_s8(wr2), vdup_n_s8(xr2_s8));

                int16x8_t ad1 = vmull_s8(vget_high_s8(wr0), vdup_n_s8(xi0_s8));
                ad1           = vmlal_s8(ad1, vget_high_s8(wr1), vdup_n_s8(xi1_s8));
                ad1           = vmlal_s8(ad1, vget_high_s8(wr2), vdup_n_s8(xi2_s8));

                int16x8_t bc1 = vmull_s8(vget_high_s8(wi0), vdup_n_s8(xr0_s8));
                bc1           = vmlal_s8(bc1, vget_high_s8(wi1), vdup_n_s8(xr1_s8));
                bc1           = vmlal_s8(bc1, vget_high_s8(wi2), vdup_n_s8(xr2_s8));

                int16x8_t bd1 = vmull_s8(vget_high_s8(wi0), vdup_n_s8(xi0_s8));
                bd1           = vmlal_s8(bd1, vget_high_s8(wi1), vdup_n_s8(xi1_s8));
                bd1           = vmlal_s8(bd1, vget_high_s8(wi2), vdup_n_s8(xi2_s8));

                int8x16x4_t out;
                out.val[0] = vcombine_s8(vqmovn_s16(ac0), vqmovn_s16(ac1));
                out.val[1] = vcombine_s8(vqmovn_s16(ad0), vqmovn_s16(ad1));
                out.val[2] = vcombine_s8(vqmovn_s16(bc0), vqmovn_s16(bc1));
                out.val[3] = vcombine_s8(vqmovn_s16(bd0), vqmovn_s16(bd1));
                vst4q_s8(tbl + (size_t) pat * 4u, out);
            }
#else
            for (int pat = 0; pat < 64; ++pat) {
                const uint8_t c0 = (uint8_t) (pat & 3);
                const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                const uint8_t c2 = (uint8_t) (pat >> 4);

                int wr0 = 0, wi0 = 0;
                int wr1 = 0, wi1 = 0;
                int wr2 = 0, wi2 = 0;

                switch (c0) {
                    case 0:
                        wr0 = -1;
                        break;
                    case 1:
                        wr0 = 1;
                        break;
                    case 2:
                        wi0 = -1;
                        break;
                    case 3:
                        wi0 = 1;
                        break;
                }
                switch (c1) {
                    case 0:
                        wr1 = -1;
                        break;
                    case 1:
                        wr1 = 1;
                        break;
                    case 2:
                        wi1 = -1;
                        break;
                    case 3:
                        wi1 = 1;
                        break;
                }
                switch (c2) {
                    case 0:
                        wr2 = -1;
                        break;
                    case 1:
                        wr2 = 1;
                        break;
                    case 2:
                        wi2 = -1;
                        break;
                    case 3:
                        wi2 = 1;
                        break;
                }

                const int sum_ac = xr0 * wr0 + xr1 * wr1 + xr2 * wr2;
                const int sum_ad = xi0 * wr0 + xi1 * wr1 + xi2 * wr2;
                const int sum_bc = xr0 * wi0 + xr1 * wi1 + xr2 * wi2;
                const int sum_bd = xi0 * wi0 + xi1 * wi1 + xi2 * wi2;

                tbl[pat * 4 + 0] = ggml_ifairy_lut_sat_s8(sum_ac);
                tbl[pat * 4 + 1] = ggml_ifairy_lut_sat_s8(sum_ad);
                tbl[pat * 4 + 2] = ggml_ifairy_lut_sat_s8(sum_bc);
                tbl[pat * 4 + 3] = ggml_ifairy_lut_sat_s8(sum_bd);
            }
#endif
        }
    }
}

void ggml_ifairy_lut_preprocess_ex_merged64(int          m,
                                            int          k,
                                            int          n,
                                            const void * act,
                                            size_t       act_stride,
                                            void *       lut_scales,
                                            void *       lut_buf,
                                            int          ith,
                                            int          nth) {
    ggml_ifairy_lut_preprocess_merged64(m, k, n, act, act_stride, lut_scales, lut_buf, ith, nth);
}
