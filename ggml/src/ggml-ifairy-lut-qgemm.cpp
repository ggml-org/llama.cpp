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
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#endif

// iFairy LUT V2: keep a single production kernel/layout (merged64).
// The merged64 kernel is carried forward from the V1 implementation with the same default tuning knobs,
// but the knobs themselves are removed to reduce surface area and branches.

void ggml_ifairy_lut_qgemm_merged64(int             m,
                                    int             k,
                                    int             n,
                                    const void *    qweights,
                                    const uint8_t * indexes,
                                    const void *    lut,
                                    const void *    lut_scales,
                                    float *         dst,
                                    size_t          dst_col_stride,
                                    size_t          dst_row_stride,
                                    bool            pack_bf16,
                                    bool            add) {
    if (!indexes || !dst || !qweights || !lut || !lut_scales) {
        return;
    }

    const int64_t K                = k;
    const int64_t blocks           = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups           = blocks * groups_per_block;

    const block_ifairy * w_blocks  = (const block_ifairy *) qweights;
    const int8_t *       lut_base  = (const int8_t *) lut;
    const float *        scales_in = (const float *) lut_scales;

#if defined(__ARM_NEON) && defined(__aarch64__)
    const int    prefetch_dist   = 2;
    const bool   prefetch        = prefetch_dist > 0;
    const bool   prefetch_index  = false;
    const size_t prefetch_groups = prefetch ? (size_t) prefetch_dist : 0;
    const bool   acc16           = true;
    const bool   n1_fastpath     = true;
    const bool   n1_stream_add   = true;
    const int    unroll          = 8;
    const bool   unroll8_2x4     = false;

    auto load_s16x4 = [](const int8_t * ptr) -> int16x4_t {
        const int32x2_t p   = vld1_dup_s32((const int32_t *) ptr);
        const int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p));
        return vget_low_s16(s16);
    };

    if (n == 1 && n1_fastpath) {
        const int8_t * lut_col = lut_base;
        const float *  scales  = scales_in;

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row   = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t *      idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float32x2_t acc01 = vdup_n_f32(0.0f);  // {ac, ad}
            float32x2_t acc23 = vdup_n_f32(0.0f);  // {bc, bd}

            const uint8_t * idx_blk = idx_row;
            const int8_t *  grp_blk = lut_col;
            const float *   scl_blk = scales;

            for (int64_t blk = 0; blk < blocks; ++blk) {
                int64_t   gi = 0;
                int32x4_t isum;

                if (acc16) {
                    int16x4_t isum16_0 = vdup_n_s16(0);
                    int16x4_t isum16_1 = vdup_n_s16(0);
                    int16x4_t isum16_2 = vdup_n_s16(0);
                    int16x4_t isum16_3 = vdup_n_s16(0);

                    if (n1_stream_add) {
                        if (unroll >= 8) {
                            if (unroll8_2x4) {
                                for (; gi + 7 < groups_per_block; gi += 8) {
                                    if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                                        const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                                        const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                                        if (prefetch_index) {
                                            __builtin_prefetch(idx_blk + gi_pf);
                                        }
                                        const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                                        __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                                    }

                                    const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                                    const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                                    const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                                    const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);
                                    const uint8_t pat4 = (uint8_t) (idx_blk[gi + 4] & 0x3f);
                                    const uint8_t pat5 = (uint8_t) (idx_blk[gi + 5] & 0x3f);
                                    const uint8_t pat6 = (uint8_t) (idx_blk[gi + 6] & 0x3f);
                                    const uint8_t pat7 = (uint8_t) (idx_blk[gi + 7] & 0x3f);

                                    const int8_t * grp0 =
                                        grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp1 =
                                        grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp2 =
                                        grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp3 =
                                        grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp4 =
                                        grp_blk + (size_t) (gi + 4) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp5 =
                                        grp_blk + (size_t) (gi + 5) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp6 =
                                        grp_blk + (size_t) (gi + 6) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp7 =
                                        grp_blk + (size_t) (gi + 7) * k_ifairy_lut_merged64_group_bytes;

                                    isum16_0 = vadd_s16(isum16_0, load_s16x4(grp0 + ((size_t) pat0 << 2)));
                                    isum16_1 = vadd_s16(isum16_1, load_s16x4(grp1 + ((size_t) pat1 << 2)));
                                    isum16_0 = vadd_s16(isum16_0, load_s16x4(grp2 + ((size_t) pat2 << 2)));
                                    isum16_1 = vadd_s16(isum16_1, load_s16x4(grp3 + ((size_t) pat3 << 2)));
                                    isum16_0 = vadd_s16(isum16_0, load_s16x4(grp4 + ((size_t) pat4 << 2)));
                                    isum16_1 = vadd_s16(isum16_1, load_s16x4(grp5 + ((size_t) pat5 << 2)));
                                    isum16_0 = vadd_s16(isum16_0, load_s16x4(grp6 + ((size_t) pat6 << 2)));
                                    isum16_1 = vadd_s16(isum16_1, load_s16x4(grp7 + ((size_t) pat7 << 2)));
                                }
                            } else {
                                for (; gi + 7 < groups_per_block; gi += 8) {
                                    if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                                        const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                                        const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                                        if (prefetch_index) {
                                            __builtin_prefetch(idx_blk + gi_pf);
                                        }
                                        const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                                        __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                                    }

                                    const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                                    const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                                    const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                                    const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);
                                    const uint8_t pat4 = (uint8_t) (idx_blk[gi + 4] & 0x3f);
                                    const uint8_t pat5 = (uint8_t) (idx_blk[gi + 5] & 0x3f);
                                    const uint8_t pat6 = (uint8_t) (idx_blk[gi + 6] & 0x3f);
                                    const uint8_t pat7 = (uint8_t) (idx_blk[gi + 7] & 0x3f);

                                    const int8_t * grp0 =
                                        grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp1 =
                                        grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp2 =
                                        grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp3 =
                                        grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp4 =
                                        grp_blk + (size_t) (gi + 4) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp5 =
                                        grp_blk + (size_t) (gi + 5) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp6 =
                                        grp_blk + (size_t) (gi + 6) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp7 =
                                        grp_blk + (size_t) (gi + 7) * k_ifairy_lut_merged64_group_bytes;

                                    isum16_0 = vadd_s16(isum16_0, load_s16x4(grp0 + ((size_t) pat0 << 2)));
                                    isum16_1 = vadd_s16(isum16_1, load_s16x4(grp1 + ((size_t) pat1 << 2)));
                                    isum16_2 = vadd_s16(isum16_2, load_s16x4(grp2 + ((size_t) pat2 << 2)));
                                    isum16_3 = vadd_s16(isum16_3, load_s16x4(grp3 + ((size_t) pat3 << 2)));
                                    isum16_0 = vadd_s16(isum16_0, load_s16x4(grp4 + ((size_t) pat4 << 2)));
                                    isum16_1 = vadd_s16(isum16_1, load_s16x4(grp5 + ((size_t) pat5 << 2)));
                                    isum16_2 = vadd_s16(isum16_2, load_s16x4(grp6 + ((size_t) pat6 << 2)));
                                    isum16_3 = vadd_s16(isum16_3, load_s16x4(grp7 + ((size_t) pat7 << 2)));
                                }
                            }
                        }

                        for (; gi + 3 < groups_per_block; gi += 4) {
                            if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                                const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                                const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                                if (prefetch_index) {
                                    __builtin_prefetch(idx_blk + gi_pf);
                                }
                                const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                                __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                            }

                            const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                            const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                            const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                            const int8_t * grp0 = grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp1 = grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp2 = grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp3 = grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;

                            isum16_0 = vadd_s16(isum16_0, load_s16x4(grp0 + ((size_t) pat0 << 2)));
                            isum16_1 = vadd_s16(isum16_1, load_s16x4(grp1 + ((size_t) pat1 << 2)));
                            isum16_0 = vadd_s16(isum16_0, load_s16x4(grp2 + ((size_t) pat2 << 2)));
                            isum16_1 = vadd_s16(isum16_1, load_s16x4(grp3 + ((size_t) pat3 << 2)));
                        }
                    } else {
                        if (unroll >= 8) {
                            if (unroll8_2x4) {
                                for (; gi + 7 < groups_per_block; gi += 8) {
                                    if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                                        const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                                        const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                                        if (prefetch_index) {
                                            __builtin_prefetch(idx_blk + gi_pf);
                                        }
                                        const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                                        __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                                    }

                                    const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                                    const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                                    const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                                    const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);
                                    const uint8_t pat4 = (uint8_t) (idx_blk[gi + 4] & 0x3f);
                                    const uint8_t pat5 = (uint8_t) (idx_blk[gi + 5] & 0x3f);
                                    const uint8_t pat6 = (uint8_t) (idx_blk[gi + 6] & 0x3f);
                                    const uint8_t pat7 = (uint8_t) (idx_blk[gi + 7] & 0x3f);

                                    const int8_t * grp0 =
                                        grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp1 =
                                        grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp2 =
                                        grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp3 =
                                        grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp4 =
                                        grp_blk + (size_t) (gi + 4) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp5 =
                                        grp_blk + (size_t) (gi + 5) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp6 =
                                        grp_blk + (size_t) (gi + 6) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp7 =
                                        grp_blk + (size_t) (gi + 7) * k_ifairy_lut_merged64_group_bytes;

                                    const int16x4_t s0 = load_s16x4(grp0 + ((size_t) pat0 << 2));
                                    const int16x4_t s1 = load_s16x4(grp1 + ((size_t) pat1 << 2));
                                    const int16x4_t s2 = load_s16x4(grp2 + ((size_t) pat2 << 2));
                                    const int16x4_t s3 = load_s16x4(grp3 + ((size_t) pat3 << 2));
                                    const int16x4_t s4 = load_s16x4(grp4 + ((size_t) pat4 << 2));
                                    const int16x4_t s5 = load_s16x4(grp5 + ((size_t) pat5 << 2));
                                    const int16x4_t s6 = load_s16x4(grp6 + ((size_t) pat6 << 2));
                                    const int16x4_t s7 = load_s16x4(grp7 + ((size_t) pat7 << 2));

                                    isum16_0 = vadd_s16(isum16_0, s0);
                                    isum16_1 = vadd_s16(isum16_1, s1);
                                    isum16_0 = vadd_s16(isum16_0, s2);
                                    isum16_1 = vadd_s16(isum16_1, s3);
                                    isum16_0 = vadd_s16(isum16_0, s4);
                                    isum16_1 = vadd_s16(isum16_1, s5);
                                    isum16_0 = vadd_s16(isum16_0, s6);
                                    isum16_1 = vadd_s16(isum16_1, s7);
                                }
                            } else {
                                for (; gi + 7 < groups_per_block; gi += 8) {
                                    if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                                        const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                                        const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                                        if (prefetch_index) {
                                            __builtin_prefetch(idx_blk + gi_pf);
                                        }
                                        const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                                        __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                                    }

                                    const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                                    const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                                    const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                                    const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);
                                    const uint8_t pat4 = (uint8_t) (idx_blk[gi + 4] & 0x3f);
                                    const uint8_t pat5 = (uint8_t) (idx_blk[gi + 5] & 0x3f);
                                    const uint8_t pat6 = (uint8_t) (idx_blk[gi + 6] & 0x3f);
                                    const uint8_t pat7 = (uint8_t) (idx_blk[gi + 7] & 0x3f);

                                    const int8_t * grp0 =
                                        grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp1 =
                                        grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp2 =
                                        grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp3 =
                                        grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp4 =
                                        grp_blk + (size_t) (gi + 4) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp5 =
                                        grp_blk + (size_t) (gi + 5) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp6 =
                                        grp_blk + (size_t) (gi + 6) * k_ifairy_lut_merged64_group_bytes;
                                    const int8_t * grp7 =
                                        grp_blk + (size_t) (gi + 7) * k_ifairy_lut_merged64_group_bytes;

                                    const int16x4_t s0 = load_s16x4(grp0 + ((size_t) pat0 << 2));
                                    const int16x4_t s1 = load_s16x4(grp1 + ((size_t) pat1 << 2));
                                    const int16x4_t s2 = load_s16x4(grp2 + ((size_t) pat2 << 2));
                                    const int16x4_t s3 = load_s16x4(grp3 + ((size_t) pat3 << 2));
                                    const int16x4_t s4 = load_s16x4(grp4 + ((size_t) pat4 << 2));
                                    const int16x4_t s5 = load_s16x4(grp5 + ((size_t) pat5 << 2));
                                    const int16x4_t s6 = load_s16x4(grp6 + ((size_t) pat6 << 2));
                                    const int16x4_t s7 = load_s16x4(grp7 + ((size_t) pat7 << 2));

                                    isum16_0 = vadd_s16(isum16_0, s0);
                                    isum16_1 = vadd_s16(isum16_1, s1);
                                    isum16_2 = vadd_s16(isum16_2, s2);
                                    isum16_3 = vadd_s16(isum16_3, s3);
                                    isum16_0 = vadd_s16(isum16_0, s4);
                                    isum16_1 = vadd_s16(isum16_1, s5);
                                    isum16_2 = vadd_s16(isum16_2, s6);
                                    isum16_3 = vadd_s16(isum16_3, s7);
                                }
                            }
                        }

                        for (; gi + 3 < groups_per_block; gi += 4) {
                            if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                                const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                                const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                                if (prefetch_index) {
                                    __builtin_prefetch(idx_blk + gi_pf);
                                }
                                const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                                __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                            }

                            const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                            const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                            const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                            const int8_t * grp0 = grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp1 = grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp2 = grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp3 = grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;

                            const int16x4_t s0 = load_s16x4(grp0 + ((size_t) pat0 << 2));
                            const int16x4_t s1 = load_s16x4(grp1 + ((size_t) pat1 << 2));
                            const int16x4_t s2 = load_s16x4(grp2 + ((size_t) pat2 << 2));
                            const int16x4_t s3 = load_s16x4(grp3 + ((size_t) pat3 << 2));

                            isum16_0 = vadd_s16(isum16_0, s0);
                            isum16_1 = vadd_s16(isum16_1, s1);
                            isum16_0 = vadd_s16(isum16_0, s2);
                            isum16_1 = vadd_s16(isum16_1, s3);
                        }
                    }

                    int16x4_t isum16 = vadd_s16(isum16_0, isum16_1);
                    if (unroll >= 8 && !unroll8_2x4) {
                        isum16 = vadd_s16(isum16, vadd_s16(isum16_2, isum16_3));
                    }
                    for (; gi < groups_per_block; ++gi) {
                        const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                        const int8_t *  grp = grp_blk + (size_t) gi * k_ifairy_lut_merged64_group_bytes;
                        const int16x4_t s   = load_s16x4(grp + ((size_t) pat << 2));
                        isum16              = vadd_s16(isum16, s);
                    }

                    isum = vmovl_s16(isum16);
                } else {
                    int32x4_t isum0 = vdupq_n_s32(0);
                    int32x4_t isum1 = vdupq_n_s32(0);

                    for (; gi + 3 < groups_per_block; gi += 4) {
                        if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                            const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                            const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                            if (prefetch_index) {
                                __builtin_prefetch(idx_blk + gi_pf);
                            }
                            const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                            __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                        }

                        const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                        const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                        const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                        const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                        const int8_t * grp0 = grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp1 = grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp2 = grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp3 = grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;

                        const int32x2_t p0 = vld1_dup_s32((const int32_t *) (grp0 + ((size_t) pat0 << 2)));
                        const int32x2_t p1 = vld1_dup_s32((const int32_t *) (grp1 + ((size_t) pat1 << 2)));
                        const int32x2_t p2 = vld1_dup_s32((const int32_t *) (grp2 + ((size_t) pat2 << 2)));
                        const int32x2_t p3 = vld1_dup_s32((const int32_t *) (grp3 + ((size_t) pat3 << 2)));

                        const int16x8_t s16_0 = vmovl_s8(vreinterpret_s8_s32(p0));
                        const int16x8_t s16_1 = vmovl_s8(vreinterpret_s8_s32(p1));
                        const int16x8_t s16_2 = vmovl_s8(vreinterpret_s8_s32(p2));
                        const int16x8_t s16_3 = vmovl_s8(vreinterpret_s8_s32(p3));

                        isum0 = vaddw_s16(isum0, vget_low_s16(s16_0));
                        isum1 = vaddw_s16(isum1, vget_low_s16(s16_1));
                        isum0 = vaddw_s16(isum0, vget_low_s16(s16_2));
                        isum1 = vaddw_s16(isum1, vget_low_s16(s16_3));
                    }

                    isum = vaddq_s32(isum0, isum1);
                    for (; gi < groups_per_block; ++gi) {
                        const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                        const int8_t *  grp = grp_blk + (size_t) gi * k_ifairy_lut_merged64_group_bytes;
                        const int32x2_t p   = vld1_dup_s32((const int32_t *) (grp + ((size_t) pat << 2)));
                        const int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p));
                        isum                = vaddw_s16(isum, vget_low_s16(s16));
                    }
                }

                const float32x2_t srsi  = vld1_f32(scl_blk);
                const float32x2_t sum01 = vcvt_f32_s32(vget_low_s32(isum));
                const float32x2_t sum23 = vcvt_f32_s32(vget_high_s32(isum));
                acc01                   = vmla_f32(acc01, sum01, srsi);
                acc23                   = vmla_f32(acc23, sum23, srsi);

                idx_blk += groups_per_block;
                grp_blk += (size_t) groups_per_block * k_ifairy_lut_merged64_group_bytes;
                scl_blk += 2;
            }

            const float acc_ac_xr = vget_lane_f32(acc01, 0);
            const float acc_ad_xi = vget_lane_f32(acc01, 1);
            const float acc_bc_xr = vget_lane_f32(acc23, 0);
            const float acc_bd_xi = vget_lane_f32(acc23, 1);

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

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

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = lut_base + (size_t) col * (size_t) groups * k_ifairy_lut_merged64_group_bytes;
        const float *  scales  = scales_in + (size_t) col * (size_t) blocks * 2;

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row   = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t *      idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float32x2_t acc01 = vdup_n_f32(0.0f);  // {ac, ad}
            float32x2_t acc23 = vdup_n_f32(0.0f);  // {bc, bd}

            for (int64_t blk = 0; blk < blocks; ++blk) {
                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t *  grp_blk =
                    lut_col + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_merged64_group_bytes;

                int64_t   gi = 0;
                int32x4_t isum;

                if (acc16) {
                    int16x4_t isum16_0 = vdup_n_s16(0);
                    int16x4_t isum16_1 = vdup_n_s16(0);
                    int16x4_t isum16_2 = vdup_n_s16(0);
                    int16x4_t isum16_3 = vdup_n_s16(0);

                    if (unroll >= 8) {
                        for (; gi + 7 < groups_per_block; gi += 8) {
                            if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                                const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                                const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                                if (prefetch_index) {
                                    __builtin_prefetch(idx_blk + gi_pf);
                                }
                                const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                                __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                            }

                            const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                            const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                            const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                            const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);
                            const uint8_t pat4 = (uint8_t) (idx_blk[gi + 4] & 0x3f);
                            const uint8_t pat5 = (uint8_t) (idx_blk[gi + 5] & 0x3f);
                            const uint8_t pat6 = (uint8_t) (idx_blk[gi + 6] & 0x3f);
                            const uint8_t pat7 = (uint8_t) (idx_blk[gi + 7] & 0x3f);

                            const int8_t * grp0 = grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp1 = grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp2 = grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp3 = grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp4 = grp_blk + (size_t) (gi + 4) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp5 = grp_blk + (size_t) (gi + 5) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp6 = grp_blk + (size_t) (gi + 6) * k_ifairy_lut_merged64_group_bytes;
                            const int8_t * grp7 = grp_blk + (size_t) (gi + 7) * k_ifairy_lut_merged64_group_bytes;

                            const int16x4_t s0 = load_s16x4(grp0 + ((size_t) pat0 << 2));
                            const int16x4_t s1 = load_s16x4(grp1 + ((size_t) pat1 << 2));
                            const int16x4_t s2 = load_s16x4(grp2 + ((size_t) pat2 << 2));
                            const int16x4_t s3 = load_s16x4(grp3 + ((size_t) pat3 << 2));
                            const int16x4_t s4 = load_s16x4(grp4 + ((size_t) pat4 << 2));
                            const int16x4_t s5 = load_s16x4(grp5 + ((size_t) pat5 << 2));
                            const int16x4_t s6 = load_s16x4(grp6 + ((size_t) pat6 << 2));
                            const int16x4_t s7 = load_s16x4(grp7 + ((size_t) pat7 << 2));

                            isum16_0 = vadd_s16(isum16_0, s0);
                            isum16_1 = vadd_s16(isum16_1, s1);
                            isum16_2 = vadd_s16(isum16_2, s2);
                            isum16_3 = vadd_s16(isum16_3, s3);
                            isum16_0 = vadd_s16(isum16_0, s4);
                            isum16_1 = vadd_s16(isum16_1, s5);
                            isum16_2 = vadd_s16(isum16_2, s6);
                            isum16_3 = vadd_s16(isum16_3, s7);
                        }
                    }

                    for (; gi + 3 < groups_per_block; gi += 4) {
                        if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                            const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                            const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                            if (prefetch_index) {
                                __builtin_prefetch(idx_blk + gi_pf);
                            }
                            const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                            __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                        }

                        const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                        const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                        const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                        const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                        const int8_t * grp0 = grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp1 = grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp2 = grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp3 = grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;

                        const int16x4_t s0 = load_s16x4(grp0 + ((size_t) pat0 << 2));
                        const int16x4_t s1 = load_s16x4(grp1 + ((size_t) pat1 << 2));
                        const int16x4_t s2 = load_s16x4(grp2 + ((size_t) pat2 << 2));
                        const int16x4_t s3 = load_s16x4(grp3 + ((size_t) pat3 << 2));

                        isum16_0 = vadd_s16(isum16_0, s0);
                        isum16_1 = vadd_s16(isum16_1, s1);
                        isum16_0 = vadd_s16(isum16_0, s2);
                        isum16_1 = vadd_s16(isum16_1, s3);
                    }

                    int16x4_t isum16 = vadd_s16(isum16_0, isum16_1);
                    if (unroll >= 8) {
                        isum16 = vadd_s16(isum16, vadd_s16(isum16_2, isum16_3));
                    }
                    for (; gi < groups_per_block; ++gi) {
                        const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                        const int8_t *  grp = grp_blk + (size_t) gi * k_ifairy_lut_merged64_group_bytes;
                        const int16x4_t s   = load_s16x4(grp + ((size_t) pat << 2));
                        isum16              = vadd_s16(isum16, s);
                    }

                    isum = vmovl_s16(isum16);
                } else {
                    int32x4_t isum0 = vdupq_n_s32(0);
                    int32x4_t isum1 = vdupq_n_s32(0);

                    for (; gi + 3 < groups_per_block; gi += 4) {
                        if (prefetch_groups && (size_t) gi + prefetch_groups < (size_t) groups_per_block) {
                            const size_t  gi_pf  = (size_t) gi + prefetch_groups;
                            const uint8_t pat_pf = (uint8_t) (idx_blk[gi_pf] & 0x3f);
                            if (prefetch_index) {
                                __builtin_prefetch(idx_blk + gi_pf);
                            }
                            const int8_t * grp_pf = grp_blk + gi_pf * k_ifairy_lut_merged64_group_bytes;
                            __builtin_prefetch(grp_pf + ((size_t) pat_pf << 2));
                        }

                        const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                        const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                        const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                        const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                        const int8_t * grp0 = grp_blk + (size_t) (gi + 0) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp1 = grp_blk + (size_t) (gi + 1) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp2 = grp_blk + (size_t) (gi + 2) * k_ifairy_lut_merged64_group_bytes;
                        const int8_t * grp3 = grp_blk + (size_t) (gi + 3) * k_ifairy_lut_merged64_group_bytes;

                        const int32x2_t p0 = vld1_dup_s32((const int32_t *) (grp0 + ((size_t) pat0 << 2)));
                        const int32x2_t p1 = vld1_dup_s32((const int32_t *) (grp1 + ((size_t) pat1 << 2)));
                        const int32x2_t p2 = vld1_dup_s32((const int32_t *) (grp2 + ((size_t) pat2 << 2)));
                        const int32x2_t p3 = vld1_dup_s32((const int32_t *) (grp3 + ((size_t) pat3 << 2)));

                        const int16x8_t s16_0 = vmovl_s8(vreinterpret_s8_s32(p0));
                        const int16x8_t s16_1 = vmovl_s8(vreinterpret_s8_s32(p1));
                        const int16x8_t s16_2 = vmovl_s8(vreinterpret_s8_s32(p2));
                        const int16x8_t s16_3 = vmovl_s8(vreinterpret_s8_s32(p3));

                        isum0 = vaddw_s16(isum0, vget_low_s16(s16_0));
                        isum1 = vaddw_s16(isum1, vget_low_s16(s16_1));
                        isum0 = vaddw_s16(isum0, vget_low_s16(s16_2));
                        isum1 = vaddw_s16(isum1, vget_low_s16(s16_3));
                    }

                    isum = vaddq_s32(isum0, isum1);
                    for (; gi < groups_per_block; ++gi) {
                        const uint8_t   pat = (uint8_t) (idx_blk[gi] & 0x3f);
                        const int8_t *  grp = grp_blk + (size_t) gi * k_ifairy_lut_merged64_group_bytes;
                        const int32x2_t p   = vld1_dup_s32((const int32_t *) (grp + ((size_t) pat << 2)));
                        const int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p));
                        isum                = vaddw_s16(isum, vget_low_s16(s16));
                    }
                }

                const float32x2_t srsi  = vld1_f32(scales + (size_t) blk * 2u);
                const float32x2_t sum01 = vcvt_f32_s32(vget_low_s32(isum));
                const float32x2_t sum23 = vcvt_f32_s32(vget_high_s32(isum));
                acc01                   = vmla_f32(acc01, sum01, srsi);
                acc23                   = vmla_f32(acc23, sum23, srsi);
            }

            const float acc_ac_xr = vget_lane_f32(acc01, 0);
            const float acc_ad_xi = vget_lane_f32(acc01, 1);
            const float acc_bc_xr = vget_lane_f32(acc23, 0);
            const float acc_bd_xi = vget_lane_f32(acc23, 1);

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

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
#else
    for (int col = 0; col < n; ++col) {
        const int8_t * lut_col = lut_base + (size_t) col * (size_t) groups * k_ifairy_lut_merged64_group_bytes;
        const float *  scales  = scales_in + (size_t) col * (size_t) blocks * 2;

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row   = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t *      idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t *  grp_blk =
                    lut_col + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_merged64_group_bytes;

                for (int64_t gi = 0; gi < groups_per_block; ++gi) {
                    const uint8_t  pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int8_t * tbl = grp_blk + (size_t) gi * k_ifairy_lut_merged64_group_bytes + (size_t) pat * 4u;
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

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

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
#endif
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
    const size_t lut_bytes       = (size_t) n * (size_t) groups * k_ifairy_lut_merged64_group_bytes;
    const size_t scale_bytes     = (size_t) n * (size_t) blocks * 2u * sizeof(float);
    const size_t total_bytes     = index_bytes + lut_bytes + scale_bytes;

    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, total_bytes) != 0) {
        return;
    }

    uint8_t * buf     = (uint8_t *) ptr;
    uint8_t * indexes = buf;
    uint8_t * lut     = buf + index_bytes;
    float *   scales  = (float *) (buf + index_bytes + lut_bytes);

    memset(buf, 0, total_bytes);

    ggml_ifairy_3w_encode((const block_ifairy *) qweights, K, m, indexes, index_bytes_raw);
    ggml_ifairy_lut_preprocess_ex_merged64(m, k, n, act, act_stride, scales, lut, 0, 1);

    const size_t dst_col_stride = (size_t) m * 2u * sizeof(float);
    const size_t dst_row_stride = 2u * sizeof(float);
    ggml_ifairy_lut_qgemm_merged64(m, k, n, qweights, indexes, lut, scales, dst, dst_col_stride, dst_row_stride,
                                   /*pack_bf16*/ false, /*add*/ false);

    free(ptr);
}
