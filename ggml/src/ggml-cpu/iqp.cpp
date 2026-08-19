#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cstdlib>
#include <cstring>

#include "repack.h"

#include "iqp.h"

// ---------------------------------------------------------------------------------------------
// transient "Q8 panel" path for the grid based IQ quants (see iqp.h and block_iqp_x8 in repack.h)
// ---------------------------------------------------------------------------------------------

// src0 rows interleaved per panel
#define IQP_NB_ROWS 8

// ---------------------------------------------------------------------------------------------
// decode primitives
//
// All the grid based types share one inner shape: gather 8 byte (or two 4 byte) grid entries and
// negate the bytes selected by a sign byte. The helpers below do that 32 values at a time; the
// scalar bodies are the original per byte loops and are what non x86 builds compile.
// ---------------------------------------------------------------------------------------------

// the low 7 bits of v are the first 7 signs and the 8th is their parity - which is all the 128 entry
// ksigns_iq2xs table stores, so it can be computed instead of loaded (cf. unpack_ksigns in the CUDA
// backend). Folding the parity with shifts keeps this portable and off the popcount feature macros.
static inline uint8_t iqp_unpack_ksigns(uint32_t v) {
    uint32_t p = v ^ (v >> 4);

    p ^= p >> 2;
    p ^= p >> 1;

    return (uint8_t) (v ^ ((p & 1) << 7));
}

#if defined(__AVX2__)

// 0xFF in every byte whose sign bit is set. sv must hold each sign byte broadcast over the 8 bytes
// it governs; the selector is kmask_iq2xs, one vector wide.
static inline __m256i iqp_sign_mask(__m256i sv) {
    const __m256i sel = _mm256_set1_epi64x((int64_t) 0x8040201008040201ULL);

#    if defined(__GFNI__)
    // gf2p8affine gives bit j of result byte i as parity(sel.byte[i] & sv.byte[j]); with sv.byte[j]
    // constant over the qword every bit of byte i comes out as bit i of the sign byte, i.e. 0 or
    // 0xFF - the whole and + compare in one instruction
    return _mm256_gf2p8affine_epi64_epi8(sel, sv, 0);
#    else
    return _mm256_cmpeq_epi8(_mm256_and_si256(sv, sel), sel);
#    endif
}

// signs holds four sign bytes, byte l governing values 8*l .. 8*l+7 - spread each over its 8 lanes
static inline __m256i iqp_sign_bytes(uint32_t signs) {
    const __m256i bcast = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,  //
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);

    return _mm256_shuffle_epi8(_mm256_set1_epi32((int32_t) signs), bcast);
}

// x ^ m - m negates the lanes of x whose mask byte m is 0xFF and leaves the others alone
static inline __m256i iqp_apply_signs(__m256i x, __m256i m) {
    return _mm256_sub_epi8(_mm256_xor_si256(x, m), m);
}

#endif

// 32 values from four 8 byte grid entries, sign byte l of signs applied to group l
static inline void iqp_store_signed_x8(int8_t * GGML_RESTRICT dst,
                                       uint64_t               g0,
                                       uint64_t               g1,
                                       uint64_t               g2,
                                       uint64_t               g3,
                                       uint32_t               signs) {
#if defined(__AVX2__)
    const __m256i g = _mm256_set_epi64x((int64_t) g3, (int64_t) g2, (int64_t) g1, (int64_t) g0);
    const __m256i m = iqp_sign_mask(iqp_sign_bytes(signs));

    _mm256_storeu_si256((__m256i *) dst, iqp_apply_signs(g, m));
#else
    const uint64_t g[4] = { g0, g1, g2, g3 };

    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *) &g[l];
        const uint8_t   s    = (uint8_t) (signs >> 8 * l);

        for (int j = 0; j < 8; ++j) {
            dst[8 * l + j] = s & kmask_iq2xs[j] ? -grid[j] : grid[j];
        }
    }
#endif
}

// same, but the eight values of group l come from two 4 byte grid entries
static inline void iqp_store_signed_x4(int8_t * GGML_RESTRICT dst,
                                       uint32_t               g0a,
                                       uint32_t               g0b,
                                       uint32_t               g1a,
                                       uint32_t               g1b,
                                       uint32_t               g2a,
                                       uint32_t               g2b,
                                       uint32_t               g3a,
                                       uint32_t               g3b,
                                       uint32_t               signs) {
#if defined(__AVX2__)
    const __m256i g = _mm256_setr_epi32((int32_t) g0a, (int32_t) g0b, (int32_t) g1a, (int32_t) g1b, (int32_t) g2a,
                                        (int32_t) g2b, (int32_t) g3a, (int32_t) g3b);
    const __m256i m = iqp_sign_mask(iqp_sign_bytes(signs));

    _mm256_storeu_si256((__m256i *) dst, iqp_apply_signs(g, m));
#else
    const uint32_t ga[4] = { g0a, g1a, g2a, g3a };
    const uint32_t gb[4] = { g0b, g1b, g2b, g3b };

    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid1 = (const uint8_t *) &ga[l];
        const uint8_t * grid2 = (const uint8_t *) &gb[l];
        const uint8_t   s     = (uint8_t) (signs >> 8 * l);

        for (int j = 0; j < 4; ++j) {
            dst[8 * l + j + 0] = s & kmask_iq2xs[j + 0] ? -grid1[j] : grid1[j];
            dst[8 * l + j + 4] = s & kmask_iq2xs[j + 4] ? -grid2[j] : grid2[j];
        }
    }
#endif
}

// 32 values of 8 * grid + delta from four 8 byte grid entries (grid bytes are in {-1, 0, 1}), byte l
// of deltas applying to group l
static inline void iqp_store_iq1_x8(int8_t * GGML_RESTRICT dst,
                                    uint64_t               g0,
                                    uint64_t               g1,
                                    uint64_t               g2,
                                    uint64_t               g3,
                                    uint32_t               deltas) {
#if defined(__AVX2__)
    __m256i g = _mm256_set_epi64x((int64_t) g3, (int64_t) g2, (int64_t) g1, (int64_t) g0);

    // no byte shift on AVX2, and the grid bytes are tiny, so three doublings are the cheap way
    g = _mm256_add_epi8(g, g);
    g = _mm256_add_epi8(g, g);
    g = _mm256_add_epi8(g, g);

    _mm256_storeu_si256((__m256i *) dst, _mm256_add_epi8(g, iqp_sign_bytes(deltas)));
#else
    const uint64_t g[4] = { g0, g1, g2, g3 };

    for (int l = 0; l < 4; ++l) {
        const int8_t * grid  = (const int8_t *) &g[l];
        const int8_t   delta = (int8_t) (deltas >> 8 * l);

        for (int j = 0; j < 8; ++j) {
            dst[8 * l + j] = 8 * grid[j] + delta;
        }
    }
#endif
}

// 32 values from 16 packed nibbles through the kvalues_iq4nl lookup: low nibbles first, then high
static inline void iqp_store_iq4_x32(int8_t * GGML_RESTRICT dst, const uint8_t * GGML_RESTRICT qs) {
#if defined(__AVX2__)
    const __m128i q   = _mm_loadu_si128((const __m128i *) qs);
    const __m128i lut = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i m4  = _mm_set1_epi8(0xf);

    _mm_storeu_si128((__m128i *) (dst + 0), _mm_shuffle_epi8(lut, _mm_and_si128(q, m4)));
    _mm_storeu_si128((__m128i *) (dst + 16), _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4)));
#else
    for (int j = 0; j < 16; ++j) {
        dst[j + 0]  = kvalues_iq4nl[qs[j] & 0xf];
        dst[j + 16] = kvalues_iq4nl[qs[j] >> 4];
    }
#endif
}

// sum of weight * integer sub-block scale over one decoded super-block. Bounded by
// 256 * 127 * 32 = 1.04e6, and 128 times that is 1.33e8, so the int32 result cannot overflow the
// kernel accumulator (see the bounds in ggml_gemm_iqp_8x8_q8_K).
static inline int32_t iqp_weighted_sum(const int8_t * GGML_RESTRICT vals, const int8_t * GGML_RESTRICT iscales) {
#if defined(__AVX2__)
    static_assert(IQP_SB_SIZE == 16, "the vector path folds two sub-blocks per 32 byte load");

    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    __m256i acc = _mm256_setzero_si256();

    for (int i = 0; i < QK_K / 32; ++i) {
        // maddubs against ones sums byte pairs, madd against ones sums those in fours: eight int32,
        // the low four covering sub-block 2*i and the high four sub-block 2*i + 1
        const __m256i v = _mm256_loadu_si256((const __m256i *) (vals + 32 * i));
        const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, v), ones16);

        const __m256i s = _mm256_set_m128i(_mm_set1_epi32(iscales[2 * i + 1]), _mm_set1_epi32(iscales[2 * i + 0]));

        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(p, s));
    }

    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));

    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));

    return _mm_cvtsi128_si32(sum);
#else
    int32_t wsum = 0;

    for (int sb = 0; sb < IQP_NSB; ++sb) {
        int32_t vsum = 0;

        for (int k = 0; k < IQP_SB_SIZE; ++k) {
            vsum += vals[sb * IQP_SB_SIZE + k];
        }

        wsum += iscales[sb] * vsum;
    }

    return wsum;
#endif
}

// decode one super-block into QK_K signed values, IQP_NSB integer sub-block scales and the one
// float factor they share. The expressions mirror dequantize_row_iq* from ggml-quants.c, split so
// that (*dfac * iscales[sb]) * value is bit identical to the reference dequantized weight - see
// block_iqp_x8 in repack.h for why the split is exact.
static bool iqp_decode_superblock(enum ggml_type             type,
                                  const void * GGML_RESTRICT vx,
                                  int8_t * GGML_RESTRICT     vals,
                                  int8_t * GGML_RESTRICT     iscales,
                                  float * GGML_RESTRICT      dfac) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
            {
                const block_iq2_xxs * x = (const block_iq2_xxs *) vx;

                // db = d * (0.5 + ls) * 0.25 = (d / 8) * (2 * ls + 1), ls 4 bit
                *dfac = GGML_CPU_FP16_TO_FP32(x->d) * 0.125f;

                uint32_t        aux32[2];
                const uint8_t * aux8 = (const uint8_t *) aux32;

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    memcpy(aux32, x->qs + 4 * ib32, 2 * sizeof(uint32_t));
                    const int8_t ls = (int8_t) (2 * (aux32[1] >> 28) + 1);

                    iscales[2 * ib32 + 0] = ls;
                    iscales[2 * ib32 + 1] = ls;

                    const uint32_t signs = (uint32_t) iqp_unpack_ksigns((aux32[1] >> 0) & 127) |
                                           (uint32_t) iqp_unpack_ksigns((aux32[1] >> 7) & 127) << 8 |
                                           (uint32_t) iqp_unpack_ksigns((aux32[1] >> 14) & 127) << 16 |
                                           (uint32_t) iqp_unpack_ksigns((aux32[1] >> 21) & 127) << 24;

                    iqp_store_signed_x8(vals + 32 * ib32, iq2xxs_grid[aux8[0]], iq2xxs_grid[aux8[1]],
                                        iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[3]], signs);
                }
            }
            break;
        case GGML_TYPE_IQ2_XS:
            {
                const block_iq2_xs * x = (const block_iq2_xs *) vx;

                *dfac = GGML_CPU_FP16_TO_FP32(x->d) * 0.125f;

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    iscales[2 * ib32 + 0] = (int8_t) (2 * (x->scales[ib32] & 0xf) + 1);
                    iscales[2 * ib32 + 1] = (int8_t) (2 * (x->scales[ib32] >> 4) + 1);

                    const uint16_t * q = x->qs + 4 * ib32;

                    const uint32_t signs = (uint32_t) iqp_unpack_ksigns(q[0] >> 9) |
                                           (uint32_t) iqp_unpack_ksigns(q[1] >> 9) << 8 |
                                           (uint32_t) iqp_unpack_ksigns(q[2] >> 9) << 16 |
                                           (uint32_t) iqp_unpack_ksigns(q[3] >> 9) << 24;

                    iqp_store_signed_x8(vals + 32 * ib32, iq2xs_grid[q[0] & 511], iq2xs_grid[q[1] & 511],
                                        iq2xs_grid[q[2] & 511], iq2xs_grid[q[3] & 511], signs);
                }
            }
            break;
        case GGML_TYPE_IQ2_S:
            {
                const block_iq2_s * x = (const block_iq2_s *) vx;

                const uint8_t * qs    = x->qs;
                const uint8_t * qh    = x->qh;
                const uint8_t * signs = qs + QK_K / 8;

                *dfac = GGML_CPU_FP16_TO_FP32(x->d) * 0.125f;

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    iscales[2 * ib32 + 0] = (int8_t) (2 * (x->scales[ib32] & 0xf) + 1);
                    iscales[2 * ib32 + 1] = (int8_t) (2 * (x->scales[ib32] >> 4) + 1);

                    // the sign bytes are stored whole here, so there is no parity to recover
                    const uint32_t sbits = (uint32_t) signs[0] | (uint32_t) signs[1] << 8 |
                                           (uint32_t) signs[2] << 16 | (uint32_t) signs[3] << 24;

                    iqp_store_signed_x8(vals + 32 * ib32, iq2s_grid[qs[0] | (qh[ib32] << 8 & 0x300)],
                                        iq2s_grid[qs[1] | (qh[ib32] << 6 & 0x300)],
                                        iq2s_grid[qs[2] | (qh[ib32] << 4 & 0x300)],
                                        iq2s_grid[qs[3] | (qh[ib32] << 2 & 0x300)], sbits);
                    qs += 4;
                    signs += 4;
                }
            }
            break;
        case GGML_TYPE_IQ3_XXS:
            {
                const block_iq3_xxs * x = (const block_iq3_xxs *) vx;

                const uint8_t * qs               = x->qs;
                const uint8_t * scales_and_signs = qs + QK_K / 4;

                // db = d * (0.5 + ls) * 0.5 = (d / 4) * (2 * ls + 1), ls 4 bit
                *dfac = GGML_CPU_FP16_TO_FP32(x->d) * 0.25f;

                uint32_t aux32;

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
                    const int8_t ls = (int8_t) (2 * (aux32 >> 28) + 1);

                    iscales[2 * ib32 + 0] = ls;
                    iscales[2 * ib32 + 1] = ls;

                    const uint32_t signs = (uint32_t) iqp_unpack_ksigns((aux32 >> 0) & 127) |
                                           (uint32_t) iqp_unpack_ksigns((aux32 >> 7) & 127) << 8 |
                                           (uint32_t) iqp_unpack_ksigns((aux32 >> 14) & 127) << 16 |
                                           (uint32_t) iqp_unpack_ksigns((aux32 >> 21) & 127) << 24;

                    iqp_store_signed_x4(vals + 32 * ib32, iq3xxs_grid[qs[0]], iq3xxs_grid[qs[1]], iq3xxs_grid[qs[2]],
                                        iq3xxs_grid[qs[3]], iq3xxs_grid[qs[4]], iq3xxs_grid[qs[5]],
                                        iq3xxs_grid[qs[6]], iq3xxs_grid[qs[7]], signs);
                    qs += 8;
                }
            }
            break;
        case GGML_TYPE_IQ3_S:
            {
                const block_iq3_s * x = (const block_iq3_s *) vx;

                const uint8_t * qs    = x->qs;
                const uint8_t * qh    = x->qh;
                const uint8_t * signs = x->signs;

                // db = d * (1 + 2 * ls), ls 4 bit
                *dfac = GGML_CPU_FP16_TO_FP32(x->d);

                int k = 0;

                for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                    const int8_t db1 = (int8_t) (1 + 2 * (x->scales[ib32 / 2] & 0xf));
                    const int8_t db2 = (int8_t) (1 + 2 * (x->scales[ib32 / 2] >> 4));

                    iscales[2 * ib32 + 0] = db1;
                    iscales[2 * ib32 + 1] = db1;
                    iscales[2 * ib32 + 2] = db2;
                    iscales[2 * ib32 + 3] = db2;

                    // the two halves of the 64 weights differ only in which qh byte supplies the
                    // 9th bit of each grid index
                    for (int h = 0; h < 2; ++h) {
                        const uint32_t sbits = (uint32_t) signs[0] | (uint32_t) signs[1] << 8 |
                                               (uint32_t) signs[2] << 16 | (uint32_t) signs[3] << 24;

                        iqp_store_signed_x4(vals + k, iq3s_grid[qs[0] | ((qh[h] << 8) & 256)],
                                            iq3s_grid[qs[1] | ((qh[h] << 7) & 256)],
                                            iq3s_grid[qs[2] | ((qh[h] << 6) & 256)],
                                            iq3s_grid[qs[3] | ((qh[h] << 5) & 256)],
                                            iq3s_grid[qs[4] | ((qh[h] << 4) & 256)],
                                            iq3s_grid[qs[5] | ((qh[h] << 3) & 256)],
                                            iq3s_grid[qs[6] | ((qh[h] << 2) & 256)],
                                            iq3s_grid[qs[7] | ((qh[h] << 1) & 256)], sbits);

                        k += 32;
                        qs += 8;
                        signs += 4;
                    }
                    qh += 2;
                }
            }
            break;
        // the two iq1 types do not dequantize as scale * value: dequantize_row_iq1_* computes
        // y = dl * (grid[j] + delta) with grid[j] in {-1, 0, 1} and delta = +-1/8. Scaling the
        // sub-block scale down by 8 absorbs the delta:
        //
        //     y = (dl / 8) * (8 * grid[j] +- 1),   8 * grid[j] +- 1 in {-9, -7, -1, 1, 7, 9}
        //
        // which fits int8. This is bit identical to the reference and not merely close: dl is
        // d (fp16, 11 bit mantissa) times a small odd integer (at most 15), so the product is
        // exact in f32 and dividing it by the power of two 8 stays exact; grid[j] +- 1/8 is exact
        // too. Both expressions therefore round the same real number exactly once.
        case GGML_TYPE_IQ1_S:
            {
                const block_iq1_s * x = (const block_iq1_s *) vx;

                const uint8_t *  qs = x->qs;
                const uint16_t * qh = x->qh;

                // dl = d * (2 * ls + 1) * 0.125, ls 3 bit - the /8 of the delta trick is the 2^-k
                *dfac = GGML_CPU_FP16_TO_FP32(x->d) * 0.125f;

                for (int ib = 0; ib < QK_K / 32; ++ib) {
                    // one scale and one delta sign per 32 weights, so both sub-blocks share them
                    const int8_t dl    = (int8_t) (2 * ((qh[ib] >> 12) & 7) + 1);
                    const int8_t delta = qh[ib] & 0x8000 ? -1 : 1;

                    iscales[2 * ib + 0] = dl;
                    iscales[2 * ib + 1] = dl;

                    iqp_store_iq1_x8(vals + 32 * ib, iq1s_grid[qs[0] | (((qh[ib] >> 0) & 7) << 8)],
                                     iq1s_grid[qs[1] | (((qh[ib] >> 3) & 7) << 8)],
                                     iq1s_grid[qs[2] | (((qh[ib] >> 6) & 7) << 8)],
                                     iq1s_grid[qs[3] | (((qh[ib] >> 9) & 7) << 8)],
                                     ((uint8_t) delta) * 0x01010101u);
                    qs += 4;
                }
            }
            break;
        case GGML_TYPE_IQ1_M:
            {
                const block_iq1_m * x = (const block_iq1_m *) vx;

                // block_iq1_m has no d field - the fp16 super-block scale is spread over the top
                // nibbles of the four scale words
                const uint16_t * sc = (const uint16_t *) x->scales;

                iq1m_scale_t scale;
                scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

                *dfac = GGML_CPU_FP16_TO_FP32(scale.f16) * 0.125f;

                const uint8_t * qs = x->qs;
                const uint8_t * qh = x->qh;

                for (int ib = 0; ib < QK_K / 32; ++ib) {
                    // 3 bit scales are per 16 weights here, i.e. one per sub-block, and the delta
                    // signs are per group of 8 - the +-1 is folded per value, so that is free
                    iscales[2 * ib + 0] = (int8_t) (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
                    iscales[2 * ib + 1] = (int8_t) (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);

                    const uint16_t idx[4] = {
                        (uint16_t) (qs[0] | ((qh[0] << 8) & 0x700)),
                        (uint16_t) (qs[1] | ((qh[0] << 4) & 0x700)),
                        (uint16_t) (qs[2] | ((qh[1] << 8) & 0x700)),
                        (uint16_t) (qs[3] | ((qh[1] << 4) & 0x700)),
                    };
                    const uint32_t deltas = (uint32_t) (qh[0] & 0x08 ? 0xff : 0x01) |
                                            (uint32_t) (qh[0] & 0x80 ? 0xff : 0x01) << 8 |
                                            (uint32_t) (qh[1] & 0x08 ? 0xff : 0x01) << 16 |
                                            (uint32_t) (qh[1] & 0x80 ? 0xff : 0x01) << 24;

                    iqp_store_iq1_x8(vals + 32 * ib, iq1s_grid[idx[0]], iq1s_grid[idx[1]], iq1s_grid[idx[2]],
                                     iq1s_grid[idx[3]], deltas);
                    qs += 4;
                    qh += 2;
                }
            }
            break;
        case GGML_TYPE_IQ4_XS:
            {
                const block_iq4_xs * x = (const block_iq4_xs *) vx;

                const uint8_t * qs = x->qs;

                // dl = d * (ls - 32), ls 6 bit, so the integer scale is in [-32, 31]
                *dfac = GGML_CPU_FP16_TO_FP32(x->d);

                for (int ib = 0; ib < QK_K / 32; ++ib) {
                    const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
                    const int8_t dl = (int8_t) (ls - 32);

                    iscales[2 * ib + 0] = dl;
                    iscales[2 * ib + 1] = dl;

                    iqp_store_iq4_x32(vals + 32 * ib, qs);
                    qs += 16;
                }
            }
            break;
        default:
            return false;
    }

    return true;
}

// decode IQP_NB_ROWS consecutive source rows (starting at src, row stride nb01) into a panel of
// nblocks block_iqp_x8
static void iqp_decode_panel_8(enum ggml_type               type,
                               const char * GGML_RESTRICT   src,
                               size_t                       nb01,
                               int64_t                      nblocks,
                               block_iqp_x8 * GGML_RESTRICT dst) {
    const size_t bsize = ggml_type_size(type);

    int8_t vals[IQP_NB_ROWS][QK_K];
    int8_t iscales[IQP_NB_ROWS][IQP_NSB];
    float  dfac[IQP_NB_ROWS];

    for (int64_t x = 0; x < nblocks; x++) {
        for (int r = 0; r < IQP_NB_ROWS; r++) {
            const char * blk = src + r * nb01 + x * bsize;

            const bool ok = iqp_decode_superblock(type, blk, vals[r], iscales[r], &dfac[r]);
            GGML_ASSERT(ok);

#ifdef GGML_IQP_VERIFY
            // debug aid: assert that the panel reproduces the reference dequantization bit exactly
            float ref[QK_K];
            ggml_get_type_traits(type)->to_float(blk, ref, QK_K);
            for (int j = 0; j < QK_K; j++) {
                const float scale = dfac[r] * iscales[r][j / IQP_SB_SIZE];
                GGML_ASSERT(scale * vals[r][j] == ref[j]);
            }
#endif
        }

        for (int r = 0; r < IQP_NB_ROWS; r++) {
            dst->dfac[r] = dfac[r];

            const int32_t wsum = iqp_weighted_sum(vals[r], iscales[r]);

            for (int sb = 0; sb < IQP_NSB; sb++) {
                dst->iscales[sb * IQP_NB_ROWS + r] = iscales[r][sb];

                // both sides are contiguous over k, so the interleave is a run of dword copies
                for (int g = 0; g < IQP_SB_SIZE / 4; g++) {
                    memcpy(dst->qs + sb * 128 + g * 32 + r * 4, vals[r] + sb * IQP_SB_SIZE + g * 4, 4);
                }
            }

            // the gemm feeds the activations as unsigned bytes, see block_iqp_x8
            dst->bias[r] = 128 * wsum;
        }

        dst++;
    }
}

static bool iqp_type_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// the checks MUL_MAT and MUL_MAT_ID have in common; each caller adds its own batch size test and
// its own src0 shape test on top
static bool iqp_supported_common(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (!iqp_type_supported(src0->type)) {
        return false;
    }

    // the whole path (gather offsets included) assumes the src1 conversion is q8_K
    GGML_ASSERT(ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_Q8_K);

    // escape hatch to A/B the panel against the plain vec_dot path without rebuilding, read once
    // (this path is not a repack buffer, so --no-repack does not cover it, and llama-bench has no
    // equivalent flag at all)
    static const bool disabled = getenv("GGML_NO_IQ_PANEL") != nullptr;
    if (disabled) {
        return false;
    }

    if (!ggml_cpu_has_avx2()) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] % QK_K != 0 || src0->ne[1] % IQP_NB_ROWS != 0) {
        return false;
    }

    if (src0->ne[3] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src0)) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float)) {
        return false;
    }

    return true;
}

bool ggml_cpu_iqp_supported_mul_mat(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (!iqp_supported_common(dst)) {
        return false;
    }

    // small batches stay on the vec_dot path - the decode would not pay for itself
    if (src1->ne[1] < GGML_IQP_MIN_BATCH) {
        return false;
    }

    // plain 2D weight matmuls only (src1 may still be batched over ne12)
    if (src0->ne[2] != 1) {
        return false;
    }

    return true;
}

bool ggml_cpu_iqp_supported_mul_mat_id(const struct ggml_tensor * dst) {
    // src0 is the 3D expert stack here, one matrix per expert, so ne[2] is not constrained
    const struct ggml_tensor * ids = dst->src[2];

    if (!iqp_supported_common(dst)) {
        return false;
    }

    // no expert can clear the per expert threshold if the whole node routes fewer rows than that.
    // Checking it here keeps token generation off the path entirely, work buffer included.
    if (!ggml_cpu_iqp_expert_eligible(ids->ne[0] * ids->ne[1])) {
        return false;
    }

    return true;
}

size_t ggml_cpu_iqp_src1_conv_size(const struct ggml_tensor * dst) {
    const enum ggml_type vec_dot_type = ggml_get_type_traits_cpu(dst->src[0]->type)->vec_dot_type;

    return ggml_row_size(vec_dot_type, ggml_nelements(dst->src[1]));
}

size_t ggml_cpu_iqp_id_gather_size(const struct ggml_tensor * dst) {
    const struct ggml_tensor * ids = dst->src[2];

    return ids->ne[0] * ids->ne[1] * ggml_row_size(GGML_TYPE_Q8_K, dst->src[1]->ne[0]);
}

// the gemm kernels read src1 rows at a fixed stride, so each expert's rows - scattered all over
// the q8_K conversion area - are first copied into one contiguous run
void ggml_cpu_iqp_gather_mul_mat_id(const struct ggml_compute_params * params,
                                    const struct ggml_tensor *         dst,
                                    const int64_t *                    matrix_row_counts,
                                    const int32_t *                    matrix_rows,
                                    void *                             gathered) {
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t n_as         = dst->src[0]->ne[2];
    const int64_t n_rows_total = ids->ne[0] * ids->ne[1];
    const int64_t ne11         = src1->ne[1];

    const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_K, src1->ne[0]);

    const char * wdata = (const char *) params->wdata;

    // the destination run of an expert is the packed prefix over the experts that are gated in, in
    // ascending order - exactly the order the caller walks them in, so it never needs a prefix sum
    int64_t off = 0;

    for (int64_t cur_a = 0; cur_a < n_as; cur_a++) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (!ggml_cpu_iqp_expert_eligible(cne1)) {
            continue;
        }

        const int32_t * expert_rows = matrix_rows + 2 * cur_a * n_rows_total;

        // spread over the flat row index so that experts smaller than nth still keep every thread busy
        for (int64_t k = 0; k < cne1; k++) {
            if ((off + k) % nth != ith) {
                continue;
            }

            const int64_t i11 = expert_rows[2 * k + 0] % ne11;
            const int64_t i12 = expert_rows[2 * k + 1];

            memcpy((char *) gathered + (off + k) * nbw1, wdata + (i11 + i12 * ne11) * nbw1, nbw1);
        }

        off += cne1;
    }
}

void ggml_compute_forward_mul_mat_id_iqp(const struct ggml_compute_params * params,
                                         struct ggml_tensor *               dst,
                                         int64_t                            cur_a,
                                         int64_t                            cne1,
                                         const int32_t *                    expert_rows,
                                         const void *                       gathered,
                                         void *                             panels) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nblocks = ne00 / QK_K;

    const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_K, ne10);

    block_iqp_x8 * panel = (block_iqp_x8 *) ((char *) panels + (size_t) ith * ggml_cpu_iqp_scratch_size(dst));

    const char * src0_cur = (const char *) src0->data + cur_a * nb02;

    // static split: every 8 row group is multiplied against all of this expert's routed rows, so the
    // result does not depend on how the groups are handed out
    const int64_t ngroups = ne01 / IQP_NB_ROWS;

    const int64_t g0 = (ngroups * ith) / nth;
    const int64_t g1 = (ngroups * (ith + 1)) / nth;

    for (int64_t g = g0; g < g1; g++) {
        const int64_t r = g * IQP_NB_ROWS;

        iqp_decode_panel_8(src0->type, src0_cur + r * nb01, nb01, nblocks, panel);

        // the dst columns of the routed rows are scattered (one per (expert slot, token) pair), so
        // the kernels write a small contiguous tile that is then copied out
        float tmp[4 * IQP_NB_ROWS];

        int64_t k = 0;

        for (; k + 4 <= cne1; k += 4) {
            ggml_gemm_iqp_8x8_q8_K(ne00, tmp, IQP_NB_ROWS, panel, (const char *) gathered + k * nbw1, 4, IQP_NB_ROWS);

            for (int64_t m = 0; m < 4; m++) {
                float * dst_col = (float *) ((char *) dst->data + expert_rows[2 * (k + m) + 0] * nb1 +
                                             expert_rows[2 * (k + m) + 1] * nb2);
                memcpy(dst_col + r, tmp + m * IQP_NB_ROWS, IQP_NB_ROWS * sizeof(float));
            }
        }

        for (; k < cne1; k++) {
            ggml_gemv_iqp_8x8_q8_K(ne00, tmp, IQP_NB_ROWS, panel, (const char *) gathered + k * nbw1, 1, IQP_NB_ROWS);

            float * dst_col =
                (float *) ((char *) dst->data + expert_rows[2 * k + 0] * nb1 + expert_rows[2 * k + 1] * nb2);
            memcpy(dst_col + r, tmp, IQP_NB_ROWS * sizeof(float));
        }
    }
}

// the panel scratch lives past the q8_K conversion of src1 in the work buffer. ggml_graph_plan
// sizes the buffer with these two helpers and the compute pass addresses it with them, so the two
// cannot drift.
size_t ggml_cpu_iqp_scratch_offset(const struct ggml_tensor * dst) {
    return GGML_PAD(ggml_cpu_iqp_src1_conv_size(dst), 64);
}

size_t ggml_cpu_iqp_scratch_size(const struct ggml_tensor * dst) {
    return GGML_PAD((dst->src[0]->ne[0] / QK_K) * sizeof(block_iqp_x8), 64);
}

void ggml_compute_forward_mul_mat_iqp(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nblocks = ne00 / QK_K;

    // src1 has already been converted to plain (non interleaved) q8_K rows in the work buffer
    const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_K, ne10);
    const size_t nbw2 = nbw1 * ne11;

    const size_t scratch_size = ggml_cpu_iqp_scratch_size(dst);

    GGML_ASSERT(ggml_cpu_iqp_scratch_offset(dst) + (size_t) nth * scratch_size <= params->wsize);

    block_iqp_x8 * panel =
        (block_iqp_x8 *) ((char *) params->wdata + ggml_cpu_iqp_scratch_offset(dst) + (size_t) ith * scratch_size);

    // every row group is multiplied against the whole src1 batch, so the gemm / gemv split - and
    // with it the per output element operation sequence - does not depend on the chunking
    const int64_t nrows = ne11;

    const int64_t ngroups = ne01 / IQP_NB_ROWS;

    // aim for 4 chunks per thread; the caller has already reset the chunk counter to nth and
    // synchronized the threads
    const int64_t groups_per_chunk = MAX(1, (ngroups + nth * 4 - 1) / (nth * 4));
    const int64_t nchunk           = (ngroups + groups_per_chunk - 1) / groups_per_chunk;

    int current_chunk = ith;

    while (current_chunk < nchunk) {
        const int64_t g0 = current_chunk * groups_per_chunk;
        const int64_t g1 = MIN(g0 + groups_per_chunk, ngroups);

        for (int64_t g = g0; g < g1; g++) {
            const int64_t r = g * IQP_NB_ROWS;

            // src0 is 2D here, so one decode of the row group serves every src1 batch
            iqp_decode_panel_8(src0->type, (const char *) src0->data + r * nb01, nb01, nblocks, panel);

            for (int64_t i12 = 0; i12 < ne12; i12++) {
                const char * src1_ptr = (const char *) params->wdata + i12 * nbw2;
                char *       dst_ptr  = (char *) dst->data + i12 * nb2;

                // If there are more than three rows in src1, use gemm; otherwise, use gemv.
                if (nrows > 3) {
                    ggml_gemm_iqp_8x8_q8_K(ne00, (float *) dst_ptr + r, nb1 / nb0, panel, src1_ptr, nrows - (nrows % 4),
                                           IQP_NB_ROWS);
                }
                for (int64_t iter = nrows - (nrows % 4); iter < nrows; iter++) {
                    ggml_gemv_iqp_8x8_q8_K(ne00, (float *) (dst_ptr + iter * nb1) + r, ne01, panel,
                                           src1_ptr + nbw1 * iter, 1 /* nrows */, IQP_NB_ROWS);
                }
            }
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}
