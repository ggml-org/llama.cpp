#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "iqp.h"

#define UNUSED GGML_UNUSED

// smallest src1 batch for which the decode pays for itself
#define GGML_IQP_MIN_BATCH 8

// same, per expert, for MUL_MAT_ID
#define GGML_IQP_MIN_BATCH_ID 8

bool ggml_cpu_iqp_mul_mat_id_min_batch(int64_t cne1) {
    return cne1 >= GGML_IQP_MIN_BATCH_ID;
}

// src0 rows interleaved per panel
#define IQP_NB_ROWS 8

#define IQP_SB_SIZE 16                    // weights per sub-block
#define IQP_NSB     (QK_K / IQP_SB_SIZE)  // sub-blocks per super-block

// one super-block of a grid based IQ type decoded to int8, 8 rows interleaved:
// dfac[row] * iscales[sb*8 + row] * qs is bit identical to dequantize_row_iq*
struct block_iqp_x8 {
    float   dfac[8];               // f32 super-block scale, d * 2^-k
    int32_t bias[8];               // 128 * sum(qs * iscale), see GGML_IQP_USE_BIAS
    int8_t  iscales[IQP_NSB * 8];  // integer sub-block scales, in [-32, 31]
    int8_t  qs[QK_K * 8];          // qs[sb*128 + g*32 + row*4 + k] = column sb*16 + g*4 + k
};

static_assert(sizeof(block_iqp_x8) == 8 * sizeof(float) + 8 * sizeof(int32_t) + IQP_NSB * 8 + QK_K * 8,
              "wrong iqp_x8 block size/padding");

// feed the activations to VNNI as unsigned bytes (y + 128) and correct with bias[]; without VNNI the kernels use the maddubs sign trick instead and bias[] is not filled
#if defined(__AVX2__) && ((defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__))
#    define GGML_IQP_USE_BIAS 1
#else
#    define GGML_IQP_USE_BIAS 0
#endif

static inline size_t ggml_cpu_iqp_row_size(const struct ggml_tensor * dst) {
    return ggml_row_size(GGML_TYPE_Q8_K, dst->src[1]->ne[0]);
}

// the low 7 bits of v are the first 7 signs and the 8th is their parity (cf. unpack_ksigns in the CUDA backend)
static inline uint8_t iqp_unpack_ksigns(uint32_t v) {
    uint32_t p = v ^ (v >> 4);

    p ^= p >> 2;
    p ^= p >> 1;

    return (uint8_t) (v ^ ((p & 1) << 7));
}

#if defined(__AVX2__)

// 0xFF in every byte whose sign bit is set; sv holds each sign byte broadcast over the 8 bytes it governs
static inline __m256i iqp_sign_mask(__m256i sv) {
    const __m256i sel = _mm256_set1_epi64x((int64_t) 0x8040201008040201ULL);

#    if defined(__GFNI__)
    // computes the and + compare in one instruction
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

// x ^ m - m negates the lanes where m is 0xFF
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

// 32 values of 8 * grid + delta from four 8 byte grid entries (grid bytes are in {-1, 0, 1}), byte l of deltas applying to group l
static inline void iqp_store_iq1_x8(int8_t * GGML_RESTRICT dst,
                                    uint64_t               g0,
                                    uint64_t               g1,
                                    uint64_t               g2,
                                    uint64_t               g3,
                                    uint32_t               deltas) {
#if defined(__AVX2__)
    __m256i g = _mm256_set_epi64x((int64_t) g3, (int64_t) g2, (int64_t) g1, (int64_t) g0);

    // no byte shift in AVX2
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

#if GGML_IQP_USE_BIAS

// sum of qs * iscale over one super-block, at most 256 * 127 * 32 = 1.04e6
static inline int32_t iqp_weighted_sum(const int8_t * GGML_RESTRICT vals, const int8_t * GGML_RESTRICT iscales) {
#if defined(__AVX2__)
    static_assert(IQP_SB_SIZE == 16, "the vector path folds two sub-blocks per 32 byte load");

    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    __m256i acc = _mm256_setzero_si256();

    for (int i = 0; i < QK_K / 32; ++i) {
        // sum groups of 4 bytes into int32, the low four lanes cover sub-block 2*i and the high four 2*i + 1
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

#endif  // GGML_IQP_USE_BIAS

static void iqp_decode_iq2_xxs(const void * GGML_RESTRICT vx,
                               int8_t * GGML_RESTRICT     vals,
                               int8_t * GGML_RESTRICT     iscales,
                               float * GGML_RESTRICT      dfac) {
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

        iqp_store_signed_x8(vals + 32 * ib32, iq2xxs_grid[aux8[0]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[2]],
                            iq2xxs_grid[aux8[3]], signs);
    }
}

static void iqp_decode_iq2_xs(const void * GGML_RESTRICT vx,
                              int8_t * GGML_RESTRICT     vals,
                              int8_t * GGML_RESTRICT     iscales,
                              float * GGML_RESTRICT      dfac) {
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    *dfac = GGML_CPU_FP16_TO_FP32(x->d) * 0.125f;

    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        iscales[2 * ib32 + 0] = (int8_t) (2 * (x->scales[ib32] & 0xf) + 1);
        iscales[2 * ib32 + 1] = (int8_t) (2 * (x->scales[ib32] >> 4) + 1);

        const uint16_t * q = x->qs + 4 * ib32;

        const uint32_t signs = (uint32_t) iqp_unpack_ksigns(q[0] >> 9) | (uint32_t) iqp_unpack_ksigns(q[1] >> 9) << 8 |
                               (uint32_t) iqp_unpack_ksigns(q[2] >> 9) << 16 |
                               (uint32_t) iqp_unpack_ksigns(q[3] >> 9) << 24;

        iqp_store_signed_x8(vals + 32 * ib32, iq2xs_grid[q[0] & 511], iq2xs_grid[q[1] & 511], iq2xs_grid[q[2] & 511],
                            iq2xs_grid[q[3] & 511], signs);
    }
}

static void iqp_decode_iq2_s(const void * GGML_RESTRICT vx,
                             int8_t * GGML_RESTRICT     vals,
                             int8_t * GGML_RESTRICT     iscales,
                             float * GGML_RESTRICT      dfac) {
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const uint8_t * qs    = x->qs;
    const uint8_t * qh    = x->qh;
    const uint8_t * signs = qs + QK_K / 8;

    *dfac = GGML_CPU_FP16_TO_FP32(x->d) * 0.125f;

    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        iscales[2 * ib32 + 0] = (int8_t) (2 * (x->scales[ib32] & 0xf) + 1);
        iscales[2 * ib32 + 1] = (int8_t) (2 * (x->scales[ib32] >> 4) + 1);

        const uint32_t sbits =
            (uint32_t) signs[0] | (uint32_t) signs[1] << 8 | (uint32_t) signs[2] << 16 | (uint32_t) signs[3] << 24;

        iqp_store_signed_x8(vals + 32 * ib32, iq2s_grid[qs[0] | (qh[ib32] << 8 & 0x300)],
                            iq2s_grid[qs[1] | (qh[ib32] << 6 & 0x300)], iq2s_grid[qs[2] | (qh[ib32] << 4 & 0x300)],
                            iq2s_grid[qs[3] | (qh[ib32] << 2 & 0x300)], sbits);
        qs += 4;
        signs += 4;
    }
}

static void iqp_decode_iq3_xxs(const void * GGML_RESTRICT vx,
                               int8_t * GGML_RESTRICT     vals,
                               int8_t * GGML_RESTRICT     iscales,
                               float * GGML_RESTRICT      dfac) {
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
                            iq3xxs_grid[qs[3]], iq3xxs_grid[qs[4]], iq3xxs_grid[qs[5]], iq3xxs_grid[qs[6]],
                            iq3xxs_grid[qs[7]], signs);
        qs += 8;
    }
}

static void iqp_decode_iq3_s(const void * GGML_RESTRICT vx,
                             int8_t * GGML_RESTRICT     vals,
                             int8_t * GGML_RESTRICT     iscales,
                             float * GGML_RESTRICT      dfac) {
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

        for (int h = 0; h < 2; ++h) {
            const uint32_t sbits =
                (uint32_t) signs[0] | (uint32_t) signs[1] << 8 | (uint32_t) signs[2] << 16 | (uint32_t) signs[3] << 24;

            iqp_store_signed_x4(vals + k, iq3s_grid[qs[0] | ((qh[h] << 8) & 256)],
                                iq3s_grid[qs[1] | ((qh[h] << 7) & 256)], iq3s_grid[qs[2] | ((qh[h] << 6) & 256)],
                                iq3s_grid[qs[3] | ((qh[h] << 5) & 256)], iq3s_grid[qs[4] | ((qh[h] << 4) & 256)],
                                iq3s_grid[qs[5] | ((qh[h] << 3) & 256)], iq3s_grid[qs[6] | ((qh[h] << 2) & 256)],
                                iq3s_grid[qs[7] | ((qh[h] << 1) & 256)], sbits);

            k += 32;
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

// dequantize_row_iq1_* computes y = dl * (grid[j] + delta) with delta = +-1/8, so the panel stores 8 * grid[j] +- 1 and folds the /8 into dfac
static void iqp_decode_iq1_s(const void * GGML_RESTRICT vx,
                             int8_t * GGML_RESTRICT     vals,
                             int8_t * GGML_RESTRICT     iscales,
                             float * GGML_RESTRICT      dfac) {
    const block_iq1_s * x = (const block_iq1_s *) vx;

    const uint8_t *  qs = x->qs;
    const uint16_t * qh = x->qh;

    // dl = d * (2 * ls + 1) * 0.125, ls 3 bit
    *dfac = GGML_CPU_FP16_TO_FP32(x->d) * 0.125f;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int8_t dl    = (int8_t) (2 * ((qh[ib] >> 12) & 7) + 1);
        const int8_t delta = qh[ib] & 0x8000 ? -1 : 1;

        iscales[2 * ib + 0] = dl;
        iscales[2 * ib + 1] = dl;

        iqp_store_iq1_x8(vals + 32 * ib, iq1s_grid[qs[0] | (((qh[ib] >> 0) & 7) << 8)],
                         iq1s_grid[qs[1] | (((qh[ib] >> 3) & 7) << 8)], iq1s_grid[qs[2] | (((qh[ib] >> 6) & 7) << 8)],
                         iq1s_grid[qs[3] | (((qh[ib] >> 9) & 7) << 8)], ((uint8_t) delta) * 0x01010101u);
        qs += 4;
    }
}

static void iqp_decode_iq1_m(const void * GGML_RESTRICT vx,
                             int8_t * GGML_RESTRICT     vals,
                             int8_t * GGML_RESTRICT     iscales,
                             float * GGML_RESTRICT      dfac) {
    const block_iq1_m * x = (const block_iq1_m *) vx;

    // block_iq1_m has no d field - the fp16 super-block scale is spread over the top nibbles of the four scale words
    const uint16_t * sc = (const uint16_t *) x->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

    *dfac = GGML_CPU_FP16_TO_FP32(scale.f16) * 0.125f;

    const uint8_t * qs = x->qs;
    const uint8_t * qh = x->qh;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        iscales[2 * ib + 0] = (int8_t) (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        iscales[2 * ib + 1] = (int8_t) (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);

        const uint16_t idx[4] = {
            (uint16_t) (qs[0] | ((qh[0] << 8) & 0x700)),
            (uint16_t) (qs[1] | ((qh[0] << 4) & 0x700)),
            (uint16_t) (qs[2] | ((qh[1] << 8) & 0x700)),
            (uint16_t) (qs[3] | ((qh[1] << 4) & 0x700)),
        };
        const uint32_t deltas = (uint32_t) (qh[0] & 0x08 ? 0xff : 0x01) | (uint32_t) (qh[0] & 0x80 ? 0xff : 0x01) << 8 |
                                (uint32_t) (qh[1] & 0x08 ? 0xff : 0x01) << 16 |
                                (uint32_t) (qh[1] & 0x80 ? 0xff : 0x01) << 24;

        iqp_store_iq1_x8(vals + 32 * ib, iq1s_grid[idx[0]], iq1s_grid[idx[1]], iq1s_grid[idx[2]], iq1s_grid[idx[3]],
                         deltas);
        qs += 4;
        qh += 2;
    }
}

static void iqp_decode_iq4_xs(const void * GGML_RESTRICT vx,
                              int8_t * GGML_RESTRICT     vals,
                              int8_t * GGML_RESTRICT     iscales,
                              float * GGML_RESTRICT      dfac) {
    const block_iq4_xs * x = (const block_iq4_xs *) vx;

    const uint8_t * qs = x->qs;

    // dl = d * (ls - 32), ls 6 bit, so the integer scale is in [-32, 31]
    *dfac = GGML_CPU_FP16_TO_FP32(x->d);

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int    ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
        const int8_t dl = (int8_t) (ls - 32);

        iscales[2 * ib + 0] = dl;
        iscales[2 * ib + 1] = dl;

        iqp_store_iq4_x32(vals + 32 * ib, qs);
        qs += 16;
    }
}

// expanded by the eligibility test and the decode dispatch
#define IQP_TYPE_LIST(T) \
    T(IQ2_XXS, iq2_xxs)  \
    T(IQ2_XS, iq2_xs)    \
    T(IQ2_S, iq2_s)      \
    T(IQ3_XXS, iq3_xxs)  \
    T(IQ3_S, iq3_s)      \
    T(IQ1_S, iq1_s)      \
    T(IQ1_M, iq1_m)      \
    T(IQ4_XS, iq4_xs)

static bool iqp_decode_superblock(enum ggml_type             type,
                                  const void * GGML_RESTRICT vx,
                                  int8_t * GGML_RESTRICT     vals,
                                  int8_t * GGML_RESTRICT     iscales,
                                  float * GGML_RESTRICT      dfac) {
    switch (type) {
#define IQP_CASE(E, name)                           \
    case GGML_TYPE_##E:                             \
        iqp_decode_##name(vx, vals, iscales, dfac); \
        return true;
        IQP_TYPE_LIST(IQP_CASE)
#undef IQP_CASE
        default:
            return false;
    }
}

#if defined(__AVX2__)

// 8x8 int32 transpose of the 32 column group starting at column off
static inline void iqp_interleave_x8(int8_t * GGML_RESTRICT dst, const int8_t (*vals)[QK_K], int off) {
    static_assert(IQP_NB_ROWS == 8, "the transpose is 8x8");

    __m256i v[IQP_NB_ROWS];

    for (int r = 0; r < IQP_NB_ROWS; ++r) {
        v[r] = _mm256_loadu_si256((const __m256i *) (vals[r] + off));
    }

    // pair rows into dword couples, then into qword quadruples, then swap the 128 bit lanes
    const __m256i a0 = _mm256_unpacklo_epi32(v[0], v[1]);
    const __m256i a1 = _mm256_unpackhi_epi32(v[0], v[1]);
    const __m256i a2 = _mm256_unpacklo_epi32(v[2], v[3]);
    const __m256i a3 = _mm256_unpackhi_epi32(v[2], v[3]);
    const __m256i a4 = _mm256_unpacklo_epi32(v[4], v[5]);
    const __m256i a5 = _mm256_unpackhi_epi32(v[4], v[5]);
    const __m256i a6 = _mm256_unpacklo_epi32(v[6], v[7]);
    const __m256i a7 = _mm256_unpackhi_epi32(v[6], v[7]);

    const __m256i b0 = _mm256_unpacklo_epi64(a0, a2);
    const __m256i b1 = _mm256_unpackhi_epi64(a0, a2);
    const __m256i b2 = _mm256_unpacklo_epi64(a1, a3);
    const __m256i b3 = _mm256_unpackhi_epi64(a1, a3);
    const __m256i b4 = _mm256_unpacklo_epi64(a4, a6);
    const __m256i b5 = _mm256_unpackhi_epi64(a4, a6);
    const __m256i b6 = _mm256_unpacklo_epi64(a5, a7);
    const __m256i b7 = _mm256_unpackhi_epi64(a5, a7);

    _mm256_storeu_si256((__m256i *) (dst + 0 * 32), _mm256_permute2x128_si256(b0, b4, 0x20));
    _mm256_storeu_si256((__m256i *) (dst + 1 * 32), _mm256_permute2x128_si256(b1, b5, 0x20));
    _mm256_storeu_si256((__m256i *) (dst + 2 * 32), _mm256_permute2x128_si256(b2, b6, 0x20));
    _mm256_storeu_si256((__m256i *) (dst + 3 * 32), _mm256_permute2x128_si256(b3, b7, 0x20));
    _mm256_storeu_si256((__m256i *) (dst + 4 * 32), _mm256_permute2x128_si256(b0, b4, 0x31));
    _mm256_storeu_si256((__m256i *) (dst + 5 * 32), _mm256_permute2x128_si256(b1, b5, 0x31));
    _mm256_storeu_si256((__m256i *) (dst + 6 * 32), _mm256_permute2x128_si256(b2, b6, 0x31));
    _mm256_storeu_si256((__m256i *) (dst + 7 * 32), _mm256_permute2x128_si256(b3, b7, 0x31));
}

#endif

// decode IQP_NB_ROWS consecutive source rows (starting at src, row stride nb01) into a panel of nblocks block_iqp_x8
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
            // check that the panel reproduces the reference dequantization bit exactly
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

            for (int sb = 0; sb < IQP_NSB; sb++) {
                dst->iscales[sb * IQP_NB_ROWS + r] = iscales[r][sb];
            }

#if GGML_IQP_USE_BIAS
            dst->bias[r] = 128 * iqp_weighted_sum(vals[r], iscales[r]);
#endif
        }

#if defined(__AVX2__)
        for (int grp = 0; grp < QK_K / 32; grp++) {
            iqp_interleave_x8(dst->qs + grp * 256, vals, grp * 32);
        }
#else
        for (int r = 0; r < IQP_NB_ROWS; r++) {
            for (int sb = 0; sb < IQP_NSB; sb++) {
                for (int g = 0; g < IQP_SB_SIZE / 4; g++) {
                    memcpy(dst->qs + sb * 128 + g * 32 + r * 4, vals[r] + sb * IQP_SB_SIZE + g * 4, 4);
                }
            }
        }
#endif

        dst++;
    }
}

// gemm/gemv kernels: vx points at block_iqp_x8, vy at plain (non interleaved) block_q8_K rows

static void iqp_gemv_8x8_q8_K_generic(int                        n,
                                      float * GGML_RESTRICT      s,
                                      size_t                     bs,
                                      const void * GGML_RESTRICT vx,
                                      const void * GGML_RESTRICT vy,
                                      int                        nr,
                                      int                        nc) {
    const int nb                = n / QK_K;
    const int ncols_interleaved = 8;

    assert(n % QK_K == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    const block_iqp_x8 * b_ptr_start = (const block_iqp_x8 *) vx;
    const block_q8_K *   a_ptr       = (const block_q8_K *) vy;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iqp_x8 * b_ptr = b_ptr_start + x * nb;

        float sumf[8] = { 0 };

        for (int l = 0; l < nb; l++) {
            int32_t sumi[8] = { 0 };

            for (int sb = 0; sb < IQP_NSB; sb++) {
                int32_t isum[8] = { 0 };

                for (int g = 0; g < 4; g++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        for (int k = 0; k < 4; k++) {
                            isum[j] += b_ptr[l].qs[sb * 128 + g * 32 + j * 4 + k] * a_ptr[l].qs[sb * 16 + g * 4 + k];
                        }
                    }
                }

                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi[j] += isum[j] * b_ptr[l].iscales[sb * 8 + j];
                }
            }

            for (int j = 0; j < ncols_interleaved; j++) {
                sumf[j] += (float) sumi[j] * (b_ptr[l].dfac[j] * a_ptr[l].d);
            }
        }

        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// one 4 row x nc column tile; s points at the first of the four output rows, bs floats apart
static void iqp_gemm_tile_4_generic(int                                nb,
                                    float * GGML_RESTRICT              s,
                                    size_t                             bs,
                                    const block_iqp_x8 * GGML_RESTRICT b_ptr_start,
                                    const block_q8_K * const           a_ptr[4],
                                    int                                nc) {
    const int ncols_interleaved = 8;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iqp_x8 * b_ptr = b_ptr_start + x * nb;

        float sumf[4][8];
        for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
                sumf[m][j] = 0.0f;
            }
        }

        for (int l = 0; l < nb; l++) {
            for (int m = 0; m < 4; m++) {
                int32_t sumi[8] = { 0 };

                for (int sb = 0; sb < IQP_NSB; sb++) {
                    int32_t isum[8] = { 0 };

                    for (int g = 0; g < 4; g++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            for (int k = 0; k < 4; k++) {
                                isum[j] +=
                                    b_ptr[l].qs[sb * 128 + g * 32 + j * 4 + k] * a_ptr[m][l].qs[sb * 16 + g * 4 + k];
                            }
                        }
                    }

                    for (int j = 0; j < ncols_interleaved; j++) {
                        sumi[j] += isum[j] * b_ptr[l].iscales[sb * 8 + j];
                    }
                }

                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] += (float) sumi[j] * (b_ptr[l].dfac[j] * a_ptr[m][l].d);
                }
            }
        }

        for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
                s[m * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

static void iqp_gemm_8x8_q8_K_generic(int                        n,
                                      float * GGML_RESTRICT      s,
                                      size_t                     bs,
                                      const void * GGML_RESTRICT vx,
                                      const void * GGML_RESTRICT vy,
                                      int                        nr,
                                      int                        nc) {
    const int nb = n / QK_K;

    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);

    const block_iqp_x8 * b_ptr_start = (const block_iqp_x8 *) vx;
    const block_q8_K *   a_ptr_start = (const block_q8_K *) vy;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_K * a_ptr[4];
        for (int m = 0; m < 4; m++) {
            a_ptr[m] = a_ptr_start + (y * 4 + m) * nb;
        }

        iqp_gemm_tile_4_generic(nb, s + y * 4 * bs, bs, b_ptr_start, a_ptr, nc);
    }
}

static void iqp_gemm_8x8_q8_K_p4_generic(int                                n,
                                         float * GGML_RESTRICT              s,
                                         size_t                             bs,
                                         const void * GGML_RESTRICT         vx,
                                         const void * const * GGML_RESTRICT vy,
                                         int                                nc) {
    const int nb = n / QK_K;

    assert(n % QK_K == 0);
    assert(nc % 8 == 0);

    const block_q8_K * a_ptr[4];
    for (int m = 0; m < 4; m++) {
        a_ptr[m] = (const block_q8_K *) vy[m];
    }

    iqp_gemm_tile_4_generic(nb, s, bs, (const block_iqp_x8 *) vx, a_ptr, nc);
}

#if defined(__AVX2__)

// add int16_t pairwise and return as 256 bit int vector, then add the accumulator
static inline __m256i sum_i16_pairs_acc_int32x8(const __m256i acc, const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(ones, x));
}

static inline __m256i mul_sum_us8_pairs_acc_int32x8(const __m256i acc, const __m256i ax, const __m256i sy) {
#    if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#    elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, ax, sy);
#    else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x8(acc, dot);
#    endif
}

// Integer variant of the function defined in ggml-quants.c
// multiply int8_t, add results pairwise twice and return as 256 bit int vector, then add the accumulator
static inline __m256i mul_sum_i8_pairs_acc_int32x8(const __m256i acc, const __m256i x, const __m256i y) {
#    if defined(__AVXVNNIINT8__)
    return _mm256_dpbssd_epi32(acc, x, y);
#    else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_acc_int32x8(acc, ax, sy);
#    endif
}

// load the 16 activations of one sub-block, offset by 128 when they are fed to dpbusd as unsigned bytes
static inline __m256i iqp_load_y(const int8_t * GGML_RESTRICT qs) {
    __m128i y = _mm_loadu_si128((const __m128i *) qs);
#    if GGML_IQP_USE_BIAS
    y = _mm_xor_si128(y, _mm_set1_epi8((char) 0x80));
#    endif
    return _mm256_broadcastsi128_si256(y);
}

// xv: 8 rows x 4 signed weights, yb: the matching 4 activation bytes broadcast to all 8 lanes
static inline __m256i iqp_dot4(const __m256i acc, const __m256i xv, const __m256i yb) {
#    if GGML_IQP_USE_BIAS
    return mul_sum_us8_pairs_acc_int32x8(acc, yb, xv);
#    else
    return mul_sum_i8_pairs_acc_int32x8(acc, xv, yb);
#    endif
}

static inline __m256i iqp_load_iscales(const int8_t * GGML_RESTRICT iscales) {
    return _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) iscales));
}

// accumulate one super-block of 8 interleaved rows against one q8_K row in int32; worst case 16 * 32 * 16 * 255 * 127 = 2.65e8 plus a bias of at most 1.33e8 does not overflow
static inline __m256i iqp_acc_block(const block_iqp_x8 * GGML_RESTRICT b, const block_q8_K * GGML_RESTRICT a) {
    __m256i sumi = _mm256_setzero_si256();

    for (int sb = 0; sb < IQP_NSB; sb++) {
        const int8_t * qs = b->qs + sb * 128;

        const __m256i yv = iqp_load_y(a->qs + sb * 16);

        __m256i isum = _mm256_setzero_si256();

        isum = iqp_dot4(isum, _mm256_loadu_si256((const __m256i *) (qs + 0)), _mm256_shuffle_epi32(yv, 0x00));
        isum = iqp_dot4(isum, _mm256_loadu_si256((const __m256i *) (qs + 32)), _mm256_shuffle_epi32(yv, 0x55));
        isum = iqp_dot4(isum, _mm256_loadu_si256((const __m256i *) (qs + 64)), _mm256_shuffle_epi32(yv, 0xAA));
        isum = iqp_dot4(isum, _mm256_loadu_si256((const __m256i *) (qs + 96)), _mm256_shuffle_epi32(yv, 0xFF));

        sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(isum, iqp_load_iscales(b->iscales + sb * 8)));
    }

#    if GGML_IQP_USE_BIAS
    sumi = _mm256_sub_epi32(sumi, _mm256_loadu_si256((const __m256i *) b->bias));
#    endif

    return sumi;
}

// one 4 row x nc column tile; s points at the first of the four output rows, bs floats apart
static inline void iqp_gemm_tile_4(int                                nb,
                                   float * GGML_RESTRICT              s,
                                   size_t                             bs,
                                   const block_iqp_x8 * GGML_RESTRICT b_ptr_start,
                                   const block_q8_K * const           a_ptr[4],
                                   int                                nc) {
    const int ncols_interleaved = 8;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iqp_x8 * b_ptr = b_ptr_start + x * nb;

        __m256 sumf[4];
        for (int m = 0; m < 4; m++) {
            sumf[m] = _mm256_setzero_ps();
        }

        for (int l = 0; l < nb; l++) {
            __m256i sumi[4];
            for (int m = 0; m < 4; m++) {
                sumi[m] = _mm256_setzero_si256();
            }

            for (int sb = 0; sb < IQP_NSB; sb++) {
                const int8_t * qs = b_ptr[l].qs + sb * 128;

                __m256i yv[4];
                __m256i isum[4];
                for (int m = 0; m < 4; m++) {
                    yv[m]   = iqp_load_y(a_ptr[m][l].qs + sb * 16);
                    isum[m] = _mm256_setzero_si256();
                }

                const __m256i xv0 = _mm256_loadu_si256((const __m256i *) (qs + 0));
                const __m256i xv1 = _mm256_loadu_si256((const __m256i *) (qs + 32));
                const __m256i xv2 = _mm256_loadu_si256((const __m256i *) (qs + 64));
                const __m256i xv3 = _mm256_loadu_si256((const __m256i *) (qs + 96));

                for (int m = 0; m < 4; m++) {
                    isum[m] = iqp_dot4(isum[m], xv0, _mm256_shuffle_epi32(yv[m], 0x00));
                    isum[m] = iqp_dot4(isum[m], xv1, _mm256_shuffle_epi32(yv[m], 0x55));
                    isum[m] = iqp_dot4(isum[m], xv2, _mm256_shuffle_epi32(yv[m], 0xAA));
                    isum[m] = iqp_dot4(isum[m], xv3, _mm256_shuffle_epi32(yv[m], 0xFF));
                }

                const __m256i isc = iqp_load_iscales(b_ptr[l].iscales + sb * 8);
                for (int m = 0; m < 4; m++) {
                    sumi[m] = _mm256_add_epi32(sumi[m], _mm256_mullo_epi32(isum[m], isc));
                }
            }

#    if GGML_IQP_USE_BIAS
            const __m256i bias = _mm256_loadu_si256((const __m256i *) b_ptr[l].bias);
            for (int m = 0; m < 4; m++) {
                sumi[m] = _mm256_sub_epi32(sumi[m], bias);
            }
#    endif

            const __m256 dfac = _mm256_loadu_ps(b_ptr[l].dfac);
            for (int m = 0; m < 4; m++) {
                sumf[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[m]),
                                          _mm256_mul_ps(dfac, _mm256_set1_ps(a_ptr[m][l].d)), sumf[m]);
            }
        }

        for (int m = 0; m < 4; m++) {
            _mm256_storeu_ps(s + m * bs + x * ncols_interleaved, sumf[m]);
        }
    }
}

#endif  // __AVX2__

static void iqp_gemv_8x8_q8_K(int                        n,
                              float * GGML_RESTRICT      s,
                              size_t                     bs,
                              const void * GGML_RESTRICT vx,
                              const void * GGML_RESTRICT vy,
                              int                        nr,
                              int                        nc) {
    const int nb                = n / QK_K;
    const int ncols_interleaved = 8;

    assert(n % QK_K == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);
    UNUSED(nb);
    UNUSED(ncols_interleaved);

#if defined(__AVX2__)
    const block_iqp_x8 * b_ptr_start = (const block_iqp_x8 *) vx;
    const block_q8_K *   a_ptr       = (const block_q8_K *) vy;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iqp_x8 * b_ptr = b_ptr_start + x * nb;

        __m256 sumf = _mm256_setzero_ps();

        for (int l = 0; l < nb; l++) {
            const __m256 dv = _mm256_mul_ps(_mm256_loadu_ps(b_ptr[l].dfac), _mm256_set1_ps(a_ptr[l].d));

            sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iqp_acc_block(b_ptr + l, a_ptr + l)), dv, sumf);
        }

        _mm256_storeu_ps(s + x * ncols_interleaved, sumf);
    }

    return;
#endif

    iqp_gemv_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

static void iqp_gemm_8x8_q8_K(int                        n,
                              float * GGML_RESTRICT      s,
                              size_t                     bs,
                              const void * GGML_RESTRICT vx,
                              const void * GGML_RESTRICT vy,
                              int                        nr,
                              int                        nc) {
    const int nb                = n / QK_K;
    const int ncols_interleaved = 8;

    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);

#if defined(__AVX2__)
    const block_iqp_x8 * b_ptr_start = (const block_iqp_x8 *) vx;
    const block_q8_K *   a_ptr_start = (const block_q8_K *) vy;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_K * a_ptr[4];
        for (int m = 0; m < 4; m++) {
            a_ptr[m] = a_ptr_start + (y * 4 + m) * nb;
        }

        iqp_gemm_tile_4(nb, s + y * 4 * bs, bs, b_ptr_start, a_ptr, nc);
    }

    return;
#endif

    iqp_gemm_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

// same as iqp_gemm_8x8_q8_K with nr = 4, but the activation rows are passed as separate pointers (for the scattered rows of MUL_MAT_ID)
static void iqp_gemm_8x8_q8_K_p4(int                                n,
                                 float * GGML_RESTRICT              s,
                                 size_t                             bs,
                                 const void * GGML_RESTRICT         vx,
                                 const void * const * GGML_RESTRICT vy,
                                 int                                nc) {
    const int nb                = n / QK_K;
    const int ncols_interleaved = 8;

    assert(n % QK_K == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);

#if defined(__AVX2__)
    const block_q8_K * a_ptr[4];
    for (int m = 0; m < 4; m++) {
        a_ptr[m] = (const block_q8_K *) vy[m];
    }

    iqp_gemm_tile_4(nb, s, bs, (const block_iqp_x8 *) vx, a_ptr, nc);

    return;
#endif

    iqp_gemm_8x8_q8_K_p4_generic(n, s, bs, vx, vy, nc);
}

static bool iqp_type_supported(enum ggml_type type) {
    switch (type) {
#define IQP_CASE(E, name) case GGML_TYPE_##E:
        IQP_TYPE_LIST(IQP_CASE)
#undef IQP_CASE
        return true;
        default:
            return false;
    }
}

static bool iqp_supported_common(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (!iqp_type_supported(src0->type)) {
        return false;
    }

    // the path assumes the src1 conversion type is q8_K
    if (ggml_get_type_traits_cpu(src0->type)->vec_dot_type != GGML_TYPE_Q8_K) {
        return false;
    }

    // escape hatch to A/B the panel against the plain vec_dot path without rebuilding (--no-repack does not cover this path)
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

bool ggml_cpu_iqp_supports_mul_mat(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (!iqp_supported_common(dst)) {
        return false;
    }

    if (src1->ne[1] < GGML_IQP_MIN_BATCH) {
        return false;
    }

    // plain 2D weight matmuls only (src1 may still be batched over ne12)
    if (src0->ne[2] != 1) {
        return false;
    }

    return true;
}

bool ggml_cpu_iqp_supports_mul_mat_id(const struct ggml_tensor * dst) {
    const struct ggml_tensor * ids = dst->src[2];

    if (!iqp_supported_common(dst)) {
        return false;
    }

    // skip the node entirely (work buffer included) if no expert can reach the per expert threshold
    if (!ggml_cpu_iqp_mul_mat_id_min_batch(ids->ne[0] * ids->ne[1])) {
        return false;
    }

    return true;
}

void ggml_compute_forward_mul_mat_id_iqp(const struct ggml_compute_params * params,
                                         struct ggml_tensor *               dst,
                                         int64_t                            cur_a,
                                         int64_t                            cne1,
                                         const int32_t *                    expert_rows,
                                         void *                             panels) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nblocks = ne00 / QK_K;

    const size_t nbw1 = ggml_cpu_iqp_row_size(dst);

    block_iqp_x8 * panel = (block_iqp_x8 *) ((char *) panels + (size_t) ith * ggml_cpu_iqp_scratch_size(dst));

    const char * src0_cur = (const char *) src0->data + cur_a * nb02;

    const int64_t ngroups = ne01 / IQP_NB_ROWS;

    const int64_t g0 = (ngroups * ith) / nth;
    const int64_t g1 = (ngroups * (ith + 1)) / nth;

    for (int64_t g = g0; g < g1; g++) {
        const int64_t r = g * IQP_NB_ROWS;

        iqp_decode_panel_8(src0->type, src0_cur + r * nb01, nb01, nblocks, panel);

        // the dst rows are scattered, so the gemm writes into tmp and it is copied out row by row
        float tmp[4 * IQP_NB_ROWS];

        for (int64_t k = 0; k < cne1; k += 4) {
            const int64_t nrows = MIN(4, cne1 - k);

            // a short tail tile duplicates its last row into the unused slots; the padding is never copied out
            const void * rows[4];

            for (int64_t m = 0; m < 4; m++) {
                const int64_t kk = k + MIN(m, nrows - 1);

                rows[m] = (const char *) params->wdata +
                          ((expert_rows[2 * kk + 0] % ne11) + expert_rows[2 * kk + 1] * ne11) * nbw1;
            }

            iqp_gemm_8x8_q8_K_p4(ne00, tmp, IQP_NB_ROWS, panel, rows, IQP_NB_ROWS);

            for (int64_t m = 0; m < nrows; m++) {
                float * dst_col = (float *) ((char *) dst->data + expert_rows[2 * (k + m) + 0] * nb1 +
                                             expert_rows[2 * (k + m) + 1] * nb2);
                memcpy(dst_col + r, tmp + m * IQP_NB_ROWS, IQP_NB_ROWS * sizeof(float));
            }
        }
    }
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

    const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_K, ne10);
    const size_t nbw2 = nbw1 * ne11;

    const size_t scratch_size = ggml_cpu_iqp_scratch_size(dst);

    const size_t scratch_offset = GGML_PAD(nbw2 * ne12, 64);

    GGML_ASSERT(scratch_offset + (size_t) nth * scratch_size <= params->wsize);

    block_iqp_x8 * panel = (block_iqp_x8 *) ((char *) params->wdata + scratch_offset + (size_t) ith * scratch_size);

    const int64_t nrows = ne11;

    const int64_t ngroups = ne01 / IQP_NB_ROWS;

    // aim for 4 chunks per thread; the caller has already reset the chunk counter
    // on NUMA systems fall back to one chunk per thread
    const int64_t chunks_per_thread = ggml_is_numa() ? 1 : 4;
    const int64_t groups_per_chunk  = MAX(1, (ngroups + nth * chunks_per_thread - 1) / (nth * chunks_per_thread));
    const int64_t nchunk            = (ngroups + groups_per_chunk - 1) / groups_per_chunk;

    int current_chunk = ith;

    while (current_chunk < nchunk) {
        const int64_t g0 = current_chunk * groups_per_chunk;
        const int64_t g1 = MIN(g0 + groups_per_chunk, ngroups);

        for (int64_t g = g0; g < g1; g++) {
            const int64_t r = g * IQP_NB_ROWS;

            iqp_decode_panel_8(src0->type, (const char *) src0->data + r * nb01, nb01, nblocks, panel);

            for (int64_t i12 = 0; i12 < ne12; i12++) {
                const char * src1_ptr = (const char *) params->wdata + i12 * nbw2;
                char *       dst_ptr  = (char *) dst->data + i12 * nb2;

                if (nrows > 3) {
                    iqp_gemm_8x8_q8_K(ne00, (float *) dst_ptr + r, nb1 / nb0, panel, src1_ptr, nrows - (nrows % 4),
                                      IQP_NB_ROWS);
                }
                for (int64_t iter = nrows - (nrows % 4); iter < nrows; iter++) {
                    iqp_gemv_8x8_q8_K(ne00, (float *) (dst_ptr + iter * nb1) + r, ne01, panel, src1_ptr + nbw1 * iter,
                                      1 /* nrows */, IQP_NB_ROWS);
                }
            }
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}
