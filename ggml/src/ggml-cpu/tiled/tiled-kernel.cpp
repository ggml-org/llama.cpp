// mulmat microtile kernels

#include "tiled-kernel.h"

#include "ggml.h"

#include <string.h>

#if defined(__AVX512VNNI__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif


// Reference implementation, slower than existing vec_dot approach
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_microtile_scalar(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                       int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;    // subblocks per 256-K block
    constexpr int NS = SUBBLK / 16; // per-16 bsums per subblock

    float acc[TILED_MICRO][TILED_MICRO];
    memset(acc, 0, sizeof(acc));

    // subdots at subblock granularity over the 256-K block, exact integer math
    for (int s = 0; s < NB; s++) {
        for (int j = 0; j < TILED_MICRO; j++) {
            const int br = j0 + j;
            const int8_t * q1 = &src1.q[br * TILED_TILE_K + s * SUBBLK];
            int32_t bsum = 0;
            for (int u = 0; u < NS; u++) {
                bsum += src1.bsums[(s * NS + u) * TILED_TILE_ROWS + br];
            }

            for (int i = 0; i < TILED_MICRO; i++) {
                const int ar = i0 + i;
                const uint8_t * q0 = &src0.q[ar * TILED_TILE_K + s * SUBBLK];

                int32_t raw = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[e] * (int32_t) q1[e];
                }

                // BIAS: subtract BIAS*bsum (src1's per-subblock code sum) from the exact int raw
                int32_t corr = raw;
                if constexpr (BIAS != 0) {
                    corr -= BIAS * bsum;
                }
                const int32_t scales_raw = (int32_t) src0.scales[ar * NB + s] * corr;
                // d is NOT applied here: it is constant over the s-loop, so we apply it last before write-out
                if constexpr (HAS_MIN) {
                    const int32_t mins_bsum = (int32_t) src0.mins[ar * NB + s] * bsum;
                    acc[i][j] += (float) src0.d[ar] * (float) scales_raw
                              - (float) src0.dmin[ar] * (float) mins_bsum;
                } else {
                    acc[i][j] += (float) src0.d[ar] * (float) scales_raw;
                }
            }
        }
    }

    // Apply d and write out to buf
    for (int i = 0; i < TILED_MICRO; i++) {
        for (int j = 0; j < TILED_MICRO; j++) {
            buf[(i0 + i) * buf_stride + (j0 + j)] += src1.d[j0 + j] * acc[i][j];
        }
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)


// VNNI microkernel
// 8x16 band pass: 8 src0 rows (i0, i0+8) x 16 src1 cols (j0, j0+16)
// Math: s1_acc += scales_s*raw_s, s2_acc += mins_s*bsums_s per subblock; the row result
// is d0*s1_acc - dmin*s2_acc (f32, exact in range), applied once per row.
//
// Register pressure: the band pass holds acc16 + s1_acc = 16 zmm for the
// 8-row band (s2_acc has no dpbusd dependency, so it is computed after the
// band pass and adds no register pressure);
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_micro_vnni_8x16(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                          int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int NS = SUBBLK / 16;
    constexpr int NG = SUBBLK / 4;

    // band width, see the register-pressure note above
    constexpr int NUM_ROWS = 8;

    const __m512 d1_vec = _mm512_load_ps(&src1.d[j0]);

    __m512i s1_acc[NUM_ROWS];
    for (int t = 0; t < NUM_ROWS; t++) { s1_acc[t] = _mm512_setzero_si512(); }
    __m512i s2_acc[NUM_ROWS];
    if constexpr (HAS_MIN) {
        for (int t = 0; t < NUM_ROWS; t++) { s2_acc[t] = _mm512_setzero_si512(); }
    }

    const int32_t * bsums = src1.bsums; // int32 per-16 sums; NS > 1 combines NS lanes per subblock
    for (int s = 0; s < NB; s++) {
        __m512i bsums32 = _mm512_load_si512((const __m512i *) &bsums[s * NS * TILED_TILE_ROWS + j0]);
        for (int u = 1; u < NS; u++) {
            bsums32 = _mm512_add_epi32(bsums32, _mm512_load_si512((const __m512i *) &bsums[(s * NS + u) * TILED_TILE_ROWS + j0]));
        }

        __m512i bias32 = _mm512_setzero_si512();
        if constexpr (BIAS != 0) {
            bias32 = _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(BIAS));
        }

        __m512i acc16[NUM_ROWS];
        for (int t = 0; t < NUM_ROWS; t++) { acc16[t] = _mm512_setzero_si512(); }

#ifdef __GNUC__
        #pragma GCC unroll 8 // pragma unrolled justified by measuing with/without
#endif
        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            const __m512i codes = _mm512_load_si512((const __m512i *) &src1.q[kg * TILED_TILE_ROWS * 4 + j0 * 4]);
            for (int t = 0; t < NUM_ROWS; t++) {
                const uint32_t u4 = *(const uint32_t *) &src0.q[(i0 + t) * TILED_TILE_K + kg * 4];
                const __m512i u4b = _mm512_set1_epi32((int) u4);
                acc16[t] = _mm512_dpbusd_epi32(acc16[t], u4b, codes);
            }
        }

        // int correction: s1_acc += scales*(raw-BIAS*bsums)
        for (int t = 0; t < NUM_ROWS; t++) {
            const int ar = i0 + t;
            __m512i rawi = acc16[t];
            if constexpr (BIAS != 0) {
                rawi = _mm512_sub_epi32(rawi, bias32);
            }
            s1_acc[t] = _mm512_add_epi32(s1_acc[t],
                _mm512_mullo_epi32(rawi, _mm512_set1_epi32(src0.scales[ar * NB + s])));
        }
    }

    // s2_acc += mins*bsums; independent of the dpbusd results, so it runs here instead of in the
    // band pass: the band pass keeps the register budget for the 8-row band
    if constexpr (HAS_MIN) {
        for (int s = 0; s < NB; s++) {
            __m512i bsums32 = _mm512_load_si512((const __m512i *) &bsums[s * NS * TILED_TILE_ROWS + j0]);
            for (int u = 1; u < NS; u++) {
                bsums32 = _mm512_add_epi32(bsums32, _mm512_load_si512((const __m512i *) &bsums[(s * NS + u) * TILED_TILE_ROWS + j0]));
            }
            for (int t = 0; t < NUM_ROWS; t++) {
                s2_acc[t] = _mm512_add_epi32(s2_acc[t],
                    _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(src0.mins[(i0 + t) * NB + s])));
            }
        }
    }

    // epilogue: int->float, apply per-row scales, store to buf
    for (int t = 0; t < NUM_ROWS; t++) {
        const int ar = i0 + t;
        __m512 f1 = _mm512_cvtepi32_ps(s1_acc[t]);
        __m512 result = _mm512_mul_ps(f1, _mm512_set1_ps(src0.d[ar]));
        if constexpr (HAS_MIN) {
            __m512 f2 = _mm512_cvtepi32_ps(s2_acc[t]);
            result = _mm512_fnmadd_ps(_mm512_set1_ps(src0.dmin[ar]), f2, result);
        }
        float * p = &buf[(i0 + t) * buf_stride + j0];
        _mm512_store_ps(p, _mm512_add_ps(_mm512_load_ps(p), _mm512_mul_ps(result, d1_vec)));
    }
}

// 16x16 microtile as two explicit 8x16 band passes
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_microtile_vnni(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                         int i0, int j0, float * buf, int buf_stride) {
    tiled_run_micro_vnni_8x16<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0,      j0, buf, buf_stride);
    tiled_run_micro_vnni_8x16<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0 + 8,  j0, buf, buf_stride);
}

#endif // __AVX512VNNI__ && __AVX512VL__

#if defined(__AVX2__)
// AVX2 kernel.
// We're effectively applying the existing vec_dot algorithms to an 8x16 block here.
// Different paths based on subblock size as it affects when/where we multiply in scales and apply mins
// SUBBLK=32: one 256-bit maddubs per (s, column), a 256-bit set1 scale,
// one 8-lane i32 accumulator per column.
// SUBBLK=16: two subblocks per 32B load. The 256-bit maddubs product
// still gives 16 i16 lanes (0..7 = subblock sp, 8..15 = sp+1), but the
// scale is applied per 128-bit half.
// With BIAS != 0 the q0 codes are biased (up to 241) and a maddubs i16 pair
// overflows, so q0 is split into 4-bit halves and the dot is the sum of two
// maddubs, each pair then <= 2*15*127
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_microtile_avx2(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                     int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int GROUP = 8; // src1 columns per group: one acc32 per column

    static_assert(SUBBLK == 16 || SUBBLK == 32, "unsupported SUBBLK");

    for (int i = 0; i < TILED_MICRO; i++) {
        // 4-lane i32 hsum (used in the group epilogue)
        auto hsum4 = [](const __m128i v) {
            __m128i s = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtsi128_si32(s);
        };
        const int ar = i0 + i;
        const float d0 = src0.d[ar];
        const float dmin0 = src0.dmin[ar];
        const uint8_t * q0 = &src0.q[ar * TILED_TILE_K];
        const int32_t * scales_row = &src0.scales[ar * NB];
        const int32_t * mins_row = &src0.mins[ar * NB];

        for (int g = 0; g < TILED_MICRO; g += GROUP) {
            const int8_t * q1g[GROUP];
            __m256i corr = _mm256_setzero_si256(); // -BIAS * sum_s scales_s * bsums_s, 1 lane per col (vanishes via constexpr)
            __m256i s2 = _mm256_setzero_si256();   // sum_s mins_s * bsums_s
            if constexpr (SUBBLK == 32) {
                __m256i acc[GROUP];
                for (int t = 0; t < GROUP; t++) {
                    q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                    acc[t] = _mm256_setzero_si256();
                }
                // one 32B q0 load per (s); with biased codes (up to 241) the maddubs i16 pairs
                // overflow, so for BIAS != 0 split q0 into 4-bit halves, each pair then <= 2*15*127
                for (int s = 0; s < NB; s++) {
                    const __m256i q0_32 = _mm256_loadu_si256((const __m256i *) &q0[s * SUBBLK]);
                    const __m256i scales16 = _mm256_set1_epi16(scales_row[s]);
                    const __m256i bsums_v = _mm256_add_epi32(
                        _mm256_loadu_si256((const __m256i *) &src1.bsums[s * 2 * TILED_TILE_ROWS + j0 + g]),
                        _mm256_loadu_si256((const __m256i *) &src1.bsums[(s * 2 + 1) * TILED_TILE_ROWS + j0 + g]));
                    if constexpr (BIAS != 0) {
                        const __m256i q0a = _mm256_and_si256(q0_32, _mm256_set1_epi8(0x0F));
                        const __m256i q0b = _mm256_and_si256(_mm256_srli_epi16(q0_32, 4), _mm256_set1_epi8(0x0F)); // srli is per 16-bit lane, mask out the neighbor bleed
                        const __m256i scales16b = _mm256_set1_epi16(16 * scales_row[s]);
                        for (int t = 0; t < GROUP; t++) {
                            const __m256i q1_32 = _mm256_loadu_si256((const __m256i *) &q1g[t][s * SUBBLK]);
                            acc[t] = _mm256_add_epi32(acc[t], _mm256_add_epi32(
                                _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(q0a, q1_32)),
                                _mm256_madd_epi16(scales16b, _mm256_maddubs_epi16(q0b, q1_32))));
                        }
                    } else {
                        for (int t = 0; t < GROUP; t++) {
                            acc[t] = _mm256_add_epi32(acc[t],
                                _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(
                                    q0_32, _mm256_loadu_si256((const __m256i *) &q1g[t][s * SUBBLK]))));
                        }
                    }
                    if constexpr (BIAS != 0) {
                        corr = _mm256_sub_epi32(corr, _mm256_mullo_epi32(bsums_v, _mm256_set1_epi32(BIAS * scales_row[s])));
                    }
                    if constexpr (HAS_MIN) {
                        s2 = _mm256_add_epi32(s2, _mm256_mullo_epi32(bsums_v, _mm256_set1_epi32(mins_row[s])));
                    }
                }
                int32_t corr_s[GROUP] = { 0 };
                int32_t s2_s[GROUP] = { 0 };
                if constexpr (BIAS != 0) {
                    _mm256_storeu_si256((__m256i *) corr_s, corr);
                }
                if constexpr (HAS_MIN) {
                    _mm256_storeu_si256((__m256i *) s2_s, s2);
                }
                for (int t = 0; t < GROUP; t++) {
                    // 8 i32 lanes -> scalar (the per-pair dot, pre-correction)
                    const __m128i lo = _mm256_castsi256_si128(acc[t]);
                    const int32_t s1 = hsum4(_mm_add_epi32(lo, _mm256_extracti128_si256(acc[t], 1))) + corr_s[t];
                    float res = d0 * (float) s1;
                    if constexpr (HAS_MIN) {
                        res -= dmin0 * (float) s2_s[t];
                    }
                    buf[ar * buf_stride + j0 + g + t] += src1.d[j0 + g + t] * res;
                }
            } else {
                __m256i acc[GROUP];
                for (int t = 0; t < GROUP; t++) {
                    q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                    acc[t] = _mm256_setzero_si256();
                }
                // SUBBLK = 16: two subblocks per 32B load.
                // Build the interleaved scale from two 128-bit set1s (one set_m128i)
                // instead of one 8-scalar-arg 256-bit set
                for (int sp = 0; sp < NB; sp += 2) {
                    const __m256i q0_32 = _mm256_loadu_si256((const __m256i *) &q0[sp * SUBBLK]);
                    const __m256i scalesv = _mm256_set_m128i(_mm_set1_epi16(scales_row[sp + 1]), _mm_set1_epi16(scales_row[sp]));
                    const __m256i bsums0_v = _mm256_loadu_si256((const __m256i *) &src1.bsums[sp * TILED_TILE_ROWS + j0 + g]);
                    const __m256i bsums1_v = _mm256_loadu_si256((const __m256i *) &src1.bsums[(sp + 1) * TILED_TILE_ROWS + j0 + g]);
                    if constexpr (BIAS != 0) {
                        // 4-bit split as in the SUBBLK = 32 path above
                        const __m256i q0a = _mm256_and_si256(q0_32, _mm256_set1_epi8(0x0F));
                        const __m256i q0b = _mm256_and_si256(_mm256_srli_epi16(q0_32, 4), _mm256_set1_epi8(0x0F)); // srli is per 16-bit lane, mask out the neighbor bleed
                        const __m256i scalesvb = _mm256_set_m128i(_mm_set1_epi16(16 * scales_row[sp + 1]), _mm_set1_epi16(16 * scales_row[sp]));
                        for (int t = 0; t < GROUP; t++) {
                            const __m256i q1_32 = _mm256_loadu_si256((const __m256i *) &q1g[t][sp * SUBBLK]);
                            acc[t] = _mm256_add_epi32(acc[t], _mm256_add_epi32(
                                _mm256_madd_epi16(scalesv, _mm256_maddubs_epi16(q0a, q1_32)),
                                _mm256_madd_epi16(scalesvb, _mm256_maddubs_epi16(q0b, q1_32))));
                        }
                    } else {
                        for (int t = 0; t < GROUP; t++) {
                            acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(
                                scalesv, _mm256_maddubs_epi16(
                                    q0_32, _mm256_loadu_si256((const __m256i *) &q1g[t][sp * SUBBLK]))));
                        }
                    }
                    // per-subblock scales differ, so corr/s2 keep the bsums0/bsums1 split
                    if constexpr (BIAS != 0) {
                        corr = _mm256_sub_epi32(corr, _mm256_add_epi32(
                            _mm256_mullo_epi32(bsums0_v, _mm256_set1_epi32(BIAS * scales_row[sp])),
                            _mm256_mullo_epi32(bsums1_v, _mm256_set1_epi32(BIAS * scales_row[sp + 1]))));
                    }
                    if constexpr (HAS_MIN) {
                        s2 = _mm256_add_epi32(s2, _mm256_add_epi32(
                            _mm256_mullo_epi32(bsums0_v, _mm256_set1_epi32(mins_row[sp])),
                            _mm256_mullo_epi32(bsums1_v, _mm256_set1_epi32(mins_row[sp + 1]))));
                    }
                }
                int32_t corr_s[GROUP] = { 0 };
                int32_t s2_s[GROUP] = { 0 };
                if constexpr (BIAS != 0) {
                    _mm256_storeu_si256((__m256i *) corr_s, corr);
                }
                if constexpr (HAS_MIN) {
                    _mm256_storeu_si256((__m256i *) s2_s, s2);
                }
                for (int t = 0; t < GROUP; t++) {
                    // 8 i32 lanes -> scalar (the per-pair dot, pre-correction)
                    const __m128i lo = _mm256_castsi256_si128(acc[t]);
                    const int32_t s1 = hsum4(_mm_add_epi32(lo, _mm256_extracti128_si256(acc[t], 1))) + corr_s[t];
                    float res = d0 * (float) s1;
                    if constexpr (HAS_MIN) {
                        res -= dmin0 * (float) s2_s[t];
                    }
                    buf[ar * buf_stride + j0 + g + t] += src1.d[j0 + g + t] * res;
                }
            }
        }
    }
}

#endif // __AVX2__

#if defined(__AVX__) && !defined(__AVX2__)
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_microtile_avx(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                    int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int NS = SUBBLK / 16;
    constexpr int GROUP = 4; // src1 columns per group: one acc32 per column

    for (int i = 0; i < TILED_MICRO; i++) {
        const int ar = i0 + i;
        const float d0 = src0.d[ar];
        const float dmin0 = src0.dmin[ar];
        const uint8_t * q0 = &src0.q[ar * TILED_TILE_K];
        const int32_t * scales_row = &src0.scales[ar * NB];
        const int32_t * mins_row = &src0.mins[ar * NB];

        for (int g = 0; g < TILED_MICRO; g += GROUP) {
            const int8_t * q1g[GROUP];
            __m128i acc[GROUP];
            __m128i corr = _mm_setzero_si128(); // -BIAS * sum_s scales_s * bsums_s, 1 lane per col (vanishes via constexpr)
            __m128i s2 = _mm_setzero_si128();   // sum_s mins_s * bsums_s
            for (int t = 0; t < GROUP; t++) {
                q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                acc[t] = _mm_setzero_si128();
            }

            for (int s = 0; s < NB; s++) {
                const __m128i scales16 = _mm_set1_epi16(scales_row[s]);
                __m128i bsums_v = _mm_setzero_si128(); // per-col bsums as a 4-lane i32 vector
                for (int u = 0; u < NS; u++) {
                    const __m128i a16 = _mm_loadu_si128((const __m128i *) &q0[s * SUBBLK + u * 16]);
                    bsums_v = _mm_add_epi32(bsums_v, _mm_loadu_si128(
                        (const __m128i *) &src1.bsums[(s * NS + u) * TILED_TILE_ROWS + j0 + g]));
                    if constexpr (BIAS != 0) {
                        // biased codes overflow the maddubs i16 pairs, split into 4-bit halves
                        const __m128i a16a = _mm_and_si128(a16, _mm_set1_epi8(0x0F));
                        const __m128i a16b = _mm_and_si128(_mm_srli_epi16(a16, 4), _mm_set1_epi8(0x0F)); // srli is per 16-bit lane, mask out the neighbor bleed
                        const __m128i scales16b = _mm_set1_epi16(16 * scales_row[s]);
                        for (int t = 0; t < GROUP; t++) {
                            const __m128i q1_16 = _mm_loadu_si128((const __m128i *) &q1g[t][s * SUBBLK + u * 16]);
                            acc[t] = _mm_add_epi32(acc[t], _mm_add_epi32(
                                _mm_madd_epi16(scales16, _mm_maddubs_epi16(a16a, q1_16)),
                                _mm_madd_epi16(scales16b, _mm_maddubs_epi16(a16b, q1_16))));
                        }
                    } else {
                        for (int t = 0; t < GROUP; t++) {
                            acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(scales16,
                                _mm_maddubs_epi16(a16,
                                    _mm_loadu_si128((const __m128i *) &q1g[t][s * SUBBLK + u * 16]))));
                        }
                    }
                }
                if constexpr (BIAS != 0) {
                    corr = _mm_sub_epi32(corr, _mm_mullo_epi32(bsums_v, _mm_set1_epi32(BIAS * scales_row[s])));
                }
                if constexpr (HAS_MIN) {
                    s2 = _mm_add_epi32(s2, _mm_mullo_epi32(bsums_v, _mm_set1_epi32(mins_row[s])));
                }
            }

            int32_t corr_s[GROUP] = { 0, 0, 0, 0 };
            int32_t s2_s[GROUP] = { 0, 0, 0, 0 };
            if constexpr (BIAS != 0) {
                _mm_storeu_si128((__m128i *) corr_s, corr);
            }
            if constexpr (HAS_MIN) {
                _mm_storeu_si128((__m128i *) s2_s, s2);
            }

            for (int t = 0; t < GROUP; t++) {
                // 4 i32 lanes -> scalar (the per-pair dot, pre-correction)
                __m128i v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(2, 3, 0, 1));
                acc[t] = _mm_add_epi32(acc[t], v);
                v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(1, 0, 3, 2));
                acc[t] = _mm_add_epi32(acc[t], v);
                const int32_t s1 = _mm_cvtsi128_si32(acc[t]) + corr_s[t];

                float res = d0 * (float) s1;
                if constexpr (HAS_MIN) {
                    res -= dmin0 * (float) s2_s[t];
                }
                buf[ar * buf_stride + j0 + g + t] += src1.d[j0 + g + t] * res;
            }
        }
    }
}

#endif // __AVX__ && !__AVX2__

// main microtile entry point
template <int SUBBLK, bool HAS_MIN, int BIAS>
void tiled_run_microtile(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                         int i0, int j0, float * buf, int buf_stride) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    tiled_run_microtile_vnni<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, buf, buf_stride);
#elif defined(__AVX2__)
    tiled_run_microtile_avx2<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, buf, buf_stride);
#elif defined(__AVX__)
    tiled_run_microtile_avx<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, buf, buf_stride);
#else
    tiled_run_microtile_scalar<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, buf, buf_stride);
#endif
}

// explicit instantiations for the in-use formats (q4_K and q5_K share the constants)
template void tiled_run_microtile<32, true, 0>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                               int i0, int j0, float * buf, int buf_stride);
// iq4_xs and the other iq types: LUT-expanded codes, BIAS = 128
template void tiled_run_microtile<32, false, 128>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                  int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<16, false, 128>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                  int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<16, false, 32>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                 int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<16, false, 4>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<16, true, 0>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                               int i0, int j0, float * buf, int buf_stride);


#define MIN(a, b) ((a) < (b) ? (a) : (b))

// block_q8_K in 4-byte words, for the int32 gather indices
static_assert(sizeof(block_q8_K) == 292 && offsetof(block_q8_K, qs) == 4,
              "block_q8_K layout changed, fix the src1 repack");

// VNNI: the dpbusd kernel reads src1 codes in [k/4][row][4] order (one 512-bit
// load covers 16 rows x 4 k). Load one k-slab's codes as 16 rows of 16 int32s,
// transpose in registers, store as 16 rows of 16 int32s into the tile.
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)

bool tiled_repack_src1_codes(const block_q8_K * const * rows, int n_rows,
                             tiled_tile_src1 * tile, int kblk) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);

    for (int r0 = 0; r0 < n_rows; r0 += TILED_MICRO) {
        const int n = MIN(n_rows - r0, TILED_MICRO);

        // 4 chunks of 16 int32s (64 int32s total = 256 bytes = qs size)
        for (int c = 0; c < 4; c++) {
            __m512i v[16];
            for (int r = 0; r < TILED_MICRO; r++) {
                if (r < n) {
                    v[r] = _mm512_loadu_si512((const void *) ((const int32_t *) rows[r0 + r][kblk].qs + c * 16));
                } else {
                    v[r] = _mm512_setzero_si512();
                }
            }

            // 16x16 int32 transpose, 4 butterfly phases
            // Phase 1: 1-element interleave, pairs (0,1), (2,3), ..., (14,15)
            {
                const __m512i idx_a = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                const __m512i idx_b = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
                for (int i = 0; i < 16; i += 2) {
                    __m512i a = v[i], b = v[i+1];
                    v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
                    v[i+1] = _mm512_permutex2var_epi32(a, idx_b, b);
                }
            }
            // Phase 2: 2-element interleave, pairs (0,2), (1,3), (4,6), (5,7), ...
            {
                const __m512i idx_a = _mm512_setr_epi32(0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29);
                const __m512i idx_b = _mm512_setr_epi32(2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31);
                for (int i = 0; i < 16; i += 4) {
                    __m512i a = v[i],   b = v[i+2];
                    v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
                    v[i+2] = _mm512_permutex2var_epi32(a, idx_b, b);
                    a = v[i+1], b = v[i+3];
                    v[i+1] = _mm512_permutex2var_epi32(a, idx_a, b);
                    v[i+3] = _mm512_permutex2var_epi32(a, idx_b, b);
                }
            }
            // Phase 3: 4-element interleave, pairs (0,4), (1,5), ..., (7,11), (8,12), ...
            {
                const __m512i idx_a = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
                const __m512i idx_b = _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
                for (int i = 0; i < 16; i += 8) {
                    for (int j = 0; j < 4; j++) {
                        __m512i a = v[i+j], b = v[i+4+j];
                        v[i+j]   = _mm512_permutex2var_epi32(a, idx_a, b);
                        v[i+4+j] = _mm512_permutex2var_epi32(a, idx_b, b);
                    }
                }
            }
            // Phase 4: 8-element interleave, pairs (0,8), (1,9), ..., (7,15)
            {
                const __m512i idx_a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
                const __m512i idx_b = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
                for (int i = 0; i < 8; i++) {
                    __m512i a = v[i], b = v[i+8];
                    v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
                    v[i+8] = _mm512_permutex2var_epi32(a, idx_b, b);
                }
            }

            // store: v[g] holds 16 int32s for k-group (c*16+col_order[g]), rows r0..r0+15
            // the butterfly produces columns in a fixed permuted order
            static const int col_order[16] = {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15};
            for (int g = 0; g < 16; g++) {
                _mm512_store_si512((void *) (tile->q + ((c * 16 + col_order[g]) * TILED_TILE_ROWS + r0) * 4), v[g]);
            }
        }
    }
    return true;
}
#else
bool tiled_repack_src1_codes(const block_q8_K * const * rows, int n_rows,
                             tiled_tile_src1 * tile, int kblk) {
    GGML_UNUSED(rows); GGML_UNUSED(n_rows);
    GGML_UNUSED(tile); GGML_UNUSED(kblk);
    return false;
}
#endif


