// mulmat microtile kernels, only optimized for x86 archs so far.

#include "tiled-kernel.h"
#include "tiled.h"
#include "ggml-cpu-impl.h"

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

    const __m512 d1_vec = _mm512_loadu_ps(&src1.d[j0]);

    __m512i s1_acc[NUM_ROWS];
    for (int t = 0; t < NUM_ROWS; t++) { s1_acc[t] = _mm512_setzero_si512(); }
    __m512i s2_acc[NUM_ROWS];
    if constexpr (HAS_MIN) {
        for (int t = 0; t < NUM_ROWS; t++) { s2_acc[t] = _mm512_setzero_si512(); }
    }

    const int32_t * bsums = src1.bsums; // int32 per-16 sums; NS > 1 combines NS lanes per subblock
    for (int s = 0; s < NB; s++) {
        __m512i bsums32 = _mm512_loadu_si512((const __m512i *) &bsums[s * NS * TILED_TILE_ROWS + j0]);
        for (int u = 1; u < NS; u++) {
            bsums32 = _mm512_add_epi32(bsums32, _mm512_loadu_si512((const __m512i *) &bsums[(s * NS + u) * TILED_TILE_ROWS + j0]));
        }

        __m512i bias32 = _mm512_setzero_si512();
        if constexpr (BIAS != 0) {
            bias32 = _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(BIAS));
        }

        __m512i acc16[NUM_ROWS];
        for (int t = 0; t < NUM_ROWS; t++) { acc16[t] = _mm512_setzero_si512(); }

        #pragma GCC unroll 8 // pragma unrolled justified by measuing with/without
        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            const __m512i codes = _mm512_loadu_si512((const __m512i *) &src1.q[kg * TILED_TILE_ROWS * 4 + j0 * 4]);
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
            __m512i bsums32 = _mm512_loadu_si512((const __m512i *) &bsums[s * NS * TILED_TILE_ROWS + j0]);
            for (int u = 1; u < NS; u++) {
                bsums32 = _mm512_add_epi32(bsums32, _mm512_loadu_si512((const __m512i *) &bsums[(s * NS + u) * TILED_TILE_ROWS + j0]));
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
        _mm512_storeu_ps(p, _mm512_add_ps(_mm512_loadu_ps(p), _mm512_mul_ps(result, d1_vec)));
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
// scale is applied per 128-bit half
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
                // one 32B a load and one 256-bit maddubs per (s, column)
                for (int s = 0; s < NB; s++) {
                    const __m256i q0_32 = _mm256_loadu_si256((const __m256i *) &q0[s * SUBBLK]);
                    const __m256i scales16 = _mm256_set1_epi16(scales_row[s]);
                    const __m256i bsums_v = _mm256_add_epi32(
                        _mm256_loadu_si256((const __m256i *) &src1.bsums[s * 2 * TILED_TILE_ROWS + j0 + g]),
                        _mm256_loadu_si256((const __m256i *) &src1.bsums[(s * 2 + 1) * TILED_TILE_ROWS + j0 + g]));
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm256_add_epi32(acc[t],
                            _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(
                                q0_32, _mm256_loadu_si256((const __m256i *) &q1g[t][s * SUBBLK]))));
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
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(
                            scalesv, _mm256_maddubs_epi16(
                                q0_32, _mm256_loadu_si256((const __m256i *) &q1g[t][sp * SUBBLK]))));
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
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(scales16,
                            _mm_maddubs_epi16(a16,
                                _mm_loadu_si128((const __m128i *) &q1g[t][s * SUBBLK + u * 16]))));
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
template void tiled_run_microtile<16, false, 32>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                 int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<16, false, 4>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<16, true, 0>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                               int i0, int j0, float * buf, int buf_stride);


#if defined(KERNEL_SRC1_UNPACK)

// Unpack routines for VNNI
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)

// Unpack one window/macrotile in a form the VNNI kernel can work with.
// dpbusd reads the src1 codes in the [k/4][row][4] order (one 64B
// vector per k-group x 16 rows). The region was interleaved up front by
// tiled_interleave_src1_q8_K, so the tile column is 4 bytes per row, contiguous. 
void tiled_unpack_src1_q8_K_kernel(int n_rows, tiled_tile_src1 * tile,
                                   const int8_t * qv, int64_t nr1_pad, int64_t r_start, int64_t kblk) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    const int n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    const int n_qv = TILED_TILE_K / 4;
    GGML_ASSERT(qv);
    const int n_copy = (int) MIN((int64_t) n_padded, nr1_pad - r_start);
    // one k-group column per iteration: the window's dpbusd vectors, contiguous
    for (int g = 0; g < n_qv; g++) {
        memcpy(&tile->q[g * TILED_TILE_ROWS * 4],
               qv + ((int64_t) (kblk * n_qv + g) * nr1_pad + r_start) * 4, (size_t) n_copy * 4);
        memset(&tile->q[g * TILED_TILE_ROWS * 4 + (size_t) n_copy * 4], 0, (size_t) (n_padded - n_copy) * 4);
    }
}


// where the custom-packed region sits in wdata: after the F32 to q8_K conversion when src1
// is not already q8_K, else at the base (see the struct in tiled-kernel.h)
tiled_interleave_geom tiled_get_interleave_geom(const struct ggml_compute_params * params,
                                                const struct ggml_tensor * src1,
                                                enum ggml_type vec_dot_type,
                                                int64_t ne10, int64_t nr1) {
    tiled_interleave_geom geom = { nullptr, 0, 0 };
    geom.nr1_pad = (nr1 + 15) & ~15LL;
    geom.bytes = ggml_tiled_extra_wdata_len(ne10, nr1);
    const size_t off = (src1->type != vec_dot_type) ? ggml_row_size(vec_dot_type, ne10) * (size_t) nr1 : 0;
    geom.qv = (int8_t *) ((char *) params->wdata + off);
    return geom;
}
// block_q8_K in 4-byte words, for the int32 gather indices of the src1 code unpack
static constexpr int TILED_Q8_K_WORDS = sizeof(block_q8_K) / 4;
static_assert(sizeof(block_q8_K) == 292 && offsetof(block_q8_K, qs) == 4,
              "block_q8_K layout changed, fix the src1 gather indices");

// The dpbusd kernel reads the src1 codes in [k/4][row][4], computing 16
// partial dots.  This code scatters the src1 codes into the region so the
// tile columns become contiguous loads instead of scattered reads.
static void tiled_interleave_src1_q8_K(const block_q8_K * rows, int64_t row_stride,
                                       int64_t r_start, int64_t r_end,
                                       int64_t n_k, int64_t nr1, int64_t nr1_pad, int8_t * qv) {

    //   qv[(s * TILED_TILE_K/4 + g) * nr1_pad + r] = qs[4*g + 0..3]
    //     of the src1 block (global row r, slab s)

    const int64_t n_slabs = n_k / TILED_TILE_K;
    GGML_ASSERT(qv);
    GGML_ASSERT(n_k % TILED_TILE_K == 0);
    GGML_ASSERT((r_start & 15) == 0 && (r_end & 15) == 0 && r_end <= nr1_pad);
    // one masked 16-lane int32 gather + one 64B store per
    // (slab, k-group, 16-row) block; masked lanes (past nr1) gather zeros
    {
        // row_off[r] = int32 index (in words) of rows[r].qs[0] (d is 4B, qs at +4);
        // row_stride * TILED_Q8_K_WORDS stays in int32 for any realistic row count
        const int32_t row_stride_w = (int32_t) row_stride * TILED_Q8_K_WORDS;
        int32_t row_off[TILED_MICRO];
        for (int r = 0; r < TILED_MICRO; r++) row_off[r] = r * row_stride_w + 1;
        const __m512i idx_row = _mm512_loadu_si512((const __m512i *) row_off);
        const int32_t * base = (const int32_t *) rows;
        for (int64_t r0 = r_start; r0 < r_end; r0 += TILED_MICRO) {
            // masked lanes (past nr1) gather into the zero register, so the
            // 16-row pad tail of the region is zeroed here, no separate memset
            const int64_t n = (nr1 > r0) ? nr1 - r0 : 0;
            const __mmask16 k = (n >= TILED_MICRO) ? 0xffff : ((__mmask16) ((1u << n) - 1));
            const int64_t rl = r0 - r_start; // rows points at global row r_start
            for (int64_t s = 0; s < n_slabs; s++) {
                // block (row rl, slab s) is rl*row_stride + s blocks from rows
                const __m512i base_idx = _mm512_set1_epi32((int32_t) ((rl * row_stride + s) * TILED_Q8_K_WORDS));
                int8_t * out = qv + s * TILED_TILE_K * nr1_pad + r0 * 4;
                for (int g = 0; g < TILED_TILE_K / 4; g++) {
                    const __m512i idx = _mm512_add_epi32(_mm512_add_epi32(idx_row, base_idx), _mm512_set1_epi32(g));
                    const __m512i v = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), k, idx, base, 4);
                    _mm512_storeu_si512((void *) (out + g * nr1_pad * 4), v);
                }
            }
        }
        return;
    }
}

void tiled_prepare_src1_interleave(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src1,
        enum ggml_type vec_dot_type,
        int64_t ne10,
        int64_t nr1,
        int ith,
        int nth) {
    const tiled_interleave_geom geom = tiled_get_interleave_geom(params, src1, vec_dot_type, ne10, nr1);
    int8_t * qv = geom.qv;
    const int64_t nr1_pad = geom.nr1_pad;
    const int64_t k1_pad = geom.bytes / (size_t) nr1_pad; // region width in elements (bytes = k1_pad * nr1_pad)
    const block_q8_K * src1_codes;
    int64_t src1_stride;
    if (src1->type != vec_dot_type) {
        ggml_barrier(params->threadpool);
        // wdata = [F32 to q8_K conversion][interleave region]; 
        const size_t off = (size_t) ((const int8_t *) geom.qv - (const int8_t *) params->wdata);
        GGML_ASSERT(params->wsize >= off + geom.bytes);
        src1_codes = (const block_q8_K *) params->wdata;
        src1_stride = ne10 / 256; // wdata rows are contiguous 256 blocks
    } else {
        GGML_ASSERT(params->wsize >= geom.bytes);
        src1_codes = (const block_q8_K *) src1->data;
        src1_stride = src1->nb[1] / sizeof(block_q8_K);
    }
    const int64_t n_groups = nr1_pad / 16;
    const int64_t g0 = (int64_t) ith * n_groups / nth;
    const int64_t g1 = (int64_t) (ith + 1) * n_groups / nth;
    for (int64_t g = g0; g < g1; g++) {
        tiled_interleave_src1_q8_K(src1_codes + g * 16 * src1_stride, src1_stride,
                                   g * 16, (g + 1) * 16, k1_pad, nr1, nr1_pad, qv);
    }
}
#endif // #if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
#endif // #if defined(KERNEL_SRC1_UNPACK)