#pragma once

// AVX2 unpack primitives, x86 only. Each body handles 32 bytes per call (lut8/tern8 handle 16/8).

#include <stdint.h>
#include <immintrin.h>

// packed 4-bit codes -> low nibbles (lo) + high nibbles (hi)
inline void tiled_unpk_nib4(const uint8_t * src, uint8_t * lo, uint8_t * hi) {
    const __m256i v = _mm256_loadu_si256((const __m256i *) src);
    // mask before the lane shift so bits do not cross byte boundaries
    _mm256_storeu_si256((__m256i *) lo, _mm256_and_si256(v, _mm256_set1_epi8(0x0F)));
    _mm256_storeu_si256((__m256i *) hi, _mm256_srli_epi32(_mm256_and_si256(v, _mm256_set1_epi8((int8_t) 0xF0)), 4));
}
// 2-bit values at bit offset S
template <int S> inline void tiled_unpk_2bit(const uint8_t * src, uint8_t * dst) {
    _mm256_storeu_si256((__m256i *) dst, _mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256((const __m256i *) src), S), _mm256_set1_epi8(0x03)));
}
// OR the M-bit value at bit offset S of src into bit offset D of dst
template <int S, int D, int M>
inline void tiled_unpk_or(uint8_t * dst, const uint8_t * src) {
    const __m256i v = _mm256_slli_epi32(_mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256((const __m256i *) src), S), _mm256_set1_epi8((uint8_t) M)), D);
    _mm256_storeu_si256((__m256i *) dst, _mm256_or_si256(_mm256_loadu_si256((const __m256i *) dst), v));
}

// LUT value expansion for the LUT-based formats: the bit unpackers give the indices,
// these expand them to widened codes

// 16-entry byte LUT: dst[j] = lut[src[j]] (16 bytes)
inline void tiled_lut8(const uint8_t * lut, const uint8_t * src, uint8_t * dst) {
    _mm_storeu_si128((__m128i *) dst, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) lut),
                                                       _mm_loadu_si128((const __m128i *) src)));
}
// 32 grid magnitudes (4 x 64-bit groups g0..g3, 8 values each) + 4 sign bytes
// (byte l signs values 8*l .. 8*l+7) -> codes stored as (value + 128)
inline void tiled_unpk_sign32(uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3,
                              const uint8_t signs[4], uint8_t * dst32) {
    const __m256i v  = _mm256_set_epi64x((int64_t) g3, (int64_t) g2, (int64_t) g1, (int64_t) g0);
    const __m256i sv = _mm256_shuffle_epi8(_mm256_set1_epi32((int32_t) (signs[0] | signs[1] << 8 | signs[2] << 16 | signs[3] << 24)),
                                           _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i sel  = _mm256_set1_epi64x((int64_t) 0x8040201008040201ULL);
    // 0xFF in every lane whose sign bit is set; GFNI does the and+compare in one instruction
#ifdef __GFNI__
    const __m256i mask = _mm256_gf2p8affine_epi64_epi8(sel, sv, 0);
#else
    const __m256i mask = _mm256_cmpeq_epi8(_mm256_and_si256(sv, sel), sel);
#endif
    // v ^ mask - mask negates the signed lanes (v elsewhere); + 128 gives the biased code
    const __m256i sgn  = _mm256_sub_epi8(_mm256_xor_si256(v, mask), mask);
    _mm256_storeu_si256((__m256i *) dst32, _mm256_add_epi8(sgn, _mm256_set1_epi8((int8_t) 128)));
}
// 8 ternary grid bytes (0 = 0, 1 = +1, 0xFF = -1): dst[j] = 128 + delta + 8 * (int8_t) src[j]
inline void tiled_unpk_tern8(const uint8_t * src, int8_t delta, uint8_t * dst) {
    const __m128i v = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *) src));
    const __m128i p = _mm_add_epi16(_mm_slli_epi16(v, 3), _mm_set1_epi16(128 + (int) delta));
    _mm_storel_epi64((__m128i *) dst, _mm_packus_epi16(p, _mm_setzero_si128()));
}
