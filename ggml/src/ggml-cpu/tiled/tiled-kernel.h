#pragma once

// Tiled matmul kernel API: tile structs, kernel definitions

// Currently only optimized for x86, new architectures should implement:
// tiled_run_microtile:  16x16 microkernel
// bit unpacking routines: tiled_unpk_nib4, tiled_unpk_2bit, tiled_unpk_or
// LUT value expansion routines: tiled_lut8, tiled_unpk_sign8, tiled_unpk_tern8

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define TILED_TILE_K    256 // one QK_K block
#define TILED_TILE_ROWS 256 // max window rows, ragged at edges
#define TILED_MICRO     16  // microtile edge (also the bsums code-sum granularity)

// src0 tile: weight side, shared by all formats.
// scales/mins are sized for the max subblock count (SUBBLK=16);
// SUBBLK=32 formats index at stride 8 and leave the slack unused.
struct tiled_tile_src0 {
    static constexpr int NB_MAX = TILED_TILE_K / 16; // max subblocks per 256-elem block

    alignas(32) uint8_t q[TILED_TILE_ROWS * TILED_TILE_K]; // unsigned quants, widened to uint8
    float    d[TILED_TILE_ROWS];  // One d from each input block, widened to f32
    float    dmin[TILED_TILE_ROWS]; // dmin from each input block (if applicable), widened to F32
    int32_t   scales[TILED_TILE_ROWS * NB_MAX];  // per-subblock scale, stored as int32_t
    int32_t   mins[TILED_TILE_ROWS * NB_MAX];    // per-subblock min, used when HAS_MIN
};

// src1 tile: built from q8_K (wdata)
struct tiled_tile_src1 {
    // q8 codes, one byte per element. Note for VNNI these are reshaped + transposed to be suitable for dpbusd.
    alignas(64) int8_t  q[TILED_TILE_ROWS * TILED_TILE_K];
    // per-16 code sums from q8_k (int16), widened to int32 so the kernels load them directly, no per-use cvt
    alignas(64) int32_t bsums[(TILED_TILE_K / 16) * TILED_TILE_ROWS];
    // f32 (not f16): q8_k stores fp16, the unpack converts once
    float       d[TILED_TILE_ROWS];
};

// per-thread workspace: all tiled state lives here, allocated in wdata (one slot per thread)
struct tiled_ws {
    tiled_tile_src0 src0;
    tiled_tile_src1 src1;
    alignas(64) float acc[TILED_TILE_ROWS * TILED_TILE_ROWS];
};

static_assert(sizeof(tiled_ws) <= 512 * 1024, "tiled workspace exceeds 512KB per-thread budget");

// unpack primitives for reading quants, defined as inline here to keep arch-specific code in kernel.h/.cpp
// If this section gets too hairy later, we can break up into separate includes.
#if defined(__AVX2__)
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


// Unpacking kernels for IQ quants TODO normalize the layout with the bit unpackers

// LUT value expansion for the LUT-based formats (iq4_xs, iq grids): the bit unpackers
// above give the indices, these expand 8/16 of them to widened codes in one pass
// 16-entry byte LUT: dst[j] = lut[src[j]] (16 bytes)
inline void tiled_lut8(const uint8_t * lut, const uint8_t * src, uint8_t * dst) {
    _mm_storeu_si128((__m128i *) dst, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) lut),
                                                       _mm_loadu_si128((const __m128i *) src)));
}


// 8 grid values with a per-lane sign flip: bit j of sign selects -src[j], else src[j];
// result is (v + 128) or (128 - v) mod 256, safe for v < 128
// sign is expanded to per-lane byte masks (0xFF / 0x00) so the AND is a per-lane select;
// ksign_spread maps a nibble n to a word whose byte j is 0xFF if bit j of n
static const uint32_t ksign_spread[16] = {
    0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF,
    0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF,
    0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF,
    0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF,
};
inline void tiled_unpk_sign8(const uint8_t * src, uint8_t sign, uint8_t * dst) {
    const __m128i mask = _mm_setr_epi32(ksign_spread[sign & 15], ksign_spread[(sign >> 4) & 15], 0, 0);
    const __m128i v = _mm_loadl_epi64((const __m128i *) src);
    const __m128i m = _mm_and_si128(v, mask);
    _mm_storel_epi64((__m128i *) dst,
                     _mm_sub_epi8(_mm_add_epi8(v, _mm_set1_epi8((int8_t) 128)), _mm_add_epi8(m, m)));
}
// 8 ternary grid bytes (0 = 0, 1 = +1, 0xFF = -1): dst[j] = 128 + delta + 8 * (int8_t) src[j]
inline void tiled_unpk_tern8(const uint8_t * src, int8_t delta, uint8_t * dst) {
    const __m128i v = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *) src));
    const __m128i p = _mm_add_epi16(_mm_slli_epi16(v, 3), _mm_set1_epi16(128 + (int) delta));
    _mm_storel_epi64((__m128i *) dst, _mm_packus_epi16(p, _mm_setzero_si128()));
}
#else
inline void tiled_unpk_nib4(const uint8_t * src, uint8_t * lo, uint8_t * hi) {
    for (int l = 0; l < 32; l++) { lo[l] = (uint8_t) (src[l] & 0xF); hi[l] = (uint8_t) (src[l] >> 4); }
}
template <int S>
inline void tiled_unpk_2bit(const uint8_t * src, uint8_t * dst) {
    for (int l = 0; l < 32; l++) { dst[l] = (uint8_t) ((src[l] >> S) & 3); }
}
template <int S, int D, int M>
inline void tiled_unpk_or(uint8_t * dst, const uint8_t * src) {
    for (int l = 0; l < 32; l++) { dst[l] = (uint8_t) (dst[l] | (((src[l] >> S) & M) << D)); }
}
inline void tiled_lut8(const uint8_t * lut, const uint8_t * src, uint8_t * dst) {
    for (int j = 0; j < 16; j++) { dst[j] = lut[src[j]]; }
}
inline void tiled_unpk_sign8(const uint8_t * src, uint8_t sign, uint8_t * dst) {
    for (int j = 0; j < 8; j++) { dst[j] = (sign & (1 << j)) ? (uint8_t) (128 - src[j]) : (uint8_t) (128 + src[j]); }
}
// 8 ternary grid bytes (0 = 0, 1 = +1, 0xFF = -1): dst[j] = 128 + delta + 8 * (int8_t) src[j]
inline void tiled_unpk_tern8(const uint8_t * src, int8_t delta, uint8_t * dst) {
    for (int j = 0; j < 8; j++) { dst[j] = (uint8_t) (128 + (int) delta + 8 * (int8_t) src[j]); }
}
#endif

// Accumulate one 16x16 microtile (src0 rows [i0, i0+16), src1 cols [j0, j0+16))
// over the full 256-K slab held in the tiles into a j-major float buffer
// (row width buf_stride): buf[i*buf_stride + j] += partial.
// SUBBLK/HAS_MIN/BIAS are the src0 format constants (see tiled_tile_src0).
template <int SUBBLK, bool HAS_MIN, int BIAS>
void tiled_run_microtile(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                         int i0, int j0, float * buf, int buf_stride);

// If this architecture requires repacking of src1 int8 weights, do so and return true.
// Otherwise, return false and driver will fill as natural [row][k] contiguous rows.
bool tiled_repack_src1_codes(const block_q8_K * const * rows,
                             int n_rows, tiled_tile_src1 * tile, int kblk);

