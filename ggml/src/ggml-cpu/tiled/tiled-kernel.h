#pragma once

// Tiled matmul kernel API: tile structs, kernel definitions

// Currently only optimized for x86, new architectures should implement:
// tiled_run_microtile:  16x16 microkernel
// bit unpacking routines: tiled_unpk_nib4, tiled_unpk_2bit, tiled_unpk_or
#include "ggml-quants.h"
#include "ggml.h"
#include "ggml-cpu-impl.h" // ggml_compute_params; no-op for the consumers, which include it first

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

// Ensure total size of both panels plus result window under 512kb for L2 cache fit
static_assert(sizeof(tiled_tile_src0) + sizeof(tiled_tile_src1) + (sizeof(float) * 65536) < 512 * 1024,
              "tiled tile memory budget exceeded");

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
#endif


// Accumulate one 16x16 microtile (src0 rows [i0, i0+16), src1 cols [j0, j0+16))
// over the full 256-K slab held in the tiles into a j-major float buffer
// (row width buf_stride): buf[i*buf_stride + j] += partial.
// 
// No store to dst here: the driver holds the buffer across the 256-K slabs
// and transpose-stores it once, so each dst element is written a single time.
// SUBBLK/HAS_MIN/BIAS are the src0 format constants (see tiled_tile_src0).
template <int SUBBLK, bool HAS_MIN, int BIAS>
void tiled_run_microtile(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                         int i0, int j0, float * buf, int buf_stride);


// Defined when this arch's kernel reads the src1 tile codes in a non-natural order.
// If set, driver will call kernel methods `tiled_prepare_src1_interleave` and 
// `tiled_unpack_src1_q8_K_kernel` to prepare the tensor and macrotiles instead of doing a naive copy.
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
#define KERNEL_SRC1_UNPACK 1
#endif


#if defined(KERNEL_SRC1_UNPACK)
// the dpbusd kernel reads the src1 tile codes in the [k/4][row][4] order instead of standard row-major
void tiled_unpack_src1_q8_K_kernel(int n_rows, tiled_tile_src1 * tile,
                                   const int8_t * qv, int64_t nr1_pad, int64_t r_start, int64_t kblk);

// geometry of an additional src1 scratch region in wdata
struct tiled_interleave_geom {
    int8_t       * qv;      // base of the [slab][k/4][row][4] code region (null when not built)
    int64_t       nr1_pad;  // row count padded to 16 (0 then)
    size_t        bytes;    // wdata reservation size for the region (0 then)
};

tiled_interleave_geom tiled_get_interleave_geom(const struct ggml_compute_params * params,
                                                const struct ggml_tensor * src1,
                                                enum ggml_type vec_dot_type,
                                                int64_t ne10, int64_t nr1);

// The q8 codes of the whole tensor are scattered into a flat [slab][k/4-in-slab][row][4] wdata region 
// (rows padded to 16, zeroed tail) so the per (window, slab) unpack becomes a contiguous copy.
void tiled_prepare_src1_interleave(const struct ggml_compute_params * params,
                                   const struct ggml_tensor * src1,
                                   enum ggml_type vec_dot_type,
                                   int64_t ne10, int64_t nr1, int ith, int nth);

#endif

