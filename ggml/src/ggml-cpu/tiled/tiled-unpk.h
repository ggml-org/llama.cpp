#pragma once

// Tiled unpack primitives.  Defined as inline in header so they can be inlined across translation units.

#include <stdint.h>

// packed 4-bit codes -> low nibbles (lo) + high nibbles (hi), 32 bytes
inline void tiled_unpk_nib4(const uint8_t * src, uint8_t * lo, uint8_t * hi);
// 2-bit values at bit offset S, 32 bytes
template <int S> inline void tiled_unpk_2bit(const uint8_t * src, uint8_t * dst);
// OR the M-bit value at bit offset S of src into bit offset D of dst, 32 bytes
template <int S, int D, int M> inline void tiled_unpk_or(uint8_t * dst, const uint8_t * src);
// 16-entry byte LUT: dst[j] = lut[src[j]] (16 bytes)
inline void tiled_lut8(const uint8_t * lut, const uint8_t * src, uint8_t * dst);
// 32 grid magnitudes (4 x 64-bit groups) + 4 sign bytes -> biased codes (value + 128)
inline void tiled_unpk_sign32(uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3,
                              const uint8_t signs[4], uint8_t * dst32);
// 8 ternary grid bytes (0 = 0, 1 = +1, 0xFF = -1): dst[j] = 128 + delta + 8 * (int8_t) src[j]
inline void tiled_unpk_tern8(const uint8_t * src, int8_t delta, uint8_t * dst);

// Each arch should include its version of unpack primitives here.
#if defined(__AVX2__)
#    include "../arch/x86/tiled-unpk.h"
#else
#    include "tiled-unpk-generic.h"
#endif
