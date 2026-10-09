#pragma once

// scalar unpack primitives, reference implementation.

#include <stdint.h>

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
inline void tiled_unpk_sign32(uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3,
                              const uint8_t signs[4], uint8_t * dst32) {
    const uint64_t g[4] = { g0, g1, g2, g3 };
    for (int l = 0; l < 4; l++) {
        const uint8_t * v = (const uint8_t *) &g[l];
        const uint8_t s = signs[l];
        for (int j = 0; j < 8; j++) {
            dst32[8 * l + j] = (s & (1 << j)) ? (uint8_t) (128 - v[j]) : (uint8_t) (128 + v[j]);
        }
    }
}
// 8 ternary grid bytes (0 = 0, 1 = +1, 0xFF = -1): dst[j] = 128 + delta + 8 * (int8_t) src[j]
inline void tiled_unpk_tern8(const uint8_t * src, int8_t delta, uint8_t * dst) {
    for (int j = 0; j < 8; j++) { dst[j] = (uint8_t) (128 + (int) delta + 8 * (int8_t) src[j]); }
}
