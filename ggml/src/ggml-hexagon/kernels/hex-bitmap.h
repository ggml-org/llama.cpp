#ifndef HEX_BITMAP_H
#define HEX_BITMAP_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

static inline void bitmap_set(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] |= (1U << (idx % 32));
}

static inline void bitmap_clear(uint32_t * bitmap, uint32_t idx) {
    bitmap[idx / 32] &= ~(1U << (idx % 32));
}

static inline bool bitmap_test(const uint32_t * bitmap, uint32_t idx) {
    return (bitmap[idx / 32] & (1U << (idx % 32))) != 0;
}

static inline void bitmap_reset(uint32_t * bitmap, size_t size_in_bits) {
    memset(bitmap, 0, ((size_in_bits + 31) / 32) * sizeof(uint32_t));
}

#endif // HEX_BITMAP_H
