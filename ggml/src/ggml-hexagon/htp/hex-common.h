#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static inline uint32_t hex_ceil_pow2(uint32_t x) {
    if (x <= 1) { return 1; }
    int p = 2;
    x--;
    while (x >>= 1) { p <<= 1; }
    return p;
}

static inline size_t hmx_ceil_div(size_t num, size_t den) {
    return (num + den - 1) / den;
}

static inline int32_t hex_is_aligned(const void * addr, uint32_t align) {
    return ((size_t) addr & (align - 1)) == 0;
}

static inline size_t hex_align_up(size_t v, size_t align) {
    return hmx_ceil_div(v, align) * align;
}

static inline size_t hex_align_down(size_t v, size_t align) {
    return (v / align) * align;
}

static inline int32_t hex_is_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    return right_off <= chunk_size;
}

static inline uint32_t hex_round_up(uint32_t n, uint32_t m) {
    return m * ((n + m - 1) / m);
}

static inline size_t hex_smin(size_t a, size_t b) {
    return a < b ? a : b;
}

static inline size_t hex_smax(size_t a, size_t b) {
    return a > b ? a : b;
}

static inline void hex_swap_ptr(void ** p1, void ** p2) {
    void * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

static inline bool hex_mul_overflow(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

static inline bool hex_add_overflow(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return true;
    *out = a + b;
    return false;
}

#endif // HEX_COMMON_H
