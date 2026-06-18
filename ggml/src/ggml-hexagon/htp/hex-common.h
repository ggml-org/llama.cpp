#ifndef HEX_COMMON_H
#define HEX_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef HEX_UTILS_H
static inline size_t hex_align_down(size_t value, size_t alignment) {
    return value & ~(alignment - 1);
}

static inline size_t hex_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static inline size_t hex_smin(size_t a, size_t b) {
    return a < b ? a : b;
}
#endif

static inline bool hmx_mul_overflow(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

static inline bool hmx_add_overflow(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return true;
    *out = a + b;
    return false;
}

#endif // HEX_COMMON_H
