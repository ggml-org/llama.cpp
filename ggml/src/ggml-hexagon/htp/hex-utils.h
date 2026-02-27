#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#include <stdbool.h>
#include <stdint.h>

#include "hexagon_types.h"

#include "hex-fastdiv.h"
#include "hex-dump.h"

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static inline uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm volatile(" %0 = c15:14\n" : "=r"(cycles));
    return cycles;
}

static inline uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm volatile(" %0 = c19:18\n" : "=r"(pktcnt));
    return pktcnt;
}

static inline int32_t hex_is_aligned(void * addr, uint32_t align) {
    return ((size_t) addr & (align - 1)) == 0;
}

static inline int32_t hex_is_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    return right_off <= chunk_size;
}

static inline uint32_t hex_round_up(uint32_t n, uint32_t m) {
    return m * ((n + m - 1) / m);
}

static inline void hex_l2fetch(const void * p, uint32_t width, uint32_t stride, uint32_t height) {
    const uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((void *) p, control);
}

#endif /* HEX_UTILS_H */
