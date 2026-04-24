#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <qurt_memory.h>
#include <qurt.h>

#include "hexagon_types.h"
#include "hexagon_protos.h"

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

static inline void hex_l2fetch(const void * p, uint32_t width, uint32_t stride, uint32_t height) {
    const uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((void *) p, control);
}

#define HEX_L2_LINE_SIZE  64
#define HEX_L2_FLUSH_SIZE (128 * 1024)

static inline void hex_l2flush(void * addr, size_t size) {
    if (size > HEX_L2_FLUSH_SIZE) {
        qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    } else {
        const uint32_t s = (uint32_t) addr;
        const uint32_t e = s + size;
        for (uint32_t i = s; i < e; i += HEX_L2_LINE_SIZE * 4) {
            Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 0);
            Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 1);
            Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 2);
            Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 3);
        }
    }
}

static inline void hex_pause() {
    asm volatile(" pause(#255)\n");
}

#ifndef HEX_NUM_PMU_COUNTERS
#define HEX_NUM_PMU_COUNTERS 8
#endif

static inline void hex_get_pmu(uint32_t counters[]) {
#if __HVX_ARCH__ >= 79
    asm volatile("%0 = upmucnt0" : "=r"(counters[0]));
    asm volatile("%0 = upmucnt1" : "=r"(counters[1]));
    asm volatile("%0 = upmucnt2" : "=r"(counters[2]));
    asm volatile("%0 = upmucnt3" : "=r"(counters[3]));
    asm volatile("%0 = upmucnt4" : "=r"(counters[4]));
    asm volatile("%0 = upmucnt5" : "=r"(counters[5]));
    asm volatile("%0 = upmucnt6" : "=r"(counters[6]));
    asm volatile("%0 = upmucnt7" : "=r"(counters[7]));
#else
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
    // qurt_pmu_get_pmucnt(counters);
#endif
}

#endif /* HEX_UTILS_H */
