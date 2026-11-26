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
#include "hex-common.h"

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
#endif
}

static inline void hex_l2fetch(const void * p, uint32_t width, uint32_t stride, uint32_t height) {
    const uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((void *) p, control);
}

static inline void hex_l2fetch_block(const void * addr, size_t size) {
    if (size == 0) return;
    const uint32_t width = 16384; // 16KB rows
    const uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

static inline void hex_l2flush(void * addr, size_t size) {
    if (size == 0) return;
    // Step by 64B (L1D line size on v73+ and L2 line size on v73) so no L1D lines are skipped.
    const uint32_t line = 64;
    const uint32_t s = ((uint32_t) addr) & ~(line - 1);
    const uint32_t e = (((uint32_t) addr) + size + line - 1) & ~(line - 1);
    for (uint32_t i = s; i < e; i += line) {
        Q6_dccleaninva_A((void *) i);
    }
    asm volatile(" syncht\n" : : : "memory");
}

static inline void hex_pause() {
    asm volatile(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
