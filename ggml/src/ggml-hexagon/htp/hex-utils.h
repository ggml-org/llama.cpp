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

#define HEX_L2_LINE_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_LINE_SIZE * 4) // flush granularity (lines per loop iteration)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

static inline void hex_l2flush(void * addr, size_t size) {
    const uint32_t s = ((uint32_t) addr) & ~(HEX_L2_LINE_SIZE - 1);
    const uint32_t e = (((uint32_t) addr) + size + HEX_L2_LINE_SIZE - 1) & ~(HEX_L2_LINE_SIZE - 1);
    for (uint32_t i = s; i < e; i += HEX_L2_BLOCK_SIZE) {
        Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 0);
        Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 1);
        Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 2);
        Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 3);
    }
}

static inline void hex_pause() {
    asm volatile(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
