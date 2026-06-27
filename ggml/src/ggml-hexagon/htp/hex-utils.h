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

#endif /* HEX_UTILS_H */
