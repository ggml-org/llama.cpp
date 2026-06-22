#ifndef HEX_PROFILE_H
#define HEX_PROFILE_H

#include <stdbool.h>
#include <stdint.h>
#include <qurt.h>

#include "hex-utils.h"
#include "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

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

struct htp_thread_trace {
    uint32_t count;
    uint32_t max_events;
    struct htp_trace_desc * events;
};

static inline void htp_trace_event(struct htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    if (tr && tr->events && tr->count < tr->max_events) {
        uint32_t idx = tr->count;
        tr->events[idx].id = id;
        tr->events[idx].info = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[idx].cycles = (uint32_t) hex_get_cycles();
        tr->count++;
    }
}

static inline void htp_trace_event_start(struct htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

static inline void htp_trace_event_stop(struct htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HEX_PROFILE_H */
