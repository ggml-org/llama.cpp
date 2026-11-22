#ifndef HTP_TRACE_H
#define HTP_TRACE_H

#include <stdint.h>

#include "hex-utils.h"
#include "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

struct htp_thread_trace {
    uint32_t count;
    uint32_t max_events;
    struct htp_trace_desc * events;
};

static inline void htp_trace_event(struct htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    if (!tr->max_events) {
        return;
    }
    uint32_t i = tr->count++;
    if (i < tr->max_events) {
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
    }
}

static inline void htp_trace_event_start(struct htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

static inline void htp_trace_event_stop(struct htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HTP_TRACE_H */
