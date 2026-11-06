#ifndef HTP_DCACHE_H
#define HTP_DCACHE_H

#include <stdint.h>
#include "htp-ops.h"

#define HTP_MAX_DIRTY_RANGES 64

struct htp_dirty_range {
    uint32_t start;
    uint32_t end;
};

struct htp_context;

void htp_dcache_flush_dirty_ranges(struct htp_context * ctx);
void htp_dcache_flush_dirty_tensors(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);
void htp_dcache_mark_dirty_tensors(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);

#endif // HTP_DCACHE_H
