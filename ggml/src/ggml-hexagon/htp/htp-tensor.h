#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#include <stdint.h>
#include "htp-ops.h"
#include "hex-bitmap.h"

static inline void * htp_tensor_data(const struct htp_tensor * t) {
    return (void *) (uintptr_t) t->data;
}

static inline uint32_t * htp_tensor_flags(const struct htp_tensor * t) {
    return (uint32_t *) &t->flags;
}

static inline uint32_t htp_tensor_get_row_size(int type, uint32_t ne00) {
    switch (type) {
        case HTP_TYPE_F32:  return ne00 * 4;
        case HTP_TYPE_F16:  return ne00 * 2;
        case HTP_TYPE_Q8_0: return (ne00 / 32) * 34;
        default:            return 0;
    }
}

struct htp_context;
void htp_tensor_flush_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);
void htp_tensor_dirty_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);

#endif // HTP_TENSOR_H
