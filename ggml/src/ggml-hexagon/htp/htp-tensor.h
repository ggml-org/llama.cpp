#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#include <stdint.h>
#include "htp-ops.h"
#include "hex-bitmap.h"

static inline struct htp_tensor * htp_tensor_alias(const struct htp_tensor * t) {
    return (struct htp_tensor *) (uintptr_t) t->alias;
}

static inline void * htp_tensor_data(const struct htp_tensor * t) {
    return (void *) (uintptr_t) t->data;
}

static inline uint32_t * htp_tensor_flags(const struct htp_tensor * t) {
    return (uint32_t *) &t->flags;
}

static inline void htp_tensor_make_dirty(const struct htp_tensor * t, uint32_t * dirty_map) {
    struct htp_tensor * curr = (struct htp_tensor *) t;
    do {
        bitmap_set(dirty_map, curr->ti);
        curr = htp_tensor_alias(curr);
    } while (curr != t);
}

static inline void htp_tensor_make_clean(const struct htp_tensor * t, uint32_t * dirty_map) {
    bitmap_clear(dirty_map, t->ti);
}

struct htp_context;
void htp_tensor_flush(struct htp_context * ctx, const struct htp_tensor * t);
void htp_tensor_flush_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);

#endif // HTP_TENSOR_H
