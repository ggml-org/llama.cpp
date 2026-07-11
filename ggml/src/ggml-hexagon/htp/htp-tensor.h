#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#include <stdint.h>
#include "htp-ops.h"

static inline struct htp_tensor * htp_tensor_alias(const struct htp_tensor * t) {
    return (struct htp_tensor *) (uintptr_t) t->alias;
}

static inline void * htp_tensor_data(const struct htp_tensor * t) {
    return (void *) (uintptr_t) t->data;
}

static inline uint32_t * htp_tensor_flags(const struct htp_tensor * t) {
    return (uint32_t *) &t->flags;
}

static inline void htp_tensor_make_dirty(const struct htp_tensor * t) {
    struct htp_tensor * curr = (struct htp_tensor *) t;
    do {
        curr->flags |= HTP_TENSOR_DIRTY;
        curr = htp_tensor_alias(curr);
    } while (curr != t);
}

static inline void htp_tensor_make_clean(const struct htp_tensor * t) {
    if (htp_tensor_alias(t) == t) {
        *htp_tensor_flags(t) &= ~HTP_TENSOR_DIRTY;
    }
}

#endif // HTP_TENSOR_H
