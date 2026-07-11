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

#endif // HTP_TENSOR_H
