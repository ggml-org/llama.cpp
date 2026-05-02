#ifndef VTCM_UTILS_H
#define VTCM_UTILS_H

#include "hex-utils.h"

#include <assert.h>
#include <stdint.h>
#include <hexagon_types.h>

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}

#endif // VTCM_UTILS_H
