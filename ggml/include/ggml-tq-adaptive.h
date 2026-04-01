#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TQ3_AP_MAGIC 0x33504154u

struct ggml_tq3_ap_extra {
    uint32_t magic;
    int64_t  blocks_per_row;
    int64_t  nrows;
    uint8_t * bitmap;
    ggml_fp16_t * means;
    uint32_t * row_offsets;
};

static inline int ggml_tq3_ap_is_promoted(const struct ggml_tq3_ap_extra * ap, int64_t block_id) {
    return (ap->bitmap[block_id >> 3] >> (block_id & 7)) & 1u;
}

#ifdef __cplusplus
}
#endif
