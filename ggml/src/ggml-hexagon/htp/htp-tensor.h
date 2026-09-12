#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "htp-ops.h"
#include "hex-bitmap.h"
#include "hex-common.h"
#include "hex-fastdiv.h"

enum {
    HTP_TENSOR_MDEV_LINE_SIZE = 128,
};

struct htp_tensor_mdev_range {
    uint32_t start;
    uint32_t count;
};

static inline void * htp_tensor_data(const struct htp_tensor * t) {
    return (void *) (uintptr_t) t->data;
}

static inline uint32_t * htp_tensor_flags(const struct htp_tensor * t) {
    return (uint32_t *) &t->flags;
}

static inline bool htp_tensor_is_contiguous(const struct htp_tensor * t, uint32_t type_size) {
    uint32_t next_nb = type_size;
    if (t->ne[0] != 1 && t->nb[0] != next_nb) {
        return false;
    }
    next_nb *= t->ne[0];
    for (int i = 1; i < HTP_OP_MAX_DIMS; i++) {
        if (t->ne[i] != 1 && t->nb[i] != next_nb) {
            return false;
        }
        next_nb *= t->ne[i];
    }
    return true;
}

static inline bool htp_tensor_is_permuted(const struct htp_tensor * t) {
    return t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

static inline bool htp_tensor_mdev_data_aligned(const struct htp_tensor * t) {
    return ((uintptr_t) t->data & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0;
}

static inline bool htp_tensor_can_row_partition(const struct htp_tensor * t, uint32_t elem_size) {
    if (!htp_tensor_mdev_data_aligned(t)) {
        return false;
    }
    if (t->ne[0] != 1 && t->nb[0] != elem_size) {
        return false;
    }
    if (htp_tensor_is_permuted(t)) {
        return false;
    }
    if (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) return false;
    if (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) return false;
    if (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) != 0) return false;
    return true;
}

static inline bool htp_tensor_mdev_rows_per_chunk(const struct htp_tensor * t, uint32_t elem_size, uint32_t row_size, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    if (!htp_tensor_mdev_data_aligned(t)) {
        return false;
    }
    if (t->ne[0] != 1 && t->nb[0] != elem_size) {
        return false;
    }
    if (htp_tensor_is_permuted(t)) {
        return false;
    }
    if (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0)) {
        *rows_per_chunk = 1;
        return true;
    }
    if (t->nb[1] == row_size &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_size > 0) ? (HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(row_size, HTP_TENSOR_MDEV_LINE_SIZE)) : 1;
        return true;
    }
    return false;
}

static inline struct htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t total_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_count, const struct fastdiv_values * mdev_count_div) {
    struct htp_tensor_mdev_range range = { 0, total_units };

    if (mdev_count <= 1) {
        return range;
    }

    if (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        return range;
    }

    const uint32_t total_chunks = total_units / units_per_chunk;
    if (total_chunks < mdev_count) {
        range.start = (mdev_idx == 0) ? 0 : total_units;
        range.count = (mdev_idx == 0) ? total_units : 0;
        return range;
    }

    const uint32_t chunks_per_mdev = fastdiv(total_chunks + mdev_count - 1, mdev_count_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, total_units);
    if (mdev_idx == mdev_count - 1) {
        range.count = total_units - range.start;
    } else {
        range.count = MIN(chunks_per_mdev * units_per_chunk, total_units - range.start);
    }
    return range;
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
void htp_flush_dirty_ranges(struct htp_context * ctx);
void htp_tensor_flush_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);
void htp_tensor_dirty_all(struct htp_context * ctx, const struct htp_tensor * const * tensors, uint32_t n);

#endif // HTP_TENSOR_H
