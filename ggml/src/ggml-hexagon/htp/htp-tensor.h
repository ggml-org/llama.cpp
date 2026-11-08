#ifndef HTP_TENSOR_H
#define HTP_TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "htp-ops.h"
#include "hex-bitmap.h"
#include "hex-common.h"
#include "hex-fastdiv.h"

#define HTP_TENSOR_MDEV_LINE_SIZE   128

// Currently HMX requires 32x32 tile formats (do not change)
#define HTP_TENSOR_TILE_N_COLS      32
#define HTP_TENSOR_TILE_N_ROWS      32

#define HTP_TENSOR_TILE_SIZE_F16    2048
#define HTP_TENSOR_TILE_SIZE_Q4_0   576
#define HTP_TENSOR_TILE_SIZE_Q4_1   640
#define HTP_TENSOR_TILE_SIZE_Q8_0   1088
#define HTP_TENSOR_TILE_SIZE_IQ4_NL 576
#define HTP_TENSOR_TILE_SIZE_MXFP4  544
#define HTP_TENSOR_TILE_SIZE_Q4_K   640
#define HTP_TENSOR_TILE_SIZE_Q5_K   768
#define HTP_TENSOR_TILE_SIZE_Q6_K   896
#define HTP_TENSOR_TILE_SIZE_Q3_K   512
#define HTP_TENSOR_TILE_SIZE_Q2_K   336
#define HTP_TENSOR_TILE_K_Q1_0      128
#define HTP_TENSOR_TILE_SIZE_Q1_0   576
#define HTP_TENSOR_TILE_K_Q2_0      64
#define HTP_TENSOR_TILE_SIZE_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

struct htp_tensor_mdev_range {
    uint32_t start;
    uint32_t count;
};

static inline bool htp_tensor_is_extended(const struct htp_tensor * t) {
    return t && (t->data >> 32) != 0;
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

static inline uint32_t htp_tensor_get_tile_size(int type) {
    switch (type) {
        case HTP_TYPE_Q4_0:   return HTP_TENSOR_TILE_SIZE_Q4_0;
        case HTP_TYPE_IQ4_NL: return HTP_TENSOR_TILE_SIZE_IQ4_NL;
        case HTP_TYPE_Q4_1:   return HTP_TENSOR_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q4_K:   return HTP_TENSOR_TILE_SIZE_Q4_K;
        case HTP_TYPE_Q8_0:   return HTP_TENSOR_TILE_SIZE_Q8_0;
        case HTP_TYPE_Q5_K:   return HTP_TENSOR_TILE_SIZE_Q5_K;
        case HTP_TYPE_Q6_K:   return HTP_TENSOR_TILE_SIZE_Q6_K;
        case HTP_TYPE_Q3_K:   return HTP_TENSOR_TILE_SIZE_Q3_K;
        case HTP_TYPE_Q2_K:   return HTP_TENSOR_TILE_SIZE_Q2_K;
        case HTP_TYPE_Q1_0:   return HTP_TENSOR_TILE_SIZE_Q1_0;
        case HTP_TYPE_Q2_0:   return HTP_TENSOR_TILE_SIZE_Q2_0;
        case HTP_TYPE_MXFP4:  return HTP_TENSOR_TILE_SIZE_MXFP4;
        case HTP_TYPE_F16:    return HTP_TENSOR_TILE_SIZE_F16;
        default:              return 0;
    }
}

static inline uint32_t htp_tensor_get_tile_k(int type) {
    switch (type) {
        case HTP_TYPE_Q1_0: return HTP_TENSOR_TILE_K_Q1_0;
        case HTP_TYPE_Q2_0: return HTP_TENSOR_TILE_K_Q2_0;
        default:            return 32;
    }
}

static inline uint32_t htp_tensor_get_tile_padded_size(int type) {
    return (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

static inline bool htp_tensor_is_tiled(const struct htp_tensor * t) {
    return t && (t->flags & HTP_TENSOR_TILED) != 0;
}

static inline uint32_t htp_tensor_get_row_size(int type, uint32_t ne00) {
    switch (type) {
        case HTP_TYPE_F32:    return ne00 * 4;
        case HTP_TYPE_F16:
        case HTP_TYPE_BF16:   return ne00 * 2;
        case HTP_TYPE_Q8_0:   return (ne00 / 32) * 34;
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL: return (ne00 / 32) * 18;
        case HTP_TYPE_Q4_1:   return (ne00 / 32) * 20;
        case HTP_TYPE_MXFP4:  return (ne00 / 32) * 17;
        case HTP_TYPE_Q2_K:   return (ne00 / 256) * 84;
        case HTP_TYPE_Q2_0:   return (ne00 / 64) * 18;
        case HTP_TYPE_Q3_K:   return (ne00 / 256) * 110;
        case HTP_TYPE_Q4_K:   return (ne00 / 256) * 144;
        case HTP_TYPE_Q5_K:   return (ne00 / 256) * 176;
        case HTP_TYPE_Q6_K:   return (ne00 / 256) * 210;
        case HTP_TYPE_Q1_0:   return (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        case HTP_TYPE_I32:    return ne00 * 4;
        case HTP_TYPE_I64:    return ne00 * 8;
        default:              return 0;
    }
}

#endif // HTP_TENSOR_H
