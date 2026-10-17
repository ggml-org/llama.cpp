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
#define HTP_TENSOR_TILE_SIZE_Q2_0   576

// Q2_K, Q3_K, Q6_K use vrmpy byte order within each 128B HVX vector:
//   k = 0..31 is split into 8 groups g = k/4 of 4 k each; byte 4*r + (k%4) holds row r
//
// Q2_K (336B, lossless tiling with same size as block_q2_K):
//   bytes   0..255 (vec 0..1): unsigned 2-bit quants, vec m holds groups 4m..4m+3 at bit offsets 0,2,4,6
//   bytes 256..319:            raw scales[] bytes (min << 4 | sc): [0..31] for k 0..15, [32..63] for k 16..31
//   bytes 320..335:            8 fp16 in tile j (0..7) of super-block: d (j < 4) or -dmin (j >= 4) for rows 8*(j%4)..8*(j%4)+7
//
// Q3_K (512B):
//   bytes   0..255 (vec 0..1): low 2 bits, vec m holds groups 4m..4m+3 at bit offsets 0,2,4,6
//   bytes 256..383 (vec 2):    sign mask, bit g set when hmask is 0 (quant = low2 - 4)
//   bytes 384..511 (vec 3):    fp16 scales d * (scales[] - 32): fp16[0..31] for k 0..15, fp16[32..63] for k 16..31
//
// Q4_K (640B, same layout as Q4_1 tile; each 256-k super-block splits into 8 tiles of 32 k):
//   bytes   0..511 (vec 0..3): 4-bit quants, byte 32*cp + r holds k = 2*cp (low nibble) and 2*cp + 1 (high nibble)
//   bytes 512..639 (vec 4):    32x2 fp16 per row: (d * sc, -dmin * m)
//
// Q5_K (768B):
//   bytes   0..639 (vec 0..4): Q4_K tile (low 4 bits + fp16 scale/min pairs)
//   bytes 640..767 (vec 5):    5th-bit plane; byte l holds the 8 high bits for lane l across vec i = 0..3 (bit 2i = lo, 2i+1 = hi)
//
// Q6_K (896B):
//   bytes   0..511 (vec 0..3): low 4 bits, vec i holds group 2i (low nibble) and 2i+1 (high nibble)
//   bytes 512..767 (vec 4..5): high 2 bits, vec 4+m holds groups 4m..4m+3 at bit offsets 0,2,4,6
//   bytes 768..895 (vec 6):    fp16 scales d * scales[]: fp16[0..31] for k 0..15, fp16[32..63] for k 16..31

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
        case HTP_TYPE_Q2_0:   return HTP_TENSOR_TILE_SIZE_Q2_0;
        case HTP_TYPE_MXFP4:  return HTP_TENSOR_TILE_SIZE_MXFP4;
        case HTP_TYPE_F16:    return HTP_TENSOR_TILE_SIZE_F16;
        default:              return 0;
    }
}

static inline uint32_t htp_tensor_get_tile_padded_size(int type) {
    return (htp_tensor_get_tile_size(type) + 127u) & ~127u;
}

static inline uint32_t htp_tensor_get_tile_k(int type) {
    return type == HTP_TYPE_Q2_0 ? 64 : 32;
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
        case HTP_TYPE_I32:    return ne00 * 4;
        case HTP_TYPE_I64:    return ne00 * 8;
        default:              return 0;
    }
}

#endif // HTP_TENSOR_H
