#ifndef HTP_MATMUL_OPS_H
#define HTP_MATMUL_OPS_H

#include <stdint.h>
#include <stddef.h>
#include "htp-ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- Weight Repacked Tile Sizes ---
#define HTP_WEIGHT_TILE_SIZE_Q4_0   576
#define HTP_WEIGHT_TILE_SIZE_Q4_1   640
#define HTP_WEIGHT_TILE_SIZE_Q8_0   1088
#define HTP_WEIGHT_TILE_SIZE_IQ4_NL 576
#define HTP_WEIGHT_TILE_SIZE_MXFP4  544

// --- Weight Repacked Aligned Tile Sizes ---
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_Q4_0   640
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_Q4_1   640
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_Q8_0   1152
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_IQ4_NL 640
#define HTP_WEIGHT_ALIGNED_TILE_SIZE_MXFP4  640

// --- Activation Tiled Block Sizes (including padding) ---
#define HTP_ACT_TILE_SIZE_Q8_0      1152
#define HTP_ACT_TILE_SIZE_Q8_1      1280

// --- Tile Size Helpers ---
static inline uint32_t htp_get_weight_tile_size(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
            return HTP_WEIGHT_TILE_SIZE_Q4_0;
        case HTP_TYPE_Q4_1:
            return HTP_WEIGHT_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q8_0:
            return HTP_WEIGHT_TILE_SIZE_Q8_0;
        case HTP_TYPE_MXFP4:
            return HTP_WEIGHT_TILE_SIZE_MXFP4;
        default:
            return 0;
    }
}

static inline uint32_t htp_get_weight_aligned_tile_size(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
            return HTP_WEIGHT_ALIGNED_TILE_SIZE_Q4_0;
        case HTP_TYPE_Q4_1:
            return HTP_WEIGHT_ALIGNED_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q8_0:
            return HTP_WEIGHT_ALIGNED_TILE_SIZE_Q8_0;
        case HTP_TYPE_MXFP4:
            return HTP_WEIGHT_ALIGNED_TILE_SIZE_MXFP4;
        default:
            return 0;
    }
}

// --- Activation/Row Size Helpers ---
static inline size_t htp_q8_0_tiled_row_size(uint32_t ne) {
    const uint32_t nb_32 = (ne + 31) / 32;
    return nb_32 * HTP_ACT_TILE_SIZE_Q8_0;
}

static inline size_t htp_q8_1_tiled_row_size(uint32_t ne) {
    const uint32_t nb_32 = (ne + 31) / 32;
    return nb_32 * HTP_ACT_TILE_SIZE_Q8_1;
}

static inline size_t htp_get_tiled_row_stride(int weight_type, int k) {
    int nb = (k + QK_Q4_0_TILED - 1) / QK_Q4_0_TILED;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q8_0:
        case HTP_TYPE_MXFP4:
            return (size_t) nb * htp_get_weight_tile_size(weight_type);
        case HTP_TYPE_F16:
            return (size_t) k * sizeof(__fp16);
        case HTP_TYPE_F32:
            return (size_t) k * sizeof(float);
        default:
            return 0;
    }
}

#ifdef __cplusplus
}
#endif

#endif // HTP_MATMUL_OPS_H
