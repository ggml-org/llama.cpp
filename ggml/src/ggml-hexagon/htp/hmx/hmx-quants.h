// HMX quantization super-block types for tile-permuted weight format.
// Ported from htp-ops-lib/include/dsp/quants.h.
//
// my_block_q4_0 / my_block_q8_0 renamed to hmx_block_q4_0 / hmx_block_q8_0.
// Duplicate ggml_type enum and standard block_q4_0 / block_q8_0 removed —
// those are provided by ggml-common.h which is always included before this
// header in DSP translation units.

#ifndef HMX_QUANTS_H
#define HMX_QUANTS_H

#include <stdint.h>

#define HMX_QK_K 256 // super-block size (8 × QK4_0 elements)

#ifndef QK4_0
#define QK4_0 32
#endif

#ifndef QK8_0
#define QK8_0 32
#endif

// Q4_0 super-block: 8 consecutive block_q4_0 along K dimension merged.
// Layout: scales first, then quants in sequential order (no cross-row
// interleaving, no XOR 0x88 sign conversion).
typedef struct {
    __fp16  scales[8];              // 8 block scales (16 B)
    uint8_t quants[8 * QK4_0 / 2]; // 8 × 16 = 128 B  4-bit quants
} __attribute__((packed)) hmx_block_q4_0; // 144 B / 256 elements

// Q8_0 super-block: 8 consecutive block_q8_0 along K dimension merged.
typedef struct {
    __fp16 scales[8];              // 8 block scales (16 B)
    int8_t quants[8 * QK8_0];     // 8 × 32 = 256 B  8-bit quants
} __attribute__((packed)) hmx_block_q8_0; // 272 B / 256 elements

// Weight type integer values (must match ggml_type enum from ggml-common.h).
// Used by HMX ops to dispatch on weight format without pulling in ggml-common.h.
#define HMX_TYPE_F16    1
#define HMX_TYPE_Q4_0   2
#define HMX_TYPE_Q8_0   8
#define HMX_TYPE_IQ4_NL 20

#endif // HMX_QUANTS_H
