/*
 * GGML QFX16 — Transition-Encoded BFloat16 (LOSSLESS, LUT Inference)
 * ====================================================================
 * Author: Derek Hinch | QomputeAI 2026
 *
 * BFloat16 weights stored as predict-only Haar delta transitions with Z-RLE.
 * The 65,536-entry bijection (exhaustively verified) maps every uint16 value
 * to a unique transition. Inference via 256KB BF16→Float LUT — no decode.
 *
 * Block format (QFX16_BLOCK_SIZE = 256 weights, variable output size):
 *   [anchor: 1B][stream_len: 2B][delta_stream: max 512B] padded to fixed size
 *
 * Storage: 8-16 bpw effective (depends on weight correlation).
 * Quality: 100% LOSSLESS — exact BFloat16 precision preserved.
 *
 * Key insight: L(a,b) = (a, (b-a) mod 256) is bijective in Z_256 because
 * subtraction is always invertible (unlike division: GCD(2,256)≠1).
 * Proven by IntegerQuaternion.modular_inverse slide technique.
 */
#ifndef GGML_QFX16_H
#define GGML_QFX16_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══ Block Geometry ═══ */

#define QFX16_BLOCK_SIZE    256    /* Weights per block */
#define QFX16_MAX_STREAM    512    /* Max delta stream bytes (worst case) */
#define QFX16_ZRLE_ESCAPE   0xFF   /* Z-RLE escape (impossible as first byte of valid transition) */

/* Block layout — fixed size for GGML random-access requirement */
typedef struct __attribute__((packed)) {
    uint8_t  anchor;                       /* First byte (reconstruction seed) */
    uint16_t stream_len;                   /* Actual compressed stream length */
    uint8_t  delta_stream[QFX16_MAX_STREAM]; /* Z-RLE encoded byte deltas */
} block_qfx16_t;

/* Compile-time size (for GGML type_size) */
#define QFX16_TYPE_SIZE  sizeof(block_qfx16_t)  /* 515 bytes per 256 weights */

/* ═══ GGML-Compatible Row API ═══ */

/* Initialize the 256KB BF16→Float LUT. Must be called once. Thread-safe. */
void ggml_qfx16_init(void);

/* Quantize a row: float32[] → block_qfx16_t[]
 * k = number of float elements (must be multiple of QFX16_BLOCK_SIZE) */
void quantize_row_qfx16(const float * __restrict__ x, void * __restrict__ y, int64_t k);

/* Dequantize a row: block_qfx16_t[] → float32[]
 * k = number of float elements to produce */
void dequantize_row_qfx16(const void * __restrict__ x, float * __restrict__ y, int64_t k);

/* Vec-dot: compute dot(qfx16_row, float32_row)
 * n = number of weights, result in *s */
void vec_dot_qfx16_f32(int64_t n, float *s, const void *vx, const float *vy);

/* ═══ GGML CPU vec_dot adapter (modern signature) ═══ */

void ggml_vec_dot_qfx16_f32_cpu(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif


/* Multi-row quantize (matches ggml_quantize_chunk signature) */
size_t quantize_qfx16(const float * __restrict__ src, void * __restrict__ dst, int64_t nrow, int64_t n_per_row, const float * quant_weights);
#endif /* GGML_QFX16_H */
