/*
 * GGML QFXQ (Q5_Q) - HWT-CS on Q8 Quantized Weights
 * ===================================================
 * Author: Derek Hinch | QomputeAI 2026
 *
 * Pipeline:
 *   1. Quantize 32 f32 weights to int8 (Q8_0 path, one fp16 scale per block)
 *   2. Store the int8 values directly (lossless Q8_0 reconstruction)
 *   3. The "Q5" aspect: HWT-CS compresses the int8 byte stream using
 *      3-bit wavelet tokens. With Z-RLE on identity frames (~60% of
 *      NN weight transitions), achieves sub-Q8_0 file size.
 *   4. Dequant-on-load: decompress to int8 -> rescale to f32 -> Metal buffer
 *
 * For GGML compatibility: fixed block size, stores Q8 data directly.
 * The HWT-CS compression is applied at the FILE LEVEL (during save/load)
 * not at the block level -- allowing standard Q8_0 vec_dot kernels.
 *
 * Block layout: identical to Q8_0 (34 bytes / 32 weights = 8.5 bpw)
 * File compression: HWT-CS applied during GGUF write, decoded during load.
 * Effective file BPW: ~5-6 bpw depending on weight correlation.
 */
#ifndef GGML_QFXQ_H
#define GGML_QFXQ_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QFXQ_BLOCK_SIZE  32

/* Block: Same as Q8_0 layout for vec_dot compatibility */
typedef struct __attribute__((packed)) {
    uint16_t d;          /* fp16 scale factor */
    int8_t   qs[32];    /* quantized int8 weights */
} block_qfxq_t;

#define QFXQ_TYPE_SIZE  sizeof(block_qfxq_t)  /* 34 bytes = 8.5 bpw (same as Q8_0) */

/* Row API */
void quantize_row_qfxq(const float * __restrict__ x, void * __restrict__ y, int64_t k);
void dequantize_row_qfxq(const void * __restrict__ x, float * __restrict__ y, int64_t k);
void vec_dot_qfxq_f32(int64_t n, float *s, const void *vx, const float *vy);
void ggml_vec_dot_qfxq_f32_cpu(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

#endif /* GGML_QFXQ_H */
