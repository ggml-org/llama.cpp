/*
 * ggml-turboquant.h — TurboQuant type definitions for GGML
 *
 * Author: Keyvan Hardani (https://github.com/Keyvanhardani)
 * Drop this into ggml/include/ and include from ggml-quants.h
 *
 * Based on Google's TurboQuant (ICLR 2026, arXiv:2504.19874)
 * with community optimizations: WHT rotation, norm correction,
 * block-32 storage, MSE-only (no QJL).
 */

#ifndef GGML_TURBOQUANT_H
#define GGML_TURBOQUANT_H

#include <stdint.h>
#include <stddef.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Block size for all TurboQuant types */
#define QK_TQ 32

/*
 * turbo4_0: 4-bit TurboQuant (best quality)
 * 18 bytes per 32 elements = 4.5 bits/elem, 3.6x compression
 * PPL: +0.77% vs q8_0 (Madreag benchmark, RTX 5090)
 */
#define QK_TURBO4_0 QK_TQ
typedef struct {
    uint16_t d;        /* fp16 corrected norm (original_norm / ||reconstruction||) */
    uint8_t  qs[16];   /* 32 x 4-bit packed Lloyd-Max indices */
} block_turbo4_0;
_Static_assert(sizeof(block_turbo4_0) == 18, "block_turbo4_0 size");

/*
 * turbo3_0: 3-bit TurboQuant (best balance)
 * 14 bytes per 32 elements = 3.5 bits/elem, 4.6x compression
 * PPL: +1.1-2.8% vs q8_0, decode speed matches q8_0 at 32K
 */
#define QK_TURBO3_0 QK_TQ
typedef struct {
    uint16_t d;        /* fp16 corrected norm */
    uint8_t  qs[12];   /* 32 x 3-bit packed Lloyd-Max indices */
} block_turbo3_0;
_Static_assert(sizeof(block_turbo3_0) == 14, "block_turbo3_0 size");

/*
 * turbo2_0: 2-bit TurboQuant (max compression)
 * 10 bytes per 32 elements = 2.5 bits/elem, 6.4x compression
 * Beats q8_0 by 5.4% at 32K on RTX 5090 (Madreag benchmark)
 */
#define QK_TURBO2_0 QK_TQ
typedef struct {
    uint16_t d;        /* fp16 corrected norm */
    uint8_t  qs[8];    /* 32 x 2-bit packed Lloyd-Max indices */
} block_turbo2_0;
_Static_assert(sizeof(block_turbo2_0) == 10, "block_turbo2_0 size");

/* ─── Quantize/Dequantize function declarations ──────────────────────── */

void dequantize_row_turbo4_0(const block_turbo4_0 * x, float * y, int64_t k);
void dequantize_row_turbo3_0(const block_turbo3_0 * x, float * y, int64_t k);
void dequantize_row_turbo2_0(const block_turbo2_0 * x, float * y, int64_t k);

void quantize_row_turbo4_0_ref(const float * x, block_turbo4_0 * y, int64_t k);
void quantize_row_turbo3_0_ref(const float * x, block_turbo3_0 * y, int64_t k);
void quantize_row_turbo2_0_ref(const float * x, block_turbo2_0 * y, int64_t k);

size_t quantize_turbo4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_turbo3_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_turbo2_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

/* vec_dot for Flash Attention (K dot Q on quantized K blocks) */
void ggml_vec_dot_turbo4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_turbo3_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_turbo2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TURBOQUANT_H */
