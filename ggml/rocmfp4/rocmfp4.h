#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QK_ROCMFP4 32
#define QR_ROCMFP4 2
#define QI_ROCMFP4 (QK_ROCMFP4 / (4 * QR_ROCMFP4))
#define QS_ROCMFP4 32

// AMD-tuned compact layout: 16 bytes of packed E2M1-derived 4-bit codes, then
// one unsigned E4M3 scale byte per 16-weight half block.
typedef struct {
    uint8_t qs[QK_ROCMFP4/2];
    uint8_t e[2];
} block_rocmfp4;

// Speed-focused layout: same 32 packed ROCmFP4 nibbles, but one UE4M3 scale
// for the whole block. This is a separate GGUF type so fast 4.25 BPW artifacts
// never alias the safer dual-scale format above.
typedef struct {
    uint8_t qs[QK_ROCMFP4/2];
    uint8_t e;
} block_rocmfp4_fast;

#if defined(__cplusplus)
static_assert(sizeof(block_rocmfp4) == QK_ROCMFP4/2 + 2*sizeof(uint8_t), "wrong rocmfp4 block size/padding");
static_assert(sizeof(block_rocmfp4_fast) == QK_ROCMFP4/2 + sizeof(uint8_t), "wrong rocmfp4 fast block size/padding");
#else
_Static_assert(sizeof(block_rocmfp4) == QK_ROCMFP4/2 + 2*sizeof(uint8_t), "wrong rocmfp4 block size/padding");
_Static_assert(sizeof(block_rocmfp4_fast) == QK_ROCMFP4/2 + sizeof(uint8_t), "wrong rocmfp4 fast block size/padding");
#endif

GGML_API void rocmfp4_quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_rocmfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void rocmfp4_dequantize_row_q4_0(const block_rocmfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void rocmfp4_quantize_row_q4_0_fast_ref(const float * GGML_RESTRICT x, block_rocmfp4_fast * GGML_RESTRICT y, int64_t k);
GGML_API void rocmfp4_dequantize_row_q4_0_fast(const block_rocmfp4_fast * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void   rocmfp4_quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfp4_quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   rocmfp4_quantize_row_q4_0_fast(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfp4_quantize_q4_0_fast(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API bool   rocmfp4_validate_row_data(const void * data, size_t nbytes);
GGML_API bool   rocmfp4_validate_row_data_fast(const void * data, size_t nbytes);

GGML_API void rocmfp4_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void rocmfp4_vec_dot_q4_0_fast_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
