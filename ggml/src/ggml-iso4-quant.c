/*
 * IsoQuant 4-bit: quaternion 4D rotation + turbo4-format quantization.
 * Same block layout as turbo4_0. CPU path delegates to turbo4.
 * Metal set_rows kernel uses quaternion rotation instead of WHT.
 */
#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include <math.h>
#include <string.h>
#include <assert.h>

extern void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k);
extern void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_iso4_0_ref(const float * GGML_RESTRICT x, block_iso4_0 * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4_0_ref(x, (block_turbo4_0 *)y, k);
}

void dequantize_row_iso4_0(const block_iso4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    dequantize_row_turbo4_0((const block_turbo4_0 *)x, y, k);
}

size_t quantize_iso4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_ISO4 == 0);
    size_t row_size = (n_per_row / QK_ISO4) * sizeof(block_iso4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_iso4_0_ref(
            src + row * n_per_row,
            (block_iso4_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}
