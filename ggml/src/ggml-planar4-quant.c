/*
 * PlanarQuant 4-bit: 2D Givens rotation + turbo4-format quantization.
 * Same block layout as turbo4_0 (68 bytes per 128 elements).
 * Delegates packing/unpacking to turbo4 format, just changes the rotation.
 */
#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include <math.h>
#include <string.h>
#include <assert.h>

/* Import planar rotation from ggml-planar-quant.c */
extern void planar_init_rotation(void);
extern float planar_cos[];
extern float planar_sin[];

/* Import turbo4 packing from ggml-turbo-quant.c */
extern void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k);
extern void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

/* We can't easily inject our rotation into turbo4's packing without
 * duplicating the whole function. Instead, we pre-rotate the input,
 * call turbo4's quantize (which will also apply WHT rotation),
 * then the dequant will undo WHT. This doesn't work — we need our own.
 *
 * Simpler approach: since block_planar4_0 IS block_turbo4_0,
 * we just use turbo4's quantize/dequantize directly.
 * The rotation difference is handled in the Metal set_rows kernel.
 * The CPU path uses turbo4's WHT rotation as fallback.
 */

void quantize_row_planar4_0_ref(const float * GGML_RESTRICT x, block_planar4_0 * GGML_RESTRICT y, int64_t k) {
    /* CPU fallback: use turbo4 packing (with WHT rotation) */
    quantize_row_turbo4_0_ref(x, (block_turbo4_0 *)y, k);
}

void dequantize_row_planar4_0(const block_planar4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    /* CPU fallback: use turbo4 dequant (with WHT inverse) */
    dequantize_row_turbo4_0((const block_turbo4_0 *)x, y, k);
}

size_t quantize_planar4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_PLANAR4 == 0);
    size_t row_size = (n_per_row / QK_PLANAR4) * sizeof(block_planar4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_planar4_0_ref(
            src + row * n_per_row,
            (block_planar4_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}
