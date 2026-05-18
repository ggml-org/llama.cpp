/*
 * Minimal TurboQuant KV cache reference implementation.
 *
 * This ports the CPU/reference storage format for TURBO3_0 and TURBO4_0 so the
 * runtime can parse and allocate these KV cache types on non-Metal builds.
 * Fast-path backend support can be layered on top later.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#define TURBO_QJL_CONST 1.2533141373155003f

static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int i = 0; i < nb; ++i) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO3; ++j) {
            norm += x[i * QK_TURBO3 + j] * x[i * QK_TURBO3 + j];
        }
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, sizeof(y[i].qs));
        memset(y[i].signs, 0, sizeof(y[i].signs));
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; ++block) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; ++j) {
            const uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            const uint8_t hi1  = (x[block].signs[j/8] >> (j%8)) & 0x1;
            const uint8_t idx  = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);
    const size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *) ((char *) dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    for (int block = 0; block < nb; ++block) {
        const float * src = x + block * QK_TURBO4;
        float norm_sq = 0.0f;
        float abs_sum = 0.0f;
        for (int i = 0; i < QK_TURBO4; ++i) {
            norm_sq += src[i] * src[i];
            abs_sum += fabsf(src[i]);
        }
        y[block].norm = GGML_FP32_TO_FP16(sqrtf(norm_sq));
        y[block].rnorm = GGML_FP32_TO_FP16(abs_sum / (float) QK_TURBO4);
        memset(y[block].qs, 0, sizeof(y[block].qs));
        memset(y[block].signs, 0, sizeof(y[block].signs));
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    for (int block = 0; block < nb; ++block) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        const float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float) QK_TURBO4 * rnorm;

        for (int i = 0; i < QK_TURBO4; ++i) {
            const int bit_offset = i * 3;
            const int byte_idx = bit_offset / 8;
            const int bit_pos = bit_offset % 8;
            uint16_t raw = (uint16_t) x[block].qs[byte_idx];
            if (byte_idx + 1 < (int) sizeof(x[block].qs)) {
                raw |= (uint16_t) x[block].qs[byte_idx + 1] << 8;
            }
            const uint8_t idx = (raw >> bit_pos) & 0x7;
            const float sign = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
            y[block * QK_TURBO4 + i] = (CENTROIDS_3BIT[idx] + sign * qjl_scale) * norm;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);
    const size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *) ((char *) dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}
