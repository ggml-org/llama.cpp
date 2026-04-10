// TurboQuant 4-bit PolarQuant (ICLR 2026, arXiv 2504.19874)
// Implements GGML_TYPE_TURBO4_0: 4-bit KV cache compression using
// PolarQuant with Walsh-Hadamard Transform rotation.
//
// Quantize: extract L2 norm -> normalize -> WHT rotation -> 4-bit centroid -> pack
// Dequantize: unpack -> centroid lookup -> scale by norm -> inverse WHT
//
// Dequant includes inverse WHT for CPU correctness without graph modifications.

#include "ggml-impl.h"
#include "ggml-quants.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <math.h>
#include <string.h>
#include <assert.h>

// 16 optimal Lloyd-Max centroids for unit-norm Gaussian (PolarQuant paper Table 1)
static const float CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f,
};

// Walsh-Hadamard Transform sign arrays (length 128)
// Pseudo-random +/-1 sequences for randomized WHT rotation
static const float turbo_cpu_s1[128] = {
     1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1,-1,-1,
     1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1,-1,
     1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1,
     1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,
    -1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,
    -1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1,
    -1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,
     1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1,
};

static const float turbo_cpu_s2[128] = {
    -1,-1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1, 1,-1,
     1, 1,-1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,
     1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,
     1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1,
    -1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1,-1,
     1, 1, 1,-1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,
    -1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1, 1,
};

// 1/sqrt(128) for normalization after Hadamard butterfly
#define INV_SQRT_128 0.08838834764831845f

// Forward WHT: y = (1/sqrt(N)) * S2 * H * S1 * x
static void turbo_fwht_forward(float * x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] *= turbo_cpu_s1[i];
    }
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        x[i] *= INV_SQRT_128 * turbo_cpu_s2[i];
    }
}

// Inverse WHT: x = (1/sqrt(N)) * S1 * H * S2 * y
// WHT is self-inverse up to scaling; inverse just swaps s1 and s2
static void turbo_fwht_inverse(float * x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] *= turbo_cpu_s2[i];
    }
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        x[i] *= INV_SQRT_128 * turbo_cpu_s1[i];
    }
}

// Binary tree search for nearest centroid (4 comparisons for 16 centroids)
// Uses midpoints between consecutive centroids for optimal assignment
static inline uint8_t nearest_centroid_4bit(float x) {
    if (x < 0.0f) {
        if (x < -0.060009f) {
            if (x < -0.103361f) {
                return (x < -0.145561f) ? 0 : 1;
            } else {
                return (x < -0.079142f) ? 2 : 3;
            }
        } else {
            if (x < -0.028293f) {
                return (x < -0.043430f) ? 4 : 5;
            } else {
                return (x < -0.013964f) ? 6 : 7;
            }
        }
    } else {
        if (x < 0.060009f) {
            if (x < 0.028293f) {
                return (x < 0.013964f) ? 8 : 9;
            } else {
                return (x < 0.043430f) ? 10 : 11;
            }
        } else {
            if (x < 0.103361f) {
                return (x < 0.079142f) ? 12 : 13;
            } else {
                return (x < 0.145561f) ? 14 : 15;
            }
        }
    }
}

// Quantize a single row of floats to turbo4_0 blocks
// Each block covers QK_TURBO4 (128) values = one rotation group
void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = (int)(k / QK_TURBO4);

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_TURBO4;
        float tmp[QK_TURBO4];

        // Step 1: compute L2 norm
        float sum_sq = 0.0f;
        for (int i = 0; i < QK_TURBO4; i++) {
            sum_sq += src[i] * src[i];
        }
        float norm = sqrtf(sum_sq);

        if (norm < 1e-10f) {
            y[block].norm  = GGML_FP32_TO_FP16(0.0f);
            y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
            memset(y[block].qs, 0, QK_TURBO4 / 2);
            continue;
        }

        // Step 2: normalize to unit vector
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < QK_TURBO4; i++) {
            tmp[i] = src[i] * inv_norm;
        }

        // Step 3: forward WHT rotation
        turbo_fwht_forward(tmp, QK_TURBO4);

        // Step 4: 4-bit centroid quantization
        float recon_sq = 0.0f;
        uint8_t indices[QK_TURBO4];
        for (int i = 0; i < QK_TURBO4; i++) {
            uint8_t idx = nearest_centroid_4bit(tmp[i]);
            indices[i] = idx;
            recon_sq += CENTROIDS_4BIT[idx] * CENTROIDS_4BIT[idx];
        }

        // Nibble pack: low nibble = even index, high nibble = odd index
        for (int i = 0; i < QK_TURBO4 / 2; i++) {
            y[block].qs[i] = indices[2 * i] | (indices[2 * i + 1] << 4);
        }

        // Step 5: norm correction
        // corrected_norm = original_norm / reconstruction_norm
        float recon_norm = sqrtf(recon_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? (norm / recon_norm) : norm;
        y[block].norm  = GGML_FP32_TO_FP16(corrected_norm);
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

// Dequantize turbo4_0 blocks back to float
// Performs inverse WHT for CPU correctness (no graph modifications needed)
void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = (int)(k / QK_TURBO4);

    for (int block = 0; block < nb; block++) {
        float corrected_norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * QK_TURBO4;

        // Step 1: unpack nibbles and look up centroids
        for (int i = 0; i < QK_TURBO4 / 2; i++) {
            uint8_t packed = x[block].qs[i];
            dst[2 * i]     = CENTROIDS_4BIT[packed & 0x0F];
            dst[2 * i + 1] = CENTROIDS_4BIT[(packed >> 4) & 0x0F];
        }

        // Step 2: inverse WHT to recover original domain
        turbo_fwht_inverse(dst, QK_TURBO4);

        // Step 3: scale by corrected norm
        for (int i = 0; i < QK_TURBO4; i++) {
            dst[i] *= corrected_norm;
        }
    }
}

// CPU backend from_float wrapper: (const float *, void *, int64_t) signature
// Used by type_traits_cpu for set_rows and other CPU ops
void quantize_row_turbo4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_turbo4_0_ref(x, (block_turbo4_0 *)vy, k);
}

// Multi-row quantize wrapper (used by ggml_quantize_chunk)
size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row,
                         const float * imatrix) {
    (void)imatrix; // importance matrix not used for KV cache quantization
    assert(n_per_row % QK_TURBO4 == 0);
    const size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
