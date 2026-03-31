/*
 * PlanarQuant: KV cache compression via 2D Givens rotation + Lloyd-Max
 * Based on: ParaMind2025/isoquant (planar2_fused_kernel.cu)
 *
 * Instead of TurboQuant's dense d×d WHT rotation, uses independent
 * 2D Givens rotations per pair: only 4 FMAs per pair vs O(d log d) for WHT.
 * Same block layout as turbo3 (2-bit indices + 1-bit signs + norm).
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

#define PLANAR_D 128
#define PLANAR_SEED 42

/* Same centroids as turbo3 (Lloyd-Max for N(0, 1/128)) */
static const float PLANAR_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* Rotation parameters: cos/sin per pair (lazy init) */
static float planar_cos[PLANAR_D / 2];
static float planar_sin[PLANAR_D / 2];
static int planar_rotation_initialized = 0;

static uint64_t planar_prng_state;

static void planar_prng_seed(uint64_t seed) {
    planar_prng_state = seed;
}

static double planar_prng_uniform(void) {
    planar_prng_state = planar_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(planar_prng_state >> 11) / (double)(1ULL << 53);
}

static void planar_init_rotation(void) {
    if (planar_rotation_initialized) return;
    planar_prng_seed(PLANAR_SEED);
    for (int i = 0; i < PLANAR_D / 2; i++) {
        double angle = planar_prng_uniform() * 2.0 * M_PI;
        planar_cos[i] = (float)cos(angle);
        planar_sin[i] = (float)sin(angle);
    }
    planar_rotation_initialized = 1;
}

static int nearest_centroid_planar3(float val) {
    int best = 0;
    float best_d = fabsf(val - PLANAR_CENTROIDS_3BIT[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(val - PLANAR_CENTROIDS_3BIT[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

void quantize_row_planar3_0_ref(const float * GGML_RESTRICT x, block_planar3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PLANAR3 == 0);
    planar_init_rotation();

    const int nb = k / QK_PLANAR3;
    const int n_pairs = QK_PLANAR3 / 2;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_PLANAR3;
        block_planar3_0 * blk = &y[block];

        /* 1. L2 norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_PLANAR3; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize + rotate + quantize */
        memset(blk->qs, 0, QK_PLANAR3 / 4);
        memset(blk->signs, 0, QK_PLANAR3 / 8);

        float recon_sq = 0.0f;
        for (int p = 0; p < n_pairs; p++) {
            float v0 = src[p * 2] * inv_norm;
            float v1 = src[p * 2 + 1] * inv_norm;

            /* Forward Givens rotation */
            float c = planar_cos[p];
            float s = planar_sin[p];
            float r0 = c * v0 - s * v1;
            float r1 = s * v0 + c * v1;

            /* Quantize both */
            int idx0 = nearest_centroid_planar3(r0);
            int idx1 = nearest_centroid_planar3(r1);

            int j0 = p * 2;
            int j1 = p * 2 + 1;

            /* Pack 2-bit lower + 1-bit sign (same as turbo3) */
            blk->qs[j0 / 4] |= (idx0 & 0x3) << ((j0 % 4) * 2);
            if (idx0 & 0x4) blk->signs[j0 / 8] |= (1 << (j0 % 8));

            blk->qs[j1 / 4] |= (idx1 & 0x3) << ((j1 % 4) * 2);
            if (idx1 & 0x4) blk->signs[j1 / 8] |= (1 << (j1 % 8));

            recon_sq += PLANAR_CENTROIDS_3BIT[idx0] * PLANAR_CENTROIDS_3BIT[idx0];
            recon_sq += PLANAR_CENTROIDS_3BIT[idx1] * PLANAR_CENTROIDS_3BIT[idx1];
        }

        /* 3. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);
    }
}

void dequantize_row_planar3_0(const block_planar3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PLANAR3 == 0);
    planar_init_rotation();

    const int nb = k / QK_PLANAR3;
    const int n_pairs = QK_PLANAR3 / 2;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        for (int p = 0; p < n_pairs; p++) {
            int j0 = p * 2;
            int j1 = p * 2 + 1;

            /* Unpack indices */
            uint8_t low0 = (x[block].qs[j0 / 4] >> ((j0 % 4) * 2)) & 0x3;
            uint8_t hi0 = (x[block].signs[j0 / 8] >> (j0 % 8)) & 0x1;
            uint8_t idx0 = low0 | (hi0 << 2);

            uint8_t low1 = (x[block].qs[j1 / 4] >> ((j1 % 4) * 2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j1 / 8] >> (j1 % 8)) & 0x1;
            uint8_t idx1 = low1 | (hi1 << 2);

            float q0 = PLANAR_CENTROIDS_3BIT[idx0];
            float q1 = PLANAR_CENTROIDS_3BIT[idx1];

            /* Inverse Givens rotation */
            float c = planar_cos[p];
            float s = planar_sin[p];
            float f0 = c * q0 + s * q1;
            float f1 = -s * q0 + c * q1;

            y[block * QK_PLANAR3 + j0] = f0 * norm;
            y[block * QK_PLANAR3 + j1] = f1 * norm;
        }
    }
}

size_t quantize_planar3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_PLANAR3 == 0);

    size_t row_size = (n_per_row / QK_PLANAR3) * sizeof(block_planar3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_planar3_0_ref(
            src + row * n_per_row,
            (block_planar3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
