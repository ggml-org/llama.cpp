/*
 * PlanarQuant 4-bit: 2D Givens rotation + 4-bit (16 centroids) nibble packed.
 * Same block layout as turbo4_0 but uses Givens rotation instead of WHT.
 */
#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include <math.h>
#include <string.h>
#include <assert.h>

#define PLANAR4_D 128
#define PLANAR4_SEED 42

static const float PLANAR4_CENTROIDS[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

static float p4_cos[64], p4_sin[64];
static int p4_init = 0;

static void planar4_init(void) {
    if (p4_init) return;
    uint64_t s = PLANAR4_SEED;
    for (int i = 0; i < 64; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        double u = (double)(s >> 11) / (double)(1ULL << 53);
        double angle = u * 2.0 * 3.14159265358979323846;
        p4_cos[i] = (float)__builtin_cos(angle);
        p4_sin[i] = (float)__builtin_sin(angle);
    }
    p4_init = 1;
}

static int nearest_4bit(float val) {
    int best = 0;
    float best_d = fabsf(val - PLANAR4_CENTROIDS[0]);
    for (int i = 1; i < 16; i++) {
        float d = fabsf(val - PLANAR4_CENTROIDS[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

void quantize_row_planar4_0_ref(const float * GGML_RESTRICT x, block_planar4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    planar4_init();
    const int nb = k / 128;

    for (int b = 0; b < nb; b++) {
        const float * src = x + b * 128;
        block_planar4_0 * blk = &y[b];

        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        memset(blk->qs, 0, 64);

        float recon_sq = 0.0f;
        for (int p = 0; p < 64; p++) {
            float v0 = src[p*2] * inv;
            float v1 = src[p*2+1] * inv;
            float r0 = p4_cos[p]*v0 - p4_sin[p]*v1;
            float r1 = p4_sin[p]*v0 + p4_cos[p]*v1;

            int i0 = nearest_4bit(r0);
            int i1 = nearest_4bit(r1);

            int j0 = p*2, j1 = p*2+1;
            blk->qs[j0/2] |= (i0 & 0xF) << ((j0%2)*4);
            blk->qs[j1/2] |= (i1 & 0xF) << ((j1%2)*4);

            recon_sq += PLANAR4_CENTROIDS[i0]*PLANAR4_CENTROIDS[i0];
            recon_sq += PLANAR4_CENTROIDS[i1]*PLANAR4_CENTROIDS[i1];
        }

        float rn = sqrtf(recon_sq);
        float corrected = (rn > 1e-10f) ? grp_norm / rn : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);
        blk->rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_planar4_0(const block_planar4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    planar4_init();
    const int nb = k / 128;

    for (int b = 0; b < nb; b++) {
        float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int p = 0; p < 64; p++) {
            int j0 = p*2, j1 = p*2+1;
            uint8_t i0 = (x[b].qs[j0/2] >> ((j0%2)*4)) & 0xF;
            uint8_t i1 = (x[b].qs[j1/2] >> ((j1%2)*4)) & 0xF;
            float q0 = PLANAR4_CENTROIDS[i0];
            float q1 = PLANAR4_CENTROIDS[i1];
            float f0 =  p4_cos[p]*q0 + p4_sin[p]*q1;
            float f1 = -p4_sin[p]*q0 + p4_cos[p]*q1;
            y[b*128 + j0] = f0 * norm;
            y[b*128 + j1] = f1 * norm;
        }
    }
}

size_t quantize_planar4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % 128 == 0);
    size_t row_size = (n_per_row / 128) * sizeof(block_planar4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_planar4_0_ref(
            src + row * n_per_row,
            (block_planar4_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}
