/*
 * IsoQuant 4-bit: quaternion 4D rotation + 4-bit (16 centroids) nibble packed.
 * Same block layout as turbo4_0 but uses quaternion rotation instead of WHT.
 */
#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include <math.h>
#include <string.h>
#include <assert.h>

#define ISO4_D 128
#define ISO4_SEED 42
#define ISO4_N_GROUPS 32

static const float ISO4_CENTROIDS[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

static float i4_qw[32], i4_qx[32], i4_qy[32], i4_qz[32];
static int i4_init = 0;

static void iso4_init(void) {
    if (i4_init) return;
    uint64_t s = ISO4_SEED;
    for (int i = 0; i < 32; i++) {
        float q[4]; float norm = 0;
        for (int j = 0; j < 4; j++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            double u1 = (double)(s >> 11) / (double)(1ULL << 53);
            if (u1 < 1e-15) u1 = 1e-15;
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            double u2 = (double)(s >> 11) / (double)(1ULL << 53);
            q[j] = (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
            norm += q[j] * q[j];
        }
        norm = sqrtf(norm);
        i4_qw[i] = q[0]/norm; i4_qx[i] = q[1]/norm;
        i4_qy[i] = q[2]/norm; i4_qz[i] = q[3]/norm;
    }
    i4_init = 1;
}

static int nearest_16(float val) {
    int best = 0;
    float best_d = fabsf(val - ISO4_CENTROIDS[0]);
    for (int i = 1; i < 16; i++) {
        float d = fabsf(val - ISO4_CENTROIDS[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

void quantize_row_iso4_0_ref(const float * GGML_RESTRICT x, block_iso4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    iso4_init();
    const int nb = k / 128;

    for (int b = 0; b < nb; b++) {
        const float * src = x + b * 128;
        block_iso4_0 * blk = &y[b];

        float norm_sq = 0;
        for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        memset(blk->qs, 0, 64);
        float recon_sq = 0;

        for (int g = 0; g < 32; g++) {
            float v0 = src[g*4]*inv, v1 = src[g*4+1]*inv, v2 = src[g*4+2]*inv, v3 = src[g*4+3]*inv;
            float qw=i4_qw[g], qx=i4_qx[g], qy=i4_qy[g], qz=i4_qz[g];
            /* q_L * v */
            float rw = qw*v0 - qx*v1 - qy*v2 - qz*v3;
            float rx = qw*v1 + qx*v0 + qy*v3 - qz*v2;
            float ry = qw*v2 - qx*v3 + qy*v0 + qz*v1;
            float rz = qw*v3 + qx*v2 - qy*v1 + qz*v0;

            float rot[4] = {rw, rx, ry, rz};
            for (int c = 0; c < 4; c++) {
                int j = g*4 + c;
                int idx = nearest_16(rot[c]);
                blk->qs[j/2] |= (idx & 0xF) << ((j%2)*4);
                recon_sq += ISO4_CENTROIDS[idx] * ISO4_CENTROIDS[idx];
            }
        }

        float rn = sqrtf(recon_sq);
        blk->norm = GGML_FP32_TO_FP16((rn > 1e-10f) ? grp_norm / rn : grp_norm);
        blk->rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_iso4_0(const block_iso4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % 128 == 0);
    iso4_init();
    const int nb = k / 128;

    for (int b = 0; b < nb; b++) {
        float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int g = 0; g < 32; g++) {
            float qvals[4];
            for (int c = 0; c < 4; c++) {
                int j = g*4 + c;
                uint8_t idx = (x[b].qs[j/2] >> ((j%2)*4)) & 0xF;
                qvals[c] = ISO4_CENTROIDS[idx];
            }
            /* conj(q_L) * v */
            float qw=i4_qw[g], qx=-i4_qx[g], qy=-i4_qy[g], qz=-i4_qz[g];
            float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
            float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
            float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
            float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

            y[b*128 + g*4]   = rw * norm;
            y[b*128 + g*4+1] = rx * norm;
            y[b*128 + g*4+2] = ry * norm;
            y[b*128 + g*4+3] = rz * norm;
        }
    }
}

size_t quantize_iso4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % 128 == 0);
    size_t row_size = (n_per_row / 128) * sizeof(block_iso4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_iso4_0_ref(
            src + row * n_per_row,
            (block_iso4_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}
