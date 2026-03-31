/*
 * IsoQuant: KV cache compression via quaternion 4D block rotation + Lloyd-Max
 * Based on: ParaMind2025/isoquant
 *
 * Uses quaternion sandwich product T(v) = q_L * v for 4D block rotation.
 * 16 FMAs per quaternion multiply (4 groups of 4 elements = 32 groups for d=128).
 * Better decorrelation than PlanarQuant (2D) but cheaper than WHT (d log d).
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

#define ISO_D 128
#define ISO_SEED 42
#define ISO_N_GROUPS 32  /* 128 / 4 */

static const float ISO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* Unit quaternions (one per 4D group, lazy init) */
static float iso_qw[ISO_N_GROUPS];
static float iso_qx[ISO_N_GROUPS];
static float iso_qy[ISO_N_GROUPS];
static float iso_qz[ISO_N_GROUPS];
static int iso_rotation_initialized = 0;

static uint64_t iso_prng_state;

static void iso_prng_seed(uint64_t seed) {
    iso_prng_state = seed;
}

static double iso_prng_normal(void) {
    iso_prng_state = iso_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(iso_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    iso_prng_state = iso_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(iso_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void iso_init_rotation(void) {
    if (iso_rotation_initialized) return;
    iso_prng_seed(ISO_SEED);
    for (int i = 0; i < ISO_N_GROUPS; i++) {
        float q[4];
        float norm = 0.0f;
        for (int j = 0; j < 4; j++) {
            q[j] = (float)iso_prng_normal();
            norm += q[j] * q[j];
        }
        norm = sqrtf(norm);
        iso_qw[i] = q[0] / norm;
        iso_qx[i] = q[1] / norm;
        iso_qy[i] = q[2] / norm;
        iso_qz[i] = q[3] / norm;
    }
    iso_rotation_initialized = 1;
}

/* Hamilton product: q * v where v = (0, v1, v2, v3) treated as pure quaternion
 * Returns (rw, rx, ry, rz) */
static void quat_mul(float aw, float ax, float ay, float az,
                     float bw, float bx, float by, float bz,
                     float *rw, float *rx, float *ry, float *rz) {
    *rw = aw*bw - ax*bx - ay*by - az*bz;
    *rx = aw*bx + ax*bw + ay*bz - az*by;
    *ry = aw*by - ax*bz + ay*bw + az*bx;
    *rz = aw*bz + ax*by - ay*bx + az*bw;
}

static int nearest_centroid_iso3(float val) {
    int best = 0;
    float best_d = fabsf(val - ISO_CENTROIDS_3BIT[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(val - ISO_CENTROIDS_3BIT[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

void quantize_row_iso3_0_ref(const float * GGML_RESTRICT x, block_iso3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ISO3 == 0);
    iso_init_rotation();

    const int nb = k / QK_ISO3;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_ISO3;
        block_iso3_0 * blk = &y[block];

        /* 1. L2 norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_ISO3; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize + rotate + quantize */
        memset(blk->qs, 0, QK_ISO3 / 4);
        memset(blk->signs, 0, QK_ISO3 / 8);

        float recon_sq = 0.0f;
        for (int g = 0; g < ISO_N_GROUPS; g++) {
            /* Load 4D block as quaternion (w=0, x=v0, y=v1, z=v2... wait,
             * we treat 4 elements as a quaternion: (v0, v1, v2, v3) */
            float v0 = src[g*4 + 0] * inv_norm;
            float v1 = src[g*4 + 1] * inv_norm;
            float v2 = src[g*4 + 2] * inv_norm;
            float v3 = src[g*4 + 3] * inv_norm;

            /* Forward rotation: rotated = q_L * v (left multiply) */
            float rw, rx, ry, rz;
            quat_mul(iso_qw[g], iso_qx[g], iso_qy[g], iso_qz[g],
                     v0, v1, v2, v3, &rw, &rx, &ry, &rz);

            /* Quantize all 4 components */
            float rotated[4] = {rw, rx, ry, rz};
            for (int c = 0; c < 4; c++) {
                int j = g * 4 + c;
                int idx = nearest_centroid_iso3(rotated[c]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) blk->signs[j / 8] |= (1 << (j % 8));
                recon_sq += ISO_CENTROIDS_3BIT[idx] * ISO_CENTROIDS_3BIT[idx];
            }
        }

        /* 3. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        blk->norm = GGML_FP32_TO_FP16(corrected);
    }
}

void dequantize_row_iso3_0(const block_iso3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ISO3 == 0);
    iso_init_rotation();

    const int nb = k / QK_ISO3;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        for (int g = 0; g < ISO_N_GROUPS; g++) {
            /* Unpack 4 indices */
            float qvals[4];
            for (int c = 0; c < 4; c++) {
                int j = g * 4 + c;
                uint8_t low = (x[block].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
                uint8_t hi = (x[block].signs[j / 8] >> (j % 8)) & 0x1;
                uint8_t idx = low | (hi << 2);
                qvals[c] = ISO_CENTROIDS_3BIT[idx];
            }

            /* Inverse rotation: conj(q_L) * v
             * conj(q) = (w, -x, -y, -z) */
            float rw, rx, ry, rz;
            quat_mul(iso_qw[g], -iso_qx[g], -iso_qy[g], -iso_qz[g],
                     qvals[0], qvals[1], qvals[2], qvals[3],
                     &rw, &rx, &ry, &rz);

            y[block * QK_ISO3 + g*4 + 0] = rw * norm;
            y[block * QK_ISO3 + g*4 + 1] = rx * norm;
            y[block * QK_ISO3 + g*4 + 2] = ry * norm;
            y[block * QK_ISO3 + g*4 + 3] = rz * norm;
        }
    }
}

size_t quantize_iso3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_ISO3 == 0);

    size_t row_size = (n_per_row / QK_ISO3) * sizeof(block_iso3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_iso3_0_ref(
            src + row * n_per_row,
            (block_iso3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
