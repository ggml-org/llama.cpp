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
    -0.1906850000f, -0.1178320000f, -0.0657170000f, -0.0214600000f,
    0.0214600000f, 0.0657170000f, 0.1178320000f, 0.1906850000f,
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
    static const float COS[]={0.7386546135f, 0.8607548475f, -0.7411674857f, 0.9674890637f, -0.7723053098f, -0.8056974411f, -0.0412844308f, 0.2707833052f, 0.9315500855f, 0.6698185802f, 0.9167487621f, -0.8320636749f, 0.6818146110f, -0.9108457565f, -0.0559285842f, -0.9032276273f, 0.7519487143f, -0.8941103816f, -0.1039871648f, -0.6961420774f, -0.1230370328f, -0.9328963161f, -0.2905603051f, 0.4910068214f, 0.7889407277f, -0.1221836656f, -0.6316579580f, 0.3128163815f, -0.9563610554f, 0.9992509484f, 0.9540294409f, 0.8902468085f, 0.7543080449f, -0.8664138913f, -0.5232898593f, 0.3621287644f, -0.8825117350f, 0.8234673142f, -0.9416025877f, -0.5480425358f, -0.6644080281f, -0.6585279703f, -0.2460795939f, 0.9438471198f, 0.2427810431f, -0.1960992366f, 0.2403578013f, -0.8461306095f, 0.0246123374f, 0.3372744620f, 0.9994974732f, -0.3494733870f, 0.7438930869f, 0.8452339768f, -0.6177822948f, -0.2662552595f, -0.5457068086f, -0.9985070229f, 0.7757105827f, 0.6141811609f, -0.9805000424f, 0.5425475240f, -0.5663578510f, -0.4696439803f};
    static const float SIN[]={-0.6740840673f, -0.5090196729f, 0.6713201404f, -0.2529129684f, 0.6352515221f, -0.5923272967f, 0.9991474152f, -0.9626403451f, -0.3636130989f, 0.7425247431f, -0.3994642496f, -0.5546801090f, -0.7315250039f, -0.4127469361f, -0.9984347820f, 0.4291617870f, -0.6592215896f, -0.4478466809f, 0.9945786595f, -0.7179040313f, 0.9924020767f, 0.3601450622f, 0.9568566680f, -0.8711557388f, 0.6144692898f, 0.9925075173f, 0.7752471566f, 0.9498136044f, -0.2921875417f, 0.0386975110f, -0.2997128963f, 0.4554784000f, -0.6565206647f, -0.4993265271f, 0.8521547318f, -0.9321280718f, -0.4702904224f, -0.5673637390f, -0.3367263079f, 0.8364504576f, -0.7473700047f, 0.7525562644f, -0.9692496061f, -0.3303825557f, -0.9700810909f, 0.9805840850f, -0.9706843495f, -0.5329755545f, -0.9996970892f, 0.9414063692f, 0.0316982083f, 0.9369462729f, 0.6682986617f, -0.5343964100f, -0.7863491774f, -0.9639025331f, -0.8379761577f, 0.0546237342f, -0.6310887933f, 0.7891650796f, -0.1965190321f, 0.8400250673f, -0.8241594434f, 0.8828558922f};
    for(int i=0;i<PLANAR_D/2;i++){planar_cos[i]=COS[i];planar_sin[i]=SIN[i];}
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
