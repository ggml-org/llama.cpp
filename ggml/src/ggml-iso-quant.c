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
    -0.1906850000f, -0.1178320000f, -0.0657170000f, -0.0214600000f,
    0.0214600000f, 0.0657170000f, 0.1178320000f, 0.1906850000f,
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
    static const float QW[]={0.5765609741f, 0.3176580369f, -0.3234235942f, -0.5127438903f, 0.9233905673f, -0.3323571086f, 0.5468608141f, -0.2500519454f, -0.5812215805f, 0.3228830695f, -0.7299832702f, -0.4535493255f, -0.7338157296f, -0.2884652913f, -0.9000198841f, -0.0377033800f, 0.5104404092f, 0.2033989877f, -0.2462528497f, 0.2314069420f, 0.0072374810f, 0.3923372924f, 0.4958070219f, -0.7235037088f, -0.9383618832f, 0.4430379272f, -0.2075705230f, 0.1983736306f, -0.8834578991f, 0.7389573455f, -0.0156172011f, 0.7738668919f};
    static const float QX[]={0.4450169504f, -0.5780548453f, 0.7089627385f, -0.3940812945f, -0.0897334740f, 0.4727236331f, 0.5542563796f, 0.0450818054f, -0.3657043576f, -0.4298477769f, 0.4666220546f, 0.7556306720f, -0.5284956098f, 0.7042509317f, 0.0230921544f, 0.7110687494f, 0.3024962246f, -0.1157865301f, 0.7490812540f, -0.2582575679f, -0.2255804837f, 0.3838746250f, -0.3209520578f, -0.3477301002f, 0.1824720055f, 0.4032751918f, 0.8433781862f, 0.9533935785f, -0.0620501526f, 0.0927560627f, 0.2964956462f, 0.2402082384f};
    static const float QY[]={0.2695076466f, -0.0201656222f, -0.1687686443f, -0.5415957570f, -0.2796611190f, 0.3510629535f, 0.2609911859f, -0.2715902030f, -0.0937586129f, 0.3095585108f, -0.4123268127f, -0.4394895136f, 0.0626545250f, -0.4811822474f, -0.0407132693f, -0.4566248953f, 0.7834537029f, -0.6187923551f, 0.0809760988f, -0.8879503012f, -0.8928058147f, 0.8350352049f, -0.6994170547f, 0.5606835485f, 0.2933705449f, 0.7377059460f, 0.4534837306f, -0.0009816211f, -0.3632916510f, -0.3959124386f, 0.1631654203f, 0.5088164806f};
    static const float QZ[]={-0.6300023794f, -0.7513582706f, -0.6035611629f, 0.5370919704f, 0.2471584976f, 0.7367672324f, 0.5706370473f, 0.9282674193f, 0.7208684087f, -0.7843156457f, -0.2817355990f, -0.1736787707f, 0.4222335219f, -0.4350655377f, 0.4333281815f, 0.5333415866f, 0.1847889870f, 0.7498788238f, 0.6096553802f, -0.3021556735f, -0.3898189068f, 0.0377884321f, 0.4024685621f, 0.2031257302f, 0.0107116764f, -0.3112498820f, 0.1999502629f, -0.2273492515f, 0.2892593443f, 0.5372074246f, 0.9408631325f, 0.2907505929f};
    for(int i=0;i<ISO_N_GROUPS;i++){iso_qw[i]=QW[i];iso_qx[i]=QX[i];iso_qy[i]=QY[i];iso_qz[i]=QZ[i];}
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
