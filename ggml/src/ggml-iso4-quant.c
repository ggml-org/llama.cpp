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
    -0.1739260000f, -0.1171950000f, -0.0895270000f, -0.0687560000f,
    -0.0512620000f, -0.0355970000f, -0.0209890000f, -0.0069380000f,
    0.0069380000f, 0.0209890000f, 0.0355970000f, 0.0512620000f,
    0.0687560000f, 0.0895270000f, 0.1171950000f, 0.1739260000f,
};

static float i4_qw[32], i4_qx[32], i4_qy[32], i4_qz[32];
static int i4_init = 0;

static void iso4_init(void) {
    if (i4_init) return;
    static const float QW[]={0.5765609741f, 0.3176580369f, -0.3234235942f, -0.5127438903f, 0.9233905673f, -0.3323571086f, 0.5468608141f, -0.2500519454f, -0.5812215805f, 0.3228830695f, -0.7299832702f, -0.4535493255f, -0.7338157296f, -0.2884652913f, -0.9000198841f, -0.0377033800f, 0.5104404092f, 0.2033989877f, -0.2462528497f, 0.2314069420f, 0.0072374810f, 0.3923372924f, 0.4958070219f, -0.7235037088f, -0.9383618832f, 0.4430379272f, -0.2075705230f, 0.1983736306f, -0.8834578991f, 0.7389573455f, -0.0156172011f, 0.7738668919f};
    static const float QX[]={0.4450169504f, -0.5780548453f, 0.7089627385f, -0.3940812945f, -0.0897334740f, 0.4727236331f, 0.5542563796f, 0.0450818054f, -0.3657043576f, -0.4298477769f, 0.4666220546f, 0.7556306720f, -0.5284956098f, 0.7042509317f, 0.0230921544f, 0.7110687494f, 0.3024962246f, -0.1157865301f, 0.7490812540f, -0.2582575679f, -0.2255804837f, 0.3838746250f, -0.3209520578f, -0.3477301002f, 0.1824720055f, 0.4032751918f, 0.8433781862f, 0.9533935785f, -0.0620501526f, 0.0927560627f, 0.2964956462f, 0.2402082384f};
    static const float QY[]={0.2695076466f, -0.0201656222f, -0.1687686443f, -0.5415957570f, -0.2796611190f, 0.3510629535f, 0.2609911859f, -0.2715902030f, -0.0937586129f, 0.3095585108f, -0.4123268127f, -0.4394895136f, 0.0626545250f, -0.4811822474f, -0.0407132693f, -0.4566248953f, 0.7834537029f, -0.6187923551f, 0.0809760988f, -0.8879503012f, -0.8928058147f, 0.8350352049f, -0.6994170547f, 0.5606835485f, 0.2933705449f, 0.7377059460f, 0.4534837306f, -0.0009816211f, -0.3632916510f, -0.3959124386f, 0.1631654203f, 0.5088164806f};
    static const float QZ[]={-0.6300023794f, -0.7513582706f, -0.6035611629f, 0.5370919704f, 0.2471584976f, 0.7367672324f, 0.5706370473f, 0.9282674193f, 0.7208684087f, -0.7843156457f, -0.2817355990f, -0.1736787707f, 0.4222335219f, -0.4350655377f, 0.4333281815f, 0.5333415866f, 0.1847889870f, 0.7498788238f, 0.6096553802f, -0.3021556735f, -0.3898189068f, 0.0377884321f, 0.4024685621f, 0.2031257302f, 0.0107116764f, -0.3112498820f, 0.1999502629f, -0.2273492515f, 0.2892593443f, 0.5372074246f, 0.9408631325f, 0.2907505929f};
    for(int i=0;i<32;i++){i4_qw[i]=QW[i];i4_qx[i]=QX[i];i4_qy[i]=QY[i];i4_qz[i]=QZ[i];}
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
