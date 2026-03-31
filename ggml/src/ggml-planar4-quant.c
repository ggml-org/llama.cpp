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
    -0.1739260000f, -0.1171950000f, -0.0895270000f, -0.0687560000f,
    -0.0512620000f, -0.0355970000f, -0.0209890000f, -0.0069380000f,
    0.0069380000f, 0.0209890000f, 0.0355970000f, 0.0512620000f,
    0.0687560000f, 0.0895270000f, 0.1171950000f, 0.1739260000f,
};

static float p4_cos[64], p4_sin[64];
static int p4_init = 0;

static void planar4_init(void) {
    if (p4_init) return;
    static const float COS[]={0.7386546135f, 0.8607548475f, -0.7411674857f, 0.9674890637f, -0.7723053098f, -0.8056974411f, -0.0412844308f, 0.2707833052f, 0.9315500855f, 0.6698185802f, 0.9167487621f, -0.8320636749f, 0.6818146110f, -0.9108457565f, -0.0559285842f, -0.9032276273f, 0.7519487143f, -0.8941103816f, -0.1039871648f, -0.6961420774f, -0.1230370328f, -0.9328963161f, -0.2905603051f, 0.4910068214f, 0.7889407277f, -0.1221836656f, -0.6316579580f, 0.3128163815f, -0.9563610554f, 0.9992509484f, 0.9540294409f, 0.8902468085f, 0.7543080449f, -0.8664138913f, -0.5232898593f, 0.3621287644f, -0.8825117350f, 0.8234673142f, -0.9416025877f, -0.5480425358f, -0.6644080281f, -0.6585279703f, -0.2460795939f, 0.9438471198f, 0.2427810431f, -0.1960992366f, 0.2403578013f, -0.8461306095f, 0.0246123374f, 0.3372744620f, 0.9994974732f, -0.3494733870f, 0.7438930869f, 0.8452339768f, -0.6177822948f, -0.2662552595f, -0.5457068086f, -0.9985070229f, 0.7757105827f, 0.6141811609f, -0.9805000424f, 0.5425475240f, -0.5663578510f, -0.4696439803f};
    static const float SIN[]={-0.6740840673f, -0.5090196729f, 0.6713201404f, -0.2529129684f, 0.6352515221f, -0.5923272967f, 0.9991474152f, -0.9626403451f, -0.3636130989f, 0.7425247431f, -0.3994642496f, -0.5546801090f, -0.7315250039f, -0.4127469361f, -0.9984347820f, 0.4291617870f, -0.6592215896f, -0.4478466809f, 0.9945786595f, -0.7179040313f, 0.9924020767f, 0.3601450622f, 0.9568566680f, -0.8711557388f, 0.6144692898f, 0.9925075173f, 0.7752471566f, 0.9498136044f, -0.2921875417f, 0.0386975110f, -0.2997128963f, 0.4554784000f, -0.6565206647f, -0.4993265271f, 0.8521547318f, -0.9321280718f, -0.4702904224f, -0.5673637390f, -0.3367263079f, 0.8364504576f, -0.7473700047f, 0.7525562644f, -0.9692496061f, -0.3303825557f, -0.9700810909f, 0.9805840850f, -0.9706843495f, -0.5329755545f, -0.9996970892f, 0.9414063692f, 0.0316982083f, 0.9369462729f, 0.6682986617f, -0.5343964100f, -0.7863491774f, -0.9639025331f, -0.8379761577f, 0.0546237342f, -0.6310887933f, 0.7891650796f, -0.1965190321f, 0.8400250673f, -0.8241594434f, 0.8828558922f};
    for(int i=0;i<64;i++){p4_cos[i]=COS[i];p4_sin[i]=SIN[i];}
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
