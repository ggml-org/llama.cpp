#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

// ---- q3_K KV cache write: kvantuoja viena 256-super-block (verbatim quantize_row_q3_K_ref portas) ----
static __device__ __forceinline__ int q3k_nearest_int(float fval) {
    fval = fval + 12582912.0f;
    int i = __float_as_int(fval);
    return (i & 0x007fffff) - 0x00400000;
}

// make_q3_quants(16, 4, x, L, do_rmse=true): grazina scale, uzpildo L (continue-atvejui)
static __device__ float q3k_make_scale16(const float * __restrict__ x, int8_t * __restrict__ L) {
    const int n = 16, nmax = 4;
    float maxv = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; ++i) { float ax = fabsf(x[i]); if (ax > amax) { amax = ax; maxv = x[i]; } }
    if (amax < 1e-15f) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.0f; }
    float iscale = -(float)nmax / maxv;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = q3k_nearest_int(iscale * x[i]);
        l = max(-nmax, min(nmax - 1, l));
        L[i] = (int8_t) l;
        float w = x[i]*x[i];
        sumlx += w*x[i]*l;
        suml2 += w*(float)l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = x[i]*x[i];
            float slx = sumlx - w*x[i]*L[i];
            if (slx > 0.0f) {
                float sl2 = suml2 - w*(float)L[i]*L[i];
                int new_l = q3k_nearest_int(x[i] * sl2 / slx);
                new_l = max(-nmax, min(nmax - 1, new_l));
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*(float)new_l*new_l;
                    if (sl2 > 0.0f && slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = (int8_t) new_l; sumlx = slx; suml2 = sl2; ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    for (int i = 0; i < n; ++i) L[i] += nmax;
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static __device__ void quantize_f32_q3_K_block(const float * __restrict__ x, block_q3_K * __restrict__ y) {
    int8_t L[QK_K];
    float scales[QK_K/16];

    float max_scale = 0.0f, amax = 0.0f;
    for (int j = 0; j < QK_K/16; ++j) {
        scales[j] = q3k_make_scale16(x + 16*j, L + 16*j);
        float sc = fabsf(scales[j]);
        if (sc > amax) { amax = sc; max_scale = scales[j]; }
    }

    for (int t = 0; t < 12; ++t) y->scales[t] = 0;
    if (max_scale != 0.0f) {
        float iscale = -32.0f / max_scale;
        for (int j = 0; j < QK_K/16; ++j) {
            int l = q3k_nearest_int(iscale * scales[j]);
            l = max(-32, min(31, l)) + 32;
            if (j < 8) y->scales[j] = l & 0xF;
            else y->scales[j-8] |= ((l & 0xF) << 4);
            l >>= 4;
            y->scales[j%4 + 8] |= (l << (2*(j/4)));
        }
        y->d = 1.0f / iscale;
    } else {
        y->d = 0.0f;
    }

    for (int j = 0; j < QK_K/16; ++j) {
        int8_t sc = j < 8 ? (int8_t)(y->scales[j] & 0xF) : (int8_t)(y->scales[j-8] >> 4);
        sc = (int8_t)((sc | (((y->scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32);
        float d = (float) y->d * (float) sc;
        if (d == 0.0f) continue;
        for (int ii = 0; ii < 16; ++ii) {
            int l = q3k_nearest_int(x[16*j + ii] / d);
            l = max(-4, min(3, l));
            L[16*j + ii] = (int8_t)(l + 4);
        }
    }

    for (int t = 0; t < QK_K/8; ++t) y->hmask[t] = 0;
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; ++j) {
        if (L[j] > 3) { y->hmask[m] |= hm; L[j] -= 4; }
        if (++m == QK_K/8) { m = 0; hm <<= 1; }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            y->qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
