#pragma once

#include "ggml-common.h"
#include "convert.cuh"
#include "../../rocmfp4/rocmfp4_hip_scale.cuh"

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

static __device__ __forceinline__ uint8_t rocmfp4_best_index_scaled_cuda(float x, float inv_scale_half) {
    const float a = fabsf(x * inv_scale_half);
    if (a <= 0.5f) {
        return 0;
    }

    const bool neg = x < 0.0f;
    if (a <= 1.5f) {
        return neg ?  9 : 1;
    }
    if (a <= 2.5f) {
        return neg ? 10 : 2;
    }
    if (a <= 3.5f) {
        return neg ? 11 : 3;
    }
    if (a <= 5.0f) {
        return neg ? 12 : 4;
    }
    if (a <= 7.0f) {
        return neg ? 13 : 5;
    }
    if (a <= 9.0f) {
        return neg ? 14 : 6;
    }

    return neg ? 15 : 7;
}

static __device__ __forceinline__ uint8_t rocmfp4_nearest_scale_ue4m3_cuda(float target_scale_half) {
    if (!(target_scale_half > 0.0f) || !isfinite(target_scale_half)) {
        return 1;
    }

    uint8_t best_e = 1;
    float best_diff = fabsf(rocmfp4_ue4m3_to_fp32_half_finite(best_e) - target_scale_half);

    for (int e = 2; e <= 126; ++e) {
        const float diff = fabsf(rocmfp4_ue4m3_to_fp32_half_finite((uint8_t) e) - target_scale_half);
        if (diff < best_diff) {
            best_diff = diff;
            best_e = (uint8_t) e;
        }
    }

    return best_e;
}

static __device__ __forceinline__ float rocmfp4_block_mse_for_scale_cuda(
        const float * __restrict__ x, int start, int n, int e, float best_err) {
    const float scale_half = rocmfp4_ue4m3_to_fp32_half_finite((uint8_t) e);
    const float inv_scale_half = 1.0f / scale_half;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float xi = x[start + i];
        const uint8_t q = rocmfp4_best_index_scaled_cuda(xi, inv_scale_half);
        const float yi = (float) rocmfp4_decode_i8(q) * scale_half;
        const float d = xi - yi;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static __device__ __forceinline__ uint8_t rocmfp4_choose_scale_ue4m3_cuda(
        const float * __restrict__ x, int start, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float xi = x[start + i];
        max_abs = fmaxf(max_abs, fabsf(xi));
    }

    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const int start_e = rocmfp4_nearest_scale_ue4m3_cuda(max_abs / 10.0f);
    int best_e = 0;
    float best_err = FLT_MAX;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = start_e - delta;
        const int e1 = start_e + delta;

        if (e0 >= 1 && e0 <= 126) {
            const float err = rocmfp4_block_mse_for_scale_cuda(x, start, n, e0, best_err);
            if (err < best_err || (err == best_err && e0 < best_e)) {
                best_err = err;
                best_e = e0;
            }
        }

        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfp4_block_mse_for_scale_cuda(x, start, n, e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = e1;
            }
        }
    }

    return (uint8_t) best_e;
}

static __device__ void quantize_f32_rocmfp4_block(const float * __restrict__ x, block_rocmfp4 * __restrict__ y) {
    const uint8_t e0 = rocmfp4_choose_scale_ue4m3_cuda(x, 0,                 QK_ROCMFP4/2);
    const uint8_t e1 = rocmfp4_choose_scale_ue4m3_cuda(x, QK_ROCMFP4/2, QK_ROCMFP4/2);
    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(e0);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(e1);
    const float id0 = d0 > 0.0f ? 1.0f/d0 : 0.0f;
    const float id1 = d1 > 0.0f ? 1.0f/d1 : 0.0f;

    y->e[0] = e0;
    y->e[1] = e1;

    for (int j = 0; j < QK_ROCMFP4/2; ++j) {
        const float v0 = x[j];
        const float v1 = x[j + QK_ROCMFP4/2];
        const uint8_t q0 = rocmfp4_best_index_scaled_cuda(v0, id0);
        const uint8_t q1 = rocmfp4_best_index_scaled_cuda(v1, id1);
        y->qs[j] = q0 | (q1 << 4);
    }
}

static __device__ void quantize_f32_rocmfp4_fast_block(const float * __restrict__ x, block_rocmfp4_fast * __restrict__ y) {
    const uint8_t e = rocmfp4_choose_scale_ue4m3_cuda(x, 0, QK_ROCMFP4);
    const float d = rocmfp4_ue4m3_to_fp32_half_finite(e);
    const float id = d > 0.0f ? 1.0f/d : 0.0f;

    y->e = e;

    for (int j = 0; j < QK_ROCMFP4/2; ++j) {
        const float v0 = x[j];
        const float v1 = x[j + QK_ROCMFP4/2];
        const uint8_t q0 = rocmfp4_best_index_scaled_cuda(v0, id);
        const uint8_t q1 = rocmfp4_best_index_scaled_cuda(v1, id);
        y->qs[j] = q0 | (q1 << 4);
    }
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

static __device__ void cpy_blck_f32_rocmfp4(const char * cxi, char * cdsti) {
    quantize_f32_rocmfp4_block((const float *)cxi, (block_rocmfp4 *)cdsti);
}

static __device__ void cpy_blck_f32_rocmfp4_fast(const char * cxi, char * cdsti) {
    quantize_f32_rocmfp4_fast_block((const float *)cxi, (block_rocmfp4_fast *)cdsti);
}

template<typename src_t>
static __device__ void cpy_blck_scalar_rocmfp4(const char * cxi, char * cdsti) {
    const src_t * x = (const src_t *) cxi;
    float tmp[QK_ROCMFP4];

#pragma unroll
    for (int j = 0; j < QK_ROCMFP4; ++j) {
        tmp[j] = ggml_cuda_cast<float>(x[j]);
    }

    quantize_f32_rocmfp4_block(tmp, (block_rocmfp4 *) cdsti);
}

template<typename src_t>
static __device__ void cpy_blck_scalar_rocmfp4_fast(const char * cxi, char * cdsti) {
    const src_t * x = (const src_t *) cxi;
    float tmp[QK_ROCMFP4];

#pragma unroll
    for (int j = 0; j < QK_ROCMFP4; ++j) {
        tmp[j] = ggml_cuda_cast<float>(x[j]);
    }

    quantize_f32_rocmfp4_fast_block(tmp, (block_rocmfp4_fast *) cdsti);
}

static __device__ void cpy_blck_f16_rocmfp4(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4<half>(cxi, cdsti);
}

static __device__ void cpy_blck_f16_rocmfp4_fast(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4_fast<half>(cxi, cdsti);
}

static __device__ void cpy_blck_bf16_rocmfp4(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4<nv_bfloat16>(cxi, cdsti);
}

static __device__ void cpy_blck_bf16_rocmfp4_fast(const char * cxi, char * cdsti) {
    cpy_blck_scalar_rocmfp4_fast<nv_bfloat16>(cxi, cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
