// Scalar dequantization helpers and ET-side block-size aliases.

#ifndef QUANTS_H
#define QUANTS_H

#include "math_fp.h"

#include <stdint.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

// 64-byte (one cache line) F16 / F32 block sizes.
#define QK_F16 32
#define QK_F32 16

static inline void dequantize_q8_0_block(const block_q8_0 * block, float * dst) {
    const float scale = fp16_to_fp32(block->d);

    for (int i = 0; i < QK8_0; i++) {
        dst[i] = scale * (float) block->qs[i];
    }
}

// Low nibbles -> dst[0..15], high nibbles -> dst[16..31].
static inline void dequantize_q4_0_block(const block_q4_0 * block, float * dst) {
    const float scale = fp16_to_fp32(block->d);

    for (int i = 0; i < QK4_0 / 2; i++) {
        const uint8_t byte = block->qs[i];
        dst[i]             = scale * (float) ((int) (byte & 0xF) - 8);
        dst[i + QK4_0 / 2] = scale * (float) ((int) (byte >> 4) - 8);
    }
}

// Unpack the 12-byte packed Q3_K block scales into 16 signed 6-bit values.
static inline void unpack_q3_K_scales(const uint8_t * packed, int8_t * out /* [16] */) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    __builtin_memcpy(aux, packed, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    __builtin_memcpy(out, aux, 16);
}

// Dequantize one Q2_K super-block (256 elements) to F32.
static inline void dequantize_q2_K_block(const block_q2_K * block, float * dst) {
    const float     d   = fp16_to_fp32(block->d);
    const float     min = fp16_to_fp32(block->dmin);
    const uint8_t * q   = block->qs;

    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = block->scales[is++];
            float dl = d * (sc & 0xF), ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) *dst++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
            sc = block->scales[is++];
            dl = d * (sc & 0xF); ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) *dst++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
            shift += 2;
        }
        q += 32;
    }
}

// Dequantize one Q3_K super-block (256 elements) to F32.
static inline void dequantize_q3_K_block(const block_q3_K * block, float * dst) {
    const float     d_all = fp16_to_fp32(block->d);
    const uint8_t * q     = block->qs;
    const uint8_t * hm    = block->hmask;
    uint8_t         m     = 1;

    int8_t scales[16];
    unpack_q3_K_scales(block->scales, scales);

    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *dst++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *dst++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

// Unpack the 6-bit scale/min pair for Q4_K group j (groups 4-7 split their high bits).
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline void dequantize_q4_K_block(const block_q4_K * block, float * dst) {
    const uint8_t * q   = block->qs;
    const float     d   = fp16_to_fp32(block->d);
    const float     min = fp16_to_fp32(block->dmin);

    int     is = 0;
    uint8_t sc, m;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) {
            *dst++ = d1 * (q[l] & 0xF) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *dst++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

// Dequantize one Q5_K super-block (256 elements) to F32. Same affine form as
// Q4_K with an extra high bit per weight drawn from qh.
static inline void dequantize_q5_K_block(const block_q5_K * block, float * dst) {
    const uint8_t * ql  = block->qs;
    const uint8_t * qh  = block->qh;
    const float     d   = fp16_to_fp32(block->d);
    const float     min = fp16_to_fp32(block->dmin);

    int     is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, block->scales, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, block->scales, &sc, &m);
        const float d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) *dst++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *dst++ = d2 * ((ql[l] >>  4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}

// Dequantize one Q6_K super-block (256 elements) to F32. Each 6-bit weight is
// (ql nibble | qh 2-bit) - 32, scaled by an int8 per-16 scale and the fp16 d.
static inline void dequantize_q6_K_block(const block_q6_K * block, float * dst) {
    const float          d  = fp16_to_fp32(block->d);
    const uint8_t      * ql = block->ql;
    const uint8_t      * qh = block->qh;
    const int8_t       * sc = block->scales;

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            dst[l +  0] = d * sc[is + 0] * q1;
            dst[l + 32] = d * sc[is + 2] * q2;
            dst[l + 64] = d * sc[is + 4] * q3;
            dst[l + 96] = d * sc[is + 6] * q4;
        }
        dst += 128;
        ql  += 64;
        qh  += 32;
        sc  += 8;
    }
}

#endif  // QUANTS_H
