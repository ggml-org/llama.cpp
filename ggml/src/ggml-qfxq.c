/*
 * ggml-qfxq.c - Q5_Q: Standard Q8_0 quantization
 * Author: Derek Hinch | QomputeAI 2026
 *
 * This is Q8_0 at the block level. The "Q5" compression happens at the
 * file/transport layer via HWT-CS encoding of the int8 byte stream.
 * At runtime, blocks are identical to Q8_0 for maximum kernel compatibility.
 */
#include "ggml-qfxq.h"
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define QFXQ_NEON 1
#endif

/* FP16 helpers */
static inline uint16_t fp32_to_fp16_raw(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint16_t sign = (b >> 16) & 0x8000;
    int exp = ((b >> 23) & 0xFF) - 127 + 15;
    uint16_t mant = (b >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | mant;
}

static inline float fp16_to_fp32_raw(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF) << 13;
    if (exp == 0) return 0.0f;
    uint32_t b = sign | ((exp - 15 + 127) << 23) | mant;
    float f; memcpy(&f, &b, 4); return f;
}

/* Encode: f32 -> Q8_0 block */
static void encode_block(const float *src, block_qfxq_t *dst, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }

    float scale = amax / 127.0f;
    dst->d = fp32_to_fp16_raw(scale);
    float inv_scale = scale > 0.0f ? 127.0f / amax : 0.0f;

    for (int i = 0; i < n; i++) {
        int v = (int)roundf(src[i] * inv_scale);
        dst->qs[i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
    }
    for (int i = n; i < QFXQ_BLOCK_SIZE; i++) {
        dst->qs[i] = 0;
    }
}

/* Decode: Q8_0 block -> f32 */
static void decode_block(const block_qfxq_t *src, float *dst, int n) {
    float scale = fp16_to_fp32_raw(src->d);
    for (int i = 0; i < n; i++) {
        dst[i] = src->qs[i] * scale;
    }
}

/* Row API */
void quantize_row_qfxq(const float * __restrict__ x, void * __restrict__ y, int64_t k) {
    block_qfxq_t *blocks = (block_qfxq_t *)y;
    int64_t nb = (k + QFXQ_BLOCK_SIZE - 1) / QFXQ_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        int bn = (int)((b == nb-1) ? (k - b*QFXQ_BLOCK_SIZE) : QFXQ_BLOCK_SIZE);
        encode_block(&x[b*QFXQ_BLOCK_SIZE], &blocks[b], bn);
    }
}

void dequantize_row_qfxq(const void * __restrict__ x, float * __restrict__ y, int64_t k) {
    const block_qfxq_t *blocks = (const block_qfxq_t *)x;
    int64_t nb = (k + QFXQ_BLOCK_SIZE - 1) / QFXQ_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        int bn = (int)((b == nb-1) ? (k - b*QFXQ_BLOCK_SIZE) : QFXQ_BLOCK_SIZE);
        decode_block(&blocks[b], &y[b*QFXQ_BLOCK_SIZE], bn);
    }
}

/* Vec-Dot: Q8 weights . f32 input */
void vec_dot_qfxq_f32(int64_t n, float *s, const void *vx, const float *vy) {
    const block_qfxq_t *blocks = (const block_qfxq_t *)vx;
    int64_t n_blocks = (n + QFXQ_BLOCK_SIZE - 1) / QFXQ_BLOCK_SIZE;
    float total = 0.0f;

    for (int64_t b = 0; b < n_blocks; b++) {
        float scale = fp16_to_fp32_raw(blocks[b].d);
        const int8_t *qs = blocks[b].qs;
        const float *inp = &vy[b * QFXQ_BLOCK_SIZE];
        int bn = (int)((b == n_blocks-1) ? (n - b*QFXQ_BLOCK_SIZE) : QFXQ_BLOCK_SIZE);

#ifdef QFXQ_NEON
        float32x4_t vsum = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 3 < bn; i += 4) {
            int16x4_t qi = {qs[i], qs[i+1], qs[i+2], qs[i+3]};
            float32x4_t qf = vcvtq_f32_s32(vmovl_s16(qi));
            float32x4_t vf = vld1q_f32(&inp[i]);
            vsum = vfmaq_f32(vsum, qf, vf);
        }
        float block_dot = vaddvq_f32(vsum) * scale;
        for (; i < bn; i++) block_dot += qs[i] * inp[i] * scale;
#else
        float block_dot = 0.0f;
        for (int i = 0; i < bn; i++) block_dot += qs[i] * inp[i];
        block_dot *= scale;
#endif
        total += block_dot;
    }
    *s = total;
}

void ggml_vec_dot_qfxq_f32_cpu(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    vec_dot_qfxq_f32((int64_t)n, s, vx, (const float *)vy);
}
