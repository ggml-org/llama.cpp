/*
 * ggml-qfx16.c — QFX16 Implementation (Scalar + NEON)
 * Author: Derek Hinch | QomputeAI 2026
 *
 * Row-level quantize/dequantize/vec_dot matching GGML conventions.
 * All functions process entire rows split into blocks internally.
 */
#include "ggml-qfx16.h"
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ═══════════════════════════════════════════════════════════════
 * BF16 → Float LUT (256KB, global, initialized once)
 * ═══════════════════════════════════════════════════════════════ */

static float g_bf16_lut[65536];
static int   g_lut_ready = 0;

void ggml_qfx16_init(void) {
    if (g_lut_ready) return;
    for (uint32_t i = 0; i < 65536; i++) {
        uint32_t f32_bits = i << 16;
        memcpy(&g_bf16_lut[i], &f32_bits, 4);
    }
    g_lut_ready = 1;
}

/* ═══════════════════════════════════════════════════════════════
 * Internal: quantize one block (256 floats → 1 block_qfx16_t)
 * ═══════════════════════════════════════════════════════════════ */

static void encode_block(const float *src, block_qfx16_t *dst, int n) {
    /* Convert floats to BF16 byte pairs */
    uint8_t bytes[QFX16_BLOCK_SIZE * 2];
    int n_bytes = n * 2;

    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &src[i], 4);
        uint16_t bf16 = (uint16_t)(bits >> 16);
        bytes[i * 2]     = (uint8_t)(bf16 >> 8);     /* High byte (exponent+sign) */
        bytes[i * 2 + 1] = (uint8_t)(bf16 & 0xFF);   /* Low byte (mantissa) */
    }

    /* Delta encode + Z-RLE */
    dst->anchor = bytes[0];
    uint16_t sp = 0;
    memset(dst->delta_stream, 0, QFX16_MAX_STREAM);

    for (int i = 0; i < n_bytes - 1 && sp < QFX16_MAX_STREAM - 2; i++) {
        uint8_t delta = (bytes[i + 1] - bytes[i]) & 0xFF;

        if (delta == 0) {
            /* Count zero run */
            int run = 1;
            while (i + run < n_bytes - 1 &&
                   ((bytes[i + run + 1] - bytes[i + run]) & 0xFF) == 0 &&
                   run < 255) {
                run++;
            }
            if (run >= 2) {
                dst->delta_stream[sp++] = QFX16_ZRLE_ESCAPE;
                dst->delta_stream[sp++] = (uint8_t)run;
                i += (run - 1);  /* Loop will increment i once more */
            } else {
                dst->delta_stream[sp++] = 0;
            }
        } else if (delta == QFX16_ZRLE_ESCAPE) {
            /* Escape the escape */
            dst->delta_stream[sp++] = QFX16_ZRLE_ESCAPE;
            dst->delta_stream[sp++] = 0;
        } else {
            dst->delta_stream[sp++] = delta;
        }
    }
    dst->stream_len = sp;
}

/* ═══════════════════════════════════════════════════════════════
 * Internal: dequantize one block (1 block_qfx16_t → up to 256 floats)
 * ═══════════════════════════════════════════════════════════════ */

static void decode_block(const block_qfx16_t *src, float *dst, int n) {
    if (!g_lut_ready) ggml_qfx16_init();

    uint8_t state = src->anchor;
    const uint8_t *stream = src->delta_stream;
    uint16_t sp = 0;
    int wi = 0;       /* Weight index (output floats) */
    int half = 0;     /* 0 = expecting high byte, 1 = expecting low byte */
    uint8_t high = 0;

    /* First byte (anchor) is the first high byte */
    high = state;
    half = 1;

    while (sp < src->stream_len && wi < n) {
        uint8_t byte = stream[sp++];

        if (byte == QFX16_ZRLE_ESCAPE && sp < src->stream_len) {
            uint8_t count = stream[sp++];
            if (count == 0) {
                /* Literal 0xFF delta */
                state = (state + 0xFF) & 0xFF;
            } else {
                /* Zero run: state unchanged for `count` byte positions */
                for (uint8_t r = 0; r < count && wi < n; r++) {
                    if (half == 0) { high = state; half = 1; }
                    else {
                        uint16_t word = ((uint16_t)high << 8) | state;
                        dst[wi++] = g_bf16_lut[word];
                        half = 0;
                    }
                }
                continue;
            }
        } else {
            state = (state + byte) & 0xFF;
        }

        if (half == 0) { high = state; half = 1; }
        else {
            uint16_t word = ((uint16_t)high << 8) | state;
            dst[wi++] = g_bf16_lut[word];
            half = 0;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Row-Level API (GGML contract)
 * ═══════════════════════════════════════════════════════════════ */

void quantize_row_qfx16(const float * __restrict__ x, void * __restrict__ y, int64_t k) {
    if (!g_lut_ready) ggml_qfx16_init();

    block_qfx16_t *blocks = (block_qfx16_t *)y;
    int64_t n_blocks = (k + QFX16_BLOCK_SIZE - 1) / QFX16_BLOCK_SIZE;

    for (int64_t b = 0; b < n_blocks; b++) {
        int block_n = (int)((b == n_blocks - 1) ? (k - b * QFX16_BLOCK_SIZE) : QFX16_BLOCK_SIZE);
        encode_block(&x[b * QFX16_BLOCK_SIZE], &blocks[b], block_n);
    }
}

void dequantize_row_qfx16(const void * __restrict__ x, float * __restrict__ y, int64_t k) {
    if (!g_lut_ready) ggml_qfx16_init();

    const block_qfx16_t *blocks = (const block_qfx16_t *)x;
    int64_t n_blocks = (k + QFX16_BLOCK_SIZE - 1) / QFX16_BLOCK_SIZE;

    for (int64_t b = 0; b < n_blocks; b++) {
        int block_n = (int)((b == n_blocks - 1) ? (k - b * QFX16_BLOCK_SIZE) : QFX16_BLOCK_SIZE);
        decode_block(&blocks[b], &y[b * QFX16_BLOCK_SIZE], block_n);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Vec-Dot: QFX16 weights · Float32 input (LUT-based, no decode buffer)
 * ═══════════════════════════════════════════════════════════════ */

void vec_dot_qfx16_f32(int64_t n, float *s, const void *vx, const float *vy) {
    if (!g_lut_ready) ggml_qfx16_init();

    const block_qfx16_t *blocks = (const block_qfx16_t *)vx;
    int64_t n_blocks = (n + QFX16_BLOCK_SIZE - 1) / QFX16_BLOCK_SIZE;
    float total = 0.0f;

    for (int64_t b = 0; b < n_blocks; b++) {
        const block_qfx16_t *blk = &blocks[b];
        const float *inp = &vy[b * QFX16_BLOCK_SIZE];
        int block_n = (int)((b == n_blocks - 1) ? (n - b * QFX16_BLOCK_SIZE) : QFX16_BLOCK_SIZE);

        uint8_t state = blk->anchor;
        const uint8_t *stream = blk->delta_stream;
        uint16_t sp = 0;
        int wi = 0;
        int half = 0;
        uint8_t high = 0;
        float block_dot = 0.0f;

        high = state;
        half = 1;

        while (sp < blk->stream_len && wi < block_n) {
            uint8_t byte = stream[sp++];

            if (byte == QFX16_ZRLE_ESCAPE && sp < blk->stream_len) {
                uint8_t count = stream[sp++];
                if (count == 0) {
                    state = (state + 0xFF) & 0xFF;
                } else {
                    /* Zero run: same weight value repeated */
                    uint16_t word = ((uint16_t)high << 8) | state;
                    float w_val = g_bf16_lut[word];
                    for (uint8_t r = 0; r < count && wi < block_n; r++) {
                        if (half == 0) { high = state; half = 1; }
                        else {
                            block_dot += w_val * inp[wi++];
                            half = 0;
                        }
                    }
                    continue;
                }
            } else {
                state = (state + byte) & 0xFF;
            }

            if (half == 0) { high = state; half = 1; }
            else {
                uint16_t word = ((uint16_t)high << 8) | state;
                block_dot += g_bf16_lut[word] * inp[wi++];
                half = 0;
            }
        }

        total += block_dot;
    }

    *s = total;
}

/* ═══════════════════════════════════════════════════════════════
 * GGML CPU vec_dot adapter (modern multi-row signature)
 * ═══════════════════════════════════════════════════════════════ */

void ggml_vec_dot_qfx16_f32_cpu(int n, float *s, size_t bs,
    const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    vec_dot_qfx16_f32((int64_t)n, s, vx, (const float *)vy);
}

/* Multi-row quantize wrapper (matches ggml_quantize_chunk signature) */
size_t quantize_qfx16(const float * __restrict__ src, void * __restrict__ dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    size_t row_size = (n_per_row / QFX16_BLOCK_SIZE) * QFX16_TYPE_SIZE;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_qfx16(src, qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}
