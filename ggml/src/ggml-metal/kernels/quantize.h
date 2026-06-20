#pragma once

#include "common.h"

void quantize_q1_0(device const float * src, device block_q1_0 & dst) {
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    for (int j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK1_0; j++) {
        if (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_q4_0(device const float * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK4_0/2 + j]*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q4_1(device const float * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < QK4_1; j++) {
        const float v = src[j];
        if (min > v) min = v;
        if (max < v) max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK4_1/2 + j] - min)*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q5_0(device const float * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK5_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK5_0/2 + j]*id;

        const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q5_1(device const float * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    float max = src[0];
    float min = src[0];

    for (int j = 1; j < QK5_1; j++) {
        const float v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q8_0(device const float * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

void quantize_iq4_nl(device const float * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_NL; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / kvalues_iq4nl_f[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = src[0        + j]*id;
        const float x1 = src[QK4_NL/2 + j]*id;

        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        const float v0 = kvalues_iq4nl_f[xi0];
        const float v1 = kvalues_iq4nl_f[xi1];
        const float w0 = src[0        + j]*src[0        + j];
        const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}
