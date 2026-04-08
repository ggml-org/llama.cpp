/**
 * tq-quants.cuh — CUDA device functions for quant.cpp (turboquant) types
 *
 * Provides __device__ quantize and dequantize functions matching the patterns
 * used by set-rows.cu (quantize_f32_*_block) and getrows.cu (dequantize_*).
 *
 * Only compiled when LLAMA_TURBOQUANT is defined.
 */
#pragma once

#ifdef LLAMA_TURBOQUANT

#include "common.cuh"
#include "ggml.h"

// We need the block struct definitions from turboquant
extern "C" {
#include "turboquant/tq_types.h"
}

// ============================================================
// tq_uniform_4b: min-max uniform 4-bit (block_size=128)
//
// Block layout (68 bytes per 128 elements):
//   scale      (fp16, 2B) — (max - min) / 15
//   zero_point (fp16, 2B) — minimum value
//   qs[64]     (64B)      — 4-bit packed, 2 values/byte, LSB-first
// ============================================================

#define TQ_QK_UNIFORM_4B 128
#define TQ_QR_UNIFORM_4B 1

static __device__ void quantize_f32_tq_uniform_4b_block(
        const float * __restrict__ x, block_tq_uniform_4b * __restrict__ y) {
    float vmin = x[0];
    float vmax = x[0];
    for (int j = 1; j < TQ_QK_UNIFORM_4B; ++j) {
        if (x[j] < vmin) vmin = x[j];
        if (x[j] > vmax) vmax = x[j];
    }

    const float d  = (vmax - vmin) / 15.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    y->scale      = __half_as_ushort(__float2half_rn(d));
    y->zero_point = __half_as_ushort(__float2half_rn(vmin));

    for (int j = 0; j < TQ_QK_UNIFORM_4B / 2; ++j) {
        const float x0 = (x[2*j + 0] - vmin) * id;
        const float x1 = (x[2*j + 1] - vmin) * id;
        const uint8_t q0 = min(15, (int)(x0 + 0.5f));
        const uint8_t q1 = min(15, (int)(x1 + 0.5f));
        y->qs[j] = q0 | (q1 << 4);
    }
}

// With qr=1, iqs = i00 % qk (always even, incremented by 2).
// Returns elements iqs and iqs+1 as v.x and v.y.
static __device__ __forceinline__ void dequantize_tq_uniform_4b(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq_uniform_4b * x = (const block_tq_uniform_4b *) vx;

    const float d   = __half2float(__ushort_as_half(x[ib].scale));
    const float mn  = __half2float(__ushort_as_half(x[ib].zero_point));
    // iqs is the element index (always even), byte index = iqs/2
    const int   vui = x[ib].qs[iqs / 2];

    v.x = (float)(vui & 0xF)  * d + mn;
    v.y = (float)(vui >> 4)   * d + mn;
}


// ============================================================
// tq_uniform_2b: sub-block 2-bit (block_size=128, 4 sub-blocks of 32)
//
// Block layout (48 bytes per 128 elements):
//   sub_scale[4] (fp16, 8B) — per-sub-block scale
//   sub_min[4]   (fp16, 8B) — per-sub-block minimum
//   qs[32]       (32B)      — 2-bit packed, 4 values/byte, LSB-first
// ============================================================

#define TQ_QK_UNIFORM_2B 128
#define TQ_QR_UNIFORM_2B 1

static __device__ void quantize_f32_tq_uniform_2b_block(
        const float * __restrict__ x, block_tq_uniform_2b * __restrict__ y) {
    // Process 4 sub-blocks of 32 elements each
    for (int s = 0; s < TQ_2B_NSUB; ++s) {
        const float * sx = x + s * TQ_2B_SUBK;
        float vmin = sx[0], vmax = sx[0];
        for (int j = 1; j < TQ_2B_SUBK; ++j) {
            if (sx[j] < vmin) vmin = sx[j];
            if (sx[j] > vmax) vmax = sx[j];
        }
        const float d  = (vmax - vmin) / 3.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y->sub_scale[s] = __half_as_ushort(__float2half_rn(d));
        y->sub_min[s]   = __half_as_ushort(__float2half_rn(vmin));

        // Pack 2-bit values: 4 per byte, LSB-first
        for (int j = 0; j < TQ_2B_SUBK / 4; ++j) {
            uint8_t packed = 0;
            for (int k = 0; k < 4; ++k) {
                float v = (sx[4*j + k] - vmin) * id;
                uint8_t q = min(3, (int)(v + 0.5f));
                packed |= (q << (2*k));
            }
            y->qs[s * (TQ_2B_SUBK / 4) + j] = packed;
        }
    }
}

static __device__ __forceinline__ void dequantize_tq_uniform_2b(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq_uniform_2b * x = (const block_tq_uniform_2b *) vx;

    // With qr=1, iqs is the element index (always even). Return elements iqs and iqs+1.
    const int elem0 = iqs;
    const int elem1 = iqs + 1;

    // Determine sub-block for each element
    const int s0 = elem0 / TQ_2B_SUBK;
    const int s1 = elem1 / TQ_2B_SUBK;

    const float d0  = __half2float(__ushort_as_half(x[ib].sub_scale[s0]));
    const float mn0 = __half2float(__ushort_as_half(x[ib].sub_min[s0]));
    const float d1  = __half2float(__ushort_as_half(x[ib].sub_scale[s1]));
    const float mn1 = __half2float(__ushort_as_half(x[ib].sub_min[s1]));

    // Extract 2-bit values: 4 per byte, LSB-first
    const int byte0 = elem0 / 4;
    const int shift0 = (elem0 % 4) * 2;
    const int byte1 = elem1 / 4;
    const int shift1 = (elem1 % 4) * 2;

    const uint8_t q0 = (x[ib].qs[byte0] >> shift0) & 0x3;
    const uint8_t q1 = (x[ib].qs[byte1] >> shift1) & 0x3;

    v.x = (float)q0 * d0 + mn0;
    v.y = (float)q1 * d1 + mn1;
}

// ============================================================
// tq_uniform_3b: sub-block 3-bit (block_size=128, 4 sub-blocks of 32)
//
// Block layout (64 bytes per 128 elements):
//   sub_scale[4] (fp16, 8B) — per-sub-block scale
//   sub_min[4]   (fp16, 8B) — per-sub-block minimum
//   qs[48]       (48B)      — 3-bit packed data
// ============================================================

#define TQ_QK_UNIFORM_3B 128
#define TQ_QR_UNIFORM_3B 1

static __device__ void quantize_f32_tq_uniform_3b_block(
        const float * __restrict__ x, block_tq_uniform_3b * __restrict__ y) {
    for (int s = 0; s < TQ_3B_NSUB; ++s) {
        const float * sx = x + s * TQ_3B_SUBK;
        float vmin = sx[0], vmax = sx[0];
        for (int j = 1; j < TQ_3B_SUBK; ++j) {
            if (sx[j] < vmin) vmin = sx[j];
            if (sx[j] > vmax) vmax = sx[j];
        }
        const float d  = (vmax - vmin) / 7.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y->sub_scale[s] = __half_as_ushort(__float2half_rn(d));
        y->sub_min[s]   = __half_as_ushort(__float2half_rn(vmin));
    }

    // Pack 3-bit values across the entire block (128 elements -> 48 bytes)
    // 8 values per 3 bytes: bits [0..2], [3..5], [6..7|0], [1..3], ...
    for (int i = 0; i < TQ_QK_UNIFORM_3B; i += 8) {
        uint8_t q[8];
        for (int k = 0; k < 8; ++k) {
            int idx = i + k;
            int s = idx / TQ_3B_SUBK;
            float d  = __half2float(__ushort_as_half(y->sub_scale[s]));
            float mn = __half2float(__ushort_as_half(y->sub_min[s]));
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            float v = (x[idx] - mn) * id;
            q[k] = min(7, (int)(v + 0.5f));
        }
        int byte_idx = (i / 8) * 3;
        y->qs[byte_idx + 0] = q[0] | (q[1] << 3) | (q[2] << 6);
        y->qs[byte_idx + 1] = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | (q[5] << 7);
        y->qs[byte_idx + 2] = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
    }
}

static __device__ __forceinline__ void dequantize_tq_uniform_3b(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq_uniform_3b * x = (const block_tq_uniform_3b *) vx;

    // With qr=1, iqs is the element index (always even). Return elements iqs and iqs+1.
    const int elem0 = iqs;
    const int elem1 = iqs + 1;

    // Extract 3-bit value for a given element index
    auto extract_3b = [&](int elem) -> uint8_t {
        int bit_offset = elem * 3;
        int byte_idx = bit_offset / 8;
        int bit_shift = bit_offset % 8;
        uint16_t word = x[ib].qs[byte_idx];
        if (bit_shift > 5 && byte_idx + 1 < (int)sizeof(x[ib].qs)) {
            word |= ((uint16_t)x[ib].qs[byte_idx + 1]) << 8;
        }
        return (word >> bit_shift) & 0x7;
    };

    const int s0 = elem0 / TQ_3B_SUBK;
    const int s1 = elem1 / TQ_3B_SUBK;

    v.x = (float)extract_3b(elem0) * __half2float(__ushort_as_half(x[ib].sub_scale[s0])) + __half2float(__ushort_as_half(x[ib].sub_min[s0]));
    v.y = (float)extract_3b(elem1) * __half2float(__ushort_as_half(x[ib].sub_scale[s1])) + __half2float(__ushort_as_half(x[ib].sub_min[s1]));
}

#endif // LLAMA_TURBOQUANT
