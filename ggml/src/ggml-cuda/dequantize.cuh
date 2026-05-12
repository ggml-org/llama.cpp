#include "common.cuh"
#include "turbo-quant.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// Turbo4: 4-bit PolarQuant (nibble packed), block size 128
// iqs is the element index within the block (even), produces elements iqs and iqs+1
static __device__ __forceinline__ void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo4_dequant_element(&x[ib], iqs + 0, norm);
    v.y = turbo4_dequant_element(&x[ib], iqs + 1, norm);
}

// Turbo3: 3-bit PolarQuant (2-bit qs + 1-bit sign), block size 32
// iqs is the element index within the block (even), produces elements iqs and iqs+1
static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo3_dequant_element(&x[ib], iqs + 0, norm);
    v.y = turbo3_dequant_element(&x[ib], iqs + 1, norm);
}

// Turbo2: 2-bit PolarQuant (2-bit qs only, no sign), block size 32
static __device__ __forceinline__ void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo2_dequant_element(&x[ib], iqs + 0, norm);
    v.y = turbo2_dequant_element(&x[ib], iqs + 1, norm);
}

// TQ4_1S: 4-bit weight type with inverse WHT, block size 32, dual half-block scales
// Cold path only (convert.cu) — dequants full block, applies inverse RHT, returns pair
static __device__ __forceinline__ void dequantize_tq4_1s(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq4_1s * x = (const block_tq4_1s *) vx;
    const float d0 = __half2float(x[ib].d0);
    const float d1 = __half2float(x[ib].d1);

    // Dequant full block (centroid lookup + scale)
    float buf[32];
    for (int j = 0; j < 32; j++) {
        uint8_t idx = (x[ib].qs[j / 2] >> ((j & 1) * 4)) & 0xF;
        float d = (j < 16) ? d0 : d1;
        buf[j] = TQ4_CENTROIDS_WEIGHT[idx] * d;
    }

    // Inverse RHT: WHT butterfly then normalize+unsign
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j] = a + b; buf[j + step] = a - b;
            }
        }
    }
    const float inv_sqrt32 = 0.17677669529663688f;
    for (int j = 0; j < 32; j++) buf[j] *= inv_sqrt32 * TQ_WEIGHT_SIGNS[j];

    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}

// TQ3_1S: 3-bit weight type with inverse WHT, block size 32, dual half-block scales
// 3-bit packing: 4 groups of 8 indices in 3 bytes each (24 bits = 8 * 3-bit)
static __device__ __forceinline__ void dequantize_tq3_1s(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq3_1s * x = (const block_tq3_1s *) vx;
    const float d0 = __half2float(x[ib].d0);
    const float d1 = __half2float(x[ib].d1);

    // Unpack all 32 3-bit indices (4 groups of 8 in 3 bytes)
    float buf[32];
    for (int g = 0; g < 4; g++) {
        const uint8_t * qp = x[ib].qs + g * 3;
        uint8_t idx[8];
        idx[0] =  qp[0]       & 7;
        idx[1] = (qp[0] >> 3) & 7;
        idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
        idx[3] = (qp[1] >> 1) & 7;
        idx[4] = (qp[1] >> 4) & 7;
        idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
        idx[6] = (qp[2] >> 2) & 7;
        idx[7] = (qp[2] >> 5) & 7;

        for (int i = 0; i < 8; i++) {
            int j = g * 8 + i;
            float d = (j < 16) ? d0 : d1;
            buf[j] = TQ3_CENTROIDS_WEIGHT[idx[i]] * d;
        }
    }

    // Inverse RHT: WHT butterfly then normalize+unsign
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j] = a + b; buf[j + step] = a - b;
            }
        }
    }
    const float inv_sqrt32 = 0.17677669529663688f;
    for (int j = 0; j < 32; j++) buf[j] *= inv_sqrt32 * TQ_WEIGHT_SIGNS[j];

    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}
