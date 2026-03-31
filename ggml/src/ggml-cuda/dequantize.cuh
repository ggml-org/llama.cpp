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

// ── PlanarQuant / IsoQuant dequantize for flash attention ───────────
#include "planar-iso-constants.cuh"

static __constant__ float dq_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float dq_centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

// Helper: unpack 3-bit index from block
static __device__ __forceinline__ uint8_t unpack_3bit(const block_planar3_0 * blk, int j) {
    uint8_t low = (blk->qs[j/4] >> ((j%4)*2)) & 0x3;
    uint8_t hi = (blk->signs[j/8] >> (j%8)) & 0x1;
    return low | (hi << 2);
}

// Helper: unpack 4-bit index from turbo4 block
static __device__ __forceinline__ uint8_t unpack_4bit(const block_turbo4_0 * blk, int j) {
    return (blk->qs[j/2] >> ((j%2)*4)) & 0xF;
}

// Planar3: 2D Givens inverse rotation — each pair is independent
static __device__ __forceinline__ void dequantize_planar3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_planar3_0 * x = (const block_planar3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    // iqs must be even (pairs)
    float q0 = dq_centroids_3bit[unpack_3bit(&x[ib], iqs)];
    float q1 = dq_centroids_3bit[unpack_3bit(&x[ib], iqs + 1)];

    // Inverse Givens: pair index = iqs/2
    int p = iqs / 2;
    float c = PI_COS[p];
    float s = PI_SIN[p];
    v.x = ( c * q0 + s * q1) * norm;
    v.y = (-s * q0 + c * q1) * norm;
}

// Iso3: quaternion 4D inverse rotation — needs full 4-element group
static __device__ __forceinline__ void dequantize_iso3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iso3_0 * x = (const block_iso3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    // Quaternion group = iqs/4 (each group has 4 elements)
    int g = iqs / 4;
    int offset = iqs % 4;  // 0 or 2 within the group

    // Unpack all 4 elements of the group
    float qvals[4];
    for (int c = 0; c < 4; c++) {
        qvals[c] = dq_centroids_3bit[unpack_3bit((const block_planar3_0 *)&x[ib], g*4 + c)];
    }

    // Inverse quaternion: conj(q_L) * v
    float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    float results[4] = {rw, rx, ry, rz};
    v.x = results[offset] * norm;
    v.y = results[offset + 1] * norm;
}

// Planar4: 2D Givens + 4-bit
static __device__ __forceinline__ void dequantize_planar4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_planar4_0 * x = (const block_planar4_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    float q0 = dq_centroids_4bit[unpack_4bit(&x[ib], iqs)];
    float q1 = dq_centroids_4bit[unpack_4bit(&x[ib], iqs + 1)];

    int p = iqs / 2;
    float c = PI_COS[p];
    float s = PI_SIN[p];
    v.x = ( c * q0 + s * q1) * norm;
    v.y = (-s * q0 + c * q1) * norm;
}

// Iso4: quaternion + 4-bit
static __device__ __forceinline__ void dequantize_iso4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iso4_0 * x = (const block_iso4_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    int g = iqs / 4;
    int offset = iqs % 4;

    float qvals[4];
    for (int c = 0; c < 4; c++) {
        qvals[c] = dq_centroids_4bit[unpack_4bit(&x[ib], g*4 + c)];
    }

    float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    float results[4] = {rw, rx, ry, rz};
    v.x = results[offset] * norm;
    v.y = results[offset + 1] * norm;
}
