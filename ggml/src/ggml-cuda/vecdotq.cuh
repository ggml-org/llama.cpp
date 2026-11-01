#pragma once

#include "common.cuh"

#include <cstdint>

static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;

    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    return x32;
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// q4 contains 8 indices with 4 bit each.
// This function selects those bytes from table that are at those indices and returns them as int2.
// The first int contains the bytes with even indices in q4, the second int contains the bytes with odd indices in q4.
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
#if defined(GGML_USE_HIP)
    // Load the 16-byte table into four 32-bit unsigned integers.
    const uint32_t *values = (const uint32_t *)table;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // Perform lookups in the lower half of the table (indices 0-7).
    uint32_t v_even_low = ggml_hip_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low  = ggml_hip_perm(values[1], values[0], q_odd & 0x07070707);

    // Perform lookups in the upper half of the table (indices 8-15).
    uint32_t v_even_high = ggml_hip_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high  = ggml_hip_perm(values[3], values[2], q_odd & 0x07070707);

    // Select between the low and high results based on the MSB of each index nibble.
    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t res_x     = ggml_hip_perm(v_even_high, v_even_low, mask_even);
    uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
    uint32_t res_y     = ggml_hip_perm(v_odd_high, v_odd_low, mask_odd);

    return make_int2(res_x, res_y);
#elif !defined(GGML_USE_MUSA)
    // CUDA does not have an instruction for selecting bytes with 4 bit indices.
    // However, __byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead.
    const uint32_t * table32 = (const uint32_t *) table;

    // __byte_perm selects bytes based on the lower 16 bits in its third argument.
    // Therefore, do 2 iterations over the 32 bits in q4 with 0 and 16 shift.
    // To handle the fourth bit, first call _byte_perm both for the low and the high 64 bit of table, using the low 3 bits.
    // Then, call __byte_perm again to select from the low and high bytes based on the fourth bit.
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;

        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }

    // tmp contains the bytes from tyble in the same order as the 4 bit indices in q4.
    // However, for the result we need ints with all even/odd 4 bit indices in q4.
    // Therefore, 2 more calls to __byte_perm to put the bytes in the correct order.
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#else
    // Generic implementation.
    const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t * q0_8   = (const int8_t *) &q0_32;
    const char4    val0_8 = make_char4(
        table[q0_8[0]], table[q0_8[1]], table[q0_8[2]], table[q0_8[3]]);

    const int      q1_32  = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t * q1_8   = (const int8_t *) &q1_32;
    const char4    val1_8 = make_char4(
        table[q1_8[0]], table[q1_8[1]], table[q1_8[2]], table[q1_8[3]]);

    return make_int2(*((const int *) &val0_8), *((const int *) &val1_8));
#endif
}

static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    // with xor we can "correct" the bit instead of having to mask
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    // broadcast over uint to allow for 0x08040201 / 0x80402010 as selectors
    return s * 0x01010101;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q1_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism
#define VDR_Q1_0_Q8_1_MMQ  4  // Q1_0 has 128 bits (4 ints) per block

#define VDR_Q2_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism
#define VDR_Q2_0_Q8_1_MMQ  2  // Q2_0 group 64: 128 bits (4 ints) per block, 2 32-element chunks

// Ternary: one 32-element chunk per vec_dot call (QK_K = 256 elements per block, 8 chunks).
#define VDR_TQ1_0_Q8_1_MMVQ 1
// MMQ unpacks the trits to signed bytes in the tile loader, so the tile row has the Q8_0 shape:
// 8 ints (32 bytes) per 32-element chunk, consumed by the regular q8_0 vec_dot (dp4a or mma).
#define VDR_TQ1_0_Q8_1_MMQ  8

// TQ2_0: same shape as TQ1_0 (QK_K = 256 elements per block, 8 chunks of 32 elements).
#define VDR_TQ2_0_Q8_1_MMVQ 1
// MMQ expands the digits to signed bytes in the tile loader, so the tile row has the Q8_0 shape:
// 8 ints (32 bytes) per 32-element chunk, consumed by the regular q8_0 vec_dot (dp4a or mma).
#define VDR_TQ2_0_Q8_1_MMQ  8

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const T & d8_0, const T & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * ((T) sumi);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_16_q8_1_impl(
    const int * v, const int * u, const float * d8_0, const float & d8_1) {

    float sumf = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < vdr; i0 += QI8_0/2) {
        int sumi = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_0/2; ++i) {
            // SIMD dot product of quantized values
            sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
        }

        sumf += d8_0[i0/(QI8_0/2)]*sumi;
    }

    return d8_1*sumf;
}

#define VDR_MXFP4_Q8_1_MMVQ 2
#define VDR_MXFP4_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_NVFP4_Q8_1_MMVQ 4
#define VDR_NVFP4_Q8_1_MMQ  8

static __device__ __forceinline__ float vec_dot_nvfp4_q8_1(
                                        const void * __restrict__ vbq,
                                        const block_q8_1 * __restrict__ bq8_1,
                                        const int32_t & kbx,
                                        const int32_t & iqs) {

    const block_nvfp4 * bq4 = (const block_nvfp4 *) vbq + kbx;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        const int32_t iqs0 = iqs + 2*i;
        const int32_t iqs1 = iqs0 + 1;
        const int32_t is = iqs0 >> 1;
        const int2 v0 = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        const int2 v1 = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        const block_q8_1 * bq8 = bq8_1 + (is >> 1);
        const int32_t i8 = ((is & 1) << 2);

        int sumi = ggml_cuda_dp4a(v0.x, get_int_b4(bq8->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(v0.y, get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(v1.x, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(v1.y, get_int_b4(bq8->qs, i8 + 3), sumi);

        const float d = ggml_cuda_ue4m3_to_fp32(bq4->d[is]) * __low2float(bq8->ds);
        sum += d * float(sumi);
    }

    return sum;
}
#define VDR_Q2_K_Q8_1_MMVQ 2
#define VDR_Q2_K_Q8_1_MMQ  4

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int * __restrict__ v, const int * __restrict__ u, const int * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[i/2] >> (16*(i%2));

        int sumi_d = 0;
        int sumi_m = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q2_K_Q8_1_MMVQ; ++j) {
            const int vi = (v[j] >> (2*i)) & 0x03030303;

            sumi_d = ggml_cuda_dp4a(vi,         u[VDR_Q2_K_Q8_1_MMVQ*i + j], sumi_d); // SIMD dot product
            sumi_m = ggml_cuda_dp4a(0x01010101, u[VDR_Q2_K_Q8_1_MMVQ*i + j], sumi_m); // sum of q8_1 values
        }

        sumf_d += d8[i] * (sumi_d * (sc & 0xF));
        sumf_m += d8[i] * (sumi_m * ((sc >> 4) & 0xF)); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// contiguous v/x + u/y values
template <int ns8>
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const half2 * dm2, const float & d8, const half2 * s8) {

    float sumf    = 0.0f;
    float sumf_d8 = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < QR2_K*VDR_Q2_K_Q8_1_MMQ; i0 += QI8_1) {
        const float2 dm2f0 = __half22float2(dm2[i0/(QI8_1/2) + 0]);
        int sumi_d0 = 0;

        const float2 dm2f1 = __half22float2(dm2[i0/(QI8_1/2) + 1]);
        int sumi_d1 = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d0 = ggml_cuda_dp4a(v[i], u[i], sumi_d0);
        }
        sumf_d8 += dm2f0.x * sumi_d0;

#pragma unroll
        for (int i = i0 + QI8_1/2; i < i0 + QI8_1; ++i) {
            sumi_d1 = ggml_cuda_dp4a(v[i], u[i], sumi_d1);
        }
        sumf_d8 += dm2f1.x * sumi_d1;

        if (i0/QI8_1 < ns8) {
            const float2 s8f = __half22float2(s8[i0/QI8_1]);
            sumf -= dm2f0.y*s8f.x;
            sumf -= dm2f1.y*s8f.y;
        } else {
            int sumi_m0 = 0;
#pragma unroll
            for (int i = i0; i < i0 + QI8_1/2; ++i) {
                sumi_m0 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m0);
            }
            sumf_d8 -= dm2f0.y * sumi_m0;

            int sumi_m1 = 0;
#pragma unroll
            for (int i = i0 + QI8_1/2; i < i0 + QI8_1; ++i) {
                sumi_m1 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m1);
            }
            sumf_d8 -= dm2f1.y * sumi_m1;
        }
    }

    return sumf + d8*sumf_d8;
}

#define VDR_Q3_K_Q8_1_MMVQ 2
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const int * __restrict__ scales,
    const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        // 4 low bits from scales[0..7], 2 high bits from scales[8..11]
        const int sc_low  = (scales[i/2] >> (16*(i%2))) & 0x0F;
        const int sc_high = (scales[2] >> (16*(i%2) + 2*(i/2))) & 0x03;
        const int sc = (sc_low | (sc_high << 4)) - 32;

        int sumi = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q3_K_Q8_1_MMVQ; ++j) {
            const int vil = (vl[j] >> (2*i)) & 0x03030303;
            const int vih = ((vh[j] >> i) << 2) & 0x04040404;

            // (vil | vih) - 4 for each byte, the - 4 is done by the second dp4a
            sumi = ggml_cuda_dp4a(vil | vih,  u[VDR_Q3_K_Q8_1_MMVQ*i + j], sumi); // SIMD dot product
            sumi = ggml_cuda_dp4a(0xFCFCFCFC, u[VDR_Q3_K_Q8_1_MMVQ*i + j], sumi);
        }

        sumf += d8[i] * (sumi * sc);
    }

    return d3 * sumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q6_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        int sumi = 0;
#pragma unroll
        for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
            const int vil = (vl[j] >> (4*i)) & 0x0F0F0F0F;
            const int vih = ((vh[j] >> (4*i)) << 4) & 0x30303030;

            // (vil | vih) - 32 for each byte, the - 32 is done by the second dp4a
            sumi = ggml_cuda_dp4a(vil | vih,  u[VDR_Q6_K_Q8_1_MMVQ*i + j], sumi); // SIMD dot product
            sumi = ggml_cuda_dp4a(0xE0E0E0E0, u[VDR_Q6_K_Q8_1_MMVQ*i + j], sumi);
        }

        sumf += d8[i] * (sumi * sc);
    }

    return d*sumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float sumf_d = 0.0f;

    const int      sc_packed = get_int_b4(sc, 0);
    const int8_t * sc_reg    = (const int8_t *) &sc_packed;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc_reg[i0/2+0]*sumi_d.x + sc_reg[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
}

static __device__ __forceinline__ float vec_dot_q1_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q1_0 * bq1_0 = (const block_q1_0 *) vbq + kbx;

    // Q1_0: 128 elements with ONE scale
    // Q8_1: 32 elements per block with individual scales
    // iqs selects which of the 4 chunks of 32 elements to process (0-3)

    const float     d1 = bq1_0->d;
    const int16_t * qs = (const int16_t *) bq1_0->qs + iqs * 2;

    // Process only the chunk specified by iqs
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int q  = qs[j];

        const int u0 = get_int_b4(bq8_1_chunk->qs, j*4+0);
        const int u1 = get_int_b4(bq8_1_chunk->qs, j*4+1);
        const int u2 = get_int_b4(bq8_1_chunk->qs, j*4+2);
        const int u3 = get_int_b4(bq8_1_chunk->qs, j*4+3);

#if defined(GGML_USE_HIP)
        const uint32_t q_bits = (uint16_t) q;

        // Expand each group of four Q1_0 bits into byte selectors for v_perm_b32.
        const uint32_t q0_bits = q_bits >>  0;
        const uint32_t q1_bits = q_bits >>  4;
        const uint32_t q2_bits = q_bits >>  8;
        const uint32_t q3_bits = q_bits >> 12;

        const uint32_t q0_indices = (q0_bits & 0x01) | ((q0_bits & 0x02) << 7) | ((q0_bits & 0x04) << 14) | ((q0_bits & 0x08) << 21);
        const uint32_t q1_indices = (q1_bits & 0x01) | ((q1_bits & 0x02) << 7) | ((q1_bits & 0x04) << 14) | ((q1_bits & 0x08) << 21);
        const uint32_t q2_indices = (q2_bits & 0x01) | ((q2_bits & 0x02) << 7) | ((q2_bits & 0x04) << 14) | ((q2_bits & 0x08) << 21);
        const uint32_t q3_indices = (q3_bits & 0x01) | ((q3_bits & 0x02) << 7) | ((q3_bits & 0x04) << 14) | ((q3_bits & 0x08) << 21);

        const int v0 = ggml_hip_perm(0x000001FF, 0x000001FF, q0_indices);
        const int v1 = ggml_hip_perm(0x000001FF, 0x000001FF, q1_indices);
        const int v2 = ggml_hip_perm(0x000001FF, 0x000001FF, q2_indices);
        const int v3 = ggml_hip_perm(0x000001FF, 0x000001FF, q3_indices);
#else
        // unpack crumbs into nibble indices
        const int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0); // [0, 1, 4, 5] [ 8,  9, 12, 13]
        const int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2); // [2, 3, 6, 7] [10, 11, 14, 15]
        // unpack nibbles into byte values
        const int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
        const int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
        const int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
        const int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
        // unshuffle values
        const int v0 = __byte_perm(s0, s1, 0x5410);
        const int v1 = __byte_perm(s0, s1, 0x7632);
        const int v2 = __byte_perm(s2, s3, 0x5410);
        const int v3 = __byte_perm(s2, s3, 0x7632);
#endif // defined(GGML_USE_HIP)

        sumi = ggml_cuda_dp4a(v0, u0, sumi);
        sumi = ggml_cuda_dp4a(v1, u1, sumi);
        sumi = ggml_cuda_dp4a(v2, u2, sumi);
        sumi = ggml_cuda_dp4a(v3, u3, sumi);
    }

    // Apply Q1_0's single scale and this chunk's Q8_1 scale
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d1 * d8 * sumi;
}

static __device__ __forceinline__ float vec_dot_q2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_0 * bq2_0 = (const block_q2_0 *) vbq + kbx;

    // Q2_0 (group 64): 64 elements with ONE scale, 2 bits per element (4 elements per byte)
    // Q8_1: 32 elements per block with individual scales
    // iqs selects which of the 2 chunks of 32 elements to process (0-1)

    const float     d2 = bq2_0->d;
    const int16_t * qs = (const int16_t *) bq2_0->qs + iqs * 4;

    // Process only the chunk specified by iqs
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q  = qs[j];
        const int u  = get_int_b4(bq8_1_chunk->qs, j*2+0);
        const int v  = get_int_b4(bq8_1_chunk->qs, j*2+1);

#if defined(GGML_USE_HIP)
        const uint32_t qx_indices = (q & 0x03) | ((q & 0x0C) << 6) | ((q & 0x30) << 12) | ((q & 0xC0) << 18);
        const uint32_t qy_bits    = q >> 8;
        const uint32_t qy_indices = (qy_bits & 0x03) | ((qy_bits & 0x0C) << 6) | ((qy_bits & 0x30) << 12) | ((qy_bits & 0xC0) << 18);
        const int      qx         = ggml_hip_perm(0x020100FF, 0x020100FF, qx_indices);
        const int      qy         = ggml_hip_perm(0x020100FF, 0x020100FF, qy_indices);
#else
        // unpack even and odd crumbs into byte values
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        // unshuffle values
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
#endif // defined(GGML_USE_HIP)

        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }

    // Apply Q2_0's single scale and this chunk's Q8_1 scale
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

// One 32-element chunk (iqs = 0..7) of a tq1_0 block against the matching q8_1 chunk.
// Nothing is unpacked and nothing is multiplied inside the block: every weight is a ternary
// symbol (see ggml_cuda_tq1_0_symbol in common.cuh), so each activation is added, subtracted or
// skipped, the accumulator stays an integer for the whole chunk and the two scales are applied
// once at the end. The chunk is handled by a single straight-line loop for all iqs, which keeps
// the warp from splitting into one code path per chunk (the (byte, digit) mapping is uniform).
// One 32-element chunk of a TQ1_0 block against one block_q8_1 of activations.
//
// Element -> (byte, trit position) inside a chunk, straight from the CPU codec:
//   iqs 0..4: qs[0..31],  trit position n = iqs for all 32 elements (8 words of 4 bytes)
//   iqs 5..6: qs[32..47], n = 2*(iqs-5) for elements 0..15 and n+1 for elements 16..31
//   iqs 7:    qs[32..47] at n = 4 for elements 0..15, then qh[0..3] for elements 16..31, where
//             the trit position IS the word index: element 16+4t+j = qh[j] at position t
// Weights are read with get_int_b2, not get_int_b4: block_tq1_0 is 54 bytes, so odd-numbered
// blocks are only 2-byte aligned and a 4-byte load would fault (same reason q4_0/q6_K use b2).
// Activations are block_q8_1 (36 bytes, qs at offset 4) and always 4-byte aligned -> get_int_b4.
// The multiply by the scales stays where it belongs: once, after the whole chunk.
static __device__ __forceinline__ float vec_dot_tq1_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_tq1_0 * bq = (const block_tq1_0 *) vbq + kbx;

    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    int sumi = 0;

    if (iqs < 5) {
        const int p3n = ggml_cuda_tq1_0_pow3(iqs);

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int s4 = ggml_cuda_tq1_0_symbols4(get_int_b2(bq->qs, j), p3n);
            sumi = ggml_cuda_dp4a(s4, get_int_b4(bq8_1_chunk->qs, j), sumi);
        }
    } else if (iqs < 7) {
        const int n0 = 2*(iqs - 5);
        const int p0 = ggml_cuda_tq1_0_pow3(n0);
        const int p1 = ggml_cuda_tq1_0_pow3(n0 + 1);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b4 = get_int_b2(bq->qs, 8 + j); // same 4 bytes serve both halves of the chunk
            sumi = ggml_cuda_dp4a(ggml_cuda_tq1_0_symbols4(b4, p0), get_int_b4(bq8_1_chunk->qs, j),     sumi);
            sumi = ggml_cuda_dp4a(ggml_cuda_tq1_0_symbols4(b4, p1), get_int_b4(bq8_1_chunk->qs, 4 + j), sumi);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int s4 = ggml_cuda_tq1_0_symbols4(get_int_b2(bq->qs, 8 + j), 81); // n = 4
            sumi = ggml_cuda_dp4a(s4, get_int_b4(bq8_1_chunk->qs, j), sumi);
        }

        const int qh4  = get_int_b2(bq->qh, 0);
        int       v_lo = __byte_perm(qh4, 0, 0x4140); // qh[0], qh[1] in 16-bit lanes
        int       v_hi = __byte_perm(qh4, 0, 0x4342); // qh[2], qh[3]

#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int w_lo = v_lo * 3;
            const int w_hi = v_hi * 3;
            v_lo = w_lo & 0x00FF00FF; // remainder carries to the next trit position
            v_hi = w_hi & 0x00FF00FF;

            const int s4 = __vsub4(__byte_perm(w_lo, w_hi, 0x7531), 0x01010101);
            sumi = ggml_cuda_dp4a(s4, get_int_b4(bq8_1_chunk->qs, 4 + t), sumi);
        }
    }

    const float d  = bq->d;
    const float d8 = __low2float(bq8_1_chunk->ds);

    return d * d8 * sumi;
}

// Strict variant, selected with -DGGML_CUDA_TQ1_0_NOMUL=ON: no integer multiply anywhere in the
// dot product. The symbol still comes from the shift/add chain in ggml_cuda_tq1_0_symbol, and the
// accumulate is branchless -- a per-element if/else diverges the warp on every weight that is not
// zero, which on a real model costs more than the multiplies it saves. About 14 instructions per
// element against ~4 for the dp4a path above.
static __device__ __forceinline__ float vec_dot_tq1_0_q8_1_nomul(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_tq1_0 * bq = (const block_tq1_0 *) vbq + kbx;

    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    int sumi = 0;

#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int s   = ggml_cuda_tq1_0_symbol(bq, 32*iqs + j); // {-1, 0, +1}
        const int a   = bq8_1_chunk->qs[j];
        const int neg = s >> 31;        // all ones where the weight is -1
        const int pos = (-s) >> 31;     // all ones where the weight is +1

        sumi += (a & pos) - (a & neg);  // +a, -a or nothing: add/subtract/skip, no branch
    }

    const float d  = bq->d;
    const float d8 = __low2float(bq8_1_chunk->ds);

    return d * d8 * sumi;
}

// One 32-element chunk (iqs = 0..7) of a tq2_0 block against one block_q8_1 of activations.
//
// With e = 128*c + 32*l + m (see ggml_cuda_tq2_0_digit in common.cuh) the chunk iqs covers
// c = iqs/4, l = iqs%4 and all 32 values of m, i.e. the 32 CONTIGUOUS bytes
// qs[32*(iqs/4) .. 32*(iqs/4)+31] at one single bit-pair position. Eight 4-byte words, one shift,
// one mask, one dp4a each: nothing is dequantized to float, the accumulator stays an integer for
// the whole chunk and the two scales are multiplied once, at the end of the chunk.
//
// Weight words are read with get_int_b2, never get_int_b4: block_tq2_0 is 66 bytes, so odd
// numbered blocks are only 2-byte aligned and a 4-byte load would fault (same reason as
// block_tq1_0 at 54 bytes). block_q8_1 is 36 bytes with qs at offset 4, so the activation words
// are 4-byte aligned and do use get_int_b4.
static __device__ __forceinline__ float vec_dot_tq2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_tq2_0 * bq = (const block_tq2_0 *) vbq + kbx;

    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    const int shift = 2*(iqs & 3);   // bit pair position l of the whole chunk
    const int woff  = 8*(iqs >> 2);  // get_int_b2 word index of the first of the 8 words

    int sumi = 0;

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int s4 = ggml_cuda_tq2_0_symbols4(get_int_b2(bq->qs, woff + j), shift);
        sumi = ggml_cuda_dp4a(s4, get_int_b4(bq8_1_chunk->qs, j), sumi);
    }

    const float d  = bq->d;
    const float d8 = __low2float(bq8_1_chunk->ds);

    return d * d8 * sumi;
}

// Strict variant, selected with -DGGML_CUDA_TQ2_0_NOMUL=ON: no integer multiply anywhere in the
// dot product, not even dp4a. The weight of one element is digit - 1 = 2*b1 + b0 - 1 with
// (b1, b0) the two stored bits, so its contribution is (b1 ? 2a : 0) + (b0 ? a : 0) - a: two
// masked adds, one shift and one subtract per element. The trailing -a is what keeps digit 3
// (weight +2) correct, so this is exact for all four levels too. Branchless on purpose - a
// per-element if/else diverges the warp on every non-zero weight, which on a real model costs more
// than the multiplies it saves.
static __device__ __forceinline__ float vec_dot_tq2_0_q8_1_nomul(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_tq2_0 * bq = (const block_tq2_0 *) vbq + kbx;

    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    const int shift = 2*(iqs & 3);
    const int woff  = 8*(iqs >> 2);

    int sumi = 0;

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int w4 = get_int_b2(bq->qs, woff + j) >> shift;

#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int dig = (w4 >> (8*t)) & 3;
            const int a   = bq8_1_chunk->qs[4*j + t];
            const int m1  = -(dig & 1);        // all ones where the low bit is set  -> +a
            const int m2  = -((dig >> 1) & 1); // all ones where the high bit is set -> +2a

            sumi += ((a & m2) << 1) + (a & m1) - a;
        }
    }

    const float d  = bq->d;
    const float d8 = __low2float(bq8_1_chunk->ds);

    return d * d8 * sumi;
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b2(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}


static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq + kbx;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b4(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq + kbx;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b2(bq5_0->qs, iqs + i);
        vh[i]    = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq + kbx;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b4(bq5_1->qs, iqs + i);
        vh[i]    = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    // iqs is in 0,2..14, each 32 bytes of qs hold 4 q8_1 blocks (one per 2 bit shift)
    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int iqs8 = iqs % QI8_1;

    int v[VDR_Q2_K_Q8_1_MMVQ];
#pragma unroll
    for (int j = 0; j < VDR_Q2_K_Q8_1_MMVQ; ++j) {
        v[j] = get_int_b4(bq2_K->qs, iqs + j);
    }

    // scale of shift i is byte 2*i + (iqs8 >= QI8_1/2) of the 8 bytes for this half
    const int sc_shift = 8 * (iqs8 / (QI8_1/2));
    int scales[2];
    scales[0] = get_int_b4(bq2_K->scales, 2*(iqs / QI8_1) + 0) >> sc_shift;
    scales[1] = get_int_b4(bq2_K->scales, 2*(iqs / QI8_1) + 1) >> sc_shift;

    int    u[VDR_Q2_K_Q8_1_MMVQ*QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q2_K_Q8_1_MMVQ; ++j) {
            u[VDR_Q2_K_Q8_1_MMVQ*i + j] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs8 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    // iqs is in 0,2..14, each 32 bytes of qs hold 4 q8_1 blocks (one per 2 bit shift)
    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int iqs8 = iqs % QI8_1;

    const float d = bq3_K->d;

    int vl[VDR_Q3_K_Q8_1_MMVQ];
    int vh[VDR_Q3_K_Q8_1_MMVQ];
#pragma unroll
    for (int j = 0; j < VDR_Q3_K_Q8_1_MMVQ; ++j) {
        vl[j] = get_int_b2(bq3_K->qs, iqs + j);
        vh[j] = get_int_b2(bq3_K->hmask, iqs8 + j) >> bq8_offset;
    }

    // sub-block 8*(iqs/8) + 2*i + (iqs8 >= QI8_1/2) for shift i:
    // 4 low bits in nibble iqs/8 of bytes, 2 high bits at bit 4*(iqs/8) + 2*(i/2) of bytes 8..11
    const int sc_shift = 8 * (iqs8 / (QI8_1/2)) + 4 * (iqs / (QI3_K/2));
    int scales[3];
#pragma unroll
    for (int l = 0; l < 3; ++l) {
        scales[l] = get_int_b2(bq3_K->scales, l) >> sc_shift;
    }

    int    u[VDR_Q3_K_Q8_1_MMVQ*QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q3_K_Q8_1_MMVQ; ++j) {
            u[VDR_Q3_K_Q8_1_MMVQ*i + j] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs8 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    // branchless so nvcc can hoist this out of the ncols_dst loop
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    // same as q4_K
    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));

    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    int vl[VDR_Q6_K_Q8_1_MMVQ];
    int vh[VDR_Q6_K_Q8_1_MMVQ];
#pragma unroll
    for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
        vl[j] = get_int_b2(bq6_K->ql, iqs + j);
        vh[j] = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4) + j) >> vh_shift;
    }

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[VDR_Q6_K_Q8_1_MMVQ*QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
#pragma unroll
        for (int j = 0; j < VDR_Q6_K_Q8_1_MMVQ; ++j) {
            u[VDR_Q6_K_Q8_1_MMVQ*i + j] = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1 + j);
        }
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

#define VDR_IQ2_XXS_Q8_1_MMVQ 2
#define VDR_IQ2_XXS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_XS_Q8_1_MMVQ 2
#define VDR_IQ2_XS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_S_Q8_1_MMVQ 2
#define VDR_IQ2_S_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 2
#define VDR_IQ3_XXS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_S_Q8_1_MMVQ 2
#define VDR_IQ3_S_Q8_1_MMQ  2

// TODO: don't use lookup table for signs
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ1_S_Q8_1_MMVQ 1
#define VDR_IQ1_S_Q8_1_MMQ  1

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq1->qh[iqs];

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0/2] | (((qh >> 3*(l0/2)) & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const float  d1q   = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
    const float2 ds    = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x*sumi + ds.y*delta);
}

#define VDR_IQ1_M_Q8_1_MMVQ 1
#define VDR_IQ1_M_Q8_1_MMQ  1

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq1_m * bq1 = (const block_iq1_m *) vbq + kbx;

    const int       qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    int   sumi[2] = {0};
    float sumf[2] = {0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2*iqs + l0/4] >> (4 * ((l0/2) % 2));

        const int grid = iq1s_grid_gpu[qs[l0/2] | ((qhl & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi[l0/4] = ggml_cuda_dp4a(grid0, u0, sumi[l0/4]);
        sumi[l0/4] = ggml_cuda_dp4a(grid1, u1, sumi[l0/4]);

        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0/4] += delta*sumy;
    }

    const uint16_t * sc = (const uint16_t *) bq1->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);

    const int tmp = sc[iqs/2] >> (6*(iqs%2));
    const int sc0 = 2*((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2*((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

#define VDR_IQ4_NL_Q8_1_MMVQ 2
#define VDR_IQ4_NL_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_IQ4_XS_Q8_1_MMVQ 4
#define VDR_IQ4_XS_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}
