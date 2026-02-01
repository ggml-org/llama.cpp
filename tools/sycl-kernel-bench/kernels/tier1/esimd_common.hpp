#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-bench.hpp"
#include "ggml-sycl/vecdotq.hpp"

#if __has_include(<sycl/ext/intel/esimd.hpp>)
#    define SYCL_ESIMD_TIER1_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
namespace esimd = sycl::ext::intel::esimd;
#else
#    define SYCL_ESIMD_TIER1_AVAILABLE 0
#endif

namespace sycl_bench {
namespace mmvq_tier1_esimd {

SYCL_ESIMD_FUNCTION inline float esimd_e8m0_to_fp32(uint8_t x) {
    uint32_t bits = (x == 0) ? 0x00400000u : (static_cast<uint32_t>(x) << 23);
    union {
        uint32_t u;
        float    f;
    } conv;
    conv.u = bits;
    return conv.f;
}

SYCL_ESIMD_FUNCTION inline float half_to_float(ggml_half v) {
    return static_cast<float>(v);
}

struct q8_1_block_local {
    ggml_half ds[2];
    int8_t    qs[QK8_1];
};

SYCL_ESIMD_FUNCTION inline void load_half2(const ggml_half2 * src, float & a, float & b) {
    const ggml_half * vals = reinterpret_cast<const ggml_half *>(src);
    a = half_to_float(vals[0]);
    b = half_to_float(vals[1]);
}

SYCL_ESIMD_FUNCTION inline void load_q8_1_scales(const q8_1_block_local * src, float & d, float & s) {
    d = half_to_float(src->ds[0]);
    s = half_to_float(src->ds[1]);
}

SYCL_ESIMD_FUNCTION inline float load_q8_1_d(const q8_1_block_local * src) {
    float d = 0.0f;
    float s = 0.0f;
    load_q8_1_scales(src, d, s);
    return d;
}

SYCL_ESIMD_FUNCTION inline int dot4(int v, int u) {
    union {
        int    i;
        int8_t b[4];
    } vv, uu;
    vv.i = v;
    uu.i = u;
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += static_cast<int>(vv.b[i]) * static_cast<int>(uu.b[i]);
    }
    return sum;
}

SYCL_ESIMD_FUNCTION inline int sum_bytes(int u) {
    union {
        int    i;
        int8_t b[4];
    } uu;
    uu.i = u;
    return static_cast<int>(uu.b[0]) + static_cast<int>(uu.b[1]) +
           static_cast<int>(uu.b[2]) + static_cast<int>(uu.b[3]);
}

SYCL_ESIMD_FUNCTION inline int pack_int8(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    return static_cast<int>(static_cast<uint8_t>(b0)) |
           (static_cast<int>(static_cast<uint8_t>(b1)) << 8) |
           (static_cast<int>(static_cast<uint8_t>(b2)) << 16) |
           (static_cast<int>(static_cast<uint8_t>(b3)) << 24);
}

SYCL_ESIMD_FUNCTION inline float vec_dot_q4_0_q8_1_esimd(const block_q4_0 * bq4_0,
                                                         const q8_1_block_local * bq8_1,
                                                         int iqs) {
    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2 * VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi += dot4(vi0, u[2 * i + 0]);
        sumi += dot4(vi1, u[2 * i + 1]);
    }

    float d8 = 0.0f;
    float s8 = 0.0f;
    load_q8_1_scales(bq8_1, d8, s8);
    const float d4 = half_to_float(bq4_0->d);
    const float zero_corr = static_cast<float>(8 * VDR_Q4_0_Q8_1_MMVQ) / static_cast<float>(QI4_0);
    return d4 * (static_cast<float>(sumi) * d8 - zero_corr * s8);
}

SYCL_ESIMD_FUNCTION inline float vec_dot_q8_0_q8_1_esimd(const block_q8_0 * bq8_0,
                                                         const q8_1_block_local * bq8_1,
                                                         int iqs) {
    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        sumi += dot4(v[i], u[i]);
    }

    const float d8_0 = half_to_float(bq8_0->d);
    const float d8_1 = load_q8_1_d(bq8_1);
    return d8_0 * d8_1 * static_cast<float>(sumi);
}

SYCL_ESIMD_FUNCTION inline float vec_dot_mxfp4_q8_1_esimd(const block_mxfp4 * bq4,
                                                          const q8_1_block_local * bq8_1,
                                                          int iqs) {
    const int start = iqs * 4;
    const int end = start + 8;
    int sumi = 0;

    for (int j = start; j < end; ++j) {
        const uint8_t packed = bq4->qs[j];
        const int8_t  w0 = kvalues_mxfp4[packed & 0x0F];
        const int8_t  w1 = kvalues_mxfp4[packed >> 4];
        sumi += static_cast<int>(bq8_1->qs[j]) * static_cast<int>(w0);
        sumi += static_cast<int>(bq8_1->qs[j + QK_MXFP4 / 2]) * static_cast<int>(w1);
    }

    const float d8 = load_q8_1_d(bq8_1);
    const float d = esimd_e8m0_to_fp32(bq4->e) * 0.5f * d8;
    return d * static_cast<float>(sumi);
}

SYCL_ESIMD_FUNCTION inline float vec_dot_q2_K_q8_1_esimd(const block_q2_K * bq2_K,
                                                         const q8_1_block_local * bq8_1,
                                                         int iqs) {
    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

    const uint8_t * scales = bq2_K->scales + scale_offset;
    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);

    int   u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const q8_1_block_local * bq8i = bq8_1 + bq8_offset + i;
        u[i] = get_int_from_int8_aligned(bq8i->qs, iqs % QI8_1);
        d8[i] = load_q8_1_d(bq8i);
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        const int m = sc >> 4;

        sumf_d += d8[i] * static_cast<float>(dot4(vi, u[i]) * (sc & 0xF));
        sumf_m += d8[i] * static_cast<float>(sum_bytes(u[i]) * m);
    }

    float dm_d = 0.0f;
    float dm_m = 0.0f;
    load_half2(&bq2_K->dm, dm_d, dm_m);
    return dm_d * sumf_d - dm_m * sumf_m;
}

SYCL_ESIMD_FUNCTION inline float vec_dot_q3_K_q8_1_esimd(const block_q3_K * bq3_K,
                                                         const q8_1_block_local * bq8_1,
                                                         int iqs) {
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

    const float d3 = half_to_float(bq3_K->d);
    const int vl = get_int_from_uint8(bq3_K->qs, iqs);
    const int vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;

    int   u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const q8_1_block_local * bq8i = bq8_1 + bq8_offset + i;
        u[i] = get_int_from_int8_aligned(bq8i->qs, iqs % QI8_1);
        d8[i] = load_q8_1_d(bq8i);
    }

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % (QK_K / 32);
        const int sc_shift_low = 4 * (isc / (QK_K / 32));
        const int sc_low = (bq3_K->scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high = ((bq3_K->scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;

        int8_t vb[4];
        int8_t hb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            vb[j] = static_cast<int8_t>((vil >> (8 * j)) & 0xFF);
            hb[j] = static_cast<int8_t>((vih >> (8 * j)) & 0xFF);
        }

        const int vi = pack_int8(static_cast<int8_t>(vb[0] - hb[0]),
                                 static_cast<int8_t>(vb[1] - hb[1]),
                                 static_cast<int8_t>(vb[2] - hb[2]),
                                 static_cast<int8_t>(vb[3] - hb[3]));
        sumf += d8[i] * static_cast<float>(dot4(vi, u[i]) * sc);
    }

    return d3 * sumf;
}

SYCL_ESIMD_FUNCTION inline float vec_dot_q4_K_q8_1_esimd(const block_q4_K * bq4_K,
                                                         const q8_1_block_local * bq8_1,
                                                         int iqs) {
#ifndef GGML_QKK_64
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int * q4 = reinterpret_cast<const int *>(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const uint16_t * scales = reinterpret_cast<const uint16_t *>(bq4_K->scales);

    const int v0 = q4[0];
    const int v1 = q4[4];

    uint16_t aux[2];
    const int j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    const uint8_t * sc = reinterpret_cast<const uint8_t *>(aux);
    const uint8_t * m = sc + 2;

    int   u[2 * QR4_K];
    float d8[QR4_K];

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const q8_1_block_local * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = load_q8_1_d(bq8i);

        const int * q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = dot4(v0i, u[2 * i + 0]) + dot4(v1i, u[2 * i + 1]);
        const int dot2 = sum_bytes(u[2 * i + 0]) + sum_bytes(u[2 * i + 1]);

        sumf_d += d8[i] * static_cast<float>(dot1 * sc[i]);
        sumf_m += d8[i] * static_cast<float>(dot2 * m[i]);
    }

    float dm_d = 0.0f;
    float dm_m = 0.0f;
    load_half2(&bq4_K->dm, dm_d, dm_m);
    return dm_d * sumf_d - dm_m * sumf_m;
#else
    (void)bq4_K;
    (void)bq8_1;
    (void)iqs;
    return 0.0f;
#endif
}

SYCL_ESIMD_FUNCTION inline float vec_dot_q5_K_q8_1_esimd(const block_q5_K * bq5_K,
                                                         const q8_1_block_local * bq8_1,
                                                         int iqs) {
#ifndef GGML_QKK_64
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int * ql = reinterpret_cast<const int *>(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int * qh = reinterpret_cast<const int *>(bq5_K->qh + 4 * ((iqs / 2) % 4));

    int vl[2];
    int vh[2];
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = reinterpret_cast<const uint16_t *>(bq5_K->scales);
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = reinterpret_cast<const uint8_t *>(aux);
    const uint8_t * m = sc + 2;

    int   u[2 * QR5_K];
    float d8[QR5_K];

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const q8_1_block_local * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = load_q8_1_d(bq8i);

        const int * q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = dot4(v0i, u[2 * i + 0]) + dot4(v1i, u[2 * i + 1]);
        const int dot2 = sum_bytes(u[2 * i + 0]) + sum_bytes(u[2 * i + 1]);

        sumf_d += d8[i] * static_cast<float>(dot1 * sc[i]);
        sumf_m += d8[i] * static_cast<float>(dot2 * m[i]);
    }

    float dm_d = 0.0f;
    float dm_m = 0.0f;
    load_half2(&bq5_K->dm, dm_d, dm_m);
    return dm_d * sumf_d - dm_m * sumf_m;
#else
    (void)bq5_K;
    (void)bq8_1;
    (void)iqs;
    return 0.0f;
#endif
}

SYCL_ESIMD_FUNCTION inline float vec_dot_q6_K_q8_1_esimd(const block_q6_K * bq6_K,
                                                         const q8_1_block_local * bq8_1,
                                                         int iqs) {
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int   u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const q8_1_block_local * bq8i = bq8_1 + bq8_offset + 2 * i;
        u[i] = get_int_from_int8_aligned(bq8i->qs, iqs % QI8_1);
        d8[i] = load_q8_1_d(bq8i);
    }

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vraw = vil | vih;

        int8_t vb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            vb[j] = static_cast<int8_t>(((vraw >> (8 * j)) & 0xFF) - 32);
        }
        const int vi = pack_int8(vb[0], vb[1], vb[2], vb[3]);

        sumf += d8[i] * static_cast<float>(dot4(vi, u[i]) * sc);
    }

    const float d = half_to_float(bq6_K->d);
    return d * sumf;
}

template <int qtype>
struct QuantTraits;

template <>
struct QuantTraits<GGML_TYPE_Q4_0> {
    using block_t = block_q4_0;
    static constexpr int qk = QK4_0;
    static constexpr int qi = QI4_0;
    static constexpr int vdr = VDR_Q4_0_Q8_1_MMVQ;
    static constexpr int blocks_per_y = qk / QK8_1;
    SYCL_ESIMD_FUNCTION static float vec_dot(const block_t * x, const q8_1_block_local * y, int iqs) {
        return vec_dot_q4_0_q8_1_esimd(x, y, iqs);
    }
};

template <>
struct QuantTraits<GGML_TYPE_Q8_0> {
    using block_t = block_q8_0;
    static constexpr int qk = QK8_0;
    static constexpr int qi = QI8_0;
    static constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    static constexpr int blocks_per_y = qk / QK8_1;
    SYCL_ESIMD_FUNCTION static float vec_dot(const block_t * x, const q8_1_block_local * y, int iqs) {
        return vec_dot_q8_0_q8_1_esimd(x, y, iqs);
    }
};

template <>
struct QuantTraits<GGML_TYPE_MXFP4> {
    using block_t = block_mxfp4;
    static constexpr int qk = QK_MXFP4;
    static constexpr int qi = QI_MXFP4;
    static constexpr int vdr = VDR_MXFP4_Q8_1_MMVQ;
    static constexpr int blocks_per_y = qk / QK8_1;
    SYCL_ESIMD_FUNCTION static float vec_dot(const block_t * x, const q8_1_block_local * y, int iqs) {
        return vec_dot_mxfp4_q8_1_esimd(x, y, iqs);
    }
};

template <>
struct QuantTraits<GGML_TYPE_Q2_K> {
    using block_t = block_q2_K;
    static constexpr int qk = QK_K;
    static constexpr int qi = QI2_K;
    static constexpr int vdr = VDR_Q2_K_Q8_1_MMVQ;
    static constexpr int blocks_per_y = qk / QK8_1;
    SYCL_ESIMD_FUNCTION static float vec_dot(const block_t * x, const q8_1_block_local * y, int iqs) {
        return vec_dot_q2_K_q8_1_esimd(x, y, iqs);
    }
};

template <>
struct QuantTraits<GGML_TYPE_Q3_K> {
    using block_t = block_q3_K;
    static constexpr int qk = QK_K;
    static constexpr int qi = QI3_K;
    static constexpr int vdr = VDR_Q3_K_Q8_1_MMVQ;
    static constexpr int blocks_per_y = qk / QK8_1;
    SYCL_ESIMD_FUNCTION static float vec_dot(const block_t * x, const q8_1_block_local * y, int iqs) {
        return vec_dot_q3_K_q8_1_esimd(x, y, iqs);
    }
};

template <>
struct QuantTraits<GGML_TYPE_Q4_K> {
    using block_t = block_q4_K;
    static constexpr int qk = QK_K;
    static constexpr int qi = QI4_K;
    static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    static constexpr int blocks_per_y = qk / QK8_1;
    SYCL_ESIMD_FUNCTION static float vec_dot(const block_t * x, const q8_1_block_local * y, int iqs) {
        return vec_dot_q4_K_q8_1_esimd(x, y, iqs);
    }
};

template <>
struct QuantTraits<GGML_TYPE_Q5_K> {
    using block_t = block_q5_K;
    static constexpr int qk = QK_K;
    static constexpr int qi = QI5_K;
    static constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;
    static constexpr int blocks_per_y = qk / QK8_1;
    SYCL_ESIMD_FUNCTION static float vec_dot(const block_t * x, const q8_1_block_local * y, int iqs) {
        return vec_dot_q5_K_q8_1_esimd(x, y, iqs);
    }
};

template <>
struct QuantTraits<GGML_TYPE_Q6_K> {
    using block_t = block_q6_K;
    static constexpr int qk = QK_K;
    static constexpr int qi = QI6_K;
    static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    static constexpr int blocks_per_y = qk / QK8_1;
    SYCL_ESIMD_FUNCTION static float vec_dot(const block_t * x, const q8_1_block_local * y, int iqs) {
        return vec_dot_q6_K_q8_1_esimd(x, y, iqs);
    }
};

#if SYCL_ESIMD_TIER1_AVAILABLE

template <int N>
SYCL_ESIMD_FUNCTION inline esimd::simd<int8_t, N> load_int8_block(const int8_t * ptr) {
    if ((reinterpret_cast<uintptr_t>(ptr) & 0x3u) == 0u) {
        return esimd::block_load<int8_t, N>(ptr, esimd::properties{esimd::alignment<4>});
    }
    esimd::simd<int8_t, N> tmp;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        tmp[i] = ptr[i];
    }
    return tmp;
}

SYCL_ESIMD_FUNCTION inline q8_1_block_local load_q8_1_block(const block_q8_1 * src) {
    q8_1_block_local out{};
    const ggml_half * ds_ptr = reinterpret_cast<const ggml_half *>(&src->ds);
    out.ds[0] = ds_ptr[0];
    out.ds[1] = ds_ptr[1];
    auto qs = load_int8_block<QK8_1>(src->qs);
#pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        out.qs[i] = qs[i];
    }
    return out;
}

template <int Blocks>
SYCL_ESIMD_FUNCTION inline void load_q8_1_blocks(const block_q8_1 * src, q8_1_block_local (&dst)[Blocks]) {
#pragma unroll
    for (int i = 0; i < Blocks; ++i) {
        dst[i] = load_q8_1_block(src + i);
    }
}

template <int Blocks>
struct Q8SlmLayout {
    static constexpr int qs_stride = QK8_1;
    static constexpr int qs_bytes = Blocks * qs_stride;
    static constexpr int ds_offset = (qs_bytes + 3) & ~3;
    static constexpr int ds_bytes = Blocks * 2 * static_cast<int>(sizeof(ggml_half));
    static constexpr int slm_bytes = ds_offset + ds_bytes;
};

template <int Blocks>
SYCL_ESIMD_FUNCTION inline void load_q8_1_blocks_slm(const block_q8_1 * src, q8_1_block_local (&dst)[Blocks]) {
    constexpr int qs_stride = Q8SlmLayout<Blocks>::qs_stride;
    constexpr int ds_offset = Q8SlmLayout<Blocks>::ds_offset;

#pragma unroll
    for (int i = 0; i < Blocks; ++i) {
        const block_q8_1 * cur = src + i;
        const ggml_half * ds_ptr = reinterpret_cast<const ggml_half *>(&cur->ds);
        auto qs = load_int8_block<QK8_1>(cur->qs);
        esimd::slm_block_store<int8_t, QK8_1>(i * qs_stride, qs);
        esimd::simd<ggml_half, 2> ds(ds_ptr[0], ds_ptr[1]);
        esimd::slm_block_store<ggml_half, 2>(ds_offset + i * 2 * sizeof(ggml_half), ds);
    }

    esimd::barrier();

#pragma unroll
    for (int i = 0; i < Blocks; ++i) {
        auto qs = esimd::slm_block_load<int8_t, QK8_1>(i * qs_stride);
        auto ds = esimd::slm_block_load<ggml_half, 2>(ds_offset + i * 2 * sizeof(ggml_half));
        dst[i].ds[0] = ds[0];
        dst[i].ds[1] = ds[1];
#pragma unroll
        for (int j = 0; j < QK8_1; ++j) {
            dst[i].qs[j] = qs[j];
        }
    }
}

template <int qtype>
SYCL_ESIMD_FUNCTION inline float dot_block(const typename QuantTraits<qtype>::block_t * x,
                                           const q8_1_block_local * y_blocks) {
    float acc = 0.0f;
    constexpr int qi = QuantTraits<qtype>::qi;
    constexpr int vdr = QuantTraits<qtype>::vdr;
#pragma unroll
    for (int iqs = 0; iqs < qi; iqs += vdr) {
        acc += QuantTraits<qtype>::vec_dot(x, y_blocks, iqs);
    }
    return acc;
}

template <int qtype>
SYCL_ESIMD_FUNCTION inline float dot_row_block_load(const typename QuantTraits<qtype>::block_t * x_row,
                                                    const block_q8_1 * y_row,
                                                    int blocks_per_row) {
    float acc = 0.0f;
    constexpr int blocks_per_y = QuantTraits<qtype>::blocks_per_y;
    q8_1_block_local y_local[blocks_per_y];

    for (int blk = 0; blk < blocks_per_row; ++blk) {
        const block_q8_1 * y_block = y_row + blk * blocks_per_y;
        load_q8_1_blocks<blocks_per_y>(y_block, y_local);
        acc += dot_block<qtype>(x_row + blk, y_local);
    }
    return acc;
}

template <int qtype>
SYCL_ESIMD_FUNCTION inline float dot_row_slm(const typename QuantTraits<qtype>::block_t * x_row,
                                             const block_q8_1 * y_row,
                                             int blocks_per_row) {
    float acc = 0.0f;
    constexpr int blocks_per_y = QuantTraits<qtype>::blocks_per_y;
    q8_1_block_local y_local[blocks_per_y];

    for (int blk = 0; blk < blocks_per_row; ++blk) {
        const block_q8_1 * y_block = y_row + blk * blocks_per_y;
        load_q8_1_blocks_slm<blocks_per_y>(y_block, y_local);
        acc += dot_block<qtype>(x_row + blk, y_local);
    }
    return acc;
}

template <int qtype, bool UseSlm>
class esimd_kernel_name;

template <int qtype, bool UseSlm>
inline sycl::event launch_esimd_kernel(const void * weights,
                                       const void * activations,
                                       float * dst,
                                       int64_t ncols,
                                       int64_t nrows,
                                       sycl::queue & queue) {
    using Traits = QuantTraits<qtype>;
    const int blocks_per_row = static_cast<int>(ncols / Traits::qk);
    const auto * x_base = static_cast<const typename Traits::block_t *>(weights);
    const auto * y_base = static_cast<const block_q8_1 *>(activations);

    sycl::range<1> global(static_cast<size_t>(nrows));
    sycl::range<1> local(1);

    return queue.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<esimd_kernel_name<qtype, UseSlm>>(sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                if constexpr (UseSlm) {
                    constexpr int slm_bytes = Q8SlmLayout<Traits::blocks_per_y>::slm_bytes;
                    esimd::slm_init<slm_bytes>();
                }

                const int row = static_cast<int>(item.get_global_id(0));
                if (row >= nrows) {
                    return;
                }

                const auto * x_row = x_base + row * blocks_per_row;
                const auto * y_row = y_base;
                float acc = 0.0f;

                if constexpr (UseSlm) {
                    acc = dot_row_slm<qtype>(x_row, y_row, blocks_per_row);
                } else {
                    acc = dot_row_block_load<qtype>(x_row, y_row, blocks_per_row);
                }

                dst[row] = acc;
            });
    });
}

#endif  // SYCL_ESIMD_TIER1_AVAILABLE

inline bool validate_esimd_args(const ggml_sycl::mmvq_bench_args & args, std::string & error) {
    if (!args.stream || !args.weights || !args.activations || !args.output) {
        error = "Invalid null pointers for ESIMD MMVQ args.";
        return false;
    }
    if (args.ncols <= 0 || args.nrows <= 0 || args.batch <= 0) {
        error = "Invalid MMVQ dimensions.";
        return false;
    }
    if (args.ncols % QK8_1 != 0) {
        error = "MMVQ requires ncols multiple of QK8_1.";
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "ESIMD Tier1 kernels require AOS layout.";
        return false;
    }
    return true;
}

inline bool run_esimd_variant(const ggml_sycl::mmvq_bench_args & args,
                              std::vector<sycl::event> * events,
                              std::string & error,
                              bool use_slm) {
#if !SYCL_ESIMD_TIER1_AVAILABLE
    (void)args;
    (void)events;
    (void)use_slm;
    error = "SYCL ESIMD unavailable; Tier1 ESIMD kernels disabled.";
    return false;
#else
    if (!validate_esimd_args(args, error)) {
        return false;
    }

    sycl::queue & queue = *args.stream;

    const int64_t row_low = args.row_low;
    const int64_t row_high = args.row_high;
    const int64_t nrows = row_high - row_low;

    const size_t row_bytes = ggml_row_size(args.weight_type, args.ncols);
    const char * weight_base = static_cast<const char *>(args.weights);
    const char * weight_ptr = weight_base + row_low * row_bytes;

    float * output_base = args.output + row_low;

    const size_t src1_row_bytes =
        static_cast<size_t>(args.src1_padded_col_size) * sizeof(block_q8_1) / QK8_1;

    auto record_event = [&](sycl::event evt) {
        if (events) {
            events->push_back(std::move(evt));
        }
    };

    for (int64_t i = 0; i < args.batch; ++i) {
        const char * y_ptr = static_cast<const char *>(args.activations) + i * src1_row_bytes;
        float * dst_ptr = output_base + i * args.dst_row_stride;

        switch (args.weight_type) {
            case GGML_TYPE_Q4_0:
                record_event(use_slm
                    ? launch_esimd_kernel<GGML_TYPE_Q4_0, true>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue)
                    : launch_esimd_kernel<GGML_TYPE_Q4_0, false>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue));
                break;
            case GGML_TYPE_Q8_0:
                record_event(use_slm
                    ? launch_esimd_kernel<GGML_TYPE_Q8_0, true>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue)
                    : launch_esimd_kernel<GGML_TYPE_Q8_0, false>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue));
                break;
            case GGML_TYPE_MXFP4:
                record_event(use_slm
                    ? launch_esimd_kernel<GGML_TYPE_MXFP4, true>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue)
                    : launch_esimd_kernel<GGML_TYPE_MXFP4, false>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue));
                break;
            case GGML_TYPE_Q2_K:
                record_event(use_slm
                    ? launch_esimd_kernel<GGML_TYPE_Q2_K, true>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue)
                    : launch_esimd_kernel<GGML_TYPE_Q2_K, false>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue));
                break;
            case GGML_TYPE_Q3_K:
                record_event(use_slm
                    ? launch_esimd_kernel<GGML_TYPE_Q3_K, true>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue)
                    : launch_esimd_kernel<GGML_TYPE_Q3_K, false>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue));
                break;
            case GGML_TYPE_Q4_K:
                record_event(use_slm
                    ? launch_esimd_kernel<GGML_TYPE_Q4_K, true>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue)
                    : launch_esimd_kernel<GGML_TYPE_Q4_K, false>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue));
                break;
            case GGML_TYPE_Q5_K:
                record_event(use_slm
                    ? launch_esimd_kernel<GGML_TYPE_Q5_K, true>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue)
                    : launch_esimd_kernel<GGML_TYPE_Q5_K, false>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue));
                break;
            case GGML_TYPE_Q6_K:
                record_event(use_slm
                    ? launch_esimd_kernel<GGML_TYPE_Q6_K, true>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue)
                    : launch_esimd_kernel<GGML_TYPE_Q6_K, false>(weight_ptr, y_ptr, dst_ptr, args.ncols, nrows, queue));
                break;
            default:
                error = "Unsupported quant type for Tier1 ESIMD variant.";
                return false;
        }
    }

    return true;
#endif
}

}  // namespace mmvq_tier1_esimd
}  // namespace sycl_bench
