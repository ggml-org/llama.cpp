//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TURBO_QUANTS_HPP
#define GGML_SYCL_TURBO_QUANTS_HPP

#include <sycl/sycl.hpp>
#include "common.hpp"
#include "turbo-wht.hpp"
#include "../ggml-turbo-wht-signs.h"

// ---- 2-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static const float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static const float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static const float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// ---- Midpoints for nearest centroid lookup ----

static const float TURBO_MID_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// ---- 4-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static const float TURBO_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

// ---- Midpoints for nearest 4-bit centroid lookup ----

static const float TURBO_MID_4BIT[15] = {
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.000000f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f
};

// ---- Nearest 2-bit centroid index ----

static inline uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < TURBO_MID_2BIT[0]) return 0;
    else if (val < TURBO_MID_2BIT[1]) return 1;
    else if (val < TURBO_MID_2BIT[2]) return 2;
    else                              return 3;
}

// ---- Nearest 3-bit centroid index ----

static inline uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MID_3BIT[0]) return 0;
    else if (val < TURBO_MID_3BIT[1]) return 1;
    else if (val < TURBO_MID_3BIT[2]) return 2;
    else if (val < TURBO_MID_3BIT[3]) return 3;
    else if (val < TURBO_MID_3BIT[4]) return 4;
    else if (val < TURBO_MID_3BIT[5]) return 5;
    else if (val < TURBO_MID_3BIT[6]) return 6;
    else                              return 7;
}

// ---- Nearest 4-bit centroid index ----

static inline uint8_t turbo_nearest_centroid_4bit(float val) {
    if      (val < TURBO_MID_4BIT[ 0]) return  0;
    else if (val < TURBO_MID_4BIT[ 1]) return  1;
    else if (val < TURBO_MID_4BIT[ 2]) return  2;
    else if (val < TURBO_MID_4BIT[ 3]) return  3;
    else if (val < TURBO_MID_4BIT[ 4]) return  4;
    else if (val < TURBO_MID_4BIT[ 5]) return  5;
    else if (val < TURBO_MID_4BIT[ 6]) return  6;
    else if (val < TURBO_MID_4BIT[ 7]) return  7;
    else if (val < TURBO_MID_4BIT[ 8]) return  8;
    else if (val < TURBO_MID_4BIT[ 9]) return  9;
    else if (val < TURBO_MID_4BIT[10]) return 10;
    else if (val < TURBO_MID_4BIT[11]) return 11;
    else if (val < TURBO_MID_4BIT[12]) return 12;
    else if (val < TURBO_MID_4BIT[13]) return 13;
    else if (val < TURBO_MID_4BIT[14]) return 14;
    else                               return 15;
}

// ---- TurboQuant Device Functions ----

template <int DIM = 1>
static __dpct_inline__ void quantize_turbo2_0(float val, block_turbo2_0 * dst, const sycl::nd_item<DIM> & item_ct1) {
    auto sg = item_ct1.get_sub_group();
    const int lane = sg.get_local_id()[0];
    const int tid = item_ct1.get_local_id(DIM - 1);
    const int elem_in_block = tid % QK_TURBO2;
    
    const uint8_t idx = turbo_nearest_centroid_2bit(val);
    
    // Pack qs: 4 elements per byte, 2 bits each.
    const int qs_byte_idx = elem_in_block / 4;
    
    uint8_t qs_byte = 0;
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = sycl::select_from_group(sg, idx, (lane & ~3) + k);
        qs_byte |= (contrib & 0x3) << (k * 2);
    }
    if (lane % 4 == 0) dst->qs[qs_byte_idx] = qs_byte;
}

template <int DIM = 1>
static __dpct_inline__ void quantize_turbo3_0(float val, block_turbo3_0 * dst, const sycl::nd_item<DIM> & item_ct1) {
    auto sg = item_ct1.get_sub_group();
    const int lane = sg.get_local_id()[0];
    const int tid = item_ct1.get_local_id(DIM - 1);
    const int elem_in_block = tid % QK_TURBO3;
    
    const uint8_t idx = turbo_nearest_centroid_3bit(val);
    
    // Pack qs: 4 elements per byte, 2 bits each.
    const uint8_t my_low2 = idx & 0x3;
    const int qs_byte_idx = elem_in_block / 4;
    
    uint8_t qs_byte = 0;
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = sycl::select_from_group(sg, my_low2, (lane & ~3) + k);
        qs_byte |= (contrib & 0x3) << (k * 2);
    }
    if (lane % 4 == 0) dst->qs[qs_byte_idx] = qs_byte;

    // Pack signs: 8 elements per byte, 1 bit each.
    uint32_t sign_bit = (idx >> 2) & 1;
    uint8_t signs_byte = 0;
    for (int k = 0; k < 8; k++) {
        uint8_t contrib = sycl::select_from_group(sg, (uint8_t)sign_bit, (lane & ~7) + k);
        signs_byte |= (contrib & 0x1) << k;
    }
    if (lane % 8 == 0) dst->signs[elem_in_block / 8] = signs_byte;
}

template <int DIM = 1>
static __dpct_inline__ void quantize_turbo4_0(float val, block_turbo4_0 * dst, const sycl::nd_item<DIM> & item_ct1) {
    auto sg = item_ct1.get_sub_group();
    const int lane = sg.get_local_id()[0];
    const int tid = item_ct1.get_local_id(DIM - 1);
    const int elem_in_block = tid % QK_TURBO4;
    
    const uint8_t idx = turbo_nearest_centroid_4bit(val);
    
    // Pack qs: 2 elements per byte, 4 bits each.
    const int qs_byte_idx = elem_in_block / 2;
    
    uint8_t qs_byte = 0;
    for (int k = 0; k < 2; k++) {
        uint8_t contrib = sycl::select_from_group(sg, idx, (lane & ~1) + k);
        qs_byte |= (contrib & 0xF) << (k * 4);
    }
    if (lane % 2 == 0) dst->qs[qs_byte_idx] = qs_byte;
}

static __dpct_inline__ float dequantize_turbo3_0(const block_turbo3_0 * x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT[idx] * norm;
}

static __dpct_inline__ float dequantize_turbo2_0(const block_turbo2_0 * x, int j, float norm) {
    uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}

static __dpct_inline__ float dequantize_turbo4_0(const block_turbo4_0 * x, int j, float norm) {
    uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}

// ---- TQ3_1S / TQ4_1S Weight centroids (N(0,1)) ----

static const float TQ_CENTROIDS_3BIT[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

static const float TQ_CENTROIDS_4BIT[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

static const float TQ_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f
};

// ---- TQ Weight Dequantization Helpers ----

static __dpct_inline__ float dequantize_tq4_1s(const block_tq4_1s * x, int j) {
    const float d = (j < 16) ? (float)x->d0 : (float)x->d1;
    const uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TQ_CENTROIDS_4BIT[idx] * d;
}

static __dpct_inline__ float dequantize_tq3_1s(const block_tq3_1s * x, int j) {
    const float d = (j < 16) ? (float)x->d0 : (float)x->d1;
    
    // TQ3 packing: 3 bits per element, 8 elements in 3 bytes.
    // 0: [0] low 3
    // 1: [0] next 3
    // 2: [0] bit 6-7, [1] bit 0
    // Actually using Metal/CUDA reference packing:
    const int ig = j / 8; // group of 8
    const int i  = j % 8; // index in group
    const uint8_t * qs = x->qs + 3*ig;
    
    uint8_t idx = 0;
    switch (i) {
        case 0: idx =  qs[0]        & 7; break;
        case 1: idx = (qs[0] >> 3)  & 7; break;
        case 2: idx = (qs[0] >> 6)  | ((qs[1] << 2) & 7); break;
        case 3: idx = (qs[1] >> 1)  & 7; break;
        case 4: idx = (qs[1] >> 4)  & 7; break;
        case 5: idx = (qs[1] >> 7)  | ((qs[2] << 1) & 7); break;
        case 6: idx = (qs[2] >> 2)  & 7; break;
        case 7: idx = (qs[2] >> 5)  & 7; break;
    }
    return TQ_CENTROIDS_3BIT[idx] * d;
}

#endif // GGML_SYCL_TURBO_QUANTS_HPP
