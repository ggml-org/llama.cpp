#pragma once

// MXFP4 dequantization using v_perm_b32 for 8-entry table lookup
// Unaligned memory loads via memcpy (compiler optimizes to flat_load)

#include "../gfx906-config.h"

#if defined(GGML_USE_HIP) && defined(__gfx906__)

static __device__ __forceinline__ int gfx906_get_int_b1_fast(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;
    int x32;
    memcpy(&x32, x8 + 4*i32, 4);
    return x32;
}

static __device__ __forceinline__ int gfx906_get_int_b2_fast(const void * x, const int & i32) {
    int x32;
    memcpy(&x32, (const uint8_t*)x + 4*i32, 4);
    return x32;
}

// 64-bit vectorized load - emits global_load_dwordx2 when aligned
// Used for double-throughput memory access in MMQ tile loading
static __device__ __forceinline__ int2 gfx906_load_int2(const void * x, const int & i32) {
    int2 x64;
    memcpy(&x64, (const uint8_t*)x + 4*i32, 8);
    return x64;
}

__constant__ uint8_t gfx906_mxfp4_magnitudes[8] = { 0, 1, 2, 3, 4, 6, 8, 12 };

static __device__ __forceinline__ int2 gfx906_get_int_from_mxfp4_table(const uint32_t q4) {
    const uint32_t *mags32 = (const uint32_t *)gfx906_mxfp4_magnitudes;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    uint32_t sign_even = (q_even >> 3) & 0x01010101;
    uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    const uint32_t sel_even = q_even & 0x07070707;
    const uint32_t sel_odd  = q_odd  & 0x07070707;

    uint32_t mag_even = __builtin_amdgcn_perm(mags32[1], mags32[0], sel_even);
    uint32_t mag_odd  = __builtin_amdgcn_perm(mags32[1], mags32[0], sel_odd);

    const uint32_t mask_even = sign_even * 0xFFu;
    const uint32_t mask_odd  = sign_odd  * 0xFFu;

    uint32_t res_x = (mag_even ^ mask_even) + sign_even;
    uint32_t res_y = (mag_odd  ^ mask_odd)  + sign_odd;

    return make_int2(res_x, res_y);
}

#define GFX906_VEC_DOT_MXFP4_Q8_1(bq4, bq8_1, iqs, sumi) \
    do { \
        const int * q8 = (const int *) bq8_1->qs + iqs; \
        const int aux_q4_0 = gfx906_get_int_b1_fast(bq4->qs, iqs + 0); \
        const int aux_q4_1 = gfx906_get_int_b1_fast(bq4->qs, iqs + 1); \
        const int2 v0 = gfx906_get_int_from_mxfp4_table(aux_q4_0); \
        const int2 v1 = gfx906_get_int_from_mxfp4_table(aux_q4_1); \
        sumi = ggml_cuda_dp4a(v0.x, q8[0], sumi); \
        sumi = ggml_cuda_dp4a(v0.y, q8[4], sumi); \
        sumi = ggml_cuda_dp4a(v1.x, q8[1], sumi); \
        sumi = ggml_cuda_dp4a(v1.y, q8[5], sumi); \
    } while(0)

#endif
