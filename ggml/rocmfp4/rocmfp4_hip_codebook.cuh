#pragma once

#include "rocmfp4_hip_scale.cuh"

#include <cstdint>
#include <cstring>

// AMD-specific fast path for expanding eight packed ROCmFP4 nibbles into two
// int32 DP4A operands. This encodes the Codebook10 table directly as four
// 32-bit constants:
//   [0, 1, 2, 3], [4, 6, 8, 10], [0, -1, -2, -3], [-4, -6, -8, -10]
// Avoiding the table pointer keeps the ROCm/HIP MMVQ/MMQ hot path fully local
// to this format. Non-HIP builds still use llama.cpp's generic table expander.
static __device__ __forceinline__ int2 rocmfp4_get_int_from_codebook_16(const int & q4, const int8_t * fallback_table) {
#if defined(GGML_USE_HIP)
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    const uint32_t v_even_low  = __builtin_amdgcn_perm(values1, values0, q_even & 0x07070707u);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values1, values0, q_odd  & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values3, values2, q_even & 0x07070707u);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values3, values2, q_odd  & 0x07070707u);

    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    const uint32_t mask_odd  = 0x03020100u | ((q_odd  & 0x08080808u) >> 1);

    return make_int2(
        __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
        __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
#else
    return get_int_from_table_16(q4, fallback_table);
#endif
}
