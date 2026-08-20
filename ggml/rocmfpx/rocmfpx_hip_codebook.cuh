#pragma once

#include <cstdint>

static __device__ __constant__ const int8_t rocmfpx_fp3_codebook[8] = {
    0, 1, 2, 4, 0, -1, -2, -4,
};

static __device__ __constant__ const int8_t rocmfpx_fp6_codebook[64] = {
     0,   1,   2,   3,   4,   5,   6,   7,
     8,   9,  10,  11,  12,  13,  14,  15,
    16,  17,  18,  19,  20,  21,  22,  23,
    24,  25,  26,  27,  28,  29,  30,  31,
     0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,
    -8,  -9, -10, -11, -12, -13, -14, -15,
   -16, -17, -18, -19, -20, -21, -22, -23,
   -24, -25, -26, -27, -28, -29, -30, -31,
};

static __device__ __forceinline__ int rocmfpx_pack4_fp3_codes(const uint32_t bits12) {
    const uint32_t c0 = (bits12 >> 0) & 7u;
    const uint32_t c1 = (bits12 >> 3) & 7u;
    const uint32_t c2 = (bits12 >> 6) & 7u;
    const uint32_t c3 = (bits12 >> 9) & 7u;
    const char4 v = make_char4(
        rocmfpx_fp3_codebook[c0],
        rocmfpx_fp3_codebook[c1],
        rocmfpx_fp3_codebook[c2],
        rocmfpx_fp3_codebook[c3]);
    return *((const int *) &v);
}

static __device__ __forceinline__ int rocmfpx_pack4_fp6_codes(const uint32_t bits24) {
    const uint32_t c0 = (bits24 >>  0) & 63u;
    const uint32_t c1 = (bits24 >>  6) & 63u;
    const uint32_t c2 = (bits24 >> 12) & 63u;
    const uint32_t c3 = (bits24 >> 18) & 63u;

    const char4 v = make_char4(
        rocmfpx_fp6_codebook[c0],
        rocmfpx_fp6_codebook[c1],
        rocmfpx_fp6_codebook[c2],
        rocmfpx_fp6_codebook[c3]);
    return *((const int *) &v);
}
