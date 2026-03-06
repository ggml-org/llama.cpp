#pragma once

// MMQ vectorized loads: 2x int4 (128-bit) instead of 8x scalar loads
// Q8_0 software pipelining: separate load/store phases for better MLP

#include "../gfx906-config.h"
#include "../quantize/vecdotq.cuh"

#if defined(GGML_USE_HIP)

static __device__ __forceinline__ void gfx906_load_q4_0_quants_vectorized(
    const int * __restrict__ y_qs,
    const int base_addr,
    const int qi,
    int * __restrict__ u) {

    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);

    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
}

static __device__ __forceinline__ void gfx906_load_q4_1_quants_vectorized(
    const int * __restrict__ y_qs,
    const int base_addr,
    const int qi,
    int * __restrict__ u) {

    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);

    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
}

template<int VDR>
static __device__ __forceinline__ void gfx906_load_quants_vectorized(
    const int * __restrict__ y_qs,
    const int base_addr,
    const int qi,
    int * __restrict__ u) {

    static_assert(VDR == 4, "Only VDR=4 supported for vectorized loads");

    const int4 vec0 = *((const int4 *) &y_qs[base_addr]);
    const int4 vec1 = *((const int4 *) &y_qs[base_addr + qi]);

    u[0] = vec0.x; u[2] = vec0.y; u[4] = vec0.z; u[6] = vec0.w;
    u[1] = vec1.x; u[3] = vec1.y; u[5] = vec1.z; u[7] = vec1.w;
}

#if defined(__gfx906__)

#define GFX906_LOAD_TILES_Q8_0_ASYNC(cache_size, nrows, nwarps, threads_per_row, need_check, \
    x, kbx0, stride, i_max, txi, kbx, kqsx, qs0_cache, qs1_cache, i_slot_cache) \
    do { \
        _Pragma("unroll") \
        for (int iter = 0; iter < cache_size; iter++) { \
            const int i0 = iter * nrows * nwarps; \
            const int i_slot = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row); \
            const int i_read = need_check ? min(i_slot, i_max) : i_slot; \
            const bool oob = need_check && (i_slot > i_max); \
            const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i_read*stride + kbx; \
            qs0_cache[iter] = oob ? 0 : gfx906_get_int_b2_fast(bxi[0].qs, kqsx); \
            qs1_cache[iter] = oob ? 0 : gfx906_get_int_b2_fast(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx); \
            i_slot_cache[iter] = i_slot; \
        } \
    } while(0)

#define GFX906_STORE_TILES_Q8_0_LDS_MMA(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi) \
    do { \
        _Pragma("unroll") \
        for (int iter = 0; iter < cache_size; iter++) { \
            const int i_slot = i_slot_cache[iter]; \
            x_qs[i_slot*MMQ_MMA_TILE_X_K_Q8_0 + 0             + txi] = qs0_cache[iter]; \
            x_qs[i_slot*MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] = qs1_cache[iter]; \
        } \
    } while(0)

#define GFX906_STORE_TILES_Q8_0_LDS_LEGACY(cache_size, x_qs, qs0_cache, qs1_cache, i_slot_cache, txi) \
    do { \
        _Pragma("unroll") \
        for (int iter = 0; iter < cache_size; iter++) { \
            const int i_slot = i_slot_cache[iter]; \
            x_qs[i_slot*(2*MMQ_TILE_NE_K + 1) + 0             + txi] = qs0_cache[iter]; \
            x_qs[i_slot*(2*MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = qs1_cache[iter]; \
        } \
    } while(0)

#endif // defined(__gfx906__)

#endif // GGML_USE_HIP
