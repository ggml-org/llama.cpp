#pragma once

// Y-tile prefetch for MMQ: issues global_load_dword for next iteration
// Hides memory latency by overlapping loads with compute

#include "../gfx906-config.h"

#if defined(GGML_USE_HIP) && defined(__gfx906__)

template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ int gfx906_prefetch_y_tile_v4(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {

    const int kb0_next = kb0 + blocks_per_iter;

    if (kb0_next >= kb0_stop) {
        return 0;
    }

    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));

    const int * by_next = y + ncols_y * (kb0_next * stride_factor);

    // Use 16 threads from warp 0 to prefetch 16 cache lines (1KB total)
    // This uses ~16 spare VGPRs to warm L2 cache for next iteration
    const int lane_id = threadIdx.x;
    if (threadIdx.y != 0 || lane_id >= 16) {
        return 0;
    }

    // Each thread prefetches a different cache line (64 bytes apart = 16 ints)
    const int prefetch_offset = lane_id * 16;
    const int * prefetch_addr = by_next + prefetch_offset;

    int prefetch_data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(prefetch_data)
        : "v"(prefetch_addr)
        : "memory"
    );
    return prefetch_data;
}

// Prefetch second Y tile (the +sz offset one)
// Uses remaining threads from warp 0 (lanes 16-31) to prefetch second half
template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ int gfx906_prefetch_y_tile_second(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter,
    const int sz) {

    const int kb0_next = kb0 + blocks_per_iter;

    if (kb0_next >= kb0_stop) {
        return 0;
    }

    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));

    // Prefetch the second Y tile (at +sz offset)
    const int * by_next_second = y + ncols_y * (kb0_next * stride_factor + sz);

    // Use lanes 16-31 from warp 0 for second tile prefetch
    const int lane_id = threadIdx.x;
    if (threadIdx.y != 0 || lane_id < 16 || lane_id >= 32) {
        return 0;
    }

    const int prefetch_offset = (lane_id - 16) * 16;
    const int * prefetch_addr = by_next_second + prefetch_offset;

    int prefetch_data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(prefetch_data)
        : "v"(prefetch_addr)
        : "memory"
    );
    return prefetch_data;
}

static __device__ __forceinline__ void gfx906_prefetch_consume(int prefetch_data) {
    asm volatile(
        "v_mov_b32 %0, %0\n"
        : "+v"(prefetch_data)
    );
}

template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ void gfx906_prefetch_y_tile_v2(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {

    const int kb0_next = kb0 + blocks_per_iter;

    if (kb0_next >= kb0_stop) {
        return;
    }

    const int tid = threadIdx.y * warp_size + threadIdx.x;

    constexpr int total_elements = mmq_x * mmq_tile_y_k;
    if (tid >= total_elements) {
        return;
    }

    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));

    const int * by_next = y + ncols_y * (kb0_next * stride_factor);
    const int * prefetch_addr = by_next + tid;

    int dummy;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(dummy)
        : "v"(prefetch_addr)
        : "memory"
    );
}

template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ void gfx906_prefetch_y_tile_v1(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {

    const int kb0_next = kb0 + blocks_per_iter;

    if (kb0_next >= kb0_stop) {
        return;
    }

    const int tid = threadIdx.y * warp_size + threadIdx.x;
    constexpr int total_elements = mmq_x * mmq_tile_y_k;

    if (tid >= total_elements) {
        return;
    }

    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));

    const int * by_next = y + ncols_y * (kb0_next * stride_factor);
    const int * prefetch_addr = by_next + tid;

    int dummy;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(dummy)
        : "v"(prefetch_addr)
        : "memory"
    );
}

template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
static __device__ __forceinline__ void gfx906_prefetch_y_tile_noop(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {
    (void)y; (void)ncols_y; (void)kb0; (void)kb0_stop; (void)qk; (void)blocks_per_iter;
}

// X-tile prefetch: warm L2 cache for next iteration's X data
// Uses warp 1 threads to prefetch X tile data while warp 0 prefetches Y
template<int mmq_y>
static __device__ __forceinline__ int gfx906_prefetch_x_tile(
    const char * __restrict__ x,
    const int offset_x,
    const int kb0,
    const int kb0_stop,
    const int blocks_per_iter,
    const int stride_row_x) {

    const int kb0_next = kb0 + blocks_per_iter;

    if (kb0_next >= kb0_stop) {
        return 0;
    }

    // Use 16 threads from warp 1 to prefetch X tile
    const int lane_id = threadIdx.x;
    if (threadIdx.y != 1 || lane_id >= 16) {
        return 0;
    }

    // Prefetch X data for next iteration - different rows
    // Each thread prefetches from a different row (stride by row)
    const int row = lane_id;
    const char * x_row = x + (offset_x + kb0_next) + row * stride_row_x;
    const int * x_ptr = (const int *)x_row;

    int prefetch_data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(prefetch_data)
        : "v"(x_ptr)
        : "memory"
    );
    return prefetch_data;
}

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
