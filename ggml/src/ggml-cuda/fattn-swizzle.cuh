#pragma once

#include "common.cuh"
#include "mma.cuh"

// XOR swizzle for K/V SMEM tiles to avoid bank conflicts without row padding (Turing+ only).
// Stride must be a power-of-two >= 32 half2 columns,otherwise we keep +4 row padding.

namespace ggml_cuda_fattn_smem_swizzle {

static __host__ __device__ constexpr bool pow2_stride(const int nbatch_2) {
    return nbatch_2 >= 32 && (nbatch_2 & (nbatch_2 - 1)) == 0;
}

static __device__ constexpr bool enabled(const int nbatch_2) {
#if defined(TURING_MMA_AVAILABLE)
    return pow2_stride(nbatch_2);
#else
    GGML_UNUSED(nbatch_2);
    return false;
#endif // defined(TURING_MMA_AVAILABLE)
}

static __host__ bool enabled(const int nbatch_2, const int cc) {
#ifdef GGML_USE_HIP
    GGML_UNUSED(nbatch_2);
    GGML_UNUSED(cc);
    return false;
#else
    return turing_mma_available(cc) && pow2_stride(nbatch_2);
#endif // GGML_USE_HIP
}

static __device__ constexpr int tile_stride(const int nbatch_2) {
    return enabled(nbatch_2) ? nbatch_2 : nbatch_2 + 4;
}

static __host__ int tile_stride(const int nbatch_2, const int cc) {
    return enabled(nbatch_2, cc) ? nbatch_2 : nbatch_2 + 4;
}

// Swizzled byte offset for tile element (row, col_h2); same map used for writes and reads.
template<int stride_h2>
static __device__ __forceinline__ int bytes_rc(const int row, const int col_h2) {
    static_assert(pow2_stride(stride_h2), "swizzled tile needs a pow2 stride");
    return ((row * stride_h2 + col_h2) * (int) sizeof(half2)) ^ ((row & 7) << 4);
}

#if defined(TURING_MMA_AVAILABLE)
// ldmatrix.x4 via 64-bit generic pointer.
static __device__ __forceinline__ void ldmatrix_x4(int * xi, const half2 * addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(addr));
}

static __device__ __forceinline__ void ldmatrix_x4_trans(int * xi, const half2 * addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(addr));
}
#endif // defined(TURING_MMA_AVAILABLE)

// Per-lane swizzled address for one tile<16, 8, half2> ldmatrix: 16 rows, 4 half2 columns per lane.
template<int stride_h2>
static __device__ __forceinline__ const half2 * lane_addr(
        const half2 * tile_base, const int base_row, const int base_col_h2) {
    static_assert(pow2_stride(stride_h2), "swizzled tile needs a pow2 stride");
    const int row = base_row    + threadIdx.x % 16;
    const int col = base_col_h2 + (threadIdx.x / 16) * 4;
    const uint32_t byte_off = (uint32_t) ((row * stride_h2 + col) * (int) sizeof(half2)) ^ (uint32_t) ((row & 7) << 4);
    return (const half2 *) ((const char *) tile_base + byte_off);
}

template<int stride_h2>
static __device__ __forceinline__ void load_ldmatrix(
        ggml_cuda_mma::tile<16, 8, half2> & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
#if defined(TURING_MMA_AVAILABLE)
    ldmatrix_x4((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2));
#else
    GGML_UNUSED_VARS(t, tile_base, base_row, base_col_h2);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

template<int stride_h2>
static __device__ __forceinline__ void load_ldmatrix_trans(
        ggml_cuda_mma::tile<16, 8, half2> & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
#if defined(TURING_MMA_AVAILABLE)
    ldmatrix_x4_trans((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2));
#else
    GGML_UNUSED_VARS(t, tile_base, base_row, base_col_h2);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

} // namespace ggml_cuda_fattn_smem_swizzle
