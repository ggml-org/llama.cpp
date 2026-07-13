#pragma once

#include "common.cuh"
#include "mma.cuh"

// XOR swizzle for K/V SMEM tiles to avoid bank conflicts without row padding (Turing+ only).
// Stride must be a power-of-two >= 32 half2 columns,otherwise we keep +4 row padding.

static __host__ __device__ constexpr bool ggml_cuda_fattn_swz_pow2_stride(const int nbatch_2) {
    return nbatch_2 >= 32 && (nbatch_2 & (nbatch_2 - 1)) == 0;
}

static __device__ constexpr bool ggml_cuda_fattn_swz_enabled(const int nbatch_2) {
#if defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_swz_pow2_stride(nbatch_2);
#else
    GGML_UNUSED(nbatch_2);
    return false;
#endif
}

static __host__ bool ggml_cuda_fattn_swz_enabled(const int nbatch_2, const int cc) {
#ifdef GGML_USE_HIP
    GGML_UNUSED(nbatch_2);
    GGML_UNUSED(cc);
    return false;
#else
    return turing_mma_available(cc) && ggml_cuda_fattn_swz_pow2_stride(nbatch_2);
#endif
}

static __device__ constexpr int ggml_cuda_fattn_swz_tile_stride(const int nbatch_2) {
    return ggml_cuda_fattn_swz_enabled(nbatch_2) ? nbatch_2 : nbatch_2 + 4;
}

static __host__ int ggml_cuda_fattn_swz_tile_stride(const int nbatch_2, const int cc) {
    return ggml_cuda_fattn_swz_enabled(nbatch_2, cc) ? nbatch_2 : nbatch_2 + 4;
}

// Swizzled byte offset for tile element (row, col_h2); same map used for writes and reads.
template<int stride_h2>
static __device__ __forceinline__ int ggml_cuda_fattn_swz_bytes_rc(const int row, const int col_h2) {
    int off_bytes = (row * stride_h2 + col_h2) * (int) sizeof(half2);
    if constexpr (ggml_cuda_fattn_swz_enabled(stride_h2)) {
        off_bytes ^= (row & 7) << 4;
    }
    return off_bytes;
}

namespace ggml_cuda_fattn_smem_swizzle {

// ldmatrix.x4 from a 32-bit .shared address (lower register pressure than 64-bit generic pointers).
static __device__ __forceinline__ void ggml_cuda_fattn_ldmatrix_x4(int * xi, const uint32_t saddr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "r"(saddr));
}
static __device__ __forceinline__ void ggml_cuda_fattn_ldmatrix_x4_trans(int * xi, const uint32_t saddr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "r"(saddr));
}

// Per-lane swizzled .shared address for tile<16,8> ldmatrix.
template<int stride_h2>
static __device__ __forceinline__ uint32_t ggml_cuda_fattn_swz_saddr(
        const half2 * tile_base, const int base_row, const int base_col_h2, const int I, const int J) {
    const int lane_row = threadIdx.x % I;
    const int lane_col = (threadIdx.x / I) * (J / 2);
    const uint32_t base = __cvta_generic_to_shared(tile_base);
    uint32_t byte_off = (uint32_t)((base_row + lane_row) * stride_h2 + base_col_h2 + lane_col) * (uint32_t)sizeof(half2);
    if constexpr (ggml_cuda_fattn_swz_enabled(stride_h2)) {
        byte_off ^= (uint32_t)(((base_row + lane_row) & 7) << 4);
    }
    return base + byte_off;
}

template<typename TileT, int stride_h2>
static __device__ __forceinline__ void load_ldmatrix(
        TileT & t, half2 * tile_base, const int base_row, const int base_col_h2) {
    using Tile = typename std::remove_reference<TileT>::type;
    constexpr int I = Tile::I;
    constexpr int J = Tile::J;
#if defined(TURING_MMA_AVAILABLE)
    if constexpr (I == 16 && J == 8 && ggml_cuda_fattn_swz_enabled(stride_h2)) {
        const uint32_t saddr = ggml_cuda_fattn_swz_saddr<stride_h2>(tile_base, base_row, base_col_h2, I, J);
        ggml_cuda_fattn_ldmatrix_x4((int *) t.x, saddr);
        return;
    }
#endif // TURING_MMA_AVAILABLE
    ggml_cuda_mma::load_ldmatrix(t, tile_base + base_row * stride_h2 + base_col_h2, stride_h2);
}

template<typename TileT, int stride_h2>
static __device__ __forceinline__ void load_ldmatrix_trans(
        TileT & t, half2 * tile_base, const int base_row, const int base_col_h2) {
    using Tile = typename std::remove_reference<TileT>::type;
    constexpr int I = Tile::I;
    constexpr int J = Tile::J;
#if defined(TURING_MMA_AVAILABLE)
    if constexpr (I == 16 && J == 8 && ggml_cuda_fattn_swz_enabled(stride_h2)) {
        const uint32_t saddr = ggml_cuda_fattn_swz_saddr<stride_h2>(tile_base, base_row, base_col_h2, I, J);
        ggml_cuda_fattn_ldmatrix_x4_trans((int *) t.x, saddr);
        return;
    }
#endif // TURING_MMA_AVAILABLE
    ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + base_row * stride_h2 + base_col_h2, stride_h2);
}

} // namespace ggml_cuda_fattn_smem_swizzle
