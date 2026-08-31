#pragma once

#include "common.cuh"
#include "mma.cuh"

// XOR swizzle for K/V SMEM tiles to avoid bank conflicts without row padding (Turing+ only).
// Stride must be a multiple of 32 half2 columns, otherwise we keep +4 row padding.

namespace ggml_cuda_fattn_smem_swizzle {

static __host__ __device__ constexpr bool bank_aligned(const int nbatch_2) {
    return nbatch_2 >= 32 && nbatch_2 % 32 == 0;
}

static __device__ constexpr bool enabled(const int nbatch_2) {
#if defined(TURING_MMA_AVAILABLE)
    return bank_aligned(nbatch_2);
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
    return turing_mma_available(cc) && bank_aligned(nbatch_2);
#endif // GGML_USE_HIP
}

static __device__ constexpr int tile_stride(const int nbatch_2) {
    return enabled(nbatch_2) ? nbatch_2 : nbatch_2 + 4;
}

static __host__ int tile_stride(const int nbatch_2, const int cc) {
    return enabled(nbatch_2, cc) ? nbatch_2 : nbatch_2 + 4;
}

// Swizzled byte offset for tile element (row, col_h2), same map used for writes and reads.
template<int stride_h2>
static __device__ __forceinline__ int bytes_rc(const int row, const int col_h2) {
    static_assert(bank_aligned(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    return ((row * stride_h2 + col_h2) * (int) sizeof(half2)) ^ ((row & 7) << 4);
}

// ldmatrix.x4 via 64-bit generic pointer.
static __device__ __forceinline__ void ldmatrix_x4(int * xi, const half2 * addr) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(addr));
#else
    GGML_UNUSED_VARS(xi, addr);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

static __device__ __forceinline__ void ldmatrix_x4_trans(int * xi, const half2 * addr) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(addr));
#else
    GGML_UNUSED_VARS(xi, addr);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

// Per-lane swizzled address for one tile<16, 8, half2> ldmatrix: 16 rows, 4 half2 columns per lane.
template<int stride_h2>
static __device__ __forceinline__ const half2 * lane_addr(
        const half2 * tile_base, const int base_row, const int base_col_h2, const int I, const int J) {
    static_assert(bank_aligned(stride_h2), "swizzled tile needs a stride that is a multiple of 32");
    const int lane_row = threadIdx.x % I;
    const int lane_col = (threadIdx.x / I) * (J / 2);
    uint32_t byte_off = (uint32_t) ((base_row + lane_row)*stride_h2 + base_col_h2 + lane_col) * (uint32_t) sizeof(half2);
    byte_off ^= (uint32_t) (((base_row + lane_row) & 7) << 4);
    return (const half2 *) ((const char *) tile_base + byte_off);
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix(
        TileT & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
    if constexpr (swz) {
        static_assert(std::is_same_v<TileT, ggml_cuda_mma::tile<16, 8, half2>>,
            "the swizzled layout is only supported for tile<16, 8, half2>");
        ldmatrix_x4((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2, TileT::I, TileT::J));
    } else {
        ggml_cuda_mma::load_ldmatrix(t, tile_base + base_row*stride_h2 + base_col_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix(TileT & t, const half2 * tile_base, const int off_h2) {
    if constexpr (swz) {
        load_ldmatrix<stride_h2, swz>(t, tile_base, off_h2 / stride_h2, off_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix(t, tile_base + off_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix_trans(
        TileT & t, const half2 * tile_base, const int base_row, const int base_col_h2) {
    if constexpr (swz) {
        static_assert(std::is_same_v<TileT, ggml_cuda_mma::tile<16, 8, half2>>,
            "the swizzled layout is only supported for tile<16, 8, half2>");
        ldmatrix_x4_trans((int *) t.x, lane_addr<stride_h2>(tile_base, base_row, base_col_h2, TileT::I, TileT::J));
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + base_row*stride_h2 + base_col_h2, stride_h2);
    }
}

template<int stride_h2, bool swz, typename TileT>
static __device__ __forceinline__ void load_ldmatrix_trans(TileT & t, const half2 * tile_base, const int off_h2) {
    if constexpr (swz) {
        load_ldmatrix_trans<stride_h2, swz>(t, tile_base, off_h2 / stride_h2, off_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(t, tile_base + off_h2, stride_h2);
    }
}

} // namespace ggml_cuda_fattn_smem_swizzle
