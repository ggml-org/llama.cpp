#pragma once

// Fused quantized-KV dequant for the MMA FA kernel: raw q8_0/q4_0/f8_e4m3 cache rows staged with cp.async
// and dequantized into the f16 tiles in-kernel, eliminating the per-call f16 conversion and its compute-buffer scratch.
// Expressed as partial specializations of upstream's flash_attn_load_tile_impl (fattn-common.cuh) keyed on the source
// tag types below: upstream's loader machinery stays byte-identical, our specializations can never match upstream's
// instantiations (disjoint T_src), and a future signature change to the primary template breaks the build at these
// declarations instead of silently drifting the fused semantics.

#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

// The fused dequant path for a type_KV=F16 MMA kernel, 0 means the kernel reads plain f16 K/V.
constexpr int FATTN_MMA_FUSED_NONE = 0;
constexpr int FATTN_MMA_FUSED_Q8_0 = 1;
constexpr int FATTN_MMA_FUSED_Q4_0 = 2;
constexpr int FATTN_MMA_FUSED_F8_E4M3 = 3;
// All three block types cover 32 values per block; f8_e4m3 is 1 B/value, its cache scale applied outside the FA op.
constexpr int FATTN_MMA_Q8_0_BLOCK_BYTES = 34;
constexpr int FATTN_MMA_Q4_0_BLOCK_BYTES = 18;
constexpr int FATTN_MMA_F8_E4M3_BLOCK_BYTES = 32;
// Staging row bytes per fused type, single source of truth for the load / dequant / alloc-predictor sites.
static constexpr __host__ __device__ inline int ggml_cuda_fattn_mma_fused_block_bytes(const int fused) {
    return fused == FATTN_MMA_FUSED_Q4_0 ? FATTN_MMA_Q4_0_BLOCK_BYTES :
           fused == FATTN_MMA_FUSED_F8_E4M3 ? FATTN_MMA_F8_E4M3_BLOCK_BYTES : FATTN_MMA_Q8_0_BLOCK_BYTES;
}
// Maps a cache type to its fused kernel template arg (the gate rejects anything else up front).
static constexpr __host__ __device__ inline int ggml_cuda_fattn_mma_fused_type_from_ggml(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? FATTN_MMA_FUSED_Q4_0 :
           type == GGML_TYPE_F8_E4M3 ? FATTN_MMA_FUSED_F8_E4M3 : FATTN_MMA_FUSED_Q8_0;
}
static bool ggml_cuda_fattn_mma_fused_type_supported(const ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_F8_E4M3;
}
// Pipelined fused staging: two half-chunks in two slots, same total bytes as one chunk slot (smem and alloc predictor unchanged).
// The half-chunk row must be whole blocks and 16 B aligned: 8 B cp.async staging (q8_0/q4_0) measured slower than the wait it hides.
template<int nbatch2, int block_bytes>
static constexpr __device__ bool ggml_cuda_fattn_mma_fused_stage_pipelined() {
    return nbatch2 % 32 == 0 && (nbatch2 / 2 / 16) * block_bytes % 16 == 0;
}
// 8 B cp.async variant for the fused DKQ=DV=128 staging: q8_0/q4_0 rows of a 128-value head are 136/72 B (≡ 8 mod 16), never 16 B
// aligned like the cache row pitch they read from. .ca supports 4/8/16 B transfers (.cg upstream only 16 B); both pointers need 8 B alignment.
static __device__ __forceinline__ void cp_async_ca_8(const unsigned int dst, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
        : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
// Linear staging area for raw quantized tiles, located right after the mask tiles.
// Shared by flash_attn_ext_f16_process_tile and flash_attn_ext_f16_iter so the two can't drift apart.
template<int ncols1, int nbatch_fa>
static __device__ __forceinline__ char * flash_attn_ext_f16_stage(half * const __restrict__ tile_mask) {
    return (char *) (tile_mask + ncols1*(nbatch_fa + 8));
}

// Source tags for the fused flash_attn_load_tile_impl specializations: raw quantized cache rows, never a loaded dtype.
struct fattn_src_q8_0 {};
struct fattn_src_q4_0 {};
struct fattn_src_f8_e4m3 {};
template <int fused> struct fattn_src_of;
template <> struct fattn_src_of<FATTN_MMA_FUSED_Q8_0>    { using type = fattn_src_q8_0; };
template <> struct fattn_src_of<FATTN_MMA_FUSED_Q4_0>    { using type = fattn_src_q4_0; };
template <> struct fattn_src_of<FATTN_MMA_FUSED_F8_E4M3> { using type = fattn_src_f8_e4m3; };

namespace fattn_fused_detail {
// Loads a tile of raw quantized data into the shared memory staging area, to be dequantized into an f16 tile once arrived.
// The staging area is a linear row-major layout, row size = D2/16 blocks * block_bytes; D2 is the row width in half2 values.
// Row resolution mirrors upstream's flash_attn_load_tile_impl: the sparse list is readable only up to i_sup (its tail beyond
// the count is not allocated to this tile), the kv_rows list is read full-width (it is padded with -1), negative rows gather row 0.
template <int D2, int block_bytes, int nwarps, int I, bool use_sparse, bool KV_indirection>
static __device__ __forceinline__ void stage(
        const char * const __restrict__ KV, char * const __restrict__ stage, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
    static_assert(D2 % 16 == 0, "fused dequant requires whole quantization blocks");
    constexpr int nblk = D2 / 16;
    constexpr int row_bytes = nblk * block_bytes;
    static_assert(row_bytes % 8 == 0, "quantized tile rows must be a multiple of 8 B");
    // Transfer width by row alignment: 16 B wherever rows are 16 B aligned (q8_0/q4_0 at D >= 256, f8_e4m3 always),
    // 8 B for the D=128 q8_0/q4_0 rows (136/72 B, see cp_async_ca_8).
    constexpr int copy_bytes = row_bytes % 16 == 0 ? 16 : 8;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int units_per_row = row_bytes / copy_bytes; // cp.async units per row
    constexpr int preload = 64;
    GGML_UNUSED(i_sup);
    GGML_UNUSED(indices);
    const unsigned int stage_32 = ggml_cuda_cvta_generic_to_shared(stage);
#pragma unroll
    for (int c = threadIdx.y*warp_size + threadIdx.x; c < I*units_per_row; c += nwarps*warp_size) {
        const int i = c / units_per_row;
        const int q = c % units_per_row;
        int64_t i_KV;
        if constexpr (use_sparse) {
            // out-of-range and negative slots gather row 0, the -inf mask removes their contribution (mirrors upstream)
            const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : 0;
            i_KV = index >= 0 ? index : 0;
        } else if constexpr (KV_indirection) {
            // K/V rows given per mask column, negative rows are padding
            i_KV = indices ? max(indices[k_VKQ_0 + i], 0) : k_VKQ_0 + i;
        } else {
            i_KV = k_VKQ_0 + i;
        }
        if constexpr (copy_bytes == 16) {
            cp_async_cg_16<preload>(stage_32 + c*sizeof(uint4), KV + i_KV*(stride_KV*4) + q*sizeof(uint4));
        } else {
            cp_async_ca_8(stage_32 + c*sizeof(uint2), KV + i_KV*(stride_KV*4) + q*sizeof(uint2));
        }
    }
}
}  // namespace fattn_fused_detail

// Each specialization: stage() fills the linear staging area with raw rows; dequant() writes them into the f16 tile with
// the same swizzle as upstream's loader. The arithmetic is bit-identical to the standalone ggml_cuda_from_to_f16 conversion.
template <int warp_size, int nwarps, int I, int J, int stride_tile, bool use_cp_async, bool oob_check,
          bool use_sparse, bool KV_indirection, bool use_swizzle>
struct flash_attn_load_tile_impl<warp_size, nwarps, I, J, stride_tile, use_cp_async, oob_check,
        use_sparse, KV_indirection, use_swizzle, fattn_src_q8_0, half2> {
    static_assert(use_cp_async, "fused staging requires cp.async");
    static constexpr int D2 = J / 2;

    static __device__ __forceinline__ void stage(
            const char * const __restrict__ KV, char * const __restrict__ stage, const int stride_KV,
            const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
        fattn_fused_detail::stage<D2, FATTN_MMA_Q8_0_BLOCK_BYTES, nwarps, I, use_sparse, KV_indirection>(
            KV, stage, stride_KV, k_VKQ_0, i_sup, indices);
    }
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ stage, half2 * const __restrict__ tile, const int blk_off = 0) {
        constexpr int warp_size_v = ggml_cuda_get_physical_warp_size();
        constexpr int nblk = D2 / 16; // q8_0 blocks per row, 32 values each
        constexpr int row_bytes = nblk * FATTN_MMA_Q8_0_BLOCK_BYTES;
        const half2 bias1152 = __floats2half2_rn(1152.0f, 1152.0f);
#pragma unroll
        for (int u = threadIdx.y*warp_size_v + threadIdx.x; u < I*nblk; u += nwarps*warp_size_v) {
            const int i = u / nblk;
            const int j = u % nblk;
            const char * const b = stage + i*row_bytes + j*FATTN_MMA_Q8_0_BLOCK_BYTES;
            const half  d  = *(const half *) b;
            const half2 dd = __half2half2(d);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                alignas(16) half vals[8];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint32_t pair = *(const uint16_t *) (b + 2 + (4*q + l)*2);
                    // s8 -> f16 without I2F: byte as mantissa of 0x6400 (ulp = 1), xor bit 7 (weight 128, the
                    // two's complement sign) -> exact f16(1024 + 128 + v); the subtract of that bias is exact too
                    const uint32_t bits = __byte_perm(pair, 0x64006400u, 0x7150) ^ 0x00800080u;
                    const half2 v = __hmul2(__hsub2(*(const half2 *) &bits, bias1152), dd);
                    vals[2*l + 0] = v.x;
                    vals[2*l + 1] = v.y;
                }
                ggml_cuda_memcpy_1<16>(ggml_cuda_mma::swizzle<stride_tile>(tile, i*stride_tile + (4*(j + blk_off) + q)*4, i), vals);
            }
        }
    }
};

template <int warp_size, int nwarps, int I, int J, int stride_tile, bool use_cp_async, bool oob_check,
          bool use_sparse, bool KV_indirection, bool use_swizzle>
struct flash_attn_load_tile_impl<warp_size, nwarps, I, J, stride_tile, use_cp_async, oob_check,
        use_sparse, KV_indirection, use_swizzle, fattn_src_q4_0, half2> {
    static_assert(use_cp_async, "fused staging requires cp.async");
    static constexpr int D2 = J / 2;

    static __device__ __forceinline__ void stage(
            const char * const __restrict__ KV, char * const __restrict__ stage, const int stride_KV,
            const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
        fattn_fused_detail::stage<D2, FATTN_MMA_Q4_0_BLOCK_BYTES, nwarps, I, use_sparse, KV_indirection>(
            KV, stage, stride_KV, k_VKQ_0, i_sup, indices);
    }
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ stage, half2 * const __restrict__ tile, const int blk_off = 0) {
        constexpr int warp_size_v = ggml_cuda_get_physical_warp_size();
        constexpr int nblk = D2 / 16; // q4_0 blocks per row, 32 values each
        constexpr int row_bytes = nblk * FATTN_MMA_Q4_0_BLOCK_BYTES;
#pragma unroll
        for (int u = threadIdx.y*warp_size_v + threadIdx.x; u < I*nblk; u += nwarps*warp_size_v) {
            const int i = u / nblk;
            const int j = u % nblk;
            const char * const b = stage + i*row_bytes + j*FATTN_MMA_Q4_0_BLOCK_BYTES;
            const float d  = __half2float(*(const half *) b);
            const float dm = -8*d;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                alignas(16) half vals[8];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    // Values 0..15 are the low nibbles of qs[0..15], values 16..31 the high nibbles:
                    const uint16_t pair = *(const uint16_t *) (b + 2 + 8*(q&1) + 2*l);
                    const uint8_t  n0   = q < 2 ? (uint8_t) (pair & 0xF) : (uint8_t) ((pair >> 4) & 0xF);
                    const uint8_t  n1   = q < 2 ? (uint8_t) ((pair >> 8) & 0xF) : (uint8_t) (pair >> 12);
                    vals[2*l + 0] = __float2half(d * (float) n0 + dm);
                    vals[2*l + 1] = __float2half(d * (float) n1 + dm);
                }
                ggml_cuda_memcpy_1<16>(ggml_cuda_mma::swizzle<stride_tile>(tile, i*stride_tile + (4*(j + blk_off) + q)*4, i), vals);
            }
        }
    }
};

// Converts two e4m3 bytes (low byte = first value) to half2. e4m3 values are exactly representable in f16, so the
// hardware cvt is bit-identical to the reference dequantize_row_f8_e4m3 path; the fallback mirrors convert.cuh.
static __device__ __forceinline__ half2 fattn_f8x2_to_half2(const uint16_t pair) {
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    return static_cast<half2>(__nv_cvt_fp8x2_to_halfraw2((uint32_t) pair, __NV_E4M3));
#else
    return __floats2half2_rn(ggml_cuda_f8_e4m3_to_fp32((uint8_t) (pair & 0xFF)),
                             ggml_cuda_f8_e4m3_to_fp32((uint8_t) (pair >> 8)));
#endif // defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int warp_size, int nwarps, int I, int J, int stride_tile, bool use_cp_async, bool oob_check,
          bool use_sparse, bool KV_indirection, bool use_swizzle>
struct flash_attn_load_tile_impl<warp_size, nwarps, I, J, stride_tile, use_cp_async, oob_check,
        use_sparse, KV_indirection, use_swizzle, fattn_src_f8_e4m3, half2> {
    static_assert(use_cp_async, "fused staging requires cp.async");
    static constexpr int D2 = J / 2;

    static __device__ __forceinline__ void stage(
            const char * const __restrict__ KV, char * const __restrict__ stage, const int stride_KV,
            const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
        fattn_fused_detail::stage<D2, FATTN_MMA_F8_E4M3_BLOCK_BYTES, nwarps, I, use_sparse, KV_indirection>(
            KV, stage, stride_KV, k_VKQ_0, i_sup, indices);
    }
    // Rows are plain byte arrays (1 B/value, no scale), so this is just the pair-wise cvt; the stage is read in 16 B
    // units (16 values, same pairs as the byte-pair reads) because smem scalar loads dominated the kernel's L1 traffic.
    static __device__ __forceinline__ void dequant(
            const char * const __restrict__ stage, half2 * const __restrict__ tile, const int blk_off = 0) {
        constexpr int warp_size_v = ggml_cuda_get_physical_warp_size();
        constexpr int nblk = D2 / 16; // f8_e4m3 blocks per row, 32 single-byte values each
        constexpr int row_bytes = nblk * FATTN_MMA_F8_E4M3_BLOCK_BYTES;
#pragma unroll
        for (int u = threadIdx.y*warp_size_v + threadIdx.x; u < I*nblk; u += nwarps*warp_size_v) {
            const int i = u / nblk;
            const int j = u % nblk;
            const char * const b = stage + i*row_bytes + j*FATTN_MMA_F8_E4M3_BLOCK_BYTES;
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const uint4 w = *(const uint4 *) (b + v*16);
                alignas(16) half2 lo[4] = {
                    fattn_f8x2_to_half2((uint16_t) w.x), fattn_f8x2_to_half2((uint16_t) (w.x >> 16)),
                    fattn_f8x2_to_half2((uint16_t) w.y), fattn_f8x2_to_half2((uint16_t) (w.y >> 16))};
                alignas(16) half2 hi[4] = {
                    fattn_f8x2_to_half2((uint16_t) w.z), fattn_f8x2_to_half2((uint16_t) (w.z >> 16)),
                    fattn_f8x2_to_half2((uint16_t) w.w), fattn_f8x2_to_half2((uint16_t) (w.w >> 16))};
                ggml_cuda_memcpy_1<16>(ggml_cuda_mma::swizzle<stride_tile>(tile, i*stride_tile + (4*(j + blk_off) + 2*v + 0)*4, i), lo);
                ggml_cuda_memcpy_1<16>(ggml_cuda_mma::swizzle<stride_tile>(tile, i*stride_tile + (4*(j + blk_off) + 2*v + 1)*4, i), hi);
            }
        }
    }
};

// Site-facing alias: the fused loader impl for one side (K or V) of the kernel. J is the tile chunk width in half2 values
// (2*D2), mirroring upstream's load_tile call convention; I is the KV rows per tile (nbatch_fa).
template <int warp_size, int fused, int nwarps, int I, int J, int stride_tile, bool use_sparse, bool KV_indirection, bool use_swizzle>
using fattn_fused_load = flash_attn_load_tile_impl<
    warp_size, nwarps, I, J, stride_tile, /*use_cp_async=*/true, /*oob_check=*/false,
    use_sparse, KV_indirection, use_swizzle, typename fattn_src_of<fused>::type, half2>;
