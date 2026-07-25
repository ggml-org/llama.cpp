#pragma once

#include "common.cuh"

#include <climits>
#include <cstdint>

#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K             256
#define MMQ_ITER_K_FP4         512
#define MMQ_NWARPS               8

typedef void (*ggml_cuda_mmq_load_tiles_t)(const char * __restrict__ x, int * x_tile, const int kbx0, const int i_max, const int stride);
typedef void (*ggml_cuda_mmq_vec_dot_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00);
typedef void (*ggml_cuda_mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ get_rows_to_sorted,
    float * __restrict__ dst, const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max);

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

static constexpr int QK8_1_MMQ  = 4*QK8_1;
static constexpr int QK_FP4_MMQ = 2*QK8_1_MMQ;

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[QK8_1_MMQ];
};

// this struct is used for fp4 data types (currently only used for Blackwell)
// mxfp4 has block size 32, each int32 of d4 contains 2 e8m0 scales in the lower 16 bits
// nvfp4 has block size 16, each int32 of d4 contains 4 ue4m3 scales
struct block_fp4_mmq {
    uint32_t d4[4];
    int8_t   qs[QK_FP4_MMQ / 2];
};

static_assert(sizeof(block_q8_1_mmq) == QK8_1_MMQ + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_fp4_mmq)  == sizeof(block_q8_1_mmq),    "Unexpected block_fp4_mmq size");

static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q5_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q5_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q8_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_MXFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_NVFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q2_K:
            return MMQ_Q8_1_DS_LAYOUT_D2S6;
        case GGML_TYPE_Q3_K:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_IQ1_S:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

// Decouple shared memory tile sizes from WARP_SIZE to allow for different warp sizes.
// The K dimension of the tiles has either,
// 1*MMQ_TILE_NE_K==32 (always for TILE_Y_K) or 2*MMQ_TILE_NE_K==64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// In other words, the size of the quantized data in the K dimension is a multiple of MMQ_TILE_NE_K.
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma.
#define MMQ_TILE_NE_K 32

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)
#define MMQ_TILE_Y_K     (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)
#define MMQ_TILE_Y_FP4_K MMQ_TILE_Y_K

enum ggml_cuda_mmq_sram_layout {
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_FP4,   // MXFP4 and NVFP4 on Blackwell.
    GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, // Generic NVFP4
};

static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_sram_layout sram_layout) {
    switch (sram_layout) {
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_1 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K         + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2       + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K   + MMQ_TILE_NE_K/8 + 7;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_FP4:
            return 2*MMQ_TILE_NE_K + 8                     + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2       + 4;
        default:
            return -1;
    }
}

static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K)  % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_FP4)   % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4) % 8 == 4, "Wrong padding.");

static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_FP4) == ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1), "Wrong tile size for MXFP4");

// Config options for the MMQ kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct ggml_cuda_mmq_config {
    ggml_type                 type;        // src0->type
    int                       nthreads;    // Number of threads per CUDA block.
    int                       occupancy;   // Targeted occupancy for the MMA kernel.
    int                       I;           // SRAM tile width in src0->ne[1]/dst->ne[0] direction.
    int                       J;           // SRAM tile width in src1->ne[1]/dst->ne[1] direction.
    ggml_cuda_mmq_sram_layout sram_layout; // SRAM tile length in src0->ne[0]/src1->ne[0] direction (physical 32 bit elements).
    int                       K_vram;      // VRAM tile length in src0->ne[0]/src1->ne[0] direction (logical elements).
    bool                      stream_k;    // Whether or not to use stream-k decomposition.
    bool                      fallback;    // Whether a fallback for out-of-bounds check in src0->ne[1] direction is needed.

    constexpr __host__ __device__ ggml_cuda_mmq_config(
            ggml_type type, int nthreads, int occupancy, int I, int J, ggml_cuda_mmq_sram_layout sram_layout, int K_vram, bool stream_k, bool fallback) :
        type(type), nthreads(nthreads), occupancy(occupancy), I(I), J(J), sram_layout(sram_layout), K_vram(K_vram), stream_k(stream_k), fallback(fallback) {}

    constexpr __device__ int rows_per_warp() const {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        return 16;
#else
        return J >= 48 && J % 16 == 0 ? 32 : 16;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    // TODO transition all combinations of GPUs and quantizations to the MMA data layout.
    __host__ int use_mma_data_layout(const int cc) const {
        if (amd_mfma_available(cc) || amd_wmma_available(cc) || turing_mma_available(cc)) {
            return true;
        }
        return false;
    }

    constexpr __device__ bool use_mma_data_layout() const {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
        return true;
#else
        return false;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    }

};

#define CASE(type_, nthreads_, occupancy_, I_, J_, sram_layout_, K_vram_, stream_k_, fallback_)                                           \
    if (type == (type_) && J == (J_) && fallback == (fallback_)) {                                                                        \
        static_assert((nthreads_) %  32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                \
        static_assert(                          (occupancy_)      <=   8, "bad occupancy");                                               \
        static_assert((I_)        %  32 == 0,                             "bad I");                                                       \
        static_assert((J_)        %   8 == 0,                             "bad J");                                                       \
        static_assert((K_vram_)   % 256 == 0,                             "bad K_vram");                                                  \
        return ggml_cuda_mmq_config((type_), (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_), (stream_k_), (fallback_)); \
    }                                                                                                                                     \

#include "mmq-config-pascal.cuh"
#include "mmq-config-ampere.cuh"
#include "mmq-config-blackwell.cuh"

#include "mmq-config-cdna.cuh"
#include "mmq-config-rdna2.cuh"
#include "mmq-config-rdna4.cuh"

#undef CASE

static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return ggml_cuda_mmq_get_config_cdna(type, J, fallback);
        }
        if (amd_wmma_available(cc)) {
            return ggml_cuda_mmq_get_config_rdna4(type, J, fallback);
        }
        return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
    }
    if (blackwell_mma_available(cc)) {
        return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
    }
    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {
        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    }
    return ggml_cuda_mmq_get_config_pascal(type, J, fallback);
}

static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback) {
#ifdef GGML_USE_HIP
#ifdef CDNA
    return ggml_cuda_mmq_get_config_cdna(type, J, fallback);
#elif defined(AMD_WMMA_AVAILABLE)
    return ggml_cuda_mmq_get_config_rdna4(type, J, fallback);
#else
    return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
#endif // CDNA
#else
#ifdef BLACKWELL_MMA_AVAILABLE
    return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
#else
    return ggml_cuda_mmq_get_config_pascal(type, J, fallback);
#endif // BLACKWELL_MMA_AVAILABLE
#endif // GGML_USE_HIP
    GGML_UNUSED_VARS(type, J, fallback);
}

static __host__ int ggml_cuda_mmq_get_type(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).type;
}

static constexpr __device__ int ggml_cuda_mmq_get_type(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).type;
}

static __host__ int ggml_cuda_mmq_get_nthreads(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).nthreads;
}

static constexpr __device__ int ggml_cuda_mmq_get_nthreads(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).nthreads;
}

static __host__ int ggml_cuda_mmq_get_occupancy(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).occupancy;
}

static constexpr __device__ int ggml_cuda_mmq_get_occupancy(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).occupancy;
}

static __host__ int ggml_cuda_mmq_get_I(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).I;
}

static constexpr __device__ int ggml_cuda_mmq_get_I(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).I;
}

static __host__ int ggml_cuda_mmq_get_J(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).J;
}

static constexpr __device__ int ggml_cuda_mmq_get_J(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).J;
}

static __host__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).sram_layout;
}

static constexpr __device__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).sram_layout;
}

static __host__ int ggml_cuda_mmq_get_K_vram(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).K_vram;
}

static constexpr __device__ int ggml_cuda_mmq_get_K_vram(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).K_vram;
}

static __host__ bool ggml_cuda_mmq_get_stream_k(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).stream_k;
}

static constexpr __device__ bool ggml_cuda_mmq_get_stream_k(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).stream_k;
}

static __host__ int ggml_cuda_mmq_get_fallback(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_config(type, J, fallback, cc).fallback;
}

static constexpr __device__ int ggml_cuda_mmq_get_fallback(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).fallback;
}

// ---------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_mmq_get_sram_stride(const ggml_type type, const int J, const bool fallback, const int cc) {
    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback, cc));
}

static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback));
}

static __host__ int ggml_cuda_mmq_get_J_max(const ggml_type type, const bool fallback, const int cc, const int64_t ne11) {
    int ret = std::min(ne11, int64_t(512));
    ret -= ret % 8;
    for (;ret > 0; ret -= 8) {
        if (ggml_cuda_mmq_get_config(type, ret, fallback, cc).type != GGML_TYPE_COUNT) {
            return ret;
        }
    }
    return ret;
}

static constexpr __device__ int ggml_cuda_mmq_get_rows_per_warp(ggml_type type, int J, bool fallback) {
    return ggml_cuda_mmq_get_config(type, J, fallback).rows_per_warp();
}

#define MMQ_DP4A_TXS_Q4_0    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_0   + I/QI4_0,     0}
#define MMQ_DP4A_TXS_Q4_1    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_1   + I/QI4_1,     0}
#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*2/QI8_0 + I/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*4/QI8_0 + I/(QI8_0/4), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*2/QI8_1 + I/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q2_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K         + I,           0}
#define MMQ_DP4A_TXS_Q3_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I,                                     I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_K,                 I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K/QI5_K   + I/QI5_K,     I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q6_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K/QI6_K   + I/QI6_K,     I*MMQ_TILE_NE_K/8 + I/8}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int I) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q4_0:    return MMQ_DP4A_TXS_Q4_0;
        case GGML_TYPE_Q4_1:    return MMQ_DP4A_TXS_Q4_1;
        case GGML_TYPE_Q5_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q5_1:    return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_Q8_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_MXFP4:   return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_NVFP4:   return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_Q2_K:    return MMQ_DP4A_TXS_Q2_K;
        case GGML_TYPE_Q3_K:    return MMQ_DP4A_TXS_Q3_K;
        case GGML_TYPE_Q4_K:    return MMQ_DP4A_TXS_Q4_K;
        case GGML_TYPE_Q5_K:    return MMQ_DP4A_TXS_Q5_K;
        case GGML_TYPE_Q6_K:    return MMQ_DP4A_TXS_Q6_K;
        case GGML_TYPE_IQ2_XXS: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_XS:  return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ2_S:   return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ3_XXS: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ3_S:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ1_S:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_XS:  return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_NL:  return MMQ_DP4A_TXS_Q8_0;
        default:                return tile_x_sizes{0, 0, 0};
    }
}

// FIXME temporary until all combinations of data types and GPUs can use the MMA data layout
static __host__ int ggml_cuda_mmq_get_nbytes_shared_x(const ggml_cuda_mmq_config & config, const int cc) {
    if (config.use_mma_data_layout(cc)) {
        return config.I * ggml_cuda_mmq_get_sram_stride(config.sram_layout) * 4;
    }
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(config.type, config.I);
    return (txs.qs + txs.dm + txs.sc) * 4;
}

// ------------------------------------------------------------

#include "mmq-load-tiles.cuh"
#include "mmq-vec-dot.cuh"

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_write_back_dp4a(
        const float * __restrict__ sum, const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
        const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    const bool y_scale_used = y_scale != nullptr;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += warp_size) {
            const int i = i0 + threadIdx.x;

            if (fallback && i > i_max) {
                continue;
            }

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (y_scale_used) {
                    dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
                } else {
                    dst[ids_dst[j]*stride + i] = sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
                }
            } else {
                dst[ids_dst[j]*stride + i] = sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
                GGML_UNUSED(y_scale_used);
            }
        }
    }
}

template<ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
            const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
            const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max) {

#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
#else
    typedef tile<16,  8, int> tile_C;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps        = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);

    const bool y_scale_used = y_scale != nullptr;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (fallback && i > i_max) {
                    continue;
                }

                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (y_scale_used) {
                        dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    } else {
                        dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    }
                } else {
                    dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
                    GGML_UNUSED(y_scale_used);
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------

// TODO remove this struct and use ggml_cuda_mmq_sram_layout instead.
struct ggml_cuda_mmq_util_funcs {
    int              vdr;
    ggml_cuda_mmq_load_tiles_t load_tiles;
    ggml_cuda_mmq_vec_dot_t    vec_dot;
    ggml_cuda_mmq_write_back_t write_back;

    constexpr __host__ __device__ ggml_cuda_mmq_util_funcs(
            int vdr, ggml_cuda_mmq_load_tiles_t load_tiles, ggml_cuda_mmq_vec_dot_t vec_dot, ggml_cuda_mmq_write_back_t write_back) :
        vdr(vdr), load_tiles(load_tiles), vec_dot(vec_dot), write_back(write_back) {}
};

template <ggml_type type, int J, bool fallback>
static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_funcs() {
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

    if (!ggml_cuda_mmq_get_config(type, J, fallback).use_mma_data_layout()) {
        switch (type) {
            case GGML_TYPE_Q1_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q1_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q1_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q4_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q4_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q4_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q4_1:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q4_1_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q4_1<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q4_1_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q5_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q5_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q5_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q5_1:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q5_1_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q5_1<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q8_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q8_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
// ---------------------------------------------------------------------------------------------
            case GGML_TYPE_Q2_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q2_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q2_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q3_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q3_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q3_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q3_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q4_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q4_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q5_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q5_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q5_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q5_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_Q6_K:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q6_K_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
// ---------------------------------------------------------------------------------------------
            case GGML_TYPE_IQ1_S:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ1_S_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq1_s<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_IQ2_XXS:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ2_XXS_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq2_xxs<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_IQ2_XS:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ2_XS_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq2_xs<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_IQ2_S:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ2_S_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq2_s<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_IQ3_XXS:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ3_XXS_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq3_xxs<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_IQ3_S:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ3_S_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq3_s<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_IQ4_XS:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ4_XS_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq4_xs<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_IQ4_NL:
                return ggml_cuda_mmq_util_funcs(
                    VDR_IQ4_NL_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_iq4_nl<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
// ---------------------------------------------------------------------------------------------
            case GGML_TYPE_MXFP4:
                return ggml_cuda_mmq_util_funcs(
                    VDR_MXFP4_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_mxfp4<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            case GGML_TYPE_NVFP4:
                return ggml_cuda_mmq_util_funcs(
                    VDR_NVFP4_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);
            default:
                return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
        }
    }

// ---------------------------------------------------------------------------------------------

#ifdef BLACKWELL_MMA_AVAILABLE
    switch (type) {
        case GGML_TYPE_MXFP4:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_mxfp4_fp4<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_NVFP4:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_nvfp4_nvfp4<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        default:
            break;
    }
#endif // BLACKWELL_MMA_AVAILABLE

// ---------------------------------------------------------------------------------------------

    switch (type) {
        case GGML_TYPE_Q1_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q1_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q4_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q4_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_DS4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q4_1:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q4_1<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q5_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q5_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q5_1:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q5_1<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q8_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
// ---------------------------------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q2_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q3_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q3_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q4_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q5_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q5_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_Q6_K:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
// ---------------------------------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq1_s<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_IQ2_XXS:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq2_xxs<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_IQ2_XS:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq2_xs<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_IQ2_S:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq2_s<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_IQ3_XXS:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq3_xxs<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_IQ3_S:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq3_s<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_IQ4_XS:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq4_xs<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_IQ4_NL:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_iq4_nl<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
// ---------------------------------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_mxfp4<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        case GGML_TYPE_NVFP4:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
        default:
            return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
    }
}

template <ggml_type type, int J, bool fallback>
static constexpr __device__ int ggml_cuda_mmq_get_vdr() {
    return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().vdr;
}

template <ggml_type type, int J, bool fallback>
static constexpr __device__ ggml_cuda_mmq_load_tiles_t ggml_cuda_mmq_get_load_tiles() {
    return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().load_tiles;
}

template <ggml_type type, int J, bool fallback>
static constexpr __device__ ggml_cuda_mmq_vec_dot_t ggml_cuda_mmq_get_vec_dot() {
    return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().vec_dot;
}

template <ggml_type type, int J, bool fallback>
static constexpr __device__ ggml_cuda_mmq_write_back_t ggml_cuda_mmq_get_write_back() {
    return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().write_back;
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int              nwarps     = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              I          = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback>();
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback>();

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + J;
    int * tile_x = tile_y + GGML_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);

#if defined(BLACKWELL_MMA_AVAILABLE)
    // FP4 tile stores 8 blocks
    constexpr int ne_block = (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) ? QK_FP4_MMQ : QK8_1_MMQ;
#else
    constexpr int ne_block = QK8_1_MMQ;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J*I / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J);
    } else {
        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}


// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback), ggml_cuda_mmq_get_occupancy(type, J, fallback))
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx) {

    // Skip unused template specializations for faster compilation:
    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    const uint32_t nty = (nrows_x + I - 1) / I; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {
        const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
        const int wt = tmp2.x;
        const int zt = tmp2.y;
        const int jt = blockIdx.y;
        const int it = blockIdx.x;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
        int offset_y_scale;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
        } else {
            GGML_UNUSED(offset_y_scale);
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) {
                offset_y_scale = 0;
            }

            if (jt*J >= col_diff) {
                return;
            }

            // __syncthreads(); // There is no previous tile that could cause a race condition.
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > J && j >= J) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
            }
            __syncthreads();
        }

        offset_y   += (col_low + jt*J)*(sizeof(block_q8_1_mmq)/sizeof(int));
        offset_dst += it*I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt*J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

        constexpr bool fixup = false;
        mul_mat_q_process_tile<type, J, fallback, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
        return;
    }

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        int tmp = fastdiv(kbc, blocks_per_ne00);
        uint2 tmp2 = fast_div_modulo(tmp, ntx);
        const int jt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nchannels_y);
        const int zt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nsamples_y);
        const int wt = tmp2.y;
        const int it = tmp2.x;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
        int offset_y_scale;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
        } else {
            GGML_UNUSED(offset_y_scale);
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) {
                offset_y_scale = 0;
            }

            if (jt*J >= col_diff) {
                kbc += blocks_per_ne00.z;
                kbc -= fastmodulo(kbc, blocks_per_ne00);

                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));

                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > J && j >= J) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it*I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt * J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<type, J, fallback, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = fastdiv(kbc, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    // Defaults for regular matrix multiplication:
    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
    int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
    int offset_y_scale;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
    } else {
        GGML_UNUSED(offset_y_scale);
    }

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y   = 0;
        offset_dst = 0;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = 0;
        }

        if (jt*J >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
            const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

            if (j0 + nwarps*warp_size > J && j >= J) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it*I;
    const float * y_scale_tile = nullptr;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale += col_low + jt * J;
        y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
    }

    const int tile_x_max_i = nrows_x  - it*I - 1;
    const int tile_y_max_j = col_diff - jt*J - 1;

    const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<type, J, fallback, fixup>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

template <ggml_type type, int J, bool fallback>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback)/2, 1)
static __global__ void mul_mat_q_stream_k_fixup(
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
        float * __restrict__ tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
        const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
        const int stride_sample_dst, const uint3 ntx) {
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback) / 2) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J / nwarps] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int nty = (nrows_x + I - 1) / I;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;


#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(J*I) + j*I + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    int tmp = fastdiv(kbc0, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    if (!ids_dst) {
        const int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst + it*I;
        dst += offset_dst;

        const int i_max = nrows_x   - it*I - 1;
        const int j_max = ncols_dst - jt*J - 1;
        if (fallback && i > i_max) {
            return;
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j > j_max) {
                return;
            }

            dst[j*stride_col_dst + i] += sum[j0/nwarps];
        }
        return;
    }

    __shared__ int ids_dst_shared[J];
    const int col_low  = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;

    for (int j = threadIdx.y*warp_size + threadIdx.x; j < J; j += nwarps*warp_size) {
        ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
    }
    __syncthreads();

    const int offset_dst = it*I;
    dst += offset_dst;

    const int i_max = nrows_x  - it*I - 1;
    const int j_max = col_diff - jt*J - 1;
    if (fallback && i > i_max) {
        return;
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

        dst[ids_dst_shared[j]*stride_col_dst + i] += sum[j0/nwarps];
    }
}

struct mmq_args {
    const char * x; ggml_type type_x; const int * y; const int32_t * ids_dst; const int32_t * expert_bounds; float * dst;
    const float * y_scale;
    int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst; int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
    int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
    int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
    int64_t ncols_max;
};

static size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config & config, const int cc) {
    const size_t nbs_ids = config.J*sizeof(int);
    const size_t nbs_x = ggml_cuda_mmq_get_nbytes_shared_x(config, cc);
    const size_t nbs_y = config.J * (sizeof(block_q8_1_mmq));
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, config.nthreads*sizeof(int));
}

template <ggml_type type, int J, bool fallback>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    GGML_ASSERT(config.nthreads % warp_size == 0);
    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);

    const dim3 block_dims(warp_size, nwarps, 1);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J,  true>), nbytes_shared);

    const int nty  = (args.nrows_x   + config.I - 1) / config.I;
    const int ntx  = (args.ncols_max + config.J - 1) / config.J;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 block_nums_xy_tiling(nty, ntx, ntzw);

    GGML_ASSERT(args.nchannels_y % args.nchannels_x == 0);
    GGML_ASSERT(args.nsamples_y  % args.nsamples_x  == 0);
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio  = args.nsamples_y  / args.nsamples_x;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const uint3 ntx_fd             = init_fastdiv_values(ntx);
    const uint3 nchannels_y_fd     = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);

    if (!ggml_cuda_mmq_get_stream_k(type, J, fallback, cc)) {
        mul_mat_q<type, J, fallback><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr, args.y_scale,
             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
             ntx_fd);
        return;
    }

    // For the stream-k kernel it is possible to run it with tiling by setting the number of CUDA blocks equal to the number of tiles.
    // This is worthwhile if the efficiency of tiling is high and skipping the fixup kernel is more important.
    const int ntiles_dst = ntx * nty * ntzw;
    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
    const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);

    GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30)); // Assert that variable kbc will not overflow.

    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;

    ggml_cuda_pool & pool = ctx.pool(id);
    ggml_cuda_pool_alloc<float> tmp_fixup(pool);
    if (fixup_needed) {
        tmp_fixup.alloc(block_nums_stream_k.x * config.J*config.I);
    }

    const dim3 block_nums_fixup(block_nums_stream_k.x, config.I/warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y/2, block_dims.z);

    mul_mat_q<type, J, fallback><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.y_scale,
         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
         ntx_fd);

    if (!fixup_needed) {
        return;
    }

    CUDA_CHECK(cudaGetLastError());
    mul_mat_q_stream_k_fixup<type, J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
        (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
         args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
         ntx_fd);
}

template <ggml_type type, bool fallback>
void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ncols_max + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }

    switch (J_best) {
        case   8:
            launch_mul_mat_q<type,   8, fallback>(ctx, args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16, fallback>(ctx, args, stream);
            break;
        case  24:
            launch_mul_mat_q<type,  24, fallback>(ctx, args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32, fallback>(ctx, args, stream);
            break;
        case  40:
            launch_mul_mat_q<type,  40, fallback>(ctx, args, stream);
            break;
        case  48:
            launch_mul_mat_q<type,  48, fallback>(ctx, args, stream);
            break;
        case  56:
            launch_mul_mat_q<type,  56, fallback>(ctx, args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64, fallback>(ctx, args, stream);
            break;
        case  72:
            launch_mul_mat_q<type,  72, fallback>(ctx, args, stream);
            break;
        case  80:
            launch_mul_mat_q<type,  80, fallback>(ctx, args, stream);
            break;
        case  88:
            launch_mul_mat_q<type,  88, fallback>(ctx, args, stream);
            break;
        case  96:
            launch_mul_mat_q<type,  96, fallback>(ctx, args, stream);
            break;
        case 104:
            launch_mul_mat_q<type, 104, fallback>(ctx, args, stream);
            break;
        case 112:
            launch_mul_mat_q<type, 112, fallback>(ctx, args, stream);
            break;
        case 120:
            launch_mul_mat_q<type, 120, fallback>(ctx, args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128, fallback>(ctx, args, stream);
            break;
        default:
            fprintf(stderr, "J_best=%d\n", J_best);
            GGML_ABORT("fatal error");
            break;
    }
}

template <ggml_type type>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    if (args.nrows_x % 128 == 0) {
        constexpr bool fallback = false;
        mul_mat_q_switch_J<type, fallback>(ctx, args, stream);
    } else {
        constexpr bool fallback = true;
        mul_mat_q_switch_J<type, fallback>(ctx, args, stream);
    }
}

#define DECL_MMQ_CASE(type)                                                        \
    template void mul_mat_q_case<type>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \

extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q8_0);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q6_K);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_IQ1_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ2_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
extern DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_NL);
extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);
// -----------------------------------------
extern DECL_MMQ_CASE(GGML_TYPE_MXFP4);
extern DECL_MMQ_CASE(GGML_TYPE_NVFP4);

// -------------------------------------------------------------------------------------------------------------------------

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts);
