#pragma once

#include <cstddef>
#include <cstdint>

#if !defined(GGML_COMMON_DECL)
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define GGML_COMMON_DECL_CUDA
#else
#define GGML_COMMON_DECL_CPP
#endif
#endif
#include "ggml-common.h"

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GGML_ALIGN_16 __align__(16)
#else
#define GGML_ALIGN_16 alignas(16)
#endif

static constexpr int GGML_NVFP4_CUDA_LANES = QK_K / QK_NVFP4; // 4 packs in one 256 block
static_assert(GGML_NVFP4_CUDA_LANES == 4, "unexpected NVFP4 CUDA lane count");

// Full groups use 4 x 64 lane blocks
// Tails keep the compact row size
struct GGML_ALIGN_16 block_nvfp4_cuda {
    uint8_t qs[GGML_NVFP4_CUDA_LANES][QK_NVFP4 / 2];
    uint8_t scales[GGML_NVFP4_CUDA_LANES][QK_NVFP4 / QK_NVFP4_SUB];
};

static_assert(sizeof(block_nvfp4_cuda) == 144, "unexpected nvfp4 cuda block size");
static_assert(alignof(block_nvfp4_cuda) == 16, "nvfp4 cuda block must be 16B aligned");

static inline uint32_t ggml_cuda_nvfp4_pack(const uint8_t src[32], int pack) {
    const int scale = pack >> 1;
    const int nibble_shift = (pack & 1) << 2;
    uint32_t out = 0;
    for (int value = 0; value < 8; ++value) {
        const uint32_t nibble = (uint32_t) ((src[scale * 8 + value] >> nibble_shift) & 0x0F); // 8 fp4 codes into one u32
        out |= (nibble << (4 * value));
    }
    return out;
}

static inline uint8_t ggml_cuda_nvfp4_unpack(uint32_t packed, int value) {
    return (packed >> (4 * value)) & 0x0F;
}

void ggml_cuda_repack_rows_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst);
void ggml_cuda_unpack_rows_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst);
