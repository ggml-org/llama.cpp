#pragma once

#include <stdint.h>
#include <stddef.h>

// Tensor Memory Accelerator (TMA) descriptor helpers for Blackwell (SM 100+).
//
// The TMA pipeline allows the GPU to initiate direct memory transfers from
// pinned system RAM using cp.async.bulk PTX. Transfers are described by
// 16-byte TMA descriptors loaded into the TMA descriptor cache.
//
// NOTE: The exact TMA descriptor bit layout is not publicly documented by
// NVIDIA. The descriptors below encode the parameters required by
// cp.async.bulk PTX. For production use, NVIDIA's built-in TMA descriptor
// generation (via .tma.surface.desc or the linker) should be preferred.

#pragma pack(push, 1)
struct ggml_cuda_tma_desc {
    uint64_t d[2];
};
#pragma pack(pop)

static_assert(sizeof(ggml_cuda_tma_desc) == 16, "TMA descriptor must be 16 bytes");

// TMA surface types for descriptor encoding
enum ggml_cuda_tma_surface {
    GGML_CUDA_TMA_SURFACE_1D = 0,
    GGML_CUDA_TMA_SURFACE_2D = 1,
};

// -----------------------------------------------------------------------------
// Host-side descriptor creation
// -----------------------------------------------------------------------------

#ifndef __CUDA_ARCH__

/**
 * Create a TMA load descriptor for a 2D tensor (rows x cols).
 *
 * @param base_ptr        Pinned system RAM address (must be page-locked, 4K-aligned)
 * @param rows            Number of rows
 * @param cols            Number of columns
 * @param type_size       Element size in bytes (1, 2, 4, 8)
 * @param row_stride_bytes Leading dimension in bytes (pitch)
 * @return 16-byte TMA descriptor
 */
static inline ggml_cuda_tma_desc ggml_cuda_tma_make_load_desc_2d(
    const void* base_ptr,
    int64_t rows, int64_t cols,
    size_t type_size,
    int64_t row_stride_bytes) {

    ggml_cuda_tma_desc desc = {};
    uint64_t addr = (uint64_t)base_ptr & 0xFFFFFFFFFFFFUL; // lower 48 bits of VA

    // Encode address + dimensions for use with cp.async.bulk PTX
    desc.d[0] = addr | ((uint64_t)(cols & 0xFFFF) << 48);
    desc.d[1] = ((uint64_t)row_stride_bytes & 0xFFFFFFFFFFFFUL) |
                ((uint64_t)(rows & 0xFFFF) << 48);

    return desc;
}

/**
 * Create a TMA load descriptor for a 1D buffer.
 *
 * @param base_ptr  Pinned system RAM address (must be page-locked, 4K-aligned)
 * @param num_bytes Total buffer size in bytes
 * @return 16-byte TMA descriptor
 */
static inline ggml_cuda_tma_desc ggml_cuda_tma_make_load_desc_1d(
    const void* base_ptr,
    size_t num_bytes) {

    ggml_cuda_tma_desc desc = {};
    uint64_t addr = (uint64_t)base_ptr & 0xFFFFFFFFFFFFUL;
    desc.d[0] = addr;
    desc.d[1] = num_bytes & 0xFFFFFFFFFFFFUL;
    return desc;
}

#endif // !__CUDA_ARCH__

// -----------------------------------------------------------------------------
// Device-side TMA operations (requires SM 100+)
// -----------------------------------------------------------------------------

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 1000

/**
 * Issue a TMA async bulk commit. This tells the TMA engine to process all
 * previously issued cp.async.bulk requests. Must be called after the
 * descriptor and data have been set up.
 */
__device__ __forceinline__ void ggml_cuda_tma_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

/**
 * Wait for TMA group completion with priority.
 * @param priority  Completion priority (0-31, higher = more urgent)
 * @param complete  Wait mode: 1 = wait until complete, 0 = no wait
 */
__device__ __forceinline__ void ggml_cuda_tma_wait_group(uint32_t priority, uint32_t complete) {
    asm volatile("cp.async.bulk.wait_group %0, %1, -1, 1;\n"
                 :: "n"(priority), "n"(complete));
}

#endif // __CUDA_ARCH__ >= 1000
#endif // __CUDA_ARCH__
