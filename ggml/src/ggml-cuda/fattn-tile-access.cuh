#pragma once

#include "common.cuh"
#include <type_traits>

// The default tag keeps native pointer aliasing and compiler resource use separate from new readers.
struct ggml_cuda_fattn_tile_native_rows {};

// Native and complete-layer readers do not checkpoint or alter their loop bounds.
template<typename KV> struct ggml_cuda_fattn_tile_resume_traits { static constexpr bool enabled = false; };

struct ggml_cuda_fattn_tile_contiguous_f16_rows {
    const half2 * data;
    int stride;

    // Keep native affine addressing and zero selection; encoded/span readers can supply the same copy contract.
    template<int bytes>
    __device__ __forceinline__ void load(half2 * destination, int row, int pair, bool valid, const half2 * zero) const {
        ggml_cuda_memcpy_1<bytes>(destination,valid ? data+row*stride+pair : zero);
    }
};
