#pragma once

#include <cstddef>
#include <cstdint>

#define GGML_CUDA_VENDOR_NVIDIA_MMA 0
#define GGML_CUDA_VENDOR_REDUCE_ADD 0

struct ggml_maca_policy {
    static constexpr bool supports_mmf = false;
    static constexpr bool supports_mmvq = false;
    static constexpr bool supports_mmvq_fusion = false;
    static constexpr bool supports_transposed_mmvf = false;

    static constexpr __host__ __device__ int mmvq_warp_size(int) {
        return 32;
    }
};

static __device__ __forceinline__ int ggml_maca_dp4a(int a, int b, int c) {
    const char4 va = reinterpret_cast<const char4 &>(a);
    const char4 vb = reinterpret_cast<const char4 &>(b);
    return MC_KERN_FUNC(sdot4)(va, vb, c, false);
}

#define GGML_CUDA_VENDOR_POLICY ggml_maca_policy
#define GGML_CUDA_VENDOR_DP4A(a, b, c) ggml_maca_dp4a(a, b, c)
