#pragma once

#define GGML_CUDA_VENDOR_NVIDIA_MMA 0
#define GGML_CUDA_VENDOR_REDUCE_ADD 0

struct ggml_maca_policy {
    static constexpr bool supports_mmf = false;
    static constexpr bool supports_mmvq = false;
    static constexpr bool supports_mmvq_fusion = false;
    static constexpr bool supports_transposed_mmvf = false;
};

#define GGML_CUDA_VENDOR_POLICY ggml_maca_policy
