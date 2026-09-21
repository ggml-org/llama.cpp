#pragma once

#include <cstdint>

// Device-visible physical extent for one ordered interval of a logical K/V sequence.
struct ggml_cuda_kv_span {
    const char * k = nullptr;
    const char * v = nullptr;
    int32_t first = 0;
    int32_t tokens = 0;
    int64_t k_token_stride = 0;
    int64_t v_token_stride = 0;
    int64_t k_head_stride = 0;
    int64_t v_head_stride = 0;
};
