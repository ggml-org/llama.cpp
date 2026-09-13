#pragma once

#include <stdint.h>

#if defined(_WIN32)
#    if defined(GGML_VULKAN_ONEDNN_BUILD)
#        define GGML_VULKAN_ONEDNN_API __declspec(dllexport)
#    else
#        define GGML_VULKAN_ONEDNN_API __declspec(dllimport)
#    endif
#else
#    define GGML_VULKAN_ONEDNN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Run one folded-GQA compressed-KV attention operation on the selected GPU.
// Q is [16,Q,256], K is [4,256,KV], V is [4,KV,256].
GGML_VULKAN_ONEDNN_API int ggml_vulkan_onednn_sdpa(
        int q, int kv,
        const uint16_t * query,
        const int8_t * key,
        const uint16_t * key_scale,
        const int8_t * value,
        const uint16_t * value_scale,
        const uint16_t * mask,
        float divisor,
        float * output);

#ifdef __cplusplus
}
#endif
