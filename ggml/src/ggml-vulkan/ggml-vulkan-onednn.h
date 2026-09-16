#pragma once

#include <stdint.h>
#include <stddef.h>

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
// Q is [16,Q,D], K is [4,D,KV], V is [4,KV,D].
GGML_VULKAN_ONEDNN_API int ggml_vulkan_onednn_sdpa(
        int q, int kv, int d,
        const uint16_t * query,
        const int8_t * key,
        const uint16_t * key_scale,
        const int8_t * value,
        const uint16_t * value_scale,
        const uint16_t * mask,
        float divisor,
        float * output);

// Win32 Vulkan exported-memory handles for the SDPA tensors.
struct ggml_vulkan_onednn_win32_allocation {
    void * handle;
    size_t allocation_size;
    size_t query_offset, query_size;
    size_t key_offset, key_size;
    size_t key_scale_offset, key_scale_size;
    size_t value_offset, value_size;
    size_t value_scale_offset, value_scale_size;
    size_t mask_offset, mask_size;
    size_t divisor_offset, divisor_size;
    size_t output_offset, output_size;
    uint64_t allocation_id;
};

// Import Vulkan allocations into Level Zero and execute synchronously.
GGML_VULKAN_ONEDNN_API int ggml_vulkan_onednn_sdpa_win32(
        int q, int kv, int d,
        const struct ggml_vulkan_onednn_win32_allocation * allocation,
        float divisor);

GGML_VULKAN_ONEDNN_API int ggml_vulkan_onednn_release_win32(uint64_t allocation_id);

#ifdef __cplusplus
}
#endif
