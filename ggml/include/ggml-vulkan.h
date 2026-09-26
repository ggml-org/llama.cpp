#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// MoE cache support: expose the raw handles of a Vulkan backend so the
// cache provider can create its own buffers/pipelines on the same device.
// Returns VK_NULL_HANDLE / 0 when the backend is not a Vulkan backend.
// Only declared when <vulkan/vulkan.h> was included first, so that
// non-Vulkan builds of this header stay free of the Vulkan headers.
#if defined(VK_VERSION_1_0)
GGML_BACKEND_API VkDevice      ggml_backend_vk_get_device_handle(ggml_backend_t backend);
GGML_BACKEND_API VkQueue       ggml_backend_vk_get_queue_handle(ggml_backend_t backend);
GGML_BACKEND_API VkPhysicalDevice ggml_backend_vk_get_physical_device(ggml_backend_t backend);
GGML_BACKEND_API uint32_t      ggml_backend_vk_get_queue_family(ggml_backend_t backend);
#endif

#ifdef  __cplusplus
}
#endif
