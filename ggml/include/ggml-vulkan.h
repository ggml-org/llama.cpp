#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME        "Vulkan"
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

typedef struct {
    char     device_name[256];
    uint32_t vendor_id;
    uint32_t device_id;
    uint64_t total_device_local_memory;
    bool     has_memory_budget_ext;
    bool     supports_float16;
    bool     supports_16bit_storage;
    uint32_t api_version;
} ggml_vk_device_info;

GGML_BACKEND_API ggml_vk_device_info ggml_backend_vk_get_device_info(int device);
GGML_BACKEND_API int                 ggml_backend_vk_get_default_gpu_layers(int device, int default_layers);

#ifdef __cplusplus
}
#endif
