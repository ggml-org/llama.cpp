#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_backend_metal_device * ggml_backend_metal_device_t;

struct ggml_backend_metal_device_props {
    char name[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool use_residency_sets;
    bool use_shared_buffers;

    bool supports_gpu_family_apple7;
};

ggml_backend_metal_device_t ggml_backend_metal_device_init(void);
void ggml_backend_metal_device_free(ggml_backend_metal_device_t ctx);

// return a singleton that is automatically destroyed when the program exits
ggml_backend_metal_device_t ggml_backend_metal_device_get(void);

void * ggml_backend_metal_device_get_device (ggml_backend_metal_device_t ctx);
void * ggml_backend_metal_device_get_library(ggml_backend_metal_device_t ctx);
void * ggml_backend_metal_device_get_queue  (ggml_backend_metal_device_t ctx);

void ggml_backend_metal_device_get_memory(ggml_backend_metal_device_t ctx, size_t * free, size_t * total);

struct ggml_backend_metal_device_props ggml_backend_metal_device_get_props(ggml_backend_metal_device_t ctx);

#ifdef __cplusplus
}
#endif
