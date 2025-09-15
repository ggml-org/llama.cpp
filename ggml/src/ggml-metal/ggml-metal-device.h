#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// device
//

typedef struct ggml_metal_device * ggml_metal_device_t;

struct ggml_metal_device_props {
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

ggml_metal_device_t ggml_metal_device_init(void);
void ggml_metal_device_free(ggml_metal_device_t ctx);

// return a singleton that is automatically destroyed when the program exits
ggml_metal_device_t ggml_metal_device_get(void);

void * ggml_metal_device_get_device (ggml_metal_device_t ctx); // id<MTLDevice>
void * ggml_metal_device_get_library(ggml_metal_device_t ctx); // id<MTLLibrary>
void * ggml_metal_device_get_queue  (ggml_metal_device_t ctx); // id<MTLCommandQueue>

void ggml_metal_device_get_memory(ggml_metal_device_t ctx, size_t * free, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t ctx, const struct ggml_tensor * op);

struct ggml_metal_device_props ggml_metal_device_get_props(ggml_metal_device_t ctx);

//
// device buffers
//

typedef struct ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t ctx, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t ctx, void * ptr, size_t size, size_t max_tensor_size);

void   ggml_metal_buffer_free     (ggml_metal_buffer_t ctx);
void * ggml_metal_buffer_get_base (ggml_metal_buffer_t ctx);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t ctx);

void   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t ctx, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void   ggml_metal_buffer_clear        (ggml_metal_buffer_t ctx, uint8_t value);


struct ggml_metal_buffer_id {
    void * metal; // id<MTLBuffer>
    size_t offs;
};

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t ctx, const struct ggml_tensor * t);

#ifdef __cplusplus
}
#endif
