#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// Tiered-memory backend. The device is intentionally registered as an accelerator
// so it is only selected explicitly. Activations use the normal CUDA buffer type;
// model weights use a custom buffer that can independently reside in VRAM, map
// registered GGUF pages, or stage selected MoE experts from pageable storage.
enum ggml_cuda_tiered_memory_tier {
    GGML_CUDA_TIERED_MEMORY_VRAM = 0,
    GGML_CUDA_TIERED_MEMORY_DRAM = 1,
    GGML_CUDA_TIERED_MEMORY_SSD  = 2,
};

struct ggml_cuda_tiered_tensor_plan {
    const char * name;
    enum ggml_cuda_tiered_memory_tier tier;
};

struct ggml_cuda_tiered_plan_options {
    // Reserved for a future persistent expert cache. The current implementation
    // releases sparse VMM mappings after each streamed operation.
    size_t ssd_cache_bytes;

    // Treat unsupported tensor layouts or unavailable CUDA facilities as hard
    // failures instead of falling back to VRAM.
    bool strict;
};

// Registers the tiered devices for statically linked backends. Dynamic CUDA
// backends register them when the module is loaded. Safe to call repeatedly.
GGML_BACKEND_API void ggml_backend_cuda_tiered_register(void);

// These functions are also exposed from the tiered device registry through
// ggml_backend_reg_get_proc_address using the same names.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_tiered_plan_begin(
        ggml_backend_dev_t dev,
        const struct ggml_cuda_tiered_tensor_plan * entries,
        size_t n_entries,
        struct ggml_cuda_tiered_plan_options options);

GGML_BACKEND_API void ggml_backend_cuda_tiered_plan_end(ggml_backend_dev_t dev);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
