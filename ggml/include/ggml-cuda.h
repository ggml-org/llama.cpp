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

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#define NVFP4_A0 0.9918823242f
#define NVFP4_B0 0.9864501953f
#define NVFP4_STEP (1.0f / 32768.0f)

enum nvfp4_cuda_choose46_mode {
    NVFP4_CUDA_CHOOSE46_ADAPTIVE = 0,
    NVFP4_CUDA_CHOOSE46_FORCE_M6 = 1,
    NVFP4_CUDA_CHOOSE46_FORCE_M4 = 2,
};

typedef struct nvfp4_cuda_runtime_cfg {
    int32_t choose46_mode;
    int32_t refit_iters;
    int32_t use_compand_sat;
    int32_t reserved_i32;
    float cap_m6;
    float cap_m4;
} nvfp4_cuda_runtime_cfg;

GGML_BACKEND_API bool nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b);
GGML_BACKEND_API bool nvfp4_autotune_cuda(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, void * stream);

GGML_BACKEND_API void nvfp4_set_ab(float a, float b);
GGML_BACKEND_API void nvfp4_clear_ab(void);

GGML_BACKEND_API bool nvfp4_quantize_cuda(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, void * stream);
GGML_BACKEND_API bool nvfp4_quantize_cuda_ab(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, float a, float b, void * stream);
GGML_BACKEND_API bool nvfp4_quantize_cuda_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    const nvfp4_cuda_runtime_cfg * cfg, void * stream);

#ifdef  __cplusplus
}
#endif
