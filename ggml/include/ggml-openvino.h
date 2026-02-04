#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_OPENVINO_NAME        "OPENVINO"
#define GGML_OPENVINO_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device);

GGML_BACKEND_API bool ggml_backend_is_openvino(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_buffer_is_openvino(ggml_backend_buffer_t buffer);

GGML_BACKEND_API bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft);

GGML_BACKEND_API bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft);

GGML_BACKEND_API size_t ggml_backend_openvino_buffer_get_ctx_id(ggml_backend_buffer_t buffer);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(int device);

GGML_BACKEND_API int ggml_backend_openvino_get_device_count(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_openvino_reg(void);

struct ggml_openvino_device_info {
    int device_count;

    struct openvino_device_info {
        int    cc;               // compute capability
        int    nsm;              // number of streaming multiprocessors
        size_t smpb;             // max. shared memory per block
        size_t smpbo;            // max. shared memory per block (with opt-in)
        bool   vmm;              // virtual memory support
        size_t vmm_granularity;  // granularity of virtual memory
        size_t total_vram;
    };

    openvino_device_info devices[GGML_OPENVINO_MAX_DEVICES] = {};

    std::array<float, GGML_OPENVINO_MAX_DEVICES> default_tensor_split = {};
};

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
const ggml_openvino_device_info & ggml_openvino_info();
#endif
