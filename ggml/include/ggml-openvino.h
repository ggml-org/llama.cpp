#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_API ggml_backend_t ggml_backend_openvino_init(int device);

GGML_API bool ggml_backend_is_openvino(ggml_backend_t backend);

// device buffer
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_split_buffer_type(const float *tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU
// and GPU
GGML_API ggml_backend_buffer_type_t
ggml_backend_openvino_host_buffer_type(void);

// GGML_API int ggml_backend_openvino_get_device_count(void);
// GGML_API void ggml_backend_openvino_get_device_description(int device,
//                                                        char *description,
//                                                        size_t
//                                                        description_size);
// GGML_API void ggml_backend_openvino_get_device_memory(int device, size_t
// *free,
//                                                   size_t *total);

// GGML_API bool ggml_backend_openvino_register_host_buffer(void *buffer, size_t
// size); GGML_API void ggml_backend_openvino_unregister_host_buffer(void
// *buffer);

GGML_API ggml_backend_reg_t ggml_backend_openvino_reg(void);

#ifdef __cplusplus
}
#endif
