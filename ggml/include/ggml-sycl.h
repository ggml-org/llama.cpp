//
//  MIT license
//  Copyright (C) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#define GGML_SYCL_NAME "SYCL"
#define GGML_SYCL_MAX_DEVICES 48

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_API ggml_backend_t ggml_backend_sycl_init(int device);

// devide buffer
GGML_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);

GGML_API void   ggml_backend_sycl_print_sycl_devices(void);
GGML_API GGML_CALL void   ggml_sycl_get_gpu_list(int *id_list, int max_len);
GGML_API GGML_CALL void   ggml_sycl_get_device_description(int device, char *description, size_t description_size);
GGML_API GGML_CALL int   ggml_backend_sycl_get_device_count();
GGML_API GGML_CALL void ggml_backend_sycl_get_device_memory(int device, size_t *free, size_t *total);

// SYCL doesn't support registering host memory, keep here for reference
// GGML_API GGML_CALL bool ggml_backend_sycl_register_host_buffer(void * buffer, size_t size);
// GGML_API GGML_CALL void ggml_backend_sycl_unregister_host_buffer(void * buffer);
#ifdef  __cplusplus
}
#endif
