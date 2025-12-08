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
GGML_BACKEND_API ggml_backend_t ggml_backend_sycl_init(int device);

GGML_BACKEND_API bool ggml_backend_is_sycl(ggml_backend_t backend);

// devide buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);

// tensor parallel buffer type (Megatron-style column/row parallel with all-reduce)
// Initializes TP system on first call. Pass device_ids=NULL for auto-detection.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_tp_buffer_type(int n_devices, const int * device_ids);

// Get the TP world size (number of devices in TP group, 1 if TP not enabled)
GGML_BACKEND_API int ggml_backend_sycl_get_tp_world_size(void);

// Get the TP rank for this process (0 if TP not enabled or single-process mode)
GGML_BACKEND_API int ggml_backend_sycl_get_tp_rank(void);

// Check if running in multi-process TP mode
GGML_BACKEND_API bool ggml_backend_sycl_is_multiprocess_tp(void);

// Get the byte offset for reading this rank's shard from GGUF file
// For column-parallel tensors, this is the offset into the tensor data
// For row-parallel tensors, returns 0 (requires special handling due to interleaved data)
// tensor_name: the tensor name to check TP layer type
// tensor_ne: original tensor dimensions [ne0, ne1, ne2, ne3]
// tensor_type: ggml_type of the tensor
GGML_BACKEND_API size_t ggml_backend_sycl_get_tp_data_offset(
    const char * tensor_name,
    const int64_t * tensor_ne,
    enum ggml_type tensor_type);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);

// Host compute buffer type - uses SYCL host memory (malloc_host) with SYCL buffer interface
// This is used for TP compute buffers to allow cross-device data sharing.
// Unlike host_buffer_type, this uses the SYCL buffer interface so it works with SYCL kernels.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_host_compute_buffer_type(int device);

GGML_BACKEND_API void ggml_backend_sycl_print_sycl_devices(void);
GGML_BACKEND_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len);
GGML_BACKEND_API void ggml_backend_sycl_get_device_description(int device,
                                                       char *description,
                                                       size_t description_size);
GGML_BACKEND_API int  ggml_backend_sycl_get_device_count();
GGML_BACKEND_API void ggml_backend_sycl_get_device_memory(int device, size_t *free, size_t *total);

// SYCL doesn't support registering host memory, keep here for reference
// GGML_BACKEND_API bool ggml_backend_sycl_register_host_buffer(void * buffer, size_t size);
// GGML_BACKEND_API void ggml_backend_sycl_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_sycl_reg(void);

// Flash attention sequence IDs for multi-sequence batching
// Set host pointers for seq_ids arrays (called from llama layer before graph execution)
// These are stored in thread-local storage and used by fattn kernel
// The pointers must be valid USM host memory (allocated by SYCL_Host buffer)
GGML_BACKEND_API void ggml_backend_sycl_set_seq_ids_host(
    const int32_t * q_seq_ids, size_t q_count,
    const int32_t * kv_seq_ids, size_t kv_count);

// Clear the seq_ids host pointers (called after graph execution)
GGML_BACKEND_API void ggml_backend_sycl_clear_seq_ids_host(void);

#ifdef  __cplusplus
}
#endif
