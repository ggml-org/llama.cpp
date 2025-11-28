#pragma once

#include "llama-safetensors.h"
#include "ggml.h"

#include <cstddef>

// Convert safetensors dtype to GGML type
ggml_type safetensors_dtype_to_ggml_type(safetensors_dtype dtype);

// Get GGML type name
const char * ggml_type_name_safe(ggml_type type);

// Convert safetensors tensor data to GGML format
// dst_data must be pre-allocated with enough space
// Returns true on success
bool convert_safetensors_to_ggml(
    const void * src_data,
    size_t src_size,
    safetensors_dtype src_dtype,
    void * dst_data,
    size_t dst_size,
    ggml_type dst_type,
    const int64_t * shape,
    int n_dims
);

// Calculate tensor size in bytes for GGML type
size_t ggml_tensor_size(ggml_type type, const int64_t * shape, int n_dims);
