#include "llama-safetensors-types.h"

#include "ggml.h"

#include <cstring>
#include <cstdint>

ggml_type safetensors_dtype_to_ggml_type(safetensors_dtype dtype) {
    switch (dtype) {
        case safetensors_dtype::F32:  return GGML_TYPE_F32;
        case safetensors_dtype::F16:  return GGML_TYPE_F32;  // Convert to F32 for CPU compatibility
        case safetensors_dtype::BF16: return GGML_TYPE_F32;  // Convert to F32 for CPU compatibility
        case safetensors_dtype::I32:  return GGML_TYPE_I32;
        case safetensors_dtype::I16:  return GGML_TYPE_I16;
        case safetensors_dtype::I8:   return GGML_TYPE_I8;
        case safetensors_dtype::U8:   return GGML_TYPE_I8;  // Map to I8, handle signedness
        // Note: GGML doesn't have direct equivalents for all types
        case safetensors_dtype::F64:  return GGML_TYPE_F32;  // Downcast to F32
        case safetensors_dtype::I64:  return GGML_TYPE_I32;  // Downcast to I32
        case safetensors_dtype::BOOL: return GGML_TYPE_I8;   // Map to I8
        default: return GGML_TYPE_COUNT;  // Invalid
    }
}

const char * ggml_type_name_safe(ggml_type type) {
    if (type < GGML_TYPE_COUNT) {
        return ggml_type_name(type);
    }
    return "INVALID";
}

size_t ggml_tensor_size(ggml_type type, const int64_t * shape, int n_dims) {
    if (n_dims == 0 || !shape) {
        return 0;
    }

    int64_t n_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        n_elements *= shape[i];
    }

    size_t type_size = ggml_type_size(type);
    size_t row_size = ggml_row_size(type, shape[0]);

    // For quantized types, use row_size calculation
    if (type >= GGML_TYPE_Q4_0 && type < GGML_TYPE_COUNT) {
        if (n_dims == 1) {
            return row_size;
        }
        // Calculate total size for multi-dimensional tensors
        int64_t n_rows = n_elements / shape[0];
        return row_size * n_rows;
    }

    // For standard types
    return type_size * n_elements;
}

bool convert_safetensors_to_ggml(
    const void * src_data,
    size_t src_size,
    safetensors_dtype src_dtype,
    void * dst_data,
    size_t dst_size,
    ggml_type dst_type,
    const int64_t * shape,
    int n_dims
) {
    if (!src_data || !dst_data || !shape || n_dims == 0) {
        return false;
    }

    int64_t n_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        n_elements *= shape[i];
    }

    // Direct copy for matching types
    if ((src_dtype == safetensors_dtype::F32 && dst_type == GGML_TYPE_F32) ||
        (src_dtype == safetensors_dtype::F16 && dst_type == GGML_TYPE_F16) ||
        (src_dtype == safetensors_dtype::BF16 && dst_type == GGML_TYPE_BF16)) {

        if (src_size != dst_size) {
            return false;
        }
        std::memcpy(dst_data, src_data, src_size);
        return true;
    }

    // Type conversion required
    // F64 -> F32
    if (src_dtype == safetensors_dtype::F64 && dst_type == GGML_TYPE_F32) {
        const double * src = (const double *)src_data;
        float * dst = (float *)dst_data;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = (float)src[i];
        }
        return true;
    }

    // F32 -> F16
    if (src_dtype == safetensors_dtype::F32 && dst_type == GGML_TYPE_F16) {
        const float * src = (const float *)src_data;
        ggml_fp16_t * dst = (ggml_fp16_t *)dst_data;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = ggml_fp32_to_fp16(src[i]);
        }
        return true;
    }

    // F16 -> F32
    if (src_dtype == safetensors_dtype::F16 && dst_type == GGML_TYPE_F32) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)src_data;
        float * dst = (float *)dst_data;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = ggml_fp16_to_fp32(src[i]);
        }
        return true;
    }

    // BF16 -> F32
    if (src_dtype == safetensors_dtype::BF16 && dst_type == GGML_TYPE_F32) {
        const uint16_t * src = (const uint16_t *)src_data;
        float * dst = (float *)dst_data;
        for (int64_t i = 0; i < n_elements; i++) {
            // BF16 to F32: shift left 16 bits
            uint32_t f32_bits = ((uint32_t)src[i]) << 16;
            float f32_value;
            memcpy(&f32_value, &f32_bits, sizeof(float));
            dst[i] = f32_value;
        }
        return true;
    }

    // I64 -> I32
    if (src_dtype == safetensors_dtype::I64 && dst_type == GGML_TYPE_I32) {
        const int64_t * src = (const int64_t *)src_data;
        int32_t * dst = (int32_t *)dst_data;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = (int32_t)src[i];
        }
        return true;
    }

    // I32 -> I32, I16 -> I16, I8 -> I8 (direct copy)
    if ((src_dtype == safetensors_dtype::I32 && dst_type == GGML_TYPE_I32) ||
        (src_dtype == safetensors_dtype::I16 && dst_type == GGML_TYPE_I16) ||
        (src_dtype == safetensors_dtype::I8 && dst_type == GGML_TYPE_I8) ||
        (src_dtype == safetensors_dtype::U8 && dst_type == GGML_TYPE_I8)) {

        size_t expected_size = n_elements * ggml_type_size(dst_type);
        if (src_size < expected_size || dst_size < expected_size) {
            return false;
        }
        std::memcpy(dst_data, src_data, expected_size);
        return true;
    }

    // BOOL -> I8
    if (src_dtype == safetensors_dtype::BOOL && dst_type == GGML_TYPE_I8) {
        const uint8_t * src = (const uint8_t *)src_data;
        int8_t * dst = (int8_t *)dst_data;
        for (int64_t i = 0; i < n_elements; i++) {
            dst[i] = src[i] ? 1 : 0;
        }
        return true;
    }

    // Unsupported conversion
    return false;
}
