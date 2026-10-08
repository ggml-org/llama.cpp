#pragma once

#include "ggml.h"
#include "graph/value-map.h"

#include <cstddef>
#include <limits>

namespace ggml::hrx {

inline bool strided_f32_storage_span_bytes(const Value & value, size_t & byte_count) {
    if (value.type != GGML_TYPE_F32 || value.nb[0] != sizeof(float)) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (value.nb[i] % sizeof(float) != 0) {
            return false;
        }
    }

    size_t max_offset = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0) {
            return false;
        }
        const size_t extent = static_cast<size_t>(value.ne[i] - 1);
        if (extent != 0 && value.nb[i] > std::numeric_limits<size_t>::max() / extent) {
            return false;
        }
        const size_t dim_offset = extent * value.nb[i];
        if (max_offset > std::numeric_limits<size_t>::max() - dim_offset) {
            return false;
        }
        max_offset += dim_offset;
    }
    if (max_offset > std::numeric_limits<size_t>::max() - sizeof(float)) {
        return false;
    }

    byte_count = max_offset + sizeof(float);
    if (value.storage_offset > std::numeric_limits<size_t>::max() - byte_count) {
        return false;
    }
    return value.storage_offset + byte_count <= value.storage_byte_count;
}

inline bool strided_f32_storage_span_elements(const Value & value, size_t & element_count) {
    size_t byte_count = 0;
    if (!strided_f32_storage_span_bytes(value, byte_count) || byte_count % sizeof(float) != 0) {
        return false;
    }
    element_count = byte_count / sizeof(float);
    return true;
}

}  // namespace ggml::hrx
