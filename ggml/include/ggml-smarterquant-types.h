#pragma once

#include <stdint.h>
#include <stdbool.h>
// Forward declare ggml_type if it's not pulled in by stdint/stdbool,
// though it's better if this file can be self-contained for basic types
// or include a minimal ggml_core_types.h if one existed.
// For now, assuming ggml_type will be known by consumers including this after ggml.h,
// or we might need to include "ggml_core.h" or similar if such a thing exists
// that defines ggml_type without pulling all of ggml.h.
// Given its usage in ggml_tensor, it should be fine.

// C-compatible structure for SmarterQuant tensor information
struct SmarterQuantTensorInfo {
    // Specifies the ggml_type (as int8_t for storage, cast to enum ggml_type for use)
    // for each of the first four 256-column-wide blocks of the tensor.
    // Subsequent blocks will use the type specified at index 3.
    int8_t compression_types[4];

    // Defines how columns of the original tensor should be reordered.
    // Points to an array of column indices.
    // The element at new_data[col_idx_new] comes from original_data[column_permutation[col_idx_new]].
    // This memory must be managed externally (e.g., by the code loading the configuration).
    int32_t * column_permutation; // Using int32_t as column indices are usually within this range
    int64_t n_cols_for_permutation; // Number of elements in column_permutation array, should match tensor's ne[0]

    // Flag indicating if SmarterQuant is enabled for this tensor.
    bool enabled;
};
