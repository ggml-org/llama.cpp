#pragma once

#include <string>
#include <vector>
#include <unordered_map>

// Structure to hold the smarter quantization info for a single tensor,
// typically loaded from a JSON configuration file (e.g., default.smarterquant.json).
struct SmarterQuantTensorInfo {
    // Specifies the ggml_type for each of the first four 256-column-wide blocks of the tensor.
    // Subsequent blocks will use the type specified at index 3.
    // Example: {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0}
    std::vector<int8_t> compression_types;

    // Defines how columns of the original tensor should be reordered before quantization.
    // If empty, no permutation is applied. Otherwise, for a tensor with C columns,
    // this vector must contain C unique integers from 0 to C-1.
    // The element at new_data[col_idx_new] comes from original_data[column_permutation[col_idx_new]].
    std::vector<int> column_permutation;

    // Flag indicating if SmarterQuant is enabled for this tensor, either via JSON config or GGUF metadata.
    bool enabled = false;
};

// Map from tensor name (std::string) to its SmarterQuantTensorInfo.
// This is the primary data structure holding the parsed smarter quantization configuration.
using SmarterQuantConfig = std::unordered_map<std::string, SmarterQuantTensorInfo>;

// Function to load and parse a smarter quantization JSON configuration file.
// The file should be a JSON object where keys are tensor names and values are
// 2-element arrays:
//   1. An array of 4 integers (ggml_type enums) for the first four 256-column blocks.
//   2. An array of integers for column permutation (can be empty).
// Example:
// {
//   "blk.0.attn_q.weight": [
//     [10, 11, 12, 13],  // compression_types (ggml_type values)
//     [0, 2, 1, 3, ...] // column_permutation
//   ]
// }
// Implemented in llama-quant.cpp.
SmarterQuantConfig load_smarter_quant_config(const std::string & fname);
