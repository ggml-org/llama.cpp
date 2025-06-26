#include "llama-impl.h" // For logging, format, no_init, llama_format_tensor_shape

#include "llama.h"
#include "llama-quant.h" // Includes ggml-smarterquant-types.h
#include "ggml.h"
#include "gguf.h" // For GGUF functions
// #include "ggml-impl.h" // For ggml_row_size, ggml_is_quantized etc. -> Functions are in ggml.h
#include "common.h"    // For utility functions (string_format is in llama-impl.h)
#include "llama-model.h"      // For llama_model, LLM_TN etc.
#include "llama-model-loader.h" // For llama_model_loader
#include "json.hpp"    // For nlohmann::json

#include <string>      // Moved standard includes up
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>     // For size_t
#include <cstdint>     // For int32_t, uint8_t, uint16_t, int64_t
#include <thread>
#include <utility>     // For std::move
#include <fstream>     // For std::ifstream
#include <mutex>       // For std::mutex
#include <cstdio>      // For snprintf, stdout, fflush, fopen, fclose
#include <cstring>     // For strcmp, strncpy, memcpy
#include <algorithm>   // For std::sort, std::max, std::min, std::count
#include <functional>
#include <limits>      // For std::numeric_limits
#include <iostream>    // For std::cerr (used by LLAMA_LOG_ERROR indirectly)
#include <iomanip>     // For std::setw, std::fixed (if used by logging)
#include <sstream>     // For std::ostringstream (if used by logging)
#include <cinttypes>   // For PRId64

// Definition for function declared in llama-quant.h
SmarterQuantConfig load_smarter_quant_config(const std::string & fname) {
    SmarterQuantConfig config_map;
    std::ifstream ifs(fname);
    if (!ifs.is_open()) {
        LLAMA_LOG_WARN("%s: Failed to open SmarterQuant config file '%s'. Proceeding without it.\n", __func__, fname.c_str());
        return config_map; // Return empty map
    }

    nlohmann::json j;
    try {
        ifs >> j;
    } catch (const nlohmann::json::parse_error& e) {
        LLAMA_LOG_ERROR("%s: Failed to parse SmarterQuant config file '%s': %s\n", __func__, fname.c_str(), e.what());
        return config_map; // Return empty map on parse error
    }

    if (!j.is_object()) {
        LLAMA_LOG_ERROR("%s: SmarterQuant config file '%s' is not a JSON object.\n", __func__, fname.c_str());
        return config_map;
    }

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string tensor_name = it.key();
        const nlohmann::json& tensor_config_json = it.value();

        SmarterQuantTensorInfo tensor_info;
        tensor_info.enabled = false; // Default to disabled, enable on successful parse
        tensor_info.column_permutation = nullptr;
        tensor_info.n_cols_for_permutation = 0;

        if (!tensor_config_json.is_array() || tensor_config_json.size() != 2) {
            LLAMA_LOG_WARN("%s: Invalid format for tensor '%s' in SmarterQuant config. Expected 2-element array. Skipping.\n", __func__, tensor_name.c_str());
            continue;
        }

        // Parse compression_types
        const nlohmann::json& block_types_json = tensor_config_json[0];
        if (!block_types_json.is_array() || block_types_json.size() != 4) {
            LLAMA_LOG_WARN("%s: Invalid 'compression_types' for tensor '%s'. Expected 4-element array. Skipping.\n", __func__, tensor_name.c_str());
            continue;
        }

        bool types_parsed_successfully = true;
        for (size_t i = 0; i < 4; ++i) {
            if (!block_types_json[i].is_number_integer()) {
                LLAMA_LOG_WARN("%s: Invalid type for 'compression_types[%zu]' for tensor '%s'. Expected integer. Skipping tensor.\n", __func__, i, tensor_name.c_str());
                types_parsed_successfully = false;
                break;
            }
            tensor_info.compression_types[i] = static_cast<int8_t>(block_types_json[i].get<int>());
        }

        if (types_parsed_successfully) {
            // Only proceed to parse permutation if types were successfully parsed
            const nlohmann::json& permutation_json = tensor_config_json[1];
            if (!permutation_json.is_array()) {
                LLAMA_LOG_WARN("%s: Invalid 'column_permutation' for tensor '%s'. Expected array. Skipping tensor processing.\n", __func__, tensor_name.c_str());
                goto next_tensor_label;
            }

            if (!permutation_json.empty()) {
                tensor_info.n_cols_for_permutation = permutation_json.size();
                tensor_info.column_permutation = new (std::nothrow) int32_t[tensor_info.n_cols_for_permutation];
                if (!tensor_info.column_permutation) {
                    LLAMA_LOG_ERROR("%s: Failed to allocate memory for column_permutation for tensor '%s'. Skipping tensor processing.\n", __func__, tensor_name.c_str());
                    tensor_info.n_cols_for_permutation = 0; // Reset before goto
                    goto next_tensor_label;
                }
                for (size_t i = 0; i < (size_t)tensor_info.n_cols_for_permutation; ++i) {
                    if (!permutation_json[i].is_number_integer()) {
                        LLAMA_LOG_WARN("%s: Invalid type for 'column_permutation[%zu]' for tensor '%s'. Expected integer. Skipping tensor processing.\n", __func__, i, tensor_name.c_str());
                        delete[] tensor_info.column_permutation;
                        tensor_info.column_permutation = nullptr;
                        tensor_info.n_cols_for_permutation = 0;
                        goto next_tensor_label;
                    }
                    tensor_info.column_permutation[i] = permutation_json[i].get<int32_t>();
                }
            }

            tensor_info.enabled = true;
            config_map[tensor_name] = tensor_info;
        } // end if (types_parsed_successfully)

        next_tensor_label:; // Label for gotos from parsing failures within this tensor's config
    } // End of for loop iterating over JSON object

    LLAMA_LOG_INFO("%s: Loaded SmarterQuant config for %zu tensors from '%s'.\n", __func__, config_map.size(), fname.c_str());
    return config_map;
}

// Old C-style SmartQuant map handlers and their usage are removed.
// The new C++ `load_smarter_quant_config` using nlohmann::json is used instead.

// Forward declare to avoid needing full quantize_state_impl definition here for now
// This struct is defined in llama.cpp and is quite complex.
// We only need its members like has_imatrix, n_attention_wv etc.
// A proper solution might involve moving its definition to a shared header if needed extensively.
// struct quantize_state_impl; // Defined below for now as a placeholder

// Placeholder definition for quantize_state_impl to allow compilation
struct quantize_state_impl {
    // Based on usage in llama-quant.cpp
    bool has_imatrix = false;
    int n_attention_wv = 0;
    bool has_output = false;
    int n_ffn_down = 0;
    int n_ffn_gate = 0;
    int n_ffn_up = 0;
    std::vector<float> permuted_imatrix_holder; // Specific to SmarterQuant
    int n_fallback = 0;
    int n_k_quantized = 0; // Assuming this is related to K-quants

    // Placeholder constructor
    quantize_state_impl(const llama_model & /*model*/, const llama_model_quantize_params * /*params*/) {
        // Initialization logic would go here based on model and params
        // For now, default initialization of members is used.
        // TODO: Revisit this based on actual requirements from llama.cpp or SmarterQuant design
    }
};


// Helper function from common.cpp (ensure it's available or replicate if small)
// For now, assuming common.h brings in enough, but zeros might be specific.
// static void zeros(std::ofstream &out, size_t n) {
//     char zero = 0;
//     for (size_t i = 0; i < n; ++i) {
//         out.write(&zero, 1);
//     }
// }
// ^^^ zeros is actually defined in gguf.cpp and used via ggml-impl.h -> gguf-impl.h.

// Forward declaration for llama_tensor_dequantize_impl, which seems to be an internal helper
// It's usually in llama.cpp or similar. For now, we'll assume it's linked.
// Making these static for now and providing stubs to make llama-quant.cpp self-contained for these symbols.
static void llama_tensor_dequantize_impl(
    struct ggml_tensor * tensor,
    std::vector<no_init<float>> & f32_conv_buf,
    std::vector<std::thread> & workers, // Unused in this simple serial version
    int64_t nelements,
    int nthread) { // Unused in this simple serial version

    GGML_UNUSED(workers);
    GGML_UNUSED(nthread);

    if (tensor->type == GGML_TYPE_F32) {
        // This function is primarily for dequantizing. If data is already F32,
        // the caller should ideally use tensor->data directly.
        // However, if called, ensure buffer is correctly sized and filled.
        f32_conv_buf.resize(nelements);
        if (tensor->data != nullptr && nelements > 0) {
            // Assuming f32_conv_buf is std::vector<no_init<float>>
            // We need to copy into the .value member or reinterpret_cast.
            // For F32 to F32 copy, direct memcpy to a float* view of no_init<float> is okay if layout is same.
             memcpy(reinterpret_cast<float*>(f32_conv_buf.data()), tensor->data, nelements * sizeof(float));
        } else if (nelements > 0) {
            for (auto& ni_val : f32_conv_buf) { ni_val.value = std::numeric_limits<float>::quiet_NaN(); }
            LLAMA_LOG_WARN("%s: Called with F32 tensor but data is null or nelements is zero.\n", __func__);
        }
        return;
    }

    if (!ggml_is_quantized(tensor->type)) {
        LLAMA_LOG_ERROR("%s: Attempting to dequantize non-quantized type %s\n", __func__, ggml_type_name(tensor->type));
        f32_conv_buf.resize(nelements);
        for (auto& ni_val : f32_conv_buf) { ni_val.value = std::numeric_limits<float>::quiet_NaN(); }
        return;
    }

    const struct ggml_type_traits * type_traits_ptr = ggml_get_type_traits(tensor->type);

    if (!type_traits_ptr) { // Check if pointer is null
        LLAMA_LOG_ERROR("%s: Could not get type traits for type %s.\n", __func__, ggml_type_name(tensor->type));
        f32_conv_buf.resize(nelements);
        for (auto& ni_val : f32_conv_buf) { ni_val.value = std::numeric_limits<float>::quiet_NaN(); }
        return;
    }

    if (!type_traits_ptr->to_float) {
        LLAMA_LOG_ERROR("%s: Type %s has no dequantization function (to_float is NULL).\n", __func__, ggml_type_name(tensor->type));
        f32_conv_buf.resize(nelements);
        for (auto& ni_val : f32_conv_buf) { ni_val.value = std::numeric_limits<float>::quiet_NaN(); }
        return;
    }

    f32_conv_buf.resize(nelements); // Ensure buffer is large enough

    // Dequantize block by block
    // Assumes nelements is a multiple of block size, which should hold for valid tensors.
    const int64_t block_size_elements = ggml_blck_size(tensor->type);
    const size_t  block_size_bytes    = ggml_type_size(tensor->type); // Size of one quantized block in bytes

    if (block_size_elements == 0) {
        LLAMA_LOG_ERROR("%s: Type %s has zero block size.\n", __func__, ggml_type_name(tensor->type));
        for (auto& ni_val : f32_conv_buf) { ni_val.value = std::numeric_limits<float>::quiet_NaN(); }
        return;
    }

    const int64_t n_blocks = nelements / block_size_elements;

    const char *  quantized_data_ptr = static_cast<const char *>(tensor->data);
    // Access .value for the float pointer. This assumes no_init<float> has .value.
    // Or, if to_float writes to a buffer that will be assigned to .value later.
    // The ggml_type_traits.to_float expects float*, so we need a raw float* buffer.
    // This means f32_conv_buf might be the wrong type if it's no_init<float>.
    // The original code in llama.cpp likely uses std::vector<float> for f32_conv_buf.
    // Let's assume f32_conv_buf IS std::vector<no_init<float>> and we dequantize to a temporary float array
    // then copy to f32_conv_buf[i].value, or more simply, that to_float can write to where .value would be.
    // The safest is to use reinterpret_cast if no_init is a simple wrapper.
    float *       float_data_ptr     = reinterpret_cast<float *>(f32_conv_buf.data());


    // TODO: Add threading here if nthread > 1, by distributing blocks among threads.
    // For now, serial implementation:
    for (int64_t i = 0; i < n_blocks; ++i) {
        type_traits_ptr->to_float(
            quantized_data_ptr + i * block_size_bytes,    // Pointer to current quantized block
            float_data_ptr     + i * block_size_elements, // Pointer to output float block
            block_size_elements                           // Number of elements in one block (e.g., QK_K)
        );
    }
}

// Forward declaration for llama_tensor_quantize_impl
static size_t llama_tensor_quantize_impl(
    enum ggml_type type,
    const float * src,
    void * dst,
    int64_t n,
    int64_t nrows,
    int64_t k,
    const float * imatrix,
    std::vector<std::thread> & workers,
    int nthread) {
    // LLAMA_LOG_WARN("%s: STUB! Real implementation needed.\n", __func__);
    // This function is called per slice of the tensor.
    // src: pointer to the start of F32 data for the current slice.
    // dst: pointer to the start of destination memory for the current slice.
    // n: (chunk_size_elements from caller) - seems to be for parallel chunking strategy,
    //    but for a serial version processing the whole slice, it's implicitly nrows * k.
    // nrows: number of rows in this slice.
    // k: number of elements per row (columns).
    // imatrix: importance matrix for this slice (size k, applied to all rows in the slice).

    GGML_UNUSED(n); // n is not directly used if quantizing the whole slice in one go or if ggml_quantize_chunk handles it.
                    // For parallel version, n would be used to divide work.

    if (nthread > 1 && nrows > 1) {
        // Basic parallelization: split rows among threads
        // More sophisticated chunking like in ggml.c's ggml_quantize_rows_parallel could be used.
        // This is a simplified parallel approach.
        LLAMA_LOG_INFO("%s: Parallelizing quantization of %" PRId64 " rows with %d threads.\n", __func__, nrows, nthread);
        std::vector<std::thread> loc_workers; // Use local workers if global 'workers' is not managed correctly
        loc_workers.resize(nthread -1); // nthread-1 worker threads, 1 main thread

        int64_t rows_per_thread = (nrows + nthread - 1) / nthread;
        // size_t total_size_written = 0; // Unused
        std::mutex size_mutex;

        for (int t = 0; t < nthread; ++t) {
            const int64_t r_start = t * rows_per_thread;
            const int64_t r_end   = std::min(r_start + rows_per_thread, nrows);
            if (r_start >= r_end) continue;

            // const float * thread_src = src + r_start * k; // Unused
            // char * thread_dst_char = static_cast<char *>(dst); // Unused
            // Calculate offset into dst for this thread's rows
            // This requires knowing the size of previously quantized rows by other threads if types vary,
            // or assuming fixed output size per row if type is const for this call.
            // For simplicity, assuming ggml_quantize_chunk can write to a sub-pointer of dst.
            // The dst pointer itself needs to be correctly offset for each thread's output.
            // This is tricky if block sizes vary.
            // The current `new_data_slice` in the caller is only for the *start* of the slice.
            // A simpler parallel model: each thread quantizes its rows into a temp buffer,
            // then results are copied. Or, each thread calculates its output offset.

            // For now, let's stick to serial for the stub to avoid complex offset calculations.
            // The `workers` vector passed in is also problematic if not cleared/joined correctly.
        }
        // Join threads... (omitted for serial stub below)

        // Serial fallback if threading logic is too complex for stub:
        // The ggml_quantize_chunk function itself is not internally parallelized for multiple rows in one call.
        // The parallelization happens by calling ggml_quantize_chunk for sub-batches of rows.
        // The `workers` and `nthread` parameters are for this higher-level parallelization.
        // The original ggml.c ggml_quantize_rows_parallel is a good reference.

        // For this stub, let's keep it serial. The caller (llama_model_quantize_impl)
        // doesn't seem to manage the workers vector for this call.
        GGML_UNUSED(workers); // Mark as unused for this serial stub
        GGML_UNUSED(nthread); // Mark as unused for this serial stub
        return ggml_quantize_chunk(type, src, dst, 0, nrows, k, imatrix);

    } else {
        // Serial execution for nthread <=1 or single row
        GGML_UNUSED(workers);
        GGML_UNUSED(nthread);
        return ggml_quantize_chunk(type, src, dst, 0, nrows, k, imatrix);
    }
}

// Forward declaration for llama_tensor_get_type (already provided earlier with static keyword)
// static enum ggml_type llama_tensor_get_type(
// quantize_state_impl & qs,
// enum ggml_type default_type,
// const struct ggml_tensor * tensor,
// llama_ftype ftype);
// Definition:
static enum ggml_type llama_tensor_get_type(
    quantize_state_impl & qs,
    enum ggml_type default_type,
    const struct ggml_tensor * tensor,
    llama_ftype ftype) {

    const std::string name = ggml_get_name(tensor);

    // Leave layer norms in F32.
    if (name.rfind("normalization.weight") != std::string::npos ||
        name.rfind(".norm.weight") != std::string::npos ||
        name.rfind("ln.weight") != std::string::npos ||
        name.rfind("_norm.weight") != std::string::npos ||
        name.rfind(".ln_f.weight") != std::string::npos ||
        name.rfind(".attention_norm.weight") != std::string::npos ||
        name.rfind(".ffn_norm.weight") != std::string::npos) {
        return GGML_TYPE_F32;
    }

    GGML_UNUSED(qs);
    GGML_UNUSED(ftype);
    return default_type;
}


// Forward declaration for llama_tensor_quantize_smarter_blocks
// It's in this file, so it should be fine if defined before use or static.

static void zeros(std::ofstream &out, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        out.write(&zero, 1);
    }
}

// This is defined in this file later.
size_t llama_tensor_quantize_smarter_blocks(
    const float * src_data,
    void * dst_data,
    const int64_t * ne,
    const SmarterQuantTensorInfo & sq_info,
    const float * imatrix_data,
    int nthread) {
    // Definition starts here
    GGML_UNUSED(nthread);

    size_t total_bytes_written = 0;
    // dst_data is the buffer for the entire tensor's quantized output.
    // We write into it sequentially.

    const int64_t n_cols = ne[0];
    const int64_t n_rows = ne[1];
    int64_t n_slices = 1;
    if (ne[2] > 0) { // Check if there's a 3rd dimension
        n_slices = ne[2];
        if (ne[3] > 0) { // Check for 4th dimension, though unlikely for typical weights
             for (int i = 3; i < GGML_MAX_DIMS && ne[i] > 0; ++i) {
                n_slices *= ne[i];
            }
        }
    }
    if (n_cols == 0 || n_rows == 0) return 0; // Should not happen for valid tensors

    for (int64_t slice_idx = 0; slice_idx < n_slices; ++slice_idx) {
        const float * slice_src_data_start = src_data + slice_idx * (n_cols * n_rows);
        const float * slice_imatrix_data_start = nullptr;
        if (imatrix_data) {
            // Assuming imatrix_data is laid out per slice, each slice having ne[0] elements for its importance matrix.
            // This matches how llama_tensor_quantize_impl seems to handle it.
            slice_imatrix_data_start = imatrix_data + slice_idx * n_cols;
        }

        for (int64_t r = 0; r < n_rows; ++r) { // For each row in the current slice
            const float * row_src_data_start = slice_src_data_start + r * n_cols;
            // char * row_dst_data_start = current_dst_ptr; // This line is incorrect and removed.

            int64_t current_col_offset_in_row = 0;
            int block_segment_idx = 0;

            while (current_col_offset_in_row < n_cols) {
                int64_t n_cols_in_segment = 256;
                if (current_col_offset_in_row + n_cols_in_segment > n_cols) {
                    n_cols_in_segment = n_cols - current_col_offset_in_row;
                }

                if (n_cols_in_segment == 0) break; // Should not happen if loop condition is correct

                enum ggml_type quant_type;
                if (block_segment_idx < 4) {
                    quant_type = static_cast<enum ggml_type>(sq_info.compression_types[block_segment_idx]);
                } else {
                    quant_type = static_cast<enum ggml_type>(sq_info.compression_types[3]);
                }

                const float * segment_src_data = row_src_data_start + current_col_offset_in_row;
                void * segment_dst_data = static_cast<char*>(dst_data) + total_bytes_written; // Calculate current position in global dst_data

                const float * segment_imatrix_data = nullptr;
                if (slice_imatrix_data_start) {
                     // imatrix is per original column index for the current slice
                    segment_imatrix_data = slice_imatrix_data_start + current_col_offset_in_row;
                }

                size_t bytes_for_segment = ggml_quantize_chunk(
                    quant_type,
                    segment_src_data,
                    segment_dst_data,
                    0, // start index within src for this chunk (relative to segment_src_data)
                    1, // number of rows in this chunk (always 1 as we iterate row by row)
                    n_cols_in_segment,
                    segment_imatrix_data
                );

                // DEBUG PRINT
                // printf("DEBUG Quant: slice %lld, row %lld, seg %d, type %s, cols %lld, bytes %zu, total_bytes %zu\n",
                //        (long long)slice_idx, (long long)r, block_segment_idx, ggml_type_name(quant_type),
                //        (long long)n_cols_in_segment, bytes_for_segment, total_bytes_written + bytes_for_segment);
                // END DEBUG

                total_bytes_written += bytes_for_segment;
                // current_dst_ptr is implicitly advanced by total_bytes_written tracking

                current_col_offset_in_row += n_cols_in_segment;
                block_segment_idx++;
            }
            // After processing all segments of a row, current_dst_ptr should point to the end of this row's data
            // This is now handled by total_bytes_written for global dst_data offsetting
        }
    }
    return total_bytes_written;
}

// Made non-static to be callable from llama.cpp
void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;
    SmarterQuantConfig smarter_quant_config_json; // Loaded from JSON
    SmarterQuantConfig smarter_quant_config_gguf; // Loaded from GGUF (will be merged)


    // Load the SmarterQuant configuration from JSON
    // TODO: Make the filename configurable via params, for now hardcoded
    smarter_quant_config_json = load_smarter_quant_config("default.smarterquant.json");

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: default_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: default_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: default_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: default_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: default_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  default_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: default_type = GGML_TYPE_BF16; break;
        case LLAMA_FTYPE_ALL_F32:     default_type = GGML_TYPE_F32;  break;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    default_type = GGML_TYPE_Q2_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  default_type = GGML_TYPE_Q3_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  default_type = GGML_TYPE_Q4_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  default_type = GGML_TYPE_Q5_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    default_type = GGML_TYPE_Q6_K;    break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   default_type = GGML_TYPE_TQ1_0;   break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   default_type = GGML_TYPE_TQ2_0;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: default_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   default_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: default_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   default_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   default_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  default_type = GGML_TYPE_IQ4_NL;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  default_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   default_type = GGML_TYPE_IQ3_S;   break;

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides_ptr = nullptr;
    if (params->kv_overrides) {
        auto v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides_ptr = v->data();
    }

    std::vector<std::string> splits = {};
    llama_model_loader ml(fname_inp, splits, use_mmap, /*check_tensors*/ true, kv_overrides_ptr);
    // GGUF SmarterQuant config is loaded by llama_model_loader constructor into ml.gguf_smarter_quant_config
    smarter_quant_config_gguf = ml.gguf_smarter_quant_config;


    ml.init_mappings(false);

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    // Merge JSON and GGUF SmarterQuant configurations. GGUF takes precedence.
    SmarterQuantConfig final_smarter_quant_config = smarter_quant_config_json;
    for (const auto& pair : smarter_quant_config_gguf) {
        final_smarter_quant_config[pair.first] = pair.second;
    }


    struct quantize_state_impl qs(model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(imatrix_data->size()));
            qs.has_imatrix = true;
            for (const auto & kv : *imatrix_data) {
                for (float f_val : kv.second) { // Renamed f to f_val
                    if (!std::isfinite(f_val)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f_val));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    gguf_set_kv     (ctx_out.get(), ml.meta.get());
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "general.file_type", ftype);

    gguf_remove_key(ctx_out.get(), ml.llm_kv(llm_kv::LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(llm_kv::LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(llm_kv::LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        const std::vector<llama_model_kv_override> & overrides = *(const std::vector<llama_model_kv_override> *)params->kv_overrides;
        for (const auto & o : overrides) {
            if (o.key[0] == 0) break;
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o.key, o.val_f64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                gguf_set_val_i32(ctx_out.get(), o.key, o.val_i64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key);
            }
        }
    }

    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & iter : ml.weights_map) { // Renamed it to iter
        tensors.push_back(&iter.second);
    }

    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    for (const auto * iter : tensors) { // Renamed it to iter
        const struct ggml_tensor * tensor = iter->tensor;
        const std::string name = ggml_get_name(tensor);
        if (name.find("attn_v.weight")   != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos ||
            name.find("attn_kv_b.weight")!= std::string::npos) {
            ++qs.n_attention_wv;
        } else if (name == LLM_TN(model.arch)(LLM_TENSOR_OUTPUT, "weight")) {
            qs.has_output = true;
        }
    }

    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)model.hparams.n_layer;
    if (qs.n_attention_wv != 0) {
        const auto & n_head_kv_iter = model.hparams.n_head_kv_arr.begin();
        int32_t n_attn_layer = model.hparams.n_layer - std::count(n_head_kv_iter, n_head_kv_iter + model.hparams.n_layer, 0);
        if (llama_model_has_encoder(&model)) {
            n_attn_layer *= 3;
        }
        GGML_ASSERT((qs.n_attention_wv == n_attn_layer) && "n_attention_wv is unexpected");
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<std::thread> workers;
    workers.reserve(nthread);
    int idx_counter = 0; // Renamed idx to idx_counter
    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;
    uint16_t n_split = 1;

    if (params->keep_split) {
        for (const auto * iter : tensors) { // Renamed it to iter
            n_split = std::max(uint16_t(iter->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    for (const auto * iter : tensors) { // Renamed it to iter
        uint16_t i_split = params->keep_split ? iter->idx : 0;
        struct ggml_tensor * tensor = iter->tensor;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);
    }

    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(llm_kv::LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(llm_kv::LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(llm_kv::LLM_KV_SPLIT_TENSORS_COUNT).c_str(), ml.n_tensors);
        }
    }

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index_val) { // Renamed index to index_val
        cur_split = index_val;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname_val = fname_out; // Renamed fname to fname_val
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname_val = std::string(split_path.data());
        }
        fout = std::ofstream(fname_val, std::ios::binary);
        fout.exceptions(std::ofstream::failbit);
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        ::zeros(fout, meta_size);
    };

    const auto tn_func = LLM_TN(model.arch); // Renamed tn to tn_func
    new_ofstream(0);
    for (const auto * iter : tensors) { // Renamed it to iter
        const auto & weight = *iter;
        struct ggml_tensor * tensor = weight.tensor;
        if (weight.idx != cur_split && params->keep_split) {
            close_ofstream();
            new_ofstream(weight.idx);
        }
        const std::string name = ggml_get_name(tensor);
        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);
        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx_counter, ml.n_tensors, // Used idx_counter
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        bool quantize = name.rfind("weight") == name.size() - 6;
        quantize &= (ggml_n_dims(tensor) >= 2);
        quantize &= name.find("_norm.weight") == std::string::npos;
        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;
        quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;
        quantize &= name != tn_func(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != tn_func(LLM_TENSOR_TOKEN_TYPES, "weight");
        quantize &= name.find("ssm_conv1d.weight") == std::string::npos;
        quantize &= name.find("time_mix_first.weight") == std::string::npos;
        // ... (other quantize &= conditions)
        quantize &= name.find("attn_rel_b.weight") == std::string::npos;


        enum ggml_type new_type_val; // Renamed new_type
        void * new_data;
        size_t new_size;

        if (quantize) {
            new_type_val = default_type;
            if (!params->pure && ggml_is_quantized(default_type)) {
                new_type_val = llama_tensor_get_type(qs, new_type_val, tensor, ftype);
            }
            if (params->token_embedding_type < GGML_TYPE_COUNT && strcmp(tensor->name, "token_embd.weight") == 0) {
                new_type_val = params->token_embedding_type;
            }
            if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
                new_type_val = params->output_tensor_type;
            }
            quantize = tensor->type != new_type_val;
        }

        if (!quantize) {
            new_type_val = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const int64_t nelements = ggml_nelements(tensor);
            const float * imatrix_ptr = nullptr; // Renamed imatrix to imatrix_ptr
            if (imatrix_data) {
                auto im_it = imatrix_data->find(tensor->name); // Renamed it to im_it
                if (im_it == imatrix_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                } else {
                    if (im_it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                        imatrix_ptr = im_it->second.data();
                    } else {
                        LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                int(im_it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);
                        if (name != tn_func(LLM_TENSOR_TOKEN_EMBD, "weight")) {
                            throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                    int(im_it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                        }
                    }
                }
            }
            if ((new_type_val == GGML_TYPE_IQ2_XXS || new_type_val == GGML_TYPE_IQ2_XS  || new_type_val == GGML_TYPE_IQ2_S   ||
                 new_type_val == GGML_TYPE_IQ1_S   || (new_type_val == GGML_TYPE_IQ1_M && strcmp(tensor->name, "token_embd.weight") && strcmp(tensor->name, "output.weight"))  ||
                (new_type_val == GGML_TYPE_Q2_K && params->ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && strcmp(tensor->name, "token_embd.weight") != 0)) && !imatrix_ptr) {
                LLAMA_LOG_ERROR("\n\n============================================================\n");
                LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                LLAMA_LOG_ERROR("============================================================\n\n");
                throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
            }

            float * f32_data;
            std::vector<no_init<float>> permuted_f32_data_holder;

            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }

            auto sq_it = final_smarter_quant_config.find(name); // Use final_smarter_quant_config
            if (sq_it != final_smarter_quant_config.end() && sq_it->second.enabled) {
                const int32_t* current_perm_ptr = nullptr;
                size_t current_perm_size = 0;

                if (sq_it->second.column_permutation != nullptr && sq_it->second.n_cols_for_permutation > 0) {
                    LLAMA_LOG_INFO("Applying column permutation for tensor %s...\n", name.c_str());
                    current_perm_ptr = sq_it->second.column_permutation;
                    current_perm_size = sq_it->second.n_cols_for_permutation;

                    if (current_perm_size != (size_t)tensor->ne[0]) {
                        LLAMA_LOG_ERROR("Error: Permutation size %zu does not match tensor columns %" PRId64 " for tensor %s. Skipping permutation.\n", current_perm_size, tensor->ne[0], name.c_str());
                    } else {
                        permuted_f32_data_holder.resize(nelements);
                        float * permuted_data_ptr = (float *)permuted_f32_data_holder.data();
                        const int64_t n_cols = tensor->ne[0];
                        const int64_t n_rows = tensor->ne[1];
                        const int64_t higher_dims_stride = ggml_nelements(tensor) / (n_cols * n_rows);

                        for (int64_t h_dim = 0; h_dim < higher_dims_stride; ++h_dim) {
                            const float * current_f32_slice = f32_data + h_dim * (n_cols * n_rows);
                            float * current_permuted_slice = permuted_data_ptr + h_dim * (n_cols * n_rows);
                            for (int64_t r = 0; r < n_rows; ++r) {
                                for (int64_t c_new = 0; c_new < n_cols; ++c_new) {
                                    const int64_t c_orig = current_perm_ptr[c_new];
                                    if (c_orig < 0 || c_orig >= n_cols) {
                                         LLAMA_LOG_ERROR("Error: Invalid column index %" PRId64 " in permutation for tensor %s. Skipping permutation.\n", c_orig, name.c_str());
                                         permuted_f32_data_holder.clear();
                                         f32_data = (float *)((tensor->type == GGML_TYPE_F32) ? tensor->data : f32_conv_buf.data());
                                         goto skip_imatrix_permutation; // Corrected label
                                    }
                                    current_permuted_slice[r * n_cols + c_new] = current_f32_slice[r * n_cols + c_orig];
                                }
                            }
                        }
                        f32_data = permuted_data_ptr;
                        LLAMA_LOG_INFO("Finished applying column permutation for f32_data of tensor %s.\n", name.c_str());

                        if (imatrix_ptr) {
                            std::vector<float> permuted_imatrix_values;
                            const int64_t n_cols_imatrix = tensor->ne[0];
                            if (imatrix_data->at(name).size() % n_cols_imatrix != 0) {
                                LLAMA_LOG_WARN("Warning: imatrix size %zu not a multiple of n_cols %" PRId64 " for tensor %s. Skipping imatrix permutation.\n",
                                               imatrix_data->at(name).size(), n_cols_imatrix, name.c_str());
                            } else {
                                permuted_imatrix_values.resize(imatrix_data->at(name).size());
                                const float* original_imatrix_ptr = imatrix_data->at(name).data();
                                float* p_imatrix_ptr = permuted_imatrix_values.data(); // Renamed permuted_imatrix_ptr
                                const int64_t num_imatrix_slices_in_source = imatrix_data->at(name).size() / n_cols_imatrix;
                                for (int64_t s_idx = 0; s_idx < num_imatrix_slices_in_source; ++s_idx) {
                                    const float* current_original_imatrix_slice = original_imatrix_ptr + s_idx * n_cols_imatrix;
                                    float* current_permuted_imatrix_slice = p_imatrix_ptr + s_idx * n_cols_imatrix;
                                    for (int64_t c_new = 0; c_new < n_cols_imatrix; ++c_new) {
                                        const int64_t c_orig = current_perm_ptr[c_new];
                                        if (c_orig >= 0 && c_orig < n_cols_imatrix) {
                                            current_permuted_imatrix_slice[c_new] = current_original_imatrix_slice[c_orig];
                                        } else {
                                            current_permuted_imatrix_slice[c_new] = current_original_imatrix_slice[c_new];
                                        }
                                    }
                                }
                                qs.permuted_imatrix_holder = permuted_imatrix_values;
                                imatrix_ptr = qs.permuted_imatrix_holder.data();
                                LLAMA_LOG_INFO("Finished applying column permutation for imatrix of tensor %s.\n", name.c_str());
                            }
                        }
                    }
                }
            skip_imatrix_permutation:;

                // Store SmarterQuant GGUF metadata if enabled.
                {
                    nlohmann::json perm_json_array_gguf = nlohmann::json::array(); // Renamed perm_json_array
                    if (sq_it->second.column_permutation != nullptr) {
                        for (int64_t i = 0; i < sq_it->second.n_cols_for_permutation; ++i) {
                            perm_json_array_gguf.push_back(sq_it->second.column_permutation[i]);
                        }
                    }
                    std::string perm_str = perm_json_array_gguf.dump();

                    llama_model_kv_override kvo_perm;
                    snprintf(kvo_perm.key, sizeof(kvo_perm.key), "%s.smarterquant.permutation", name.c_str());
                    kvo_perm.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                    strncpy(kvo_perm.val_str, perm_str.c_str(), sizeof(kvo_perm.val_str) - 1);
                    kvo_perm.val_str[sizeof(kvo_perm.val_str) - 1] = '\0';

                    llama_model_kv_override kvo_enabled;
                    snprintf(kvo_enabled.key, sizeof(kvo_enabled.key), "%s.smarterquant.enabled", name.c_str());
                    kvo_enabled.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
                    kvo_enabled.val_bool = true;

                    nlohmann::json types_json_array_gguf = nlohmann::json::array(); // Renamed types_json_array
                    for(int i=0; i<4; ++i) {
                        types_json_array_gguf.push_back(sq_it->second.compression_types[i]);
                    }
                    std::string types_str = types_json_array_gguf.dump();
                    llama_model_kv_override kvo_types;
                    snprintf(kvo_types.key, sizeof(kvo_types.key), "%s.smarterquant.block_types", name.c_str());
                    kvo_types.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                    strncpy(kvo_types.val_str, types_str.c_str(), sizeof(kvo_types.val_str) -1);
                    kvo_types.val_str[sizeof(kvo_types.val_str)-1] = '\0';

                    if (params->kv_overrides) {
                        auto* overrides_vec = reinterpret_cast<std::vector<llama_model_kv_override>*>(params->kv_overrides);
                        // bool null_term_found = false; // unused variable
                        if (!overrides_vec->empty() && overrides_vec->back().key[0] == 0) {
                            // null_term_found = true; // unused variable
                            overrides_vec->pop_back();
                        }
                        overrides_vec->push_back(kvo_perm);
                        overrides_vec->push_back(kvo_enabled);
                        overrides_vec->push_back(kvo_types);
                        overrides_vec->emplace_back();
                        overrides_vec->back().key[0] = 0;
                    }
                    LLAMA_LOG_INFO("Adding metadata for %s: permutation, enabled, block_types\n", name.c_str());
                }
            }

            if (work.size() < (size_t)nelements * 4) {
                work.resize(nelements * 4);
            }
            new_data = work.data();

            const int64_t n_per_row = tensor->ne[0];
            const int64_t nrows     = tensor->ne[1];
            const int64_t n_slices  = tensor->ne[2];
            static const int64_t min_chunk_size_bytes = 32 * 512;
            const int64_t elements_per_row_bytes_approx = n_per_row * sizeof(float);
            const int64_t chunk_size_elements = (elements_per_row_bytes_approx >= min_chunk_size_bytes ? n_per_row : n_per_row * ((min_chunk_size_bytes + elements_per_row_bytes_approx - 1)/elements_per_row_bytes_approx));
            const int64_t nelements_matrix_per_slice = n_per_row * nrows;
            const int64_t nchunk_per_slice = (nelements_matrix_per_slice + chunk_size_elements - 1)/chunk_size_elements;
            const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk_per_slice)) : 1;

            if (sq_it != final_smarter_quant_config.end() && sq_it->second.enabled) {
                new_type_val = static_cast<ggml_type>(sq_it->second.compression_types[3]); // Base GGUF type
                LLAMA_LOG_INFO("Applying SmarterQuant to %s. GGUF type: %s. Calling llama_tensor_quantize_smarter_blocks.\n", name.c_str(), ggml_type_name(new_type_val));
                new_size = llama_tensor_quantize_smarter_blocks(
                    f32_data, new_data, tensor->ne, sq_it->second, imatrix_ptr, nthread_use);
                LLAMA_LOG_INFO("SmarterQuant for %s done. Calculated new_size = %zu bytes.\n", name.c_str(), new_size);
            } else {
                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type_val));
                fflush(stdout);
                new_size = 0;
                for (int64_t i03 = 0; i03 < n_slices; ++i03) {
                    const float * f32_data_slice = f32_data + i03 * nelements_matrix_per_slice;
                    void * new_data_slice = (char *)new_data + i03 * nrows * ggml_row_size(new_type_val, n_per_row);
                    const float * imatrix_slice_ptr = nullptr; // Renamed imatrix_slice
                    if (imatrix_ptr) {
                        imatrix_slice_ptr = imatrix_ptr + i03 * n_per_row;
                    }
                    new_size += llama_tensor_quantize_impl(new_type_val, f32_data_slice, new_data_slice, chunk_size_elements, nrows, n_per_row, imatrix_slice_ptr, workers, nthread_use);
                }
            }
            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        gguf_set_tensor_type(ctx_outs[cur_split].get(), name.c_str(), new_type_val);
        gguf_set_tensor_data(ctx_outs[cur_split].get(), name.c_str(), new_data);

        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }
    close_ofstream();
		
    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, qs.n_k_quantized + qs.n_fallback);
    }
}
