// End-to-end tests for SmarterQuant GGUF functionality
// This test covers:
// 1. Writing SmarterQuant metadata to GGUF during quantization.
// 2. Reading SmarterQuant metadata from GGUF during model loading.
// 3. Numerical correctness of dequantization through the model loading path.

#include "ggml.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "llama-quant.h"
#include "llama-model-loader.h" // For llama_model_loader, to inspect GGUF metadata directly if needed
#include "gguf.h" // For gguf_init_empty, gguf_add_tensor, gguf_write_to_file, etc.
#include "json.hpp" // For nlohmann::json to create dummy smarterquant json

#undef NDEBUG // Ensure asserts are enabled
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream> // For std::ofstream, std::ifstream
#include <stdexcept> // For std::runtime_error
#include <cstdio> // For remove()

// Helper from test-smarterquant.cpp
static bool compare_float_arrays(const float* arr1, const float* arr2, size_t size, float tolerance, const std::string& test_name) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(arr1[i] - arr2[i]) > tolerance) {
            std::cerr << std::fixed << std::setprecision(8)
                      << "Test: " << test_name << " - Mismatch at index " << i << ": arr1 = " << arr1[i]
                      << ", arr2 = " << arr2[i] << ", diff = " << fabs(arr1[i] - arr2[i])
                      << ", tolerance = " << tolerance << std::endl;
            return false;
        }
    }
    return true;
}

// Helper to create a dummy FP32 GGUF file
static bool create_dummy_fp32_gguf(
    const std::string& filename,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& tensor_infos, // name, dims
    const std::vector<std::vector<float>>& tensor_data // data for each tensor
) {
    gguf_context_ptr ctx_out { gguf_init_empty() };
    if (!ctx_out) {
        fprintf(stderr, "Failed to initialize GGUF context for %s\n", filename.c_str());
        return false;
    }

    // Add some minimal GGUF metadata
    gguf_set_val_str (ctx_out.get(), "general.architecture", "dummy");
    gguf_set_val_u32(ctx_out.get(), "dummy.block_count", 1);
    gguf_set_val_u32(ctx_out.get(), "dummy.tensor_count", tensor_infos.size());


    for (size_t i = 0; i < tensor_infos.size(); ++i) {
        const auto& info = tensor_infos[i];
        const auto& data = tensor_data[i];

        struct ggml_init_params params = { data.size() * sizeof(float) + ggml_tensor_overhead(), NULL, true };
        struct ggml_context * tensor_ctx = ggml_init(params);
        if (!tensor_ctx) {
            fprintf(stderr, "Failed to create ggml context for tensor %s\n", info.first.c_str());
            return false;
        }

        struct ggml_tensor * t = nullptr;
        if (info.second.size() == 1) {
            t = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, info.second[0]);
        } else if (info.second.size() == 2) {
            t = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, info.second[0], info.second[1]);
        } else if (info.second.size() == 3) {
            t = ggml_new_tensor_3d(tensor_ctx, GGML_TYPE_F32, info.second[0], info.second[1], info.second[2]);
        } else {
            fprintf(stderr, "Unsupported tensor dimension count %zu for %s\n", info.second.size(), info.first.c_str());
            ggml_free(tensor_ctx);
            return false;
        }
        ggml_set_name(t, info.first.c_str());
        memcpy(t->data, data.data(), ggml_nbytes(t));

        gguf_add_tensor(ctx_out.get(), t);
        ggml_free(tensor_ctx);
    }

    if (!gguf_write_to_file(ctx_out.get(), filename.c_string(), false)) {
        fprintf(stderr, "Failed to write GGUF file %s\n", filename.c_str());
        return false;
    }
    printf("    Successfully created dummy FP32 GGUF: %s\n", filename.c_str());
    return true;
}

// Helper to create a dummy smarterquant.json file
static bool create_dummy_smarterquant_json(
    const std::string& filename,
    const std::string& tensor_name_1, int64_t n_cols_1, const std::vector<int32_t>& perm_1,
    const std::string& tensor_name_2, int64_t n_cols_2, const std::vector<int32_t>& perm_2
) {
    nlohmann::json j;

    nlohmann::json t1_config = nlohmann::json::array();
    nlohmann::json t1_types = nlohmann::json::array({GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K});
    nlohmann::json t1_perm = nlohmann::json::array();
    if (!perm_1.empty()) {
        for (int32_t idx : perm_1) t1_perm.push_back(idx);
    } else { // identity
        for (int64_t i=0; i<n_cols_1; ++i) t1_perm.push_back(i);
    }
    t1_config.push_back(t1_types);
    t1_config.push_back(t1_perm);
    j[tensor_name_1] = t1_config;

    nlohmann::json t2_config = nlohmann::json::array();
    // Use different types for the second tensor for variety
    nlohmann::json t2_types = nlohmann::json::array({GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K});
    nlohmann::json t2_perm = nlohmann::json::array();
     if (!perm_2.empty()) {
        for (int32_t idx : perm_2) t2_perm.push_back(idx);
    } else { // identity
        for (int64_t i=0; i<n_cols_2; ++i) t2_perm.push_back(i);
    }
    t2_config.push_back(t2_types);
    t2_config.push_back(t2_perm);
    j[tensor_name_2] = t2_config;

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        fprintf(stderr, "Failed to open %s for writing.\n", filename.c_str());
        return false;
    }
    ofs << j.dump(4);
    ofs.close();
    printf("    Successfully created dummy smarterquant JSON: %s\n", filename.c_str());
    return true;
}


int main(int argc, char **argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);
    printf("Testing SmarterQuant GGUF end-to-end functionality...\n");
    int overall_status = 0;

    const std::string dummy_fp32_gguf_name = "dummy_fp32_input.gguf";
    const std::string dummy_sq_json_name = "dummy_smarterquant.json";
    const std::string dummy_quantized_gguf_name = "dummy_quantized_output.gguf";

    // Define tensors for the dummy model
    std::string tensor1_name = "tensor_one";
    std::vector<int64_t> tensor1_dims = {512, 2}; // 512 cols, 2 rows
    std::vector<float> tensor1_data(tensor1_dims[0] * tensor1_dims[1]);
    for(size_t i=0; i<tensor1_data.size(); ++i) tensor1_data[i] = static_cast<float>(i % 128) - 64.f;

    std::string tensor2_name = "tensor_two";
    std::vector<int64_t> tensor2_dims = {1280, 1}; // 1280 cols, 1 row (5 * 256)
    std::vector<float> tensor2_data(tensor2_dims[0] * tensor2_dims[1]);
     for(size_t i=0; i<tensor2_data.size(); ++i) tensor2_data[i] = static_cast<float>((i % 200) * (i%2==0 ? 1 : -1)) * 0.5f;


    // Create dummy FP32 GGUF
    if (!create_dummy_fp32_gguf(dummy_fp32_gguf_name,
                                {{tensor1_name, tensor1_dims}, {tensor2_name, tensor2_dims}},
                                {tensor1_data, tensor2_data})) {
        fprintf(stderr, "Failed to create dummy FP32 GGUF.\n");
        return 1;
    }

    // Create dummy smarterquant.json
    std::vector<int32_t> perm1(tensor1_dims[0]); // Reverse for tensor1
    for(int64_t i=0; i<tensor1_dims[0]; ++i) perm1[i] = (tensor1_dims[0] - 1) - i;
    std::vector<int32_t> perm2; // Identity for tensor2 (empty means identity in helper)

    if (!create_dummy_smarterquant_json(dummy_sq_json_name,
                                        tensor1_name, tensor1_dims[0], perm1,
                                        tensor2_name, tensor2_dims[0], perm2)) {
        fprintf(stderr, "Failed to create dummy smarterquant.json.\n");
        remove(dummy_fp32_gguf_name.c_str());
        return 1;
    }

    // Quantize
    printf("  Attempting quantization...\n");
    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    qparams.ftype = LLAMA_FTYPE_MOSTLY_Q8_0; // Base type, SmarterQuant will override
    qparams.smarter_quant_json_path = dummy_sq_json_name.c_str(); // Specify our JSON

    // We need a kv_overrides vector even if empty, for the SmarterQuant metadata to be added to.
    std::vector<llama_model_kv_override> kv_overrides;
    kv_overrides.emplace_back(); // Add the null terminator
    kv_overrides.back().key[0] = 0;
    qparams.kv_overrides = &kv_overrides;


    try {
        llama_model_quantize_impl(dummy_fp32_gguf_name, dummy_quantized_gguf_name, &qparams);
        printf("    Quantization call completed.\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "    ERROR: Quantization failed with exception: %s\n", e.what());
        overall_status = 1;
        goto cleanup;
    }

    // Load the quantized model and verify
    printf("  Loading quantized GGUF and verifying...\n");
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(dummy_quantized_gguf_name.c_str(), mparams);

    if (!model) {
        fprintf(stderr, "    ERROR: Failed to load quantized GGUF model %s.\n", dummy_quantized_gguf_name.c_str());
        overall_status = 1;
        goto cleanup;
    }

    // Verify tensor1
    {
        const ggml_tensor* t1 = llama_get_model_tensor(model, tensor1_name.c_str());
        if (!t1) {
            fprintf(stderr, "    ERROR: Tensor '%s' not found in quantized model.\n", tensor1_name.c_str());
            overall_status = 1;
        } else {
            if (!t1->sq_info || !t1->sq_info->enabled) {
                fprintf(stderr, "    ERROR: Tensor '%s' does not have SmarterQuant info enabled after loading.\n", tensor1_name.c_str());
                overall_status = 1;
            } else {
                printf("    Tensor '%s' SmarterQuant info loaded successfully.\n", tensor1_name.c_str());
                // Check types (example for first block)
                if (t1->sq_info->compression_types[0] != GGML_TYPE_Q4_0) {
                     fprintf(stderr, "    ERROR: Tensor '%s' expected type0 %d, got %d.\n", tensor1_name.c_str(), GGML_TYPE_Q4_0, t1->sq_info->compression_types[0]);
                     overall_status = 1;
                }
                // Check permutation (example for first element)
                if (t1->sq_info->column_permutation[0] != perm1[0]) {
                     fprintf(stderr, "    ERROR: Tensor '%s' expected perm[0] %d, got %d.\n", tensor1_name.c_str(), perm1[0], t1->sq_info->column_permutation[0]);
                     overall_status = 1;
                }

                // Numerical check for tensor1
                std::vector<float> t1_dequant_data(tensor1_dims[0] * tensor1_dims[1]);
                // Simulate getting rows (simplified for test - assumes CPU context and direct call)
                // In a real scenario, this would be through ggml_compute_forward or similar.
                for(int r=0; r<tensor1_dims[1]; ++r) {
                    // Calculate the byte offset for the current row in the ggml_tensor's data
                    // This is a simplified calculation. A real scenario might need to consider
                    // the actual byte layout if rows are not simply (total_size / num_rows).
                    // llama_tensor_quantize_smarter_blocks writes data sequentially, so this should be okay for this test.
                    size_t row_byte_size = 0;
                    for(int64_t c_seg = 0; c_seg < tensor1_dims[0]; c_seg += 256) {
                        int64_t seg_cols = std::min((int64_t)256, tensor1_dims[0] - c_seg);
                        int block_idx_in_row = c_seg / 256;
                        ggml_type seg_type = (block_idx_in_row < 4) ? (ggml_type)t1->sq_info->compression_types[block_idx_in_row] : (ggml_type)t1->sq_info->compression_types[3];
                        row_byte_size += ggml_type_size(seg_type) * (seg_cols / ggml_blck_size(seg_type));
                    }

                    const char * t1_row_data = (const char*)t1->data + r * row_byte_size;
                    float* t1_dequant_row_ptr = t1_dequant_data.data() + r * tensor1_dims[0];
                    ggml_get_rows_smarterquant(t1, t1_row_data, t1_dequant_row_ptr);
                }
                if (!compare_float_arrays(tensor1_data.data(), t1_dequant_data.data(), tensor1_data.size(), 0.15f, tensor1_name)) {
                    fprintf(stderr, "    ERROR: Numerical mismatch for tensor '%s'.\n", tensor1_name.c_str());
                    overall_status = 1;
                } else {
                    printf("    Tensor '%s' numerical check PASSED.\n", tensor1_name.c_str());
                }
            }
        }
    }

    // Verify tensor2
    {
        const ggml_tensor* t2 = llama_get_model_tensor(model, tensor2_name.c_str());
        if (!t2) {
            fprintf(stderr, "    ERROR: Tensor '%s' not found in quantized model.\n", tensor2_name.c_str());
            overall_status = 1;
        } else {
            if (!t2->sq_info || !t2->sq_info->enabled) {
                fprintf(stderr, "    ERROR: Tensor '%s' does not have SmarterQuant info enabled after loading.\n", tensor2_name.c_str());
                overall_status = 1;
            } else {
                 printf("    Tensor '%s' SmarterQuant info loaded successfully.\n", tensor2_name.c_str());
                if (t2->sq_info->compression_types[0] != GGML_TYPE_Q8_0) { // Matching dummy_smarterquant.json
                     fprintf(stderr, "    ERROR: Tensor '%s' expected type0 %d, got %d.\n", tensor2_name.c_str(), GGML_TYPE_Q8_0, t2->sq_info->compression_types[0]);
                     overall_status = 1;
                }
                 // Check permutation (identity for tensor2)
                if (t2->sq_info->column_permutation[0] != 0) {
                     fprintf(stderr, "    ERROR: Tensor '%s' expected perm[0] 0 (identity), got %d.\n", tensor2_name.c_str(), t2->sq_info->column_permutation[0]);
                     overall_status = 1;
                }
                // Numerical check for tensor2
                std::vector<float> t2_dequant_data(tensor2_dims[0] * tensor2_dims[1]);
                for(int r=0; r<tensor2_dims[1]; ++r) {
                    size_t row_byte_size = 0;
                     for(int64_t c_seg = 0; c_seg < tensor2_dims[0]; c_seg += 256) {
                        int64_t seg_cols = std::min((int64_t)256, tensor2_dims[0] - c_seg);
                        int block_idx_in_row = c_seg / 256;
                        ggml_type seg_type = (block_idx_in_row < 4) ? (ggml_type)t2->sq_info->compression_types[block_idx_in_row] : (ggml_type)t2->sq_info->compression_types[3];
                        row_byte_size += ggml_type_size(seg_type) * (seg_cols / ggml_blck_size(seg_type));
                    }
                    const char * t2_row_data = (const char*)t2->data + r * row_byte_size;
                    float* t2_dequant_row_ptr = t2_dequant_data.data() + r * tensor2_dims[0];
                    ggml_get_rows_smarterquant(t2, t2_row_data, t2_dequant_row_ptr);
                }
                 if (!compare_float_arrays(tensor2_data.data(), t2_dequant_data.data(), tensor2_data.size(), 0.15f, tensor2_name)) {
                    fprintf(stderr, "    ERROR: Numerical mismatch for tensor '%s'.\n", tensor2_name.c_str());
                    overall_status = 1;
                } else {
                    printf("    Tensor '%s' numerical check PASSED.\n", tensor2_name.c_str());
                }
            }
        }
    }


    if (model) {
        llama_free_model(model);
    }

cleanup:
    // Clean up dummy files
    remove(dummy_fp32_gguf_name.c_str());
    remove(dummy_sq_json_name.c_str());
    remove(dummy_quantized_gguf_name.c_str());

    printf("SmarterQuant GGUF end-to-end test finished.\n");
    return overall_status;
}
