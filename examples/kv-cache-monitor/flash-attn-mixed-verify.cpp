#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <memory>
#include <iomanip>

// Import the custom flash attention function from mixed KV cache
// Note: We declare it here instead of including the header to avoid linking issues
void ggml_custom_flash_attn_mixed_simple(
    ggml_tensor * dst,
    int ith,
    int nth,
    void* wdata,
    size_t wsize,
    void * userdata);

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef CACHE_LINE_SIZE_F32
#define CACHE_LINE_SIZE_F32 16
#endif

/**
 * Test data structure with standard ggml_tensor usage
 */
struct flash_attn_test_data {
    ggml_context* reference_ctx = nullptr;  // Context for reference tensors
    gguf_context* reference_gguf = nullptr; // GGUF context for cleanup
    std::unordered_map<std::string, ggml_tensor*> reference_tensors;
    int target_step = 1; // Which step to test
    bool verbose = false;
    
    ~flash_attn_test_data() {
        // Cleanup resources
        if (reference_ctx) {
            ggml_free(reference_ctx);
            reference_ctx = nullptr;
        }
        if (reference_gguf) {
            gguf_free(reference_gguf);
            reference_gguf = nullptr;
        }
    }
};

/**
 * Load tensors from GGUF file using standard ggml_tensor
 */
static bool load_tensors_from_gguf(flash_attn_test_data* test_data, const std::string& filename) {
    LOG("[VERIFY] Loading tensors from: %s\n", filename.c_str());
    
    // Initialize GGUF context with data context
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &test_data->reference_ctx,
    };
    
    test_data->reference_gguf = gguf_init_from_file(filename.c_str(), params);
    if (!test_data->reference_gguf) {
        LOG_ERR("[VERIFY] Failed to load GGUF file: %s\n", filename.c_str());
        return false;
    }
    
    if (!test_data->reference_ctx) {
        LOG_ERR("[VERIFY] Failed to create reference context\n");
        gguf_free(test_data->reference_gguf);
        test_data->reference_gguf = nullptr;
        return false;
    }
    
    // Load all tensors from the context
    const int n_tensors = gguf_get_n_tensors(test_data->reference_gguf);
    LOG("[VERIFY] Found %d tensors\n", n_tensors);
    
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(test_data->reference_gguf, i);
        
        ggml_tensor* tensor = ggml_get_tensor(test_data->reference_ctx, name);
        if (tensor) {
            test_data->reference_tensors[std::string(name)] = tensor;
            
            if (test_data->verbose) {
                LOG("[VERIFY] Loaded tensor: %s [%ld,%ld,%ld,%ld] type=%s\n", 
                    name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    ggml_type_name(tensor->type));
            }
        }
    }
    
    LOG("[VERIFY] Loaded %zu tensors\n", test_data->reference_tensors.size());
    return !test_data->reference_tensors.empty();
}

/**
 * Find Q tensor (Qcur with permuted)
 */
static ggml_tensor* find_q_tensor(flash_attn_test_data* test_data, int step) {
    std::string step_suffix = "_step_" + std::to_string(step);
    
    for (auto& pair : test_data->reference_tensors) {
        const std::string& name = pair.first;
        
        // Look for: src0_Qcur-0 (permuted)_step_N (WITH "permuted")
        if (name.find("src0_") == 0 && 
            name.find("Qcur") != std::string::npos && 
            name.find("permuted") != std::string::npos &&
            name.find(step_suffix) != std::string::npos) {
            
            LOG("[VERIFY] Found Q tensor: %s\n", name.c_str());
            return pair.second;
        }
    }
    
    LOG("[VERIFY] No Q tensor found for step %d\n", step);
    return nullptr;
}

/**
 * Find K tensor (cache_k with permuted)
 */
static ggml_tensor* find_k_tensor(flash_attn_test_data* test_data, int step) {
    std::string step_suffix = "_step_" + std::to_string(step);
    
    for (auto& pair : test_data->reference_tensors) {
        const std::string& name = pair.first;
        
        // Look for: src1_cache_k_l0 (view) (permuted)_step_N
        if (name.find("src1_") == 0 && 
            name.find("cache_k") != std::string::npos && 
            name.find("permuted") != std::string::npos &&
            name.find(step_suffix) != std::string::npos) {
            
            LOG("[VERIFY] Found K tensor: %s\n", name.c_str());
            return pair.second;
        }
    }
    
    LOG("[VERIFY] No K tensor found for step %d\n", step);
    return nullptr;
}

/**
 * Find V tensor (cache_v with permuted)
 */
static ggml_tensor* find_v_tensor(flash_attn_test_data* test_data, int step) {
    std::string step_suffix = "_step_" + std::to_string(step);
    
    for (auto& pair : test_data->reference_tensors) {
        const std::string& name = pair.first;
        
        // Look for: src2_cache_v_l0 (view) (permuted)_step_N
        if (name.find("src2_") == 0 && 
            name.find("cache_v") != std::string::npos && 
            name.find("permuted") != std::string::npos &&
            name.find(step_suffix) != std::string::npos) {
            
            LOG("[VERIFY] Found V tensor: %s\n", name.c_str());
            return pair.second;
        }
    }
    
    LOG("[VERIFY] No V tensor found for step %d\n", step);
    return nullptr;
}

/**
 * Find output tensor for a specific step
 */
static ggml_tensor* find_output_tensor(flash_attn_test_data* test_data, int step) {
    // Look for kqv_out tensor for the specified step
    for (auto& pair : test_data->reference_tensors) {
        const std::string& name = pair.first;
        
        // Check if this is an output tensor for the target step
        if (name.find("kqv_out") != std::string::npos && name.find("_step_" + std::to_string(step)) != std::string::npos) {
            return pair.second;
        }
    }
    return nullptr;
}

/**
 * Convert tensor data to float array for comparison
 */
static std::vector<float> tensor_to_float_array(const uint8_t* data, ggml_type type, size_t n_elements) {
    std::vector<float> result(n_elements);
    
    switch (type) {
        case GGML_TYPE_F32: {
            const float* f32_data = (const float*)data;
            for (size_t i = 0; i < n_elements; ++i) {
                result[i] = f32_data[i];
            }
            break;
        }
        case GGML_TYPE_F16: {
            const ggml_fp16_t* f16_data = (const ggml_fp16_t*)data;
            for (size_t i = 0; i < n_elements; ++i) {
                result[i] = ggml_fp16_to_fp32(f16_data[i]);
            }
            break;
        }
        default:
            // For unsupported types, fill with zeros
            std::fill(result.begin(), result.end(), 0.0f);
            break;
    }
    
    return result;
}

/**
 * Copy tensor data to a new tensor in target context
 */
static ggml_tensor* copy_tensor_to_context(ggml_context* target_ctx, const ggml_tensor* source_tensor) {
    if (!target_ctx || !source_tensor) {
        return nullptr;
    }
    
    // Create new tensor with same properties
    ggml_tensor* new_tensor = ggml_new_tensor(target_ctx, source_tensor->type, GGML_MAX_DIMS, source_tensor->ne);
    if (!new_tensor) {
        return nullptr;
    }
    
    // Copy data
    size_t data_size = ggml_nbytes(source_tensor);
    memcpy(new_tensor->data, source_tensor->data, data_size);
    
    return new_tensor;
}

/**
 * Compare two tensors and print detailed statistics
 */
static void compare_tensors(const ggml_tensor* expected, const ggml_tensor* actual, const std::string& name) {
    LOG("[VERIFY] Comparing tensor: %s\n", name.c_str());
    
    // Check shapes
    bool shapes_match = true;
    size_t total_elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (expected->ne[i] != actual->ne[i]) {
            shapes_match = false;
        }
        if (expected->ne[i] > 0) {
            total_elements *= expected->ne[i];
        }
    }
    
    if (!shapes_match) {
        LOG_ERR("[VERIFY] Shape mismatch for %s\n", name.c_str());
        return;
    }
    
    // Convert both to float arrays
    std::vector<float> expected_data = tensor_to_float_array((const uint8_t*)expected->data, expected->type, total_elements);
    std::vector<float> actual_data = tensor_to_float_array((const uint8_t*)actual->data, actual->type, total_elements);
    
    // Calculate statistics
    double sum_abs_diff = 0.0;
    double sum_rel_diff = 0.0;
    double sum_squared_diff = 0.0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    
    for (size_t i = 0; i < total_elements; ++i) {
        float expected_val = expected_data[i];
        float actual_val = actual_data[i];
        
        // Check for NaN and Inf
        if (std::isnan(expected_val) || std::isnan(actual_val)) {
            nan_count++;
            continue;
        }
        if (std::isinf(expected_val) || std::isinf(actual_val)) {
            inf_count++;
            continue;
        }
        
        // Absolute difference
        double abs_diff = std::abs(expected_val - actual_val);
        sum_abs_diff += abs_diff;
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        
        // Relative difference
        double expected_abs = std::abs(expected_val);
        if (expected_abs > 1e-12) {
            double rel_diff = abs_diff / expected_abs;
            sum_rel_diff += rel_diff;
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
        
        // For RMSE
        double diff = expected_val - actual_val;
        sum_squared_diff += diff * diff;
    }
    
    size_t valid_elements = total_elements - nan_count - inf_count;
    
    if (valid_elements > 0) {
        double mean_abs_diff = sum_abs_diff / valid_elements;
        double mean_rel_diff = sum_rel_diff / valid_elements;
        double rmse = std::sqrt(sum_squared_diff / valid_elements);
        
        LOG("[VERIFY] Results for %s:\n", name.c_str());
        LOG("[VERIFY]   Total elements: %zu\n", total_elements);
        LOG("[VERIFY]   Mean abs diff: %.6e\n", mean_abs_diff);
        LOG("[VERIFY]   Max abs diff: %.6e\n", max_abs_diff);
        LOG("[VERIFY]   Mean rel diff: %.6e\n", mean_rel_diff);
        LOG("[VERIFY]   Max rel diff: %.6e\n", max_rel_diff);
        LOG("[VERIFY]   RMSE: %.6e\n", rmse);
        
        if (nan_count > 0 || inf_count > 0) {
            LOG("[VERIFY]   WARNING: NaN count: %zu, Inf count: %zu\n", nan_count, inf_count);
        }
        
        // Print first 10 elements comparison
        LOG("[VERIFY]   First 10 elements comparison:\n");
        LOG("[VERIFY]   Index | Expected Value |  Actual Value  | Abs Diff | Rel Diff\n");
        LOG("[VERIFY]   ------|----------------|----------------|----------|----------\n");
        
        size_t elements_to_show = std::min(static_cast<size_t>(1024), total_elements);
        for (size_t i = 0; i < elements_to_show; ++i) {
            float expected_val = expected_data[i];
            float actual_val = actual_data[i];
            double abs_diff = std::abs(expected_val - actual_val);
            double rel_diff = 0.0;
            
            double expected_abs = std::abs(expected_val);
            if (expected_abs > 1e-12) {
                rel_diff = abs_diff / expected_abs;
            }
            
            LOG("[VERIFY]   %5zu | %14.6e | %15.6e | %8.2e | %8.2e\n", 
                i, expected_val, actual_val, abs_diff, rel_diff);
        }
        
        // Quality assessment
        const double tolerance_abs = 1e-4;
        const double tolerance_rel = 1e-3;
        bool within_tolerance = (mean_abs_diff <= tolerance_abs) && (mean_rel_diff <= tolerance_rel);
        
        LOG("[VERIFY]   Quality assessment: %s\n", within_tolerance ? "PASS" : "FAIL");
        LOG("[VERIFY]   ----------------------------------------\n");
    }
}

/**
 * Calculate workspace size for flash attention
 */
static size_t calculate_workspace_size(const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v, int n_threads) {
    GGML_UNUSED(k); // k is not needed for workspace calculation
    
    const int64_t DK = q->ne[0];        // head_dim for queries/keys
    const int64_t DV = v->ne[0];        // head_dim for values  
    const int64_t SEQ_LEN = q->ne[1];   // sequence length (Q: [head_dim, seq_len, n_heads, batch])
    const int64_t N_Q_HEADS = q->ne[2]; // number of query heads
    
    // Follow the mixed KV cache flash attention workspace layout:
    // OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 2 * DV + 1 * DK + 1 + CACHE_LINE_SIZE_F32
    const size_t OUTPUT_SIZE = DV * N_Q_HEADS * SEQ_LEN;
    const size_t LOCAL_MAX_SIZE = N_Q_HEADS * SEQ_LEN;
    const size_t cache_line_size_f32 = 16;
    
    size_t per_thread_size = (OUTPUT_SIZE + 2 * LOCAL_MAX_SIZE + 2 * DV + 1 * DK + 1 + cache_line_size_f32) * sizeof(float);
    
    return per_thread_size * n_threads;
}

/**
 * Test flash attention for a specific step
 */
static bool test_flash_attention_step(flash_attn_test_data* test_data, int step) {
    LOG("[VERIFY] Testing flash attention for step %d\n", step);
    
    // Find input tensors using the correct naming convention
    ggml_tensor* q_tensor = find_q_tensor(test_data, step);            // Q input: Qcur with permuted
    ggml_tensor* k_tensor = find_k_tensor(test_data, step);            // K input: cache_k with permuted  
    ggml_tensor* v_tensor = find_v_tensor(test_data, step);            // V input: cache_v with permuted
    ggml_tensor* mask_tensor = nullptr;                                // mask input: set to null for now
    ggml_tensor* expected_output = find_output_tensor(test_data, step);
    
    if (!q_tensor || !k_tensor || !v_tensor || !expected_output) {
        LOG_ERR("[VERIFY] Missing required tensors for step %d\n", step);
        LOG_ERR("[VERIFY]   Q: %s, K: %s, V: %s, Output: %s\n", 
                q_tensor ? "found" : "missing",
                k_tensor ? "found" : "missing", 
                v_tensor ? "found" : "missing",
                expected_output ? "found" : "missing");
        return false;
    }
    
    LOG("[VERIFY] Found all required tensors for step %d\n", step);
    
    // Create GGML context for computation
    size_t ctx_size = 1024 * 1024 * 16; // 16MB should be enough
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERR("[VERIFY] Failed to create GGML context\n");
        return false;
    }
    
    // Copy tensors to computation context
    ggml_tensor* q = copy_tensor_to_context(ctx, q_tensor);
    ggml_tensor* k = copy_tensor_to_context(ctx, k_tensor);
    ggml_tensor* v = copy_tensor_to_context(ctx, v_tensor);
    ggml_tensor* mask = nullptr;
    if (mask_tensor) {
        mask = copy_tensor_to_context(ctx, mask_tensor);
    }
    
    if (!q || !k || !v) {
        LOG_ERR("[VERIFY] Failed to copy input tensors to computation context\n");
        ggml_free(ctx);
        return false;
    }
    
    // Print tensor information
    LOG("[VERIFY] Q tensor: [%ld, %ld, %ld, %ld] type=%s\n", 
        q->ne[0], q->ne[1], q->ne[2], q->ne[3], ggml_type_name(q->type));
    LOG("[VERIFY] K tensor: [%ld, %ld, %ld, %ld] type=%s\n", 
        k->ne[0], k->ne[1], k->ne[2], k->ne[3], ggml_type_name(k->type));
    LOG("[VERIFY] V tensor: [%ld, %ld, %ld, %ld] type=%s\n", 
        v->ne[0], v->ne[1], v->ne[2], v->ne[3], ggml_type_name(v->type));
    LOG("[VERIFY] Expected output: [%ld, %ld, %ld, %ld] type=%s\n", 
        expected_output->ne[0], expected_output->ne[1], expected_output->ne[2], expected_output->ne[3], 
        ggml_type_name(expected_output->type));
    
    // CRITICAL FIX: Extract dimensions correctly for all steps
    // Expected output format: [head_dim * n_heads, seq_len, 1, batch]
    // We need to derive the correct dimensions from the expected output, not make assumptions
    const int64_t expected_total_dim = expected_output->ne[0];  // head_dim * n_heads (e.g., 4096)
    const int64_t expected_seq_len = expected_output->ne[1];    // actual sequence length from expected output
    const int64_t expected_batch = expected_output->ne[3];      // batch size
    
    // Q tensor format: [head_dim, seq_len, n_heads, batch] (after permutation)
    const int64_t head_dim = q->ne[0];        // 128
    const int64_t q_seq_len = q->ne[1];       // actual sequence length from Q
    const int64_t n_heads = q->ne[2];         // 32
    const int64_t batch_size = q->ne[3];      // 1
    
    // Verify that dimensions are consistent
    if (expected_total_dim != head_dim * n_heads) {
        LOG_ERR("[VERIFY] ERROR: Expected total dimension (%ld) != head_dim * n_heads (%ld * %ld = %ld)\n", 
                expected_total_dim, head_dim, n_heads, head_dim * n_heads);
        ggml_free(ctx);
        return false;
    }
    
    if (expected_seq_len != q_seq_len) {
        LOG_ERR("[VERIFY] ERROR: Expected sequence length (%ld) != Q sequence length (%ld)\n", 
                expected_seq_len, q_seq_len);
        ggml_free(ctx);
        return false;
    }
    
    LOG("[VERIFY] Verified dimensions: head_dim=%ld, n_heads=%ld, seq_len=%ld, batch=%ld\n", 
        head_dim, n_heads, expected_seq_len, batch_size);
    
    // Create custom flash attention operation using ggml_custom_4d
    // Use the verified dimensions from expected output
    ggml_tensor* args[] = { q, k, v, mask };
    
    const int n_threads = 4; // Use 4 threads for testing
    
    LOG("[VERIFY] Creating custom flash attention operation...\n");
    LOG("[VERIFY] Output dimensions: [%ld, %ld, %ld, %ld]\n", 
        head_dim, n_heads, expected_seq_len, batch_size);
    
    ggml_tensor* custom_output = ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,                                          // output type
        head_dim, n_heads, expected_seq_len, batch_size,        // FIXED: use expected_seq_len
        args,                                                   // input tensors
        4,                                                      // number of arguments
        (ggml_custom_op_t)ggml_custom_flash_attn_mixed_simple,  // custom function
        n_threads,                                              // number of threads
        nullptr                                                 // userdata
    );
    
    if (!custom_output) {
        LOG_ERR("[VERIFY] Failed to create custom flash attention operation\n");
        ggml_free(ctx);
        return false;
    }
    
    // Build computation graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, custom_output);
    
    // Calculate workspace size and allocate
    struct ggml_cplan cplan = ggml_graph_plan(graph, n_threads, nullptr);
    size_t workspace_size = cplan.work_size;
    
    // If workspace size is 0 or too small, calculate manually
    if (workspace_size == 0) {
        workspace_size = calculate_workspace_size(q, k, v, n_threads);
    }
    
    LOG("[VERIFY] Workspace size: %zu bytes (%.2f MB)\n", workspace_size, workspace_size / (1024.0 * 1024.0));
    
    std::vector<uint8_t> workspace(workspace_size);
    cplan.work_data = workspace.data();
    cplan.work_size = workspace_size;
    
    // Execute computation graph
    LOG("[VERIFY] Executing custom flash attention computation graph...\n");
    
    enum ggml_status status = ggml_graph_compute(graph, &cplan);
    
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("[VERIFY] Flash attention computation failed with status: %d\n", status);
        ggml_free(ctx);
        return false;
    }
    
    LOG("[VERIFY] Custom flash attention computation completed successfully\n");
        
    // Create expected output tensor for comparison in computation context
    ggml_tensor* expected = copy_tensor_to_context(ctx, expected_output);
    if (!expected) {
        LOG_ERR("[VERIFY] Failed to copy expected output to computation context\n");
        ggml_free(ctx);
        return false;
    }
    
    // Reshape custom output to match expected output format
    // Custom output: [head_dim, n_heads, seq_len, batch] 
    // Expected output: [head_dim * n_heads, seq_len, 1, batch]
    // We need to reshape our output to match the expected format
    
    LOG("[VERIFY] Custom output shape: [%ld, %ld, %ld, %ld]\n", 
        custom_output->ne[0], custom_output->ne[1], custom_output->ne[2], custom_output->ne[3]);
    LOG("[VERIFY] Expected output shape: [%ld, %ld, %ld, %ld]\n", 
        expected->ne[0], expected->ne[1], expected->ne[2], expected->ne[3]);
    
    // Create a reshaped view of custom output to match expected format
    // Reshape from [head_dim, n_heads, seq_len, batch] to [head_dim * n_heads, seq_len, 1, batch]
    ggml_tensor* custom_reshaped = ggml_reshape_4d(ctx, custom_output, 
                                                   head_dim * n_heads,  // head_dim * n_heads
                                                   expected_seq_len,    // seq_len  
                                                   1,                   // 1
                                                   batch_size);         // batch
    
    if (!custom_reshaped) {
        LOG_ERR("[VERIFY] Failed to reshape custom output\n");
        ggml_free(ctx);
        return false;
    }
    
    LOG("[VERIFY] Reshaped custom output shape: [%ld, %ld, %ld, %ld]\n", 
        custom_reshaped->ne[0], custom_reshaped->ne[1], custom_reshaped->ne[2], custom_reshaped->ne[3]);
    
    // Compare results
    compare_tensors(expected, custom_reshaped, "Flash Attention Output");
    
    ggml_free(ctx);
    return true;
}

/**
 * Run all tests
 */
static bool run_tests(flash_attn_test_data* test_data) {
    LOG("[VERIFY] Running flash attention verification tests\n");
    
    bool all_passed = true;
    
    // Test the target step
    if (!test_flash_attention_step(test_data, test_data->target_step)) {
        LOG_ERR("[VERIFY] Test failed for step %d\n", test_data->target_step);
        all_passed = false;
    }
    
    LOG("[VERIFY] All tests completed. Result: %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed;
}

int main(int argc, char** argv) {
    flash_attn_test_data test_data;

    // Parse command line arguments
    std::string input_file;
    int target_step = 1;
    bool verbose = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            target_step = std::atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s --input <gguf_file> [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --input <file>     Input GGUF file with saved tensors (required)\n");
            printf("  --step <n>         Target step to verify (default: 1)\n");
            printf("  --verbose          Enable verbose output\n");
            printf("  --help, -h         Show this help message\n");
            printf("\nExample:\n");
            printf("  %s --input trace_data.gguf --step 1 --verbose\n", argv[0]);
            return 0;
        }
    }
    
    if (input_file.empty()) {
        LOG_ERR("Error: --input parameter is required\n");
        LOG_ERR("Use --help for usage information\n");
        return 1;
    }
    
    test_data.target_step = target_step;
    test_data.verbose = verbose;
    
    LOG_INF("Flash Attention Mixed KV Cache Verification Tool\n");
    LOG_INF("Input file: %s\n", input_file.c_str());
    LOG_INF("Target step: %d\n", target_step);
    LOG_INF("Verbose mode: %s\n", verbose ? "enabled" : "disabled");
    
    // Load tensors from GGUF file using standard ggml_tensor
    if (!load_tensors_from_gguf(&test_data, input_file)) {
        LOG_ERR("Failed to load tensors from %s\n", input_file.c_str());
        return 1;
    }
    
    // Print all loaded tensor names
    LOG_INF("\nLoaded tensors (%zu total):\n", test_data.reference_tensors.size());
    
    if (test_data.reference_tensors.empty()) {
        LOG_ERR("No tensors were loaded from the file!\n");
        return 1;
    }
    
    // Collect tensor names for sorted output
    std::vector<std::string> tensor_names;
    for (const auto& tensor_pair : test_data.reference_tensors) {
        tensor_names.push_back(tensor_pair.first);
    }
    
    // Sort tensor names for more readable output
    std::sort(tensor_names.begin(), tensor_names.end());
    
    // Print tensor details
    for (const auto& name : tensor_names) {
        const auto& tensor = test_data.reference_tensors[name];
        
        // Create shape string showing ne dimensions
        std::string shape_str = "[";
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            shape_str += std::to_string(tensor->ne[i]);
            if (i < GGML_MAX_DIMS - 1) {
                shape_str += ",";
            }
        }
        shape_str += "]";
        
        LOG_INF("  %s - shape: %s\n", name.c_str(), shape_str.c_str());
        
        // Print additional details if verbose mode is enabled
        if (verbose) {
            LOG_INF("    type: %s, size: %zu bytes\n", 
                ggml_type_name(tensor->type),
                ggml_nbytes(tensor));
        }
    }
    
    LOG_INF("\n");
    
    // Run tests
    bool success = run_tests(&test_data);
    
    return success ? 0 : 1;
} 