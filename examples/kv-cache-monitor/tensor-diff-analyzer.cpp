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
#include <cctype>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <sstream>
#include <functional>

/**
 * Structure to hold reference tensor data loaded from GGUF file
 */
struct reference_tensor {
    std::string name;
    ggml_type type;
    std::vector<int64_t> ne;
    std::vector<uint8_t> data;
    int step;
    
    reference_tensor(const std::string& n, ggml_type t, const int64_t* dims, 
                    const uint8_t* d, size_t data_size, int s) 
        : name(n), type(t), step(s), data(d, d + data_size) {
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            ne.push_back(dims[i]);
        }
    }
};

/**
 * Tensor difference statistics
 */
struct tensor_diff_stats {
    std::string tensor_name;
    int step;
    double mean_abs_diff = 0.0;
    double max_abs_diff = 0.0;
    double mean_rel_diff = 0.0;
    double max_rel_diff = 0.0;
    double rmse = 0.0;
    double cosine_similarity = 0.0;
    size_t total_elements = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    bool shapes_match = false;
    bool types_match = false;
};

/**
 * Callback data structure for tensor comparison
 */
struct tensor_diff_data {
    std::vector<uint8_t> temp_data;
    int step_count = 0;
    std::unordered_map<std::string, int> tensor_counts;
    int target_layer = -1;
    std::string reference_file;
    std::vector<reference_tensor> reference_tensors;
    std::vector<tensor_diff_stats> diff_results;
    bool analysis_enabled = false;
    double tolerance_abs = 1e-6;
    double tolerance_rel = 1e-4;
};

// Helper functions for tensor name matching
static std::string extract_base_name(const std::string& full_name) {
    // Extract base name from names like "kqv_out_kqv_out-0_step_1" -> "kqv_out-0"
    // or from current names like "kqv_out-0" -> "kqv_out-0"
    size_t step_pos = full_name.find("_step_");
    if (step_pos != std::string::npos) {
        std::string without_step = full_name.substr(0, step_pos);
        
        // Remove prefix like "kqv_out_" or "src0_"
        size_t prefix_end = without_step.find('_');
        if (prefix_end != std::string::npos && prefix_end + 1 < without_step.length()) {
            return without_step.substr(prefix_end + 1);
        }
        return without_step;
    }
    return full_name;
}

static std::string create_reference_name(const std::string& current_name, const std::string& prefix, int step) {
    // Create expected reference name: prefix + "_" + current_name + "_step_" + step
    return prefix + "_" + current_name + "_step_" + std::to_string(step);
}

static int extract_step_number(const std::string& full_name) {
    size_t step_pos = full_name.find("_step_");
    if (step_pos != std::string::npos) {
        std::string step_str = full_name.substr(step_pos + 6);
        try {
            return std::stoi(step_str);
        } catch (...) {
            return -1;
        }
    }
    return -1;
}

static bool is_kqv_out_tensor(const char* tensor_name) {
    if (!tensor_name) return false;
    std::string name(tensor_name);
    return name.find("kqv_out") != std::string::npos;
}

static int extract_layer_number(const char* tensor_name) {
    if (!tensor_name) return -1;
    
    std::string name(tensor_name);
    
    // Look for kqv_out-N pattern
    size_t kqv_pos = name.find("kqv_out-");
    if (kqv_pos != std::string::npos) {
        size_t dash_pos = kqv_pos + 8;
        if (dash_pos < name.length()) {
            std::string layer_str = name.substr(dash_pos);
            size_t end_pos = 0;
            while (end_pos < layer_str.length() && std::isdigit(layer_str[end_pos])) {
                end_pos++;
            }
            if (end_pos > 0) {
                try {
                    return std::stoi(layer_str.substr(0, end_pos));
                } catch (...) {
                    return -1;
                }
            }
        }
    }
    
    return -1;
}

static bool should_monitor_tensor(const char* tensor_name, int target_layer) {
    if (!is_kqv_out_tensor(tensor_name)) {
        return false;
    }
    
    if (target_layer == -1) {
        return true;
    }
    
    int layer_num = extract_layer_number(tensor_name);
    return layer_num == target_layer;
}

/**
 * Load reference tensors from GGUF file
 */
static bool load_reference_tensors(tensor_diff_data* diff_data) {
    if (diff_data->reference_file.empty()) {
        return false;
    }
    
    LOG("[DIFF-ANALYZER] Loading reference tensors from: %s\n", diff_data->reference_file.c_str());
    
    struct ggml_context* ctx_data = nullptr;
    
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };
    
    struct gguf_context* ctx = gguf_init_from_file(diff_data->reference_file.c_str(), params);
    if (!ctx) {
        LOG_ERR("[DIFF-ANALYZER] Failed to load reference GGUF file: %s\n", diff_data->reference_file.c_str());
        return false;
    }
    
    // Load all tensors
    const int n_tensors = gguf_get_n_tensors(ctx);
    LOG("[DIFF-ANALYZER] Found %d reference tensors\n", n_tensors);
    
    for (int i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx, i);
        
        if (ctx_data) {
            struct ggml_tensor* tensor = ggml_get_tensor(ctx_data, name);
            if (tensor) {
                int step = extract_step_number(std::string(name));
                
                diff_data->reference_tensors.emplace_back(
                    std::string(name),
                    tensor->type,
                    tensor->ne,
                    (const uint8_t*)tensor->data,
                    ggml_nbytes(tensor),
                    step
                );
                
                LOG("[DIFF-ANALYZER] Loaded reference tensor: %s (step %d)\n", name, step);
            }
        }
    }
    
    // Cleanup
    if (ctx_data) {
        ggml_free(ctx_data);
    }
    gguf_free(ctx);
    
    LOG("[DIFF-ANALYZER] Loaded %zu reference tensors\n", diff_data->reference_tensors.size());
    return !diff_data->reference_tensors.empty();
}

/**
 * Find matching reference tensor
 */
static const reference_tensor* find_reference_tensor(const tensor_diff_data* diff_data, 
                                                   const std::string& current_name, 
                                                   int current_step,
                                                   const std::string& prefix) {
    // Create expected reference name: prefix + "_" + current_name + "_step_" + step
    std::string expected_ref_name = create_reference_name(current_name, prefix, current_step);
    
    for (const auto& ref_tensor : diff_data->reference_tensors) {
        if (ref_tensor.name == expected_ref_name) {
            return &ref_tensor;
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
 * Calculate comprehensive tensor difference statistics
 */
static tensor_diff_stats calculate_tensor_diff(const std::string& tensor_name, int step,
                                             const uint8_t* current_data, ggml_type current_type, 
                                             const int64_t* current_ne,
                                             const reference_tensor& ref_tensor) {
    tensor_diff_stats stats;
    stats.tensor_name = tensor_name;
    stats.step = step;
    
    // Check shape compatibility
    stats.shapes_match = true;
    stats.total_elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (current_ne[i] != ref_tensor.ne[i]) {
            stats.shapes_match = false;
        }
        if (current_ne[i] > 0) {
            stats.total_elements *= current_ne[i];
        }
    }
    
    // Check type compatibility
    stats.types_match = (current_type == ref_tensor.type);
    
    if (!stats.shapes_match) {
        LOG_ERR("[DIFF-ANALYZER] Shape mismatch for %s: current vs reference\n", tensor_name.c_str());
        return stats;
    }
    
    // Convert both tensors to float arrays
    std::vector<float> current_float = tensor_to_float_array(current_data, current_type, stats.total_elements);
    std::vector<float> ref_float = tensor_to_float_array(ref_tensor.data.data(), ref_tensor.type, stats.total_elements);
    
    // Calculate statistics
    double sum_abs_diff = 0.0;
    double sum_rel_diff = 0.0;
    double sum_squared_diff = 0.0;
    double sum_current_squared = 0.0;
    double sum_ref_squared = 0.0;
    double dot_product = 0.0;
    
    stats.max_abs_diff = 0.0;
    stats.max_rel_diff = 0.0;
    stats.nan_count = 0;
    stats.inf_count = 0;
    
    for (size_t i = 0; i < stats.total_elements; ++i) {
        float current_val = current_float[i];
        float ref_val = ref_float[i];
        
        // Check for NaN and Inf
        if (std::isnan(current_val) || std::isnan(ref_val)) {
            stats.nan_count++;
            continue;
        }
        if (std::isinf(current_val) || std::isinf(ref_val)) {
            stats.inf_count++;
            continue;
        }
        
        // Absolute difference
        double abs_diff = std::abs(current_val - ref_val);
        sum_abs_diff += abs_diff;
        stats.max_abs_diff = std::max(stats.max_abs_diff, abs_diff);
        
        // Relative difference
        double ref_abs = std::abs(ref_val);
        if (ref_abs > 1e-12) {
            double rel_diff = abs_diff / ref_abs;
            sum_rel_diff += rel_diff;
            stats.max_rel_diff = std::max(stats.max_rel_diff, rel_diff);
        }
        
        // For RMSE and cosine similarity
        double diff = current_val - ref_val;
        sum_squared_diff += diff * diff;
        sum_current_squared += current_val * current_val;
        sum_ref_squared += ref_val * ref_val;
        dot_product += current_val * ref_val;
    }
    
    size_t valid_elements = stats.total_elements - stats.nan_count - stats.inf_count;
    
    if (valid_elements > 0) {
        stats.mean_abs_diff = sum_abs_diff / valid_elements;
        stats.mean_rel_diff = sum_rel_diff / valid_elements;
        stats.rmse = std::sqrt(sum_squared_diff / valid_elements);
        
        // Cosine similarity
        double norm_current = std::sqrt(sum_current_squared);
        double norm_ref = std::sqrt(sum_ref_squared);
        if (norm_current > 1e-12 && norm_ref > 1e-12) {
            stats.cosine_similarity = dot_product / (norm_current * norm_ref);
        }
    }
    
    return stats;
}

/**
 * Compare current tensor with reference
 */
static void compare_tensor_with_reference(tensor_diff_data* diff_data, 
                                        struct ggml_tensor* current_tensor,
                                        const std::string& prefix = "") {
    if (!diff_data->analysis_enabled || !current_tensor) return;
    
    // Get current tensor data
    const bool is_host = ggml_backend_buffer_is_host(current_tensor->buffer);
    uint8_t* current_data = nullptr;
    
    if (!is_host) {
        auto n_bytes = ggml_nbytes(current_tensor);
        diff_data->temp_data.resize(n_bytes);
        ggml_backend_tensor_get(current_tensor, diff_data->temp_data.data(), 0, n_bytes);
        current_data = diff_data->temp_data.data();
    } else {
        current_data = (uint8_t*)current_tensor->data;
    }
    
    // Use the actual tensor name directly
    std::string tensor_name = std::string(current_tensor->name ? current_tensor->name : "unnamed");
    
    // Find matching reference tensor
    const reference_tensor* ref_tensor = find_reference_tensor(diff_data, tensor_name, diff_data->step_count, prefix);
    
    if (!ref_tensor) {
        LOG("[DIFF-ANALYZER] No reference tensor found for: %s (step %d, prefix: %s)\n", 
            tensor_name.c_str(), diff_data->step_count, prefix.c_str());
        return;
    }
    
    // Calculate differences
    tensor_diff_stats stats = calculate_tensor_diff(
        tensor_name, diff_data->step_count,
        current_data, current_tensor->type, current_tensor->ne,
        *ref_tensor
    );
    
    diff_data->diff_results.push_back(stats);
    
    // Log results
    LOG("[DIFF-ANALYZER] Tensor: %s (step %d)\n", tensor_name.c_str(), diff_data->step_count);
    LOG("[DIFF-ANALYZER]   Shape match: %s, Type match: %s\n", 
        stats.shapes_match ? "YES" : "NO", stats.types_match ? "YES" : "NO");
    LOG("[DIFF-ANALYZER]   Mean abs diff: %.6e, Max abs diff: %.6e\n", 
        stats.mean_abs_diff, stats.max_abs_diff);
    LOG("[DIFF-ANALYZER]   Mean rel diff: %.6e, Max rel diff: %.6e\n", 
        stats.mean_rel_diff, stats.max_rel_diff);
    LOG("[DIFF-ANALYZER]   RMSE: %.6e, Cosine similarity: %.6f\n", 
        stats.rmse, stats.cosine_similarity);
    
    if (stats.nan_count > 0 || stats.inf_count > 0) {
        LOG("[DIFF-ANALYZER]   WARNING: NaN count: %zu, Inf count: %zu\n", 
            stats.nan_count, stats.inf_count);
    }
    
    // Print first 10 elements comparison
    if (stats.shapes_match && stats.total_elements > 0) {
        LOG("[DIFF-ANALYZER]   First 10 elements comparison:\n");
        LOG("[DIFF-ANALYZER]   Index | Current Value | Reference Value | Abs Diff | Rel Diff\n");
        LOG("[DIFF-ANALYZER]   ------|---------------|-----------------|----------|----------\n");
        
        // Convert tensor data to float arrays for element comparison
        std::vector<float> current_float = tensor_to_float_array(current_data, current_tensor->type, stats.total_elements);
        std::vector<float> ref_float = tensor_to_float_array(ref_tensor->data.data(), ref_tensor->type, stats.total_elements);
        
        size_t elements_to_show = std::min(static_cast<size_t>(10), stats.total_elements);
        for (size_t i = 0; i < elements_to_show; ++i) {
            float current_val = current_float[i];
            float ref_val = ref_float[i];
            double abs_diff = std::abs(current_val - ref_val);
            double rel_diff = 0.0;
            
            // Calculate relative difference
            double ref_abs = std::abs(ref_val);
            if (ref_abs > 1e-12) {
                rel_diff = abs_diff / ref_abs;
            }
            
            LOG("[DIFF-ANALYZER]   %5zu | %13.6e | %15.6e | %8.2e | %8.2e\n", 
                i, current_val, ref_val, abs_diff, rel_diff);
        }
        
        if (stats.total_elements > 10) {
            LOG("[DIFF-ANALYZER]   ... (%zu more elements)\n", stats.total_elements - 10);
        }
    }
    
    // Check tolerances
    bool within_tolerance = (stats.mean_abs_diff <= diff_data->tolerance_abs) && 
                           (stats.mean_rel_diff <= diff_data->tolerance_rel);
    LOG("[DIFF-ANALYZER]   Within tolerance: %s\n", within_tolerance ? "YES" : "NO");
    LOG("[DIFF-ANALYZER]   ----------------------------------------\n");
}

/**
 * Print final analysis summary
 */
static void print_analysis_summary(const tensor_diff_data* diff_data) {
    if (diff_data->diff_results.empty()) {
        LOG("[DIFF-ANALYZER] No tensor comparisons performed\n");
        return;
    }
    
    LOG("\n=== TENSOR DIFFERENCE ANALYSIS SUMMARY ===\n");
    LOG("Reference file: %s\n", diff_data->reference_file.c_str());
    LOG("Total comparisons: %zu\n", diff_data->diff_results.size());
    LOG("Tolerance - Absolute: %.2e, Relative: %.2e\n", 
        diff_data->tolerance_abs, diff_data->tolerance_rel);
    
    // Calculate overall statistics
    size_t within_tolerance_count = 0;
    size_t shape_mismatch_count = 0;
    size_t type_mismatch_count = 0;
    double max_abs_diff_overall = 0.0;
    double max_rel_diff_overall = 0.0;
    double avg_cosine_similarity = 0.0;
    
    for (const auto& stats : diff_data->diff_results) {
        if (!stats.shapes_match) shape_mismatch_count++;
        if (!stats.types_match) type_mismatch_count++;
        
        bool within_tolerance = (stats.mean_abs_diff <= diff_data->tolerance_abs) && 
                               (stats.mean_rel_diff <= diff_data->tolerance_rel);
        if (within_tolerance) within_tolerance_count++;
        
        max_abs_diff_overall = std::max(max_abs_diff_overall, stats.max_abs_diff);
        max_rel_diff_overall = std::max(max_rel_diff_overall, stats.max_rel_diff);
        avg_cosine_similarity += stats.cosine_similarity;
    }
    
    avg_cosine_similarity /= diff_data->diff_results.size();
    
    LOG("\n--- Overall Results ---\n");
    LOG("Tensors within tolerance: %zu/%zu (%.1f%%)\n", 
        within_tolerance_count, diff_data->diff_results.size(),
        100.0 * within_tolerance_count / diff_data->diff_results.size());
    LOG("Shape mismatches: %zu\n", shape_mismatch_count);
    LOG("Type mismatches: %zu\n", type_mismatch_count);
    LOG("Maximum absolute difference: %.6e\n", max_abs_diff_overall);
    LOG("Maximum relative difference: %.6e\n", max_rel_diff_overall);
    LOG("Average cosine similarity: %.6f\n", avg_cosine_similarity);
    
    // List problematic tensors
    LOG("\n--- Tensors exceeding tolerance ---\n");
    for (const auto& stats : diff_data->diff_results) {
        bool within_tolerance = (stats.mean_abs_diff <= diff_data->tolerance_abs) && 
                               (stats.mean_rel_diff <= diff_data->tolerance_rel);
        if (!within_tolerance) {
            LOG("  %s (step %d): abs=%.2e, rel=%.2e\n", 
                stats.tensor_name.c_str(), stats.step,
                stats.mean_abs_diff, stats.mean_rel_diff);
        }
    }
    
    LOG("==========================================\n\n");
}

/**
 * GGML operations callback for tensor comparison
 */
static bool ggml_debug_tensor_diff(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * diff_data = (tensor_diff_data *) user_data;

    if (ask) {
        return should_monitor_tensor(t->name, diff_data->target_layer);
    }

    if (!should_monitor_tensor(t->name, diff_data->target_layer)) {
        return true;
    }

    diff_data->step_count++;
    diff_data->tensor_counts[std::string(t->name)]++;

    LOG("\n=== TENSOR DIFFERENCE ANALYSIS ===\n");
    LOG("Analyzing tensor: %s (step %d)\n", t->name, diff_data->step_count);

    // Recursive function to compare all tensors in the computation graph
    std::function<void(struct ggml_tensor*, const std::string&, int)> compare_tensor_recursive = 
        [&](struct ggml_tensor* tensor, const std::string& prefix, int depth) {
            if (!tensor || depth > 3) return; // Limit recursion depth to avoid infinite loops
            
            // Try to find and compare this tensor with reference
            std::string tensor_name = std::string(tensor->name ? tensor->name : "unnamed");
            
            // Check if this tensor exists in our reference data
            const reference_tensor* ref_tensor = find_reference_tensor(diff_data, tensor_name, diff_data->step_count, prefix);
            
            if (ref_tensor) {
                LOG("[DIFF-ANALYZER] Found reference for %s with prefix %s\n", tensor_name.c_str(), prefix.c_str());
                compare_tensor_with_reference(diff_data, tensor, prefix);
            } else {
                // Try different common prefixes if direct match fails
                std::vector<std::string> common_prefixes = {"kqv_out", "src0", "src1", "src2", "src3"};
                bool found_match = false;
                
                for (const auto& test_prefix : common_prefixes) {
                    const reference_tensor* test_ref = find_reference_tensor(diff_data, tensor_name, diff_data->step_count, test_prefix);
                    if (test_ref) {
                        LOG("[DIFF-ANALYZER] Found reference for %s with prefix %s\n", tensor_name.c_str(), test_prefix.c_str());
                        compare_tensor_with_reference(diff_data, tensor, test_prefix);
                        found_match = true;
                        break;
                    }
                }
                
                if (!found_match) {
                    LOG("[DIFF-ANALYZER] No reference tensor found for: %s (step %d, tried prefixes: %s + common)\n", 
                        tensor_name.c_str(), diff_data->step_count, prefix.c_str());
                }
            }
            
            // Recursively process source tensors
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (tensor->src[i]) {
                    std::string src_prefix = "src" + std::to_string(i);
                    compare_tensor_recursive(const_cast<struct ggml_tensor*>(tensor->src[i]), src_prefix, depth + 1);
                }
            }
        };

    // Start recursive comparison from the main tensor
    compare_tensor_recursive(t, "kqv_out", 0);
    
    LOG("===================================\n\n");

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    LOG("Initial prompt tokens: %zu\n", tokens.size());
    LOG("Starting generation with %d tokens to generate\n", params.n_predict);
    LOG("========================================\n\n");

    // Process initial prompt
    LOG("=== PROCESSING INITIAL PROMPT ===\n");
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval initial prompt\n", __func__);
        return false;
    }
    LOG("=== INITIAL PROMPT PROCESSED ===\n\n");

    // Generate tokens one by one
    for (int i = 0; i < params.n_predict; ++i) {
        LOG("=== GENERATION STEP %d/%d ===\n", i + 1, params.n_predict);
        
        auto logits = llama_get_logits_ith(ctx, -1);
        auto n_vocab = llama_vocab_n_tokens(vocab);
        
        // Find token with highest probability (greedy sampling)
        llama_token new_token = 0;
        float max_logit = logits[0];
        for (llama_token token_id = 1; token_id < n_vocab; token_id++) {
            if (logits[token_id] > max_logit) {
                max_logit = logits[token_id];
                new_token = token_id;
            }
        }
        
        // Simple check for common EOS tokens
        if (new_token == 2 || new_token == 0) {
            LOG("Generated potential EOS token (id: %d), stopping generation\n", new_token);
            break;
        }
        
        LOG("Generated token %d: (id: %d, logit: %.4f)\n", i + 1, new_token, max_logit);
        
        // Decode the new token
        LOG("--- Decoding token %d ---\n", i + 1);
        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
            LOG_ERR("%s : failed to eval token %d\n", __func__, i + 1);
            return false;
        }
        LOG("--- Token %d decoded ---\n\n", i + 1);
        
        tokens.push_back(new_token);
    }

    LOG("=== GENERATION COMPLETED ===\n");
    LOG("Total tokens generated: %zu\n", tokens.size());
    
    return true;
}

int main(int argc, char ** argv) {
    tensor_diff_data diff_data;

    common_params params;

    // Add custom parameter parsing
    int target_layer = -1;
    std::string reference_file;
    double tolerance_abs = 1e-6;
    double tolerance_rel = 1e-4;
    
    // Create new argument list, excluding our custom parameters
    std::vector<char*> new_argv;
    new_argv.push_back(argv[0]);
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            target_layer = std::atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            reference_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--tolerance-abs") == 0 && i + 1 < argc) {
            tolerance_abs = std::atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--tolerance-rel") == 0 && i + 1 < argc) {
            tolerance_rel = std::atof(argv[i + 1]);
            i++;
        } else {
            new_argv.push_back(argv[i]);
        }
    }
    
    diff_data.target_layer = target_layer;
    diff_data.reference_file = reference_file;
    diff_data.tolerance_abs = tolerance_abs;
    diff_data.tolerance_rel = tolerance_rel;
    diff_data.analysis_enabled = !reference_file.empty();

    if (!common_params_parse(new_argv.size(), new_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        LOG_ERR("Usage: %s [options] --reference <gguf_file> [analysis_options]\n", argv[0]);
        LOG_ERR("  --reference <file>    Reference GGUF file with saved tensors\n");
        LOG_ERR("  --layer <n>           Monitor only layer n (0-based). Use -1 or omit to monitor all layers\n");
        LOG_ERR("  --tolerance-abs <f>   Absolute tolerance for differences (default: 1e-6)\n");
        LOG_ERR("  --tolerance-rel <f>   Relative tolerance for differences (default: 1e-4)\n");
        LOG_ERR("Examples:\n");
        LOG_ERR("  %s -m model.gguf -p \"Hello\" --reference saved_tensors.gguf\n", argv[0]);
        LOG_ERR("  %s -m model.gguf -p \"Hello\" --reference saved_tensors.gguf --layer 0 --tolerance-abs 1e-5\n", argv[0]);
        return 1;
    }

    if (!diff_data.analysis_enabled) {
        LOG_ERR("Error: --reference parameter is required\n");
        return 1;
    }

    LOG_INF("Tensor Difference Analyzer\n");
    LOG_INF("Reference file: %s\n", reference_file.c_str());
    if (target_layer >= 0) {
        LOG_INF("Monitoring layer: %d\n", target_layer);
    } else {
        LOG_INF("Monitoring all layers\n");
    }
    LOG_INF("Tolerance - Absolute: %.2e, Relative: %.2e\n", tolerance_abs, tolerance_rel);

    // Load reference tensors
    if (!load_reference_tensors(&diff_data)) {
        LOG_ERR("Failed to load reference tensors\n");
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // Set callback for tensor comparison
    params.cb_eval = ggml_debug_tensor_diff;
    params.cb_eval_user_data = &diff_data;
    params.warmup = false;

    // Initialize model and context
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // Print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    // Print analysis summary
    print_analysis_summary(&diff_data);

    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
} 