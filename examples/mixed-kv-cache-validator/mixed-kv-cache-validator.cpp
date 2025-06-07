#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <memory>
#include <functional>

/**
 * Mixed KV Cache Validation Tool
 * 
 * This tool validates the accuracy of mixed precision KV cache by:
 * 1. Running inference with standard unified cache
 * 2. Running the same inference with mixed precision cache
 * 3. Comparing the outputs (kqv_out tensors) to measure numerical differences
 * 4. Reporting detailed statistics about cache behavior and accuracy
 */

struct validation_data {
    std::vector<uint8_t> temp_data;
    
    // Reference outputs from unified cache
    std::unordered_map<std::string, std::vector<float>> reference_outputs;
    
    // Outputs from mixed cache
    std::unordered_map<std::string, std::vector<float>> mixed_outputs;
    
    // Statistics
    int step_count = 0;
    int layer_count = 0;
    std::unordered_map<std::string, int> tensor_counts;
    
    // Configuration
    int target_layer = -1; // -1 means validate all layers
    bool save_outputs = false;
    std::string output_file;
    
    // Validation state
    enum validation_mode {
        MODE_REFERENCE,  // Collecting reference outputs
        MODE_MIXED,      // Collecting mixed cache outputs
        MODE_COMPARE     // Comparing outputs
    } current_mode = MODE_REFERENCE;
};

static int extract_layer_number(const char* tensor_name) {
    if (!tensor_name) return -1;

    std::string name(tensor_name);

    // Look for kqv_out-N pattern
    size_t kqv_pos = name.find("kqv_out-");
    if (kqv_pos != std::string::npos) {
        size_t dash_pos = kqv_pos + 8; // Position after "kqv_out-"
        if (dash_pos < name.length()) {
            std::string layer_str = name.substr(dash_pos);
            // Extract only the numeric part
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

static bool is_kqv_out_tensor(const char* tensor_name) {
    if (!tensor_name) return false;
    std::string name(tensor_name);
    return name.find("kqv_out") != std::string::npos;
}

static bool should_validate_tensor(const char* tensor_name, int target_layer) {
    if (!is_kqv_out_tensor(tensor_name)) {
        return false;
    }

    if (target_layer == -1) {
        return true; // Validate all layers
    }

    int layer_num = extract_layer_number(tensor_name);
    return layer_num == target_layer;
}

static std::vector<float> extract_tensor_data(ggml_tensor* tensor, std::vector<uint8_t>& temp_buffer) {
    if (!tensor) return {};
    
    const bool is_host = ggml_backend_buffer_is_host(tensor->buffer);
    uint8_t* data = nullptr;

    if (!is_host) {
        auto n_bytes = ggml_nbytes(tensor);
        temp_buffer.resize(n_bytes);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), 0, n_bytes);
        data = temp_buffer.data();
    } else {
        data = (uint8_t*)tensor->data;
    }

    // Convert to float vector
    std::vector<float> result;
    
    size_t total_elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS && tensor->ne[i] > 0; ++i) {
        total_elements *= tensor->ne[i];
    }

    result.reserve(total_elements);

    for (size_t idx = 0; idx < total_elements; ++idx) {
        float v = 0.0f;

        if (tensor->type == GGML_TYPE_F32) {
            v = ((float*)data)[idx];
        } else if (tensor->type == GGML_TYPE_F16) {
            v = ggml_fp16_to_fp32(((ggml_fp16_t*)data)[idx]);
        } else {
            // Unsupported type, skip
            continue;
        }

        result.push_back(v);
    }

    return result;
}

static void compute_tensor_diff_stats(const std::vector<float>& ref, const std::vector<float>& mixed, 
                                     const std::string& tensor_name) {
    if (ref.size() != mixed.size()) {
        LOG_ERR("[VALIDATION] Size mismatch for %s: ref=%zu, mixed=%zu\n", 
                tensor_name.c_str(), ref.size(), mixed.size());
        return;
    }

    if (ref.empty()) {
        LOG("[VALIDATION] Empty tensor: %s\n", tensor_name.c_str());
        return;
    }

    // Compute statistics
    double sum_abs_diff = 0.0;
    double sum_rel_diff = 0.0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t valid_elements = 0;
    size_t large_diff_count = 0;
    
    const double LARGE_DIFF_THRESHOLD = 1e-3; // 0.1%
    
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(mixed[i])) {
            continue;
        }
        
        double abs_diff = std::abs(ref[i] - mixed[i]);
        double rel_diff = 0.0;
        
        if (std::abs(ref[i]) > 1e-8) {
            rel_diff = abs_diff / std::abs(ref[i]);
        }
        
        sum_abs_diff += abs_diff;
        sum_rel_diff += rel_diff;
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
        
        if (rel_diff > LARGE_DIFF_THRESHOLD) {
            large_diff_count++;
        }
        
        valid_elements++;
    }

    if (valid_elements == 0) {
        LOG("[VALIDATION] No valid elements in tensor: %s\n", tensor_name.c_str());
        return;
    }

    double avg_abs_diff = sum_abs_diff / valid_elements;
    double avg_rel_diff = sum_rel_diff / valid_elements;
    double large_diff_pct = (double)large_diff_count / valid_elements * 100.0;

    int layer_num = extract_layer_number(tensor_name.c_str());

    LOG("[VALIDATION] Layer %d - %s (elements: %zu)\n", 
        layer_num >= 0 ? layer_num : -1, tensor_name.c_str(), valid_elements);
    LOG("[VALIDATION]   Avg absolute diff: %.8f\n", avg_abs_diff);
    LOG("[VALIDATION]   Max absolute diff: %.8f\n", max_abs_diff);
    LOG("[VALIDATION]   Avg relative diff: %.6f%% \n", avg_rel_diff * 100.0);
    LOG("[VALIDATION]   Max relative diff: %.6f%%\n", max_rel_diff * 100.0);
    LOG("[VALIDATION]   Large diffs (>0.1%%): %zu (%.2f%%)\n", large_diff_count, large_diff_pct);
    
    // Quality assessment
    if (max_rel_diff < 0.001) { // < 0.1%
        LOG("[VALIDATION]   Quality: EXCELLENT (< 0.1%% diff)\n");
    } else if (max_rel_diff < 0.01) { // < 1%
        LOG("[VALIDATION]   Quality: GOOD (< 1%% diff)\n");
    } else if (max_rel_diff < 0.05) { // < 5%
        LOG("[VALIDATION]   Quality: ACCEPTABLE (< 5%% diff)\n");
    } else {
        LOG("[VALIDATION]   Quality: POOR (>= 5%% diff)\n");
    }
}

static bool ggml_validation_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (validation_data *) user_data;

    if (ask) {
        // Only interested in kqv_out related tensors
        return should_validate_tensor(t->name, cb_data->target_layer);
    }

    // Only process kqv_out related tensors
    if (!should_validate_tensor(t->name, cb_data->target_layer)) {
        return true;
    }

    cb_data->step_count++;
    cb_data->tensor_counts[std::string(t->name)]++;

    std::string tensor_key = std::string(t->name);
    
    LOG("[VALIDATION] Processing %s tensor: %s (mode: %s)\n", 
        cb_data->current_mode == validation_data::MODE_REFERENCE ? "REFERENCE" : "MIXED",
        tensor_key.c_str(),
        cb_data->current_mode == validation_data::MODE_REFERENCE ? "reference" : "mixed");

    // Extract tensor data
    std::vector<float> tensor_data = extract_tensor_data(t, cb_data->temp_data);
    
    if (tensor_data.empty()) {
        LOG("[VALIDATION] Failed to extract data from tensor: %s\n", tensor_key.c_str());
        return true;
    }

    // Store based on current mode
    if (cb_data->current_mode == validation_data::MODE_REFERENCE) {
        cb_data->reference_outputs[tensor_key] = tensor_data;
        LOG("[VALIDATION] Stored reference data for %s (%zu elements)\n", 
            tensor_key.c_str(), tensor_data.size());
    } else if (cb_data->current_mode == validation_data::MODE_MIXED) {
        cb_data->mixed_outputs[tensor_key] = tensor_data;
        LOG("[VALIDATION] Stored mixed data for %s (%zu elements)\n", 
            tensor_key.c_str(), tensor_data.size());
            
        // If we have both reference and mixed data, compare them
        auto ref_it = cb_data->reference_outputs.find(tensor_key);
        if (ref_it != cb_data->reference_outputs.end()) {
            LOG("\n=== COMPARING %s ===\n", tensor_key.c_str());
            compute_tensor_diff_stats(ref_it->second, tensor_data, tensor_key);
            LOG("=====================================\n\n");
        } else {
            LOG("[VALIDATION] No reference data found for %s\n", tensor_key.c_str());
        }
    }

    return true;
}

static bool run_validation_pass(llama_context * ctx, const common_params & params, 
                               validation_data* cb_data, const std::string& mode_name) {
    LOG("=== STARTING %s PASS ===\n", mode_name.c_str());
    
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    LOG("Processing %zu tokens with %s\n", tokens.size(), mode_name.c_str());

    // Process initial prompt
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("Failed to process initial prompt in %s\n", mode_name.c_str());
        return false;
    }

    // Generate a few tokens to test the cache
    for (int i = 0; i < std::min(8, params.n_predict); ++i) {
        LOG("=== %s: Generation step %d ===\n", mode_name.c_str(), i + 1);

        // Simple greedy sampling
        auto logits = llama_get_logits_ith(ctx, -1);
        auto n_vocab = llama_n_vocab(vocab);

        llama_token new_token = 0;
        float max_logit = logits[0];
        for (llama_token token_id = 1; token_id < n_vocab; token_id++) {
            if (logits[token_id] > max_logit) {
                max_logit = logits[token_id];
                new_token = token_id;
            }
        }

        LOG("%s: Generated token %d (id: %d, logit: %.4f)\n", 
            mode_name.c_str(), i + 1, new_token, max_logit);

        // Check for EOS
        if (new_token == 2 || new_token == 0) {
            LOG("%s: EOS token detected, stopping\n", mode_name.c_str());
            break;
        }

        // Decode the new token
        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
            LOG_ERR("%s: Failed to decode token %d\n", mode_name.c_str(), i + 1);
            return false;
        }

        tokens.push_back(new_token);
    }

    LOG("=== %s PASS COMPLETED ===\n\n", mode_name.c_str());
    return true;
}

static void print_validation_summary(const validation_data* cb_data) {
    LOG("\n=== MIXED KV CACHE VALIDATION SUMMARY ===\n");
    if (cb_data->target_layer >= 0) {
        LOG("Validated layer: %d\n", cb_data->target_layer);
    } else {
        LOG("Validated layers: All layers\n");
    }
    LOG("Total callback steps: %d\n", cb_data->step_count);
    LOG("Reference outputs collected: %zu\n", cb_data->reference_outputs.size());
    LOG("Mixed outputs collected: %zu\n", cb_data->mixed_outputs.size());
    
    LOG("\nTensors processed:\n");
    for (const auto& pair : cb_data->tensor_counts) {
        int layer_num = extract_layer_number(pair.first.c_str());
        LOG("  %s (layer %d): %d times\n", pair.first.c_str(), layer_num, pair.second);
    }
    
    // Overall assessment
    size_t compared_tensors = 0;
    for (const auto& mixed_pair : cb_data->mixed_outputs) {
        if (cb_data->reference_outputs.find(mixed_pair.first) != cb_data->reference_outputs.end()) {
            compared_tensors++;
        }
    }
    
    LOG("\nComparisons completed: %zu/%zu tensors\n", compared_tensors, cb_data->mixed_outputs.size());
    
    if (compared_tensors == cb_data->mixed_outputs.size() && compared_tensors > 0) {
        LOG("Status: SUCCESS - All mixed cache outputs validated\n");
    } else if (compared_tensors > 0) {
        LOG("Status: PARTIAL - Some outputs validated (%zu/%zu)\n", compared_tensors, cb_data->mixed_outputs.size());
    } else {
        LOG("Status: FAILED - No outputs could be compared\n");
    }
    LOG("==========================================\n\n");
}

int main(int argc, char ** argv) {
    validation_data cb_data;

    common_params params;

    // Parse custom parameters
    int target_layer = -1;
    bool save_outputs = false;
    std::string output_file;

    std::vector<char*> new_argv;
    new_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            target_layer = std::atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--save-outputs") == 0 && i + 1 < argc) {
            save_outputs = true;
            output_file = argv[i + 1];
            i++;
        } else {
            new_argv.push_back(argv[i]);
        }
    }

    cb_data.target_layer = target_layer;
    cb_data.save_outputs = save_outputs;
    cb_data.output_file = output_file;

    if (!common_params_parse(new_argv.size(), new_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        LOG_ERR("Usage: %s [options] [--layer <layer_number>] [--save-outputs <filename>]\n", argv[0]);
        LOG_ERR("  --layer <n>           Validate only layer n (0-based). Use -1 or omit to validate all layers.\n");
        LOG_ERR("  --save-outputs <file> Save comparison results to file.\n");
        LOG_ERR("Examples:\n");
        LOG_ERR("  %s -m model.gguf -p \"Hello\" --layer 0    # Validate only layer 0\n", argv[0]);
        LOG_ERR("  %s -m model.gguf -p \"Hello\"              # Validate all layers\n", argv[0]);
        return 1;
    }

    if (target_layer >= 0) {
        LOG_INF("Validating mixed KV cache for layer %d only\n", target_layer);
    } else {
        LOG_INF("Validating mixed KV cache for all layers\n");
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Force specific cache types for comparison
    params.warmup = false;
    
    // Phase 1: Run with unified cache (reference)
    LOG_INF("\n=== PHASE 1: COLLECTING REFERENCE DATA (UNIFIED CACHE) ===\n");
    params.use_mixed_kv_cache = false;  // Use unified cache
    params.cb_eval = ggml_validation_callback;
    params.cb_eval_user_data = &cb_data;
    cb_data.current_mode = validation_data::MODE_REFERENCE;

    common_init_result ref_init = common_init_from_params(params);
    if (!ref_init.model || !ref_init.context) {
        LOG_ERR("Failed to initialize reference model/context\n");
        return 1;
    }

    if (!run_validation_pass(ref_init.context.get(), params, &cb_data, "REFERENCE")) {
        LOG_ERR("Reference pass failed\n");
        return 1;
    }

    // Clear context for next phase
    ref_init.context.reset();
    ref_init.model.reset();

    // Phase 2: Run with mixed cache 
    LOG_INF("\n=== PHASE 2: COLLECTING MIXED CACHE DATA ===\n");
    params.use_mixed_kv_cache = true;   // Use mixed cache
    cb_data.current_mode = validation_data::MODE_MIXED;
    cb_data.step_count = 0;  // Reset counter

    common_init_result mixed_init = common_init_from_params(params);
    if (!mixed_init.model || !mixed_init.context) {
        LOG_ERR("Failed to initialize mixed cache model/context\n");
        return 1;
    }

    if (!run_validation_pass(mixed_init.context.get(), params, &cb_data, "MIXED")) {
        LOG_ERR("Mixed cache pass failed\n");
        return 1;
    }

    // Print final summary
    print_validation_summary(&cb_data);

    llama_backend_free();
    return 0;
} 