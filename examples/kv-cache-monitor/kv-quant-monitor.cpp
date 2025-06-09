#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>    // for std::pair
#include <algorithm>  // for std::min
#include <cmath>      // for std::isfinite
#include <limits>     // for std::numeric_limits

// Enhanced data structure for KV quantization monitoring
struct kv_quant_trace_data {
    int step_count = 0;
    ggml_context * trace_ctx = nullptr;
    
    std::unordered_map<std::string, int> string_counter;
    
    int increment_string_count(const std::string& str) {
        return ++string_counter[str];
    }
    
    // Helper function to check if a string exists in the counter
    bool contains_string(const std::string& str) const {
        return string_counter.find(str) != string_counter.end();
    }
    
    bool insert_string(const std::string& str) {
        if (contains_string(str)) {
            return false;
        }
        string_counter[str] = 0;
        return true;
    }
    
    // Helper function to get the count for a given string
    int get_string_count(const std::string& str) const {
        auto it = string_counter.find(str);
        return (it != string_counter.end()) ? it->second : 0;
    }
    
    // Clear all string counts
    void clear_string_counts() {
        string_counter.clear();
    }
    
    std::vector<std::pair<int, ggml_tensor*>> k_quant_ref_tensors;   
    std::vector<std::pair<int, ggml_tensor*>> v_quant_ref_tensors;   

    // Error analysis storage
    struct error_record {
        std::string name;           // tensor name (k_quant or v_quant)
        int step;                   // quantization step
        double mse;                 // mean squared error
        double mae;                 // mean absolute error
        double rmse;                // root mean squared error
        double max_error;           // maximum error
        double ref_norm;            // reference norm
        double relative_error;      // relative error (RMSE/RMS)
        double sqnr;                // signal-to-quantization-noise ratio (dB)
        double valid_elements;      // number of valid elements
        size_t elements;            // number of elements
        std::string assessment;     // quality assessment
    };
    
    std::vector<error_record> error_records;
    
    // Add error record
    void add_error_record(const std::string& name, int step, double mse, double mae, 
                         double rmse, double max_error, double ref_norm, 
                         double relative_error, double sqnr, double valid_elements, size_t elements, const std::string& assessment) {
        error_records.push_back({name, step, mse, mae, rmse, max_error, ref_norm, 
                               relative_error, sqnr, valid_elements, elements, assessment});
    }
};

// Helper function to get tensor shape as string
static std::string ggml_ne_string(const ggml_tensor * t) {
    if (!t) return "null";
    return "[" + std::to_string(t->ne[0]) + "," +
               std::to_string(t->ne[1]) + "," +
               std::to_string(t->ne[2]) + "," +
               std::to_string(t->ne[3]) + "]";
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    LOG("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG(" shape = [%ld, %ld, %ld, %ld]\n", ne[0], ne[1], ne[2], ne[3]);
    }
}

// Helper function to calculate numerical error between two tensors
static void calculate_tensor_error(ggml_tensor* ref_tensor, float* dequant_tensor, double* mse, double* mae, double* max_error, double* ref_norm, double* signal_power, double* valid_elements) {
    *mse = 0.0;
    *mae = 0.0;
    *max_error = 0.0;
    *ref_norm = 0.0;
    *signal_power = 0.0;
    *valid_elements = 0;
    
    size_t total_elements = ggml_nelements(ref_tensor);
    
    // Use linear indexing to avoid stride issues
    for (size_t i = 0; i < total_elements; i++) {
        float ref_val = *((float*)ref_tensor->data + i);
        float test_val = *(dequant_tensor + i);
        
        // Check for invalid values first
        if (!std::isfinite(ref_val) || !std::isfinite(test_val)) {
            continue;
        }

        (*valid_elements)++;
        float error = std::abs(ref_val - test_val);
        *mse += error * error;
        *mae += error;
        *max_error = std::max(*max_error, (double)error);
        *ref_norm += ref_val * ref_val;
        *signal_power += ref_val * ref_val;
    }
    
    if (*valid_elements > 0) {
        *mse /= *valid_elements;
        *mae /= *valid_elements;
        *signal_power /= *valid_elements;  // Average signal power
        *ref_norm = std::sqrt(*signal_power);  // RMS of reference signal
    } else {
        // Handle case where no valid elements found
        *mse = 0.0;
        *mae = 0.0;
        *max_error = 0.0;
        *ref_norm = 0.0;
        *signal_power = 0.0;
    }
}

// Function to get quality assessment string
static std::string get_quality_assessment(double relative_error) {
    if (relative_error < 0.01) {
        return "EXCELLENT";
    } else if (relative_error < 0.05) {
        return "GOOD";
    } else if (relative_error < 0.10) {
        return "ACCEPTABLE";
    } else {
        return "POOR";
    }
}

// Function to print error analysis table
static void print_error_table(const kv_quant_trace_data& data) {
    if (data.error_records.empty()) {
        LOG("No quantization error records found.\n");
        return;
    }
    
    LOG("\n");
    LOG("======================================================================================================\n");
    LOG("                             KV CACHE QUANTIZATION ERROR ANALYSIS                                     \n");
    LOG("======================================================================================================\n");
    LOG("| %-12s | %-4s | %-10s | %-10s | %-10s | %-10s | %-10s | %-9s | %-10s | %-10s |\n", 
        "Tensor", "Step", "MAE", "RMSE", "Max Error", "Ref Norm", "Rel Error", "SQNR(dB)", "Valid/Total", "Assessment");
    LOG("|--------------|------|------------|------------|------------|------------|------------|-----------|------------|------------|\n");
    
    for (const auto& record : data.error_records) {
        double valid_ratio = record.valid_elements / record.elements;
        LOG("| %-12s | %-4d | %10.6f | %10.6f | %10.6f | %10.6f | %9.4f%% | %9.2f | %5.1f | %-10s |\n",
            record.name.c_str(),
            record.step,
            record.mae,
            record.rmse,
            record.max_error,
            record.ref_norm,
            record.relative_error * 100.0,
            record.sqnr,
            valid_ratio,
            record.assessment.c_str());
    }
    
    LOG("======================================================================================================\n");
    
    // Summary statistics
    if (!data.error_records.empty()) {
        double avg_mae = 0.0, avg_rmse = 0.0, avg_rel_error = 0.0, avg_sqnr = 0.0, avg_valid_ratio = 0.0;
        double max_mae = 0.0, max_rmse = 0.0, max_rel_error = 0.0, max_sqnr = 0.0;
        double min_sqnr = std::numeric_limits<double>::max();
        size_t total_elements = 0;
        double total_valid_elements = 0.0;
        
        for (const auto& record : data.error_records) {
            avg_mae += record.mae;
            avg_rmse += record.rmse;
            avg_rel_error += record.relative_error;
            avg_sqnr += record.sqnr;
            total_valid_elements += record.valid_elements;
            total_elements += record.elements;
            
            max_mae = std::max(max_mae, record.mae);
            max_rmse = std::max(max_rmse, record.rmse);
            max_rel_error = std::max(max_rel_error, record.relative_error);
            max_sqnr = std::max(max_sqnr, record.sqnr);
            min_sqnr = std::min(min_sqnr, record.sqnr);
        }
        
        size_t count = data.error_records.size();
        avg_mae /= count;
        avg_rmse /= count;
        avg_rel_error /= count;
        avg_sqnr /= count;
        avg_valid_ratio = total_valid_elements / total_elements;
        
        LOG("\nSUMMARY STATISTICS:\n");
        LOG("------------------\n");
        LOG("Total quantization events: %zu\n", count);
        LOG("Average MAE:  %10.6f  |  Maximum MAE:  %10.6f\n", avg_mae, max_mae);
        LOG("Average RMSE: %10.6f  |  Maximum RMSE: %10.6f\n", avg_rmse, max_rmse);
        LOG("Average Rel:  %9.4f%%  |  Maximum Rel:  %9.4f%%\n", avg_rel_error * 100.0, max_rel_error * 100.0);
        LOG("Average SQNR: %9.2f dB |  Maximum SQNR: %9.2f dB |  Minimum SQNR: %9.2f dB\n", avg_sqnr, max_sqnr, min_sqnr);
        LOG("Valid/Total Elements: %zu/%zu (%.2f%%)\n", (size_t)total_valid_elements, total_elements, avg_valid_ratio * 100.0);
        LOG("Overall Assessment: %s\n", get_quality_assessment(avg_rel_error).c_str());
        LOG("======================================================================================================\n");
    }
}

// Enhanced detection for k_quant and v_quant tensors
static bool is_kv_quant_tensor(const char * name) {
    if (!name) return false;
    std::string s(name);

    // Exclude tensors whose names start with "cache"
    if (s.rfind("cache", 0) == 0) {
        return false;
    }

    // Only match exact names "k_quant-0" and "v_quant-0"
    return s == "k_quant_data-0" || s == "v_quant_data-0";
}

// Enhanced detection for cache-prefixed k_quant and v_quant tensors
static bool is_cache_kv_quant_tensor(const char * name) {
    if (!name) return false;
    std::string s(name);

    // Match tensors starting with "cache_k_quant" or "cache_v_quant"
    return s.rfind("cache_k_quant_l0", 0) == 0 ||
           s.rfind("cache_v_quant_l0", 0) == 0;
}

static bool is_cache_kv_tensor(const char * name) {
    if (!name) return false;
    std::string s(name);
    return s.rfind("cache_k_l0", 0) == 0 ||
           s.rfind("cache_v_l0", 0) == 0;
}

// Helper function to dequantize a tensor
static void dequantize_tensor(ggml_tensor * src, float * dst) {
    // Get the type traits for the source tensor
    const ggml_type_traits * traits = ggml_get_type_traits(src->type);

    size_t all_elements = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];

    // Perform the dequantization
    try {
        traits->to_float(src->data, dst, all_elements);
    } catch (...) {
        LOG("[KV-QUANT] ERROR: Exception during traits->to_float operation\n");
        return;
    }

    src->nb[0] = sizeof(float);
    src->nb[1] = sizeof(float) * src->ne[0];
    src->nb[2] = sizeof(float) * src->ne[0] * src->ne[1];
    src->nb[3] = sizeof(float) * src->ne[0] * src->ne[1] * src->ne[2];
}

// Enhanced callback to trace k/v quant tensors
static bool ggml_debug_kv_quant(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (kv_quant_trace_data *)user_data;

    if (t->name[0] && (strncmp(t->name, "k_quant_ref-0", 13) == 0 || strncmp(t->name, "v_quant_ref-0", 13) == 0)) {
        int step = data->increment_string_count(t->name);
        
        if (t->name[0] == 'k') {
            data->k_quant_ref_tensors.push_back(std::make_pair(step, t));
        } else if (t->name[0] == 'v') {
            data->v_quant_ref_tensors.push_back(std::make_pair(step, t));
        }
    }

    if (t->name[0] && (strcmp(t->name, "k_quant_data-0") == 0 || strcmp(t->name, "v_quant_data-0") == 0)) {
        int step = data->increment_string_count(t->name);

        ggml_tensor* quant_ref = nullptr;
        
        if (t->name[0] == 'k') {
            for (const auto &entry : data->k_quant_ref_tensors) {
                if (entry.first == step) {
                    quant_ref = entry.second;
                    break;
                }
            }
        } else {
            for (const auto &entry : data->v_quant_ref_tensors) {
                if (entry.first == step) {
                    quant_ref = entry.second;
                    break;
                }
            }
        }
        
        // LOG("[Quant] %s captured - Shape: %s, Type: %s, Elements: %zu\n", 
        //     t->name, ggml_ne_string(t).c_str(), ggml_type_name(t->type), ggml_nelements(t));
        
        float* dequantized_data = (float*)malloc(ggml_nelements(t) * sizeof(float));
        dequantize_tensor(t, dequantized_data);

        double mse = 0.0;
        double mae = 0.0;
        double max_error = 0.0;
        double ref_norm = 0.0;
        double signal_power = 0.0;
        double valid_elements = 0;

        //> Make sure the reference tensor exists and has valid data
        if (quant_ref && ggml_nelements(quant_ref) > 0) {
            calculate_tensor_error(quant_ref, dequantized_data, &mse, &mae, &max_error, &ref_norm, &signal_power, &valid_elements);
            
            double rmse = std::sqrt(mse);
            double relative_error = ref_norm > 0 ? rmse / ref_norm : 0.0;
            std::string assessment = get_quality_assessment(relative_error);
            
            // Calculate SQNR in dB: 10 * log10(signal_power / noise_power)
            double sqnr = (mse > 0.0 && signal_power > 0.0) ? 
                         10.0 * std::log10(signal_power / mse) : 0.0;
            
            data->add_error_record(t->name, step, mse, mae, rmse, max_error, ref_norm, 
                                  relative_error, sqnr, valid_elements, ggml_nelements(t), assessment);
            
            // Print both tensors for comparison
            LOG("[TENSOR COMPARISON] %s (step %d)\n", t->name, step);
            LOG("Reference tensor (original):\n");
            ggml_print_tensor((uint8_t*)quant_ref->data, quant_ref->type, quant_ref->ne, quant_ref->nb, 3);
            
            LOG("Dequantized tensor (after quantization):\n");
            // Create a temporary view of the dequantized data with the same dimensions
            ggml_tensor* dequant_view = ggml_new_tensor_4d(data->trace_ctx, GGML_TYPE_F32, 
                                                          quant_ref->ne[0], quant_ref->ne[1], 
                                                          quant_ref->ne[2], quant_ref->ne[3]);
            memcpy(dequant_view->data, dequantized_data, ggml_nelements(quant_ref) * sizeof(float));
            ggml_print_tensor((uint8_t*)dequant_view->data, GGML_TYPE_F32, dequant_view->ne, dequant_view->nb, 3);
            
            LOG("===========================================================================\n");
        } else {
            // Log when no reference tensor is found for debugging
            LOG("[DEBUG] No matching reference tensor found for %s step %d\n", t->name, step);
        }
        
        free(dequantized_data);
    }

    return true;
}

static void print_usage(const char* program_name) {
    fprintf(stderr, "Usage: %s [options]\n", program_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -v, --verbose     Enable verbose output with detailed tensor stats\n");
    fprintf(stderr, "  -h, --help        Show this help message\n");
    fprintf(stderr, "\nThis tool monitors KV cache quantization tensors during inference.\n");
}

int main(int argc, char ** argv) {
    kv_quant_trace_data trace_data;
    common_params params;
    // Create a temporary GGML context for tensor operations
    struct ggml_init_params ctx_params = {
        .mem_size   = 128 * 1024 * 1024,  // 16 MB should be enough for temporary operations
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    struct ggml_context * trace_ctx = ggml_init(ctx_params);
    if (!trace_ctx) {
        LOG_ERR("[KV-QUANT] ERROR: Failed to create temporary GGML context\n");
        return 1;
    }
    trace_data.trace_ctx = trace_ctx;

    // Parse custom arguments first
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            // Remove this argument from argv for common_params_parse
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc--;
            i--;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Parse common parameters
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        print_usage(argv[0]);
        return 1;
    }

    // Initialize llama backend
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Set up the callback
    params.cb_eval = ggml_debug_kv_quant;
    params.cb_eval_user_data = &trace_data;
    params.warmup = false;  // Disable warmup to see actual quantization

    LOG("=== KV Cache Quantization Monitor ===\n");
    LOG("Monitoring k_quant and v_quant tensors...\n\n");

    // NOTE: Following code will call graph_build, BUT it will not allocate the graph.
    auto init = common_init_from_params(params);
    auto * model    = init.model.get();
    auto * ctx    = init.context.get();

    if (!model || !ctx) {
        LOG_ERR("Failed to load model or create context\n");
        llama_backend_free();
        return 1;
    }

    // Tokenize prompt
    const auto prompt_tokens = common_tokenize(ctx, params.prompt, /*add_bos=*/true);
    if (prompt_tokens.empty()) {
        LOG_ERR("No tokens to process. Prompt: '%s'\n", params.prompt.c_str());
        llama_backend_free();
        return 1;
    }

    LOG("Processing %zu tokens from prompt: '%s'\n\n", prompt_tokens.size(), params.prompt.c_str());

    // Run initial prompt evaluation
    auto batch = llama_batch_get_one(const_cast<llama_token*>(prompt_tokens.data()), prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("Failed to decode prompt batch\n");
        llama_backend_free();
        return 1;
    }

    // Continue with generation to trigger more quantization events
    int n_predict = params.n_predict;
    int n_generated = 0;

    if (n_predict <= 0) {
        n_predict = 32; // Default to 32 tokens if not specified
    }

    LOG("\nGenerating %d tokens to trigger more quantization events...\n", n_predict);

    // Get model vocabulary for API calls
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Initialize sampler (using greedy sampling for simplicity)
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Main generation loop
    llama_token new_token_id = 0;
    while (n_generated < n_predict) {
        // Sample next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token_id) && !params.sampling.ignore_eos) {
            LOG("End of sequence reached\n");
            break;
        }

        // Add token to the context
        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode generation batch\n");
            break;
        }

        n_generated++;

        // Print token for visual feedback
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string token_str(buf, n);
            printf("%s", token_str.c_str());
            fflush(stdout);
        }

        // Check if we've accumulated enough quantization events
        if (trace_data.step_count > 50) {
            LOG("\nReached sufficient quantization events, stopping generation early.\n");
            break;
        }
    }

    printf("\n"); // New line after generation

    // Clean up sampler
    llama_sampler_free(smpl);
    
    LOG("\n=== QUANTIZATION ANALYSIS COMPLETE ===\n");
    LOG("Preparing error analysis table...\n\n");
    
    print_error_table(trace_data);
    
    // Clean up GGML context
    ggml_free(trace_data.trace_ctx);
    llama_backend_free();

    return 0;
}
