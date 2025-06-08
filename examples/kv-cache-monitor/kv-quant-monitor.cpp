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
#include <algorithm>  // for std::min
#include <cmath>      // for std::isfinite

// Enhanced data structure for KV quantization monitoring
struct kv_quant_trace_data {
    std::vector<uint8_t> temp_data;
    int step_count = 0;
    std::unordered_map<std::string, int> tensor_counts;
    int count_k = 0;
    int count_v = 0;
    bool enabled = true;
    bool verbose = false;
};

// Helper function to get tensor shape as string
static std::string ggml_ne_string(const ggml_tensor * t) {
    if (!t) return "null";
    return "[" + std::to_string(t->ne[0]) + "," +
               std::to_string(t->ne[1]) + "," +
               std::to_string(t->ne[2]) + "," +
               std::to_string(t->ne[3]) + "]";
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

static bool is_kv_quant_ref_tensor(const char * name) {
    if (!name) return false;
    std::string s(name);
    return s.rfind("k_quant_ref-0", 0) == 0 ||
           s.rfind("v_quant_ref-0", 0) == 0;
}

// Print basic tensor statistics
static void print_kv_quant_tensor_stats(const ggml_tensor * t, const char* tensor_name) {
    if (!t || !tensor_name) return;

    const int64_t nelements = ggml_nelements(t);
    const size_t type_size = ggml_type_size(t->type);
    const size_t total_bytes = ggml_nbytes(t);

    LOG("[KV-QUANT] %s:\n", tensor_name);
    LOG("  - Shape: %s\n", ggml_ne_string(t).c_str());
    LOG("  - Type: %s\n", ggml_type_name(t->type));
    LOG("  - Elements: %lld\n", (long long)nelements);
    LOG("  - Type size: %zu bytes\n", type_size);
    LOG("  - Total size: %zu bytes (%.2f KB)\n", total_bytes, total_bytes / 1024.0);
    LOG("\n");
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
        LOG("                                     sum = %f\n", sum);
    }
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

    const size_t new_nb[GGML_MAX_DIMS] = {
        sizeof(float), 
        sizeof(float) * src->ne[0], 
        sizeof(float) * src->ne[0] * src->ne[1], 
        sizeof(float) * src->ne[0] * src->ne[1] * src->ne[2]
    };
    
    LOG("DEQUANTIZED TENSOR: \n");
    ggml_print_tensor((uint8_t *)dst, GGML_TYPE_F32, src->ne, new_nb, 3);
}

static void print_tensor_shape_recursive(struct ggml_tensor * t, int depth = 0) {
    if (t == nullptr) return;

    // DEFENSIVE FIX: Prevent excessive recursion to avoid stack overflow
    if (depth > 10) {
        LOG("  [max recursion depth reached]\n");
        return;
    }

    //> raw kvcache tensor.
    if (t->name && (strcmp(t->name, "cache_k_quant_l0") == 0 || strcmp(t->name, "cache_v_quant_l0") == 0)) {
        // CRITICAL FIX: Allocate sufficient buffer to prevent overflow
        // We're processing up to 32 elements, so allocate 32 * sizeof(float) bytes
        const size_t all_elements = ggml_nelements(t);

        float* dst = (float*)malloc(all_elements * sizeof(float));
        if (!dst) {
            LOG("[KV-QUANT] ERROR: Failed to allocate %zu bytes for dequantization buffer\n", all_elements * sizeof(float));
            return;
        }

        // Initialize buffer to prevent using uninitialized memory
        memset(dst, 0, all_elements * sizeof(float));

        try {
            dequantize_tensor(t, dst);
        } catch (...) {
            LOG("[KV-QUANT] ERROR: Exception during dequantization\n");
        }

        // Safely free the buffer
        free(dst);
        dst = nullptr;
    }

    // Print indentation based on recursion depth
    std::string indent(depth * 2, ' ');

    // DEFENSIVE FIX: Add bounds checking for recursive calls
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] != nullptr) {
            // LOG("%s  Source %d:\n", indent.c_str(), i);
            print_tensor_shape_recursive(t->src[i], depth + 1);
        }
    }
}

// Enhanced callback to trace k/v quant tensors
static bool ggml_debug_kv_quant(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (kv_quant_trace_data *)user_data;

    if (t->name && (strncmp(t->name, "k_quant_ref-0", 13) == 0 || strncmp(t->name, "v_quant_ref-0", 13) == 0)) {
        LOG("+-----------------------------------------------------------------------------------------------+\n");
        ggml_print_tensor((uint8_t *)t->data, t->type, t->ne, t->nb, 3);
    }

    // Process the tensor if it's a KV quantization tensor
    if (is_kv_quant_tensor(t->name)) {
        const size_t all_elements = ggml_nelements(t);
        const size_t buffer_size = all_elements * sizeof(float);

        float* dst = (float*)malloc(buffer_size);
        if (!dst) {
            LOG("[KV-QUANT] ERROR: Failed to allocate %zu bytes for dequantization buffer\n", 4096 * sizeof(float));
        }

        // Initialize buffer to prevent using uninitialized memory
        memset(dst, 0, buffer_size);

        try {
            dequantize_tensor(t, dst);
        } catch (...) {
            LOG("[KV-QUANT] ERROR: Exception during dequantization\n");
        }
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

    // Parse custom arguments first
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            trace_data.verbose = true;
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
    LOG("Verbose mode: %s\n", trace_data.verbose ? "enabled" : "disabled");
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
    llama_backend_free();

    return 0;
}
