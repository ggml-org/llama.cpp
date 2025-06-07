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
 * Structure to hold tensor data for saving to GGUF
 */
struct tensor_save_info {
    std::string name;
    ggml_type type;
    std::vector<int64_t> ne;
    std::vector<uint8_t> data;

    tensor_save_info(const std::string& n, ggml_type t, const int64_t* dims, const uint8_t* d, size_t data_size)
        : name(n), type(t), data(d, d + data_size) {
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            ne.push_back(dims[i]);
        }
    }
};

/**
 * Callback data structure for tracking kqv_out tensors and their sources
 */
struct kqv_trace_data {
    std::vector<uint8_t> temp_data;
    int step_count = 0;
    std::unordered_map<std::string, int> tensor_counts;
    int target_layer = -1; // -1 means monitor all layers, >= 0 means monitor specific layer
    bool trace_sources = true; // whether to trace source tensors
    std::string save_file; // GGUF file to save tensors to
    std::vector<tensor_save_info> saved_tensors; // tensors to save
    bool save_enabled = false; // whether saving is enabled
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

    // Look for "_l" pattern (e.g., "kqv_out_l0")
    size_t l_pos = name.find("_l");
    if (l_pos != std::string::npos) {
        size_t start = l_pos + 2;
        if (start < name.length() && std::isdigit(name[start])) {
            size_t end = start;
            while (end < name.length() && std::isdigit(name[end])) {
                end++;
            }

            if (end > start) {
                std::string layer_str = name.substr(start, end - start);
                return std::stoi(layer_str);
            }
        }
    }

    // Look for "layer" or "blk" pattern
    size_t layer_pos = name.find("layer");
    if (layer_pos == std::string::npos) {
        layer_pos = name.find("blk");
    }

    if (layer_pos != std::string::npos) {
        size_t start = layer_pos;
        while (start < name.length() && !std::isdigit(name[start])) {
            start++;
        }

        if (start < name.length()) {
            size_t end = start;
            while (end < name.length() && std::isdigit(name[end])) {
                end++;
            }

            if (end > start) {
                std::string layer_str = name.substr(start, end - start);
                return std::stoi(layer_str);
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

static bool should_monitor_tensor(const char* tensor_name, int target_layer) {
    LOG("[KQV-TRACE] Checking tensor: %s, target_layer: %d\n", tensor_name, target_layer);
    if (!is_kqv_out_tensor(tensor_name)) {
        return false;
    }

    if (target_layer == -1) {
        return true; // 监控所有层
    }

    int layer_num = extract_layer_number(tensor_name);
    return layer_num == target_layer;
}

static void print_tensor_stats(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, const char* tensor_name) {
    if (data == nullptr || ne == nullptr) return;

    size_t total_elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS && ne[i] > 0; ++i) {
        total_elements *= ne[i];
    }

    if (total_elements == 0) return;

    double sum = 0.0, sum_sq = 0.0;
    double min_val = DBL_MAX, max_val = -DBL_MAX;
    size_t valid_elements = 0;

    for (size_t idx = 0; idx < total_elements; ++idx) {
        float v = 0.0f;

        if (type == GGML_TYPE_F32) {
            v = ((float*)data)[idx];
        } else if (type == GGML_TYPE_F16) {
            v = ggml_fp16_to_fp32(((ggml_fp16_t*)data)[idx]);
        } else {
            continue;
        }

        sum += v;
        sum_sq += v * v;
        min_val = std::min(min_val, (double)v);
        max_val = std::max(max_val, (double)v);
        valid_elements++;
    }

    if (valid_elements == 0) return;

    double mean = sum / valid_elements;
    double variance = (sum_sq / valid_elements) - (mean * mean);
    double std_dev = std::sqrt(variance);

    int layer_num = extract_layer_number(tensor_name);

    LOG("[KQV-TRACE] Layer %d - %s: shape=[%ld,%ld,%ld,%ld] type=%s elements=%zu\n",
        layer_num >= 0 ? layer_num : -1,
        tensor_name ? tensor_name : "unknown",
        ne[0], ne[1], ne[2], ne[3],
        ggml_type_name(type), valid_elements);

    LOG("[KQV-TRACE]   stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
        mean, std_dev, min_val, max_val);
}

static void print_source_tensor_info(struct ggml_tensor * tensor, int depth = 0) {
    if (!tensor || depth > 3) return; // Limit recursion depth

    std::string indent(depth * 2, ' ');

    if (depth == 0) {
        LOG("%s[OP] %s: op=%s, shape=[%ld,%ld,%ld,%ld], type=%s\n",
            indent.c_str(),
            tensor->name ? tensor->name : "unnamed",
            ggml_op_name(tensor->op),
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            ggml_type_name(tensor->type ));
    }

    // Recursively print source tensors
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (tensor->src[i]) {
            LOG("%s[SRC-%d] %s: op=%s, shape=[%ld,%ld,%ld,%ld], type=%s\n",
                indent.c_str(), i,
                tensor->name ? tensor->name : "unnamed",
                ggml_op_name(tensor->src[i]->op),
                tensor->src[i]->ne[0], tensor->src[i]->ne[1], tensor->src[i]->ne[2], tensor->src[i]->ne[3],
                ggml_type_name(tensor->src[i]->type));
            print_source_tensor_info(tensor->src[i], depth + 1);
        }
    }
}

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

/**
 * Save tensor data for later writing to GGUF file
 */
static void save_tensor_data(kqv_trace_data* cb_data, struct ggml_tensor* tensor, const std::string& prefix = "") {
    if (!cb_data->save_enabled || !tensor) return;

    // Get tensor data
    const bool is_host = ggml_backend_buffer_is_host(tensor->buffer);
    uint8_t* data = nullptr;

    if (!is_host) {
        auto n_bytes = ggml_nbytes(tensor);
        cb_data->temp_data.resize(n_bytes);
        ggml_backend_tensor_get(tensor, cb_data->temp_data.data(), 0, n_bytes);
        data = cb_data->temp_data.data();
    } else {
        data = (uint8_t*)tensor->data;
    }

    // Create unique name with prefix and step count
    std::string save_name = prefix.empty() ?
        std::string(tensor->name ? tensor->name : "unnamed") :
        prefix + "_" + std::string(tensor->name ? tensor->name : "unnamed");
    save_name += "_step_" + std::to_string(cb_data->step_count);

    // Save tensor info
    cb_data->saved_tensors.emplace_back(
        save_name,
        tensor->type,
        tensor->ne,
        data,
        ggml_nbytes(tensor)
    );

    LOG("[GGUF-SAVE] Saved tensor: %s, type: %s, size: %zu bytes\n",
        save_name.c_str(), ggml_type_name(tensor->type), ggml_nbytes(tensor));
}

/**
 * Write all saved tensors to GGUF file
 */
static bool write_tensors_to_gguf(const kqv_trace_data* cb_data) {
    if (!cb_data->save_enabled || cb_data->save_file.empty() || cb_data->saved_tensors.empty()) {
        return true; // Nothing to save
    }

    LOG("[GGUF-SAVE] Writing %zu tensors to file: %s\n", cb_data->saved_tensors.size(), cb_data->save_file.c_str());

    // Create GGUF context
    struct gguf_context* ctx = gguf_init_empty();
    if (!ctx) {
        LOG_ERR("[GGUF-SAVE] Failed to create GGUF context\n");
        return false;
    }

    // Add metadata
    gguf_set_val_str(ctx, "kqv_trace.description", "KQV output tensors and their inputs traced from llama.cpp");
    gguf_set_val_i32(ctx, "kqv_trace.total_steps", cb_data->step_count);
    gguf_set_val_i32(ctx, "kqv_trace.target_layer", cb_data->target_layer);
    gguf_set_val_bool(ctx, "kqv_trace.trace_sources", cb_data->trace_sources);
    gguf_set_val_i32(ctx, "kqv_trace.tensor_count", (int32_t)cb_data->saved_tensors.size());

    // Create GGML context for tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024ull * 1024ull * 1024ull, // 1GB should be enough
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context* ctx_data = ggml_init(params);
    if (!ctx_data) {
        LOG_ERR("[GGUF-SAVE] Failed to create GGML context\n");
        gguf_free(ctx);
        return false;
    }

    // Add tensors to GGUF
    for (const auto& tensor_info : cb_data->saved_tensors) {
        // Create GGML tensor
        struct ggml_tensor* tensor = ggml_new_tensor(ctx_data, tensor_info.type, GGML_MAX_DIMS, tensor_info.ne.data());
        if (!tensor) {
            LOG_ERR("[GGUF-SAVE] Failed to create tensor: %s\n", tensor_info.name.c_str());
            continue;
        }

        ggml_set_name(tensor, tensor_info.name.c_str());

        // Copy data
        memcpy(tensor->data, tensor_info.data.data(), tensor_info.data.size());

        // Add to GGUF
        gguf_add_tensor(ctx, tensor);

        LOG("[GGUF-SAVE] Added tensor to GGUF: %s\n", tensor_info.name.c_str());
    }

    // Write to file
    bool success = gguf_write_to_file(ctx, cb_data->save_file.c_str(), false);
    if (success) {
        LOG("[GGUF-SAVE] Successfully wrote GGUF file: %s\n", cb_data->save_file.c_str());
    } else {
        LOG_ERR("[GGUF-SAVE] Failed to write GGUF file: %s\n", cb_data->save_file.c_str());
    }

    // Cleanup
    ggml_free(ctx_data);
    gguf_free(ctx);

    return success;
}

/**
 * GGML operations callback during the graph execution.
 */
static bool ggml_debug_kqv_trace(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (kqv_trace_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        // Only interested in kqv_out related tensors
        return should_monitor_tensor(t->name, cb_data->target_layer);
    }

    // Only process kqv_out related tensors
    if (!should_monitor_tensor(t->name, cb_data->target_layer)) {
        return true;
    }

    cb_data->step_count++;
    cb_data->tensor_counts[std::string(t->name)]++;

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    LOG("\n=== KQV_OUT TENSOR DETECTED ===\n");
    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
         t->name, ggml_type_name(t->type), ggml_op_desc(t),
         src0 ? src0->name : "NULL", src0 ? ggml_ne_string(src0).c_str() : "",
         src1 ? src1_str : "",
         ggml_ne_string(t).c_str());

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->temp_data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->temp_data.data(), 0, n_bytes);
    }

    // Print kqv_out tensor statistics
    uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->temp_data.data();
    print_tensor_stats(data, t->type, t->ne, t->nb, t->name);

    // Save tensors recursively if enabled
    if (cb_data->save_enabled) {
        // Recursive function to save all tensors in the computation graph
        std::function<void(struct ggml_tensor*, const std::string&, int)> save_tensor_recursive =
            [&](struct ggml_tensor* tensor, const std::string& prefix, int depth) {
                if (!tensor || depth > 3) return; // Limit recursion depth to avoid infinite loops

                // Save current tensor
                std::string tensor_name = std::string(tensor->name ? tensor->name : "unnamed");
                LOG("[KQV-TRACE] Saving tensor: %s with prefix %s (depth %d)\n",
                    tensor_name.c_str(), prefix.c_str(), depth);

                save_tensor_data(cb_data, tensor, prefix);

                // Recursively save source tensors
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (tensor->src[i]) {
                        std::string src_prefix = "src" + std::to_string(i);
                        save_tensor_recursive(const_cast<struct ggml_tensor*>(tensor->src[i]), src_prefix, depth + 1);
                    }
                }
            };

        // Start recursive saving from the main tensor
        save_tensor_recursive(t, "kqv_out", 0);
    }

    // Trace source tensors
    if (cb_data->trace_sources) {
        LOG("\n[KQV-TRACE] Source tensor hierarchy:\n");
        print_source_tensor_info(t);
    }

    LOG("===============================\n\n");

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

        // Sample next token using simple greedy approach
        auto logits = llama_get_logits_ith(ctx, -1);
        auto n_vocab = llama_n_vocab(vocab);

        // Find token with highest probability (greedy sampling)
        llama_token new_token = 0;
        float max_logit = logits[0];
        for (llama_token token_id = 1; token_id < n_vocab; token_id++) {
            if (logits[token_id] > max_logit) {
                max_logit = logits[token_id];
                new_token = token_id;
            }
        }

        // Simple check for common EOS tokens (this is a simplified approach)
        if (new_token == 2 || new_token == 0) { // Common EOS token IDs
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

        // Add to tokens for potential future use
        tokens.push_back(new_token);
    }

    LOG("=== GENERATION COMPLETED ===\n");
    LOG("Total tokens generated: %zu\n", tokens.size());

    return true;
}

int main(int argc, char ** argv) {
    kqv_trace_data cb_data;

    common_params params;

    // Add custom parameter parsing
    int target_layer = -1; // Default: monitor all layers
    bool trace_sources = true; // Default: trace source tensors
    std::string save_file; // GGUF file to save tensors to

    // Create new argument list, excluding our custom parameters
    std::vector<char*> new_argv;
    new_argv.push_back(argv[0]); // Keep program name

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            target_layer = std::atoi(argv[i + 1]);
            i++; // Skip next parameter (layer number)
        } else if (strcmp(argv[i], "--no-trace-sources") == 0) {
            trace_sources = false;
        } else if (strcmp(argv[i], "--save-gguf") == 0 && i + 1 < argc) {
            save_file = argv[i + 1];
            i++; // Skip next parameter (filename)
        } else {
            new_argv.push_back(argv[i]);
        }
    }

    cb_data.target_layer = target_layer;
    cb_data.trace_sources = trace_sources;
    cb_data.save_file = save_file;
    cb_data.save_enabled = !save_file.empty();

    if (!common_params_parse(new_argv.size(), new_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        LOG_ERR("Usage: %s [options] [--layer <layer_number>] [--no-trace-sources] [--save-gguf <filename>]\n", argv[0]);
        LOG_ERR("  --layer <n>           Monitor only layer n (0-based). Use -1 or omit to monitor all layers.\n");
        LOG_ERR("  --no-trace-sources    Disable tracing of source tensors.\n");
        LOG_ERR("  --save-gguf <file>    Save traced tensors to GGUF file.\n");
        LOG_ERR("Examples:\n");
        LOG_ERR("  %s -m model.gguf -p \"Hello\" --layer 0    # Monitor only layer 0\n", argv[0]);
        LOG_ERR("  %s -m model.gguf -p \"Hello\"              # Monitor all layers\n", argv[0]);
        LOG_ERR("  %s -m model.gguf -p \"Hello\" --save-gguf tensors.gguf  # Save tensors to file\n", argv[0]);
        return 1;
    }

    if (target_layer >= 0) {
        LOG_INF("Monitoring kqv_out tensors for layer %d only\n", target_layer);
    } else {
        LOG_INF("Monitoring kqv_out tensors for all layers\n");
    }

    if (trace_sources) {
        LOG_INF("Source tensor tracing enabled\n");
    } else {
        LOG_INF("Source tensor tracing disabled\n");
    }

    if (cb_data.save_enabled) {
        LOG_INF("Tensor saving enabled, output file: %s\n", save_file.c_str());
    } else {
        LOG_INF("Tensor saving disabled\n");
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug_kqv_trace;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    // Write saved tensors to GGUF file
    if (cb_data.save_enabled) {
        if (!write_tensors_to_gguf(&cb_data)) {
            LOG_ERR("Failed to write tensors to GGUF file\n");
            return 1;
        }
    }

    // Output kqv_out monitoring statistics
    LOG("\n=== KQV_OUT Monitoring Summary ===\n");
    if (cb_data.target_layer >= 0) {
        LOG("Monitored layer: %d\n", cb_data.target_layer);
    } else {
        LOG("Monitored layers: All layers\n");
    }
    LOG("Source tracing: %s\n", cb_data.trace_sources ? "Enabled" : "Disabled");
    LOG("Tensor saving: %s\n", cb_data.save_enabled ? "Enabled" : "Disabled");
    if (cb_data.save_enabled) {
        LOG("Output file: %s\n", cb_data.save_file.c_str());
        LOG("Tensors saved: %zu\n", cb_data.saved_tensors.size());
    }
    LOG("Total callback steps: %d\n", cb_data.step_count);
    LOG("KQV_OUT tensors encountered:\n");
    for (const auto& pair : cb_data.tensor_counts) {
        int layer_num = extract_layer_number(pair.first.c_str());
        LOG("  %s (layer %d): %d times\n", pair.first.c_str(), layer_num, pair.second);
    }
    LOG("===================================\n\n");

    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
