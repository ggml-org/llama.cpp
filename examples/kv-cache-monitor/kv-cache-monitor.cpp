#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cctype>
#include <algorithm>

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
    int step_count = 0;
    std::unordered_map<std::string, int> tensor_counts;
    int target_layer = -1; // -1 means monitor all layers, >= 0 means monitor specific layer
};

static int extract_layer_number(const char* tensor_name) {
    if (!tensor_name) return -1;
    
    std::string name(tensor_name);
    
    size_t layer_pos = name.find("layer");
    if (layer_pos == std::string::npos) {
        layer_pos = name.find("blk");
    }
    
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

static bool is_kv_cache_tensor(const char* tensor_name) {
    if (!tensor_name) return false;
    std::string name(tensor_name);
    return name.find("mixedcache_k") != std::string::npos || 
           name.find("mixedcache_v") != std::string::npos ||
           name.find("kv_cache") != std::string::npos ||
           (name.find(".k") != std::string::npos && name.find("layer") != std::string::npos) ||
           (name.find(".v") != std::string::npos && name.find("layer") != std::string::npos);
}

// 检查是否应该监控这个张量（基于层过滤）
static bool should_monitor_tensor(const char* tensor_name, int target_layer) {
    if (!is_kv_cache_tensor(tensor_name)) {
        return false;
    }
    int layer_num = extract_layer_number(tensor_name);

    // 如果包含"copy of"这个字符串，可以return true
    if (tensor_name && strstr(tensor_name, "copy of") != nullptr && layer_num == target_layer) {
        return true;
    }
    
    // 只处理严格以 "(view)" 结尾的张量
    std::string name(tensor_name);
    if (name.length() < 6 || name.substr(name.length() - 6) != "(view)") {
        return false;
    }
    
    if (target_layer == -1) {
        return true; // 监控所有层
    }

    return layer_num == target_layer;
}

static void print_kv_cache_stats(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, const char* tensor_name) {
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
    
    LOG("[KV-CACHE] Layer %d - %s: shape=[%ld,%ld,%ld,%ld], stride=[%ld,%ld,%ld,%ld], type=%s elements=%zu\n",
        layer_num >= 0 ? layer_num : -1,
        tensor_name ? tensor_name : "unknown",
        ne[0], ne[1], ne[2], ne[3], 
        nb[0], nb[1], nb[2], nb[3],
        ggml_type_name(type), valid_elements);
    
    LOG("[KV-CACHE]   stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
        mean, std_dev, min_val, max_val);
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

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, const char* tensor_name) {
    GGML_ASSERT(n > 0);
    
    std::string name(tensor_name ? tensor_name : "");
    
    // 判断是否为KV cache（仅包含 "(view)" 后缀）还是projection层输出（"copy of ..."）
    bool is_pure_kv_cache = (name.find(" (view)") != std::string::npos) && 
                           (name.find("copy of") == std::string::npos) &&
                           (name.find(" (view)") + 7 == name.length());
    
    if (is_pure_kv_cache) {
        // 这是纯KV cache，按照token顺序打印
        bool is_v_cache = (tensor_name && strstr(tensor_name, "cache_v") && 
                          name.find(" (view)") != std::string::npos &&
                          name.find(" (view)") + 7 == name.length());
        
        int64_t head_dim, n_head, n_tokens, batch;
        int64_t max_head_dim, max_n_head, max_n_tokens, max_batch;
        
        if (is_v_cache) {
            // V cache layout: [tokens, n_head, head_dim, batch]
            head_dim    = ne[0];
            n_head      = ne[1];
            n_tokens    = ne[2];
            batch       = ne[3];
            
            max_n_tokens = std::min(n_tokens, (int64_t)16);
            max_n_head = std::min(n_head, (int64_t)2);
            max_head_dim = std::min(head_dim, (int64_t)4);
            max_batch = batch;
            
            LOG("V Cache tensor shape: [tokens=%ld, n_head=%ld, head_dim=%ld, batch=%ld]\n", 
                n_tokens, n_head, head_dim, batch);
            LOG("Showing: [tokens=0..%ld, n_head=0..%ld, head_dim=0..%ld, batch=0..%ld]\n",
                max_n_tokens-1, max_n_head-1, max_head_dim-1, max_batch-1);
        } else {
            // K cache layout: [head_dim, n_head, tokens, batch]
            head_dim    = ne[0];
            n_head      = ne[1];
            n_tokens    = ne[2];
            batch       = ne[3];
            
            max_head_dim = std::min(head_dim, (int64_t)4);
            max_n_head = std::min(n_head, (int64_t)2);
            max_n_tokens = std::min(n_tokens, (int64_t)16);
            max_batch = batch;
            
            LOG("K Cache tensor shape: [head_dim=%ld, n_head=%ld, tokens=%ld, batch=%ld]\n", 
                head_dim, n_head, n_tokens, batch);
            LOG("Showing: [head_dim=0..%ld, n_head=0..%ld, tokens=0..%ld, batch=0..%ld]\n",
                max_head_dim-1, max_n_head-1, max_n_tokens-1, max_batch-1);
        }
        
        float total_sum = 0;
        
        // 按照token顺序打印KV cache
        for (int64_t b = 0; b < max_batch; b++) {
            LOG("  Batch[%ld]:\n", b);
            
            for (int64_t token = 0; token < max_n_tokens; token++) {
                LOG("    Token[%ld]:\n", token);
                
                for (int64_t head = 0; head < max_n_head; head++) {
                    LOG("      Head[%ld]: [", head);
                    
                    float head_sum = 0;
                    for (int64_t dim = 0; dim < max_head_dim; dim++) {
                        size_t i;
                        if (is_v_cache) {
                            // V cache: [tokens, n_head, head_dim, batch]
                            // i = b * nb[3] + dim * nb[2] + head * nb[1] + token * nb[0];
                            i = b * nb[3] + token * nb[2] + head * nb[1] + dim * nb[0];
                        } else {
                            // K cache: [head_dim, n_head, tokens, batch]
                            i = b * nb[3] + token * nb[2] + head * nb[1] + dim * nb[0];
                        }
                        
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
                        
                        LOG("%8.4f", v);
                        head_sum += v;
                        total_sum += v;
                        
                        if (dim < max_head_dim - 1) LOG(", ");
                    }
                    
                    if (head_dim > max_head_dim) {
                        LOG(", ... (%ld more dims)", head_dim - max_head_dim);
                    }
                    LOG("] sum=%.4f\n", head_sum);
                }
                
                if (n_head > max_n_head) {
                    LOG("      ... (%ld more heads)\n", n_head - max_n_head);
                }
            }
            
            if (n_tokens > max_n_tokens) {
                LOG("    ... (%ld more tokens)\n", n_tokens - max_n_tokens);
            }
        }
        
        LOG("Total sum = %.6f\n", total_sum);
    } else {
        // 这是projection层的输出（"copy of ..."），按照正常多头方式打印
        LOG("Projection tensor shape: [%ld, %ld, %ld, %ld]\n", ne[0], ne[1], ne[2], ne[3]);
        
        // 假设projection层输出的维度排布为 [head_dim, n_head, n_tokens, batch]
        int64_t head_dim = ne[0];
        int64_t n_head = ne[1];
        int64_t n_tokens = ne[2];
        int64_t batch = ne[3];
        
        int64_t max_head_dim = std::min(head_dim, (int64_t)4);
        int64_t max_n_head = std::min(n_head, (int64_t)2);
        int64_t max_n_tokens = std::min(n_tokens, (int64_t)4);
        int64_t max_batch = batch;
        
        LOG("Showing: [head_dim=0..%ld, n_head=0..%ld, n_tokens=0..%ld, batch=0..%ld]\n",
            max_head_dim-1, max_n_head-1, max_n_tokens-1, max_batch-1);
        
        float total_sum = 0;
        
        // 按照多头方式打印projection输出
        for (int64_t b = 0; b < max_batch; b++) {
            LOG("  Batch[%ld]:\n", b);
            
            for (int64_t head = 0; head < max_n_head; head++) {
                LOG("    Head[%ld]:\n", head);
                
                for (int64_t token = 0; token < max_n_tokens; token++) {
                    LOG("      Token[%ld]: [", token);
                    
                    float token_sum = 0;
                    for (int64_t dim = 0; dim < max_head_dim; dim++) {
                        // projection输出: [head_dim, n_head, n_tokens, batch]
                        size_t i = b * nb[3] + token * nb[2] + head * nb[1] + dim * nb[0];
                        
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
                        
                        LOG("%8.4f", v);
                        token_sum += v;
                        total_sum += v;
                        
                        if (dim < max_head_dim - 1) LOG(", ");
                    }
                    
                    if (head_dim > max_head_dim) {
                        LOG(", ... (%ld more dims)", head_dim - max_head_dim);
                    }
                    LOG("] sum=%.4f\n", token_sum);
                }
                
                if (n_tokens > max_n_tokens) {
                    LOG("      ... (%ld more tokens)\n", n_tokens - max_n_tokens);
                }
            }
            
            if (n_head > max_n_head) {
                LOG("    ... (%ld more heads)\n", n_head - max_n_head);
            }
        }
        
        LOG("Total sum = %.6f\n", total_sum);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        // 只对 KV cache 相关的张量感兴趣
        return should_monitor_tensor(t->name, cb_data->target_layer);
    }

    // 只处理 KV cache 相关的张量
    if (!should_monitor_tensor(t->name, cb_data->target_layer)) {
        return true;
    }

    cb_data->step_count++;
    cb_data->tensor_counts[std::string(t->name)]++;

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
         t->name, ggml_type_name(t->type), ggml_op_desc(t),
         src0 ? src0->name : "NULL", src0 ? ggml_ne_string(src0).c_str() : "",
         src1 ? src1_str : "",
         ggml_ne_string(t).c_str());

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    // 对 KV cache 张量进行统计分析
    uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
    print_kv_cache_stats(data, t->type, t->ne, t->nb, t->name);

    // 如果不是量化类型，也打印详细数据（限制输出量）
    if (!ggml_is_quantized(t->type)) {
        ggml_print_tensor(data, t->type, t->ne, t->nb, 4, t->name); // 减少输出量
    }

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;

    common_params params;

    // 添加自定义参数解析
    int target_layer = -1; // 默认监控所有层
    
    // 简单的参数解析，查找 --layer 参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            target_layer = std::atoi(argv[i + 1]);
            // 从参数列表中移除这两个参数，避免影响common_params_parse
            for (int j = i; j < argc - 2; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }
    
    cb_data.target_layer = target_layer;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        LOG_ERR("Usage: %s [options] --layer <layer_number>\n", argv[0]);
        LOG_ERR("  --layer <n>  Monitor only layer n (0-based). Use -1 or omit to monitor all layers.\n");
        LOG_ERR("Examples:\n");
        LOG_ERR("  %s -m model.gguf -p \"Hello\" --layer 0    # Monitor only layer 0\n", argv[0]);
        LOG_ERR("  %s -m model.gguf -p \"Hello\"              # Monitor all layers\n", argv[0]);
        return 1;
    }

    if (target_layer >= 0) {
        LOG_INF("Monitoring KV cache for layer %d only\n", target_layer);
    } else {
        LOG_INF("Monitoring KV cache for all layers\n");
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
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

    // 输出 KV cache 监控统计信息
    LOG("\n=== KV Cache Monitoring Summary ===\n");
    if (cb_data.target_layer >= 0) {
        LOG("Monitored layer: %d\n", cb_data.target_layer);
    } else {
        LOG("Monitored layers: All layers\n");
    }
    LOG("Total callback steps: %d\n", cb_data.step_count);
    LOG("KV Cache tensors encountered:\n");
    for (const auto& pair : cb_data.tensor_counts) {
        int layer_num = extract_layer_number(pair.first.c_str());
        LOG("  %s (layer %d): %d times\n", pair.first.c_str(), layer_num, pair.second);
    }
    LOG("=====================================\n\n");

    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
