#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <map>

struct kqv_tensor_params {
    std::string input_file;
    bool verbose = false;
    bool show_data_stats = false;
    bool show_shape_details = false;
    int target_step = -1; // -1 means show all steps
    int target_layer = -1; // -1 means show all layers
};

static void print_usage(const char* program_name) {
    LOG_INF("Usage: %s [options]\n", program_name);
    LOG_INF("Options:\n");
    LOG_INF("  -i, --input <file>        Input GGUF file to read (required)\n");
    LOG_INF("  --shapes                  Show detailed shape and stride information\n");
    LOG_INF("  -h, --help                Show this help message\n");
    LOG_INF("\n");
    LOG_INF("Description:\n");
    LOG_INF("  Specialized tool to read and analyze kqv_out tensors and their direct\n");
    LOG_INF("  source tensors (QKV, mask) from GGUF files saved by kqv-trace-monitor.\n");
    LOG_INF("\n");
    LOG_INF("Examples:\n");
    LOG_INF("  %s -i tensors.gguf                # Basic tensor listing\n", program_name);
    LOG_INF("  %s -i tensors.gguf --shapes       # Show detailed shape information\n", program_name);
}

static bool parse_args(int argc, char** argv, kqv_tensor_params& params) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (++i >= argc) {
                LOG_ERR("Error: --input requires a filename\n");
                return false;
            }
            params.input_file = argv[i];
        } else if (strcmp(argv[i], "--shapes") == 0) {
            params.show_shape_details = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return false;
        } else {
            LOG_ERR("Error: Unknown argument '%s'\n", argv[i]);
            return false;
        }
    }

    if (params.input_file.empty()) {
        LOG_ERR("Error: Input file is required (use -i or --input)\n");
        return false;
    }

    return true;
}

static int extract_step_from_name(const std::string& name) {
    size_t step_pos = name.find("_step_");
    if (step_pos != std::string::npos) {
        size_t start = step_pos + 6; // Position after "_step_"
        if (start < name.length()) {
            size_t end = start;
            while (end < name.length() && std::isdigit(name[end])) {
                end++;
            }
            if (end > start) {
                try {
                    return std::stoi(name.substr(start, end - start));
                } catch (...) {
                    return -1;
                }
            }
        }
    }
    return -1;
}

static int extract_layer_from_name(const std::string& name) {
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

static bool is_kqv_out_tensor(const std::string& name) {
    return name.find("kqv_out_") == 0;
}

static bool is_src_tensor(const std::string& name) {
    return name.find("src") == 0;
}

struct tensor_stats {
    double mean = 0.0;
    double std_dev = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    size_t elements = 0;
};

static tensor_stats calculate_tensor_stats(const ggml_tensor* tensor) {
    tensor_stats stats;
    
    if (!tensor || !tensor->data) {
        return stats;
    }

    size_t total_elements = ggml_nelements(tensor);
    if (total_elements == 0) {
        return stats;
    }

    float sum = 0.0, sum_sq = 0.0;
    size_t valid_elements = 0;

    for (size_t i = 0; i < total_elements; ++i) {
        float value = 0.0f;
        
        if (tensor->type == GGML_TYPE_F32) {
            value = ((float*)tensor->data)[i];
        } else if (tensor->type == GGML_TYPE_F16) {
            value = ggml_fp16_to_fp32(((ggml_fp16_t*)tensor->data)[i]);
        } else {
            LOG_ERR("Unsupported Type.");
            return stats;
        }

        sum += value;
        sum_sq += value * value;
        stats.min_val = std::min(stats.min_val, (double)value);
        stats.max_val = std::max(stats.max_val, (double)value);
        valid_elements++;
    }

    if (valid_elements > 0) {
        stats.mean = sum / valid_elements;
        double variance = (sum_sq / valid_elements) - (stats.mean * stats.mean);
        stats.std_dev = std::sqrt(variance);
        stats.elements = valid_elements;
    }

    return stats;
}

static void print_tensor_info(const ggml_tensor* tensor, const std::string& name, 
                             const kqv_tensor_params& params, int index) {
    
    int step = extract_step_from_name(name);
    int layer = extract_layer_from_name(name);
    std::string tensor_type = is_kqv_out_tensor(name) ? "KQV_OUT" : "SRC";
    
    // Print basic tensor info in a more compact format
    LOG_INF("[%d] %s: %s %s", index, name.c_str(), ggml_type_name(tensor->type), tensor_type.c_str());
    if (step >= 0) LOG_INF(" step=%d", step);
    if (layer >= 0) LOG_INF(" layer=%d", layer);
    LOG_INF(" shape=[%ld,%ld,%ld,%ld] size=%zu\n", 
           tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], ggml_nbytes(tensor));

    // Only print detailed shape info if requested
    if (params.verbose && params.show_shape_details) {
        LOG_INF("    stride=[%zu,%zu,%zu,%zu] ptr=%p\n", 
               tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3], tensor->data);
    }

    // Print statistics if requested
    if (params.show_data_stats) {
        tensor_stats stats = calculate_tensor_stats(tensor);
        if (stats.elements > 0) {
            LOG_INF("    stats: n=%zu mean=%.4f std=%.4f min=%.4f max=%.4f\n",
                   stats.elements, stats.mean, stats.std_dev, stats.min_val, stats.max_val);
        }
    }
}

static void print_tensors_ctx(struct ggml_context* tensor_ctx) {
    for (ggml_tensor* tensor = ggml_get_first_tensor(tensor_ctx); tensor; tensor = ggml_get_next_tensor(tensor_ctx, tensor)) {
        std::string name = tensor->name ? tensor->name : "unnamed";
        std::cout << "tensor name: " << name << std::endl;
    }
}

static bool read_kqv_tensors(const kqv_tensor_params& params) {
    LOG_INF("Reading KQV trace file: %s\n", params.input_file.c_str());
    LOG_INF("=====================================\n\n");

    // Load GGUF file
    struct ggml_context* ggml_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ggml_ctx,
    };

    struct gguf_context* ctx = gguf_init_from_file(params.input_file.c_str(), gguf_params);
    if (!ctx) {
        LOG_ERR("Error: Failed to load GGUF file: %s\n", params.input_file.c_str());
        return false;
    }

    // Get tensor context
    struct ggml_context* tensor_ctx = ggml_ctx;
    if (!tensor_ctx) {
        LOG_ERR("Error: Failed to get tensor context\n");
        gguf_free(ctx);
        return false;
    }
    
    // step -> vector of (tensor, name)
    std::map<int, std::vector<std::pair<ggml_tensor*, std::string>>> step_tensor_map;
    for (ggml_tensor* tensor = ggml_get_first_tensor(tensor_ctx); tensor; tensor = ggml_get_next_tensor(tensor_ctx, tensor)) {
        std::string name = tensor->name != nullptr ? tensor->name : "unnamed";
        int step = extract_step_from_name(name);
        step_tensor_map[step].emplace_back(tensor, name);
    }

    // Output by step
    int global_index = 0;
    for (const auto& [step, tensors] : step_tensor_map) {
        LOG_INF("\n==== Step %d ====%s\n", step, (step == -1 ? " (unknown)" : ""));
        int local_index = 0;
        
        if (tensors.size() < 2) {
            continue;
        }
        
        ggml_tensor * kqv_out = tensors[0].first;
        ggml_tensor * Q = tensors[1].first;
        ggml_tensor * K = tensors[2].first;
        ggml_tensor * V = tensors[3].first;
        ggml_tensor * kq_mask = tensors[4].first;
        if (tensors.size() > 5) {
            ggml_tensor * Q_quant = tensors[5].first;
            ggml_tensor * K_quant = tensors[6].first;
            ggml_tensor * V_quant = tensors[7].first;
            LOG_INF("Q: %s, K: %s, V: %s, Q_quant: %s, K_quant: %s, V_quant: %s\n", Q->name, K->name, V->name, Q_quant->name, K_quant->name, V_quant->name);
        } else {
            LOG_INF("Q: %s, K: %s, V: %s\n", Q->name, K->name, V->name);
        }
        
        

    }

    // Cleanup
    gguf_free(ctx);
    
    return true;
}

int main(int argc, char** argv) {
    kqv_tensor_params params;

    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    if (!read_kqv_tensors(params)) {
        return 1;
    }

    return 0;
} 




