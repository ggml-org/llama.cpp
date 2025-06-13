#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cassert>
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
#include <algorithm>

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
    LOG_INF("  Flash attention computation is automatically performed on all detected steps.\n");
    LOG_INF("\n");
    LOG_INF("Examples:\n");
    LOG_INF("  %s -i tensors.gguf                # Basic tensor listing with flash attention\n", program_name);
    LOG_INF("  %s -i tensors.gguf --shapes       # Show detailed shape information with flash attention\n", program_name);
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

struct tensor_stats {
    double mean = 0.0;
    double std_dev = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    size_t elements = 0;
};

// Flash attention model structure
struct flash_attn_model {
    struct ggml_tensor * Q;
    struct ggml_tensor * K;
    struct ggml_tensor * V;
    struct ggml_tensor * mask;
    struct ggml_context * ctx;
};

// Initialize flash attention model with Q, K, V tensors
static bool init_flash_attn_model(flash_attn_model & model, ggml_tensor* q_src, ggml_tensor* k_src, ggml_tensor* v_src, ggml_tensor* mask_src = nullptr) {
    // Calculate context size needed
    size_t ctx_size = 0;
    ctx_size += ggml_nbytes(q_src);
    ctx_size += ggml_nbytes(k_src);
    ctx_size += ggml_nbytes(v_src);
    if (mask_src) {
        ctx_size += ggml_nbytes(mask_src);
    }
    
    // Add space for result tensor (estimated)
    size_t result_size = q_src->ne[0] * q_src->ne[1] * q_src->ne[2] * q_src->ne[3] * ggml_type_size(GGML_TYPE_F32);
    ctx_size += result_size;
    
    ctx_size += 4 * ggml_tensor_overhead(); // tensors
    ctx_size += ggml_graph_overhead(); // compute graph
    ctx_size += 1024 * 1024; // extra overhead

    struct ggml_init_params params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    // create context
    model.ctx = ggml_init(params);
    if (!model.ctx) {
        LOG_ERR("Failed to create ggml context for flash attention\n");
        return false;
    }

    // Create new tensors with same shapes and copy data
    model.Q = ggml_new_tensor_4d(model.ctx, q_src->type, q_src->ne[0], q_src->ne[1], q_src->ne[2], q_src->ne[3]);
    model.K = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, k_src->ne[0], k_src->ne[1], k_src->ne[2], k_src->ne[3]);
    model.V = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, v_src->ne[0], v_src->ne[1], v_src->ne[2], v_src->ne[3]);
    
    if (mask_src) {
        model.mask = ggml_new_tensor_4d(model.ctx, mask_src->type, mask_src->ne[0], mask_src->ne[1], mask_src->ne[2], mask_src->ne[3]);
        memcpy(model.mask->data, mask_src->data, ggml_nbytes(mask_src));
    } else {
        model.mask = nullptr;
    }

    // Copy data
    memcpy(model.Q->data, q_src->data, ggml_nbytes(q_src));

    ggml_fp32_to_fp16_row((const float*)k_src->data, (ggml_fp16_t*)model.K->data, ggml_nelements(k_src));
    ggml_fp32_to_fp16_row((const float*)v_src->data, (ggml_fp16_t*)model.V->data, ggml_nelements(v_src));

    return true;
}

// Build computation graph for flash attention
static struct ggml_cgraph * build_flash_attn_graph(const flash_attn_model& model, float scale = 1.0f, float max_bias = 0.0f, float logit_softcap = 0.0f) {
    struct ggml_cgraph * gf = ggml_new_graph(model.ctx);

    // Perform flash attention: result = flash_attn_ext(Q, K, V, mask)
    struct ggml_tensor * result = ggml_flash_attn_ext(
        model.ctx, 
        model.Q, 
        model.K, 
        model.V, 
        model.mask,
        scale,
        max_bias,
        logit_softcap
    );
    result = ggml_reshape_2d(model.ctx, result, result->ne[0] * result->ne[1], result->ne[2]);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// Compute flash attention
static struct ggml_tensor * compute_flash_attn(const flash_attn_model & model, float scale = 1.0f) {
    struct ggml_cgraph * gf = build_flash_attn_graph(model, scale);

    int n_threads = 1; // number of threads

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // return the result tensor (last node in graph)
    return ggml_graph_node(gf, -1);
}

// Professional tensor printing function similar to ggml_print_tensor
static void ggml_print_tensor_info(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, const std::string& name, int64_t n = 3) {
    if (!data || n <= 0) {
        LOG_INF("Tensor %s: NULL or invalid data\n", name.c_str());
        return;
    }

    LOG_INF("\n=== Tensor: %s ===\n", name.c_str());
    LOG_INF("Type: %s, Shape: [%ld, %ld, %ld, %ld]\n", ggml_type_name(type), ne[0], ne[1], ne[2], ne[3]);
    
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG_INF("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG_INF("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG_INF("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG_INF("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG_INF("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG_INF("..., ");
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
                        v = 0.0f; // fallback for unsupported types
                    }
                    LOG_INF("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG_INF(", ");
                }
                LOG_INF("],\n");
            }
            LOG_INF("                                      ],\n");
        }
        LOG_INF("                                     ]\n");
    }
    LOG_INF("Sum: %.6f\n", sum);
    LOG_INF("================\n\n");
}

// Simple tensor info without detailed data
static void print_tensor_summary(ggml_tensor* tensor, const std::string& name) {
    if (!tensor) {
        LOG_INF("Tensor %s: NULL\n", name.c_str());
        return;
    }
    LOG_INF("%s: shape=[%ld,%ld,%ld,%ld], type=%s, elements=%zu\n", 
            name.c_str(), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            ggml_type_name(tensor->type), ggml_nelements(tensor));
}

static bool read_kqv_tensors(const kqv_tensor_params& params) {
    LOG_INF("Reading KQV trace file: %s\n", params.input_file.c_str());
    LOG_INF("Flash attention computation enabled for all steps\n");
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
        std::string name = tensor->name && tensor->name[0] ? tensor->name : "unnamed";
        int step = extract_step_from_name(name);
        step_tensor_map[step].emplace_back(tensor, name);
    }

    // Output by step
    for (const auto& [step, tensors] : step_tensor_map) {
        LOG_INF("\n==== Step %d ====%s\n", step, (step == -1 ? " (unknown)" : ""));
        
        if (tensors.size() < 4) {
            LOG_INF("Insufficient tensors in step %d (need at least Q, K, V, mask)\n", step);
            continue;
        }
        
        ggml_tensor * kqv_out = tensors[0].first;
        ggml_tensor * Q = tensors[1].first;
        ggml_tensor * K = tensors[2].first;
        ggml_tensor * V = tensors[3].first;
        ggml_tensor * kq_mask = tensors.size() > 4 ? tensors[4].first : nullptr;
        
        LOG_INF("Found tensors - Q: %s, K: %s, V: %s", Q->name, K->name, V->name);
        if (kq_mask) {
            LOG_INF(", Mask: %s", kq_mask->name);
        }
        LOG_INF("\n");
        
        if (tensors.size() > 5) {
            ggml_tensor * Q_quant = tensors[5].first;
            ggml_tensor * K_quant = tensors[6].first;
            ggml_tensor * V_quant = tensors[7].first;
            LOG_INF("Quantized tensors - Q_quant: %s, K_quant: %s, V_quant: %s\n", 
                    Q_quant->name, K_quant->name, V_quant->name);
        }
        
        // Run flash attention for all steps
        LOG_INF("\n🔥 Running Flash Attention at Step %d 🔥\n", step);
        
        // Print input tensor summary (without detailed data)
        print_tensor_summary(Q, "Q (Query)");
        print_tensor_summary(K, "K (Key)");
        print_tensor_summary(V, "V (Value)");
        if (kq_mask) {
            print_tensor_summary(kq_mask, "Mask");
        }
        
        // Initialize flash attention model
        flash_attn_model flash_model;
        if (!init_flash_attn_model(flash_model, Q, K, V, kq_mask)) {
            LOG_ERR("Failed to initialize flash attention model\n");
            continue;
        }
        
        // Compute flash attention
        float scale = 1.0f / sqrtf((float)Q->ne[0]); // Standard attention scaling
        LOG_INF("Computing flash attention with scale: %.6f\n", scale);
        
        struct ggml_tensor * flash_result = compute_flash_attn(flash_model, scale);
        
        if (flash_result) {
            LOG_INF("✅ Flash Attention computation successful!\n");
            ggml_print_tensor_info((uint8_t*)flash_result->data, flash_result->type, 
                                 flash_result->ne, flash_result->nb, "Flash Attention Result", 2);
            
            // Compare with original kqv_out if available
            if (kqv_out && kqv_out->data) {
                LOG_INF("📊 Comparing with original kqv_out:\n");
                ggml_print_tensor_info((uint8_t*)kqv_out->data, kqv_out->type, 
                                     kqv_out->ne, kqv_out->nb, "Original KQV_OUT", 2);
                
                // Calculate difference if same size
                if (ggml_nelements(flash_result) == ggml_nelements(kqv_out) && 
                    flash_result->type == GGML_TYPE_F32 && kqv_out->type == GGML_TYPE_F32) {
                    
                    float* flash_data = (float*)flash_result->data;
                    float* orig_data = (float*)kqv_out->data;
                    size_t n_elements = ggml_nelements(flash_result);
                    
                    double mse = 0.0;
                    double max_diff = 0.0;
                    for (size_t i = 0; i < n_elements; i++) {
                        double diff = fabs(flash_data[i] - orig_data[i]);
                        mse += diff * diff;
                        max_diff = std::max(max_diff, diff);
                    }
                    mse /= n_elements;
                    
                    LOG_INF("🔍 Difference Analysis:\n");
                    LOG_INF("  Mean Squared Error: %.10f\n", mse);
                    LOG_INF("  Max Absolute Difference: %.10f\n", max_diff);
                    LOG_INF("  RMSE: %.10f\n", sqrt(mse));
                }
            }
        } else {
            LOG_ERR("❌ Flash Attention computation failed!\n");
        }
        
        // Free flash attention model
        ggml_free(flash_model.ctx);
    }

    // Cleanup
    gguf_free(ctx);
    
    return true;
}

int main(int argc, char** argv) {
    ggml_time_init();

    kqv_tensor_params params;

    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    if (!read_kqv_tensors(params)) {
        return 1;
    }

    return 0;
} 




