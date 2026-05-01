// capture-layer-data.cpp
// Captures intermediate activation tensors during model inference
// and saves them as .f32bin files for the quantization laboratory.
//
// Usage:
//   llama-capture-layer-data -m MODEL_PATH -l LAYER [-p PROMPT] [-o OUTPUT_DIR]
//
// Example:
//   llama-capture-layer-data -m /devel/models/Qwen_Qwen3-4B-Instruct-2507-bf16.gguf -l 0 -o data

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct TensorMapping {
    const char * graph_name_prefix;
    const char * output_suffix;
};

static const TensorMapping mappings[] = {
    { "attn_norm",  "attn_input"        },
    { "kqv_out",    "attn_output_input" },
    { "ffn_norm",   "ffn_input"         },
    { "ffn_swiglu", "ffn_down_input"    },
};
static constexpr int N_MAPPINGS = sizeof(mappings) / sizeof(mappings[0]);

struct CaptureState {
    int         target_layer;
    std::string output_dir;
    int         captured_count = 0;
    std::string pending_name;

    std::string graph_to_filename(const char * graph_name) const {
        for (int i = 0; i < N_MAPPINGS; i++) {
            std::string prefix = mappings[i].graph_name_prefix;
            if (strncmp(graph_name, prefix.c_str(), prefix.size()) == 0) {
                char buf[256];
                snprintf(buf, sizeof(buf), "act_blk%d_%s.f32bin", target_layer, mappings[i].output_suffix);
                return std::string(buf);
            }
        }
        return "";
    }
};

static CaptureState * g_capture_state = nullptr;

static void save_tensor_as_f32bin(const ggml_tensor * t, const std::string & filepath) {
    int64_t n_rows  = t->ne[1];
    int64_t row_len = t->ne[0];

    int64_t total = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        total *= t->ne[i];
    }

    std::vector<float> f32_data(total);

    if (t->type == GGML_TYPE_F32) {
        const float * src = (const float *) t->data;
        if (!src) {
            LOG_ERR("Tensor %s has null data pointer\n", t->name);
            return;
        }
        memcpy(f32_data.data(), src, total * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *) t->data;
        for (int64_t i = 0; i < total; i++) {
            f32_data[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else if (t->type == GGML_TYPE_BF16) {
        const ggml_bf16_t * src = (const ggml_bf16_t *) t->data;
        for (int64_t i = 0; i < total; i++) {
            f32_data[i] = ggml_bf16_to_fp32(src[i]);
        }
    } else {
        LOG_ERR("Unsupported tensor type %s for %s\n", ggml_type_name(t->type), t->name);
        return;
    }

    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        LOG_ERR("Failed to open %s for writing\n", filepath.c_str());
        return;
    }

    file.write(reinterpret_cast<const char *>(&n_rows), sizeof(int64_t));
    file.write(reinterpret_cast<const char *>(&row_len), sizeof(int64_t));
    file.write(reinterpret_cast<const char *>(f32_data.data()), total * sizeof(float));

    file.close();
    LOG("  Captured: %s -> %s (%lld x %lld, %s)\n", t->name, filepath.c_str(), (long long) n_rows, (long long) row_len,
        ggml_type_name(t->type));
}

static bool capture_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (CaptureState *) user_data;

    if (ask) {
        char target[128];
        for (int i = 0; i < N_MAPPINGS; i++) {
            snprintf(target, sizeof(target), "%s-%d", mappings[i].graph_name_prefix, state->target_layer);
            if (strcmp(t->name, target) == 0) {
                state->pending_name = t->name;
                return true;
            }
        }
        return false;
    }

    if (state->pending_name.empty()) {
        return true;
    }
    if (strcmp(t->name, state->pending_name.c_str()) != 0) {
        return true;
    }

    if (!ggml_backend_buffer_is_host(t->buffer)) {
        size_t               nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tmp(nbytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
        LOG_WRN("Tensor %s is not host-accessible, data copied via backend\n", t->name);
    }

    std::string filename = state->graph_to_filename(t->name);
    if (!filename.empty()) {
        std::filesystem::create_directories(state->output_dir);
        std::string filepath = (std::filesystem::path(state->output_dir) / filename).string();
        save_tensor_as_f32bin(t, filepath);
        state->captured_count++;
    }

    state->pending_name.clear();
    return true;
}

static void print_usage(void) {
    LOG("Usage: llama-capture-layer-data -m MODEL_PATH [-l LAYER] [-p PROMPT] [-o OUTPUT_DIR]\n");
    LOG("\n");
    LOG("  -m MODEL      Path to GGUF model (BF16/F16 recommended)\n");
    LOG("  -l LAYER      Target layer index (default: 0)\n");
    LOG("  -p PROMPT     Inference prompt (default: \"The quick brown fox...\")\n");
    LOG("  -o DIR        Output directory for .f32bin files (default: data)\n");
}

int main(int argc, char ** argv) {
    if (argc < 3 || (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage();
        return 1;
    }

    common_params params;
    int           layer      = 0;
    std::string   output_dir = "data";
    std::string   prompt     = "The quick brown fox jumps over the lazy dog.";
    std::string   model_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "-l" && i + 1 < argc) {
            layer = atoi(argv[++i]);
        } else if (arg == "-p" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_dir = argv[++i];
        }
    }

    if (model_path.empty()) {
        LOG_ERR("Error: -m MODEL_PATH is required\n\n");
        print_usage();
        return 1;
    }

    params.model.path   = model_path;
    params.prompt       = prompt;
    params.n_batch      = 512;
    params.n_ubatch     = 512;
    params.n_gpu_layers = 0;
    params.fit_params   = false;

    CaptureState state;
    state.target_layer = layer;
    state.output_dir   = output_dir;
    g_capture_state    = &state;

    params.cb_eval           = capture_callback;
    params.cb_eval_user_data = &state;

    LOG("Loading model: %s\n", model_path.c_str());
    LOG("Target layer: %d\n", layer);
    LOG("Output directory: %s\n", output_dir.c_str());

    common_init();
    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    if (!llama_init) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("Failed to initialize context\n");
        return 1;
    }

    LOG("Model loaded successfully\n");

    const llama_vocab *      vocab   = llama_model_get_vocab(model);
    const bool               add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens  = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("No tokens generated from prompt\n");
        return 1;
    }

    LOG("Tokenizing prompt: %zu tokens\n", tokens.size());
    LOG("Running inference...\n");

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("llama_decode failed\n");
        return 1;
    }

    LOG("\nDone. Captured %d tensors to %s/\n", state.captured_count, output_dir.c_str());

    llama_backend_free();

    return state.captured_count == 0 ? 1 : 0;
}
