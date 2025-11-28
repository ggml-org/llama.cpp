#include "llama-model-from-safetensors.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-safetensors-types.h"

#include "../vendor/nlohmann/json.hpp"

#include <cinttypes>
#include <fstream>
#include <filesystem>

// Helper function to apply head permutation for HuggingFace attention weights
// This reverses the HF permutation: reshape(n_head, 2, dim/(n_head*2), *) -> swap(1,2) -> reshape
static void apply_head_permutation(
    std::vector<char> & data,
    size_t elem_size,
    size_t out_dim,
    size_t in_dim,
    int n_head
) {
    // Verify dimensions are compatible
    if (out_dim % (n_head * 2) != 0) {
        LLAMA_LOG_WARN("%s: out_dim %zu not divisible by n_head*2 (%d), skipping permutation\n",
                       __func__, out_dim, n_head * 2);
        return;
    }

    size_t head_dim = out_dim / n_head;  // Dimension per head
    size_t half_head = head_dim / 2;     // Half of head dimension

    std::vector<char> permuted(data.size());

    // Apply permutation: swap pairs within each head
    // Original layout: [n_head, 2, half_head, in_dim]
    // Permuted layout: [n_head, half_head, 2, in_dim]
    for (size_t h = 0; h < (size_t)n_head; h++) {          // For each head
        for (size_t hh = 0; hh < half_head; hh++) {        // For each half-head element
            for (size_t pair = 0; pair < 2; pair++) {      // For the pair (0 or 1)
                for (size_t i = 0; i < in_dim; i++) {      // For each input dimension
                    // Source: [h, pair, hh, i] in original [n_head, 2, half_head, in_dim]
                    size_t src_idx = ((h * 2 + pair) * half_head + hh) * in_dim + i;
                    // Dest: [h, hh, pair, i] in permuted [n_head, half_head, 2, in_dim]
                    size_t dst_idx = ((h * half_head + hh) * 2 + pair) * in_dim + i;

                    memcpy(permuted.data() + dst_idx * elem_size,
                           data.data() + src_idx * elem_size,
                           elem_size);
                }
            }
        }
    }

    data = std::move(permuted);
}

// Main entry point
llama_model * llama_model_load_from_safetensors(
    const char * model_path,
    const llama_model_params & params
) {
    if (!model_path) {
        LLAMA_LOG_ERROR("%s: model_path is null\n", __func__);
        return nullptr;
    }

    // Determine if path is directory or file
    std::string path_str(model_path);
    std::filesystem::path path(path_str);

    std::string model_dir;
    if (std::filesystem::is_directory(path)) {
        model_dir = path_str;
    } else if (std::filesystem::is_regular_file(path)) {
        model_dir = path.parent_path().string();
    } else {
        LLAMA_LOG_ERROR("%s: invalid path: %s\n", __func__, model_path);
        return nullptr;
    }

    // Create builder and build model
    safetensors_model_builder builder(model_dir, params);
    llama_model * model = builder.build();

    if (!model) {
        LLAMA_LOG_ERROR("%s: failed to load model: %s\n", __func__, builder.get_error().c_str());
    }

    return model;
}

// Implementation
safetensors_model_builder::safetensors_model_builder(
    const std::string & model_dir,
    const llama_model_params & params
) : model_dir(model_dir), params(params) {
}

safetensors_model_builder::~safetensors_model_builder() {
    // Clean up backend buffers from map
    for (auto & pair : buffer_map) {
        if (pair.second) {
            ggml_backend_buffer_free(pair.second);
        }
    }
    buffer_map.clear();

    // Clean up legacy backend buffer if allocated
    if (backend_buffer) {
        ggml_backend_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }

    // Clean up GGML contexts from map
    for (auto & pair : ctx_map) {
        if (pair.second) {
            ggml_free(pair.second);
        }
    }
    ctx_map.clear();

    // Clean up legacy GGML contexts
    if (ctx_meta) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }

    if (ctx_data) {
        ggml_free(ctx_data);
        ctx_data = nullptr;
    }
}

llama_model * safetensors_model_builder::build() {
    LLAMA_LOG_INFO("%s: loading model from safetensors: %s\n", __func__, model_dir.c_str());

    // Step 1: Load config.json
    if (!load_config()) {
        return nullptr;
    }

    // Step 2: Load safetensors files
    if (!load_safetensors_files()) {
        return nullptr;
    }

    // Step 3: Detect architecture
    if (!detect_architecture()) {
        return nullptr;
    }

    // Step 4: Create model structure
    if (!create_model_structure()) {
        return nullptr;
    }

    // Step 4.5: Initialize backend devices
    if (!init_devices()) {
        return nullptr;
    }

    // Step 5: Allocate tensors
    if (!allocate_tensors()) {
        return nullptr;
    }

    // Step 6: Load tensor data
    if (!load_tensor_data()) {
        return nullptr;
    }

    // Step 7: Link tensors to model structure
    if (!link_tensors_to_model()) {
        return nullptr;
    }

    // Step 8: Register buffers with model (transfer ownership)
    if (!register_buffers_with_model()) {
        return nullptr;
    }

    // Step 9: Initialize vocabulary
    if (!init_vocabulary()) {
        return nullptr;
    }

    // Step 10: Finalize
    if (!finalize_model()) {
        return nullptr;
    }

    LLAMA_LOG_INFO("%s: model loaded successfully\n", __func__);
    return model;
}

bool safetensors_model_builder::load_config() {
    std::string config_path = model_dir + "/config.json";

    config = std::make_unique<hf_config>();
    if (!config->load_from_file(config_path)) {
        error_msg = "Failed to load config.json: " + config->get_error();
        return false;
    }

    LLAMA_LOG_INFO("%s: loaded config.json\n", __func__);
    return true;
}

bool safetensors_model_builder::load_safetensors_files() {
    st_loader = std::make_unique<safetensors_loader>();

    // Try single file first
    std::string single_file = model_dir + "/model.safetensors";
    if (std::filesystem::exists(single_file)) {
        if (st_loader->load_single(single_file)) {
            LLAMA_LOG_INFO("%s: loaded single safetensors file\n", __func__);
            return true;
        }
    }

    // Try sharded model
    std::string index_file = model_dir + "/model.safetensors.index.json";
    if (std::filesystem::exists(index_file)) {
        if (st_loader->load_sharded(index_file, model_dir)) {
            LLAMA_LOG_INFO("%s: loaded sharded safetensors files\n", __func__);
            return true;
        }
    }

    error_msg = "No safetensors files found in: " + model_dir;
    return false;
}

bool safetensors_model_builder::detect_architecture() {
    std::string hf_arch = config->get_architecture();
    if (hf_arch.empty()) {
        error_msg = "Could not detect architecture from config.json";
        return false;
    }

    mapper = create_tensor_mapper(hf_arch);
    if (!mapper) {
        error_msg = "Unsupported architecture: " + hf_arch;
        return false;
    }

    LLAMA_LOG_INFO("%s: detected architecture: %s\n", __func__, hf_arch.c_str());
    return true;
}

bool safetensors_model_builder::create_model_structure() {
    // Step 1: Allocate llama_model
    model = new llama_model(params);
    if (!model) {
        error_msg = "Failed to allocate llama_model";
        return false;
    }

    // Step 2: Set architecture
    model->arch = mapper->get_arch();
    if (model->arch == LLM_ARCH_UNKNOWN) {
        error_msg = "Unknown architecture";
        delete model;
        model = nullptr;
        return false;
    }

    // Step 3: Initialize hparams from HF config
    // Get basic hyperparameters
    model->hparams.n_embd = config->get_hidden_size();
    model->hparams.n_layer = config->get_num_hidden_layers();

    // Get context length
    int64_t max_pos = config->get_max_position_embeddings();
    model->hparams.n_ctx_train = max_pos > 0 ? max_pos : 2048;

    // Get attention parameters
    uint32_t n_head = config->get_num_attention_heads();
    int64_t n_head_kv_val = config->get_num_key_value_heads();
    uint32_t n_head_kv = (n_head_kv_val > 0) ? n_head_kv_val : n_head;  // Default to n_head for MHA

    // Fill per-layer arrays with same values (uniform layers)
    std::fill(model->hparams.n_head_arr.begin(), model->hparams.n_head_arr.end(), n_head);
    std::fill(model->hparams.n_head_kv_arr.begin(), model->hparams.n_head_kv_arr.end(), n_head_kv);

    // Get feed-forward dimension
    int64_t n_ff_val = config->get_intermediate_size();
    if (n_ff_val > 0) {
        std::fill(model->hparams.n_ff_arr.begin(), model->hparams.n_ff_arr.end(), static_cast<uint32_t>(n_ff_val));
    }

    // Calculate head dimensions
    if (n_head > 0) {
        model->hparams.n_embd_head_k = model->hparams.n_embd / n_head;
        model->hparams.n_embd_head_v = model->hparams.n_embd / n_head;
        model->hparams.n_rot = model->hparams.n_embd_head_k;  // Full rotary
    }

    // Get normalization epsilon
    double norm_eps = config->get_rms_norm_eps();
    if (norm_eps > 0.0) {
        model->hparams.f_norm_rms_eps = static_cast<float>(norm_eps);
    } else {
        // Try layer_norm_eps as fallback
        double layer_norm_eps;
        if (config->get_float("layer_norm_eps", layer_norm_eps)) {
            model->hparams.f_norm_rms_eps = static_cast<float>(layer_norm_eps);
        } else {
            model->hparams.f_norm_rms_eps = 1e-5f;  // Default
        }
    }
    model->hparams.f_norm_eps = model->hparams.f_norm_rms_eps;

    // Get RoPE parameters
    double rope_theta;
    if (config->get_float("rope_theta", rope_theta)) {
        model->hparams.rope_freq_base_train = static_cast<float>(rope_theta);
    } else {
        model->hparams.rope_freq_base_train = 10000.0f;  // Default
    }

    // Check for RoPE scaling
    if (config->has_key("rope_scaling")) {
        // TODO: Parse rope_scaling dict if present
        model->hparams.rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    } else {
        model->hparams.rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;
    }

    // Default rope parameters
    model->hparams.rope_freq_scale_train = 1.0f;
    model->hparams.n_ctx_orig_yarn = model->hparams.n_ctx_train;

    // Set rope type based on architecture
    model->hparams.rope_type = llama_model_rope_type(model);

    // Initialize SWA (Sliding Window Attention) layers array - default to no SWA
    std::fill(model->hparams.swa_layers.begin(), model->hparams.swa_layers.end(), false);

    // Initialize recurrent layer array - default to no recurrent layers
    std::fill(model->hparams.recurrent_layer_arr.begin(), model->hparams.recurrent_layer_arr.end(), false);

    // Step 4: Determine model type based on architecture and size
    model->type = LLM_TYPE_UNKNOWN;

    switch (model->arch) {
        case LLM_ARCH_LLAMA:
            // SmolLM2-135M has 30 layers, which maps to 256M type
            switch (model->hparams.n_layer) {
                case 30: model->type = LLM_TYPE_256M; break;  // SmolLM2-135M
                case 16: model->type = LLM_TYPE_1B; break;
                case 22: model->type = LLM_TYPE_1B; break;
                case 26: model->type = LLM_TYPE_3B; break;
                case 28: model->type = LLM_TYPE_3B; break;
                case 32: model->type = LLM_TYPE_7B; break;
                case 40: model->type = LLM_TYPE_13B; break;
                case 48: model->type = LLM_TYPE_34B; break;
                case 60: model->type = LLM_TYPE_30B; break;
                case 80: model->type = LLM_TYPE_70B; break;
                default: model->type = LLM_TYPE_UNKNOWN;
            }
            break;

        case LLM_ARCH_PHI3:
            switch (model->hparams.n_layer) {
                case 24: model->type = LLM_TYPE_1_3B; break;
                case 32: model->type = LLM_TYPE_3B; break;
                case 40: model->type = LLM_TYPE_14B; break;
                default: model->type = LLM_TYPE_UNKNOWN;
            }
            break;

        case LLM_ARCH_QWEN2:
            switch (model->hparams.n_layer) {
                case 24: model->type = LLM_TYPE_0_5B; break;
                case 28: model->type = LLM_TYPE_1_5B; break;
                case 32: model->type = LLM_TYPE_7B; break;
                case 40: model->type = LLM_TYPE_13B; break;
                case 80: model->type = LLM_TYPE_70B; break;
                default: model->type = LLM_TYPE_UNKNOWN;
            }
            break;

        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
            switch (model->hparams.n_layer) {
                case 18: model->type = LLM_TYPE_2B; break;
                case 26: model->type = LLM_TYPE_7B; break;
                case 42: model->type = LLM_TYPE_9B; break;
                case 46: model->type = LLM_TYPE_27B; break;
                default: model->type = LLM_TYPE_UNKNOWN;
            }
            break;

        default:
            model->type = LLM_TYPE_UNKNOWN;
    }

    // Step 5: Allocate layers vector
    model->layers.resize(model->hparams.n_layer);

    // Set model name from config
    std::string model_name;
    if (config->get_string("_name_or_path", model_name)) {
        model->name = model_name;
    } else {
        model->name = "unknown";
    }

    LLAMA_LOG_INFO("%s: created model structure: arch=%s, layers=%d, type=%s\n",
                   __func__,
                   llm_arch_name(model->arch),
                   model->hparams.n_layer,
                   model->type_name().c_str());

    return true;
}

bool safetensors_model_builder::init_devices() {
    LLAMA_LOG_INFO("%s: initializing backend devices\n", __func__);

    const int n_gpu_layers = params.n_gpu_layers;

    // Initialize GPU backends if requested
    if (n_gpu_layers > 0) {
        LLAMA_LOG_INFO("%s: GPU offloading enabled with %d layers\n", __func__, n_gpu_layers);

        // Get available GPU backends
        size_t n_devices = ggml_backend_dev_count();
        for (size_t i = 0; i < n_devices; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);

            // Add GPU/Metal backends to model->devices
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU) {
                model->devices.push_back(dev);
                LLAMA_LOG_INFO("%s: added GPU device: %s\n", __func__, ggml_backend_dev_name(dev));
            }
        }

        if (model->devices.empty()) {
            LLAMA_LOG_WARN("%s: no GPU backends found, falling back to CPU\n", __func__);
        }
    } else {
        LLAMA_LOG_INFO("%s: GPU offloading disabled (n_gpu_layers=0)\n", __func__);
    }

    // Initialize buffer type lists and layer device mappings
    try {
        model->init_layer_devices();
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to initialize layer devices: ") + e.what();
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error_msg.c_str());
        return false;
    }

    return true;
}

// Helper function to parse layer number from tensor name
// Returns -1 for non-layer tensors (embeddings, output, etc.)
static int parse_layer_number(const std::string & name) {
    // Look for pattern like "blk.5." or "layers.5."
    size_t pos = name.find("blk.");
    if (pos == std::string::npos) {
        pos = name.find("layers.");
    }

    if (pos != std::string::npos) {
        size_t start = pos + (name[pos] == 'b' ? 4 : 7); // Skip "blk." or "layers."
        size_t end = name.find('.', start);
        if (end != std::string::npos) {
            std::string layer_str = name.substr(start, end - start);
            try {
                return std::stoi(layer_str);
            } catch (...) {
                return -1;
            }
        }
    }
    return -1;
}

bool safetensors_model_builder::allocate_tensors() {
    // Step 1: Get list of all tensors from safetensors
    std::vector<std::string> tensor_names = st_loader->get_tensor_names();

    if (tensor_names.empty()) {
        error_msg = "No tensors found in safetensors files";
        return false;
    }

    LLAMA_LOG_INFO("%s: found %zu tensors in safetensors\n", __func__, tensor_names.size());

    // Step 2: Create GGML contexts for tensor metadata (one per buffer type)
    // This follows the pattern from GGUF loader in llama-model.cpp

    size_t ctx_size = tensor_names.size() * ggml_tensor_overhead();

    // Helper lambda to get or create context for a given buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            struct ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,  // Don't allocate data yet, just metadata
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                error_msg = "Failed to initialize GGML context for buffer type";
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);
            LLAMA_LOG_DEBUG("%s: created GGML context for buffer type: %s\n",
                           __func__, ggml_backend_buft_name(buft));
            return ctx;
        }
        return it->second;
    };

    // Helper lambda to determine buffer type for a tensor based on its name
    auto get_tensor_buft = [&](const std::string & name) -> ggml_backend_buffer_type_t {
        // Parse layer number from tensor name
        int layer_idx = parse_layer_number(name);

        // Input layer tensors (token_embd)
        if (name.find("token_embd") != std::string::npos) {
            return model->get_layer_buft(-1); // -1 = input layer
        }

        // Output norm and output tensors
        if (name.find("output_norm") != std::string::npos || name == "output.weight") {
            return model->get_layer_buft(-2); // -2 = output layer
        }

        // Layer tensors - use layer assignment
        if (layer_idx >= 0) {
            return model->get_layer_buft(layer_idx);
        }

        // Default to CPU for other tensors
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            return ggml_backend_dev_buffer_type(cpu_dev);
        }
        return nullptr;
    };

    // Step 3: Create tensor metadata for each safetensors tensor in appropriate context
    int tensors_created = 0;
    std::map<ggml_backend_buffer_type_t, int> tensor_counts;

    for (const std::string & hf_name : tensor_names) {
        // Get tensor info from safetensors
        const safetensors_tensor_info * info = st_loader->get_tensor_info(hf_name);
        if (!info) {
            LLAMA_LOG_WARN("%s: could not find tensor info for %s, skipping\n", __func__, hf_name.c_str());
            continue;
        }

        // Map HuggingFace tensor name to llama.cpp internal name
        std::string internal_name = mapper->map_tensor_name(hf_name);
        if (internal_name.empty()) {
            LLAMA_LOG_DEBUG("%s: no mapping for tensor %s, skipping\n", __func__, hf_name.c_str());
            continue;
        }

        // Convert safetensors dtype to GGML type
        ggml_type ggml_type = safetensors_dtype_to_ggml_type(info->dtype);
        if (ggml_type == GGML_TYPE_COUNT) {
            LLAMA_LOG_WARN("%s: unsupported dtype for tensor %s, skipping\n", __func__, hf_name.c_str());
            continue;
        }

        // Determine which buffer type (CPU or GPU) this tensor should use
        ggml_backend_buffer_type_t buft = get_tensor_buft(internal_name);
        if (!buft) {
            error_msg = "Failed to determine buffer type for tensor: " + internal_name;
            return false;
        }

        // Get or create context for this buffer type
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            return false;
        }

        // Create tensor in the appropriate context
        struct ggml_tensor * tensor = nullptr;

        switch (info->shape.size()) {
            case 1:
                tensor = ggml_new_tensor_1d(ctx, ggml_type, info->shape[0]);
                break;
            case 2:
                // GGML expects dimensions REVERSED from PyTorch/HuggingFace
                tensor = ggml_new_tensor_2d(ctx, ggml_type, info->shape[1], info->shape[0]);
                break;
            case 3:
                tensor = ggml_new_tensor_3d(ctx, ggml_type, info->shape[2], info->shape[1], info->shape[0]);
                break;
            case 4:
                tensor = ggml_new_tensor_4d(ctx, ggml_type, info->shape[3], info->shape[2], info->shape[1], info->shape[0]);
                break;
            default:
                LLAMA_LOG_WARN("%s: tensor %s has unsupported number of dimensions: %zu\n",
                              __func__, hf_name.c_str(), info->shape.size());
                continue;
        }

        if (!tensor) {
            error_msg = "Failed to create tensor: " + internal_name;
            return false;
        }

        // Set tensor name
        ggml_set_name(tensor, internal_name.c_str());

        tensors_created++;
        tensor_counts[buft]++;

        // Debug log for key tensors
        if (tensors_created <= 10 || tensors_created % 100 == 0) {
            int layer_idx = parse_layer_number(internal_name);
            LLAMA_LOG_DEBUG("%s: Created %s (layer %d): ne[0]=%" PRId64 ", ne[1]=%" PRId64 " in context for %s\n",
                          __func__, internal_name.c_str(), layer_idx,
                          tensor->ne[0], tensor->ne[1],
                          ggml_backend_buft_name(buft));
        }

        if (tensors_created % 100 == 0) {
            LLAMA_LOG_INFO("%s: created %d tensor metadata entries...\n", __func__, tensors_created);
        }
    }

    LLAMA_LOG_INFO("%s: created %d tensor metadata entries total\n", __func__, tensors_created);

    // Step 4: Allocate backend buffers for each context
    LLAMA_LOG_INFO("%s: allocating backend buffers for %zu buffer types\n", __func__, ctx_map.size());

    for (auto & pair : ctx_map) {
        ggml_backend_buffer_type_t buft = pair.first;
        ggml_context * ctx = pair.second;

        int count = tensor_counts[buft];
        LLAMA_LOG_INFO("%s: allocating buffer for %s (%d tensors)\n",
                      __func__, ggml_backend_buft_name(buft), count);

        // Allocate backend buffer for all tensors in this context
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buffer) {
            error_msg = std::string("Failed to allocate backend buffer for ") + ggml_backend_buft_name(buft);
            return false;
        }

        size_t buffer_size = ggml_backend_buffer_get_size(buffer);
        LLAMA_LOG_INFO("%s:   allocated %zu bytes for %s\n",
                      __func__, buffer_size, ggml_backend_buft_name(buft));

        // Mark buffer as containing weights for scheduler optimization
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Store buffer for use during loading (ownership will be transferred later)
        buffer_map.emplace(buft, buffer);
    }

    LLAMA_LOG_INFO("%s: tensor allocation complete - %d tensors ready\n", __func__, tensors_created);
    return true;
}

bool safetensors_model_builder::load_tensor_data() {
    if (ctx_map.empty()) {
        error_msg = "Cannot load tensor data: no contexts initialized";
        return false;
    }

    if (buffer_map.empty()) {
        error_msg = "Cannot load tensor data: no backend buffers allocated";
        return false;
    }

    LLAMA_LOG_INFO("%s: loading tensor data from safetensors\n", __func__);

    int tensors_loaded = 0;
    int tensors_skipped = 0;
    int tensors_failed = 0;

    // Get all safetensors tensor names
    std::vector<std::string> st_tensor_names = st_loader->get_tensor_names();

    for (const std::string & hf_name : st_tensor_names) {
        // Map HF name to internal name
        std::string internal_name = mapper->map_tensor_name(hf_name);
        if (internal_name.empty()) {
            // This tensor doesn't map to anything (might be optional)
            LLAMA_LOG_DEBUG("%s: no mapping for HF tensor %s, skipping\n", __func__, hf_name.c_str());
            tensors_skipped++;
            continue;
        }

        // Find the tensor across all GGML contexts
        struct ggml_tensor * tensor = nullptr;
        for (auto & pair : ctx_map) {
            tensor = ggml_get_tensor(pair.second, internal_name.c_str());
            if (tensor) {
                break;
            }
        }

        if (!tensor) {
            LLAMA_LOG_WARN("%s: tensor %s (HF: %s) not found in any GGML context\n",
                          __func__, internal_name.c_str(), hf_name.c_str());
            tensors_skipped++;
            continue;
        }

        // Verify tensor has allocated data
        if (!tensor->data) {
            LLAMA_LOG_ERROR("%s: tensor %s has no data buffer allocated\n", __func__, internal_name.c_str());
            tensors_failed++;
            continue;
        }

        // Get tensor info from safetensors
        const safetensors_tensor_info * info = st_loader->get_tensor_info(hf_name);
        if (!info) {
            LLAMA_LOG_ERROR("%s: could not get info for tensor %s\n", __func__, hf_name.c_str());
            tensors_failed++;
            continue;
        }

        // Don't transpose - match Python converter behavior which only reverses dimensions
        bool needs_transpose = false;

        // Read data from safetensors into temporary buffer
        size_t st_data_size = info->size();
        std::vector<char> temp_buffer(st_data_size);

        if (!st_loader->read_tensor_data(hf_name, temp_buffer.data(), st_data_size)) {
            LLAMA_LOG_ERROR("%s: failed to read tensor data for %s\n", __func__, hf_name.c_str());
            tensors_failed++;
            continue;
        }

        // Convert types and copy to GGML tensor
        size_t ggml_data_size = ggml_nbytes(tensor);
        ggml_type tensor_type = tensor->type;

        // If transposition is needed, we need to create a temporary buffer with transposed data
        std::vector<char> transposed_buffer;
        std::vector<int64_t> transposed_shape;
        const char * source_data = temp_buffer.data();
        size_t source_size = st_data_size;
        const int64_t * shape_ptr = reinterpret_cast<const int64_t *>(info->shape.data());
        size_t shape_size = info->shape.size();

        // For 2D weight tensors: we need to physically transpose the data
        // because we create the tensor with swapped dimensions [dim1, dim0]
        // but the safetensors data is in [dim0, dim1] layout
        if (needs_transpose && info->shape.size() == 2) {
            size_t dim0 = info->shape[0];
            size_t dim1 = info->shape[1];
            size_t elem_size = st_data_size / (dim0 * dim1);

            // Physically transpose the data: [dim0, dim1] -> [dim1, dim0]
            transposed_buffer.resize(st_data_size);
            const char * src = temp_buffer.data();
            char * dst = transposed_buffer.data();

            for (size_t row = 0; row < dim0; row++) {
                for (size_t col = 0; col < dim1; col++) {
                    size_t src_idx = (row * dim1 + col) * elem_size;
                    size_t dst_idx = (col * dim0 + row) * elem_size;
                    memcpy(dst + dst_idx, src + src_idx, elem_size);
                }
            }

            source_data = transposed_buffer.data();

            LLAMA_LOG_DEBUG("%s: Transposed %s from [%zu, %zu] to [%zu, %zu]\n",
                           __func__, internal_name.c_str(), dim0, dim1, dim1, dim0);
        }

        // Apply head permutation for attention query/key weights
        // This reverses HuggingFace's permutation that was applied during training
        // Apply head permutation for attention query/key weights
        // This reverses HuggingFace's permutation that was applied during training
        if ((internal_name.find("attn_q.weight") != std::string::npos ||
             internal_name.find("attn_k.weight") != std::string::npos) &&
            info->shape.size() == 2 && !needs_transpose) {

            // Get n_head from model hparams
            int n_head = model->hparams.n_head();
            if (internal_name.find("attn_k.weight") != std::string::npos) {
                // For key weights, use n_head_kv if available
                n_head = model->hparams.n_head_kv();
            }

            if (n_head > 0) {
                size_t out_dim = info->shape[0];  // Output dimension
                size_t in_dim = info->shape[1];   // Input dimension
                size_t elem_size = st_data_size / (out_dim * in_dim);

                LLAMA_LOG_DEBUG("%s: Applying head permutation to %s (n_head=%d)\n",
                               __func__, internal_name.c_str(), n_head);

                // Apply permutation in-place on temp_buffer
                apply_head_permutation(temp_buffer, elem_size, out_dim, in_dim, n_head);

                // Update source_data pointer to use the permuted data
                source_data = temp_buffer.data();
            }
        }

        if (!convert_safetensors_to_ggml(
                source_data, source_size, info->dtype,
                tensor->data, ggml_data_size, tensor_type,
                shape_ptr, shape_size)) {
            LLAMA_LOG_ERROR("%s: failed to convert tensor data for %s\n", __func__, hf_name.c_str());
            tensors_failed++;
            continue;
        }

        // DEBUG: Log first few values of key tensors
        if (internal_name == "token_embd.weight" || internal_name == "blk.0.attn_q.weight") {
            const float * data_f32 = (const float *)tensor->data;
            LLAMA_LOG_INFO("%s: [DEBUG] %s first 8 F32 values: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                          __func__, internal_name.c_str(),
                          data_f32[0], data_f32[1], data_f32[2], data_f32[3],
                          data_f32[4], data_f32[5], data_f32[6], data_f32[7]);
        }

        tensors_loaded++;

        if (tensors_loaded % 50 == 0) {
            LLAMA_LOG_INFO("%s: loaded %d tensors...\n", __func__, tensors_loaded);
        }
    }

    LLAMA_LOG_INFO("%s: loaded %d tensors, skipped %d, failed %d\n",
                   __func__, tensors_loaded, tensors_skipped, tensors_failed);

    if (tensors_failed > 0) {
        error_msg = "Some tensors failed to load";
        return false;
    }

    if (tensors_loaded == 0) {
        error_msg = "No tensors were loaded";
        return false;
    }

    return true;
}

bool safetensors_model_builder::link_tensors_to_model() {
    if (!model) {
        error_msg = "Cannot link tensors: model not created";
        return false;
    }

    if (ctx_map.empty()) {
        error_msg = "Cannot link tensors: no contexts initialized";
        return false;
    }

    LLAMA_LOG_INFO("%s: linking tensors to model structure\n", __func__);

    // Helper lambda to get tensor (returns nullptr if not found, which is ok for optional tensors)
    // Search across all GGML contexts (CPU, GPU, etc.)
    auto get_tensor = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * tensor = nullptr;

        // Search all contexts for this tensor
        for (auto & pair : ctx_map) {
            tensor = ggml_get_tensor(pair.second, name);
            if (tensor) {
                break;
            }
        }

        if (tensor) {
            // Add to tensors_by_name for n_tensors() to work correctly
            model->tensors_by_name.emplace_back(name, tensor);
        }
        return tensor;
    };

    int tensors_linked = 0;

    // Link input embedding
    model->tok_embd = get_tensor("token_embd.weight");
    if (model->tok_embd) {
        tensors_linked++;
        LLAMA_LOG_INFO("%s: linked token_embd: ne[0]=%" PRId64 ", ne[1]=%" PRId64 ", ne[2]=%" PRId64 ", ne[3]=%" PRId64 "\n",
                      __func__, model->tok_embd->ne[0], model->tok_embd->ne[1], model->tok_embd->ne[2], model->tok_embd->ne[3]);
    } else {
        LLAMA_LOG_WARN("%s: token_embd.weight not found\n", __func__);
    }

    // Link output norm and output
    model->output_norm = get_tensor("output_norm.weight");
    if (model->output_norm) {
        tensors_linked++;
    }

    model->output = get_tensor("output.weight");
    if (model->output) {
        tensors_linked++;
    } else {
        // output might share with tok_embd
        model->output = model->tok_embd;
        LLAMA_LOG_DEBUG("%s: output shares with token_embd\n", __func__);
    }

    // Link layer tensors based on architecture
    switch (model->arch) {
        case LLM_ARCH_LLAMA:
            {
                LLAMA_LOG_INFO("%s: linking Llama layer tensors\n", __func__);

                for (size_t i = 0; i < model->layers.size(); ++i) {
                    auto & layer = model->layers[i];
                    char buf[256];

                    // Attention norm
                    snprintf(buf, sizeof(buf), "blk.%zu.attn_norm.weight", i);
                    layer.attn_norm = get_tensor(buf);
                    if (layer.attn_norm) tensors_linked++;

                    // Attention Q, K, V, O
                    snprintf(buf, sizeof(buf), "blk.%zu.attn_q.weight", i);
                    layer.wq = get_tensor(buf);
                    if (layer.wq) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_k.weight", i);
                    layer.wk = get_tensor(buf);
                    if (layer.wk) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_v.weight", i);
                    layer.wv = get_tensor(buf);
                    if (layer.wv) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_output.weight", i);
                    layer.wo = get_tensor(buf);
                    if (layer.wo) tensors_linked++;

                    // FFN norm
                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_norm.weight", i);
                    layer.ffn_norm = get_tensor(buf);
                    if (layer.ffn_norm) tensors_linked++;

                    // FFN gate, down, up
                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_gate.weight", i);
                    layer.ffn_gate = get_tensor(buf);
                    if (layer.ffn_gate) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_down.weight", i);
                    layer.ffn_down = get_tensor(buf);
                    if (layer.ffn_down) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_up.weight", i);
                    layer.ffn_up = get_tensor(buf);
                    if (layer.ffn_up) tensors_linked++;

                    if (i % 10 == 0 && i > 0) {
                        LLAMA_LOG_INFO("%s: linked layer %zu/%zu\n", __func__, i, model->layers.size());
                    }
                }

                LLAMA_LOG_INFO("%s: linked all %zu layers\n", __func__, model->layers.size());
            }
            break;

        case LLM_ARCH_PHI3:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
            {
                // These architectures have similar structure to Llama
                // For now, use the same linking pattern
                LLAMA_LOG_WARN("%s: using Llama-style linking for %s - may need adjustments\n",
                              __func__, llm_arch_name(model->arch));

                for (size_t i = 0; i < model->layers.size(); ++i) {
                    auto & layer = model->layers[i];
                    char buf[256];

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_norm.weight", i);
                    layer.attn_norm = get_tensor(buf);
                    if (layer.attn_norm) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_q.weight", i);
                    layer.wq = get_tensor(buf);
                    if (layer.wq) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_k.weight", i);
                    layer.wk = get_tensor(buf);
                    if (layer.wk) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_v.weight", i);
                    layer.wv = get_tensor(buf);
                    if (layer.wv) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_output.weight", i);
                    layer.wo = get_tensor(buf);
                    if (layer.wo) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_norm.weight", i);
                    layer.ffn_norm = get_tensor(buf);
                    if (layer.ffn_norm) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_gate.weight", i);
                    layer.ffn_gate = get_tensor(buf);
                    if (layer.ffn_gate) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_down.weight", i);
                    layer.ffn_down = get_tensor(buf);
                    if (layer.ffn_down) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_up.weight", i);
                    layer.ffn_up = get_tensor(buf);
                    if (layer.ffn_up) tensors_linked++;
                }
            }
            break;

        default:
            error_msg = "Tensor linking not implemented for this architecture";
            return false;
    }

    LLAMA_LOG_INFO("%s: linked %d tensors to model structure\n", __func__, tensors_linked);

    if (tensors_linked == 0) {
        error_msg = "No tensors were linked to model - tensor names may not match";
        return false;
    }

    return true;
}

bool safetensors_model_builder::register_buffers_with_model() {
    LLAMA_LOG_INFO("%s: registering buffers with model\n", __func__);

    if (ctx_map.empty()) {
        error_msg = "Cannot register buffers: no contexts allocated";
        return false;
    }

    if (buffer_map.empty()) {
        error_msg = "Cannot register buffers: no backend buffers allocated";
        return false;
    }

    // Transfer ownership of contexts and buffers to model
    // This follows the pattern from GGUF loader in llama-model.cpp line 6688-6703
    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;

        // Get the buffer for this buffer type
        auto buf_it = buffer_map.find(buft);
        if (buf_it == buffer_map.end()) {
            error_msg = std::string("No buffer found for buffer type: ") + ggml_backend_buft_name(buft);
            return false;
        }

        ggml_backend_buffer_t buf = buf_it->second;

        // Wrap buffer in unique_ptr and move to vector
        std::vector<ggml_backend_buffer_ptr> bufs;
        bufs.emplace_back(buf);

        // Add context and buffers to model (transfers ownership)
        model->add_context_with_buffers(ctx, std::move(bufs));

        LLAMA_LOG_DEBUG("%s: registered context and buffer for %s\n",
                       __func__, ggml_backend_buft_name(buft));
    }

    // Clear maps to prevent double-free in destructor
    // The model now owns these resources
    ctx_map.clear();
    buffer_map.clear();

    LLAMA_LOG_INFO("%s: successfully registered all buffers with model\n", __func__);
    return true;
}

bool safetensors_model_builder::init_vocabulary() {
    LLAMA_LOG_INFO("%s: initializing vocabulary\n", __func__);

    // Check if tokenizer.json exists
    std::string tokenizer_path = model_dir + "/tokenizer.json";
    std::string tokenizer_config_path = model_dir + "/tokenizer_config.json";

    bool has_tokenizer = std::filesystem::exists(tokenizer_path);
    bool has_config = std::filesystem::exists(tokenizer_config_path);

    if (!has_tokenizer) {
        LLAMA_LOG_ERROR("%s: tokenizer.json not found in %s\n", __func__, model_dir.c_str());
        error_msg = "tokenizer.json not found - cannot load vocabulary";
        return false;
    }

    LLAMA_LOG_INFO("%s: found tokenizer.json\n", __func__);
    if (has_config) {
        LLAMA_LOG_INFO("%s: found tokenizer_config.json\n", __func__);
    }

    // Load vocabulary from HuggingFace tokenizer format
    bool success = model->vocab.load_from_hf_tokenizer(
        tokenizer_path,
        has_config ? tokenizer_config_path : ""
    );

    if (!success) {
        error_msg = "Failed to load vocabulary from tokenizer.json";
        LLAMA_LOG_ERROR("%s: failed to load vocabulary\n", __func__);
        return false;
    }

    LLAMA_LOG_INFO("%s: vocabulary loaded successfully - %u tokens\n",
                   __func__, model->vocab.n_tokens());

    return true;
}

bool safetensors_model_builder::finalize_model() {
    if (!model) {
        error_msg = "Cannot finalize: model not created";
        return false;
    }

    LLAMA_LOG_INFO("%s: finalizing model\n", __func__);

    // Validate that critical tensors are linked
    bool has_tok_embd = (model->tok_embd != nullptr);
    bool has_output = (model->output != nullptr);
    bool has_output_norm = (model->output_norm != nullptr);

    if (!has_tok_embd) {
        LLAMA_LOG_WARN("%s: token embedding tensor not linked\n", __func__);
    }

    if (!has_output) {
        LLAMA_LOG_WARN("%s: output tensor not linked\n", __func__);
    }

    if (!has_output_norm) {
        LLAMA_LOG_WARN("%s: output norm tensor not linked\n", __func__);
    }

    // Validate layers have critical tensors
    int layers_valid = 0;
    for (size_t i = 0; i < model->layers.size(); ++i) {
        const auto & layer = model->layers[i];
        bool layer_ok = (layer.attn_norm && layer.wq && layer.wk && layer.wv && layer.wo &&
                        layer.ffn_norm && layer.ffn_gate && layer.ffn_down && layer.ffn_up);
        if (layer_ok) {
            layers_valid++;
        } else {
            LLAMA_LOG_WARN("%s: layer %zu missing some tensors\n", __func__, i);
        }
    }

    LLAMA_LOG_INFO("%s: validated %d/%zu layers\n", __func__, layers_valid, model->layers.size());

    // Log final model info
    LLAMA_LOG_INFO("%s: model finalized:\n", __func__);
    LLAMA_LOG_INFO("%s:   architecture: %s\n", __func__, llm_arch_name(model->arch));
    LLAMA_LOG_INFO("%s:   type: %s\n", __func__, model->type_name().c_str());
    LLAMA_LOG_INFO("%s:   layers: %zu\n", __func__, model->layers.size());
    LLAMA_LOG_INFO("%s:   embedding dim: %d\n", __func__, model->hparams.n_embd);
    LLAMA_LOG_INFO("%s:   attention heads: %d\n", __func__, model->hparams.n_head());
    LLAMA_LOG_INFO("%s:   context length: %d\n", __func__, model->hparams.n_ctx_train);

    // Set model stats (number of elements and bytes)
    // These are used for various calculations in the backend
    uint64_t n_elements = 0;
    size_t n_bytes = 0;
    std::vector<std::string> tensor_names = st_loader->get_tensor_names();
    for (const auto & name : tensor_names) {
        const safetensors_tensor_info * info = st_loader->get_tensor_info(name);
        if (info) {
            n_elements += info->n_elements();
            n_bytes += info->size();
        }
    }
    model->set_stats(n_elements, n_bytes);
    LLAMA_LOG_INFO("%s: model stats: n_elements=%" PRIu64 ", n_bytes=%zu\n", __func__, n_elements, n_bytes);

    // Transfer ownership of context and buffer to the model
    // This prevents them from being freed when safetensors_model_builder is destroyed
    if (ctx_meta && backend_buffer) {
        // Create unique pointer for buffer to manage lifetime
        ggml_backend_buffer_ptr buf_ptr(backend_buffer);

        // Add to model's context/buffer list
        std::vector<ggml_backend_buffer_ptr> bufs;
        bufs.push_back(std::move(buf_ptr));
        model->add_context_with_buffers(ctx_meta, std::move(bufs));

        // Release ownership from builder so destructor doesn't free them
        ctx_meta = nullptr;
        backend_buffer = nullptr;

        LLAMA_LOG_INFO("%s: transferred context and buffer ownership to model\n", __func__);
    }

    // Load chat template from tokenizer_config.json if available
    std::string tokenizer_config_path = model_dir + "/tokenizer_config.json";
    std::ifstream config_file(tokenizer_config_path);
    if (config_file.is_open()) {
        try {
            nlohmann::json tokenizer_config = nlohmann::json::parse(config_file);
            if (tokenizer_config.contains("chat_template") && tokenizer_config["chat_template"].is_string()) {
                std::string chat_template = tokenizer_config["chat_template"];
                std::string key = LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
                model->gguf_kv.emplace(key, chat_template);
                LLAMA_LOG_INFO("%s: loaded chat template from tokenizer_config.json\n", __func__);
            }
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: failed to parse tokenizer_config.json for chat template: %s\n", __func__, e.what());
        }
    }

    return true;
}
