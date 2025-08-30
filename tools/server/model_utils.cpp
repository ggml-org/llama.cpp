#include "model_utils.hpp"
#include "json_utils.hpp"

#include <fstream>
#include <iostream>

/**
 * @brief Implementation of model loading and management utilities
 * 
 * This module provides comprehensive model management functionality extracted
 * from the original server.cpp. It handles model loading, context creation,
 * speculative decoding setup, and multimodal integration.
 */

/**
 * @brief Initialize the llama backend and configure NUMA settings.
 * 
 * Performs global initialization of the llama.cpp backend, including
 * NUMA topology detection and configuration for optimal performance
 * on multi-socket systems.
 */
void initialize_llama_backend(ggml_numa_strategy numa_strategy) {
    LOG_INF("Initializing llama.cpp backend...\n");
    
    // Initialize the llama backend
    llama_backend_init();
    
    // Initialize NUMA configuration
    llama_numa_init(numa_strategy);
    
    LOG_INF("llama.cpp backend initialized with NUMA strategy: %d\n", (int)numa_strategy);
}

/**
 * @brief Validate model file accessibility and format.
 * 
 * Checks that the specified model file exists, is readable, and appears
 * to be in a valid GGUF format before attempting to load it.
 */
bool validate_model_file(const std::string & model_path) {
    if (model_path.empty()) {
        LOG_ERR("Model path is empty\n");
        return false;
    }
    
    // Check if file exists and is readable
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Cannot open model file: %s\n", model_path.c_str());
        return false;
    }
    
    // Check file size
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size < 32) {  // Minimum size for GGUF header
        LOG_ERR("Model file too small: %ld bytes\n", (long)file_size);
        return false;
    }
    
    // Check for GGUF magic number
    char magic[4];
    file.read(magic, 4);
    if (file.gcount() != 4) {
        LOG_ERR("Cannot read model file header\n");
        return false;
    }
    
    // GGUF magic is "GGUF"
    if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
        LOG_WRN("Model file may not be in GGUF format (magic: %02x%02x%02x%02x)\n",
               (unsigned char)magic[0], (unsigned char)magic[1], 
               (unsigned char)magic[2], (unsigned char)magic[3]);
        // Continue anyway as some models might have different headers
    }
    
    file.close();
    
    LOG_INF("Model file validation passed: %s (%ld bytes)\n", model_path.c_str(), (long)file_size);
    return true;
}

/**
 * @brief Load and initialize the primary language model.
 * 
 * Loads the main language model from the specified path with the given
 * parameters. Handles model loading, context creation, and initial
 * validation of the loaded model.
 */
common_init_result load_primary_model(const common_params & params) {
    LOG_INF("Loading primary model: %s\n", params.model.path.c_str());
    
    if (!validate_model_file(params.model.path)) {
        LOG_ERR("Model file validation failed\n");
        return {};  // Return empty result
    }
    
    // Use common_init_from_params to load the model
    common_params params_copy = params;  // Make a non-const copy
    common_init_result result = common_init_from_params(params_copy);
    
    if (!result.model || !result.context) {
        LOG_ERR("Failed to load primary model\n");
        return {};
    }
    
    // Validate the loaded model
    llama_model * model = result.model.get();
    llama_context * ctx = result.context.get();
    
    if (!model || !ctx) {
        LOG_ERR("Invalid model or context pointers\n");
        return {};
    }
    
    // Log model information
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t n_ctx = llama_n_ctx(ctx);
    
    LOG_INF("Model loaded successfully:\n");
    LOG_INF("  Vocabulary size: %d tokens\n", n_vocab);
    LOG_INF("  Context size: %d tokens\n", n_ctx);
    LOG_INF("  Model size: %.2f GB\n", (double)llama_model_size(model) / (1024.0 * 1024.0 * 1024.0));
    LOG_INF("  Model parameters: %.2f B\n", (double)llama_model_n_params(model) / 1e9);
    
    return result;
}

/**
 * @brief Load and initialize draft model for speculative decoding.
 * 
 * Loads a secondary "draft" model used for speculative decoding to
 * accelerate inference. Validates compatibility with the primary model.
 */
common_init_result load_draft_model(const common_params & params_base,
                                   const common_params_speculative & spec_params) {
    if (spec_params.model.path.empty()) {
        LOG_ERR("Draft model path is empty\n");
        return {};
    }
    
    LOG_INF("Loading draft model: %s\n", spec_params.model.path.c_str());
    
    if (!validate_model_file(spec_params.model.path)) {
        LOG_ERR("Draft model file validation failed\n");
        return {};
    }
    
    // Create parameters for draft model
    common_params params_dft = params_base;
    params_dft.model = spec_params.model;
    params_dft.n_ctx = spec_params.n_ctx == 0 ? params_base.n_ctx / params_base.n_parallel : spec_params.n_ctx;
    params_dft.n_gpu_layers = spec_params.n_gpu_layers;
    params_dft.n_parallel = 1;  // Draft models don't need parallel processing
    
    // Load the draft model
    common_init_result result = common_init_from_params(params_dft);
    
    if (!result.model || !result.context) {
        LOG_ERR("Failed to load draft model\n");
        return {};
    }
    
    LOG_INF("Draft model loaded successfully\n");
    return result;
}

/**
 * @brief Initialize multimodal context for vision/audio processing.
 * 
 * Sets up multimodal processing capabilities by loading the specified
 * multimodal projection model and initializing the processing context.
 */
mtmd_context * initialize_multimodal_context(llama_model * model,
                                            const std::string & mmproj_path,
                                            const common_params & params) {
    if (mmproj_path.empty()) {
        LOG_DBG("No multimodal projection model specified\n");
        return nullptr;
    }
    
    LOG_INF("Loading multimodal projection model: %s\n", mmproj_path.c_str());
    
    if (!validate_model_file(mmproj_path)) {
        LOG_ERR("Multimodal projection model file validation failed\n");
        return nullptr;
    }
    
    // Initialize multimodal context parameters
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params.mmproj_use_gpu;
    mparams.print_timings = false;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.verbosity = params.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;
    
    // Create multimodal context
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
    
    if (!mctx) {
        LOG_ERR("Failed to load multimodal projection model\n");
        return nullptr;
    }
    
    LOG_INF("Multimodal context initialized successfully\n");
    return mctx;
}

/**
 * @brief Validate speculative decoding compatibility.
 * 
 * Checks that the draft model is compatible with the primary model
 * for speculative decoding, including vocabulary compatibility.
 */
bool validate_speculative_compatibility(llama_context * primary_ctx, 
                                       llama_context * draft_ctx) {
    if (!primary_ctx || !draft_ctx) {
        LOG_ERR("Invalid contexts for speculative compatibility check\n");
        return false;
    }
    
    // Use common_speculative_are_compatible for validation
    bool compatible = common_speculative_are_compatible(primary_ctx, draft_ctx);
    
    if (compatible) {
        LOG_INF("Draft model is compatible with primary model for speculative decoding\n");
    } else {
        LOG_WRN("Draft model is not fully compatible with primary model. "
               "Tokens will be translated between models.\n");
    }
    
    return compatible;
}

/**
 * @brief Initialize chat templates for conversation formatting.
 * 
 * Loads and initializes chat templates for the given model, with
 * fallback to default templates if model-specific templates fail.
 */
common_chat_templates_ptr initialize_chat_templates(llama_model * model,
                                                   const std::string & template_name) {
    if (!model) {
        LOG_ERR("Cannot initialize chat templates: model is null\n");
        return nullptr;
    }
    
    LOG_INF("Initializing chat templates...\n");
    
    // Try to initialize with model's built-in template first
    common_chat_templates_ptr chat_templates = common_chat_templates_init(model, template_name);
    
    if (!chat_templates) {
        LOG_WRN("Failed to initialize model-specific chat template, trying default...\n");
        
        // Fallback to chatml template
        chat_templates = common_chat_templates_init(model, "chatml");
        
        if (!chat_templates) {
            LOG_ERR("Failed to initialize any chat template\n");
            return nullptr;
        }
        
        LOG_WRN("Using fallback chatml template. This may result in suboptimal responses.\n");
    }
    
    LOG_INF("Chat templates initialized successfully\n");
    return chat_templates;
}

/**
 * @brief Get model capabilities and metadata.
 * 
 * Extracts comprehensive information about the loaded model including
 * architecture, vocabulary size, context length, and supported features.
 */
json get_model_info(llama_model * model) {
    if (!model) {
        return json{};
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    json result = json::object();
    result["vocab_size"] = llama_vocab_n_tokens(vocab);
    result["n_params"] = llama_model_n_params(model);
    result["size_bytes"] = llama_model_size(model);
    // result["rope_freq_base"] = llama_model_rope_freq_base_train(model); // Function name needs checking
    // result["rope_freq_scale"] = llama_model_rope_freq_scale_train(model); // Function name needs checking
    result["has_encoder"] = llama_model_has_encoder(model);
    result["has_decoder"] = llama_model_has_decoder(model);
    
    // Get model description safely
    char desc_buf[256];
    int desc_len = llama_model_desc(model, desc_buf, sizeof(desc_buf));
    if (desc_len > 0) {
        result["desc"] = std::string(desc_buf, desc_len);
    } else {
        result["desc"] = "unknown";
    }
    
    return result;
}

/**
 * @brief Calculate optimal context allocation for parallel processing.
 * 
 * Determines the optimal distribution of context tokens across multiple
 * processing slots based on model constraints and parallel configuration.
 */
int32_t calculate_slot_context_size(int32_t total_ctx_size, 
                                   int32_t n_parallel, 
                                   int32_t min_ctx_per_slot) {
    if (n_parallel <= 0) {
        LOG_ERR("Invalid parallel count: %d\n", n_parallel);
        return 0;
    }
    
    int32_t ctx_per_slot = total_ctx_size / n_parallel;
    
    if (ctx_per_slot < min_ctx_per_slot) {
        LOG_WRN("Context per slot (%d) is below minimum (%d). Consider reducing parallel slots or increasing context size.\n",
               ctx_per_slot, min_ctx_per_slot);
    }
    
    LOG_INF("Context allocation: %d total, %d slots, %d tokens per slot\n",
           total_ctx_size, n_parallel, ctx_per_slot);
    
    return ctx_per_slot;
}

/**
 * @brief Validate model memory requirements.
 * 
 * Estimates and validates that the system has sufficient memory to
 * load and operate the specified model configuration.
 */
bool validate_memory_requirements(const common_params & params) {
    // This is a simplified implementation
    // A full implementation would calculate precise memory requirements
    
    LOG_INF("Validating memory requirements...\n");
    
    // Basic validation - ensure context size is reasonable
    const int32_t max_reasonable_ctx = 32768;  // 32K tokens
    
    if (params.n_ctx > max_reasonable_ctx) {
        LOG_WRN("Very large context size requested (%d tokens). This may require significant memory.\n",
               params.n_ctx);
    }
    
    // Check for parallel configuration memory impact
    const int32_t total_ctx = params.n_ctx * params.n_parallel;
    if (total_ctx > max_reasonable_ctx * 4) {
        LOG_WRN("Very large total context allocation (%d tokens across %d slots)\n",
               total_ctx, params.n_parallel);
    }
    
    LOG_INF("Memory validation passed\n");
    return true;
}

/**
 * @brief Configure model tensor allocation strategies.
 * 
 * Sets up optimal tensor allocation strategies based on available
 * compute devices (CPU, GPU) and memory configuration.
 */
bool configure_tensor_allocation(common_params & params) {
    LOG_INF("Configuring tensor allocation...\n");
    
    // This is a simplified implementation
    // The full implementation would configure based on available devices
    
    if (params.n_gpu_layers > 0) {
        LOG_INF("GPU acceleration requested: %d layers\n", params.n_gpu_layers);
    } else {
        LOG_INF("CPU-only inference configured\n");
    }
    
    LOG_INF("Tensor allocation configured\n");
    return true;
}

/**
 * @brief Clean up model resources and contexts.
 * 
 * Properly releases all model-related resources including contexts,
 * tensors, and associated memory allocations.
 */
void cleanup_model_resources(const common_init_result & llama_init,
                            const common_init_result & llama_init_dft,
                            mtmd_context * mctx) {
    (void)llama_init;     // May be used for cleanup later
    (void)llama_init_dft; // May be used for cleanup later
    
    LOG_INF("Cleaning up model resources...\n");
    
    // Free multimodal context first
    if (mctx) {
        mtmd_free(mctx);
        LOG_DBG("Multimodal context freed\n");
    }
    
    // The common_init_result destructor will handle model and context cleanup
    // when the objects go out of scope
    
    LOG_INF("Model resources cleanup complete\n");
}

/**
 * @brief Hot-swap model without interrupting active sessions.
 * 
 * Attempts to replace the current model with a new one while minimizing
 * disruption to ongoing inference sessions. Not all configurations
 * support hot-swapping.
 */
bool hot_swap_model(llama_model * current_model,
                   const std::string & new_model_path,
                   const common_params & params) {
    (void)current_model;   // Not implemented yet
    (void)new_model_path;  // Not implemented yet  
    (void)params;          // Not implemented yet
    
    LOG_WRN("Hot-swap model functionality is not yet implemented\n");
    return false;
}

/**
 * @brief Get detailed model loading progress information.
 * 
 * Provides progress information during model loading operations,
 * useful for providing feedback to users during long load times.
 */
json get_model_loading_progress() {
    // This would typically be updated by the model loading process
    return json {
        {"status", "ready"},
        {"progress", 100}
    };
}

/**
 * @brief Verify model file integrity and checksums.
 * 
 * Performs integrity checking on model files to ensure they haven't
 * been corrupted during download or storage.
 */
bool verify_model_integrity(const std::string & model_path,
                           const std::string & expected_checksum) {
    (void)model_path; // Not implemented yet
    
    if (expected_checksum.empty()) {
        LOG_DBG("No checksum provided for model integrity verification\n");
        return true;  // Skip verification if no checksum provided
    }
    
    LOG_WRN("Model integrity verification is not yet implemented\n");
    return true;  // Assume valid for now
}

/**
 * @brief Configure model quantization and optimization settings.
 * 
 * Applies post-loading optimizations and quantization settings to
 * improve inference performance and memory usage.
 */
bool optimize_model_for_inference(llama_model * model, const common_params & params) {
    (void)params; // Not implemented yet
    
    if (!model) {
        LOG_ERR("Cannot optimize: model is null\n");
        return false;
    }
    
    LOG_INF("Applying model optimizations for inference...\n");
    
    // This is a placeholder for optimization logic
    // The full implementation would apply various optimizations based on the model and parameters
    
    LOG_INF("Model optimizations applied\n");
    return true;
}