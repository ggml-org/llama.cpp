#pragma once

#include "server_app.hpp"

/**
 * @brief Model loading and management utilities for the llama.cpp server
 * 
 * This module provides comprehensive model management functionality including:
 * - Language model loading and initialization
 * - Context creation and management
 * - Speculative decoding setup
 * - Multimodal model integration
 * - Model validation and compatibility checking
 * - Resource cleanup and lifecycle management
 * 
 * The utilities handle all aspects of model loading, from file validation
 * to context initialization, ensuring proper setup for inference operations.
 */

/**
 * @brief Initialize the llama backend and configure NUMA settings.
 * 
 * Performs global initialization of the llama.cpp backend, including
 * NUMA topology detection and configuration for optimal performance
 * on multi-socket systems.
 * 
 * @param numa_strategy NUMA initialization strategy
 */
void initialize_llama_backend(ggml_numa_strategy numa_strategy = GGML_NUMA_STRATEGY_DISABLED);

/**
 * @brief Validate model file accessibility and format.
 * 
 * Checks that the specified model file exists, is readable, and appears
 * to be in a valid GGUF format before attempting to load it.
 * 
 * @param model_path Path to the model file to validate
 * @return true if the model file is valid and accessible
 * @return false if validation fails
 */
bool validate_model_file(const std::string & model_path);

/**
 * @brief Load and initialize the primary language model.
 * 
 * Loads the main language model from the specified path with the given
 * parameters. Handles model loading, context creation, and initial
 * validation of the loaded model.
 * 
 * @param params Model loading parameters including paths and configuration
 * @return common_init_result Result containing model and context pointers
 */
common_init_result load_primary_model(const common_params & params);

/**
 * @brief Load and initialize draft model for speculative decoding.
 * 
 * Loads a secondary "draft" model used for speculative decoding to
 * accelerate inference. Validates compatibility with the primary model.
 * 
 * @param params_base Base parameters for the primary model
 * @param spec_params Speculative decoding specific parameters
 * @return common_init_result Result containing draft model and context
 */
common_init_result load_draft_model(const common_params & params_base,
                                   const common_params_speculative & spec_params);

/**
 * @brief Initialize multimodal context for vision/audio processing.
 * 
 * Sets up multimodal processing capabilities by loading the specified
 * multimodal projection model and initializing the processing context.
 * 
 * @param model Primary language model
 * @param mmproj_path Path to the multimodal projection model
 * @param params Multimodal initialization parameters
 * @return mtmd_context* Multimodal context or nullptr on failure
 */
mtmd_context * initialize_multimodal_context(llama_model * model,
                                            const std::string & mmproj_path,
                                            const common_params & params);

/**
 * @brief Validate speculative decoding compatibility.
 * 
 * Checks that the draft model is compatible with the primary model
 * for speculative decoding, including vocabulary compatibility and
 * architectural requirements.
 * 
 * @param primary_ctx Primary model context
 * @param draft_ctx Draft model context
 * @return true if models are compatible for speculative decoding
 * @return false if compatibility check fails
 */
bool validate_speculative_compatibility(llama_context * primary_ctx, 
                                       llama_context * draft_ctx);

/**
 * @brief Initialize chat templates for conversation formatting.
 * 
 * Loads and initializes chat templates for the given model, with
 * fallback to default templates if model-specific templates fail.
 * 
 * @param model Language model to initialize templates for
 * @param template_name Optional specific template name to use
 * @return common_chat_templates_ptr Initialized chat templates
 */
common_chat_templates_ptr initialize_chat_templates(llama_model * model,
                                                   const std::string & template_name = "");

/**
 * @brief Get model capabilities and metadata.
 * 
 * Extracts comprehensive information about the loaded model including
 * architecture, vocabulary size, context length, and supported features.
 * 
 * @param model Language model to query
 * @return json Model metadata and capabilities
 */
json get_model_info(llama_model * model);

/**
 * @brief Calculate optimal context allocation for parallel processing.
 * 
 * Determines the optimal distribution of context tokens across multiple
 * processing slots based on model constraints and parallel configuration.
 * 
 * @param total_ctx_size Total context size available
 * @param n_parallel Number of parallel processing slots
 * @param min_ctx_per_slot Minimum context tokens per slot
 * @return int32_t Recommended context size per slot
 */
int32_t calculate_slot_context_size(int32_t total_ctx_size, 
                                   int32_t n_parallel, 
                                   int32_t min_ctx_per_slot = 512);

/**
 * @brief Validate model memory requirements.
 * 
 * Estimates and validates that the system has sufficient memory to
 * load and operate the specified model configuration.
 * 
 * @param params Model parameters to validate
 * @return true if system has sufficient resources
 * @return false if resource requirements cannot be met
 */
bool validate_memory_requirements(const common_params & params);

/**
 * @brief Configure model tensor allocation strategies.
 * 
 * Sets up optimal tensor allocation strategies based on available
 * compute devices (CPU, GPU) and memory configuration.
 * 
 * @param params Model parameters to configure
 * @return true if tensor allocation was configured successfully
 * @return false if configuration failed
 */
bool configure_tensor_allocation(common_params & params);

/**
 * @brief Clean up model resources and contexts.
 * 
 * Properly releases all model-related resources including contexts,
 * tensors, and associated memory allocations.
 * 
 * @param llama_init Primary model initialization result
 * @param llama_init_dft Draft model initialization result (optional)
 * @param mctx Multimodal context (optional)
 */
void cleanup_model_resources(const common_init_result & llama_init,
                            const common_init_result & llama_init_dft,
                            mtmd_context * mctx = nullptr);

/**
 * @brief Hot-swap model without interrupting active sessions.
 * 
 * Attempts to replace the current model with a new one while minimizing
 * disruption to ongoing inference sessions. Not all configurations
 * support hot-swapping.
 * 
 * @param current_model Current model to replace
 * @param new_model_path Path to the new model
 * @param params Loading parameters for the new model
 * @return true if hot-swap was successful
 * @return false if hot-swap failed or is not supported
 */
bool hot_swap_model(llama_model * current_model,
                   const std::string & new_model_path,
                   const common_params & params);

/**
 * @brief Get detailed model loading progress information.
 * 
 * Provides progress information during model loading operations,
 * useful for providing feedback to users during long load times.
 * 
 * @return json Current loading progress and status
 */
json get_model_loading_progress();

/**
 * @brief Verify model file integrity and checksums.
 * 
 * Performs integrity checking on model files to ensure they haven't
 * been corrupted during download or storage.
 * 
 * @param model_path Path to model file to verify
 * @param expected_checksum Optional expected checksum for verification
 * @return true if model file integrity is verified
 * @return false if verification fails
 */
bool verify_model_integrity(const std::string & model_path,
                           const std::string & expected_checksum = "");

/**
 * @brief Configure model quantization and optimization settings.
 * 
 * Applies post-loading optimizations and quantization settings to
 * improve inference performance and memory usage.
 * 
 * @param model Model to optimize
 * @param params Optimization parameters
 * @return true if optimization was successful
 * @return false if optimization failed
 */
bool optimize_model_for_inference(llama_model * model, const common_params & params);