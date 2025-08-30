#pragma once

#include "server_app.hpp"

/**
 * @brief JSON utilities for the llama.cpp server
 * 
 * This module provides comprehensive JSON handling functionality including:
 * - Request/response serialization and deserialization
 * - Parameter validation and conversion
 * - Error response formatting
 * - Safe JSON string conversion
 * - Schema validation helpers
 * 
 * The utilities handle all JSON communication between the server and clients,
 * ensuring proper formatting, validation, and error handling for all endpoints.
 */

/**
 * @brief Safely convert a JSON object to string with error handling.
 * 
 * This function provides safe JSON serialization with proper error handling
 * for cases where JSON serialization might fail due to invalid characters
 * or other issues.
 * 
 * @param j JSON object to convert
 * @return std::string Serialized JSON string or error placeholder
 */
std::string safe_json_to_str(const json & j);

/**
 * @brief Format a standardized error response JSON object.
 * 
 * Creates a properly formatted error response following OpenAI-compatible
 * error format standards. Used throughout the server to ensure consistent
 * error reporting.
 * 
 * @param message Human-readable error message
 * @param type Type of error for categorization
 * @return json Formatted error response object
 */
json format_error_response(const std::string & message, const enum error_type type);

/**
 * @brief Parse and validate completion request parameters from JSON.
 * 
 * Extracts and validates completion parameters from a JSON request body,
 * applying defaults from the server configuration where appropriate.
 * 
 * @param ctx Llama context for model-specific validation
 * @param params_base Base server parameters for defaults
 * @param data JSON request data to parse
 * @return slot_params Parsed and validated parameters
 */
struct slot_params parse_params_from_json_cmpl(
    const llama_context * ctx,
    const common_params & params_base,
    const json & data);

/**
 * @brief Parse and validate embedding request parameters from JSON.
 * 
 * Similar to completion parameter parsing but specific to embedding
 * requests with different validation rules and default values.
 * 
 * @param ctx Llama context for model-specific validation
 * @param params_base Base server parameters for defaults
 * @param data JSON request data to parse
 * @return slot_params Parsed and validated parameters for embeddings
 */
struct slot_params parse_params_from_json_embd(
    const llama_context * ctx,
    const common_params & params_base,
    const json & data);

/**
 * @brief Convert slot parameters to JSON representation.
 * 
 * Serializes slot parameters to JSON format for response inclusion
 * or configuration display purposes.
 * 
 * @param params Slot parameters to serialize
 * @return json JSON representation of the parameters
 */
json slot_params_to_json(const struct slot_params & params);

/**
 * @brief Validate and parse logit bias from JSON array/object.
 * 
 * Handles the parsing of logit bias parameters which can be provided
 * in various JSON formats (array of pairs, object mapping, etc.).
 * 
 * @param data JSON logit bias data
 * @param vocab Vocabulary for token validation
 * @return std::unordered_map<llama_token, float> Parsed logit bias mapping
 */
std::unordered_map<llama_token, float> parse_logit_bias(const json & data, const llama_vocab * vocab);

/**
 * @brief Format logit bias mapping for JSON response.
 * 
 * Converts internal logit bias representation to JSON format suitable
 * for including in response objects.
 * 
 * @param logit_bias Internal logit bias mapping
 * @return json Formatted JSON representation
 */
json format_logit_bias(const std::unordered_map<llama_token, float> & logit_bias);

/**
 * @brief Parse tokens from JSON input with validation.
 * 
 * Handles parsing of token arrays from JSON input, with proper validation
 * and error handling for invalid token values.
 * 
 * @param data JSON token data (array of integers)
 * @param vocab Vocabulary for token validation
 * @return server_tokens Vector of validated tokens
 */
std::vector<llama_token> parse_tokens_from_json(const json & data, const llama_vocab * vocab);

/**
 * @brief Convert token array to JSON representation.
 * 
 * Serializes a vector of tokens to JSON array format for response
 * inclusion or debugging purposes.
 * 
 * @param tokens Vector of tokens to serialize
 * @return json JSON array of token values
 */
json tokens_to_json(const std::vector<llama_token> & tokens);

/**
 * @brief Parse and validate LoRA adapter configuration from JSON.
 * 
 * Extracts LoRA adapter configuration from JSON request data,
 * validating file paths and scale parameters.
 * 
 * @param data JSON LoRA configuration data
 * @return std::vector<common_adapter_lora_info> Parsed LoRA configurations
 */
std::vector<common_adapter_lora_info> parse_lora_adapters(const json & data);

/**
 * @brief Convert LoRA adapter configuration to JSON.
 * 
 * Serializes LoRA adapter configuration to JSON format for
 * response inclusion or status reporting.
 * 
 * @param adapters Vector of LoRA adapter configurations
 * @return json JSON representation of LoRA adapters
 */
json lora_adapters_to_json(const std::vector<common_adapter_lora_info> & adapters);

/**
 * @brief Validate and parse multimodal content from JSON.
 * 
 * Handles parsing of multimodal content (text, images, audio) from
 * JSON request data with proper validation and format checking.
 * 
 * @param data JSON multimodal content data
 * @return json Parsed and validated multimodal content
 */
json parse_multimodal_content(const json & data);

/**
 * @brief Create standard OpenAI-compatible model listing response.
 * 
 * Generates a properly formatted model listing response following
 * OpenAI API standards for /v1/models endpoint.
 * 
 * @param model_path Path to the loaded model file
 * @param model_alias Optional alias for the model
 * @return json OpenAI-compatible model listing response
 */
json create_models_response(const std::string & model_path, const std::string & model_alias = "");

/**
 * @brief Create completion response in OpenAI-compatible format.
 * 
 * Formats completion results into OpenAI-compatible response structure
 * with proper choice formatting, usage statistics, and metadata.
 * 
 * @param result Completion task result
 * @param model_name Name of the model used
 * @param request_id Unique identifier for the request
 * @return json OpenAI-compatible completion response
 */
json create_completion_response(const struct server_task_result & result, 
                               const std::string & model_name,
                               const std::string & request_id);

/**
 * @brief Create chat completion response in OpenAI-compatible format.
 * 
 * Similar to completion response but formatted specifically for chat
 * completion endpoints with message-based structure.
 * 
 * @param result Chat completion task result
 * @param model_name Name of the model used
 * @param request_id Unique identifier for the request
 * @return json OpenAI-compatible chat completion response
 */
json create_chat_completion_response(const struct server_task_result & result,
                                    const std::string & model_name,
                                    const std::string & request_id);

/**
 * @brief Create embedding response in OpenAI-compatible format.
 * 
 * Formats embedding results into OpenAI-compatible response structure
 * with proper data formatting and usage statistics.
 * 
 * @param result Embedding task result
 * @param model_name Name of the model used
 * @param request_id Unique identifier for the request
 * @return json OpenAI-compatible embedding response
 */
json create_embedding_response(const struct server_task_result & result,
                              const std::string & model_name,
                              const std::string & request_id);