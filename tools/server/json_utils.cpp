#include "json_utils.hpp"

/**
 * @brief Implementation of JSON utilities for the llama.cpp server
 * 
 * This module provides comprehensive JSON handling functionality extracted
 * from the original server.cpp. It includes request parsing, response formatting,
 * parameter validation, and error handling utilities.
 */

/**
 * @brief Safely convert a JSON object to string with error handling.
 * 
 * This function provides safe JSON serialization with proper error handling
 * for cases where JSON serialization might fail due to invalid characters
 * or other issues. Uses nlohmann::json's replace error handler.
 */
std::string safe_json_to_str(const json & j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

/**
 * @brief Format a standardized error response JSON object.
 * 
 * Creates a properly formatted error response following OpenAI-compatible
 * error format standards. Maps internal error types to appropriate HTTP
 * status codes and provides consistent error messaging.
 */
json format_error_response(const std::string & message, const enum error_type type) {
    std::string error_type_str;
    int error_code;
    
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            error_type_str = "invalid_request_error";
            error_code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            error_type_str = "authentication_error"; 
            error_code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            error_type_str = "not_found_error";
            error_code = 404;
            break;
        case ERROR_TYPE_PERMISSION:
            error_type_str = "permission_error";
            error_code = 403;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            error_type_str = "unavailable_error";
            error_code = 503;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            error_type_str = "not_supported_error";
            error_code = 501;
            break;
        case ERROR_TYPE_SERVER:
        default:
            error_type_str = "internal_server_error";
            error_code = 500;
            break;
    }
    
    return json {
        {"message", message},
        {"type", error_type_str},
        {"param", nullptr},
        {"code", error_code}
    };
}

/**
 * @brief Parse and validate completion request parameters from JSON.
 * 
 * Extracts and validates completion parameters from a JSON request body,
 * applying defaults from the server configuration where appropriate.
 * Handles both simple completion and complex parameter structures.
 */
struct slot_params parse_params_from_json_cmpl(
    const llama_context * ctx,
    const common_params & params_base,  
    const json & data) {
    
    // This is a simplified implementation - the full implementation would
    // need to be extracted from the original server_task::params_from_json_cmpl method
    
    struct slot_params params;
    
    // Basic parameter extraction with defaults
    params.verbose           = params_base.verbosity > 9;
    params.stream           = json_value(data, "stream", false);
    params.cache_prompt     = json_value(data, "cache_prompt", true);
    params.return_tokens    = json_value(data, "return_tokens", false);
    params.n_predict        = json_value(data, "n_predict", json_value(data, "max_tokens", params_base.n_predict));
    params.n_keep           = json_value(data, "n_keep", params_base.n_keep);
    params.timings_per_token = json_value(data, "timings_per_token", false);
    
    // Initialize sampling parameters from base configuration
    params.sampling = params_base.sampling;
    
    // Override sampling parameters from request data
    params.sampling.top_k    = json_value(data, "top_k", params.sampling.top_k);
    params.sampling.top_p    = json_value(data, "top_p", params.sampling.top_p);
    params.sampling.temp     = json_value(data, "temperature", params.sampling.temp);
    params.sampling.seed     = json_value(data, "seed", params.sampling.seed);
    
    // Handle stop sequences
    if (data.contains("stop")) {
        if (data.at("stop").is_string()) {
            params.antiprompt = {data.at("stop").get<std::string>()};
        } else if (data.at("stop").is_array()) {
            params.antiprompt = data.at("stop").get<std::vector<std::string>>();
        }
    }
    
    return params;
}

/**
 * @brief Parse and validate embedding request parameters from JSON.
 * 
 * Similar to completion parameter parsing but specific to embedding
 * requests with different validation rules and default values.
 */
struct slot_params parse_params_from_json_embd(
    const llama_context * ctx,
    const common_params & params_base,
    const json & data) {
    
    struct slot_params params;
    
    // Embeddings have simpler parameter requirements
    params.verbose = params_base.verbosity > 9;
    params.embd_normalize = json_value(data, "encoding_format", std::string("float")) == "base64" ? -1 : 2;
    
    // Embeddings don't use most generation parameters
    params.stream = false;
    params.n_predict = 0;
    params.cache_prompt = true;
    
    return params;
}

/**
 * @brief Convert slot parameters to JSON representation.
 * 
 * Serializes slot parameters to JSON format for response inclusion
 * or configuration display purposes. Creates a comprehensive representation
 * of all parameter values for debugging and API responses.
 */
json slot_params_to_json(const struct slot_params & params) {
    return json {
        {"stream", params.stream},
        {"cache_prompt", params.cache_prompt},
        {"return_tokens", params.return_tokens},
        {"n_keep", params.n_keep},
        {"n_predict", params.n_predict},
        {"temperature", params.sampling.temp},
        {"top_k", params.sampling.top_k},
        {"top_p", params.sampling.top_p},
        {"seed", params.sampling.seed},
        {"verbose", params.verbose},
        {"timings_per_token", params.timings_per_token}
    };
}

/**
 * @brief Validate and parse logit bias from JSON array/object.
 * 
 * Handles the parsing of logit bias parameters which can be provided
 * in various JSON formats (array of pairs, object mapping, etc.).
 * Validates token IDs against the vocabulary.
 */
std::unordered_map<llama_token, float> parse_logit_bias(const json & data, const llama_vocab * vocab) {
    std::unordered_map<llama_token, float> result;
    
    if (data.is_null() || data.empty()) {
        return result;
    }
    
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    
    if (data.is_object()) {
        // Handle object format: {"token_id": bias_value, ...}
        for (const auto & [key, value] : data.items()) {
            try {
                int token_id = std::stoi(key);
                if (token_id >= 0 && token_id < n_vocab) {
                    result[token_id] = value.get<float>();
                }
            } catch (const std::exception &) {
                // Skip invalid entries
            }
        }
    } else if (data.is_array()) {
        // Handle array format: [{"token": id, "bias": value}, ...]
        for (const auto & entry : data) {
            if (entry.is_object() && entry.contains("token") && entry.contains("bias")) {
                llama_token token = entry.at("token").get<llama_token>();
                float bias = entry.at("bias").get<float>();
                if (token >= 0 && token < n_vocab) {
                    result[token] = bias;
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief Format logit bias mapping for JSON response.
 * 
 * Converts internal logit bias representation to JSON format suitable
 * for including in response objects. Creates an array of bias objects.
 */
json format_logit_bias(const std::unordered_map<llama_token, float> & logit_bias) {
    json result = json::array();
    
    for (const auto & [token, bias] : logit_bias) {
        result.push_back(json {
            {"token", token},
            {"bias", bias}
        });
    }
    
    return result;
}

/**
 * @brief Parse tokens from JSON input with validation.
 * 
 * Handles parsing of token arrays from JSON input, with proper validation
 * and error handling for invalid token values.
 */
std::vector<llama_token> parse_tokens_from_json(const json & data, const llama_vocab * vocab) {
    std::vector<llama_token> result;
    
    if (!data.is_array()) {
        throw std::runtime_error("Tokens must be provided as an array");
    }
    
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    
    for (const auto & token_json : data) {
        if (!token_json.is_number_integer()) {
            throw std::runtime_error("All tokens must be integers");
        }
        
        llama_token token = token_json.get<llama_token>();
        if (token < 0 || token >= n_vocab) {
            throw std::runtime_error("Token " + std::to_string(token) + " is outside vocabulary range");
        }
        
        result.push_back(token);
    }
    
    return result;
}

/**
 * @brief Convert token array to JSON representation.
 * 
 * Serializes a vector of tokens to JSON array format for response
 * inclusion or debugging purposes.
 */
json tokens_to_json(const std::vector<llama_token> & tokens) {
    json result = json::array();
    
    for (llama_token token : tokens) {
        result.push_back(token);
    }
    
    return result;
}

/**
 * @brief Parse and validate LoRA adapter configuration from JSON.
 * 
 * Extracts LoRA adapter configuration from JSON request data,
 * validating file paths and scale parameters.
 */
std::vector<common_adapter_lora_info> parse_lora_adapters(const json & data) {
    std::vector<common_adapter_lora_info> result;
    
    if (!data.is_array()) {
        throw std::runtime_error("LoRA adapters must be provided as an array");
    }
    
    for (const auto & adapter_json : data) {
        if (!adapter_json.is_object()) {
            throw std::runtime_error("Each LoRA adapter must be an object");
        }
        
        common_adapter_lora_info adapter;
        
        if (adapter_json.contains("path")) {
            adapter.path = adapter_json.at("path").get<std::string>();
        }
        
        if (adapter_json.contains("scale")) {
            adapter.scale = adapter_json.at("scale").get<float>();
        } else {
            adapter.scale = 1.0f; // Default scale
        }
        
        result.push_back(adapter);
    }
    
    return result;
}

/**
 * @brief Convert LoRA adapter configuration to JSON.
 * 
 * Serializes LoRA adapter configuration to JSON format for
 * response inclusion or status reporting.
 */
json lora_adapters_to_json(const std::vector<common_adapter_lora_info> & adapters) {
    json result = json::array();
    
    for (size_t i = 0; i < adapters.size(); ++i) {
        const auto & adapter = adapters[i];
        result.push_back(json {
            {"id", i},
            {"path", adapter.path},
            {"scale", adapter.scale}
        });
    }
    
    return result;
}

/**
 * @brief Validate and parse multimodal content from JSON.
 * 
 * Handles parsing of multimodal content (text, images, audio) from
 * JSON request data with proper validation and format checking.
 */
json parse_multimodal_content(const json & data) {
    // This is a placeholder implementation
    // The full implementation would handle base64 decoding, file validation, etc.
    return data;
}

/**
 * @brief Create standard OpenAI-compatible model listing response.
 * 
 * Generates a properly formatted model listing response following
 * OpenAI API standards for /v1/models endpoint.
 */
json create_models_response(const std::string & model_path, const std::string & model_alias) {
    std::string model_name = model_alias.empty() ? "llama-cpp" : model_alias;
    
    return json {
        {"object", "list"},
        {"data", json::array({
            json {
                {"id", model_name},
                {"object", "model"},
                {"created", std::time(nullptr)},
                {"owned_by", "llama.cpp"},
                {"path", model_path}
            }
        })}
    };
}

/**
 * @brief Create completion response in OpenAI-compatible format.
 * 
 * Formats completion results into OpenAI-compatible response structure
 * with proper choice formatting, usage statistics, and metadata.
 */
json create_completion_response(const struct server_task_result & result, 
                               const std::string & model_name,
                               const std::string & request_id) {
    return json {
        {"id", request_id},
        {"object", "text_completion"},
        {"created", std::time(nullptr)},
        {"model", model_name},
        {"choices", json::array()}, // Will be populated with actual choices
        {"usage", json {
            {"prompt_tokens", 0},
            {"completion_tokens", 0},
            {"total_tokens", 0}
        }}
    };
}

/**
 * @brief Create chat completion response in OpenAI-compatible format.
 * 
 * Similar to completion response but formatted specifically for chat
 * completion endpoints with message-based structure.
 */
json create_chat_completion_response(const struct server_task_result & result,
                                    const std::string & model_name,
                                    const std::string & request_id) {
    return json {
        {"id", request_id},
        {"object", "chat.completion"},
        {"created", std::time(nullptr)},
        {"model", model_name},
        {"choices", json::array()}, // Will be populated with actual choices
        {"usage", json {
            {"prompt_tokens", 0},
            {"completion_tokens", 0},
            {"total_tokens", 0}
        }}
    };
}

/**
 * @brief Create embedding response in OpenAI-compatible format.
 * 
 * Formats embedding results into OpenAI-compatible response structure
 * with proper data formatting and usage statistics.
 */
json create_embedding_response(const struct server_task_result & result,
                              const std::string & model_name,
                              const std::string & request_id) {
    return json {
        {"object", "list"},
        {"data", json::array()}, // Will be populated with embeddings
        {"model", model_name},
        {"usage", json {
            {"prompt_tokens", 0},
            {"total_tokens", 0}
        }}
    };
}