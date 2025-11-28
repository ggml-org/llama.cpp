#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../vendor/nlohmann/json.hpp"

// HuggingFace model configuration
class hf_config {
public:
    hf_config() = default;
    ~hf_config() = default;

    // Load config from file
    bool load_from_file(const std::string & config_path);

    // Load config from JSON string
    bool load_from_string(const std::string & json_str);

    // Get architecture name (e.g., "LlamaForCausalLM", "MistralForCausalLM")
    std::string get_architecture() const;

    // Get a configuration value as integer
    bool get_int(const std::string & key, int64_t & out) const;

    // Get a configuration value as float
    bool get_float(const std::string & key, double & out) const;

    // Get a configuration value as string
    bool get_string(const std::string & key, std::string & out) const;

    // Get a configuration value as bool
    bool get_bool(const std::string & key, bool & out) const;

    // Check if a key exists
    bool has_key(const std::string & key) const;

    // Get raw JSON object (for advanced users)
    const nlohmann::json * get_json() const;

    // Get last error message
    const std::string & get_error() const { return error_msg; }

    // Common configuration getters
    int64_t get_hidden_size() const;
    int64_t get_num_hidden_layers() const;
    int64_t get_num_attention_heads() const;
    int64_t get_num_key_value_heads() const;
    int64_t get_intermediate_size() const;
    int64_t get_vocab_size() const;
    int64_t get_max_position_embeddings() const;
    double get_rms_norm_eps() const;
    std::string get_rope_scaling_type() const;

private:
    std::unique_ptr<nlohmann::json> config;
    std::string error_msg;

    // Helper to get value, checking nested configs (text_config, vision_config)
    template<typename T>
    bool get_value_with_fallback(const std::string & key, T & out) const;
};
