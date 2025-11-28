#include "llama-hf-config.h"

#include <fstream>
#include "../vendor/nlohmann/json.hpp"

using json = nlohmann::json;

bool hf_config::load_from_file(const std::string & config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        error_msg = "Failed to open config file: " + config_path;
        return false;
    }

    try {
        config = std::make_unique<json>();
        f >> *config;
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to parse config JSON: ") + e.what();
        return false;
    }

    return true;
}

bool hf_config::load_from_string(const std::string & json_str) {
    try {
        config = std::make_unique<json>(json::parse(json_str));
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to parse config JSON: ") + e.what();
        return false;
    }

    return true;
}

std::string hf_config::get_architecture() const {
    if (!config) {
        return "";
    }

    // Check for architectures array (most common)
    if (config->contains("architectures") && (*config)["architectures"].is_array()) {
        const auto & archs = (*config)["architectures"];
        if (!archs.empty() && archs[0].is_string()) {
            return archs[0].get<std::string>();
        }
    }

    // Check text_config (for multimodal models)
    if (config->contains("text_config") && (*config)["text_config"].is_object()) {
        const auto & text_config = (*config)["text_config"];
        if (text_config.contains("architectures") && text_config["architectures"].is_array()) {
            const auto & archs = text_config["architectures"];
            if (!archs.empty() && archs[0].is_string()) {
                return archs[0].get<std::string>();
            }
        }
    }

    // Check for ssm_cfg (Mamba models)
    if (config->contains("ssm_cfg") && (*config)["ssm_cfg"].is_object()) {
        const auto & ssm_cfg = (*config)["ssm_cfg"];
        if (ssm_cfg.contains("layer") && ssm_cfg["layer"].is_string()) {
            return ssm_cfg["layer"].get<std::string>() + "ForCausalLM";
        }
    }

    return "";
}

template<typename T>
bool hf_config::get_value_with_fallback(const std::string & key, T & out) const {
    if (!config) {
        return false;
    }

    // First try root level
    if (config->contains(key)) {
        try {
            out = (*config)[key].get<T>();
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }

    // Try text_config (for multimodal models)
    if (config->contains("text_config") && (*config)["text_config"].is_object()) {
        const auto & text_config = (*config)["text_config"];
        if (text_config.contains(key)) {
            try {
                out = text_config[key].get<T>();
                return true;
            } catch (const std::exception &) {
                return false;
            }
        }
    }

    return false;
}

bool hf_config::get_int(const std::string & key, int64_t & out) const {
    return get_value_with_fallback(key, out);
}

bool hf_config::get_float(const std::string & key, double & out) const {
    return get_value_with_fallback(key, out);
}

bool hf_config::get_string(const std::string & key, std::string & out) const {
    return get_value_with_fallback(key, out);
}

bool hf_config::get_bool(const std::string & key, bool & out) const {
    return get_value_with_fallback(key, out);
}

bool hf_config::has_key(const std::string & key) const {
    if (!config) {
        return false;
    }

    if (config->contains(key)) {
        return true;
    }

    // Check text_config
    if (config->contains("text_config") && (*config)["text_config"].is_object()) {
        return (*config)["text_config"].contains(key);
    }

    return false;
}

const nlohmann::json * hf_config::get_json() const {
    return config.get();
}

// Common configuration getters

int64_t hf_config::get_hidden_size() const {
    int64_t val = 0;
    // Try multiple possible keys
    if (get_int("hidden_size", val)) return val;
    if (get_int("d_model", val)) return val;
    if (get_int("n_embd", val)) return val;
    return 0;
}

int64_t hf_config::get_num_hidden_layers() const {
    int64_t val = 0;
    if (get_int("num_hidden_layers", val)) return val;
    if (get_int("n_layers", val)) return val;
    if (get_int("n_layer", val)) return val;
    if (get_int("num_layers", val)) return val;
    return 0;
}

int64_t hf_config::get_num_attention_heads() const {
    int64_t val = 0;
    if (get_int("num_attention_heads", val)) return val;
    if (get_int("n_heads", val)) return val;
    if (get_int("n_head", val)) return val;
    return 0;
}

int64_t hf_config::get_num_key_value_heads() const {
    int64_t val = 0;
    if (get_int("num_key_value_heads", val)) return val;
    // If not specified, defaults to num_attention_heads (MHA)
    return get_num_attention_heads();
}

int64_t hf_config::get_intermediate_size() const {
    int64_t val = 0;
    if (get_int("intermediate_size", val)) return val;
    if (get_int("n_inner", val)) return val;
    return 0;
}

int64_t hf_config::get_vocab_size() const {
    int64_t val = 0;
    if (get_int("vocab_size", val)) return val;
    if (get_int("padded_vocab_size", val)) return val;
    return 0;
}

int64_t hf_config::get_max_position_embeddings() const {
    int64_t val = 0;
    if (get_int("max_position_embeddings", val)) return val;
    if (get_int("n_positions", val)) return val;
    if (get_int("n_ctx", val)) return val;
    return 0;
}

double hf_config::get_rms_norm_eps() const {
    double val = 0;
    if (get_float("rms_norm_eps", val)) return val;
    if (get_float("layer_norm_eps", val)) return val;
    if (get_float("layer_norm_epsilon", val)) return val;
    return 1e-5;  // common default
}

std::string hf_config::get_rope_scaling_type() const {
    if (!config) {
        return "";
    }

    // Check for rope_scaling object
    if (config->contains("rope_scaling") && (*config)["rope_scaling"].is_object()) {
        const auto & rope_scaling = (*config)["rope_scaling"];
        if (rope_scaling.contains("type") && rope_scaling["type"].is_string()) {
            return rope_scaling["type"].get<std::string>();
        }
    }

    return "";
}
