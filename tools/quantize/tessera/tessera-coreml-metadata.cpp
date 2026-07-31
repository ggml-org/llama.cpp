#include "tessera-coreml-metadata.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

// Minimal JSON value extraction (no dependency on common/json.hpp for
// portability; the sidecar format is flat key-value).
static std::string ts_json_get_string(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

static int64_t ts_json_get_int(const std::string & json, const std::string & key, int64_t def) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return std::atoll(json.c_str() + pos);
}

static float ts_json_get_float(const std::string & json, const std::string & key, float def) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return (float) std::atof(json.c_str() + pos);
}

static bool ts_json_get_bool(const std::string & json, const std::string & key, bool def) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    return json.find("true", pos) != std::string::npos &&
           json.find("true", pos) < json.find('\n', pos);
}

int ts_coreml_config_from_gguf(const char * gguf_path,
                               ts_coreml_config * config,
                               std::string * err_msg) {
    if (gguf_path == nullptr || config == nullptr) {
        if (err_msg) *err_msg = "null argument";
        return -1;
    }

    // For v1, we read the GGUF as a binary file and look for tessera
    // metadata keys in the header. A full implementation would use
    // gguf_init_from_file, but that requires libgguf linkage.
    // For now, check if the file exists and return "absent" since
    // the GGUF metadata reading is wired through the dispatch layer.
    std::ifstream f(gguf_path, std::ios::binary);
    if (!f) {
        if (err_msg) *err_msg = std::string("cannot open ") + gguf_path;
        return -1;
    }

    // Check GGUF magic
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        if (err_msg) *err_msg = "not a GGUF file";
        return -1;
    }

    // Full metadata parsing requires libgguf. Return 1 (absent) to
    // signal that the caller should use the dispatch layer's metadata.
    *config = {};
    return 1;
}

int ts_coreml_config_from_sidecar(const char * sidecar_path,
                                  ts_coreml_config * config,
                                  std::string * err_msg) {
    if (sidecar_path == nullptr || config == nullptr) {
        if (err_msg) *err_msg = "null argument";
        return -1;
    }

    std::ifstream f(sidecar_path);
    if (!f) {
        return 1; // not found, not an error
    }

    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    config->n_layers    = ts_json_get_int(json, "n_layers", 0);
    config->hidden_dim  = ts_json_get_int(json, "hidden_dim", 0);
    config->n_heads     = ts_json_get_int(json, "n_heads", 0);
    config->n_kv_heads  = ts_json_get_int(json, "n_kv_heads", 0);
    config->vocab_size  = ts_json_get_int(json, "vocab_size", 0);
    config->quant_type  = (int32_t) ts_json_get_int(json, "quant_type", 43);
    config->has_act_scale = ts_json_get_bool(json, "has_act_scale", false);
    config->calibration_mse = ts_json_get_float(json, "calibration_mse", 0.0f);
    config->calibration_policy = ts_json_get_string(json, "calibration_policy");
    config->model_name  = ts_json_get_string(json, "model_name");

    return 0;
}

int ts_coreml_config_merge(const ts_coreml_config * gguf_config,
                           const ts_coreml_config * sidecar_config,
                           ts_coreml_config * merged,
                           std::vector<std::string> * warnings) {
    if (merged == nullptr) return -1;

    // Start with GGUF as primary
    if (gguf_config) {
        *merged = *gguf_config;
    } else {
        *merged = {};
    }

    // Override with sidecar values where present
    if (sidecar_config) {
        auto warn = [&](const char * field, int64_t gv, int64_t sv) {
            if (gv != 0 && sv != 0 && gv != sv && warnings) {
                warnings->push_back(std::string(field) + ": GGUF=" +
                    std::to_string(gv) + " sidecar=" + std::to_string(sv) +
                    " (using sidecar)");
            }
        };

        if (sidecar_config->n_layers > 0) {
            warn("n_layers", merged->n_layers, sidecar_config->n_layers);
            merged->n_layers = sidecar_config->n_layers;
        }
        if (sidecar_config->hidden_dim > 0) {
            warn("hidden_dim", merged->hidden_dim, sidecar_config->hidden_dim);
            merged->hidden_dim = sidecar_config->hidden_dim;
        }
        if (sidecar_config->n_heads > 0) {
            warn("n_heads", merged->n_heads, sidecar_config->n_heads);
            merged->n_heads = sidecar_config->n_heads;
        }
        if (!sidecar_config->model_name.empty()) {
            merged->model_name = sidecar_config->model_name;
        }
        if (!sidecar_config->calibration_policy.empty()) {
            merged->calibration_policy = sidecar_config->calibration_policy;
        }
        if (sidecar_config->calibration_mse > 0.0f) {
            merged->calibration_mse = sidecar_config->calibration_mse;
        }
        merged->has_act_scale = sidecar_config->has_act_scale || merged->has_act_scale;
    }

    return 0;
}
