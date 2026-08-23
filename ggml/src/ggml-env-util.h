#pragma once

#include <cstring>

enum class ggml_env_flag_value {
    unset,
    enabled,
    disabled,
    invalid,
};

// Parse the conventional, case-sensitive boolean spellings used by GGML
// feature flags. Callers choose the default and invalid-value policy.
inline ggml_env_flag_value ggml_env_parse_flag(const char * value) {
    if (value == nullptr) {
        return ggml_env_flag_value::unset;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0 ||
            std::strcmp(value, "true") == 0 || std::strcmp(value, "yes") == 0) {
        return ggml_env_flag_value::enabled;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0 ||
            std::strcmp(value, "false") == 0 || std::strcmp(value, "no") == 0) {
        return ggml_env_flag_value::disabled;
    }
    return ggml_env_flag_value::invalid;
}
