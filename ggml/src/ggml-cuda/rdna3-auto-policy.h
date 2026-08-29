#pragma once

#include "../ggml-env-util.h"

#include <cstdlib>

enum class ggml_cuda_rdna3_auto_flag {
    disabled,
    enabled,
    invalid,
};

// The RDNA3 umbrella is deliberately opt-in. An unset value leaves existing
// runtime behavior unchanged; invalid values fail closed and are diagnosed by
// the HIP backend during device initialization.
inline ggml_cuda_rdna3_auto_flag ggml_cuda_rdna3_auto_parse(const char * value) {
    switch (ggml_env_parse_flag(value)) {
        case ggml_env_flag_value::enabled:
            return ggml_cuda_rdna3_auto_flag::enabled;
        case ggml_env_flag_value::disabled:
        case ggml_env_flag_value::unset:
            return ggml_cuda_rdna3_auto_flag::disabled;
        case ggml_env_flag_value::invalid:
            return ggml_cuda_rdna3_auto_flag::invalid;
    }
    return ggml_cuda_rdna3_auto_flag::invalid;
}

inline bool ggml_cuda_rdna3_auto_enabled() {
    return ggml_cuda_rdna3_auto_parse(std::getenv("GGML_HIP_RDNA3_AUTO")) ==
        ggml_cuda_rdna3_auto_flag::enabled;
}

// The automatic profile may use every visible physical GPU, but must never
// turn on for a virtual-device configuration or a single-device launch.
inline bool ggml_cuda_rdna3_auto_counts_qualified(int physical_device_count, int device_count) {
    return physical_device_count >= 2 && device_count == physical_device_count;
}
