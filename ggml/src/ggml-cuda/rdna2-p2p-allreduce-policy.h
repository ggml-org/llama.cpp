#pragma once

#include "../ggml-env-util.h"

#include <cstdint>
#include <cstring>

enum ggml_cuda_rdna2_p2p_host_mode {
    GGML_CUDA_RDNA2_P2P_HOST_OFF           = 0,
    GGML_CUDA_RDNA2_P2P_HOST_SIMPLE        = 1,
    GGML_CUDA_RDNA2_P2P_HOST_FUSED         = 2,
    GGML_CUDA_RDNA2_P2P_HOST_MTP           = 3,
    GGML_CUDA_RDNA2_P2P_HOST_AUTO          = 4,
    GGML_CUDA_RDNA2_P2P_HOST_AUTO_EXPANDED = 5,
};

struct ggml_cuda_rdna2_p2p_host_mode_result {
    int mode;
    bool recognized;
};

inline ggml_cuda_rdna2_p2p_host_mode_result ggml_cuda_rdna2_p2p_host_parse_mode(
        bool rdna2_auto_enabled, const char * value) {
    if (!rdna2_auto_enabled) {
        return { GGML_CUDA_RDNA2_P2P_HOST_OFF, true };
    }
    if (value == nullptr || std::strcmp(value, "auto") == 0 ||
            std::strcmp(value, "auto-expanded") == 0) {
        return { GGML_CUDA_RDNA2_P2P_HOST_AUTO_EXPANDED, true };
    }

    switch (ggml_env_parse_flag(value)) {
        case ggml_env_flag_value::enabled:
            return { GGML_CUDA_RDNA2_P2P_HOST_AUTO_EXPANDED, true };
        case ggml_env_flag_value::disabled:
            return { GGML_CUDA_RDNA2_P2P_HOST_OFF, true };
        case ggml_env_flag_value::unset:
            return { GGML_CUDA_RDNA2_P2P_HOST_AUTO_EXPANDED, true };
        case ggml_env_flag_value::invalid:
            break;
    }

    if (std::strcmp(value, "auto-basic") == 0) {
        return { GGML_CUDA_RDNA2_P2P_HOST_AUTO, true };
    }
    if (std::strcmp(value, "host") == 0) {
        return { GGML_CUDA_RDNA2_P2P_HOST_SIMPLE, true };
    }
    if (std::strcmp(value, "host-fused") == 0) {
        return { GGML_CUDA_RDNA2_P2P_HOST_FUSED, true };
    }
    if (std::strcmp(value, "host-mtp") == 0) {
        return { GGML_CUDA_RDNA2_P2P_HOST_MTP, true };
    }
    return { GGML_CUDA_RDNA2_P2P_HOST_OFF, false };
}

enum class ggml_cuda_rdna2_p2p_host_route {
    fallback,
    qwen4exp_width1,
    ordinary_width1,
    speculative_width5,
    speculative_width6,
};

enum class ggml_cuda_rdna2_p2p_host_fallback_reason {
    none,
    unrelated_shape,
    unsupported_width,
    policy_disabled,
    self_test_failed,
};

struct ggml_cuda_rdna2_p2p_host_route_result {
    ggml_cuda_rdna2_p2p_host_route route;
    ggml_cuda_rdna2_p2p_host_fallback_reason fallback_reason;
};

inline ggml_cuda_rdna2_p2p_host_route_result ggml_cuda_rdna2_p2p_host_select_route(
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        bool speculative_widths_enabled, bool exact_qwen4exp_width1,
        bool exact_width1, bool exact_width5, bool exact_width6) {
    if (ne0 == 2560 && ne1 == 1 && ne2 == 1 && ne3 == 1) {
        if (!exact_qwen4exp_width1) {
            return { ggml_cuda_rdna2_p2p_host_route::fallback,
                     ggml_cuda_rdna2_p2p_host_fallback_reason::self_test_failed };
        }
        return { ggml_cuda_rdna2_p2p_host_route::qwen4exp_width1,
                 ggml_cuda_rdna2_p2p_host_fallback_reason::none };
    }
    if (ne0 != 5120 || ne2 != 1 || ne3 != 1) {
        return { ggml_cuda_rdna2_p2p_host_route::fallback,
                 ggml_cuda_rdna2_p2p_host_fallback_reason::unrelated_shape };
    }
    if (ne1 == 1) {
        if (!exact_width1) {
            return { ggml_cuda_rdna2_p2p_host_route::fallback,
                     ggml_cuda_rdna2_p2p_host_fallback_reason::self_test_failed };
        }
        return { ggml_cuda_rdna2_p2p_host_route::ordinary_width1,
                 ggml_cuda_rdna2_p2p_host_fallback_reason::none };
    }
    if (ne1 == 5) {
        if (!speculative_widths_enabled) {
            return { ggml_cuda_rdna2_p2p_host_route::fallback,
                     ggml_cuda_rdna2_p2p_host_fallback_reason::policy_disabled };
        }
        if (!exact_width5) {
            return { ggml_cuda_rdna2_p2p_host_route::fallback,
                     ggml_cuda_rdna2_p2p_host_fallback_reason::self_test_failed };
        }
        return { ggml_cuda_rdna2_p2p_host_route::speculative_width5,
                 ggml_cuda_rdna2_p2p_host_fallback_reason::none };
    }
    if (ne1 == 6) {
        if (!speculative_widths_enabled) {
            return { ggml_cuda_rdna2_p2p_host_route::fallback,
                     ggml_cuda_rdna2_p2p_host_fallback_reason::policy_disabled };
        }
        if (!exact_width6) {
            return { ggml_cuda_rdna2_p2p_host_route::fallback,
                     ggml_cuda_rdna2_p2p_host_fallback_reason::self_test_failed };
        }
        return { ggml_cuda_rdna2_p2p_host_route::speculative_width6,
                 ggml_cuda_rdna2_p2p_host_fallback_reason::none };
    }
    return { ggml_cuda_rdna2_p2p_host_route::fallback,
             ggml_cuda_rdna2_p2p_host_fallback_reason::unsupported_width };
}

inline uint32_t ggml_cuda_rdna2_p2p_host_width_bit(int64_t width) {
    return width >= 1 && width <= 31 ? 1u << uint32_t(width) : 0;
}
