#pragma once

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ggml::hrx {

struct RopeFrequencyTable {
    std::vector<uint8_t> data;
    float                mscale = 1.0f;
};

inline float rope_yarn_ramp(float low, float high, int64_t i0) {
    const float y = (static_cast<float>(i0 / 2) - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

inline float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return static_cast<float>(n_dims) * std::log(static_cast<float>(n_ctx_orig) / (n_rot * 2.0f * static_cast<float>(M_PI))) /
           (2.0f * std::log(base));
}

inline void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    const float start = std::floor(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end   = std::ceil(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0]           = std::max(0.0f, start);
    dims[1]           = std::min(static_cast<float>(n_dims - 1), end);
}

inline bool build_rope_frequency_table(const RopeParams & params, int64_t n_dims, RopeFrequencyTable & table) {
    if (n_dims <= 0 || n_dims % 2 != 0 || params.freq_base <= 0.0f || !std::isfinite(params.freq_base) ||
        params.freq_scale <= 0.0f || !std::isfinite(params.freq_scale) || !std::isfinite(params.ext_factor) ||
        !std::isfinite(params.attn_factor)) {
        return false;
    }

    float corr_dims[2] = { 0.0f, 0.0f };
    rope_yarn_corr_dims(static_cast<int>(n_dims), params.n_ctx_orig, params.freq_base, params.beta_fast, params.beta_slow,
                        corr_dims);

    table.mscale = params.attn_factor;
    if (params.ext_factor != 0.0f) {
        table.mscale *= 1.0f + 0.1f * std::log(1.0f / params.freq_scale);
    }

    table.data.resize(static_cast<size_t>(n_dims / 2) * sizeof(float));
    const float theta_scale = std::pow(params.freq_base, -2.0f / static_cast<float>(n_dims));
    float       theta       = 1.0f;
    for (int64_t i = 0; i < n_dims / 2; ++i) {
        const int64_t dim        = 2 * i;
        const float   ramp_mix   = rope_yarn_ramp(corr_dims[0], corr_dims[1], dim) * params.ext_factor;
        const float   frequency  = params.freq_scale * theta * (1.0f - ramp_mix) + theta * ramp_mix;
        std::memcpy(table.data.data() + static_cast<size_t>(i) * sizeof(float), &frequency, sizeof(frequency));
        theta *= theta_scale;
    }
    return true;
}

inline std::string rope_mscale_config_value(float value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.9g", static_cast<double>(value));
    return std::string(buffer);
}

}  // namespace ggml::hrx
