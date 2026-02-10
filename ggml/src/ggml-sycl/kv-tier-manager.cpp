#include "kv-tier-manager.hpp"
#include "unified-cache.hpp"
#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdlib>

namespace ggml_sycl {

static std::array<kv_tier_manager, GGML_SYCL_MAX_DEVICES> g_kv_tier_managers;

kv_tier_manager & get_kv_tier_manager(int device) {
    return g_kv_tier_managers[device];
}

bool kv_tier_manager::configure(int device, uint32_t hot_tokens, uint32_t total_tokens) {
    device_       = device;
    total_tokens_ = total_tokens;

    // Check env var override: GGML_SYCL_KV_HOT_TOKENS=N
    const char * env = std::getenv("GGML_SYCL_KV_HOT_TOKENS");
    if (env) {
        int val = std::atoi(env);
        if (val > 0) {
            hot_tokens_ = static_cast<uint32_t>(val);
        } else {
            hot_tokens_ = hot_tokens;
        }
    } else {
        hot_tokens_ = hot_tokens;
    }

    // Minimum hot window: 1024 tokens
    hot_tokens_ = std::max(hot_tokens_, uint32_t(1024));

    // If hot window >= total, no tiering needed — everything fits in VRAM
    if (hot_tokens_ >= total_tokens_) {
        active_ = false;
        return false;
    }

    active_ = true;
    GGML_LOG_INFO("[KV-TIER] Device %d: hot=%u tokens (VRAM), cold=%u tokens (host pinned), total=%u\n",
                  device_, hot_tokens_, total_tokens_ - hot_tokens_, total_tokens_);
    return true;
}

uint32_t kv_tier_manager::compute_hot_window(int device, size_t kv_bytes_per_token) {
    if (kv_bytes_per_token == 0) {
        return UINT32_MAX;
    }

    auto info = unified_cache_get_budget_info(device);

    // Available VRAM after weights
    size_t available = info.available_for_weights > info.weight_bytes
                         ? info.available_for_weights - info.weight_bytes : 0;

    // Reserve VRAM for compute scratch (256 MB or 10% of budget, whichever is larger)
    size_t compute_reserve = std::max(size_t(256) << 20, info.budget_bytes / 10);
    if (available > compute_reserve) {
        available -= compute_reserve;
    } else {
        available = 0;
    }

    return static_cast<uint32_t>(available / kv_bytes_per_token);
}

bool kv_tier_manager::is_hot(uint32_t token_pos) const {
    if (!active_) {
        return true;  // All hot when tiering inactive
    }
    return token_pos < hot_tokens_;
}

void kv_tier_manager::get_region_sizes(size_t total_bytes, size_t & hot_bytes, size_t & cold_bytes) const {
    if (!active_ || total_tokens_ == 0) {
        hot_bytes  = total_bytes;
        cold_bytes = 0;
        return;
    }
    // Proportional split based on token counts
    double hot_fraction = static_cast<double>(hot_tokens_) / total_tokens_;
    hot_bytes  = static_cast<size_t>(total_bytes * hot_fraction);
    // Align hot_bytes to 512 bytes (device allocation alignment)
    hot_bytes  = (hot_bytes + 511) & ~size_t(511);
    if (hot_bytes > total_bytes) {
        hot_bytes = total_bytes;
    }
    cold_bytes = total_bytes - hot_bytes;
}

} // namespace ggml_sycl
