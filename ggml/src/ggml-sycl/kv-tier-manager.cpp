#include "kv-tier-manager.hpp"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cstdlib>

namespace ggml_sycl {

static std::array<kv_tier_manager, GGML_SYCL_MAX_DEVICES> g_kv_tier_managers;

kv_tier_manager & get_kv_tier_manager(int device) {
    return g_kv_tier_managers[device];
}

bool kv_tier_manager::configure(int device, size_t hot_bytes, size_t total_bytes, size_t kv_bytes_per_token) {
    device_             = device;
    kv_bytes_per_token_ = kv_bytes_per_token;

    // Convert byte counts to token counts
    total_tokens_ = (kv_bytes_per_token > 0)
                      ? static_cast<uint32_t>(std::min(total_bytes / kv_bytes_per_token,
                                                       static_cast<size_t>(UINT32_MAX)))
                      : static_cast<uint32_t>(std::min(total_bytes, static_cast<size_t>(UINT32_MAX)));

    // Check env var override: GGML_SYCL_KV_HOT_TOKENS=N (token count)
    const char * env = std::getenv("GGML_SYCL_KV_HOT_TOKENS");
    int val = env ? std::atoi(env) : 0;
    if (val > 0) {
        hot_tokens_ = static_cast<uint32_t>(val);
    } else {
        hot_tokens_ = (kv_bytes_per_token > 0)
                        ? static_cast<uint32_t>(hot_bytes / kv_bytes_per_token)
                        : static_cast<uint32_t>(std::min(hot_bytes, static_cast<size_t>(UINT32_MAX)));
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
    // Convert hot token count back to bytes
    if (kv_bytes_per_token_ > 0) {
        hot_bytes = std::min(static_cast<size_t>(hot_tokens_) * kv_bytes_per_token_, total_bytes);
    } else {
        // Fallback: proportional split
        double hot_fraction = static_cast<double>(hot_tokens_) / total_tokens_;
        hot_bytes = static_cast<size_t>(total_bytes * hot_fraction);
        if (hot_bytes > total_bytes) {
            hot_bytes = total_bytes;
        }
    }
    cold_bytes = total_bytes - hot_bytes;
}

} // namespace ggml_sycl
