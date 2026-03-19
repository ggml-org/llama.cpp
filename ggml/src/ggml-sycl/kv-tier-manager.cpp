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

bool kv_tier_manager::configure(int device, uint32_t n_layers, size_t kv_vram_cap, size_t total_bytes) {
    device_       = device;
    total_layers_ = n_layers;

    if (n_layers == 0) {
        active_       = false;
        hot_layers_   = 0;
        kv_per_layer_ = 0;
        return false;
    }

    kv_per_layer_ = total_bytes / n_layers;

    // Check env var override: GGML_SYCL_KV_HOT_LAYERS=N
    const char * env = std::getenv("GGML_SYCL_KV_HOT_LAYERS");
    int          val = env ? std::atoi(env) : -1;
    if (val >= 0) {
        hot_layers_ = static_cast<uint32_t>(std::min(val, static_cast<int>(n_layers)));
    } else if (kv_vram_cap == 0 || kv_per_layer_ == 0) {
        hot_layers_ = 0;
    } else {
        hot_layers_ = static_cast<uint32_t>(std::min(static_cast<size_t>(n_layers), kv_vram_cap / kv_per_layer_));
    }

    if (hot_layers_ >= total_layers_) {
        active_ = false;
        return false;
    }

    active_ = true;
    return true;
}

bool kv_tier_manager::is_hot(uint32_t layer_id) const {
    if (!active_) {
        return true;  // All hot when tiering inactive
    }
    return layer_id < hot_layers_;
}

void kv_tier_manager::get_region_sizes(size_t total_bytes, size_t & hot_bytes, size_t & cold_bytes) const {
    if (!active_ || total_layers_ == 0) {
        hot_bytes  = total_bytes;
        cold_bytes = 0;
        return;
    }
    hot_bytes  = std::min(static_cast<size_t>(hot_layers_) * kv_per_layer_, total_bytes);
    cold_bytes = total_bytes - hot_bytes;
}

}  // namespace ggml_sycl
