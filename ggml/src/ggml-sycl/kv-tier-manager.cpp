#include "kv-tier-manager.hpp"

#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

namespace ggml_sycl {

// Forward-declare to avoid heavyweight unified-cache.hpp include
size_t unified_cache_get_layer_vram_bytes(int device, int layer_id);

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

void kv_tier_manager::set_actual_hot_layers(uint32_t n_hot) {
    if (n_hot > total_layers_) {
        n_hot = total_layers_;
    }
    hot_layers_ = n_hot;
    active_     = (hot_layers_ < total_layers_);
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

void kv_tier_manager::configure_with_weights(int device, uint32_t n_layers,
                                              size_t kv_vram_cap, size_t total_bytes) {
    device_       = device;
    total_layers_ = n_layers;

    if (n_layers == 0 || total_bytes == 0) {
        active_       = false;
        hot_layers_   = 0;
        kv_per_layer_ = 0;
        return;
    }

    kv_per_layer_ = total_bytes / n_layers;

    // Query unified cache for per-layer weight residency
    int device_layers = 0;
    for (uint32_t l = 0; l < n_layers; l++) {
        if (unified_cache_get_layer_vram_bytes(device, static_cast<int>(l)) > 0) {
            device_layers++;
        }
    }

    // Fall back to budget-based configure() when cache has no layer data
    if (device_layers == 0) {
        configure(device, n_layers, kv_vram_cap, total_bytes);
        return;
    }

    // Co-locate KV with device-resident weights: fill hot layers contiguously
    // from layer 0 while weights are on device and VRAM cap permits
    hot_layers_    = 0;
    size_t hot_bytes = 0;
    for (uint32_t l = 0; l < n_layers && hot_bytes + kv_per_layer_ <= kv_vram_cap; l++) {
        if (unified_cache_get_layer_vram_bytes(device, static_cast<int>(l)) > 0) {
            hot_layers_ = l + 1;  // contiguous from layer 0
            hot_bytes += kv_per_layer_;
        }
    }

    // Env var override takes precedence
    const char * hot_layers_env = std::getenv("GGML_SYCL_KV_HOT_LAYERS");
    if (hot_layers_env) {
        int val = std::atoi(hot_layers_env);
        if (val >= 0) {
            hot_layers_ = std::min(static_cast<uint32_t>(val), n_layers);
        }
    }

    active_ = (hot_layers_ < total_layers_);

    GGML_LOG_INFO("[KV-TIER] Weight-aware: %u/%u layers hot (%d with device weights, "
                  "%.1f MB hot, %.1f MB cold)\n",
                  hot_layers_, total_layers_, device_layers,
                  (hot_layers_ * kv_per_layer_) / (1024.0 * 1024.0),
                  ((total_layers_ - hot_layers_) * kv_per_layer_) / (1024.0 * 1024.0));
}

}  // namespace ggml_sycl
