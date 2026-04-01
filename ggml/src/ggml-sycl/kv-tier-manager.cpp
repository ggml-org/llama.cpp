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
    if (layer_id < layer_on_device_.size()) {
        return layer_on_device_[layer_id];
    }
    // Fallback for budget-only configure() path (no per-layer vector)
    return layer_id < hot_layers_;
}

void kv_tier_manager::set_actual_hot_layers(uint32_t n_hot) {
    if (n_hot > total_layers_) {
        n_hot = total_layers_;
    }
    hot_layers_ = n_hot;
    active_     = (hot_layers_ < total_layers_);

    // Sync per-layer vector if it was populated by configure_with_weights().
    // The allocation retry loop may have reduced the device budget, so trim
    // device-placed layers from the back until the count matches n_hot.
    if (!layer_on_device_.empty()) {
        uint32_t current_on_device = 0;
        for (bool v : layer_on_device_) {
            if (v) current_on_device++;
        }
        // Remove device-placed layers from the back until count matches
        for (uint32_t l = total_layers_; l > 0 && current_on_device > n_hot; l--) {
            if (layer_on_device_[l - 1]) {
                layer_on_device_[l - 1] = false;
                current_on_device--;
            }
        }
    }
}

void kv_tier_manager::get_region_sizes(size_t total_bytes, size_t & hot_bytes, size_t & cold_bytes) const {
    if (!active_ || total_layers_ == 0) {
        hot_bytes  = total_bytes;
        cold_bytes = 0;
        return;
    }
    // Use per-layer vector when available (non-contiguous support)
    if (!layer_on_device_.empty()) {
        uint32_t device_count = 0;
        for (bool v : layer_on_device_) {
            if (v) device_count++;
        }
        hot_bytes  = std::min(static_cast<size_t>(device_count) * kv_per_layer_, total_bytes);
        cold_bytes = total_bytes - hot_bytes;
        return;
    }
    // Fallback: contiguous hot_layers_ count
    hot_bytes  = std::min(static_cast<size_t>(hot_layers_) * kv_per_layer_, total_bytes);
    cold_bytes = total_bytes - hot_bytes;
}

std::vector<layer_region> kv_tier_manager::compute_region_layout(size_t total_bytes) const {
    GGML_UNUSED(total_bytes);
    const size_t aligned_per_layer = (kv_per_layer_ + 511) & ~size_t(511);

    std::vector<layer_region> regions(total_layers_);
    size_t device_offset = 0;
    size_t host_offset   = 0;
    for (uint32_t l = 0; l < total_layers_; l++) {
        regions[l].layer_id  = l;
        regions[l].size      = aligned_per_layer;
        regions[l].on_device = is_hot(l);
        if (regions[l].on_device) {
            regions[l].offset = device_offset;
            device_offset += aligned_per_layer;
        } else {
            regions[l].offset = host_offset;
            host_offset += aligned_per_layer;
        }
    }
    return regions;
}

void kv_tier_manager::configure_with_weights(int device, uint32_t n_layers,
                                              size_t kv_vram_cap, size_t total_bytes) {
    device_       = device;
    total_layers_ = n_layers;

    if (n_layers == 0 || total_bytes == 0) {
        active_       = false;
        hot_layers_   = 0;
        kv_per_layer_ = 0;
        layer_on_device_.clear();
        return;
    }

    kv_per_layer_ = total_bytes / n_layers;

    // PHASE 1: Query unified cache for per-layer weight residency
    std::vector<bool> layer_has_device_weights(n_layers, false);
    int               device_weight_count = 0;
    for (uint32_t l = 0; l < n_layers; l++) {
        if (unified_cache_get_layer_vram_bytes(device, static_cast<int>(l)) > 0) {
            layer_has_device_weights[l] = true;
            device_weight_count++;
        }
    }

    // Fall back to budget-based configure() when cache has no layer data.
    // Budget-only path uses contiguous hot_layers_ without per-layer vector.
    if (device_weight_count == 0) {
        layer_on_device_.clear();
        configure(device, n_layers, kv_vram_cap, total_bytes);
        return;
    }

    // PHASE 2: Check env var override — takes precedence over weight residency
    const char * hot_layers_env = std::getenv("GGML_SYCL_KV_HOT_LAYERS");
    if (hot_layers_env) {
        int val = std::atoi(hot_layers_env);
        if (val >= 0) {
            // Env override: use contiguous placement (layers 0..N-1)
            hot_layers_ = std::min(static_cast<uint32_t>(val), n_layers);
            layer_on_device_.assign(n_layers, false);
            for (uint32_t l = 0; l < hot_layers_; l++) {
                layer_on_device_[l] = true;
            }
            active_ = (hot_layers_ < total_layers_);

            GGML_LOG_INFO("[KV-TIER] Weight-aware (env override): %u/%u layers on device "
                          "(%.1f MB device, %.1f MB host)\n",
                          hot_layers_, total_layers_,
                          (hot_layers_ * kv_per_layer_) / (1024.0 * 1024.0),
                          ((total_layers_ - hot_layers_) * kv_per_layer_) / (1024.0 * 1024.0));
            return;
        }
    }

    // PHASE 3: Per-layer placement — co-locate KV with device-resident weights
    // within the VRAM budget.  Any layer with device weights gets placed on
    // device if the budget allows; this supports non-contiguous placement.
    layer_on_device_.assign(n_layers, false);
    size_t vram_used = 0;
    for (uint32_t l = 0; l < n_layers; l++) {
        if (layer_has_device_weights[l] && vram_used + kv_per_layer_ <= kv_vram_cap) {
            layer_on_device_[l] = true;
            vram_used += kv_per_layer_;
        }
    }

    // Update hot_layers_ for backward-compatible logging and fallback paths
    hot_layers_ = 0;
    for (bool on_dev : layer_on_device_) {
        if (on_dev) hot_layers_++;
    }

    active_ = (hot_layers_ < total_layers_);

    GGML_LOG_INFO("[KV-TIER] Weight-aware: %u/%u layers on device (%d with device weights, "
                  "%.1f MB device, %.1f MB host)\n",
                  hot_layers_, total_layers_, device_weight_count,
                  (hot_layers_ * kv_per_layer_) / (1024.0 * 1024.0),
                  ((total_layers_ - hot_layers_) * kv_per_layer_) / (1024.0 * 1024.0));
}

}  // namespace ggml_sycl
