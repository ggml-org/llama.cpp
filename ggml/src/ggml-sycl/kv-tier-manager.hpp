#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

// Forward-declare GGML_SYCL_MAX_DEVICES from ggml-sycl.h
#ifndef GGML_SYCL_MAX_DEVICES
#define GGML_SYCL_MAX_DEVICES 48
#endif

namespace ggml_sycl {

// Manages hot/cold tiering for KV cache memory.
// Hot window: recent tokens in VRAM (fast GPU access)
// Cold tier: older tokens in pinned host memory (PCIe access via USM)
class kv_tier_manager {
public:
    kv_tier_manager() = default;

    // Configure the tier split for a device.
    // hot_bytes: VRAM bytes available for the hot KV region
    // total_bytes: total KV buffer size in bytes
    // kv_bytes_per_token: bytes per KV token (all layers combined)
    // Returns true if tiering is active (hot < total)
    bool configure(int device, size_t hot_bytes, size_t total_bytes, size_t kv_bytes_per_token);

    // Calculate optimal hot window size based on available VRAM.
    // kv_bytes_per_token: bytes per KV entry per token (sum across all layers)
    // Returns suggested hot_tokens count.
    static uint32_t compute_hot_window(int device, size_t kv_bytes_per_token);

    // Query tier state
    bool     is_active()     const { return active_; }
    uint32_t hot_tokens()    const { return hot_tokens_; }
    uint32_t total_tokens()  const { return total_tokens_; }
    int      device_id()     const { return device_; }

    // For the tiered buffer type: determine if a KV position is hot or cold.
    // When tiering is inactive, all positions are considered hot.
    bool is_hot(uint32_t token_pos) const;

    // Get hot/cold byte sizes for a given total buffer size.
    // hot_bytes: bytes for device memory region
    // cold_bytes: bytes for host pinned memory region
    void get_region_sizes(size_t total_bytes, size_t & hot_bytes, size_t & cold_bytes) const;

private:
    bool     active_              = false;
    int      device_              = -1;
    uint32_t hot_tokens_          = 0;
    uint32_t total_tokens_        = 0;
    size_t   kv_bytes_per_token_  = 0;
};

// Per-device singleton accessor
kv_tier_manager & get_kv_tier_manager(int device);

} // namespace ggml_sycl
