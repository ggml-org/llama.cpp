#pragma once

#include <cstddef>
#include <cstdint>

// Forward-declare GGML_SYCL_MAX_DEVICES from ggml-sycl.h
#ifndef GGML_SYCL_MAX_DEVICES
#    define GGML_SYCL_MAX_DEVICES 48
#endif

namespace ggml_sycl {

// Manages hot/cold tiering for KV cache memory on a per-layer basis.
// Hot layers: KV on device VRAM (fast GPU access, co-located with attention weights)
// Cold layers: KV on pinned host memory (PCIe access via USM)
class kv_tier_manager {
  public:
    kv_tier_manager() = default;

    // Configure the layer-based tier split for a device.
    // n_layers: total number of transformer layers
    // kv_vram_cap: VRAM bytes available for KV cache
    // total_bytes: total KV buffer size in bytes
    // Returns true if tiering is active (some layers on host)
    bool configure(int device, uint32_t n_layers, size_t kv_vram_cap, size_t total_bytes);

    // Query tier state
    bool is_active() const { return active_; }

    uint32_t hot_layers() const { return hot_layers_; }

    uint32_t total_layers() const { return total_layers_; }

    int device_id() const { return device_; }

    size_t kv_per_layer() const { return kv_per_layer_; }

    // Returns true if the given layer should be placed in device VRAM (hot).
    bool is_hot(uint32_t layer_id) const;

    // Get hot/cold byte sizes for a given total buffer size.
    // hot_bytes: bytes for device memory region (first N layers)
    // cold_bytes: bytes for host pinned memory region (remaining layers)
    void get_region_sizes(size_t total_bytes, size_t & hot_bytes, size_t & cold_bytes) const;

    // Override the hot layer count after allocation.  Called when device
    // allocation fails and the retry loop settles on fewer hot layers than
    // configure() initially computed (e.g. due to VRAM fragmentation or
    // env-var override that exceeds actual capacity).
    void set_actual_hot_layers(uint32_t n_hot);

  private:
    bool     active_       = false;
    int      device_       = -1;
    uint32_t hot_layers_   = 0;
    uint32_t total_layers_ = 0;
    size_t   kv_per_layer_ = 0;
};

// Per-device singleton accessor
kv_tier_manager & get_kv_tier_manager(int device);

}  // namespace ggml_sycl
