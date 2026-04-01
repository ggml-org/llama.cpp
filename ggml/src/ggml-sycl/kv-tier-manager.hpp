#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Forward-declare GGML_SYCL_MAX_DEVICES from ggml-sycl.h
#ifndef GGML_SYCL_MAX_DEVICES
#    define GGML_SYCL_MAX_DEVICES 48
#endif

namespace ggml_sycl {

// Per-layer region descriptor for KV cache placement.
struct layer_region {
    uint32_t layer_id;
    size_t   offset;     // Offset within device or host region
    size_t   size;       // Bytes for this layer's KV
    bool     on_device;  // true = VRAM, false = host-pinned
};

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

    // Weight-aware configuration: queries unified cache for per-layer weight
    // residency and co-locates KV with device-resident weights.
    // Populates per-layer placement vector for non-contiguous placement.
    // Falls back to budget-based configure() when cache data is unavailable.
    void configure_with_weights(int device, uint32_t n_layers, size_t kv_vram_cap, size_t total_bytes);

    // Query tier state
    bool is_active() const { return active_; }

    uint32_t hot_layers() const { return hot_layers_; }

    uint32_t total_layers() const { return total_layers_; }

    int device_id() const { return device_; }

    size_t kv_per_layer() const { return kv_per_layer_; }

    // Returns true if the given layer should be placed in device VRAM (hot).
    // Supports non-contiguous placement when configure_with_weights() was used.
    bool is_hot(uint32_t layer_id) const;

    // Get hot/cold byte sizes for a given total buffer size.
    // hot_bytes: bytes for device-placed layers, cold_bytes: for host-placed layers.
    // Supports non-contiguous placement (sums device vs host layers).
    void get_region_sizes(size_t total_bytes, size_t & hot_bytes, size_t & cold_bytes) const;

    // Compute per-layer region layout with offsets within device/host regions.
    // Layer sizes are aligned to 512 bytes.
    std::vector<layer_region> compute_region_layout(size_t total_bytes) const;

    // Return the per-layer placement vector.
    // Empty when only budget-based configure() was used.
    const std::vector<bool> & get_layer_placement() const { return layer_on_device_; }

    // Return the number of device-placed layers.
    uint32_t get_device_layer_count() const { return hot_layers_; }

    // Override the hot layer count after allocation.  Called when device
    // allocation fails and the retry loop settles on fewer hot layers than
    // configure() initially computed (e.g. due to VRAM fragmentation or
    // env-var override that exceeds actual capacity).
    void set_actual_hot_layers(uint32_t n_hot);

  private:
    bool                active_       = false;
    int                 device_       = -1;
    uint32_t            hot_layers_   = 0;
    uint32_t            total_layers_ = 0;
    size_t              kv_per_layer_ = 0;
    std::vector<bool>   layer_on_device_;  // Per-layer: true = VRAM, false = host
};

// Per-device singleton accessor
kv_tier_manager & get_kv_tier_manager(int device);

}  // namespace ggml_sycl
