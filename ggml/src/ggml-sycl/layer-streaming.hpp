//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_LAYER_STREAMING_HPP
#define GGML_SYCL_LAYER_STREAMING_HPP

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"

namespace ggml_sycl {

// Double-buffered layer streaming manager.
// Manages two device buffers that alternate holding transformer layer weights.
// While GPU computes on buffer A (layer N), DMA loads layer N+1 into buffer B.
//
// Usage:
//   1. build_layer_map(inventory, count) — once after tensor inventory is set
//   2. allocate_buffers(queue) — once to create device buffers
//   3. ensure_layer(layer_id) — before each layer's first kernel
//   4. get_weight_device_ptr(name) — to get device pointer for a specific weight
//   5. prefetch_next_layer(layer_id+1) — after layer N's first kernel launches
//   6. shutdown() — cleanup

class layer_stream_manager {
  public:
    layer_stream_manager() = default;
    ~layer_stream_manager();

    // Non-copyable, non-movable (owns device memory)
    layer_stream_manager(const layer_stream_manager &)            = delete;
    layer_stream_manager & operator=(const layer_stream_manager &) = delete;

    // Build the layer weight map from tensor inventory.
    // inventory: array of (name, size) pairs
    // count: number of entries
    // Filters to only include "blk.N.*" tensors (layer tensors).
    void build_layer_map(const std::pair<std::string, size_t> * inventory, size_t count);

    // Allocate two device buffers sized to max_layer_size.
    // Returns false if allocation fails.
    // queue: the SYCL queue for device allocation.
    bool allocate_buffers(sycl::queue & queue);

    // Free device buffers and reset state.
    void shutdown();

    // Synchronously ensure a layer's weights are loaded into a device buffer.
    // If the layer is already loaded (from a previous prefetch or ensure call), this is a no-op.
    // If not loaded, performs synchronous DMA from host pointers to the buffer.
    // host_ptrs: map from tensor name to host pointer (from tensor->data or host cache).
    // Returns true if the layer is now loaded.
    bool ensure_layer(int layer_id, sycl::queue & queue);

    // Start async DMA of a layer into the alternate buffer.
    // Non-blocking — the DMA runs on copy_queue (if provided) or main queue.
    // Call await_prefetch() before accessing the layer's data.
    void prefetch_next_layer(int layer_id, sycl::queue & queue);

    // Block until the most recent async prefetch completes.
    void await_prefetch();

    // Get the device pointer for a specific weight tensor.
    // Returns nullptr if the weight's layer is not currently loaded.
    // The pointer is valid until the layer is evicted (overwritten by a new layer load).
    void * get_weight_device_ptr(const char * tensor_name) const;

    // Check if a layer is currently loaded in a buffer.
    bool is_layer_loaded(int layer_id) const;

    // Get which buffer index (0 or 1) a layer is in. Returns -1 if not loaded.
    int buffer_for_layer(int layer_id) const;

    // Check if the manager is active (has buffers allocated).
    bool is_active() const { return buffers_[0] != nullptr; }

    // Total bytes allocated for the two buffers.
    size_t allocated_bytes() const { return buffer_size_ * 2; }

    // Max layer size across all layers.
    size_t max_layer_size() const { return max_layer_size_; }

    // Number of layers in the model.
    int n_layers() const { return static_cast<int>(layers_.size()); }

    // Register the host pointer for a tensor (called during model load/cache setup).
    // This is needed because the tensor inventory only stores names and sizes,
    // not the actual host pointers. The pointers are registered as tensors are
    // encountered during inference.
    void register_host_ptr(const char * tensor_name, const void * host_ptr, size_t size);

  private:
    // Per-weight entry within a layer
    struct weight_entry {
        std::string  name;
        size_t       size            = 0;
        size_t       offset_in_layer = 0;  // Byte offset within layer buffer
        const void * host_ptr        = nullptr;  // Host-resident data pointer (registered later)
    };

    // Per-layer info
    struct layer_info {
        std::vector<weight_entry> weights;
        size_t                    total_size = 0;
    };

    // Layer data
    std::vector<layer_info>                               layers_;         // Indexed by layer_id
    size_t                                                max_layer_size_ = 0;
    std::unordered_map<std::string, std::pair<int, size_t>> name_to_location_;  // name -> (layer_id, weight_idx)

    // Double buffers
    void *  buffers_[2]       = {nullptr, nullptr};
    size_t  buffer_size_      = 0;
    int     loaded_layers_[2] = {-1, -1};  // Which layer is in each buffer (-1 = empty)

    // Async prefetch state
    int         prefetch_target_layer_ = -1;
    int         prefetch_buffer_       = -1;
    sycl::event prefetch_event_;
    bool        prefetch_pending_      = false;
    mutable std::mutex prefetch_mutex_;

    // Host pointer registration
    mutable std::mutex host_ptr_mutex_;

    // Internal helpers
    int  pick_buffer_for_layer(int layer_id) const;
    bool load_layer_sync(int layer_id, int buffer_idx, sycl::queue & queue);
};

// Global accessor — returns the singleton layer stream manager for a device.
// Creates on first access. Thread-safe.
layer_stream_manager & get_layer_stream_manager(int device_id);

// Free function API for use from ggml-sycl.cpp
bool layer_streaming_active(int device_id);
void * layer_streaming_get_weight_ptr(int device_id, const char * name);
void layer_streaming_ensure_layer(int device_id, int layer_id, sycl::queue & queue);
void layer_streaming_prefetch_next(int device_id, int layer_id, sycl::queue & queue);
void layer_streaming_await_prefetch(int device_id);
void layer_streaming_register_host_ptr(int device_id, const char * name, const void * ptr, size_t size);

}  // namespace ggml_sycl

#endif  // GGML_SYCL_LAYER_STREAMING_HPP
