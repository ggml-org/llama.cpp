//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_DENSE_SCHEDULER_HPP
#define GGML_SYCL_DENSE_SCHEDULER_HPP

#include <cstddef>
#include <functional>
#include <mutex>
#include <sycl/sycl.hpp>

namespace ggml_sycl {

// Double-buffered scheduler for dense layers when VRAM is insufficient.
// Uses two VRAM slots with async prefetch for compute/transfer overlap.
//
// Usage pattern:
//   1. Call get_dense_layer() to get VRAM pointer for current layer
//   2. Call prefetch_next() to start async copy of next layer
//   3. Execute compute on current layer
//   4. Call advance_slot() to swap buffers
//   5. Repeat for next layer
//
// The scheduler overlaps PCIe transfer with GPU computation by maintaining
// two VRAM slots and using a separate copy queue for async transfers.
class dense_layer_scheduler {
  public:
    // Callback to get host pointer for a layer
    using host_ptr_callback = std::function<void *(int layer_id)>;

    // Create scheduler with two VRAM slots of the given size
    dense_layer_scheduler(sycl::queue & compute_queue, size_t max_layer_size);
    ~dense_layer_scheduler();

    // Non-copyable, non-movable
    dense_layer_scheduler(const dense_layer_scheduler &)             = delete;
    dense_layer_scheduler & operator=(const dense_layer_scheduler &) = delete;
    dense_layer_scheduler(dense_layer_scheduler &&)                  = delete;
    dense_layer_scheduler & operator=(dense_layer_scheduler &&)      = delete;

    // Register callback to get host pointer for a layer
    void set_host_ptr_callback(host_ptr_callback cb);

    // Get VRAM pointer for layer (loads from host if needed)
    // Returns nullptr on error
    void * get_dense_layer(int layer_id, size_t size);

    // Prefetch next layer into alternate slot (async)
    void prefetch_next(int next_layer_id, size_t size);

    // Wait for any pending prefetch operations
    void wait_prefetch();

    // Advance to the prefetched slot (call after layer computation completes)
    void advance_slot();

    // Check if scheduler is properly initialized
    bool is_initialized() const { return vram_slot_[0] != nullptr && vram_slot_[1] != nullptr; }

    // Statistics
    size_t slot_size() const { return slot_size_; }

    int current_slot() const { return current_slot_; }

    int layer_in_slot(int slot) const { return (slot >= 0 && slot < 2) ? layer_in_slot_[slot] : -1; }

    size_t prefetch_count() const { return prefetch_count_; }

    size_t sync_load_count() const { return sync_load_count_; }

  private:
    sycl::queue & compute_queue_;
    sycl::queue   copy_queue_;
    int           device_id_ = -1;

    void * vram_slot_[2] = { nullptr, nullptr };
    size_t slot_size_;
    int    current_slot_     = 0;
    int    layer_in_slot_[2] = { -1, -1 };

    sycl::event pending_prefetch_;
    bool        has_pending_prefetch_ = false;

    host_ptr_callback get_host_ptr_;

    size_t prefetch_count_  = 0;
    size_t sync_load_count_ = 0;

    mutable std::mutex mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_DENSE_SCHEDULER_HPP
