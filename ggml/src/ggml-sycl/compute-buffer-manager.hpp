//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Compute Buffer Manager for SYCL unified memory management
// Part of unified memory system (epic llama.cpp-v3n, task llama.cpp-6s5)
//
// Manages compute buffers (activations, intermediate results) with P0 priority.
// P0 buffers are NEVER evicted - they must remain in VRAM for correct operation.
//
// Features:
// - Pool-based allocation: Reuses buffers to avoid expensive SYCL mallocs
// - P0 priority: All buffers are pinned (never evicted)
// - Scratch space: Resizable buffer for temporary operations
// - Thread-safe: Concurrent allocations are safe
//

#ifndef GGML_SYCL_COMPUTE_BUFFER_MANAGER_HPP
#define GGML_SYCL_COMPUTE_BUFFER_MANAGER_HPP

#include "eviction-policy.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace ggml_sycl {

// Information about a pooled compute buffer
struct ComputeBuffer {
    void *       ptr        = nullptr;  // GPU pointer
    size_t       size       = 0;        // Buffer size in bytes
    bool         in_use     = false;    // Currently allocated to caller
    uint64_t     alloc_time = 0;        // Timestamp of allocation (for debugging)
    const char * name       = nullptr;  // Current user name (for debugging)
    alloc_handle handle{};
};

// Compute Buffer Manager
//
// Manages compute buffers with P0 (CRITICAL) priority, meaning they are
// NEVER evicted. All buffers are pinned automatically.
//
// Usage:
//   ComputeBufferManager mgr(queue);
//   void* buf = mgr.allocate(size, "my_buffer");
//   // use buf...
//   mgr.release(buf);
//
class ComputeBufferManager {
  public:
    // Initialize with SYCL queue
    explicit ComputeBufferManager(sycl::queue & queue);
    ~ComputeBufferManager();

    // Non-copyable, non-movable
    ComputeBufferManager(const ComputeBufferManager &)             = delete;
    ComputeBufferManager & operator=(const ComputeBufferManager &) = delete;
    ComputeBufferManager(ComputeBufferManager &&)                  = delete;
    ComputeBufferManager & operator=(ComputeBufferManager &&)      = delete;

    // === Allocation API ===

    // Allocate a compute buffer from the pool (P0 priority)
    // Returns nullptr if allocation fails (OOM)
    // name is optional, for debugging purposes
    void * allocate(size_t size, const char * name = nullptr);

    // Release a buffer back to the pool
    // Does NOT free the memory - it returns to the pool for reuse
    void release(void * ptr);

    // === Scratch Space API ===

    // Get scratch buffer of at least min_size bytes
    // Scratch grows as needed but never shrinks
    void * get_scratch(size_t min_size);

    // Current scratch allocation size (last requested size)
    size_t get_scratch_size() const { return scratch_size_; }

    // Current scratch capacity (actual allocated size)
    size_t get_scratch_capacity() const { return scratch_capacity_; }

    // Whether scratch buffer is pinned (should always be true)
    bool is_scratch_pinned() const { return scratch_ptr_ != nullptr; }

    // === Buffer Properties ===

    // Get priority of a buffer (should always be P0_COMPUTE)
    EvictionPriority get_priority(void * ptr) const;

    // Check if a buffer is pinned (should always be true for compute buffers)
    bool is_pinned(void * ptr) const;

    // Check if a pointer is a valid managed buffer
    bool is_valid(void * ptr) const;

    // Attempt to evict buffers to free space (always fails for P0 buffers)
    // Returns true if any space was freed (always false for compute manager)
    bool try_evict_for_space(size_t bytes_needed);

    // === Memory Accounting ===

    // Total size of all buffers in pool (including released ones)
    size_t pool_total_size() const;

    // Size of buffers currently in use
    size_t pool_used_size() const;

    // === Statistics ===

    // Total number of allocate() calls
    size_t num_allocations() const { return total_allocations_.load(); }

    // Number of allocations satisfied from pool (reuse)
    size_t num_pool_hits() const { return pool_hits_.load(); }

    // Number of allocations that required new SYCL malloc
    size_t num_pool_misses() const { return pool_misses_.load(); }

    // Print statistics to stdout
    void print_stats() const;

  private:
    // Find a free buffer in pool that can satisfy size request
    // Returns nullptr if no suitable buffer found
    ComputeBuffer * find_free_buffer(size_t size);

    // Allocate a new SYCL buffer
    // Returns nullptr on failure
    void * allocate_new_buffer(size_t size);

    // Grow scratch buffer to new_size
    void grow_scratch(size_t new_size);

    // Get current timestamp (monotonic counter)
    uint64_t current_time();

    sycl::queue & queue_;

    // Buffer pool
    std::vector<ComputeBuffer> pool_;
    mutable std::mutex         pool_mutex_;

    // Scratch space
    void *             scratch_ptr_      = nullptr;
    size_t             scratch_size_     = 0;  // Last requested size
    size_t             scratch_capacity_ = 0;  // Actual allocated size
    alloc_handle       scratch_handle_{};
    mutable std::mutex scratch_mutex_;

    // Monotonic time counter
    std::atomic<uint64_t> time_counter_{ 0 };
    int                   device_id_ = 0;

    // Statistics
    std::atomic<size_t> total_allocations_{ 0 };
    std::atomic<size_t> pool_hits_{ 0 };
    std::atomic<size_t> pool_misses_{ 0 };
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_COMPUTE_BUFFER_MANAGER_HPP
