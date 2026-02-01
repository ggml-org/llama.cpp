//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_VRAM_POOL_HPP
#define GGML_SYCL_VRAM_POOL_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>

namespace ggml_sycl {

// VRAM pool allocator for device memory.
// Unlike pinned_chunk_pool, allocates per-tensor (no 11GB limit on device).
// Tracks allocations by tensor_id for deallocation.
//
// Thread-safe: all public methods can be called from multiple threads.
class vram_pool {
  public:
    static constexpr size_t DEFAULT_ALIGNMENT = 64;  // Cache line alignment

    // Create a pool with the given VRAM budget
    vram_pool(sycl::queue & queue, size_t budget);
    ~vram_pool();

    // Non-copyable, non-movable (owns SYCL allocations)
    vram_pool(const vram_pool &)             = delete;
    vram_pool & operator=(const vram_pool &) = delete;
    vram_pool(vram_pool &&)                  = delete;
    vram_pool & operator=(vram_pool &&)      = delete;

    // Allocate VRAM for a tensor. Returns nullptr if over budget.
    // tensor_id: unique identifier for later deallocation
    void * allocate(size_t size, uint64_t tensor_id, size_t alignment = DEFAULT_ALIGNMENT);

    // Deallocate VRAM for a tensor
    void deallocate(uint64_t tensor_id);

    // Check if tensor is allocated
    bool is_allocated(uint64_t tensor_id) const;

    // Get pointer for allocated tensor (nullptr if not allocated)
    void * get(uint64_t tensor_id) const;

    // Statistics
    size_t budget() const { return budget_; }

    size_t used() const { return used_; }

    size_t available() const { return budget_ > used_ ? budget_ - used_ : 0; }

    size_t allocation_count() const;

  private:
    struct allocation {
        void * ptr;
        size_t size;
    };

    sycl::queue &                            queue_;
    int                                     device_id_ = -1;
    size_t                                   budget_;
    size_t                                   used_ = 0;
    std::unordered_map<uint64_t, allocation> allocations_;
    mutable std::mutex                       mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_VRAM_POOL_HPP
