//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_PINNED_POOL_HPP
#define GGML_SYCL_PINNED_POOL_HPP

#include <cstddef>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace ggml_sycl {

// Pool allocator for pinned host memory using multiple chunks.
// Bypasses Intel Level Zero's ~11GB per-allocation limit by using
// multiple 8GB malloc_host allocations.
// Optional: set GGML_SYCL_PINNED_CHUNK_MB to override the default chunk size.
// Optional: set GGML_SYCL_PINNED_ALLOC_TIMEOUT_MS to abort if a chunk allocation hangs.
//
// Uses bump allocation with 64-byte alignment for cache line efficiency.
// Chunks are only released when the pool is destroyed.
class pinned_chunk_pool {
  public:
    static constexpr size_t CHUNK_SIZE        = 8ULL * 1024 * 1024 * 1024;  // 8GB default chunk
    static constexpr size_t DEFAULT_ALIGNMENT = 64;                         // Cache line alignment

    // Create a pool with the given budget (maximum total memory to allocate)
    pinned_chunk_pool(sycl::queue & queue, size_t budget);
    ~pinned_chunk_pool();

    // Non-copyable, non-movable (owns SYCL allocations)
    pinned_chunk_pool(const pinned_chunk_pool &)             = delete;
    pinned_chunk_pool & operator=(const pinned_chunk_pool &) = delete;
    pinned_chunk_pool(pinned_chunk_pool &&)                  = delete;
    pinned_chunk_pool & operator=(pinned_chunk_pool &&)      = delete;

    // Allocate from pool. Returns nullptr if over budget or allocation fails.
    // All allocations are aligned to DEFAULT_ALIGNMENT (64 bytes).
    void * allocate(size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Mark allocation as free. Note: bump allocator - individual deallocations
    // are tracked but memory is only reclaimed when the pool is destroyed.
    void deallocate(void * ptr, size_t size);

    // Statistics
    size_t budget() const { return budget_; }

    size_t allocated() const;  // Total bytes allocated (in chunks)
    size_t chunk_count() const;

  private:
    struct chunk {
        void * base;   // malloc_host result
        size_t size;   // CHUNK_SIZE
        size_t used;   // Bump pointer offset
        size_t freed;  // Bytes deallocated (for tracking, not reclaimed)
    };

    // Allocate a new chunk (>= min_size). Returns false if over budget or allocation fails.
    bool grow(size_t min_size);

    sycl::queue &      queue_;
    size_t             budget_;
    size_t             total_allocated_ = 0;
    size_t             chunk_size_      = CHUNK_SIZE;
    size_t             alloc_timeout_ms_ = 0;
    std::vector<chunk> chunks_;
    mutable std::mutex mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_PINNED_POOL_HPP
