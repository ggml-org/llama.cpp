//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Bump-allocator pool for SYCL device memory.
// Pre-allocates large chunks and sub-allocates from them to reduce the number
// of individual sycl::malloc_device() calls. This consolidates hundreds of
// small USM allocations into a few large contiguous regions, improving GPU TLB
// hit rates and reducing virtual address space fragmentation.
//
// The pool does not support individual frees. All memory is released when the
// pool is destroyed or reset(). This is appropriate for weight tensors and
// layout buffers that live for the model's lifetime.

#ifndef GGML_SYCL_DEVICE_POOL_HPP
#define GGML_SYCL_DEVICE_POOL_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml-impl.h"
#include "alloc-registry.hpp"

// Forward-declare to avoid circular include with common.hpp
void * ggml_sycl_malloc_device(size_t size, const sycl::queue & queue, const char * tag);

namespace ggml_sycl {

class sycl_device_pool {
  public:
    // chunk_size: default size for each large allocation (256 MB).
    // Requests larger than chunk_size get their own dedicated chunk.
    sycl_device_pool(sycl::queue & queue, size_t chunk_size = 256 * 1024 * 1024)
        : queue_(queue), default_chunk_size_(chunk_size) {}

    ~sycl_device_pool() {
        if (!abandoned_) {
            reset();
        }
    }

    // Non-copyable, non-movable
    sycl_device_pool(const sycl_device_pool &)             = delete;
    sycl_device_pool & operator=(const sycl_device_pool &) = delete;
    sycl_device_pool(sycl_device_pool &&)                  = delete;
    sycl_device_pool & operator=(sycl_device_pool &&)      = delete;

    // Result of a pool allocation: pointer + how much NEW physical memory was consumed.
    struct alloc_result {
        void * ptr                = nullptr;
        size_t new_physical_bytes = 0;  // >0 only if a new chunk was allocated
    };

    // Allocate size bytes with the given alignment (must be power of 2).
    // Returns {nullptr, 0} if the underlying sycl::malloc_device fails.
    // new_physical_bytes tells the caller how much additional device memory was consumed
    // (equals chunk_size when a new chunk is needed, 0 for sub-allocations within existing chunks).
    // Thread-safe.
    alloc_result allocate(size_t size, size_t align = 256) {
        if (size == 0) {
            return {};
        }

        // Validate alignment is power of 2
        if (align == 0 || (align & (align - 1)) != 0) {
            align = 256;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Try to sub-allocate from existing chunks (most recent first)
        for (auto it = chunks_.rbegin(); it != chunks_.rend(); ++it) {
            void * ptr = try_suballoc(*it, size, align);
            if (ptr) {
                alloc_count_++;
                return { ptr, 0 };
            }
        }

        // Need a new chunk
        size_t chunk_size = default_chunk_size_;
        // For oversized requests, allocate a chunk that exactly fits
        size_t padded = size + align;  // worst-case alignment padding
        if (padded > chunk_size) {
            chunk_size = padded;
        }

        void * base = ggml_sycl_malloc_device(chunk_size, queue_, "layout_pool:chunk");
        if (!base) {
            GGML_LOG_ERROR("[DEVICE-POOL] chunk alloc failed (%zu bytes)\n", chunk_size);
            return {};
        }

        chunks_.push_back({ base, chunk_size, 0 });
        chunk_bytes_ += chunk_size;

        GGML_LOG_INFO("[DEVICE-POOL] new chunk #%zu: %.1f MB (total %.1f MB in %zu chunks)\n",
                      chunks_.size(), chunk_size / (1024.0 * 1024.0),
                      chunk_bytes_ / (1024.0 * 1024.0), chunks_.size());

        void * ptr = try_suballoc(chunks_.back(), size, align);
        if (ptr) {
            alloc_count_++;
        }
        return { ptr, chunk_size };
    }

    // Check if a pointer was allocated from this pool.
    // Thread-safe.
    bool owns(const void * ptr) const {
        if (!ptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto p = reinterpret_cast<uintptr_t>(ptr);
        for (const auto & c : chunks_) {
            const auto base = reinterpret_cast<uintptr_t>(c.base);
            if (p >= base && p < base + c.size) {
                return true;
            }
        }
        return false;
    }

    // Free all chunks. After this, all pointers returned by allocate() are invalid.
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto & c : chunks_) {
            if (c.base) {
                try {
                    ggml_sycl::alloc_registry::instance().unregister_alloc(c.base);
                    sycl::free(c.base, queue_);
                } catch (...) {
                }
            }
        }
        chunks_.clear();
        chunk_bytes_ = 0;
        alloc_count_ = 0;
    }

    // Abandon all chunks without freeing (for use during SYCL shutdown when
    // the runtime context may already be torn down).  The memory is leaked
    // intentionally — the process is exiting.
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        abandoned_ = true;
        chunks_.clear();
        chunk_bytes_ = 0;
        alloc_count_ = 0;
    }

    // Check if the requested size can be sub-allocated from an existing chunk
    // without requiring a new chunk allocation. Thread-safe.
    bool can_fit(size_t size, size_t align = 256) const {
        if (size == 0) {
            return true;
        }
        if (align == 0 || (align & (align - 1)) != 0) {
            align = 256;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = chunks_.rbegin(); it != chunks_.rend(); ++it) {
            size_t aligned_offset = (it->used + align - 1) & ~(align - 1);
            if (aligned_offset + size <= it->size) {
                return true;
            }
        }
        return false;
    }

    // Get the default chunk size (used for budget pre-checks).
    size_t get_default_chunk_size() const { return default_chunk_size_; }

    // Statistics
    size_t chunk_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return chunks_.size();
    }
    size_t total_chunk_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return chunk_bytes_;
    }
    size_t total_used_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t used = 0;
        for (const auto & c : chunks_) {
            used += c.used;
        }
        return used;
    }
    int alloc_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return alloc_count_;
    }

  private:
    struct chunk {
        void * base = nullptr;
        size_t size = 0;
        size_t used = 0;  // bump pointer offset
    };

    // Try to sub-allocate from a chunk. Returns nullptr if not enough space.
    static void * try_suballoc(chunk & c, size_t size, size_t align) {
        size_t aligned_offset = (c.used + align - 1) & ~(align - 1);
        if (aligned_offset + size > c.size) {
            return nullptr;
        }
        void * ptr = static_cast<char *>(c.base) + aligned_offset;
        c.used     = aligned_offset + size;
        return ptr;
    }

    sycl::queue &        queue_;
    size_t               default_chunk_size_;
    std::vector<chunk>   chunks_;
    size_t               chunk_bytes_ = 0;
    int                  alloc_count_ = 0;
    bool                 abandoned_   = false;
    mutable std::mutex   mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_DEVICE_POOL_HPP
