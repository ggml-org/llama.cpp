//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_PINNED_POOL_HPP
#define GGML_SYCL_PINNED_POOL_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace ggml_sycl {

// Host memory zone identifiers. Mirrors the VRAM arena zoning model.
enum class host_zone_id : uint8_t {
    WEIGHT  = 0,
    KV      = 1,
    STAGING = 2,
    SCRATCH = 3,
    COUNT   = 4,
};

const char * host_zone_name(host_zone_id zone);

struct host_zone {
    size_t              start = 0;
    size_t              size  = 0;
    std::atomic<size_t> used{ 0 };
};

// Segment of a pinned buffer.
struct buffer_segment {
    void * ptr  = nullptr;
    size_t size = 0;
};

// A buffer that may consist of one or more non-contiguous pinned segments.
struct segmented_buffer {
    std::vector<buffer_segment> segments;
    size_t                      total_size = 0;

    void * contiguous_ptr() const { return segments.size() == 1 ? segments[0].ptr : nullptr; }

    bool is_contiguous() const { return segments.size() <= 1; }
};

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

    // Allocate from pool. Returns a segmented_buffer containing one or more pinned segments.
    // All segments are aligned to DEFAULT_ALIGNMENT (64 bytes).
    segmented_buffer allocate_segmented(size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Legacy wrapper: returns first segment's pointer. For large allocations, use allocate_segmented().
    [[deprecated("use allocate_segmented() for large allocations")]] void * allocate(
        size_t size,
        size_t alignment = DEFAULT_ALIGNMENT);

    // Runtime allocation: uses a separate pool of chunks that are not part of
    // the zone layout.  Suitable for large contiguous allocations (e.g., 615 MB
    // reorder buffers) that don't fit within a single 256 MB zone chunk.
    // Falls back to growing the runtime pool if no existing chunk has space.
    void * allocate_runtime(size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Mark allocation as free. Note: bump allocator - individual deallocations
    // are tracked but memory is only reclaimed when the pool is destroyed.
    void deallocate(void * ptr, size_t size);

    // Pre-allocate enough chunks to hold total_bytes without any runtime allocation.
    // Call at init time (e.g. after MoE prestage) so that inference-time allocate()
    // never triggers sycl::malloc_host which blocks the Level Zero driver.
    // Returns the number of chunks actually allocated.
    size_t pre_allocate(size_t total_bytes);

    // Pre-allocate ALL chunks needed for the full model weight set plus headroom.
    // Called once after model header is parsed and total weight size is known.
    // Returns the number of new chunks allocated.
    size_t pre_allocate_all(size_t model_weight_bytes);

    // Configure logical zones across the pool's free capacity.
    // Must be called after pre-allocation and before any zone_alloc().
    void configure_zones(size_t weight_bytes, size_t kv_bytes, size_t staging_bytes, size_t scratch_bytes);

    // Allocate from a specific host zone. Returns a segmented_buffer.
    segmented_buffer zone_alloc_segmented(host_zone_id zone, size_t size, size_t alignment = DEFAULT_ALIGNMENT);

    // Reset a zone's bump pointer. Callers must ensure no outstanding users remain.
    void zone_reset(host_zone_id zone);

    // Roll back a zone's bump pointer to a previously saved position.
    // Unlike zone_reset (which sets used=0), this restores to a specific
    // offset, preserving earlier allocations in the zone.
    void zone_rollback(host_zone_id zone, size_t saved_used);

    size_t zone_used(host_zone_id zone) const;
    size_t zone_capacity(host_zone_id zone) const;

    // Grow a zone by allocating additional chunks and extending the zone's span.
    // Used when compute buffer needs exceed the initially planned SCRATCH zone.
    // Returns true if the zone was successfully grown.
    bool grow_zone(host_zone_id zone, size_t additional_bytes);

    bool zones_configured() const { return zones_configured_; }

    // Check if a pointer belongs to this pool (falls within any chunk).
    bool contains(const void * ptr) const;

    // Statistics
    size_t budget() const { return budget_; }

    size_t allocated() const;  // Total bytes allocated (in chunks)
    size_t chunk_count() const;

  private:
    struct chunk {
        void * base;         // malloc_host result
        size_t size;         // CHUNK_SIZE
        size_t used;         // Bump pointer offset
        size_t freed;        // Number of deallocations (count, not bytes)
        size_t alloc_count;  // Total allocations from this chunk
    };

    struct zone_chunk_span {
        size_t logical_start = 0;
        size_t chunk_idx     = 0;
        size_t chunk_start   = 0;
        size_t span_size     = 0;
    };

    void * allocate_from_chunks(std::vector<chunk> & chunks, size_t size, size_t alignment, bool runtime_pool);
    bool   deallocate_from_chunks(std::vector<chunk> & chunks, void * ptr);
    bool   grow_into(std::vector<chunk> & chunks, size_t min_size, bool runtime_pool);

    // Allocate a new chunk (>= min_size). Returns false if over budget or allocation fails.
    bool grow(size_t min_size);

    sycl::queue &                                                   queue_;
    size_t                                                          budget_;
    size_t                                                          total_allocated_  = 0;
    size_t                                                          chunk_size_       = CHUNK_SIZE;
    size_t                                                          alloc_timeout_ms_ = 0;
    std::vector<chunk>                                              chunks_;
    std::vector<chunk>                                              runtime_chunks_;
    std::array<host_zone, static_cast<size_t>(host_zone_id::COUNT)> zones_{};
    bool                                                            zones_configured_ = false;
    std::vector<zone_chunk_span>                                    flat_spans_;
    mutable std::mutex                                              mutex_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_PINNED_POOL_HPP
