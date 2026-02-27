//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Async Expert Prefetch DMA Engine for MoE Hybrid Inference
//
// Schedules non-blocking H2D DMA to prefetch predicted expert weights from
// host RAM to VRAM while the GPU is busy computing attention. This overlaps
// PCIe transfer with GPU compute, hiding latency for cache-miss experts.
//
// Uses an out-of-order SYCL queue (dma_queue_) for DMA, separate from the
// compute queue. hint() submits memcpy on dma_queue_ via
// ExpertCache::prefetch_async() and stores the returned sycl::event in a
// PrefetchRequest. await() waits on the per-expert event for granular
// synchronization.
//
// L2 coherency: BCS H2D to malloc_device completes BEFORE the kernel
// launches because await() is called before kernel submission, and the
// in-order compute queue serializes after await().

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

#include "expert-cache.hpp"

namespace ggml_sycl {

// Tracks a single in-flight DMA prefetch operation.
struct PrefetchRequest {
    expert_key   key;
    void *       device_dst = nullptr;  // VRAM slot (from ExpertCache)
    const void * host_src   = nullptr;  // Host RAM source
    size_t       bytes      = 0;
    sycl::event  event;                 // DMA completion event from dma_queue_
    bool         completed  = false;
};

// Async DMA engine for prefetching MoE expert weights from host RAM to VRAM.
//
// Schedules non-blocking H2D DMA using an out-of-order SYCL queue (separate
// from the compute queue). Supports multiple prefetches in flight and
// per-expert await for compute/transfer overlap.
//
// Thread-safe: hint() can be called from a prediction thread while the GPU
// thread calls await().
//
// Usage:
//   ExpertPrefetcher prefetcher;
//   prefetcher.init(compute_queue, &cache);
//   prefetcher.hint(layer + 2, expert_id);    // non-blocking H2D on OOQ
//   void * ptr = prefetcher.await(layer, id); // waits on per-expert event
//
class ExpertPrefetcher {
  public:
    ExpertPrefetcher() = default;
    ~ExpertPrefetcher();

    // Non-copyable, non-movable
    ExpertPrefetcher(const ExpertPrefetcher &)             = delete;
    ExpertPrefetcher & operator=(const ExpertPrefetcher &) = delete;
    ExpertPrefetcher(ExpertPrefetcher &&)                  = delete;
    ExpertPrefetcher & operator=(ExpertPrefetcher &&)      = delete;

    // Initialize the prefetcher.
    // compute_q: the primary in-order compute queue (used to derive context/device)
    // cache: the expert VRAM cache (for slot allocation + host pointer lookup)
    void init(sycl::queue & compute_q, ExpertCache * cache);

    // Shut down: cancel all in-flight prefetches, wait for completion.
    void shutdown();

    // Schedule async prefetch of a single expert (non-blocking).
    // Submits H2D memcpy on dma_queue_ via ExpertCache::prefetch_async().
    // Returns true if a new prefetch was scheduled.
    // Returns false if: already cached in VRAM, already in-flight, no capacity,
    //                   cache is null, or expert is not registered.
    bool hint(int layer_idx, int expert_idx);

    // Schedule async prefetch of multiple experts for a layer (non-blocking).
    void hint_batch(int layer_idx, const std::vector<int> & expert_indices);

    // Wait for a specific expert's prefetch to complete and return its VRAM ptr.
    // Waits on the per-expert sycl::event (not a global queue wait).
    // If the expert is already cached (no in-flight prefetch), returns the
    // cached ptr via ExpertCache::lookup(). Returns nullptr if not registered.
    void * await(int layer_idx, int expert_idx);

    // Cancel all pending prefetches and wait for in-flight DMAs to complete.
    void cancel_all();

    // Return the configured prefetch depth (layers ahead to look).
    int prefetch_depth() const { return prefetch_depth_; }

    // Statistics
    int  pending_count() const;
    int  completed_count() const;
    bool is_active() const { return initialized_; }

  private:
    sycl::queue     dma_queue_;                // OOQ for async H2D DMA
    ExpertCache *   cache_         = nullptr;
    int             prefetch_depth_ = 2;       // Default: 2 layers ahead
    bool            initialized_   = false;

    // Max concurrent DMA operations. MoE models activate up to 8 experts
    // per layer, so 8 in-flight requests covers a full layer's worth of misses.
    static constexpr int max_inflight_ = 8;

    // In-flight prefetch tracking. Key = expert_key.
    std::unordered_map<expert_key, PrefetchRequest, expert_key_hash> inflight_;

    mutable std::mutex mutex_;

    // Stats
    int completed_count_ = 0;

    // Garbage-collect completed requests to free tracking slots.
    void gc_completed();

    // Check if we have room for more in-flight requests.
    bool has_capacity() const;
};

// Backward-compatible type alias.
using expert_prefetcher = ExpertPrefetcher;

}  // namespace ggml_sycl
