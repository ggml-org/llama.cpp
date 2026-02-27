//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Async Expert Prefetch DMA Engine implementation.
// See expert-prefetch.hpp for design overview.

#include "expert-prefetch.hpp"

#include "common.hpp"
#include "cpu-dispatch.hpp"   // expert_miss_precision, burst threshold config

#include <cstdlib>

namespace ggml_sycl {

// ============================================================================
// Lifecycle
// ============================================================================

ExpertPrefetcher::~ExpertPrefetcher() {
    if (initialized_) {
        shutdown();
    }
}

void ExpertPrefetcher::init(sycl::queue & compute_q, ExpertCache * cache) {
    if (initialized_) {
        return;
    }
    if (!cache) {
        GGML_LOG_WARN("[SYCL] ExpertPrefetcher::init called with null cache\n");
        return;
    }

    cache_ = cache;

    // Create an out-of-order queue on the same device/context for DMA.
    // OOQ allows multiple H2D transfers to overlap and run concurrently.
    dma_queue_ = sycl::queue(compute_q.get_context(), compute_q.get_device());

    // Read prefetch depth from environment
    const char * depth_env = std::getenv("GGML_SYCL_EXPERT_PREFETCH_DEPTH");
    if (depth_env) {
        int d = std::atoi(depth_env);
        if (d > 0 && d <= 16) {
            prefetch_depth_ = d;
        }
    }

    initialized_ = true;
    GGML_LOG_INFO("[SYCL] Expert prefetcher initialized (depth=%d, max_inflight=%d)\n",
                  prefetch_depth_, max_inflight_);
}

void ExpertPrefetcher::shutdown() {
    if (!initialized_) {
        return;
    }

    cancel_all();
    initialized_ = false;
    cache_       = nullptr;
    GGML_LOG_INFO("[SYCL] Expert prefetcher shut down (completed=%d)\n", completed_count_);
}

// ============================================================================
// Hint: schedule a non-blocking async H2D prefetch on dma_queue_
// ============================================================================

bool ExpertPrefetcher::hint(int layer_idx, int expert_idx) {
    if (!initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    expert_key key{ layer_idx, expert_idx };

    // Already in-flight?
    if (inflight_.find(key) != inflight_.end()) {
        return false;
    }

    // Already cached in VRAM?
    ExpertLookup lk = cache_->lookup(layer_idx, expert_idx);
    if (lk.is_cached) {
        return false;
    }

    // Not registered in cache (no host pointer)?
    if (!lk.host_ptr) {
        return false;
    }

    // Room for more in-flight?
    if (!has_capacity()) {
        gc_completed();
        if (!has_capacity()) {
            return false;
        }
    }

    // Submit async H2D DMA on our OOQ via ExpertCache::prefetch_async().
    // This allocates a VRAM slot (evicting if needed) and submits the memcpy
    // on dma_queue_, returning a per-expert sycl::event for granular await.
    sycl::event ev = cache_->prefetch_async(layer_idx, expert_idx, dma_queue_);

    PrefetchRequest req;
    req.key       = key;
    req.event     = ev;
    req.completed = false;
    inflight_[key] = req;

    return true;
}

void ExpertPrefetcher::hint_batch(int layer_idx, const std::vector<int> & expert_indices) {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    for (int eidx : expert_indices) {
        expert_key key{ layer_idx, eidx };

        // Skip if already in-flight or already cached
        if (inflight_.find(key) != inflight_.end()) {
            continue;
        }

        ExpertLookup lk = cache_->lookup(layer_idx, eidx);
        if (lk.is_cached) {
            continue;
        }
        if (!lk.host_ptr) {
            continue;
        }

        if (!has_capacity()) {
            gc_completed();
            if (!has_capacity()) {
                break;
            }
        }

        // Submit async H2D on our OOQ
        sycl::event ev = cache_->prefetch_async(layer_idx, eidx, dma_queue_);

        PrefetchRequest req;
        req.key       = key;
        req.event     = ev;
        req.completed = false;
        inflight_[key] = req;
    }
}

// ============================================================================
// Adaptive prefetch: first N experts get DMA, rest dispatched to CPU
// ============================================================================

std::vector<int> ExpertPrefetcher::hint_batch_adaptive(
    int layer_idx,
    const std::vector<int> & expert_indices,
    int n_miss_total)
{
    std::vector<int> cpu_indices;

    if (!initialized_ || expert_indices.empty()) {
        return cpu_indices;
    }

    const expert_miss_precision mode = ggml_sycl_expert_miss_precision_mode();
    const int threshold = ggml_sycl_expert_miss_burst_threshold();

    // If full precision or below threshold: prefetch all
    if (mode == expert_miss_precision::FULL || n_miss_total <= threshold) {
        hint_batch(layer_idx, expert_indices);
        return cpu_indices;  // empty: all prefetched
    }

    // Split: first `threshold` experts get prefetch, rest go to CPU
    const int n_prefetch = std::min(threshold, static_cast<int>(expert_indices.size()));

    std::vector<int> prefetch_indices(expert_indices.begin(),
                                      expert_indices.begin() + n_prefetch);
    cpu_indices.assign(expert_indices.begin() + n_prefetch,
                       expert_indices.end());

    // Schedule DMA for the prefetch batch
    if (!prefetch_indices.empty()) {
        hint_batch(layer_idx, prefetch_indices);
    }

    GGML_SYCL_DEBUG("[PREFETCH] Adaptive: %d prefetch, %d cpu-fallback "
                    "(threshold=%d, miss=%d)\n",
                    n_prefetch, (int) cpu_indices.size(),
                    threshold, n_miss_total);

    return cpu_indices;
}

// ============================================================================
// Await: block until a specific expert's DMA completes, return VRAM ptr
// ============================================================================

void * ExpertPrefetcher::await(int layer_idx, int expert_idx) {
    if (!initialized_) {
        return nullptr;
    }

    expert_key key{ layer_idx, expert_idx };

    // Extract event under lock, then release before waiting.
    // This avoids blocking hint() callers during DMA completion.
    sycl::event ev_copy;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = inflight_.find(key);
        if (it != inflight_.end() && !it->second.completed) {
            ev_copy = it->second.event;
            found = true;
        }
    }

    if (found) {
        // Wait on the per-expert sycl::event (granular, not global queue wait).
        // This blocks only until THIS expert's H2D DMA completes on dma_queue_.
        // Mutex is NOT held during the wait, so hint() can proceed concurrently.
        ev_copy.wait();

        // Re-acquire lock to update state. Re-lookup because the map could
        // have changed while the lock was released (e.g. cancel_all()).
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = inflight_.find(key);
        if (it != inflight_.end() && !it->second.completed) {
            it->second.completed = true;
            completed_count_++;
        }
    }

    // Return the expert's VRAM pointer. After the per-expert event completes,
    // the data is guaranteed visible (BCS H2D to malloc_device completes
    // before kernel launch because await() is called before submission on
    // the in-order compute queue).
    ExpertLookup lk = cache_->lookup(layer_idx, expert_idx);
    return lk.is_cached ? lk.device_ptr : lk.host_ptr;
}

// ============================================================================
// Cancel: drain all in-flight prefetches
// ============================================================================

void ExpertPrefetcher::cancel_all() {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!inflight_.empty()) {
        // Wait for all pending DMAs on our OOQ
        dma_queue_.wait();

        for (auto & [key, req] : inflight_) {
            if (!req.completed) {
                req.completed = true;
                completed_count_++;
            }
        }
        inflight_.clear();
    }
}

// ============================================================================
// Statistics
// ============================================================================

int ExpertPrefetcher::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int pending = 0;
    for (const auto & [key, req] : inflight_) {
        if (!req.completed) {
            pending++;
        }
    }
    return pending;
}

int ExpertPrefetcher::completed_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_count_;
}

// ============================================================================
// Internal helpers
// ============================================================================

void ExpertPrefetcher::gc_completed() {
    // Called with mutex_ held.
    // Remove entries that have been completed and consumed by await().
    auto it = inflight_.begin();
    while (it != inflight_.end()) {
        if (it->second.completed) {
            it = inflight_.erase(it);
        } else {
            ++it;
        }
    }
}

bool ExpertPrefetcher::has_capacity() const {
    // Called with mutex_ held.
    return static_cast<int>(inflight_.size()) < max_inflight_;
}

}  // namespace ggml_sycl
