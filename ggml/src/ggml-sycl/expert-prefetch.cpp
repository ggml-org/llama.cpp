//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Async Expert Prefetch DMA Engine implementation.
// See expert-prefetch.hpp for design overview.

#include "expert-prefetch.hpp"

#include "common.hpp"

#include <cstdlib>

namespace ggml_sycl {

// ============================================================================
// Lifecycle
// ============================================================================

expert_prefetcher::~expert_prefetcher() {
    if (initialized_) {
        shutdown();
    }
}

void expert_prefetcher::init(sycl::queue & compute_q, expert_cache * cache) {
    if (initialized_) {
        return;
    }
    if (!cache) {
        GGML_LOG_WARN("[SYCL] expert_prefetcher::init called with null cache\n");
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

void expert_prefetcher::shutdown() {
    if (!initialized_) {
        return;
    }

    cancel_all();
    initialized_ = false;
    cache_       = nullptr;
    GGML_LOG_INFO("[SYCL] Expert prefetcher shut down (completed=%d)\n", completed_count_);
}

// ============================================================================
// Hint: schedule a non-blocking async H2D prefetch
// ============================================================================

bool expert_prefetcher::hint(int layer_idx, int expert_idx) {
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
    if (cache_->is_cached_in_vram(layer_idx, expert_idx)) {
        return false;
    }

    // Room for more in-flight?
    if (!has_capacity()) {
        gc_completed();
        if (!has_capacity()) {
            return false;
        }
    }

    // Use expert_cache to allocate a VRAM slot and get the host source pointer.
    // expert_cache::prefetch does allocation + async copy internally on its
    // copy_queue_. We wrap it here to track per-expert completion.
    //
    // However, expert_cache::prefetch uses its own copy_queue_ and does NOT
    // return per-expert events. Instead, we use our own OOQ DMA queue to
    // submit the memcpy directly, leveraging the cache's host registration
    // to get host_ptr and allocate_vram for device_ptr.
    //
    // For now, delegate to expert_cache::prefetch for the batch and track
    // via cache's wait_prefetch. This is simpler and avoids duplicating
    // VRAM allocation logic.

    // Trigger the cache to prefetch this single expert
    std::vector<expert_key> batch = { key };
    // Use expert_size = 0 to let the cache use its registered size.
    // We need to know the expert size -- look it up from the cache's host entries.
    // Since expert_cache doesn't expose expert size directly, we use a reasonable
    // default and let the cache handle it.

    // We actually want to do the DMA ourselves for per-expert event tracking.
    // But expert_cache owns VRAM allocation, so we call get_expert which does
    // sync load if needed. For truly async, we'd need to expose allocate_vram
    // from expert_cache.
    //
    // Compromise: use expert_cache::prefetch for the actual transfer (it uses
    // its own OOQ), and store a "pending" marker. await() calls wait_prefetch().

    cache_->prefetch(batch, 0);  // size 0: cache uses registered size

    prefetch_request req;
    req.key       = key;
    req.completed = false;
    inflight_[key] = req;

    return true;
}

void expert_prefetcher::hint_batch(int layer_idx, const std::vector<int> & expert_indices) {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<expert_key> batch;
    batch.reserve(expert_indices.size());

    for (int eidx : expert_indices) {
        expert_key key{ layer_idx, eidx };

        // Skip if already in-flight or already cached
        if (inflight_.find(key) != inflight_.end()) {
            continue;
        }
        if (cache_->is_cached_in_vram(layer_idx, eidx)) {
            continue;
        }
        if (!has_capacity()) {
            gc_completed();
            if (!has_capacity()) {
                break;
            }
        }

        batch.push_back(key);

        prefetch_request req;
        req.key       = key;
        req.completed = false;
        inflight_[key] = req;
    }

    if (!batch.empty()) {
        cache_->prefetch(batch, 0);
    }
}

// ============================================================================
// Await: block until a specific expert is available in VRAM
// ============================================================================

void * expert_prefetcher::await(int layer_idx, int expert_idx) {
    if (!initialized_) {
        return nullptr;
    }

    expert_key key{ layer_idx, expert_idx };

    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = inflight_.find(key);
        if (it != inflight_.end()) {
            // This expert has an in-flight prefetch. Wait for the cache's
            // copy queue to complete all pending transfers.
            cache_->wait_prefetch();

            // Mark completed
            it->second.completed = true;
            completed_count_++;
        }
    }

    // Return the expert's pointer (VRAM if cached, host fallback otherwise).
    // After wait_prefetch(), the expert should be in VRAM.
    return cache_->get_expert(layer_idx, expert_idx, 0);
}

// ============================================================================
// Cancel: drain all in-flight prefetches
// ============================================================================

void expert_prefetcher::cancel_all() {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!inflight_.empty()) {
        // Wait for all pending DMAs via the cache's copy queue
        cache_->wait_prefetch();

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

int expert_prefetcher::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int pending = 0;
    for (const auto & [key, req] : inflight_) {
        if (!req.completed) {
            pending++;
        }
    }
    return pending;
}

int expert_prefetcher::completed_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_count_;
}

// ============================================================================
// Internal helpers
// ============================================================================

void expert_prefetcher::gc_completed() {
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

bool expert_prefetcher::has_capacity() const {
    // Called with mutex_ held.
    return static_cast<int>(inflight_.size()) < max_inflight_;
}

}  // namespace ggml_sycl
