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
#include "unified-cache.hpp"  // ggml_sycl_is_shutting_down()

#include <algorithm>
#include <cstdlib>
#include <numeric>

namespace ggml_sycl {

// ============================================================================
// Lifecycle
// ============================================================================

ExpertPrefetcher::~ExpertPrefetcher() {
    if (initialized_ && !ggml_sycl_is_shutting_down()) {
        shutdown();
    }
    // During static destruction, intentionally leak the queue handle.
    // The OS reclaims all process memory at exit.
    if (ggml_sycl_is_shutting_down() && dma_queue_) {
        (void) dma_queue_.release();
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
    dma_queue_ = std::make_unique<sycl::queue>(compute_q.get_context(), compute_q.get_device());

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
    sycl::event ev = cache_->prefetch_async(layer_idx, expert_idx, *dma_queue_);

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
        sycl::event ev = cache_->prefetch_async(layer_idx, eidx, *dma_queue_);

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
        dma_queue_->wait();

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

// ============================================================================
// ExpertPredictor: pre-attention expert prediction
// ============================================================================

void ExpertPredictor::init(int n_layers, int n_experts, int n_experts_used) {
    if (initialized_) {
        return;
    }
    if (n_layers <= 0 || n_experts <= 0 || n_experts_used <= 0) {
        return;
    }

    n_layers_       = n_layers;
    n_experts_      = n_experts;
    n_experts_used_ = n_experts_used;

    last_experts_.resize(n_layers);
    freq_table_.resize(n_layers, std::vector<uint32_t>(n_experts, 0));
    last_prediction_.resize(n_layers);

    accuracy_ring_.resize(ACCURACY_WINDOW);
    accuracy_ring_pos_ = 0;
    accuracy_hits_     = 0;
    accuracy_total_    = 0;

    initialized_ = true;
    GGML_LOG_INFO("[SYCL] Expert predictor initialized (layers=%d, experts=%d, top_k=%d)\n",
                  n_layers, n_experts, n_experts_used);
}

std::vector<int> ExpertPredictor::predict(int layer_idx, const float * /*hidden_state*/) {
    if (!initialized_ || layer_idx < 0 || layer_idx >= n_layers_) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<int> predicted;
    predicted.reserve(n_experts_used_);

    // Heuristic 1: Reuse last token's experts for this layer.
    // Expert access has strong temporal locality (~70% overlap between
    // consecutive tokens in the same layer).
    const auto & last = last_experts_[layer_idx];
    for (int eidx : last) {
        if (static_cast<int>(predicted.size()) >= n_experts_used_) {
            break;
        }
        predicted.push_back(eidx);
    }

    // Heuristic 2: Fill remaining slots from global frequency table.
    // Picks the most commonly activated experts (excluding already-predicted ones).
    if (static_cast<int>(predicted.size()) < n_experts_used_) {
        int remaining = n_experts_used_ - static_cast<int>(predicted.size());
        auto freq_fill = top_k_by_freq(layer_idx, predicted, remaining);
        predicted.insert(predicted.end(), freq_fill.begin(), freq_fill.end());
    }

    // Store prediction for accuracy tracking
    last_prediction_[layer_idx] = predicted;

    return predicted;
}

void ExpertPredictor::record_actual(int layer_idx, const std::vector<int> & actual_experts) {
    if (!initialized_ || layer_idx < 0 || layer_idx >= n_layers_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Update last-token expert selections for this layer
    last_experts_[layer_idx] = actual_experts;

    // Update global frequency table
    for (int eidx : actual_experts) {
        if (eidx >= 0 && eidx < n_experts_) {
            freq_table_[layer_idx][eidx]++;
        }
    }

    // Accuracy tracking: compare last prediction vs actual
    const auto & pred = last_prediction_[layer_idx];
    if (!pred.empty()) {
        // Count how many predicted experts were actually selected
        int hits = 0;
        for (int p : pred) {
            for (int a : actual_experts) {
                if (p == a) {
                    hits++;
                    break;
                }
            }
        }

        // Hit if we predicted at least half of the actual experts
        // (integer division: lenient for odd sizes).
        bool sample_hit = (hits > 0 && !actual_experts.empty() &&
                           hits >= static_cast<int>(actual_experts.size()) / 2);

        // Update rolling window
        if (accuracy_total_ >= ACCURACY_WINDOW) {
            // Evict oldest sample
            if (accuracy_ring_[accuracy_ring_pos_]) {
                accuracy_hits_--;
            }
        } else {
            accuracy_total_++;
        }

        accuracy_ring_[accuracy_ring_pos_] = sample_hit ? 1 : 0;
        if (sample_hit) {
            accuracy_hits_++;
        }
        accuracy_ring_pos_ = (accuracy_ring_pos_ + 1) % ACCURACY_WINDOW;

        // Periodic logging every ACCURACY_WINDOW predictions
        if (accuracy_total_ >= ACCURACY_WINDOW && accuracy_ring_pos_ == 0) {
            float rate = static_cast<float>(accuracy_hits_) / static_cast<float>(accuracy_total_);
            GGML_LOG_INFO("[EXPERT-PREDICT] accuracy=%.1f%% (hits=%d, window=%d)\n",
                          rate * 100.0f, accuracy_hits_, accuracy_total_);
        }
    }
}

float ExpertPredictor::hit_rate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (accuracy_total_ == 0) {
        return 0.0f;
    }
    return static_cast<float>(accuracy_hits_) / static_cast<float>(accuracy_total_);
}

int ExpertPredictor::window_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return accuracy_total_;
}

int ExpertPredictor::window_hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return accuracy_hits_;
}

std::vector<int> ExpertPredictor::top_k_by_freq(int layer_idx,
                                                 const std::vector<int> & exclude,
                                                 int k) const {
    // Called with mutex_ held.
    if (layer_idx < 0 || layer_idx >= n_layers_ || k <= 0) {
        return {};
    }

    const auto & freq = freq_table_[layer_idx];

    // Build indices sorted by frequency (descending)
    std::vector<int> indices(n_experts_);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(),
                      indices.begin() + std::min(k + static_cast<int>(exclude.size()), n_experts_),
                      indices.end(),
                      [&freq](int a, int b) { return freq[a] > freq[b]; });

    // Pick top-k that aren't in the exclude set
    std::vector<int> result;
    result.reserve(k);
    for (int idx : indices) {
        if (static_cast<int>(result.size()) >= k) {
            break;
        }
        bool excluded = false;
        for (int ex : exclude) {
            if (idx == ex) {
                excluded = true;
                break;
            }
        }
        if (!excluded) {
            result.push_back(idx);
        }
    }

    return result;
}

// ============================================================================
// Environment variable helper
// ============================================================================

bool ggml_sycl_expert_predict_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = std::getenv("GGML_SYCL_EXPERT_PREDICT");
        // Default: ON (1) unless explicitly set to 0
        cached = (!env || std::atoi(env) != 0) ? 1 : 0;
    }
    return cached != 0;
}

}  // namespace ggml_sycl
