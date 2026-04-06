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
    // All VRAM is owned by the unified cache — nothing to free here.
    if (ggml_sycl_is_shutting_down() && dma_queue_) {
        (void) dma_queue_.release();
    }
}

void ExpertPrefetcher::init(sycl::queue & compute_q) {
    if (initialized_) {
        return;
    }

    // Create an out-of-order queue on the same device/context for DMA.
    // OOQ allows multiple H2D transfers to overlap and run concurrently.
    dma_queue_ = std::make_unique<sycl::queue>(compute_q.get_context(), compute_q.get_device());

    // Derive device ID from the compute queue for VRAM budget tracking.
    device_id_ = ggml_sycl_get_device_id_from_queue(compute_q);

    // Read prefetch depth from environment
    const char * depth_env = std::getenv("GGML_SYCL_EXPERT_PREFETCH_DEPTH");
    if (depth_env) {
        int d = std::atoi(depth_env);
        if (d > 0 && d <= 16) {
            prefetch_depth_ = d;
        }
    }

    initialized_ = true;
    GGML_LOG_INFO("[SYCL] Expert prefetcher initialized (depth=%d, dynamic pool)\n",
                  prefetch_depth_);
}

void ExpertPrefetcher::shutdown() {
    if (!initialized_) {
        return;
    }

    cancel_all();
    initialized_ = false;
    GGML_LOG_INFO("[SYCL] Expert prefetcher shut down (prefetched=%d)\n",
                  completed_count_);

    // Print final MoE dispatch statistics
    if (MoeDispatchStats::enabled()) {
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; d++) {
            auto & stats = get_moe_dispatch_stats(d);
            if (stats.total_experts_dispatched.load(std::memory_order_relaxed) > 0) {
                stats.print_summary("FINAL");
            }
        }
    }
}

// ============================================================================
// Hint: schedule a non-blocking async H2D prefetch on dma_queue_
// ============================================================================

bool ExpertPrefetcher::hint(int layer_idx, int expert_idx) {
    if (!initialized_ || !dma_queue_ || prefetch_disabled_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    return hint_locked(layer_idx, expert_idx);
}

// Internal implementation of hint(). Caller must hold mutex_.
//
// Uses moe_get_expert_stage_info() to read metadata, then submits async DMA +
// AOS->SOA reorder via direct_stage_expert(). Returns immediately after DMA
// submission -- does NOT block on DMA completion.
//
// The returned sycl::event is stored in inflight_ and waited on in await().
bool ExpertPrefetcher::hint_locked(int layer_idx, int expert_idx) {
    expert_key key{ layer_idx, expert_idx };

    // Already in-flight -- skip.
    if (inflight_.count(key)) {
        return false;
    }

    // Garbage-collect completed in-flight entries.
    gc_completed();

    // Step 1: Get expert metadata via read-only accessor.
    expert_stage_info info;
    if (!moe_get_expert_stage_info(layer_idx, expert_idx, device_id_, info)) {
        return false;
    }

    // Step 2: Check unified cache -- may already be READY.
    unified_cache * cache = get_unified_cache_for_device(device_id_);
    if (!cache) {
        return false;
    }

    void * cached_ptr = cache->lookup(info.cache_key, info.layout);
    if (cached_ptr) {
        completed_count_++;
        GGML_SYCL_DEBUG("[PREFETCH] hint L%d E%d: already cached in unified cache, ptr=%p\n",
                        layer_idx, expert_idx, cached_ptr);
        return true;
    }

    // Step 3: Build reorder context on the stack.
    // fill_fn reads ctx synchronously during submission (before
    // direct_stage_expert returns), so stack allocation is safe.
    reorder_fill_ctx rctx{};
    rctx.type          = info.type;
    rctx.ncols         = info.ncols;
    rctx.nrows         = info.nrows;
    rctx.nbytes        = info.src_size;
    rctx.dst_bytes     = info.dst_size;
    rctx.layout        = info.layout;
    rctx.src_is_device = false;
    rctx.device_id     = device_id_;

    // Step 4: Submit async DMA via direct_stage_expert.
    // Route H2D to BCS (copy-only) queue so prefetch DMA doesn't monopolize
    // CCS and trigger xe driver GT engine resets during inference.
    sycl::queue * bcs_q = &cache->get_bcs_queue();

    direct_stage_result sr;
    try {
        sr = cache->direct_stage_expert(
            info.cache_key, info.src_ptr, info.src_size, info.dst_size,
            info.layout, fill_reordered_host, &rctx, bcs_q);
    } catch (...) {
        return false;
    }

    if (!sr.ok || !sr.ptr) {
        return false;
    }

    // DMA submitted -- track in-flight for await().
    prefetch_request req;
    req.key        = key;
    req.event      = sr.event;
    req.device_ptr = sr.ptr;
    req.completed  = false;
    req.cache_key  = info.cache_key;
    req.layout     = info.layout;
    req.size       = info.dst_size;
    req.layer_id   = info.layer_id;
    req.expert_id  = info.expert_id;

    inflight_[key] = std::move(req);
    GGML_SYCL_DEBUG("[PREFETCH] hint L%d E%d: async DMA submitted, dev=%d ptr=%p\n",
                    layer_idx, expert_idx, device_id_, sr.ptr);
    return true;
}

void ExpertPrefetcher::hint_batch(int layer_idx, const std::vector<int> & expert_indices) {
    if (!initialized_ || !dma_queue_ || prefetch_disabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (int eid : expert_indices) {
        hint_locked(layer_idx, eid);
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
    std::vector<int> cpu_dispatch;

    if (!initialized_ || !dma_queue_ || prefetch_disabled_) {
        return cpu_dispatch;
    }

    // Hold the lock across the entire function to prevent TOCTOU races:
    // budget is computed and consumed atomically within a single critical section.
    std::lock_guard<std::mutex> lock(mutex_);

    gc_completed();
    int max_dma = max_concurrent_dma_;
    int budget = max_dma - static_cast<int>(inflight_.size());

    int scheduled = 0;
    for (int eid : expert_indices) {
        // Schedule prefetch when: (1) we have remaining DMA budget, AND
        // (2) the total miss count across all experts is within our capacity.
        // When n_miss_total > max_dma, even the first batch of experts
        // would saturate DMA bandwidth, so overflow to CPU instead.
        if (scheduled < budget && n_miss_total <= max_dma) {
            if (hint_locked(layer_idx, eid)) {
                scheduled++;
            }
        } else {
            cpu_dispatch.push_back(eid);
        }
    }

    return cpu_dispatch;
}

// ============================================================================
// Await: block until a specific expert's DMA completes, return VRAM ptr
// ============================================================================

void * ExpertPrefetcher::await(int layer_idx, int expert_idx) {
    if (!initialized_) {
        return nullptr;
    }

    expert_key key{ layer_idx, expert_idx };

    // Phase 1: Check in-flight completed or extract event.
    // Release mutex before blocking on event.wait() to avoid deadlock.
    sycl::event ev_copy;
    bool        need_wait = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = inflight_.find(key);
        if (it == inflight_.end()) {
            return nullptr;
        }
        if (it->second.completed) {
            return it->second.device_ptr;
        }
        ev_copy   = it->second.event;
        need_wait = true;
    }

    // Phase 2: Wait on event WITHOUT holding the lock.
    if (need_wait) {
        try {
            ev_copy.wait();
        } catch (const sycl::exception & e) {
            GGML_LOG_WARN("[SYCL] Prefetch await failed for L%d E%d: %s\n",
                          layer_idx, expert_idx, e.what());
            std::lock_guard<std::mutex> lock(mutex_);
            inflight_.erase(key);
            return nullptr;
        }
    }

    // Phase 3: Re-acquire lock, finalize cache entry, and update state.
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = inflight_.find(key);
    if (it == inflight_.end()) {
        return nullptr;
    }

    if (!it->second.completed) {
        it->second.completed = true;
        completed_count_++;

        // Entry is already stored in direct_expert_entries_ by
        // direct_stage_expert() and findable via lookup_expert().
        // No register_ready() call needed.

        GGML_SYCL_DEBUG("[PREFETCH] await L%d E%d: DMA complete, device_ptr=%p\n",
                        layer_idx, expert_idx, it->second.device_ptr);
    }

    return it->second.device_ptr;
}

// ============================================================================
// Cancel: drain all in-flight prefetches
// ============================================================================

void ExpertPrefetcher::cancel_all() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Wait for all in-flight DMAs.
    // First: wait on individual per-expert events (handles unified cache queue DMA).
    for (auto & [key, req] : inflight_) {
        if (!req.completed) {
            try {
                req.event.wait();
            } catch (const sycl::exception &) {
                // Best effort during shutdown.
            }
        }
    }
    // Also drain the prefetcher's own OOQ for any legacy DMA.
    if (dma_queue_) {
        try {
            dma_queue_->wait();
        } catch (const sycl::exception &) {
            // Best effort during shutdown.
        }
    }

    // Clear in-flight tracking. All VRAM is owned by the unified cache.
    inflight_.clear();
}

// ============================================================================
// Non-blocking cache query
// ============================================================================

void * ExpertPrefetcher::get_cached_ptr(int layer_idx, int expert_idx) const {
    if (!initialized_) {
        return nullptr;
    }

    // Query the unified cache for this expert's device pointer.
    // The prefetcher no longer maintains a private pool.
    expert_stage_info info;
    if (!moe_get_expert_stage_info(layer_idx, expert_idx, device_id_, info)) {
        return nullptr;
    }

    unified_cache * cache = get_unified_cache_for_device(device_id_);
    if (!cache) {
        return nullptr;
    }

    return cache->lookup(info.cache_key, info.layout);
}

// ============================================================================
// Demand load: synchronous hint + await for cache-miss experts
// ============================================================================

void * ExpertPrefetcher::demand_load(int layer_idx, int expert_idx) {
    if (!initialized_ || !dma_queue_) {
        return nullptr;
    }

    // Submit DMA via unified cache (hint_locked checks cache first).
    {
        std::lock_guard<std::mutex> lock(mutex_);
        hint_locked(layer_idx, expert_idx);
    }

    // Wait for DMA completion and return VRAM pointer.
    return await(layer_idx, expert_idx);
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
    // Remove completed entries from in-flight map. The unified cache tracks
    // all cached expert data — the prefetcher only needs to track active DMA.
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
    // Count only active (non-completed) in-flight entries.
    int active = 0;
    for (const auto & [k, req] : inflight_) {
        if (!req.completed) {
            active++;
        }
    }
    return active < max_concurrent_dma_;
}

// ============================================================================
// Pre-load popular experts into VRAM via unified cache at model init time
// ============================================================================

void ExpertPrefetcher::preload_experts(int layer_idx, const std::vector<int> & expert_ids) {
    if (!initialized_ || !dma_queue_ || prefetch_disabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    int preloaded = 0;
    for (int eid : expert_ids) {
        // Use SOA-correct caching via unified cache (same as hint_locked).
        // This fixes the pre-existing bug where raw memcpy copied AOS layout
        // to VRAM but MMVQ kernels expect SOA layout.
        void * ptr = moe_expert_ensure_soa_cached(layer_idx, eid, device_id_);
        if (ptr) {
            preloaded++;
        }
    }

    if (preloaded > 0) {
        GGML_LOG_INFO("[SYCL] Preloaded %d experts for layer %d into VRAM (SOA via unified cache)\n",
                      preloaded, layer_idx);
    }
}

// ============================================================================
// ExpertPredictor: pre-attention expert prediction
// ============================================================================

ExpertPredictor::~ExpertPredictor() {
    // Free pre-allocated device scores buffer.
    // Skip during static destruction (SYCL context may be invalid).
    if (scores_dev_ && scores_queue_ && !ggml_sycl_is_shutting_down()) {
        if (scores_dev_n_ > 0 && !scores_from_arena_) {
            int scores_device = ggml_sycl_get_device_id_from_queue(*scores_queue_);
            unified_cache_sub_runtime_bytes(scores_device, scores_dev_n_ * sizeof(float),
                                            runtime_category::EXPERT_CACHE);
        }
        if (!scores_from_arena_) {
            try {
                sycl::free(scores_dev_, *scores_queue_);
            } catch (...) {
                // SYCL runtime may be partially torn down
            }
        }
    }
    scores_dev_        = nullptr;
    scores_dev_n_      = 0;
    scores_from_arena_ = false;
}

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
    window_total_    = 0;

    // Read prediction depth from environment
    const char * depth_env = std::getenv("GGML_SYCL_EXPERT_PREDICT_DEPTH");
    if (depth_env) {
        int d = std::atoi(depth_env);
        if (d >= 1 && d <= 8) {
            predict_depth_ = d;
        }
    }

    initialized_ = true;
    GGML_LOG_INFO("[SYCL] Expert predictor initialized (layers=%d, experts=%d, top_k=%d, depth=%d)\n",
                  n_layers, n_experts, n_experts_used, predict_depth_);
}

// ============================================================================
// argsort_top_k: host-side top-K selection from score array
// ============================================================================

static std::vector<int> argsort_top_k(const std::vector<float> & scores, int k) {
    const int n = static_cast<int>(scores.size());
    if (n == 0 || k <= 0) {
        return {};
    }
    k = std::min(k, n);

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });
    indices.resize(k);
    return indices;
}

// ============================================================================
// ExpertPredictor: heuristic prediction
// ============================================================================

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

    // Heuristic 2: Cross-layer correlation — experts active in a recent
    // preceding layer for the CURRENT token are likely active in this layer.
    // Sequential IDs include gate/up/down tensors, but record_actual() only
    // fills gate tensor entries. Scan backwards to find the nearest non-empty
    // last_experts_ entry (i.e., previous block's gate routing).
    if (static_cast<int>(predicted.size()) < n_experts_used_) {
        for (int prev = layer_idx - 1; prev >= 0 && prev >= layer_idx - 4; prev--) {
            const auto & prev_layer = last_experts_[prev];
            if (prev_layer.empty()) continue;
            for (int eidx : prev_layer) {
                if (static_cast<int>(predicted.size()) >= n_experts_used_) {
                    break;
                }
                // Skip duplicates (already predicted by Heuristic 1)
                bool dup = false;
                for (int p : predicted) {
                    if (p == eidx) { dup = true; break; }
                }
                if (!dup) {
                    predicted.push_back(eidx);
                }
            }
            break;  // Only use the nearest non-empty predecessor
        }
    }

    // Heuristic 3: Fill remaining slots from global frequency table.
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

    // Update last-token expert selections for this layer (ALWAYS, even during warmup)
    last_experts_[layer_idx] = actual_experts;

    // Update global frequency table (ALWAYS, needed for learning)
    for (int eidx : actual_experts) {
        if (eidx >= 0 && eidx < n_experts_) {
            freq_table_[layer_idx][eidx]++;
        }
    }

    // Detect token boundary for warmup tracking.
    // When layer_idx wraps around to a previously-seen layer, a new token has started.
    // NOTE: This heuristic assumes record_actual is called once per MoE layer
    // per token in increasing layer_idx order. Speculative decoding or out-of-order
    // dispatch can cause warmup_tokens_ to advance faster than actual token count.
    if (warmup_layer_max_ >= 0 && layer_idx < warmup_layer_max_) {
        warmup_tokens_ = std::min(warmup_tokens_ + 1, 2);
    }
    warmup_layer_max_ = std::max(warmup_layer_max_, layer_idx);

    // Accuracy tracking: compare last prediction vs actual.
    // Skip during warmup (first 2 tokens) to avoid cold-start misses
    // poisoning the accuracy window and triggering permanent disable.
    const auto & pred = last_prediction_[layer_idx];
    if (!pred.empty() && warmup_tokens_ >= 2) {
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

        // Feed per-expert stats to MoeDispatchStats (lock-free, outside mutex scope)
        if (MoeDispatchStats::enabled()) {
            // Use device 0 for stats — record_actual is called from the primary device
            auto & stats = get_moe_dispatch_stats(0);
            stats.record_prediction_accuracy(pred, actual_experts);
        }

        // Hit if we predicted at least half of the actual experts
        // (integer division: lenient for odd sizes).
        bool sample_hit = (hits > 0 && !actual_experts.empty() &&
                           hits >= static_cast<int>(actual_experts.size()) / 2);

        // Update rolling window
        if (window_total_ >= ACCURACY_WINDOW) {
            // Evict oldest sample
            if (accuracy_ring_[accuracy_ring_pos_]) {
                accuracy_hits_--;
            }
        } else {
            window_total_++;
        }

        accuracy_ring_[accuracy_ring_pos_] = sample_hit ? 1 : 0;
        if (sample_hit) {
            accuracy_hits_++;
        }
        accuracy_ring_pos_ = (accuracy_ring_pos_ + 1) % ACCURACY_WINDOW;

        // Periodic logging every ACCURACY_WINDOW predictions
        static constexpr float DISABLE_THRESHOLD = 0.3f;
        if (window_total_ >= ACCURACY_WINDOW && accuracy_ring_pos_ == 0) {
            float rate = static_cast<float>(accuracy_hits_) / static_cast<float>(window_total_);
            GGML_LOG_INFO("[EXPERT-PREDICT] accuracy=%.1f%% (hits=%d, window=%d)\n",
                          rate * 100.0f, accuracy_hits_, window_total_);

            // Disable prefetching when prediction accuracy drops below 30%.
            // At this accuracy, most prefetches are wasted DMA bandwidth.
            if (!prefetch_disabled_ && rate < DISABLE_THRESHOLD) {
                prefetch_disabled_ = true;
                GGML_LOG_WARN("[EXPERT-PREDICT] hit rate %.1f%% below threshold %.0f%% — "
                              "prefetching disabled\n",
                              rate * 100.0f, DISABLE_THRESHOLD * 100.0f);
            } else if (prefetch_disabled_ && rate >= DISABLE_THRESHOLD) {
                // Re-enable prefetching if accuracy recovers above threshold.
                // This handles the case where cold-start disabled prefetch but
                // the predictor later achieves good accuracy after warmup.
                prefetch_disabled_ = false;
                GGML_LOG_INFO("[EXPERT-PREDICT] hit rate %.1f%% recovered above threshold "
                              "(hits=%d, window=%d) — prefetching re-enabled\n",
                              rate * 100.0f, accuracy_hits_, window_total_);
            }
        }
    }
}

float ExpertPredictor::hit_rate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (window_total_ == 0) {
        return 0.0f;
    }
    return static_cast<float>(accuracy_hits_) / static_cast<float>(window_total_);
}

int ExpertPredictor::window_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return window_total_;
}

int ExpertPredictor::window_hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return accuracy_hits_;
}

bool ExpertPredictor::is_prefetch_disabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return prefetch_disabled_;
}

std::vector<std::pair<int, uint32_t>> ExpertPredictor::get_frequency_ranking(int layer_idx) const {
    if (!initialized_ || layer_idx < 0 || layer_idx >= n_layers_) {
        return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<int, uint32_t>> ranked;
    for (int e = 0; e < n_experts_; e++) {
        if (freq_table_[layer_idx][e] > 0) {
            ranked.push_back({e, freq_table_[layer_idx][e]});
        }
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });
    return ranked;
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
// ExpertPredictor: pre-gated router prediction via inline SYCL GEMV
// ============================================================================

std::vector<int> ExpertPredictor::predict_pregate(int           next_layer_idx,
                                                  const void *  gate_weights,
                                                  const void *  hidden_state,
                                                  sycl::queue & compute_q) {
    if (!initialized_ || next_layer_idx < 0 || next_layer_idx >= n_layers_) {
        return predict(next_layer_idx);
    }

    // If gate_weights not provided explicitly, look up from registered pointers
    if (!gate_weights) {
        if (next_layer_idx < static_cast<int>(gate_weight_ptrs_.size())) {
            gate_weights = gate_weight_ptrs_[next_layer_idx];
        }
    }

    // Fallback to heuristic if inputs are unavailable
    if (!gate_weights || !hidden_state) {
        return predict(next_layer_idx);
    }

    if (n_embd_ <= 0 || n_experts_ <= 0) {
        return predict(next_layer_idx);
    }

    const int    K          = n_embd_;
    const int    M          = n_experts_;
    const auto * gate_f32   = static_cast<const float *>(gate_weights);
    const auto * hidden_f32 = static_cast<const float *>(hidden_state);

    // Allocate host buffer for scores (tiny: n_experts floats, e.g. 512 bytes for 128 experts)
    std::vector<float> scores_host(M);

    // Reuse pre-allocated device buffer for output scores, or allocate on first use.
    // This avoids sycl::malloc_device/free per call (3 calls with 3-layer lookahead).
    if (!scores_dev_ || scores_dev_n_ < M) {
        int scores_device = ggml_sycl_get_device_id_from_queue(compute_q);
        if (scores_dev_ && scores_queue_) {
            if (scores_dev_n_ > 0 && !scores_from_arena_) {
                unified_cache_sub_runtime_bytes(scores_device, scores_dev_n_ * sizeof(float),
                                                runtime_category::EXPERT_CACHE);
            }
            if (!scores_from_arena_) {
                sycl::free(scores_dev_, *scores_queue_);
            }
        }
        scores_from_arena_ = false;
        if (vram_arena_enabled() && scores_device >= 0) {
            void * ptr = unified_cache_arena_alloc(scores_device, M * sizeof(float));
            if (ptr) {
                scores_dev_        = static_cast<float *>(ptr);
                scores_from_arena_ = true;
            }
        }
        if (!scores_from_arena_) {
            scores_dev_ = static_cast<float *>(ggml_sycl_malloc_device(M * sizeof(float), compute_q, "scores_dev"));
        }
        scores_dev_n_ = M;
        scores_queue_ = &compute_q;
        if (!scores_dev_) {
            scores_dev_n_ = 0;
            GGML_LOG_WARN("[EXPERT-PREDICT] Failed to allocate device scores buffer, falling back to heuristic\n");
            return predict(next_layer_idx);
        }
        if (!scores_from_arena_) {
            unified_cache_add_runtime_bytes(scores_device, M * sizeof(float), runtime_category::EXPERT_CACHE);
        }
    }
    float * scores_dev = scores_dev_;

    // Inline SYCL GEMV kernel:
    //   scores[j] = sum_k(gate_weights[j * K + k] * hidden_state[k])
    //   n_experts work groups, each computing one output element via
    //   subgroup reduction + SLM cross-subgroup accumulation.
    const int wg_size = std::min(256, ((K + 15) / 16) * 16);  // Clamp WG size, round up to 16
    const int n_wgs   = M;

    try {
        auto ev = compute_q.submit([&](sycl::handler & cgh) {
            // Allocate SLM for cross-subgroup reduction.  Use wg_size as upper
            // bound on subgroup count (covers any subgroup size the runtime picks).
            sycl::local_accessor<float, 1> slm(sycl::range<1>(wg_size + 1), cgh);

            cgh.parallel_for(sycl::nd_range<1>(n_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                const int j     = item.get_group_linear_id();  // expert index
                const int lid   = item.get_local_linear_id();
                const int wg_sz = item.get_local_range(0);

                const float * gate_row = gate_f32 + j * K;
                float         sum      = 0.0f;
                for (int k = lid; k < K; k += wg_sz) {
                    sum += gate_row[k] * hidden_f32[k];
                }

                // Subgroup reduction
                auto sg = item.get_sub_group();
                sum     = sycl::reduce_over_group(sg, sum, sycl::plus<float>());

                // Cross-subgroup reduction via SLM
                const int sg_id  = sg.get_group_linear_id();
                const int sg_lid = sg.get_local_linear_id();
                const int n_sgs  = wg_sz / sg.get_local_linear_range();

                if (sg_lid == 0) {
                    slm[sg_id] = sum;
                }
                sycl::group_barrier(item.get_group());

                // Thread 0 accumulates across subgroups
                if (lid == 0) {
                    float total = 0.0f;
                    for (int s = 0; s < n_sgs; s++) {
                        total += slm[s];
                    }
                    scores_dev[j] = total;
                }
            });
        });

        // D2H copy of scores (tiny: M floats, e.g. 512 bytes for 128 experts)
        compute_q.memcpy(scores_host.data(), scores_dev, M * sizeof(float), ev).wait();

    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[EXPERT-PREDICT] Pre-gate GEMV failed: %s, falling back to heuristic\n", e.what());
        return predict(next_layer_idx);
    }

    // Top-K selection on host
    auto result = argsort_top_k(scores_host, n_experts_used_);

    GGML_SYCL_DEBUG("[EXPERT-PREDICT] Pre-gate layer=%d: top-%d experts = [", next_layer_idx, n_experts_used_);
    for (int i = 0; i < static_cast<int>(result.size()); i++) {
        GGML_SYCL_DEBUG("%s%d", i > 0 ? "," : "", result[i]);
    }
    GGML_SYCL_DEBUG("]\n");

    return result;
}

// ============================================================================
// ExpertPredictor: multi-layer lookahead prediction
// ============================================================================

std::vector<std::pair<int, std::vector<int>>> ExpertPredictor::predict_multi_layer(
    int current_seq_layer,
    const void * hidden_state,
    sycl::queue & compute_q)
{
    std::vector<std::pair<int, std::vector<int>>> results;

    for (int depth = 1; depth <= predict_depth_; depth++) {
        int target = current_seq_layer + depth;
        if (target >= n_layers_) {
            break;
        }

        auto predicted = predict_pregate(target, nullptr, hidden_state, compute_q);
        if (!predicted.empty()) {
            results.push_back({ target, std::move(predicted) });
        }
    }

    return results;
}

void ExpertPredictor::register_gate_weights(int layer_idx, const void * gate_ptr, int n_embd) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (layer_idx < 0) {
        return;
    }

    // Grow the gate pointer vector if needed
    if (layer_idx >= static_cast<int>(gate_weight_ptrs_.size())) {
        gate_weight_ptrs_.resize(layer_idx + 1, nullptr);
    }

    gate_weight_ptrs_[layer_idx] = gate_ptr;
    n_embd_                      = n_embd;
}

bool ExpertPredictor::has_gate_weights(int layer_idx) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (layer_idx < 0 || layer_idx >= static_cast<int>(gate_weight_ptrs_.size())) {
        return false;
    }
    return gate_weight_ptrs_[layer_idx] != nullptr;
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

// ============================================================================
// MoE Dispatch Statistics
// ============================================================================

bool MoeDispatchStats::enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = std::getenv("GGML_SYCL_MOE_STATS");
        // Default: ON (1) when expert prediction is active, unless explicitly 0
        cached = (!env || std::atoi(env) != 0) ? 1 : 0;
    }
    return cached != 0;
}

void MoeDispatchStats::record_dispatch(int n_vram, int n_host, int n_staging,
                                        int n_miss, int n_prefetched) {
    const int total = n_vram + n_host + n_staging + n_miss;
    total_experts_dispatched.fetch_add(total, std::memory_order_relaxed);
    total_vram_hits.fetch_add(n_vram, std::memory_order_relaxed);
    total_host_hits.fetch_add(n_host, std::memory_order_relaxed);
    total_staging.fetch_add(n_staging, std::memory_order_relaxed);
    total_cpu_fallbacks.fetch_add(n_miss, std::memory_order_relaxed);
    total_prefetch_hits.fetch_add(n_prefetched, std::memory_order_relaxed);
    total_layers.fetch_add(1, std::memory_order_relaxed);

    interval_experts.fetch_add(total, std::memory_order_relaxed);
    interval_vram_hits.fetch_add(n_vram, std::memory_order_relaxed);
    interval_host_hits.fetch_add(n_host, std::memory_order_relaxed);
    interval_staging.fetch_add(n_staging, std::memory_order_relaxed);
    interval_cpu_fallbacks.fetch_add(n_miss, std::memory_order_relaxed);
}

void MoeDispatchStats::record_prediction_accuracy(const std::vector<int> & predicted,
                                                   const std::vector<int> & actual) {
    if (predicted.empty() || actual.empty()) {
        return;
    }

    int correct = 0;
    for (int a : actual) {
        for (int p : predicted) {
            if (a == p) {
                correct++;
                break;
            }
        }
    }

    pred_total_experts.fetch_add(static_cast<int64_t>(actual.size()), std::memory_order_relaxed);
    pred_correct_experts.fetch_add(correct, std::memory_order_relaxed);
    pred_total_layers.fetch_add(1, std::memory_order_relaxed);

    interval_pred_total.fetch_add(static_cast<int64_t>(actual.size()), std::memory_order_relaxed);
    interval_pred_correct.fetch_add(correct, std::memory_order_relaxed);
}

void MoeDispatchStats::tick_token() {
    int64_t tok = total_tokens.fetch_add(1, std::memory_order_relaxed) + 1;
    interval_tokens.fetch_add(1, std::memory_order_relaxed);

    if (report_interval > 0 && (tok % report_interval) == 0) {
        print_interval();
    }
}

void MoeDispatchStats::print_summary(const char * tag) const {
    int64_t dispatched = total_experts_dispatched.load(std::memory_order_relaxed);
    int64_t vram       = total_vram_hits.load(std::memory_order_relaxed);
    int64_t host       = total_host_hits.load(std::memory_order_relaxed);
    int64_t staging    = total_staging.load(std::memory_order_relaxed);
    int64_t cpu        = total_cpu_fallbacks.load(std::memory_order_relaxed);
    int64_t prefetch   = total_prefetch_hits.load(std::memory_order_relaxed);
    int64_t tokens     = total_tokens.load(std::memory_order_relaxed);
    int64_t layers     = total_layers.load(std::memory_order_relaxed);
    int64_t p_total    = pred_total_experts.load(std::memory_order_relaxed);
    int64_t p_correct  = pred_correct_experts.load(std::memory_order_relaxed);
    int64_t p_layers   = pred_total_layers.load(std::memory_order_relaxed);

    if (dispatched == 0) {
        return;
    }

    float vram_pct    = dispatched > 0 ? 100.0f * static_cast<float>(vram) / static_cast<float>(dispatched) : 0.0f;
    float host_pct    = dispatched > 0 ? 100.0f * static_cast<float>(host) / static_cast<float>(dispatched) : 0.0f;
    float staging_pct = dispatched > 0 ? 100.0f * static_cast<float>(staging) / static_cast<float>(dispatched) : 0.0f;
    float pred_pct    = p_total > 0 ? 100.0f * static_cast<float>(p_correct) / static_cast<float>(p_total) : 0.0f;
    float pcie_per_tok = tokens > 0 ? static_cast<float>(host + staging) / static_cast<float>(tokens) : 0.0f;

    GGML_LOG_INFO("\n[MOE-STATS %s] ===== MoE Dispatch Statistics =====\n", tag);
    GGML_LOG_INFO("[MOE-STATS %s] Tokens: %lld, Layers: %lld, Experts: %lld\n", tag,
                  (long long) tokens, (long long) layers, (long long) dispatched);
    GGML_LOG_INFO("[MOE-STATS %s] VRAM hits:  %lld (%.1f%%) - fast, no PCIe\n",
                  tag, (long long) vram, vram_pct);
    GGML_LOG_INFO("[MOE-STATS %s] Host hits:  %lld (%.1f%%) - PCIe streaming\n",
                  tag, (long long) host, host_pct);
    GGML_LOG_INFO("[MOE-STATS %s] Staging:    %lld (%.1f%%) - fresh upload\n",
                  tag, (long long) staging, staging_pct);
    GGML_LOG_INFO("[MOE-STATS %s] CPU miss:   %lld (%.0f%%) - fallback\n",
                  tag, (long long) cpu,
                  dispatched > 0 ? 100.0f * static_cast<float>(cpu) / static_cast<float>(dispatched) : 0.0f);
    GGML_LOG_INFO("[MOE-STATS %s] PCIe experts/token: %.1f (host + staging)\n", tag, pcie_per_tok);
    GGML_LOG_INFO("[MOE-STATS %s] Prefetch: %lld DMA-awaited\n", tag, (long long) prefetch);
    GGML_LOG_INFO("[MOE-STATS %s] Prediction: %lld/%lld correct (%.1f%%) across %lld layers\n",
                  tag, (long long) p_correct, (long long) p_total, pred_pct, (long long) p_layers);
    GGML_LOG_INFO("[MOE-STATS %s] ================================\n\n", tag);
}

void MoeDispatchStats::print_interval() {
    int64_t i_experts = interval_experts.exchange(0, std::memory_order_relaxed);
    int64_t i_vram    = interval_vram_hits.exchange(0, std::memory_order_relaxed);
    int64_t i_host    = interval_host_hits.exchange(0, std::memory_order_relaxed);
    int64_t i_staging = interval_staging.exchange(0, std::memory_order_relaxed);
    int64_t i_cpu     = interval_cpu_fallbacks.exchange(0, std::memory_order_relaxed);
    int64_t i_pt      = interval_pred_total.exchange(0, std::memory_order_relaxed);
    int64_t i_pc      = interval_pred_correct.exchange(0, std::memory_order_relaxed);
    int64_t i_tok     = interval_tokens.exchange(0, std::memory_order_relaxed);
    int64_t tok_total = total_tokens.load(std::memory_order_relaxed);

    if (i_experts == 0) {
        return;
    }

    float vram_pct = i_experts > 0 ? 100.0f * static_cast<float>(i_vram) / static_cast<float>(i_experts) : 0.0f;
    float host_pct = i_experts > 0 ? 100.0f * static_cast<float>(i_host) / static_cast<float>(i_experts) : 0.0f;
    float pcie_per_tok = i_tok > 0 ? static_cast<float>(i_host + i_staging) / static_cast<float>(i_tok) : 0.0f;
    float pred_pct  = i_pt > 0 ? 100.0f * static_cast<float>(i_pc) / static_cast<float>(i_pt) : 0.0f;

    GGML_LOG_INFO("[MOE-STATS T%lld] vram=%.1f%% host=%.1f%% miss=%lld (%lld/%lld) "
                  "pcie/tok=%.1f pred=%.1f%% (%lld tok)\n",
                  (long long) tok_total, vram_pct, host_pct, (long long) i_cpu,
                  (long long) (i_vram + i_host + i_staging), (long long) i_experts,
                  pcie_per_tok, pred_pct, (long long) i_tok);
}

MoeDispatchStats & get_moe_dispatch_stats(int device) {
    static MoeDispatchStats stats[GGML_SYCL_MAX_DEVICES];
    static std::once_flag   init_flags[GGML_SYCL_MAX_DEVICES];

    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        device = 0;
    }

    std::call_once(init_flags[device], [device]() {
        const char * env = std::getenv("GGML_SYCL_MOE_STATS_INTERVAL");
        if (env) {
            int val = std::atoi(env);
            if (val > 0) {
                stats[device].report_interval = val;
            }
        }
    });

    return stats[device];
}

}  // namespace ggml_sycl
