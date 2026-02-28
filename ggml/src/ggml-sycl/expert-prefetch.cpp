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
    if (!cache_ || !dma_queue_ || !initialized_) {
        return;
    }

    // Build batch request, filtering out already in-flight experts.
    std::vector<std::pair<int, int>> batch;
    batch.reserve(expert_indices.size());

    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int eid : expert_indices) {
            expert_key key{ layer_idx, eid };
            if (inflight_.find(key) != inflight_.end()) {
                continue;  // Already in-flight
            }
            batch.push_back({ layer_idx, eid });
        }
    }

    if (batch.empty()) {
        return;
    }

    // Batch prefetch: Phase 1 (eviction planning) and Phase 2 (DMA submission)
    // happen inside ExpertCache with split-phase locking.
    PrefetchResult result = cache_->prefetch_batch_async(batch, *dma_queue_);

    // Track per-expert events in inflight_ map for granular await.
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & [ek, ev] : result.events) {
        if (inflight_.size() >= static_cast<size_t>(max_inflight_)) {
            gc_completed();
        }

        PrefetchRequest req;
        req.key       = ek;
        req.event     = ev;
        req.completed = false;
        inflight_[ek] = req;
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

    // Allocate a small device buffer for output scores
    float * scores_dev = sycl::malloc_device<float>(M, compute_q);
    if (!scores_dev) {
        GGML_LOG_WARN("[EXPERT-PREDICT] Failed to allocate device scores buffer, falling back to heuristic\n");
        return predict(next_layer_idx);
    }

    // Inline SYCL GEMV kernel:
    //   scores[j] = sum_k(gate_weights[j * K + k] * hidden_state[k])
    //   n_experts work groups, each computing one output element via
    //   subgroup reduction + SLM cross-subgroup accumulation.
    const int wg_size = std::min(256, ((K + 15) / 16) * 16);  // Clamp WG size, round up to 16
    const int n_wgs   = M;

    try {
        auto ev = compute_q.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> slm(sycl::range<1>(wg_size / 16 + 1), cgh);

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
        sycl::free(scores_dev, compute_q);
        return predict(next_layer_idx);
    }

    sycl::free(scores_dev, compute_q);

    // Top-K selection on host
    auto result = argsort_top_k(scores_host, n_experts_used_);

    GGML_SYCL_DEBUG("[EXPERT-PREDICT] Pre-gate layer=%d: top-%d experts = [", next_layer_idx, n_experts_used_);
    for (int i = 0; i < static_cast<int>(result.size()); i++) {
        GGML_SYCL_DEBUG("%s%d", i > 0 ? "," : "", result[i]);
    }
    GGML_SYCL_DEBUG("]\n");

    return result;
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

}  // namespace ggml_sycl
