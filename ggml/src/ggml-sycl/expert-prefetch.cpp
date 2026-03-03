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
    // Free VRAM pool slots.
    if (!ggml_sycl_is_shutting_down() && dma_queue_) {
        for (auto & slot : vram_pool_) {
            if (slot.ptr) {
                sycl::free(slot.ptr, *dma_queue_);
                slot.ptr = nullptr;
            }
        }
    }
    // During static destruction, intentionally leak the queue handle + VRAM pool.
    // The OS reclaims all process memory at exit.
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
    GGML_LOG_INFO("[SYCL] Expert prefetcher shut down (prefetched=%d, already_cached=%d)\n",
                  completed_count_, prefetch_hits_);
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
bool ExpertPrefetcher::hint_locked(int layer_idx, int expert_idx) {
    expert_key key{ layer_idx, expert_idx };

    // Already in-flight or completed — skip.
    if (inflight_.count(key)) {
        return false;
    }

    // Check capacity (GC first to free completed slots).
    gc_completed();
    if (!has_capacity()) {
        return false;
    }

    // Look up expert in placement table.
    auto & ptable = get_expert_placement_table();
    if (!ptable.is_initialized()) {
        return false;
    }

    auto placement = ptable.get(layer_idx, expert_idx);

    // Already in VRAM — nothing to prefetch.
    if (placement.device_ptr) {
        prefetch_hits_++;
        return false;
    }

    // No host pointer — cannot prefetch.
    if (!placement.host_ptr || placement.weight_bytes == 0) {
        return false;
    }

    // Lazily allocate VRAM pool on first use, sized to this expert's weight_bytes.
    if (vram_pool_.empty()) {
        vram_slot_bytes_ = placement.weight_bytes;
        vram_pool_.resize(max_inflight_);
        size_t allocated = 0;
        for (auto & slot : vram_pool_) {
            try {
                slot.ptr  = sycl::malloc_device(vram_slot_bytes_, *dma_queue_);
                slot.free = (slot.ptr != nullptr);
                if (slot.ptr) { allocated++; }
            } catch (const sycl::exception &) {
                slot.ptr  = nullptr;
                slot.free = false;
            }
        }
        GGML_LOG_INFO("[SYCL] Expert prefetch VRAM pool: %zu/%d slots (%.1f MB each)\n",
                      allocated, max_inflight_, vram_slot_bytes_ / (1024.0 * 1024.0));
        if (allocated == 0) {
            GGML_LOG_WARN("[SYCL] Expert prefetch VRAM pool: ALL slots failed to allocate — "
                          "prefetching permanently disabled\n");
            prefetch_disabled_ = true;
            return false;
        }
    }

    // Skip if expert is larger than pool slots (model changed mid-run).
    if (placement.weight_bytes > vram_slot_bytes_) {
        return false;
    }

    // Acquire a VRAM slot.
    int slot = acquire_vram_slot();
    if (slot < 0) {
        return false;
    }

    // Submit async H2D DMA on the OOQ.
    void * dst = vram_pool_[slot].ptr;
    try {
        sycl::event ev = dma_queue_->memcpy(dst, placement.host_ptr, placement.weight_bytes);

        prefetch_request req;
        req.key        = key;
        req.event      = ev;
        req.device_ptr = dst;
        req.pool_slot  = slot;
        req.completed  = false;
        inflight_[key] = std::move(req);

        GGML_SYCL_DEBUG("[PREFETCH] hint L%d E%d: H2D %.1f KB -> slot %d\n",
                        layer_idx, expert_idx, placement.weight_bytes / 1024.0, slot);
        return true;
    } catch (const sycl::exception & e) {
        release_vram_slot(slot);
        GGML_LOG_WARN("[SYCL] Prefetch H2D failed for L%d E%d: %s\n",
                      layer_idx, expert_idx, e.what());
        return false;
    }
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
    int budget = max_inflight_ - static_cast<int>(inflight_.size());

    int scheduled = 0;
    for (int eid : expert_indices) {
        // Schedule prefetch when: (1) we have remaining DMA budget, AND
        // (2) the total miss count across all experts is within our capacity.
        // When n_miss_total > max_inflight_, even the first batch of experts
        // would saturate DMA bandwidth, so overflow to CPU instead.
        if (scheduled < budget && n_miss_total <= max_inflight_) {
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

    // Phase 1: Check if already completed or extract event for waiting.
    // Release mutex before blocking on event.wait() to avoid deadlock
    // (another thread calling hint() needs the lock).
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
            auto it = inflight_.find(key);
            if (it != inflight_.end()) {
                release_vram_slot(it->second.pool_slot);
                inflight_.erase(it);
            }
            return nullptr;
        }
    }

    // Phase 3: Re-acquire lock and update state.
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = inflight_.find(key);
    if (it == inflight_.end()) {
        return nullptr;
    }

    if (!it->second.completed) {
        it->second.completed = true;
        completed_count_++;

        // Update placement table so the dispatch path finds device_ptr.
        auto & ptable = get_expert_placement_table();
        if (ptable.is_initialized()) {
            ptable.set_device_ptr(layer_idx, expert_idx, 0, it->second.device_ptr);
        }

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
    if (dma_queue_) {
        try {
            dma_queue_->wait();
        } catch (const sycl::exception &) {
            // Best effort during shutdown.
        }
    }

    // Release all pool slots and clear tracking.
    for (auto & [key, req] : inflight_) {
        release_vram_slot(req.pool_slot);
    }
    inflight_.clear();
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
    // Remove completed tracking entries and release their VRAM pool slots.
    //
    // Safety: gc_completed() is called from hint_locked(), which runs for
    // future layers (L+1..L+depth). By the time hint_locked(L+1) runs,
    // layer L-1's GPU dispatch has completed because:
    //   1. ggml_sycl_mul_mat_id() runs synchronously per layer
    //   2. dispatch_cpu_and_scatter() includes stream->wait() before return
    //   3. await() is called before kernel submission in ggml_sycl_mul_mat_id()
    // So completed entries from layer L-1 are safe to gc because the GPU
    // has consumed their pool slot data.
    //
    // Note: entries become completed in await() for layer L, and are gc'd
    // by hint_locked() for layer L+1 or later. Since hint runs 1+ layers
    // ahead, there's at least one full dispatch cycle between completed and gc.
    //
    // SAFETY INVARIANT: This guarantee depends on the synchronous call chain:
    //   ggml_sycl_mul_mat_id() -> await() -> [kernel dispatch] -> stream->wait()
    // If this call chain becomes async, gc_completed() must be revisited.
    //
    // Callers: hint_locked(), hint_batch_adaptive() — all hold mutex_ before calling.
    auto it = inflight_.begin();
    while (it != inflight_.end()) {
        if (it->second.completed) {
            // Clear placement table device_ptr since the pool slot
            // will be recycled for a different expert.
            auto & ptable = get_expert_placement_table();
            if (ptable.is_initialized()) {
                ptable.set_device_ptr(it->second.key.layer, it->second.key.expert_id, 0, nullptr);
            }
            release_vram_slot(it->second.pool_slot);
            it = inflight_.erase(it);
        } else {
            ++it;
        }
    }
}

bool ExpertPrefetcher::has_capacity() const {
    // Called with mutex_ held.
    // Count only active (non-completed) entries. Completed-but-not-gc'd entries
    // should not count against capacity since their DMA slots are reclaimable.
    int active = 0;
    for (const auto & [k, req] : inflight_) {
        if (!req.completed) {
            active++;
        }
    }
    return active < max_inflight_;
}

int ExpertPrefetcher::acquire_vram_slot() {
    // Called with mutex_ held.
    for (int i = 0; i < static_cast<int>(vram_pool_.size()); i++) {
        if (vram_pool_[i].free && vram_pool_[i].ptr) {
            vram_pool_[i].free = false;
            return i;
        }
    }
    return -1;
}

void ExpertPrefetcher::release_vram_slot(int slot) {
    // Called with mutex_ held.
    if (slot >= 0 && slot < static_cast<int>(vram_pool_.size())) {
        vram_pool_[slot].free = true;
    }
}

// ============================================================================
// ExpertPredictor: pre-attention expert prediction
// ============================================================================

ExpertPredictor::~ExpertPredictor() {
    // Free pre-allocated device scores buffer.
    // Skip during static destruction (SYCL context may be invalid).
    if (scores_dev_ && scores_queue_ && !ggml_sycl_is_shutting_down()) {
        try {
            sycl::free(scores_dev_, *scores_queue_);
        } catch (...) {
            // SYCL runtime may be partially torn down
        }
    }
    scores_dev_   = nullptr;
    scores_dev_n_ = 0;
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
        if (window_total_ >= ACCURACY_WINDOW && accuracy_ring_pos_ == 0) {
            float rate = static_cast<float>(accuracy_hits_) / static_cast<float>(window_total_);
            GGML_LOG_INFO("[EXPERT-PREDICT] accuracy=%.1f%% (hits=%d, window=%d)\n",
                          rate * 100.0f, accuracy_hits_, window_total_);

            // Disable prefetching when prediction accuracy drops below 30%.
            // At this accuracy, most prefetches are wasted DMA bandwidth.
            static constexpr float DISABLE_THRESHOLD = 0.3f;
            if (!prefetch_disabled_ && rate < DISABLE_THRESHOLD) {
                prefetch_disabled_ = true;
                GGML_LOG_WARN("[EXPERT-PREDICT] hit rate %.1f%% below threshold %.0f%% — "
                              "prefetching disabled\n",
                              rate * 100.0f, DISABLE_THRESHOLD * 100.0f);
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
        if (scores_dev_ && scores_queue_) {
            sycl::free(scores_dev_, *scores_queue_);
        }
        scores_dev_   = sycl::malloc_device<float>(M, compute_q);
        scores_dev_n_ = M;
        scores_queue_ = &compute_q;
        if (!scores_dev_) {
            scores_dev_n_ = 0;
            GGML_LOG_WARN("[EXPERT-PREDICT] Failed to allocate device scores buffer, falling back to heuristic\n");
            return predict(next_layer_idx);
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

}  // namespace ggml_sycl
