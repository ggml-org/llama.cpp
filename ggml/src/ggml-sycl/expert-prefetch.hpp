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
#include <memory>
#include <mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

#include "expert-cache.hpp"

namespace ggml_sycl {

// Tracks a single in-flight DMA prefetch operation.
struct PrefetchRequest {
    expert_key  key;
    sycl::event event;                 // DMA completion event from dma_queue_
    bool        completed = false;
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

    // Adaptive prefetch: schedules prefetch for first `threshold` experts at
    // full precision.  When n_miss_total > burst threshold AND mixed-precision
    // mode is active, remaining experts are NOT prefetched — they will be
    // dispatched to CPU compute via cpu_expert_mul_mat_int4() instead.
    // Returns the indices of experts that should use CPU compute (not prefetched).
    //
    // Usage:
    //   auto cpu_indices = prefetcher.hint_batch_adaptive(layer, experts, n_miss);
    //   // cpu_indices: experts to dispatch via cpu_expert_mul_mat_int4()
    //   // remaining: await() as normal (they were prefetched to VRAM)
    std::vector<int> hint_batch_adaptive(int layer_idx,
                                         const std::vector<int> & expert_indices,
                                         int n_miss_total);

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
    std::unique_ptr<sycl::queue> dma_queue_;   // OOQ for async H2D DMA (unique_ptr to avoid static init + enable leak-on-exit)
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

// ============================================================================
// ExpertPredictor: pre-attention expert prediction for MoE prefetching
// ============================================================================
//
// Predicts which experts will be needed AFTER attention completes, giving
// a full attention computation's worth of time (~17ms) to prefetch cache-miss
// experts. Runs on CPU only (no GPU involvement), <0.5ms.
//
// Heuristic (no learned predictor):
//   1. Reuse last token's experts for the same layer (~70% accuracy due to
//      expert access temporal locality)
//   2. Fill remaining slots from global frequency table (experts most commonly
//      activated across all tokens)
//
// Integration: after each layer's attention, call predict() for the next MoE
// layer and feed results to ExpertPrefetcher::hint_batch().
//
// Accuracy tracking: record_actual() compares predictions vs router selections,
// maintains a rolling hit rate.
//
// Env var: GGML_SYCL_EXPERT_PREDICT=1 enables prediction (default: ON when
// expert cache is active).
//
class ExpertPredictor {
  public:
    ExpertPredictor() = default;
    ~ExpertPredictor();

    // Non-copyable, non-movable
    ExpertPredictor(const ExpertPredictor &)             = delete;
    ExpertPredictor & operator=(const ExpertPredictor &) = delete;
    ExpertPredictor(ExpertPredictor &&)                  = delete;
    ExpertPredictor & operator=(ExpertPredictor &&)      = delete;

    // Initialize the predictor.
    //   n_layers:       total number of transformer layers
    //   n_experts:      total experts per MoE layer
    //   n_experts_used: experts activated per token (top-K)
    void init(int n_layers, int n_experts, int n_experts_used);

    // Predict which experts will be needed for a given layer.
    // Returns up to n_experts_used predicted expert indices.
    // hidden_state is unused in heuristic mode (reserved for future learned predictor).
    std::vector<int> predict(int layer_idx, const float * hidden_state = nullptr);

    // Pre-gated router: compute actual gate scores 1 layer ahead using a
    // small inline SYCL GEMV kernel. Returns top-K expert indices from the
    // real router gate weights, giving ~3ms of DMA prefetch overlap.
    //
    // Falls back to heuristic predict() if gate_weights or hidden_state is
    // nullptr, or if gate weight pointers are not registered.
    //
    //   next_layer_idx: sequential layer index for the NEXT MoE layer
    //   gate_weights:   device ptr to f32 gate weights [n_experts x n_embd]
    //   hidden_state:   device ptr to f32 hidden state [1 x n_embd]
    //   compute_q:      SYCL queue for kernel submission + D2H copy
    std::vector<int> predict_pregate(int           next_layer_idx,
                                     const void *  gate_weights,
                                     const void *  hidden_state,
                                     sycl::queue & compute_q);

    // Record actual expert selections from the router for accuracy tracking.
    // Called after MUL_MAT_ID with the real expert indices chosen by the gating network.
    void record_actual(int layer_idx, const std::vector<int> & actual_experts);

    // Register gate weight pointer for a specific layer.
    // Called during moe_hybrid_init_once() after scanning graph for ffn_gate_inp tensors.
    void register_gate_weights(int layer_idx, const void * gate_ptr, int n_embd);

    // Check if pre-gated routing is available for a given layer.
    bool has_gate_weights(int layer_idx) const;

    // Multi-layer lookahead prediction: predict experts for layers L+1..L+depth.
    // Returns a vector of (target_layer_idx, predicted_experts) pairs.
    // Uses predict_pregate() for each target layer with correct gate weights.
    std::vector<std::pair<int, std::vector<int>>> predict_multi_layer(
        int current_seq_layer,
        const void * hidden_state,
        sycl::queue & compute_q);

    // Return the configured prediction depth (layers ahead to predict).
    int predict_depth() const { return predict_depth_; }

    // Statistics (rolling window of last ACCURACY_WINDOW predictions)
    float hit_rate() const;       // Rolling prediction accuracy (0.0 - 1.0)
    int   window_size() const;    // Current window sample count (up to ACCURACY_WINDOW)
    int   window_hits() const;    // Hits within current window
    bool  is_active() const { return initialized_; }

  private:
    bool initialized_ = false;
    int  n_layers_      = 0;
    int  n_experts_     = 0;
    int  n_experts_used_ = 0;
    int  predict_depth_  = 3;  // Number of layers to predict ahead (default: 3)

    // Per-layer last-token expert selections.
    // last_experts_[layer] = vector of expert indices used by previous token.
    std::vector<std::vector<int>> last_experts_;

    // Global frequency table: freq_table_[layer][expert] = access count.
    std::vector<std::vector<uint32_t>> freq_table_;

    // Last prediction per layer (for accuracy comparison).
    std::vector<std::vector<int>> last_prediction_;

    // Pre-gated router: cache of gate weight pointers per layer.
    // gate_weight_ptrs_[seq_layer_idx] = device ptr to f32 gate weights.
    // Empty if model has no MoE or gate weights not yet registered.
    std::vector<const void *> gate_weight_ptrs_;
    int                       n_embd_ = 0;  // Embedding dimension for GEMV kernel

    // Pre-allocated device buffer for predict_pregate() GEMV output scores.
    // Avoids sycl::malloc_device/free per call (3 calls per MoE dispatch with 3-layer lookahead).
    float *      scores_dev_    = nullptr;
    int          scores_dev_n_  = 0;       // Number of floats allocated
    sycl::queue * scores_queue_ = nullptr;  // Queue used for allocation (for deallocation)

    // Rolling accuracy stats (last 100 predictions).
    static constexpr int ACCURACY_WINDOW = 100;
    int accuracy_hits_  = 0;
    int accuracy_total_ = 0;

    // Circular buffer for rolling window eviction.
    // uint8_t avoids std::vector<bool> specialization issues.
    std::vector<uint8_t> accuracy_ring_;
    int                  accuracy_ring_pos_ = 0;

    mutable std::mutex mutex_;

    // Get top-K experts by frequency for a layer (excluding already-selected ones).
    std::vector<int> top_k_by_freq(int layer_idx, const std::vector<int> & exclude, int k) const;
};

// Check if expert prediction is enabled via environment variable.
// Default: ON (returns true unless GGML_SYCL_EXPERT_PREDICT=0).
bool ggml_sycl_expert_predict_enabled();

}  // namespace ggml_sycl
