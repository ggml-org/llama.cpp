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
// compute queue. hint() submits memcpy on dma_queue_ and stores the returned
// sycl::event in a prefetch_request. await() waits on the per-expert event
// for granular synchronization.
//
// L2 coherency: BCS H2D to malloc_device completes BEFORE the kernel
// launches because await() is called before kernel submission, and the
// in-order compute queue serializes after await().

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

#include "expert-key.hpp"

namespace ggml_sycl {

// Tracks a single in-flight DMA prefetch operation.
struct prefetch_request {
    expert_key  key;
    sycl::event event;                 // DMA completion event from dma_queue_
    void *      device_ptr = nullptr;  // VRAM destination of the H2D DMA
    int         pool_slot  = -1;       // Index into vram_pool_ (-1 = no slot)
    bool        completed  = false;
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
    void init(sycl::queue & compute_q);

    // Shut down: cancel all in-flight prefetches, wait for completion.
    void shutdown();

    // Schedule async prefetch of a single expert (non-blocking).
    // Submits H2D memcpy on dma_queue_.
    // Returns true if a new prefetch was scheduled.
    // Returns false if: already cached in VRAM, already in-flight, no capacity,
    //                   or expert is not registered.
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
    // cached ptr from the unified cache. Returns nullptr if not registered.
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
    std::unique_ptr<sycl::queue> dma_queue_;    // OOQ for async H2D DMA (unique_ptr to avoid static init + enable leak-on-exit)
    int                          prefetch_depth_ = 2;  // Default: 2 layers ahead
    bool                         initialized_   = false;

    // Max concurrent DMA operations. MoE models activate up to 8 experts
    // per layer, so 8 in-flight requests covers a full layer's worth of misses.
    static constexpr int max_inflight_ = 8;

    // In-flight prefetch tracking. Key = expert_key.
    std::unordered_map<expert_key, prefetch_request, expert_key_hash> inflight_;

    // VRAM prefetch pool: ring buffer of pre-allocated device memory slots.
    // Each slot holds one expert's worth of weight data.
    // Allocated lazily on first hint() with weight_bytes > 0.
    struct vram_slot {
        void * ptr  = nullptr;
        bool   free = true;
    };
    std::vector<vram_slot> vram_pool_;
    size_t                 vram_slot_bytes_ = 0;  // Size of each pool slot

    // Acquire a free VRAM slot. Returns slot index or -1 if none available.
    int acquire_vram_slot();
    // Release a VRAM slot back to the pool.
    void release_vram_slot(int slot);

    mutable std::mutex mutex_;

    // Stats
    int completed_count_ = 0;
    int prefetch_hits_   = 0;  // Experts found already in VRAM (no DMA needed)

    // Garbage-collect completed requests to free tracking slots.
    void gc_completed();

    // Check if we have room for more in-flight requests.
    // Counts only active (non-completed) entries.
    bool has_capacity() const;

    // Internal locked implementation of hint(). Caller must hold mutex_.
    bool hint_locked(int layer_idx, int expert_idx);

    // Set when prediction hit rate drops below threshold; disables prefetching.
    std::atomic<bool> prefetch_disabled_{false};
};

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

    // Returns true when prediction accuracy is too low for useful prefetching.
    // Checked by ExpertPrefetcher to short-circuit hint().
    bool is_prefetch_disabled() const;

    // Get top-K experts by frequency for a layer (excluding already-selected ones).
    // Public wrapper for hot expert pool population.
    std::vector<int> get_top_k_by_freq(int layer_idx, const std::vector<int> & exclude, int k) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return top_k_by_freq(layer_idx, exclude, k);
    }

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

    // Rolling accuracy stats (last ACCURACY_WINDOW predictions).
    static constexpr int ACCURACY_WINDOW = 100;
    int accuracy_hits_   = 0;
    int window_total_    = 0;  // Number of samples in the rolling accuracy window (up to ACCURACY_WINDOW)

    // Set when hit rate drops below threshold; signals prefetcher to disable.
    std::atomic<bool> prefetch_disabled_{false};

    // Circular buffer for rolling window eviction.
    // uint8_t avoids std::vector<bool> specialization issues.
    std::vector<uint8_t> accuracy_ring_;
    int                  accuracy_ring_pos_ = 0;

    mutable std::mutex mutex_;

    // Get top-K experts by frequency for a layer (excluding already-selected ones).
    std::vector<int> top_k_by_freq(int layer_idx, const std::vector<int> & exclude, int k) const;
};

// ============================================================================
// HotExpertPool: dedicated VRAM cache for most-accessed MoE experts
// ============================================================================
//
// Allocates a fixed-size VRAM pool for caching hot experts identified by the
// ExpertPredictor's frequency table. After a warmup period, the top-K most
// frequent experts per MoE tensor are promoted from host-pinned to VRAM,
// eliminating PCIe streaming for the most commonly accessed weights.
//
// This pool is SEPARATE from the unified cache - it doesn't interfere with
// the cache's budget accounting or eviction. It uses VRAM headroom that's
// free after the unified cache has finished loading non-expert weights.
//
// Env var: GGML_SYCL_HOT_EXPERTS=N sets the number of hot experts per
// tensor (default: 2). Set to 0 to disable hot expert VRAM caching.
// GGML_SYCL_HOT_EXPERT_BUDGET_MB=N sets the VRAM budget in MB (default: 512).
//
class HotExpertPool {
  public:
    HotExpertPool() = default;
    ~HotExpertPool();

    // Non-copyable, non-movable
    HotExpertPool(const HotExpertPool &)             = delete;
    HotExpertPool & operator=(const HotExpertPool &) = delete;
    HotExpertPool(HotExpertPool &&)                  = delete;
    HotExpertPool & operator=(HotExpertPool &&)      = delete;

    // Initialize the pool.
    //   queue: SYCL queue for device allocation and memcpy
    //   budget_mb: VRAM budget in MB for this pool (default from env)
    //   expert_bytes: size of one expert-tensor in bytes
    void init(sycl::queue & queue, size_t expert_bytes);

    // Promote the top-K experts for a given tensor to VRAM.
    // Called after warmup, triggered by the predictor's frequency data.
    //   tensor_name_hash: FNV hash identifying the MoE weight tensor
    //   expert_ids: sorted list of expert IDs to promote (most frequent first)
    //   host_ptrs: corresponding host-pinned pointers for each expert
    // Returns number of experts actually promoted (may be less if pool full).
    int promote(int tensor_name_hash,
                const std::vector<int> & expert_ids,
                const std::vector<void *> & host_ptrs);

    // Look up a hot expert. Returns VRAM pointer if promoted, nullptr otherwise.
    void * lookup(int tensor_name_hash, int expert_id) const;

    // Check if pool is initialized and has capacity.
    bool is_active() const { return initialized_; }

    // Stats
    int promoted_count() const { return promoted_count_; }
    int capacity() const { return total_slots_; }

    // How many hot experts per tensor (from env or default)
    static int hot_experts_per_tensor();

  private:
    bool         initialized_  = false;
    sycl::queue * queue_       = nullptr;
    void *       pool_base_    = nullptr;   // Base of VRAM allocation
    size_t       pool_size_    = 0;         // Total bytes allocated
    size_t       expert_bytes_ = 0;         // Bytes per expert slot
    int          total_slots_  = 0;         // Number of expert slots
    int          promoted_count_ = 0;       // Currently promoted experts
    int          next_slot_    = 0;         // Next free slot index

    // Lookup: (tensor_hash, expert_id) -> slot index
    // Stored as (tensor_hash << 16 | expert_id) for fast lookup
    struct hot_entry {
        int64_t key;           // composite key
        int     slot;          // index into pool
        void *  vram_ptr;      // pointer within pool_base_
    };
    std::vector<hot_entry> entries_;

    mutable std::mutex mutex_;
};

// ============================================================================
// MoE Dispatch Statistics: per-token cache hit/miss + prediction accuracy
// ============================================================================
//
// Tracks detailed per-expert-level hit rates (not just binary per-layer).
// Integrated at the dispatch partition point in ggml_sycl_mul_mat_id() to
// measure actual cache residency and prediction overlap.
//
// Env var: GGML_SYCL_MOE_STATS=1 enables stats collection (default: ON when
// expert prediction is active). GGML_SYCL_MOE_STATS_INTERVAL=N controls
// reporting interval in tokens (default: 10).
//
struct MoeDispatchStats {
    // Cumulative counters (lifetime)
    std::atomic<int64_t> total_experts_dispatched{0};  // Total expert dispatches
    std::atomic<int64_t> total_vram_hits{0};           // Expert in VRAM cache (fastest)
    std::atomic<int64_t> total_host_hits{0};           // Expert in host-pinned cache (PCIe streaming)
    std::atomic<int64_t> total_staging{0};             // Expert freshly staged from host (IN_PROGRESS)
    std::atomic<int64_t> total_cpu_fallbacks{0};       // Expert fell to CPU (cache miss)
    std::atomic<int64_t> total_prefetch_hits{0};       // Expert was in-flight prefetched and awaited
    std::atomic<int64_t> total_tokens{0};              // Token counter
    std::atomic<int64_t> total_layers{0};              // Layer dispatch counter

    // Per-expert prediction accuracy (cumulative)
    std::atomic<int64_t> pred_total_experts{0};     // Total experts in actual selections
    std::atomic<int64_t> pred_correct_experts{0};   // Experts that were in prediction set
    std::atomic<int64_t> pred_total_layers{0};      // Layers where prediction was available

    // Interval counters (reset each report)
    std::atomic<int64_t> interval_experts{0};
    std::atomic<int64_t> interval_vram_hits{0};
    std::atomic<int64_t> interval_host_hits{0};
    std::atomic<int64_t> interval_staging{0};
    std::atomic<int64_t> interval_cpu_fallbacks{0};
    std::atomic<int64_t> interval_pred_total{0};
    std::atomic<int64_t> interval_pred_correct{0};
    std::atomic<int64_t> interval_tokens{0};

    // Reporting interval in tokens
    int report_interval = 10;

    // Record a dispatch partition for one MUL_MAT_ID call.
    // n_vram: experts found in VRAM (device-resident)
    // n_host: experts found in host-pinned cache (PCIe streaming)
    // n_staging: experts freshly being staged (IN_PROGRESS)
    // n_miss: cache misses (nullptr, falls to CPU)
    // n_prefetched: experts from async DMA prefetch
    void record_dispatch(int n_vram, int n_host, int n_staging, int n_miss, int n_prefetched);

    // Record prediction accuracy for one layer: predicted vs actual expert sets
    void record_prediction_accuracy(const std::vector<int> & predicted,
                                    const std::vector<int> & actual);

    // Called once per token (after all layers) to check if we should report
    void tick_token();

    // Print summary statistics
    void print_summary(const char * tag = "FINAL") const;

    // Print interval statistics and reset interval counters
    void print_interval();

    // Check if stats collection is enabled
    static bool enabled();
};

// Global stats instance (one per device, indexed by device id)
MoeDispatchStats & get_moe_dispatch_stats(int device);

// Check if expert prediction is enabled via environment variable.
// Default: ON (returns true unless GGML_SYCL_EXPERT_PREDICT=0).
bool ggml_sycl_expert_predict_enabled();

}  // namespace ggml_sycl
