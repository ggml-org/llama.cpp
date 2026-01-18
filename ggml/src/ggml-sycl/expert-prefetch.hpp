// Expert Prefetcher for MoE models
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-eqa)
//
// This header defines the interface. Implementation in expert-prefetch.cpp.
// TDD: Interface defined by tests in test-sycl-expert-prefetch.cpp
//
// Hybrid Prefetch Strategy:
// - Predictive: Prefetch top-K experts based on router scores BEFORE ARGSORT
// - On-demand: If router selects unexpected expert, stream immediately
// - Sorted: Prefetch in descending score order for optimal cache access
// - Adaptive: Track prediction accuracy per layer, adjust prefetch count
//
// Priority Integration (via eviction-policy.hpp):
// - P2_HOT_EXPERT: Recently used, high confidence
// - P3_WARM_EXPERT: Prefetched/predicted, not yet confirmed
// - P4_COLD_EXPERT: Not recently used, evict first

#ifndef GGML_SYCL_EXPERT_PREFETCH_HPP
#define GGML_SYCL_EXPERT_PREFETCH_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// Forward declarations
class ChunkManager;
class EvictionPolicy;

// Single expert prediction with score and metadata
struct ExpertPrediction {
    uint32_t expert_id;  // Expert index
    float    score;      // Router score for this expert
    bool     selected;   // Was this expert actually selected by router?

    ExpertPrediction() : expert_id(0), score(0.0f), selected(false) {}
};

// Batch of predictions for one layer invocation
struct PrefetchBatch {
    uint32_t                      layer_id;     // Layer this batch is for
    std::vector<ExpertPrediction> predictions;  // Sorted by score descending

    // Stats for this batch
    size_t num_prefetched;  // How many experts we prefetched
    size_t num_selected;    // How many were actually selected (set by record_selections)
    size_t num_hits;        // Prefetched AND selected
    size_t num_misses;      // Selected but NOT prefetched (had to stream on-demand)

    PrefetchBatch() : layer_id(0), num_prefetched(0), num_selected(0), num_hits(0), num_misses(0) {}
};

// Per-layer accuracy statistics
struct LayerStats {
    std::atomic<uint64_t> total_predictions{ 0 };
    std::atomic<uint64_t> correct_predictions{ 0 };

    LayerStats() = default;

    // Copy constructor (load from source atomics)
    LayerStats(const LayerStats & other) :
        total_predictions(other.total_predictions.load(std::memory_order_relaxed)),
        correct_predictions(other.correct_predictions.load(std::memory_order_relaxed)) {}

    // Move constructor (load from source atomics)
    LayerStats(LayerStats && other) noexcept :
        total_predictions(other.total_predictions.load(std::memory_order_relaxed)),
        correct_predictions(other.correct_predictions.load(std::memory_order_relaxed)) {}

    // Copy assignment
    LayerStats & operator=(const LayerStats & other) {
        if (this != &other) {
            total_predictions.store(other.total_predictions.load(std::memory_order_relaxed), std::memory_order_relaxed);
            correct_predictions.store(other.correct_predictions.load(std::memory_order_relaxed),
                                      std::memory_order_relaxed);
        }
        return *this;
    }

    // Move assignment
    LayerStats & operator=(LayerStats && other) noexcept {
        if (this != &other) {
            total_predictions.store(other.total_predictions.load(std::memory_order_relaxed), std::memory_order_relaxed);
            correct_predictions.store(other.correct_predictions.load(std::memory_order_relaxed),
                                      std::memory_order_relaxed);
        }
        return *this;
    }

    // Calculate accuracy ratio
    float accuracy() const {
        uint64_t total = total_predictions.load(std::memory_order_relaxed);
        if (total == 0) {
            return 0.0f;
        }
        return static_cast<float>(correct_predictions.load(std::memory_order_relaxed)) / static_cast<float>(total);
    }

    // Minimum samples required before adaptive prefetch kicks in
    // This ensures we have stable accuracy measurement before adjusting
    // Set to 50 to allow ~12 batches of 4 predictions to establish baseline
    static constexpr uint64_t MIN_SAMPLES_FOR_ADAPTIVE = 50;

    // Adaptive prefetch count based on accuracy
    // Higher accuracy -> prefetch exactly top_k
    // Lower accuracy -> prefetch extra buffer
    // No data yet or insufficient samples -> prefetch exactly top_k
    size_t recommended_prefetch_count(size_t top_k) const {
        uint64_t total = total_predictions.load(std::memory_order_relaxed);
        // Need minimum samples before adaptive prefetch kicks in
        if (total < MIN_SAMPLES_FOR_ADAPTIVE) {
            return top_k;
        }

        float acc = accuracy();
        if (acc >= 0.95f) {
            return top_k;  // Very accurate, prefetch exactly K
        }
        if (acc >= 0.80f) {
            return top_k + 1;  // Good, small buffer
        }
        if (acc >= 0.60f) {
            return top_k + 2;  // Moderate, more buffer
        }
        return top_k + 4;      // Poor, significant buffer
    }
};

// Aggregate statistics across all layers
struct PrefetchStats {
    uint64_t total_predictions;
    uint64_t total_correct;
    float    overall_accuracy;

    PrefetchStats() : total_predictions(0), total_correct(0), overall_accuracy(0.0f) {}
};

// Main prefetcher class
// Manages predictive prefetching of MoE experts based on router scores
class ExpertPrefetcher {
  public:
    ExpertPrefetcher();
    ~ExpertPrefetcher();

    // Non-copyable, non-movable (contains atomics)
    ExpertPrefetcher(const ExpertPrefetcher &)             = delete;
    ExpertPrefetcher & operator=(const ExpertPrefetcher &) = delete;
    ExpertPrefetcher(ExpertPrefetcher &&)                  = delete;
    ExpertPrefetcher & operator=(ExpertPrefetcher &&)      = delete;

    // Configuration
    // num_layers: Number of MoE layers in the model
    // num_experts: Number of experts per layer
    // expert_size: Size of each expert in bytes (for streaming estimation)
    void configure(uint32_t num_layers, uint32_t num_experts, size_t expert_size);

    // Start prefetching experts for a layer based on router scores
    // Returns a PrefetchBatch with predictions sorted by score (highest first)
    // This initiates async DMA for top-K experts
    //
    // layer_id: Which MoE layer (0 to num_layers-1)
    // router_scores: Array of scores for each expert (size = num_experts)
    // num_experts: Number of experts (size of router_scores array)
    // top_k: How many experts will be selected by router
    PrefetchBatch start_prefetch(uint32_t layer_id, const float * router_scores, size_t num_experts, size_t top_k);

    // Get expert data - uses prefetch if available, else streams on-demand
    // Updates batch.num_hits or batch.num_misses accordingly
    //
    // layer_id: Which MoE layer
    // expert_id: Which expert to retrieve
    // batch: The PrefetchBatch from start_prefetch (updated with hit/miss stats)
    void * get_expert_data(uint32_t layer_id, uint32_t expert_id, PrefetchBatch & batch);

    // Record which experts were actually selected by the router
    // Updates accuracy tracking for adaptive prefetch count
    //
    // layer_id: Which MoE layer
    // selected_experts: Vector of expert IDs that were selected
    // batch: The PrefetchBatch from start_prefetch (updated with selection info)
    void record_selections(uint32_t layer_id, const std::vector<uint32_t> & selected_experts, PrefetchBatch & batch);

    // Get the prefetch order index for an expert in a batch
    // Returns the position (0 = first prefetched, 1 = second, etc.)
    // Returns -1 if expert was not prefetched
    int get_prefetch_order(const PrefetchBatch & batch, uint32_t expert_id) const;

    // Get prediction accuracy for a specific layer
    // Returns 0.0 if no data or invalid layer
    float get_layer_accuracy(uint32_t layer_id) const;

    // Get aggregate statistics across all layers
    PrefetchStats get_stats() const;

    // Reset all accuracy statistics
    void reset_stats();

    // Print statistics to stdout (for debugging)
    void print_stats() const;

  private:
    struct Impl;
    Impl * impl_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_EXPERT_PREFETCH_HPP
