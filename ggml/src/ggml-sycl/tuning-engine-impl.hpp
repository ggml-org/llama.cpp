//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TUNING_ENGINE_IMPL_HPP
#define GGML_SYCL_TUNING_ENGINE_IMPL_HPP

// TuningEngine: Progressive background tuning engine
//
// This header provides the TuningEngine class that:
// - Accumulates operation histograms during inference
// - Progresses through phases: OBSERVATION -> BENCHMARKING -> REFINEMENT
// - Provides O(1) parameter lookup with cold-start heuristics
// - Runs micro-benchmarks in background thread

#include "tuning-engine.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ggml_sycl_tuning {

// =============================================================================
// TuningPhase: Current state of the tuning engine
// =============================================================================
enum class TuningPhase {
    OBSERVATION,   // Collecting operation frequency data
    BENCHMARKING,  // Running micro-benchmarks on top patterns
    REFINEMENT     // Periodic re-testing and confidence decay
};

// =============================================================================
// TuningEngineConfig: Configuration for TuningEngine behavior
// =============================================================================
struct TuningEngineConfig {
    // Number of total observations before transitioning to BENCHMARKING
    uint64_t observation_threshold = 1000;

    // Number of top patterns to benchmark
    size_t top_k_patterns = 5;

    // Background thread sleep interval (ms) when idle
    uint32_t idle_sleep_ms = 10;

    // Confidence decay rate for refinement phase (not yet implemented)
    float confidence_decay = 0.99f;
};

// =============================================================================
// ObservationRecord: Queued observation for async processing
// =============================================================================
struct ObservationRecord {
    TuningKey   key;
    TunedParams params;
};

// =============================================================================
// TuningEngine: Main class for progressive background tuning
// =============================================================================
class TuningEngine {
  public:
    // Constructor with default configuration
    TuningEngine();

    // Constructor with custom configuration
    explicit TuningEngine(const TuningEngineConfig & config);

    // Destructor - stops background thread
    ~TuningEngine();

    // Non-copyable, non-movable
    TuningEngine(const TuningEngine &)             = delete;
    TuningEngine & operator=(const TuningEngine &) = delete;
    TuningEngine(TuningEngine &&)                  = delete;
    TuningEngine & operator=(TuningEngine &&)      = delete;

    // =========================================================================
    // Core API (called from hot path)
    // =========================================================================

    // Get tuned parameters for a given key
    // O(1) for cache hits, derives heuristics for cache misses
    // Thread-safe, non-blocking
    TunedParams get_params(const TuningKey & key);

    // Record an observation asynchronously (fire-and-forget)
    // Non-blocking (<100ns), queues work for background thread
    // Thread-safe
    void record_observation_async(const TuningKey & key, const TunedParams & used);

    // Record an observation with timing data (for confidence building)
    // Non-blocking, queues work for background thread
    // Thread-safe
    void record_observation_async(const TuningKey & key, const TunedParams & used, double time_ms);

    // Get confidence score for a key (0.0 = no data, 1.0 = high confidence)
    // Thread-safe, non-blocking
    double get_confidence(const TuningKey & key) const;

    // Get full statistics for a key
    // Thread-safe, non-blocking
    ObservationStats get_stats(const TuningKey & key) const;

    // =========================================================================
    // Control API
    // =========================================================================

    // Start background tuning thread (called automatically by constructor)
    void start_background_tuning();

    // Stop background tuning thread (called automatically by destructor)
    void stop();

    // Get current tuning phase
    TuningPhase current_phase() const { return phase_.load(std::memory_order_relaxed); }

    // =========================================================================
    // Testing/Debug API
    // =========================================================================

    // Get snapshot of histogram (for testing)
    std::unordered_map<TuningKey, uint64_t> get_histogram_snapshot() const;

    // Get top-K patterns by frequency
    std::vector<std::pair<TuningKey, uint64_t>> get_top_patterns(size_t k) const;

    // Insert cached params directly (for testing/preloading)
    void insert_cached_params(const TuningKey & key, const TunedParams & params, float tflops);

    // Get total observation count
    uint64_t total_observations() const { return total_observations_.load(std::memory_order_relaxed); }

  private:
    // Configuration
    TuningEngineConfig config_;

    // Cache for tuned parameters
    TuningCache cache_;

    // Current phase
    std::atomic<TuningPhase> phase_{ TuningPhase::OBSERVATION };

    // Histogram of operation counts
    mutable std::mutex                      histogram_mutex_;
    std::unordered_map<TuningKey, uint64_t> histogram_;

    // Total observation count
    std::atomic<uint64_t> total_observations_{ 0 };

    // Background thread
    std::thread       tuning_thread_;
    std::atomic<bool> should_stop_{ false };

    // Observation queue for async processing (legacy - frequency counting)
    mutable std::mutex            queue_mutex_;
    std::condition_variable       queue_cv_;
    std::queue<ObservationRecord> observation_queue_;

    // Performance observation queue for async processing (with timing)
    ObservationQueue perf_observation_queue_;

    // Performance statistics per key
    mutable std::mutex                              perf_stats_mutex_;
    std::unordered_map<TuningKey, ObservationStats> perf_stats_;

    // =========================================================================
    // Internal methods
    // =========================================================================

    // Background thread main loop
    void background_thread_main();

    // Process queued observations (frequency counting)
    void process_observations();

    // Process queued performance observations (timing stats)
    void process_perf_observations();

    // Check if should transition to next phase
    void check_phase_transition();

    // Derive heuristic params for cache miss
    TunedParams derive_heuristic_params(const TuningKey & key) const;

    // Run benchmarks for top patterns (Phase 2)
    void run_benchmarks();
};

}  // namespace ggml_sycl_tuning

#endif  // GGML_SYCL_TUNING_ENGINE_IMPL_HPP
