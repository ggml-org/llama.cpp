//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TUNING_ENGINE_HPP
#define GGML_SYCL_TUNING_ENGINE_HPP

// Tuning Engine: Auto-tuning infrastructure for SYCL kernel dispatch
//
// This header provides data structures for:
// - TunedParams: Kernel tuning parameters (tile sizes, workgroup, etc.)
// - BatchBucket: M dimension bucketing for batch-aware dispatch
// - TuningKey: Lookup key combining quant_type, batch_bucket, K, N
// - TuningEntry: Cache entry with key + params + metrics
// - TuningCache: Thread-safe O(1) lookup cache
// - TuningObservation: Runtime performance observation
// - ObservationStats: Online statistics with Welford's algorithm
// - ObservationQueue: Thread-safe observation queue

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <unordered_map>

namespace ggml_sycl_tuning {

// =============================================================================
// TunedParams: Kernel tuning parameters
// =============================================================================
struct TunedParams {
    // Tile dimensions for tiled algorithms
    uint16_t tile_m = 0;  // M dimension tile size (batch/output rows)
    uint16_t tile_n = 0;  // N dimension tile size (hidden dim/output cols)
    uint16_t tile_k = 0;  // K dimension tile size (reduction dim)

    // Workgroup configuration
    uint16_t workgroup_size = 0;  // Total workgroup size (WG_X * WG_Y)

    // Memory configuration
    uint8_t slm_kb         = 0;  // Shared local memory in KB
    uint8_t prefetch_depth = 0;  // Prefetch depth (0 = none, 1-4 typical)

    // Hardware feature flags
    bool use_dpas = false;  // Use XMX/dpas instructions

    // Layout mode (maps to reorder_mode enum values)
    // 0=NONE, 1=SOA, 2=COALESCED, 3=XMX_COALESCED, 4=XMX_GEMM_TILED
    uint8_t layout_mode = 0;

    // Confidence score for these parameters (0.0 to 1.0)
    // Updated by observation stats or decayed during transfer learning
    float confidence = 0.0f;
};

// =============================================================================
// BatchBucket: M dimension bucketing for batch-aware kernel dispatch
// =============================================================================
// Different batch sizes have very different optimal configurations:
// - SINGLE (M=1): Pure memory-bound, maximize bandwidth
// - SMALL (M=2-8): Transitional, may benefit from small tiles
// - MEDIUM (M=9-64): Can utilize some compute, moderate tiles
// - LARGE (M=65-128): Compute starts to dominate
// - XLARGE (M>128): Fully compute-bound, large XMX tiles optimal

enum class BatchBucket : uint8_t {
    SINGLE = 0,  // M = 1 (single token generation)
    SMALL  = 1,  // M = 2-8 (small batch, mostly memory-bound)
    MEDIUM = 2,  // M = 9-64 (transitional)
    LARGE  = 3,  // M = 65-128 (compute starts to dominate)
    XLARGE = 4,  // M > 128 (fully compute-bound)
};

// Helper function to map batch size (M) to bucket
inline BatchBucket bucket_for_batch(int M) {
    if (M <= 1) {
        return BatchBucket::SINGLE;
    }
    if (M <= 8) {
        return BatchBucket::SMALL;
    }
    if (M <= 64) {
        return BatchBucket::MEDIUM;
    }
    if (M <= 128) {
        return BatchBucket::LARGE;
    }
    return BatchBucket::XLARGE;
}

// =============================================================================
// TuningKey: Cache lookup key
// =============================================================================
// Combines all factors that affect optimal kernel parameters:
// - quant_type: Different types have different memory patterns
// - batch_bucket: Batch size regime affects compute/memory balance
// - K: Reduction dimension (affects tile sizing)
// - N: Output columns (affects parallelization strategy)

struct TuningKey {
    int32_t     quant_type;    // GGML_TYPE_* enum value
    BatchBucket batch_bucket;  // Batch size bucket
    int32_t     K;             // Reduction dimension (typically hidden_dim)
    int32_t     N;             // Output columns (typically n_vocab or hidden_dim)

    bool operator==(const TuningKey & other) const {
        return quant_type == other.quant_type && batch_bucket == other.batch_bucket && K == other.K && N == other.N;
    }
};

}  // namespace ggml_sycl_tuning

// Hash specialization for TuningKey (must be in std namespace)
namespace std {
template <> struct hash<ggml_sycl_tuning::TuningKey> {
    size_t operator()(const ggml_sycl_tuning::TuningKey & key) const noexcept {
        // Combine all fields using a standard hash combining technique
        // Based on boost::hash_combine pattern
        size_t h            = 0;
        auto   hash_combine = [&h](size_t value) {
            h ^= value + 0x9e3779b9 + (h << 6) + (h >> 2);
        };

        hash_combine(std::hash<int32_t>{}(key.quant_type));
        hash_combine(std::hash<uint8_t>{}(static_cast<uint8_t>(key.batch_bucket)));
        hash_combine(std::hash<int32_t>{}(key.K));
        hash_combine(std::hash<int32_t>{}(key.N));

        return h;
    }
};
}  // namespace std

namespace ggml_sycl_tuning {

// =============================================================================
// TuningEntry: Cache entry with key + params + metrics
// =============================================================================
struct TuningEntry {
    TuningKey   key;
    TunedParams params;

    // Performance metrics from tuning runs
    float   measured_tflops = 0.0f;  // Measured throughput in TFLOPS
    int64_t timestamp       = 0;     // Unix timestamp of measurement
};

// =============================================================================
// TuningObservation: Runtime performance observation
// =============================================================================
// Captures actual kernel execution data for building confidence over time.
// Observations are queued asynchronously and processed in background.

struct TuningObservation {
    TuningKey   key;               // Operation identification
    TunedParams params;            // Parameters used for this execution
    double      time_ms    = 0.0;  // Kernel execution time in milliseconds
    int64_t     batch_size = 0;    // Actual batch size (M dimension)
    int64_t     flops      = 0;    // Estimated FLOPS for this operation
    uint64_t    timestamp  = 0;    // When recorded (steady_clock nanoseconds)
};

// =============================================================================
// ObservationStats: Online statistics with Welford's algorithm
// =============================================================================
// Tracks running mean, variance, and confidence using Welford's numerically
// stable online algorithm. Confidence is computed from sample count and
// coefficient of variation (CV = stddev / mean).

struct ObservationStats {
    double mean_time_ms = 0.0;  // Running mean of execution time
    double variance     = 0.0;  // Running variance (using Welford's algorithm)
    int    sample_count = 0;    // Number of observations
    double confidence   = 0.0;  // Confidence score 0.0 to 1.0

    // Update statistics with a new observation using Welford's online algorithm.
    // This is numerically stable for computing variance incrementally.
    void update(double new_time) {
        sample_count++;
        if (sample_count == 1) {
            // First sample: mean = value, variance = 0
            mean_time_ms = new_time;
            variance     = 0.0;
        } else {
            // Welford's online algorithm for variance
            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            double delta = new_time - mean_time_ms;
            mean_time_ms += delta / sample_count;
            double delta2 = new_time - mean_time_ms;
            // Update variance using sample variance formula
            variance      = variance * (sample_count - 2.0) / (sample_count - 1.0) + delta * delta2 / sample_count;
        }

        // Compute confidence based on:
        // 1. Sample count (more samples = higher confidence, max at 100)
        // 2. Coefficient of variation (lower CV = higher confidence)
        //
        // confidence = sample_factor * stability_factor
        // where sample_factor = min(1.0, sample_count / 100)
        //       stability_factor = max(0.0, 1.0 - CV)
        //       CV = stddev / mean (coefficient of variation)
        double cv               = (mean_time_ms > 0.0) ? std::sqrt(std::max(0.0, variance)) / mean_time_ms : 1.0;
        double sample_factor    = std::min(1.0, static_cast<double>(sample_count) / 100.0);
        double stability_factor = std::max(0.0, 1.0 - cv);
        confidence              = sample_factor * stability_factor;
    }
};

// =============================================================================
// ObservationQueue: Thread-safe observation queue for async processing
// =============================================================================
// Non-blocking queue for recording observations from hot path.
// Push is fast (mutex + push), pop is non-blocking (try_pop).
// Background thread processes observations to update statistics.

class ObservationQueue {
  public:
    ObservationQueue() = default;

    // Push observation to queue (thread-safe, may briefly block)
    void push(const TuningObservation & obs) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(obs);
        }
        cv_.notify_one();
    }

    // Try to pop observation from queue (non-blocking)
    // Returns true if an observation was popped, false if queue is empty
    bool try_pop(TuningObservation & obs) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        obs = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // Wait for an observation or shutdown signal (blocking)
    // Returns true if an observation was popped, false on shutdown
    bool wait_and_pop(TuningObservation & obs, uint32_t timeout_ms = 10) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool                         has_data = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                                             [this]() { return !queue_.empty() || shutdown_.load(std::memory_order_relaxed); });
        (void) has_data;  // Suppress unused warning

        if (shutdown_.load(std::memory_order_relaxed) && queue_.empty()) {
            return false;
        }

        if (!queue_.empty()) {
            obs = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;
    }

    // Signal shutdown to wake up waiting threads
    void shutdown() {
        shutdown_.store(true, std::memory_order_relaxed);
        cv_.notify_all();
    }

    // Check if shutdown has been requested
    bool is_shutdown() const { return shutdown_.load(std::memory_order_relaxed); }

    // Get current queue size (for debugging/testing)
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

  private:
    mutable std::mutex            mutex_;
    std::condition_variable       cv_;
    std::queue<TuningObservation> queue_;
    std::atomic<bool>             shutdown_{ false };
};

// =============================================================================
// TuningCache: Thread-safe O(1) lookup cache
// =============================================================================
// Uses read-write lock for concurrent reads with exclusive writes.
// Optimized for many reads, infrequent writes (typical inference pattern).

class TuningCache {
  public:
    TuningCache() = default;

    // Lookup: Returns entry if found, nullopt if miss
    // Thread-safe for concurrent reads
    std::optional<TuningEntry> lookup(const TuningKey & key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto                                it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Insert or update entry
    // Thread-safe, blocks concurrent reads during write
    void insert(const TuningEntry & entry) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_[entry.key] = entry;
    }

    // Check if key exists (without returning data)
    bool contains(const TuningKey & key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_.find(key) != cache_.end();
    }

    // Get number of cached entries
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_.size();
    }

    // Clear all entries
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_.clear();
    }

    // Iterate over all entries (thread-safe read)
    // Callback signature: void(const TuningKey& key, const TuningEntry& entry)
    template <typename Func> void for_each(Func && fn) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (const auto & [key, entry] : cache_) {
            fn(key, entry);
        }
    }

  private:
    mutable std::shared_mutex                  mutex_;
    std::unordered_map<TuningKey, TuningEntry> cache_;
};

}  // namespace ggml_sycl_tuning

#endif  // GGML_SYCL_TUNING_ENGINE_HPP
