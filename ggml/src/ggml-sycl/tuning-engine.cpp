//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// TuningEngine implementation
// Progressive background tuning with three phases:
// 1. OBSERVATION: Collect operation frequency data
// 2. BENCHMARKING: Run micro-benchmarks on top patterns
// 3. REFINEMENT: Periodic re-testing (future work)

#include "tuning-engine-impl.hpp"

#include <algorithm>
#include <chrono>

namespace ggml_sycl_tuning {

// =============================================================================
// Constructor / Destructor
// =============================================================================

TuningEngine::TuningEngine() : TuningEngine(TuningEngineConfig{}) {}

TuningEngine::TuningEngine(const TuningEngineConfig & config) : config_(config) {
    start_background_tuning();
}

TuningEngine::~TuningEngine() {
    stop();
}

// =============================================================================
// Core API
// =============================================================================

TunedParams TuningEngine::get_params(const TuningKey & key) {
    // O(1) cache lookup
    auto cached = cache_.lookup(key);
    if (cached.has_value()) {
        return cached->params;
    }

    // Cache miss - derive heuristic params
    return derive_heuristic_params(key);
}

void TuningEngine::record_observation_async(const TuningKey & key, const TunedParams & used) {
    // Quick increment of total counter (atomic, no lock)
    total_observations_.fetch_add(1, std::memory_order_relaxed);

    // Queue observation for background processing
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        observation_queue_.push(ObservationRecord{ key, used });
    }
    queue_cv_.notify_one();
}

void TuningEngine::record_observation_async(const TuningKey & key, const TunedParams & used, double time_ms) {
    // Quick increment of total counter (atomic, no lock)
    total_observations_.fetch_add(1, std::memory_order_relaxed);

    // Queue for frequency counting (legacy path)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        observation_queue_.push(ObservationRecord{ key, used });
    }
    queue_cv_.notify_one();

    // Queue for performance statistics (new path)
    TuningObservation obs;
    obs.key       = key;
    obs.params    = used;
    obs.time_ms   = time_ms;
    obs.timestamp = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    perf_observation_queue_.push(obs);
}

double TuningEngine::get_confidence(const TuningKey & key) const {
    std::lock_guard<std::mutex> lock(perf_stats_mutex_);
    auto                        it = perf_stats_.find(key);
    if (it == perf_stats_.end()) {
        return 0.0;
    }
    return it->second.confidence;
}

ObservationStats TuningEngine::get_stats(const TuningKey & key) const {
    std::lock_guard<std::mutex> lock(perf_stats_mutex_);
    auto                        it = perf_stats_.find(key);
    if (it == perf_stats_.end()) {
        return ObservationStats{};
    }
    return it->second;
}

// =============================================================================
// Control API
// =============================================================================

void TuningEngine::start_background_tuning() {
    if (tuning_thread_.joinable()) {
        return;  // Already running
    }

    should_stop_.store(false, std::memory_order_relaxed);
    tuning_thread_ = std::thread(&TuningEngine::background_thread_main, this);
}

void TuningEngine::stop() {
    should_stop_.store(true, std::memory_order_relaxed);
    queue_cv_.notify_all();
    perf_observation_queue_.shutdown();

    if (tuning_thread_.joinable()) {
        tuning_thread_.join();
    }
}

// =============================================================================
// Testing/Debug API
// =============================================================================

std::unordered_map<TuningKey, uint64_t> TuningEngine::get_histogram_snapshot() const {
    std::lock_guard<std::mutex> lock(histogram_mutex_);
    return histogram_;
}

std::vector<std::pair<TuningKey, uint64_t>> TuningEngine::get_top_patterns(size_t k) const {
    std::lock_guard<std::mutex> lock(histogram_mutex_);

    // Convert to vector for sorting
    std::vector<std::pair<TuningKey, uint64_t>> patterns(histogram_.begin(), histogram_.end());

    // Sort by count descending
    std::sort(patterns.begin(), patterns.end(), [](const auto & a, const auto & b) { return a.second > b.second; });

    // Return top-k
    if (patterns.size() > k) {
        patterns.resize(k);
    }
    return patterns;
}

void TuningEngine::insert_cached_params(const TuningKey & key, const TunedParams & params, float tflops) {
    TuningEntry entry;
    entry.key             = key;
    entry.params          = params;
    entry.measured_tflops = tflops;
    entry.timestamp =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    cache_.insert(entry);
}

// =============================================================================
// Internal methods
// =============================================================================

void TuningEngine::background_thread_main() {
    while (!should_stop_.load(std::memory_order_relaxed)) {
        // Wait for observations or stop signal
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(config_.idle_sleep_ms), [this]() {
                return !observation_queue_.empty() || should_stop_.load(std::memory_order_relaxed);
            });
        }

        if (should_stop_.load(std::memory_order_relaxed)) {
            break;
        }

        // Process any queued observations
        process_observations();

        // Process performance observations
        process_perf_observations();

        // Check if we should transition phases
        check_phase_transition();

        // If in BENCHMARKING phase, run benchmarks
        if (phase_.load(std::memory_order_relaxed) == TuningPhase::BENCHMARKING) {
            run_benchmarks();
        }
    }

    // Drain remaining observations before exit
    process_observations();
    process_perf_observations();
}

void TuningEngine::process_observations() {
    // Batch process all queued observations
    std::queue<ObservationRecord> to_process;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::swap(to_process, observation_queue_);
    }

    if (to_process.empty()) {
        return;
    }

    // Update histogram
    {
        std::lock_guard<std::mutex> lock(histogram_mutex_);
        while (!to_process.empty()) {
            const auto & record = to_process.front();
            histogram_[record.key]++;
            to_process.pop();
        }
    }
}

void TuningEngine::process_perf_observations() {
    // Process all queued performance observations
    TuningObservation obs;
    while (perf_observation_queue_.try_pop(obs)) {
        std::lock_guard<std::mutex> lock(perf_stats_mutex_);
        perf_stats_[obs.key].update(obs.time_ms);
    }
}

void TuningEngine::check_phase_transition() {
    TuningPhase current = phase_.load(std::memory_order_relaxed);

    switch (current) {
        case TuningPhase::OBSERVATION:
            // Transition to BENCHMARKING after threshold observations
            if (total_observations_.load(std::memory_order_relaxed) >= config_.observation_threshold) {
                phase_.store(TuningPhase::BENCHMARKING, std::memory_order_relaxed);
            }
            break;

        case TuningPhase::BENCHMARKING:
            // Transition to REFINEMENT after benchmarks complete
            // (For now, we stay in BENCHMARKING - refinement is future work)
            break;

        case TuningPhase::REFINEMENT:
            // Stay in refinement
            break;
    }
}

TunedParams TuningEngine::derive_heuristic_params(const TuningKey & key) const {
    TunedParams params;

    // Heuristics based on batch bucket and quant type
    switch (key.batch_bucket) {
        case BatchBucket::SINGLE:
            // M=1: Memory-bound, use simple configuration
            params.tile_m         = 1;
            params.tile_n         = 64;
            params.tile_k         = 32;
            params.workgroup_size = 64;
            params.slm_kb         = 0;      // No SLM needed for single row
            params.prefetch_depth = 2;
            params.use_dpas       = false;  // Not beneficial for M=1
            params.layout_mode    = 1;      // SOA for bandwidth
            break;

        case BatchBucket::SMALL:
            // M=2-8: Transitional, small tiles
            params.tile_m         = 8;
            params.tile_n         = 32;
            params.tile_k         = 32;
            params.workgroup_size = 128;
            params.slm_kb         = 8;
            params.prefetch_depth = 2;
            params.use_dpas       = false;  // Small batch, dpas overhead not worth it
            params.layout_mode    = 2;      // COALESCED
            break;

        case BatchBucket::MEDIUM:
            // M=9-64: Can benefit from some compute optimization
            params.tile_m         = 16;
            params.tile_n         = 64;
            params.tile_k         = 32;
            params.workgroup_size = 256;
            params.slm_kb         = 16;
            params.prefetch_depth = 2;
            params.use_dpas       = true;  // Start using dpas
            params.layout_mode    = 3;     // XMX_COALESCED
            break;

        case BatchBucket::LARGE:
            // M=65-128: Compute starts to dominate
            params.tile_m         = 32;
            params.tile_n         = 64;
            params.tile_k         = 32;
            params.workgroup_size = 256;
            params.slm_kb         = 32;
            params.prefetch_depth = 2;
            params.use_dpas       = true;
            params.layout_mode    = 3;  // XMX_COALESCED
            break;

        case BatchBucket::XLARGE:
            // M>128: Fully compute-bound, large tiles
            params.tile_m         = 64;
            params.tile_n         = 64;
            params.tile_k         = 32;
            params.workgroup_size = 256;
            params.slm_kb         = 48;
            params.prefetch_depth = 4;
            params.use_dpas       = true;
            params.layout_mode    = 4;  // XMX_GEMM_TILED
            break;
    }

    // Adjust for quant type if needed
    // Q8_0 has better INT8 dpas support
    if (key.quant_type == 8) {  // GGML_TYPE_Q8_0
        // Q8_0 is native INT8, can use dpas more aggressively
        if (key.batch_bucket == BatchBucket::SMALL) {
            params.use_dpas = true;
        }
    }

    // Adjust tile_k based on K dimension (avoid oversized tiles)
    if (key.K < 1024) {
        params.tile_k = std::min(params.tile_k, static_cast<uint16_t>(key.K));
    }

    return params;
}

void TuningEngine::run_benchmarks() {
    // Get top-K patterns to benchmark
    auto top_patterns = get_top_patterns(config_.top_k_patterns);

    if (top_patterns.empty()) {
        return;
    }

    // For now, just populate cache with heuristic-derived params
    // Actual micro-benchmarks would require SYCL queue access (future work)
    for (const auto & [key, count] : top_patterns) {
        // Only cache if not already cached
        if (!cache_.contains(key)) {
            TunedParams params = derive_heuristic_params(key);

            // Insert with a reasonable estimated TFLOPS
            // (In real implementation, we'd measure this)
            float estimated_tflops = 0.0f;

            // Rough estimate based on batch bucket
            switch (key.batch_bucket) {
                case BatchBucket::SINGLE:
                    estimated_tflops = 10.0f;  // Memory bound
                    break;
                case BatchBucket::SMALL:
                    estimated_tflops = 25.0f;
                    break;
                case BatchBucket::MEDIUM:
                    estimated_tflops = 50.0f;
                    break;
                case BatchBucket::LARGE:
                    estimated_tflops = 100.0f;
                    break;
                case BatchBucket::XLARGE:
                    estimated_tflops = 200.0f;  // Compute bound
                    break;
            }

            insert_cached_params(key, params, estimated_tflops);
        }
    }

    // Transition to REFINEMENT phase after initial benchmarks
    phase_.store(TuningPhase::REFINEMENT, std::memory_order_relaxed);
}

}  // namespace ggml_sycl_tuning
