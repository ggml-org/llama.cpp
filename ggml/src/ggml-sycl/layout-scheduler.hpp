//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Layout conversion scheduling for SYCL memory management
// Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.16)
//
// Schedules tensor layout conversions during GPU idle periods:
// - Detects idle time between inference batches
// - Batches conversions to minimize overhead
// - Prioritizes HOT tensors for conversion first
// - Respects memory and batch size limits
//

#ifndef GGML_SYCL_LAYOUT_SCHEDULER_HPP
#define GGML_SYCL_LAYOUT_SCHEDULER_HPP

#include "heat-classifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <queue>
#include <vector>

namespace ggml_sycl {

// =============================================================================
// Target Layout Definitions
// =============================================================================

// Target layout for conversion
enum class TargetLayout {
    SOA,           // Structure of Arrays - contiguous qs bytes, then d values
    COALESCED,     // Word-major interleaved - tile-based for MMVQ
    XMX_COALESCED  // XMX-aligned for dpas - K_TILE=32 aligned
};

// Convert target layout to string for debugging
inline const char * target_layout_to_string(TargetLayout layout) {
    switch (layout) {
        case TargetLayout::SOA:
            return "SOA";
        case TargetLayout::COALESCED:
            return "COALESCED";
        case TargetLayout::XMX_COALESCED:
            return "XMX_COALESCED";
        default:
            return "UNKNOWN";
    }
}

// =============================================================================
// Conversion Request
// =============================================================================

// Conversion request for a tensor
struct ConversionRequest {
    const void * tensor_data;   // Pointer to tensor data
    size_t       size_bytes;    // Size of tensor in bytes
    TargetLayout target;        // Target layout for conversion
    TensorHeat   heat;          // Heat classification from heat-classifier
    int          priority;      // Higher = more urgent
    int          access_count;  // Times accessed since last conversion

    // Priority comparator for max-heap (higher priority = top)
    bool operator<(const ConversionRequest & other) const {
        return priority < other.priority;  // Lower priority = earlier in min-heap
    }
};

// =============================================================================
// Conversion Status
// =============================================================================

// Status of a conversion operation
enum class ConversionStatus {
    NOT_STARTED,  // Conversion not yet queued
    QUEUED,       // In the pending queue
    IN_PROGRESS,  // Currently being converted
    COMPLETED,    // Successfully converted
    FAILED        // Conversion failed
};

// Convert status to string for debugging
inline const char * conversion_status_to_string(ConversionStatus status) {
    switch (status) {
        case ConversionStatus::NOT_STARTED:
            return "NOT_STARTED";
        case ConversionStatus::QUEUED:
            return "QUEUED";
        case ConversionStatus::IN_PROGRESS:
            return "IN_PROGRESS";
        case ConversionStatus::COMPLETED:
            return "COMPLETED";
        case ConversionStatus::FAILED:
            return "FAILED";
        default:
            return "UNKNOWN";
    }
}

// =============================================================================
// Scheduler Configuration
// =============================================================================

// Configuration for the layout conversion scheduler
struct SchedulerConfig {
    size_t min_idle_ms        = 5;                 // Minimum idle time (ms) to trigger conversion
    size_t batch_size_limit   = 4;                 // Max tensors per conversion batch
    size_t batch_bytes_limit  = 64 * 1024 * 1024;  // 64 MB max bytes per batch
    int    min_access_count   = 3;                 // Min accesses before conversion eligible
    bool   enable_scheduling  = true;              // Enable/disable scheduling
    bool   prefer_hot_tensors = true;              // Prioritize HOT tensors
};

// =============================================================================
// Layout Conversion Scheduler
// =============================================================================

// Schedules layout conversions during GPU idle periods
//
// Usage:
//   LayoutConversionScheduler scheduler;
//   scheduler.queue_conversion(request);
//   scheduler.on_idle_start();
//   if (scheduler.should_convert()) {
//       auto batch = scheduler.get_conversion_batch();
//       // Perform conversions...
//   }
//
class LayoutConversionScheduler {
  public:
    // Callback type for conversion execution
    using ConversionCallback = std::function<bool(const ConversionRequest &)>;

    LayoutConversionScheduler() = default;

    // Configuration
    void set_config(const SchedulerConfig & config) { config_ = config; }

    const SchedulerConfig & get_config() const { return config_; }

    // Queue a tensor for conversion
    // Only queues if access_count >= min_access_count
    void queue_conversion(const ConversionRequest & request) {
        if (!config_.enable_scheduling) {
            return;
        }

        if (request.access_count >= config_.min_access_count) {
            pending_.push(request);
        }
    }

    // Record idle period start
    // Call this when the GPU becomes idle (e.g., between inference batches)
    void on_idle_start() { idle_start_ = std::chrono::steady_clock::now(); }

    // Check if we should convert (called during idle)
    // Returns true if:
    // - There are pending conversions
    // - Sufficient idle time has elapsed
    bool should_convert() const {
        if (pending_.empty()) {
            return false;
        }

        auto now     = std::chrono::steady_clock::now();
        auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - idle_start_).count();

        return idle_ms >= static_cast<long long>(config_.min_idle_ms);
    }

    // Get batch of conversions to perform
    // Returns up to batch_size_limit tensors, up to batch_bytes_limit bytes
    // Batch is sorted by heat (HOT first) for optimal conversion order
    std::vector<ConversionRequest> get_conversion_batch() {
        std::vector<ConversionRequest> batch;
        size_t                         batch_bytes = 0;

        while (!pending_.empty() && batch.size() < config_.batch_size_limit &&
               batch_bytes < config_.batch_bytes_limit) {
            ConversionRequest req = pending_.top();
            pending_.pop();

            batch_bytes += req.size_bytes;
            batch.push_back(req);
        }

        // Sort by heat for optimal conversion order (HOT first)
        // Heat enum: HOT=0, WARM=1, COLD=2, FROZEN=3
        if (config_.prefer_hot_tensors) {
            std::sort(batch.begin(), batch.end(), [](const auto & a, const auto & b) {
                return static_cast<int>(a.heat) < static_cast<int>(b.heat);
            });
        }

        return batch;
    }

    // Compute conversion priority based on heat, size, and access count
    //
    // Priority formula:
    //   priority = heat_score + size_score + access_score
    //
    // Where:
    //   heat_score: HOT=1000, WARM=100, FROZEN=50, COLD=10
    //   size_score: log2(size_MB + 1) * 10 (larger tensors benefit more)
    //   access_score: access_count * 5 (frequently accessed = higher priority)
    //
    static int compute_priority(TensorHeat heat, size_t size_bytes, int access_count) {
        // Base priority from heat level
        int heat_score = 0;
        switch (heat) {
            case TensorHeat::HOT:
                heat_score = 1000;
                break;
            case TensorHeat::WARM:
                heat_score = 100;
                break;
            case TensorHeat::COLD:
                heat_score = 10;
                break;
            case TensorHeat::FROZEN:
                heat_score = 50;
                break;
        }

        // Size bonus: larger tensors benefit more from optimized layout
        // log2(size_MB + 1) * 10, where size_MB = size_bytes / (1024 * 1024)
        double size_mb    = static_cast<double>(size_bytes) / (1024.0 * 1024.0);
        int    size_score = static_cast<int>(std::log2(size_mb + 1.0) * 10.0);

        // Access frequency bonus
        int access_score = access_count * 5;

        return heat_score + size_score + access_score;
    }

    // Get pending count
    size_t get_pending_count() const { return pending_.size(); }

    // Clear pending conversions
    void clear() {
        while (!pending_.empty()) {
            pending_.pop();
        }
    }

    // Get total pending bytes
    // Note: This is O(n), avoid calling frequently
    size_t get_pending_bytes() const {
        size_t total = 0;
        auto   temp  = pending_;
        while (!temp.empty()) {
            total += temp.top().size_bytes;
            temp.pop();
        }
        return total;
    }

    // Check if scheduling is enabled
    bool is_enabled() const { return config_.enable_scheduling; }

    // Enable/disable scheduling
    void set_enabled(bool enabled) { config_.enable_scheduling = enabled; }

    // Get time since idle start (milliseconds)
    long long get_idle_duration_ms() const {
        auto now     = std::chrono::steady_clock::now();
        auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - idle_start_).count();
        return idle_ms;
    }

  private:
    SchedulerConfig                        config_;
    std::priority_queue<ConversionRequest> pending_;
    std::chrono::steady_clock::time_point  idle_start_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_LAYOUT_SCHEDULER_HPP
