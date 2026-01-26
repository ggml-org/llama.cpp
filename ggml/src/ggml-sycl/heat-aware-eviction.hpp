//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Heat-aware eviction policy for SYCL unified memory management
// Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.12)
//
// Uses heat classification (HOT/WARM/COLD/FROZEN), access patterns, and
// memory pressure to make intelligent eviction decisions.
//
// Priority Formula:
//   priority = heat_score * frequency / (recency + 1.0) - size_penalty
//
// Where:
//   - heat_score: HOT=1000, WARM=100, COLD=10, FROZEN=50
//   - frequency: access_count / session_duration_seconds
//   - recency: seconds since last access
//   - size_penalty: log2(size_bytes / 1MB)
//
// Higher priority = keep longer, lower priority = evict first

#ifndef GGML_SYCL_HEAT_AWARE_EVICTION_HPP
#define GGML_SYCL_HEAT_AWARE_EVICTION_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Forward declaration for ggml_tensor (avoid heavy includes in header)
struct ggml_tensor;

namespace ggml_sycl {

// =============================================================================
// Heat Classification
// =============================================================================

// Heat classification for tensors based on access patterns
// Used by the eviction policy to prioritize what to keep in memory
enum class TensorHeat {
    FROZEN,  // Static tensors that never change (e.g., model weights that are rarely accessed)
    COLD,    // Infrequently accessed tensors
    WARM,    // Moderately accessed tensors
    HOT      // Frequently accessed tensors (e.g., active KV cache, current layer weights)
};

// Get heat score multiplier for priority calculation
// Higher score = higher priority = evicted last
inline float heat_multiplier(TensorHeat heat) {
    switch (heat) {
        case TensorHeat::HOT:
            return 1000.0f;
        case TensorHeat::WARM:
            return 100.0f;
        case TensorHeat::FROZEN:
            return 50.0f;  // FROZEN between WARM and COLD - static but may be needed
        case TensorHeat::COLD:
            return 10.0f;
        default:
            return 1.0f;
    }
}

// Convert heat to string for debugging
inline const char * heat_to_string(TensorHeat heat) {
    switch (heat) {
        case TensorHeat::HOT:
            return "HOT";
        case TensorHeat::WARM:
            return "WARM";
        case TensorHeat::COLD:
            return "COLD";
        case TensorHeat::FROZEN:
            return "FROZEN";
        default:
            return "UNKNOWN";
    }
}

// =============================================================================
// Memory Pressure Levels
// =============================================================================

// Memory pressure levels based on budget utilization
// Determines aggressiveness of eviction
enum class MemoryPressure {
    NONE,       // < 70% utilization - no eviction needed
    MODERATE,   // 70-85% - evict only COLD tensors
    HIGH,       // 85-95% - evict COLD and WARM
    CRITICAL,   // 95-100% - evict everything except HOT
    IMPOSSIBLE  // > 100% - cannot satisfy request even with full eviction
};

// Get pressure level from memory usage
// budget: total memory budget
// current_usage: currently used memory
// pending: memory needed for pending allocation
inline MemoryPressure get_pressure(size_t budget, size_t current_usage, size_t pending) {
    if (budget == 0) {
        return MemoryPressure::IMPOSSIBLE;
    }

    // Calculate effective utilization after pending allocation
    size_t total_needed = current_usage + pending;
    float  utilization  = static_cast<float>(total_needed) / static_cast<float>(budget);

    if (utilization > 1.0f) {
        return MemoryPressure::IMPOSSIBLE;
    }
    if (utilization > 0.95f) {
        return MemoryPressure::CRITICAL;
    }
    if (utilization > 0.85f) {
        return MemoryPressure::HIGH;
    }
    if (utilization > 0.70f) {
        return MemoryPressure::MODERATE;
    }
    return MemoryPressure::NONE;
}

// Convert pressure to string for debugging
inline const char * pressure_to_string(MemoryPressure pressure) {
    switch (pressure) {
        case MemoryPressure::NONE:
            return "NONE";
        case MemoryPressure::MODERATE:
            return "MODERATE";
        case MemoryPressure::HIGH:
            return "HIGH";
        case MemoryPressure::CRITICAL:
            return "CRITICAL";
        case MemoryPressure::IMPOSSIBLE:
            return "IMPOSSIBLE";
        default:
            return "UNKNOWN";
    }
}

// =============================================================================
// Cache Entry
// =============================================================================

// Cache entry with heat-based metadata for eviction scoring
struct CacheEntry {
    ggml_tensor *                            tensor;       // Pointer to the tensor (may be null for testing)
    TensorHeat                               heat;         // Current heat classification
    size_t                                   size_bytes;   // Size of the cached data
    std::chrono::steady_clock::time_point    last_access;  // Time of last access
    uint64_t                                 access_count; // Number of accesses since creation
    uint64_t                                 id;           // Unique identifier for the entry
};

// =============================================================================
// Priority Computation
// =============================================================================

// Global session start time for frequency calculation
// In production, this should be managed by the cache manager
inline std::chrono::steady_clock::time_point & get_session_start() {
    static std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    return start;
}

// Reset session start time (for testing)
inline void reset_session_start() {
    get_session_start() = std::chrono::steady_clock::now();
}

// Get session duration in seconds
inline float session_duration_seconds() {
    auto now      = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - get_session_start());
    // Minimum 1 second to avoid division issues
    return std::max(1.0f, static_cast<float>(duration.count()) / 1000.0f);
}

// Get seconds since a time point
inline float seconds_since(const std::chrono::steady_clock::time_point & tp) {
    auto now      = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - tp);
    return static_cast<float>(duration.count()) / 1000.0f;
}

// Compute eviction priority for a cache entry
// Higher priority = keep longer (evicted last)
// Lower priority = evict first
//
// Formula: heat_score * frequency / (recency + 1.0) - size_penalty
//
// Components:
// - heat_score: Based on TensorHeat classification
// - frequency: access_count / session_duration (accesses per second)
// - recency: seconds since last access (older = lower priority)
// - size_penalty: log2(size_MB) - larger tensors penalized slightly
inline float compute_priority(const CacheEntry & entry) {
    float heat_score = heat_multiplier(entry.heat);

    // Frequency: accesses per second of session time
    float frequency = static_cast<float>(entry.access_count) / session_duration_seconds();

    // Recency: seconds since last access
    float recency = seconds_since(entry.last_access);

    // Size penalty: log2 of size in MB (larger = more penalty)
    // Minimum 1 MB to avoid log(0) or negative values
    float size_mb      = static_cast<float>(entry.size_bytes) / (1024.0f * 1024.0f);
    float size_penalty = 0.0f;
    if (size_mb > 1.0f) {
        size_penalty = std::log2(size_mb);
    }

    // Combine: high heat + high frequency + recent access - large size = high priority
    return heat_score * frequency / (recency + 1.0f) - size_penalty;
}

// =============================================================================
// Eviction Functions
// =============================================================================

// Evict entries to meet a target budget
// Modifies the cache vector in place, removing evicted entries
// HOT tensors are protected and never evicted
//
// Parameters:
//   cache: Vector of cache entries to consider for eviction (modified in place)
//   target_bytes: Target memory usage to achieve (evict until usage <= target)
//
// Returns: Total bytes freed by eviction
//
// Algorithm:
// 1. Sort entries by priority (ascending - lowest priority first)
// 2. Evict lowest priority entries until current <= target
// 3. Skip HOT tensors (they are protected)
inline size_t evict_to_budget(std::vector<CacheEntry> & cache, size_t target_bytes) {
    // Calculate current usage
    size_t current_usage = 0;
    for (const auto & entry : cache) {
        current_usage += entry.size_bytes;
    }

    // If already under budget, nothing to do
    if (current_usage <= target_bytes) {
        return 0;
    }

    // Build list of evictable entries with their priorities
    struct PrioritizedEntry {
        size_t index;
        float  priority;
    };

    std::vector<PrioritizedEntry> candidates;
    candidates.reserve(cache.size());

    for (size_t i = 0; i < cache.size(); ++i) {
        // HOT tensors are protected - never evict them
        if (cache[i].heat == TensorHeat::HOT) {
            continue;
        }
        candidates.push_back({ i, compute_priority(cache[i]) });
    }

    // Sort by priority (ascending - lowest priority first = evict first)
    std::sort(candidates.begin(), candidates.end(), [](const PrioritizedEntry & a, const PrioritizedEntry & b) {
        return a.priority < b.priority;
    });

    // Mark entries for removal
    std::vector<bool> to_remove(cache.size(), false);
    size_t            bytes_freed = 0;

    for (const auto & candidate : candidates) {
        if (current_usage <= target_bytes) {
            break;
        }

        to_remove[candidate.index] = true;
        bytes_freed += cache[candidate.index].size_bytes;
        current_usage -= cache[candidate.index].size_bytes;
    }

    // Remove marked entries (iterate backwards to preserve indices)
    for (size_t i = cache.size(); i > 0; --i) {
        if (to_remove[i - 1]) {
            cache.erase(cache.begin() + static_cast<std::ptrdiff_t>(i - 1));
        }
    }

    return bytes_freed;
}

// Check if eviction can satisfy a request given current state and pressure
// Returns true if eviction + allocation is possible
inline bool can_satisfy_request(const std::vector<CacheEntry> & cache, size_t budget, size_t request_bytes,
                                MemoryPressure pressure) {
    if (pressure == MemoryPressure::IMPOSSIBLE) {
        return false;
    }

    // Calculate total evictable bytes (exclude HOT)
    size_t evictable = 0;
    size_t total     = 0;

    for (const auto & entry : cache) {
        total += entry.size_bytes;
        if (entry.heat != TensorHeat::HOT) {
            evictable += entry.size_bytes;
        }
    }

    // After evicting all evictable entries, is there room?
    size_t after_eviction = total - evictable;
    return (after_eviction + request_bytes) <= budget;
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_HEAT_AWARE_EVICTION_HPP
