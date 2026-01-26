//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Memory budget management for SYCL unified memory system
// Part of unified memory system (epic llama.cpp-6hp, task llama.cpp-6hp.13)
//
// Provides MemoryBudget class for:
// - Reading GGML_SYCL_MEM_BUDGET environment variable
// - Tier allocation (HOT/WARM/COLD/workspace percentages)
// - Memory pressure detection (GREEN/YELLOW/RED/CRITICAL)
// - Graceful degradation under memory pressure
//
// Environment variable formats:
// - "80%" - percentage of device memory
// - "12G" or "12g" - absolute gigabytes
// - "8192M" or "8192m" - absolute megabytes
// - "12345678" - absolute bytes
//

#ifndef GGML_SYCL_MEMORY_BUDGET_HPP
#define GGML_SYCL_MEMORY_BUDGET_HPP

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ggml_sycl {

// =============================================================================
// Memory Pressure Levels
// =============================================================================

// Memory pressure levels based on budget utilization
// Determines behavior for preloading and eviction
enum class MemoryPressure {
    GREEN,    // < 70% - aggressive preloading allowed
    YELLOW,   // 70-85% - normal operation, LRU eviction
    RED,      // 85-95% - emergency eviction, no preloading
    CRITICAL  // > 95% - evict everything non-essential
};

// =============================================================================
// Tier Allocation Configuration
// =============================================================================

// Memory tier allocation (percentages of available budget)
// These percentages define soft limits for each tensor heat category
struct TierAllocation {
    float hot_pct       = 0.40f;  // 40% for HOT tensors (attention)
    float warm_pct      = 0.35f;  // 35% for WARM tensors (FFN)
    float cold_pct      = 0.20f;  // 20% for COLD tensors (MoE experts)
    float workspace_pct = 0.05f;  // 5% for scratch/workspace
};

// =============================================================================
// Memory Budget Manager
// =============================================================================

class MemoryBudget {
  public:
    MemoryBudget() = default;

    // Initialize with device memory size
    // Reads GGML_SYCL_MEM_BUDGET env var or defaults to 90% of device memory
    void init(size_t device_memory_bytes) {
        total_memory_ = device_memory_bytes;
        budget_       = parse_budget_env(device_memory_bytes);

        // Calculate tier limits based on budget
        tier_limits_[0] = static_cast<size_t>(budget_ * tiers_.hot_pct);        // HOT
        tier_limits_[1] = static_cast<size_t>(budget_ * tiers_.warm_pct);       // WARM
        tier_limits_[2] = static_cast<size_t>(budget_ * tiers_.cold_pct);       // COLD
        tier_limits_[3] = static_cast<size_t>(budget_ * tiers_.workspace_pct);  // workspace
    }

    // Get current memory pressure level based on allocation percentage
    MemoryPressure get_pressure() const {
        size_t alloc     = allocated_.load(std::memory_order_relaxed);
        float  usage_pct = (budget_ > 0) ? static_cast<float>(alloc) / static_cast<float>(budget_) : 0.0f;

        if (usage_pct < 0.70f) {
            return MemoryPressure::GREEN;
        }
        if (usage_pct < 0.85f) {
            return MemoryPressure::YELLOW;
        }
        if (usage_pct < 0.95f) {
            return MemoryPressure::RED;
        }
        return MemoryPressure::CRITICAL;
    }

    // Check if allocation of given size can proceed without exceeding budget
    bool can_allocate(size_t bytes) const { return (allocated_.load(std::memory_order_relaxed) + bytes) <= budget_; }

    // Track a new allocation
    // Returns the pointer unchanged for chaining
    void * track_allocation(void * ptr, size_t bytes) {
        if (ptr && bytes > 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            allocations_[ptr] = bytes;
            allocated_.fetch_add(bytes, std::memory_order_relaxed);
        }
        return ptr;
    }

    // Track deallocation
    void track_free(void * ptr) {
        if (ptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto                        it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                allocated_.fetch_sub(it->second, std::memory_order_relaxed);
                allocations_.erase(it);
            }
        }
    }

    // Get total memory budget
    size_t get_budget() const { return budget_; }

    // Get currently allocated bytes
    size_t get_allocated() const { return allocated_.load(std::memory_order_relaxed); }

    // Get available bytes (budget - allocated)
    size_t get_available() const {
        size_t alloc = allocated_.load(std::memory_order_relaxed);
        return (alloc < budget_) ? (budget_ - alloc) : 0;
    }

    // Get tier limit by index (0=HOT, 1=WARM, 2=COLD, 3=workspace)
    size_t get_tier_limit(int tier) const {
        if (tier >= 0 && tier < 4) {
            return tier_limits_[tier];
        }
        return 0;
    }

    // Get total device memory
    size_t get_total_memory() const { return total_memory_; }

    // Get usage as percentage (0-100)
    float get_usage_percent() const {
        if (budget_ == 0) {
            return 0.0f;
        }
        return static_cast<float>(allocated_.load(std::memory_order_relaxed)) / static_cast<float>(budget_) * 100.0f;
    }

    // Get tier allocation configuration
    const TierAllocation & get_tier_allocation() const { return tiers_; }

  private:
    // Parse GGML_SYCL_MEM_BUDGET environment variable
    // Supports formats: "80%", "12G", "8192M", or raw bytes
    static size_t parse_budget_env(size_t device_memory) {
        const char * env = std::getenv("GGML_SYCL_MEM_BUDGET");
        if (!env || !env[0]) {
            // Default: 90% of device memory
            return static_cast<size_t>(static_cast<double>(device_memory) * 0.90);
        }

        std::string val(env);

        // Empty string check
        if (val.empty()) {
            return static_cast<size_t>(static_cast<double>(device_memory) * 0.90);
        }

        // Percentage format (e.g., "80%")
        if (val.back() == '%') {
            try {
                float pct = std::stof(val.substr(0, val.size() - 1)) / 100.0f;
                // Clamp to reasonable range
                if (pct < 0.1f) {
                    pct = 0.1f;
                }
                if (pct > 1.0f) {
                    pct = 1.0f;
                }
                return static_cast<size_t>(static_cast<double>(device_memory) * pct);
            } catch (...) {
                return static_cast<size_t>(static_cast<double>(device_memory) * 0.90);
            }
        }

        // Gigabytes suffix (e.g., "12G" or "12g")
        char suffix = val.back();
        if (suffix == 'G' || suffix == 'g') {
            try {
                double gb = std::stod(val.substr(0, val.size() - 1));
                return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
            } catch (...) {
                return static_cast<size_t>(static_cast<double>(device_memory) * 0.90);
            }
        }

        // Megabytes suffix (e.g., "8192M" or "8192m")
        if (suffix == 'M' || suffix == 'm') {
            try {
                double mb = std::stod(val.substr(0, val.size() - 1));
                return static_cast<size_t>(mb * 1024.0 * 1024.0);
            } catch (...) {
                return static_cast<size_t>(static_cast<double>(device_memory) * 0.90);
            }
        }

        // Raw bytes (e.g., "12345678")
        try {
            return std::stoull(val);
        } catch (...) {
            return static_cast<size_t>(static_cast<double>(device_memory) * 0.90);
        }
    }

    size_t              total_memory_ = 0;
    size_t              budget_       = 0;
    std::atomic<size_t> allocated_{ 0 };
    TierAllocation      tiers_;
    size_t              tier_limits_[4] = { 0 };

    std::unordered_map<void *, size_t> allocations_;
    mutable std::mutex                 mutex_;
};

// =============================================================================
// Pressure Action Helpers
// =============================================================================

// Get recommended action string for a given pressure level
// Used for logging and debugging
inline const char * get_pressure_action(MemoryPressure pressure) {
    switch (pressure) {
        case MemoryPressure::GREEN:
            return "preload_aggressive";
        case MemoryPressure::YELLOW:
            return "evict_lru";
        case MemoryPressure::RED:
            return "evict_emergency";
        case MemoryPressure::CRITICAL:
            return "evict_all_nonessential";
        default:
            return "unknown";
    }
}

// Convert pressure level to string for debugging
inline const char * pressure_to_string(MemoryPressure pressure) {
    switch (pressure) {
        case MemoryPressure::GREEN:
            return "GREEN";
        case MemoryPressure::YELLOW:
            return "YELLOW";
        case MemoryPressure::RED:
            return "RED";
        case MemoryPressure::CRITICAL:
            return "CRITICAL";
        default:
            return "UNKNOWN";
    }
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_MEMORY_BUDGET_HPP
