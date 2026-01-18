//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Priority-based LRU eviction policy for SYCL unified memory management
// Part of unified memory system (epic llama.cpp-v3n, task llama.cpp-2oy)
//
// Priority Levels (P0-P4):
// - P0: Compute buffers - NEVER evict (pinned=true)
// - P1: Active KV cache heads - evict last
// - P2: Hot experts (recently used) - evict after KV
// - P3: Warm experts (predicted) - evict before hot
// - P4: Cold experts - evict first
//
// Eviction targets (in order):
// 1. VRAM -> HOST (fast, keep in pinned RAM)
// 2. HOST -> MMAP (slower, page out)
// 3. MMAP can be discarded (reload from file)

#ifndef GGML_SYCL_EVICTION_POLICY_HPP
#define GGML_SYCL_EVICTION_POLICY_HPP

#include "unified-cache.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ggml_sycl {

// Priority levels for eviction policy
// Lower number = higher priority = evicted LAST
enum class EvictionPriority {
    P0_COMPUTE     = 0,  // Compute buffers - NEVER evict
    P1_ACTIVE_KV   = 1,  // Active KV cache heads - evict last
    P2_HOT_EXPERT  = 2,  // Hot experts (recently used) - evict after KV
    P3_WARM_EXPERT = 3,  // Warm experts (predicted) - evict before hot
    P4_COLD_EXPERT = 4,  // Cold experts - evict first
};

// Information about a victim selected for eviction
struct EvictionVictim {
    uint64_t         id;        // Unique identifier
    size_t           size;      // Size in bytes
    cache_location   location;  // Current memory location
    EvictionPriority priority;  // Current priority
};

// Priority-based LRU eviction policy
//
// Selects victims for eviction based on:
// 1. Priority level (P4 evicted first, P0 never evicted)
// 2. LRU within same priority (least recently used first)
// 3. Memory location (VRAM entries from VRAM, HOST from HOST, etc.)
class EvictionPolicy {
  public:
    EvictionPolicy();
    ~EvictionPolicy();

    // Non-copyable, non-movable
    EvictionPolicy(const EvictionPolicy &)             = delete;
    EvictionPolicy & operator=(const EvictionPolicy &) = delete;
    EvictionPolicy(EvictionPolicy &&)                  = delete;
    EvictionPolicy & operator=(EvictionPolicy &&)      = delete;

    // Add an entry to the eviction tracker
    // If entry with same id exists, updates it instead
    void add_entry(uint64_t id, EvictionPriority priority, size_t size, cache_location location);

    // Remove an entry from the eviction tracker
    void remove_entry(uint64_t id);

    // Update last access time (touch) for an entry
    void touch_entry(uint64_t id);

    // Update priority for an entry (e.g., cold expert becomes hot)
    void update_priority(uint64_t id, EvictionPriority new_priority);

    // Update location for an entry (e.g., VRAM -> HOST after eviction)
    void update_location(uint64_t id, cache_location new_location);

    // Select a single victim for eviction from the specified location
    // Returns std::nullopt if no evictable entry exists
    std::optional<EvictionVictim> select_victim(cache_location from_location);

    // Select multiple victims to free at least `bytes_needed` from the specified location
    // May return fewer if not enough evictable entries exist
    std::vector<EvictionVictim> select_victims_for_size(cache_location from_location, size_t bytes_needed);

    // Get total size of all tracked entries
    size_t get_total_size() const;

    // Get total size of entries in a specific location
    size_t get_size_by_location(cache_location location) const;

    // Get number of tracked entries
    size_t get_entry_count() const;

  private:
    struct Impl;
    Impl * impl_;
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_EVICTION_POLICY_HPP
