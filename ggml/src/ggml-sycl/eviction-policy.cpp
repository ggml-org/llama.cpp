//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Priority-based LRU eviction policy for SYCL unified memory management
// Part of unified memory system (epic llama.cpp-v3n, task llama.cpp-2oy)

#include "eviction-policy.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// Internal entry representation
struct EvictionEntry {
    uint64_t         id;
    EvictionPriority priority;
    size_t           size;
    cache_location   location;
    int64_t          last_access;  // Monotonic timestamp (lower = older = evict first)
};

struct EvictionPolicy::Impl {
    std::unordered_map<uint64_t, EvictionEntry> entries;
    int64_t                                     time_counter = 0;
    mutable std::mutex                          mutex;

    int64_t now() { return ++time_counter; }

    // Select victim based on priority (P4 first) then LRU within priority
    // Returns nullptr if no evictable entry exists at the given location
    const EvictionEntry * find_best_victim(cache_location from_location) const {
        const EvictionEntry * best = nullptr;

        for (const auto & [id, entry] : entries) {
            // Skip P0 entries - they are NEVER evicted
            if (entry.priority == EvictionPriority::P0_COMPUTE) {
                continue;
            }

            // Skip entries not at the requested location
            if (entry.location != from_location) {
                continue;
            }

            if (best == nullptr) {
                best = &entry;
                continue;
            }

            // Priority comparison: higher number = lower priority = evict first
            // P4 (4) evicts before P3 (3) before P2 (2) before P1 (1)
            if (static_cast<int>(entry.priority) > static_cast<int>(best->priority)) {
                best = &entry;
            } else if (entry.priority == best->priority) {
                // Same priority: LRU - lower timestamp = older = evict first
                if (entry.last_access < best->last_access) {
                    best = &entry;
                }
            }
        }

        return best;
    }
};

EvictionPolicy::EvictionPolicy() : impl_(new Impl()) {}

EvictionPolicy::~EvictionPolicy() {
    delete impl_;
}

void EvictionPolicy::add_entry(uint64_t id, EvictionPriority priority, size_t size, cache_location location) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // If entry exists, update it; otherwise add new entry
    auto it = impl_->entries.find(id);
    if (it != impl_->entries.end()) {
        // Update existing entry
        it->second.priority    = priority;
        it->second.size        = size;
        it->second.location    = location;
        it->second.last_access = impl_->now();
    } else {
        // Add new entry
        impl_->entries[id] = EvictionEntry{ id, priority, size, location, impl_->now() };
    }
}

void EvictionPolicy::remove_entry(uint64_t id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->entries.erase(id);
}

void EvictionPolicy::touch_entry(uint64_t id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(id);
    if (it != impl_->entries.end()) {
        it->second.last_access = impl_->now();
    }
}

void EvictionPolicy::update_priority(uint64_t id, EvictionPriority new_priority) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(id);
    if (it != impl_->entries.end()) {
        it->second.priority = new_priority;
    }
}

void EvictionPolicy::update_location(uint64_t id, cache_location new_location) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(id);
    if (it != impl_->entries.end()) {
        it->second.location = new_location;
    }
}

std::optional<EvictionVictim> EvictionPolicy::select_victim(cache_location from_location) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    const EvictionEntry * best = impl_->find_best_victim(from_location);

    if (best == nullptr) {
        return std::nullopt;
    }

    return EvictionVictim{ best->id, best->size, best->location, best->priority };
}

std::vector<EvictionVictim> EvictionPolicy::select_victims_for_size(cache_location from_location, size_t bytes_needed) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<EvictionVictim> victims;
    size_t                      total = 0;

    // Build a list of all evictable entries at this location
    std::vector<const EvictionEntry *> candidates;
    for (const auto & [id, entry] : impl_->entries) {
        if (entry.priority != EvictionPriority::P0_COMPUTE && entry.location == from_location) {
            candidates.push_back(&entry);
        }
    }

    // Sort by priority (descending) then by last_access (ascending)
    // Higher priority number (P4=4) first, then older entries first
    std::sort(candidates.begin(), candidates.end(), [](const EvictionEntry * a, const EvictionEntry * b) {
        if (static_cast<int>(a->priority) != static_cast<int>(b->priority)) {
            return static_cast<int>(a->priority) > static_cast<int>(b->priority);
        }
        return a->last_access < b->last_access;
    });

    // Select victims until we have enough bytes
    for (const auto * entry : candidates) {
        if (total >= bytes_needed) {
            break;
        }
        victims.push_back(EvictionVictim{ entry->id, entry->size, entry->location, entry->priority });
        total += entry->size;
    }

    return victims;
}

size_t EvictionPolicy::get_total_size() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    size_t                      total = 0;
    for (const auto & [id, entry] : impl_->entries) {
        total += entry.size;
    }
    return total;
}

size_t EvictionPolicy::get_size_by_location(cache_location location) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    size_t                      total = 0;
    for (const auto & [id, entry] : impl_->entries) {
        if (entry.location == location) {
            total += entry.size;
        }
    }
    return total;
}

size_t EvictionPolicy::get_entry_count() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->entries.size();
}

}  // namespace ggml_sycl
