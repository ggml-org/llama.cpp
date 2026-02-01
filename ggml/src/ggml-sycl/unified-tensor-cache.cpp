//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "unified-tensor-cache.hpp"

#include "common.hpp"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ggml_sycl {

unified_tensor_cache::unified_tensor_cache(sycl::queue & queue, size_t vram_budget, size_t host_budget) :
    queue_(queue),
    copy_queue_(queue.get_context(), queue.get_device()),
    vram_budget_(vram_budget),
    host_budget_(host_budget),
    vram_pool_(queue, vram_budget) {
    GGML_LOG_INFO("[SYCL] Unified tensor cache created: VRAM %.2f GB, Host %.2f GB\n",
                  vram_budget / (1024.0 * 1024.0 * 1024.0), host_budget / (1024.0 * 1024.0 * 1024.0));
}

unified_tensor_cache::~unified_tensor_cache() {
    print_stats();

    // Free any host memory we allocated
    for (auto & [id, entry] : entries_) {
        if (entry.owns_host && entry.host_ptr) {
            free_host(entry.host_ptr, entry.info.size);
        }
    }
}

void unified_tensor_cache::set_inventory(const tensor_inventory & inventory) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Store tensor info and build name-to-ID mapping
    for (size_t i = 0; i < inventory.tensors.size(); i++) {
        tensor_entry entry;
        entry.info         = inventory.tensors[i];
        entry.planned_tier = memory_tier::MMAP;  // Will be computed
        entries_[i]        = entry;

        // Build name -> ID mapping for lookups
        if (!entry.info.name.empty()) {
            name_to_id_[entry.info.name] = i;
        }
    }

    // Check if tiered mode needed
    tiered_enabled_ = should_enable_tiered(inventory, vram_budget_);

    if (tiered_enabled_) {
        GGML_LOG_INFO("[SYCL] Tiered memory enabled: model %.2f GB exceeds VRAM %.2f GB\n",
                      inventory.total_size / (1024.0 * 1024.0 * 1024.0), vram_budget_ / (1024.0 * 1024.0 * 1024.0));
    }

    compute_placement();
}

void unified_tensor_cache::compute_placement() {
    // Sort entries by priority (lower priority value = higher priority)
    std::vector<std::pair<uint64_t, tensor_entry *>> sorted;
    for (auto & [id, entry] : entries_) {
        sorted.push_back({ id, &entry });
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) {
        return a.second->info.static_priority < b.second->info.static_priority;
    });

    size_t vram_remaining = vram_budget_;
    size_t host_remaining = host_budget_;

    for (auto & [id, entry] : sorted) {
        if (entry->info.size <= vram_remaining) {
            entry->planned_tier = memory_tier::VRAM;
            vram_remaining -= entry->info.size;
        } else if (entry->info.size <= host_remaining) {
            entry->planned_tier = memory_tier::PINNED_HOST;
            host_remaining -= entry->info.size;
        } else {
            entry->planned_tier = memory_tier::MMAP;
        }
    }

    // Log placement summary
    size_t vram_count = 0, host_count = 0, mmap_count = 0;
    for (const auto & [id, entry] : entries_) {
        switch (entry.planned_tier) {
            case memory_tier::VRAM:
                vram_count++;
                break;
            case memory_tier::PINNED_HOST:
                host_count++;
                break;
            case memory_tier::MMAP:
                mmap_count++;
                break;
        }
    }
    GGML_LOG_INFO("[SYCL] Placement plan: %zu VRAM, %zu Host, %zu mmap\n", vram_count, host_count, mmap_count);
}

memory_tier unified_tensor_cache::get_planned_tier(uint64_t tensor_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = entries_.find(tensor_id);
    if (it == entries_.end()) {
        return memory_tier::MMAP;
    }
    return it->second.planned_tier;
}

std::optional<uint64_t> unified_tensor_cache::get_tensor_id(const std::string & name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = name_to_id_.find(name);
    if (it == name_to_id_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void unified_tensor_cache::load_tensor_data(uint64_t tensor_id, const void * src_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = entries_.find(tensor_id);
    if (it == entries_.end()) {
        GGML_LOG_ERROR("[SYCL] Unknown tensor ID: %llu\n", (unsigned long long) tensor_id);
        return;
    }

    tensor_entry & entry = it->second;
    if (entry.loaded) {
        return;  // Already loaded
    }

    const size_t size = entry.info.size;

    // Allocate based on planned tier
    if (entry.planned_tier == memory_tier::VRAM) {
        // Allocate VRAM and copy
        void * vram_ptr = vram_pool_.allocate(size, tensor_id);
        if (vram_ptr) {
            queue_.memcpy(vram_ptr, src_ptr, size).wait();
            entry.vram_ptr     = vram_ptr;
            entry.current_tier = memory_tier::VRAM;
            entry.loaded       = true;
            return;
        }
        // Fallback to host if VRAM allocation failed
        entry.planned_tier = memory_tier::PINNED_HOST;
    }

    if (entry.planned_tier == memory_tier::PINNED_HOST) {
        // Allocate host and copy
        void * host_ptr = allocate_host(size);
        if (host_ptr) {
            std::memcpy(host_ptr, src_ptr, size);
            entry.host_ptr     = host_ptr;
            entry.owns_host    = true;
            entry.current_tier = memory_tier::PINNED_HOST;
            entry.loaded       = true;
            return;
        }
        // Fallback to mmap
        entry.planned_tier = memory_tier::MMAP;
    }

    // Keep as mmap - just reference the source pointer
    entry.host_ptr     = const_cast<void *>(src_ptr);
    entry.owns_host    = false;
    entry.current_tier = memory_tier::MMAP;
    entry.loaded       = true;
}

tensor_location unified_tensor_cache::get_tensor_with_location(uint64_t tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = entries_.find(tensor_id);
    if (it == entries_.end()) {
        return { nullptr, memory_tier::MMAP };
    }

    tensor_entry & entry = it->second;
    entry.last_access    = time_++;
    entry.access_count++;

    // Return VRAM pointer if available
    if (entry.vram_ptr) {
        cache_hits_++;
        return { entry.vram_ptr, memory_tier::VRAM };
    }

    // Return host pointer
    if (entry.host_ptr) {
        cache_misses_++;
        return { entry.host_ptr, entry.current_tier };
    }

    cache_misses_++;
    return { nullptr, memory_tier::MMAP };
}

void unified_tensor_cache::request_promotion(uint64_t tensor_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = entries_.find(tensor_id);
    if (it == entries_.end()) {
        return;
    }

    tensor_entry & entry = it->second;

    // Already in VRAM
    if (entry.vram_ptr) {
        return;
    }

    // No host data to promote
    if (!entry.host_ptr) {
        return;
    }

    const size_t size = entry.info.size;

    // Try to allocate VRAM, evicting if needed
    while (vram_pool_.available() < size) {
        if (!evict_one(size)) {
            return;  // Can't evict enough
        }
    }

    void * vram_ptr = vram_pool_.allocate(size, tensor_id);
    if (!vram_ptr) {
        return;
    }

    // Async copy
    copy_queue_.memcpy(vram_ptr, entry.host_ptr, size);
    entry.vram_ptr     = vram_ptr;
    entry.current_tier = memory_tier::VRAM;
    promotions_++;
}

void unified_tensor_cache::prefetch(const std::vector<uint64_t> & tensor_ids) {
    for (uint64_t id : tensor_ids) {
        request_promotion(id);
    }
}

void unified_tensor_cache::wait_pending_transfers() {
    copy_queue_.wait();
}

bool unified_tensor_cache::evict_one(size_t needed_size) {
    (void) needed_size;  // Currently unused, may be used for smarter eviction

    // Find lowest-score VRAM-resident tensor
    uint64_t worst_id    = UINT64_MAX;
    float    worst_score = std::numeric_limits<float>::max();

    for (const auto & [id, entry] : entries_) {
        if (!entry.vram_ptr) {
            continue;  // Not in VRAM
        }
        if (entry.info.static_priority <= 1) {
            continue;  // Don't evict high-priority (embedding, output, attention)
        }

        float score = compute_score(entry);
        if (score < worst_score) {
            worst_score = score;
            worst_id    = id;
        }
    }

    if (worst_id == UINT64_MAX) {
        return false;  // Nothing to evict
    }

    // Evict
    tensor_entry & entry = entries_[worst_id];
    vram_pool_.deallocate(worst_id);
    entry.vram_ptr     = nullptr;
    entry.current_tier = entry.host_ptr ? memory_tier::PINNED_HOST : memory_tier::MMAP;
    evictions_++;

    return true;
}

float unified_tensor_cache::compute_score(const tensor_entry & entry) const {
    // Combined LRU + frequency scoring
    // Higher score = more valuable = don't evict
    float recency   = 1.0f / static_cast<float>(time_ - entry.last_access + 1);
    float frequency = static_cast<float>(entry.access_count);
    return 0.3f * recency + 0.7f * frequency;
}

void * unified_tensor_cache::allocate_host(size_t size) {
    if (host_used_ + size > host_budget_) {
        GGML_LOG_WARN("[SYCL] Host budget exceeded, can't allocate %zu bytes\n", size);
        return nullptr;
    }

    try {
        void * ptr = ggml_sycl_malloc_host(size, queue_, "unified_tensor_cache:host");
        if (ptr) {
            host_used_ += size;
        }
        return ptr;
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[SYCL] Host allocation failed: %s\n", e.what());
        return nullptr;
    }
}

void unified_tensor_cache::free_host(void * ptr, size_t size) {
    if (ptr) {
        try {
            sycl::free(ptr, queue_);
            host_used_ -= size;
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[SYCL] Host free failed: %s\n", e.what());
        }
    }
}

void unified_tensor_cache::print_stats() const {
    GGML_LOG_INFO("[SYCL] Unified tensor cache stats:\n");
    GGML_LOG_INFO("  VRAM: %.2f / %.2f GB\n", vram_pool_.used() / (1024.0 * 1024.0 * 1024.0),
                  vram_budget_ / (1024.0 * 1024.0 * 1024.0));
    GGML_LOG_INFO("  Host: %.2f / %.2f GB\n", host_used_ / (1024.0 * 1024.0 * 1024.0),
                  host_budget_ / (1024.0 * 1024.0 * 1024.0));
    GGML_LOG_INFO("  Hits: %zu, Misses: %zu, Promotions: %zu, Evictions: %zu\n", cache_hits_.load(),
                  cache_misses_.load(), promotions_.load(), evictions_.load());
}

// === Global Functions ===

bool should_enable_tiered(const tensor_inventory & inv, size_t vram_available) {
    // Enable tiered mode if model exceeds 90% of VRAM
    if (inv.total_size <= vram_available * 0.9) {
        return false;  // Fits in VRAM
    }
    return true;
}

}  // namespace ggml_sycl
