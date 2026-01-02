//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "unified-cache.hpp"
#include "common.hpp"
#include "ggml-sycl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace ggml_sycl {

// Per-device cache storage (for PER_DEVICE and AUTO modes)
static std::unordered_map<int, std::unique_ptr<unified_cache>> g_device_caches;
static std::mutex g_cache_mutex;
static size_t g_unified_cache_budget = 0;  // 0 = auto-calculate
static unified_cache_mode g_cache_mode = unified_cache_mode::AUTO;
static std::atomic<bool> g_cache_mode_locked{false};  // Locked after first cache access
static std::atomic<bool> g_sycl_shutting_down{false};  // Set during shutdown to skip sycl::free()
static std::atomic<bool> g_atexit_registered{false};   // Ensure atexit handler registered once

// atexit handler to prevent SYCL cleanup during static destruction
static void unified_cache_atexit_handler() {
    g_sycl_shutting_down.store(true);
}

unified_cache::unified_cache(sycl::queue& queue, size_t budget_bytes, size_t staging_size)
    : queue_(queue), budget_(budget_bytes) {

    // Register atexit handler once to set shutdown flag before static destructors run
    // This prevents the destructor from calling sycl::free() on invalid queue
    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(unified_cache_atexit_handler);
    }

    // Allocate staging buffer (pinned host memory)
    try {
        staging_ = sycl::malloc_host(staging_size, queue_);
        if (staging_) {
            staging_size_ = staging_size;
        }
    } catch (const sycl::exception& e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Failed to allocate staging buffer: %s\n", e.what());
        staging_ = nullptr;
        staging_size_ = 0;
    }

    GGML_LOG_INFO("[UNIFIED-CACHE] Initialized: budget=%.1f MB, staging=%.1f MB\n",
                  budget_ / (1024.0f * 1024.0f),
                  staging_size_ / (1024.0f * 1024.0f));
}

unified_cache::~unified_cache() {
    // Skip cleanup if SYCL runtime is shutting down (static destruction order issue)
    // This can happen when the program exits and static destructors run in undefined order
    if (g_sycl_shutting_down.load()) {
        return;
    }

    // Try to verify SYCL context is still valid before freeing
    // This guards against static destruction order issues where SYCL runtime
    // has been torn down before this destructor runs
    try {
        // Simple validity check - if this throws, SYCL is gone
        (void)queue_.get_context();
    } catch (...) {
        // SYCL runtime already destroyed, skip cleanup
        return;
    }

    // Free all cached entries
    for (auto& pair : entries_) {
        if (pair.second.device_ptr) {
            try {
                sycl::free(pair.second.device_ptr, queue_);
            } catch (...) {}
        }
    }

    // Free staging buffer
    if (staging_) {
        try {
            sycl::free(staging_, queue_);
        } catch (...) {}
    }
}

// Fast 64-bit hash of entire data buffer (xxHash-style)
// Computes full content hash for robust change detection
// ~10 GB/s on modern CPUs - acceptable for one-time cache miss
static uint64_t compute_content_hash(const void* data, size_t size) {
    if (!data || size == 0) return 0;

    const uint8_t* bytes = static_cast<const uint8_t*>(data);

    // xxHash-style constants
    constexpr uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
    constexpr uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
    constexpr uint64_t PRIME3 = 0x165667B19E3779F9ULL;
    constexpr uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
    constexpr uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

    uint64_t hash = PRIME5 + size;

    // Process 32-byte chunks for speed
    const uint64_t* chunks = reinterpret_cast<const uint64_t*>(bytes);
    size_t num_chunks = size / 32;

    if (num_chunks > 0) {
        uint64_t v1 = hash + PRIME1 + PRIME2;
        uint64_t v2 = hash + PRIME2;
        uint64_t v3 = hash;
        uint64_t v4 = hash - PRIME1;

        for (size_t i = 0; i < num_chunks; i++) {
            v1 += chunks[i * 4 + 0] * PRIME2;
            v1 = (v1 << 31) | (v1 >> 33);
            v1 *= PRIME1;

            v2 += chunks[i * 4 + 1] * PRIME2;
            v2 = (v2 << 31) | (v2 >> 33);
            v2 *= PRIME1;

            v3 += chunks[i * 4 + 2] * PRIME2;
            v3 = (v3 << 31) | (v3 >> 33);
            v3 *= PRIME1;

            v4 += chunks[i * 4 + 3] * PRIME2;
            v4 = (v4 << 31) | (v4 >> 33);
            v4 *= PRIME1;
        }

        hash = ((v1 << 1) | (v1 >> 63)) + ((v2 << 7) | (v2 >> 57)) +
               ((v3 << 12) | (v3 >> 52)) + ((v4 << 18) | (v4 >> 46));

        hash ^= ((v1 * PRIME2) << 31 | (v1 * PRIME2) >> 33) * PRIME1;
        hash = hash * PRIME1 + PRIME4;
        hash ^= ((v2 * PRIME2) << 31 | (v2 * PRIME2) >> 33) * PRIME1;
        hash = hash * PRIME1 + PRIME4;
        hash ^= ((v3 * PRIME2) << 31 | (v3 * PRIME2) >> 33) * PRIME1;
        hash = hash * PRIME1 + PRIME4;
        hash ^= ((v4 * PRIME2) << 31 | (v4 * PRIME2) >> 33) * PRIME1;
        hash = hash * PRIME1 + PRIME4;
    }

    // Process remaining 8-byte chunks
    size_t remaining = size - (num_chunks * 32);
    const uint8_t* tail = bytes + (num_chunks * 32);

    while (remaining >= 8) {
        uint64_t k = *reinterpret_cast<const uint64_t*>(tail);
        k *= PRIME2;
        k = (k << 31) | (k >> 33);
        k *= PRIME1;
        hash ^= k;
        hash = ((hash << 27) | (hash >> 37)) * PRIME1 + PRIME4;
        tail += 8;
        remaining -= 8;
    }

    // Process remaining bytes
    while (remaining > 0) {
        hash ^= static_cast<uint64_t>(*tail) * PRIME5;
        hash = ((hash << 11) | (hash >> 53)) * PRIME1;
        tail++;
        remaining--;
    }

    // Final avalanche
    hash ^= hash >> 33;
    hash *= PRIME2;
    hash ^= hash >> 29;
    hash *= PRIME3;
    hash ^= hash >> 32;

    return hash;
}

void* unified_cache::ensure_cached(const void* key_ptr, const void* src_ptr, size_t size,
                                    cache_entry_type type,
                                    int layer_id, int expert_id,
                                    bool validate_content) {
    if (!key_ptr || !src_ptr || size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Create key for lookup (uses stable key_ptr, not source data pointer)
    unified_cache_key key{type, key_ptr, layer_id, expert_id};

    // Check if already cached
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        // Entry exists - check if size or content changed
        // This handles the ABA problem: same key_ptr may be reused after free/alloc cycle
        bool need_realloc = (size != it->second.size);
        bool need_recopy = need_realloc || (it->second.src_ptr != src_ptr) || validate_content;

        if (need_recopy) {
            uint64_t new_hash = compute_content_hash(src_ptr, size);
            bool content_changed = (it->second.content_hash != new_hash);

            if (need_realloc) {
                // Size changed - need to reallocate device buffer
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Size changed for key %p (%zu -> %zu bytes), reallocating\n",
                                key_ptr, it->second.size, size);

                // Free old buffer
                sycl::free(it->second.device_ptr, queue_);
                used_ -= it->second.size;

                // Allocate new buffer with correct size
                void* new_device_ptr = nullptr;
                try {
                    new_device_ptr = sycl::malloc_device(size, queue_);
                } catch (const sycl::exception& e) {
                    GGML_LOG_ERROR("[UNIFIED-CACHE] realloc malloc_device failed: %s\n", e.what());
                    entries_.erase(it);
                    return nullptr;
                }

                if (!new_device_ptr) {
                    GGML_LOG_ERROR("[UNIFIED-CACHE] realloc malloc_device returned nullptr\n");
                    entries_.erase(it);
                    return nullptr;
                }

                // Copy new data
                copy_to_device(new_device_ptr, src_ptr, size);

                // Update entry with new allocation
                it->second.device_ptr = new_device_ptr;
                it->second.size = size;
                it->second.content_hash = new_hash;
                it->second.src_ptr = src_ptr;
                used_ += size;
            } else if (content_changed) {
                // Same size but content changed - just re-upload
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Content changed for key %p (hash %llx -> %llx), re-uploading\n",
                                key_ptr, (unsigned long long)it->second.content_hash, (unsigned long long)new_hash);
                copy_to_device(it->second.device_ptr, src_ptr, size);
                it->second.content_hash = new_hash;
                it->second.src_ptr = src_ptr;
            } else {
                // Same content from different pointer - just update src_ptr
                it->second.src_ptr = src_ptr;
            }
        }
        hits_++;
        // Update access stats
        it->second.access_count++;
        it->second.last_access = time_++;
        return it->second.device_ptr;
    }

    misses_++;

    // Need to allocate - check if we have space
    while (used_.load() + size > budget_) {
        // Need to evict
        if (!evict_one(size)) {
            // All entries pinned, cannot evict
            GGML_LOG_WARN("[UNIFIED-CACHE] Cannot evict: all entries pinned (used=%.1f MB, need=%.1f MB)\n",
                          used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
            return nullptr;
        }
    }

    // Allocate device memory
    void* device_ptr = nullptr;
    try {
        device_ptr = sycl::malloc_device(size, queue_);
    } catch (const sycl::exception& e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] malloc_device failed: %s\n", e.what());
        return nullptr;
    }

    if (!device_ptr) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] malloc_device returned nullptr\n");
        return nullptr;
    }

    // Copy data from source to device
    copy_to_device(device_ptr, src_ptr, size);

    // Compute content hash for new entry (only computed once on cache miss)
    uint64_t content_hash = compute_content_hash(src_ptr, size);

    // Create cache entry
    unified_cache_entry entry{};
    entry.device_ptr = device_ptr;
    entry.src_ptr = src_ptr;  // Track source for change detection
    entry.content_hash = content_hash;  // Track content for change detection
    entry.size = size;
    entry.type = type;
    entry.layer_id = layer_id;
    entry.expert_id = expert_id;
    entry.access_count = 1;
    entry.last_access = time_++;
    entry.pinned = false;
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here

    // Store in cache
    entries_[key] = entry;
    ptr_to_key_[key_ptr] = key;
    used_ += size;

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cached %s: %.2f MB (used=%.1f/%.1f MB)\n",
                    type == cache_entry_type::DENSE_WEIGHT ? "dense" : "expert",
                    size / (1024.0f * 1024.0f),
                    used_.load() / (1024.0f * 1024.0f),
                    budget_ / (1024.0f * 1024.0f));

    return device_ptr;
}

bool unified_cache::is_cached(const void* mmap_ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ptr_to_key_.find(mmap_ptr) != ptr_to_key_.end();
}

void* unified_cache::get(const void* mmap_ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_key_.find(mmap_ptr);
    if (it == ptr_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    return entry_it->second.device_ptr;
}

// NOTE: mark_reordered/is_reordered removed - reorder state tracked in tensor->extra->optimized_feature

void unified_cache::pin(const void* mmap_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_key_.find(mmap_ptr);
    if (it != ptr_to_key_.end()) {
        auto entry_it = entries_.find(it->second);
        if (entry_it != entries_.end()) {
            entry_it->second.pinned = true;
        }
    }
}

void unified_cache::unpin(const void* mmap_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_key_.find(mmap_ptr);
    if (it != ptr_to_key_.end()) {
        auto entry_it = entries_.find(it->second);
        if (entry_it != entries_.end()) {
            entry_it->second.pinned = false;
        }
    }
}

void unified_cache::unpin_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : entries_) {
        pair.second.pinned = false;
    }
}

bool unified_cache::is_pinned(const void* mmap_ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_key_.find(mmap_ptr);
    if (it == ptr_to_key_.end()) {
        return false;
    }
    auto entry_it = entries_.find(it->second);
    return entry_it != entries_.end() && entry_it->second.pinned;
}

size_t unified_cache::evict(size_t bytes_needed) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t freed = 0;
    while (freed < bytes_needed && !entries_.empty()) {
        if (!evict_one(0)) {
            break;  // All entries pinned
        }
        freed += 0;  // evict_one already updates used_
    }
    return freed;
}

bool unified_cache::evict_one(size_t /* new_size */) {
    // Find lowest-scoring non-pinned entry
    float min_score = std::numeric_limits<float>::max();
    unified_cache_key evict_key{};
    bool found = false;

    for (const auto& pair : entries_) {
        if (pair.second.pinned) continue;

        float score = compute_score(pair.second);
        if (score < min_score) {
            min_score = score;
            evict_key = pair.first;
            found = true;
        }
    }

    if (!found) {
        return false;  // All entries pinned
    }

    // Evict the entry
    auto it = entries_.find(evict_key);
    if (it != entries_.end()) {
        size_t entry_size = it->second.size;
        void* ptr = it->second.device_ptr;

        // Free device memory
        if (ptr) {
            try {
                sycl::free(ptr, queue_);
            } catch (...) {}
        }

        // Remove from lookup
        ptr_to_key_.erase(evict_key.key_ptr);

        // Remove from entries
        entries_.erase(it);

        // Update usage
        used_ -= entry_size;

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Evicted: %.2f MB (used=%.1f/%.1f MB)\n",
                        entry_size / (1024.0f * 1024.0f),
                        used_.load() / (1024.0f * 1024.0f),
                        budget_ / (1024.0f * 1024.0f));
    }

    return true;
}

float unified_cache::compute_score(const unified_cache_entry& entry) const {
    int64_t age = time_.load() - entry.last_access;
    float decay = std::exp(-DECAY_ALPHA * static_cast<float>(age));
    float base_score = static_cast<float>(entry.access_count) * decay;

    // Dense weights get 2x priority (harder to evict than MoE experts)
    // Rationale: Dense weights are accessed every token, experts only when selected
    if (entry.type == cache_entry_type::DENSE_WEIGHT) {
        return base_score * 2.0f;
    }
    return base_score;
}

void unified_cache::copy_to_device(void* dst, const void* src, size_t size) {
    // Use staging buffer for mmap'd data
    if (staging_ && size <= staging_size_) {
        // Copy mmap -> staging (may trigger page fault)
        std::memcpy(staging_, src, size);
        // Copy staging -> device
        queue_.memcpy(dst, staging_, size).wait();
    } else if (staging_) {
        // Chunked transfer for large entries
        const char* src_ptr = static_cast<const char*>(src);
        char* dst_ptr = static_cast<char*>(dst);
        size_t remaining = size;

        while (remaining > 0) {
            size_t chunk = std::min(remaining, staging_size_);
            std::memcpy(staging_, src_ptr, chunk);
            queue_.memcpy(dst_ptr, staging_, chunk).wait();
            src_ptr += chunk;
            dst_ptr += chunk;
            remaining -= chunk;
        }
    } else {
        // No staging - need temp allocation
        void* temp = sycl::malloc_host(size, queue_);
        if (temp) {
            std::memcpy(temp, src, size);
            queue_.memcpy(dst, temp, size).wait();
            sycl::free(temp, queue_);
        } else {
            GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to allocate temp staging\n");
        }
    }
}

size_t unified_cache::dense_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& pair : entries_) {
        if (pair.second.type == cache_entry_type::DENSE_WEIGHT) {
            count++;
        }
    }
    return count;
}

size_t unified_cache::expert_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT) {
            count++;
        }
    }
    return count;
}

void unified_cache::print_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = hits_.load() + misses_.load();
    float rate = total > 0 ? 100.0f * hits_.load() / total : 0.0f;

    size_t dense = 0, experts = 0;
    size_t dense_bytes = 0, expert_bytes = 0;
    for (const auto& pair : entries_) {
        if (pair.second.type == cache_entry_type::DENSE_WEIGHT) {
            dense++;
            dense_bytes += pair.second.size;
        } else {
            experts++;
            expert_bytes += pair.second.size;
        }
    }

    GGML_LOG_INFO("[UNIFIED-CACHE] Stats: %zu hits, %zu misses (%.1f%% hit rate)\n",
                  hits_.load(), misses_.load(), rate);
    GGML_LOG_INFO("[UNIFIED-CACHE] Entries: %zu dense (%.1f MB), %zu experts (%.1f MB), total %.1f/%.1f MB\n",
                  dense, dense_bytes / (1024.0f * 1024.0f),
                  experts, expert_bytes / (1024.0f * 1024.0f),
                  used_.load() / (1024.0f * 1024.0f),
                  budget_ / (1024.0f * 1024.0f));
}

void unified_cache::reset_stats() {
    hits_ = 0;
    misses_ = 0;
}

// === Mode and Global Functions ===

unified_cache_mode get_unified_cache_mode() {
    // Check environment variable
    const char* env = std::getenv("GGML_SYCL_UNIFIED_CACHE_MODE");
    if (env) {
        if (strcmp(env, "global") == 0) return unified_cache_mode::GLOBAL;
        if (strcmp(env, "per_device") == 0) return unified_cache_mode::PER_DEVICE;
        if (strcmp(env, "auto") == 0) return unified_cache_mode::AUTO;
    }
    return g_cache_mode;
}

void set_unified_cache_mode(unified_cache_mode mode) {
    if (g_cache_mode_locked) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Mode change ignored: cache already initialized\n");
        return;
    }
    g_cache_mode = mode;
}

// Helper: Determine effective mode (resolves AUTO)
static unified_cache_mode get_effective_mode() {
    unified_cache_mode mode = get_unified_cache_mode();
    if (mode == unified_cache_mode::AUTO) {
        // Auto-detect: use per_device if multiple GPUs available
        int device_count = dpct::dev_mgr::instance().device_count();
        return (device_count > 1) ? unified_cache_mode::PER_DEVICE : unified_cache_mode::GLOBAL;
    }
    return mode;
}

// Helper: Get device ID from queue
static int get_device_id_from_queue(sycl::queue& queue) {
    try {
        sycl::device dev = queue.get_device();
        int device_count = dpct::dev_mgr::instance().device_count();
        for (int i = 0; i < device_count; i++) {
            if (dpct::dev_mgr::instance().get_device(i) == dev) {
                return i;
            }
        }
    } catch (...) {}
    return dpct::dev_mgr::instance().current_device_id();
}

// Helper: Create cache for a device
static unified_cache* create_cache_for_device(int device_id) {
    // Get queue for this device
    sycl::queue& queue = dpct::dev_mgr::instance().get_device(device_id).default_queue();

    // Auto-calculate budget if not set
    size_t budget = g_unified_cache_budget;
    if (budget == 0) {
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);

        // Use 70% of free memory for unified cache
        budget = static_cast<size_t>(free_mem * 0.70);

        GGML_LOG_INFO("[UNIFIED-CACHE] Device %d: budget=%.1f MB (70%% of %.1f MB free)\n",
                      device_id,
                      budget / (1024.0f * 1024.0f),
                      free_mem / (1024.0f * 1024.0f));
    }

    try {
        g_device_caches[device_id] = std::make_unique<unified_cache>(queue, budget);
        return g_device_caches[device_id].get();
    } catch (const sycl::exception& e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to initialize device %d: %s\n", device_id, e.what());
        return nullptr;
    }
}

unified_cache* get_unified_cache(sycl::queue& queue) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache_mode_locked = true;

    unified_cache_mode mode = get_effective_mode();
    int device_id = (mode == unified_cache_mode::GLOBAL) ? 0 : get_device_id_from_queue(queue);

    auto it = g_device_caches.find(device_id);
    if (it != g_device_caches.end()) {
        return it->second.get();
    }

    return create_cache_for_device(device_id);
}

unified_cache* get_unified_cache_for_device(int device_id) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache_mode_locked = true;

    unified_cache_mode mode = get_effective_mode();
    int effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device_id;

    auto it = g_device_caches.find(effective_device);
    if (it != g_device_caches.end()) {
        return it->second.get();
    }

    return create_cache_for_device(effective_device);
}

bool unified_cache_enabled() {
    // Check if explicitly disabled
    const char* env = std::getenv("GGML_SYCL_UNIFIED_CACHE");
    if (env && std::atoi(env) == 0) {
        return false;  // Explicitly disabled
    }
    // Unified cache is now the default for MoE expert caching
    // Set GGML_SYCL_UNIFIED_CACHE=0 to disable
    return true;
}

void set_unified_cache_budget(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (g_cache_mode_locked) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Budget change ignored: cache already initialized\n");
        return;
    }
    g_unified_cache_budget = bytes;
}

// === MoE Expert Caching API ===

void* cache_moe_expert(sycl::queue& queue, const void* mmap_ptr, size_t expert_size,
                       int layer_id, int expert_id) {
    unified_cache* cache = get_unified_cache(queue);
    if (!cache) {
        return nullptr;
    }

    // For MoE experts, use mmap_ptr as both key and source
    // (mmap pointers are stable for MoE during inference)
    return cache->ensure_cached(mmap_ptr, mmap_ptr, expert_size,
                                 cache_entry_type::MOE_EXPERT,
                                 layer_id, expert_id);
}

bool is_expert_cached(const void* mmap_ptr) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Search all device caches
    for (const auto& [device_id, cache] : g_device_caches) {
        if (cache && cache->is_cached(mmap_ptr)) {
            return true;
        }
    }
    return false;
}

void* get_cached_expert(const void* mmap_ptr) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Search all device caches
    for (const auto& [device_id, cache] : g_device_caches) {
        if (cache) {
            void* ptr = cache->get(mmap_ptr);
            if (ptr) return ptr;
        }
    }
    return nullptr;
}

// NOTE: mark_expert_reordered/is_expert_reordered removed - reorder state tracked in tensor->extra->optimized_feature

void pin_expert(const void* mmap_ptr) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Pin in all caches that have this entry
    for (auto& [device_id, cache] : g_device_caches) {
        if (cache) {
            cache->pin(mmap_ptr);
        }
    }
}

void unpin_all_experts() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Unpin in all caches
    for (auto& [device_id, cache] : g_device_caches) {
        if (cache) {
            cache->unpin_all();
        }
    }
}

// Global reorder callback (set once, used for all MXFP4 experts)
static moe_reorder_callback_fn g_moe_reorder_callback = nullptr;

void set_moe_reorder_callback(moe_reorder_callback_fn callback) {
    g_moe_reorder_callback = callback;
}

void* cache_moe_expert_with_reorder(sycl::queue& queue, const void* mmap_ptr, size_t expert_size,
                                     int layer_id, int expert_id, int ncols, int nrows) {
    unified_cache* cache = get_unified_cache(queue);
    if (!cache) {
        return nullptr;
    }

    // Check if already cached
    void* existing = cache->get(mmap_ptr);
    if (existing) {
        // Already cached - caller should check tensor->extra->optimized_feature for reorder state
        return existing;
    }

    // Cache the expert (newly cached)
    // For MoE experts, use mmap_ptr as both key and source
    void* device_ptr = cache->ensure_cached(mmap_ptr, mmap_ptr, expert_size,
                                             cache_entry_type::MOE_EXPERT,
                                             layer_id, expert_id);
    if (!device_ptr) {
        return nullptr;
    }

    // Apply reorder callback if set (caller should update tensor->extra->optimized_feature)
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not in cache
    if (g_moe_reorder_callback && ncols > 0 && nrows > 0) {
        // Apply SoA reorder transformation
        g_moe_reorder_callback(static_cast<uint8_t*>(device_ptr), ncols, nrows, expert_size, &queue);
        queue.wait();  // Wait for reorder to complete

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reordered expert L%d:E%d (%dx%d)\n",
                        layer_id, expert_id, ncols, nrows);
    }

    return device_ptr;
}

void shutdown_unified_cache() {
    // Set shutdown flag FIRST so destructors skip sycl::free() calls
    g_sycl_shutting_down.store(true);

    // Clear all device caches
    // The destructors will skip cleanup due to the shutdown flag
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_device_caches.clear();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Shutdown complete\n");
}

} // namespace ggml_sycl
