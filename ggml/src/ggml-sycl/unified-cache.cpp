//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "unified-cache.hpp"

#include "common.hpp"
#include "ggml-impl.h"
#include "ggml-sycl.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <unistd.h>
#endif

namespace ggml_sycl {

// Per-device cache storage (for PER_DEVICE and AUTO modes)
static std::unordered_map<int, std::unique_ptr<unified_cache>> g_device_caches;
static std::unordered_map<int, std::unique_ptr<host_cache>>    g_host_caches;
static std::mutex                                              g_cache_mutex;
static size_t                                                  g_unified_cache_budget     = 0;  // 0 = auto-calculate
static int                                                     g_unified_cache_budget_pct = 90;
static size_t                                                  g_unified_cache_host_budget     = 0;  // 0 = auto-calc
static int                                                     g_unified_cache_host_budget_pct = 90;
static unified_cache_mode                                      g_cache_mode               = unified_cache_mode::AUTO;
static std::atomic<bool> g_cache_mode_locked{ false };   // Locked after first cache access
static std::atomic<bool> g_sycl_shutting_down{ false };  // Set during shutdown to skip sycl::free()
static std::array<std::atomic<size_t>, GGML_SYCL_MAX_DEVICES> g_runtime_reserved_bytes{};
static std::atomic<bool> g_atexit_registered{ false };   // Ensure atexit handler registered once
// atexit handler to prevent SYCL cleanup during static destruction
static void unified_cache_atexit_handler() {
    g_sycl_shutting_down.store(true);
}

unified_cache::unified_cache(sycl::queue & queue, size_t budget_bytes, size_t staging_size) :
    queue_(queue),
    budget_(budget_bytes),
    base_budget_(budget_bytes),
    reserved_(0) {
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
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Failed to allocate staging buffer: %s\n", e.what());
        staging_      = nullptr;
        staging_size_ = 0;
    }

    GGML_LOG_INFO("[UNIFIED-CACHE] Initialized: budget=%.1f MB, staging=%.1f MB\n", budget_ / (1024.0f * 1024.0f),
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
        (void) queue_.get_context();
    } catch (...) {
        // SYCL runtime already destroyed, skip cleanup
        return;
    }

    // Free all cached entries
    for (auto & pair : entries_) {
        if (pair.second.device_ptr) {
            try {
                sycl::free(pair.second.device_ptr, queue_);
            } catch (...) {
            }
        }
    }

    // Free staging buffer
    if (staging_) {
        try {
            sycl::free(staging_, queue_);
        } catch (...) {
        }
    }

    // Free any deferred frees that haven't been released yet.
    for (auto & entry : deferred_frees_) {
        if (entry.ptr) {
            try {
                sycl::free(entry.ptr, queue_);
            } catch (...) {
            }
        }
    }
    deferred_frees_.clear();
}

// Fast 64-bit hash of entire data buffer (xxHash-style)
// Computes full content hash for robust change detection
// ~10 GB/s on modern CPUs - acceptable for one-time cache miss
static uint64_t compute_content_hash(const void * data, size_t size) {
    if (!data || size == 0) {
        return 0;
    }

    const uint8_t * bytes = static_cast<const uint8_t *>(data);

    // xxHash-style constants
    constexpr uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
    constexpr uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
    constexpr uint64_t PRIME3 = 0x165667B19E3779F9ULL;
    constexpr uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
    constexpr uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

    uint64_t hash = PRIME5 + size;

    auto load_u64_unaligned = [](const void * ptr) -> uint64_t {
        uint64_t value;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    };

    // Process 32-byte chunks for speed
    size_t num_chunks = size / 32;

    if (num_chunks > 0) {
        uint64_t v1 = hash + PRIME1 + PRIME2;
        uint64_t v2 = hash + PRIME2;
        uint64_t v3 = hash;
        uint64_t v4 = hash - PRIME1;

        for (size_t i = 0; i < num_chunks; i++) {
            const uint8_t * chunk = bytes + i * 32;
            uint64_t        k1    = load_u64_unaligned(chunk + 0);
            uint64_t        k2    = load_u64_unaligned(chunk + 8);
            uint64_t        k3    = load_u64_unaligned(chunk + 16);
            uint64_t        k4    = load_u64_unaligned(chunk + 24);

            v1 += k1 * PRIME2;
            v1 = (v1 << 31) | (v1 >> 33);
            v1 *= PRIME1;

            v2 += k2 * PRIME2;
            v2 = (v2 << 31) | (v2 >> 33);
            v2 *= PRIME1;

            v3 += k3 * PRIME2;
            v3 = (v3 << 31) | (v3 >> 33);
            v3 *= PRIME1;

            v4 += k4 * PRIME2;
            v4 = (v4 << 31) | (v4 >> 33);
            v4 *= PRIME1;
        }

        hash =
            ((v1 << 1) | (v1 >> 63)) + ((v2 << 7) | (v2 >> 57)) + ((v3 << 12) | (v3 >> 52)) + ((v4 << 18) | (v4 >> 46));

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
    size_t          remaining = size - (num_chunks * 32);
    const uint8_t * tail      = bytes + (num_chunks * 32);

    while (remaining >= 8) {
        uint64_t k = load_u64_unaligned(tail);
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

static size_t get_total_system_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullTotalPhys);
    }
    return 0;
#else
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
#endif
}

static bool is_host_accessible_ptr(const void * ptr, const sycl::queue & queue) {
    if (!ptr) {
        return false;
    }
    try {
        const sycl::usm::alloc alloc = sycl::get_pointer_type(ptr, queue.get_context());
        return alloc == sycl::usm::alloc::host || alloc == sycl::usm::alloc::shared;
    } catch (...) {
        return false;
    }
}

host_cache::host_cache(sycl::queue & queue, size_t budget_bytes) :
    queue_(queue),
    budget_(budget_bytes) {
    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(unified_cache_atexit_handler);
    }

    GGML_LOG_INFO("[UNIFIED-CACHE] Host cache initialized: budget=%.1f MB\n", budget_ / (1024.0f * 1024.0f));
}

host_cache::~host_cache() {
    if (g_sycl_shutting_down.load()) {
        return;
    }

    try {
        (void) queue_.get_context();
    } catch (...) {
        return;
    }

    for (auto & pair : entries_) {
        free_entry(pair.second);
    }
    entries_.clear();
    ptr_to_key_.clear();
}

void * host_cache::ensure_cached_alloc(const void *           key_ptr,
                                       const void *           src_ptr,
                                       size_t                 src_size,
                                       size_t                 dst_size,
                                       cache_entry_type       type,
                                       int                    layer_id,
                                       int                    expert_id,
                                       ggml_layout_mode       layout,
                                       bool                   validate_content,
                                       bool *                 needs_fill,
                                       bool *                 pinned_alloc_out,
                                       const cache_layout_xmx_info * xmx_info) {
    if (needs_fill) {
        *needs_fill = true;
    }
    if (pinned_alloc_out) {
        *pinned_alloc_out = false;
    }
    if (!key_ptr || !src_ptr || src_size == 0 || dst_size == 0) {
        return nullptr;
    }

    const bool   can_hash = validate_content && is_host_accessible_ptr(src_ptr, queue_);
    const uint64_t new_hash = can_hash ? compute_content_hash(src_ptr, src_size) : 0;

    std::lock_guard<std::mutex> lock(mutex_);

    unified_cache_key key{ type, key_ptr, layer_id, expert_id, layout };
    auto              it = entries_.find(key);
    if (it != entries_.end()) {
        auto & entry = it->second;
        bool size_changed   = (dst_size != entry.size);
        bool content_changed =
            validate_content && can_hash && (entry.content_hash != new_hash);
        bool src_changed = entry.src_ptr != src_ptr;
        bool needs_realloc = size_changed;
        bool needs_refill  = needs_realloc || src_changed || content_changed;

        if (needs_realloc) {
            const bool was_pinned = entry.pinned;
            entry.pinned          = true;
            while (used_.load() + dst_size > budget_) {
                if (evict_one() == 0) {
                    entry.pinned = was_pinned;
                    return nullptr;
                }
            }

            void * new_ptr = nullptr;
            bool   pinned_alloc = true;
            try {
                new_ptr = sycl::malloc_host(dst_size, queue_);
            } catch (...) {
                new_ptr = nullptr;
            }
            if (!new_ptr) {
                pinned_alloc = false;
                new_ptr      = std::malloc(dst_size);
            }
            if (!new_ptr) {
                entry.pinned = was_pinned;
                return nullptr;
            }

            free_entry(entry);

            entry.host_ptr     = new_ptr;
            entry.size         = dst_size;
            entry.pinned_alloc = pinned_alloc;
            entry.pinned       = was_pinned;
            used_ += dst_size;
        }

        entry.src_ptr      = src_ptr;
        entry.content_hash = can_hash ? new_hash : 0;
        entry.access_count++;
        entry.last_access = time_++;

        if (needs_fill) {
            *needs_fill = needs_refill;
        }
        if (pinned_alloc_out) {
            *pinned_alloc_out = entry.pinned_alloc;
        }
        if (xmx_info) {
            entry.xmx_info = *xmx_info;
        }
        return entry.host_ptr;
    }

    while (used_.load() + dst_size > budget_) {
        if (evict_one() == 0) {
            return nullptr;
        }
    }

    void * host_ptr = nullptr;
    bool   pinned_alloc = true;
    try {
        host_ptr = sycl::malloc_host(dst_size, queue_);
    } catch (...) {
        host_ptr = nullptr;
    }
    if (!host_ptr) {
        pinned_alloc = false;
        host_ptr     = std::malloc(dst_size);
    }
    if (!host_ptr) {
        return nullptr;
    }

    host_cache_entry entry{};
    entry.host_ptr     = host_ptr;
    entry.src_ptr      = src_ptr;
    entry.content_hash = can_hash ? new_hash : 0;
    entry.size         = dst_size;
    entry.type         = type;
    entry.layer_id     = layer_id;
    entry.expert_id    = expert_id;
    entry.layout       = layout;
    if (xmx_info) {
        entry.xmx_info = *xmx_info;
    }
    entry.access_count = 1;
    entry.last_access  = time_++;
    entry.pinned       = false;
    entry.owns_ptr     = true;
    entry.pinned_alloc = pinned_alloc;

    entries_[key]                    = entry;
    ptr_to_key_[{ key_ptr, layout }] = key;
    used_ += dst_size;

    if (needs_fill) {
        *needs_fill = true;
    }
    if (pinned_alloc_out) {
        *pinned_alloc_out = pinned_alloc;
    }

    return host_ptr;
}

bool host_cache::is_cached(const void * key_ptr, ggml_layout_mode layout) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ptr_to_key_.find({ key_ptr, layout }) != ptr_to_key_.end();
}

void * host_cache::get(const void * key_ptr, ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = ptr_to_key_.find({ key_ptr, layout });
    if (it == ptr_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    entry_it->second.access_count++;
    entry_it->second.last_access = time_++;
    return entry_it->second.host_ptr;
}

void host_cache::remove(const void *     key_ptr,
                        cache_entry_type type,
                        int              layer_id,
                        int              expert_id,
                        ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    unified_cache_key           key{ type, key_ptr, layer_id, expert_id, layout };
    auto                        it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }
    free_entry(it->second);
    entries_.erase(it);
    auto ptr_it = ptr_to_key_.find({ key_ptr, layout });
    if (ptr_it != ptr_to_key_.end() && ptr_it->second == key) {
        ptr_to_key_.erase(ptr_it);
    }
}

void host_cache::pin(const void * key_ptr, ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = ptr_to_key_.find({ key_ptr, layout });
    if (it != ptr_to_key_.end()) {
        auto entry_it = entries_.find(it->second);
        if (entry_it != entries_.end()) {
            entry_it->second.pinned = true;
        }
    }
}

void host_cache::unpin(const void * key_ptr, ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = ptr_to_key_.find({ key_ptr, layout });
    if (it != ptr_to_key_.end()) {
        auto entry_it = entries_.find(it->second);
        if (entry_it != entries_.end()) {
            entry_it->second.pinned = false;
        }
    }
}

void host_cache::unpin_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & pair : entries_) {
        pair.second.pinned = false;
    }
}

size_t host_cache::evict(size_t bytes_needed) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      freed = 0;
    while (freed < bytes_needed && !entries_.empty()) {
        size_t evicted = evict_one();
        if (evicted == 0) {
            break;
        }
        freed += evicted;
    }
    return freed;
}

size_t host_cache::evict_one() {
    float             min_score = std::numeric_limits<float>::max();
    unified_cache_key evict_key{};
    bool              found = false;

    for (auto & pair : entries_) {
        const auto & entry = pair.second;
        if (entry.pinned) {
            continue;
        }
        float score = compute_score(entry);
        if (score < min_score) {
            min_score = score;
            evict_key = pair.first;
            found     = true;
        }
    }

    if (!found) {
        return 0;
    }

    size_t evicted_bytes = 0;
    auto   it            = entries_.find(evict_key);
    if (it != entries_.end()) {
        evicted_bytes = it->second.size;
        free_entry(it->second);
        ptr_to_key_.erase({ evict_key.key_ptr, evict_key.layout });
        entries_.erase(it);
    }
    return evicted_bytes;
}

float host_cache::compute_score(const host_cache_entry & entry) const {
    int64_t age        = time_.load() - entry.last_access;
    float   decay      = std::exp(-DECAY_ALPHA * static_cast<float>(age));
    float   base_score = static_cast<float>(entry.access_count) * decay;
    if (entry.type == cache_entry_type::DENSE_WEIGHT) {
        return base_score * 2.0f;
    }
    return base_score;
}

void host_cache::free_entry(host_cache_entry & entry) {
    if (entry.host_ptr && entry.owns_ptr) {
        if (entry.pinned_alloc) {
            try {
                sycl::free(entry.host_ptr, queue_);
            } catch (...) {
            }
        } else {
            std::free(entry.host_ptr);
        }
        used_ -= entry.size;
    }
    entry.host_ptr = nullptr;
    entry.size     = 0;
}

void * unified_cache::ensure_cached(const void *     key_ptr,
                                    const void *     src_ptr,
                                    size_t           size,
                                    cache_entry_type type,
                                    int              layer_id,
                                    int              expert_id,
                                    ggml_layout_mode layout,
                                    bool             validate_content) {
    if (!key_ptr || !src_ptr || size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    process_deferred_frees();

    // Create key for lookup (uses stable key_ptr, not source data pointer)
    unified_cache_key key{ type, key_ptr, layer_id, expert_id, layout };

    // Check if already cached
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        // Entry exists - check if size or content changed
        // This handles the ABA problem: same key_ptr may be reused after free/alloc cycle
        bool need_realloc = (size != it->second.size);
        bool need_recopy  = need_realloc || (it->second.src_ptr != src_ptr) || validate_content;

        if (need_recopy) {
            uint64_t new_hash        = compute_content_hash(src_ptr, size);
            bool     content_changed = (it->second.content_hash != new_hash);

            if (need_realloc) {
                // Size changed - need to reallocate device buffer
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Size changed for key %p (%zu -> %zu bytes), reallocating\n", key_ptr,
                                it->second.size, size);

                const bool   was_pinned = it->second.pinned;
                const size_t old_size   = it->second.size;
                it->second.pinned       = true;
                while (used_.load() - old_size + size > budget_) {
                    if (evict_one(size) == 0) {
                        it->second.pinned = was_pinned;
                        GGML_LOG_WARN("[UNIFIED-CACHE] Cannot evict for realloc (used=%.1f MB, need=%.1f MB)\n",
                                      used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
                        return nullptr;
                    }
                }

                // Allocate new buffer with correct size
                void * new_device_ptr = nullptr;
                try {
                    new_device_ptr = sycl::malloc_device(size, queue_);
                } catch (const sycl::exception & e) {
                    GGML_LOG_ERROR("[UNIFIED-CACHE] realloc malloc_device failed: %s\n", e.what());
                    it->second.pinned = was_pinned;
                    return nullptr;
                }

                if (!new_device_ptr) {
                    GGML_LOG_ERROR("[UNIFIED-CACHE] realloc malloc_device returned nullptr\n");
                    it->second.pinned = was_pinned;
                    return nullptr;
                }

                // Release old buffer after new allocation succeeds
                enqueue_deferred_free(it->second.device_ptr, it->second.size);

                // Copy new data
                copy_to_device(new_device_ptr, src_ptr, size);

                // Update entry with new allocation
                it->second.device_ptr   = new_device_ptr;
                it->second.size         = size;
                it->second.content_hash = new_hash;
                it->second.src_ptr      = src_ptr;
                used_ += (size - old_size);
                it->second.pinned = was_pinned;
            } else if (content_changed) {
                // Same size but content changed - just re-upload
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Content changed for key %p (hash %llx -> %llx), re-uploading\n",
                                key_ptr, (unsigned long long) it->second.content_hash, (unsigned long long) new_hash);
                copy_to_device(it->second.device_ptr, src_ptr, size);
                it->second.content_hash = new_hash;
                it->second.src_ptr      = src_ptr;
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
        if (evict_one(size) == 0) {
            // All entries pinned, cannot evict
            GGML_LOG_WARN("[UNIFIED-CACHE] Cannot evict: all entries pinned (used=%.1f MB, need=%.1f MB)\n",
                          used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
            return nullptr;
        }
    }

    // Allocate device memory
    void * device_ptr = nullptr;
    try {
        device_ptr = sycl::malloc_device(size, queue_);
    } catch (const sycl::exception & e) {
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
    entry.device_ptr      = device_ptr;
    entry.src_ptr         = src_ptr;       // Track source for change detection
    entry.content_hash    = content_hash;  // Track content for change detection
    entry.size            = size;
    entry.type            = type;
    entry.layer_id        = layer_id;
    entry.expert_id       = expert_id;
    entry.layout          = layout;
    entry.access_count    = 1;
    entry.last_access     = time_++;
    entry.pinned          = false;
    entry.hot             = false;
    entry.state           = cache_entry_state::READY;
    entry.has_ready_event = false;
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here

    // Store in cache
    entries_[key]                    = entry;
    ptr_to_key_[{ key_ptr, layout }] = key;
    used_ += size;

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cached %s: %.2f MB (used=%.1f/%.1f MB)\n",
                    type == cache_entry_type::DENSE_WEIGHT ? "dense" : "expert", size / (1024.0f * 1024.0f),
                    used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f));

    return device_ptr;
}

void * unified_cache::ensure_cached_alloc(const void *     key_ptr,
                                          const void *     src_ptr,
                                          size_t           src_size,
                                          size_t           alloc_size,
                                          cache_entry_type type,
                                          int              layer_id,
                                          int              expert_id,
                                          ggml_layout_mode layout,
                                          bool             validate_content,
                                          bool *           needs_fill) {
    if (needs_fill) {
        *needs_fill = true;
    }
    if (!key_ptr || !src_ptr || src_size == 0 || alloc_size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    process_deferred_frees();

    unified_cache_key key{ type, key_ptr, layer_id, expert_id, layout };
    const uint64_t    new_hash = compute_content_hash(src_ptr, src_size);

    auto it = entries_.find(key);
    if (it != entries_.end()) {
        bool need_realloc = (alloc_size != it->second.size);
        bool content_changed =
            validate_content || (it->second.src_ptr != src_ptr) || (it->second.content_hash != new_hash);

        if (need_realloc) {
            const bool   was_pinned = it->second.pinned;
            const size_t old_size   = it->second.size;
            it->second.pinned       = true;
            // Ensure space for new allocation
            while (used_.load() - old_size + alloc_size > budget_) {
                if (evict_one(alloc_size) == 0) {
                    GGML_LOG_WARN("[UNIFIED-CACHE] Cannot evict for alloc (used=%.1f MB, need=%.1f MB)\n",
                                  used_.load() / (1024.0f * 1024.0f), alloc_size / (1024.0f * 1024.0f));
                    it->second.pinned = was_pinned;
                    if (needs_fill) {
                        *needs_fill = false;
                    }
                    return nullptr;
                }
            }

            void * new_device_ptr = nullptr;
            try {
                new_device_ptr = sycl::malloc_device(alloc_size, queue_);
            } catch (const sycl::exception & e) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] alloc malloc_device failed: %s\n", e.what());
                it->second.pinned = was_pinned;
                if (needs_fill) {
                    *needs_fill = false;
                }
                return nullptr;
            }

            if (!new_device_ptr) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] alloc malloc_device returned nullptr\n");
                it->second.pinned = was_pinned;
                if (needs_fill) {
                    *needs_fill = false;
                }
                return nullptr;
            }
            it->second.pinned = was_pinned;

            // Free old buffer after new allocation succeeds
            enqueue_deferred_free(it->second.device_ptr, it->second.size);

            it->second.device_ptr = new_device_ptr;
            it->second.size       = alloc_size;
            used_ += (alloc_size - old_size);
            content_changed = true;
        }

        it->second.src_ptr      = src_ptr;
        it->second.content_hash = new_hash;
        it->second.access_count++;
        it->second.last_access     = time_++;
        it->second.state           = cache_entry_state::READY;
        it->second.has_ready_event = false;

        if (needs_fill) {
            *needs_fill = need_realloc || content_changed;
        }
        return it->second.device_ptr;
    }

    // Need to allocate new entry
    while (used_.load() + alloc_size > budget_) {
        if (evict_one(alloc_size) == 0) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Cannot evict for alloc (used=%.1f MB, need=%.1f MB)\n",
                          used_.load() / (1024.0f * 1024.0f), alloc_size / (1024.0f * 1024.0f));
            return nullptr;
        }
    }

    void * device_ptr = nullptr;
    try {
        device_ptr = sycl::malloc_device(alloc_size, queue_);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] alloc malloc_device failed: %s\n", e.what());
        return nullptr;
    }

    if (!device_ptr) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] alloc malloc_device returned nullptr\n");
        return nullptr;
    }

    unified_cache_entry entry{};
    entry.device_ptr      = device_ptr;
    entry.src_ptr         = src_ptr;
    entry.content_hash    = new_hash;
    entry.size            = alloc_size;
    entry.type            = type;
    entry.layer_id        = layer_id;
    entry.expert_id       = expert_id;
    entry.layout          = layout;
    entry.access_count    = 1;
    entry.last_access     = time_++;
    entry.pinned          = false;
    entry.hot             = false;
    entry.state           = cache_entry_state::READY;
    entry.has_ready_event = false;

    entries_[key]                    = entry;
    ptr_to_key_[{ key_ptr, layout }] = key;
    used_ += alloc_size;

    if (needs_fill) {
        *needs_fill = true;
    }

    return device_ptr;
}

cache_layout_result unified_cache::ensure_cached_layout(const cache_layout_request &     request,
                                                        const std::vector<sycl::event> & deps) {
    cache_layout_result result{};
    result.layout   = request.layout;
    result.xmx_info = request.xmx_info;

    if (!request.key_ptr || !request.src_ptr || request.src_size == 0 || request.dst_size == 0) {
        result.status = cache_layout_status::INVALID;
        return result;
    }

    const unified_cache_key key{ request.type, request.key_ptr, request.layer_id, request.expert_id, request.layout };
    const bool              can_hash = request.validate_content && is_host_accessible_ptr(request.src_ptr, queue_);
    uint64_t                new_hash = can_hash ? compute_content_hash(request.src_ptr, request.src_size) : 0;

    void * device_ptr = nullptr;
    bool   needs_fill = false;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        process_deferred_frees();
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            while (it != entries_.end() && it->second.state == cache_entry_state::IN_PROGRESS &&
                   !it->second.has_ready_event) {
                entry_cv_.wait(lock, [&]() {
                    auto it_wait = entries_.find(key);
                    return it_wait == entries_.end() || it_wait->second.state != cache_entry_state::IN_PROGRESS ||
                           it_wait->second.has_ready_event;
                });
                it = entries_.find(key);
            }
        }

        if (it != entries_.end()) {
            auto & entry = it->second;
            // Refresh in-progress state if ready_event has completed
            if (entry.state == cache_entry_state::IN_PROGRESS && entry.has_ready_event &&
                event_complete(entry.ready_event)) {
                entry.state           = cache_entry_state::READY;
                entry.has_ready_event = false;
            }

            entry.access_count++;
            entry.last_access = time_++;

            if (entry.state == cache_entry_state::IN_PROGRESS) {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] layout pending: key=%p layout=%d size=%zu has_event=%d\n",
                                request.key_ptr, (int) request.layout, entry.size, entry.has_ready_event ? 1 : 0);
                result.device_ptr                      = entry.device_ptr;
                result.size                            = entry.size;
                result.status                          = cache_layout_status::IN_PROGRESS;
                std::vector<sycl::event> combined_deps = deps;
                if (entry.has_ready_event) {
                    combined_deps.push_back(entry.ready_event);
                }
                result.event = submit_barrier(combined_deps);
                return result;
            }

            if (entry.size != request.dst_size) {
                GGML_LOG_WARN("[UNIFIED-CACHE] layout size mismatch: key=%p cached=%zu req=%zu\n", request.key_ptr,
                              entry.size, request.dst_size);
                result.status = cache_layout_status::FAILED;
                return result;
            }

            bool content_changed = (entry.src_ptr != request.src_ptr);
            if (request.validate_content && can_hash) {
                content_changed = (entry.content_hash != new_hash) || content_changed;
            }

            if (!content_changed) {
                result.device_ptr = entry.device_ptr;
                result.size       = entry.size;
                result.status     = cache_layout_status::READY;
                result.event      = submit_barrier(deps);
                return result;
            }

            entry.src_ptr         = request.src_ptr;
            entry.content_hash    = can_hash ? new_hash : 0;
            entry.state           = cache_entry_state::IN_PROGRESS;
            entry.has_ready_event = false;
            entry.xmx_info        = request.xmx_info;
            device_ptr            = entry.device_ptr;
            needs_fill            = true;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] layout refresh: key=%p layout=%d size=%zu\n", request.key_ptr,
                            (int) request.layout, entry.size);
        } else {
            // Allocate new entry
            while (used_.load() + request.dst_size > budget_) {
                if (evict_one(request.dst_size) == 0) {
                    GGML_LOG_WARN("[UNIFIED-CACHE] Cannot evict for layout (used=%.1f MB, need=%.1f MB)\n",
                                  used_.load() / (1024.0f * 1024.0f), request.dst_size / (1024.0f * 1024.0f));
                    result.status = cache_layout_status::FAILED;
                    return result;
                }
            }

            void * new_device_ptr = nullptr;
            try {
                new_device_ptr = sycl::malloc_device(request.dst_size, queue_);
            } catch (const sycl::exception & e) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] layout malloc_device failed: %s\n", e.what());
                result.status = cache_layout_status::FAILED;
                return result;
            }

            if (!new_device_ptr) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] layout malloc_device returned nullptr\n");
                result.status = cache_layout_status::FAILED;
                return result;
            }

            if (can_hash) {
                new_hash = compute_content_hash(request.src_ptr, request.src_size);
            } else {
                new_hash = 0;
            }

            unified_cache_entry entry{};
            entry.device_ptr      = new_device_ptr;
            entry.src_ptr         = request.src_ptr;
            entry.content_hash    = can_hash ? new_hash : 0;
            entry.size            = request.dst_size;
            entry.type            = request.type;
            entry.layer_id        = request.layer_id;
            entry.expert_id       = request.expert_id;
            entry.layout          = request.layout;
            entry.xmx_info        = request.xmx_info;
            entry.access_count    = 1;
            entry.last_access     = time_++;
            entry.pinned          = false;
            entry.hot             = false;
            entry.state           = cache_entry_state::IN_PROGRESS;
            entry.has_ready_event = false;

            entries_[key]                                    = entry;
            ptr_to_key_[{ request.key_ptr, request.layout }] = key;
            used_ += request.dst_size;
            device_ptr = new_device_ptr;
            needs_fill = true;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] layout allocate: key=%p layout=%d size=%zu\n", request.key_ptr,
                            (int) request.layout, request.dst_size);
        }
    }

    if (!needs_fill || device_ptr == nullptr) {
        result.device_ptr = device_ptr;
        result.size       = request.dst_size;
        result.status     = cache_layout_status::FAILED;
        return result;
    }

    sycl::event fill_event;
    try {
        if (request.fill_fn) {
            fill_event = request.fill_fn(queue_, device_ptr, request.dst_size, request.src_ptr, request.src_size,
                                         request.fill_ctx, deps);
        } else {
            fill_event = copy_to_device_async(device_ptr, request.src_ptr, request.src_size, deps);
        }

        if (request.layout != GGML_LAYOUT_XMX_TILED && request.dst_size > request.src_size) {
            const size_t pad_bytes = request.dst_size - request.src_size;
            void *       pad_ptr   = static_cast<char *>(device_ptr) + request.src_size;
            fill_event             = queue_.submit([&](sycl::handler & cgh) {
                cgh.depends_on(fill_event);
                cgh.memset(pad_ptr, 0, pad_bytes);
            });
        }
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout fill failed: %s\n", e.what());
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = entries_.find(key);
        if (it != entries_.end()) {
            sycl::free(it->second.device_ptr, queue_);
            used_ -= it->second.size;
            entries_.erase(it);
            ptr_to_key_.erase({ request.key_ptr, request.layout });
        }
        entry_cv_.notify_all();
        result.status = cache_layout_status::FAILED;
        return result;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.ready_event     = fill_event;
            it->second.has_ready_event = true;
            it->second.state           = cache_entry_state::IN_PROGRESS;
        }
    }
    entry_cv_.notify_all();

    result.device_ptr = device_ptr;
    result.size       = request.dst_size;
    result.status     = cache_layout_status::IN_PROGRESS;
    result.event      = fill_event;
    return result;
}

bool unified_cache::is_cached(const void * key_ptr, ggml_layout_mode layout) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ptr_to_key_.find({ key_ptr, layout }) != ptr_to_key_.end();
}

bool unified_cache::is_cached_any(const void * key_ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & pair : entries_) {
        if (pair.first.key_ptr == key_ptr) {
            return true;
        }
    }
    return false;
}

void * unified_cache::get(const void * key_ptr, ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = ptr_to_key_.find({ key_ptr, layout });
    if (it == ptr_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    auto & entry = entry_it->second;
    if (entry.state == cache_entry_state::IN_PROGRESS) {
        if (entry.has_ready_event && event_complete(entry.ready_event)) {
            entry.state           = cache_entry_state::READY;
            entry.has_ready_event = false;
        } else {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] get pending: key=%p layout=%d size=%zu has_event=%d\n", key_ptr,
                            (int) layout, entry.size, entry.has_ready_event ? 1 : 0);
            return nullptr;
        }
    }
    return entry.device_ptr;
}

void unified_cache::remove(const void *     key_ptr,
                           cache_entry_type type,
                           int              layer_id,
                           int              expert_id,
                           ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    process_deferred_frees();
    unified_cache_key key{ type, key_ptr, layer_id, expert_id, layout };

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }
    if (it->second.state == cache_entry_state::IN_PROGRESS) {
        GGML_LOG_WARN("[UNIFIED-CACHE] remove skipped: entry in progress (key=%p)\n", key_ptr);
        return;
    }

    enqueue_deferred_free(it->second.device_ptr, it->second.size);

    entries_.erase(it);

    auto ptr_it = ptr_to_key_.find({ key_ptr, layout });
    if (ptr_it != ptr_to_key_.end() && ptr_it->second == key) {
        ptr_to_key_.erase(ptr_it);
    }
}

// NOTE: mark_reordered/is_reordered removed - reorder state tracked in tensor->extra->optimized_feature

void unified_cache::pin(const void * key_ptr, ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = ptr_to_key_.find({ key_ptr, layout });
    if (it != ptr_to_key_.end()) {
        auto entry_it = entries_.find(it->second);
        if (entry_it != entries_.end()) {
            entry_it->second.pinned = true;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin key=%p layout=%d\n", key_ptr, (int) layout);
        }
    }
}

void unified_cache::unpin(const void * key_ptr, ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = ptr_to_key_.find({ key_ptr, layout });
    if (it != ptr_to_key_.end()) {
        auto entry_it = entries_.find(it->second);
        if (entry_it != entries_.end()) {
            entry_it->second.pinned = false;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin key=%p layout=%d\n", key_ptr, (int) layout);
        }
    }
}

void unified_cache::unpin_experts() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT) {
            pair.second.pinned = false;
        }
    }
}

void unified_cache::unpin_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & pair : entries_) {
        pair.second.pinned = false;
    }
}

bool unified_cache::is_pinned(const void * key_ptr, ggml_layout_mode layout) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = ptr_to_key_.find({ key_ptr, layout });
    if (it == ptr_to_key_.end()) {
        return false;
    }
    auto entry_it = entries_.find(it->second);
    return entry_it != entries_.end() && entry_it->second.pinned;
}

size_t unified_cache::evict(size_t bytes_needed) {
    std::lock_guard<std::mutex> lock(mutex_);
    process_deferred_frees();

    size_t freed = 0;
    while (freed < bytes_needed && !entries_.empty()) {
        size_t evicted = evict_one(0);
        if (evicted == 0) {
            break;  // All entries pinned
        }
        freed += evicted;
    }
    return freed;
}

static int eviction_tier(const unified_cache_entry & entry) {
    // Tiered eviction priority:
    // 0: MoE experts (cold), 1: MoE experts (hot), 2: dense (cold), 3: dense (hot)
    const int base = (entry.type == cache_entry_type::DENSE_WEIGHT) ? 2 : 0;
    return base + (entry.hot ? 1 : 0);
}

size_t unified_cache::evict_one(size_t /* new_size */) {
    process_deferred_frees();

    unified_cache_key evict_key{};
    int               best_tier       = std::numeric_limits<int>::max();
    int64_t           best_last_access = std::numeric_limits<int64_t>::max();
    bool              found           = false;

    for (auto & pair : entries_) {
        auto & entry = pair.second;
        if (entry.state == cache_entry_state::IN_PROGRESS) {
            if (entry.has_ready_event && event_complete(entry.ready_event)) {
                entry.state           = cache_entry_state::READY;
                entry.has_ready_event = false;
            } else {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict skip: key=%p layout=%d in-progress size=%zu\n",
                                pair.first.key_ptr, (int) pair.first.layout, entry.size);
                continue;
            }
        }
        if (entry.pinned) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict skip: key=%p layout=%d pinned size=%zu\n", pair.first.key_ptr,
                            (int) pair.first.layout, entry.size);
            continue;
        }

        const int tier = eviction_tier(entry);
        if (tier < best_tier || (tier == best_tier && entry.last_access < best_last_access)) {
            best_tier        = tier;
            best_last_access = entry.last_access;
            evict_key = pair.first;
            found     = true;
        }
    }

    if (!found) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict failed: no eligible entries\n");
        return 0;  // All entries pinned
    }

    // Evict the entry
    size_t evicted_bytes = 0;
    auto   it            = entries_.find(evict_key);
    if (it != entries_.end()) {
        size_t entry_size = it->second.size;
        void * ptr        = it->second.device_ptr;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict key=%p layout=%d size=%zu\n", evict_key.key_ptr,
                        (int) evict_key.layout, entry_size);
        enqueue_deferred_free(ptr, entry_size);

        // Remove from lookup
        ptr_to_key_.erase({ evict_key.key_ptr, evict_key.layout });

        // Remove from entries
        entries_.erase(it);

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Evicted: key=%p layout=%d %.2f MB (used=%.1f/%.1f MB)\n", evict_key.key_ptr,
                        (int) evict_key.layout, entry_size / (1024.0f * 1024.0f), used_.load() / (1024.0f * 1024.0f),
                        budget_ / (1024.0f * 1024.0f));
        evicted_bytes = entry_size;
    }

    return evicted_bytes;
}

float unified_cache::compute_score(const unified_cache_entry & entry) const {
    int64_t age        = time_.load() - entry.last_access;
    float   decay      = std::exp(-DECAY_ALPHA * static_cast<float>(age));
    float   base_score = static_cast<float>(entry.access_count) * decay;

    // Dense weights get 2x priority (harder to evict than MoE experts)
    // Rationale: Dense weights are accessed every token, experts only when selected
    if (entry.type == cache_entry_type::DENSE_WEIGHT) {
        return base_score * 2.0f;
    }
    if (entry.hot) {
        constexpr float k_hot_boost = 1.5f;
        return base_score * k_hot_boost;
    }
    return base_score;
}

void unified_cache::copy_to_device(void * dst, const void * src, size_t size) {
    // Use staging buffer for mmap'd data
    if (staging_ && size <= staging_size_) {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        // Copy mmap -> staging (may trigger page fault)
        std::memcpy(staging_, src, size);
        // Copy staging -> device
        queue_.memcpy(dst, staging_, size).wait();
    } else if (staging_) {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        // Chunked transfer for large entries
        const char *                src_ptr   = static_cast<const char *>(src);
        char *                      dst_ptr   = static_cast<char *>(dst);
        size_t                      remaining = size;

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
        void * temp = sycl::malloc_host(size, queue_);
        if (temp) {
            std::memcpy(temp, src, size);
            queue_.memcpy(dst, temp, size).wait();
            sycl::free(temp, queue_);
        } else {
            GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to allocate temp staging\n");
        }
    }
}

sycl::event unified_cache::copy_to_device_async(void *                           dst,
                                                const void *                     src,
                                                size_t                           size,
                                                const std::vector<sycl::event> & deps) {
    const sycl::usm::alloc src_type = sycl::get_pointer_type(src, queue_.get_context());
    if (src_type == sycl::usm::alloc::unknown) {
        // Stage mmap'd or non-USM pointers through host memory.
        void * temp = sycl::malloc_host(size, queue_);
        if (temp) {
            std::memcpy(temp, src, size);
            sycl::event ev;
            if (deps.empty()) {
                ev = queue_.memcpy(dst, temp, size);
            } else {
                ev = queue_.submit([&](sycl::handler & cgh) {
                    cgh.depends_on(deps);
                    cgh.memcpy(dst, temp, size);
                });
            }
            auto * queue_ptr = &queue_;
            queue_.submit([&](sycl::handler & cgh) {
                cgh.depends_on(ev);
                cgh.host_task([temp, queue_ptr]() { sycl::free(temp, *queue_ptr); });
            });
            return ev;
        }

        if (staging_) {
            std::lock_guard<std::mutex> lock(staging_mutex_);
            if (!deps.empty()) {
                queue_.ext_oneapi_submit_barrier(deps).wait();
            }
            if (size <= staging_size_) {
                std::memcpy(staging_, src, size);
                sycl::event ev = queue_.memcpy(dst, staging_, size);
                ev.wait();
                return ev;
            }

            const char * src_ptr   = static_cast<const char *>(src);
            char *       dst_ptr   = static_cast<char *>(dst);
            size_t       remaining = size;
            sycl::event  last;

            while (remaining > 0) {
                const size_t chunk = std::min(remaining, staging_size_);
                std::memcpy(staging_, src_ptr, chunk);
                last = queue_.memcpy(dst_ptr, staging_, chunk);
                last.wait();
                src_ptr += chunk;
                dst_ptr += chunk;
                remaining -= chunk;
            }

            return last;
        }
    }

    if (deps.empty()) {
        return queue_.memcpy(dst, src, size);
    }
    return queue_.submit([&](sycl::handler & cgh) {
        cgh.depends_on(deps);
        cgh.memcpy(dst, src, size);
    });
}

bool unified_cache::event_complete(const sycl::event & evt) {
    try {
        auto status = evt.get_info<sycl::info::event::command_execution_status>();
        return status == sycl::info::event_command_status::complete;
    } catch (...) {
        return false;
    }
}

sycl::event unified_cache::submit_barrier(const std::vector<sycl::event> & deps) {
    if (deps.empty()) {
        return sycl::event{};
    }
    return queue_.ext_oneapi_submit_barrier(deps);
}

sycl::event unified_cache::submit_barrier_all() {
    return queue_.ext_oneapi_submit_barrier(std::vector<sycl::event>{});
}

void unified_cache::enqueue_deferred_free(void * ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }

    deferred_free_entry entry{};
    entry.ptr  = ptr;
    entry.size = size;
    try {
        entry.event     = submit_barrier_all();
        entry.has_event = true;
    } catch (...) {
        entry.has_event = false;
    }

    deferred_frees_.push_back(entry);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free: ptr=%p size=%zu\n", ptr, size);
}

void unified_cache::process_deferred_frees() {
    auto it = deferred_frees_.begin();
    while (it != deferred_frees_.end()) {
        const bool ready = !it->has_event || event_complete(it->event);
        if (!ready) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free pending: ptr=%p size=%zu\n", it->ptr, it->size);
            ++it;
            continue;
        }

        if (it->ptr) {
            if (!it->has_event) {
                try {
                    queue_.wait();
                } catch (...) {
                }
            }
            try {
                sycl::free(it->ptr, queue_);
            } catch (...) {
            }
            used_ -= it->size;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free done: ptr=%p size=%zu\n", it->ptr, it->size);
        }

        it = deferred_frees_.erase(it);
    }

    auto pin_it = inflight_unpins_.begin();
    while (pin_it != inflight_unpins_.end()) {
        const bool ready = !pin_it->has_event || event_complete(pin_it->event);
        if (!ready) {
            ++pin_it;
            continue;
        }
        auto key_it = ptr_to_key_.find({ pin_it->key_ptr, pin_it->layout });
        if (key_it != ptr_to_key_.end()) {
            auto entry_it = entries_.find(key_it->second);
            if (entry_it != entries_.end()) {
                entry_it->second.pinned = false;
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] in-flight unpin key=%p layout=%d\n", pin_it->key_ptr,
                                (int) pin_it->layout);
            }
        }
        pin_it = inflight_unpins_.erase(pin_it);
    }
}

size_t unified_cache::dense_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::DENSE_WEIGHT) {
            count++;
        }
    }
    return count;
}

size_t unified_cache::expert_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT) {
            count++;
        }
    }
    return count;
}

size_t unified_cache::used_bytes(cache_entry_type type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      total = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == type) {
            total += pair.second.size;
        }
    }
    return total;
}

void unified_cache::print_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = hits_.load() + misses_.load();
    float  rate  = total > 0 ? 100.0f * hits_.load() / total : 0.0f;

    size_t dense = 0, experts = 0;
    size_t dense_bytes = 0, expert_bytes = 0;
    size_t layout_counts[4] = { 0, 0, 0, 0 };
    size_t layout_bytes[4]  = { 0, 0, 0, 0 };
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::DENSE_WEIGHT) {
            dense++;
            dense_bytes += pair.second.size;
        } else {
            experts++;
            expert_bytes += pair.second.size;
        }
        const int layout_idx = static_cast<int>(pair.second.layout);
        if (layout_idx >= 0 && layout_idx < 4) {
            layout_counts[layout_idx]++;
            layout_bytes[layout_idx] += pair.second.size;
        }
    }

    GGML_LOG_INFO("[UNIFIED-CACHE] Stats: %zu hits, %zu misses (%.1f%% hit rate)\n", hits_.load(), misses_.load(),
                  rate);
    GGML_LOG_INFO("[UNIFIED-CACHE] Entries: %zu dense (%.1f MB), %zu experts (%.1f MB), total %.1f/%.1f MB\n", dense,
                  dense_bytes / (1024.0f * 1024.0f), experts, expert_bytes / (1024.0f * 1024.0f),
                  used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f));
    GGML_LOG_INFO(
        "[UNIFIED-CACHE] Layouts: aos=%zu (%.1f MB), soa=%zu (%.1f MB), coalesced=%zu (%.1f MB), xmx_tiled=%zu (%.1f "
        "MB)\n",
        layout_counts[GGML_LAYOUT_AOS], layout_bytes[GGML_LAYOUT_AOS] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_SOA], layout_bytes[GGML_LAYOUT_SOA] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_COALESCED], layout_bytes[GGML_LAYOUT_COALESCED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_XMX_TILED], layout_bytes[GGML_LAYOUT_XMX_TILED] / (1024.0f * 1024.0f));
}

void unified_cache::reset_stats() {
    hits_   = 0;
    misses_ = 0;
}

void unified_cache::update_reserved_bytes(size_t reserved_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    reserved_ = reserved_bytes;
    if (reserved_ >= base_budget_) {
        budget_ = 0;
    } else {
        budget_ = base_budget_ - reserved_;
    }
    while (used_.load() > budget_ && !entries_.empty()) {
        if (evict_one(0) == 0) {
            break;
        }
    }
    const size_t used = used_.load();
    if (used > budget_) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Cache usage (%.1f MB) exceeds budget (%.1f MB) after reserving %.1f MB\n",
                      used / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f),
                      reserved_ / (1024.0f * 1024.0f));
    }
}

void unified_cache::unpin_on_event(const void * key_ptr, ggml_layout_mode layout, const sycl::event & event) {
    if (!key_ptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_unpin_entry entry{};
    entry.key_ptr   = key_ptr;
    entry.layout    = layout;
    entry.event     = event;
    entry.has_event = true;
    inflight_unpins_.push_back(entry);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] in-flight pin key=%p layout=%d\n", key_ptr, (int) layout);
}

void unified_cache::set_hot(const void *     key_ptr,
                            cache_entry_type type,
                            int              layer_id,
                            int              expert_id,
                            ggml_layout_mode layout,
                            bool             hot) {
    std::lock_guard<std::mutex> lock(mutex_);
    unified_cache_key           key{ type, key_ptr, layer_id, expert_id, layout };
    auto                        it = entries_.find(key);
    if (it != entries_.end()) {
        it->second.hot = hot;
    }
}

void unified_cache::clear_hot_experts(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT && pair.second.layer_id == layer_id) {
            pair.second.hot = false;
        }
    }
}

// === Mode and Global Functions ===

unified_cache_mode get_unified_cache_mode() {
    // Check environment variable
    const char * env = std::getenv("GGML_SYCL_UNIFIED_CACHE_MODE");
    if (env) {
        if (strcmp(env, "global") == 0) {
            return unified_cache_mode::GLOBAL;
        }
        if (strcmp(env, "per_device") == 0) {
            return unified_cache_mode::PER_DEVICE;
        }
        if (strcmp(env, "auto") == 0) {
            return unified_cache_mode::AUTO;
        }
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
static int get_device_id_from_queue(sycl::queue & queue) {
    try {
        sycl::device dev          = queue.get_device();
        int          device_count = dpct::dev_mgr::instance().device_count();
        for (int i = 0; i < device_count; i++) {
            if (dpct::dev_mgr::instance().get_device(i) == dev) {
                return i;
            }
        }
    } catch (...) {
    }
    return dpct::dev_mgr::instance().current_device_id();
}

static size_t runtime_reserved_bytes_nolock(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    return g_runtime_reserved_bytes[device_id].load(std::memory_order_relaxed);
}

// Helper: Create cache for a device
static unified_cache * create_cache_for_device(int device_id) {
    // Get queue for this device
    sycl::queue & queue = dpct::dev_mgr::instance().get_device(device_id).default_queue();

    // Auto-calculate budget if not set
    size_t budget = g_unified_cache_budget;
    if (budget == 0) {
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
        const size_t base_mem = total_mem > 0 ? total_mem : free_mem;

        int pct = g_unified_cache_budget_pct;
        if (pct < 1) {
            pct = 1;
        } else if (pct > 100) {
            pct = 100;
        }

        budget = static_cast<size_t>(base_mem * (static_cast<double>(pct) / 100.0));

        const size_t min_headroom = 256ull * 1024ull * 1024ull;
        const size_t headroom     = std::max(min_headroom, base_mem / 10);
        if (base_mem > headroom && budget > base_mem - headroom) {
            budget = base_mem - headroom;
        }

        GGML_LOG_INFO("[UNIFIED-CACHE] Device %d: budget=%.1f MB (%d%% of %.1f MB total, headroom=%.1f MB)\n", device_id,
                      budget / (1024.0f * 1024.0f), pct, base_mem / (1024.0f * 1024.0f),
                      headroom / (1024.0f * 1024.0f));
    }

    const size_t reserved = runtime_reserved_bytes_nolock(device_id);
    if (reserved > 0) {
        if (reserved >= budget) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Runtime buffers reserve %.1f MB exceeds cache budget %.1f MB\n",
                          reserved / (1024.0f * 1024.0f), budget / (1024.0f * 1024.0f));
            budget = 0;
        } else {
            budget -= reserved;
        }
    }

    try {
        g_device_caches[device_id] = std::make_unique<unified_cache>(queue, budget);
        return g_device_caches[device_id].get();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to initialize device %d: %s\n", device_id, e.what());
        return nullptr;
    }
}

// Helper: Create host cache for a device
static host_cache * create_host_cache_for_device(int device_id) {
    sycl::queue & queue = dpct::dev_mgr::instance().get_device(device_id).default_queue();

    size_t budget = g_unified_cache_host_budget;
    if (budget == 0) {
        size_t total_mem = get_total_system_memory_bytes();
        if (total_mem == 0) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Host cache budget: unable to query system RAM, disabling host cache\n");
            return nullptr;
        }

        int pct = g_unified_cache_host_budget_pct;
        if (pct < 1) {
            pct = 1;
        } else if (pct > 100) {
            pct = 100;
        }

        budget = static_cast<size_t>(total_mem * (static_cast<double>(pct) / 100.0));
        GGML_LOG_INFO("[UNIFIED-CACHE] Host cache budget=%.1f MB (%d%% of %.1f MB total RAM)\n",
                      budget / (1024.0 * 1024.0), pct, total_mem / (1024.0 * 1024.0));
    }

    try {
        g_host_caches[device_id] = std::make_unique<host_cache>(queue, budget);
        return g_host_caches[device_id].get();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to initialize host cache for device %d: %s\n", device_id, e.what());
        return nullptr;
    }
}

unified_cache * get_unified_cache(sycl::queue & queue) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache_mode_locked = true;

    unified_cache_mode mode      = get_effective_mode();
    int                device_id = (mode == unified_cache_mode::GLOBAL) ? 0 : get_device_id_from_queue(queue);

    auto it = g_device_caches.find(device_id);
    if (it != g_device_caches.end()) {
        return it->second.get();
    }

    return create_cache_for_device(device_id);
}

host_cache * get_host_cache(sycl::queue & queue) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache_mode_locked = true;

    unified_cache_mode mode      = get_effective_mode();
    int                device_id = (mode == unified_cache_mode::GLOBAL) ? 0 : get_device_id_from_queue(queue);

    auto it = g_host_caches.find(device_id);
    if (it != g_host_caches.end()) {
        return it->second.get();
    }

    return create_host_cache_for_device(device_id);
}

unified_cache * get_unified_cache_for_device(int device_id) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache_mode_locked = true;

    unified_cache_mode mode             = get_effective_mode();
    int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device_id;

    auto it = g_device_caches.find(effective_device);
    if (it != g_device_caches.end()) {
        return it->second.get();
    }

    return create_cache_for_device(effective_device);
}

host_cache * get_host_cache_for_device(int device_id) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache_mode_locked = true;

    unified_cache_mode mode             = get_effective_mode();
    int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device_id;

    auto it = g_host_caches.find(effective_device);
    if (it != g_host_caches.end()) {
        return it->second.get();
    }

    return create_host_cache_for_device(effective_device);
}

bool unified_cache_enabled() {
    // Check if explicitly disabled
    const char * env = std::getenv("GGML_SYCL_UNIFIED_CACHE");
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

void set_unified_cache_budget_pct(int pct) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (g_cache_mode_locked) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Budget pct change ignored: cache already initialized\n");
        return;
    }
    if (pct < 1) {
        pct = 1;
    } else if (pct > 100) {
        pct = 100;
    }
    g_unified_cache_budget_pct = pct;
}

void set_unified_cache_host_budget_pct(int pct) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (g_cache_mode_locked) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Host budget pct change ignored: cache already initialized\n");
        return;
    }
    if (pct < 1) {
        pct = 1;
    } else if (pct > 100) {
        pct = 100;
    }
    g_unified_cache_host_budget_pct = pct;
}

void unified_cache_add_runtime_bytes(int device, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode mode             = get_effective_mode();
    int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    g_runtime_reserved_bytes[effective_device].fetch_add(bytes, std::memory_order_relaxed);
    auto it = g_device_caches.find(effective_device);
    if (it != g_device_caches.end()) {
        it->second->update_reserved_bytes(g_runtime_reserved_bytes[effective_device].load(std::memory_order_relaxed));
    }
}

void unified_cache_sub_runtime_bytes(int device, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode mode             = get_effective_mode();
    int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    size_t cur  = g_runtime_reserved_bytes[effective_device].load(std::memory_order_relaxed);
    size_t next = cur > bytes ? cur - bytes : 0;
    g_runtime_reserved_bytes[effective_device].store(next, std::memory_order_relaxed);
    auto it = g_device_caches.find(effective_device);
    if (it != g_device_caches.end()) {
        it->second->update_reserved_bytes(next);
    }
}

size_t unified_cache_get_runtime_bytes(int device) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode mode             = get_effective_mode();
    int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    return g_runtime_reserved_bytes[effective_device].load(std::memory_order_relaxed);
}

// === MoE Expert Caching API ===

void * cache_moe_expert(sycl::queue & queue, const void * mmap_ptr, size_t expert_size, int layer_id, int expert_id) {
    unified_cache * cache = get_unified_cache(queue);
    if (!cache) {
        return nullptr;
    }

    // For MoE experts, use mmap_ptr as both key and source
    // (mmap pointers are stable for MoE during inference)
    return cache->ensure_cached(mmap_ptr, mmap_ptr, expert_size, cache_entry_type::MOE_EXPERT, layer_id, expert_id,
                                GGML_LAYOUT_AOS);
}

bool is_expert_cached(const void * mmap_ptr) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Search all device caches
    for (const auto & [device_id, cache] : g_device_caches) {
        if (cache && cache->is_cached_any(mmap_ptr)) {
            return true;
        }
    }
    return false;
}

void * get_cached_expert(const void * mmap_ptr) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Search all device caches
    for (auto & [device_id, cache] : g_device_caches) {
        if (cache) {
            void * ptr = cache->get(mmap_ptr, GGML_LAYOUT_AOS);
            if (ptr) {
                return ptr;
            }
        }
    }
    return nullptr;
}

// NOTE: mark_expert_reordered/is_expert_reordered removed - reorder state tracked in tensor->extra->optimized_feature

void pin_expert(const void * mmap_ptr) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Pin in all caches that have this entry
    for (auto & [device_id, cache] : g_device_caches) {
        if (cache) {
            cache->pin(mmap_ptr, GGML_LAYOUT_AOS);
        }
    }
}

void unpin_all_experts() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Unpin in all caches
    for (auto & [device_id, cache] : g_device_caches) {
        if (cache) {
            cache->unpin_experts();
        }
    }
}

// Global reorder callback (set once, used for all MXFP4 experts)
static moe_reorder_callback_fn g_moe_reorder_callback = nullptr;

void set_moe_reorder_callback(moe_reorder_callback_fn callback) {
    g_moe_reorder_callback = callback;
}

void * cache_moe_expert_with_reorder(sycl::queue & queue,
                                     const void *  mmap_ptr,
                                     size_t        expert_size,
                                     int           layer_id,
                                     int           expert_id,
                                     int           ncols,
                                     int           nrows) {
    unified_cache * cache = get_unified_cache(queue);
    if (!cache) {
        return nullptr;
    }

    // Check if already cached
    void * existing = cache->get(mmap_ptr, GGML_LAYOUT_AOS);
    if (existing) {
        // Already cached - caller should check tensor->extra->optimized_feature for reorder state
        return existing;
    }

    // Cache the expert (newly cached)
    // For MoE experts, use mmap_ptr as both key and source
    void * device_ptr = cache->ensure_cached(mmap_ptr, mmap_ptr, expert_size, cache_entry_type::MOE_EXPERT, layer_id,
                                             expert_id, GGML_LAYOUT_AOS);
    if (!device_ptr) {
        return nullptr;
    }

    // Apply reorder callback if set (caller should update tensor->extra->optimized_feature)
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not in cache
    if (g_moe_reorder_callback && ncols > 0 && nrows > 0) {
        // Apply SoA reorder transformation
        g_moe_reorder_callback(static_cast<uint8_t *>(device_ptr), ncols, nrows, expert_size, &queue);
        if (!g_ggml_sycl_graph_recording) {
            queue.wait();  // Wait for reorder to complete
        }

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reordered expert L%d:E%d (%dx%d)\n", layer_id, expert_id, ncols, nrows);
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
    g_host_caches.clear();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Shutdown complete\n");
}

}  // namespace ggml_sycl
