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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <unordered_set>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <unistd.h>
#endif

namespace ggml_sycl {

// Per-device cache storage (for PER_DEVICE and AUTO modes)
static std::unordered_map<int, std::unique_ptr<unified_cache>> g_device_caches;
static std::unique_ptr<host_cache>                             g_host_cache_shared;
static std::mutex                                              g_cache_mutex;
static size_t                                                  g_unified_cache_budget      = 0;  // 0 = auto-calculate
static int                                                     g_unified_cache_budget_pct  = 90;
static size_t                                                  g_unified_cache_host_budget = 0;  // 0 = auto-calc
static int                                                     g_unified_cache_host_budget_pct = 90;
static unified_cache_mode                                      g_cache_mode = unified_cache_mode::AUTO;
static std::atomic<bool> g_cache_mode_locked{ false };   // Locked after first cache access
static std::atomic<bool> g_sycl_shutting_down{ false };  // Set during shutdown to skip sycl::free()
static std::array<std::atomic<size_t>, GGML_SYCL_MAX_DEVICES> g_runtime_reserved_bytes{};
static std::atomic<size_t>                                   g_runtime_reserved_host_bytes{};
static std::atomic<bool> g_atexit_registered{ false };   // Ensure atexit handler registered once
static std::atomic<int> g_host_cache_guard_errors{ 0 };
static std::atomic<int> g_host_cache_guard_enabled{ -1 };
static constexpr size_t k_host_cache_guard_bytes   = 64;
static constexpr uint8_t k_host_cache_guard_pattern = 0xA5;
static std::atomic<int> g_cache_assert_enabled{ -1 };
static std::atomic<int> g_copy_trace_enabled{ -1 };

static int get_device_id_from_queue(sycl::queue & queue);

static bool parse_env_mb_value(const char * name, size_t & out_mb) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    char * end = nullptr;
    long   mb  = std::strtol(env, &end, 10);
    if (end == env || mb < 0) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Invalid %s='%s'\n", name, env);
        return false;
    }
    out_mb = static_cast<size_t>(mb);
    return true;
}

static bool parse_env_count_value(const char * name, size_t & out_count) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    char * end  = nullptr;
    long   count = std::strtol(env, &end, 10);
    if (end == env || count < 0) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Invalid %s='%s'\n", name, env);
        return false;
    }
    out_count = static_cast<size_t>(count);
    return true;
}

static void resolve_dma_defaults(size_t & slice_bytes, size_t & buffer_count) {
    size_t slice_mb = 1024;
    size_t buffers  = 2;
    size_t env_val  = 0;

    const bool slice_env_set = parse_env_mb_value("GGML_SYCL_DMA_SLICE_MB", env_val);
    if (slice_env_set) {
        slice_mb = env_val;
    }
    const bool buffers_env_set = parse_env_count_value("GGML_SYCL_DMA_BUFFERS", env_val) ||
                                 parse_env_count_value("GGML_SYCL_DMA_SLICES", env_val);
    if (buffers_env_set) {
        buffers = env_val;
    }
    if (!slice_env_set && !buffers_env_set && ggml_backend_sycl_weights_evictable()) {
        // Use smaller defaults for evictable weights to reduce staging OOM risk.
        slice_mb = std::min<size_t>(slice_mb, 32);
        buffers  = std::min<size_t>(buffers, 1);
    }

    if (slice_bytes == 0) {
        slice_bytes = slice_mb * 1024ULL * 1024ULL;
    }
    if (buffer_count == 0) {
        buffer_count = buffers;
    }
}

static size_t resolve_host_staging_bytes() {
    size_t staging_mb = 64;
    size_t env_mb     = 0;
    if (parse_env_mb_value("GGML_SYCL_HOST_STAGING_MB", env_mb) ||
        parse_env_mb_value("GGML_SYCL_MMAP_STAGING_MB", env_mb)) {
        staging_mb = env_mb;
    }
    return staging_mb * 1024ULL * 1024ULL;
}

static size_t resolve_host_reserve_bytes(size_t staging_bytes) {
    size_t reserve_mb = 0;
    size_t env_mb     = 0;
    if (parse_env_mb_value("GGML_SYCL_HOST_RESERVE_MB", env_mb)) {
        reserve_mb = env_mb;
    } else {
        reserve_mb = staging_bytes / (1024ULL * 1024ULL);
    }
    return reserve_mb * 1024ULL * 1024ULL;
}

static bool onednn_pack_m_mismatch(const unified_cache_entry & entry, const cache_layout_request & request) {
    if (request.layout != GGML_LAYOUT_ONEDNN_PACKED && request.layout != GGML_LAYOUT_ONEDNN_WOQ) {
        return false;
    }
    return entry.onednn_pack_m != request.onednn_pack_m;
}

static void * host_cache_alloc_unpinned(size_t size, size_t alignment) {
    void * ptr = nullptr;
#if defined(_POSIX_C_SOURCE) || defined(__linux__)
    if (alignment < sizeof(void *)) {
        alignment = sizeof(void *);
    }
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#else
    (void) alignment;
    ptr = std::malloc(size);
#endif
    return ptr;
}

static bool host_cache_prefer_unpinned(cache_entry_type type) {
    // Default to pinned host allocations for GPU-accessed weights to avoid non-USM
    // pointers reaching kernels (can trigger device loss on Level Zero).
    // Allow opt-in to unpinned via env for debugging.
    const char * env = std::getenv("GGML_SYCL_HOST_CACHE_UNPINNED");
    if (env && std::atoi(env) != 0) {
        return ggml_backend_sycl_weights_evictable() && type == cache_entry_type::DENSE_WEIGHT;
    }
    return false;
}

static bool cache_assert_enabled() {
    int enabled = g_cache_assert_enabled.load(std::memory_order_acquire);
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_CACHE_ASSERT");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    g_cache_assert_enabled.store(enabled, std::memory_order_release);
    return enabled != 0;
}

static bool copy_trace_enabled() {
    int enabled = g_copy_trace_enabled.load(std::memory_order_acquire);
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_COPY_TRACE");
    enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    g_copy_trace_enabled.store(enabled, std::memory_order_release);
    return enabled != 0;
}

static bool host_cache_guard_enabled() {
    int enabled = g_host_cache_guard_enabled.load(std::memory_order_acquire);
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_HOST_CACHE_GUARD");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    g_host_cache_guard_enabled.store(enabled, std::memory_order_release);
    return enabled != 0;
}

static bool host_cache_check_guard_locked(const host_cache_entry & entry,
                                          const ggml_sycl_cache_id & key,
                                          const char * where) {
    if (entry.guard_size == 0 || entry.host_ptr == nullptr) {
        return true;
    }
    const uint8_t * guard = static_cast<const uint8_t *>(entry.host_ptr) + entry.size;
    for (size_t i = 0; i < entry.guard_size; ++i) {
        if (guard[i] != k_host_cache_guard_pattern) {
            g_host_cache_guard_errors.fetch_add(1, std::memory_order_relaxed);
            GGML_LOG_ERROR(
                "[UNIFIED-CACHE] host_cache guard corrupted at %s: model=%llu name_hash=0x%llx layout=%d size=%zu guard=%zu type=%d L%d E%d\n",
                where,
                (unsigned long long) key.model_id,
                (unsigned long long) key.name_hash,
                (int) entry.layout,
                entry.size,
                entry.guard_size,
                (int) entry.type,
                entry.layer_id,
                entry.expert_id);
            return false;
        }
    }
    return true;
}

static bool host_cache_check_pinned_guard_locked(const host_cache_entry & entry,
                                                 const ggml_sycl_cache_id & key,
                                                 const char * where) {
    if (!host_cache_guard_enabled()) {
        return true;
    }
    if (!entry.pinned_alloc || !entry.owns_ptr || entry.guard_size == 0 || entry.host_ptr == nullptr) {
        return true;
    }
    const uint8_t * guard = static_cast<const uint8_t *>(entry.host_ptr) + entry.size;
    for (size_t i = 0; i < entry.guard_size; ++i) {
        if (guard[i] != k_host_cache_guard_pattern) {
            g_host_cache_guard_errors.fetch_add(1, std::memory_order_relaxed);
            GGML_LOG_ERROR(
                "[UNIFIED-CACHE] pinned pool guard corrupted at %s: model=%llu name_hash=0x%llx layout=%d size=%zu guard=%zu type=%d L%d E%d\n",
                where,
                (unsigned long long) key.model_id,
                (unsigned long long) key.name_hash,
                (int) entry.layout,
                entry.size,
                entry.guard_size,
                (int) entry.type,
                entry.layer_id,
                entry.expert_id);
            return false;
        }
    }
    return true;
}

int host_cache_guard_error_count() {
    return g_host_cache_guard_errors.load(std::memory_order_acquire);
}

void host_cache_guard_reset() {
    g_host_cache_guard_errors.store(0, std::memory_order_release);
    g_host_cache_guard_enabled.store(-1, std::memory_order_release);
}

bool host_cache_guard_check_all(int device_id, const char * where) {
    host_cache * hcache = get_host_cache_for_device(device_id);
    if (!hcache) {
        return true;
    }
    return hcache->check_all_guards(where);
}

// atexit handler to prevent SYCL cleanup during static destruction
static void unified_cache_atexit_handler() {
    g_sycl_shutting_down.store(true);
}

unified_cache::unified_cache(sycl::queue & queue,
                             size_t       budget_bytes,
                             size_t       staging_size,
                             size_t       dma_reserved_bytes) :
    queue_(queue),
    budget_(budget_bytes),
    base_budget_(budget_bytes),
    reserved_(0),
    dma_reserved_bytes_(dma_reserved_bytes) {
    // Register atexit handler once to set shutdown flag before static destructors run
    // This prevents the destructor from calling sycl::free() on invalid queue
    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(unified_cache_atexit_handler);
    }

    // Allocate staging buffer (pinned host memory)
    try {
        staging_ = ggml_sycl_malloc_host(staging_size, queue_, "unified_cache:staging");
        if (staging_) {
            staging_size_ = staging_size;
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to allocate staging buffer: %s\n", e.what());
        staging_      = nullptr;
        staging_size_ = 0;
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Initialized: budget=%.1f MB, staging=%.1f MB, dma-reserve=%.1f MB\n",
                  budget_ / (1024.0f * 1024.0f),
                  staging_size_ / (1024.0f * 1024.0f),
                  dma_reserved_bytes_ / (1024.0f * 1024.0f));

    // Ensure unordered_map has buckets before any find() calls.
    entries_.rehash(1);
    id_to_key_.rehash(1);
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

    // Free DMA staging buffers
    for (void * ptr : dma_staging_buffers_) {
        if (!ptr) {
            continue;
        }
        try {
            sycl::free(ptr, queue_);
        } catch (...) {
        }
    }
    dma_staging_buffers_.clear();

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

static const char * usm_alloc_name(sycl::usm::alloc alloc) {
    switch (alloc) {
        case sycl::usm::alloc::host:
            return "host";
        case sycl::usm::alloc::shared:
            return "shared";
        case sycl::usm::alloc::device:
            return "device";
        default:
            return "unknown";
    }
}

host_cache::host_cache(sycl::queue & queue, size_t budget_bytes) :
    queue_(queue), budget_(budget_bytes), base_budget_(budget_bytes) {
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DEBUG: host_cache constructor started\n");
    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(unified_cache_atexit_handler);
    }

    // Create pinned pool with same budget
    // This uses 8GB chunks to bypass Intel Level Zero's ~11GB per-allocation limit
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DEBUG: Creating pinned pool\n");
    pinned_pool_ = std::make_unique<pinned_chunk_pool>(queue_, budget_bytes);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Host cache initialized: budget=%.1f MB (using pinned pool)\n",
                  budget_ / (1024.0f * 1024.0f));
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DEBUG: host_cache constructor finished\n");

    // Ensure unordered_maps have buckets before any find() calls.
    entries_.rehash(1);
}

void host_cache::update_reserved_bytes(size_t reserved_bytes) {
    reserved_ = reserved_bytes;
    if (reserved_ >= base_budget_) {
        budget_ = 0;
        GGML_LOG_INFO("[UNIFIED-CACHE] Host reserve %.1f MB >= base budget %.1f MB; host cache budget now 0 (used %.1f MB)\n",
                      reserved_ / (1024.0f * 1024.0f), base_budget_ / (1024.0f * 1024.0f),
                      used_.load() / (1024.0f * 1024.0f));
    } else {
        budget_ = base_budget_ - reserved_;
    }
    while (used_.load() > budget_ && !entries_.empty()) {
        if (evict_one() == 0) {
            break;
        }
    }
    const size_t used = used_.load();
    if (used > budget_) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Host cache usage (%.1f MB) exceeds budget (%.1f MB) after reserving %.1f MB\n",
                      used / (1024.0f * 1024.0f),
                      budget_ / (1024.0f * 1024.0f),
                      reserved_ / (1024.0f * 1024.0f));
    }
}

host_cache::~host_cache() {
    if (g_sycl_shutting_down.load()) {
        // During SYCL shutdown, we can't safely free memory
        // Release the pool without calling its destructor (which would try sycl::free)
        // This intentionally leaks memory during shutdown to avoid crashes
        if (pinned_pool_) {
            (void) pinned_pool_.release();  // Leak the pool to avoid sycl::free during shutdown
        }
        return;
    }

    try {
        (void) queue_.get_context();
    } catch (...) {
        // Context already destroyed - can't safely free memory
        if (pinned_pool_) {
            (void) pinned_pool_.release();  // Leak to avoid crash
        }
        return;
    }

    for (auto & pair : entries_) {
        free_entry(pair.second);
    }
    entries_.clear();
    // pinned_pool_ will be destroyed normally here (its destructor calls sycl::free)
}

void * host_cache::allocate_pinned_runtime(size_t size, size_t alignment) {
    if (!pinned_pool_) {
        return nullptr;
    }
    return pinned_pool_->allocate(size, alignment);
}

void host_cache::free_pinned_runtime(void * ptr, size_t size) {
    if (!pinned_pool_ || !ptr || size == 0) {
        return;
    }
    pinned_pool_->deallocate(ptr, size);
}


void * host_cache::ensure_cached_alloc(const ggml_sycl_cache_id &     key_id,
                                       const void *                  src_ptr,
                                       size_t                        src_size,
                                       size_t                        dst_size,
                                       cache_entry_type              type,
                                       int                           layer_id,
                                       int                           expert_id,
                                       ggml_layout_mode              layout,
                                       bool                          validate_content,
                                       bool *                        needs_fill,
                                       bool *                        pinned_alloc_out,
                                       cache_location *              location_out,
                                       const cache_layout_xmx_info * xmx_info) {
    if (needs_fill) {
        *needs_fill = true;
    }
    if (pinned_alloc_out) {
        *pinned_alloc_out = false;
    }
    if (location_out) {
        *location_out = cache_location::HOST_MMAP;
    }
    if (!key_id.valid || !src_ptr || src_size == 0 || dst_size == 0) {
        return nullptr;
    }

    const bool     host_accessible = is_host_accessible_ptr(src_ptr, queue_);
    const bool     can_hash = validate_content && host_accessible;
    const uint64_t new_hash = can_hash ? compute_content_hash(src_ptr, src_size) : 0;
    const bool     can_alias = host_accessible && layout == GGML_LAYOUT_AOS && src_size == dst_size;
    const bool     prefer_unpinned = host_cache_prefer_unpinned(type);

    std::lock_guard<std::mutex> lock(mutex_);
    const ggml_sycl_cache_id & key_id_ref = key_id;

    unified_cache_key key{ type, key_id_ref, layer_id, expert_id };
    auto              it = entries_.find(key);
    if (it != entries_.end() && it->second.layout != layout) {
        if (it->second.pinned) {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] host_cache layout switch blocked (pinned) model=%llu name_hash=0x%llx have=%d want=%d\n",
                (unsigned long long) key_id_ref.model_id,
                (unsigned long long) key_id_ref.name_hash,
                (int) it->second.layout,
                (int) layout);
            return nullptr;
        }
        free_entry(it->second);
        entries_.erase(it);
        it = entries_.end();
    }
    if (it != entries_.end()) {
        auto & entry           = it->second;
        if (!host_cache_check_guard_locked(entry, key_id_ref, "ensure_cached_alloc") ||
            !host_cache_check_pinned_guard_locked(entry, key_id_ref, "ensure_cached_alloc")) {
            if (cache_assert_enabled()) {
                GGML_ABORT("host_cache guard corruption detected");
            }
            return nullptr;
        }
        bool   size_changed    = (dst_size != entry.size);
        bool   content_changed = validate_content && can_hash && (entry.content_hash != new_hash);
        bool   src_changed     = entry.src_ptr != src_ptr;
        bool   needs_realloc   = size_changed;
        bool   needs_refill    = needs_realloc || src_changed || content_changed;

        if (needs_realloc) {
            const bool was_pinned = entry.pinned;
            entry.pinned          = true;
            while (used_.load() + dst_size > budget_) {
                if (evict_one() == 0) {
                    entry.pinned = was_pinned;
                    return nullptr;
                }
            }

            void * new_ptr      = nullptr;
            bool   pooled_alloc = false;
            size_t alloc_size = dst_size;
            size_t guard_size = 0;
            if (host_cache_guard_enabled()) {
                guard_size = k_host_cache_guard_bytes;
                alloc_size += guard_size;
            }

            if (can_alias && prefer_unpinned) {
                free_entry(entry);
                entry.host_ptr     = const_cast<void *>(src_ptr);
                entry.size         = dst_size;
                entry.guard_size   = 0;
                entry.pinned_alloc = false;
                entry.pinned       = was_pinned;
                entry.owns_ptr     = false;
                entry.location     = cache_location::HOST_MMAP;
                entry.src_ptr      = src_ptr;
                entry.content_hash = can_hash ? new_hash : 0;
                entry.access_count++;
                entry.last_access = time_++;
                if (needs_fill) {
                    *needs_fill = false;
                }
                if (pinned_alloc_out) {
                    *pinned_alloc_out = false;
                }
                if (location_out) {
                    *location_out = entry.location;
                }
                if (xmx_info) {
                    entry.xmx_info = *xmx_info;
                }
                if (g_ggml_sycl_debug >= 2) {
                    GGML_SYCL_DEBUG(
                        "[HOST-CACHE] alias reuse: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p pinned=%d owns=%d loc=%d\n",
                        (unsigned long long) key_id_ref.model_id,
                        (unsigned long long) key_id_ref.name_hash,
                        (int) layout,
                        entry.size,
                        entry.host_ptr,
                        entry.pinned_alloc ? 1 : 0,
                        entry.owns_ptr ? 1 : 0,
                        (int) entry.location);
                }
                return entry.host_ptr;
            }

            if (pinned_pool_ && !prefer_unpinned) {
                new_ptr = pinned_pool_->allocate(alloc_size);
                pooled_alloc = (new_ptr != nullptr);
            }

            if (!new_ptr) {
                new_ptr = host_cache_alloc_unpinned(alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
                pooled_alloc = false;
            }

            if (!new_ptr) {
                if (can_alias) {
                    free_entry(entry);
                    entry.host_ptr     = const_cast<void *>(src_ptr);
                    entry.size         = dst_size;
                    entry.guard_size   = 0;
                    entry.pinned_alloc = false;
                    entry.pinned       = was_pinned;
                    entry.owns_ptr     = false;
                    entry.location     = cache_location::HOST_MMAP;
                    entry.src_ptr      = src_ptr;
                    entry.content_hash = can_hash ? new_hash : 0;
                    entry.access_count++;
                    entry.last_access = time_++;
                    if (needs_fill) {
                        *needs_fill = false;
                    }
                    if (pinned_alloc_out) {
                        *pinned_alloc_out = false;
                    }
                    if (location_out) {
                        *location_out = entry.location;
                    }
                    if (xmx_info) {
                        entry.xmx_info = *xmx_info;
                    }
                    if (g_ggml_sycl_debug >= 2) {
                        GGML_SYCL_DEBUG(
                            "[HOST-CACHE] alias reuse: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p pinned=%d owns=%d loc=%d\n",
                            (unsigned long long) key_id_ref.model_id,
                            (unsigned long long) key_id_ref.name_hash,
                            (int) layout,
                            entry.size,
                            entry.host_ptr,
                            entry.pinned_alloc ? 1 : 0,
                            entry.owns_ptr ? 1 : 0,
                            (int) entry.location);
                    }
                    return entry.host_ptr;
                }
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] pinned pool alloc failed during realloc (%zu bytes)\n", alloc_size);
                entry.pinned = was_pinned;
                return nullptr;
            }

            if (guard_size > 0) {
                std::memset(static_cast<uint8_t *>(new_ptr) + dst_size, k_host_cache_guard_pattern, guard_size);
            }

            free_entry(entry);

            entry.host_ptr     = new_ptr;
            entry.size         = dst_size;
            entry.guard_size   = guard_size;
            entry.pinned_alloc = pooled_alloc;
            entry.pinned       = was_pinned;
            entry.owns_ptr     = true;
            entry.location     = pooled_alloc ? cache_location::HOST_PINNED : cache_location::HOST_MMAP;
            used_ += dst_size;
            if (g_ggml_sycl_debug >= 2) {
                GGML_SYCL_DEBUG(
                    "[HOST-CACHE] realloc: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p guard=%zu pinned=%d owns=%d loc=%d\n",
                    (unsigned long long) key_id_ref.model_id,
                    (unsigned long long) key_id_ref.name_hash,
                    (int) layout,
                    entry.size,
                    entry.host_ptr,
                    entry.guard_size,
                    entry.pinned_alloc ? 1 : 0,
                    entry.owns_ptr ? 1 : 0,
                    (int) entry.location);
            }
        }

        entry.src_ptr      = src_ptr;
        entry.content_hash = can_hash ? new_hash : 0;
        if (!entry.owns_ptr && src_changed) {
            entry.host_ptr = const_cast<void *>(src_ptr);
        }
        entry.access_count++;
        entry.last_access = time_++;

        if (needs_fill) {
            *needs_fill = entry.owns_ptr ? needs_refill : false;
        }
        if (pinned_alloc_out) {
            *pinned_alloc_out = entry.pinned_alloc;
        }
        if (location_out) {
            *location_out = entry.location;
        }
        if (xmx_info) {
            entry.xmx_info = *xmx_info;
        }
        return entry.host_ptr;
    }

    if (can_alias && prefer_unpinned) {
        host_cache_entry entry{};
        entry.host_ptr     = const_cast<void *>(src_ptr);
        entry.src_ptr      = src_ptr;
        entry.content_hash = can_hash ? new_hash : 0;
        entry.size         = dst_size;
        entry.guard_size   = 0;
        entry.type         = type;
        entry.layer_id     = layer_id;
        entry.expert_id    = expert_id;
        entry.layout       = layout;
        entry.access_count = 1;
        entry.last_access  = time_++;
        entry.pinned       = false;
        entry.owns_ptr     = false;
        entry.pinned_alloc = false;
        entry.location     = cache_location::HOST_MMAP;
        if (xmx_info) {
            entry.xmx_info = *xmx_info;
        }
        entries_[key] = entry;
        if (g_ggml_sycl_debug >= 2) {
            GGML_SYCL_DEBUG(
                "[HOST-CACHE] alias insert: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p pinned=%d owns=%d loc=%d\n",
                (unsigned long long) key_id_ref.model_id,
                (unsigned long long) key_id_ref.name_hash,
                (int) layout,
                entry.size,
                entry.host_ptr,
                entry.pinned_alloc ? 1 : 0,
                entry.owns_ptr ? 1 : 0,
                (int) entry.location);
        }
        if (needs_fill) {
            *needs_fill = false;
        }
        if (pinned_alloc_out) {
            *pinned_alloc_out = false;
        }
        if (location_out) {
            *location_out = entry.location;
        }
        return entry.host_ptr;
    }

    while (used_.load() + dst_size > budget_) {
        if (evict_one() == 0) {
            return nullptr;
        }
    }

    void * host_ptr      = nullptr;
    bool   pooled_alloc  = false;
    size_t alloc_size    = dst_size;
    size_t guard_size    = 0;
    if (host_cache_guard_enabled()) {
        guard_size = k_host_cache_guard_bytes;
        alloc_size += guard_size;
    }
    if (pinned_pool_ && !prefer_unpinned) {
        host_ptr     = pinned_pool_->allocate(alloc_size);
        pooled_alloc = (host_ptr != nullptr);
    }
    if (!host_ptr) {
        host_ptr = host_cache_alloc_unpinned(alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
        pooled_alloc = false;
    }
    if (!host_ptr) {
        if (can_alias) {
            host_cache_entry entry{};
            entry.host_ptr     = const_cast<void *>(src_ptr);
            entry.src_ptr      = src_ptr;
            entry.content_hash = can_hash ? new_hash : 0;
            entry.size         = dst_size;
            entry.guard_size   = 0;
            entry.type         = type;
            entry.layer_id     = layer_id;
            entry.expert_id    = expert_id;
            entry.layout       = layout;
            entry.access_count = 1;
            entry.last_access  = time_++;
            entry.pinned       = false;
            entry.owns_ptr     = false;
            entry.pinned_alloc = false;
            entry.location     = cache_location::HOST_MMAP;
            if (xmx_info) {
                entry.xmx_info = *xmx_info;
            }
            entries_[key] = entry;
            if (g_ggml_sycl_debug >= 2) {
                GGML_SYCL_DEBUG(
                    "[HOST-CACHE] alias insert: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p pinned=%d owns=%d loc=%d\n",
                    (unsigned long long) key_id_ref.model_id,
                    (unsigned long long) key_id_ref.name_hash,
                    (int) layout,
                    entry.size,
                    entry.host_ptr,
                    entry.pinned_alloc ? 1 : 0,
                    entry.owns_ptr ? 1 : 0,
                    (int) entry.location);
            }
            if (needs_fill) {
                *needs_fill = false;
            }
            if (pinned_alloc_out) {
                *pinned_alloc_out = false;
            }
            if (location_out) {
                *location_out = entry.location;
            }
            return entry.host_ptr;
        }
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] pinned pool alloc failed (%zu bytes)\n", alloc_size);
        return nullptr;
    }
    if (guard_size > 0) {
        std::memset(static_cast<uint8_t *>(host_ptr) + dst_size, k_host_cache_guard_pattern, guard_size);
    }

    host_cache_entry entry{};
    entry.host_ptr     = host_ptr;
    entry.src_ptr      = src_ptr;
    entry.content_hash = can_hash ? new_hash : 0;
    entry.size         = dst_size;
    entry.guard_size   = guard_size;
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
    entry.pinned_alloc = pooled_alloc;
    entry.location     = pooled_alloc ? cache_location::HOST_PINNED : cache_location::HOST_MMAP;

    entries_[key] = entry;
    used_ += dst_size;

    if (needs_fill) {
        *needs_fill = true;
    }
    if (pinned_alloc_out) {
        *pinned_alloc_out = pooled_alloc;
    }
    if (location_out) {
        *location_out = entry.location;
    }

    if (g_ggml_sycl_debug >= 2) {
        GGML_SYCL_DEBUG(
            "[HOST-CACHE] alloc: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p guard=%zu pinned=%d owns=%d loc=%d\n",
            (unsigned long long) key_id_ref.model_id,
            (unsigned long long) key_id_ref.name_hash,
            (int) layout,
            entry.size,
            entry.host_ptr,
            entry.guard_size,
            entry.pinned_alloc ? 1 : 0,
            entry.owns_ptr ? 1 : 0,
            (int) entry.location);
    }
    return host_ptr;
}

bool host_cache::is_cached(const ggml_sycl_cache_id & key_id,
                           cache_entry_type type,
                           int layer_id,
                           int expert_id,
                           ggml_layout_mode layout) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return false;
    }
    if (entries_.bucket_count() == 0) {
        const_cast<decltype(entries_) &>(entries_).rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto              it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }
    if (it->second.layout != layout) {
        return false;
    }
    return true;
}

void * host_cache::get(const ggml_sycl_cache_id & key_id,
                       cache_entry_type type,
                       int layer_id,
                       int expert_id,
                       ggml_layout_mode layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return nullptr;
    }
    if (entries_.bucket_count() == 0) {
        entries_.rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto entry_it = entries_.find(key);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    if (entry_it->second.layout != layout) {
        return nullptr;
    }
    if (!host_cache_check_guard_locked(entry_it->second, key_id, "get") ||
        !host_cache_check_pinned_guard_locked(entry_it->second, key_id, "get")) {
        if (cache_assert_enabled()) {
            GGML_ABORT("host_cache guard corruption detected");
        }
        return nullptr;
    }
    entry_it->second.access_count++;
    entry_it->second.last_access = time_++;
    return entry_it->second.host_ptr;
}

cache_location host_cache::get_location(const ggml_sycl_cache_id & key_id,
                                        cache_entry_type type,
                                        int layer_id,
                                        int expert_id,
                                        ggml_layout_mode layout) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return cache_location::HOST_MMAP;
    }
    if (entries_.bucket_count() == 0) {
        const_cast<decltype(entries_) &>(entries_).rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto entry_it = entries_.find(key);
    if (entry_it == entries_.end()) {
        return cache_location::HOST_MMAP;
    }
    if (entry_it->second.layout != layout) {
        return cache_location::HOST_MMAP;
    }
    if (!host_cache_check_guard_locked(entry_it->second, key_id, "get_location") ||
        !host_cache_check_pinned_guard_locked(entry_it->second, key_id, "get_location")) {
        if (cache_assert_enabled()) {
            GGML_ABORT("host_cache guard corruption detected");
        }
        return cache_location::HOST_MMAP;
    }
    return entry_it->second.location;
}

bool host_cache::check_guard(const ggml_sycl_cache_id & key_id,
                             cache_entry_type type,
                             int layer_id,
                             int expert_id,
                             ggml_layout_mode layout) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return true;
    }
    if (entries_.bucket_count() == 0) {
        const_cast<decltype(entries_) &>(entries_).rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto entry_it = entries_.find(key);
    if (entry_it == entries_.end()) {
        return true;
    }
    if (entry_it->second.layout != layout) {
        GGML_LOG_ERROR(
            "[UNIFIED-CACHE] host_cache layout mismatch in check_guard model=%llu name_hash=0x%llx have=%d want=%d\n",
            (unsigned long long) key_id.model_id,
            (unsigned long long) key_id.name_hash,
            (int) entry_it->second.layout,
            (int) layout);
        if (cache_assert_enabled()) {
            GGML_ABORT("host_cache layout mismatch");
        }
        return false;
    }
    return host_cache_check_guard_locked(entry_it->second, key_id, "check_guard") &&
           host_cache_check_pinned_guard_locked(entry_it->second, key_id, "check_guard");
}

bool host_cache::check_all_guards(const char * where) {
    if (!host_cache_guard_enabled()) {
        return true;
    }
    const char * tag = (where && where[0]) ? where : "check_all_guards";
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.bucket_count() == 0) {
        entries_.rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    bool ok = true;
    for (const auto & pair : entries_) {
        const unified_cache_key & key   = pair.first;
        const host_cache_entry &  entry = pair.second;
        if (!host_cache_check_guard_locked(entry, key.id, tag) ||
            !host_cache_check_pinned_guard_locked(entry, key.id, tag)) {
            ok = false;
        }
    }
    if (!ok && cache_assert_enabled()) {
        GGML_ABORT("host_cache guard corruption detected");
    }
    return ok;
}

void host_cache::remove(const ggml_sycl_cache_id & key_id,
                        cache_entry_type          type,
                        int                       layer_id,
                        int                       expert_id,
                        ggml_layout_mode          layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return;
    }
    if (entries_.bucket_count() == 0) {
        entries_.rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key           key{ type, key_id, layer_id, expert_id };
    auto                        it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }
    if (it->second.layout != layout) {
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] host_cache remove layout mismatch model=%llu name_hash=0x%llx have=%d want=%d (removing cached)\n",
            (unsigned long long) key_id.model_id,
            (unsigned long long) key_id.name_hash,
            (int) it->second.layout,
            (int) layout);
    }
    if (!host_cache_check_guard_locked(it->second, key_id, "remove") ||
        !host_cache_check_pinned_guard_locked(it->second, key_id, "remove")) {
        if (cache_assert_enabled()) {
            GGML_ABORT("host_cache guard corruption detected");
        }
    }
    free_entry(it->second);
    entries_.erase(it);
}

void host_cache::pin(const ggml_sycl_cache_id & key_id,
                     cache_entry_type          type,
                     int                       layer_id,
                     int                       expert_id,
                     ggml_layout_mode          layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return;
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto entry_it = entries_.find(key);
    if (entry_it != entries_.end() && entry_it->second.layout == layout) {
        entry_it->second.pinned = true;
    }
}

void host_cache::unpin(const ggml_sycl_cache_id & key_id,
                       cache_entry_type          type,
                       int                       layer_id,
                       int                       expert_id,
                       ggml_layout_mode          layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return;
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto entry_it = entries_.find(key);
    if (entry_it != entries_.end() && entry_it->second.layout == layout) {
        entry_it->second.pinned = false;
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
        if (entry.guard_size > 0) {
            const uint8_t * guard = static_cast<const uint8_t *>(entry.host_ptr) + entry.size;
            for (size_t i = 0; i < entry.guard_size; ++i) {
                if (guard[i] != k_host_cache_guard_pattern) {
                    g_host_cache_guard_errors.fetch_add(1, std::memory_order_relaxed);
                    GGML_LOG_ERROR("[UNIFIED-CACHE] host_cache guard corrupted: ptr=%p size=%zu guard=%zu layout=%d\n",
                                   entry.host_ptr, entry.size, entry.guard_size, (int) entry.layout);
                    break;
                }
            }
        }
        if (entry.pinned_alloc) {
            // Return to pinned pool
            pinned_pool_->deallocate(entry.host_ptr, entry.size + entry.guard_size);
        } else {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Freeing non-pinned host cache entry (unexpected)\n");
            std::free(entry.host_ptr);
        }
        used_ -= entry.size;
    }
    entry.host_ptr = nullptr;
    entry.size     = 0;
    entry.guard_size = 0;
}

void * unified_cache::ensure_cached(const ggml_sycl_cache_id & key_id,
                                    const void *     src_ptr,
                                    size_t           size,
                                    cache_entry_type type,
                                    int              layer_id,
                                    int              expert_id,
                                    ggml_layout_mode layout,
                                    bool             validate_content) {
    if (!key_id.valid || !src_ptr || size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    process_deferred_frees();

    // Create key for lookup (identity-only, no layout)
    unified_cache_key key{ type, key_id, layer_id, expert_id };

    // Check if already cached
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        auto id_it = id_to_key_.find(key_id);
        if (id_it == id_to_key_.end()) {
            id_to_key_.emplace(key_id, key);
        } else if (!(id_it->second == key)) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision in ensure_cached model=%llu name_hash=0x%llx\n",
                           (unsigned long long) key_id.model_id,
                           (unsigned long long) key_id.name_hash);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache id_to_key mismatch");
            }
        }
        if (it->second.layout != layout) {
            GGML_LOG_ERROR(
                "[UNIFIED-CACHE] layout mismatch in ensure_cached model=%llu name_hash=0x%llx have=%d want=%d\n",
                (unsigned long long) key_id.model_id,
                (unsigned long long) key_id.name_hash,
                (int) it->second.layout,
                (int) layout);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return nullptr;
        }
        // Entry exists - check if size or content changed
        // This handles ABA: same identity with new src_ptr/size
        bool need_realloc = (size != it->second.size);
        bool need_recopy  = need_realloc || (it->second.src_ptr != src_ptr) || validate_content;

        if (need_recopy) {
            uint64_t new_hash        = compute_content_hash(src_ptr, size);
            bool     content_changed = (it->second.content_hash != new_hash);

            if (need_realloc) {
                // Size changed - need to reallocate device buffer
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] Size changed for model=%llu name_hash=0x%llx (%zu -> %zu bytes), reallocating\n",
                    (unsigned long long) key_id.model_id,
                    (unsigned long long) key_id.name_hash,
                    it->second.size,
                    size);

                const bool   was_pinned = it->second.pinned;
                const size_t old_size   = it->second.size;
                it->second.pinned       = true;
                while (used_.load() - old_size + size > budget_) {
                    if (evict_one(size) == 0) {
                        it->second.pinned = was_pinned;
                        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict for realloc (used=%.1f MB, need=%.1f MB)\n",
                                      used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
                        return nullptr;
                    }
                }

                // Allocate new buffer with correct size
                void * new_device_ptr = nullptr;
                try {
                    new_device_ptr = ggml_sycl_malloc_device(size, queue_, "unified_cache:realloc");
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
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] Content changed for model=%llu name_hash=0x%llx (hash %llx -> %llx), re-uploading\n",
                    (unsigned long long) key_id.model_id,
                    (unsigned long long) key_id.name_hash,
                    (unsigned long long) it->second.content_hash,
                    (unsigned long long) new_hash);
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
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict: all entries pinned (used=%.1f MB, need=%.1f MB)\n",
                          used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
            return nullptr;
        }
    }

    // Allocate device memory
    void * device_ptr = nullptr;
    try {
        device_ptr = ggml_sycl_malloc_device(size, queue_, "unified_cache:alloc");
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
    entry.host_resident   = false;
    entry.location        = cache_location::DEVICE;
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here

    // Store in cache
    entries_[key] = entry;
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        id_to_key_.emplace(key_id, key);
    } else if (!(id_it->second == key)) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision on insert model=%llu name_hash=0x%llx\n",
                       (unsigned long long) key_id.model_id,
                       (unsigned long long) key_id.name_hash);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache id_to_key mismatch");
        }
    }
    used_ += size;

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cached %s: %.2f MB (used=%.1f/%.1f MB)\n",
                    type == cache_entry_type::DENSE_WEIGHT ? "dense" : "expert", size / (1024.0f * 1024.0f),
                    used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f));

    return device_ptr;
}

void * unified_cache::ensure_cached_alloc(const ggml_sycl_cache_id & key_id,
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
    if (!key_id.valid || !src_ptr || src_size == 0 || alloc_size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    process_deferred_frees();

    unified_cache_key key{ type, key_id, layer_id, expert_id };
    const uint64_t    new_hash = compute_content_hash(src_ptr, src_size);

    auto it = entries_.find(key);
    if (it != entries_.end()) {
        auto id_it = id_to_key_.find(key_id);
        if (id_it == id_to_key_.end()) {
            id_to_key_.emplace(key_id, key);
        } else if (!(id_it->second == key)) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision in ensure_cached_alloc model=%llu name_hash=0x%llx\n",
                           (unsigned long long) key_id.model_id,
                           (unsigned long long) key_id.name_hash);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache id_to_key mismatch");
            }
        }
        if (it->second.layout != layout) {
            if (it->second.pinned) {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout switch blocked (pinned) model=%llu name_hash=0x%llx have=%d want=%d\n",
                    (unsigned long long) key_id.model_id,
                    (unsigned long long) key_id.name_hash,
                    (int) it->second.layout,
                    (int) layout);
                return nullptr;
            }
            void * stale_ptr      = it->second.device_ptr;
            size_t stale_size     = it->second.size;
            bool   stale_host_res = it->second.host_resident;
            entries_.erase(it);
            it = entries_.end();
            if (!stale_host_res && stale_ptr && stale_size > 0) {
                enqueue_deferred_free(stale_ptr, stale_size);
            }
        }
        if (it == entries_.end()) {
            // Fall through to allocation path below
        } else {
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
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict for alloc (used=%.1f MB, need=%.1f MB)\n",
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
                new_device_ptr = ggml_sycl_malloc_device(alloc_size, queue_, "unified_cache:alloc");
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
    }

    // Need to allocate new entry
    while (used_.load() + alloc_size > budget_) {
        if (evict_one(alloc_size) == 0) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict for alloc (used=%.1f MB, need=%.1f MB)\n",
                          used_.load() / (1024.0f * 1024.0f), alloc_size / (1024.0f * 1024.0f));
            return nullptr;
        }
    }

    void * device_ptr = nullptr;
    try {
        device_ptr = ggml_sycl_malloc_device(alloc_size, queue_, "unified_cache:alloc");
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
    entry.host_resident   = false;
    entry.location        = cache_location::DEVICE;

    entries_[key] = entry;
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        id_to_key_.emplace(key_id, key);
    } else if (!(id_it->second == key)) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision on insert model=%llu name_hash=0x%llx\n",
                       (unsigned long long) key_id.model_id,
                       (unsigned long long) key_id.name_hash);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache id_to_key mismatch");
        }
    }
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
    result.onednn_pack_m = request.onednn_pack_m;
    result.xmx_info = request.xmx_info;

    if (!request.key.valid || !request.src_ptr || request.src_size == 0 || request.dst_size == 0) {
        result.status = cache_layout_status::INVALID;
        return result;
    }
    if (request.dst_size < request.src_size) {
        GGML_LOG_ERROR(
            "[UNIFIED-CACHE] invalid size: dst_size(%zu) < src_size(%zu) model=%llu name_hash=0x%llx layout=%d type=%d layer=%d expert=%d\n",
            request.dst_size,
            request.src_size,
            (unsigned long long) request.key.model_id,
            (unsigned long long) request.key.name_hash,
            (int) request.layout,
            (int) request.type,
            request.layer_id,
            request.expert_id);
        GGML_ASSERT(false && "cache layout dst_size < src_size");
    }

    if (g_ggml_sycl_debug) {
        sycl::usm::alloc alloc = sycl::usm::alloc::unknown;
        try {
            alloc = sycl::get_pointer_type(request.src_ptr, queue_.get_context());
        } catch (...) {
        }
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] layout request model=%llu name_hash=0x%llx type=%d layer=%d expert=%d layout=%d src=%p (%s) src_size=%zu "
            "dst_size=%zu used=%.1f MB budget=%.1f MB base=%.1f MB reserved=%.1f MB avail=%.1f MB\n",
            (unsigned long long) request.key.model_id,
            (unsigned long long) request.key.name_hash,
            static_cast<int>(request.type),
            request.layer_id,
            request.expert_id,
            static_cast<int>(request.layout),
            request.src_ptr,
            usm_alloc_name(alloc),
            request.src_size,
            request.dst_size,
            used_.load() / (1024.0f * 1024.0f),
            budget_ / (1024.0f * 1024.0f),
            base_budget_ / (1024.0f * 1024.0f),
            reserved_ / (1024.0f * 1024.0f),
            available() / (1024.0f * 1024.0f));
    }

    const unified_cache_key key{ request.type, request.key, request.layer_id, request.expert_id };
    if (ggml_sycl_graph_recording_active()) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = entries_.find(key);
        if (it != entries_.end() && it->second.state == cache_entry_state::READY &&
            it->second.size == request.dst_size) {
            if (it->second.layout != request.layout || onednn_pack_m_mismatch(it->second, request)) {
                GGML_LOG_ERROR(
                    "[UNIFIED-CACHE] layout mismatch in graph mode model=%llu name_hash=0x%llx have=%d want=%d\n",
                    (unsigned long long) request.key.model_id,
                    (unsigned long long) request.key.name_hash,
                    (int) it->second.layout,
                    (int) request.layout);
                if (cache_assert_enabled()) {
                    GGML_ABORT("unified_cache layout mismatch");
                }
                result.status = cache_layout_status::FAILED;
                return result;
            }
            result.device_ptr    = it->second.device_ptr;
            result.size          = it->second.size;
            result.status        = cache_layout_status::READY;
            result.host_resident = it->second.host_resident;
            result.location      = it->second.location;
            result.onednn_pack_m = it->second.onednn_pack_m;
            result.event         = submit_barrier(deps);
            return result;
        }
        result.status = cache_layout_status::FAILED;
        return result;
    }
    const bool              can_hash = request.validate_content && is_host_accessible_ptr(request.src_ptr, queue_);
    uint64_t                new_hash = can_hash ? compute_content_hash(request.src_ptr, request.src_size) : 0;

    auto try_host_fallback = [&](const char * reason) -> bool {
        host_cache * hcache = get_host_cache(queue_);
        if (!hcache) {
            return false;
        }
        cache_location host_loc = hcache->get_location(request.key, request.type, request.layer_id, request.expert_id,
                                                       request.layout);
        void *         host_ptr = hcache->get(request.key, request.type, request.layer_id, request.expert_id,
                                              request.layout);
        if (!host_ptr) {
            bool needs_host_fill = false;
            bool pinned_alloc    = false;
            host_ptr = hcache->ensure_cached_alloc(request.key, request.src_ptr, request.src_size, request.dst_size,
                                                   request.type, request.layer_id, request.expert_id, request.layout,
                                                   false, &needs_host_fill, &pinned_alloc, &host_loc,
                                                   &request.xmx_info);
            if (host_ptr && needs_host_fill) {
                if (request.fill_fn) {
                    request.fill_fn(queue_, host_ptr, request.dst_size, request.src_ptr, request.src_size,
                                    request.fill_ctx, {});
                } else {
                    std::memcpy(host_ptr, request.src_ptr, std::min(request.dst_size, request.src_size));
                    if (request.dst_size > request.src_size) {
                        std::memset(static_cast<char *>(host_ptr) + request.src_size, 0,
                                    request.dst_size - request.src_size);
                    }
                }
            }
        }
        if (!host_ptr) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto                        it = entries_.find(key);
            if (it != entries_.end()) {
                if (!it->second.host_resident && it->second.device_ptr) {
                    try {
                        queue_.wait();
                    } catch (...) {
                    }
                    try {
                        sycl::free(it->second.device_ptr, queue_);
                        used_ -= it->second.size;
                    } catch (...) {
                    }
                }
                it->second.device_ptr      = host_ptr;
                it->second.src_ptr         = request.src_ptr;
                it->second.content_hash    = can_hash ? new_hash : 0;
                it->second.size            = request.dst_size;
                it->second.type            = request.type;
                it->second.layer_id        = request.layer_id;
                it->second.expert_id       = request.expert_id;
                it->second.layout          = request.layout;
                it->second.onednn_pack_m   = request.onednn_pack_m;
                it->second.xmx_info        = request.xmx_info;
                it->second.access_count++;
                it->second.last_access     = time_++;
                it->second.state           = cache_entry_state::READY;
                it->second.has_ready_event = false;
                it->second.host_resident   = true;
                it->second.location        = host_loc;
            } else {
                unified_cache_entry entry{};
                entry.device_ptr      = host_ptr;
                entry.src_ptr         = request.src_ptr;
                entry.content_hash    = can_hash ? new_hash : 0;
                entry.size            = request.dst_size;
                entry.type            = request.type;
                entry.layer_id        = request.layer_id;
                entry.expert_id       = request.expert_id;
                entry.layout          = request.layout;
                entry.onednn_pack_m   = request.onednn_pack_m;
                entry.xmx_info        = request.xmx_info;
                entry.access_count    = 1;
                entry.last_access     = time_++;
                entry.pinned          = false;
                entry.hot             = false;
                entry.state           = cache_entry_state::READY;
                entry.has_ready_event = false;
                entry.host_resident   = true;
                entry.location        = host_loc;
                entries_[key]         = entry;
                auto id_it = id_to_key_.find(request.key);
                if (id_it == id_to_key_.end()) {
                    if (id_to_key_.bucket_count() == 0) {
                        id_to_key_.rehash(1);
                    }
                    id_to_key_.emplace(request.key, key);
                } else if (!(id_it->second == key)) {
                    GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision on host fallback model=%llu name_hash=0x%llx\n",
                                   (unsigned long long) request.key.model_id,
                                   (unsigned long long) request.key.name_hash);
                    if (cache_assert_enabled()) {
                        GGML_ABORT("unified_cache id_to_key mismatch");
                    }
                }
            }
            entry_cv_.notify_all();
        }
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] Host fallback (%s): model=%llu name_hash=0x%llx layout=%d size=%.1f MB\n",
            reason ? reason : "unknown",
            (unsigned long long) request.key.model_id,
            (unsigned long long) request.key.name_hash,
            (int) request.layout,
            request.dst_size / (1024.0f * 1024.0f));
        result.device_ptr    = host_ptr;
        result.size          = request.dst_size;
        result.status        = cache_layout_status::READY;
        result.host_resident = true;
        result.location      = host_loc;
        result.onednn_pack_m = request.onednn_pack_m;
        result.event         = sycl::event{};
        return true;
    };

    void * device_ptr = nullptr;
    bool   needs_fill = false;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        process_deferred_frees();

        auto it = entries_.find(key);
        if (it != entries_.end()) {
            auto id_it = id_to_key_.find(request.key);
            if (id_it == id_to_key_.end()) {
                if (g_ggml_sycl_debug >= 2) {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] id_to_key pre-insert (existing): size=%zu buckets=%zu load=%.3f\n",
                                    id_to_key_.size(), id_to_key_.bucket_count(), id_to_key_.load_factor());
                }
                if (id_to_key_.bucket_count() == 0) {
                    id_to_key_.rehash(1);
                }
                id_to_key_.emplace(request.key, key);
            } else if (!(id_it->second == key)) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision in ensure_cached_layout model=%llu name_hash=0x%llx\n",
                               (unsigned long long) request.key.model_id,
                               (unsigned long long) request.key.name_hash);
                if (cache_assert_enabled()) {
                    GGML_ABORT("unified_cache id_to_key mismatch");
                }
            }
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
            const bool layout_mismatch = entry.layout != request.layout || onednn_pack_m_mismatch(entry, request);
            if (layout_mismatch) {
                if (entry.pinned || entry.state == cache_entry_state::IN_PROGRESS) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] layout switch blocked (pinned/in-progress) model=%llu name_hash=0x%llx have=%d want=%d\n",
                        (unsigned long long) request.key.model_id,
                        (unsigned long long) request.key.name_hash,
                        (int) entry.layout,
                        (int) request.layout);
                    result.status = cache_layout_status::FAILED;
                    return result;
                }
                void * stale_ptr      = entry.device_ptr;
                size_t stale_size     = entry.size;
                bool   stale_host_res = entry.host_resident;
                entries_.erase(it);
                it = entries_.end();
                if (!stale_host_res && stale_ptr && stale_size > 0) {
                    enqueue_deferred_free(stale_ptr, stale_size);
                }
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
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout pending: model=%llu name_hash=0x%llx layout=%d size=%zu has_event=%d\n",
                    (unsigned long long) request.key.model_id,
                    (unsigned long long) request.key.name_hash,
                    (int) request.layout,
                    entry.size,
                    entry.has_ready_event ? 1 : 0);
                result.device_ptr                      = entry.device_ptr;
                result.size                            = entry.size;
                result.status                          = cache_layout_status::IN_PROGRESS;
                result.host_resident                   = entry.host_resident;
                result.location                        = entry.location;
                result.onednn_pack_m                   = entry.onednn_pack_m;
                std::vector<sycl::event> combined_deps = deps;
                if (entry.has_ready_event) {
                    combined_deps.push_back(entry.ready_event);
                }
                result.event = submit_barrier(combined_deps);
                return result;
            }

            // Handle previously failed entries - clean up and retry allocation
            if (entry.state == cache_entry_state::FAILED) {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] Cleaning up failed entry: model=%llu name_hash=0x%llx layout=%d, will retry\n",
                    (unsigned long long) request.key.model_id,
                    (unsigned long long) request.key.name_hash,
                    (int) request.layout);
                // Memory was already freed and used_ decremented in the exception handler.
                // If device_ptr is still set (shouldn't happen), try to free it.
                if (entry.device_ptr) {
                    try {
                        sycl::free(entry.device_ptr, queue_);
                        used_ -= entry.size;
                    } catch (...) {
                        // Ignore - may leak memory but avoid crash
                    }
                }
                entries_.erase(it);
                it = entries_.end();  // Force fall-through to allocation path below
                // NOTE: 'entry' is now a dangling reference, do not use it after this point
            }
        }

        // Process existing valid entry (not IN_PROGRESS or FAILED)
        if (it != entries_.end()) {
            auto & entry = it->second;

            if (entry.size != request.dst_size) {
                if (entry.pinned) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] layout size mismatch: model=%llu name_hash=0x%llx layout=%d cached=%zu req=%zu (pinned)\n",
                        (unsigned long long) request.key.model_id,
                        (unsigned long long) request.key.name_hash,
                        (int) request.layout,
                        entry.size,
                        request.dst_size);
                    result.status = cache_layout_status::FAILED;
                    return result;
                }

                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout size mismatch: model=%llu name_hash=0x%llx layout=%d cached=%zu req=%zu, evicting\n",
                    (unsigned long long) request.key.model_id,
                    (unsigned long long) request.key.name_hash,
                    (int) request.layout,
                    entry.size,
                    request.dst_size);
                void * stale_ptr       = entry.device_ptr;
                size_t stale_size      = entry.size;
                bool   stale_host_res  = entry.host_resident;
                entries_.erase(it);
                it = entries_.end();
                if (!stale_host_res && stale_ptr && stale_size > 0) {
                    enqueue_deferred_free(stale_ptr, stale_size);
                }
            }
        }

        if (it != entries_.end()) {
            auto & entry = it->second;
            bool   content_changed = (entry.src_ptr != request.src_ptr);
            if (request.validate_content && can_hash) {
                content_changed = (entry.content_hash != new_hash) || content_changed;
            }

            if (cache_assert_enabled()) {
                GGML_ASSERT(entry.device_ptr != nullptr);
                GGML_ASSERT(entry.size == request.dst_size);
            }
            if (g_ggml_sycl_debug >= 2 && entry.src_ptr != request.src_ptr) {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout src change model=%llu name_hash=0x%llx layout=%d cached_src=%p new_src=%p size=%zu\n",
                    (unsigned long long) request.key.model_id,
                    (unsigned long long) request.key.name_hash,
                    (int) request.layout,
                    entry.src_ptr,
                    request.src_ptr,
                    entry.size);
            }
            if (!content_changed) {
                result.device_ptr = entry.device_ptr;
                result.size       = entry.size;
                result.status     = cache_layout_status::READY;
                result.host_resident = entry.host_resident;
                result.location      = entry.location;
                result.onednn_pack_m = entry.onednn_pack_m;
                result.event      = submit_barrier(deps);
                return result;
            }

            entry.src_ptr         = request.src_ptr;
            entry.content_hash    = can_hash ? new_hash : 0;
            entry.state           = cache_entry_state::IN_PROGRESS;
            entry.has_ready_event = false;
            entry.xmx_info        = request.xmx_info;
            entry.onednn_pack_m   = request.onednn_pack_m;
            device_ptr            = entry.device_ptr;
            needs_fill            = true;
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] layout refresh: model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                (unsigned long long) request.key.model_id,
                (unsigned long long) request.key.name_hash,
                (int) request.layout,
                entry.size);
        }

        // Allocate new entry if not found or was cleaned up due to FAILED state
        if (it == entries_.end()) {
            // Allocate new entry
            const size_t base_budget    = budget_;
            const size_t allowed_budget = base_budget;  // No overcommit; keep DMA headroom intact.

            while (used_.load() + request.dst_size > base_budget) {
                if (evict_one(request.dst_size) == 0) {
                    break;
                }
            }

            bool force_host = false;
            if (used_.load() + request.dst_size > allowed_budget) {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict for layout (used=%.1f MB, need=%.1f MB)\n",
                              used_.load() / (1024.0f * 1024.0f), request.dst_size / (1024.0f * 1024.0f));
                force_host = true;
            }
            host_cache * hcache = get_host_cache(queue_);
            if (request.prefer_host && hcache) {
                force_host = true;
            }
            if (!force_host && hcache) {
                size_t free_mem = 0;
                size_t total_mem = 0;
                try {
                    const int device_id = get_device_id_from_queue(queue_);
                    ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
                } catch (...) {
                    free_mem  = 0;
                    total_mem = 0;
                }
                if (total_mem > 0) {
                    const size_t min_headroom = 256ull * 1024ull * 1024ull;
                    const size_t headroom     = std::max(min_headroom, total_mem / 10);
                    const size_t usable_free  = free_mem > headroom ? free_mem - headroom : 0;
                    if (request.dst_size > usable_free) {
                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] live VRAM low (free=%.1f MB, headroom=%.1f MB, need=%.1f MB) - using host\n",
                            free_mem / (1024.0f * 1024.0f),
                            headroom / (1024.0f * 1024.0f),
                            request.dst_size / (1024.0f * 1024.0f));
                        force_host = true;
                    }
                }
            }

            void *         new_device_ptr   = nullptr;
            bool           is_host_resident = false;
            cache_location host_location    = cache_location::HOST_MMAP;
            if (!force_host) {
                try {
                    new_device_ptr = ggml_sycl_malloc_device(request.dst_size, queue_, "unified_cache:layout");
                } catch (const sycl::exception & e) {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] layout malloc_device failed: %s, trying host fallback\n", e.what());
                    new_device_ptr = nullptr;
                }
            }

            if (!new_device_ptr) {
                // Try host_cache fallback when device allocation fails
                if (!hcache) {
                    hcache = get_host_cache(queue_);
                }
                if (hcache) {
                    cache_location host_loc = hcache->get_location(request.key, request.type, request.layer_id,
                                                                   request.expert_id, request.layout);
                    void *         host_ptr = hcache->get(request.key, request.type, request.layer_id,
                                                          request.expert_id, request.layout);
                    if (!host_ptr) {
                        // Try to create in host cache
                        bool           needs_host_fill = false;
                        bool           pinned_alloc    = false;
                        host_ptr = hcache->ensure_cached_alloc(request.key, request.src_ptr, request.src_size,
                                                               request.dst_size, request.type, request.layer_id,
                                                               request.expert_id, request.layout, false,
                                                               &needs_host_fill, &pinned_alloc, &host_loc,
                                                               &request.xmx_info);

                        if (host_ptr && needs_host_fill) {
                            if (request.fill_fn) {
                                // Fill host buffer synchronously (no device queue for host memory)
                                request.fill_fn(queue_, host_ptr, request.dst_size, request.src_ptr, request.src_size,
                                                request.fill_ctx, {});
                            } else {
                                std::memcpy(host_ptr, request.src_ptr, std::min(request.dst_size, request.src_size));
                                if (request.dst_size > request.src_size) {
                                    std::memset(static_cast<char *>(host_ptr) + request.src_size, 0,
                                                request.dst_size - request.src_size);
                                }
                            }
                        }
                    }

                    if (host_ptr) {
                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] Device full, using host-resident pointer model=%llu name_hash=0x%llx layout=%d\n",
                            (unsigned long long) request.key.model_id,
                            (unsigned long long) request.key.name_hash,
                            (int) request.layout);
                        new_device_ptr   = host_ptr;
                        is_host_resident = true;
                        host_location    = host_loc;
                    }
                }

                if (!new_device_ptr) {
                    // Only fail if no host fallback available
                    if (force_host) {
                        GGML_LOG_ERROR("[UNIFIED-CACHE] layout fallback failed (budget exhausted)\n");
                    } else {
                        GGML_LOG_ERROR("[UNIFIED-CACHE] layout allocation failed, no host fallback available\n");
                    }
                    result.status = cache_layout_status::FAILED;
                    return result;
                }
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
            entry.onednn_pack_m   = request.onednn_pack_m;
            entry.xmx_info        = request.xmx_info;
            entry.access_count    = 1;
            entry.last_access     = time_++;
            entry.pinned          = false;
            entry.hot             = false;
            entry.state           = is_host_resident ? cache_entry_state::READY : cache_entry_state::IN_PROGRESS;
            entry.has_ready_event = false;
            entry.host_resident   = is_host_resident;
            entry.location        = is_host_resident ? host_location : cache_location::DEVICE;

            if (g_ggml_sycl_debug >= 2) {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout insert model=%llu name_hash=0x%llx layout=%d size=%zu entries=%zu buckets=%zu\n",
                    (unsigned long long) request.key.model_id,
                    (unsigned long long) request.key.name_hash,
                    (int) request.layout,
                    request.dst_size,
                    entries_.size(),
                    entries_.bucket_count());
            }
            entries_[key] = entry;
            auto id_it = id_to_key_.find(request.key);
            if (id_it == id_to_key_.end()) {
                if (g_ggml_sycl_debug >= 2) {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] id_to_key pre-insert (new): size=%zu buckets=%zu load=%.3f\n",
                                    id_to_key_.size(), id_to_key_.bucket_count(), id_to_key_.load_factor());
                }
                if (id_to_key_.bucket_count() == 0) {
                    id_to_key_.rehash(1);
                }
                id_to_key_.emplace(request.key, key);
            } else if (!(id_it->second == key)) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision on layout insert model=%llu name_hash=0x%llx\n",
                               (unsigned long long) request.key.model_id,
                               (unsigned long long) request.key.name_hash);
                if (cache_assert_enabled()) {
                    GGML_ABORT("unified_cache id_to_key mismatch");
                }
            }
            if (!is_host_resident) {
                // Only count device memory against unified cache budget
                used_ += request.dst_size;
            }
            device_ptr = new_device_ptr;
            needs_fill = !is_host_resident;  // Host-resident already filled above
            if (is_host_resident) {
                // Return immediately for host-resident entries (already filled)
                result.device_ptr    = device_ptr;
                result.size          = request.dst_size;
                result.status        = cache_layout_status::READY;
                result.host_resident = true;
                result.location      = host_location;
                result.onednn_pack_m = request.onednn_pack_m;
                result.event         = sycl::event{};
                return result;
            }
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] layout allocate: model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                (unsigned long long) request.key.model_id,
                (unsigned long long) request.key.name_hash,
                (int) request.layout,
                request.dst_size);
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
        GGML_SYCL_DEBUG(
            "[DEBUG-FILL] Starting fill: device_ptr=%p src_ptr=%p dst_size=%zu src_size=%zu layout=%d fill_fn=%s\n",
            device_ptr, request.src_ptr, request.dst_size, request.src_size, (int) request.layout,
            request.fill_fn ? "yes" : "no");
        if (copy_trace_enabled()) {
            GGML_LOG_INFO(
                "[SYCL] layout fill begin: model=%llu name_hash=0x%llx layout=%d dst=%p dst_size=%zu src=%p src_size=%zu fill_fn=%s\n",
                (unsigned long long) request.key.model_id,
                (unsigned long long) request.key.name_hash,
                (int) request.layout,
                device_ptr,
                request.dst_size,
                request.src_ptr,
                request.src_size,
                request.fill_fn ? "yes" : "no");
            fflush(stderr);
        }

        if (request.fill_fn) {
            GGML_SYCL_DEBUG("[DEBUG-FILL] Calling fill_fn...\n");
            fill_event = request.fill_fn(queue_, device_ptr, request.dst_size, request.src_ptr, request.src_size,
                                         request.fill_ctx, deps);
            GGML_SYCL_DEBUG("[DEBUG-FILL] fill_fn returned\n");
        } else {
            GGML_SYCL_DEBUG("[DEBUG-FILL] Calling copy_to_device_async...\n");
            fill_event = copy_to_device_async(device_ptr, request.src_ptr, request.src_size, deps);
            GGML_SYCL_DEBUG("[DEBUG-FILL] copy_to_device_async returned\n");
        }

        // Wait for fill to complete before any padding memset
        GGML_SYCL_DEBUG("[DEBUG-FILL] About to wait on fill_event...\n");
        fill_event.wait();
        GGML_SYCL_DEBUG("[DEBUG-FILL] fill_event.wait() completed\n");

        if (request.layout != GGML_LAYOUT_XMX_TILED && request.layout != GGML_LAYOUT_XMX_GEMM_TILED &&
            request.layout != GGML_LAYOUT_ONEDNN_PACKED && request.layout != GGML_LAYOUT_ONEDNN_WOQ &&
            request.dst_size > request.src_size) {
            const size_t pad_bytes = request.dst_size - request.src_size;
            void *       pad_ptr   = static_cast<char *>(device_ptr) + request.src_size;
            GGML_SYCL_DEBUG("[DEBUG-FILL] About to memset padding: pad_ptr=%p pad_bytes=%zu\n", pad_ptr, pad_bytes);
            // Do padding synchronously - Level Zero has issues with event chains
            queue_.memset(pad_ptr, 0, pad_bytes).wait();
            GGML_SYCL_DEBUG("[DEBUG-FILL] Padding memset completed\n");
        }
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout fill failed (sycl): %s\n", e.what());
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout fill context model=%llu name_hash=0x%llx type=%d layer=%d expert=%d layout=%d src=%p "
                       "src_size=%zu dst_size=%zu\n",
                       (unsigned long long) request.key.model_id,
                       (unsigned long long) request.key.name_hash,
                       static_cast<int>(request.type),
                       request.layer_id,
                       request.expert_id,
                       static_cast<int>(request.layout),
                       request.src_ptr,
                       request.src_size,
                       request.dst_size);
        if (try_host_fallback("sycl_fill")) {
            return result;
        }
        if (const char * msg = e.what()) {
            if (std::strstr(msg, "DEVICE_LOST") || std::strstr(msg, "device lost")) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] Device lost during cache fill - aborting to preserve backtrace.\n");
                std::abort();
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = entries_.find(key);
        if (it != entries_.end()) {
            // Mark entry as FAILED instead of deleting it immediately.
            // This allows waiting threads to see the failure and fall back gracefully.
            it->second.state           = cache_entry_state::FAILED;
            it->second.has_ready_event = false;
            // Try to free the device memory directly rather than deferring.
            // The queue may be in a bad state, so wrap in try-catch.
            if (it->second.device_ptr) {
                try {
                    // Try to synchronize before freeing to avoid use-after-free
                    queue_.wait();
                } catch (...) {
                    // Queue in bad state, ignore
                }
                try {
                    sycl::free(it->second.device_ptr, queue_);
                    used_ -= it->second.size;
                } catch (...) {
                    // Free failed - memory may leak, but avoid crash
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to free device memory during error recovery\n");
                }
                it->second.device_ptr = nullptr;
                it->second.size       = 0;
            }
        }
        entry_cv_.notify_all();
        result.status     = cache_layout_status::FAILED;
        result.device_ptr = nullptr;
        return result;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout fill failed (std): %s\n", e.what());
        if (try_host_fallback("std_fill")) {
            return result;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.state           = cache_entry_state::FAILED;
            it->second.has_ready_event = false;
            if (it->second.device_ptr) {
                try {
                    queue_.wait();
                } catch (...) {
                }
                try {
                    sycl::free(it->second.device_ptr, queue_);
                    used_ -= it->second.size;
                } catch (...) {
                }
                it->second.device_ptr = nullptr;
                it->second.size       = 0;
            }
        }
        entry_cv_.notify_all();
        result.status     = cache_layout_status::FAILED;
        result.device_ptr = nullptr;
        return result;
    } catch (...) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout fill failed (unknown exception)\n");
        if (try_host_fallback("unknown_fill")) {
            return result;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.state           = cache_entry_state::FAILED;
            it->second.has_ready_event = false;
            if (it->second.device_ptr) {
                try {
                    queue_.wait();
                } catch (...) {
                }
                try {
                    sycl::free(it->second.device_ptr, queue_);
                    used_ -= it->second.size;
                } catch (...) {
                }
                it->second.device_ptr = nullptr;
                it->second.size       = 0;
            }
        }
        entry_cv_.notify_all();
        result.status     = cache_layout_status::FAILED;
        result.device_ptr = nullptr;
        return result;
    }

    // All operations completed synchronously - mark as READY immediately.
    // We avoid returning events because Level Zero driver has issues with event handling
    // that can cause crashes when events are waited on multiple times or in certain patterns.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.has_ready_event = false;
            it->second.state           = cache_entry_state::READY;
        }
    }
    entry_cv_.notify_all();

    result.device_ptr = device_ptr;
    result.size       = request.dst_size;
    result.status     = cache_layout_status::READY;
    result.host_resident = false;
    result.location      = cache_location::DEVICE;
    // Don't set result.event - no need to wait since everything is done synchronously
    return result;
}

bool unified_cache::is_cached(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) const {
    if (!key_id.valid) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return false;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return false;
    }
    if (entry_it->second.layout != layout) {
        return false;
    }
    return true;
}

bool unified_cache::is_cached_any(const ggml_sycl_cache_id & key_id) const {
    if (!key_id.valid) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return false;
    }
    return entries_.find(id_it->second) != entries_.end();
}

void * unified_cache::get(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }
    if (entry.state == cache_entry_state::IN_PROGRESS) {
        if (entry.has_ready_event && event_complete(entry.ready_event)) {
            entry.state           = cache_entry_state::READY;
            entry.has_ready_event = false;
        } else {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] get pending: model=%llu name_hash=0x%llx layout=%d size=%zu has_event=%d\n",
                (unsigned long long) key_id.model_id,
                (unsigned long long) key_id.name_hash,
                (int) layout,
                entry.size,
                entry.has_ready_event ? 1 : 0);
            return nullptr;
        }
    }
    return entry.device_ptr;
}

void * unified_cache::get_or_wait(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return nullptr;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }

    // Wait for IN_PROGRESS entries to complete (prevents mmap fallback)
    while (entry_it->second.state == cache_entry_state::IN_PROGRESS) {
        auto & e = entry_it->second;
        if (e.has_ready_event) {
            // Wait on the ready event (release lock during wait)
            sycl::event evt = e.ready_event;
            lock.unlock();
            try {
                evt.wait();
            } catch (const sycl::exception & ex) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] get_or_wait event wait failed: %s\n", ex.what());
                return nullptr;
            }
            lock.lock();

            // Re-lookup after releasing lock
            id_it = id_to_key_.find(key_id);
            if (id_it == id_to_key_.end()) {
                return nullptr;
            }
            entry_it = entries_.find(id_it->second);
            if (entry_it == entries_.end()) {
                return nullptr;
            }
            if (entry_it->second.layout != layout) {
                return nullptr;
            }

            // Update state if event completed
            if (entry_it->second.state == cache_entry_state::IN_PROGRESS &&
                entry_it->second.has_ready_event &&
                event_complete(entry_it->second.ready_event)) {
                entry_it->second.state           = cache_entry_state::READY;
                entry_it->second.has_ready_event = false;
            }
        } else {
            // No event yet - spin wait briefly then check again
            // This handles the case where entry is created but event not yet assigned
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            lock.lock();

            // Re-lookup
            id_it = id_to_key_.find(key_id);
            if (id_it == id_to_key_.end()) {
                return nullptr;
            }
            entry_it = entries_.find(id_it->second);
            if (entry_it == entries_.end()) {
                return nullptr;
            }
        }
    }

    if (entry_it->second.state == cache_entry_state::FAILED) {
        return nullptr;
    }

    return entry_it->second.device_ptr;
}

void * unified_cache::get_by_data_ptr(void * data_ptr, size_t nbytes, ggml_layout_mode layout) {
    if (!data_ptr || nbytes == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Search all entries for one that matches by source pointer and size.
    // This is O(N) but only used as a fallback during graph recording when
    // the primary name-based lookup fails due to tensor name aliasing.
    for (const auto & [key, entry] : entries_) {
        if (entry.state != cache_entry_state::READY) {
            continue;
        }
        if (entry.layout != layout) {
            continue;
        }
        if (entry.size != nbytes) {
            continue;
        }
        if (entry.src_ptr == data_ptr) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] get_by_data_ptr: found alias data=%p size=%zu -> device=%p\n",
                           data_ptr, nbytes, entry.device_ptr);
            return entry.device_ptr;
        }
    }

    return nullptr;
}

cache_ptr_view unified_cache::get_view(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    cache_ptr_view view{};
    if (!key_id.valid) {
        return view;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return view;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return view;
    }
    auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return view;
    }
    if (entry.state == cache_entry_state::IN_PROGRESS) {
        if (entry.has_ready_event && event_complete(entry.ready_event)) {
            entry.state           = cache_entry_state::READY;
            entry.has_ready_event = false;
        } else {
            return view;
        }
    }
    view.ptr      = entry.device_ptr;
    view.size     = entry.size;
    view.layout   = entry.layout;
    view.onednn_pack_m = entry.onednn_pack_m;
    view.location = entry.location;
    view.type     = entry.type;
    view.layer_id = entry.layer_id;
    view.expert_id = entry.expert_id;
    view.xmx_info = entry.xmx_info;
    return view;
}

void unified_cache::remove(const ggml_sycl_cache_id & key_id,
                           cache_entry_type          type,
                           int                       layer_id,
                           int                       expert_id,
                           ggml_layout_mode          layout) {
    if (!key_id.valid) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    process_deferred_frees();
    unified_cache_key key{ type, key_id, layer_id, expert_id };

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }
    if (it->second.layout != layout) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in remove model=%llu name_hash=0x%llx have=%d want=%d\n",
                       (unsigned long long) key_id.model_id,
                       (unsigned long long) key_id.name_hash,
                       (int) it->second.layout,
                       (int) layout);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache layout mismatch");
        }
        return;
    }
    if (it->second.state == cache_entry_state::IN_PROGRESS) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] remove skipped: entry in progress model=%llu name_hash=0x%llx\n",
                        (unsigned long long) key_id.model_id,
                        (unsigned long long) key_id.name_hash);
        return;
    }

    enqueue_deferred_free(it->second.device_ptr, it->second.size);

    entries_.erase(it);
    id_to_key_.erase(key_id);
}

// NOTE: mark_reordered/is_reordered removed - reorder state tracked in tensor->extra->optimized_feature

void unified_cache::pin(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it != entries_.end()) {
        if (entry_it->second.layout != layout) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in pin model=%llu name_hash=0x%llx have=%d want=%d\n",
                           (unsigned long long) key_id.model_id,
                           (unsigned long long) key_id.name_hash,
                           (int) entry_it->second.layout,
                           (int) layout);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return;
        }
        entry_it->second.pinned = true;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) key_id.model_id,
                        (unsigned long long) key_id.name_hash,
                        (int) layout);
    }
}

void unified_cache::unpin(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it != entries_.end()) {
        if (entry_it->second.layout != layout) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in unpin model=%llu name_hash=0x%llx have=%d want=%d\n",
                           (unsigned long long) key_id.model_id,
                           (unsigned long long) key_id.name_hash,
                           (int) entry_it->second.layout,
                           (int) layout);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return;
        }
        entry_it->second.pinned = false;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) key_id.model_id,
                        (unsigned long long) key_id.name_hash,
                        (int) layout);
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

bool unified_cache::is_pinned(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) const {
    if (!key_id.valid) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return false;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return false;
    }
    if (entry_it->second.layout != layout) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in is_pinned model=%llu name_hash=0x%llx have=%d want=%d\n",
                       (unsigned long long) key_id.model_id,
                       (unsigned long long) key_id.name_hash,
                       (int) entry_it->second.layout,
                       (int) layout);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache layout mismatch");
        }
        return false;
    }
    return entry_it->second.pinned;
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
    // Tiered eviction priority (lower = evict first):
    // -1: host-resident (already slow, evict first to reclaim tracking)
    //  0: MoE experts (cold), 1: MoE experts (hot), 2: dense (cold), 3: dense (hot)
    if (entry.host_resident) {
        return -1;  // Host-resident entries evict first (they're already slow)
    }
    const int base = (entry.type == cache_entry_type::DENSE_WEIGHT) ? 2 : 0;
    return base + (entry.hot ? 1 : 0);
}

size_t unified_cache::evict_one(size_t /* new_size */) {
    process_deferred_frees();

    unified_cache_key evict_key{};
    int               best_tier        = std::numeric_limits<int>::max();
    int64_t           best_last_access = std::numeric_limits<int64_t>::max();
    bool              found            = false;

    for (auto & pair : entries_) {
        auto & entry = pair.second;
        if (entry.state == cache_entry_state::IN_PROGRESS) {
            if (entry.has_ready_event && event_complete(entry.ready_event)) {
                entry.state           = cache_entry_state::READY;
                entry.has_ready_event = false;
            } else {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict skip: model=%llu name_hash=0x%llx layout=%d in-progress size=%zu\n",
                                (unsigned long long) pair.first.id.model_id,
                                (unsigned long long) pair.first.id.name_hash,
                                (int) entry.layout,
                                entry.size);
                continue;
            }
        }
        if (entry.pinned) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict skip: model=%llu name_hash=0x%llx layout=%d pinned size=%zu\n",
                            (unsigned long long) pair.first.id.model_id,
                            (unsigned long long) pair.first.id.name_hash,
                            (int) entry.layout,
                            entry.size);
            continue;
        }

        const int tier = eviction_tier(entry);
        if (tier < best_tier || (tier == best_tier && entry.last_access < best_last_access)) {
            best_tier        = tier;
            best_last_access = entry.last_access;
            evict_key        = pair.first;
            found            = true;
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
        size_t entry_size    = it->second.size;
        void * ptr           = it->second.device_ptr;
        bool   host_resident = it->second.host_resident;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict model=%llu name_hash=0x%llx layout=%d size=%zu host_resident=%d\n",
                        (unsigned long long) evict_key.id.model_id,
                        (unsigned long long) evict_key.id.name_hash,
                        (int) it->second.layout,
                        entry_size,
                        host_resident ? 1 : 0);

        if (!host_resident) {
            // Only free device memory; host-resident entries are managed by host_cache
            enqueue_deferred_free(ptr, entry_size);
        }
        // Note: For host-resident entries, we just remove tracking here.
        // The host_cache still owns the memory and will evict it via its own LRU policy.

        // Remove from lookup
        id_to_key_.erase(evict_key.id);

        // Remove from entries
        entries_.erase(it);

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Evicted: model=%llu name_hash=0x%llx layout=%d %.2f MB (used=%.1f/%.1f MB) host_resident=%d\n",
                        (unsigned long long) evict_key.id.model_id,
                        (unsigned long long) evict_key.id.name_hash,
                        (int) it->second.layout,
                        entry_size / (1024.0f * 1024.0f),
                        used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f), host_resident ? 1 : 0);
        evicted_bytes = host_resident ? 0 : entry_size;  // Only count device bytes freed
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
        void * temp = ggml_sycl_malloc_host(size, queue_, "unified_cache:host_temp");
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
    const sycl::usm::alloc dst_type = sycl::get_pointer_type(dst, queue_.get_context());
    if (g_ggml_sycl_debug >= 2 || copy_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] copy_to_device_async ptr types: dst=%p type=%d src=%p type=%d size=%zu\n",
                      dst, (int) dst_type, src, (int) src_type, size);
        if (copy_trace_enabled()) {
            fflush(stderr);
        }
    }
    if (dst_type == sycl::usm::alloc::unknown && cache_assert_enabled()) {
        GGML_ABORT("copy_to_device_async called with non-USM destination");
    }
    // Stage any non-device source memory through host buffer.
    // This handles:
    // - unknown: mmap'd or non-USM pointers
    // - shared: can fail on Level Zero if allocated on different context
    // - host: generally works, but staging is safer
    // Only device-to-device copies skip staging.
    const bool needs_staging = (src_type != sycl::usm::alloc::device);
    if (needs_staging) {
        // Stage through host memory.
        // NOTE: We perform synchronous staging to avoid complex event dependency chains
        // that can cause issues with Level Zero driver's event handling.
        const bool avoid_wait = ggml_sycl_graph_recording_active() || ggml_sycl_graph_inflight_count() > 0;
        if (avoid_wait) {
            // Allocate per-copy staging to avoid blocking waits in graph/inflight mode.
            constexpr size_t k_fallback_chunk = 64 * 1024 * 1024;
            const size_t     chunk_size       = std::min(size, staging_size_ > 0 ? staging_size_ : k_fallback_chunk);
            const char *     src_ptr          = static_cast<const char *>(src);
            char *           dst_ptr          = static_cast<char *>(dst);
            size_t           remaining        = size;
            sycl::event      last;
            std::vector<sycl::event> chain = deps;

            while (remaining > 0) {
                const size_t chunk = std::min(remaining, chunk_size);
                void *       temp  = ggml_sycl_malloc_host(chunk, queue_, "unified_cache:host_chunk");
                if (!temp) {
                    throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                          "Cannot copy non-USM pointer to device: staging allocation failed");
                }
                std::memcpy(temp, src_ptr, chunk);
                sycl::event ev;
                if (chain.empty()) {
                    ev = queue_.memcpy(dst_ptr, temp, chunk);
                } else {
                    ev = queue_.submit([&](sycl::handler & cgh) {
                        cgh.depends_on(chain);
                        cgh.memcpy(dst_ptr, temp, chunk);
                    });
                }
                enqueue_deferred_host_free(temp, 0, ev);
                last = ev;
                chain.clear();
                chain.push_back(ev);
                src_ptr += chunk;
                dst_ptr += chunk;
                remaining -= chunk;
            }
            return last;
        }

        // Wait for any dependencies first.
        // NOTE: Avoid ext_oneapi_submit_barrier which can corrupt Level Zero event state.
        // Instead, wait on each dep individually.
        for (const auto & dep : deps) {
            // sycl::event::wait() is non-const, so use const_cast
            // Let exceptions propagate - if device is lost, we must abort
            const_cast<sycl::event &>(dep).wait();
        }

        if (staging_) {
            std::lock_guard<std::mutex> lock(staging_mutex_);
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

        // No pre-allocated staging available; fall back to chunked temp allocations.
        constexpr size_t k_fallback_chunk = 64 * 1024 * 1024;
        const size_t     chunk_size       = std::min(size, k_fallback_chunk);
        void *           temp             = ggml_sycl_malloc_host(chunk_size, queue_, "unified_cache:host_chunk");
        if (!temp) {
            throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                  "Cannot copy non-USM pointer to device: staging allocation failed");
        }

        const char * src_ptr   = static_cast<const char *>(src);
        char *       dst_ptr   = static_cast<char *>(dst);
        size_t       remaining = size;
        sycl::event  last;

        while (remaining > 0) {
            const size_t chunk = std::min(remaining, chunk_size);
            std::memcpy(temp, src_ptr, chunk);
            last = queue_.memcpy(dst_ptr, temp, chunk);
            last.wait();
            src_ptr += chunk;
            dst_ptr += chunk;
            remaining -= chunk;
        }

        sycl::free(temp, queue_);
        return last;
    }

    if (deps.empty()) {
        auto ev = queue_.memcpy(dst, src, size);
        ev.wait();  // DEBUG: Make synchronous to rule out async issues
        return ev;
    }
    auto ev = queue_.submit([&](sycl::handler & cgh) {
        cgh.depends_on(deps);
        cgh.memcpy(dst, src, size);
    });
    return ev;
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

void unified_cache::enqueue_deferred_host_free(void * ptr, size_t size, const sycl::event & event) {
    if (!ptr) {
        return;
    }
    deferred_host_free_entry entry{};
    entry.ptr       = ptr;
    entry.size      = size;
    entry.has_event = true;
    entry.event     = event;
    deferred_host_frees_.push_back(entry);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred host free: ptr=%p\n", ptr);
}

void unified_cache::defer_host_free(void * ptr, size_t size, const sycl::event & event) {
    enqueue_deferred_host_free(ptr, size, event);
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

    auto host_it = deferred_host_frees_.begin();
    while (host_it != deferred_host_frees_.end()) {
        const bool ready = !host_it->has_event || event_complete(host_it->event);
        if (!ready) {
            ++host_it;
            continue;
        }
        if (host_it->ptr) {
            try {
                if (host_it->has_event) {
                    host_it->event.wait_and_throw();
                }
            } catch (...) {
            }
            try {
                sycl::free(host_it->ptr, queue_);
            } catch (...) {
            }
            if (host_it->size > 0) {
                ggml_sycl::unified_cache_sub_runtime_host_bytes(host_it->size);
            }
        }
        host_it = deferred_host_frees_.erase(host_it);
    }

    auto pin_it = inflight_unpins_.begin();
    while (pin_it != inflight_unpins_.end()) {
        const bool ready = !pin_it->has_event || event_complete(pin_it->event);
        if (g_ggml_sycl_debug) {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] unpin check model=%llu name_hash=0x%llx layout=%d has_event=%d ready=%d\n",
                (unsigned long long) pin_it->key.model_id,
                (unsigned long long) pin_it->key.name_hash,
                (int) pin_it->layout,
                pin_it->has_event ? 1 : 0,
                ready ? 1 : 0);
        }
        if (!ready) {
            ++pin_it;
            continue;
        }
        if (pin_it->has_event) {
            try {
                pin_it->event.wait_and_throw();
            } catch (...) {
                // Best-effort cleanup; event_complete already said ready.
            }
        }
        auto id_it = id_to_key_.find(pin_it->key);
        if (id_it != id_to_key_.end()) {
            auto entry_it = entries_.find(id_it->second);
            if (entry_it != entries_.end()) {
                if (entry_it->second.layout != pin_it->layout) {
                    GGML_LOG_ERROR(
                        "[UNIFIED-CACHE] layout mismatch in inflight unpin model=%llu name_hash=0x%llx have=%d want=%d\n",
                        (unsigned long long) pin_it->key.model_id,
                        (unsigned long long) pin_it->key.name_hash,
                        (int) entry_it->second.layout,
                        (int) pin_it->layout);
                    if (cache_assert_enabled()) {
                        GGML_ABORT("unified_cache layout mismatch");
                    }
                } else {
                    entry_it->second.pinned = false;
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] in-flight unpin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) pin_it->key.model_id,
                        (unsigned long long) pin_it->key.name_hash,
                        (int) pin_it->layout);
                }
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
    constexpr int layout_count = GGML_LAYOUT_ONEDNN_WOQ + 1;
    size_t layout_counts[layout_count] = {};
    size_t layout_bytes[layout_count]  = {};
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::DENSE_WEIGHT) {
            dense++;
            dense_bytes += pair.second.size;
        } else {
            experts++;
            expert_bytes += pair.second.size;
        }
        const int layout_idx = static_cast<int>(pair.second.layout);
        if (layout_idx >= 0 && layout_idx < layout_count) {
            layout_counts[layout_idx]++;
            layout_bytes[layout_idx] += pair.second.size;
        }
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Stats: %zu hits, %zu misses (%.1f%% hit rate)\n", hits_.load(), misses_.load(),
                  rate);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Entries: %zu dense (%.1f MB), %zu experts (%.1f MB), total %.1f/%.1f MB\n", dense,
                  dense_bytes / (1024.0f * 1024.0f), experts, expert_bytes / (1024.0f * 1024.0f),
                  used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f));
    GGML_LOG_INFO(
        "[UNIFIED-CACHE] Layouts: aos=%zu (%.1f MB), soa=%zu (%.1f MB), coalesced=%zu (%.1f MB), xmx_tiled=%zu (%.1f "
        "MB), xmx_gemm_tiled=%zu (%.1f MB), onednn_packed=%zu (%.1f MB), onednn_woq=%zu (%.1f MB)\n",
        layout_counts[GGML_LAYOUT_AOS], layout_bytes[GGML_LAYOUT_AOS] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_SOA], layout_bytes[GGML_LAYOUT_SOA] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_COALESCED], layout_bytes[GGML_LAYOUT_COALESCED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_XMX_TILED], layout_bytes[GGML_LAYOUT_XMX_TILED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_XMX_GEMM_TILED], layout_bytes[GGML_LAYOUT_XMX_GEMM_TILED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_ONEDNN_PACKED], layout_bytes[GGML_LAYOUT_ONEDNN_PACKED] / (1024.0f * 1024.0f),
        layout_counts[GGML_LAYOUT_ONEDNN_WOQ], layout_bytes[GGML_LAYOUT_ONEDNN_WOQ] / (1024.0f * 1024.0f));
}

void unified_cache::reset_stats() {
    hits_   = 0;
    misses_ = 0;
}

bool unified_cache::validate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    bool                        ok = true;

    for (const auto & pair : entries_) {
        const auto & key   = pair.first;
        const auto & entry = pair.second;
        auto         it    = id_to_key_.find(key.id);
        if (it == id_to_key_.end() || !(it->second == key)) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] validate: id_to_key mismatch model=%llu name_hash=0x%llx\n",
                          (unsigned long long) key.id.model_id,
                          (unsigned long long) key.id.name_hash);
            ok = false;
        }
        if (!entry.device_ptr || entry.size == 0) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] validate: entry missing data model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                          (unsigned long long) key.id.model_id,
                          (unsigned long long) key.id.name_hash,
                          (int) entry.layout,
                          entry.size);
            ok = false;
        }
    }

    for (const auto & pair : id_to_key_) {
        auto it = entries_.find(pair.second);
        if (it == entries_.end()) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] validate: dangling id_to_key entry model=%llu name_hash=0x%llx\n",
                          (unsigned long long) pair.first.model_id,
                          (unsigned long long) pair.first.name_hash);
            ok = false;
        }
    }

    return ok;
}

void unified_cache::update_reserved_bytes(size_t reserved_bytes) {
    size_t effective_budget = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        reserved_ = reserved_bytes;
        if (reserved_ >= base_budget_) {
            budget_ = 0;
            GGML_LOG_INFO("[UNIFIED-CACHE] Reserve %.1f MB >= base budget %.1f MB; cache budget now 0 (used %.1f MB)\n",
                          reserved_ / (1024.0f * 1024.0f), base_budget_ / (1024.0f * 1024.0f),
                          used_.load() / (1024.0f * 1024.0f));
        } else {
            budget_ = base_budget_ - reserved_;
        }
        effective_budget = budget_;
        while (used_.load() > budget_ && !entries_.empty()) {
            if (evict_one(0) == 0) {
                break;
            }
        }
        const size_t used = used_.load();
        if (used > budget_) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cache usage (%.1f MB) exceeds budget (%.1f MB) after reserving %.1f MB\n",
                          used / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f), reserved_ / (1024.0f * 1024.0f));
        }
    }
    // Recalculate model placement decision based on new effective budget.
    // This ensures g_model_exceeds_vram reflects actual available VRAM after KV cache allocation.
    ggml_sycl_recalc_model_exceeds_vram(effective_budget);
}

void unified_cache::unpin_on_event(const ggml_sycl_cache_id & key_id,
                                   ggml_layout_mode          layout,
                                   const sycl::event &       event) {
    if (!key_id.valid) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    inflight_unpin_entry        entry{};
    entry.key       = key_id;
    entry.layout    = layout;
    entry.event     = event;
    entry.has_event = true;
    inflight_unpins_.push_back(entry);
    if (g_ggml_sycl_debug) {
        const bool ready = entry.has_event ? event_complete(entry.event) : true;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] in-flight pin model=%llu name_hash=0x%llx layout=%d ready=%d\n",
                      (unsigned long long) key_id.model_id,
                      (unsigned long long) key_id.name_hash,
                      (int) layout,
                      ready ? 1 : 0);
    }
}

bool unified_cache::get_dma_staging_buffers(size_t slice_bytes, size_t count, dma_staging_buffers & out) {
    out = {};
    if (slice_bytes == 0 || count == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(dma_staging_mutex_);
    if (!dma_staging_buffers_.empty()) {
        if (dma_buffer_count_ >= count && dma_slice_bytes_ >= slice_bytes) {
            out.buffers     = dma_staging_buffers_.data();
            out.count       = count;
            out.slice_bytes = slice_bytes;
            return true;
        }
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA staging pool mismatch: have=%zu x %.1f MB, need=%zu x %.1f MB; reallocating\n",
                      dma_buffer_count_, dma_slice_bytes_ / (1024.0 * 1024.0), count,
                      slice_bytes / (1024.0 * 1024.0));
        for (void * ptr : dma_staging_buffers_) {
            if (!ptr) {
                continue;
            }
            // Avoid blocking frees while DMA ops may still be in-flight.
            enqueue_deferred_free(ptr, dma_slice_bytes_);
        }
        dma_staging_buffers_.clear();
        dma_slice_bytes_  = 0;
        dma_buffer_count_ = 0;
    }

    const int    device_id    = get_device_id_from_queue(queue_);
    const size_t old_reserved = dma_reserved_bytes_;
    const size_t new_reserved = slice_bytes * count;
    if (new_reserved > old_reserved && device_id >= 0 && device_id < GGML_SYCL_MAX_DEVICES) {
        unified_cache_add_runtime_bytes(device_id, new_reserved - old_reserved);
        dma_reserved_bytes_ = new_reserved;
    }

    dma_staging_buffers_.resize(count, nullptr);
    size_t allocated = 0;
    for (size_t i = 0; i < count; ++i) {
        void * ptr = nullptr;
        try {
            ptr = ggml_sycl_malloc_device(slice_bytes, queue_, "unified_cache:dma_stage");
        } catch (const sycl::exception & e) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA staging malloc_device failed: %s\n", e.what());
            ptr = nullptr;
        }
        if (!ptr) {
            break;
        }
        dma_staging_buffers_[i] = ptr;
        allocated++;
    }

    if (allocated != count) {
        for (void * ptr : dma_staging_buffers_) {
            if (!ptr) {
                continue;
            }
            try {
                sycl::free(ptr, queue_);
            } catch (...) {
            }
        }
        dma_staging_buffers_.clear();
        dma_slice_bytes_  = 0;
        dma_buffer_count_ = 0;
        if (dma_reserved_bytes_ != old_reserved && device_id >= 0 && device_id < GGML_SYCL_MAX_DEVICES) {
            unified_cache_sub_runtime_bytes(device_id, dma_reserved_bytes_ - old_reserved);
            dma_reserved_bytes_ = old_reserved;
        }
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA staging pool allocation failed (need=%zu x %.1f MB)\n", count,
                      slice_bytes / (1024.0 * 1024.0));
        return false;
    }

    dma_slice_bytes_  = slice_bytes;
    dma_buffer_count_ = count;
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA staging pool allocated: %zu x %.1f MB\n", count,
                  slice_bytes / (1024.0 * 1024.0));
    out.buffers     = dma_staging_buffers_.data();
    out.count       = count;
    out.slice_bytes = dma_slice_bytes_;
    return true;
}

unified_cache::dma_stream_result unified_cache::stream_dma(const cache_ptr_view &            src,
                                                           size_t                            total_bytes,
                                                           size_t                            slice_bytes,
                                                           size_t                            buffer_count,
                                                           dma_stream_slice_fn               slice_fn,
                                                           const void *                      ctx,
                                                           const std::vector<sycl::event> &  deps,
                                                           dma_stream_copy_fn                copy_fn) {
    dma_stream_result result{};
    if (!src.ptr || !slice_fn) {
        return result;
    }

    size_t bytes = src.size;
    if (total_bytes > 0) {
        bytes = std::min(total_bytes, src.size);
    }
    if (bytes == 0) {
        return result;
    }

    resolve_dma_defaults(slice_bytes, buffer_count);
    if (slice_bytes == 0 || buffer_count == 0) {
        return result;
    }
    if (slice_bytes > bytes) {
        slice_bytes = bytes;
    }

    result.slice_bytes  = slice_bytes;
    result.buffer_count = buffer_count;
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA stream: ptr=%p bytes=%zu slice=%.1f MB buffers=%zu loc=%d type=%d\n",
                    src.ptr,
                    bytes,
                    slice_bytes / (1024.0 * 1024.0),
                    buffer_count,
                    static_cast<int>(src.location),
                    static_cast<int>(src.type));

    if (src.location == cache_location::DEVICE) {
        result.event = slice_fn(queue_, src.ptr, bytes, 0, ctx, deps);
        result.ok     = true;
        result.slices = 1;
        return result;
    }

    if (src.location == cache_location::HOST_MMAP) {
        result.used_mmap_direct = true;
        if (std::getenv("GGML_SYCL_TEST_DMA_FAIL") != nullptr) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA test override: forcing mmap DMA failure\n");
            result.mmap_direct_failed = true;
            return result;
        }
    }

    dma_staging_buffers staging{};
    if (!get_dma_staging_buffers(slice_bytes, buffer_count, staging)) {
        return result;
    }

    if (src.location == cache_location::HOST_MMAP) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA streaming from mmap pointer %p (bytes=%zu)\n", src.ptr, bytes);
    }

    std::vector<sycl::event> all_events;
    std::vector<sycl::event> buffer_events(buffer_count);
    std::vector<bool>        buffer_has_event(buffer_count, false);

    size_t offset = 0;
    size_t slices = 0;
    while (offset < bytes) {
        const size_t cur  = std::min(slice_bytes, bytes - offset);
        const size_t slot = slices % buffer_count;

        std::vector<sycl::event> copy_deps = deps;
        if (buffer_has_event[slot]) {
            copy_deps.push_back(buffer_events[slot]);
        }

        sycl::event copy_evt;
        try {
            if (copy_fn) {
                copy_evt = copy_fn(queue_, staging.buffers[slot], cur, offset, src.ptr, src.size, ctx, copy_deps);
            } else if (src.location == cache_location::HOST_MMAP) {
                // Avoid direct queue_.memcpy from mmap'd pointers (can trigger device loss).
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA stream staging mmap slice offset=%zu size=%zu\n", offset, cur);
                copy_evt = copy_to_device_async(staging.buffers[slot],
                                                static_cast<const char *>(src.ptr) + offset,
                                                cur,
                                                copy_deps);
            } else {
                copy_evt = queue_.memcpy(staging.buffers[slot],
                                         static_cast<const char *>(src.ptr) + offset,
                                         cur,
                                         copy_deps);
            }
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] DMA copy failed: %s\n", e.what());
            if (src.location == cache_location::HOST_MMAP) {
                result.mmap_direct_failed = true;
            }
            return result;
        }

        std::vector<sycl::event> kernel_deps;
        kernel_deps.push_back(copy_evt);
        sycl::event kernel_evt = slice_fn(queue_, staging.buffers[slot], cur, offset, ctx, kernel_deps);

        buffer_events[slot]     = kernel_evt;
        buffer_has_event[slot]  = true;
        all_events.push_back(kernel_evt);

        offset += cur;
        slices++;
    }

    result.slices = slices;
    if (!all_events.empty()) {
        if (queue_.has_property<sycl::property::queue::in_order>()) {
            // In-order queues already serialize submissions; avoid ext_oneapi_submit_barrier.
            result.event = all_events.back();
        } else {
            result.event = submit_barrier(all_events);
        }
    }
    result.ok     = true;
    if (result.used_mmap_direct && !result.mmap_direct_failed) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA mmap direct ok: slices=%zu bytes=%zu\n", result.slices, bytes);
    }
    return result;
}

void unified_cache::set_hot(const ggml_sycl_cache_id & key_id,
                            cache_entry_type          type,
                            int                       layer_id,
                            int                       expert_id,
                            ggml_layout_mode          layout,
                            bool                      hot) {
    if (!key_id.valid) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    unified_cache_key           key{ type, key_id, layer_id, expert_id };
    auto                        it = entries_.find(key);
    if (it != entries_.end()) {
        if (it->second.layout != layout) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in set_hot model=%llu name_hash=0x%llx have=%d want=%d\n",
                           (unsigned long long) key_id.model_id,
                           (unsigned long long) key_id.name_hash,
                           (int) it->second.layout,
                           (int) layout);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return;
        }
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
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Mode change ignored: cache already initialized\n");
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
            if (ggml_sycl_get_device(i) == dev) {
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

static size_t runtime_reserved_host_bytes_nolock() {
    return g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
}

// Helper: Create cache for a device
static unified_cache * create_cache_for_device(int device_id) {
    // Get queue for this device
    sycl::queue & queue = ggml_sycl_get_device(device_id).default_queue();

    // Reserve VRAM headroom for DMA staging buffers.
    // Match the resolved defaults (including evictable-weight sizing) unless explicitly overridden.
    size_t dma_reserve_mb    = 0;
    size_t dma_reserve_bytes = 0;
    size_t reserve_mb_env    = 0;
    bool   reserve_env_set   = parse_env_mb_value("GGML_SYCL_DMA_RESERVE_MB", reserve_mb_env);
    size_t slice_bytes       = 0;
    size_t buffers           = 0;
    if (reserve_env_set) {
        dma_reserve_mb    = reserve_mb_env;
        dma_reserve_bytes = dma_reserve_mb * 1024ULL * 1024ULL;
    } else {
        resolve_dma_defaults(slice_bytes, buffers);
        if (slice_bytes == 0 || buffers == 0) {
            dma_reserve_bytes = 0;
        } else {
            dma_reserve_bytes = slice_bytes * buffers;
        }
    }

    if (dma_reserve_bytes > 0) {
        g_runtime_reserved_bytes[device_id].fetch_add(dma_reserve_bytes, std::memory_order_relaxed);
        if (reserve_env_set) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reserving %.1f MB for DMA staging (fixed)\n",
                          dma_reserve_bytes / (1024.0 * 1024.0));
        } else {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reserving %.1f MB for DMA staging (buffers=%zu, slice=%.1f MB)\n",
                          dma_reserve_bytes / (1024.0 * 1024.0),
                          buffers,
                          slice_bytes / (1024.0 * 1024.0));
        }
    }

    // Auto-calculate budget if not set
    size_t budget = g_unified_cache_budget;
    if (budget == 0) {
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
        size_t base_mem = ggml_sycl_info().devices[device_id].total_vram;
        if (base_mem == 0) {
            base_mem = total_mem > 0 ? total_mem : free_mem;
        }

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

        char desc[256] = { 0 };
        ggml_backend_sycl_get_device_description(device_id, desc, sizeof(desc));
        GGML_LOG_INFO("[UNIFIED-CACHE] Device %d (%s): total=%.1f MB free=%.1f MB budget=%.1f MB (%d%%, headroom=%.1f MB)\n",
                      device_id,
                      desc,
                      base_mem / (1024.0f * 1024.0f),
                      free_mem / (1024.0f * 1024.0f),
                      budget / (1024.0f * 1024.0f),
                      pct,
                      headroom / (1024.0f * 1024.0f));
    }


    const size_t staging_bytes = resolve_host_staging_bytes();
    try {
        g_device_caches[device_id] = std::make_unique<unified_cache>(queue, budget, staging_bytes, dma_reserve_bytes);
        const size_t reserved = runtime_reserved_bytes_nolock(device_id);
        if (reserved > 0) {
            g_device_caches[device_id]->update_reserved_bytes(reserved);
        }
        return g_device_caches[device_id].get();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to initialize device %d: %s\n", device_id, e.what());
        return nullptr;
    }
}

// Helper: Create host cache for a device
static host_cache * create_host_cache_for_device(int device_id) {
    if (g_host_cache_shared) {
        return g_host_cache_shared.get();
    }
    sycl::queue & queue = ggml_sycl_get_device(device_id).default_queue();

    size_t budget = g_unified_cache_host_budget;
    if (budget == 0) {
        size_t total_mem = get_total_system_memory_bytes();
        if (total_mem == 0) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Host cache budget: unable to query system RAM, disabling host cache\n");
            return nullptr;
        }

        int pct = g_unified_cache_host_budget_pct;
        if (pct < 1) {
            pct = 1;
        } else if (pct > 100) {
            pct = 100;
        }

        budget = static_cast<size_t>(total_mem * (static_cast<double>(pct) / 100.0));
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Host cache budget=%.1f MB (%d%% of %.1f MB total RAM)\n",
                      budget / (1024.0 * 1024.0), pct, total_mem / (1024.0 * 1024.0));
    }

    const size_t staging_bytes = resolve_host_staging_bytes();
    size_t       reserve_bytes = resolve_host_reserve_bytes(staging_bytes);
    const int    device_count  = std::max(1, static_cast<int>(dpct::dev_mgr::instance().device_count()));
    if (reserve_bytes > 0) {
        const size_t total_reserve = reserve_bytes * static_cast<size_t>(device_count);
        if (total_reserve >= budget) {
            GGML_LOG_INFO("[UNIFIED-CACHE] Host reserve %.1f MB >= host budget %.1f MB; host cache disabled\n",
                          total_reserve / (1024.0 * 1024.0), budget / (1024.0 * 1024.0));
            return nullptr;
        }
        budget -= total_reserve;
        GGML_LOG_INFO("[UNIFIED-CACHE] Host reserve %.1f MB (staging %.1f MB x %d devices), host budget now %.1f MB\n",
                      total_reserve / (1024.0 * 1024.0),
                      staging_bytes / (1024.0 * 1024.0),
                      device_count,
                      budget / (1024.0 * 1024.0));
    }

    try {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Creating shared host_cache (device %d context)\n", device_id);
        g_host_cache_shared = std::make_unique<host_cache>(queue, budget);
        const size_t reserved_host = runtime_reserved_host_bytes_nolock();
        if (reserved_host > 0) {
            g_host_cache_shared->update_reserved_bytes(reserved_host);
        }
        host_cache * result = g_host_cache_shared.get();
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Shared host_cache ready (ptr=%p)\n", (void *) result);
        return result;
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

    int device_id = get_device_id_from_queue(queue);
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

    return create_host_cache_for_device(device_id);
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
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Budget change ignored: cache already initialized\n");
        return;
    }
    g_unified_cache_budget = bytes;
}

void set_unified_cache_budget_pct(int pct) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (g_cache_mode_locked) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Budget pct change ignored: cache already initialized\n");
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
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Host budget pct change ignored: cache already initialized\n");
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
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
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
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
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
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    return g_runtime_reserved_bytes[effective_device].load(std::memory_order_relaxed);
}

void unified_cache_add_runtime_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_runtime_reserved_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (g_host_cache_shared) {
        g_host_cache_shared->update_reserved_bytes(g_runtime_reserved_host_bytes.load(std::memory_order_relaxed));
    }
}

void unified_cache_sub_runtime_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    size_t cur  = g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
    size_t next = cur > bytes ? cur - bytes : 0;
    g_runtime_reserved_host_bytes.store(next, std::memory_order_relaxed);
    if (g_host_cache_shared) {
        g_host_cache_shared->update_reserved_bytes(next);
    }
}

size_t unified_cache_get_runtime_host_bytes() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    return g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
}

// === MoE Cache Helpers ===

void unpin_all_experts() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Unpin in all caches
    for (auto & [device_id, cache] : g_device_caches) {
        if (cache) {
            cache->unpin_experts();
        }
    }
}

// === Routing-Aware Expert Pre-staging ===

// Helper: Create a cache ID from expert pointer and metadata
// This is used for routing-aware pre-staging where we don't have a tensor object
static ggml_sycl_cache_id make_expert_ptr_cache_id(const void * expert_ptr,
                                                   size_t       expert_size,
                                                   int          layer_id,
                                                   int          expert_id) {
    ggml_sycl_cache_id id{};
    if (!expert_ptr) {
        return id;
    }

    // Use pointer address as unique identifier
    // Combined with layer_id and expert_id for full uniqueness
    const uint64_t ptr_hash = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(expert_ptr));

    id.valid         = true;
    id.model_id      = 0;  // Not model-specific for ptr-based keys
    id.has_gguf      = false;
    id.file_idx      = 0;
    id.file_offs     = 0;
    id.nbytes        = expert_size;
    id.name_hash     = ptr_hash;  // Use pointer as hash for uniqueness
    id.type          = GGML_TYPE_COUNT;
    id.tp_sharded    = false;
    id.tp_rank       = 0;
    id.tp_world_size = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = 0;
        id.tp_local_ne[i]  = 0;
        id.tp_offset_ne[i] = 0;
    }

    // Combine layer_id and expert_id into aux_id for uniqueness
    uint64_t aux = static_cast<uint64_t>(layer_id);
    aux          = detail::cache_hash_combine(aux, static_cast<uint64_t>(expert_id));
    aux          = detail::cache_hash_combine(aux, ptr_hash);
    id.aux_id    = aux;

    return id;
}

prestage_result prestage_routed_experts(void *          queue_ptr,
                                        const int32_t * expert_ids,
                                        int             n_expert_used,
                                        int             n_tokens,
                                        const void *    weight_base_ptr,
                                        size_t          expert_stride,
                                        size_t          expert_size,
                                        int             layer_id,
                                        int             n_experts_total,
                                        int             device_id) {
    prestage_result result{};
    result.n_staged  = 0;
    result.n_pinned  = 0;
    result.n_unique  = 0;
    result.success   = false;

    // Validate inputs
    if (!expert_ids || n_expert_used <= 0 || n_tokens <= 0 || !weight_base_ptr) {
        GGML_SYCL_DEBUG("[PRESTAGE] Invalid inputs: expert_ids=%p, n_expert_used=%d, n_tokens=%d, weight_base=%p\n",
                        (const void *) expert_ids, n_expert_used, n_tokens, weight_base_ptr);
        return result;
    }

    // Get unified cache for this device
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        GGML_SYCL_DEBUG("[PRESTAGE] No unified cache for device %d\n", device_id);
        return result;
    }

    // Step 1: Deduplicate expert IDs with bounds checking
    std::unordered_set<int32_t> unique_experts;
    const int total_ids = n_expert_used * n_tokens;

    for (int i = 0; i < total_ids; i++) {
        const int32_t expert_id = expert_ids[i];
        if (expert_id >= 0 && expert_id < n_experts_total) {
            unique_experts.insert(expert_id);
        }
    }

    result.n_unique = static_cast<int>(unique_experts.size());

    GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: %d unique experts from %d IDs (n_experts_total=%d)\n",
                    layer_id, result.n_unique, total_ids, n_experts_total);

    if (result.n_unique == 0) {
        result.success = true;  // Nothing to do, but not an error
        return result;
    }

    // Step 2: Check cache hits and build list of experts to stage
    std::vector<int32_t> experts_to_stage;
    experts_to_stage.reserve(result.n_unique);

    for (int32_t expert_id : unique_experts) {
        const void *       expert_ptr = static_cast<const char *>(weight_base_ptr) + expert_id * expert_stride;
        ggml_sycl_cache_id key        = make_expert_ptr_cache_id(expert_ptr, expert_size, layer_id, expert_id);

        // Check if already cached (any layout)
        if (!cache->is_cached_any(key)) {
            experts_to_stage.push_back(expert_id);
        }
    }

    GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: %zu cache hits, %zu to stage\n",
                    layer_id, unique_experts.size() - experts_to_stage.size(), experts_to_stage.size());

    // Step 3: Stage missing experts
    // NOTE: This is a placeholder - actual staging requires layout decisions and fill callbacks
    // For now, we just use ensure_cached with AOS layout (passthrough)
    for (int32_t expert_id : experts_to_stage) {
        const void *       expert_ptr = static_cast<const char *>(weight_base_ptr) + expert_id * expert_stride;
        ggml_sycl_cache_id key        = make_expert_ptr_cache_id(expert_ptr, expert_size, layer_id, expert_id);

        // Stage expert with AOS layout (passthrough - no reorder)
        void * cached_ptr = cache->ensure_cached(key,
                                                 expert_ptr,
                                                 expert_size,
                                                 cache_entry_type::MOE_EXPERT,
                                                 layer_id,
                                                 expert_id,
                                                 GGML_LAYOUT_AOS,
                                                 false);  // No content validation

        if (cached_ptr) {
            result.n_staged++;
        } else {
            GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: Failed to stage expert %d\n", layer_id, expert_id);
        }
    }

    // Step 4: Pin all unique experts (including those already cached)
    for (int32_t expert_id : unique_experts) {
        const void *       expert_ptr = static_cast<const char *>(weight_base_ptr) + expert_id * expert_stride;
        ggml_sycl_cache_id key        = make_expert_ptr_cache_id(expert_ptr, expert_size, layer_id, expert_id);

        cache->pin(key, GGML_LAYOUT_AOS);
        result.n_pinned++;
    }

    result.success = true;

    GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: Completed - staged=%d, pinned=%d, unique=%d\n",
                    layer_id, result.n_staged, result.n_pinned, result.n_unique);

    // Unused for now but may be used for async staging
    (void) queue_ptr;

    return result;
}

void unpin_routed_experts(const int32_t * expert_ids,
                          int             n_expert_used,
                          int             n_tokens,
                          const void *    weight_base_ptr,
                          size_t          expert_stride,
                          int             layer_id,
                          int             n_experts_total,
                          int             device_id) {
    // Validate inputs
    if (!expert_ids || n_expert_used <= 0 || n_tokens <= 0 || !weight_base_ptr) {
        return;
    }

    // Get unified cache for this device
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return;
    }

    // Deduplicate expert IDs (same as prestage)
    std::unordered_set<int32_t> unique_experts;
    const int                   total_ids = n_expert_used * n_tokens;

    for (int i = 0; i < total_ids; i++) {
        const int32_t expert_id = expert_ids[i];
        if (expert_id >= 0 && expert_id < n_experts_total) {
            unique_experts.insert(expert_id);
        }
    }

    // Unpin all unique experts
    for (int32_t expert_id : unique_experts) {
        const void *       expert_ptr = static_cast<const char *>(weight_base_ptr) + expert_id * expert_stride;
        ggml_sycl_cache_id key        = make_expert_ptr_cache_id(expert_ptr, expert_stride, layer_id, expert_id);

        cache->unpin(key, GGML_LAYOUT_AOS);
    }

    GGML_SYCL_DEBUG("[UNPIN] Layer %d: Unpinned %zu experts\n", layer_id, unique_experts.size());
}

void shutdown_unified_cache() {
    // Set shutdown flag FIRST so destructors skip sycl::free() calls
    g_sycl_shutting_down.store(true);

    // Clear all device caches
    // The destructors will skip cleanup due to the shutdown flag
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_device_caches.clear();
    g_host_cache_shared.reset();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Shutdown complete\n");
}

}  // namespace ggml_sycl
