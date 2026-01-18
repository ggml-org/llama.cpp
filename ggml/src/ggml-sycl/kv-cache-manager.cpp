//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// KV Cache Manager with per-attention-head granularity - STUB IMPLEMENTATION
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-vr5)
//
// TDD RED PHASE: This stub implementation should compile but all tests should fail.
// Real implementation comes in GREEN phase.
//

#include "kv-cache-manager.hpp"

#include <algorithm>
#include <cstring>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

// Internal KV entry representation
struct KVHeadEntry {
    uint64_t         id;
    uint32_t         layer_id;
    uint32_t         head_id;
    size_t           seq_start;
    size_t           seq_len;
    EvictionPriority priority;
    uint64_t         last_access;
    bool             pinned;
    MemoryTier       tier;

    // Data pointers
    void * device_ptr;
    void * host_ptr;

    // Size in bytes
    size_t size;
};

struct KVCacheManager::Impl {
    // Configuration
    uint32_t  num_layers               = 0;
    uint32_t  num_heads                = 0;
    size_t    head_dim                 = 0;
    ggml_type kv_type                  = GGML_TYPE_F16;
    size_t    bytes_per_token_per_head = 0;

    // Entry tracking
    std::unordered_map<uint64_t, KVHeadEntry> entries;
    uint64_t                                  next_id      = 1;
    int64_t                                   time_counter = 0;

    // Memory tracking
    size_t total_vram = 0;
    size_t total_host = 0;

    mutable std::mutex mutex;

    uint64_t allocate_id() { return next_id++; }

    int64_t now() { return ++time_counter; }
};

KVCacheManager::KVCacheManager() : impl_(new Impl()) {}

KVCacheManager::~KVCacheManager() {
    // Free any allocated memory
    for (auto & [id, entry] : impl_->entries) {
        if (entry.device_ptr) {
            // In real impl: sycl::free(entry.device_ptr, queue)
            free(entry.device_ptr);
        }
        if (entry.host_ptr) {
            free(entry.host_ptr);
        }
    }
    delete impl_;
}

void KVCacheManager::configure(uint32_t num_layers, uint32_t num_heads, size_t head_dim, ggml_type kv_type) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->num_layers               = num_layers;
    impl_->num_heads                = num_heads;
    impl_->head_dim                 = head_dim;
    impl_->kv_type                  = kv_type;
    // K and V interleaved: head_dim * type_size * 2 per token
    impl_->bytes_per_token_per_head = head_dim * ggml_type_size(kv_type) * 2;
}

uint32_t KVCacheManager::get_num_layers() const {
    return impl_->num_layers;
}

uint32_t KVCacheManager::get_num_heads() const {
    return impl_->num_heads;
}

size_t KVCacheManager::get_head_dim() const {
    return impl_->head_dim;
}

ggml_type KVCacheManager::get_kv_type() const {
    return impl_->kv_type;
}

size_t KVCacheManager::get_bytes_per_token_per_head() const {
    return impl_->bytes_per_token_per_head;
}

KVHandle KVCacheManager::allocate(uint32_t layer, uint32_t head, size_t seq_pos, size_t num_tokens) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    const size_t size = num_tokens * impl_->bytes_per_token_per_head;

    KVHeadEntry entry;
    entry.id          = impl_->allocate_id();
    entry.layer_id    = layer;
    entry.head_id     = head;
    entry.seq_start   = seq_pos;
    entry.seq_len     = num_tokens;
    entry.priority    = EvictionPriority::P4_COLD_EXPERT;
    entry.last_access = impl_->now();
    entry.pinned      = false;
    entry.tier        = MemoryTier::HOST;  // Start in HOST
    entry.size        = size;

    // Allocate host memory
    entry.host_ptr   = malloc(size);
    entry.device_ptr = nullptr;

    impl_->entries[entry.id] = entry;
    impl_->total_host += size;

    return KVHandle{ entry.id };
}

KVHandle KVCacheManager::extend(KVHandle handle, size_t additional_tokens) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto it = impl_->entries.find(handle.id);
    if (it == impl_->entries.end()) {
        return KVHandle{ 0 };
    }

    const auto & base          = it->second;
    const size_t new_seq_start = base.seq_start + base.seq_len;
    const size_t size          = additional_tokens * impl_->bytes_per_token_per_head;

    KVHeadEntry entry;
    entry.id          = impl_->allocate_id();
    entry.layer_id    = base.layer_id;
    entry.head_id     = base.head_id;
    entry.seq_start   = new_seq_start;
    entry.seq_len     = additional_tokens;
    entry.priority    = EvictionPriority::P4_COLD_EXPERT;
    entry.last_access = impl_->now();
    entry.pinned      = false;
    entry.tier        = MemoryTier::HOST;
    entry.size        = size;

    entry.host_ptr   = malloc(size);
    entry.device_ptr = nullptr;

    impl_->entries[entry.id] = entry;
    impl_->total_host += size;

    return KVHandle{ entry.id };
}

bool KVCacheManager::is_valid_handle(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->entries.find(handle.id) != impl_->entries.end();
}

KVHandle KVCacheManager::find_handle(uint32_t layer, uint32_t head, size_t token_pos) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    for (const auto & [id, entry] : impl_->entries) {
        if (entry.layer_id == layer && entry.head_id == head) {
            if (token_pos >= entry.seq_start && token_pos < entry.seq_start + entry.seq_len) {
                return KVHandle{ id };
            }
        }
    }
    return KVHandle{ 0 };
}

std::vector<KVHandle> KVCacheManager::get_handles_for_layer(uint32_t layer) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<KVHandle> handles;
    for (const auto & [id, entry] : impl_->entries) {
        if (entry.layer_id == layer) {
            handles.push_back(KVHandle{ id });
        }
    }
    return handles;
}

uint32_t KVCacheManager::get_layer_id(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.layer_id : 0;
}

uint32_t KVCacheManager::get_head_id(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.head_id : 0;
}

size_t KVCacheManager::get_seq_start(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.seq_start : 0;
}

size_t KVCacheManager::get_seq_len(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.seq_len : 0;
}

size_t KVCacheManager::get_allocation_size(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.size : 0;
}

size_t KVCacheManager::get_total_tokens(uint32_t layer, uint32_t head) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    size_t total = 0;
    for (const auto & [id, entry] : impl_->entries) {
        if (entry.layer_id == layer && entry.head_id == head) {
            total += entry.seq_len;
        }
    }
    return total;
}

void * KVCacheManager::get_k_data(KVHandle handle) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    if (it == impl_->entries.end()) {
        return nullptr;
    }

    auto & entry      = it->second;
    entry.last_access = impl_->now();

    // Return appropriate pointer based on tier
    if (entry.tier == MemoryTier::VRAM && entry.device_ptr) {
        return entry.device_ptr;
    }
    return entry.host_ptr;
}

void * KVCacheManager::get_v_data(KVHandle handle) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    if (it == impl_->entries.end()) {
        return nullptr;
    }

    auto & entry      = it->second;
    entry.last_access = impl_->now();

    // V is after K in interleaved layout
    // K size = seq_len * head_dim * type_size
    const size_t k_size = entry.seq_len * impl_->head_dim * ggml_type_size(impl_->kv_type);

    void * base = (entry.tier == MemoryTier::VRAM && entry.device_ptr) ? entry.device_ptr : entry.host_ptr;

    return base ? static_cast<uint8_t *>(base) + k_size : nullptr;
}

std::vector<KVPtrs> KVCacheManager::get_layer_kv(uint32_t                      layer,
                                                 const std::vector<uint32_t> & heads,
                                                 size_t                        seq_start,
                                                 size_t                        seq_len) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    (void) seq_start;
    (void) seq_len;

    std::vector<KVPtrs> result;
    result.reserve(heads.size());

    for (uint32_t head : heads) {
        KVPtrs ptrs{ nullptr, nullptr };

        // Find entry for this layer/head
        for (auto & [id, entry] : impl_->entries) {
            if (entry.layer_id == layer && entry.head_id == head) {
                entry.last_access = impl_->now();

                void * base = (entry.tier == MemoryTier::VRAM && entry.device_ptr) ? entry.device_ptr : entry.host_ptr;

                if (base) {
                    const size_t k_size = entry.seq_len * impl_->head_dim * ggml_type_size(impl_->kv_type);
                    ptrs.k              = base;
                    ptrs.v              = static_cast<uint8_t *>(base) + k_size;
                }
                break;
            }
        }

        result.push_back(ptrs);
    }

    return result;
}

void KVCacheManager::convert_to_interleaved(const void * grouped_data, void * interleaved_data, size_t seq_len) {
    const size_t head_k_size = impl_->head_dim * seq_len * ggml_type_size(impl_->kv_type);
    const size_t head_v_size = head_k_size;

    const uint8_t * src = static_cast<const uint8_t *>(grouped_data);
    uint8_t *       dst = static_cast<uint8_t *>(interleaved_data);

    // Source layout: [K0,K1,...,KN,V0,V1,...,VN]
    const uint8_t * src_k = src;
    const uint8_t * src_v = src + impl_->num_heads * head_k_size;

    // Dest layout: [K0,V0,K1,V1,...,KN,VN]
    for (uint32_t h = 0; h < impl_->num_heads; h++) {
        memcpy(dst, src_k + h * head_k_size, head_k_size);
        dst += head_k_size;
        memcpy(dst, src_v + h * head_v_size, head_v_size);
        dst += head_v_size;
    }
}

void KVCacheManager::convert_to_grouped(const void * interleaved_data, void * grouped_data, size_t seq_len) {
    const size_t head_k_size = impl_->head_dim * seq_len * ggml_type_size(impl_->kv_type);
    const size_t head_v_size = head_k_size;

    const uint8_t * src = static_cast<const uint8_t *>(interleaved_data);
    uint8_t *       dst = static_cast<uint8_t *>(grouped_data);

    // Source layout: [K0,V0,K1,V1,...,KN,VN]
    // Dest layout: [K0,K1,...,KN,V0,V1,...,VN]
    uint8_t * dst_k = dst;
    uint8_t * dst_v = dst + impl_->num_heads * head_k_size;

    for (uint32_t h = 0; h < impl_->num_heads; h++) {
        memcpy(dst_k + h * head_k_size, src, head_k_size);
        src += head_k_size;
        memcpy(dst_v + h * head_v_size, src, head_v_size);
        src += head_v_size;
    }
}

EvictionPriority KVCacheManager::get_priority(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.priority : EvictionPriority::P4_COLD_EXPERT;
}

bool KVCacheManager::is_pinned(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.pinned : false;
}

void KVCacheManager::mark_heads_active(uint32_t layer, const std::vector<uint32_t> & heads) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    for (uint32_t head : heads) {
        for (auto & [id, entry] : impl_->entries) {
            if (entry.layer_id == layer && entry.head_id == head) {
                entry.priority    = EvictionPriority::P1_ACTIVE_KV;
                entry.pinned      = true;
                entry.last_access = impl_->now();
            }
        }
    }
}

void KVCacheManager::mark_heads_cold(uint32_t layer, const std::vector<uint32_t> & heads) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    for (uint32_t head : heads) {
        for (auto & [id, entry] : impl_->entries) {
            if (entry.layer_id == layer && entry.head_id == head) {
                entry.priority = EvictionPriority::P4_COLD_EXPERT;
                entry.pinned   = false;
            }
        }
    }
}

void KVCacheManager::touch(KVHandle handle) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    if (it != impl_->entries.end()) {
        it->second.last_access = impl_->now();
    }
}

uint64_t KVCacheManager::get_last_access(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.last_access : 0;
}

MemoryTier KVCacheManager::get_tier(KVHandle handle) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    return it != impl_->entries.end() ? it->second.tier : MemoryTier::NONE;
}

void KVCacheManager::evict_to_host(KVHandle handle) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    if (it == impl_->entries.end()) {
        return;
    }

    auto & entry = it->second;
    if (entry.tier != MemoryTier::VRAM) {
        return;  // Already in host or other tier
    }

    // In real impl: copy device -> host, then free device memory
    // For now, just update tracking
    if (entry.device_ptr) {
        if (!entry.host_ptr) {
            entry.host_ptr = malloc(entry.size);
        }
        memcpy(entry.host_ptr, entry.device_ptr, entry.size);
        free(entry.device_ptr);
        entry.device_ptr = nullptr;
    }

    impl_->total_vram -= entry.size;
    impl_->total_host += entry.size;
    entry.tier = MemoryTier::HOST;
}

void KVCacheManager::prefetch_to_vram(KVHandle handle) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto                        it = impl_->entries.find(handle.id);
    if (it == impl_->entries.end()) {
        return;
    }

    auto & entry = it->second;
    if (entry.tier == MemoryTier::VRAM) {
        return;  // Already in VRAM
    }

    // In real impl: allocate device memory, copy host -> device
    // For now, simulate with malloc (tests don't use actual GPU)
    if (!entry.device_ptr) {
        entry.device_ptr = malloc(entry.size);
    }
    if (entry.host_ptr) {
        memcpy(entry.device_ptr, entry.host_ptr, entry.size);
    }

    if (entry.tier == MemoryTier::HOST) {
        impl_->total_host -= entry.size;
    }
    impl_->total_vram += entry.size;
    entry.tier = MemoryTier::VRAM;
}

bool KVCacheManager::is_prefetching(KVHandle handle) const {
    // In real impl, would track async prefetch operations
    // For now, always return false (prefetch is synchronous)
    (void) handle;
    return false;
}

void KVCacheManager::prefetch_layer(uint32_t layer) {
    // Get handles outside lock to avoid deadlock
    std::vector<KVHandle> handles;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (const auto & [id, entry] : impl_->entries) {
            if (entry.layer_id == layer) {
                handles.push_back(KVHandle{ id });
            }
        }
    }

    // Prefetch each handle
    for (const auto & handle : handles) {
        prefetch_to_vram(handle);
    }
}

std::vector<KVHandle> KVCacheManager::get_eviction_candidates(size_t count) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Collect evictable entries
    std::vector<const KVHeadEntry *> candidates;
    for (const auto & [id, entry] : impl_->entries) {
        // Skip pinned/active entries
        if (entry.pinned || entry.priority == EvictionPriority::P1_ACTIVE_KV) {
            continue;
        }
        candidates.push_back(&entry);
    }

    // Sort by priority (higher number first), then by last_access (older first)
    std::sort(candidates.begin(), candidates.end(), [](const KVHeadEntry * a, const KVHeadEntry * b) {
        if (static_cast<int>(a->priority) != static_cast<int>(b->priority)) {
            return static_cast<int>(a->priority) > static_cast<int>(b->priority);
        }
        return a->last_access < b->last_access;
    });

    // Return up to count handles
    std::vector<KVHandle> result;
    for (size_t i = 0; i < std::min(count, candidates.size()); i++) {
        result.push_back(KVHandle{ candidates[i]->id });
    }

    return result;
}

size_t KVCacheManager::evict_cold_heads(size_t bytes_needed) {
    size_t evicted = 0;

    // Get all candidates and filter to those in VRAM
    std::vector<KVHandle> vram_candidates;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (const auto & [id, entry] : impl_->entries) {
            // Skip pinned/active entries
            if (entry.pinned || entry.priority == EvictionPriority::P1_ACTIVE_KV) {
                continue;
            }
            // Only consider VRAM entries for eviction
            if (entry.tier == MemoryTier::VRAM) {
                vram_candidates.push_back(KVHandle{ id });
            }
        }

        // Sort by priority (higher number = lower priority = evict first), then LRU
        std::sort(vram_candidates.begin(), vram_candidates.end(), [this](const KVHandle & a, const KVHandle & b) {
            auto it_a = impl_->entries.find(a.id);
            auto it_b = impl_->entries.find(b.id);
            if (it_a == impl_->entries.end()) {
                return false;
            }
            if (it_b == impl_->entries.end()) {
                return true;
            }
            const auto & ea = it_a->second;
            const auto & eb = it_b->second;
            if (static_cast<int>(ea.priority) != static_cast<int>(eb.priority)) {
                return static_cast<int>(ea.priority) > static_cast<int>(eb.priority);
            }
            return ea.last_access < eb.last_access;
        });
    }

    // Evict candidates until we free enough space
    for (const auto & handle : vram_candidates) {
        if (evicted >= bytes_needed) {
            break;
        }

        size_t size = get_allocation_size(handle);
        evict_to_host(handle);
        evicted += size;
    }

    return evicted;
}

size_t KVCacheManager::get_total_vram_usage() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->total_vram;
}

size_t KVCacheManager::get_total_host_usage() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->total_host;
}

}  // namespace ggml_sycl
