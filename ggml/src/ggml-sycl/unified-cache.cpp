//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "unified-cache.hpp"

#include "alloc-registry.hpp"
#include "expert-prefetch.hpp"
#include "kv-tier-manager.hpp"
#include "common.hpp"
#include "ggml-impl.h"
#include "ggml-sycl.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
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
static std::shared_mutex                                       g_cache_rw_mutex;
static size_t                                                  g_unified_cache_budget      = 0;  // 0 = auto-calculate
static int                                                     g_unified_cache_budget_pct  = 100;
static size_t                                                  g_unified_cache_host_budget = 0;  // 0 = auto-calc
static int                                                     g_unified_cache_host_budget_pct = 90;
static unified_cache_mode                                      g_cache_mode = unified_cache_mode::AUTO;
static std::atomic<bool> g_cache_mode_locked{ false };   // Locked after first cache access
static std::atomic<bool> g_sycl_shutting_down{ false };  // Set during shutdown to skip sycl::free()
static std::array<std::atomic<size_t>, GGML_SYCL_MAX_DEVICES> g_runtime_reserved_bytes{};
static std::array<std::atomic<size_t>, GGML_SYCL_MAX_DEVICES> g_runtime_reserved_baseline{};
static std::atomic<size_t> g_runtime_cat_bytes[GGML_SYCL_MAX_DEVICES][static_cast<int>(runtime_category::COUNT)]{};
static std::atomic<size_t> g_runtime_reserved_host_bytes{};
static std::atomic<size_t> g_runtime_host_cat_bytes[static_cast<int>(runtime_category::COUNT)]{};
static std::array<std::atomic<size_t>, GGML_SYCL_MAX_DEVICES> g_runtime_managed_reserved_bytes{};
static std::atomic<size_t>                                    g_runtime_managed_reserved_host_bytes{};
static std::atomic<bool>     g_atexit_registered{ false };  // Ensure atexit handler registered once
static std::atomic<int>      g_host_cache_guard_errors{ 0 };
static std::atomic<int>      g_host_cache_guard_enabled{ -1 };
static constexpr size_t      k_host_cache_guard_bytes   = 64;
static constexpr uint8_t     k_host_cache_guard_pattern = 0xA5;
static std::atomic<int>      g_cache_assert_enabled{ -1 };
static std::atomic<int>      g_copy_trace_enabled{ -1 };
static std::atomic<bool>     g_graph_compute_active{ false };
static std::mutex            g_runtime_alloc_mutex;
static std::atomic<uint64_t> g_runtime_alloc_id{ 1 };

struct runtime_alloc_record {
    alloc_handle  handle{};
    sycl::queue * queue            = nullptr;
    bool          uses_pinned_pool = false;
    bool          from_arena       = false;  // True if sub-allocated from vram_arena (KV zone)
    std::string   cohort_id;
};

static std::unordered_map<void *, runtime_alloc_record> g_runtime_alloc_registry;
static std::unordered_map<std::string, alloc_tier>      g_runtime_cohort_tier;
static std::mutex                                       g_offload_pool_mutex;
static std::atomic<uint64_t>                            g_offload_pool_lease_id{ 1 };

struct offload_pool_key {
    int                 device    = -1;
    offload_buffer_role role      = offload_buffer_role::OTHER;
    alloc_tier          tier      = alloc_tier::HOST_PINNED;
    runtime_category    category  = runtime_category::OTHER;
    size_t              alignment = 64;
};

struct offload_pool_key_hash {
    size_t operator()(const offload_pool_key & key) const {
        size_t h = 0;
        h        = detail::cache_hash_combine(h, std::hash<int>()(key.device));
        h        = detail::cache_hash_combine(h, std::hash<int>()(static_cast<int>(key.role)));
        h        = detail::cache_hash_combine(h, std::hash<int>()(static_cast<int>(key.tier)));
        h        = detail::cache_hash_combine(h, std::hash<int>()(static_cast<int>(key.category)));
        h        = detail::cache_hash_combine(h, std::hash<size_t>()(key.alignment));
        return h;
    }
};

static bool operator==(const offload_pool_key & a, const offload_pool_key & b) {
    return a.device == b.device && a.role == b.role && a.tier == b.tier && a.category == b.category &&
           a.alignment == b.alignment;
}

struct offload_pool_slot {
    alloc_handle     handle{};
    offload_pool_key key{};
    bool             in_use   = false;
    uint64_t         lease_id = 0;
};

static std::unordered_map<void *, offload_pool_slot>                                    g_offload_pool_slots;
static std::unordered_map<offload_pool_key, std::vector<void *>, offload_pool_key_hash> g_offload_pool_free;

static std::atomic<uint64_t> g_offload_wait_count{ 0 };
static std::atomic<uint64_t> g_offload_wait_count_forced{ 0 };
static std::atomic<uint64_t> g_offload_wait_count_fallback{ 0 };
static std::atomic<uint64_t> g_offload_alloc_count_host{ 0 };
static std::atomic<uint64_t> g_offload_alloc_count_device{ 0 };
static std::atomic<uint64_t> g_offload_alloc_count_shared{ 0 };
static std::atomic<uint64_t> g_offload_pool_hit_count{ 0 };
static std::atomic<uint64_t> g_offload_pool_miss_count{ 0 };
static std::atomic<uint64_t> g_offload_cross_domain_transfer_count{ 0 };
static std::atomic<uint64_t> g_offload_cross_domain_transfer_count_pp{ 0 };
static std::atomic<uint64_t> g_offload_cross_domain_transfer_count_tg{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_h2d{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_d2h{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_h2d_pp{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_h2d_tg{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_d2h_pp{ 0 };
static std::atomic<uint64_t> g_offload_transfer_bytes_d2h_tg{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_cpu{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_island{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_cpu_pp{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_cpu_tg{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_pp{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_tg{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_island_pp{ 0 };
static std::atomic<uint64_t> g_offload_dispatch_count_gpu_island_tg{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_count{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_count_pp{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_count_tg{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_elided_count{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_elided_count_pp{ 0 };
static std::atomic<uint64_t> g_offload_transition_wait_elided_count_tg{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_call_count{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_calls_unified_alloc_host{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes_unified_alloc_host{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_calls_unified_cache_host_chunk{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes_unified_cache_host_chunk{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_calls_host_malloc{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes_host_malloc{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_calls_other{ 0 };
static std::atomic<uint64_t> g_offload_host_alloc_bytes_other{ 0 };
static std::mutex            g_offload_host_alloc_by_tag_mutex;
static std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> g_offload_host_alloc_by_tag;
static std::atomic<int> g_offload_phase{ static_cast<int>(offload_phase::UNKNOWN) };

static int get_device_id_from_queue(sycl::queue & queue);

static const char * alloc_tier_name(alloc_tier tier) {
    switch (tier) {
        case alloc_tier::DEVICE_VRAM:
            return "device_vram";
        case alloc_tier::HOST_PINNED:
            return "host_pinned";
        case alloc_tier::MMAP_TRACKED:
            return "mmap_tracked";
        default:
            return "unknown";
    }
}

static inline size_t align_up(size_t value, size_t alignment) {
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

static runtime_category category_from_role(alloc_role role) {
    switch (role) {
        case alloc_role::KV:
            return runtime_category::KV_CACHE;
        case alloc_role::COMPUTE:
            return runtime_category::COMPUTE;
        case alloc_role::STAGING:
            return runtime_category::STAGING;
        case alloc_role::GRAPH_TMP:
            return runtime_category::GRAPH;
        case alloc_role::TP_TMP:
            return runtime_category::GRAPH;
        case alloc_role::EXPERT_STAGING:
            return runtime_category::EXPERT_CACHE;
        case alloc_role::WEIGHT:
        case alloc_role::OTHER:
        default:
            return runtime_category::OTHER;
    }
}

static inline void unified_managed_add_device_bytes(int device, size_t bytes) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES || bytes == 0) {
        return;
    }
    g_runtime_managed_reserved_bytes[device].fetch_add(bytes, std::memory_order_relaxed);
}

static inline void unified_managed_sub_device_bytes(int device, size_t bytes) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES || bytes == 0) {
        return;
    }
    g_runtime_managed_reserved_bytes[device].fetch_sub(bytes, std::memory_order_relaxed);
}

static inline void unified_managed_add_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    g_runtime_managed_reserved_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

static inline void unified_managed_sub_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    g_runtime_managed_reserved_host_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

bool unified_alloc_strict_mode() {
    static std::atomic<int> s_strict{ -1 };
    int                     cached = s_strict.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached != 0;
    }
    const char * env    = std::getenv("GGML_SYCL_UNIFIED_ALLOC_STRICT");
    const int    strict = (env && std::atoi(env) != 0) ? 1 : 0;
    s_strict.store(strict, std::memory_order_release);
    return strict != 0;
}

bool offload_stats_enabled() {
    const char * env = std::getenv("GGML_SYCL_OFFLOAD_STATS");
    return env && std::atoi(env) != 0;
}

void offload_stats_reset() {
    g_offload_wait_count.store(0, std::memory_order_relaxed);
    g_offload_wait_count_forced.store(0, std::memory_order_relaxed);
    g_offload_wait_count_fallback.store(0, std::memory_order_relaxed);
    g_offload_alloc_count_host.store(0, std::memory_order_relaxed);
    g_offload_alloc_count_device.store(0, std::memory_order_relaxed);
    g_offload_alloc_count_shared.store(0, std::memory_order_relaxed);
    g_offload_pool_hit_count.store(0, std::memory_order_relaxed);
    g_offload_pool_miss_count.store(0, std::memory_order_relaxed);
    g_offload_cross_domain_transfer_count.store(0, std::memory_order_relaxed);
    g_offload_cross_domain_transfer_count_pp.store(0, std::memory_order_relaxed);
    g_offload_cross_domain_transfer_count_tg.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_h2d.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_d2h.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_h2d_pp.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_h2d_tg.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_d2h_pp.store(0, std::memory_order_relaxed);
    g_offload_transfer_bytes_d2h_tg.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_cpu.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_island.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_cpu_pp.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_cpu_tg.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_pp.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_tg.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_island_pp.store(0, std::memory_order_relaxed);
    g_offload_dispatch_count_gpu_island_tg.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_count.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_count_pp.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_count_tg.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_elided_count.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_elided_count_pp.store(0, std::memory_order_relaxed);
    g_offload_transition_wait_elided_count_tg.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_call_count.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_calls_unified_alloc_host.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_unified_alloc_host.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_calls_unified_cache_host_chunk.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_unified_cache_host_chunk.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_calls_host_malloc.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_host_malloc.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_calls_other.store(0, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_other.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_offload_host_alloc_by_tag_mutex);
        g_offload_host_alloc_by_tag.clear();
    }
    g_offload_phase.store(static_cast<int>(offload_phase::UNKNOWN), std::memory_order_relaxed);
}

void offload_stats_set_phase(offload_phase phase) {
    g_offload_phase.store(static_cast<int>(phase), std::memory_order_relaxed);
}

offload_phase offload_stats_phase() {
    const int phase = g_offload_phase.load(std::memory_order_relaxed);
    switch (phase) {
        case static_cast<int>(offload_phase::PP):
            return offload_phase::PP;
        case static_cast<int>(offload_phase::TG):
            return offload_phase::TG;
        default:
            return offload_phase::UNKNOWN;
    }
}

static inline offload_phase offload_stats_current_phase() {
    const int phase = g_offload_phase.load(std::memory_order_relaxed);
    switch (phase) {
        case static_cast<int>(offload_phase::PP):
            return offload_phase::PP;
        case static_cast<int>(offload_phase::TG):
            return offload_phase::TG;
        default:
            return offload_phase::UNKNOWN;
    }
}

void offload_stats_note_wait(bool fallback) {
    g_offload_wait_count.fetch_add(1, std::memory_order_relaxed);
    if (fallback) {
        g_offload_wait_count_fallback.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_offload_wait_count_forced.fetch_add(1, std::memory_order_relaxed);
    }
}

void offload_stats_note_alloc(alloc_tier tier) {
    switch (tier) {
        case alloc_tier::DEVICE_VRAM:
            g_offload_alloc_count_device.fetch_add(1, std::memory_order_relaxed);
            break;
        case alloc_tier::HOST_PINNED:
            g_offload_alloc_count_host.fetch_add(1, std::memory_order_relaxed);
            break;
        case alloc_tier::MMAP_TRACKED:
            g_offload_alloc_count_shared.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
    }
}

void offload_stats_note_pool_hit() {
    g_offload_pool_hit_count.fetch_add(1, std::memory_order_relaxed);
}

void offload_stats_note_pool_miss() {
    g_offload_pool_miss_count.fetch_add(1, std::memory_order_relaxed);
}

void offload_stats_note_transfer(bool h2d, size_t bytes) {
    const offload_phase phase = offload_stats_current_phase();
    if (h2d) {
        g_offload_transfer_bytes_h2d.fetch_add(bytes, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_transfer_bytes_h2d_pp.fetch_add(bytes, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_transfer_bytes_h2d_tg.fetch_add(bytes, std::memory_order_relaxed);
        }
    } else {
        g_offload_transfer_bytes_d2h.fetch_add(bytes, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_transfer_bytes_d2h_pp.fetch_add(bytes, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_transfer_bytes_d2h_tg.fetch_add(bytes, std::memory_order_relaxed);
        }
    }
}

void offload_stats_note_cross_domain_transfer(size_t bytes) {
    GGML_UNUSED(bytes);
    const offload_phase phase = offload_stats_current_phase();
    g_offload_cross_domain_transfer_count.fetch_add(1, std::memory_order_relaxed);
    if (phase == offload_phase::PP) {
        g_offload_cross_domain_transfer_count_pp.fetch_add(1, std::memory_order_relaxed);
    } else if (phase == offload_phase::TG) {
        g_offload_cross_domain_transfer_count_tg.fetch_add(1, std::memory_order_relaxed);
    }
}

void offload_stats_note_dispatch(bool cpu, bool gpu_island) {
    const offload_phase phase = offload_stats_current_phase();
    if (cpu) {
        g_offload_dispatch_count_cpu.fetch_add(1, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_dispatch_count_cpu_pp.fetch_add(1, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_dispatch_count_cpu_tg.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }

    g_offload_dispatch_count_gpu.fetch_add(1, std::memory_order_relaxed);
    if (phase == offload_phase::PP) {
        g_offload_dispatch_count_gpu_pp.fetch_add(1, std::memory_order_relaxed);
    } else if (phase == offload_phase::TG) {
        g_offload_dispatch_count_gpu_tg.fetch_add(1, std::memory_order_relaxed);
    }

    if (gpu_island) {
        g_offload_dispatch_count_gpu_island.fetch_add(1, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_dispatch_count_gpu_island_pp.fetch_add(1, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_dispatch_count_gpu_island_tg.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void offload_stats_note_transition_wait(bool waited) {
    const offload_phase phase = offload_stats_current_phase();
    if (waited) {
        g_offload_transition_wait_count.fetch_add(1, std::memory_order_relaxed);
        if (phase == offload_phase::PP) {
            g_offload_transition_wait_count_pp.fetch_add(1, std::memory_order_relaxed);
        } else if (phase == offload_phase::TG) {
            g_offload_transition_wait_count_tg.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }

    g_offload_transition_wait_elided_count.fetch_add(1, std::memory_order_relaxed);
    if (phase == offload_phase::PP) {
        g_offload_transition_wait_elided_count_pp.fetch_add(1, std::memory_order_relaxed);
    } else if (phase == offload_phase::TG) {
        g_offload_transition_wait_elided_count_tg.fetch_add(1, std::memory_order_relaxed);
    }
}

void offload_stats_note_host_alloc(const char * tag, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    g_offload_host_alloc_call_count.fetch_add(1, std::memory_order_relaxed);
    g_offload_host_alloc_bytes.fetch_add(bytes, std::memory_order_relaxed);

    const std::string tag_s = tag && tag[0] != '\0' ? std::string(tag) : std::string("(unknown)");
    {
        std::lock_guard<std::mutex> lock(g_offload_host_alloc_by_tag_mutex);
        auto &                      entry = g_offload_host_alloc_by_tag[tag_s];
        entry.first += 1;
        entry.second += bytes;
    }

    if (tag_s == "unified_alloc:host") {
        g_offload_host_alloc_calls_unified_alloc_host.fetch_add(1, std::memory_order_relaxed);
        g_offload_host_alloc_bytes_unified_alloc_host.fetch_add(bytes, std::memory_order_relaxed);
        return;
    }
    if (tag_s.find("unified_cache:host_chunk") != std::string::npos ||
        tag_s.find("unified_cache:host_temp") != std::string::npos) {
        g_offload_host_alloc_calls_unified_cache_host_chunk.fetch_add(1, std::memory_order_relaxed);
        g_offload_host_alloc_bytes_unified_cache_host_chunk.fetch_add(bytes, std::memory_order_relaxed);
        return;
    }
    if (tag_s == "host_malloc" || tag_s == "tp_shared_host") {
        g_offload_host_alloc_calls_host_malloc.fetch_add(1, std::memory_order_relaxed);
        g_offload_host_alloc_bytes_host_malloc.fetch_add(bytes, std::memory_order_relaxed);
        return;
    }
    g_offload_host_alloc_calls_other.fetch_add(1, std::memory_order_relaxed);
    g_offload_host_alloc_bytes_other.fetch_add(bytes, std::memory_order_relaxed);
}

offload_stats_snapshot offload_stats_get() {
    offload_stats_snapshot s{};
    s.wait_count                      = g_offload_wait_count.load(std::memory_order_relaxed);
    s.wait_count_forced               = g_offload_wait_count_forced.load(std::memory_order_relaxed);
    s.wait_count_fallback             = g_offload_wait_count_fallback.load(std::memory_order_relaxed);
    s.alloc_count_host                = g_offload_alloc_count_host.load(std::memory_order_relaxed);
    s.alloc_count_device              = g_offload_alloc_count_device.load(std::memory_order_relaxed);
    s.alloc_count_shared              = g_offload_alloc_count_shared.load(std::memory_order_relaxed);
    s.pool_hit_count                  = g_offload_pool_hit_count.load(std::memory_order_relaxed);
    s.pool_miss_count                 = g_offload_pool_miss_count.load(std::memory_order_relaxed);
    s.cross_domain_transfer_count     = g_offload_cross_domain_transfer_count.load(std::memory_order_relaxed);
    s.cross_domain_transfer_count_pp  = g_offload_cross_domain_transfer_count_pp.load(std::memory_order_relaxed);
    s.cross_domain_transfer_count_tg  = g_offload_cross_domain_transfer_count_tg.load(std::memory_order_relaxed);
    s.transfer_bytes_h2d              = g_offload_transfer_bytes_h2d.load(std::memory_order_relaxed);
    s.transfer_bytes_d2h              = g_offload_transfer_bytes_d2h.load(std::memory_order_relaxed);
    s.transfer_bytes_h2d_pp           = g_offload_transfer_bytes_h2d_pp.load(std::memory_order_relaxed);
    s.transfer_bytes_h2d_tg           = g_offload_transfer_bytes_h2d_tg.load(std::memory_order_relaxed);
    s.transfer_bytes_d2h_pp           = g_offload_transfer_bytes_d2h_pp.load(std::memory_order_relaxed);
    s.transfer_bytes_d2h_tg           = g_offload_transfer_bytes_d2h_tg.load(std::memory_order_relaxed);
    s.dispatch_count_cpu              = g_offload_dispatch_count_cpu.load(std::memory_order_relaxed);
    s.dispatch_count_gpu              = g_offload_dispatch_count_gpu.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_island       = g_offload_dispatch_count_gpu_island.load(std::memory_order_relaxed);
    s.dispatch_count_cpu_pp           = g_offload_dispatch_count_cpu_pp.load(std::memory_order_relaxed);
    s.dispatch_count_cpu_tg           = g_offload_dispatch_count_cpu_tg.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_pp           = g_offload_dispatch_count_gpu_pp.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_tg           = g_offload_dispatch_count_gpu_tg.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_island_pp    = g_offload_dispatch_count_gpu_island_pp.load(std::memory_order_relaxed);
    s.dispatch_count_gpu_island_tg    = g_offload_dispatch_count_gpu_island_tg.load(std::memory_order_relaxed);
    s.transition_wait_count           = g_offload_transition_wait_count.load(std::memory_order_relaxed);
    s.transition_wait_count_pp        = g_offload_transition_wait_count_pp.load(std::memory_order_relaxed);
    s.transition_wait_count_tg        = g_offload_transition_wait_count_tg.load(std::memory_order_relaxed);
    s.transition_wait_elided_count    = g_offload_transition_wait_elided_count.load(std::memory_order_relaxed);
    s.transition_wait_elided_count_pp = g_offload_transition_wait_elided_count_pp.load(std::memory_order_relaxed);
    s.transition_wait_elided_count_tg = g_offload_transition_wait_elided_count_tg.load(std::memory_order_relaxed);
    s.host_alloc_call_count           = g_offload_host_alloc_call_count.load(std::memory_order_relaxed);
    s.host_alloc_bytes                = g_offload_host_alloc_bytes.load(std::memory_order_relaxed);
    s.host_alloc_calls_unified_alloc_host =
        g_offload_host_alloc_calls_unified_alloc_host.load(std::memory_order_relaxed);
    s.host_alloc_bytes_unified_alloc_host =
        g_offload_host_alloc_bytes_unified_alloc_host.load(std::memory_order_relaxed);
    s.host_alloc_calls_unified_cache_host_chunk =
        g_offload_host_alloc_calls_unified_cache_host_chunk.load(std::memory_order_relaxed);
    s.host_alloc_bytes_unified_cache_host_chunk =
        g_offload_host_alloc_bytes_unified_cache_host_chunk.load(std::memory_order_relaxed);
    s.host_alloc_calls_host_malloc = g_offload_host_alloc_calls_host_malloc.load(std::memory_order_relaxed);
    s.host_alloc_bytes_host_malloc = g_offload_host_alloc_bytes_host_malloc.load(std::memory_order_relaxed);
    s.host_alloc_calls_other       = g_offload_host_alloc_calls_other.load(std::memory_order_relaxed);
    s.host_alloc_bytes_other       = g_offload_host_alloc_bytes_other.load(std::memory_order_relaxed);
    return s;
}

static std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> offload_host_alloc_top_by_calls(
    size_t top_n) {
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> rows;
    {
        std::lock_guard<std::mutex> lock(g_offload_host_alloc_by_tag_mutex);
        rows.reserve(g_offload_host_alloc_by_tag.size());
        for (const auto & it : g_offload_host_alloc_by_tag) {
            rows.push_back(it);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        if (a.second.first != b.second.first) {
            return a.second.first > b.second.first;
        }
        return a.second.second > b.second.second;
    });
    if (rows.size() > top_n) {
        rows.resize(top_n);
    }
    return rows;
}

void offload_stats_log_summary(const char * tag, int device) {
    if (!offload_stats_enabled()) {
        return;
    }
    const offload_stats_snapshot s     = offload_stats_get();
    const offload_phase          phase = offload_stats_current_phase();
    const char * phase_name = phase == offload_phase::PP ? "pp" : phase == offload_phase::TG ? "tg" : "unknown";
    fprintf(
        stderr,
        "[SYCL-OFFLOAD-STATS] tag=%s device=%d wait_count=%llu wait_count_forced=%llu "
        "wait_count_fallback=%llu alloc_count_host=%llu alloc_count_device=%llu "
        "alloc_count_shared=%llu pool_hit_count=%llu pool_miss_count=%llu phase=%s "
        "cross_domain_transfer_count=%llu cross_domain_transfer_count_pp=%llu cross_domain_transfer_count_tg=%llu "
        "transfer_bytes_h2d=%llu transfer_bytes_h2d_pp=%llu transfer_bytes_h2d_tg=%llu "
        "transfer_bytes_d2h=%llu transfer_bytes_d2h_pp=%llu transfer_bytes_d2h_tg=%llu "
        "dispatch_count_cpu=%llu dispatch_count_gpu=%llu dispatch_count_gpu_island=%llu "
        "dispatch_count_cpu_pp=%llu dispatch_count_cpu_tg=%llu "
        "dispatch_count_gpu_pp=%llu dispatch_count_gpu_tg=%llu "
        "dispatch_count_gpu_island_pp=%llu dispatch_count_gpu_island_tg=%llu "
        "transition_wait_count=%llu transition_wait_count_pp=%llu transition_wait_count_tg=%llu "
        "transition_wait_elided_count=%llu transition_wait_elided_count_pp=%llu transition_wait_elided_count_tg=%llu "
        "host_alloc_call_count=%llu host_alloc_bytes=%llu "
        "host_alloc_calls_unified_alloc_host=%llu host_alloc_bytes_unified_alloc_host=%llu "
        "host_alloc_calls_unified_cache_host_chunk=%llu host_alloc_bytes_unified_cache_host_chunk=%llu "
        "host_alloc_calls_host_malloc=%llu host_alloc_bytes_host_malloc=%llu "
        "host_alloc_calls_other=%llu host_alloc_bytes_other=%llu\n",
        tag ? tag : "graph", device, (unsigned long long) s.wait_count, (unsigned long long) s.wait_count_forced,
        (unsigned long long) s.wait_count_fallback, (unsigned long long) s.alloc_count_host,
        (unsigned long long) s.alloc_count_device, (unsigned long long) s.alloc_count_shared,
        (unsigned long long) s.pool_hit_count, (unsigned long long) s.pool_miss_count, phase_name,
        (unsigned long long) s.cross_domain_transfer_count, (unsigned long long) s.cross_domain_transfer_count_pp,
        (unsigned long long) s.cross_domain_transfer_count_tg, (unsigned long long) s.transfer_bytes_h2d,
        (unsigned long long) s.transfer_bytes_h2d_pp, (unsigned long long) s.transfer_bytes_h2d_tg,
        (unsigned long long) s.transfer_bytes_d2h, (unsigned long long) s.transfer_bytes_d2h_pp,
        (unsigned long long) s.transfer_bytes_d2h_tg, (unsigned long long) s.dispatch_count_cpu,
        (unsigned long long) s.dispatch_count_gpu, (unsigned long long) s.dispatch_count_gpu_island,
        (unsigned long long) s.dispatch_count_cpu_pp, (unsigned long long) s.dispatch_count_cpu_tg,
        (unsigned long long) s.dispatch_count_gpu_pp, (unsigned long long) s.dispatch_count_gpu_tg,
        (unsigned long long) s.dispatch_count_gpu_island_pp, (unsigned long long) s.dispatch_count_gpu_island_tg,
        (unsigned long long) s.transition_wait_count, (unsigned long long) s.transition_wait_count_pp,
        (unsigned long long) s.transition_wait_count_tg, (unsigned long long) s.transition_wait_elided_count,
        (unsigned long long) s.transition_wait_elided_count_pp, (unsigned long long) s.transition_wait_elided_count_tg,
        (unsigned long long) s.host_alloc_call_count, (unsigned long long) s.host_alloc_bytes,
        (unsigned long long) s.host_alloc_calls_unified_alloc_host,
        (unsigned long long) s.host_alloc_bytes_unified_alloc_host,
        (unsigned long long) s.host_alloc_calls_unified_cache_host_chunk,
        (unsigned long long) s.host_alloc_bytes_unified_cache_host_chunk,
        (unsigned long long) s.host_alloc_calls_host_malloc, (unsigned long long) s.host_alloc_bytes_host_malloc,
        (unsigned long long) s.host_alloc_calls_other, (unsigned long long) s.host_alloc_bytes_other);

    int top_n = 5;
    if (const char * env = std::getenv("GGML_SYCL_HOST_ALLOC_TOP")) {
        top_n = std::max(1, std::atoi(env));
    }
    const auto rows = offload_host_alloc_top_by_calls(static_cast<size_t>(top_n));
    if (!rows.empty()) {
        fprintf(stderr, "[SYCL-OFFLOAD-HOST-ALLOC] tag=%s device=%d top=%d", tag ? tag : "graph", device, top_n);
        for (const auto & row : rows) {
            fprintf(stderr, " [%s calls=%llu bytes=%llu]", row.first.c_str(), (unsigned long long) row.second.first,
                    (unsigned long long) row.second.second);
        }
        fprintf(stderr, "\n");
    }

    long warn_calls = 0;
    if (const char * env = std::getenv("GGML_SYCL_HOST_ALLOC_WARN_CALLS")) {
        warn_calls = std::strtol(env, nullptr, 10);
    }
    if (warn_calls > 0 && s.host_alloc_call_count > static_cast<uint64_t>(warn_calls)) {
        fprintf(stderr,
                "[SYCL-OFFLOAD-HOST-ALLOC] WARN tag=%s device=%d host_alloc_call_count=%llu exceeds threshold=%ld\n",
                tag ? tag : "graph", device, (unsigned long long) s.host_alloc_call_count, warn_calls);
    }
}

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
    char * end   = nullptr;
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
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    g_copy_trace_enabled.store(enabled, std::memory_order_release);
    return enabled != 0;
}

static bool copy_to_device_sync_enabled() {
    static std::atomic<int> cached{ -1 };
    int                     enabled = cached.load(std::memory_order_acquire);
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char * env = std::getenv("GGML_SYCL_COPY_TO_DEVICE_SYNC");
    enabled          = (env && std::atoi(env) != 0) ? 1 : 0;
    cached.store(enabled, std::memory_order_release);
    return enabled != 0;
}

static size_t copy_to_device_stage_slots() {
    static std::atomic<size_t> cached{ 0 };
    size_t                     slots = cached.load(std::memory_order_acquire);
    if (slots != 0) {
        return slots;
    }
    size_t parsed = 3;
    if (const char * env = std::getenv("GGML_SYCL_COPY_TO_DEVICE_STAGE_SLOTS")) {
        parsed = static_cast<size_t>(std::max(1, std::atoi(env)));
    }
    parsed = std::min<size_t>(parsed, 16);
    cached.store(parsed, std::memory_order_release);
    return parsed;
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

static bool host_cache_check_guard_locked(const host_cache_entry &   entry,
                                          const ggml_sycl_cache_id & key,
                                          const char *               where) {
    if (entry.guard_size == 0 || entry.host_ptr == nullptr) {
        return true;
    }
    const uint8_t * guard = static_cast<const uint8_t *>(entry.host_ptr) + entry.size;
    for (size_t i = 0; i < entry.guard_size; ++i) {
        if (guard[i] != k_host_cache_guard_pattern) {
            g_host_cache_guard_errors.fetch_add(1, std::memory_order_relaxed);
            GGML_LOG_ERROR(
                "[UNIFIED-CACHE] host_cache guard corrupted at %s: model=%llu name_hash=0x%llx layout=%d size=%zu "
                "guard=%zu type=%d L%d E%d\n",
                where, (unsigned long long) key.model_id, (unsigned long long) key.name_hash, (int) entry.layout,
                entry.size, entry.guard_size, (int) entry.type, entry.layer_id, entry.expert_id);
            return false;
        }
    }
    return true;
}

static bool host_cache_check_pinned_guard_locked(const host_cache_entry &   entry,
                                                 const ggml_sycl_cache_id & key,
                                                 const char *               where) {
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
                "[UNIFIED-CACHE] pinned pool guard corrupted at %s: model=%llu name_hash=0x%llx layout=%d size=%zu "
                "guard=%zu type=%d L%d E%d\n",
                where, (unsigned long long) key.model_id, (unsigned long long) key.name_hash, (int) entry.layout,
                entry.size, entry.guard_size, (int) entry.type, entry.layer_id, entry.expert_id);
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
    g_sycl_shutting_down.store(true, std::memory_order_release);
}

bool ggml_sycl_is_shutting_down() {
    return g_sycl_shutting_down.load(std::memory_order_acquire);
}

unified_cache::unified_cache(sycl::queue & queue, size_t budget_bytes, size_t staging_size, size_t dma_reserved_bytes) :
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
                    budget_ / (1024.0f * 1024.0f), staging_size_ / (1024.0f * 1024.0f),
                    dma_reserved_bytes_ / (1024.0f * 1024.0f));

    // Pre-allocate reusable host-pinned staging slots for copy_to_device_async.
    // This eliminates per-expert sycl::malloc_host / sycl::free churn during
    // inference — each alloc/free does GGTT page table ops in the kernel driver.
    // NOTE: We use ggml_sycl_malloc_host directly (not unified_alloc) because
    // the constructor runs under g_cache_rw_mutex and unified_alloc would deadlock
    // via unified_cache_add_runtime_host_bytes -> g_cache_rw_mutex.
    {
        constexpr size_t k_fallback_chunk = 64 * 1024 * 1024;
        const size_t     slot_capacity    = staging_size_ > 0 ? staging_size_ : k_fallback_chunk;
        const size_t     n_slots          = copy_to_device_stage_slots();

        copy_stage_slots_.reserve(n_slots);
        for (size_t i = 0; i < n_slots; ++i) {
            try {
                void * ptr = ggml_sycl_malloc_host(slot_capacity, queue_, "copy_stage_slot_prealloc");
                if (ptr) {
                    copy_stage_slot slot{};
                    slot.ptr       = ptr;
                    slot.device    = -1;  // host-pinned, no device association
                    slot.capacity  = slot_capacity;
                    slot.in_flight = false;
                    copy_stage_slots_.push_back(slot);
                } else {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to pre-allocate staging slot %zu\n", i);
                }
            } catch (const sycl::exception & e) {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to pre-allocate staging slot %zu: %s\n", i, e.what());
            }
        }
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Pre-allocated %zu/%zu staging slots (%.1f MB each)\n",
                        copy_stage_slots_.size(), n_slots, slot_capacity / (1024.0f * 1024.0f));
    }

    // Initialize layout pool for consolidating layout allocations into
    // contiguous chunks (reduces GPU TLB misses from scattered USM mappings).
    layout_pool_ = std::make_unique<sycl_device_pool>(queue_, ggml_sycl_get_device_id_from_queue(queue_));

    // VRAM Arena: pre-allocate a single VRAM block when GGML_SYCL_VRAM_ARENA=1.
    if (vram_arena_enabled()) {
        const int dev_id = ggml_sycl_get_device_id_from_queue(queue_);
        size_t max_alloc = queue_.get_device().get_info<sycl::info::device::max_mem_alloc_size>();

        // Default zone sizes.  Compute arena default is 256 MB.
        size_t compute_zone = 256 * 1024 * 1024;
        const char * arena_mb_env = std::getenv("GGML_SYCL_COMPUTE_ARENA_MB");
        if (arena_mb_env) {
            compute_zone = static_cast<size_t>(std::max(0, std::atoi(arena_mb_env))) * 1024 * 1024;
        }

        // oneDNN scratch: 0 by default, sized later by reserve_onednn_scratch.
        // Pre-reserve a generous 256 MB for oneDNN to avoid later realloc.
        size_t onednn_zone = 256 * 1024 * 1024;

        if (arena_.reserve(queue_, budget_bytes, max_alloc, compute_zone, onednn_zone)) {
            // Mark all arena bytes as used in the cache budget.
            used_.fetch_add(arena_.total_size(), std::memory_order_relaxed);
            GGML_LOG_INFO("[VRAM-ARENA] Active on device %d: %d chunk(s), %.1f MB total\n",
                          dev_id, arena_.chunk_count(), arena_.total_size() / (1024.0 * 1024.0));
            // Bind layout pool to the arena so new layout allocations come from the
            // arena's weight zone instead of allocating separate chunks.
            if (layout_pool_) {
                layout_pool_->set_arena(&arena_);
            }
        } else {
            GGML_LOG_WARN("[VRAM-ARENA] Failed on device %d, falling back to per-entry allocation\n", dev_id);
        }
    }

    // Create a separate in-order DMA queue for cache operations (CCS engine).
    // This keeps cache DMA/fill work off the compute queue, preventing
    // >20s accumulated queue work that triggers L0 DirectSubmission timeouts.
    try {
        dma_queue_ =
            std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), sycl::property::queue::in_order{});
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Created separate DMA queue for cache operations\n");
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Failed to create DMA queue, falling back to compute queue: %s\n", e.what());
        dma_queue_.reset();
    }

    // Create a BCS (copy-only) queue for H2D transfers during expert prestaging.
    // On Intel GPUs, Level Zero exposes queue groups: ordinal 0 = CCS (compute+copy),
    // ordinal 1 = BCS (copy-only / blitter).  By routing H2D memcpy to a separate
    // in-order queue, the runtime can assign it to the BCS engine, keeping CCS free
    // for SOA reorder kernels.  This prevents CCS monopolization during the ~6000
    // kernel submissions of MoE expert prestaging that trigger GT engine resets.
    //
    // Even if the runtime routes both queues to CCS, having separate queues still
    // enables pipelining: H2D copies and reorder kernels interleave via event deps
    // instead of serializing on a single command list.
    try {
        bcs_queue_ =
            std::make_unique<sycl::queue>(queue_.get_context(), queue_.get_device(), sycl::property::queue::in_order{});
        GGML_LOG_INFO("[UNIFIED-CACHE] Created BCS queue for H2D copy pipelining\n");
        // Give the device pool a reference so it can drain BCS before chunk allocs.
        if (layout_pool_) {
            layout_pool_->set_bcs_queue(bcs_queue_.get());
        }
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Failed to create BCS queue, falling back to DMA queue: %s\n", e.what());
        bcs_queue_.reset();
    }

    // P7: async DMA eviction — default ON when arena is active.
    // Env var GGML_SYCL_ASYNC_EVICT=0 disables.
    {
        const char * env  = std::getenv("GGML_SYCL_ASYNC_EVICT");
        bool         dflt = arena_.active();  // Default ON when arena active
        async_evict_enabled_ = env ? (std::string(env) != "0") : dflt;
        if (async_evict_enabled_) {
            GGML_LOG_INFO("[UNIFIED-CACHE] Async DMA eviction enabled (preserves layouts during migration)\n");
        }
    }

    // Ensure unordered_map has buckets before any find() calls.
    entries_.rehash(1);
    id_to_key_.rehash(1);
}

unified_cache::~unified_cache() {
    // Stop the prefetch worker thread first (before any resource cleanup).
    // This is safe even if the SYCL runtime is shutting down since the worker
    // only does cache lookups and pinning, not SYCL memory operations.
    stop_prefetch_worker();

    // Skip cleanup if SYCL runtime is shutting down (static destruction order issue)
    // This can happen when the program exits and static destructors run in undefined order
    if (g_sycl_shutting_down.load()) {
        // Abandon pool chunks without calling sycl::free() (context is invalid)
        if (layout_pool_) {
            layout_pool_->abandon();
        }
        // Abandon arena without calling sycl::free — context is invalid.
        arena_.abandon();
        compute_arena_ptr_ = nullptr;
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
        if (layout_pool_) {
            layout_pool_->abandon();
        }
        compute_arena_ptr_ = nullptr;
        return;
    }

    // Check arena state before destroying anything.
    const bool had_arena = arena_.active();

    // Free all cached entries (skip pool-allocated and arena-owned entries)
    for (auto & pair : entries_) {
        if (pair.second.device_ptr && !pair.second.pool_allocated &&
            !(had_arena && arena_.owns(pair.second.device_ptr))) {
            try {
                sycl::free(pair.second.device_ptr, queue_);
            } catch (...) {
            }
        }
    }

    // Free compute arena BEFORE destroying the VRAM arena (which would invalidate owns check).
    if (compute_arena_ptr_ && !(had_arena && arena_.owns(compute_arena_ptr_))) {
        try {
            sycl::free(compute_arena_ptr_, queue_);
        } catch (...) {}
        saturating_sub_used(compute_arena_size_);
    }
    compute_arena_ptr_  = nullptr;
    compute_arena_size_ = 0;
    compute_arena_off_.store(0, std::memory_order_relaxed);

    // Free scratch pool BEFORE destroying the VRAM arena (which would invalidate owns check).
    if (scratch_pool_ptr_ && !(had_arena && arena_.owns(scratch_pool_ptr_))) {
        try {
            sycl::free(scratch_pool_ptr_, queue_);
        } catch (...) {}
        saturating_sub_used(scratch_pool_size_);
    }
    scratch_pool_ptr_  = nullptr;
    scratch_pool_size_ = 0;
    scratch_pool_off_.store(0, std::memory_order_relaxed);

    // Destroy the VRAM arena (frees the pre-allocated chunks).
    arena_.destroy();

    // Destroy layout pool before SYCL context goes away.
    // The pool's reset() returns physical bytes freed so we can decrement used_.
    if (layout_pool_) {
        const size_t pool_freed = layout_pool_->reset();
        if (pool_freed > 0 && used_.load(std::memory_order_relaxed) >= pool_freed) {
            used_.fetch_sub(pool_freed, std::memory_order_relaxed);
        }
        layout_pool_.reset();
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

    // Free reusable host-pinned async copy staging slots.
    // These were allocated with ggml_sycl_malloc_host (not unified_alloc) to
    // avoid g_cache_rw_mutex deadlock during construction, so free via sycl::free.
    for (auto & slot : copy_stage_slots_) {
        if (slot.ptr != nullptr) {
            try {
                sycl::free(slot.ptr, queue_);
            } catch (...) {
            }
            slot.ptr = nullptr;
        }
    }
    copy_stage_slots_.clear();

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

    // Free oneDNN scratch buffers
    if (onednn_weights_scratch_) {
        try {
            sycl::free(onednn_weights_scratch_, queue_);
        } catch (...) {
        }
        onednn_weights_scratch_ = nullptr;
    }
    if (onednn_activations_scratch_) {
        try {
            sycl::free(onednn_activations_scratch_, queue_);
        } catch (...) {
        }
        onednn_activations_scratch_ = nullptr;
    }

    // Free reorder temp buffer
    if (reorder_temp_buffer_) {
        alloc_registry::instance().unregister_alloc(reorder_temp_buffer_);
        saturating_sub_used(reorder_temp_size_);
        try {
            sycl::free(reorder_temp_buffer_, queue_);
        } catch (...) {
        }
        reorder_temp_buffer_ = nullptr;
        reorder_temp_size_   = 0;
    }

    // Free persistent scratch buffers
    for (auto & pair : persistent_scratches_) {
        if (pair.second.device_ptr) {
            try {
                sycl::free(pair.second.device_ptr, queue_);
            } catch (...) {
            }
        }
    }
    persistent_scratches_.clear();

    // Free partial row entries (multi-device tensor split)
    for (auto & pair : partial_cache_) {
        if (pair.second.ptr) {
            try {
                sycl::free(pair.second.ptr, queue_);
            } catch (...) {
            }
        }
    }
    partial_cache_.clear();
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
        const sycl::usm::alloc alloc = ggml_sycl_get_alloc_type(ptr);
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
    queue_(queue),
    budget_(budget_bytes),
    base_budget_(budget_bytes) {
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DEBUG: host_cache constructor started\n");
    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(unified_cache_atexit_handler);
    }

    // Create pinned pool with capped budget.
    // Without the cap, the pool inherits the full host memory budget (~227 GB)
    // and grows unboundedly during MoE warmup profiling (36 layers × 538 MB
    // = 19.4 GB of pinned host memory).  Cap at 4 GB — enough for working set
    // of 2-3 layers of expert staging plus CPU dispatch buffers.
    const size_t pinned_cap = size_t(16) << 30;  // 16 GB
    const size_t pinned_budget = std::min(budget_bytes, pinned_cap);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DEBUG: Creating pinned pool\n");
    pinned_pool_ = std::make_unique<pinned_chunk_pool>(queue_, pinned_budget);
    GGML_LOG_INFO("[SYCL] Pinned chunk pool created with %.1f GB budget\n",
                  pinned_budget / (1024.0 * 1024.0 * 1024.0));

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
        GGML_LOG_INFO(
            "[UNIFIED-CACHE] Host reserve %.1f MB >= base budget %.1f MB; host cache budget now 0 (used %.1f MB)\n",
            reserved_ / (1024.0f * 1024.0f), base_budget_ / (1024.0f * 1024.0f), used_.load() / (1024.0f * 1024.0f));
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
                        used / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f), reserved_ / (1024.0f * 1024.0f));
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

bool host_cache::contains_pinned(const void * ptr) const {
    if (!pinned_pool_ || !ptr) {
        return false;
    }
    return pinned_pool_->contains(ptr);
}

size_t host_cache::pre_allocate_pinned(size_t total_bytes) {
    if (!pinned_pool_) {
        return 0;
    }
    return pinned_pool_->pre_allocate(total_bytes);
}

size_t host_cache::pre_allocate_all(size_t model_weight_bytes) {
    if (!pinned_pool_) {
        return 0;
    }
    return pinned_pool_->pre_allocate_all(model_weight_bytes);
}

void * host_cache::ensure_cached_alloc(const ggml_sycl_cache_id &    key_id,
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
    const bool     can_hash        = validate_content && host_accessible;
    const uint64_t new_hash        = can_hash ? compute_content_hash(src_ptr, src_size) : 0;
    // Always allow aliasing for host-accessible AOS entries with matching sizes.
    // The unified non-blocking cache handles all model sizes without branching
    // on model_exceeds_vram.  Pinning of host cache entries is handled separately.
    const bool     can_alias       = host_accessible && layout == GGML_LAYOUT_AOS && src_size == dst_size;
    const bool     prefer_unpinned = host_cache_prefer_unpinned(type);

    std::lock_guard<std::mutex> lock(mutex_);
    const ggml_sycl_cache_id &  key_id_ref = key_id;

    unified_cache_key key{ type, key_id_ref, layer_id, expert_id };
    auto              it = entries_.find(key);
    if (it != entries_.end() && it->second.layout != layout) {
        if (it->second.pinned) {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] host_cache layout switch: unpinning model=%llu name_hash=0x%llx have=%d "
                "want=%d\n",
                (unsigned long long) key_id_ref.model_id, (unsigned long long) key_id_ref.name_hash,
                (int) it->second.layout, (int) layout);
            it->second.pinned = false;
        }
        free_entry(it->second);
        entries_.erase(it);
        it = entries_.end();
    }
    if (it != entries_.end()) {
        auto & entry = it->second;
        if (!host_cache_check_guard_locked(entry, key_id_ref, "ensure_cached_alloc") ||
            !host_cache_check_pinned_guard_locked(entry, key_id_ref, "ensure_cached_alloc")) {
            if (cache_assert_enabled()) {
                GGML_ABORT("host_cache guard corruption detected");
            }
            return nullptr;
        }
        bool size_changed    = (dst_size != entry.size);
        bool content_changed = validate_content && can_hash && (entry.content_hash != new_hash);
        bool src_changed     = entry.src_ptr != src_ptr;
        bool needs_realloc   = size_changed;
        bool needs_refill    = needs_realloc || src_changed || content_changed;

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
            size_t alloc_size   = dst_size;
            size_t guard_size   = 0;
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
                        "[HOST-CACHE] alias reuse: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p pinned=%d "
                        "owns=%d loc=%d\n",
                        (unsigned long long) key_id_ref.model_id, (unsigned long long) key_id_ref.name_hash,
                        (int) layout, entry.size, entry.host_ptr, entry.pinned_alloc ? 1 : 0, entry.owns_ptr ? 1 : 0,
                        (int) entry.location);
                }
                return entry.host_ptr;
            }

            if (pinned_pool_ && !prefer_unpinned) {
                new_ptr      = pinned_pool_->allocate(alloc_size);
                pooled_alloc = (new_ptr != nullptr);
            }

            if (!new_ptr) {
                new_ptr      = host_cache_alloc_unpinned(alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
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
                            "[HOST-CACHE] alias reuse: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p pinned=%d "
                            "owns=%d loc=%d\n",
                            (unsigned long long) key_id_ref.model_id, (unsigned long long) key_id_ref.name_hash,
                            (int) layout, entry.size, entry.host_ptr, entry.pinned_alloc ? 1 : 0,
                            entry.owns_ptr ? 1 : 0, (int) entry.location);
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
            used_.fetch_add(dst_size, std::memory_order_relaxed);
            if (g_ggml_sycl_debug >= 2) {
                GGML_SYCL_DEBUG(
                    "[HOST-CACHE] realloc: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p guard=%zu pinned=%d "
                    "owns=%d loc=%d\n",
                    (unsigned long long) key_id_ref.model_id, (unsigned long long) key_id_ref.name_hash, (int) layout,
                    entry.size, entry.host_ptr, entry.guard_size, entry.pinned_alloc ? 1 : 0, entry.owns_ptr ? 1 : 0,
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
                "[HOST-CACHE] alias insert: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p pinned=%d owns=%d "
                "loc=%d\n",
                (unsigned long long) key_id_ref.model_id, (unsigned long long) key_id_ref.name_hash, (int) layout,
                entry.size, entry.host_ptr, entry.pinned_alloc ? 1 : 0, entry.owns_ptr ? 1 : 0, (int) entry.location);
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

    void * host_ptr     = nullptr;
    bool   pooled_alloc = false;
    size_t alloc_size   = dst_size;
    size_t guard_size   = 0;
    if (host_cache_guard_enabled()) {
        guard_size = k_host_cache_guard_bytes;
        alloc_size += guard_size;
    }
    if (pinned_pool_ && !prefer_unpinned) {
        host_ptr     = pinned_pool_->allocate(alloc_size);
        pooled_alloc = (host_ptr != nullptr);
    }
    if (!host_ptr) {
        host_ptr     = host_cache_alloc_unpinned(alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
        pooled_alloc = false;
    }
    if (!host_ptr) {
        // Last resort: alias the source pointer directly (mmap or heap).
        // For AOS layout (no reorder needed), the source data can be read as-is by CPU
        // dispatch.  Allow aliasing even for non-USM mmap pointers when layout is AOS
        // and sizes match — the pointer is only used on the CPU side.
        // Allow mmap alias when layout is AOS and source data is at least as large as
        // needed.  Padded dst_size > src_size is safe because AOS padding bytes at the
        // end of a weight tensor are never accessed by compute kernels (the kernel
        // operates on aligned blocks within the valid data range).
        const bool can_mmap_alias = (layout == GGML_LAYOUT_AOS && src_ptr &&
                                     src_size > 0 && src_size <= dst_size);
        if (can_alias || can_mmap_alias) {
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
                    "[HOST-CACHE] alias insert: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p pinned=%d "
                    "owns=%d loc=%d\n",
                    (unsigned long long) key_id_ref.model_id, (unsigned long long) key_id_ref.name_hash, (int) layout,
                    entry.size, entry.host_ptr, entry.pinned_alloc ? 1 : 0, entry.owns_ptr ? 1 : 0,
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
    used_.fetch_add(dst_size, std::memory_order_relaxed);

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
            "[HOST-CACHE] alloc: model=%llu name_hash=0x%llx layout=%d size=%zu ptr=%p guard=%zu pinned=%d owns=%d "
            "loc=%d\n",
            (unsigned long long) key_id_ref.model_id, (unsigned long long) key_id_ref.name_hash, (int) layout,
            entry.size, entry.host_ptr, entry.guard_size, entry.pinned_alloc ? 1 : 0, entry.owns_ptr ? 1 : 0,
            (int) entry.location);
    }
    return host_ptr;
}

bool host_cache::is_cached(const ggml_sycl_cache_id & key_id,
                           cache_entry_type           type,
                           int                        layer_id,
                           int                        expert_id,
                           ggml_layout_mode           layout) const {
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
                       cache_entry_type           type,
                       int                        layer_id,
                       int                        expert_id,
                       ggml_layout_mode           layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return nullptr;
    }
    if (entries_.bucket_count() == 0) {
        entries_.rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto              entry_it = entries_.find(key);
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
                                        cache_entry_type           type,
                                        int                        layer_id,
                                        int                        expert_id,
                                        ggml_layout_mode           layout) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return cache_location::HOST_MMAP;
    }
    if (entries_.bucket_count() == 0) {
        const_cast<decltype(entries_) &>(entries_).rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto              entry_it = entries_.find(key);
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
                             cache_entry_type           type,
                             int                        layer_id,
                             int                        expert_id,
                             ggml_layout_mode           layout) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return true;
    }
    if (entries_.bucket_count() == 0) {
        const_cast<decltype(entries_) &>(entries_).rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto              entry_it = entries_.find(key);
    if (entry_it == entries_.end()) {
        return true;
    }
    if (entry_it->second.layout != layout) {
        GGML_LOG_ERROR(
            "[UNIFIED-CACHE] host_cache layout mismatch in check_guard model=%llu name_hash=0x%llx have=%d want=%d\n",
            (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) entry_it->second.layout,
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
    const char *                tag = (where && where[0]) ? where : "check_all_guards";
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
                        cache_entry_type           type,
                        int                        layer_id,
                        int                        expert_id,
                        ggml_layout_mode           layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return;
    }
    if (entries_.bucket_count() == 0) {
        entries_.rehash(1);
        GGML_LOG_WARN("[UNIFIED-CACHE] host_cache entries_ had zero buckets; rehashing\n");
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto              it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }
    if (it->second.layout != layout) {
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] host_cache remove layout mismatch model=%llu name_hash=0x%llx have=%d want=%d (removing "
            "cached)\n",
            (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) it->second.layout,
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

bool host_cache::adopt_evicted(const ggml_sycl_cache_id &    key_id,
                               void *                        host_ptr,
                               size_t                        size,
                               cache_entry_type              type,
                               int                           layer_id,
                               int                           expert_id,
                               ggml_layout_mode              layout,
                               const cache_layout_xmx_info * xmx_info) {
    if (!key_id.valid || !host_ptr || size == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.bucket_count() == 0) {
        entries_.rehash(1);
    }

    unified_cache_key key{ type, key_id, layer_id, expert_id };

    // Remove any existing entry for this key
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        free_entry(it->second);
        entries_.erase(it);
    }

    // Evict host entries if needed to make room
    while (used_.load() + size > budget_) {
        if (evict_one() == 0) {
            GGML_LOG_WARN("[UNIFIED-CACHE] host_cache adopt_evicted: cannot make room for %zu bytes\n", size);
            return false;
        }
    }

    host_cache_entry entry{};
    entry.host_ptr     = host_ptr;
    entry.src_ptr      = nullptr;  // No original source — this is preserved device data
    entry.content_hash = 0;
    entry.size         = size;
    entry.type         = type;
    entry.layer_id     = layer_id;
    entry.expert_id    = expert_id;
    entry.layout       = layout;
    entry.access_count = 1;
    entry.last_access  = time_++;
    entry.pinned       = false;
    entry.owns_ptr     = true;
    entry.pinned_alloc = true;   // Was allocated via sycl::malloc_host
    entry.location     = cache_location::HOST_PINNED;
    if (xmx_info) {
        entry.xmx_info = *xmx_info;
    }

    entries_.emplace(key, std::move(entry));
    used_.fetch_add(size, std::memory_order_relaxed);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] host_cache adopted evicted: model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                    (int) layout, size);
    return true;
}

void host_cache::pin(const ggml_sycl_cache_id & key_id,
                     cache_entry_type           type,
                     int                        layer_id,
                     int                        expert_id,
                     ggml_layout_mode           layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return;
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto              entry_it = entries_.find(key);
    if (entry_it != entries_.end() && entry_it->second.layout == layout) {
        entry_it->second.pinned = true;
    }
}

void host_cache::unpin(const ggml_sycl_cache_id & key_id,
                       cache_entry_type           type,
                       int                        layer_id,
                       int                        expert_id,
                       ggml_layout_mode           layout) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key_id.valid) {
        return;
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };
    auto              entry_it = entries_.find(key);
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

    // Priority-based score boost (consistent with unified_cache::compute_score)
    const alloc_category cat = (entry.type == cache_entry_type::MOE_EXPERT)
                                   ? alloc_category::EXPERT_CACHE
                                   : alloc_category::WEIGHT;
    constexpr int k_max_priority = 4;
    const float   priority_boost = static_cast<float>(k_max_priority - alloc_category_priority(cat) + 1);
    base_score *= priority_boost;

    if (entry.type == cache_entry_type::DENSE_WEIGHT) {
        return base_score;
    }
    // Boost MoE experts with high popularity (low rank = more popular).
    // This makes popular experts resist eviction after warmup profiling.
    if (entry.type == cache_entry_type::MOE_EXPERT && entry.layer_id >= 0 && entry.expert_id >= 0) {
        if (is_expert_popularity_initialized()) {
            int pop_rank = get_expert_popularity_rank(entry.layer_id, entry.expert_id);
            if (pop_rank >= 0) {
                // Top experts (rank 0-3) get 4x-1x boost; rank 4+ get no boost
                int boost_slots = 4;
                if (pop_rank < boost_slots) {
                    float boost = static_cast<float>(boost_slots - pop_rank);
                    base_score *= (1.0f + boost);
                }
            }
        }
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
        saturating_sub_used(entry.size);
    }
    entry.host_ptr   = nullptr;
    entry.size       = 0;
    entry.guard_size = 0;
}

void * unified_cache::ensure_cached(const ggml_sycl_cache_id & key_id,
                                    const void *               src_ptr,
                                    size_t                     size,
                                    cache_entry_type           type,
                                    int                        layer_id,
                                    int                        expert_id,
                                    ggml_layout_mode           layout,
                                    bool                       validate_content) {
    if (!key_id.valid || !src_ptr || size == 0) {
        return nullptr;
    }

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    // Only process deferred frees when no GPU kernels are in-flight.
    // During graph_compute (MoE inference), freed VRAM pages may still be
    // referenced by earlier MUL_MAT_ID kernels — processing frees here
    // causes GPU page faults (DEVICE_LOST).
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }

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
                           (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache id_to_key mismatch");
            }
        }
        if (it->second.layout != layout) {
            GGML_LOG_ERROR(
                "[UNIFIED-CACHE] layout mismatch in ensure_cached model=%llu name_hash=0x%llx have=%d want=%d\n",
                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) it->second.layout,
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
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, it->second.size, size);

                const bool   was_pinned          = it->second.pinned;
                const size_t old_size            = it->second.size;
                const bool   was_device_resident = !it->second.host_resident;
                bool         use_host_fallback = false;
                it->second.pinned              = true;
                while (used_.load() - old_size + size > budget_) {
                    if (evict_one(size) == 0) {
                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] Cannot evict for realloc (used=%.1f MB, need=%.1f MB), trying host "
                            "fallback\n",
                            used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
                        use_host_fallback = true;
                        break;
                    }
                }

                // Allocate new buffer with correct size
                void *         new_device_ptr   = nullptr;
                bool           is_host_resident = false;
                cache_location new_location     = cache_location::DEVICE;

                if (!use_host_fallback) {
                    try {
                        new_device_ptr = ggml_sycl_malloc_device_raw(size, queue_, "unified_cache:realloc");
                    } catch (const sycl::exception & e) {
                        GGML_SYCL_DEBUG("[UNIFIED-CACHE] realloc malloc_device failed: %s, trying host fallback\n",
                                        e.what());
                        use_host_fallback = true;
                    }

                    if (!new_device_ptr && !use_host_fallback) {
                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] realloc malloc_device returned nullptr, trying host fallback\n");
                        use_host_fallback = true;
                    }
                }

                // Host fallback for realloc
                if (use_host_fallback) {
                    host_cache * hcache = get_host_cache(queue_);
                    if (hcache) {
                        bool           needs_host_fill = false;
                        bool           pinned_alloc    = false;
                        cache_location host_loc        = cache_location::HOST_MMAP;
                        void *         host_ptr        = hcache->ensure_cached_alloc(
                            key_id, src_ptr, size, size, type, layer_id, expert_id, layout, validate_content,
                            &needs_host_fill, &pinned_alloc, &host_loc, nullptr);

                        if (host_ptr) {
                            if (needs_host_fill) {
                                std::memcpy(host_ptr, src_ptr, size);
                            }

                            GGML_SYCL_DEBUG(
                                "[UNIFIED-CACHE] Realloc to host-resident for model=%llu name_hash=0x%llx (%.2f MB)\n",
                                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                                size / (1024.0f * 1024.0f));

                            new_device_ptr   = host_ptr;
                            is_host_resident = true;
                            new_location     = host_loc;
                        }
                    }

                    if (!new_device_ptr) {
                        GGML_LOG_ERROR("[UNIFIED-CACHE] Both device and host realloc failed\n");
                        it->second.pinned = was_pinned;
                        return nullptr;
                    }
                }

                // Release old buffer after new allocation succeeds (only if it was on device)
                if (!it->second.host_resident && it->second.device_ptr) {
                    enqueue_deferred_free(it->second.device_ptr, it->second.size);
                    // Device pointer freed — baked graph pointers to this entry are now stale
                    has_evictions_.store(true, std::memory_order_release);
                }

                // Copy new data (only if on device, host_cache already filled host buffer)
                if (!is_host_resident) {
                    copy_to_device(new_device_ptr, src_ptr, size).wait();
                }

                // Update entry with new allocation
                it->second.device_ptr    = new_device_ptr;
                it->second.size          = size;
                it->second.content_hash  = new_hash;
                it->second.src_ptr       = src_ptr;
                it->second.host_resident = is_host_resident;
                it->second.location      = new_location;
                if (!is_host_resident) {
                    if (size > old_size) {
                        used_.fetch_add(size - old_size, std::memory_order_relaxed);
                    } else if (old_size > size) {
                        saturating_sub_used(old_size - size);
                    }
                } else if (was_device_resident) {
                    // Migrated from device to host, reduce device usage
                    saturating_sub_used(old_size);
                }
                it->second.pinned = was_pinned;
            } else if (content_changed) {
                // Same size but content changed - just re-upload
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] Content changed for model=%llu name_hash=0x%llx (hash %llx -> %llx), "
                    "re-uploading\n",
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                    (unsigned long long) it->second.content_hash, (unsigned long long) new_hash);
                copy_to_device(it->second.device_ptr, src_ptr, size).wait();
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
    bool use_host_fallback = false;
    while (used_.load() + size > budget_) {
        // Need to evict
        if (evict_one(size) == 0) {
            // All entries pinned, cannot evict - try host fallback
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] Cannot evict: all entries pinned (used=%.1f MB, need=%.1f MB), trying host fallback\n",
                used_.load() / (1024.0f * 1024.0f), size / (1024.0f * 1024.0f));
            use_host_fallback = true;
            break;
        }
    }

    // Allocate device memory (unless we need host fallback)
    void *         device_ptr       = nullptr;
    bool           is_host_resident = false;
    cache_location entry_location   = cache_location::DEVICE;

    if (!use_host_fallback) {
        try {
            device_ptr = ggml_sycl_malloc_device_raw(size, queue_, "unified_cache:alloc");
        } catch (const sycl::exception & e) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] malloc_device failed: %s, trying host fallback\n", e.what());
            use_host_fallback = true;
        }

        if (!device_ptr && !use_host_fallback) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] malloc_device returned nullptr, trying host fallback\n");
            use_host_fallback = true;
        }
    }

    // Host fallback when device allocation fails or eviction is impossible
    if (use_host_fallback) {
        host_cache * hcache = get_host_cache(queue_);
        if (hcache) {
            // For simple ensure_cached, src_size == dst_size == size
            bool           needs_host_fill = false;
            bool           pinned_alloc    = false;
            cache_location host_loc        = cache_location::HOST_MMAP;
            void *         host_ptr =
                hcache->ensure_cached_alloc(key_id, src_ptr, size, size, type, layer_id, expert_id, layout,
                                            validate_content, &needs_host_fill, &pinned_alloc, &host_loc, nullptr);

            if (host_ptr) {
                // Fill host buffer if needed (synchronous since host memory)
                if (needs_host_fill) {
                    std::memcpy(host_ptr, src_ptr, size);
                }

                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] Device full, using host-resident for model=%llu name_hash=0x%llx (%.2f MB)\n",
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                    size / (1024.0f * 1024.0f));

                device_ptr       = host_ptr;
                is_host_resident = true;
                entry_location   = host_loc;
            }
        }

        if (!device_ptr) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] Both device and host allocation failed\n");
            return nullptr;
        }
    } else {
        // Copy data from source to device
        copy_to_device(device_ptr, src_ptr, size).wait();
    }

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
    entry.host_resident   = is_host_resident;
    entry.location        = entry_location;
    // NOTE: Reorder state is tracked in tensor->extra->optimized_feature, not here

    // Store in cache
    entries_[key] = entry;
    auto id_it    = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        id_to_key_.emplace(key_id, key);
    } else if (!(id_it->second == key)) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision on insert model=%llu name_hash=0x%llx\n",
                       (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache id_to_key mismatch");
        }
    }

    // Only track device memory usage, not host-resident entries
    if (!is_host_resident) {
        used_.fetch_add(size, std::memory_order_relaxed);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cached %s%s: %.2f MB (used=%.1f/%.1f MB)\n",
                    type == cache_entry_type::DENSE_WEIGHT ? "dense" : "expert",
                    is_host_resident ? " (host-resident)" : "", size / (1024.0f * 1024.0f),
                    used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f));

    return device_ptr;
}

void * unified_cache::ensure_cached_alloc(const ggml_sycl_cache_id & key_id,
                                          const void *               src_ptr,
                                          size_t                     src_size,
                                          size_t                     alloc_size,
                                          cache_entry_type           type,
                                          int                        layer_id,
                                          int                        expert_id,
                                          ggml_layout_mode           layout,
                                          bool                       validate_content,
                                          bool *                     needs_fill) {
    if (needs_fill) {
        *needs_fill = true;
    }
    if (!key_id.valid || !src_ptr || src_size == 0 || alloc_size == 0) {
        return nullptr;
    }

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }

    unified_cache_key key{ type, key_id, layer_id, expert_id };
    const uint64_t    new_hash = compute_content_hash(src_ptr, src_size);

    auto it = entries_.find(key);
    if (it != entries_.end()) {
        auto id_it = id_to_key_.find(key_id);
        if (id_it == id_to_key_.end()) {
            id_to_key_.emplace(key_id, key);
        } else if (!(id_it->second == key)) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision in ensure_cached_alloc model=%llu name_hash=0x%llx\n",
                           (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache id_to_key mismatch");
            }
        }
        if (it->second.layout != layout) {
            if (it->second.pinned) {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout switch: unpinning model=%llu name_hash=0x%llx have=%d want=%d\n",
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                    (int) it->second.layout, (int) layout);
                it->second.pinned = false;
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
                    new_device_ptr = ggml_sycl_malloc_device_raw(alloc_size, queue_, "unified_cache:alloc");
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
                if (alloc_size > old_size) {
                    used_.fetch_add(alloc_size - old_size, std::memory_order_relaxed);
                } else if (old_size > alloc_size) {
                    saturating_sub_used(old_size - alloc_size);
                }
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
        device_ptr = ggml_sycl_malloc_device_raw(alloc_size, queue_, "unified_cache:alloc");
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
    auto id_it    = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        id_to_key_.emplace(key_id, key);
    } else if (!(id_it->second == key)) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] identity collision on insert model=%llu name_hash=0x%llx\n",
                       (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache id_to_key mismatch");
        }
    }
    used_.fetch_add(alloc_size, std::memory_order_relaxed);

    if (needs_fill) {
        *needs_fill = true;
    }

    return device_ptr;
}

cache_layout_result unified_cache::ensure_cached_layout(const cache_layout_request &     request,
                                                        const std::vector<sycl::event> & deps,
                                                        sycl::queue * override_queue,
                                                        bool non_blocking) {
    cache_layout_result result{};
    result.layout        = request.layout;
    result.onednn_pack_m = request.onednn_pack_m;
    result.xmx_info      = request.xmx_info;

    if (!request.key.valid || !request.src_ptr || request.src_size == 0 || request.dst_size == 0) {
        result.status = cache_layout_status::INVALID;
        return result;
    }
    if (request.dst_size < request.src_size) {
        GGML_LOG_ERROR(
            "[UNIFIED-CACHE] invalid size: dst_size(%zu) < src_size(%zu) model=%llu name_hash=0x%llx layout=%d type=%d "
            "layer=%d expert=%d\n",
            request.dst_size, request.src_size, (unsigned long long) request.key.model_id,
            (unsigned long long) request.key.name_hash, (int) request.layout, (int) request.type, request.layer_id,
            request.expert_id);
        GGML_ASSERT(false && "cache layout dst_size < src_size");
    }

    if (g_ggml_sycl_debug) {
        sycl::usm::alloc alloc = sycl::usm::alloc::unknown;
        try {
            alloc = ggml_sycl_get_alloc_type(request.src_ptr);
        } catch (...) {
        }
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] layout request model=%llu name_hash=0x%llx type=%d layer=%d expert=%d layout=%d src=%p "
            "(%s) src_size=%zu "
            "dst_size=%zu used=%.1f MB budget=%.1f MB base=%.1f MB reserved=%.1f MB avail=%.1f MB\n",
            (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
            static_cast<int>(request.type), request.layer_id, request.expert_id, static_cast<int>(request.layout),
            request.src_ptr, usm_alloc_name(alloc), request.src_size, request.dst_size,
            used_.load() / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f), base_budget_ / (1024.0f * 1024.0f),
            reserved_ / (1024.0f * 1024.0f), available() / (1024.0f * 1024.0f));
    }

    const unified_cache_key key{ request.type, request.key, request.layer_id, request.expert_id };

    // Fast path: try read-only lookup first to avoid mutex contention
    // This uses shared_lock internally, allowing concurrent readers
    if (!ggml_sycl_graph_recording_active()) {
        bool entry_found = false;
        {
            std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
            auto                                id_it = id_to_key_.find(request.key);
            if (id_it != id_to_key_.end()) {
                entry_found       = true;
                auto entry_it = entries_.find(id_it->second);
                if (entry_it != entries_.end()) {
                    const auto & entry = entry_it->second;
                    // Fast path: entry exists, is READY, and layout matches
                    if (entry.state == cache_entry_state::READY && entry.layout == request.layout &&
                        entry.size == request.dst_size && !onednn_pack_m_mismatch(entry, request)) {
                        // Update hit count atomically (acceptable perf tradeoff vs full LRU update)
                        hits_++;
                        result.device_ptr    = entry.device_ptr;
                        result.size          = entry.size;
                        result.status        = cache_layout_status::READY;
                        result.host_resident = entry.host_resident;
                        result.location      = entry.location;
                        result.onednn_pack_m = entry.onednn_pack_m;
                        result.event         = submit_barrier(deps);
                        return result;
                    }
                    // Non-blocking: IN_PROGRESS entry with matching layout — return pointer + event
                    if (non_blocking && entry.state == cache_entry_state::IN_PROGRESS &&
                        entry.layout == request.layout) {
                        result.device_ptr    = entry.device_ptr;
                        result.size          = entry.size;
                        result.status        = cache_layout_status::IN_PROGRESS;
                        result.host_resident = entry.host_resident;
                        result.location      = entry.location;
                        result.onednn_pack_m = entry.onednn_pack_m;
                        std::vector<sycl::event> combined_deps = deps;
                        if (entry.has_ready_event) {
                            combined_deps.push_back(entry.ready_event);
                        }
                        result.event = submit_barrier(combined_deps);
                        return result;
                    }
                    // Entry exists but layout/size mismatch. Fall through to exclusive lock
                    // which handles layout conversion (evict old, allocate new).
                }
            }
        }
        // Non-blocking: entry truly not found in cache — return FAILED immediately.
        // Don't allocate, don't fill, don't block. Caller falls back to host-pinned.
        // When the entry exists (even with wrong layout), allow fall-through to the
        // exclusive lock path for layout conversion.
        if (non_blocking && !entry_found) {
            result.status = cache_layout_status::FAILED;
            return result;
        }
    }
    // Fall through to slow path with exclusive lock

    if (ggml_sycl_graph_recording_active()) {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        auto                                it = entries_.find(key);
        if (it != entries_.end() && it->second.state == cache_entry_state::READY &&
            it->second.size == request.dst_size) {
            if (it->second.layout != request.layout || onednn_pack_m_mismatch(it->second, request)) {
                GGML_LOG_ERROR(
                    "[UNIFIED-CACHE] layout mismatch in graph mode model=%llu name_hash=0x%llx have=%d want=%d\n",
                    (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                    (int) it->second.layout, (int) request.layout);
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
    const bool can_hash = request.validate_content && is_host_accessible_ptr(request.src_ptr, queue_);
    uint64_t   new_hash = can_hash ? compute_content_hash(request.src_ptr, request.src_size) : 0;

    auto try_host_fallback = [&](const char * reason) -> bool {
        // Skip host fallback for SOA layout requests — SOA is only useful on GPU.
        // CPU reads AOS directly from the original host-pinned buffer.
        // Without this check, falling back to host creates SOA copies of ALL
        // experts in pinned memory (~100 GB for 120B), wasting memory and time.
        if (request.layout == GGML_LAYOUT_SOA) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] skipping host fallback for SOA layout "
                            "(CPU reads AOS directly): %s\n", reason);
            return false;
        }
        host_cache * hcache = get_host_cache(queue_);
        if (!hcache) {
            return false;
        }
        cache_location host_loc =
            hcache->get_location(request.key, request.type, request.layer_id, request.expert_id, request.layout);
        void * host_ptr = hcache->get(request.key, request.type, request.layer_id, request.expert_id, request.layout);
        if (!host_ptr) {
            bool needs_host_fill = false;
            bool pinned_alloc    = false;
            host_ptr =
                hcache->ensure_cached_alloc(request.key, request.src_ptr, request.src_size, request.dst_size,
                                            request.type, request.layer_id, request.expert_id, request.layout, false,
                                            &needs_host_fill, &pinned_alloc, &host_loc, &request.xmx_info);
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
            std::unique_lock<std::shared_mutex> lock(rw_mutex_);
            auto                                it = entries_.find(key);
            if (it != entries_.end()) {
                if (!it->second.host_resident && it->second.device_ptr) {
                    if (!it->second.pool_allocated) {
                        // Use deferred free to avoid queue_.wait() under rw_mutex_
                        // which can deadlock when GPU work is in-flight.
                        enqueue_deferred_free(it->second.device_ptr, it->second.size);
                    }
                    // Pool entries: used_ stays at chunk level
                }
                it->second.device_ptr    = host_ptr;
                it->second.src_ptr       = request.src_ptr;
                it->second.content_hash  = can_hash ? new_hash : 0;
                it->second.size          = request.dst_size;
                it->second.type          = request.type;
                it->second.layer_id      = request.layer_id;
                it->second.expert_id     = request.expert_id;
                it->second.layout        = request.layout;
                it->second.onednn_pack_m = request.onednn_pack_m;
                it->second.xmx_info      = request.xmx_info;
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
                auto id_it            = id_to_key_.find(request.key);
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
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Host fallback (%s): model=%llu name_hash=0x%llx layout=%d size=%.1f MB\n",
                        reason ? reason : "unknown", (unsigned long long) request.key.model_id,
                        (unsigned long long) request.key.name_hash, (int) request.layout,
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
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        if (!g_graph_compute_active.load(std::memory_order_acquire)) {
            process_deferred_frees();
        }

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
                GGML_LOG_ERROR(
                    "[UNIFIED-CACHE] identity collision in ensure_cached_layout model=%llu name_hash=0x%llx\n",
                    (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash);
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
            auto &     entry           = it->second;
            const bool layout_mismatch = entry.layout != request.layout || onednn_pack_m_mismatch(entry, request);
            if (layout_mismatch) {
                if (entry.state == cache_entry_state::IN_PROGRESS) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] layout switch blocked (in-progress) model=%llu name_hash=0x%llx "
                        "have=%d want=%d\n",
                        (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                        (int) entry.layout, (int) request.layout);
                    result.status = cache_layout_status::FAILED;
                    return result;
                }
                if (entry.pinned) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] layout switch: unpinning model=%llu name_hash=0x%llx have=%d want=%d\n",
                        (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                        (int) entry.layout, (int) request.layout);
                    entry.pinned = false;
                }
                void * stale_ptr      = entry.device_ptr;
                size_t stale_size     = entry.size;
                bool   stale_host_res = entry.host_resident;
                bool   stale_pool     = entry.pool_allocated;
                if (stale_pool) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] layout switch (pool): model=%llu name_hash=0x%llx "
                        "have=%d want=%d size=%zu (abandoned in pool)\n",
                        (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                        (int) entry.layout, (int) request.layout, stale_size);
                }
                entries_.erase(it);
                it = entries_.end();
                if (!stale_host_res && stale_ptr && stale_size > 0) {
                    if (!stale_pool) {
                        // Always use deferred free for layout switches during inference.
                        // The previous S1-mode immediate free (queue_.wait() + sycl::free)
                        // caused a hang on 120B MoE models: the queue_.wait() blocks under
                        // the rw_mutex_ lock waiting for in-flight MoE kernels to complete,
                        // but those kernels may take very long (MXFP4 on host-pinned memory
                        // over PCIe) or never complete if the GPU is stalled.
                        // Deferred free is safe: the stale pointer won't be reused until
                        // process_deferred_frees() runs at the next ensure_cached_layout
                        // entry, by which point the GPU queue has drained.
                        enqueue_deferred_free(stale_ptr, stale_size);
                    }
                    // Pool entries: sub-allocation abandoned (dead space), chunk stays
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
                    (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                    (int) request.layout, entry.size, entry.has_ready_event ? 1 : 0);
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
                    (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                    (int) request.layout);
                // Memory was already freed and used_ decremented in the exception handler.
                // If device_ptr is still set (shouldn't happen), try to free it.
                if (entry.device_ptr) {
                    if (!entry.pool_allocated) {
                        try {
                            sycl::free(entry.device_ptr, queue_);
                        } catch (...) {
                            // Ignore - may leak memory but avoid crash
                        }
                        saturating_sub_used(entry.size);
                    }
                    // Pool entries: used_ stays at chunk level
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
                        "[UNIFIED-CACHE] layout size mismatch: unpinning model=%llu name_hash=0x%llx layout=%d "
                        "cached=%zu req=%zu\n",
                        (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                        (int) request.layout, entry.size, request.dst_size);
                    entry.pinned = false;
                }

                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout size mismatch: model=%llu name_hash=0x%llx layout=%d cached=%zu req=%zu, "
                    "evicting\n",
                    (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                    (int) request.layout, entry.size, request.dst_size);
                void * stale_ptr      = entry.device_ptr;
                size_t stale_size     = entry.size;
                bool   stale_host_res = entry.host_resident;
                bool   stale_pool     = entry.pool_allocated;
                entries_.erase(it);
                it = entries_.end();
                if (!stale_host_res && stale_ptr && stale_size > 0) {
                    if (!stale_pool) {
                        // Always defer free — see comment at layout_mismatch path above.
                        enqueue_deferred_free(stale_ptr, stale_size);
                    }
                    // Pool entries: memory stays in pool, used_ stays at chunk level
                }
            }
        }

        if (it != entries_.end()) {
            auto & entry           = it->second;
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
                    "[UNIFIED-CACHE] layout src change model=%llu name_hash=0x%llx layout=%d cached_src=%p new_src=%p "
                    "size=%zu\n",
                    (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                    (int) request.layout, entry.src_ptr, request.src_ptr, entry.size);
            }
            if (!content_changed) {
                result.device_ptr    = entry.device_ptr;
                result.size          = entry.size;
                result.status        = cache_layout_status::READY;
                result.host_resident = entry.host_resident;
                result.location      = entry.location;
                result.onednn_pack_m = entry.onednn_pack_m;
                result.event         = submit_barrier(deps);
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
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] layout refresh: model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                            (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                            (int) request.layout, entry.size);
        }

        // Allocate new entry if not found or was cleaned up due to FAILED state
        if (it == entries_.end()) {
            // Determine allocation cost: pool may need a new chunk (256 MB) or can sub-allocate (0 cost)
            const bool   pool_can_fit = layout_pool_ && layout_pool_->can_fit(request.dst_size);
            const size_t alloc_cost   = (layout_pool_ && !pool_can_fit) ? layout_pool_->get_default_chunk_size() :
                                                                          (pool_can_fit ? 0 : request.dst_size);

            // Defer get_host_cache() to avoid acquiring g_cache_rw_mutex eagerly.
            // ensure_cached_layout can be called from ggml_sycl_get_weight_layout_ptr
            // during inference while another code path (MoE prestage, runtime budget
            // update, cache registration) holds g_cache_rw_mutex.  Eagerly calling
            // get_host_cache() here caused a deadlock on 120B MoE inference because
            // std::shared_mutex is not reentrant.  Resolve hcache lazily only when a host
            // fallback is actually needed (device alloc failed or prefer_host).
            host_cache * hcache    = nullptr;
            auto         lazy_host_cache = [&]() -> host_cache * {
                if (!hcache) {
                    hcache = get_host_cache(queue_);
                }
                return hcache;
            };
            bool         force_host = false;

            // 0. During graph_compute_impl, force ALL new allocations to host.
            // No new VRAM allocations during inference — prevents both:
            //   (a) eviction-induced stale pointers (evict_one blocked)
            //   (b) L0 OUT_OF_DEVICE_MEMORY from near-full VRAM
            // Existing VRAM-cached experts are used as-is (cache hit path).
            if (g_graph_compute_active.load(std::memory_order_acquire) && lazy_host_cache()) {
                force_host = true;
            }

            // 1. Honor explicit host preference
            if (request.prefer_host && lazy_host_cache()) {
                force_host = true;
            }

            // 2. Check live VRAM FIRST — driver-queried free memory is ground truth.
            //    The budget_ field can be stale (deflated by untracked runtime
            //    reservations), but actual free VRAM is authoritative.
            if (!force_host && alloc_cost > 0) {
                size_t free_mem  = 0;
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
                    if (alloc_cost > usable_free) {
                        // Try eviction before falling back to host.
                        // SOA/COALESCED layouts are only useful on device — host fallback
                        // would degrade to AOS, so eviction is the only viable path.
                        size_t evicted_total = 0;
                        while (evicted_total < alloc_cost) {
                            size_t evicted = evict_one(alloc_cost);
                            if (evicted == 0) {
                                break;
                            }
                            evicted_total += evicted;
                        }
                        if (evicted_total < alloc_cost) {
                            GGML_SYCL_DEBUG(
                                "[UNIFIED-CACHE] live VRAM low after eviction (free=%.1f MB, headroom=%.1f MB, "
                                "need=%.1f MB, evicted=%.1f MB) - using host\n",
                                free_mem / (1024.0f * 1024.0f), headroom / (1024.0f * 1024.0f),
                                request.dst_size / (1024.0f * 1024.0f), evicted_total / (1024.0f * 1024.0f));
                            force_host = true;
                        } else {
                            GGML_SYCL_DEBUG("[UNIFIED-CACHE] evicted %.1f MB to make room for layout (need=%.1f MB)\n",
                                            evicted_total / (1024.0f * 1024.0f),
                                            request.dst_size / (1024.0f * 1024.0f));
                        }
                    }
                    // VRAM is sufficient (or eviction freed enough) — skip budget-based eviction/rejection
                } else if (lazy_host_cache()) {
                    // VRAM query failed — fall back to budget-based check
                    const size_t base_budget = budget_;
                    while (used_.load() + alloc_cost > base_budget) {
                        if (evict_one(alloc_cost) == 0) {
                            break;
                        }
                    }
                    if (used_.load() + alloc_cost > base_budget) {
                        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict for layout (used=%.1f MB, need=%.1f MB)\n",
                                        used_.load() / (1024.0f * 1024.0f), alloc_cost / (1024.0f * 1024.0f));
                        force_host = true;
                    }
                }
            }

            void *         new_device_ptr   = nullptr;
            bool           is_host_resident = false;
            bool           is_pool_alloc    = false;
            cache_location host_location    = cache_location::HOST_MMAP;
            // In S1 mode (all-weights-host), skip pool for DENSE weights that may
            // switch layouts between PP (AOS/oneDNN) and TG (SOA). Pool entries
            // can't be individually freed, so layout switches would leak pool space.
            // MoE EXPERT entries and S1-PRELOAD entries always use pool because:
            // - Experts stay AOS throughout inference (no layout switch)
            // - S1-PRELOAD weights are pinned (never freed)
            // - Without pool, each expert gets individual sycl::malloc_device
            //   causing 21% of PP time in kernel_init_pages (VRAM page clearing)
            const bool is_expert = (request.type == cache_entry_type::MOE_EXPERT);
            const bool skip_pool = ggml_backend_sycl_all_weights_host()
                                   && !request.force_pool
                                   && !is_expert;
            if (!force_host && layout_pool_ && !skip_pool) {
                // Budget guard: check if a potential new pool chunk would exceed budget.
                // The pool allocates 256+ MB chunks but the budget check at the caller
                // only validated the expert size (4.3 KB). Without this guard, the pool
                // can silently blow past the budget by allocating large chunks.
                const size_t pool_headroom = (budget_ > used_.load(std::memory_order_relaxed))
                                             ? budget_ - used_.load(std::memory_order_relaxed)
                                             : 0;
                const bool pool_has_space = layout_pool_->can_fit(request.dst_size);
                if (pool_has_space || pool_headroom >= layout_pool_->get_default_chunk_size()) {
                    // Drain ALL queues before pool growth to prevent BCS stall.
                    // Level Zero's sycl::malloc_device during pool chunk allocation
                    // can stall BCS H2D events, causing staging pool event waits to
                    // hang indefinitely.  Draining before growth ensures no in-flight
                    // operations conflict with the new device memory allocation.
                    if (!pool_has_space) {
                        try { queue_.wait(); } catch (...) {}
                        if (bcs_queue_) {
                            try { bcs_queue_->wait(); } catch (...) {}
                        }
                    }
                    auto pool_result = layout_pool_->allocate(request.dst_size);
                    new_device_ptr   = pool_result.ptr;
                    if (new_device_ptr) {
                        is_pool_alloc = true;
                        if (pool_result.new_physical_bytes > 0) {
                            used_.fetch_add(pool_result.new_physical_bytes, std::memory_order_relaxed);
                        }
                    }
                } else {
                    // Pool would need a new chunk but budget doesn't allow it.
                    // Fall through to host fallback instead of OOM.
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Pool chunk would exceed budget "
                                    "(headroom=%.1f MB, chunk=%.1f MB), using host fallback\n",
                                    pool_headroom / (1024.0f * 1024.0f),
                                    layout_pool_->get_default_chunk_size() / (1024.0f * 1024.0f));
                }
            }
            if (!force_host && !new_device_ptr) {
                // Pool allocation failed; fall back to individual malloc_device
                try {
                    new_device_ptr = ggml_sycl_malloc_device_raw(request.dst_size, queue_, "unified_cache:layout");
                } catch (const sycl::exception & e) {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] layout malloc_device failed: %s, trying host fallback\n",
                                    e.what());
                    new_device_ptr = nullptr;
                }
            }

            bool layout_degraded_to_aos = false;
            if (!new_device_ptr) {
                // Try host_cache fallback when device allocation fails.
                // Skip host fallback for SOA layout — SOA is only useful on GPU.
                // CPU reads AOS directly; creating a host SOA copy wastes memory.
                if (request.layout == GGML_LAYOUT_SOA) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] skipping host_cache fallback for SOA layout "
                        "(CPU reads AOS directly)\n");
                } else if (!hcache) {
                    hcache = lazy_host_cache();
                }
                if (request.layout != GGML_LAYOUT_SOA && hcache) {
                    cache_location host_loc = hcache->get_location(request.key, request.type, request.layer_id,
                                                                   request.expert_id, request.layout);
                    void *         host_ptr =
                        hcache->get(request.key, request.type, request.layer_id, request.expert_id, request.layout);
                    if (!host_ptr) {
                        // Try to create in host cache
                        bool needs_host_fill = false;
                        bool pinned_alloc    = false;
                        host_ptr             = hcache->ensure_cached_alloc(
                            request.key, request.src_ptr, request.src_size, request.dst_size, request.type,
                            request.layer_id, request.expert_id, request.layout, false, &needs_host_fill, &pinned_alloc,
                            &host_loc, &request.xmx_info);

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
                            "[UNIFIED-CACHE] Device full, using host-resident pointer model=%llu name_hash=0x%llx "
                            "layout=%d\n",
                            (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                            (int) request.layout);
                        new_device_ptr   = host_ptr;
                        is_host_resident = true;
                        host_location    = host_loc;
                    }
                }

                if (!new_device_ptr) {
                    // Last resort: alias the mmap source pointer directly for host-side
                    // routing.  For AOS layout, the source data matches directly.  For
                    // non-AOS layouts (SOA, COALESCED), we degrade to AOS — the caller
                    // will use host-side routing with the original AOS data, losing the
                    // layout optimization but avoiding a complete cache failure.
                    if (request.src_ptr && request.src_size > 0) {
                        const bool is_aos = (request.layout == GGML_LAYOUT_AOS);
                        const bool can_degrade = (request.layout != GGML_LAYOUT_AOS &&
                                                  request.src_size <= request.dst_size);
                        if (is_aos || can_degrade) {
                            const char * degrade_label = can_degrade ? " (AOS degraded)" : "";
                            GGML_SYCL_DEBUG(
                                "[UNIFIED-CACHE] Aliasing mmap src%s: model=%llu name_hash=0x%llx "
                                "layout=%d->AOS size=%zu\n",
                                degrade_label,
                                (unsigned long long) request.key.model_id,
                                (unsigned long long) request.key.name_hash,
                                (int) request.layout, request.src_size);
                            new_device_ptr          = const_cast<void *>(request.src_ptr);
                            is_host_resident        = true;
                            host_location           = cache_location::HOST_MMAP;
                            layout_degraded_to_aos  = can_degrade;
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
            entry.size            = layout_degraded_to_aos ? request.src_size : request.dst_size;
            entry.type            = request.type;
            entry.layer_id        = request.layer_id;
            entry.expert_id       = request.expert_id;
            entry.layout          = layout_degraded_to_aos ? GGML_LAYOUT_AOS : request.layout;
            entry.onednn_pack_m   = request.onednn_pack_m;
            entry.xmx_info        = request.xmx_info;
            entry.access_count    = 1;
            entry.last_access     = time_++;
            entry.pinned          = is_pool_alloc;  // Pool entries are immediately pinned (can't be individually freed)
            entry.hot             = false;
            entry.state           = is_host_resident ? cache_entry_state::READY : cache_entry_state::IN_PROGRESS;
            entry.has_ready_event = false;
            entry.host_resident   = is_host_resident;
            entry.location        = is_host_resident ? host_location : cache_location::DEVICE;
            entry.pool_allocated  = is_pool_alloc;

            if (g_ggml_sycl_debug >= 2) {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout insert model=%llu name_hash=0x%llx layout=%d size=%zu entries=%zu "
                    "buckets=%zu\n",
                    (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                    (int) request.layout, request.dst_size, entries_.size(), entries_.bucket_count());
            }
            entries_[key] = entry;
            auto id_it    = id_to_key_.find(request.key);
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
                               (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash);
                if (cache_assert_enabled()) {
                    GGML_ABORT("unified_cache id_to_key mismatch");
                }
            }
            if (!is_host_resident && !is_pool_alloc) {
                // Only count device memory against unified cache budget.
                // Pool entries are tracked at chunk granularity (added when chunk is allocated).
                used_.fetch_add(request.dst_size, std::memory_order_relaxed);
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
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] layout allocate: model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                            (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
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
        GGML_SYCL_DEBUG(
            "[DEBUG-FILL] Starting fill: device_ptr=%p src_ptr=%p dst_size=%zu src_size=%zu layout=%d fill_fn=%s\n",
            device_ptr, request.src_ptr, request.dst_size, request.src_size, (int) request.layout,
            request.fill_fn ? "yes" : "no");
        if (copy_trace_enabled()) {
            GGML_LOG_INFO(
                "[SYCL] layout fill begin: model=%llu name_hash=0x%llx layout=%d dst=%p dst_size=%zu src=%p "
                "src_size=%zu fill_fn=%s\n",
                (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                (int) request.layout, device_ptr, request.dst_size, request.src_ptr, request.src_size,
                request.fill_fn ? "yes" : "no");
            fflush(stderr);
        }

        // Use override_queue when provided to drive H2D transfers on the
        // caller's queue instead of the cache's internal queue_.
        sycl::queue & fill_queue = override_queue ? *override_queue : queue_;
        if (request.fill_fn) {
            GGML_SYCL_DEBUG("[DEBUG-FILL] Calling fill_fn...\n");
            fill_event = request.fill_fn(fill_queue, device_ptr, request.dst_size, request.src_ptr, request.src_size,
                                         request.fill_ctx, deps);
            GGML_SYCL_DEBUG("[DEBUG-FILL] fill_fn returned\n");
        } else {
            GGML_SYCL_DEBUG("[DEBUG-FILL] Calling copy_to_device_async...\n");
            if (!request.src_ptr || !device_ptr) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] ensure_cached_layout: null pointer in fill "
                               "(src=%p dst=%p size=%zu)\n",
                               request.src_ptr, device_ptr, request.src_size);
                result.status = cache_layout_status::FAILED;
                return result;
            }
            fill_event = copy_to_device_async(device_ptr, request.src_ptr, request.src_size, deps,
                                                override_queue);
            GGML_SYCL_DEBUG("[DEBUG-FILL] copy_to_device_async returned\n");
        }

        // When skip_fill_wait is set (bulk preload), skip the synchronous wait
        // and return the event for the caller to batch-wait later.
        if (request.skip_fill_wait) {
            GGML_SYCL_DEBUG("[DEBUG-FILL] skip_fill_wait: returning event for batch wait\n");

            // Submit padding async if needed (implicitly ordered after fill on in-order queue)
            sycl::event last_event = fill_event;
            if (request.layout != GGML_LAYOUT_XMX_TILED && request.layout != GGML_LAYOUT_XMX_GEMM_TILED &&
                request.layout != GGML_LAYOUT_ONEDNN_PACKED && request.layout != GGML_LAYOUT_ONEDNN_WOQ &&
                request.dst_size > request.src_size) {
                const size_t pad_bytes = request.dst_size - request.src_size;
                void *       pad_ptr   = static_cast<char *>(device_ptr) + request.src_size;
                last_event = fill_queue.memset(pad_ptr, 0, pad_bytes);
            }

            // Mark entry as IN_PROGRESS with ready_event so concurrent lookups
            // can see the pending fill and wait on it if needed.
            {
                std::unique_lock<std::shared_mutex> lock(rw_mutex_);
                auto                                it = entries_.find(key);
                if (it != entries_.end()) {
                    it->second.has_ready_event  = true;
                    it->second.ready_event      = last_event;
                    it->second.state            = cache_entry_state::IN_PROGRESS;
                    it->second.last_write_event = last_event;
                    it->second.has_write_event  = true;
                }
            }
            entry_cv_.notify_all();

            result.device_ptr    = device_ptr;
            result.size          = request.dst_size;
            result.status        = cache_layout_status::IN_PROGRESS;
            result.host_resident = false;
            result.location      = cache_location::DEVICE;
            result.event         = last_event;
            return result;
        }

        // Chain padding after fill via depends_on instead of separate .wait() calls.
        // Single wait at the end ensures both fill and padding complete before marking READY.
        sycl::event last_fill_event = fill_event;
        if (request.layout != GGML_LAYOUT_XMX_TILED && request.layout != GGML_LAYOUT_XMX_GEMM_TILED &&
            request.layout != GGML_LAYOUT_ONEDNN_PACKED && request.layout != GGML_LAYOUT_ONEDNN_WOQ &&
            request.dst_size > request.src_size) {
            const size_t pad_bytes = request.dst_size - request.src_size;
            void *       pad_ptr   = static_cast<char *>(device_ptr) + request.src_size;
            GGML_SYCL_DEBUG("[DEBUG-FILL] About to memset padding: pad_ptr=%p pad_bytes=%zu\n", pad_ptr, pad_bytes);
            // Chain padding memset after fill via depends_on — no intermediate CPU stall
            last_fill_event = queue_.submit([&](sycl::handler & cgh) {
                cgh.depends_on(fill_event);
                cgh.memset(pad_ptr, 0, pad_bytes);
            });
            GGML_SYCL_DEBUG("[DEBUG-FILL] Padding memset submitted\n");
        }
        last_fill_event.wait();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout fill failed (sycl): %s\n", e.what());
        GGML_LOG_ERROR(
            "[UNIFIED-CACHE] layout fill context model=%llu name_hash=0x%llx type=%d layer=%d expert=%d layout=%d "
            "src=%p "
            "src_size=%zu dst_size=%zu\n",
            (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
            static_cast<int>(request.type), request.layer_id, request.expert_id, static_cast<int>(request.layout),
            request.src_ptr, request.src_size, request.dst_size);
        if (try_host_fallback("sycl_fill")) {
            return result;
        }
        if (const char * msg = e.what()) {
            if (std::strstr(msg, "DEVICE_LOST") || std::strstr(msg, "device lost")) {
                GGML_LOG_ERROR("[UNIFIED-CACHE] Device lost during cache fill - aborting to preserve backtrace.\n");
                std::abort();
            }
        }
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        auto                                it = entries_.find(key);
        if (it != entries_.end()) {
            // Mark entry as FAILED instead of deleting it immediately.
            // This allows waiting threads to see the failure and fall back gracefully.
            it->second.state           = cache_entry_state::FAILED;
            it->second.has_ready_event = false;
            // Defer the free to avoid queue_.wait() under rw_mutex_ (deadlock risk).
            // The deferred free will synchronize via barrier event instead.
            if (it->second.device_ptr) {
                if (!it->second.pool_allocated) {
                    enqueue_deferred_free(it->second.device_ptr, it->second.size);
                }
                // Pool entries: used_ stays at chunk level
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
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        auto                                it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.state           = cache_entry_state::FAILED;
            it->second.has_ready_event = false;
            if (it->second.device_ptr) {
                if (!it->second.pool_allocated) {
                    enqueue_deferred_free(it->second.device_ptr, it->second.size);
                }
                // Pool entries: used_ stays at chunk level
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
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        auto                                it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.state           = cache_entry_state::FAILED;
            it->second.has_ready_event = false;
            if (it->second.device_ptr) {
                if (!it->second.pool_allocated) {
                    enqueue_deferred_free(it->second.device_ptr, it->second.size);
                }
                // Pool entries: used_ stays at chunk level
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
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        auto                                it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.has_ready_event  = false;
            it->second.state            = cache_entry_state::READY;
            it->second.last_write_event = fill_event;
            it->second.has_write_event  = true;
        }
    }
    entry_cv_.notify_all();

    result.device_ptr    = device_ptr;
    result.size          = request.dst_size;
    result.status        = cache_layout_status::READY;
    result.host_resident = false;
    result.location      = cache_location::DEVICE;
    // Don't set result.event - no need to wait since everything is done synchronously
    return result;
}

void unified_cache::finalize_pending_fills() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    size_t finalized = 0;
    for (auto & [key, entry] : entries_) {
        if (entry.state == cache_entry_state::IN_PROGRESS && entry.has_ready_event) {
            entry.state           = cache_entry_state::READY;
            entry.has_ready_event = false;
            finalized++;
        }
    }
    size_t in_progress = 0, ready = 0, no_event = 0;
    for (auto & [k, e] : entries_) {
        if (e.state == cache_entry_state::IN_PROGRESS) {
            if (e.has_ready_event) in_progress++;
            else no_event++;
        } else if (e.state == cache_entry_state::READY) ready++;
    }
    GGML_LOG_INFO("[UNIFIED-CACHE] finalize: finalized=%zu, remaining: ready=%zu in_progress_with_event=%zu in_progress_no_event=%zu total_entries=%zu id_to_key=%zu\n",
                  finalized, ready, in_progress, no_event, entries_.size(), id_to_key_.size());
    if (finalized > 0) {
        entry_cv_.notify_all();
    }
}

bool unified_cache::is_cached(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) const {
    if (!key_id.valid) {
        return false;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
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
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return false;
    }
    return entries_.find(id_it->second) != entries_.end();
}

void * unified_cache::get(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return nullptr;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
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
                (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) layout, entry.size,
                entry.has_ready_event ? 1 : 0);
            return nullptr;
        }
    }
    return entry.device_ptr;
}

void * unified_cache::try_get_cached_fast(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return nullptr;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        static std::atomic<int> miss_log{0};
        if (miss_log.fetch_add(1, std::memory_order_relaxed) < 3) {
            GGML_LOG_WARN("[CACHE-LOOKUP] MISS: model=%llu hash=0x%llx aux=0x%llx layout=%d id_to_key_size=%zu entries_size=%zu\n",
                          (unsigned long long)key_id.model_id, (unsigned long long)key_id.name_hash,
                          (unsigned long long)key_id.aux_id, (int)layout, id_to_key_.size(), entries_.size());
        }
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    const auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }
    if (entry.state != cache_entry_state::READY) {
        return nullptr;
    }
    // HOST_MMAP entries contain raw mmap pointers that are NOT GPU-accessible.
    // Returning them from lookup would cause GPU page faults when kernels
    // dereference the pointer.  Only DEVICE and HOST_PINNED (sycl::malloc_host,
    // GPU-accessible via PCIe zero-copy) entries are safe for GPU dispatch.
    if (entry.location == cache_location::HOST_MMAP) {
        return nullptr;
    }
    return entry.device_ptr;
}

void * unified_cache::try_get_cached_with_event(const ggml_sycl_cache_id & key_id,
                                                ggml_layout_mode           layout,
                                                sycl::event *              out_event,
                                                bool *                     out_has_event) {
    if (out_event) {
        *out_event = sycl::event{};
    }
    if (out_has_event) {
        *out_has_event = false;
    }
    if (!key_id.valid) {
        return nullptr;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    const auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }
    if (entry.location == cache_location::HOST_MMAP) {
        return nullptr;
    }
    // READY entries: return pointer, no event needed.
    if (entry.state == cache_entry_state::READY) {
        return entry.device_ptr;
    }
    // IN_PROGRESS entries: return pointer + ready_event so the caller can
    // chain the subsequent kernel/memcpy after the fill completes.
    if (entry.state == cache_entry_state::IN_PROGRESS && entry.device_ptr) {
        if (entry.has_ready_event) {
            if (out_event) {
                *out_event = entry.ready_event;
            }
            if (out_has_event) {
                *out_has_event = true;
            }
        }
        return entry.device_ptr;
    }
    return nullptr;
}

// --- Decomposed cache operations ---

void * unified_cache::allocate_slot(const ggml_sycl_cache_id & key,
                                    size_t                     size,
                                    ggml_layout_mode           layout,
                                    cache_entry_type           type,
                                    int                        layer_id,
                                    int                        expert_id) {
    if (!key.valid || size == 0) {
        return nullptr;
    }

    const unified_cache_key cache_key{ type, key, layer_id, expert_id };

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }

    // Check if entry already exists with matching layout and size
    auto it = entries_.find(cache_key);
    if (it != entries_.end()) {
        auto & entry = it->second;
        if (entry.layout == layout && entry.size == size && entry.device_ptr &&
            entry.location != cache_location::HOST_MMAP) {
            // Already allocated (may be READY or IN_PROGRESS) — return existing ptr.
            // HOST_MMAP entries are raw mmap pointers, not device allocations;
            // treat them as a mismatch so a real device allocation is made.
            return entry.device_ptr;
        }
        // Layout/size mismatch or HOST_MMAP — evict old entry
        if (entry.device_ptr && !entry.host_resident) {
            if (!entry.pool_allocated) {
                enqueue_deferred_free(entry.device_ptr, entry.size);
            }
        }
        entries_.erase(it);
    }

    // Check VRAM budget and evict if needed
    const bool skip_pool     = ggml_backend_sycl_all_weights_host();
    bool       is_pool_alloc = false;
    void *     device_ptr    = nullptr;

    if (layout_pool_ && !skip_pool) {
        auto pool_result = layout_pool_->allocate(size);
        device_ptr       = pool_result.ptr;
        if (device_ptr) {
            is_pool_alloc = true;
            if (pool_result.new_physical_bytes > 0) {
                used_.fetch_add(pool_result.new_physical_bytes, std::memory_order_relaxed);
            }
        }
    }

    if (!device_ptr && arena_.active()) {
        // VRAM arena path: sub-allocate from the weight zone.
        device_ptr = arena_.zone_alloc(vram_zone_id::WEIGHT, size);
        if (device_ptr) {
            is_pool_alloc = true;  // Arena-owned: don't free individually.
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate_slot: arena weight zone alloc %zu bytes\n", size);
        }
    }

    if (!device_ptr) {
        // Check live VRAM and evict if needed
        size_t free_mem = 0, total_mem = 0;
        try {
            const int device_id = get_device_id_from_queue(queue_);
            ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
        } catch (...) {
        }

        if (total_mem > 0) {
            const size_t min_headroom = 256ull * 1024ull * 1024ull;
            const size_t headroom     = std::max(min_headroom, total_mem / 10);
            const size_t usable_free  = free_mem > headroom ? free_mem - headroom : 0;
            if (size > usable_free) {
                size_t evicted_total = 0;
                while (evicted_total < size) {
                    size_t evicted = evict_one(size);
                    if (evicted == 0) {
                        break;
                    }
                    evicted_total += evicted;
                }
                if (evicted_total < size) {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate_slot: VRAM insufficient after eviction\n");
                    return nullptr;
                }
            }
        }

        try {
            device_ptr = ggml_sycl_malloc_device_raw(size, queue_, "unified_cache:slot");
        } catch (const sycl::exception & e) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate_slot: malloc_device failed: %s\n", e.what());
            return nullptr;
        }
    }

    if (!device_ptr) {
        return nullptr;
    }

    // Create entry in IN_PROGRESS state (not yet READY — caller must register_ready)
    unified_cache_entry entry{};
    entry.device_ptr      = device_ptr;
    entry.src_ptr         = nullptr;
    entry.content_hash    = 0;
    entry.size            = size;
    entry.type            = type;
    entry.layer_id        = layer_id;
    entry.expert_id       = expert_id;
    entry.layout          = layout;
    entry.onednn_pack_m   = 0;
    entry.xmx_info        = {};
    entry.access_count    = 1;
    entry.last_access     = time_++;
    entry.pinned          = is_pool_alloc;
    entry.hot             = false;
    entry.state           = cache_entry_state::IN_PROGRESS;
    entry.has_ready_event = false;
    entry.host_resident   = false;
    entry.location        = cache_location::DEVICE;
    entry.pool_allocated  = is_pool_alloc;

    entries_[cache_key] = entry;
    auto id_it          = id_to_key_.find(key);
    if (id_it == id_to_key_.end()) {
        if (id_to_key_.bucket_count() == 0) {
            id_to_key_.rehash(1);
        }
        id_to_key_.emplace(key, cache_key);
    }

    if (!is_pool_alloc) {
        used_.fetch_add(size, std::memory_order_relaxed);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate_slot: layout=%d size=%zu ptr=%p\n", (int) layout, size, device_ptr);
    return device_ptr;
}

void unified_cache::register_ready(const ggml_sycl_cache_id & key,
                                   void *                     device_ptr,
                                   ggml_layout_mode           layout,
                                   size_t                     size,
                                   cache_entry_type           type,
                                   int                        layer_id,
                                   int                        expert_id,
                                   const void *               src_ptr,
                                   int64_t                    onednn_pack_m) {
    if (!key.valid || !device_ptr) {
        return;
    }

    const unified_cache_key cache_key{ type, key, layer_id, expert_id };

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                it = entries_.find(cache_key);
    if (it != entries_.end()) {
        auto & entry          = it->second;
        entry.device_ptr      = device_ptr;
        entry.state           = cache_entry_state::READY;
        entry.has_ready_event = false;
        entry.layout          = layout;
        entry.size            = size;
        entry.src_ptr         = src_ptr;
        entry.onednn_pack_m   = onednn_pack_m;
        entry.access_count++;
        entry.last_access = time_++;
    } else {
        // Entry was not pre-allocated via allocate_slot — create it directly as READY
        unified_cache_entry entry{};
        entry.device_ptr      = device_ptr;
        entry.src_ptr         = src_ptr;
        entry.content_hash    = 0;
        entry.size            = size;
        entry.type            = type;
        entry.layer_id        = layer_id;
        entry.expert_id       = expert_id;
        entry.layout          = layout;
        entry.onednn_pack_m   = onednn_pack_m;
        entry.xmx_info        = {};
        entry.access_count    = 1;
        entry.last_access     = time_++;
        entry.pinned          = false;
        entry.hot             = false;
        entry.state           = cache_entry_state::READY;
        entry.has_ready_event = false;
        entry.host_resident   = false;
        entry.location        = cache_location::DEVICE;
        entry.pool_allocated  = false;

        entries_[cache_key] = entry;
        auto id_it          = id_to_key_.find(key);
        if (id_it == id_to_key_.end()) {
            if (id_to_key_.bucket_count() == 0) {
                id_to_key_.rehash(1);
            }
            id_to_key_.emplace(key, cache_key);
        }
    }
    entry_cv_.notify_all();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] register_ready: layout=%d size=%zu ptr=%p\n", (int) layout, size, device_ptr);
}

// ---------------------------------------------------------------------------
// Atomic expert group staging: all 3 tensors stage or none stage.
// ---------------------------------------------------------------------------
bool unified_cache::stage_expert_group(int                          block_id,
                                       int                          expert_id_arg,
                                       const expert_tensor_group &  keys,
                                       const staging_tensor_data &  gate_data,
                                       const staging_tensor_data &  up_data,
                                       const staging_tensor_data &  down_data,
                                       ggml_layout_mode             layout,
                                       const cache_layout_request * gate_req,
                                       const cache_layout_request * up_req,
                                       const cache_layout_request * down_req) {
    // Collect the tensors that need staging (skip unregistered roles)
    struct slot_info {
        const ggml_sycl_cache_id *   key;
        const staging_tensor_data *  data;
        const cache_layout_request * req;
        void *                       ptr;
        bool                         was_existing;
    };
    std::vector<slot_info> slots;
    slots.reserve(3);

    if (keys.has_gate && gate_data.src_ptr && gate_data.dst_size > 0) {
        slots.push_back({ &keys.gate_key, &gate_data, gate_req, nullptr, false });
    }
    if (keys.has_up && up_data.src_ptr && up_data.dst_size > 0) {
        slots.push_back({ &keys.up_key, &up_data, up_req, nullptr, false });
    }
    if (keys.has_down && down_data.src_ptr && down_data.dst_size > 0) {
        slots.push_back({ &keys.down_key, &down_data, down_req, nullptr, false });
    }

    if (slots.empty()) {
        return false;
    }

    // Calculate total VRAM needed (only for tensors not already cached)
    size_t total_needed = 0;
    for (auto & s : slots) {
        void * existing = lookup(*s.key, layout);
        if (existing) {
            s.ptr          = existing;  // Already cached -- skip allocation
            s.was_existing = true;
        } else {
            total_needed += s.data->dst_size;
        }
    }

    // All already cached? Success.
    if (total_needed == 0) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] stage_expert_group: blk=%d exp=%d "
                        "all %zu tensors already cached\n",
                        block_id, expert_id_arg, slots.size());
        return true;
    }

    // Check VRAM availability and evict if needed
    if (available() < total_needed) {
        size_t freed = evict(total_needed - available());
        (void) freed;
    }

    if (available() < total_needed) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] stage_expert_group: blk=%d exp=%d "
                        "insufficient VRAM (need=%zu avail=%zu)\n",
                        block_id, expert_id_arg, total_needed, available());
        return false;
    }

    // Explicit allocate + fill + register path — NO ensure_cached_layout.
    // This avoids the black-box behavior of ensure_cached_layout which can
    // silently fall back to host cache, fail to register entries, or use
    // the wrong queue.
    {
        bool all_ok = true;
        for (auto & s : slots) {
            if (s.was_existing) continue;

            // 1. Allocate VRAM slot (device only, no host fallback)
            s.ptr = allocate_slot(*s.key, s.data->dst_size, layout,
                                  cache_entry_type::MOE_EXPERT,
                                  s.data->layer_id, s.data->expert_id);
            if (!s.ptr) {
                all_ok = false;
                break;
            }
        }

        if (!all_ok) {
            // Rollback partial allocations
            for (auto & s : slots) {
                if (s.was_existing || !s.ptr) continue;
                remove(*s.key, cache_entry_type::MOE_EXPERT,
                       s.data->layer_id, s.data->expert_id, layout);
                s.ptr = nullptr;
            }
            return false;
        }

        // 2. Fill all slots (DMA + optional reorder via fill_fn from request).
        //    Fills are submitted WITHOUT a per-expert dq.wait().  This allows
        //    BCS H2D copies and CCS reorder kernels to pipeline across experts.
        //    The caller batches experts and calls get_dma_queue().wait() +
        //    get_bcs_queue().wait() + finalize_pending_fills() at batch boundaries.
        //    Raw H2D copies go through BCS (copy-only engine) to keep CCS free.
        //    Fill functions receive DMA queue and internally route H2D to BCS.
        //
        //    IMPORTANT: When using a pre-allocated temp buffer (prealloc_temp),
        //    consecutive fills share the same staging VRAM.  The BCS H2D for
        //    tensor N+1 must not overwrite the temp buffer while the CCS reorder
        //    for tensor N is still reading it.  We chain fills via deps: each
        //    fill's H2D depends on the previous fill's reorder completion event.
        //    Without this, the BCS and CCS queues race on the shared temp buffer
        //    (WAR hazard) causing corrupted SOA layouts or BCS CAT faults.
        sycl::queue & dq  = get_dma_queue();
        sycl::queue & bcs = get_bcs_queue();
        sycl::event   last_event;
        bool          has_last_event = false;
        for (auto & s : slots) {
            if (s.was_existing) continue;
            if (s.req && s.req->fill_fn) {
                // Use the caller-provided fill function (GPU reorder, CPU reorder, etc.)
                // Fill functions route H2D to BCS internally via ctx->bcs_queue.
                // Pass last_event as dependency so BCS H2D waits for previous
                // CCS reorder to finish reading the shared prealloc_temp buffer.
                std::vector<sycl::event> fill_deps;
                if (has_last_event) {
                    fill_deps.push_back(last_event);
                }
                last_event =
                    s.req->fill_fn(dq, s.ptr, s.data->dst_size, s.data->src_ptr, s.data->src_size, s.req->fill_ctx, fill_deps);
                has_last_event = true;
            } else if (s.data->src_ptr && s.data->src_size > 0) {
                // Raw DMA copy — route to BCS (copy engine) to keep CCS free
                last_event = bcs.memcpy(s.ptr, s.data->src_ptr, std::min(s.data->dst_size, s.data->src_size));
                has_last_event = true;
            }
        }

        // 3. Mark entries as IN_PROGRESS with ready_event (deferred READY).
        //    Caller calls finalize_pending_fills() after batch wait to promote
        //    these to READY state.  This avoids the per-expert dq.wait() that
        //    serialized all BCS/CCS work and caused GT engine resets.
        {
            std::unique_lock<std::shared_mutex> lock(rw_mutex_);
            for (auto & s : slots) {
                if (s.was_existing) {
                    continue;
                }
                unified_cache_key ckey{ cache_entry_type::MOE_EXPERT, *s.key, s.data->layer_id, s.data->expert_id };
                auto              it = entries_.find(ckey);
                if (it != entries_.end()) {
                    it->second.has_ready_event  = true;
                    it->second.ready_event      = last_event;
                    it->second.layout           = layout;
                    it->second.size             = s.data->dst_size;
                    if (has_last_event) {
                        it->second.last_write_event = last_event;
                        it->second.has_write_event  = true;
                    }
                }
            }
        }

        // Verify (first 3 experts only)
        for (auto & s : slots) {
            if (s.was_existing) continue;
            static std::atomic<int> verify_log{ 0 };
            if (verify_log.fetch_add(1, std::memory_order_relaxed) < 3) {
                fprintf(stderr,
                        "[STAGE-SUBMIT] blk=%d exp=%d layout=%d ptr=%p "
                        "model=%llu hash=0x%llx aux=0x%llx (deferred READY)\n",
                        block_id, expert_id_arg, (int) layout, s.ptr, (unsigned long long) s.key->model_id,
                        (unsigned long long) s.key->name_hash, (unsigned long long) s.key->aux_id);
            }
        }

        return true;
    }

    // Fallback: allocate + raw DMA fill + register_ready
    bool alloc_ok = true;
    for (auto & s : slots) {
        if (s.was_existing) {
            continue;  // Already cached
        }
        s.ptr = allocate_slot(*s.key, s.data->dst_size, layout,
                              cache_entry_type::MOE_EXPERT,
                              s.data->layer_id, s.data->expert_id);
        if (!s.ptr) {
            alloc_ok = false;
            break;
        }
    }

    // Rollback on partial allocation failure
    if (!alloc_ok) {
        for (auto & s : slots) {
            if (s.ptr && !s.was_existing) {
                std::shared_lock<std::shared_mutex> lock(rw_mutex_);
                unified_cache_key ckey{ cache_entry_type::MOE_EXPERT, *s.key,
                                        s.data->layer_id, s.data->expert_id };
                auto it = entries_.find(ckey);
                if (it != entries_.end() &&
                    it->second.state == cache_entry_state::IN_PROGRESS) {
                    lock.unlock();
                    remove(*s.key, cache_entry_type::MOE_EXPERT,
                           s.data->layer_id, s.data->expert_id, layout);
                }
            }
        }
        GGML_LOG_WARN("[UNIFIED-CACHE] stage_expert_group: blk=%d exp=%d "
                      "allocation rollback\n", block_id, expert_id_arg);
        return false;
    }

    // Fill all tensors using BCS queue (copy-only engine) — no per-expert wait.
    // Raw H2D copies go through BCS to keep CCS free and prevent GT engine resets.
    sycl::queue & bcs = get_bcs_queue();
    sycl::event   last_event;
    for (auto & s : slots) {
        if (s.was_existing) continue;

        try {
            size_t copy_size = std::min(s.data->src_size, s.data->dst_size);
            last_event       = bcs.memcpy(s.ptr, s.data->src_ptr, copy_size);
        } catch (...) {
            GGML_LOG_WARN("[UNIFIED-CACHE] stage_expert_group: DMA memcpy "
                          "failed blk=%d exp=%d\n", block_id, expert_id_arg);
            // Rollback all new allocations
            for (auto & s2 : slots) {
                if (s2.ptr && !s2.was_existing) {
                    remove(*s2.key, cache_entry_type::MOE_EXPERT,
                           s2.data->layer_id, s2.data->expert_id, layout);
                }
            }
            return false;
        }
    }

    // Mark entries as IN_PROGRESS with ready_event (deferred READY).
    // Caller waits at batch boundaries + calls finalize_pending_fills().
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        for (auto & s : slots) {
            if (s.was_existing) {
                continue;
            }
            unified_cache_key ckey{ cache_entry_type::MOE_EXPERT, *s.key, s.data->layer_id, s.data->expert_id };
            auto              it = entries_.find(ckey);
            if (it != entries_.end()) {
                it->second.has_ready_event  = true;
                it->second.ready_event      = last_event;
                it->second.layout           = layout;
                it->second.size             = s.data->dst_size;
                it->second.src_ptr          = s.data->src_ptr;
                it->second.last_write_event = last_event;
                it->second.has_write_event  = true;
            }
        }
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] stage_expert_group: blk=%d exp=%d "
                    "staged %zu tensors (total=%zu bytes)\n",
                    block_id, expert_id_arg, slots.size(), total_needed);
    return true;
}

// ---------------------------------------------------------------------------
// Atomic expert group eviction: all 3 tensors evict together.
// ---------------------------------------------------------------------------
void unified_cache::evict_expert_group(const expert_tensor_group & keys,
                                       ggml_layout_mode            layout) {
    auto evict_one_key = [&](const ggml_sycl_cache_id & key) {
        if (!key.valid) return;
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        auto id_it = id_to_key_.find(key);
        if (id_it == id_to_key_.end()) return;
        auto entry_it = entries_.find(id_it->second);
        if (entry_it == entries_.end()) return;
        int lid = entry_it->second.layer_id;
        int eid = entry_it->second.expert_id;
        lock.unlock();
        remove(key, cache_entry_type::MOE_EXPERT, lid, eid, layout);
    };

    if (keys.has_gate) evict_one_key(keys.gate_key);
    if (keys.has_up)   evict_one_key(keys.up_key);
    if (keys.has_down) evict_one_key(keys.down_key);
}

// ---------------------------------------------------------------------------
// Expert-granularity LRU eviction: find coldest expert group, evict all 3.
// ---------------------------------------------------------------------------
size_t unified_cache::evict_coldest_expert_group(
    const std::unordered_map<int64_t, expert_tensor_group> & expert_groups,
    ggml_layout_mode                                         layout) {

    // Scan all expert groups to find the coldest one (lowest combined frequency).
    int64_t  coldest_gkey        = -1;
    uint32_t coldest_freq        = UINT32_MAX;
    int64_t  coldest_last_access = std::numeric_limits<int64_t>::max();
    size_t   coldest_total_bytes = 0;

    auto get_entry_info = [&](const ggml_sycl_cache_id & key)
        -> std::tuple<bool, uint32_t, int64_t, size_t, bool> {
        // Returns: (found, access_count, last_access, size, pinned)
        if (!key.valid) {
            return { false, 0, 0, 0, false };
        }
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        auto id_it = id_to_key_.find(key);
        if (id_it == id_to_key_.end()) {
            return { false, 0, 0, 0, false };
        }
        auto entry_it = entries_.find(id_it->second);
        if (entry_it == entries_.end()) {
            return { false, 0, 0, 0, false };
        }
        const auto & e = entry_it->second;
        return { true, e.access_count, e.last_access, e.size, e.pinned };
    };

    for (const auto & [gkey, grp] : expert_groups) {
        uint32_t combined_freq = 0;
        int64_t  oldest_access = std::numeric_limits<int64_t>::max();
        size_t   total_bytes   = 0;
        bool     any_found     = false;
        bool     any_pinned    = false;

        auto check_tensor = [&](const ggml_sycl_cache_id & key,
                                bool has_key) {
            if (!has_key || !key.valid) return;
            auto [found, freq, last_access, sz, pinned] = get_entry_info(key);
            if (!found) return;
            any_found = true;
            combined_freq += freq;
            oldest_access = std::min(oldest_access, last_access);
            total_bytes += sz;
            if (pinned) any_pinned = true;
        };

        check_tensor(grp.gate_key, grp.has_gate);
        check_tensor(grp.up_key,   grp.has_up);
        check_tensor(grp.down_key, grp.has_down);

        if (!any_found || any_pinned || total_bytes == 0) {
            continue;
        }

        // Compare: lower frequency wins, then older last_access as tiebreaker
        bool is_colder = (combined_freq < coldest_freq) ||
                         (combined_freq == coldest_freq &&
                          oldest_access < coldest_last_access);

        if (is_colder) {
            coldest_gkey        = gkey;
            coldest_freq        = combined_freq;
            coldest_last_access = oldest_access;
            coldest_total_bytes = total_bytes;
        }
    }

    if (coldest_gkey < 0) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict_coldest_expert_group: "
                        "no evictable group found\n");
        return 0;
    }

    // Evict the coldest group
    auto it = expert_groups.find(coldest_gkey);
    if (it == expert_groups.end()) {
        return 0;
    }

    const int block = static_cast<int>(coldest_gkey >> 16);
    const int exp   = static_cast<int>(coldest_gkey & 0xFFFF);
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict_coldest_expert_group: blk=%d exp=%d "
                    "freq=%u bytes=%zu\n", block, exp,
                    coldest_freq, coldest_total_bytes);

    evict_expert_group(it->second, layout);
    return coldest_total_bytes;
}

void * unified_cache::lookup(const ggml_sycl_cache_id & key, ggml_layout_mode layout) {
    // Identical semantics to try_get_cached_fast -- shared_lock read-only path
    // NOTE: try_get_cached_fast already filters HOST_MMAP entries.
    return try_get_cached_fast(key, layout);
}

void * unified_cache::lookup_device_only(const ggml_sycl_cache_id & key, ggml_layout_mode layout) {
    if (!key.valid) {
        return nullptr;
    }
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key);
    if (id_it == id_to_key_.end()) {
        return nullptr;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return nullptr;
    }
    const auto & entry = entry_it->second;
    if (entry.layout != layout) {
        return nullptr;
    }
    if (entry.state != cache_entry_state::READY) {
        return nullptr;
    }
    // Only return VRAM-resident entries. Host-pinned and mmap entries are excluded.
    if (entry.host_resident) {
        return nullptr;
    }
    return entry.device_ptr;
}

unified_cache::weight_ptr_result unified_cache::get_weight_ptr(const ggml_sycl_cache_id & key) {
    weight_ptr_result result{};
    if (!key.valid) {
        return result;
    }
    // Try layouts in priority order: COALESCED > SOA > AOS.
    // This ensures the best available layout is returned, not whatever
    // id_to_key_ happens to point at (which can be ONEDNN_PACKED from PP).
    static const ggml_layout_mode try_layouts[] = {
        GGML_LAYOUT_COALESCED, GGML_LAYOUT_SOA, GGML_LAYOUT_AOS
    };
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    for (auto layout : try_layouts) {
        // Build the full cache key with this layout
        unified_cache_key ckey{ cache_entry_type::DENSE_WEIGHT, key, -1, -1 };
        auto entry_it = entries_.find(ckey);
        if (entry_it == entries_.end()) continue;
        const auto & entry = entry_it->second;
        if (entry.state != cache_entry_state::READY) continue;
        if (!entry.device_ptr) continue;
        // HOST_MMAP entries contain raw mmap pointers that are NOT GPU-accessible.
        // Returning them would cause CCS page faults when GPU kernels dereference
        // the pointer.  Only DEVICE and HOST_PINNED entries are safe.
        if (entry.location == cache_location::HOST_MMAP) continue;
        result.ptr       = entry.device_ptr;
        result.layout    = entry.layout;
        result.on_device = !entry.host_resident;
        return result;
    }
    return result;
}

sycl::queue & unified_cache::get_dma_queue() {
    // Return dedicated DMA queue if available, otherwise fall back to compute queue
    if (dma_queue_) {
        return *dma_queue_;
    }
    return queue_;
}

sycl::queue & unified_cache::get_bcs_queue() {
    // Return BCS (copy-only) queue if available, otherwise fall back to DMA queue
    if (bcs_queue_) {
        return *bcs_queue_;
    }
    return get_dma_queue();
}

// get_or_wait REMOVED — legacy synchronous blocking pattern
// get_by_data_ptr REMOVED — O(N) scan with zero callers

cache_ptr_view unified_cache::get_view(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    cache_ptr_view view{};
    if (!key_id.valid) {
        return view;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
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
    if (entry.state == cache_entry_state::EVICTING) {
        // Entry is being async-evicted to host — device pointer is stale-bound.
        // Return empty view so caller falls back to host_cache / mmap.
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
    view.ptr           = entry.device_ptr;
    view.size          = entry.size;
    view.layout        = entry.layout;
    view.onednn_pack_m = entry.onednn_pack_m;
    view.location      = entry.location;
    view.type          = entry.type;
    view.layer_id      = entry.layer_id;
    view.expert_id     = entry.expert_id;
    view.xmx_info      = entry.xmx_info;
    return view;
}

void unified_cache::remove(const ggml_sycl_cache_id & key_id,
                           cache_entry_type           type,
                           int                        layer_id,
                           int                        expert_id,
                           ggml_layout_mode           layout) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }
    unified_cache_key key{ type, key_id, layer_id, expert_id };

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }
    if (it->second.layout != layout) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in remove model=%llu name_hash=0x%llx have=%d want=%d\n",
                       (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                       (int) it->second.layout, (int) layout);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache layout mismatch");
        }
        return;
    }
    if (it->second.state == cache_entry_state::IN_PROGRESS) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] remove skipped: entry in progress model=%llu name_hash=0x%llx\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
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
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it != entries_.end()) {
        if (entry_it->second.layout != layout) {
            // Layout changed since caller's last lookup (PP→TG switch).
            // Pin with the entry's current layout — caller still gets protection.
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin layout mismatch model=%llu name_hash=0x%llx have=%d want=%d — pinning with current layout\n",
                            (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                            (int) entry_it->second.layout, (int) layout);
        }
        entry_it->second.pinned = true;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                        (int) entry_it->second.layout);
    }
}

void unified_cache::unpin(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it != entries_.end()) {
        if (entry_it->second.layout != layout) {
            // Entry layout changed (e.g. PP→TG layout switch).  Unpin anyway
            // since the caller's pinned handle is stale but the entry must be
            // released for future eviction.
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin layout mismatch model=%llu name_hash=0x%llx have=%d want=%d — unpinning anyway\n",
                            (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                            (int) entry_it->second.layout, (int) layout);
        }
        entry_it->second.pinned = false;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                        (int) entry_it->second.layout);
    }
}

void unified_cache::unpin_experts() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    for (auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT) {
            pair.second.pinned = false;
        }
    }
}

void unified_cache::unpin_all() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    for (auto & pair : entries_) {
        pair.second.pinned = false;
    }
}

bool unified_cache::is_pinned(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) const {
    if (!key_id.valid) {
        return false;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto                                id_it = id_to_key_.find(key_id);
    if (id_it == id_to_key_.end()) {
        return false;
    }
    auto entry_it = entries_.find(id_it->second);
    if (entry_it == entries_.end()) {
        return false;
    }
    if (entry_it->second.layout != layout) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in is_pinned model=%llu name_hash=0x%llx have=%d want=%d\n",
                       (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                       (int) entry_it->second.layout, (int) layout);
        if (cache_assert_enabled()) {
            GGML_ABORT("unified_cache layout mismatch");
        }
        return false;
    }
    return entry_it->second.pinned;
}

// === Bulk Pinning Implementation ===

int unified_cache::pin_layer_weights(int layer_id, const layer_weight_set & weights, ggml_layout_mode layout) {
    int pinned = 0;

    // Helper lambda to try pinning a single key
    auto try_pin = [&](const ggml_sycl_cache_id & key) {
        if (!key.valid) {
            return;
        }
        // Use fast path to check if entry exists with correct layout
        void * ptr = try_get_cached_fast(key, layout);
        if (ptr) {
            pin(key, layout);
            pinned++;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] bulk pin layer=%d model=%llu name_hash=0x%llx layout=%d\n", layer_id,
                            (unsigned long long) key.model_id, (unsigned long long) key.name_hash, (int) layout);
        }
    };

    // Pin all weights in the set
    try_pin(weights.attn_norm);
    try_pin(weights.q_proj);
    try_pin(weights.k_proj);
    try_pin(weights.v_proj);
    try_pin(weights.o_proj);
    try_pin(weights.ffn_norm);
    try_pin(weights.gate_proj);
    try_pin(weights.up_proj);
    try_pin(weights.down_proj);
    // Optional fused weights
    try_pin(weights.attn_qkv_proj);
    try_pin(weights.ffn_gate_up_proj);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin_layer_weights layer=%d pinned=%d layout=%d\n", layer_id, pinned, (int) layout);
    return pinned;
}

void unified_cache::unpin_layer_weights(int layer_id, const layer_weight_set & weights, ggml_layout_mode layout) {
    // Helper lambda to try unpinning a single key
    auto try_unpin = [&](const ggml_sycl_cache_id & key) {
        if (!key.valid) {
            return;
        }
        unpin(key, layout);
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] bulk unpin layer=%d model=%llu name_hash=0x%llx layout=%d\n", layer_id,
                        (unsigned long long) key.model_id, (unsigned long long) key.name_hash, (int) layout);
    };

    // Unpin all weights in the set
    try_unpin(weights.attn_norm);
    try_unpin(weights.q_proj);
    try_unpin(weights.k_proj);
    try_unpin(weights.v_proj);
    try_unpin(weights.o_proj);
    try_unpin(weights.ffn_norm);
    try_unpin(weights.gate_proj);
    try_unpin(weights.up_proj);
    try_unpin(weights.down_proj);
    // Optional fused weights
    try_unpin(weights.attn_qkv_proj);
    try_unpin(weights.ffn_gate_up_proj);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin_layer_weights layer=%d layout=%d\n", layer_id, (int) layout);
}

int unified_cache::pin_model_weights(int                                   n_layers,
                                     const std::vector<layer_weight_set> & layers,
                                     ggml_layout_mode                      layout) {
    if (n_layers <= 0 || layers.empty()) {
        return 0;
    }

    int       total_pinned  = 0;
    const int actual_layers = std::min(n_layers, (int) layers.size());

    for (int layer_id = 0; layer_id < actual_layers; layer_id++) {
        total_pinned += pin_layer_weights(layer_id, layers[layer_id], layout);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin_model_weights n_layers=%d total_pinned=%d layout=%d\n", actual_layers,
                    total_pinned, (int) layout);
    return total_pinned;
}

// === Async Layer Prefetch Implementation ===

void unified_cache::queue_layer_prefetch(int                      layer_id,
                                         const layer_weight_set & weights,
                                         ggml_layout_mode         layout,
                                         prefetch_priority        priority) {
    // Lazily start the worker thread on first call
    if (!prefetch_started_.load()) {
        start_prefetch_worker();
    }

    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        prefetch_request            req;
        req.layer_id = layer_id;
        req.weights  = weights;
        req.layout   = layout;
        req.priority = priority;

        // HIGH priority goes to the front of the queue
        if (priority == prefetch_priority::HIGH) {
            prefetch_queue_.push_front(std::move(req));
        } else {
            prefetch_queue_.push_back(std::move(req));
        }
    }
    prefetch_cv_.notify_one();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] queue_layer_prefetch layer=%d priority=%d layout=%d\n", layer_id, (int) priority,
                    (int) layout);
}

void unified_cache::prefetch_worker_loop() {
    while (true) {
        prefetch_request req;

        // Wait for a request or shutdown signal
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            prefetch_cv_.wait_for(lock, std::chrono::seconds(2),
                                  [this] { return !prefetch_queue_.empty() || prefetch_shutdown_.load(); });

            if (prefetch_shutdown_.load() && prefetch_queue_.empty()) {
                return;
            }

            // Spurious wakeup or timeout with empty queue — loop back
            if (prefetch_queue_.empty()) {
                continue;
            }

            req = std::move(prefetch_queue_.front());
            prefetch_queue_.pop_front();
        }

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch_worker processing layer=%d layout=%d\n", req.layer_id,
                        (int) req.layout);

        // Pin all weights for this layer.
        // The weights should already be in cache from model loading; we just pin them
        // to prevent eviction during persistent kernel execution.
        int pinned = pin_layer_weights(req.layer_id, req.weights, req.layout);

        // Store the weight set and layout for later release_layer
        {
            std::lock_guard<std::mutex> lock(layer_state_mutex_);
            layer_weights_[req.layer_id] = req.weights;
            layer_layouts_[req.layer_id] = req.layout;
            layer_ready_[req.layer_id]   = true;
        }
        layer_ready_cv_.notify_all();

        GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch_worker layer=%d ready (pinned=%d)\n", req.layer_id, pinned);
    }
}

layer_weight_pointers unified_cache::await_layer(int layer_id) {
    // Wait until the layer is marked ready by the prefetch worker,
    // then read layout and weights in the same critical section to avoid TOCTOU.
    ggml_layout_mode layout;
    layer_weight_set weights;
    {
        std::unique_lock<std::mutex> lock(layer_state_mutex_);
        bool ready = layer_ready_cv_.wait_for(lock, std::chrono::seconds(5), [this, layer_id] {
            auto it = layer_ready_.find(layer_id);
            return it != layer_ready_.end() && it->second;
        });
        if (!ready) {
            GGML_LOG_WARN("[PREFETCH] await_layer %d timed out after 5s, falling back to direct lookup\n", layer_id);
            return {};
        }
        layout  = layer_layouts_[layer_id];
        weights = layer_weights_[layer_id];
    }

    // Build the result by looking up each weight pointer in the cache.
    // try_get_cached_fast uses a shared_lock on rw_mutex_ (read-only, no deadlock risk).
    layer_weight_pointers ptrs;
    ptrs.attn_norm = try_get_cached_fast(weights.attn_norm, layout);
    ptrs.q_proj    = try_get_cached_fast(weights.q_proj, layout);
    ptrs.k_proj    = try_get_cached_fast(weights.k_proj, layout);
    ptrs.v_proj    = try_get_cached_fast(weights.v_proj, layout);
    ptrs.o_proj    = try_get_cached_fast(weights.o_proj, layout);
    ptrs.ffn_norm  = try_get_cached_fast(weights.ffn_norm, layout);
    ptrs.gate_proj = try_get_cached_fast(weights.gate_proj, layout);
    ptrs.up_proj   = try_get_cached_fast(weights.up_proj, layout);
    ptrs.down_proj = try_get_cached_fast(weights.down_proj, layout);

    // Fused weight lookups (optional, zero cache_id means not set)
    if (weights.attn_qkv_proj.valid) {
        ptrs.attn_qkv_proj = try_get_cached_fast(weights.attn_qkv_proj, layout);
    }
    if (weights.ffn_gate_up_proj.valid) {
        ptrs.ffn_gate_up_proj = try_get_cached_fast(weights.ffn_gate_up_proj, layout);
    }

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] await_layer layer=%d pointers resolved\n", layer_id);
    return ptrs;
}

bool unified_cache::is_layer_ready(int layer_id) const {
    std::lock_guard<std::mutex> lock(layer_state_mutex_);
    auto                        it = layer_ready_.find(layer_id);
    return it != layer_ready_.end() && it->second;
}

void unified_cache::release_layer(int layer_id) {
    layer_weight_set weights;
    ggml_layout_mode layout;

    // Retrieve and remove the layer's tracking state
    {
        std::lock_guard<std::mutex> lock(layer_state_mutex_);
        auto                        wit = layer_weights_.find(layer_id);
        auto                        lit = layer_layouts_.find(layer_id);
        if (wit == layer_weights_.end() || lit == layer_layouts_.end()) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] release_layer layer=%d not found\n", layer_id);
            return;
        }
        weights = wit->second;
        layout  = lit->second;

        layer_ready_.erase(layer_id);
        layer_weights_.erase(wit);
        layer_layouts_.erase(lit);
    }

    // Unpin the layer weights (uses rw_mutex_ internally, safe since we dropped layer_state_mutex_)
    unpin_layer_weights(layer_id, weights, layout);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] release_layer layer=%d unpinned\n", layer_id);
}

void unified_cache::start_prefetch_worker() {
    std::lock_guard<std::mutex> lock(prefetch_lifecycle_mutex_);
    if (prefetch_started_.load()) {
        return;  // Already started
    }

    prefetch_shutdown_.store(false);
    prefetch_worker_ = std::thread([this] { prefetch_worker_loop(); });
    prefetch_started_.store(true);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch worker started\n");
}

void unified_cache::stop_prefetch_worker() {
    std::lock_guard<std::mutex> lock(prefetch_lifecycle_mutex_);
    if (!prefetch_started_.load()) {
        return;  // Never started
    }

    // Signal shutdown and wake the worker
    {
        std::lock_guard<std::mutex> qlock(prefetch_mutex_);
        prefetch_shutdown_.store(true);
    }
    prefetch_cv_.notify_one();

    // Join the worker thread with timeout
    if (prefetch_worker_.joinable()) {
        auto future = std::async(std::launch::async, [this] { prefetch_worker_.join(); });
        if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
            GGML_LOG_WARN("[UNIFIED-CACHE] prefetch worker did not exit within 5s\n");
            prefetch_worker_.detach();
        }
    }

    prefetch_started_.store(false);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch worker stopped\n");
}

size_t unified_cache::evict(size_t bytes_needed) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!g_graph_compute_active.load(std::memory_order_acquire)) {
        process_deferred_frees();
    }

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

size_t unified_cache::evict_and_flush(size_t bytes_needed) {
    // Phase 1: evict entries (defers the actual sycl::free behind barrier events)
    size_t evicted = evict(bytes_needed);
    if (evicted == 0) {
        return 0;
    }

    // Phase 2: wait on the queue so all pending operations complete, making
    //          the deferred frees eligible for processing.
    try {
        queue_.wait_and_throw();
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] evict_and_flush: queue wait failed: %s\n", e.what());
    } catch (...) {
        GGML_LOG_WARN("[UNIFIED-CACHE] evict_and_flush: queue wait failed (unknown)\n");
    }

    // Phase 3: process deferred frees — this calls sycl::free and
    //          saturating_sub_used, so used_ reflects the freed memory.
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        if (!g_graph_compute_active.load(std::memory_order_acquire)) {
            process_deferred_frees();
        }
    }

    return evicted;
}

static int eviction_tier(const unified_cache_entry & entry) {
    // Tiered eviction priority (lower = evict first):
    // -1: host-resident (already slow, evict first to reclaim tracking)
    // Derived from alloc_category_priority (lower priority number = higher VRAM
    // priority = harder to evict = HIGHER eviction tier).
    //   MoE experts: category priority 2 → inverted base 2 → tiers 4-5 (cold/hot)
    //   Dense weights: category priority 1 → inverted base 3 → tiers 6-7 (cold/hot)
    // Hot entries get +1 within their category to resist eviction.
    if (entry.host_resident) {
        return -1;  // Host-resident entries evict first (they're already slow)
    }
    const alloc_category cat = (entry.type == cache_entry_type::MOE_EXPERT)
                                   ? alloc_category::EXPERT_CACHE
                                   : alloc_category::WEIGHT;
    constexpr int k_max_priority = 4;  // max value from alloc_category_priority
    const int     inverted       = k_max_priority - alloc_category_priority(cat);
    const int     base           = inverted * 2;
    return base + (entry.hot ? 1 : 0);
}

size_t unified_cache::evict_one(size_t /* new_size */) {
    // NOTE: process_deferred_frees() was removed from here to fix a BCS CAT
    // error [18].  The race: evict_one is called from stage_expert_group
    // during prestage, which also submits BCS copies.  Processing deferred
    // frees here would call sycl::free on entries whose barrier was submitted
    // BEFORE the current batch's BCS copies — unmapping VRAM while BCS is
    // still writing to nearby pages.  Callers must drain all queues and call
    // process_deferred_frees() explicitly at safe synchronization points
    // (e.g., the prestage yield loop, finalize_pending_fills).

    // Block eviction while GPU kernels are in flight (graph_compute_impl).
    // Evicting frees VRAM that MUL_MAT_ID kernels may still reference via
    // the expert pointer table → GPU page fault → DEVICE_LOST.
    // Callers fall back to host-pinned zero-copy when eviction returns 0.
    if (g_graph_compute_active.load(std::memory_order_acquire)) {
        return 0;
    }

    unified_cache_key evict_key{};
    int               best_tier        = std::numeric_limits<int>::max();
    uint32_t          best_freq        = UINT32_MAX;    // Lower frequency = evict first (MoE only)
    int64_t           best_last_access = std::numeric_limits<int64_t>::max();
    bool              found            = false;

    for (auto & pair : entries_) {
        auto & entry = pair.second;
        if (entry.state == cache_entry_state::EVICTING) {
            continue;  // Already being evicted asynchronously
        }
        if (entry.state == cache_entry_state::IN_PROGRESS) {
            if (entry.has_ready_event && event_complete(entry.ready_event)) {
                entry.state           = cache_entry_state::READY;
                entry.has_ready_event = false;
            } else {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] evict skip: model=%llu name_hash=0x%llx layout=%d in-progress size=%zu\n",
                    (unsigned long long) pair.first.id.model_id, (unsigned long long) pair.first.id.name_hash,
                    (int) entry.layout, entry.size);
                continue;
            }
        }
        if (entry.pinned) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict skip: model=%llu name_hash=0x%llx layout=%d pinned size=%zu\n",
                            (unsigned long long) pair.first.id.model_id, (unsigned long long) pair.first.id.name_hash,
                            (int) entry.layout, entry.size);
            continue;
        }

        const int tier = eviction_tier(entry);

        // For MoE expert entries (tier 0/1): use frequency as primary signal
        // within the same tier, with last_access as tiebreaker.
        // For dense weights (tier 2/3) and host-resident (tier -1): pure LRU.
        uint32_t freq = 0;
        if (entry.type == cache_entry_type::MOE_EXPERT && entry.expert_id >= 0) {
            freq = ggml_sycl::get_expert_frequency(entry.layer_id, entry.expert_id);
        }

        bool is_better = false;
        if (tier < best_tier) {
            is_better = true;
        } else if (tier == best_tier) {
            if (entry.type == cache_entry_type::MOE_EXPERT && entry.expert_id >= 0) {
                // Frequency-weighted: evict lowest frequency first, LRU tiebreaker
                is_better = (freq < best_freq) ||
                            (freq == best_freq && entry.last_access < best_last_access);
            } else {
                // Pure LRU for dense weights and host-resident entries
                is_better = (entry.last_access < best_last_access);
            }
        }

        if (is_better) {
            best_tier        = tier;
            best_freq        = freq;
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
        int    entry_layout  = static_cast<int>(it->second.layout);
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] evict model=%llu name_hash=0x%llx layout=%d size=%zu host_resident=%d\n",
                        (unsigned long long) evict_key.id.model_id, (unsigned long long) evict_key.id.name_hash,
                        (int) it->second.layout, entry_size, host_resident ? 1 : 0);

        if (!host_resident) {
            // Check if async D2H eviction is available: device entry with a
            // transformed layout (SOA/COALESCED/XMX) worth preserving.
            const bool has_transformed_layout =
                it->second.layout != GGML_LAYOUT_AOS;
            const bool async_evict_enabled = async_evict_enabled_ && has_transformed_layout;

            if (async_evict_enabled) {
                // P7: Async D2H eviction — preserve transformed layout in host-pinned memory
                void * host_dst = nullptr;
                try {
                    host_dst = sycl::malloc_host(entry_size, queue_.get_context());
                } catch (...) {
                    host_dst = nullptr;
                }

                if (host_dst) {
                    // Issue async D2H copy via DMA queue
                    sycl::queue & dq = get_dma_queue();
                    sycl::event   evt;
                    try {
                        evt = dq.memcpy(host_dst, ptr, entry_size);
                    } catch (...) {
                        sycl::free(host_dst, queue_.get_context());
                        host_dst = nullptr;
                    }

                    if (host_dst) {
                        // Mark entry as EVICTING — VRAM stays occupied until finalize
                        it->second.state              = cache_entry_state::EVICTING;
                        it->second.eviction_event     = evt;
                        it->second.has_eviction_event = true;
                        it->second.eviction_host_ptr  = host_dst;
                        it->second.pinned             = true;  // Prevent re-eviction

                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] async evict started: model=%llu name_hash=0x%llx layout=%d "
                            "size=%zu\n",
                            (unsigned long long) evict_key.id.model_id,
                            (unsigned long long) evict_key.id.name_hash,
                            entry_layout, entry_size);

                        // Return 0: VRAM not yet freed. Caller should call finalize_evictions()
                        // or retry with another entry. VRAM is reclaimed asynchronously.
                        // We still report success so the caller knows progress was made.
                        evicted_bytes = 0;
                        // Increment eviction counter so callers can poll finalize
                        evictions_in_flight_++;
                        has_evictions_.store(true, std::memory_order_release);

                        // Return entry_size to indicate the eviction is in progress
                        // even though VRAM hasn't been freed yet. The caller can
                        // call finalize_evictions() to reclaim after DMA completes.
                        return entry_size;
                    }
                }
            }

            // Synchronous eviction fallback (original path)
            // Always defer device memory frees.  Direct sycl::free inside evict_one
            // causes BCS CAT errors: eviction during prestage unmaps VRAM pages
            // while concurrent BCS copies (from stage_expert_group in the same
            // prestage loop) are still writing to nearby pages in the same L0
            // allocation region.  Deferred frees are processed at safe sync points
            // (after queue drains) where no BCS work is in-flight.
            const bool is_arena = arena_.owns(ptr);
            const bool is_pool  = !is_arena && layout_pool_ && layout_pool_->owns(ptr);
            if (is_arena) {
                // Arena entries: reclaim space in weight zone free-list.
                size_t offset = arena_.ptr_to_offset(ptr);
                if (offset != SIZE_MAX) {
                    arena_.weight_reclaim(offset, entry_size);
                }
                // No budget adjustment — arena bytes stay in used_ until arena is destroyed.
            } else if (!is_pool) {
                enqueue_deferred_free(ptr, entry_size);
            } else {
                // Pool entries: memory stays in pool, just update accounting
                saturating_sub_used(entry_size);
            }
            has_evictions_.store(true, std::memory_order_release);
        }
        // Note: For host-resident entries, we just remove tracking here.
        // The host_cache still owns the memory and will evict it via its own LRU policy.

        // Remove from lookup
        id_to_key_.erase(evict_key.id);

        // Remove from entries — invalidates iterator, must not dereference `it` after this
        entries_.erase(it);

        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] Evicted: model=%llu name_hash=0x%llx layout=%d %.2f MB (used=%.1f/%.1f MB) "
            "host_resident=%d\n",
            (unsigned long long) evict_key.id.model_id, (unsigned long long) evict_key.id.name_hash,
            entry_layout, entry_size / (1024.0f * 1024.0f), used_.load() / (1024.0f * 1024.0f),
            budget_ / (1024.0f * 1024.0f), host_resident ? 1 : 0);
        evicted_bytes = host_resident ? 0 : entry_size;  // Only count device bytes freed
    }

    return evicted_bytes;
}

// Internal: caller MUST already hold rw_mutex_ (unique).
size_t unified_cache::finalize_evictions_locked() {
    if (evictions_in_flight_.load(std::memory_order_relaxed) == 0) {
        return 0;
    }

    std::vector<unified_cache_key> finalized_keys;
    finalized_keys.reserve(4);

    for (auto & pair : entries_) {
        auto & entry = pair.second;
        if (entry.state != cache_entry_state::EVICTING || !entry.has_eviction_event) {
            continue;
        }

        // Check if D2H copy is complete
        if (!event_complete(entry.eviction_event)) {
            continue;
        }

        // DMA complete — adopt into host_cache
        auto * hcache = try_get_host_cache();
        if (hcache && entry.eviction_host_ptr) {
            const cache_layout_xmx_info * xmx_ptr =
                (entry.xmx_info.tile_n > 0) ? &entry.xmx_info : nullptr;
            bool adopted = hcache->adopt_evicted(
                pair.first.id, entry.eviction_host_ptr, entry.size,
                entry.type, entry.layer_id, entry.expert_id,
                entry.layout, xmx_ptr);
            if (!adopted) {
                // Host cache full — free the buffer
                try { sycl::free(entry.eviction_host_ptr, queue_.get_context()); } catch (...) {}
            }
        } else if (entry.eviction_host_ptr) {
            try { sycl::free(entry.eviction_host_ptr, queue_.get_context()); } catch (...) {}
        }

        // Reclaim device VRAM
        void * ptr        = entry.device_ptr;
        size_t entry_size = entry.size;
        const bool is_arena = arena_.owns(ptr);
        const bool is_pool  = !is_arena && layout_pool_ && layout_pool_->owns(ptr);
        if (is_arena) {
            size_t offset = arena_.ptr_to_offset(ptr);
            if (offset != SIZE_MAX) {
                arena_.weight_reclaim(offset, entry_size);
            }
        } else if (!is_pool) {
            enqueue_deferred_free(ptr, entry_size);
        } else {
            saturating_sub_used(entry_size);
        }

        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] async evict finalized: model=%llu name_hash=0x%llx layout=%d size=%zu\n",
            (unsigned long long) pair.first.id.model_id, (unsigned long long) pair.first.id.name_hash,
            (int) entry.layout, entry_size);

        finalized_keys.push_back(pair.first);
    }

    // Remove finalized entries
    for (const auto & key : finalized_keys) {
        id_to_key_.erase(key.id);
        entries_.erase(key);
        evictions_in_flight_--;
    }

    return finalized_keys.size();
}

size_t unified_cache::finalize_evictions() {
    // Poll in-flight async D2H evictions.  For each completed one:
    // 1. Adopt the host-pinned buffer into host_cache (preserves layout)
    // 2. Reclaim VRAM (arena weight_reclaim or deferred free)
    // 3. Remove entry from device cache
    if (evictions_in_flight_.load(std::memory_order_relaxed) == 0) {
        return 0;
    }

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    return finalize_evictions_locked();
}

void * unified_cache::promote_to_device(const unified_cache_key & key, size_t size) {
    // Re-promote a weight from host cache back to device VRAM.
    // The host_cache holds a preserved copy with the original transformed layout.
    auto * hcache = try_get_host_cache();
    if (!hcache) {
        return nullptr;
    }

    // Look up in host cache
    void * host_ptr = hcache->get(key.id, key.type, key.layer_id, key.expert_id,
                                  GGML_LAYOUT_AOS);  // Try AOS first
    ggml_layout_mode found_layout = GGML_LAYOUT_AOS;

    // Try transformed layouts if AOS not found
    if (!host_ptr) {
        const ggml_layout_mode layouts[] = { GGML_LAYOUT_SOA, GGML_LAYOUT_COALESCED };
        for (auto layout : layouts) {
            host_ptr = hcache->get(key.id, key.type, key.layer_id, key.expert_id, layout);
            if (host_ptr) {
                found_layout = layout;
                break;
            }
        }
    }

    if (!host_ptr) {
        return nullptr;  // Not in host cache
    }

    // Ensure VRAM is available
    if (used_.load() + size > budget_) {
        size_t freed = evict(size - available());
        if (freed == 0 && used_.load() + size > budget_) {
            return nullptr;  // Cannot make room
        }
    }

    // Allocate device memory (arena or direct)
    void * device_ptr = nullptr;
    bool   from_arena = false;
    if (arena_.active()) {
        device_ptr = arena_.zone_alloc(vram_zone_id::WEIGHT, size);
        from_arena = (device_ptr != nullptr);
    }
    if (!device_ptr) {
        try {
            device_ptr = sycl::malloc_device(size, queue_);
        } catch (...) {
            return nullptr;
        }
        if (device_ptr) {
            used_.fetch_add(size, std::memory_order_relaxed);
        }
    }

    if (!device_ptr) {
        return nullptr;
    }

    // Issue async H2D copy via DMA queue
    sycl::queue & dq = get_dma_queue();
    sycl::event   evt;
    try {
        evt = dq.memcpy(device_ptr, host_ptr, size);
    } catch (...) {
        if (!from_arena) {
            enqueue_deferred_free(device_ptr, size);
        }
        return nullptr;
    }

    // Create cache entry as IN_PROGRESS
    unified_cache_entry entry{};
    entry.device_ptr      = device_ptr;
    entry.src_ptr         = nullptr;
    entry.content_hash    = 0;
    entry.size            = size;
    entry.type            = key.type;
    entry.layer_id        = key.layer_id;
    entry.expert_id       = key.expert_id;
    entry.layout          = found_layout;
    entry.access_count    = 1;
    entry.last_access     = time_++;
    entry.pinned          = false;
    entry.hot             = false;
    entry.state           = cache_entry_state::IN_PROGRESS;
    entry.has_ready_event = true;
    entry.ready_event     = evt;
    entry.host_resident   = false;
    entry.location        = cache_location::DEVICE;
    entry.pool_allocated  = false;

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    entries_[key]       = std::move(entry);
    id_to_key_[key.id] = key;

    GGML_SYCL_DEBUG(
        "[UNIFIED-CACHE] promote_to_device: model=%llu name_hash=0x%llx layout=%d size=%zu\n",
        (unsigned long long) key.id.model_id, (unsigned long long) key.id.name_hash,
        (int) found_layout, size);

    return device_ptr;
}

float unified_cache::compute_score(const unified_cache_entry & entry) const {
    int64_t age        = time_.load() - entry.last_access;
    float   decay      = std::exp(-DECAY_ALPHA * static_cast<float>(age));
    float   base_score = static_cast<float>(entry.access_count) * decay;

    // Higher VRAM priority (lower priority number) → higher score → harder to evict.
    // Dense weights (priority 1) get more boost than MoE experts (priority 2).
    const alloc_category cat = (entry.type == cache_entry_type::MOE_EXPERT)
                                   ? alloc_category::EXPERT_CACHE
                                   : alloc_category::WEIGHT;
    constexpr int k_max_priority = 4;
    const float   priority_boost = static_cast<float>(k_max_priority - alloc_category_priority(cat) + 1);
    base_score *= priority_boost;  // WEIGHT → 4x, EXPERT_CACHE → 3x

    if (entry.type == cache_entry_type::DENSE_WEIGHT) {
        return base_score;
    }
    if (entry.hot) {
        constexpr float k_hot_boost = 1.5f;
        return base_score * k_hot_boost;
    }
    // Boost MoE experts with high popularity (low rank = more popular).
    // This makes popular experts resist VRAM eviction after warmup profiling.
    if (entry.type == cache_entry_type::MOE_EXPERT && entry.layer_id >= 0 && entry.expert_id >= 0) {
        if (is_expert_popularity_initialized()) {
            int pop_rank = get_expert_popularity_rank(entry.layer_id, entry.expert_id);
            if (pop_rank >= 0) {
                int boost_slots = 4;
                if (pop_rank < boost_slots) {
                    float boost = static_cast<float>(boost_slots - pop_rank);
                    base_score *= (1.0f + boost);
                }
            }
        }
    }
    return base_score;
}

sycl::event unified_cache::copy_to_device(void * dst, const void * src, size_t size) {
    // Host-pinned USM can be read directly by the GPU via DMA — skip staging.
    const sycl::usm::alloc src_type = ggml_sycl_get_alloc_type(src);
    if (src_type == sycl::usm::alloc::host || src_type == sycl::usm::alloc::device) {
        return queue_.memcpy(dst, src, size);
    }
    // Use staging buffer for mmap'd / non-USM data
    if (staging_ && size <= staging_size_) {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        // Copy mmap -> staging (may trigger page fault)
        std::memcpy(staging_, src, size);
        // Copy staging -> device — return event instead of blocking
        return queue_.memcpy(dst, staging_, size);
    } else if (staging_) {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        // Chunked transfer for large entries — must serialize chunks through
        // the single staging buffer, so each chunk waits on prior via depends_on.
        const char * src_ptr   = static_cast<const char *>(src);
        char *       dst_ptr   = static_cast<char *>(dst);
        size_t       remaining = size;
        sycl::event  last_event;

        while (remaining > 0) {
            size_t chunk = std::min(remaining, staging_size_);
            // Must wait for previous chunk's GPU read of staging_ to complete
            // before overwriting staging_ with next chunk's memcpy.
            if (src_ptr != static_cast<const char *>(src)) {
                last_event.wait();
            }
            std::memcpy(staging_, src_ptr, chunk);
            last_event = queue_.memcpy(dst_ptr, staging_, chunk);
            src_ptr += chunk;
            dst_ptr += chunk;
            remaining -= chunk;
        }
        // Wait for the final chunk since we hold staging_mutex_ and the staging
        // buffer must not be reused until the GPU finishes reading it.
        last_event.wait();
        return sycl::event{};
    } else {
        // No staging buffer — this should not happen since staging_ is always
        // pre-allocated in the constructor.  Fall back to host_cache pinned pool
        // to avoid runtime sycl::malloc_host.
        auto * hcache = try_get_host_cache();
        void * temp = hcache ? hcache->allocate_pinned_runtime(size, 64) : nullptr;
        if (temp) {
            std::memcpy(temp, src, size);
            sycl::event evt = queue_.memcpy(dst, temp, size);
            // Must wait before freeing temp buffer back to pool
            evt.wait();
            if (hcache) {
                hcache->free_pinned_runtime(temp, size);
            }
            return sycl::event{};
        } else {
            GGML_LOG_ERROR("[UNIFIED-CACHE] copy_to_device: no staging buffer and pinned pool exhausted\n");
            return sycl::event{};
        }
    }
}

sycl::event unified_cache::copy_to_device_async(void *                           dst,
                                                const void *                     src,
                                                size_t                           size,
                                                const std::vector<sycl::event> & deps,
                                                sycl::queue *                    override_q) {
    if (!src || !dst) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] copy_to_device_async: null pointer (src=%p dst=%p size=%zu)\n", src, dst, size);
        return sycl::event{};
    }
    const sycl::usm::alloc src_type = ggml_sycl_get_alloc_type(src);
    const sycl::usm::alloc dst_type = ggml_sycl_get_alloc_type(dst);
    if (g_ggml_sycl_debug >= 2 || copy_trace_enabled()) {
        GGML_LOG_INFO("[SYCL] copy_to_device_async ptr types: dst=%p type=%d src=%p type=%d size=%zu\n", dst,
                      (int) dst_type, src, (int) src_type, size);
        if (copy_trace_enabled()) {
            fflush(stderr);
        }
    }
    if (dst_type == sycl::usm::alloc::unknown && cache_assert_enabled()) {
        GGML_ABORT("copy_to_device_async called with non-USM destination");
    }
    // Route memcpy through override queue when provided (e.g. BCS queue for
    // expert prefetch).  Falls back to the cache's internal queue_ otherwise.
    sycl::queue & q = override_q ? *override_q : queue_;
    if (copy_to_device_sync_enabled()) {
        for (const auto & dep : deps) {
            const_cast<sycl::event &>(dep).wait();
        }
        copy_to_device(dst, src, size).wait();
        return submit_barrier_all();
    }

    // Stage non-USM source memory through host-pinned buffer.
    // This handles:
    // - unknown: mmap'd or non-USM pointers (must stage — GPU cannot DMA)
    // - shared: can fail on Level Zero if allocated on different context
    // Host-pinned (sycl::usm::alloc::host) and device sources skip staging —
    // the GPU can DMA directly from host-pinned USM via queue.memcpy.
    // This is critical for large tensors (e.g. 615 MB token_embd.weight in
    // 120B models) where staging through intermediate buffers can segfault.
    const bool needs_staging = (src_type != sycl::usm::alloc::device &&
                                src_type != sycl::usm::alloc::host);
    if (needs_staging) {
        // Non-USM source pointers are staged through reusable host-pinned chunks.
        constexpr size_t         k_fallback_chunk = 64 * 1024 * 1024;
        const size_t             chunk_size = std::min(size, staging_size_ > 0 ? staging_size_ : k_fallback_chunk);
        const char *             src_ptr    = static_cast<const char *>(src);
        char *                   dst_ptr    = static_cast<char *>(dst);
        size_t                   remaining  = size;
        sycl::event              last;
        std::vector<sycl::event> chain = deps;

        while (remaining > 0) {
            const size_t chunk = std::min(remaining, chunk_size);

            // Acquire a pre-allocated staging slot.  Slots are allocated once at
            // cache construction — no sycl::malloc_host during inference.
            size_t slot_idx = std::numeric_limits<size_t>::max();
            if (copy_stage_slots_.empty()) {
                throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                      "Cannot copy non-USM pointer to device: no staging slots pre-allocated");
            }
            while (slot_idx == std::numeric_limits<size_t>::max()) {
                sycl::event wait_evt{};
                bool        has_wait_evt = false;

                {
                    std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                    // Scan for a free slot with sufficient capacity.
                    for (size_t i = 0; i < copy_stage_slots_.size(); ++i) {
                        const size_t idx  = (copy_stage_next_ + i) % copy_stage_slots_.size();
                        auto &       slot = copy_stage_slots_[idx];
                        if (slot.capacity < chunk) {
                            continue;
                        }
                        if (slot.in_flight && !event_complete(slot.done_event)) {
                            continue;
                        }
                        slot.in_flight   = false;
                        copy_stage_next_ = (idx + 1) % std::max<size_t>(copy_stage_slots_.size(), 1);
                        slot_idx         = idx;
                        break;
                    }
                    if (slot_idx != std::numeric_limits<size_t>::max()) {
                        break;
                    }
                    // All slots busy — wait on the next one in round-robin order.
                    const size_t idx = copy_stage_next_ % copy_stage_slots_.size();
                    copy_stage_next_ = (idx + 1) % copy_stage_slots_.size();
                    auto & slot      = copy_stage_slots_[idx];
                    if (slot.in_flight) {
                        wait_evt     = slot.done_event;
                        has_wait_evt = true;
                    }
                    slot_idx = idx;
                }

                if (has_wait_evt) {
                    wait_evt.wait();
                    std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                    if (slot_idx < copy_stage_slots_.size()) {
                        copy_stage_slots_[slot_idx].in_flight = false;
                    }
                }
            }

            void * stage_ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                GGML_ASSERT(slot_idx < copy_stage_slots_.size());
                stage_ptr = copy_stage_slots_[slot_idx].ptr;
            }

            std::memcpy(stage_ptr, src_ptr, chunk);
            sycl::event ev;
            if (chain.empty()) {
                ev = q.memcpy(dst_ptr, stage_ptr, chunk);
            } else {
                ev = q.submit([&](sycl::handler & cgh) {
                    cgh.depends_on(chain);
                    cgh.memcpy(dst_ptr, stage_ptr, chunk);
                });
            }
            {
                std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                GGML_ASSERT(slot_idx < copy_stage_slots_.size());
                copy_stage_slots_[slot_idx].in_flight  = true;
                copy_stage_slots_[slot_idx].done_event = ev;
            }

            src_ptr += chunk;
            dst_ptr += chunk;
            remaining -= chunk;
            last = ev;
            chain.clear();
            chain.push_back(ev);
        }
        return last;
    }

    if (deps.empty()) {
        return q.memcpy(dst, src, size);
    }
    return q.submit([&](sycl::handler & cgh) {
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
    // Submit barrier that depends on ALL queues — cache, DMA, BCS, and compute.
    // This ensures deferred frees don't execute until in-flight work on every
    // queue has completed.  Missing any queue causes use-after-free:
    //   - compute queue: MUL_MAT_ID kernels reference expert pointer table
    //   - dma queue: CCS reorder kernels write to freshly-allocated VRAM slots
    //   - bcs queue: H2D copies write to temp VRAM or destination slots
    // Without the dma_queue_ barrier, sycl::free() in process_deferred_frees()
    // can unmap pages while dma_queue_ reorder kernels still reference VRAM,
    // causing BCS CAT errors when the L0 driver reuses freed pages under
    // high VRAM pressure (85-90% budget with 1000+ expert groups).
    std::vector<sycl::event> no_deps;
    std::vector<sycl::event> deps;
    try {
        deps.push_back(queue_.ext_oneapi_submit_barrier(no_deps));
    } catch (...) {}
    if (dma_queue_) {
        try {
            deps.push_back(dma_queue_->ext_oneapi_submit_barrier(no_deps));
        } catch (...) {}
    }
    if (bcs_queue_) {
        try {
            deps.push_back(bcs_queue_->ext_oneapi_submit_barrier(no_deps));
        } catch (...) {}
    }
    if (compute_queue_) {
        try {
            deps.push_back(compute_queue_->ext_oneapi_submit_barrier(no_deps));
        } catch (...) {}
    }
    // Return a barrier on the cache queue that depends on all collected events
    return queue_.ext_oneapi_submit_barrier(deps);
}

void unified_cache::enqueue_deferred_free(void * ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }

    // Pool-owned and arena-owned pointers cannot be individually freed; skip the
    // deferred free entirely to avoid unnecessary barrier events and invalid sycl::free() calls.
    if (layout_pool_ && layout_pool_->owns(ptr)) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] skipping deferred free for pool-owned ptr=%p size=%zu\n", ptr, size);
        return;
    }
    if (arena_.owns(ptr)) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] skipping deferred free for arena-owned ptr=%p size=%zu\n", ptr, size);
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
    // P7: finalize any completed async D2H evictions first.
    // This reclaims VRAM from entries whose D2H copies have completed.
    // NOTE: Use _locked variant — caller already holds rw_mutex_.
    if (evictions_in_flight_.load(std::memory_order_relaxed) > 0) {
        finalize_evictions_locked();
    }

    auto it = deferred_frees_.begin();
    while (it != deferred_frees_.end()) {
        const bool ready = !it->has_event || event_complete(it->event);
        if (!ready) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free pending: ptr=%p size=%zu\n", it->ptr, it->size);
            ++it;
            continue;
        }

        if (it->ptr) {
            const bool is_arena = arena_.owns(it->ptr);
            const bool is_pool  = !is_arena && layout_pool_ && layout_pool_->owns(it->ptr);
            if (is_arena) {
                // Arena entries: just remove from deferred list, no sycl::free needed.
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free skip arena ptr=%p size=%zu\n",
                                it->ptr, it->size);
            } else if (!is_pool) {
                if (!it->has_event) {
                    // Instead of queue_.wait() under rw_mutex_ (deadlock risk),
                    // submit a barrier event and defer to the next cycle.
                    try {
                        it->event     = submit_barrier_all();
                        it->has_event = true;
                    } catch (...) {
                        // If barrier submission fails, fall back to skipping
                    }
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free: added barrier for ptr=%p\n", it->ptr);
                    ++it;
                    continue;
                }
                try {
                    sycl::free(it->ptr, queue_);
                } catch (...) {
                }
                saturating_sub_used(it->size);
            }
            // Pool entries: used_ stays at chunk level, memory stays in pool
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] deferred free done: ptr=%p size=%zu pool=%d\n", it->ptr, it->size,
                            is_pool ? 1 : 0);
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
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin check model=%llu name_hash=0x%llx layout=%d has_event=%d ready=%d\n",
                            (unsigned long long) pin_it->key.model_id, (unsigned long long) pin_it->key.name_hash,
                            (int) pin_it->layout, pin_it->has_event ? 1 : 0, ready ? 1 : 0);
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
                    // Rate-limit: log once, then suppress. Common for MoE models where
                    // MMVQ unpins with AOS but unified cache stores SOA/COALESCED for experts.
                    static std::atomic<int> unpin_mismatch_count{ 0 };
                    int                     count = unpin_mismatch_count.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (count == 1) {
                        GGML_LOG_WARN(
                            "[UNIFIED-CACHE] layout mismatch in inflight unpin: have=%d want=%d "
                            "(MoE expert layout mismatch, benign — further occurrences suppressed)\n",
                            (int) entry_it->second.layout, (int) pin_it->layout);
                    }
                    if (cache_assert_enabled()) {
                        GGML_ABORT("unified_cache layout mismatch");
                    }
                } else {
                    entry_it->second.pinned = false;
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] in-flight unpin model=%llu name_hash=0x%llx layout=%d\n",
                                    (unsigned long long) pin_it->key.model_id,
                                    (unsigned long long) pin_it->key.name_hash, (int) pin_it->layout);
                }
            }
        }
        pin_it = inflight_unpins_.erase(pin_it);
    }
}

bool unified_cache::has_pending_deferred_frees() const {
    return !deferred_frees_.empty() || !deferred_host_frees_.empty();
}

size_t unified_cache::dense_count() const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              count = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::DENSE_WEIGHT) {
            count++;
        }
    }
    return count;
}

size_t unified_cache::expert_count() const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              count = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == cache_entry_type::MOE_EXPERT) {
            count++;
        }
    }
    return count;
}

size_t unified_cache::used_bytes(cache_entry_type type) const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              total = 0;
    for (const auto & pair : entries_) {
        if (pair.second.type == type) {
            total += pair.second.size;
        }
    }
    return total;
}

size_t unified_cache::get_layer_vram_bytes(int layer_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              total = 0;
    for (const auto & [key, entry] : entries_) {
        if (entry.layer_id == layer_id &&
            entry.state == cache_entry_state::READY &&
            !entry.host_resident &&
            entry.location != cache_location::HOST_MMAP &&
            entry.location != cache_location::HOST_PINNED) {
            total += entry.size;
        }
    }
    return total;
}

size_t unified_cache::evictable_expert_bytes() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    size_t                              total = 0;
    for (const auto & [key, entry] : entries_) {
        if (entry.type == cache_entry_type::MOE_EXPERT &&
            entry.state == cache_entry_state::READY &&
            !entry.pinned && !entry.host_resident) {
            total += entry.size;
        }
    }
    return total;
}

void unified_cache::print_stats() const {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    size_t total = hits_.load() + misses_.load();
    float  rate  = total > 0 ? 100.0f * hits_.load() / total : 0.0f;

    size_t        dense = 0, experts = 0;
    size_t        dense_bytes = 0, expert_bytes = 0;
    constexpr int layout_count                = GGML_LAYOUT_ONEDNN_WOQ + 1;
    size_t        layout_counts[layout_count] = {};
    size_t        layout_bytes[layout_count]  = {};
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
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    bool                                ok = true;

    for (const auto & pair : entries_) {
        const auto & key   = pair.first;
        const auto & entry = pair.second;
        auto         it    = id_to_key_.find(key.id);
        if (it == id_to_key_.end() || !(it->second == key)) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] validate: id_to_key mismatch model=%llu name_hash=0x%llx\n",
                            (unsigned long long) key.id.model_id, (unsigned long long) key.id.name_hash);
            ok = false;
        }
        if (!entry.device_ptr || entry.size == 0) {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] validate: entry missing data model=%llu name_hash=0x%llx layout=%d size=%zu\n",
                (unsigned long long) key.id.model_id, (unsigned long long) key.id.name_hash, (int) entry.layout,
                entry.size);
            ok = false;
        }
    }

    for (const auto & pair : id_to_key_) {
        auto it = entries_.find(pair.second);
        if (it == entries_.end()) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] validate: dangling id_to_key entry model=%llu name_hash=0x%llx\n",
                            (unsigned long long) pair.first.model_id, (unsigned long long) pair.first.name_hash);
            ok = false;
        }
    }

    return ok;
}

void unified_cache::update_reserved_bytes(size_t reserved_bytes) {
    size_t effective_budget = 0;
    {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
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
            if (!budget_exceeded_) {
                GGML_LOG_WARN(
                    "[UNIFIED-CACHE] Budget exceeded: used %.1f MB > budget %.1f MB, "
                    "eviction exhausted (reserved %.1f MB)\n",
                    used / (1024.0f * 1024.0f), budget_ / (1024.0f * 1024.0f), reserved_ / (1024.0f * 1024.0f));
            }
            budget_exceeded_ = true;
        } else {
            budget_exceeded_ = false;
        }
    }
}

void unified_cache::unpin_on_event(const ggml_sycl_cache_id & key_id,
                                   ggml_layout_mode           layout,
                                   const sycl::event &        event) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    inflight_unpin_entry                entry{};
    entry.key       = key_id;
    entry.layout    = layout;
    entry.event     = event;
    entry.has_event = true;
    inflight_unpins_.push_back(entry);
    if (g_ggml_sycl_debug) {
        const bool ready = entry.has_event ? event_complete(entry.event) : true;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] in-flight pin model=%llu name_hash=0x%llx layout=%d ready=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) layout,
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
        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] DMA staging pool mismatch: have=%zu x %.1f MB, need=%zu x %.1f MB; reallocating\n",
            dma_buffer_count_, dma_slice_bytes_ / (1024.0 * 1024.0), count, slice_bytes / (1024.0 * 1024.0));
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
            ptr = ggml_sycl_malloc_device_raw(slice_bytes, queue_, "unified_cache:dma_stage");
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

unified_cache::dma_stream_result unified_cache::stream_dma(const cache_ptr_view &           src,
                                                           size_t                           total_bytes,
                                                           size_t                           slice_bytes,
                                                           size_t                           buffer_count,
                                                           dma_stream_slice_fn              slice_fn,
                                                           const void *                     ctx,
                                                           const std::vector<sycl::event> & deps,
                                                           dma_stream_copy_fn               copy_fn) {
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
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA stream: ptr=%p bytes=%zu slice=%.1f MB buffers=%zu loc=%d type=%d\n", src.ptr,
                    bytes, slice_bytes / (1024.0 * 1024.0), buffer_count, static_cast<int>(src.location),
                    static_cast<int>(src.type));

    if (src.location == cache_location::DEVICE) {
        result.event  = slice_fn(queue_, src.ptr, bytes, 0, ctx, deps);
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
                copy_evt = copy_to_device_async(staging.buffers[slot], static_cast<const char *>(src.ptr) + offset, cur,
                                                copy_deps);
            } else {
                copy_evt =
                    queue_.memcpy(staging.buffers[slot], static_cast<const char *>(src.ptr) + offset, cur, copy_deps);
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

        buffer_events[slot]    = kernel_evt;
        buffer_has_event[slot] = true;
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
    result.ok = true;
    if (result.used_mmap_direct && !result.mmap_direct_failed) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] DMA mmap direct ok: slices=%zu bytes=%zu\n", result.slices, bytes);
    }
    return result;
}

void unified_cache::set_hot(const ggml_sycl_cache_id & key_id,
                            cache_entry_type           type,
                            int                        layer_id,
                            int                        expert_id,
                            ggml_layout_mode           layout,
                            bool                       hot) {
    if (!key_id.valid) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    unified_cache_key                   key{ type, key_id, layer_id, expert_id };
    auto                                it = entries_.find(key);
    if (it != entries_.end()) {
        if (it->second.layout != layout) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in set_hot model=%llu name_hash=0x%llx have=%d want=%d\n",
                           (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                           (int) it->second.layout, (int) layout);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return;
        }
        it->second.hot = hot;
    }
}

void unified_cache::clear_hot_experts(int layer_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
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

// Helper: Get device ID from queue.
// Uses gpu_dpct_ids[] (pre-scheduler-hiding GPU map) for secondary devices that
// are hidden from the scheduler.  The scheduler-filtered map's identity fallback
// is wrong when non-GPU devices are interleaved in dpct enumeration.
static int get_device_id_from_queue(sycl::queue & queue) {
    try {
        sycl::device   dev  = queue.get_device();
        const auto &   info = ggml_sycl_info();
        // First: check all physical GPUs via gpu_dpct_ids[] (pre-hiding map).
        // This correctly finds secondary GPUs that were hidden from the scheduler.
        for (int i = 0; i < info.total_gpu_count && i < GGML_SYCL_MAX_DEVICES; i++) {
            if (ggml_sycl_get_gpu_device(i) == dev) {
                return i;
            }
        }
        // Fallback: check all dpct devices via scheduler-filtered map.
        int device_count = dpct::dev_mgr::instance().device_count();
        for (int i = 0; i < device_count; i++) {
            if (ggml_sycl_get_device(i) == dev) {
                return i;
            }
        }
    } catch (...) {
    }
    return dpct::dev_mgr::instance().current_device_id();
}

// Helper: Resolve effective device ID (GLOBAL mode maps everything to device 0)
static int resolve_effective_device(int device) {
    unified_cache_mode mode = get_effective_mode();
    int effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return -1;
    }
    return effective_device;
}

// Helper: Look up existing cache under shared (read) lock.
// Returns nullptr if cache doesn't exist yet.  Safe for hot-path use
// because g_device_caches entries are never erased during inference.
static unified_cache * get_cache_shared(int effective_device) {
    std::shared_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    auto it = g_device_caches.find(effective_device);
    if (it == g_device_caches.end() || !it->second) {
        return nullptr;
    }
    return it->second.get();
}

static size_t runtime_reserved_bytes_nolock(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    return g_runtime_reserved_bytes[device_id].load(std::memory_order_relaxed);
}

static size_t runtime_reserved_adjusted_nolock(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    const size_t total = g_runtime_reserved_bytes[device_id].load(std::memory_order_relaxed);
    const size_t base  = g_runtime_reserved_baseline[device_id].load(std::memory_order_relaxed);
    return total > base ? total - base : 0;
}

static size_t runtime_reserved_host_bytes_nolock() {
    return g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
}

// Helper: Create cache for a device.
// deferred_reserved_out: if non-null, stores the reserved bytes that the
// caller must apply via update_reserved_bytes() AFTER releasing g_cache_rw_mutex.
// This prevents a deadlock: update_reserved_bytes() → recalc → layer streaming
// → unified_cache_add_runtime_bytes() → tries to re-lock g_cache_rw_mutex.
static unified_cache * create_cache_for_device(int device_id, size_t * deferred_reserved_out = nullptr) {
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
                            dma_reserve_bytes / (1024.0 * 1024.0), buffers, slice_bytes / (1024.0 * 1024.0));
        }
    }

    // Auto-calculate budget if not set
    size_t budget                = g_unified_cache_budget;
    bool   budget_capped_to_free = false;
    if (budget == 0) {
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
        size_t base_mem = ggml_sycl_info().devices[device_id].total_vram;
        if (base_mem == 0) {
            base_mem = total_mem > 0 ? total_mem : free_mem;
        }

        int          pct     = g_unified_cache_budget_pct;
        // Allow env var override for testing host fallback paths
        const char * env_pct = getenv("GGML_SYCL_VRAM_BUDGET_PCT");
        if (env_pct) {
            pct = std::atoi(env_pct);
            GGML_LOG_INFO("[UNIFIED-CACHE] Budget override via GGML_SYCL_VRAM_BUDGET_PCT=%d%%\n", pct);
        }
        if (pct < 1) {
            pct = 1;
        } else if (pct > 100) {
            pct = 100;
        }

        budget = static_cast<size_t>(base_mem * (static_cast<double>(pct) / 100.0));

        // Cap budget to actual free VRAM at startup to account for system overhead
        // (display compositor, driver structures, etc.) which can be 1-2 GB
        if (free_mem > 0 && budget > free_mem) {
            GGML_LOG_INFO("[UNIFIED-CACHE] Capping budget from %.1f MB to %.1f MB (actual free VRAM)\n",
                          budget / (1024.0f * 1024.0f), free_mem / (1024.0f * 1024.0f));
            budget                = free_mem;
            budget_capped_to_free = true;
        }

        char desc[256] = { 0 };
        ggml_backend_sycl_get_device_description(device_id, desc, sizeof(desc));
        GGML_LOG_INFO(
            "[UNIFIED-CACHE] Device %d (%s): total=%.1f MB free=%.1f MB budget=%.1f MB (%d%%)\n",
            device_id, desc, base_mem / (1024.0f * 1024.0f), free_mem / (1024.0f * 1024.0f),
            budget / (1024.0f * 1024.0f), pct);
    }

    const size_t staging_bytes = resolve_host_staging_bytes();
    try {
        g_device_caches[device_id]  = std::make_unique<unified_cache>(queue, budget, staging_bytes, dma_reserve_bytes);
        // Always baseline pre-existing runtime reservations so they don't
        // eat into the cache's weight budget.  Allocations that existed before
        // cache creation (DMA pre-reserve, probe residuals) are already
        // accounted for in the budget calculation — only NEW runtime
        // allocations after this point should reduce available cache space.
        const size_t reserved_total = runtime_reserved_bytes_nolock(device_id);
        const size_t baseline       = reserved_total;
        g_runtime_reserved_baseline[device_id].store(baseline, std::memory_order_relaxed);
        const size_t reserved_adjusted = runtime_reserved_adjusted_nolock(device_id);
        // Defer update_reserved_bytes to caller (after releasing g_cache_rw_mutex)
        // to avoid deadlock: update_reserved_bytes → recalc → layer streaming
        // → unified_cache_add_runtime_bytes → re-lock g_cache_rw_mutex
        if (deferred_reserved_out) {
            *deferred_reserved_out = reserved_adjusted;
        } else if (reserved_adjusted > 0) {
            g_device_caches[device_id]->update_reserved_bytes(reserved_adjusted);
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
                      total_reserve / (1024.0 * 1024.0), staging_bytes / (1024.0 * 1024.0), device_count,
                      budget / (1024.0 * 1024.0));
    }

    try {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Creating shared host_cache (device %d context)\n", device_id);
        g_host_cache_shared        = std::make_unique<host_cache>(queue, budget);
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
    unified_cache_mode mode      = get_effective_mode();
    int                device_id = (mode == unified_cache_mode::GLOBAL) ? 0 : get_device_id_from_queue(queue);

    // Fast path: check under shared lock (no contention during inference)
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        auto it = g_device_caches.find(device_id);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }
    }

    // Slow path: create cache under exclusive lock
    unified_cache * result           = nullptr;
    size_t          deferred_reserve = 0;
    {
        std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
        g_cache_mode_locked = true;

        // Double-check after acquiring write lock
        auto it = g_device_caches.find(device_id);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }

        result = create_cache_for_device(device_id, &deferred_reserve);
    }
    // Apply deferred reserved bytes outside the mutex to avoid deadlock
    if (result && deferred_reserve > 0) {
        result->update_reserved_bytes(deferred_reserve);
    }
    return result;
}

host_cache * get_host_cache(sycl::queue & queue) {
    int device_id = get_device_id_from_queue(queue);

    // Fast path: check under shared lock
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        if (g_host_cache_shared) {
            return g_host_cache_shared.get();
        }
    }

    // Slow path: create under exclusive lock
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    g_cache_mode_locked = true;

    return create_host_cache_for_device(device_id);
}

unified_cache * get_unified_cache_for_device(int device_id) {
    unified_cache_mode mode             = get_effective_mode();
    int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device_id;

    // Fast path: check under shared lock (no contention during inference)
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        auto it = g_device_caches.find(effective_device);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }
    }

    // Slow path: create cache under exclusive lock
    unified_cache * result           = nullptr;
    size_t          deferred_reserve = 0;
    {
        std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
        g_cache_mode_locked = true;

        // Double-check after acquiring write lock
        auto it = g_device_caches.find(effective_device);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }

        result = create_cache_for_device(effective_device, &deferred_reserve);
    }
    // Apply deferred reserved bytes outside the mutex to avoid deadlock
    if (result && deferred_reserve > 0) {
        result->update_reserved_bytes(deferred_reserve);
    }
    return result;
}

host_cache * get_host_cache_for_device(int device_id) {
    // Fast path: check under shared lock
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        if (g_host_cache_shared) {
            return g_host_cache_shared.get();
        }
    }

    // Slow path: create under exclusive lock
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    g_cache_mode_locked = true;

    return create_host_cache_for_device(device_id);
}

host_cache * try_get_host_cache() {
    // Lock-free read: g_host_cache_shared is set once, cleared only at shutdown when no inference is active.
    // Pointer read is atomic on x86_64. No mutex needed for the fast path.
    return g_host_cache_shared.get();
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
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    if (g_cache_mode_locked) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Budget change ignored: cache already initialized\n");
        return;
    }
    g_unified_cache_budget = bytes;
}

void set_unified_cache_budget_pct(int pct) {
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
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
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
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

alloc_tier unified_select_tier(const alloc_request & req) {
    if (req.intent.constraints.must_device) {
        return alloc_tier::DEVICE_VRAM;
    }
    if (req.intent.constraints.must_host_pinned) {
        return alloc_tier::HOST_PINNED;
    }

    if (req.intent.constraints.prefer_same_tier_as_cohort && req.intent.cohort_id && req.intent.cohort_id[0] != '\0') {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        auto                        it = g_runtime_cohort_tier.find(req.intent.cohort_id);
        if (it != g_runtime_cohort_tier.end()) {
            return it->second;
        }
    }

    if (req.device >= 0) {
        if (auto * cache = get_unified_cache_for_device(req.device)) {
            if (req.size > cache->available()) {
                return alloc_tier::HOST_PINNED;
            }
        }
    }
    return alloc_tier::DEVICE_VRAM;
}

static bool unified_free_record(const runtime_alloc_record & rec) {
    bool ok = true;
    if (rec.handle.tier == alloc_tier::DEVICE_VRAM) {
        if (rec.handle.role != alloc_role::WEIGHT) {
            unified_cache_sub_runtime_bytes(rec.handle.device, rec.handle.size, rec.handle.category);
        }
        unified_managed_sub_device_bytes(rec.handle.device, rec.handle.size);
    } else if (rec.handle.tier == alloc_tier::HOST_PINNED || rec.handle.tier == alloc_tier::MMAP_TRACKED) {
        unified_cache_sub_runtime_host_bytes(rec.handle.size);
        unified_managed_sub_host_bytes(rec.handle.size);
    }

    try {
        if (rec.handle.tier == alloc_tier::MMAP_TRACKED) {
            return true;
        }
        // Arena sub-allocations (KV zone) are freed when the arena is destroyed,
        // not individually.  Just remove the tracking record.
        if (rec.from_arena) {
            return true;
        }
        // Check if this pointer was allocated via regular malloc (pinned cap overflow).
        // The alloc_registry tracks these as MMAP type.  Using sycl::free on them
        // would crash, so use ::free instead.
        const auto * reg_info = ggml_sycl::alloc_registry::instance().lookup(rec.handle.ptr);
        if (reg_info && reg_info->type == ggml_sycl::alloc_type::MMAP) {
            ggml_sycl::alloc_registry::instance().unregister_alloc(rec.handle.ptr);
            ::free(rec.handle.ptr);
            return true;
        }
        if (rec.uses_pinned_pool) {
            if (auto * hcache = try_get_host_cache()) {
                hcache->free_pinned_runtime(rec.handle.ptr, rec.handle.size);
            }
        } else if (rec.queue != nullptr && rec.handle.ptr != nullptr) {
            sycl::free(rec.handle.ptr, *rec.queue);
        } else if (rec.handle.ptr != nullptr && rec.handle.device >= 0 && rec.handle.device < GGML_SYCL_MAX_DEVICES) {
            auto & q = ggml_sycl_get_device(rec.handle.device).default_queue();
            sycl::free(rec.handle.ptr, q);
        }
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] free failed ptr=%p size=%zu dev=%d tier=%s: %s\n", rec.handle.ptr,
                       rec.handle.size, rec.handle.device, alloc_tier_name(rec.handle.tier), e.what());
        ok = false;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] free failed ptr=%p size=%zu dev=%d tier=%s: %s\n", rec.handle.ptr,
                       rec.handle.size, rec.handle.device, alloc_tier_name(rec.handle.tier), e.what());
        ok = false;
    }

    if (!ok) {
        if (rec.handle.tier == alloc_tier::DEVICE_VRAM) {
            if (rec.handle.role != alloc_role::WEIGHT) {
                unified_cache_add_runtime_bytes(rec.handle.device, rec.handle.size, rec.handle.category);
            }
            unified_managed_add_device_bytes(rec.handle.device, rec.handle.size);
        } else if (rec.handle.tier == alloc_tier::HOST_PINNED || rec.handle.tier == alloc_tier::MMAP_TRACKED) {
            unified_cache_add_runtime_host_bytes(rec.handle.size);
            unified_managed_add_host_bytes(rec.handle.size);
        }
    }
    return ok;
}

bool unified_alloc(const alloc_request & req_in, alloc_handle * out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};

    if (req_in.size == 0) {
        return true;
    }

    alloc_request req = req_in;
    if (req.queue == nullptr && req.device >= 0 && req.device < GGML_SYCL_MAX_DEVICES) {
        req.queue = &ggml_sycl_get_device(req.device).default_queue();
    }
    if (req.queue == nullptr) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] invalid request: missing queue\n");
        return false;
    }
    if (req.device < 0) {
        req.device = get_device_id_from_queue(*req.queue);
    }
    if (req.device < 0 || req.device >= GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] invalid request: device=%d\n", req.device);
        return false;
    }

    if (req.intent.constraints.must_device && req.intent.constraints.must_host_pinned) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] invalid request: both must_device and must_host_pinned are set\n");
        return false;
    }

    runtime_category cat =
        req.intent.category == runtime_category::OTHER ? category_from_role(req.intent.role) : req.intent.category;
    alloc_tier tier = unified_select_tier(req);
    if (tier == alloc_tier::MMAP_TRACKED) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] MMAP_TRACKED runtime allocations are not supported in unified_alloc\n");
        return false;
    }
    if (req.intent.constraints.must_device) {
        tier = alloc_tier::DEVICE_VRAM;
    }
    if (req.intent.constraints.must_host_pinned) {
        tier = alloc_tier::HOST_PINNED;
    }

    const size_t alloc_size     = std::max<size_t>(req.size, 1);
    bool         reserve_device = (tier == alloc_tier::DEVICE_VRAM);
    bool         reserve_host   = (tier == alloc_tier::HOST_PINNED);

    bool   uses_pinned_pool = false;
    bool   from_arena       = false;
    void * ptr              = nullptr;
    if (tier == alloc_tier::DEVICE_VRAM) {
        // Guard against Level Zero overcommit: if this allocation would exceed
        // the device's total VRAM, fail early so the caller can retry with
        // host-pinned.  Without this check, L0 may return a valid pointer but
        // a subsequent memset/memcpy triggers DEVICE_LOST.
        if (req.device >= 0 && req.device < GGML_SYCL_MAX_DEVICES) {
            const size_t total_vram    = ggml_sycl_info().devices[req.device].total_vram;
            const size_t runtime_vram  = g_runtime_managed_reserved_bytes[req.device].load(std::memory_order_relaxed);
            // Include weight cache usage (SOA entries on device VRAM)
            size_t       cache_vram    = 0;
            auto *       cache         = get_unified_cache_for_device(req.device);
            if (cache) {
                cache_vram = cache->used();
            }
            size_t used_vram = runtime_vram + cache_vram;
            if (total_vram > 0 && used_vram + alloc_size > total_vram) {
                // Try evicting cache entries to make room before failing.
                // This is the key mechanism that lets the 120B model work:
                // model load fills VRAM with cached weights, then compute
                // buffer allocation triggers eviction to free space.
                if (cache) {
                    const size_t needed = used_vram + alloc_size - total_vram;
                    GGML_LOG_INFO("[UNIFIED-ALLOC] Device %d VRAM pressure: "
                                  "used=%.1f MB + alloc=%.1f MB > total=%.1f MB, "
                                  "evicting %.1f MB from cache\n",
                                  req.device, used_vram / (1024.0 * 1024.0),
                                  alloc_size / (1024.0 * 1024.0),
                                  total_vram / (1024.0 * 1024.0),
                                  needed / (1024.0 * 1024.0));
                    const size_t freed = cache->evict_and_flush(needed);
                    // Re-check after eviction
                    cache_vram = cache->used();
                    used_vram  = g_runtime_managed_reserved_bytes[req.device].load(std::memory_order_relaxed)
                               + cache_vram;
                    if (freed > 0) {
                        GGML_LOG_INFO("[UNIFIED-ALLOC] Evicted %.1f MB, "
                                      "used now=%.1f MB (cache=%.1f MB + runtime=%.1f MB)\n",
                                      freed / (1024.0 * 1024.0),
                                      used_vram / (1024.0 * 1024.0),
                                      cache_vram / (1024.0 * 1024.0),
                                      g_runtime_managed_reserved_bytes[req.device].load(std::memory_order_relaxed)
                                          / (1024.0 * 1024.0));
                    }
                }
                if (total_vram > 0 && used_vram + alloc_size > total_vram) {
                    GGML_SYCL_DEBUG("[UNIFIED-ALLOC] Device %d VRAM overcommit guard: "
                                    "used=%.1f MB + alloc=%.1f MB > total=%.1f MB, failing\n",
                                    req.device, used_vram / (1024.0f * 1024.0f),
                                    alloc_size / (1024.0f * 1024.0f),
                                    total_vram / (1024.0f * 1024.0f));
                    return false;
                }
            }
        }
        // P5: Route KV allocations through the arena's KV zone when active.
        // This co-locates KV cache with weights in the same pre-allocated VRAM block,
        // eliminating separate sycl::malloc_device calls during context creation.
        if (req.intent.role == alloc_role::KV && vram_arena_enabled()) {
            auto * cache = get_unified_cache_for_device(req.device);
            if (cache && cache->get_arena().active()) {
                ptr = cache->get_arena().zone_alloc(vram_zone_id::KV, alloc_size);
                if (ptr) {
                    from_arena = true;
                    GGML_SYCL_DEBUG("[UNIFIED-ALLOC] KV arena alloc: dev=%d size=%.1f MB ptr=%p\n",
                                    req.device, alloc_size / (1024.0 * 1024.0), ptr);
                }
            }
        }
        if (!ptr) {
            ptr = ggml_sycl_malloc_device_raw(alloc_size, *req.queue, "unified_alloc:device");
        }
    } else {
        // Always try the pre-allocated pinned chunk pool first (lock-free path).
        // Uses try_get_host_cache() to avoid g_cache_rw_mutex which would deadlock
        // during S1-PRELOAD when the caller already holds the cache lock.
        if (auto * hcache = try_get_host_cache()) {
            ptr              = hcache->allocate_pinned_runtime(alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
            uses_pinned_pool = (ptr != nullptr);
        }
        if (!ptr) {
            // Pinned pool exhausted — fall back to raw malloc_host.
            // After pre_allocate_all this should not happen during inference;
            // log a warning so we can identify any remaining leak paths.
            GGML_LOG_WARN("[UNIFIED-ALLOC] pinned pool exhausted for %zu bytes, "
                          "falling back to malloc_host (should not happen during inference)\n",
                          alloc_size);
            ptr = ggml_sycl_malloc_host(alloc_size, *req.queue, "unified_alloc:host");
        }
    }

    if (!ptr) {
        return false;
    }

    // Track the allocation now that it succeeded.
    // Weight buffers are the primary model data allocated by ggml framework;
    // they must NOT count against the cache budget (reserved_) because the
    // cache manages SOA/XMX layouts in the REMAINING VRAM after weights.
    // We still track them in g_runtime_managed_reserved_bytes (overcommit guard).
    if (reserve_device) {
        if (req.intent.role != alloc_role::WEIGHT) {
            unified_cache_add_runtime_bytes(req.device, alloc_size, cat);
        }
        unified_managed_add_device_bytes(req.device, alloc_size);
    } else if (reserve_host) {
        unified_cache_add_runtime_host_bytes(alloc_size);
        unified_managed_add_host_bytes(alloc_size);
    }

    runtime_alloc_record rec;
    rec.handle.ptr       = ptr;
    rec.handle.size      = alloc_size;
    rec.handle.device    = req.device;
    rec.handle.tier      = tier;
    rec.handle.role      = req.intent.role;
    rec.handle.category  = cat;
    rec.handle.alloc_id  = g_runtime_alloc_id.fetch_add(1, std::memory_order_relaxed);
    rec.queue            = req.queue;
    rec.uses_pinned_pool = uses_pinned_pool;
    rec.from_arena       = from_arena;
    if (req.intent.cohort_id && req.intent.cohort_id[0] != '\0') {
        rec.cohort_id = req.intent.cohort_id;
    }

    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        if (g_runtime_alloc_registry.find(ptr) != g_runtime_alloc_registry.end()) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] duplicate pointer registration ptr=%p size=%zu tier=%s\n", ptr, alloc_size,
                           alloc_tier_name(tier));
            if (reserve_device) {
                if (req.intent.role != alloc_role::WEIGHT) {
                    unified_cache_sub_runtime_bytes(req.device, alloc_size, cat);
                }
                unified_managed_sub_device_bytes(req.device, alloc_size);
            } else if (reserve_host) {
                unified_cache_sub_runtime_host_bytes(alloc_size);
                unified_managed_sub_host_bytes(alloc_size);
            }
            if (uses_pinned_pool) {
                if (auto * hcache = try_get_host_cache()) {
                    hcache->free_pinned_runtime(ptr, alloc_size);
                }
            } else {
                sycl::free(ptr, *req.queue);
            }
            return false;
        }
        g_runtime_alloc_registry.emplace(ptr, rec);
        if (!rec.cohort_id.empty()) {
            g_runtime_cohort_tier[rec.cohort_id] = tier;
        }
    }

    *out = rec.handle;
    offload_stats_note_alloc(tier);
    return true;
}

bool acquire_offload_buffer(const offload_buffer_request & req_in, offload_buffer_lease * out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};

    if (req_in.queue == nullptr) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] offload pool acquire failed: missing queue\n");
        return false;
    }

    offload_buffer_request req = req_in;
    if (req.device < 0) {
        req.device = get_device_id_from_queue(*req.queue);
    }
    if (req.device < 0 || req.device >= GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("[UNIFIED-ALLOC] offload pool acquire failed: invalid device=%d\n", req.device);
        return false;
    }

    const size_t alignment  = std::max<size_t>(req.alignment, 64);
    const size_t alloc_size = align_up(std::max<size_t>(req.size, 1), alignment);

    alloc_request areq{};
    areq.queue  = req.queue;
    areq.device = req.device;
    areq.size   = alloc_size;
    areq.intent = req.intent;
    if (areq.intent.role == alloc_role::OTHER) {
        areq.intent.role = alloc_role::STAGING;
    }
    if (areq.intent.category == runtime_category::OTHER) {
        areq.intent.category = runtime_category::HOST_COMPUTE;
    }
    const bool has_explicit_tier_constraint =
        areq.intent.constraints.must_host_pinned || areq.intent.constraints.must_device;

    // Default staging roles to host-pinned, but preserve explicit caller
    // constraints so compute-buffer users can intentionally target VRAM.
    if (!has_explicit_tier_constraint) {
        switch (req.role) {
            case offload_buffer_role::STAGING_SRC0:
            case offload_buffer_role::STAGING_SRC1:
            case offload_buffer_role::STAGING_DST:
            case offload_buffer_role::RETAINED_SCRATCH:
                areq.intent.constraints.must_host_pinned = true;
                break;
            default:
                break;
        }
    }

    const alloc_tier tier = unified_select_tier(areq);
    offload_pool_key key{};
    key.device    = req.device;
    key.role      = req.role;
    key.tier      = tier;
    key.category  = areq.intent.category;
    key.alignment = alignment;

    {
        std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
        auto                        it_bucket = g_offload_pool_free.find(key);
        if (it_bucket != g_offload_pool_free.end()) {
            size_t best_idx  = std::numeric_limits<size_t>::max();
            size_t best_size = std::numeric_limits<size_t>::max();
            for (size_t i = 0; i < it_bucket->second.size(); ++i) {
                void * ptr     = it_bucket->second[i];
                auto   slot_it = g_offload_pool_slots.find(ptr);
                if (slot_it == g_offload_pool_slots.end() || slot_it->second.in_use) {
                    continue;
                }
                if (slot_it->second.handle.size < alloc_size) {
                    continue;
                }
                if (slot_it->second.handle.size < best_size) {
                    best_size = slot_it->second.handle.size;
                    best_idx  = i;
                }
            }
            if (best_idx != std::numeric_limits<size_t>::max()) {
                void * ptr = it_bucket->second[best_idx];
                it_bucket->second.erase(it_bucket->second.begin() + static_cast<ptrdiff_t>(best_idx));
                if (it_bucket->second.empty()) {
                    g_offload_pool_free.erase(it_bucket);
                }

                auto slot_it = g_offload_pool_slots.find(ptr);
                GGML_ASSERT(slot_it != g_offload_pool_slots.end());
                slot_it->second.in_use   = true;
                slot_it->second.lease_id = g_offload_pool_lease_id.fetch_add(1, std::memory_order_relaxed);
                out->handle              = slot_it->second.handle;
                out->lease_id            = slot_it->second.lease_id;
                out->valid               = true;
                offload_stats_note_pool_hit();
                return true;
            }
        }
    }

    alloc_handle h{};
    if (!unified_alloc(areq, &h) || h.ptr == nullptr) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
        offload_pool_slot           slot{};
        slot.handle                 = h;
        slot.key                    = key;
        slot.in_use                 = true;
        slot.lease_id               = g_offload_pool_lease_id.fetch_add(1, std::memory_order_relaxed);
        g_offload_pool_slots[h.ptr] = slot;
        out->handle                 = h;
        out->lease_id               = slot.lease_id;
        out->valid                  = true;
    }
    offload_stats_note_pool_miss();
    return true;
}

bool release_offload_buffer(const offload_buffer_lease & lease) {
    if (!lease.valid || lease.handle.ptr == nullptr) {
        return true;
    }

    std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
    auto                        it = g_offload_pool_slots.find(lease.handle.ptr);
    if (it == g_offload_pool_slots.end()) {
        if (unified_alloc_strict_mode()) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] strict offload-pool release unknown ptr=%p lease=%llu\n", lease.handle.ptr,
                           static_cast<unsigned long long>(lease.lease_id));
        }
        return false;
    }
    if (!it->second.in_use || it->second.lease_id != lease.lease_id) {
        if (unified_alloc_strict_mode()) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] strict offload-pool stale lease ptr=%p lease=%llu active=%llu in_use=%d\n",
                           lease.handle.ptr, static_cast<unsigned long long>(lease.lease_id),
                           static_cast<unsigned long long>(it->second.lease_id), it->second.in_use ? 1 : 0);
        }
        return false;
    }

    it->second.in_use = false;
    g_offload_pool_free[it->second.key].push_back(lease.handle.ptr);
    return true;
}

void offload_buffer_pool_trim(int device) {
    std::vector<alloc_handle> free_list;
    {
        std::lock_guard<std::mutex> lock(g_offload_pool_mutex);
        for (auto it = g_offload_pool_slots.begin(); it != g_offload_pool_slots.end();) {
            const offload_pool_slot & slot = it->second;
            if (!slot.in_use && (device < 0 || slot.key.device == device)) {
                free_list.push_back(slot.handle);
                it = g_offload_pool_slots.erase(it);
            } else {
                ++it;
            }
        }
        g_offload_pool_free.clear();
        for (const auto & kv : g_offload_pool_slots) {
            if (!kv.second.in_use) {
                g_offload_pool_free[kv.second.key].push_back(kv.first);
            }
        }
    }

    for (const alloc_handle & h : free_list) {
        (void) unified_free(h);
    }
}

bool unified_lookup(void * ptr, alloc_handle * out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};
    if (ptr == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
    auto                        it = g_runtime_alloc_registry.find(ptr);
    if (it == g_runtime_alloc_registry.end()) {
        return false;
    }
    *out = it->second.handle;
    return true;
}

bool unified_free_ptr(void * ptr, int expected_device) {
    if (ptr == nullptr) {
        return true;
    }

    runtime_alloc_record rec;
    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        auto                        it = g_runtime_alloc_registry.find(ptr);
        if (it == g_runtime_alloc_registry.end()) {
            if (unified_alloc_strict_mode()) {
                GGML_LOG_ERROR("[UNIFIED-ALLOC] strict unknown free ptr=%p expected_device=%d\n", ptr, expected_device);
            }
            return false;
        }
        if (expected_device >= 0 && expected_device != it->second.handle.device) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] free device mismatch ptr=%p expected=%d actual=%d\n", ptr, expected_device,
                           it->second.handle.device);
            return false;
        }
        rec = it->second;
    }

    if (!unified_free_record(rec)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
    auto                        it = g_runtime_alloc_registry.find(ptr);
    if (it != g_runtime_alloc_registry.end() && it->second.handle.alloc_id == rec.handle.alloc_id) {
        g_runtime_alloc_registry.erase(it);
    }
    return true;
}

bool unified_free(const alloc_handle & handle) {
    if (handle.ptr == nullptr) {
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        auto                        it = g_runtime_alloc_registry.find(handle.ptr);
        if (it == g_runtime_alloc_registry.end()) {
            if (unified_alloc_strict_mode()) {
                GGML_LOG_ERROR("[UNIFIED-ALLOC] strict stale/unknown handle free ptr=%p alloc_id=%llu\n", handle.ptr,
                               static_cast<unsigned long long>(handle.alloc_id));
            }
            return false;
        }
        if (handle.alloc_id != 0 && it->second.handle.alloc_id != handle.alloc_id) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] stale handle free ptr=%p expected_alloc_id=%llu actual_alloc_id=%llu\n",
                           handle.ptr, static_cast<unsigned long long>(handle.alloc_id),
                           static_cast<unsigned long long>(it->second.handle.alloc_id));
            return false;
        }
    }

    return unified_free_ptr(handle.ptr, handle.device);
}

// ============================================================================
// Simplified allocation facade
// ============================================================================

// Map alloc_category to the internal alloc_role / runtime_category / constraints
// used by the existing unified_alloc machinery.
static void category_to_intent(alloc_category cat, alloc_intent & intent) {
    switch (cat) {
        case alloc_category::WEIGHT:
            intent.role     = alloc_role::WEIGHT;
            intent.category = runtime_category::OTHER;
            break;
        case alloc_category::KV_CACHE:
            intent.role     = alloc_role::KV;
            intent.category = runtime_category::KV_CACHE;
            break;
        case alloc_category::COMPUTE_SCRATCH:
            intent.role     = alloc_role::COMPUTE;
            intent.category = runtime_category::COMPUTE;
            break;
        case alloc_category::STAGING:
            intent.role     = alloc_role::STAGING;
            intent.category = runtime_category::STAGING;
            break;
        case alloc_category::CONTROL:
            intent.role                       = alloc_role::GRAPH_TMP;
            intent.category                   = runtime_category::GRAPH;
            intent.constraints.must_device    = true;
            break;
        case alloc_category::EXPERT_CACHE:
            intent.role     = alloc_role::EXPERT_STAGING;
            intent.category = runtime_category::EXPERT_CACHE;
            break;
    }
}

// Returns true if a category is eligible for host-pinned fallback when VRAM is full.
static bool category_allows_host_fallback(alloc_category cat) {
    return cat == alloc_category::COMPUTE_SCRATCH || cat == alloc_category::STAGING;
}

int alloc_category_priority(alloc_category cat) {
    switch (cat) {
        case alloc_category::COMPUTE_SCRATCH: return 0;  // pre-reserved arena, must stay in VRAM
        case alloc_category::KV_CACHE:        return 1;  // hot KV: latency-critical, always VRAM
        case alloc_category::WEIGHT:          return 2;  // attention weights: used every token
        case alloc_category::CONTROL:         return 2;  // tiny, must be on device
        case alloc_category::EXPERT_CACHE:    return 3;  // MoE experts: evictable via LRU/frequency
        case alloc_category::STAGING:         return 4;  // always host-ok
    }
    return 4;  // unknown → treat as staging
}

unified_alloc_result unified_cache_allocate(
    int             device,
    size_t          size,
    alloc_category  category,
    sycl::queue &   queue) {

    unified_alloc_result result{};
    if (size == 0) {
        return result;
    }

    alloc_request req{};
    req.queue  = &queue;
    req.device = device;
    req.size   = size;
    category_to_intent(category, req.intent);

    alloc_handle handle{};
    if (unified_alloc(req, &handle)) {
        result.ptr  = handle.ptr;
        result.tier = handle.tier;
        result.size = handle.size;
        return result;
    }

    // VRAM allocation failed — try host-pinned fallback for eligible categories.
    if (category_allows_host_fallback(category) && !req.intent.constraints.must_device) {
        req.intent.constraints.must_host_pinned = true;
        req.intent.constraints.must_device      = false;
        if (unified_alloc(req, &handle)) {
            result.ptr  = handle.ptr;
            result.tier = handle.tier;
            result.size = handle.size;
            return result;
        }
    }

    // All attempts failed
    return result;
}

void unified_cache_deallocate(void * ptr, int device) {
    if (ptr == nullptr) {
        return;
    }
    unified_free_ptr(ptr, device);
}

bool unified_alloc_validate_registry(int device, const char * where) {
    std::array<size_t, GGML_SYCL_MAX_DEVICES> registry_device{};
    size_t                                    registry_host = 0;
    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        for (const auto & kv : g_runtime_alloc_registry) {
            const alloc_handle & h = kv.second.handle;
            if (h.tier == alloc_tier::DEVICE_VRAM) {
                if (h.device >= 0 && h.device < GGML_SYCL_MAX_DEVICES) {
                    registry_device[h.device] += h.size;
                }
            } else if (h.tier == alloc_tier::HOST_PINNED || h.tier == alloc_tier::MMAP_TRACKED) {
                registry_host += h.size;
            }
        }
    }

    bool ok = true;
    for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
        if (device >= 0 && d != device) {
            continue;
        }
        const size_t tracked = g_runtime_managed_reserved_bytes[d].load(std::memory_order_relaxed);
        const size_t reg     = registry_device[d];
        if (tracked != reg) {
            ok = false;
            GGML_LOG_WARN("[UNIFIED-ALLOC] managed registry mismatch%s%s dev=%d tracked=%zu registry=%zu\n",
                          where ? " at " : "", where ? where : "", d, tracked, reg);
        }
    }
    const size_t tracked_host = g_runtime_managed_reserved_host_bytes.load(std::memory_order_relaxed);
    if (tracked_host != registry_host) {
        ok = false;
        GGML_LOG_WARN("[UNIFIED-ALLOC] managed registry mismatch%s%s host tracked=%zu registry=%zu\n",
                      where ? " at " : "", where ? where : "", tracked_host, registry_host);
    }
    return ok;
}

void unified_cache_set_graph_compute_active(bool active) {
    g_graph_compute_active.store(active, std::memory_order_release);
}

bool unified_cache_is_graph_compute_active() {
    return g_graph_compute_active.load(std::memory_order_acquire);
}

bool unified_cache_has_pending_deferred_frees(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return false;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return false;
    }
    return cache->has_pending_deferred_frees();
}

void unified_cache_add_runtime_bytes(int device, size_t bytes, runtime_category cat) {
    if (bytes == 0) {
        return;
    }
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return;
    }
    // Atomic counter updates — no lock needed
    const size_t new_total =
        g_runtime_reserved_bytes[effective_device].fetch_add(bytes, std::memory_order_relaxed) + bytes;
    g_runtime_cat_bytes[effective_device][static_cast<int>(cat)].fetch_add(bytes, std::memory_order_relaxed);
    GGML_SYCL_DEBUG("[BUDGET] add dev=%d(eff=%d) +%.3f MB cat=%d total=%.1f MB\n",
                    device, effective_device, bytes / (1024.0 * 1024.0),
                    static_cast<int>(cat), new_total / (1024.0 * 1024.0));
    // Look up cache under shared lock, call update_reserved_bytes outside lock
    unified_cache * cache = get_cache_shared(effective_device);
    if (cache) {
        const size_t baseline = g_runtime_reserved_baseline[effective_device].load(std::memory_order_relaxed);
        const size_t adjusted = new_total > baseline ? new_total - baseline : 0;
        cache->update_reserved_bytes(adjusted);
    }
}

void unified_cache_sub_runtime_bytes(int device, size_t bytes, runtime_category cat) {
    if (bytes == 0) {
        return;
    }
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return;
    }
    // Atomic saturating subtract via CAS loop — no lock needed
    size_t cur = g_runtime_reserved_bytes[effective_device].load(std::memory_order_relaxed);
    size_t next;
    do {
        next = cur > bytes ? cur - bytes : 0;
    } while (!g_runtime_reserved_bytes[effective_device].compare_exchange_weak(
        cur, next, std::memory_order_relaxed, std::memory_order_relaxed));
    size_t cat_cur = g_runtime_cat_bytes[effective_device][static_cast<int>(cat)].load(std::memory_order_relaxed);
    size_t cat_next;
    do {
        cat_next = cat_cur > bytes ? cat_cur - bytes : 0;
    } while (!g_runtime_cat_bytes[effective_device][static_cast<int>(cat)].compare_exchange_weak(
        cat_cur, cat_next, std::memory_order_relaxed, std::memory_order_relaxed));
    // Look up cache under shared lock, call update_reserved_bytes outside lock
    unified_cache * cache = get_cache_shared(effective_device);
    if (cache) {
        const size_t baseline = g_runtime_reserved_baseline[effective_device].load(std::memory_order_relaxed);
        const size_t adjusted = next > baseline ? next - baseline : 0;
        cache->update_reserved_bytes(adjusted);
    }
}

size_t unified_cache_get_runtime_bytes(int device) {
    // Pure atomic read — no lock needed
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    return g_runtime_reserved_bytes[effective_device].load(std::memory_order_relaxed);
}

size_t unified_cache_get_runtime_bytes_by_category(int device, runtime_category cat) {
    // Pure atomic read — no lock needed
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    if (static_cast<int>(cat) >= static_cast<int>(runtime_category::COUNT)) {
        return 0;
    }
    return g_runtime_cat_bytes[effective_device][static_cast<int>(cat)].load(std::memory_order_relaxed);
}

void unified_cache_add_runtime_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    // Atomic counter update — no exclusive lock needed
    const size_t new_total = g_runtime_reserved_host_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    // Look up host cache under shared lock for update_reserved_bytes
    host_cache * hcache = nullptr;
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        if (g_host_cache_shared) {
            hcache = g_host_cache_shared.get();
        }
    }
    if (hcache) {
        hcache->update_reserved_bytes(new_total);
    }
}

void unified_cache_sub_runtime_host_bytes(size_t bytes) {
    if (bytes == 0) {
        return;
    }
    // Atomic saturating subtract — no exclusive lock needed
    size_t cur  = g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
    size_t next = cur > bytes ? cur - bytes : 0;
    g_runtime_reserved_host_bytes.store(next, std::memory_order_relaxed);
    // Look up host cache under shared lock for update_reserved_bytes
    host_cache * hcache = nullptr;
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        if (g_host_cache_shared) {
            hcache = g_host_cache_shared.get();
        }
    }
    if (hcache) {
        hcache->update_reserved_bytes(next);
    }
}

size_t unified_cache_get_runtime_host_bytes() {
    // Pure atomic read — no lock needed
    return g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
}

// === Budget Query API ===

size_t unified_cache_available_for_compute(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }

    // Use the unified total-available view that sums BOTH budget channels
    // (weights from used_ + runtime from g_runtime_reserved_bytes).
    // Previously this only checked cache->available_for_compute() which relies
    // on reserved_ being kept in sync via update_reserved_bytes() — a TOCTOU
    // gap that could let the compute pool over-allocate when weights consumed
    // VRAM between the last update_reserved_bytes call and this query.
    //
    // We still query live VRAM as a safety cap: if the driver reports less free
    // memory than the budget says, use the lower value.
    const size_t budget_avail = unified_cache_total_available_bytes(device);
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_sycl_get_device_memory(effective_device, &free_vram, &total_vram);
    if (free_vram == 0) {
        return budget_avail;
    }
    // Reserve 256 MB headroom for driver structures, compute scratch, and
    // transient allocations that come and go during inference.
    const size_t headroom   = size_t(256) << 20;
    const size_t live_avail = free_vram > headroom ? free_vram - headroom : 0;
    return std::min(budget_avail, live_avail);
}

size_t unified_cache_total_committed_bytes(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    // Use baseline-adjusted runtime bytes: base_budget_ was computed from free
    // VRAM at init, which already excluded pre-existing runtime allocations.
    // Using raw g_runtime_reserved_bytes would double-count the baseline.
    const size_t runtime = runtime_reserved_adjusted_nolock(effective_device);
    unified_cache * cache = get_cache_shared(effective_device);
    const size_t weights = cache ? cache->weight_bytes() : 0;
    return runtime + weights;
}

size_t unified_cache_total_available_bytes(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    const size_t base      = cache->base_budget();
    const size_t committed = unified_cache_total_committed_bytes(device);
    return base > committed ? base - committed : 0;
}

size_t unified_cache_total_managed(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    return cache->base_budget();
}

size_t unified_cache_weight_bytes(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    return cache->weight_bytes();
}

size_t unified_cache_get_layer_vram_bytes(int device, int layer_id) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    return cache->get_layer_vram_bytes(layer_id);
}

size_t unified_cache_evictable_expert_bytes(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return 0;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return 0;
    }
    return cache->evictable_expert_bytes();
}

// === Budget Summary Diagnostic ===

void unified_cache_log_budget_summary(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return;
    }
    unified_cache * cache_ptr = get_cache_shared(effective_device);
    if (!cache_ptr) {
        return;
    }
    auto &       cache = *cache_ptr;
    const size_t base  = cache.base_budget();
    const size_t wt    = cache.weight_bytes();
    const size_t rt    = g_runtime_reserved_bytes[effective_device].load(std::memory_order_relaxed);
    const size_t eff   = cache.budget();
    const size_t avl   = cache.available();

    const size_t avail_for_wt = base > rt ? base - rt : 0;
    int          budget_pct   = 90;
    const char * env_pct      = std::getenv("GGML_SYCL_VRAM_BUDGET_PCT");
    if (env_pct) {
        budget_pct = std::max(1, std::min(100, std::atoi(env_pct)));
    }
    // Compute model-exceeds-VRAM directly from model size vs available budget
    const size_t model_size = ggml_sycl_get_model_size();
    size_t       moe_total_log = 0;
    int          n_exp_log = 0, n_exp_used_log = 0;
    ggml_sycl_get_moe_info(&moe_total_log, &n_exp_log, &n_exp_used_log);
    const size_t effective_model = compute_moe_effective_weight_bytes(
        model_size, moe_total_log, n_exp_log, n_exp_used_log);
    const bool exceeds = (model_size > 0) && (effective_model > avail_for_wt);

    const size_t committed = unified_cache_total_committed_bytes(device);
    const size_t total_avl = unified_cache_total_available_bytes(device);

    GGML_LOG_INFO(
        "[UNIFIED-CACHE] Budget summary for device %d:\n"
        "  Total VRAM budget:    %8.1f MB\n"
        "  Weight bytes (used_): %8.1f MB\n"
        "  Runtime reserved:     %8.1f MB\n"
        "  Total committed:      %8.1f MB  (weights + runtime)\n"
        "  Total available:      %8.1f MB  (budget - committed)\n"
        "  Effective budget:     %8.1f MB\n"
        "  Available for alloc:  %8.1f MB\n"
        "  Avail for weights:    %8.1f MB\n"
        "  Budget pct:           %8d %%\n"
        "  Model exceeds VRAM:   %8s\n",
        device, base / (1024.0f * 1024.0f), wt / (1024.0f * 1024.0f), rt / (1024.0f * 1024.0f),
        committed / (1024.0f * 1024.0f), total_avl / (1024.0f * 1024.0f),
        eff / (1024.0f * 1024.0f), avl / (1024.0f * 1024.0f), avail_for_wt / (1024.0f * 1024.0f), budget_pct,
        exceeds ? "yes" : "no");

    // Per-category runtime breakdown
    static const char * cat_names[] = { "KV_CACHE", "COMPUTE", "STAGING", "GRAPH", "HOST_COMPUTE", "EXPERT_CACHE", "OTHER" };
    GGML_LOG_INFO("[UNIFIED-CACHE] Runtime breakdown for device %d:\n", device);
    for (int c = 0; c < static_cast<int>(runtime_category::COUNT); c++) {
        const size_t cat_bytes = g_runtime_cat_bytes[effective_device][c].load(std::memory_order_relaxed);
        if (cat_bytes > 0) {
            GGML_LOG_INFO("  %-12s %8.1f MB\n", cat_names[c], cat_bytes / (1024.0f * 1024.0f));
        }
    }
    // Show untagged delta (total - sum of categories)
    size_t cat_sum = 0;
    for (int c = 0; c < static_cast<int>(runtime_category::COUNT); c++) {
        cat_sum += g_runtime_cat_bytes[effective_device][c].load(std::memory_order_relaxed);
    }
    if (rt > cat_sum + (1024 * 1024)) {  // >1 MB untagged
        GGML_LOG_INFO("  %-12s %8.1f MB (tracked outside categories)\n", "UNTAGGED",
                      (rt - cat_sum) / (1024.0f * 1024.0f));
    }

    // Validate accounting consistency:
    //   weight_bytes + available should equal effective budget
    //   (effective budget = base_budget - internal reserved)
    const size_t tolerance   = 1024 * 1024;  // 1 MB
    const size_t wt_plus_avl = wt + avl;
    if (wt_plus_avl > eff + tolerance || eff > wt_plus_avl + tolerance) {
        GGML_LOG_WARN(
            "[UNIFIED-CACHE] Accounting mismatch on device %d: "
            "weights(%.1f) + available(%.1f) = %.1f MB, "
            "but effective_budget = %.1f MB (delta = %.1f MB)\n",
            device, wt / (1024.0f * 1024.0f), avl / (1024.0f * 1024.0f), wt_plus_avl / (1024.0f * 1024.0f),
            eff / (1024.0f * 1024.0f),
            (double) (wt_plus_avl > eff ? wt_plus_avl - eff : eff - wt_plus_avl) / (1024.0 * 1024.0));
    }

    // Sanity: used_ should not exceed effective budget
    if (wt > eff + tolerance) {
        GGML_LOG_WARN(
            "[UNIFIED-CACHE] Over-allocation on device %d: "
            "weight_bytes(%.1f MB) > effective_budget(%.1f MB)\n",
            device, wt / (1024.0f * 1024.0f), eff / (1024.0f * 1024.0f));
    }

    // Diagnostic: flag if external runtime tracker diverges from internal reserved
    const size_t implied_reserved = (base > eff) ? (base - eff) : 0;
    if (rt > implied_reserved + tolerance || implied_reserved > rt + tolerance) {
        GGML_LOG_INFO(
            "[UNIFIED-CACHE] Note: external runtime tracker (%.1f MB) "
            "differs from internal reserved (%.1f MB) on device %d\n",
            rt / (1024.0f * 1024.0f), implied_reserved / (1024.0f * 1024.0f), device);
    }
}

void unified_cache_seal_layout_pool(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return;
    }
    cache->seal_layout_pool();
    GGML_LOG_INFO("[UNIFIED-CACHE] Layout pool sealed on device %d\n", effective_device);
}

bool unified_cache_is_budget_exceeded(int device) {
    int effective_device = resolve_effective_device(device);
    if (effective_device < 0) {
        return false;
    }
    unified_cache * cache = get_cache_shared(effective_device);
    if (!cache) {
        return false;
    }
    // Check the unified view: total committed (weights + runtime) > base budget.
    // The per-cache is_budget_exceeded() only checks used_ > budget_ which can
    // miss over-allocation when runtime_bytes grew since the last
    // update_reserved_bytes() call.
    const size_t committed = unified_cache_total_committed_bytes(device);
    const size_t base      = cache->base_budget();
    return cache->is_budget_exceeded() || committed > base;
}

bool unified_cache_has_evictions() {
    std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
    for (auto & [device_id, cache] : g_device_caches) {
        if (cache && cache->has_evictions()) {
            return true;
        }
    }
    return false;
}

// === Budget Export API ===

size_t compute_moe_effective_weight_bytes(size_t total_weight_bytes,
                                          size_t expert_total_bytes,
                                          int    n_expert,
                                          int    n_expert_used) {
    if (n_expert <= 0 || n_expert_used <= 0 || expert_total_bytes == 0) {
        return total_weight_bytes;  // Dense model, no savings
    }
    // Active expert fraction + headroom for expert cache churn
    // Use 1.5x active ratio to account for recently-used experts still in cache
    double active_ratio    = static_cast<double>(n_expert_used) / n_expert;
    double effective_ratio = std::min(1.0, active_ratio * 1.5);
    size_t expert_savings  = static_cast<size_t>(expert_total_bytes * (1.0 - effective_ratio));
    return (expert_savings <= total_weight_bytes) ? total_weight_bytes - expert_savings : 0;
}

unified_budget_info unified_cache_get_budget_info(int device) {
    unified_budget_info info = {};
    info.device_id           = device;

    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return info;  // Return zeroed struct for invalid device
    }

    // Read budget percentage once, clamp to [1,100]
    int          pct     = 90;
    const char * env_pct = std::getenv("GGML_SYCL_VRAM_BUDGET_PCT");
    if (env_pct) {
        pct = std::max(1, std::min(100, std::atoi(env_pct)));
    }
    info.budget_pct = pct;

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_sycl_get_device_memory(device, &free_mem, &total_mem);
    info.total_vram = ggml_sycl_info().devices[device].total_vram;
    if (info.total_vram == 0) {
        info.total_vram = total_mem > 0 ? total_mem : free_mem;
    }

    auto * cache = get_unified_cache_for_device(device);
    if (cache) {
        info.budget_bytes  = unified_cache_total_managed(device);
        info.weight_bytes  = unified_cache_weight_bytes(device);
        info.runtime_bytes = unified_cache_get_runtime_bytes(device);
        info.total_committed = unified_cache_total_committed_bytes(device);
        info.available_for_weights =
            info.budget_bytes > info.runtime_bytes ? info.budget_bytes - info.runtime_bytes : 0;
        info.total_available = unified_cache_total_available_bytes(device);
    } else {
        // Cache not yet initialized — use raw calculation
        info.budget_bytes     = static_cast<size_t>(info.total_vram * (static_cast<double>(pct) / 100.0));
        const size_t headroom = std::max(size_t(256) << 20, info.total_vram / 10);
        if (info.total_vram > headroom && info.budget_bytes > info.total_vram - headroom) {
            info.budget_bytes = info.total_vram - headroom;
        }
        info.available_for_weights = info.budget_bytes;
        info.total_committed       = 0;
        info.total_available       = info.budget_bytes;
    }

    // Populate MoE fields from tensor inventory
    size_t moe_total = 0;
    int    n_exp = 0, n_exp_used = 0;
    ggml_sycl_get_moe_info(&moe_total, &n_exp, &n_exp_used);
    info.expert_weight_bytes = moe_total;
    info.n_expert_total      = n_exp;
    info.n_expert_used       = n_exp_used;
    info.active_expert_bytes = compute_moe_effective_weight_bytes(moe_total, moe_total, n_exp, n_exp_used);

    // model_exceeds_vram removed — unified non-blocking cache handles all model sizes

    return info;
}

size_t unified_cache_get_margin_bytes(int device) {
    auto info = unified_cache_get_budget_info(device);
    if (info.available_for_weights > info.weight_bytes) {
        return info.available_for_weights - info.weight_bytes;
    }
    return 0;
}

bool unified_cache_should_offload_kv(int device, size_t kv_estimate_bytes) {
    // Check env var override first
    static std::atomic<int> cached_env{ -2 };  // -2 = not checked
    int                     env_val = cached_env.load(std::memory_order_acquire);
    if (env_val == -2) {
        const char * env_kv = std::getenv("GGML_SYCL_KV_HOST");
        env_val             = env_kv ? std::atoi(env_kv) : -1;
        cached_env.store(env_val, std::memory_order_release);
    }
    if (env_val == 1) {
        return true;
    }
    if (env_val == 0) {
        return false;
    }

    // HOST_COMPUTE mode: auto-offload KV to host-pinned memory.
    // When CPU offload runs with host-pinned compute buffers, KV on host
    // eliminates GPU islands (SET_ROWS writes + FLASH_ATTN reads go through
    // PCIe zero-copy instead of requiring CPU↔GPU transitions).
    static std::atomic<int> cached_hc{ -1 };
    int hc_val = cached_hc.load(std::memory_order_acquire);
    if (hc_val == -1) {
        const char * env_hc = std::getenv("GGML_SYCL_HOST_COMPUTE");
        hc_val              = (env_hc && std::atoi(env_hc) != 0) ? 1 : 0;
        cached_hc.store(hc_val, std::memory_order_release);
    }
    if (hc_val == 1) {
        return true;
    }

    auto info = unified_cache_get_budget_info(device);

    // Check if weight bytes already exceed available budget (model doesn't fit in VRAM)
    if (info.weight_bytes > 0 && info.weight_bytes > info.available_for_weights) {
        return true;
    }

    // Check total model size vs budget for models not yet fully loaded
    const size_t model_size = ggml_sycl_get_model_size();
    if (model_size > 0 && model_size > info.available_for_weights) {
        return true;
    }

    // If KV estimate provided, check if it would push us over budget
    if (kv_estimate_bytes > 0 && info.available_for_weights > info.weight_bytes) {
        size_t margin = info.available_for_weights - info.weight_bytes;
        if (kv_estimate_bytes > margin) {
            return true;
        }
    }

    return false;
}

// === MoE Cache Helpers ===

void unpin_all_experts() {
    std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
    // Unpin in all caches (cache pointers stable during inference)
    for (auto & [device_id, cache] : g_device_caches) {
        if (cache) {
            cache->unpin_experts();
        }
    }
}

// === Routing-Aware Expert Pre-staging ===

// Helper: Create a cache ID for an expert that matches the dispatch path's key generation.
// Uses tensor name hash + cache_uuid + model_id (same as ggml_sycl_get_moe_expert_cache_key
// in ggml-sycl.cpp) so prestaged entries are found during dispatch.
static ggml_sycl_cache_id make_expert_cache_id(const char * tensor_name,
                                               uint64_t     cache_uuid,
                                               uint32_t     model_id,
                                               int          expert_id,
                                               ggml_type    tensor_type = GGML_TYPE_COUNT,
                                               int64_t      ne0         = 0,
                                               int64_t      ne1         = 0) {
    ggml_sycl_cache_id id{};

    // Use name-based key with expert_id suffix for per-expert uniqueness.
    // Matches ggml_sycl_get_moe_expert_cache_key in ggml-sycl.cpp.
    std::string expert_name = (tensor_name && tensor_name[0]) ? std::string(tensor_name) : std::string("unknown");
    expert_name += ":e";
    expert_name += std::to_string(expert_id);
    uint64_t name_hash = static_cast<uint64_t>(std::hash<std::string>()(expert_name));

    id.valid         = true;
    id.model_id      = model_id;
    id.has_gguf      = false;
    id.file_idx      = 0;
    id.file_offs     = 0;
    id.nbytes        = 0;
    id.name_hash     = name_hash;
    id.type          = tensor_type;
    id.tp_sharded    = false;
    id.tp_rank       = 0;
    id.tp_world_size = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = 0;
        id.tp_local_ne[i]  = 0;
        id.tp_offset_ne[i] = 0;
    }
    id.ne[0] = ne0;
    id.ne[1] = ne1;
    id.ne[2] = 1;
    id.ne[3] = 1;

    // Combine cache_uuid with expert_id — matches ggml_sycl_get_moe_expert_cache_key
    uint64_t aux = cache_uuid;
    aux          = detail::cache_hash_combine(aux, static_cast<uint64_t>(expert_id));
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
                                        int             device_id,
                                        const char *    tensor_name,
                                        uint64_t        cache_uuid,
                                        uint32_t        model_id,
                                        ggml_type       tensor_type,
                                        int64_t         ne0,
                                        int64_t         ne1) {
    prestage_result result{};
    result.n_staged = 0;
    result.n_pinned = 0;
    result.n_unique = 0;
    result.success  = false;

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
    const int                   total_ids = n_expert_used * n_tokens;

    for (int i = 0; i < total_ids; i++) {
        const int32_t expert_id = expert_ids[i];
        if (expert_id >= 0 && expert_id < n_experts_total) {
            unique_experts.insert(expert_id);
        }
    }

    result.n_unique = static_cast<int>(unique_experts.size());

    GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: %d unique experts from %d IDs (n_experts_total=%d)\n", layer_id,
                    result.n_unique, total_ids, n_experts_total);

    if (result.n_unique == 0) {
        result.success = true;  // Nothing to do, but not an error
        return result;
    }

    // Step 2: Check cache hits and build list of experts to stage
    std::vector<int32_t> experts_to_stage;
    experts_to_stage.reserve(result.n_unique);

    for (int32_t expert_id : unique_experts) {
        const void *       expert_ptr = static_cast<const char *>(weight_base_ptr) + expert_id * expert_stride;
        ggml_sycl_cache_id key        = make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id,
                                                             tensor_type, ne0, ne1);

        // Check if already cached (any layout)
        if (!cache->is_cached_any(key)) {
            experts_to_stage.push_back(expert_id);
        }
    }

    GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: %zu cache hits, %zu to stage\n", layer_id,
                    unique_experts.size() - experts_to_stage.size(), experts_to_stage.size());

    // Use queue_ptr for async staging — submit H2D transfers on the caller's
    // queue instead of the cache's internal queue for better pipelining.
    sycl::queue * staging_queue = static_cast<sycl::queue *>(queue_ptr);

    // Step 3: Stage missing experts via ensure_cached_layout to get fill events.
    // CRITICAL: If the cache is near capacity, ensure_cached_layout triggers
    // eviction which calls enqueue_deferred_free. Drain ALL queues (compute,
    // cache, BCS) before staging so no stale pointers are freed while kernels run.
    if (!experts_to_stage.empty() && cache->budget_utilization() > 0.5f) {
        GGML_LOG_INFO("[PRESTAGE-DRAIN] Layer %d: draining queues before staging %zu experts "
                      "(utilization=%.1f%%)\n",
                      layer_id, experts_to_stage.size(), cache->budget_utilization() * 100.0f);
        try { cache->get_queue().wait(); } catch (...) {}
        if (staging_queue) {
            try { staging_queue->wait(); } catch (...) {}
        }
        try { cache->get_bcs_queue().wait(); } catch (...) {}
        // Deferred frees are processed by process_deferred_frees_public()
        // which checks g_graph_compute_active.  During inference it will
        // correctly skip the free — deferred frees drain later in
        // ggml_backend_sycl_synchronize() after all GPU work completes.
    }

    // Yield every 4 experts to drain all queues (CCS + BCS + staging) and
    // prevent xe driver GT engine resets from unbounded command list growth
    // during inference dispatch.  The ensure_cached_layout path submits
    // malloc_device + fill_fn + H2D memcpy per expert -- without periodic
    // draining, a layer with many cache-miss experts (e.g. 120B model cold
    // start) can accumulate >10s of non-preemptible work.
    // Yield every 2 experts to stay well under the xe driver's 10s job timeout.
    // Previous value of 4 could accumulate >10s of non-preemptible work on cold
    // start (all experts miss → eviction cascades + H2D + CCS reorder per expert).
    constexpr int PRESTAGE_YIELD_BATCH = 2;
    int           experts_staged_count = 0;
    for (int32_t expert_id : experts_to_stage) {
        const void *       expert_ptr = static_cast<const char *>(weight_base_ptr) + expert_id * expert_stride;
        ggml_sycl_cache_id key        = make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id,
                                                             tensor_type, ne0, ne1);

        cache_layout_request req{};
        req.key              = key;
        req.src_ptr          = expert_ptr;
        req.src_size         = expert_size;
        req.dst_size         = expert_size;
        req.type             = cache_entry_type::MOE_EXPERT;
        req.layer_id         = layer_id;
        req.expert_id        = expert_id;
        req.layout           = GGML_LAYOUT_AOS;
        req.validate_content = false;

        cache_layout_result layout_result = cache->ensure_cached_layout(req, {}, staging_queue);

        if (layout_result.status == cache_layout_status::READY ||
            layout_result.status == cache_layout_status::IN_PROGRESS) {
            result.n_staged++;
            // Collect async fill event so callers can depends_on() it
            if (layout_result.status == cache_layout_status::IN_PROGRESS) {
                result.staging_events.push_back(layout_result.event);
            }
        } else {
            GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: Failed to stage expert %d\n", layer_id, expert_id);
        }

        // Periodic yield: drain all queues to prevent engine reset
        experts_staged_count++;
        if (experts_staged_count % PRESTAGE_YIELD_BATCH == 0) {
            try {
                cache->get_queue().wait();       // CCS compute queue
                cache->get_dma_queue().wait();   // DMA reorder queue
                cache->get_bcs_queue().wait();   // BCS copy queue
                if (staging_queue) {
                    staging_queue->wait();        // caller's staging queue
                }
                cache->finalize_pending_fills();
                cache->process_deferred_frees_public();
            } catch (...) {
            }
        }
    }

    // Final flush after staging loop: drain any remaining in-flight work
    if (experts_staged_count > 0 && (experts_staged_count % PRESTAGE_YIELD_BATCH) != 0) {
        try {
            cache->get_queue().wait();
            cache->get_dma_queue().wait();
            cache->get_bcs_queue().wait();
            if (staging_queue) {
                staging_queue->wait();
            }
            cache->finalize_pending_fills();
        } catch (...) {
        }
    }

    // Step 4: Pin all unique experts (including those already cached)
    for (int32_t expert_id : unique_experts) {
        ggml_sycl_cache_id key = make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id,
                                                      tensor_type, ne0, ne1);

        cache->pin(key, GGML_LAYOUT_AOS);
        result.n_pinned++;
    }

    result.success = true;

    GGML_SYCL_DEBUG("[PRESTAGE] Layer %d: Completed - staged=%d, pinned=%d, unique=%d, async_events=%zu\n",
                    layer_id, result.n_staged, result.n_pinned, result.n_unique, result.staging_events.size());

    return result;
}

void unpin_routed_experts(const int32_t * expert_ids,
                          int             n_expert_used,
                          int             n_tokens,
                          const void *    weight_base_ptr,
                          size_t          expert_stride,
                          int             layer_id,
                          int             n_experts_total,
                          int             device_id,
                          const char *    tensor_name,
                          uint64_t        cache_uuid,
                          uint32_t        model_id,
                          ggml_type       tensor_type,
                          int64_t         ne0,
                          int64_t         ne1) {
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
        ggml_sycl_cache_id key = make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id,
                                                      tensor_type, ne0, ne1);

        cache->unpin(key, GGML_LAYOUT_AOS);
    }

    GGML_SYCL_DEBUG("[UNPIN] Layer %d: Unpinned %zu experts\n", layer_id, unique_experts.size());
}

// (ExpertPlacementTable removed — the cache IS the placement.
//  Popularity tracking now in g_expert_popularity via
//  get_expert_popularity_rank() / set_expert_popularity_rank().)

// === OneDNN FP16 Scratch Buffer Implementation ===

bool unified_cache::reserve_onednn_scratch(size_t weights_size, size_t activations_size) {
    std::lock_guard<std::mutex> lock(onednn_scratch_mutex_);

    // Already reserved with sufficient size?
    if (onednn_weights_scratch_ && onednn_activations_scratch_ && onednn_weights_scratch_size_ >= weights_size &&
        onednn_activations_scratch_size_ >= activations_size) {
        return true;
    }

    // VRAM arena path: sub-allocate from the oneDNN zone.
    if (arena_.active()) {
        const size_t total_needed = weights_size + activations_size;
        size_t zone_cap = arena_.zone_capacity(vram_zone_id::ONEDNN);
        if (total_needed <= zone_cap) {
            // Reset the oneDNN zone to reclaim any previous allocation.
            arena_.zone_reset(vram_zone_id::ONEDNN);

            void * w = arena_.zone_alloc(vram_zone_id::ONEDNN, weights_size);
            void * a = w ? arena_.zone_alloc(vram_zone_id::ONEDNN, activations_size) : nullptr;
            if (w && a) {
                onednn_weights_scratch_          = w;
                onednn_weights_scratch_size_     = weights_size;
                onednn_activations_scratch_      = a;
                onednn_activations_scratch_size_ = activations_size;
                // Budget already charged when arena was reserved.
                GGML_LOG_INFO("[UNIFIED-CACHE] oneDNN scratch from arena: weights=%.1f MB, activations=%.1f MB\n",
                              weights_size / (1024.0f * 1024.0f), activations_size / (1024.0f * 1024.0f));
                return true;
            }
            // Reset zone on partial failure.
            arena_.zone_reset(vram_zone_id::ONEDNN);
        }
        GGML_LOG_WARN("[UNIFIED-CACHE] oneDNN scratch arena zone too small (need %.1f MB, have %.1f MB), "
                      "falling back to direct alloc\n",
                      total_needed / (1024.0f * 1024.0f), zone_cap / (1024.0f * 1024.0f));
    }

    // Free existing if resizing — subtract old sizes from budget first
    const size_t old_total = onednn_weights_scratch_size_ + onednn_activations_scratch_size_;
    if (onednn_weights_scratch_ && !arena_.owns(onednn_weights_scratch_)) {
        try {
            sycl::free(onednn_weights_scratch_, queue_);
        } catch (...) {
        }
        onednn_weights_scratch_      = nullptr;
        onednn_weights_scratch_size_ = 0;
    } else {
        onednn_weights_scratch_      = nullptr;
        onednn_weights_scratch_size_ = 0;
    }
    if (onednn_activations_scratch_ && !arena_.owns(onednn_activations_scratch_)) {
        try {
            sycl::free(onednn_activations_scratch_, queue_);
        } catch (...) {
        }
        onednn_activations_scratch_      = nullptr;
        onednn_activations_scratch_size_ = 0;
    } else {
        onednn_activations_scratch_      = nullptr;
        onednn_activations_scratch_size_ = 0;
    }
    if (old_total > 0 && !arena_.active()) {
        saturating_sub_used(old_total);
    }

    // Note: we do NOT check cache budget here.  oneDNN scratch is a temporary
    // compute buffer (not cached weights), so it should not be gated by the
    // weight-cache available() budget.  When all weights are device-resident
    // (must_device=true), available() is near-zero but the device still has
    // physical VRAM for scratch.  If the device truly lacks memory,
    // sycl::malloc_device will fail and we handle it in the catch blocks below.
    const size_t total_needed = weights_size + activations_size;
    GGML_SYCL_DEBUG("[UNIFIED-CACHE] oneDNN scratch: need %.1f MB, cache-available %.1f MB (bypassing budget check)\n",
                    total_needed / (1024.0f * 1024.0f), available() / (1024.0f * 1024.0f));

    // Allocate weights scratch
    try {
        onednn_weights_scratch_ = sycl::malloc_device(weights_size, queue_);
        if (!onednn_weights_scratch_) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to allocate oneDNN weights scratch (%.1f MB)\n",
                            weights_size / (1024.0f * 1024.0f));
            return false;
        }
        alloc_registry::instance().register_alloc(onednn_weights_scratch_, weights_size,
                                                  ggml_sycl_get_device_id_from_queue(queue_), alloc_type::DEVICE);
        onednn_weights_scratch_size_ = weights_size;
    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] oneDNN weights scratch allocation failed: %s\n", e.what());
        return false;
    }

    // Allocate activations scratch
    try {
        onednn_activations_scratch_ = sycl::malloc_device(activations_size, queue_);
        if (!onednn_activations_scratch_) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to allocate oneDNN activations scratch (%.1f MB)\n",
                            activations_size / (1024.0f * 1024.0f));
            // Cleanup weights
            alloc_registry::instance().unregister_alloc(onednn_weights_scratch_);
            sycl::free(onednn_weights_scratch_, queue_);
            onednn_weights_scratch_      = nullptr;
            onednn_weights_scratch_size_ = 0;
            return false;
        }
        alloc_registry::instance().register_alloc(onednn_activations_scratch_, activations_size,
                                                  ggml_sycl_get_device_id_from_queue(queue_), alloc_type::DEVICE);
        onednn_activations_scratch_size_ = activations_size;
    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] oneDNN activations scratch allocation failed: %s\n", e.what());
        sycl::free(onednn_weights_scratch_, queue_);
        onednn_weights_scratch_      = nullptr;
        onednn_weights_scratch_size_ = 0;
        return false;
    }

    // Track in budget
    used_.fetch_add(total_needed, std::memory_order_relaxed);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reserved oneDNN scratch: weights=%.1f MB, activations=%.1f MB\n",
                    weights_size / (1024.0f * 1024.0f), activations_size / (1024.0f * 1024.0f));
    return true;
}

bool unified_cache::get_onednn_scratch(size_t weights_needed, size_t activations_needed, onednn_scratch_buffers & out) {
    // Note: caller must hold onednn_scratch_mutex_ via lock_onednn_scratch()
    if (!onednn_weights_scratch_ || !onednn_activations_scratch_) {
        return false;
    }
    if (weights_needed > onednn_weights_scratch_size_ || activations_needed > onednn_activations_scratch_size_) {
        return false;
    }
    out.weights          = onednn_weights_scratch_;
    out.activations      = onednn_activations_scratch_;
    out.weights_size     = onednn_weights_scratch_size_;
    out.activations_size = onednn_activations_scratch_size_;
    return true;
}

bool unified_cache::reserve_reorder_temp(size_t size_bytes) {
    // Already reserved with sufficient size?
    if (reorder_temp_buffer_ && reorder_temp_size_ >= size_bytes) {
        return true;
    }

    // Free existing if resizing
    if (reorder_temp_buffer_) {
        alloc_registry::instance().unregister_alloc(reorder_temp_buffer_);
        try {
            sycl::free(reorder_temp_buffer_, queue_);
        } catch (...) {
        }
        saturating_sub_used(reorder_temp_size_);
        reorder_temp_buffer_ = nullptr;
        reorder_temp_size_   = 0;
    }

    // Allocate temp buffer for GPU-side AOS→SOA reorder.
    // Called from moe_hybrid_init_once under std::call_once — single-threaded.
    try {
        reorder_temp_buffer_ = sycl::malloc_device(size_bytes, queue_);
        if (!reorder_temp_buffer_) {
            GGML_LOG_WARN("[UNIFIED-CACHE] Failed to allocate reorder temp buffer (%.1f MB)\n",
                          size_bytes / (1024.0f * 1024.0f));
            return false;
        }
        reorder_temp_size_ = size_bytes;
        used_.fetch_add(size_bytes, std::memory_order_relaxed);
        const int dev_id = ggml_sycl_get_device_id_from_queue(queue_);
        alloc_registry::instance().register_alloc(reorder_temp_buffer_, size_bytes,
                                                  dev_id, alloc_type::DEVICE);
        GGML_LOG_INFO("[UNIFIED-CACHE] Reserved GPU reorder temp buffer: %.1f MB\n",
                      size_bytes / (1024.0f * 1024.0f));
        return true;
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[UNIFIED-CACHE] Reorder temp buffer allocation failed: %s\n", e.what());
        reorder_temp_buffer_ = nullptr;
        reorder_temp_size_   = 0;
        return false;
    }
}

// Global scratch buffer state for lock management
static std::mutex                                            g_onednn_scratch_lock_mutex;
static std::unordered_map<int, std::unique_lock<std::mutex>> g_onednn_scratch_locks;

bool unified_cache_reserve_onednn_scratch(int device_id, size_t weights_size, size_t activations_size) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->reserve_onednn_scratch(weights_size, activations_size);
}

onednn_scratch_result unified_cache_get_onednn_scratch(int    device_id,
                                                       size_t weights_needed,
                                                       size_t activations_needed) {
    onednn_scratch_result result;
    unified_cache *       cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return result;
    }

    // Acquire lock and store it for later release
    auto lock = cache->lock_onednn_scratch();

    unified_cache::onednn_scratch_buffers buffers;
    if (!cache->get_onednn_scratch(weights_needed, activations_needed, buffers)) {
        return result;
    }

    // Store lock for release
    {
        std::lock_guard<std::mutex> guard(g_onednn_scratch_lock_mutex);
        g_onednn_scratch_locks[device_id] = std::move(lock);
    }

    result.weights     = buffers.weights;
    result.activations = buffers.activations;
    result.ok          = true;
    return result;
}

void unified_cache_release_onednn_scratch(int device_id) {
    std::lock_guard<std::mutex> guard(g_onednn_scratch_lock_mutex);
    auto                        it = g_onednn_scratch_locks.find(device_id);
    if (it != g_onednn_scratch_locks.end()) {
        // Unlock by destroying the unique_lock
        g_onednn_scratch_locks.erase(it);
    }
}

bool unified_cache_has_onednn_scratch(int device_id) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->has_onednn_scratch();
}

// === Persistent Scratch Buffer Implementation ===

bool unified_cache::reserve_persistent_scratch(const std::string & buffer_name, size_t size_bytes, bool pin) {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);

    // Check if we already have this buffer with sufficient size
    auto it = persistent_scratches_.find(buffer_name);
    if (it != persistent_scratches_.end()) {
        auto & entry = it->second;
        if (entry.size >= size_bytes) {
            // Already have sufficient size, just update pin state if needed
            entry.pinned = pin;
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] Persistent scratch '%s' already reserved (%.1f MB >= %.1f MB)\n",
                            buffer_name.c_str(), entry.size / (1024.0f * 1024.0f), size_bytes / (1024.0f * 1024.0f));
            return true;
        }
        // Existing buffer too small, need to free and reallocate
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Persistent scratch '%s' resize: %.1f MB -> %.1f MB\n", buffer_name.c_str(),
                        entry.size / (1024.0f * 1024.0f), size_bytes / (1024.0f * 1024.0f));
        if (entry.device_ptr) {
            try {
                sycl::free(entry.device_ptr, queue_);
            } catch (...) {
            }
            saturating_sub_used(entry.size);
        }
        persistent_scratches_.erase(it);
    }

    // Check if we have budget
    if (size_bytes > available()) {
        // Try to evict to make room
        size_t freed = evict(size_bytes - available());
        if (freed < size_bytes - available()) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] Cannot reserve persistent scratch '%s': need %.1f MB, available %.1f MB\n",
                           buffer_name.c_str(), size_bytes / (1024.0f * 1024.0f), available() / (1024.0f * 1024.0f));
            return false;
        }
    }

    // Allocate device memory
    void * ptr = nullptr;
    try {
        ptr = sycl::malloc_device(size_bytes, queue_);
        if (!ptr) {
            GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to allocate persistent scratch '%s' (%.1f MB)\n",
                           buffer_name.c_str(), size_bytes / (1024.0f * 1024.0f));
            return false;
        }
        alloc_registry::instance().register_alloc(ptr, size_bytes, ggml_sycl_get_device_id_from_queue(queue_),
                                                  alloc_type::DEVICE);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] Persistent scratch '%s' allocation failed: %s\n", buffer_name.c_str(),
                       e.what());
        return false;
    }

    // Track in budget and store entry
    used_.fetch_add(size_bytes, std::memory_order_relaxed);
    persistent_scratches_[buffer_name] = { ptr, size_bytes, pin };

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Reserved persistent scratch '%s': %.1f MB (pinned=%d)\n", buffer_name.c_str(),
                    size_bytes / (1024.0f * 1024.0f), pin ? 1 : 0);
    return true;
}

void * unified_cache::get_persistent_scratch(const std::string & buffer_name) {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);

    auto it = persistent_scratches_.find(buffer_name);
    if (it == persistent_scratches_.end()) {
        return nullptr;
    }
    return it->second.device_ptr;
}

void unified_cache::release_persistent_scratch(const std::string & buffer_name) {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);

    auto it = persistent_scratches_.find(buffer_name);
    if (it == persistent_scratches_.end()) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Persistent scratch '%s' not found for release\n", buffer_name.c_str());
        return;
    }

    auto & entry = it->second;
    if (entry.device_ptr) {
        try {
            sycl::free(entry.device_ptr, queue_);
        } catch (...) {
        }
        saturating_sub_used(entry.size);
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Released persistent scratch '%s' (%.1f MB)\n", buffer_name.c_str(),
                        entry.size / (1024.0f * 1024.0f));
    }
    persistent_scratches_.erase(it);
}

bool unified_cache::has_persistent_scratch(const std::string & buffer_name) const {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);
    return persistent_scratches_.find(buffer_name) != persistent_scratches_.end();
}

size_t unified_cache::get_persistent_scratch_size(const std::string & buffer_name) const {
    std::lock_guard<std::mutex> lock(persistent_scratch_mutex_);
    auto                        it = persistent_scratches_.find(buffer_name);
    if (it == persistent_scratches_.end()) {
        return 0;
    }
    return it->second.size;
}

// === Persistent Scratch Buffer C API Wrappers ===

bool unified_cache_reserve_persistent_scratch(int device_id, const char * buffer_name, size_t size_bytes, bool pin) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->reserve_persistent_scratch(buffer_name, size_bytes, pin);
}

void * unified_cache_get_persistent_scratch(int device_id, const char * buffer_name) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->get_persistent_scratch(buffer_name);
}

void unified_cache_release_persistent_scratch(int device_id, const char * buffer_name) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->release_persistent_scratch(buffer_name);
    }
}

bool unified_cache_has_persistent_scratch(int device_id, const char * buffer_name) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->has_persistent_scratch(buffer_name);
}

size_t unified_cache_get_persistent_scratch_size(int device_id, const char * buffer_name) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return 0;
    }
    return cache->get_persistent_scratch_size(buffer_name);
}

// === Bulk Weight Pinning C API Wrappers ===

int unified_cache_pin_layer_weights(int device_id, int layer_id, const layer_weight_set * weights, int layout) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache || !weights) {
        return 0;
    }
    return cache->pin_layer_weights(layer_id, *weights, static_cast<ggml_layout_mode>(layout));
}

void unified_cache_unpin_layer_weights(int device_id, int layer_id, const layer_weight_set * weights, int layout) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache || !weights) {
        return;
    }
    cache->unpin_layer_weights(layer_id, *weights, static_cast<ggml_layout_mode>(layout));
}

int unified_cache_pin_model_weights(int device_id, int n_layers, const layer_weight_set * layers, int layout) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache || !layers || n_layers <= 0) {
        return 0;
    }
    // Convert C array to vector
    std::vector<layer_weight_set> layers_vec(layers, layers + n_layers);
    return cache->pin_model_weights(n_layers, layers_vec, static_cast<ggml_layout_mode>(layout));
}

void unified_cache_unpin_model_weights(int device_id, int n_layers, const layer_weight_set * layers, int layout) {
    unified_cache * cache = get_unified_cache_for_device(device_id);
    if (!cache || !layers || n_layers <= 0) {
        return;
    }
    // Unpin each layer
    for (int i = 0; i < n_layers; i++) {
        cache->unpin_layer_weights(i, layers[i], static_cast<ggml_layout_mode>(layout));
    }
}

// =============================================================================
// Multi-Device Partial Row Loading
// =============================================================================

void * unified_cache::load_partial_rows(const char * tensor_name,
                                         const void * src_host,
                                         ggml_type    type,
                                         int64_t      ncols,
                                         int64_t      row_count,
                                         int          device_idx) {
    if (!tensor_name || !src_host || row_count <= 0 || ncols <= 0) {
        return nullptr;
    }

    // Build cache key: "tensor_name:device_idx"
    std::string key = std::string(tensor_name) + ":" + std::to_string(device_idx);

    // Check if already loaded
    {
        std::lock_guard<std::mutex> lock(partial_mutex_);
        auto it = partial_cache_.find(key);
        if (it != partial_cache_.end()) {
            return it->second.ptr;
        }
    }

    const size_t row_bytes     = ggml_row_size(type, ncols);
    const size_t partial_bytes = static_cast<size_t>(row_count) * row_bytes;

    if (partial_bytes == 0) {
        return nullptr;
    }

    // Allocate device memory on this cache's queue
    void * dev_ptr = nullptr;
    try {
        dev_ptr = ggml_sycl_malloc_device_raw(partial_bytes, queue_, "unified_cache:partial_rows");
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[PARTIAL-ROWS] malloc_device failed for '%s' device %d: %s\n",
                       tensor_name, device_idx, e.what());
        return nullptr;
    }
    if (!dev_ptr) {
        GGML_LOG_ERROR("[PARTIAL-ROWS] malloc_device returned nullptr for '%s' device %d (%.2f MB)\n",
                       tensor_name, device_idx, partial_bytes / (1024.0f * 1024.0f));
        return nullptr;
    }

    // Copy AOS data from host to device — no CPU wait needed since the reorder
    // kernel below is submitted to the same in-order queue_ and will implicitly
    // depend on this memcpy completing on the GPU.
    queue_.memcpy(dev_ptr, src_host, partial_bytes);

    // Apply in-place SOA reorder on device (same queue, implicitly ordered)
    bool reordered = reorder_rows_to_soa(static_cast<uint8_t *>(dev_ptr), type,
                                          ncols, row_count, partial_bytes, &queue_);
    if (!reordered) {
        GGML_LOG_ERROR("[PARTIAL-ROWS] SOA reorder failed for '%s' device %d type %d\n",
                       tensor_name, device_idx, (int) type);
        sycl::free(dev_ptr, queue_);
        return nullptr;
    }

    // Track in partial cache
    {
        std::lock_guard<std::mutex> lock(partial_mutex_);
        partial_cache_[key] = { dev_ptr, device_idx, partial_bytes };
    }

    // Update budget tracking (count as weight bytes on this device)
    used_.fetch_add(partial_bytes, std::memory_order_relaxed);

    GGML_SYCL_DEBUG("[PARTIAL-ROWS] Loaded '%s' device %d: %lld rows, %.2f MB SOA\n",
                    tensor_name, device_idx, (long long) row_count,
                    partial_bytes / (1024.0f * 1024.0f));

    return dev_ptr;
}

void * unified_cache::get_split_weight_ptr(const char * tensor_name, int device_idx) {
    if (!tensor_name) {
        return nullptr;
    }
    std::string key = std::string(tensor_name) + ":" + std::to_string(device_idx);

    std::lock_guard<std::mutex> lock(partial_mutex_);
    auto it = partial_cache_.find(key);
    return (it != partial_cache_.end()) ? it->second.ptr : nullptr;
}

void unified_cache::free_partial_entries() {
    std::lock_guard<std::mutex> lock(partial_mutex_);
    for (auto & pair : partial_cache_) {
        if (pair.second.ptr && !g_sycl_shutting_down.load()) {
            sycl::free(pair.second.ptr, queue_);
            saturating_sub_used(pair.second.bytes);
        }
    }
    partial_cache_.clear();
}

// Free-standing wrappers for multi-device partial row API

void * unified_cache_load_partial_rows(const char * tensor_name,
                                        const void * src_host,
                                        ggml_type    type,
                                        int64_t      ncols,
                                        int64_t      row_count,
                                        int          target_device) {
    auto * cache = get_unified_cache_for_device(target_device);
    if (!cache) {
        GGML_LOG_ERROR("[PARTIAL-ROWS] No cache for device %d\n", target_device);
        return nullptr;
    }
    return cache->load_partial_rows(tensor_name, src_host, type, ncols, row_count, target_device);
}

void * unified_cache_get_split_weight_ptr(const char * tensor_name, int device) {
    auto * cache = get_unified_cache_for_device(device);
    if (!cache) {
        return nullptr;
    }
    return cache->get_split_weight_ptr(tensor_name, device);
}

void unified_cache_free_partial_entries(int device) {
    auto * cache = get_unified_cache_for_device(device);
    if (cache) {
        cache->free_partial_entries();
    }
}

unified_cache * unified_cache_register_for_queue(int device_id, sycl::queue & queue) {
    // Fast path: check under shared lock
    {
        std::shared_lock<std::shared_mutex> read_lock(g_cache_rw_mutex);
        auto it = g_device_caches.find(device_id);
        if (it != g_device_caches.end()) {
            return it->second.get();
        }
    }

    // Slow path: create under exclusive lock
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);

    // Double-check after acquiring write lock
    auto it = g_device_caches.find(device_id);
    if (it != g_device_caches.end()) {
        return it->second.get();
    }

    // Query VRAM from the queue's device (no dpct dependency)
    sycl::device dev    = queue.get_device();
    size_t       total  = dev.get_info<sycl::info::device::global_mem_size>();
    size_t       budget = static_cast<size_t>(total * 0.80);  // 80% budget for secondary GPU

    const size_t min_headroom = 256ull * 1024ull * 1024ull;
    if (total > min_headroom && budget > total - min_headroom) {
        budget = total - min_headroom;
    }

    GGML_LOG_INFO("[UNIFIED-CACHE] Registering device %d (%s): total=%.1f MB budget=%.1f MB\n",
                  device_id, dev.get_info<sycl::info::device::name>().c_str(),
                  total / (1024.0f * 1024.0f), budget / (1024.0f * 1024.0f));

    const size_t staging_bytes = 16 * 1024 * 1024;  // 16 MB staging for secondary device
    try {
        g_device_caches[device_id] = std::make_unique<unified_cache>(queue, budget, staging_bytes, 0);
        return g_device_caches[device_id].get();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to register device %d: %s\n", device_id, e.what());
        return nullptr;
    }
}

void shutdown_unified_cache() {
    // Set shutdown flag FIRST so destructors skip sycl::free() calls
    g_sycl_shutting_down.store(true);

    // Clear all device caches
    // The destructors will skip cleanup due to the shutdown flag
    std::unique_lock<std::shared_mutex> lock(g_cache_rw_mutex);
    g_device_caches.clear();
    g_host_cache_shared.reset();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Shutdown complete\n");
}

// ============================================================================
// Phase 3: Unified Allocation API
// ============================================================================

// --- unified_cache::available_device() ---
// Queries Level Zero for the actual free VRAM on the device, subtracts a safety
// margin for driver internals.  Unlike available() which is pure budget math,
// this reflects real hardware state.

size_t unified_cache::available_device() const {
    const int device_id = get_device_id_from_queue(const_cast<sycl::queue &>(queue_));
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
    if (free_mem <= VRAM_SAFETY_MARGIN) {
        return 0;
    }
    return free_mem - VRAM_SAFETY_MARGIN;
}

// --- unified_cache::allocate() ---
// The primary allocation path for Phase 3.  Steps:
//   1. Check budget: used_ + runtime_reserved + size <= budget_
//   2. If insufficient, try evict_and_flush
//   3. Query L0 free VRAM as a second safety guard
//   4. Allocate on device or fall back to host-pinned
//   5. Track in managed_allocs_ and adjust budget

unified_cache::vram_alloc_result unified_cache::allocate(size_t          size,
                                                         alloc_lifetime  lifetime,
                                                         const char *    tag) {
    vram_alloc_result result{};
    if (size == 0) {
        return result;
    }

    const char * label = tag ? tag : "unified_cache::allocate";

    // Step 1: Check budget headroom (available() = budget_ - used_,
    // where budget_ = base_budget_ - reserved_, and reserved_ tracks
    // KV + compute + staging + graph runtime bytes).
    bool try_device = true;
    if (size > available()) {
        // Try eviction to make room.
        const size_t needed = size - available();
        const size_t freed  = evict_and_flush(needed);
        if (size > available()) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate(%s): budget exhausted after evicting "
                            "%.1f MB (need %.1f MB, avail %.1f MB)\n",
                            label, freed / (1024.0 * 1024.0),
                            size / (1024.0 * 1024.0),
                            available() / (1024.0 * 1024.0));
            try_device = false;
        }
    }

    // Step 2: Cross-check with L0 free VRAM.
    if (try_device) {
        const size_t hw_avail = available_device();
        if (size > hw_avail) {
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] allocate(%s): L0 free VRAM too low "
                            "(need %.1f MB, L0 avail %.1f MB), falling back\n",
                            label, size / (1024.0 * 1024.0),
                            hw_avail / (1024.0 * 1024.0));
            try_device = false;
        }
    }

    // Step 3: Attempt device allocation.
    void * ptr = nullptr;
    if (try_device) {
        try {
            ptr = ggml_sycl_malloc_device_raw(size, queue_, label);
        } catch (...) {
            ptr = nullptr;
        }
    }

    if (ptr) {
        // Device allocation succeeded.
        result.ptr       = ptr;
        result.on_device = true;
        result.size      = size;

        // Track against budget as runtime reservation.
        used_.fetch_add(size, std::memory_order_relaxed);

        std::lock_guard<std::mutex> lock(managed_allocs_mutex_);
        managed_allocs_[ptr] = { size, true, lifetime };
        return result;
    }

    // Step 4: Host-pinned fallback — route through the pre-allocated pinned pool.
    // All host-pinned memory is pre-allocated at init; zero runtime malloc_host.
    if (auto * hcache = try_get_host_cache()) {
        ptr = hcache->allocate_pinned_runtime(size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
    }
    if (!ptr) {
        // Pool exhausted — last resort: raw malloc_host (should not happen
        // after pre_allocate_all, but avoids hard failure during init).
        try {
            ptr = ggml_sycl_malloc_host(size, queue_, label);
        } catch (...) {
            ptr = nullptr;
        }
    }

    if (!ptr) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] allocate(%s): both device and host alloc failed "
                       "(size=%.1f MB)\n",
                       label, size / (1024.0 * 1024.0));
        return result;
    }

    result.ptr       = ptr;
    result.on_device = false;
    result.size      = size;

    // Host-pinned does NOT consume VRAM budget (it is malloc_host).
    std::lock_guard<std::mutex> lock(managed_allocs_mutex_);
    managed_allocs_[ptr] = { size, false, lifetime };
    return result;
}

// --- unified_cache::deallocate() ---

void unified_cache::deallocate(void * ptr, size_t size, alloc_lifetime lifetime) {
    if (!ptr) {
        return;
    }
    (void) lifetime;  // Used for future per-lifetime diagnostics.

    managed_alloc_entry entry{};
    {
        std::lock_guard<std::mutex> lock(managed_allocs_mutex_);
        auto it = managed_allocs_.find(ptr);
        if (it == managed_allocs_.end()) {
            GGML_LOG_WARN("[UNIFIED-CACHE] deallocate: unknown pointer %p (size=%zu)\n", ptr, size);
            return;
        }
        entry = it->second;
        managed_allocs_.erase(it);
    }

    if (entry.on_device) {
        // Reverse the budget charge.
        saturating_sub_used(entry.size);
    }

    try {
        sycl::free(ptr, queue_);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[UNIFIED-CACHE] deallocate: sycl::free failed ptr=%p: %s\n", ptr, e.what());
    }
}

// --- Compute Arena ---
// Pre-reserved VRAM bump allocator for compute scratch buffers.
// Single sycl::malloc_device allocation, bump-allocated during graph_compute,
// reset between invocations.  Must be called BEFORE S1-PRELOAD.

bool unified_cache::reserve_compute_arena(size_t arena_bytes) {
    if (compute_arena_ptr_ && compute_arena_size_ >= arena_bytes) {
        return true;  // Already reserved with sufficient capacity.
    }

    // VRAM arena path: compute zone is already pre-allocated.
    if (arena_.active()) {
        size_t zone_cap = arena_.zone_capacity(vram_zone_id::COMPUTE);
        if (zone_cap >= arena_bytes) {
            // Point compute_arena at the arena's compute zone base.
            compute_arena_ptr_  = arena_.offset_to_ptr(0);  // Compute zone starts at offset 0.
            compute_arena_size_ = zone_cap;
            compute_arena_off_.store(0, std::memory_order_relaxed);
            GGML_LOG_INFO("[COMPUTE-ARENA] Using VRAM arena compute zone: %.1f MB\n",
                          zone_cap / (1024.0 * 1024.0));
            // Budget already charged when arena was reserved — don't double-count.
            return true;
        }
        GGML_LOG_WARN("[COMPUTE-ARENA] Arena compute zone (%.1f MB) < requested (%.1f MB), "
                      "falling back to direct alloc\n",
                      zone_cap / (1024.0 * 1024.0), arena_bytes / (1024.0 * 1024.0));
    }

    // Free existing arena if resizing.
    if (compute_arena_ptr_ && !arena_.active()) {
        try {
            sycl::free(compute_arena_ptr_, queue_);
        } catch (...) {}
        saturating_sub_used(compute_arena_size_);
        compute_arena_ptr_  = nullptr;
        compute_arena_size_ = 0;
        compute_arena_off_.store(0, std::memory_order_relaxed);
    }

    // Allocate a single contiguous VRAM block.
    try {
        compute_arena_ptr_ = sycl::malloc_device(arena_bytes, queue_);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("[COMPUTE-ARENA] sycl::malloc_device failed (%.1f MB): %s\n",
                       arena_bytes / (1024.0 * 1024.0), e.what());
        compute_arena_ptr_ = nullptr;
    }

    if (!compute_arena_ptr_) {
        GGML_LOG_ERROR("[COMPUTE-ARENA] Failed to reserve %.1f MB of VRAM\n",
                       arena_bytes / (1024.0 * 1024.0));
        return false;
    }

    compute_arena_size_ = arena_bytes;
    compute_arena_off_.store(0, std::memory_order_relaxed);

    // Track arena bytes against the unified cache budget so S1-PRELOAD
    // sees reduced available VRAM and loads fewer weights.
    used_.fetch_add(arena_bytes, std::memory_order_relaxed);

    GGML_LOG_INFO("[COMPUTE-ARENA] Reserved %.1f MB VRAM for compute scratch\n",
                  arena_bytes / (1024.0 * 1024.0));
    return true;
}

void * unified_cache::arena_alloc(size_t size) {
    if (!compute_arena_ptr_ || size == 0) {
        return nullptr;
    }

    // Align to 256 bytes for GPU coalescing.
    const size_t aligned = (size + 255) & ~size_t(255);

    // Atomic bump allocator — lock-free.
    size_t off = compute_arena_off_.fetch_add(aligned, std::memory_order_relaxed);
    if (off + aligned > compute_arena_size_) {
        // Arena exhausted — roll back.
        compute_arena_off_.fetch_sub(aligned, std::memory_order_relaxed);
        return nullptr;
    }

    return static_cast<uint8_t *>(compute_arena_ptr_) + off;
}

void unified_cache::arena_reset() {
    compute_arena_off_.store(0, std::memory_order_relaxed);
}

bool unified_cache::arena_owns(const void * ptr) const {
    if (!compute_arena_ptr_ || !ptr) {
        return false;
    }
    const auto p    = reinterpret_cast<uintptr_t>(ptr);
    const auto base = reinterpret_cast<uintptr_t>(compute_arena_ptr_);
    return p >= base && p < base + compute_arena_size_;
}

size_t unified_cache::compute_arena_capacity() const {
    return compute_arena_size_;
}

size_t unified_cache::compute_arena_used() const {
    return compute_arena_off_.load(std::memory_order_relaxed);
}

// --- Inference Scratch Pool ---

bool unified_cache::reserve_scratch_pool(size_t pool_bytes) {
    if (scratch_pool_ptr_ && scratch_pool_size_ >= pool_bytes) {
        return true;  // Already large enough.
    }

    // Free existing pool if it exists but is too small.
    // Arena-owned pointers must NOT be sycl::free'd — reclaim to the weight zone instead.
    if (scratch_pool_ptr_) {
        if (arena_.active() && arena_.owns(scratch_pool_ptr_)) {
            size_t offset = arena_.ptr_to_offset(scratch_pool_ptr_);
            if (offset != SIZE_MAX) {
                arena_.weight_reclaim(offset, scratch_pool_size_);
            }
            // Budget was charged to the arena's bulk reservation — don't sub from used_.
        } else {
            saturating_sub_used(scratch_pool_size_);
            try {
                sycl::free(scratch_pool_ptr_, queue_);
            } catch (...) {}
        }
        scratch_pool_ptr_  = nullptr;
        scratch_pool_size_ = 0;
        scratch_pool_off_.store(0, std::memory_order_relaxed);
    }

    // VRAM arena path: sub-allocate from the weight zone (persistent allocation).
    if (arena_.active()) {
        void * ptr = arena_.zone_alloc(vram_zone_id::WEIGHT, pool_bytes);
        if (ptr) {
            scratch_pool_ptr_  = ptr;
            scratch_pool_size_ = pool_bytes;
            scratch_pool_off_.store(0, std::memory_order_relaxed);
            scratch_pool_hwm_  = 0;
            GGML_LOG_INFO("[UNIFIED-CACHE] Scratch pool reserved from arena weight zone: %.1f MB\n",
                          pool_bytes / (1024.0 * 1024.0));
            return true;
        }
        GGML_LOG_WARN("[UNIFIED-CACHE] Arena weight zone full for scratch pool (%.1f MB), "
                      "falling back to direct alloc\n", pool_bytes / (1024.0 * 1024.0));
    }

    // Allocate through our own allocate() path so budget/L0 checks apply.
    // Scratch pool MUST be on device — if allocate() falls back to host, reject it.
    vram_alloc_result res = allocate(pool_bytes, alloc_lifetime::PERSISTENT, "scratch_pool");
    if (!res.ptr || !res.on_device) {
        if (res.ptr && !res.on_device) {
            // Got host-pinned — not useful as scratch pool, release it.
            deallocate(res.ptr, pool_bytes, alloc_lifetime::PERSISTENT);
        }
        res.ptr = nullptr;
    }
    if (!res.ptr) {
        GGML_LOG_WARN("[UNIFIED-CACHE] reserve_scratch_pool: failed to allocate %.1f MB\n",
                      pool_bytes / (1024.0 * 1024.0));
        return false;
    }

    scratch_pool_ptr_  = res.ptr;
    scratch_pool_size_ = pool_bytes;
    scratch_pool_off_.store(0, std::memory_order_relaxed);
    scratch_pool_hwm_  = 0;

    GGML_LOG_INFO("[UNIFIED-CACHE] Scratch pool reserved: %.1f MB on device\n",
                  pool_bytes / (1024.0 * 1024.0));
    return true;
}

void * unified_cache::get_scratch(size_t size) {
    if (!scratch_pool_ptr_ || size == 0) {
        return nullptr;
    }

    // Align to 256 bytes for GPU coalescing.
    const size_t aligned = (size + 255) & ~size_t(255);

    // Atomic bump allocator — lock-free.
    size_t off = scratch_pool_off_.fetch_add(aligned, std::memory_order_relaxed);
    if (off + aligned > scratch_pool_size_) {
        // Pool exhausted — roll back.
        scratch_pool_off_.fetch_sub(aligned, std::memory_order_relaxed);
        return nullptr;
    }

    // Track high-water mark (relaxed — diagnostic only).
    size_t new_hwm = off + aligned;
    size_t cur_hwm = scratch_pool_hwm_;
    while (new_hwm > cur_hwm) {
        // Not atomic — this is best-effort diagnostic.
        scratch_pool_hwm_ = new_hwm;
        cur_hwm = new_hwm;
    }

    return static_cast<uint8_t *>(scratch_pool_ptr_) + off;
}

void unified_cache::return_scratch(void * ptr, size_t size) {
    // Stack discipline: we don't actually free individual allocations.
    // The pool is reset wholesale via reset_scratch_pool().
    (void) ptr;
    (void) size;
}

void unified_cache::reset_scratch_pool() {
    scratch_pool_off_.store(0, std::memory_order_relaxed);
}

// --- Expert Allocation ---

unified_cache::vram_alloc_result unified_cache::allocate_expert(size_t size) {
    return allocate(size, alloc_lifetime::SCRATCH, "expert");
}

// ============================================================================
// Phase 3: Free-Standing Wrappers
// ============================================================================

unified_cache::vram_alloc_result unified_cache_allocate(
    int                           device_id,
    size_t                        size,
    unified_cache::alloc_lifetime lifetime,
    const char *                  tag) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return {};
    }
    return cache->allocate(size, lifetime, tag);
}

void unified_cache_deallocate(int                           device_id,
                              void *                        ptr,
                              size_t                        size,
                              unified_cache::alloc_lifetime lifetime) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->deallocate(ptr, size, lifetime);
    }
}

bool unified_cache_reserve_compute_arena(int device_id, size_t arena_bytes) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->reserve_compute_arena(arena_bytes);
}

void * unified_cache_arena_alloc(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->arena_alloc(size);
}

void * unified_cache_arena_alloc_weight(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache || !cache->get_arena().active()) {
        return nullptr;
    }
    return cache->get_arena().zone_alloc(vram_zone_id::WEIGHT, size, 64);
}

void unified_cache_arena_reset(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->arena_reset();
    }
}

bool unified_cache_arena_owns(int device_id, const void * ptr) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->arena_owns(ptr);
}

size_t unified_cache_compute_arena_capacity(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    return cache ? cache->compute_arena_capacity() : 0;
}

size_t unified_cache_compute_arena_used(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    return cache ? cache->compute_arena_used() : 0;
}

size_t unified_cache_kv_arena_capacity(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache || !cache->get_arena().active()) {
        return 0;
    }
    return cache->get_arena().zone_capacity(vram_zone_id::KV);
}

size_t unified_cache_kv_arena_used(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache || !cache->get_arena().active()) {
        return 0;
    }
    return cache->get_arena().zone_used(vram_zone_id::KV);
}

void * unified_cache_kv_arena_alloc(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache || !cache->get_arena().active()) {
        return nullptr;
    }
    return cache->get_arena().zone_alloc(vram_zone_id::KV, size);
}

memory_location query_location(const void * ptr, int device_hint) {
    memory_location loc{};
    if (!ptr) {
        return loc;
    }

    loc.ptr = const_cast<void *>(ptr);

    // Step 1: Check alloc_registry for tier classification (O(1) binary search).
    const auto * info = alloc_registry::instance().lookup(ptr);
    if (info) {
        loc.device = info->device_id;
        switch (info->type) {
            case alloc_type::DEVICE:
                loc.tier = alloc_tier::DEVICE_VRAM;
                break;
            case alloc_type::HOST_PINNED:
                loc.tier = alloc_tier::HOST_PINNED;
                break;
            case alloc_type::SHARED:
                loc.tier = alloc_tier::HOST_PINNED;  // treat shared as host-accessible
                break;
            case alloc_type::MMAP:
                loc.tier = alloc_tier::MMAP_TRACKED;
                break;
            default:
                loc.tier = alloc_tier::MMAP_TRACKED;
                break;
        }
    } else if (device_hint >= 0) {
        // Not registered — assume host/mmap (conservative).
        loc.device = device_hint;
        loc.tier   = alloc_tier::MMAP_TRACKED;
    }

    // Step 2: Check arena zone ownership (device allocations only).
    if (loc.tier == alloc_tier::DEVICE_VRAM) {
        int dev = (loc.device >= 0) ? loc.device : device_hint;
        if (dev >= 0) {
            auto * cache = get_unified_cache_for_device(dev);
            if (cache && cache->get_arena().active()) {
                auto & arena = cache->get_arena();
                if (arena.owns(ptr)) {
                    loc.from_arena = true;
                    if (arena.zone_owns(vram_zone_id::COMPUTE, ptr)) {
                        loc.zone = vram_zone_id::COMPUTE;
                        loc.role = alloc_role::COMPUTE;
                    } else if (arena.zone_owns(vram_zone_id::KV, ptr)) {
                        loc.zone = vram_zone_id::KV;
                        loc.role = alloc_role::KV;
                    } else if (arena.zone_owns(vram_zone_id::ONEDNN, ptr)) {
                        loc.zone = vram_zone_id::ONEDNN;
                        loc.role = alloc_role::COMPUTE;
                    } else if (arena.zone_owns(vram_zone_id::WEIGHT, ptr)) {
                        loc.zone = vram_zone_id::WEIGHT;
                        loc.role = alloc_role::WEIGHT;
                    }
                }
            }
        }
    }

    return loc;
}

memory_location query_kv_location(int layer_id, int device) {
    memory_location loc{};
    loc.device = device;
    loc.role   = alloc_role::KV;
    loc.layout = GGML_LAYOUT_AOS;  // KV cache is always row-major

    if (device < 0) {
        // No device specified — assume host
        loc.tier = alloc_tier::HOST_PINNED;
        return loc;
    }

    auto & mgr = get_kv_tier_manager(device);
    if (!mgr.is_active()) {
        // No tiering configured — all KV on device (default path)
        loc.tier = alloc_tier::DEVICE_VRAM;
        return loc;
    }

    if (mgr.is_hot(static_cast<uint32_t>(layer_id))) {
        loc.tier = alloc_tier::DEVICE_VRAM;
        loc.zone = vram_zone_id::KV;
    } else {
        loc.tier = alloc_tier::HOST_PINNED;
    }

    return loc;
}

bool unified_cache_reserve_scratch_pool(int device_id, size_t pool_bytes) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return false;
    }
    return cache->reserve_scratch_pool(pool_bytes);
}

void * unified_cache_get_scratch(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return nullptr;
    }
    return cache->get_scratch(size);
}

void unified_cache_return_scratch(int device_id, void * ptr, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->return_scratch(ptr, size);
    }
}

void unified_cache_reset_scratch_pool(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (cache) {
        cache->reset_scratch_pool();
    }
}

unified_cache::vram_alloc_result unified_cache_allocate_expert(int device_id, size_t size) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return {};
    }
    return cache->allocate_expert(size);
}

size_t unified_cache_available_device(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return 0;
    }
    return cache->available_device();
}

size_t unified_cache_available_budget(int device_id) {
    auto * cache = get_unified_cache_for_device(device_id);
    if (!cache) {
        return 0;
    }
    return cache->available_budget();
}

// ============================================================================
// Phase 4: Pre-allocated MoE Inference Buffers
// ============================================================================

// Per-device storage for pre-allocated MoE buffers.
static std::array<moe_inference_buffers, GGML_SYCL_MAX_DEVICES> g_moe_buffers{};
static std::mutex                                                g_moe_buffers_mutex;

// Internal: allocate the expert pointer table block as a single contiguous
// allocation, then partition it into per-table pointers.
// Each table is `n_experts * sizeof(void*)` bytes.
// We allocate one big block: n_tables * table_bytes, then point into it.
// This minimizes the number of cache allocations.

bool moe_preallocate_inference_buffers(int device_id, const moe_buffer_params & params) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("[MOE-PREALLOC] invalid device_id=%d\n", device_id);
        return false;
    }
    if (params.n_experts <= 0) {
        GGML_LOG_WARN("[MOE-PREALLOC] n_experts=%d, skipping pre-allocation\n", params.n_experts);
        return true;  // Not an error — model may not be MoE.
    }

    std::lock_guard<std::mutex> lock(g_moe_buffers_mutex);
    moe_inference_buffers & bufs = g_moe_buffers[device_id];
    if (bufs.initialized) {
        return true;  // Already done.
    }

    const int    n_tables    = params.n_moe_layers * std::max(params.n_moe_tensors, 1);
    const size_t table_bytes = static_cast<size_t>(params.n_experts) * sizeof(void *);
    const size_t total_table_bytes = static_cast<size_t>(n_tables) * table_bytes;

    // IDs staging: n_expert_used * max_batch * sizeof(int32_t)
    const size_t ids_bytes = static_cast<size_t>(params.n_expert_used) *
                             static_cast<size_t>(params.max_batch) * sizeof(int32_t);

    GGML_LOG_INFO("[MOE-PREALLOC] device %d: n_tables=%d table_bytes=%zu "
                  "total_table=%.1f KB ids_staging=%.1f KB\n",
                  device_id, n_tables, table_bytes,
                  total_table_bytes / 1024.0, ids_bytes / 1024.0);

    // --- Allocate expert pointer tables ---
    if (n_tables > 0 && table_bytes > 0) {
        auto table_result = unified_cache_allocate(
            device_id, total_table_bytes,
            unified_cache::alloc_lifetime::PERSISTENT,
            "moe_expert_ptr_tables");

        if (!table_result) {
            GGML_LOG_ERROR("[MOE-PREALLOC] failed to allocate expert pointer tables "
                           "(%.1f KB)\n", total_table_bytes / 1024.0);
            return false;
        }

        // Allocate the host-side array of pointers into the contiguous block.
        bufs.expert_ptr_tables = static_cast<void **>(::malloc(
            static_cast<size_t>(n_tables) * sizeof(void *)));
        if (!bufs.expert_ptr_tables) {
            unified_cache_deallocate(device_id, table_result.ptr, total_table_bytes,
                                     unified_cache::alloc_lifetime::PERSISTENT);
            GGML_LOG_ERROR("[MOE-PREALLOC] failed to allocate host table pointer array\n");
            return false;
        }

        // Partition the contiguous device block into per-table pointers.
        auto * base = static_cast<uint8_t *>(table_result.ptr);
        for (int i = 0; i < n_tables; i++) {
            bufs.expert_ptr_tables[i] = base + static_cast<size_t>(i) * table_bytes;
        }

        bufs.n_tables         = n_tables;
        bufs.table_bytes      = table_bytes;
        bufs.tables_on_device = table_result.on_device;

        // Zero-fill the tables (they'll be populated by update_moe_ptr_table).
        auto * cache = get_unified_cache_for_device(device_id);
        if (cache && table_result.on_device) {
            try {
                cache->get_queue().memset(table_result.ptr, 0, total_table_bytes).wait();
            } catch (...) {
                GGML_LOG_WARN("[MOE-PREALLOC] memset of expert pointer tables failed\n");
            }
        }
    }

    // --- Allocate MoE IDs staging ---
    if (ids_bytes > 0) {
        auto ids_result = unified_cache_allocate(
            device_id, ids_bytes,
            unified_cache::alloc_lifetime::PERSISTENT,
            "moe_ids_staging");

        if (!ids_result) {
            GGML_LOG_ERROR("[MOE-PREALLOC] failed to allocate IDs staging "
                           "(%.1f KB)\n", ids_bytes / 1024.0);
            // Tables were allocated — leave them, partial success.
        } else {
            bufs.ids_staging       = ids_result.ptr;
            bufs.ids_staging_bytes = ids_bytes;
            bufs.ids_on_device     = ids_result.on_device;
        }
    }

    bufs.initialized = true;

    GGML_LOG_INFO("[MOE-PREALLOC] device %d: tables=%s (%d x %zu B) ids=%s (%.1f KB)\n",
                  device_id,
                  bufs.tables_on_device ? "VRAM" : "host",
                  bufs.n_tables, bufs.table_bytes,
                  bufs.ids_on_device ? "VRAM" : "host",
                  bufs.ids_staging_bytes / 1024.0);

    return true;
}

const moe_inference_buffers * moe_get_inference_buffers(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_moe_buffers_mutex);
    if (!g_moe_buffers[device_id].initialized) {
        return nullptr;
    }
    return &g_moe_buffers[device_id];
}

void * moe_get_expert_ptr_table(int device_id, int table_index) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    // No lock needed — read-only after initialization.
    const moe_inference_buffers & bufs = g_moe_buffers[device_id];
    if (!bufs.initialized || !bufs.expert_ptr_tables) {
        return nullptr;
    }
    if (table_index < 0 || table_index >= bufs.n_tables) {
        return nullptr;
    }
    return bufs.expert_ptr_tables[table_index];
}

void * moe_get_ids_staging(int device_id, size_t needed_bytes) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }
    // No lock needed — read-only after initialization.
    const moe_inference_buffers & bufs = g_moe_buffers[device_id];
    if (!bufs.initialized || !bufs.ids_staging) {
        return nullptr;
    }
    if (needed_bytes > bufs.ids_staging_bytes) {
        GGML_SYCL_DEBUG("[MOE-PREALLOC] ids_staging too small: need %zu have %zu\n",
                        needed_bytes, bufs.ids_staging_bytes);
        return nullptr;
    }
    return bufs.ids_staging;
}

void moe_free_inference_buffers(int device_id) {
    if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_moe_buffers_mutex);
    moe_inference_buffers & bufs = g_moe_buffers[device_id];
    if (!bufs.initialized) {
        return;
    }

    // Free the contiguous table block (first table pointer is the base).
    if (bufs.expert_ptr_tables && bufs.n_tables > 0) {
        void * base = bufs.expert_ptr_tables[0];
        if (base) {
            const size_t total = static_cast<size_t>(bufs.n_tables) * bufs.table_bytes;
            unified_cache_deallocate(device_id, base, total,
                                     unified_cache::alloc_lifetime::PERSISTENT);
        }
        ::free(bufs.expert_ptr_tables);
    }

    // Free IDs staging.
    if (bufs.ids_staging) {
        unified_cache_deallocate(device_id, bufs.ids_staging, bufs.ids_staging_bytes,
                                 unified_cache::alloc_lifetime::PERSISTENT);
    }

    bufs = {};  // Reset to zero state.
}

// === VRAM Arena Implementation ===

bool vram_arena_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_SYCL_VRAM_ARENA");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

// --- Device pool arena helpers (called from device-pool.hpp inline methods) ---

void * device_pool_arena_alloc(vram_arena * arena, size_t size, size_t align) {
    if (!arena || !arena->active()) {
        return nullptr;
    }
    return arena->zone_alloc(vram_zone_id::WEIGHT, size, align);
}

bool device_pool_arena_owns(const vram_arena * arena, const void * ptr) {
    if (!arena || !arena->active()) {
        return false;
    }
    return arena->owns(ptr);
}

vram_arena::~vram_arena() {
    destroy();
}

bool vram_arena::reserve(sycl::queue & queue, size_t budget_bytes, size_t max_alloc_size,
                         size_t compute_bytes, size_t onednn_bytes) {
    if (arena_base_) {
        return true;  // Already reserved.
    }

    queue_ = &queue;

    // Align zone sizes to 256 bytes.
    compute_bytes = (compute_bytes + 255) & ~size_t(255);
    onednn_bytes  = (onednn_bytes + 255) & ~size_t(255);

    // Try single allocation first.
    void * ptr = nullptr;
    size_t alloc_size = budget_bytes;

    if (alloc_size <= max_alloc_size) {
        try {
            ptr = sycl::malloc_device(alloc_size, queue);
        } catch (const sycl::exception & e) {
            GGML_LOG_WARN("[VRAM-ARENA] Single alloc (%.1f MB) failed: %s\n",
                          alloc_size / (1024.0 * 1024.0), e.what());
            ptr = nullptr;
        }
    }

    if (!ptr) {
        // Try 2-chunk split: 50% + 40% of budget (leaves 10% margin for driver)
        const size_t chunk0_size = (budget_bytes * 50) / 100;
        const size_t chunk1_size = (budget_bytes * 40) / 100;

        if (chunk0_size > max_alloc_size || chunk1_size > max_alloc_size) {
            GGML_LOG_WARN("[VRAM-ARENA] 2-chunk sizes (%.1f + %.1f MB) exceed max_alloc (%.1f MB)\n",
                          chunk0_size / (1024.0 * 1024.0), chunk1_size / (1024.0 * 1024.0),
                          max_alloc_size / (1024.0 * 1024.0));
            return false;
        }

        void * p0 = nullptr;
        void * p1 = nullptr;
        try {
            p0 = sycl::malloc_device(chunk0_size, queue);
        } catch (...) { p0 = nullptr; }
        if (!p0) {
            GGML_LOG_WARN("[VRAM-ARENA] 2-chunk: first chunk (%.1f MB) failed\n",
                          chunk0_size / (1024.0 * 1024.0));
            return false;
        }
        try {
            p1 = sycl::malloc_device(chunk1_size, queue);
        } catch (...) { p1 = nullptr; }
        if (!p1) {
            sycl::free(p0, queue);
            GGML_LOG_WARN("[VRAM-ARENA] 2-chunk: second chunk (%.1f MB) failed\n",
                          chunk1_size / (1024.0 * 1024.0));
            return false;
        }

        // chunk0: compute+oneDNN+KV, chunk1: weights
        chunks_[0] = { p0, chunk0_size };
        chunks_[1] = { p1, chunk1_size };
        n_chunks_  = 2;
        arena_base_ = p0;
        arena_size_ = chunk0_size + chunk1_size;

        // Layout zones within 2-chunk arena:
        // chunk0: [Compute zone] [oneDNN zone] [KV zone]
        // chunk1: [Weight zone — entire chunk]
        size_t off = 0;
        auto & cz  = zones_[static_cast<int>(vram_zone_id::COMPUTE)];
        cz.start   = off;
        cz.size    = std::min(compute_bytes, chunk0_size);
        cz.used.store(0, std::memory_order_relaxed);
        off += cz.size;

        auto & oz  = zones_[static_cast<int>(vram_zone_id::ONEDNN)];
        oz.start   = off;
        oz.size    = std::min(onednn_bytes, chunk0_size - off);
        oz.used.store(0, std::memory_order_relaxed);
        off += oz.size;

        auto & kz  = zones_[static_cast<int>(vram_zone_id::KV)];
        kz.start   = off;
        kz.size    = chunk0_size - off;
        kz.used.store(0, std::memory_order_relaxed);

        // Weight zone occupies all of chunk1.
        auto & wz  = zones_[static_cast<int>(vram_zone_id::WEIGHT)];
        wz.start   = chunk0_size;  // Offset from arena_base
        wz.size    = chunk1_size;
        wz.used.store(0, std::memory_order_relaxed);

        GGML_LOG_INFO("[VRAM-ARENA] Reserved 2-chunk: %.1f + %.1f MB "
                      "(compute=%.1f, oneDNN=%.1f, KV=%.1f, weight=%.1f MB)\n",
                      chunk0_size / (1024.0 * 1024.0), chunk1_size / (1024.0 * 1024.0),
                      cz.size / (1024.0 * 1024.0), oz.size / (1024.0 * 1024.0),
                      kz.size / (1024.0 * 1024.0), wz.size / (1024.0 * 1024.0));
        return true;
    }

    // Single chunk succeeded.
    chunks_[0]  = { ptr, alloc_size };
    n_chunks_   = 1;
    arena_base_ = ptr;
    arena_size_ = alloc_size;

    // Layout: [Compute] [oneDNN] [KV (grows→) ... Weight (←grows)]
    size_t off = 0;
    auto & cz  = zones_[static_cast<int>(vram_zone_id::COMPUTE)];
    cz.start   = off;
    cz.size    = compute_bytes;
    cz.used.store(0, std::memory_order_relaxed);
    off += compute_bytes;

    auto & oz  = zones_[static_cast<int>(vram_zone_id::ONEDNN)];
    oz.start   = off;
    oz.size    = onednn_bytes;
    oz.used.store(0, std::memory_order_relaxed);
    off += onednn_bytes;

    // KV and Weight share the remaining space.  KV grows right, Weight grows left.
    const size_t shared = alloc_size - off;
    auto & kz  = zones_[static_cast<int>(vram_zone_id::KV)];
    kz.start   = off;
    kz.size    = shared;
    kz.used.store(0, std::memory_order_relaxed);

    auto & wz  = zones_[static_cast<int>(vram_zone_id::WEIGHT)];
    wz.start   = off;      // Same region as KV (they grow toward each other)
    wz.size    = shared;
    wz.used.store(0, std::memory_order_relaxed);

    GGML_LOG_INFO("[VRAM-ARENA] Reserved single chunk: %.1f MB "
                  "(compute=%.1f, oneDNN=%.1f, shared KV+weight=%.1f MB)\n",
                  alloc_size / (1024.0 * 1024.0), cz.size / (1024.0 * 1024.0),
                  oz.size / (1024.0 * 1024.0), shared / (1024.0 * 1024.0));
    return true;
}

void * vram_arena::zone_alloc(vram_zone_id zone, size_t size, size_t align) {
    if (!arena_base_ || size == 0) {
        return nullptr;
    }

    auto & z = zones_[static_cast<int>(zone)];

    if (align == 0 || (align & (align - 1)) != 0) {
        align = 256;
    }

    if (zone == vram_zone_id::WEIGHT) {
        // Weight zone: bump-LEFT from end.
        // Check free-list first (best-fit).
        {
            std::lock_guard<std::mutex> lock(z.free_list_mutex);
            size_t best_idx   = SIZE_MAX;
            size_t best_waste = SIZE_MAX;
            for (size_t i = 0; i < z.free_list.size(); i++) {
                auto & blk = z.free_list[i];
                size_t aligned_off = (blk.offset + align - 1) & ~(align - 1);
                size_t avail = (blk.offset + blk.size > aligned_off)
                                   ? (blk.offset + blk.size - aligned_off) : 0;
                if (avail >= size) {
                    size_t waste = avail - size;
                    if (waste < best_waste) {
                        best_waste = waste;
                        best_idx   = i;
                    }
                }
            }
            if (best_idx != SIZE_MAX) {
                auto & blk = z.free_list[best_idx];
                size_t aligned_off = (blk.offset + align - 1) & ~(align - 1);
                void * ptr = offset_to_ptr(aligned_off);

                size_t used_end = aligned_off + size;
                if (used_end >= blk.offset + blk.size) {
                    if (aligned_off > blk.offset) {
                        blk.size = aligned_off - blk.offset;
                    } else {
                        z.free_list.erase(z.free_list.begin() +
                                          static_cast<ptrdiff_t>(best_idx));
                    }
                } else {
                    size_t remain_off  = used_end;
                    size_t remain_size = (blk.offset + blk.size) - remain_off;
                    if (aligned_off > blk.offset) {
                        blk.size = aligned_off - blk.offset;
                        z.free_list.push_back({ remain_off, remain_size });
                    } else {
                        blk.offset = remain_off;
                        blk.size   = remain_size;
                    }
                }
                return ptr;
            }
        }

        // No free-list hit.  Bump-left from end of zone.
        const size_t aligned_size = (size + align - 1) & ~(align - 1);

        if (n_chunks_ == 1) {
            // Collision check: KV used + weight used must not exceed shared zone.
            // Safe: KV allocation (context creation) and weight allocation (model load) never overlap in practice.
            const auto & kz = zones_[static_cast<int>(vram_zone_id::KV)];
            size_t prev = z.used.fetch_add(aligned_size, std::memory_order_relaxed);
            size_t kv_used = kz.used.load(std::memory_order_relaxed);
            if (kv_used + prev + aligned_size > z.size) {
                z.used.fetch_sub(aligned_size, std::memory_order_relaxed);
                return nullptr;
            }
            size_t zone_end = z.start + z.size;
            size_t alloc_off = zone_end - prev - aligned_size;
            alloc_off = alloc_off & ~(align - 1);
            return offset_to_ptr(alloc_off);
        }

        // 2-chunk: weight zone is standalone.
        size_t prev = z.used.fetch_add(aligned_size, std::memory_order_relaxed);
        if (prev + aligned_size > z.size) {
            z.used.fetch_sub(aligned_size, std::memory_order_relaxed);
            return nullptr;
        }
        size_t zone_end = z.start + z.size;
        size_t alloc_off = zone_end - prev - aligned_size;
        alloc_off = alloc_off & ~(align - 1);
        return offset_to_ptr(alloc_off);
    }

    // Non-weight zones: bump-right from zone start.
    const size_t aligned_size = (size + align - 1) & ~(align - 1);

    if (zone == vram_zone_id::KV && n_chunks_ == 1) {
        // Collision check against weight zone.
        const auto & wz = zones_[static_cast<int>(vram_zone_id::WEIGHT)];
        size_t prev = z.used.fetch_add(aligned_size, std::memory_order_relaxed);
        size_t weight_used = wz.used.load(std::memory_order_relaxed);
        if (prev + aligned_size + weight_used > z.size) {
            z.used.fetch_sub(aligned_size, std::memory_order_relaxed);
            return nullptr;
        }
        return offset_to_ptr(z.start + prev);
    }

    // Simple bump-right for compute and onednn zones.
    size_t prev = z.used.fetch_add(aligned_size, std::memory_order_relaxed);
    if (prev + aligned_size > z.size) {
        z.used.fetch_sub(aligned_size, std::memory_order_relaxed);
        return nullptr;
    }

    return offset_to_ptr(z.start + prev);
}

void vram_arena::zone_reset(vram_zone_id zone) {
    zones_[static_cast<int>(zone)].used.store(0, std::memory_order_relaxed);
}

void vram_arena::weight_reclaim(size_t offset, size_t size) {
    auto & wz = zones_[static_cast<int>(vram_zone_id::WEIGHT)];
    std::lock_guard<std::mutex> lock(wz.free_list_mutex);

    // Insert sorted by offset, then coalesce adjacent blocks.
    auto it = wz.free_list.begin();
    while (it != wz.free_list.end() && it->offset < offset) {
        ++it;
    }
    it = wz.free_list.insert(it, { offset, size });

    // Coalesce with next.
    auto next = std::next(it);
    if (next != wz.free_list.end() && it->offset + it->size == next->offset) {
        it->size += next->size;
        wz.free_list.erase(next);
    }
    // Coalesce with prev.
    if (it != wz.free_list.begin()) {
        auto prev_it = std::prev(it);
        if (prev_it->offset + prev_it->size == it->offset) {
            prev_it->size += it->size;
            wz.free_list.erase(it);
        }
    }
}

bool vram_arena::owns(const void * ptr) const {
    if (!arena_base_ || !ptr) {
        return false;
    }
    for (int i = 0; i < n_chunks_; i++) {
        auto base = reinterpret_cast<uintptr_t>(chunks_[i].ptr);
        auto p    = reinterpret_cast<uintptr_t>(ptr);
        if (p >= base && p < base + chunks_[i].size) {
            return true;
        }
    }
    return false;
}

bool vram_arena::zone_owns(vram_zone_id zone, const void * ptr) const {
    if (!arena_base_ || !ptr) {
        return false;
    }
    size_t off = ptr_to_offset(ptr);
    if (off == SIZE_MAX) {
        return false;
    }
    const auto & z = zones_[static_cast<int>(zone)];
    return off >= z.start && off < z.start + z.size;
}

size_t vram_arena::ptr_to_offset(const void * ptr) const {
    if (!ptr) {
        return SIZE_MAX;
    }
    for (int i = 0; i < n_chunks_; i++) {
        auto base = reinterpret_cast<uintptr_t>(chunks_[i].ptr);
        auto p    = reinterpret_cast<uintptr_t>(ptr);
        if (p >= base && p < base + chunks_[i].size) {
            if (i == 0) {
                return static_cast<size_t>(p - base);
            }
            return chunks_[0].size + static_cast<size_t>(p - base);
        }
    }
    return SIZE_MAX;
}

void * vram_arena::offset_to_ptr(size_t offset) const {
    if (n_chunks_ == 1) {
        return static_cast<uint8_t *>(arena_base_) + offset;
    }
    if (offset < chunks_[0].size) {
        return static_cast<uint8_t *>(chunks_[0].ptr) + offset;
    }
    size_t chunk1_off = offset - chunks_[0].size;
    return static_cast<uint8_t *>(chunks_[1].ptr) + chunk1_off;
}

size_t vram_arena::zone_capacity(vram_zone_id zone) const {
    return zones_[static_cast<int>(zone)].size;
}

size_t vram_arena::zone_used(vram_zone_id zone) const {
    return zones_[static_cast<int>(zone)].used.load(std::memory_order_relaxed);
}

void vram_arena::destroy() {
    if (!queue_) {
        return;
    }
    for (int i = 0; i < n_chunks_; i++) {
        if (chunks_[i].ptr) {
            try {
                sycl::free(chunks_[i].ptr, *queue_);
            } catch (...) {}
            chunks_[i] = {};
        }
    }
    arena_base_ = nullptr;
    arena_size_ = 0;
    n_chunks_   = 0;
    for (int i = 0; i < static_cast<int>(vram_zone_id::COUNT); i++) {
        zones_[i].start = 0;
        zones_[i].size  = 0;
        zones_[i].used.store(0, std::memory_order_relaxed);
        zones_[i].free_list.clear();
    }
}

void vram_arena::abandon() {
    // Null everything without calling sycl::free — used during shutdown
    // when the SYCL context is already invalid.
    for (int i = 0; i < n_chunks_; i++) {
        chunks_[i] = {};
    }
    arena_base_ = nullptr;
    arena_size_ = 0;
    n_chunks_   = 0;
    for (int i = 0; i < static_cast<int>(vram_zone_id::COUNT); i++) {
        zones_[i].start = 0;
        zones_[i].size  = 0;
        zones_[i].used.store(0, std::memory_order_relaxed);
        zones_[i].free_list.clear();
    }
}

// === P4: Priority-based static placement planner ===

// Map tensor_usage + name to placement priority.
// For MoE experts, name sub-classification distinguishes gate/down/up.
static placement_priority tensor_to_placement_priority(tensor_usage usage, const char * name) {
    switch (usage) {
        case tensor_usage::NORM:
        case tensor_usage::EMBEDDING:
            return placement_priority::NORM_EMBED;
        case tensor_usage::ATTENTION_WEIGHT:
            return placement_priority::ATTENTION;
        case tensor_usage::FFN_WEIGHT:
            return placement_priority::FFN;
        case tensor_usage::MOE_GATE:
            return placement_priority::MOE_GATE;
        case tensor_usage::MOE_EXPERT_WEIGHT:
            // Unsloth-informed sub-classification: down > up > gate
            if (name && strstr(name, "ffn_down_exps")) {
                return placement_priority::MOE_DOWN;
            }
            if (name && strstr(name, "ffn_up_exps")) {
                return placement_priority::MOE_UP;
            }
            // ffn_gate_exps or unknown expert pattern
            return placement_priority::MOE_GATE_PROJ;
        case tensor_usage::MOE_INTERMEDIATE:
            return placement_priority::NORM_EMBED;  // Small, treat like norms
        case tensor_usage::UNKNOWN:
        default:
            // Unknown tensors get FFN priority (middle of the pack)
            return placement_priority::FFN;
    }
}

// Extract layer number from tensor name (e.g. "blk.5.attn_q" -> 5).
static int p4_extract_layer_id(const char * name) {
    if (!name) return -1;
    const char * blk = strstr(name, "blk.");
    if (!blk) return -1;
    return std::atoi(blk + 4);
}

placement_plan compute_placement_plan(
    const std::vector<std::pair<std::string, size_t>> & tensor_inventory,
    size_t vram_budget,
    int    device_id) {

    placement_plan plan;
    plan.vram_budget  = vram_budget;
    plan.device_id    = device_id;
    plan.multi_device = false;
    plan.vram_bytes   = 0;
    plan.host_bytes   = 0;

    // Build entries with priority classification
    plan.entries.reserve(tensor_inventory.size());
    for (const auto & [name, src_size] : tensor_inventory) {
        placement_entry entry;
        entry.name     = name;
        entry.src_size = src_size;
        entry.dst_size = src_size;  // Default: same as source (AOS)
        entry.layer_id = p4_extract_layer_id(name.c_str());

        // Classify usage and map to priority
        const tensor_usage usage = infer_tensor_usage(name.c_str());
        entry.priority = tensor_to_placement_priority(usage, name.c_str());

        // MoE expert tensors are composite (contain all experts for a layer).
        // They compete for VRAM at the full tensor level during S1-PRELOAD.
        entry.on_device      = false;      // Will be set during packing below
        entry.target_device  = -1;         // -1 = host (updated during packing)
        plan.entries.push_back(std::move(entry));
    }

    // Sort by (priority ASC, layer_id ASC, dst_size DESC).
    // This ensures highest-priority weights fill VRAM first, earlier layers
    // before later ones, and larger weights before smaller within the same
    // priority+layer (to minimize fragmentation from small leftovers).
    std::sort(plan.entries.begin(), plan.entries.end(),
        [](const placement_entry & a, const placement_entry & b) {
            if (a.priority != b.priority) {
                return static_cast<uint8_t>(a.priority) < static_cast<uint8_t>(b.priority);
            }
            if (a.layer_id != b.layer_id) {
                return a.layer_id < b.layer_id;
            }
            return a.dst_size > b.dst_size;
        });

    // Greedy bin-packing: fill VRAM up to budget, spill the rest to host.
    // When the VRAM arena is active, subtract compute scratch and oneDNN
    // scratch zone capacities from the budget BEFORE packing weights.
    // Without this, weights fill all available VRAM and leave zero space
    // for compute scratch → "VRAM exhaustion" abort on large models.
    size_t remaining = vram_budget;
    if (vram_arena_enabled()) {
        auto * cache = get_unified_cache_for_device(device_id);
        if (cache && cache->get_arena().active()) {
            const size_t compute_reserve = cache->get_arena().zone_capacity(vram_zone_id::COMPUTE);
            const size_t onednn_reserve  = cache->get_arena().zone_capacity(vram_zone_id::ONEDNN);
            const size_t total_reserve   = compute_reserve + onednn_reserve;
            if (remaining > total_reserve) {
                remaining -= total_reserve;
            } else {
                remaining = 0;
            }
            GGML_LOG_INFO("[PLACEMENT] Scratch reservation: compute=%.1f MB + oneDNN=%.1f MB = %.1f MB "
                          "(weight budget=%.1f MB)\n",
                          compute_reserve / (1024.0 * 1024.0),
                          onednn_reserve / (1024.0 * 1024.0),
                          total_reserve / (1024.0 * 1024.0),
                          remaining / (1024.0 * 1024.0));
        }
    }
    for (auto & entry : plan.entries) {
        if (entry.dst_size <= remaining) {
            entry.on_device     = true;
            entry.target_device = device_id;
            remaining -= entry.dst_size;
            plan.vram_bytes += entry.dst_size;
        } else {
            entry.on_device     = false;
            entry.target_device = -1;
            plan.host_bytes += entry.dst_size;
        }
    }

    // Build the name->index lookup for O(1) queries
    plan.build_index();

    // Log placement summary per priority level
    static const char * priority_names[] = {
        "NORM/EMBED", "ATTENTION", "FFN", "MOE_GATE",
        "MOE_DOWN", "MOE_UP", "MOE_GATE_PROJ"
    };
    for (int p = 0; p < static_cast<int>(placement_priority::COUNT); ++p) {
        size_t device_count = 0, host_count = 0;
        size_t device_bytes = 0, host_bytes = 0;
        for (const auto & e : plan.entries) {
            if (static_cast<int>(e.priority) == p) {
                if (e.on_device) {
                    device_count++;
                    device_bytes += e.dst_size;
                } else {
                    host_count++;
                    host_bytes += e.dst_size;
                }
            }
        }
        if (device_count + host_count > 0) {
            GGML_LOG_INFO("[PLACEMENT] %-14s  device=%3zu (%.1f MB)  host=%3zu (%.1f MB)\n",
                          priority_names[p],
                          device_count, device_bytes / (1024.0 * 1024.0),
                          host_count, host_bytes / (1024.0 * 1024.0));
        }
    }
    GGML_LOG_INFO("[PLACEMENT] Total: %.1f MB device + %.1f MB host (budget=%.1f MB)\n",
                  plan.vram_bytes / (1024.0 * 1024.0),
                  plan.host_bytes / (1024.0 * 1024.0),
                  vram_budget / (1024.0 * 1024.0));

    return plan;
}

// ---------------------------------------------------------------------------
// P4.5: Multi-device placement planning with hybrid parallelism.
// ---------------------------------------------------------------------------
//
// Algorithm:
//   1. Compute per-device budgets and total VRAM pool.
//   2. For DENSE layers: assign contiguous layer ranges proportional to VRAM.
//      Each device gets a range [layer_start, layer_end).  Dense weights for
//      a layer go entirely to the owning device.
//   3. For MoE EXPERT tensors: pool remaining VRAM across all devices and
//      fill by Unsloth priority (gate > down > up, earlier layers first).
//      Each expert tensor is assigned to the device with the most remaining
//      budget (first-fit-decreasing across devices).
//   4. Overflow (dense or expert) spills to host (-1).
//
// Falls back to single-device compute_placement_plan() when only 1 device.

multi_gpu_mode get_multi_gpu_mode(bool is_moe) {
    const char * env = std::getenv("GGML_SYCL_MULTI_GPU_MODE");
    if (env) {
        if (std::strcmp(env, "layer") == 0)  return multi_gpu_mode::LAYER;
        if (std::strcmp(env, "expert") == 0) return multi_gpu_mode::EXPERT;
        if (std::strcmp(env, "hybrid") == 0) return multi_gpu_mode::HYBRID;
        GGML_LOG_WARN("[PLACEMENT-MULTI] Unknown GGML_SYCL_MULTI_GPU_MODE='%s', "
                      "using default\n", env);
    }
    // Default: hybrid for MoE, layer for dense-only
    return is_moe ? multi_gpu_mode::HYBRID : multi_gpu_mode::LAYER;
}

placement_plan compute_multi_device_plan(
    const std::vector<device_budget> &                  device_budgets,
    const std::vector<std::pair<std::string, size_t>> & tensor_inventory,
    int                                                 n_layers,
    multi_gpu_mode                                      mode) {

    // Single device: delegate to existing P4 path
    if (device_budgets.size() <= 1) {
        const int    dev    = device_budgets.empty() ? 0 : device_budgets[0].device_id;
        const size_t budget = device_budgets.empty() ? 0 : device_budgets[0].vram_budget;
        return compute_placement_plan(tensor_inventory, budget, dev);
    }

    const size_t n_devs = device_budgets.size();

    placement_plan plan;
    plan.device_id    = -1;  // Multi-device
    plan.multi_device = true;
    plan.vram_bytes   = 0;
    plan.host_bytes   = 0;
    plan.vram_budget  = 0;

    // Store device list and per-device tracking
    plan.devices.resize(n_devs);
    plan.per_device_vram.resize(n_devs, 0);
    std::vector<size_t> remaining(n_devs);
    for (size_t d = 0; d < n_devs; d++) {
        plan.devices[d]    = device_budgets[d].device_id;
        remaining[d]       = device_budgets[d].vram_budget;

        // Subtract arena compute + oneDNN zone reservations from per-device budget
        // so weights don't fill space needed for compute scratch.
        if (vram_arena_enabled()) {
            auto * cache = get_unified_cache_for_device(device_budgets[d].device_id);
            if (cache && cache->get_arena().active()) {
                const size_t reserve = cache->get_arena().zone_capacity(vram_zone_id::COMPUTE)
                                     + cache->get_arena().zone_capacity(vram_zone_id::ONEDNN);
                remaining[d] = remaining[d] > reserve ? remaining[d] - reserve : 0;
            }
        }

        plan.vram_budget  += device_budgets[d].vram_budget;
    }

    // Step 1: Compute layer-to-device assignment (dense layers).
    // Proportional to total VRAM, so bigger GPU gets more layers.
    // layer_owner[l] = device index in device_budgets (not device_id).
    std::vector<int> layer_owner(n_layers, 0);
    {
        size_t total_vram = 0;
        for (const auto & db : device_budgets) {
            total_vram += db.total_vram;
        }
        int layer_cursor = 0;
        for (size_t d = 0; d < n_devs; d++) {
            double fraction = static_cast<double>(device_budgets[d].total_vram)
                              / static_cast<double>(total_vram);
            int n_dev_layers = static_cast<int>(fraction * n_layers + 0.5);
            if (d == n_devs - 1) {
                n_dev_layers = n_layers - layer_cursor;
            }
            for (int l = 0; l < n_dev_layers && layer_cursor < n_layers; l++, layer_cursor++) {
                layer_owner[layer_cursor] = static_cast<int>(d);
            }
        }
    }

    // Step 2: Build entries with priority classification (same as P4).
    plan.entries.reserve(tensor_inventory.size());
    for (const auto & [name, src_size] : tensor_inventory) {
        placement_entry entry;
        entry.name          = name;
        entry.src_size      = src_size;
        entry.dst_size      = src_size;
        entry.layer_id      = p4_extract_layer_id(name.c_str());
        entry.on_device     = false;
        entry.target_device = -1;

        const tensor_usage usage = infer_tensor_usage(name.c_str());
        entry.priority = tensor_to_placement_priority(usage, name.c_str());

        plan.entries.push_back(std::move(entry));
    }

    // Step 3: Sort by (priority ASC, layer_id ASC, dst_size DESC).
    std::sort(plan.entries.begin(), plan.entries.end(),
        [](const placement_entry & a, const placement_entry & b) {
            if (a.priority != b.priority) {
                return static_cast<uint8_t>(a.priority) < static_cast<uint8_t>(b.priority);
            }
            if (a.layer_id != b.layer_id) {
                return a.layer_id < b.layer_id;
            }
            return a.dst_size > b.dst_size;
        });

    // Step 4: Pack entries into devices.
    // Behavior depends on parallelism mode:
    //   LAYER:  All tensors (dense + MoE) assigned by layer owner.
    //   EXPERT: Dense to primary device; MoE experts distributed across all.
    //   HYBRID: Dense by layer owner; MoE experts distributed across all.

    auto is_moe_priority = [](placement_priority p) {
        return p == placement_priority::MOE_GATE ||
               p == placement_priority::MOE_DOWN ||
               p == placement_priority::MOE_UP   ||
               p == placement_priority::MOE_GATE_PROJ;
    };

    static const char * mode_names[] = { "LAYER", "EXPERT", "HYBRID" };
    GGML_LOG_INFO("[PLACEMENT-MULTI] Mode: %s\n",
                  mode_names[static_cast<int>(mode)]);

    for (auto & entry : plan.entries) {
        int target_dev_idx = -1;

        const bool is_moe = is_moe_priority(entry.priority);

        if (is_moe && mode != multi_gpu_mode::LAYER) {
            // EXPERT or HYBRID: distribute MoE experts across all devices
            size_t best_remaining = 0;
            for (size_t d = 0; d < n_devs; d++) {
                if (remaining[d] >= entry.dst_size && remaining[d] > best_remaining) {
                    best_remaining = remaining[d];
                    target_dev_idx = static_cast<int>(d);
                }
            }
        } else if (!is_moe && mode == multi_gpu_mode::EXPERT) {
            // EXPERT mode: all dense layers on primary device
            target_dev_idx = 0;
            if (remaining[0] < entry.dst_size) {
                target_dev_idx = -1;  // Spill to host
            }
        } else {
            // LAYER or HYBRID for dense: assign by layer owner
            int layer_id = entry.layer_id;
            if (layer_id >= 0 && layer_id < n_layers) {
                target_dev_idx = layer_owner[layer_id];
            } else {
                target_dev_idx = 0;
            }
            // Check if owning device has budget
            if (target_dev_idx >= 0 &&
                static_cast<size_t>(target_dev_idx) < n_devs &&
                remaining[target_dev_idx] < entry.dst_size) {
                // Owning device full — try any other device with space
                int fallback = -1;
                size_t best_remaining = 0;
                for (size_t d = 0; d < n_devs; d++) {
                    if (remaining[d] >= entry.dst_size && remaining[d] > best_remaining) {
                        best_remaining = remaining[d];
                        fallback       = static_cast<int>(d);
                    }
                }
                target_dev_idx = fallback;
            }
        }

        if (target_dev_idx >= 0 &&
            static_cast<size_t>(target_dev_idx) < n_devs &&
            remaining[target_dev_idx] >= entry.dst_size) {
            entry.on_device     = true;
            entry.target_device = device_budgets[target_dev_idx].device_id;
            remaining[target_dev_idx] -= entry.dst_size;
            plan.vram_bytes += entry.dst_size;
            plan.per_device_vram[target_dev_idx] += entry.dst_size;
        } else {
            entry.on_device     = false;
            entry.target_device = -1;
            plan.host_bytes += entry.dst_size;
        }
    }

    plan.build_index();

    // Build runtime query maps for multi-device inference routing.

    // device_layers: per-device contiguous layer range
    for (size_t d = 0; d < n_devs; d++) {
        int dev_id = device_budgets[d].device_id;
        int first  = n_layers, last = -1;
        for (int l = 0; l < n_layers; l++) {
            if (layer_owner[l] == static_cast<int>(d)) {
                first = std::min(first, l);
                last  = std::max(last, l);
            }
        }
        if (first <= last) {
            plan.device_layers[dev_id] = { first, last };
        }
    }

    // kv_device: KV cache for each layer co-located with its dense weight device
    for (int l = 0; l < n_layers; l++) {
        int owner_idx = layer_owner[l];
        plan.kv_device[l] = device_budgets[owner_idx].device_id;
    }

    // expert_device: per-layer per-expert device assignment (from entry target_device).
    // MoE expert tensors contain ALL experts for a layer in one blob.
    // The plan assigns the whole tensor to one device, so all experts in that
    // tensor share the same target.  Individual expert routing is handled
    // at runtime by the unified cache's per-expert caching.
    for (const auto & entry : plan.entries) {
        if (!is_moe_priority(entry.priority)) continue;
        if (entry.layer_id < 0) continue;
        // All experts within this tensor go to target_device
        plan.expert_device[entry.layer_id][0] = entry.target_device;
    }

    // Log multi-device placement summary
    GGML_LOG_INFO("[PLACEMENT-MULTI] %zu devices, %d layers, hybrid parallelism\n",
                  n_devs, n_layers);
    for (size_t d = 0; d < n_devs; d++) {
        int dev_id      = device_budgets[d].device_id;
        int first_layer = n_layers, last_layer = -1;
        for (int l = 0; l < n_layers; l++) {
            if (layer_owner[l] == static_cast<int>(d)) {
                first_layer = std::min(first_layer, l);
                last_layer  = std::max(last_layer, l);
            }
        }
        GGML_LOG_INFO("[PLACEMENT-MULTI] Device %d: layers [%d, %d], "
                      "%.1f MB VRAM used (%.1f MB budget)\n",
                      device_budgets[d].device_id, first_layer, last_layer,
                      plan.per_device_vram[d] / (1024.0 * 1024.0),
                      device_budgets[d].vram_budget / (1024.0 * 1024.0));
    }

    // Per-priority breakdown
    static const char * priority_names[] = {
        "NORM/EMBED", "ATTENTION", "FFN", "MOE_GATE",
        "MOE_DOWN", "MOE_UP", "MOE_GATE_PROJ"
    };
    for (int p = 0; p < static_cast<int>(placement_priority::COUNT); ++p) {
        size_t dev_count = 0, host_count = 0;
        size_t dev_bytes = 0, host_bytes = 0;
        for (const auto & e : plan.entries) {
            if (static_cast<int>(e.priority) == p) {
                if (e.on_device) {
                    dev_count++;
                    dev_bytes += e.dst_size;
                } else {
                    host_count++;
                    host_bytes += e.dst_size;
                }
            }
        }
        if (dev_count + host_count > 0) {
            GGML_LOG_INFO("[PLACEMENT-MULTI] %-14s  device=%3zu (%.1f MB)  host=%3zu (%.1f MB)\n",
                          priority_names[p],
                          dev_count, dev_bytes / (1024.0 * 1024.0),
                          host_count, host_bytes / (1024.0 * 1024.0));
        }
    }
    GGML_LOG_INFO("[PLACEMENT-MULTI] Total: %.1f MB device + %.1f MB host (budget=%.1f MB)\n",
                  plan.vram_bytes / (1024.0 * 1024.0),
                  plan.host_bytes / (1024.0 * 1024.0),
                  plan.vram_budget / (1024.0 * 1024.0));

    return plan;
}

}  // namespace ggml_sycl
