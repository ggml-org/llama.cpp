//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "unified-cache.hpp"

#include "alloc-registry.hpp"
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
static std::mutex            g_runtime_alloc_mutex;
static std::atomic<uint64_t> g_runtime_alloc_id{ 1 };

struct runtime_alloc_record {
    alloc_handle  handle{};
    sycl::queue * queue            = nullptr;
    bool          uses_pinned_pool = false;
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

    // Initialize layout pool for consolidating layout allocations into
    // contiguous chunks (reduces GPU TLB misses from scattered USM mappings).
    layout_pool_ = std::make_unique<sycl_device_pool>(queue_);

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
        return;
    }

    // Free all cached entries (skip pool-allocated entries; the pool destructor frees those)
    for (auto & pair : entries_) {
        if (pair.second.device_ptr && !pair.second.pool_allocated) {
            try {
                sycl::free(pair.second.device_ptr, queue_);
            } catch (...) {
            }
        }
    }
    // Destroy layout pool before SYCL context goes away
    layout_pool_.reset();

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
    for (auto & slot : copy_stage_slots_) {
        if (slot.ptr != nullptr) {
            (void) unified_free_ptr(slot.ptr, slot.device);
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
    // When the model exceeds VRAM, GPU must DMA weights from host memory every token.
    // Pinned host memory allows direct DMA; mmap requires an extra staging copy.
    // Disable aliasing so weights get copied to the pinned pool instead.
    const bool     model_exceeds   = ggml_backend_sycl_model_exceeds_vram(nullptr);
    const bool     can_alias       = host_accessible && layout == GGML_LAYOUT_AOS && src_size == dst_size
                                     && !model_exceeds;
    const bool     prefer_unpinned = host_cache_prefer_unpinned(type);

    std::lock_guard<std::mutex> lock(mutex_);
    const ggml_sycl_cache_id &  key_id_ref = key_id;

    unified_cache_key key{ type, key_id_ref, layer_id, expert_id };
    auto              it = entries_.find(key);
    if (it != entries_.end() && it->second.layout != layout) {
        if (it->second.pinned) {
            GGML_SYCL_DEBUG(
                "[UNIFIED-CACHE] host_cache layout switch blocked (pinned) model=%llu name_hash=0x%llx have=%d "
                "want=%d\n",
                (unsigned long long) key_id_ref.model_id, (unsigned long long) key_id_ref.name_hash,
                (int) it->second.layout, (int) layout);
            return nullptr;
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

                const bool   was_pinned        = it->second.pinned;
                const size_t old_size          = it->second.size;
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
                        new_device_ptr = ggml_sycl_malloc_device(size, queue_, "unified_cache:realloc");
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
                    copy_to_device(new_device_ptr, src_ptr, size);
                }

                // Update entry with new allocation
                it->second.device_ptr    = new_device_ptr;
                it->second.size          = size;
                it->second.content_hash  = new_hash;
                it->second.src_ptr       = src_ptr;
                it->second.host_resident = is_host_resident;
                it->second.location      = new_location;
                if (!is_host_resident) {
                    used_.fetch_add(size - old_size, std::memory_order_relaxed);
                } else if (!it->second.host_resident) {
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
            device_ptr = ggml_sycl_malloc_device(size, queue_, "unified_cache:alloc");
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
        copy_to_device(device_ptr, src_ptr, size);
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
                           (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache id_to_key mismatch");
            }
        }
        if (it->second.layout != layout) {
            if (it->second.pinned) {
                GGML_SYCL_DEBUG(
                    "[UNIFIED-CACHE] layout switch blocked (pinned) model=%llu name_hash=0x%llx have=%d want=%d\n",
                    (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                    (int) it->second.layout, (int) layout);
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
                used_.fetch_add(alloc_size - old_size, std::memory_order_relaxed);
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
                                                        sycl::queue * override_queue) {
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
        std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
        auto                                id_it = id_to_key_.find(request.key);
        if (id_it != id_to_key_.end()) {
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
            }
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
                        try {
                            queue_.wait();
                        } catch (...) {
                        }
                        try {
                            sycl::free(it->second.device_ptr, queue_);
                        } catch (...) {
                        }
                        saturating_sub_used(it->second.size);
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
                if (entry.pinned || entry.state == cache_entry_state::IN_PROGRESS) {
                    GGML_SYCL_DEBUG(
                        "[UNIFIED-CACHE] layout switch blocked (pinned/in-progress) model=%llu name_hash=0x%llx "
                        "have=%d want=%d\n",
                        (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                        (int) entry.layout, (int) request.layout);
                    result.status = cache_layout_status::FAILED;
                    return result;
                }
                void * stale_ptr      = entry.device_ptr;
                size_t stale_size     = entry.size;
                bool   stale_host_res = entry.host_resident;
                bool   stale_pool     = entry.pool_allocated;
                entries_.erase(it);
                it = entries_.end();
                if (!stale_host_res && stale_ptr && stale_size > 0) {
                    if (!stale_pool) {
                        enqueue_deferred_free(stale_ptr, stale_size);
                    }
                    // Pool entries: memory stays in pool, used_ stays at chunk level
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
                        "[UNIFIED-CACHE] layout size mismatch: model=%llu name_hash=0x%llx layout=%d cached=%zu "
                        "req=%zu (pinned)\n",
                        (unsigned long long) request.key.model_id, (unsigned long long) request.key.name_hash,
                        (int) request.layout, entry.size, request.dst_size);
                    result.status = cache_layout_status::FAILED;
                    return result;
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
            // Allocate new entry
            const size_t base_budget    = budget_;
            const size_t allowed_budget = base_budget;  // No overcommit; keep DMA headroom intact.

            // Determine allocation cost: pool may need a new chunk (256 MB) or can sub-allocate (0 cost)
            const bool   pool_can_fit = layout_pool_ && layout_pool_->can_fit(request.dst_size);
            const size_t alloc_cost   = (layout_pool_ && !pool_can_fit) ? layout_pool_->get_default_chunk_size() :
                                                                          (pool_can_fit ? 0 : request.dst_size);

            while (alloc_cost > 0 && used_.load() + alloc_cost > base_budget) {
                if (evict_one(alloc_cost) == 0) {
                    break;
                }
            }

            bool force_host = false;
            if (alloc_cost > 0 && used_.load() + alloc_cost > allowed_budget) {
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cannot evict for layout (used=%.1f MB, need=%.1f MB)\n",
                                used_.load() / (1024.0f * 1024.0f), alloc_cost / (1024.0f * 1024.0f));
                force_host = true;
            }
            host_cache * hcache = get_host_cache(queue_);
            if (request.prefer_host && hcache) {
                force_host = true;
            }
            if (!force_host && hcache) {
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
                        GGML_SYCL_DEBUG(
                            "[UNIFIED-CACHE] live VRAM low (free=%.1f MB, headroom=%.1f MB, need=%.1f MB) - using "
                            "host\n",
                            free_mem / (1024.0f * 1024.0f), headroom / (1024.0f * 1024.0f),
                            request.dst_size / (1024.0f * 1024.0f));
                        force_host = true;
                    }
                }
            }

            void *         new_device_ptr   = nullptr;
            bool           is_host_resident = false;
            bool           is_pool_alloc    = false;
            cache_location host_location    = cache_location::HOST_MMAP;
            if (!force_host && layout_pool_) {
                // Use layout pool for contiguous sub-allocation (reduces TLB misses)
                auto pool_result = layout_pool_->allocate(request.dst_size);
                new_device_ptr   = pool_result.ptr;
                if (new_device_ptr) {
                    is_pool_alloc = true;
                    // Account for any new physical memory consumed by new chunks
                    if (pool_result.new_physical_bytes > 0) {
                        used_.fetch_add(pool_result.new_physical_bytes, std::memory_order_relaxed);
                    }
                }
            }
            if (!force_host && !new_device_ptr) {
                // Pool allocation failed; fall back to individual malloc_device
                try {
                    new_device_ptr = ggml_sycl_malloc_device(request.dst_size, queue_, "unified_cache:layout");
                } catch (const sycl::exception & e) {
                    GGML_SYCL_DEBUG("[UNIFIED-CACHE] layout malloc_device failed: %s, trying host fallback\n",
                                    e.what());
                    new_device_ptr = nullptr;
                }
            }

            bool layout_degraded_to_aos = false;
            if (!new_device_ptr) {
                // Try host_cache fallback when device allocation fails
                if (!hcache) {
                    hcache = get_host_cache(queue_);
                }
                if (hcache) {
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
            if (override_queue) {
                // Use caller's queue for the H2D transfer
                fill_event = override_queue->memcpy(device_ptr, request.src_ptr, request.src_size);
            } else {
                fill_event = copy_to_device_async(device_ptr, request.src_ptr, request.src_size, deps);
            }
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
            // Try to free the device memory directly rather than deferring.
            // The queue may be in a bad state, so wrap in try-catch.
            if (it->second.device_ptr) {
                if (!it->second.pool_allocated) {
                    try {
                        // Try to synchronize before freeing to avoid use-after-free
                        queue_.wait();
                    } catch (...) {
                        // Queue in bad state, ignore
                    }
                    try {
                        sycl::free(it->second.device_ptr, queue_);
                    } catch (...) {
                        // Free failed - memory may leak, but avoid crash
                        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Failed to free device memory during error recovery\n");
                    }
                    saturating_sub_used(it->second.size);
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
                    try {
                        queue_.wait();
                    } catch (...) {
                    }
                    try {
                        sycl::free(it->second.device_ptr, queue_);
                    } catch (...) {
                    }
                    saturating_sub_used(it->second.size);
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
                    try {
                        queue_.wait();
                    } catch (...) {
                    }
                    try {
                        sycl::free(it->second.device_ptr, queue_);
                    } catch (...) {
                    }
                    saturating_sub_used(it->second.size);
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
            it->second.has_ready_event = false;
            it->second.state           = cache_entry_state::READY;
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

bool unified_cache::is_cached(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) const {
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
        return false;
    }
    return true;
}

bool unified_cache::is_cached_any(const ggml_sycl_cache_id & key_id) const {
    if (!key_id.valid) {
        return false;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
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
    return entry.device_ptr;
}

void * unified_cache::get_or_wait(const ggml_sycl_cache_id & key_id, ggml_layout_mode layout) {
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
            if (entry_it->second.state == cache_entry_state::IN_PROGRESS && entry_it->second.has_ready_event &&
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

    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

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
            GGML_SYCL_DEBUG("[UNIFIED-CACHE] get_by_data_ptr: found alias data=%p size=%zu -> device=%p\n", data_ptr,
                            nbytes, entry.device_ptr);
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
    process_deferred_frees();
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
            // Rate-limit: log once, then suppress. Common for MoE models where
            // MMVQ pins with AOS but unified cache stores SOA/COALESCED for experts.
            static std::atomic<int> mismatch_count{ 0 };
            int                     count = mismatch_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (count == 1) {
                GGML_LOG_WARN(
                    "[UNIFIED-CACHE] layout mismatch in pin: have=%d want=%d "
                    "(MoE expert layout mismatch, benign — further occurrences suppressed)\n",
                    (int) entry_it->second.layout, (int) layout);
            }
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return;
        }
        entry_it->second.pinned = true;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] pin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) layout);
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
            GGML_LOG_ERROR("[UNIFIED-CACHE] layout mismatch in unpin model=%llu name_hash=0x%llx have=%d want=%d\n",
                           (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash,
                           (int) entry_it->second.layout, (int) layout);
            if (cache_assert_enabled()) {
                GGML_ABORT("unified_cache layout mismatch");
            }
            return;
        }
        entry_it->second.pinned = false;
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] unpin model=%llu name_hash=0x%llx layout=%d\n",
                        (unsigned long long) key_id.model_id, (unsigned long long) key_id.name_hash, (int) layout);
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
            prefetch_cv_.wait(lock, [this] { return !prefetch_queue_.empty() || prefetch_shutdown_.load(); });

            if (prefetch_shutdown_.load() && prefetch_queue_.empty()) {
                return;
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
        layer_ready_cv_.wait(lock, [this, layer_id] {
            auto it = layer_ready_.find(layer_id);
            return it != layer_ready_.end() && it->second;
        });
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

    // Join the worker thread
    if (prefetch_worker_.joinable()) {
        prefetch_worker_.join();
    }

    prefetch_started_.store(false);

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] prefetch worker stopped\n");
}

size_t unified_cache::evict(size_t bytes_needed) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
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
                        (unsigned long long) evict_key.id.model_id, (unsigned long long) evict_key.id.name_hash,
                        (int) it->second.layout, entry_size, host_resident ? 1 : 0);

        if (!host_resident) {
            // Only free device memory; host-resident entries are managed by host_cache
            enqueue_deferred_free(ptr, entry_size);
            // Signal that device-resident weight pointers may now be stale.
            // Graph replay / persistent TG must check this before using baked pointers.
            has_evictions_.store(true, std::memory_order_release);
        }
        // Note: For host-resident entries, we just remove tracking here.
        // The host_cache still owns the memory and will evict it via its own LRU policy.

        // Remove from lookup
        id_to_key_.erase(evict_key.id);

        // Remove from entries
        entries_.erase(it);

        GGML_SYCL_DEBUG(
            "[UNIFIED-CACHE] Evicted: model=%llu name_hash=0x%llx layout=%d %.2f MB (used=%.1f/%.1f MB) "
            "host_resident=%d\n",
            (unsigned long long) evict_key.id.model_id, (unsigned long long) evict_key.id.name_hash,
            (int) it->second.layout, entry_size / (1024.0f * 1024.0f), used_.load() / (1024.0f * 1024.0f),
            budget_ / (1024.0f * 1024.0f), host_resident ? 1 : 0);
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
    if (copy_to_device_sync_enabled()) {
        for (const auto & dep : deps) {
            const_cast<sycl::event &>(dep).wait();
        }
        copy_to_device(dst, src, size);
        return submit_barrier_all();
    }

    // Stage any non-device source memory through host buffer.
    // This handles:
    // - unknown: mmap'd or non-USM pointers
    // - shared: can fail on Level Zero if allocated on different context
    // - host: generally works, but staging is safer
    // Only device-to-device copies skip staging.
    const bool needs_staging = (src_type != sycl::usm::alloc::device);
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

            size_t slot_idx = std::numeric_limits<size_t>::max();
            while (slot_idx == std::numeric_limits<size_t>::max()) {
                bool        need_new_slot = false;
                sycl::event wait_evt{};
                bool        has_wait_evt = false;

                {
                    std::lock_guard<std::mutex> lock(copy_stage_mutex_);
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
                    if (copy_stage_slots_.size() < copy_to_device_stage_slots()) {
                        need_new_slot = true;
                    } else if (!copy_stage_slots_.empty()) {
                        const size_t idx = copy_stage_next_ % copy_stage_slots_.size();
                        copy_stage_next_ = (idx + 1) % copy_stage_slots_.size();
                        auto & slot      = copy_stage_slots_[idx];
                        if (slot.in_flight) {
                            wait_evt     = slot.done_event;
                            has_wait_evt = true;
                        }
                        slot_idx = idx;
                    } else {
                        need_new_slot = true;
                    }
                }

                if (need_new_slot) {
                    alloc_request req{};
                    req.queue                               = &queue_;
                    req.device                              = get_device_id_from_queue(queue_);
                    req.size                                = chunk;
                    req.intent.role                         = alloc_role::STAGING;
                    req.intent.category                     = runtime_category::STAGING;
                    req.intent.cohort_id                    = "copy_to_device_async";
                    req.intent.constraints.must_host_pinned = true;
                    alloc_handle handle{};
                    if (!unified_alloc(req, &handle) || handle.ptr == nullptr) {
                        throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                              "Cannot copy non-USM pointer to device: staging allocation failed");
                    }
                    std::lock_guard<std::mutex> lock(copy_stage_mutex_);
                    copy_stage_slot             slot{};
                    slot.ptr       = handle.ptr;
                    slot.device    = handle.device;
                    slot.capacity  = handle.size;
                    slot.in_flight = false;
                    copy_stage_slots_.push_back(slot);
                    slot_idx = copy_stage_slots_.size() - 1;
                    continue;
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
                ev = queue_.memcpy(dst_ptr, stage_ptr, chunk);
            } else {
                ev = queue_.submit([&](sycl::handler & cgh) {
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

    // Pool-owned pointers cannot be individually freed; skip the deferred free
    // entirely to avoid unnecessary barrier events and invalid sycl::free() calls.
    if (layout_pool_ && layout_pool_->owns(ptr)) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] skipping deferred free for pool-owned ptr=%p size=%zu\n", ptr, size);
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
            const bool is_pool = layout_pool_ && layout_pool_->owns(it->ptr);
            if (!is_pool) {
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
    // Recalculate model placement decision based on new effective budget.
    // This ensures g_model_exceeds_vram reflects actual available VRAM after KV cache allocation.
    ggml_sycl_recalc_model_exceeds_vram(effective_budget);
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
// caller must apply via update_reserved_bytes() AFTER releasing g_cache_mutex.
// This prevents a deadlock: update_reserved_bytes() → recalc → layer streaming
// → unified_cache_add_runtime_bytes() → tries to re-lock g_cache_mutex.
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

        const size_t min_headroom = 256ull * 1024ull * 1024ull;
        const size_t headroom     = std::max(min_headroom, base_mem / 10);
        if (base_mem > headroom && budget > base_mem - headroom) {
            budget = base_mem - headroom;
        }

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
            "[UNIFIED-CACHE] Device %d (%s): total=%.1f MB free=%.1f MB budget=%.1f MB (%d%%, headroom=%.1f MB)\n",
            device_id, desc, base_mem / (1024.0f * 1024.0f), free_mem / (1024.0f * 1024.0f),
            budget / (1024.0f * 1024.0f), pct, headroom / (1024.0f * 1024.0f));
    }

    const size_t staging_bytes = resolve_host_staging_bytes();
    try {
        g_device_caches[device_id]  = std::make_unique<unified_cache>(queue, budget, staging_bytes, dma_reserve_bytes);
        const size_t reserved_total = runtime_reserved_bytes_nolock(device_id);
        const size_t baseline       = budget_capped_to_free ? reserved_total : 0;
        g_runtime_reserved_baseline[device_id].store(baseline, std::memory_order_relaxed);
        const size_t reserved_adjusted = runtime_reserved_adjusted_nolock(device_id);
        // Defer update_reserved_bytes to caller (after releasing g_cache_mutex)
        // to avoid deadlock: update_reserved_bytes → recalc → layer streaming
        // → unified_cache_add_runtime_bytes → re-lock g_cache_mutex
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
    unified_cache * result           = nullptr;
    size_t          deferred_reserve = 0;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache_mode_locked = true;

        unified_cache_mode mode      = get_effective_mode();
        int                device_id = (mode == unified_cache_mode::GLOBAL) ? 0 : get_device_id_from_queue(queue);

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
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache_mode_locked = true;

    int device_id = get_device_id_from_queue(queue);
    return create_host_cache_for_device(device_id);
}

unified_cache * get_unified_cache_for_device(int device_id) {
    unified_cache * result           = nullptr;
    size_t          deferred_reserve = 0;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache_mode_locked = true;

        unified_cache_mode mode             = get_effective_mode();
        int                effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device_id;

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
        unified_cache_sub_runtime_bytes(rec.handle.device, rec.handle.size, rec.handle.category);
        unified_managed_sub_device_bytes(rec.handle.device, rec.handle.size);
    } else if (rec.handle.tier == alloc_tier::HOST_PINNED || rec.handle.tier == alloc_tier::MMAP_TRACKED) {
        unified_cache_sub_runtime_host_bytes(rec.handle.size);
        unified_managed_sub_host_bytes(rec.handle.size);
    }

    try {
        if (rec.handle.tier == alloc_tier::MMAP_TRACKED) {
            return true;
        }
        if (rec.uses_pinned_pool) {
            if (auto * hcache = get_host_cache_for_device(rec.handle.device)) {
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
            unified_cache_add_runtime_bytes(rec.handle.device, rec.handle.size, rec.handle.category);
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
    if (reserve_device) {
        unified_cache_add_runtime_bytes(req.device, alloc_size, cat);
        unified_managed_add_device_bytes(req.device, alloc_size);
    } else if (reserve_host) {
        unified_cache_add_runtime_host_bytes(alloc_size);
        unified_managed_add_host_bytes(alloc_size);
    }

    bool   uses_pinned_pool = false;
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
            if (auto * cache = get_unified_cache_for_device(req.device)) {
                cache_vram = cache->used();
            }
            const size_t used_vram = runtime_vram + cache_vram;
            if (total_vram > 0 && used_vram + alloc_size > total_vram) {
                GGML_SYCL_DEBUG("[UNIFIED-ALLOC] Device %d VRAM overcommit guard: "
                                "used=%.1f MB + alloc=%.1f MB > total=%.1f MB, failing\n",
                                req.device, used_vram / (1024.0f * 1024.0f),
                                alloc_size / (1024.0f * 1024.0f),
                                total_vram / (1024.0f * 1024.0f));
                // Roll back the reserve we just added above
                if (reserve_device) {
                    unified_cache_sub_runtime_bytes(req.device, alloc_size, cat);
                    unified_managed_sub_device_bytes(req.device, alloc_size);
                }
                return false;
            }
        }
        ptr = ggml_sycl_malloc_device(alloc_size, *req.queue, "unified_alloc:device");
    } else {
        if (req.intent.constraints.use_pinned_pool) {
            if (auto * hcache = get_host_cache_for_device(req.device)) {
                ptr              = hcache->allocate_pinned_runtime(alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
                uses_pinned_pool = (ptr != nullptr);
            }
        }
        if (!ptr) {
            ptr = ggml_sycl_malloc_host(alloc_size, *req.queue, "unified_alloc:host");
        }
    }

    if (!ptr) {
        if (reserve_device) {
            unified_cache_sub_runtime_bytes(req.device, alloc_size, cat);
            unified_managed_sub_device_bytes(req.device, alloc_size);
        } else if (reserve_host) {
            unified_cache_sub_runtime_host_bytes(alloc_size);
            unified_managed_sub_host_bytes(alloc_size);
        }
        return false;
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
    if (req.intent.cohort_id && req.intent.cohort_id[0] != '\0') {
        rec.cohort_id = req.intent.cohort_id;
    }

    {
        std::lock_guard<std::mutex> lock(g_runtime_alloc_mutex);
        if (g_runtime_alloc_registry.find(ptr) != g_runtime_alloc_registry.end()) {
            GGML_LOG_ERROR("[UNIFIED-ALLOC] duplicate pointer registration ptr=%p size=%zu tier=%s\n", ptr, alloc_size,
                           alloc_tier_name(tier));
            if (reserve_device) {
                unified_cache_sub_runtime_bytes(req.device, alloc_size, cat);
                unified_managed_sub_device_bytes(req.device, alloc_size);
            } else if (reserve_host) {
                unified_cache_sub_runtime_host_bytes(alloc_size);
                unified_managed_sub_host_bytes(alloc_size);
            }
            if (uses_pinned_pool) {
                if (auto * hcache = get_host_cache_for_device(req.device)) {
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

void unified_cache_add_runtime_bytes(int device, size_t bytes, runtime_category cat) {
    if (bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    const size_t new_total =
        g_runtime_reserved_bytes[effective_device].fetch_add(bytes, std::memory_order_relaxed) + bytes;
    g_runtime_cat_bytes[effective_device][static_cast<int>(cat)].fetch_add(bytes, std::memory_order_relaxed);
    auto it = g_device_caches.find(effective_device);
    if (it != g_device_caches.end()) {
        const size_t baseline = g_runtime_reserved_baseline[effective_device].load(std::memory_order_relaxed);
        const size_t adjusted = new_total > baseline ? new_total - baseline : 0;
        it->second->update_reserved_bytes(adjusted);
    }
}

void unified_cache_sub_runtime_bytes(int device, size_t bytes, runtime_category cat) {
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
    size_t cat_cur  = g_runtime_cat_bytes[effective_device][static_cast<int>(cat)].load(std::memory_order_relaxed);
    size_t cat_next = cat_cur > bytes ? cat_cur - bytes : 0;
    g_runtime_cat_bytes[effective_device][static_cast<int>(cat)].store(cat_next, std::memory_order_relaxed);
    auto it = g_device_caches.find(effective_device);
    if (it != g_device_caches.end()) {
        const size_t baseline = g_runtime_reserved_baseline[effective_device].load(std::memory_order_relaxed);
        const size_t adjusted = next > baseline ? next - baseline : 0;
        it->second->update_reserved_bytes(adjusted);
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

size_t unified_cache_get_runtime_bytes_by_category(int device, runtime_category cat) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
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
    size_t                      cur  = g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
    size_t                      next = cur > bytes ? cur - bytes : 0;
    g_runtime_reserved_host_bytes.store(next, std::memory_order_relaxed);
    if (g_host_cache_shared) {
        g_host_cache_shared->update_reserved_bytes(next);
    }
}

size_t unified_cache_get_runtime_host_bytes() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    return g_runtime_reserved_host_bytes.load(std::memory_order_relaxed);
}

// === Budget Query API ===

size_t unified_cache_available_for_compute(int device) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    auto it = g_device_caches.find(effective_device);
    if (it == g_device_caches.end() || !it->second) {
        return 0;
    }
    return it->second->available_for_compute();
}

size_t unified_cache_total_managed(int device) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    auto it = g_device_caches.find(effective_device);
    if (it == g_device_caches.end() || !it->second) {
        return 0;
    }
    return it->second->base_budget();
}

size_t unified_cache_weight_bytes(int device) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return 0;
    }
    auto it = g_device_caches.find(effective_device);
    if (it == g_device_caches.end() || !it->second) {
        return 0;
    }
    return it->second->weight_bytes();
}

// === Budget Summary Diagnostic ===

void unified_cache_log_budget_summary(int device) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    auto it = g_device_caches.find(effective_device);
    if (it == g_device_caches.end() || !it->second) {
        return;
    }
    auto &       cache = *it->second;
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
    const bool exceeds = ggml_backend_sycl_model_exceeds_vram(nullptr);

    GGML_LOG_INFO(
        "[UNIFIED-CACHE] Budget summary for device %d:\n"
        "  Total VRAM budget:    %8.1f MB\n"
        "  Weight bytes (used_): %8.1f MB\n"
        "  Runtime reserved:     %8.1f MB\n"
        "  Effective budget:     %8.1f MB\n"
        "  Available for alloc:  %8.1f MB\n"
        "  Avail for weights:    %8.1f MB\n"
        "  Budget pct:           %8d %%\n"
        "  Model exceeds VRAM:   %8s\n",
        device, base / (1024.0f * 1024.0f), wt / (1024.0f * 1024.0f), rt / (1024.0f * 1024.0f),
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

bool unified_cache_is_budget_exceeded(int device) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    unified_cache_mode          mode             = get_effective_mode();
    int                         effective_device = (mode == unified_cache_mode::GLOBAL) ? 0 : device;
    if (effective_device < 0 || effective_device >= GGML_SYCL_MAX_DEVICES) {
        return false;
    }
    auto it = g_device_caches.find(effective_device);
    if (it == g_device_caches.end() || !it->second) {
        return false;
    }
    return it->second->is_budget_exceeded();
}

bool unified_cache_has_evictions() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
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
        info.available_for_weights =
            info.budget_bytes > info.runtime_bytes ? info.budget_bytes - info.runtime_bytes : 0;
    } else {
        // Cache not yet initialized — use raw calculation
        info.budget_bytes     = static_cast<size_t>(info.total_vram * (static_cast<double>(pct) / 100.0));
        const size_t headroom = std::max(size_t(256) << 20, info.total_vram / 10);
        if (info.total_vram > headroom && info.budget_bytes > info.total_vram - headroom) {
            info.budget_bytes = info.total_vram - headroom;
        }
        info.available_for_weights = info.budget_bytes;
    }

    info.model_exceeds_vram = ggml_backend_sycl_model_exceeds_vram(nullptr);

    // Populate MoE fields from tensor inventory
    size_t moe_total = 0;
    int    n_exp = 0, n_exp_used = 0;
    ggml_sycl_get_moe_info(&moe_total, &n_exp, &n_exp_used);
    info.expert_weight_bytes = moe_total;
    info.n_expert_total      = n_exp;
    info.n_expert_used       = n_exp_used;
    info.active_expert_bytes = compute_moe_effective_weight_bytes(moe_total, moe_total, n_exp, n_exp_used);

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

    // If model already exceeds VRAM, definitely offload KV
    if (info.model_exceeds_vram) {
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
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    // Unpin in all caches
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
                                               int          expert_id) {
    ggml_sycl_cache_id id{};

    uint64_t name_hash = static_cast<uint64_t>(std::hash<std::string>()(tensor_name ? tensor_name : "unknown"));

    id.valid         = true;
    id.model_id      = model_id;
    id.has_gguf      = false;
    id.file_idx      = 0;
    id.file_offs     = 0;
    id.nbytes        = 0;
    id.name_hash     = name_hash;
    id.type          = GGML_TYPE_COUNT;
    id.tp_sharded    = false;
    id.tp_rank       = 0;
    id.tp_world_size = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = 0;
        id.tp_local_ne[i]  = 0;
        id.tp_offset_ne[i] = 0;
    }

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
                                        uint32_t        model_id) {
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
        ggml_sycl_cache_id key        = make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id);

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

    // Step 3: Stage missing experts via ensure_cached_layout to get fill events
    for (int32_t expert_id : experts_to_stage) {
        const void *       expert_ptr = static_cast<const char *>(weight_base_ptr) + expert_id * expert_stride;
        ggml_sycl_cache_id key        = make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id);

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
    }

    // Step 4: Pin all unique experts (including those already cached)
    for (int32_t expert_id : unique_experts) {
        ggml_sycl_cache_id key = make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id);

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
                          uint32_t        model_id) {
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
        ggml_sycl_cache_id key = make_expert_cache_id(tensor_name, cache_uuid, model_id, expert_id);

        cache->unpin(key, GGML_LAYOUT_AOS);
    }

    GGML_SYCL_DEBUG("[UNPIN] Layer %d: Unpinned %zu experts\n", layer_id, unique_experts.size());
}

// --- ExpertPlacementTable implementation ---

void ExpertPlacementTable::init(int n_layers, int n_experts_per_layer) {
    std::unique_lock lock(mutex_);
    n_layers_  = n_layers;
    n_experts_ = n_experts_per_layer;
    table_.reserve(static_cast<size_t>(n_layers) * n_experts_per_layer);
}

void ExpertPlacementTable::set(int layer_id, int expert_id,
                                const ExpertPlacement & placement) {
    std::unique_lock lock(mutex_);
    table_[make_key(layer_id, expert_id)] = placement;
}

ExpertPlacement ExpertPlacementTable::get(int layer_id, int expert_id) const {
    std::shared_lock lock(mutex_);
    auto it = table_.find(make_key(layer_id, expert_id));
    if (it != table_.end()) {
        return it->second;
    }
    return {};  // Invalid placement (device_id = -1, ptrs = nullptr)
}

void ExpertPlacementTable::set_device_ptr(int layer_id, int expert_id,
                                           int device_id, void * ptr) {
    std::unique_lock lock(mutex_);
    auto it = table_.find(make_key(layer_id, expert_id));
    if (it != table_.end()) {
        it->second.device_id  = device_id;
        it->second.device_ptr = ptr;
    }
}

void ExpertPlacementTable::set_popularity(int layer_id, int expert_id, int rank) {
    std::unique_lock lock(mutex_);
    auto it = table_.find(make_key(layer_id, expert_id));
    if (it != table_.end()) {
        it->second.popularity_rank = rank;
    }
}

std::vector<std::pair<int, ExpertPlacement>>
ExpertPlacementTable::get_layer_experts(int layer_id) const {
    std::shared_lock lock(mutex_);
    std::vector<std::pair<int, ExpertPlacement>> result;
    for (int e = 0; e < n_experts_; e++) {
        auto it = table_.find(make_key(layer_id, e));
        if (it != table_.end()) {
            result.push_back({e, it->second});
        }
    }
    std::sort(result.begin(), result.end(),
              [](const auto & a, const auto & b) {
                  return a.second.popularity_rank < b.second.popularity_rank;
              });
    return result;
}

ExpertPlacementTable & get_expert_placement_table() {
    static ExpertPlacementTable table;
    return table;
}

// === OneDNN FP16 Scratch Buffer Implementation ===

bool unified_cache::reserve_onednn_scratch(size_t weights_size, size_t activations_size) {
    std::lock_guard<std::mutex> lock(onednn_scratch_mutex_);

    // Already reserved with sufficient size?
    if (onednn_weights_scratch_ && onednn_activations_scratch_ && onednn_weights_scratch_size_ >= weights_size &&
        onednn_activations_scratch_size_ >= activations_size) {
        return true;
    }

    // Free existing if resizing — subtract old sizes from budget first
    const size_t old_total = onednn_weights_scratch_size_ + onednn_activations_scratch_size_;
    if (onednn_weights_scratch_) {
        try {
            sycl::free(onednn_weights_scratch_, queue_);
        } catch (...) {
        }
        onednn_weights_scratch_      = nullptr;
        onednn_weights_scratch_size_ = 0;
    }
    if (onednn_activations_scratch_) {
        try {
            sycl::free(onednn_activations_scratch_, queue_);
        } catch (...) {
        }
        onednn_activations_scratch_      = nullptr;
        onednn_activations_scratch_size_ = 0;
    }
    if (old_total > 0) {
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
        dev_ptr = ggml_sycl_malloc_device(partial_bytes, queue_, "unified_cache:partial_rows");
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

    // Copy AOS data from host to device
    queue_.memcpy(dev_ptr, src_host, partial_bytes).wait();

    // Apply in-place SOA reorder on device
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
    std::lock_guard<std::mutex> lock(g_cache_mutex);

    // Return existing cache if already registered
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
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_device_caches.clear();
    g_host_cache_shared.reset();

    GGML_SYCL_DEBUG("[UNIFIED-CACHE] Shutdown complete\n");
}

}  // namespace ggml_sycl
