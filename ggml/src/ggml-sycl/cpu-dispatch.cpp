//
// cpu-dispatch.cpp — CPU compute path for data-local inference
//
// Executes operations on a CPU SYCL queue when weight data resides in host
// pinned memory (unified cache PINNED_HOST or MMAP tier).  Avoids unnecessary
// host-to-device transfers for layers evicted from VRAM.
//
// MUL_MAT uses quantized vec_dot (e.g., Q4_0×Q8_0) for small M (TG), or
// dnnl_sgemm for larger M (PP) after dequantizing to F32.  Element-wise ops
// (RMS_NORM, ADD, MUL) use portable SYCL parallel_for on the CPU queue.
//
// Supports F32, F16, and quantized types (via dequantize-to-F32 + sgemm).
//
// Activation staging: When compute buffers are device-resident (no HOST_COMPUTE),
// CPU kernels can't access device memory directly.  Double-buffered staging
// copies activations host↔device with SYCL event-based overlap.  Weights are
// already host-pinned so they need no staging.  With GGML_SYCL_HOST_COMPUTE=1,
// compute buffers are host-pinned and staging is bypassed entirely.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "cpu-dispatch.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tensor-types.hpp"
#include "unified-cache.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if __has_include(<oneapi/tbb/blocked_range.h>) && __has_include(<oneapi/tbb/parallel_for.h>)
#    include <oneapi/tbb/blocked_range.h>
#    include <oneapi/tbb/parallel_for.h>
#    include <oneapi/tbb/task_arena.h>
#    define GGML_SYCL_HAS_TBB 1
namespace ggml_sycl_tbb = oneapi::tbb;
#elif __has_include(<tbb/blocked_range.h>) && __has_include(<tbb/parallel_for.h>)
#    include <tbb/blocked_range.h>
#    include <tbb/parallel_for.h>
#    include <tbb/task_arena.h>
#    define GGML_SYCL_HAS_TBB 1
namespace ggml_sycl_tbb = tbb;
#else
#    define GGML_SYCL_HAS_TBB 0
#endif

#if GGML_SYCL_DNNL
#    include "gemm.hpp"  // Provides dnnl.hpp → dnnl_sgemm()
#endif

// ---------------------------------------------------------------------------
// Host pointer registry: stores original mmap pointers for weight tensors.
// Populated during set_tensor (when the host data from the GGUF mmap is still
// available) and read during CPU dispatch to access quantized weight data
// directly without dequantization.
// ---------------------------------------------------------------------------

static std::mutex                                    g_host_ptr_mutex;
static std::unordered_map<std::string, const void *> g_host_ptr_map;
static bool                                          g_host_ptr_owns_memory = false;

// Generation counter for activation quantization cache invalidation.
// Bumped at the start of each graph compute to prevent stale cache hits
// across tokens (same tensor pointer, different data due to buffer reuse).
static std::atomic<uint64_t> g_quant_cache_generation{0};

void ggml_sycl_cpu_quant_cache_new_graph() {
    g_quant_cache_generation.fetch_add(1, std::memory_order_relaxed);
}

enum class offload_wait_reason : uint8_t {
    FORCED   = 0,
    FALLBACK = 1,
};

static inline bool offload_event_waitable(const sycl::event & evt) {
    try {
        return ggml_sycl_should_add_dependency(evt);
    } catch (...) {
        return false;
    }
}

static inline void offload_wait_event(sycl::event & evt, offload_wait_reason reason = offload_wait_reason::FORCED) {
    if (!offload_event_waitable(evt)) {
        return;
    }
    evt.wait();
    ggml_sycl::offload_stats_note_wait(reason == offload_wait_reason::FALLBACK);
}

static inline void offload_wait_queue(sycl::queue * q, offload_wait_reason reason = offload_wait_reason::FORCED) {
    q->wait();
    ggml_sycl::offload_stats_note_wait(reason == offload_wait_reason::FALLBACK);
}

static void staging_track_cpu_event(const sycl::event & evt);

static sycl::event g_cpu_chain_event{};
static bool        g_cpu_chain_event_valid  = false;
static bool        g_cpu_chain_on_cpu_queue = false;

static inline void wait_dependency_if_needed(const sycl::event & evt) {
    if (!offload_event_waitable(evt)) {
        return;
    }
    sycl::event evt_copy = evt;
    offload_wait_event(evt_copy, offload_wait_reason::FORCED);
}

static inline void append_dependency(std::vector<sycl::event> & deps, const sycl::event & evt) {
    try {
        if (ggml_sycl_should_add_dependency(evt)) {
            deps.push_back(evt);
        }
    } catch (...) {
    }
}

static inline std::vector<sycl::event> cpu_collect_deps(const sycl::event * e0       = nullptr,
                                                        const sycl::event * e1       = nullptr,
                                                        sycl::queue *       target_q = nullptr) {
    std::vector<sycl::event> deps;
    deps.reserve(3);
    if (e0) {
        append_dependency(deps, *e0);
    }
    if (e1) {
        append_dependency(deps, *e1);
    }
    if (g_cpu_chain_event_valid) {
        const bool target_is_cpu = (target_q != nullptr && target_q == ggml_sycl_get_cpu_queue());
        if (target_is_cpu == g_cpu_chain_on_cpu_queue) {
            append_dependency(deps, g_cpu_chain_event);
        } else {
            wait_dependency_if_needed(g_cpu_chain_event);
            g_cpu_chain_event_valid = false;
        }
    }
    return deps;
}

template <typename SubmitFn>
static sycl::event cpu_submit_async(sycl::queue * cpu_q, const std::vector<sycl::event> & deps, SubmitFn && fn) {
    const bool               submit_on_cpu_queue = (cpu_q != nullptr && cpu_q == ggml_sycl_get_cpu_queue());
    std::vector<sycl::event> submit_deps;
    submit_deps.reserve(deps.size());
    if (submit_on_cpu_queue) {
        for (const auto & dep : deps) {
            wait_dependency_if_needed(dep);
        }
    } else {
        for (const auto & dep : deps) {
            append_dependency(submit_deps, dep);
        }
    }

    sycl::event evt = cpu_q->submit([&](sycl::handler & cgh) {
        if (!submit_deps.empty()) {
            cgh.depends_on(submit_deps);
        }
        fn(cgh);
    });
    if (ggml_sycl_cpu_offload_async_enabled()) {
        g_cpu_chain_event        = evt;
        g_cpu_chain_event_valid  = true;
        g_cpu_chain_on_cpu_queue = submit_on_cpu_queue;
        staging_track_cpu_event(evt);
    } else {
        offload_wait_event(evt, offload_wait_reason::FALLBACK);
        g_cpu_chain_event_valid = false;
    }
    return evt;
}

static inline void cpu_wait_chain_event() {
    if (!g_cpu_chain_event_valid) {
        return;
    }
    offload_wait_event(g_cpu_chain_event);
    g_cpu_chain_event_valid = false;
}

void ggml_sycl_cpu_dispatch_register_host_ptr(const char * name, const void * host_ptr, size_t size) {
    if (!name || !host_ptr || size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_host_ptr_mutex);

    if (ggml_sycl_cpu_offload_enabled()) {
        // CPU offload mode: copy weight data to persistent host memory.
        // The original mmap pointer may be released by the model loader after
        // set_tensor completes, so we need our own copy for inference.
        // aligned_alloc(64) ensures AVX-512 alignment for vec_dot.
        size_t aligned_size = (size + 63) & ~size_t(63);
        void * copy         = aligned_alloc(64, aligned_size);
        if (copy) {
            memcpy(copy, host_ptr, size);
            // Free any previous copy for this tensor
            if (g_host_ptr_owns_memory) {
                auto it = g_host_ptr_map.find(name);
                if (it != g_host_ptr_map.end()) {
                    free(const_cast<void *>(it->second));
                }
            }
            g_host_ptr_map[name]   = copy;
            g_host_ptr_owns_memory = true;
        }
    } else {
        g_host_ptr_map[name] = host_ptr;
    }
}

static const void * cpu_dispatch_lookup_host_ptr(const char * name) {
    if (!name) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_host_ptr_mutex);
    auto                        it = g_host_ptr_map.find(name);
    return (it != g_host_ptr_map.end()) ? it->second : nullptr;
}

// ---------------------------------------------------------------------------
// Retained activation state: eliminates per-op staging overhead
// ---------------------------------------------------------------------------
//
// When active, CPU op outputs stay in host scratch memory instead of being
// flushed to device. The next CPU op can read them directly without D2H copy.
// Activated at GPU→CPU transitions, flushed at CPU→GPU transitions.

static void *                          g_retained_scratch     = nullptr;
static size_t                          g_retained_scratch_cap = 0;
static size_t                          g_retained_scratch_off = 0;  // bump allocator offset
static int                             g_retained_device      = -1;
static ggml_sycl::offload_buffer_lease g_retained_scratch_lease{};

struct retained_entry {
    void * host_ptr;  // pointer into g_retained_scratch
    size_t size;      // byte size of retained data
};

static std::unordered_map<const ggml_tensor *, retained_entry> g_retained_map;
static bool                                                    g_retained_active = false;
static sycl::queue *                                           g_retained_gpu_q  = nullptr;
static sycl::event                                             g_retained_flush_evt{};
static bool                                                    g_retained_flush_pending = false;

static inline void retained_wait_flush_event(offload_wait_reason reason = offload_wait_reason::FORCED) {
    if (!g_retained_flush_pending) {
        return;
    }
    offload_wait_event(g_retained_flush_evt, reason);
    g_retained_flush_pending = false;
}

// Allocate from scratch buffer (64-byte aligned for AVX-512)
static void * scratch_alloc(size_t size) {
    size_t aligned_off = (g_retained_scratch_off + 63) & ~size_t(63);
    if (aligned_off + size > g_retained_scratch_cap) {
        return nullptr;  // scratch full, fall back to staging
    }
    void * ptr             = static_cast<char *>(g_retained_scratch) + aligned_off;
    g_retained_scratch_off = aligned_off + size;
    return ptr;
}

static void scratch_reset() {
    g_retained_scratch_off = 0;
}

void ggml_sycl_cpu_retained_init(int device, sycl::queue * gpu_q) {
    retained_wait_flush_event();
    cpu_wait_chain_event();
    GGML_ASSERT(!g_retained_scratch || gpu_q == g_retained_gpu_q);
    if (!g_retained_scratch) {
        constexpr size_t                  DEFAULT_SCRATCH_SIZE = 32 * 1024 * 1024;  // 32MB
        ggml_sycl::offload_buffer_request req{};
        req.queue                                         = gpu_q;
        req.device                                        = device;
        req.size                                          = DEFAULT_SCRATCH_SIZE;
        req.alignment                                     = 64;
        req.role                                          = ggml_sycl::offload_buffer_role::RETAINED_SCRATCH;
        req.intent.role                                   = ggml_sycl::alloc_role::COMPUTE;
        req.intent.category                               = ggml_sycl::runtime_category::HOST_COMPUTE;
        req.intent.cohort_id                              = "cpu_offload";
        req.intent.constraints.must_host_pinned           = true;
        req.intent.constraints.prefer_same_tier_as_cohort = true;
        if (ggml_sycl::acquire_offload_buffer(req, &g_retained_scratch_lease)) {
            g_retained_scratch     = g_retained_scratch_lease.handle.ptr;
            g_retained_scratch_cap = DEFAULT_SCRATCH_SIZE;
        }
    }
    g_retained_scratch_off = 0;
    g_retained_map.clear();
    g_retained_active = true;
    g_retained_gpu_q  = gpu_q;
    g_retained_device = device;
}

void ggml_sycl_cpu_retained_cleanup() {
    retained_wait_flush_event();
    cpu_wait_chain_event();
    if (g_retained_scratch && g_retained_gpu_q) {
        (void) ggml_sycl::release_offload_buffer(g_retained_scratch_lease);
        g_retained_scratch       = nullptr;
        g_retained_scratch_cap   = 0;
        g_retained_scratch_lease = {};
    }
    g_retained_map.clear();
    scratch_reset();
    g_retained_active = false;
    g_retained_gpu_q  = nullptr;
    g_retained_device = -1;
}

bool ggml_sycl_cpu_retained_active() {
    return g_retained_active && g_retained_scratch;
}

void * ggml_sycl_cpu_retained_alloc_output(const ggml_tensor * dst) {
    if (!g_retained_active || !g_retained_scratch) {
        return nullptr;
    }
    size_t nbytes = ggml_nbytes(dst);
    void * ptr    = scratch_alloc(nbytes);
    if (ptr) {
        g_retained_map[dst] = { ptr, nbytes };
    }
    return ptr;  // nullptr if scratch full → staging fallback
}

void ggml_sycl_cpu_retained_flush_all(int device, sycl::queue * gpu_q) {
    if (g_retained_map.empty()) {
        return;
    }
    retained_wait_flush_event();
    cpu_wait_chain_event();

    std::vector<sycl::event> events;
    events.reserve(g_retained_map.size());

    for (auto & [tensor, entry] : g_retained_map) {
        if (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer)) {
            continue;
        }
        void * device_ptr = ggml_sycl_get_data_ptr(tensor, device);
        if (!device_ptr) {
            continue;
        }
        ggml_sycl::offload_stats_note_transfer(true, entry.size);
        events.push_back(gpu_q->memcpy(device_ptr, entry.host_ptr, entry.size));
    }

    if (!events.empty()) {
        offload_wait_queue(gpu_q);
    }
    g_retained_flush_pending = false;

    g_retained_map.clear();
    scratch_reset();
}

void ggml_sycl_cpu_retained_flush_selective(int                         device,
                                            sycl::queue *               gpu_q,
                                            const ggml_tensor * const * gpu_nodes,
                                            int                         n_gpu_nodes) {
    if (g_retained_map.empty() || !gpu_nodes || n_gpu_nodes <= 0) {
        g_retained_map.clear();
        scratch_reset();
        return;
    }

    // Collect retained tensors needed by ANY upcoming GPU node.  GPU node inputs
    // may be views (RESHAPE/VIEW/PERMUTE) of retained tensors — follow view_src
    // chain to find the underlying retained entry and flush to the VIEW's device
    // address (which is a valid subregion of the original allocation).
    //
    // Use (retained_key, view_tensor) pairs to copy from the retained host buffer
    // to the view tensor's device address (accounting for view_offs).
    struct flush_entry {
        const ggml_tensor * retained_key;  // key in g_retained_map
        const ggml_tensor * view_tensor;   // actual tensor the GPU node reads
    };

    std::vector<flush_entry>                to_flush;
    std::unordered_set<const ggml_tensor *> seen;  // avoid duplicate flushes

    for (int n = 0; n < n_gpu_nodes; n++) {
        const ggml_tensor * gnode = gpu_nodes[n];
        if (!gnode) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = gnode->src[s];
            if (!src) {
                break;
            }
            // Follow view_src chain to find retained entry
            const ggml_tensor * lookup = src;
            while (lookup) {
                if (g_retained_map.count(lookup)) {
                    if (seen.insert(src).second) {
                        to_flush.push_back({ lookup, src });
                    }
                    break;
                }
                lookup = lookup->view_src;
            }
        }
    }

    if (to_flush.empty()) {
        g_retained_map.clear();
        scratch_reset();
        return;
    }

    retained_wait_flush_event();
    cpu_wait_chain_event();

    struct copy_req {
        const ggml_tensor * retained_key = nullptr;
        char *              dst          = nullptr;
        char *              src          = nullptr;
        size_t              size         = 0;
        size_t              src_off      = 0;
    };

    std::vector<copy_req> copies;
    copies.reserve(to_flush.size());

    // Flush all collected tensors — their device addresses are guaranteed valid
    // (live DAG dependencies of upcoming GPU nodes can't be recycled).
    for (auto & [retained_key, view_tensor] : to_flush) {
        auto it = g_retained_map.find(retained_key);
        if (it == g_retained_map.end()) {
            continue;
        }
        if (view_tensor->buffer && !ggml_backend_buffer_is_host(view_tensor->buffer)) {
            void * device_ptr = ggml_sycl_get_data_ptr(view_tensor, device);
            if (device_ptr) {
                // Compute offset within the retained buffer for views
                char * host_base = static_cast<char *>(it->second.host_ptr);
                size_t off    = (view_tensor == retained_key) ? 0 : (view_tensor->view_offs - retained_key->view_offs);
                size_t nbytes = ggml_nbytes(view_tensor);
                if (nbytes > 0) {
                    copies.push_back({ retained_key, static_cast<char *>(device_ptr), host_base + off, nbytes, off });
                }
            }
        }
    }

    if (!copies.empty()) {
        std::sort(copies.begin(), copies.end(), [](const copy_req & a, const copy_req & b) {
            if (a.retained_key != b.retained_key) {
                return reinterpret_cast<uintptr_t>(a.retained_key) < reinterpret_cast<uintptr_t>(b.retained_key);
            }
            if (a.src_off != b.src_off) {
                return a.src_off < b.src_off;
            }
            return reinterpret_cast<uintptr_t>(a.dst) < reinterpret_cast<uintptr_t>(b.dst);
        });

        std::vector<copy_req> merged;
        merged.reserve(copies.size());
        for (const copy_req & req : copies) {
            if (merged.empty()) {
                merged.push_back(req);
                continue;
            }
            copy_req & last         = merged.back();
            const bool adjacent_src = last.src + last.size == req.src;
            const bool adjacent_dst = last.dst + last.size == req.dst;
            if (last.retained_key == req.retained_key && adjacent_src && adjacent_dst) {
                last.size += req.size;
            } else {
                merged.push_back(req);
            }
        }

        sycl::event last_evt{};
        bool        has_copy = false;
        for (const copy_req & req : merged) {
            ggml_sycl::offload_stats_note_transfer(true, req.size);
            last_evt = gpu_q->memcpy(req.dst, req.src, req.size);
            has_copy = true;
        }

        if (has_copy) {
            if (ggml_sycl_cpu_offload_async_enabled()) {
                g_retained_flush_evt     = last_evt;
                g_retained_flush_pending = true;
            } else {
                offload_wait_queue(gpu_q, offload_wait_reason::FALLBACK);
                g_retained_flush_pending = false;
            }
        }
    }

    // Discard ALL retained data. Only the tensors above were flushed.
    // Other tensors' device addresses may have been recycled — must NOT write.
    g_retained_map.clear();
    scratch_reset();
}

void ggml_sycl_cpu_retained_deactivate() {
    g_retained_map.clear();
    scratch_reset();
    g_retained_active = false;
}

// ---------------------------------------------------------------------------
// Activation staging: double-buffered host-pinned buffers for device↔host transfer
// ---------------------------------------------------------------------------
//
// GPU compute buffers use sycl::malloc_device() which CPU can't access.
// Weight tensors are already host-pinned (that's why we're here).
// Activation/output tensors need staging: copy device→host before CPU compute,
// then host→device after CPU writes output.
//
// Double-buffered staging with SYCL events (adapted from layer-streaming.hpp):
//   Two banks of 3 slots each (slot 0=src0, 1=src1, 2=dst).
//   Ops alternate between banks.  The previous op's flush can overlap with
//   the current op's stage-in because they use different buffers.
//   Events replace synchronous .wait() — we wait only when data is needed.

static constexpr int STAGING_SLOTS_PER_BANK = 3;
static constexpr int STAGING_BANKS          = 2;

static struct {
    void *                          ptr = nullptr;
    size_t                          cap = 0;
    ggml_sycl::offload_buffer_lease lease{};
} g_cpu_staging[STAGING_BANKS][STAGING_SLOTS_PER_BANK];

// Persistent staging cache for leaf tensors (RoPE freqs, masks, constants).
// Key: tensor data pointer (stable across tokens for leaf tensors).
// Value: host-accessible pointer (either direct host ptr or cached staging copy).
// Cleared on graph shape change (new token count changes masks).
static std::unordered_map<const void *, void *> g_leaf_staging_cache;

void ggml_sycl_cpu_staging_cache_clear() {
    g_leaf_staging_cache.clear();
}

// Current bank index (alternates per op) and event tracking
static int         g_staging_bank = 0;
static sycl::event g_staging_flush_evt[STAGING_BANKS];
static bool        g_staging_flush_pending[STAGING_BANKS] = { false, false };
static sycl::event g_staging_compute_evt[STAGING_BANKS];
static bool        g_staging_compute_pending[STAGING_BANKS] = { false, false };

// ---------------------------------------------------------------------------
// HOST_COMPUTE host_task mode: when active, CPU ops run as host_task callbacks
// on gpu_q instead of parallel_for on cpu_q.  This eliminates cross-queue
// sync overhead (10x faster per op for TG sizes).  The in-order gpu_q
// naturally serializes GPU kernels and host_tasks.
//
// Activated when GGML_SYCL_HOST_COMPUTE=1 — compute buffers are allocated as
// host-pinned USM (already host-accessible, no mirror/staging needed).
// get_host_ptr() returns t->data directly for these buffers.
// ---------------------------------------------------------------------------

static thread_local bool g_host_task_mode = false;

// BATCHED host_task mode: when active, CPU ops run as direct function calls
// inside a single batched host_task, not individual submissions.
// Activated by graph_compute_impl when collecting CPU segments.
// thread_local: host_task runs on a SYCL worker thread — each thread
// needs its own copy to avoid data races with the main submission thread.
static thread_local bool g_batched_mode = false;

static inline bool batched_mode_active() {
    return g_batched_mode;
}

static inline bool host_task_mode_active() {
    return g_host_task_mode;
}

// Called from graph_compute_impl when HOST_COMPUTE + CPU offload is active.
void ggml_sycl_host_task_mode_set(bool active) {
    g_host_task_mode = active;
}

void ggml_sycl_batched_mode_set(bool active) {
    g_batched_mode = active;
}

bool ggml_sycl_batched_mode_active() {
    return g_batched_mode;
}

static void staging_track_cpu_event(const sycl::event & evt) {
    try {
        if (!ggml_sycl_should_add_dependency(evt)) {
            return;
        }
    } catch (...) {
        return;
    }
    g_staging_compute_evt[g_staging_bank]     = evt;
    g_staging_compute_pending[g_staging_bank] = true;
}

static ggml_sycl::offload_buffer_role staging_role_for_slot(int slot) {
    switch (slot) {
        case 0:
            return ggml_sycl::offload_buffer_role::STAGING_SRC0;
        case 1:
            return ggml_sycl::offload_buffer_role::STAGING_SRC1;
        case 2:
            return ggml_sycl::offload_buffer_role::STAGING_DST;
        default:
            return ggml_sycl::offload_buffer_role::OTHER;
    }
}

static size_t staging_growth_granularity_bytes() {
    static size_t granularity = []() {
        const char * env = std::getenv("GGML_SYCL_CPU_STAGING_GROW_GRANULARITY_KB");
        const size_t kb  = env ? static_cast<size_t>(std::max(1, std::atoi(env))) : 256;
        return kb * 1024;
    }();
    return std::max<size_t>(64, granularity);
}

static size_t staging_target_capacity(size_t requested, size_t current_capacity) {
    size_t target = requested;
    if (current_capacity > 0) {
        const size_t grown = current_capacity + current_capacity / 2;  // 1.5x growth to reduce realloc churn
        if (grown > target) {
            target = grown;
        }
    }
    const size_t granularity = staging_growth_granularity_bytes();
    const size_t rounded     = ((target + granularity - 1) / granularity) * granularity;
    return std::max(rounded, requested);
}

static void * staging_ensure(int bank, int slot, size_t nbytes, sycl::queue * gpu_q) {
    if (bank < 0 || bank >= STAGING_BANKS || slot < 0 || slot >= STAGING_SLOTS_PER_BANK) {
        return nullptr;
    }
    auto & entry = g_cpu_staging[bank][slot];
    if (nbytes <= entry.cap && entry.ptr) {
        return entry.ptr;
    }
    // Return old buffer to the unified offload pool.
    if (entry.ptr) {
        (void) ggml_sycl::release_offload_buffer(entry.lease);
        entry.lease = {};
    }
    const size_t                      target_capacity = staging_target_capacity(nbytes, entry.cap);
    ggml_sycl::offload_buffer_request req{};
    req.queue                                         = gpu_q;
    req.device                                        = ggml_sycl_get_device_id_from_queue(*gpu_q);
    req.size                                          = target_capacity;
    req.alignment                                     = 64;
    req.role                                          = staging_role_for_slot(slot);
    req.intent.role                                   = ggml_sycl::alloc_role::STAGING;
    req.intent.category                               = ggml_sycl::runtime_category::STAGING;
    req.intent.cohort_id                              = "cpu_offload";
    req.intent.constraints.must_host_pinned           = true;
    req.intent.constraints.prefer_same_tier_as_cohort = true;
    if (!ggml_sycl::acquire_offload_buffer(req, &entry.lease)) {
        entry.ptr = nullptr;
        entry.cap = 0;
        return nullptr;
    }
    entry.ptr = entry.lease.handle.ptr;
    entry.cap = target_capacity;
    return entry.ptr;
}

// Begin a new staging operation.  Alternates to the next bank and waits for
// the previous op's flush to complete.  Since we alternate banks, the pending
// flush used the OTHER bank.  By waiting here we ensure the global flush
// event is drained before submitting new memcpys to the GPU queue.  The
// staging buffers for the bank we're about to use were last touched 2 ops ago
// and are already safe (waited on by the intervening op).
static void staging_begin_op() {
    // Batched mode in HOST_COMPUTE: no staging buffers used.
    if (g_batched_mode) {
        return;
    }
    const int next_bank = 1 - g_staging_bank;
    if (ggml_sycl_cpu_offload_async_enabled()) {
        // Async mode: wait only when reusing the target bank.
        // If no flush was scheduled (retained output), still wait for the
        // bank's last CPU compute event before reuse.
        if (g_staging_flush_pending[next_bank]) {
            offload_wait_event(g_staging_flush_evt[next_bank]);
            g_staging_flush_pending[next_bank]   = false;
            g_staging_compute_pending[next_bank] = false;
        } else if (g_staging_compute_pending[next_bank]) {
            offload_wait_event(g_staging_compute_evt[next_bank]);
            g_staging_compute_pending[next_bank] = false;
        }
    } else {
        // Legacy mode: preserve eager drain behavior.
        for (int b = 0; b < STAGING_BANKS; ++b) {
            if (g_staging_flush_pending[b]) {
                offload_wait_event(g_staging_flush_evt[b], offload_wait_reason::FALLBACK);
                g_staging_flush_pending[b]   = false;
                g_staging_compute_pending[b] = false;
            } else if (g_staging_compute_pending[b]) {
                offload_wait_event(g_staging_compute_evt[b], offload_wait_reason::FALLBACK);
                g_staging_compute_pending[b] = false;
            }
        }
    }
    g_staging_bank = next_bank;
}

// Get host-accessible pointer for a tensor.
// If tensor is in host-accessible memory, returns original pointer.
// For weight tensors: tries host_cache first (AOS data, no device copy needed).
// For activations/compute tensors: copies device→host via staging (event-based).
//
// out_event: if non-null, set to the memcpy event that must complete before
//            reading from the returned pointer.  If no staging was needed,
//            the event is left unchanged.
// Shared helper: check if a pointer is host-accessible via USM type.
// Caches results per base pointer to avoid repeated runtime queries.
static bool is_host_accessible_usm(void * ptr, int device) {
    static std::unordered_map<void *, bool> cache;
    auto it = cache.find(ptr);
    if (it != cache.end()) {
        return it->second;
    }
    bool is_host = true;  // assume host unless proven device
    try {
        sycl::context    ctx = ggml_sycl_get_device(device).default_queue().get_context();
        sycl::usm::alloc pt  = sycl::get_pointer_type(ptr, ctx);
        is_host              = (pt != sycl::usm::alloc::device);
    } catch (...) {}
    cache[ptr] = is_host;
    return is_host;
}

static void * get_host_ptr(const ggml_tensor * t,
                           int                 device,
                           int                 slot,
                           sycl::queue *       gpu_q,
                           sycl::event *       out_event = nullptr) {
    // Check retained activation map first — if this tensor's data was
    // produced by a prior CPU op in the same layer block, return the
    // host pointer directly without any D2H copy.
    // Follow view_src chain for RESHAPE/VIEW/PERMUTE noops: these create
    // new tensor objects that point to the same underlying data, but the
    // retained map keys are the original tensor pointers.
    if (g_retained_active) {
        const ggml_tensor * lookup = t;
        while (lookup) {
            auto it = g_retained_map.find(lookup);
            if (it != g_retained_map.end()) {
                // Found the source in retained map.  Apply view offset if we
                // traversed a view chain (RESHAPE view_offs is typically 0).
                char * base = static_cast<char *>(it->second.host_ptr);
                size_t off  = (lookup == t) ? 0 : (t->view_offs - lookup->view_offs);
                if (out_event) {
                    *out_event = sycl::event{};
                }
                return base + off;
            }
            lookup = lookup->view_src;
        }
    }

    // Host-accessible buffers (weight mmap, host-pinned) → use tensor->data directly
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }

    // For weight tensors: look up host-accessible data from cache or mmap.
    if (ggml_sycl_tensor_is_weight(t)) {
        if (ggml_sycl::unified_cache_enabled()) {
            ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(t, device);
            if (key.valid) {
                // Try unified cache — PINNED_HOST and MMAP entries are host-accessible.
                auto * cache = ggml_sycl::get_unified_cache_for_device(device);
                if (cache) {
                    ggml_sycl::cache_ptr_view view = cache->get_view(key, GGML_LAYOUT_AOS);
                    if (view.ptr && view.location != ggml_sycl::cache_location::DEVICE) {
                        return view.ptr;
                    }
                }

                // Try host_cache (pinned AOS copies of device-resident weights).
                auto * hcache = ggml_sycl::get_host_cache_for_device(device);
                if (hcache) {
                    int    layer_id  = ggml_sycl::extract_layer_id(t->name);
                    int    expert_id = ggml_sycl::extract_expert_id(t->name);
                    void * hp        = hcache->get(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, layer_id, expert_id,
                                                   GGML_LAYOUT_AOS);
                    if (hp) {
                        return hp;
                    }
                }
            }
        }

        // Fallback: retrieve original mmap host pointer from our static registry.
        // During set_tensor, we store the host data pointer (from the mmap'd GGUF
        // file) before the SYCL backend copies it to device memory.
        if (t->name) {
            const void * mmap_ptr = cpu_dispatch_lookup_host_ptr(t->name);
            if (mmap_ptr) {
                return const_cast<void *>(mmap_ptr);
            }
        }
        return nullptr;
    }

    // Non-weight tensors (activations, compute buffers).
    // Check persistent staging cache for leaf tensors.
    // Leaf tensors (RoPE freqs, masks) have stable data between tokens.
    if (t->data) {
        auto it = g_leaf_staging_cache.find(t->data);
        if (it != g_leaf_staging_cache.end()) {
            if (out_event) {
                *out_event = sycl::event{};
            }
            return it->second;
        }
    }

    // Non-contiguous tensors (e.g. permuted KV cache views) cannot be
    // copied with a linear memcpy.  Reject so the caller falls back to GPU.
    if (!ggml_is_contiguous(t)) {
        GGML_SYCL_DEBUG("[CPU-STAGE] Rejecting non-contiguous tensor %s\n", t->name ? t->name : "(null)");
        return nullptr;
    }

    // Check if tensor data is already host-accessible.  SYCL buffer backing
    // may be host memory (e.g. KV cache when VRAM is constrained) but
    // ggml_backend_buffer_is_host() returns false for SYCL buffer types.
    // Detect this by checking the USM pointer type of the base tensor's data.
    {
        const ggml_tensor * base = t;
        while (base->view_src) {
            base = base->view_src;
        }
        void * base_data = base->data;
        if (base_data && is_host_accessible_usm(base_data, device)) {
            if (out_event) {
                *out_event = sycl::event{};
            }
            GGML_SYCL_DEBUG("[CPU-STAGE] Host-accessible %s (base=%p) — no staging\n", t->name ? t->name : "(null)",
                            base_data);
            return t->data;
        }
    }

    // Batched mode: we're inside a host_task on gpu_q.  Cannot submit memcpy
    // to gpu_q (deadlock — in-order queue blocked by this host_task).  If we
    // reached here, the tensor isn't host-accessible → return nullptr so the
    // caller falls back to GPU dispatch or fails gracefully.
    if (g_batched_mode) {
        GGML_SYCL_DEBUG("[CPU-STAGE] Skipping staging in batched mode for %s\n", t->name ? t->name : "(null)");
        return nullptr;
    }

    // Resolve the device pointer.  For view/permute tensors (e.g. KV cache
    // views), extra->data_device may be NULL.  Walk the view_src chain to
    // find a base tensor with a known device pointer, then add the offset.
    void * dev_ptr = nullptr;
    {
        if (t->extra) {
            void * ed = static_cast<ggml_tensor_extra_gpu *>(t->extra)->data_device[device];
            if (ed) {
                dev_ptr = ed;
            }
        }

        if (!dev_ptr) {
            // Follow view_src chain to find base tensor with device pointer
            const ggml_tensor * base = t;
            while (base->view_src) {
                base = base->view_src;
            }
            if (base->extra) {
                void * base_dev = static_cast<ggml_tensor_extra_gpu *>(base->extra)->data_device[device];
                if (base_dev && base->data) {
                    ptrdiff_t offset = static_cast<char *>(t->data) - static_cast<char *>(base->data);
                    dev_ptr          = static_cast<char *>(base_dev) + offset;
                }
            }
        }

        // If view_src resolution failed, use ggml_sycl_get_data_ptr as fallback.
        if (!dev_ptr) {
            dev_ptr = ggml_sycl_get_data_ptr(t, device);
        }
    }

    if (!dev_ptr) {
        return nullptr;
    }

    size_t nbytes = ggml_nbytes(t);
    void * host   = staging_ensure(g_staging_bank, slot, nbytes, gpu_q);
    if (!host) {
        return nullptr;
    }
    GGML_SYCL_DEBUG("[CPU-STAGE] memcpy %s: host=%p <- dev=%p, %zu bytes (bank=%d slot=%d)\n", t->name ? t->name : "?",
                    host, dev_ptr, nbytes, g_staging_bank, slot);
    sycl::event evt = gpu_q->memcpy(host, dev_ptr, nbytes);
    ggml_sycl::offload_stats_note_transfer(false, nbytes);
    if (out_event) {
        *out_event = evt;
    } else {
        // Fallback: if caller doesn't handle events, wait synchronously
        offload_wait_event(evt, offload_wait_reason::FALLBACK);
    }
    // Cache for leaf tensors (stable data pointers between tokens).
    // Only cache non-weight tensors that aren't activations (no src[0]).
    // Leaf tensors in ggml have no source tensors.
    if (t->data && !t->src[0]) {
        g_leaf_staging_cache[t->data] = host;
    }
    return host;
}

// Copy output from host staging back to device memory (event-based).
// No-op if tensor is already in host-accessible memory.
// The flush event is tracked internally and awaited at the start of the next op.
static void flush_output(ggml_tensor *       t,
                         int                 device,
                         sycl::queue *       gpu_q,
                         const sycl::event * dep_evt              = nullptr,
                         bool                dep_event_same_queue = false) {
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return;
    }
    // Batched mode: inside a host_task on gpu_q — cannot submit memcpy (deadlock).
    if (g_batched_mode) {
        return;
    }
    // HOST_COMPUTE: host-pinned buffers don't need staging flush.
    if (t->data && is_host_accessible_usm(t->data, device)) {
        return;
    }
    // When retained mode is active, outputs stay in host scratch.
    // flush_all at CPU→GPU boundary handles the final copy.
    if (g_retained_active && g_retained_map.count(t)) {
        return;
    }
    void * dev_ptr = ggml_sycl_get_data_ptr(t, device);
    auto & entry   = g_cpu_staging[g_staging_bank][2];
    if (!dev_ptr || !entry.ptr) {
        return;
    }
    size_t nbytes = ggml_nbytes(t);
    ggml_sycl::offload_stats_note_transfer(true, nbytes);
    std::vector<sycl::event> deps;
    deps.reserve(2);
    if (dep_evt) {
        if (dep_event_same_queue) {
            append_dependency(deps, *dep_evt);
        } else {
            wait_dependency_if_needed(*dep_evt);
        }
    }
    if (g_staging_flush_pending[g_staging_bank]) {
        append_dependency(deps, g_staging_flush_evt[g_staging_bank]);
    }
    if (deps.empty()) {
        g_staging_flush_evt[g_staging_bank] = gpu_q->memcpy(dev_ptr, entry.ptr, nbytes);
    } else {
        g_staging_flush_evt[g_staging_bank] = gpu_q->submit([&](sycl::handler & cgh) {
            cgh.depends_on(deps);
            cgh.memcpy(dev_ptr, entry.ptr, nbytes);
        });
    }
    g_staging_flush_pending[g_staging_bank] = true;
}

// Get host pointer for output tensor.
// Uses staging slot 2 of the current bank.
static void * get_host_output_ptr(ggml_tensor * t, int device, sycl::queue * gpu_q) {
    // Host-accessible buffer → use tensor->data directly
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }
    // HOST_COMPUTE: SYCL-allocated host-pinned USM buffers are host-accessible
    // but ggml_backend_buffer_is_host() returns false for SYCL buffer types.
    // Check USM pointer type to detect host-accessible compute buffers.
    if (t->data && is_host_accessible_usm(t->data, device)) {
        return t->data;
    }
    // Device-resident: allocate staging but don't copy (will be written by kernel)
    size_t nbytes = ggml_nbytes(t);
    return staging_ensure(g_staging_bank, 2, nbytes, gpu_q);
}

// Helper: get output pointer from retained scratch or staging fallback.
// Sets *retained to true if output goes to scratch, false for staging.
static void * get_retained_or_staging_output(ggml_tensor * dst, int device, sycl::queue * gpu_q, bool * retained) {
    // Batched mode: output directly to host-pinned t->data
    if (g_batched_mode && dst->data && is_host_accessible_usm(dst->data, device)) {
        *retained = false;
        return dst->data;
    }
    void * scratch_ptr = ggml_sycl_cpu_retained_alloc_output(dst);
    if (scratch_ptr) {
        *retained = true;
        return scratch_ptr;
    }
    *retained = false;
    return get_host_output_ptr(dst, device, gpu_q);
}

// Wait for all pending staging events (call at boundary sync points).
void ggml_sycl_cpu_staging_drain() {
    cpu_wait_chain_event();
    for (int b = 0; b < STAGING_BANKS; ++b) {
        if (g_staging_flush_pending[b]) {
            offload_wait_event(g_staging_flush_evt[b]);
            g_staging_flush_pending[b]   = false;
            g_staging_compute_pending[b] = false;
        } else if (g_staging_compute_pending[b]) {
            offload_wait_event(g_staging_compute_evt[b]);
            g_staging_compute_pending[b] = false;
        }
    }
}

// Thread count hint for CPU vec_dot path.
// GGML_SYCL_CPU_THREADS=1 forces serial execution.
static int ggml_sycl_cpu_threads_hint() {
    static int n_threads = []() {
        const char * env   = getenv("GGML_SYCL_CPU_THREADS");
        const int    hw    = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        int          value = env ? std::max(1, atoi(env)) : std::max(1, hw - 2);
        return std::min(value, 32);
    }();
    return n_threads;
}

#if GGML_SYCL_HAS_TBB
// Persistent task arena: keeps TBB worker threads alive between parallel_for
// calls, eliminating the ~12% overhead from repeated wake/sleep cycles across
// the ~90 MUL_MAT ops per TG token.
static ggml_sycl_tbb::task_arena & ggml_sycl_cpu_arena() {
    static ggml_sycl_tbb::task_arena arena(ggml_sycl_cpu_threads_hint());
    return arena;
}
#endif

// Minimum output work (N*M) before enabling TBB in vec_dot path.
// Keeps tiny TG workloads on the serial fast path to avoid scheduler overhead.
static int ggml_sycl_cpu_vecdot_min_parallel_work() {
    static int min_work = []() {
        const char * env   = getenv("GGML_SYCL_CPU_OFFLOAD_VECDOT_MIN_WORK");
        const int    value = env ? atoi(env) : 512;
        return std::max(1, value);
    }();
    return min_work;
}

// Lower bound on rows-per-task for vec_dot TBB partitioning.
static int ggml_sycl_cpu_vecdot_min_rows_per_task() {
    static int rows = []() {
        const char * env   = getenv("GGML_SYCL_CPU_OFFLOAD_VECDOT_MIN_ROWS_PER_TASK");
        const int    value = env ? atoi(env) : 4;
        return std::max(1, value);
    }();
    return rows;
}

// Target number of tasks per thread in vec_dot TBB partitioning.
static int ggml_sycl_cpu_vecdot_tasks_per_thread() {
    static int tasks = []() {
        const char * env   = getenv("GGML_SYCL_CPU_OFFLOAD_VECDOT_TASKS_PER_THREAD");
        const int    value = env ? atoi(env) : 2;
        return std::max(1, value);
    }();
    return tasks;
}

// ---------------------------------------------------------------------------
// MUL_MAT  (oneDNN on host, async host_task when enabled)
// ---------------------------------------------------------------------------

static bool cpu_mul_mat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // activations

    if (!src0 || !src1) {
        return false;
    }

    // Activations must be F32, output must be F32
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Supported weight types: F32, F16, or any quantized type with to_float
    const bool src0_f32       = (src0->type == GGML_TYPE_F32);
    const bool src0_quantized = !src0_f32 && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));

    if (!src0_f32 && !src0_quantized) {
        return false;
    }

    const auto * type_traits = src0_quantized ? ggml_get_type_traits(src0->type) : nullptr;
    if (src0_quantized && (!type_traits || !type_traits->to_float)) {
        return false;
    }

    // ggml MUL_MAT convention:  C^T = A * B^T  =>  C = B * A^T
    //   src0 = A  (weights)     [ne00 x ne01]  (ne00 = K, ne01 = N)
    //   src1 = B  (activations) [ne10 x ne11]  (ne10 = K, ne11 = M)
    //   dst  = C  (output)      [ne0  x ne1 ]  (ne0  = N, ne1  = M)
    //
    // We need: C[M,N] = src1[M,K] * src0^T[K,N]

    GGML_ASSERT(src0->ne[0] == src1->ne[0]);

    const dnnl_dim_t M = static_cast<dnnl_dim_t>(src1->ne[1]);
    const dnnl_dim_t N = static_cast<dnnl_dim_t>(src0->ne[1]);
    const dnnl_dim_t K = static_cast<dnnl_dim_t>(src0->ne[0]);

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    // Stage tensors to host-accessible memory (event-based).
    sycl::event e0, e1;
    const bool  async_requested = ggml_sycl_cpu_offload_async_enabled();
    const bool  async_mode      = async_requested && gpu_q;

    // Async path safety: prefer persistent registered host copy for weights.
    // Host cache/unified-cache views are not lease-pinned for async task lifetime.
    const void * src0_data = nullptr;
    if (async_mode) {
        src0_data = cpu_dispatch_lookup_host_ptr(src0->name);
    }
    if (!src0_data) {
        src0_data = get_host_ptr(src0, device, 0, gpu_q, &e0);
    }
    const void * src1_data = get_host_ptr(src1, device, 1, gpu_q, &e1);

    bool   retained_output;
    void * dst_data = get_retained_or_staging_output(dst, device, gpu_q, &retained_output);

    if (!src0_data || !src1_data || !dst_data) {
        return false;
    }

    // Batch dimensions (broadcast src0 if ne02/ne03 < ne12/ne13)
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    // AOS stride for host data (tensor nb may be SOA, but mmap/cache data is AOS)
    const int64_t nb01 = static_cast<int64_t>(ggml_row_size(src0->type, K));
    const int64_t nb02 = nb01 * src0->ne[1];
    const int64_t nb03 = nb02 * src0->ne[2];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];
    const int64_t nb2  = dst->nb[2];
    const int64_t nb3  = dst->nb[3];

    const dnnl_dim_t ldc = static_cast<dnnl_dim_t>(dst->nb[1] / sizeof(float));

    // For small M (TG batch=1..4), use quantized dot product when available.
    // This avoids dequantizing the entire N×K weight matrix to F32 and replaces
    // dnnl_sgemm GEMV with direct quantized vec_dot (e.g., Q4_0 × Q8_0).
    // Benefits: ~5x less memory bandwidth (quantized reads), no BLAS overhead,
    // L1-friendly access pattern (one 2KB weight row + 4KB activation per dot).
    const auto * cpu_traits  = src0_quantized ? ggml_get_type_traits_cpu(src0->type) : nullptr;
    const bool   use_vec_dot = (M <= 4 && cpu_traits && cpu_traits->vec_dot);

    auto run_mul_mat = [=]() {
        ggml_from_float_t    from_float_fn = nullptr;
        size_t               q_row_size    = 0;
        static thread_local std::vector<uint8_t> src1_q_buf;

        // Activation quantization cache: Q/K/V projections share the same src1
        // tensor (hidden state), so we can skip re-quantizing when the tensor
        // identity and dimensions haven't changed.  Saves ~3x quantization per
        // layer.  Keyed on tensor pointer + graph generation to prevent stale
        // hits across tokens (graph replay reuses tensor pointers with new data).
        struct quant_cache_key {
            const ggml_tensor * tensor     = nullptr;
            uint64_t            generation = 0;
            dnnl_dim_t          M          = 0;
            dnnl_dim_t          K          = 0;
            ggml_from_float_t   fn         = nullptr;

            bool matches(const ggml_tensor * t, uint64_t gen, dnnl_dim_t m, dnnl_dim_t k,
                         ggml_from_float_t f) const {
                return tensor == t && generation == gen && M == m && K == k && fn == f;
            }
        };
        static thread_local quant_cache_key src1_q_cache;

        if (use_vec_dot) {
            const ggml_type vec_dot_type   = cpu_traits->vec_dot_type;
            const auto *    vdt_cpu_traits = ggml_get_type_traits_cpu(vec_dot_type);
            from_float_fn                  = vdt_cpu_traits ? vdt_cpu_traits->from_float : nullptr;
            if (from_float_fn) {
                q_row_size = ggml_row_size(vec_dot_type, K);
                src1_q_buf.resize(static_cast<size_t>(M) * q_row_size);
            }
        }

        // Dequant/conversion buffer for non-F32 weights (only for GEMM fallback path)
        static thread_local std::vector<float> src0_f32_buf;
        if (src0_quantized && !use_vec_dot) {
            src0_f32_buf.resize(static_cast<size_t>(N) * K);
        }

        for (int64_t i13 = 0; i13 < ne13; i13++) {
            for (int64_t i12 = 0; i12 < ne12; i12++) {
                const int64_t i02 = i12 % ne02;
                const int64_t i03 = i13 % ne03;

                const char *  src0_batch = static_cast<const char *>(src0_data) + i02 * nb02 + i03 * nb03;
                const float * src1_batch =
                    reinterpret_cast<const float *>(static_cast<const char *>(src1_data) + i12 * nb12 + i13 * nb13);
                float * dst_batch = reinterpret_cast<float *>(static_cast<char *>(dst_data) + i12 * nb2 + i13 * nb3);

                if (use_vec_dot && from_float_fn) {
                    // Quantized dot product path: quantize activations, then vec_dot
                    // per output element.  No weight dequantization needed.
                    const uint64_t gen = g_quant_cache_generation.load(std::memory_order_relaxed);
                    if (!src1_q_cache.matches(src1, gen, M, K, from_float_fn)) {
                        for (dnnl_dim_t m = 0; m < M; m++) {
                            from_float_fn(src1_batch + m * K, src1_q_buf.data() + m * q_row_size, K);
                        }
                        src1_q_cache = { src1, gen, M, K, from_float_fn };
                    }

                    // Parallel vec_dot over N (output rows).
                    // Each thread processes a contiguous chunk of weight rows.
                    // Thread-safe: each (n,m) writes to a unique dst_batch location.
                    const int N_int          = static_cast<int>(N);
                    const int n_threads_hint = ggml_sycl_cpu_threads_hint();

                    const int64_t total_work = static_cast<int64_t>(N_int) * M;
                    if (N_int > 1 && n_threads_hint > 1 && total_work >= ggml_sycl_cpu_vecdot_min_parallel_work()) {
#    if GGML_SYCL_HAS_TBB
                        const int target_tasks = std::max(1, n_threads_hint * ggml_sycl_cpu_vecdot_tasks_per_thread());
                        const int grain_from_target = std::max(1, (N_int + target_tasks - 1) / target_tasks);
                        const int grain = std::max(grain_from_target, ggml_sycl_cpu_vecdot_min_rows_per_task());
                        // Extract pointer before parallel_for: src1_q_buf is static thread_local,
                        // so TBB worker threads would see their own empty instances.
                        // Capturing the raw pointer ensures all workers use the populated buffer.
                        uint8_t * src1_q_data = src1_q_buf.data();
                        ggml_sycl_cpu_arena().execute([&] {
                            ggml_sycl_tbb::parallel_for(
                                ggml_sycl_tbb::blocked_range<int>(0, N_int, grain),
                                [&, src1_q_data](const ggml_sycl_tbb::blocked_range<int> & r) {
                                    for (int n = r.begin(); n < r.end(); n++) {
                                        const void * weight_row = src0_batch + n * nb01;
                                        for (dnnl_dim_t m = 0; m < M; m++) {
                                            float dot_result = 0.0f;
                                            cpu_traits->vec_dot(static_cast<int>(K), &dot_result, sizeof(float),
                                                                weight_row, 0, src1_q_data + m * q_row_size, 0, 1);
                                            dst_batch[m * ldc + n] = dot_result;
                                        }
                                    }
                                });
                        });
#    else
                        for (int n = 0; n < N_int; n++) {
                            const void * weight_row = src0_batch + static_cast<dnnl_dim_t>(n) * nb01;
                            for (dnnl_dim_t m = 0; m < M; m++) {
                                float dot_result = 0.0f;
                                cpu_traits->vec_dot(static_cast<int>(K), &dot_result, sizeof(float), weight_row, 0,
                                                    src1_q_buf.data() + m * q_row_size, 0, 1);
                                dst_batch[m * ldc + static_cast<dnnl_dim_t>(n)] = dot_result;
                            }
                        }
#    endif
                    } else {
                        // Small N or single thread: use original serial path
                        for (dnnl_dim_t n = 0; n < N; n++) {
                            const void * weight_row = src0_batch + n * nb01;
                            for (dnnl_dim_t m = 0; m < M; m++) {
                                float dot_result = 0.0f;
                                cpu_traits->vec_dot(static_cast<int>(K), &dot_result, sizeof(float), weight_row, 0,
                                                    src1_q_buf.data() + m * q_row_size, 0, 1);
                                dst_batch[m * ldc + n] = dot_result;
                            }
                        }
                    }
                } else {
                    // GEMM fallback: dequantize weights to F32, then dnnl_sgemm
                    const float * weight_f32;
                    dnnl_dim_t    weight_ld = K;

                    if (src0_f32) {
                        weight_f32 = reinterpret_cast<const float *>(src0_batch);
                        weight_ld  = static_cast<dnnl_dim_t>(nb01 / sizeof(float));
                    } else {
                        for (dnnl_dim_t row = 0; row < N; row++) {
                            const void * row_data = src0_batch + row * nb01;
                            type_traits->to_float(row_data, src0_f32_buf.data() + row * K, K);
                        }
                        weight_f32 = src0_f32_buf.data();
                    }

                    dnnl_sgemm('T', 'N', N, M, K, 1.0f, weight_f32, weight_ld, src1_batch, K, 0.0f, dst_batch, ldc);
                }
            }
        }
    };

    if (batched_mode_active()) {
        // Direct execution inside batched host_task — no submission, no events.
        // In HOST_COMPUTE mode: src0 from mmap, src1/dst from host-pinned t->data.
        run_mul_mat();
        // No flush_output: dst_data already points to host-pinned t->data
        // (after Change 4 fixes get_host_output_ptr).
        return true;
    }

    if (async_mode) {
        // Async path: submit host compute on the GPU queue so completion is
        // naturally visible to graph scheduling and downstream GPU kernels.
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, &e1, gpu_q);
        sycl::event              cpu_evt =
            cpu_submit_async(gpu_q, deps, [=](sycl::handler & cgh) { cgh.host_task([=]() { run_mul_mat(); }); });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt, true);
        }
        return true;
    }

    // Legacy sync path
    cpu_wait_chain_event();
    offload_wait_event(e0, offload_wait_reason::FALLBACK);
    offload_wait_event(e1, offload_wait_reason::FALLBACK);
    run_mul_mat();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
#endif
}

// ---------------------------------------------------------------------------
// RMS_NORM  (SYCL parallel_for on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0)) {
        return false;
    }

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event   e0;
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        for (int64_t row = 0; row < nrows; row++) {
            const float * src_row = src_data + row * ne00;
            float *       dst_row = dst_data + row * ne00;

            float sum_sq = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                sum_sq += src_row[j] * src_row[j];
            }

            const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(ne00) + eps);

            for (int64_t j = 0; j < ne00; j++) {
                dst_row[j] = src_row[j] * scale;
            }
        }
    } else if (host_task) {
        // host_task on gpu_q: in-order queue serializes with GPU kernels.
        // No cross-queue sync needed.  std:: math is faster than sycl:: on host.
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() {
                for (int64_t row = 0; row < nrows; row++) {
                    const float * src_row = src_data + row * ne00;
                    float *       dst_row = dst_data + row * ne00;

                    float sum_sq = 0.0f;
                    for (int64_t j = 0; j < ne00; j++) {
                        sum_sq += src_row[j] * src_row[j];
                    }

                    const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(ne00) + eps);

                    for (int64_t j = 0; j < ne00; j++) {
                        dst_row[j] = src_row[j] * scale;
                    }
                }
            });
        });
    } else {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, nullptr, cpu_q);

        // One work-item per row — each computes RMS and normalizes.
        sycl::event cpu_evt = cpu_submit_async(cpu_q, deps, [&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
                const int64_t row     = static_cast<int64_t>(row_id[0]);
                const float * src_row = src_data + row * ne00;
                float *       dst_row = dst_data + row * ne00;

                float sum_sq = 0.0f;
                for (int64_t j = 0; j < ne00; j++) {
                    sum_sq += src_row[j] * src_row[j];
                }

                const float scale = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(ne00) + eps);

                for (int64_t j = 0; j < ne00; j++) {
                    dst_row[j] = src_row[j] * scale;
                }
            });
        });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// ADD / MUL  (SYCL parallel_for on CPU queue, general ND broadcast)
// ---------------------------------------------------------------------------
//
// General broadcast: src1 dimensions can be 1 where src0 dimensions are > 1.
// The broadcast pattern uses modulo indexing across all 4 dimensions,
// following the same stride-based approach as ggml-cpu/binary-ops.cpp.
// src0 and dst always have the same shape.

enum class binary_op_type { OP_ADD, OP_MUL };

static bool cpu_binary_op(ggml_backend_sycl_context & ctx, ggml_tensor * dst, binary_op_type op) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || !src1) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // src0 and dst must be contiguous; src1 must be contiguous along dim 0
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float)) {
        return false;
    }

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event   e0, e1;
    const float * src0_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));
    const float * src1_data = static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q, &e1));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src0_data || !src1_data || !dst_data) {
        return false;
    }

    // dst/src0 dimensions
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    // src1 dimensions (may be smaller for broadcasting)
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    // src1 strides in floats
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);

    // Number of column repetitions within a row
    const int64_t nr0 = ne00 / ne10;

    // Total rows = ne01 * ne02 * ne03
    const int64_t total_rows = ne01 * ne02 * ne03;

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        for (int64_t ir = 0; ir < total_rows; ir++) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            const float * src1_row = src1_data + i13 * s13 + i12 * s12 + i11 * s11;
            const float * sp0      = src0_data + ir * ne00;
            float *       dp       = dst_data + ir * ne00;

            if (op == binary_op_type::OP_ADD) {
                for (int64_t r = 0; r < nr0; r++) {
                    for (int64_t j = 0; j < ne10; j++) {
                        dp[r * ne10 + j] = sp0[r * ne10 + j] + src1_row[j];
                    }
                }
            } else {
                for (int64_t r = 0; r < nr0; r++) {
                    for (int64_t j = 0; j < ne10; j++) {
                        dp[r * ne10 + j] = sp0[r * ne10 + j] * src1_row[j];
                    }
                }
            }
        }
    } else if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() {
                for (int64_t ir = 0; ir < total_rows; ir++) {
                    const int64_t i03 = ir / (ne02 * ne01);
                    const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
                    const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

                    const int64_t i13 = i03 % ne13;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i11 = i01 % ne11;

                    const float * src1_row = src1_data + i13 * s13 + i12 * s12 + i11 * s11;
                    const float * sp0      = src0_data + ir * ne00;
                    float *       dp       = dst_data + ir * ne00;

                    if (op == binary_op_type::OP_ADD) {
                        for (int64_t r = 0; r < nr0; r++) {
                            for (int64_t j = 0; j < ne10; j++) {
                                dp[r * ne10 + j] = sp0[r * ne10 + j] + src1_row[j];
                            }
                        }
                    } else {
                        for (int64_t r = 0; r < nr0; r++) {
                            for (int64_t j = 0; j < ne10; j++) {
                                dp[r * ne10 + j] = sp0[r * ne10 + j] * src1_row[j];
                            }
                        }
                    }
                }
            });
        });
    } else {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, &e1, cpu_q);

        sycl::event cpu_evt = cpu_submit_async(cpu_q, deps, [&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_rows)), [=](sycl::id<1> row_id) {
                const int64_t ir  = static_cast<int64_t>(row_id[0]);
                const int64_t i03 = ir / (ne02 * ne01);
                const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
                const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

                const int64_t i13 = i03 % ne13;
                const int64_t i12 = i02 % ne12;
                const int64_t i11 = i01 % ne11;

                const int64_t src0_row_off = ir * ne00;
                const int64_t dst_row_off  = ir * ne00;

                const float * src1_row = src1_data + i13 * s13 + i12 * s12 + i11 * s11;
                const float * sp0      = src0_data + src0_row_off;
                float *       dp       = dst_data + dst_row_off;

                if (op == binary_op_type::OP_ADD) {
                    for (int64_t r = 0; r < nr0; r++) {
                        for (int64_t j = 0; j < ne10; j++) {
                            dp[r * ne10 + j] = sp0[r * ne10 + j] + src1_row[j];
                        }
                    }
                } else {
                    for (int64_t r = 0; r < nr0; r++) {
                        for (int64_t j = 0; j < ne10; j++) {
                            dp[r * ne10 + j] = sp0[r * ne10 + j] * src1_row[j];
                        }
                    }
                }
            });
        });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt);
        }
    }
    return true;
}

static bool cpu_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    return cpu_binary_op(ctx, dst, binary_op_type::OP_ADD);
}

static bool cpu_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    return cpu_binary_op(ctx, dst, binary_op_type::OP_MUL);
}

// ---------------------------------------------------------------------------
// SILU  (x * sigmoid(x), element-wise)
// ---------------------------------------------------------------------------

static bool cpu_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event   e0;
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t n = ggml_nelements(dst);

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        for (int64_t i = 0; i < n; i++) {
            const float x = src_data[i];
            dst_data[i]   = x / (1.0f + std::exp(-x));
        }
    } else if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() {
                for (int64_t i = 0; i < n; i++) {
                    const float x = src_data[i];
                    dst_data[i]   = x / (1.0f + std::exp(-x));
                }
            });
        });
    } else {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, nullptr, cpu_q);

        sycl::event cpu_evt = cpu_submit_async(cpu_q, deps, [&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
                const float x = src_data[i];
                dst_data[i]   = x / (1.0f + sycl::exp(-x));
            });
        });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// GLU  (SWIGLU, REGLU, GEGLU variants — fused gate*up)
// ---------------------------------------------------------------------------

static bool cpu_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1 && !ggml_is_contiguous(src1)) {
        return false;
    }

    const enum ggml_glu_op glu_op  = ggml_get_glu_op(dst);
    const int32_t          swapped = ((const int32_t *) (dst->op_params))[1];

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event   e0, e1;
    const float * src0_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));
    const float * src1_data = src1 ? static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q, &e1)) : src0_data;

    if (!src0_data || !src1_data) {
        return false;
    }

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));
    if (!dst_data) {
        return false;
    }

    const int64_t nc    = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nrows = ggml_nrows(src0);

    const int64_t src0_row_stride = src0->nb[1] / sizeof(float);
    const int64_t src1_row_stride = src1 ? (src1->nb[1] / sizeof(float)) : src0_row_stride;
    const int64_t dst_row_stride  = dst->nb[1] / sizeof(float);

    const int64_t gate_offset = (!src1 && swapped) ? nc : 0;
    const int64_t up_offset   = (!src1 && swapped) ? 0 : nc;
    const bool    has_src1    = (src1 != nullptr);

    // GLU activation helper — shared between host_task and parallel_for paths.
    // Uses template to select std:: (host) vs sycl:: (device) math.
    auto glu_activate = [](float gate_val, ggml_glu_op op, auto exp_fn, auto erf_fn) -> float {
        switch (op) {
            case GGML_GLU_OP_SWIGLU:
            case GGML_GLU_OP_SWIGLU_OAI:
                return gate_val / (1.0f + exp_fn(-gate_val));
            case GGML_GLU_OP_REGLU:
                return gate_val > 0.0f ? gate_val : 0.0f;
            case GGML_GLU_OP_GEGLU:
            case GGML_GLU_OP_GEGLU_ERF:
                return 0.5f * gate_val * (1.0f + erf_fn(gate_val * 0.7071067811865475f));
            case GGML_GLU_OP_GEGLU_QUICK:
                return gate_val / (1.0f + exp_fn(-1.702f * gate_val));
            default:
                return gate_val;
        }
    };

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        for (int64_t row = 0; row < nrows; row++) {
            for (int64_t col = 0; col < nc; col++) {
                float gate_val, up_val;
                if (has_src1) {
                    gate_val = src0_data[row * src0_row_stride + col];
                    up_val   = src1_data[row * src1_row_stride + col];
                } else {
                    gate_val = src0_data[row * src0_row_stride + gate_offset + col];
                    up_val   = src0_data[row * src0_row_stride + up_offset + col];
                }

                float activated = glu_activate(
                    gate_val, glu_op, [](float x) { return std::exp(x); },
                    [](float x) { return std::erf(x); });

                dst_data[row * dst_row_stride + col] = activated * up_val;
            }
        }
    } else if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() {
                for (int64_t row = 0; row < nrows; row++) {
                    for (int64_t col = 0; col < nc; col++) {
                        float gate_val, up_val;
                        if (has_src1) {
                            gate_val = src0_data[row * src0_row_stride + col];
                            up_val   = src1_data[row * src1_row_stride + col];
                        } else {
                            gate_val = src0_data[row * src0_row_stride + gate_offset + col];
                            up_val   = src0_data[row * src0_row_stride + up_offset + col];
                        }

                        float activated = glu_activate(
                            gate_val, glu_op, [](float x) { return std::exp(x); },
                            [](float x) { return std::erf(x); });

                        dst_data[row * dst_row_stride + col] = activated * up_val;
                    }
                }
            });
        });
    } else {
        std::vector<sycl::event> deps =
            src1 ? cpu_collect_deps(&e0, &e1, cpu_q) : cpu_collect_deps(&e0, nullptr, cpu_q);

        sycl::event cpu_evt = cpu_submit_async(cpu_q, deps, [&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows * nc)), [=](sycl::id<1> idx) {
                const int64_t row = static_cast<int64_t>(idx[0]) / nc;
                const int64_t col = static_cast<int64_t>(idx[0]) % nc;

                float gate_val, up_val;
                if (has_src1) {
                    gate_val = src0_data[row * src0_row_stride + col];
                    up_val   = src1_data[row * src1_row_stride + col];
                } else {
                    gate_val = src0_data[row * src0_row_stride + gate_offset + col];
                    up_val   = src0_data[row * src0_row_stride + up_offset + col];
                }

                float activated = glu_activate(
                    gate_val, glu_op, [](float x) { return sycl::exp(x); },
                    [](float x) { return sycl::erf(x); });

                dst_data[row * dst_row_stride + col] = activated * up_val;
            });
        });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// SOFT_MAX  (row-wise softmax with scale and optional mask)
// ---------------------------------------------------------------------------

static bool cpu_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // Optional mask (src1) — F32 only for CPU path
    const ggml_tensor * src1 = dst->src[1];
    if (src1 && src1->type != GGML_TYPE_F32) {
        return false;
    }

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    // ALiBi not supported on CPU path for simplicity
    if (max_bias != 0.0f) {
        return false;
    }

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event   e0, e1;
    const float * src_data  = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));
    const float * mask_data = src1 ? static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q, &e1)) : nullptr;

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t ne00  = src0->ne[0];
    const int64_t ne01  = src0->ne[1];
    const int64_t ne02  = src0->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    const int64_t mask_nb11 = src1 ? (int64_t) (src1->nb[1] / sizeof(float)) : 0;
    const int64_t mask_nb12 = src1 ? (int64_t) (src1->nb[2] / sizeof(float)) : 0;
    const int64_t mask_nb13 = src1 ? (int64_t) (src1->nb[3] / sizeof(float)) : 0;
    const int64_t mask_ne12 = src1 ? src1->ne[2] : 1;
    const int64_t mask_ne13 = src1 ? src1->ne[3] : 1;

    // Softmax row kernel — shared between host_task and parallel_for paths.
    auto softmax_row = [](const float * sp, float * dp, const float * mp, int64_t width, float sc,
                          auto exp_fn) {
        float max_val = -INFINITY;
        for (int64_t j = 0; j < width; j++) {
            float v = sp[j] * sc;
            if (mp) {
                v += mp[j];
            }
            dp[j] = v;
            if (v > max_val) {
                max_val = v;
            }
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < width; j++) {
            dp[j] = exp_fn(dp[j] - max_val);
            sum += dp[j];
        }
        const float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < width; j++) {
            dp[j] *= inv_sum;
        }
    };

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        for (int64_t row = 0; row < nrows; row++) {
            const float * sp = src_data + row * ne00;
            float *       dp = dst_data + row * ne00;

            const float * mp = nullptr;
            if (mask_data) {
                const int64_t i01 = row % ne01;
                const int64_t i02 = (row / ne01) % ne02;
                const int64_t i03 = row / (ne01 * ne02);
                mp = mask_data + (i01 * mask_nb11) + (i02 % mask_ne12) * mask_nb12 +
                     (i03 % mask_ne13) * mask_nb13;
            }

            softmax_row(sp, dp, mp, ne00, scale, [](float x) { return std::exp(x); });
        }
    } else if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() {
                for (int64_t row = 0; row < nrows; row++) {
                    const float * sp = src_data + row * ne00;
                    float *       dp = dst_data + row * ne00;

                    const float * mp = nullptr;
                    if (mask_data) {
                        const int64_t i01 = row % ne01;
                        const int64_t i02 = (row / ne01) % ne02;
                        const int64_t i03 = row / (ne01 * ne02);
                        mp = mask_data + (i01 * mask_nb11) + (i02 % mask_ne12) * mask_nb12 +
                             (i03 % mask_ne13) * mask_nb13;
                    }

                    softmax_row(sp, dp, mp, ne00, scale, [](float x) { return std::exp(x); });
                }
            });
        });
    } else {
        std::vector<sycl::event> deps =
            src1 ? cpu_collect_deps(&e0, &e1, cpu_q) : cpu_collect_deps(&e0, nullptr, cpu_q);

        sycl::event cpu_evt = cpu_submit_async(cpu_q, deps, [&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
                const int64_t row = static_cast<int64_t>(row_id[0]);
                const float * sp  = src_data + row * ne00;
                float *       dp  = dst_data + row * ne00;

                const float * mp = nullptr;
                if (mask_data) {
                    const int64_t i01 = row % ne01;
                    const int64_t i02 = (row / ne01) % ne02;
                    const int64_t i03 = row / (ne01 * ne02);
                    mp = mask_data + (i01 * mask_nb11) + (i02 % mask_ne12) * mask_nb12 +
                         (i03 % mask_ne13) * mask_nb13;
                }

                softmax_row(sp, dp, mp, ne00, scale, [](float x) { return sycl::exp(x); });
            });
        });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// NORM  (layer normalization: mean-subtract, variance-normalize)
// ---------------------------------------------------------------------------

static bool cpu_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0)) {
        return false;
    }

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event   e0;
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        for (int64_t row = 0; row < nrows; row++) {
            const float * src_row = src_data + row * ne00;
            float *       dst_row = dst_data + row * ne00;

            float sum = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                sum += src_row[j];
            }
            const float mean = sum / static_cast<float>(ne00);

            float var = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                float d    = src_row[j] - mean;
                dst_row[j] = d;
                var += d * d;
            }
            var /= static_cast<float>(ne00);

            const float sc = 1.0f / std::sqrt(var + eps);
            for (int64_t j = 0; j < ne00; j++) {
                dst_row[j] *= sc;
            }
        }
    } else if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() {
                for (int64_t row = 0; row < nrows; row++) {
                    const float * src_row = src_data + row * ne00;
                    float *       dst_row = dst_data + row * ne00;

                    float sum = 0.0f;
                    for (int64_t j = 0; j < ne00; j++) {
                        sum += src_row[j];
                    }
                    const float mean = sum / static_cast<float>(ne00);

                    float var = 0.0f;
                    for (int64_t j = 0; j < ne00; j++) {
                        float d    = src_row[j] - mean;
                        dst_row[j] = d;
                        var += d * d;
                    }
                    var /= static_cast<float>(ne00);

                    const float sc = 1.0f / std::sqrt(var + eps);
                    for (int64_t j = 0; j < ne00; j++) {
                        dst_row[j] *= sc;
                    }
                }
            });
        });
    } else {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, nullptr, cpu_q);

        sycl::event cpu_evt = cpu_submit_async(cpu_q, deps, [&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
                const int64_t row     = static_cast<int64_t>(row_id[0]);
                const float * src_row = src_data + row * ne00;
                float *       dst_row = dst_data + row * ne00;

                float sum = 0.0f;
                for (int64_t j = 0; j < ne00; j++) {
                    sum += src_row[j];
                }
                const float mean = sum / static_cast<float>(ne00);

                float var = 0.0f;
                for (int64_t j = 0; j < ne00; j++) {
                    float d    = src_row[j] - mean;
                    dst_row[j] = d;
                    var += d * d;
                }
                var /= static_cast<float>(ne00);

                const float sc = 1.0f / sycl::sqrt(var + eps);
                for (int64_t j = 0; j < ne00; j++) {
                    dst_row[j] *= sc;
                }
            });
        });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// SCALE  (multiply all elements by a scalar)
// ---------------------------------------------------------------------------

static bool cpu_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event   e0;
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    const int64_t n = ggml_nelements(dst);

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        for (int64_t i = 0; i < n; i++) {
            dst_data[i] = src_data[i] * scale;
        }
    } else if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() {
                for (int64_t i = 0; i < n; i++) {
                    dst_data[i] = src_data[i] * scale;
                }
            });
        });
    } else {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, nullptr, cpu_q);

        sycl::event cpu_evt = cpu_submit_async(cpu_q, deps, [&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(n)),
                             [=](sycl::id<1> i) { dst_data[i] = src_data[i] * scale; });
        });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CPY / CONT  (copy or contiguify tensor data)
// ---------------------------------------------------------------------------

static bool cpu_cpy(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }

    // Simple case: both contiguous, same type → memcpy
    if (src0->type != dst->type) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event  e0;
    const void * src_data = get_host_ptr(src0, device, 0, gpu_q, &e0);

    bool   retained_output;
    void * dst_data = get_retained_or_staging_output(dst, device, gpu_q, &retained_output);

    if (!src_data || !dst_data) {
        return false;
    }

    const size_t nbytes = ggml_nbytes(dst);

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        memcpy(dst_data, src_data, nbytes);
    } else if (host_task) {
        // host_task on gpu_q: in-order queue ensures prior GPU writes complete.
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() { memcpy(dst_data, src_data, nbytes); });
        });
    } else {
        cpu_wait_chain_event();
        offload_wait_event(e0);

        memcpy(dst_data, src_data, nbytes);

        if (!retained_output) {
            flush_output(dst, device, gpu_q);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// ROPE  (rotary positional embeddings — NEOX and NORMAL modes, F32 only)
// ---------------------------------------------------------------------------

static bool cpu_rope(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const bool batched   = batched_mode_active();
    const bool host_task = !batched && host_task_mode_active();

    sycl::queue * cpu_q = (host_task || batched) ? nullptr : ggml_sycl_get_cpu_queue();
    if (!host_task && !batched && !cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];  // input tensor
    const ggml_tensor * src1 = dst->src[1];  // positions (int32)
    const ggml_tensor * src2 = dst->src[2];  // freq_factors (optional)

    if (!src0 || !src1) {
        return false;
    }
    // F32 only for CPU path
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Extract op_params
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));

    // Only support NORMAL and NEOX modes on CPU
    const bool is_neox   = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    const bool is_normal = (mode == GGML_ROPE_TYPE_NORMAL);
    if (!is_neox && !is_normal) {
        return false;  // MROPE, VISION, IMROPE not supported
    }

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    sycl::event     e0, e1;
    const float *   src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));
    const int32_t * pos_data = static_cast<const int32_t *>(get_host_ptr(src1, device, 1, gpu_q, &e1));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !pos_data || !dst_data) {
        return false;
    }

    // Freq factors (optional src2) — must be host-accessible (no staging slot available)
    const float * freq_factors_data = nullptr;
    if (src2) {
        void * src2_ptr = ggml_sycl_get_data_ptr(src2, device);
        if (!src2_ptr) {
            return false;
        }
        if (src2->buffer && !ggml_backend_buffer_is_host(src2->buffer)) {
            return false;
        }
        freq_factors_data = static_cast<const float *>(src2_ptr);
    }

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const int64_t s01 = src0->nb[1] / sizeof(float);
    const int64_t s02 = src0->nb[2] / sizeof(float);
    const int64_t s03 = src0->nb[3] / sizeof(float);
    const int64_t d01 = dst->nb[1] / sizeof(float);
    const int64_t d02 = dst->nb[2] / sizeof(float);
    const int64_t d03 = dst->nb[3] / sizeof(float);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    const int64_t total_rows = ne3 * ne2 * ne1;

    const float cd0 = corr_dims[0];
    const float cd1 = corr_dims[1];

    // RoPE row kernel — shared between host_task and parallel_for paths.
    // Uses generic math functions passed as arguments.
    auto rope_row = [](const float * src_row, float * dst_row, int32_t p, int n_dims, int64_t ne0, bool is_normal,
                       float freq_scale, float ext_factor, float attn_factor, float theta_scale,
                       const float * freq_factors_data, float cd0, float cd1, auto cos_fn, auto sin_fn,
                       auto fmax_fn, auto fmin_fn, auto log_fn) {
        float theta = static_cast<float>(p);
        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
            const float ff            = freq_factors_data ? freq_factors_data[i0 / 2] : 1.0f;
            const float theta_extrap  = theta / ff;
            float       theta_interp  = freq_scale * theta_extrap;
            float       theta_val     = theta_interp;
            float       mscale        = attn_factor;

            if (ext_factor != 0.0f) {
                const float y        = (i0 / 2.0f - cd0) / fmax_fn(0.001f, cd1 - cd0);
                const float ramp_mix = (1.0f - fmin_fn(1.0f, fmax_fn(0.0f, y))) * ext_factor;
                theta_val            = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * log_fn(1.0f / freq_scale);
            }

            const float cos_theta = cos_fn(theta_val) * mscale;
            const float sin_theta = sin_fn(theta_val) * mscale;

            if (is_normal) {
                const float x0  = src_row[i0];
                const float x1  = src_row[i0 + 1];
                dst_row[i0]     = x0 * cos_theta - x1 * sin_theta;
                dst_row[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
            } else {
                const int64_t ic         = i0 / 2;
                const float   x0         = src_row[ic];
                const float   x1         = src_row[ic + n_dims / 2];
                dst_row[ic]              = x0 * cos_theta - x1 * sin_theta;
                dst_row[ic + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            }

            theta *= theta_scale;
        }

        for (int64_t i0 = n_dims; i0 < ne0; i0++) {
            dst_row[i0] = src_row[i0];
        }
    };

    if (batched) {
        // Direct synchronous execution inside batched host_task — no queue submission
        for (int64_t idx = 0; idx < total_rows; idx++) {
            const int64_t i3 = idx / (ne2 * ne1);
            const int64_t i2 = (idx / ne1) % ne2;
            const int64_t i1 = idx % ne1;

            const float * src_row = src_data + i3 * s03 + i2 * s02 + i1 * s01;
            float *       dst_row = dst_data + i3 * d03 + i2 * d02 + i1 * d01;

            rope_row(
                src_row, dst_row, pos_data[i2], n_dims, ne0, is_normal, freq_scale, ext_factor, attn_factor,
                theta_scale, freq_factors_data, cd0, cd1, [](float x) { return std::cos(x); },
                [](float x) { return std::sin(x); }, [](float a, float b) { return std::fmax(a, b); },
                [](float a, float b) { return std::fmin(a, b); }, [](float x) { return std::log(x); });
        }
    } else if (host_task) {
        gpu_q->submit([&](sycl::handler & cgh) {
            cgh.host_task([=]() {
                for (int64_t idx = 0; idx < total_rows; idx++) {
                    const int64_t i3 = idx / (ne2 * ne1);
                    const int64_t i2 = (idx / ne1) % ne2;
                    const int64_t i1 = idx % ne1;

                    const float * src_row = src_data + i3 * s03 + i2 * s02 + i1 * s01;
                    float *       dst_row = dst_data + i3 * d03 + i2 * d02 + i1 * d01;

                    rope_row(
                        src_row, dst_row, pos_data[i2], n_dims, ne0, is_normal, freq_scale, ext_factor, attn_factor,
                        theta_scale, freq_factors_data, cd0, cd1, [](float x) { return std::cos(x); },
                        [](float x) { return std::sin(x); }, [](float a, float b) { return std::fmax(a, b); },
                        [](float a, float b) { return std::fmin(a, b); }, [](float x) { return std::log(x); });
                }
            });
        });
    } else {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, &e1, cpu_q);

        sycl::event cpu_evt = cpu_submit_async(cpu_q, deps, [&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_rows)), [=](sycl::id<1> work_id) {
                const int64_t idx = static_cast<int64_t>(work_id[0]);
                const int64_t i3  = idx / (ne2 * ne1);
                const int64_t i2  = (idx / ne1) % ne2;
                const int64_t i1  = idx % ne1;

                const float * src_row = src_data + i3 * s03 + i2 * s02 + i1 * s01;
                float *       dst_row = dst_data + i3 * d03 + i2 * d02 + i1 * d01;

                rope_row(
                    src_row, dst_row, pos_data[i2], n_dims, ne0, is_normal, freq_scale, ext_factor, attn_factor,
                    theta_scale, freq_factors_data, cd0, cd1, [](float x) { return sycl::cos(x); },
                    [](float x) { return sycl::sin(x); }, [](float a, float b) { return sycl::fmax(a, b); },
                    [](float a, float b) { return sycl::fmin(a, b); }, [](float x) { return sycl::log(x); });
            });
        });

        if (!retained_output) {
            flush_output(dst, device, gpu_q, &cpu_evt);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// UNARY dispatch (SILU and others)
// ---------------------------------------------------------------------------

static bool cpu_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const enum ggml_unary_op op = ggml_get_unary_op(dst);
    switch (op) {
        case GGML_UNARY_OP_SILU:
            return cpu_silu(ctx, dst);
        default:
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported unary op %d on CPU\n", static_cast<int>(op));
            return false;
    }
}

// ---------------------------------------------------------------------------
// Fused RMS_NORM + MUL  (single staging pass, saves 1 flush + 1 stage-in)
// ---------------------------------------------------------------------------
//
// Pattern: rms_dst = RMS_NORM(x), mul_dst = MUL(rms_dst, w)
// Fused:   mul_dst[j] = x[j] * rms_scale * w[j]  (no intermediate flush)
//
// Saves 2 staging transfers per fusion (2x per transformer layer):
//   - RMS_NORM output flush to device
//   - MUL input re-stage of that same data

bool ggml_sycl_compute_fused_rms_norm_mul(ggml_backend_sycl_context & ctx,
                                          ggml_tensor *               rms_dst,
                                          ggml_tensor *               mul_dst) {
    const ggml_tensor * rms_src0 = rms_dst->src[0];  // input to normalize
    const ggml_tensor * mul_src1 = mul_dst->src[1];  // element-wise weight

    if (!rms_src0 || !mul_src1) {
        return false;
    }
    if (rms_src0->type != GGML_TYPE_F32 || rms_dst->type != GGML_TYPE_F32 || mul_src1->type != GGML_TYPE_F32 ||
        mul_dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(rms_src0) || !ggml_is_contiguous(mul_dst)) {
        return false;
    }

    float eps;
    memcpy(&eps, rms_dst->op_params, sizeof(float));

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    // Stage: rms input (slot 0) + mul weight (slot 1) + mul output (slot 2)
    sycl::event   e0, e1;
    const float * rms_in_data = static_cast<const float *>(get_host_ptr(rms_src0, device, 0, gpu_q, &e0));
    const float * mul_wt_data = static_cast<const float *>(get_host_ptr(mul_src1, device, 1, gpu_q, &e1));

    bool    retained_output;
    float * out_data = static_cast<float *>(get_retained_or_staging_output(mul_dst, device, gpu_q, &retained_output));

    if (!rms_in_data || !mul_wt_data || !out_data) {
        return false;
    }

    const int64_t ne00  = rms_src0->ne[0];
    const int64_t nrows = ggml_nrows(rms_src0);

    // mul_src1 dimensions for broadcasting
    const int64_t ne10 = mul_src1->ne[0];
    const int64_t ne11 = mul_src1->ne[1];
    const int64_t s11  = mul_src1->nb[1] / sizeof(float);

    auto run_fused = [=]() {
        for (int64_t row = 0; row < nrows; row++) {
            const float * src_row = rms_in_data + row * ne00;
            float *       dst_row = out_data + row * ne00;

            float sum_sq = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                sum_sq += src_row[j] * src_row[j];
            }
            const float scale = 1.0f / sqrtf(sum_sq / static_cast<float>(ne00) + eps);

            const int64_t wt_row_idx = row % ne11;
            const float * wt_row     = mul_wt_data + wt_row_idx * s11;
            const int64_t nr0        = ne00 / ne10;
            for (int64_t r = 0; r < nr0; r++) {
                for (int64_t j = 0; j < ne10; j++) {
                    dst_row[r * ne10 + j] = src_row[r * ne10 + j] * scale * wt_row[j];
                }
            }
        }
    };

    if (batched_mode_active()) {
        run_fused();
        return true;
    }

    if (ggml_sycl_cpu_offload_async_enabled()) {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, &e1, gpu_q);
        sycl::event              cpu_evt =
            cpu_submit_async(gpu_q, deps, [=](sycl::handler & cgh) { cgh.host_task([=]() { run_fused(); }); });
        if (!retained_output) {
            flush_output(mul_dst, device, gpu_q, &cpu_evt, true);
        }
        return true;
    }

    cpu_wait_chain_event();
    offload_wait_event(e0, offload_wait_reason::FALLBACK);
    offload_wait_event(e1, offload_wait_reason::FALLBACK);
    run_fused();
    if (!retained_output) {
        flush_output(mul_dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fused ADD + RMS_NORM  (single staging pass, saves 1 stage-in)
// ---------------------------------------------------------------------------
//
// Pattern: add_dst = ADD(a, b), rms_dst = RMS_NORM(add_dst)
// Both outputs needed: add_dst is the residual (consumed downstream),
// rms_dst is the normalized input to attention/FFN.
//
// Saves 1 staging transfer per fusion (1x per transformer layer):
//   - RMS_NORM re-staging of add_dst from device

bool ggml_sycl_compute_fused_add_rms_norm(ggml_backend_sycl_context & ctx,
                                          ggml_tensor *               add_dst,
                                          ggml_tensor *               rms_dst) {
    const ggml_tensor * add_src0 = add_dst->src[0];
    const ggml_tensor * add_src1 = add_dst->src[1];
    const ggml_tensor * rms_src0 = rms_dst->src[0];  // should == add_dst

    if (!add_src0 || !add_src1 || rms_src0 != add_dst) {
        return false;
    }
    if (add_src0->type != GGML_TYPE_F32 || add_src1->type != GGML_TYPE_F32 || add_dst->type != GGML_TYPE_F32 ||
        rms_dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(add_src0) || !ggml_is_contiguous(add_dst) || !ggml_is_contiguous(rms_dst)) {
        return false;
    }
    if (add_src1->nb[0] != sizeof(float)) {
        return false;
    }

    float eps;
    memcpy(&eps, rms_dst->op_params, sizeof(float));

    const int     device = ctx.device;
    sycl::queue * gpu_q  = ctx.stream();

    staging_begin_op();

    // Stage add inputs (slots 0, 1) + add output (slot 2)
    sycl::event   e0, e1;
    const float * a_data = static_cast<const float *>(get_host_ptr(add_src0, device, 0, gpu_q, &e0));
    const float * b_data = static_cast<const float *>(get_host_ptr(add_src1, device, 1, gpu_q, &e1));

    // Output: use retained scratch for both add_dst and rms_dst if active
    bool    retained_add;
    float * add_out = static_cast<float *>(get_retained_or_staging_output(add_dst, device, gpu_q, &retained_add));

    if (!a_data || !b_data || !add_out) {
        return false;
    }

    const int64_t ne00 = add_src0->ne[0];
    const int64_t ne01 = add_src0->ne[1];
    const int64_t ne02 = add_src0->ne[2];
    const int64_t ne03 = add_src0->ne[3];
    const int64_t ne10 = add_src1->ne[0];
    const int64_t ne11 = add_src1->ne[1];
    const int64_t ne12 = add_src1->ne[2];
    const int64_t ne13 = add_src1->ne[3];
    const int64_t s11  = add_src1->nb[1] / sizeof(float);
    const int64_t s12  = add_src1->nb[2] / sizeof(float);
    const int64_t s13  = add_src1->nb[3] / sizeof(float);
    const int64_t nr0  = ne00 / ne10;

    // rms output: use retained scratch if active, else staging slot 0.
    // Cannot use the helper here because slot 2 may already be used for add_dst.
    const size_t rms_nbytes = ggml_nbytes(rms_dst);
    float *      rms_out;
    bool         retained_rms = false;
    // Batched mode: write directly to rms_dst->data (host-pinned USM).
    // The batched path returns early before flush logic, so staging would
    // leave rms_dst->data stale — subsequent ops in the batch would read
    // wrong values.
    if (g_batched_mode && rms_dst->data && is_host_accessible_usm(rms_dst->data, device)) {
        rms_out = static_cast<float *>(rms_dst->data);
    } else if (g_retained_active) {
        rms_out      = static_cast<float *>(ggml_sycl_cpu_retained_alloc_output(rms_dst));
        retained_rms = (rms_out != nullptr);
        if (!retained_rms) {
            rms_out = static_cast<float *>(staging_ensure(g_staging_bank, 0, rms_nbytes, gpu_q));
        }
    } else {
        rms_out = static_cast<float *>(staging_ensure(g_staging_bank, 0, rms_nbytes, gpu_q));
    }
    if (!rms_out) {
        return false;
    }

    const int64_t total_rows = ne01 * ne02 * ne03;

    auto run_fused = [=]() {
        for (int64_t ir = 0; ir < total_rows; ir++) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            const float * a_row   = a_data + ir * ne00;
            const float * b_row   = b_data + i13 * s13 + i12 * s12 + i11 * s11;
            float *       add_row = add_out + ir * ne00;
            float *       rms_row = rms_out + ir * ne00;

            float sum_sq = 0.0f;
            for (int64_t r = 0; r < nr0; r++) {
                for (int64_t j = 0; j < ne10; j++) {
                    const float v         = a_row[r * ne10 + j] + b_row[j];
                    add_row[r * ne10 + j] = v;
                    sum_sq += v * v;
                }
            }

            const float scale = 1.0f / sqrtf(sum_sq / static_cast<float>(ne00) + eps);
            for (int64_t j = 0; j < ne00; j++) {
                rms_row[j] = add_row[j] * scale;
            }
        }
    };

    if (batched_mode_active()) {
        run_fused();
        return true;
    }

    const bool  async_mode = ggml_sycl_cpu_offload_async_enabled();
    sycl::event cpu_evt{};
    bool        has_cpu_evt = false;
    if (async_mode) {
        std::vector<sycl::event> deps = cpu_collect_deps(&e0, &e1, gpu_q);
        cpu_evt = cpu_submit_async(gpu_q, deps, [=](sycl::handler & cgh) { cgh.host_task([=]() { run_fused(); }); });
        has_cpu_evt = true;
    } else {
        cpu_wait_chain_event();
        offload_wait_event(e0, offload_wait_reason::FALLBACK);
        offload_wait_event(e1, offload_wait_reason::FALLBACK);
        run_fused();
    }

    if (!retained_add) {
        if (has_cpu_evt) {
            flush_output(add_dst, device, gpu_q, &cpu_evt, true);
        } else {
            flush_output(add_dst, device, gpu_q);
        }
    }

    if (!retained_rms) {
        if (!rms_dst->buffer || ggml_backend_buffer_is_host(rms_dst->buffer)) {
            if (has_cpu_evt) {
                offload_wait_event(cpu_evt, offload_wait_reason::FALLBACK);
            }
            memcpy(rms_dst->data, rms_out, rms_nbytes);
        } else {
            void * rms_dev_ptr = ggml_sycl_get_data_ptr(rms_dst, device);
            if (rms_dev_ptr) {
                ggml_sycl::offload_stats_note_transfer(true, rms_nbytes);
                std::vector<sycl::event> deps;
                deps.reserve(2);
                if (has_cpu_evt) {
                    append_dependency(deps, cpu_evt);
                }
                if (g_staging_flush_pending[g_staging_bank]) {
                    append_dependency(deps, g_staging_flush_evt[g_staging_bank]);
                }
                if (deps.empty()) {
                    g_staging_flush_evt[g_staging_bank] = gpu_q->memcpy(rms_dev_ptr, rms_out, rms_nbytes);
                } else {
                    g_staging_flush_evt[g_staging_bank] = gpu_q->submit([&](sycl::handler & cgh) {
                        cgh.depends_on(deps);
                        cgh.memcpy(rms_dev_ptr, rms_out, rms_nbytes);
                    });
                }
                g_staging_flush_pending[g_staging_bank] = true;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Main dispatch entry point
// ---------------------------------------------------------------------------

bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) {
    GGML_SYCL_DEBUG("[CPU-FWD] op=%s name=%s\n", ggml_op_name(dst->op), dst->name ? dst->name : "(null)");
    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            return cpu_mul_mat(ctx, dst);
        case GGML_OP_RMS_NORM:
            return cpu_rms_norm(ctx, dst);
        case GGML_OP_ADD:
            return cpu_add(ctx, dst);
        case GGML_OP_MUL:
            return cpu_mul(ctx, dst);
        case GGML_OP_UNARY:
            return cpu_unary(ctx, dst);
        case GGML_OP_GLU:
            return cpu_glu(ctx, dst);
        case GGML_OP_SOFT_MAX:
            return cpu_soft_max(ctx, dst);
        case GGML_OP_NORM:
            return cpu_norm(ctx, dst);
        case GGML_OP_SCALE:
            return cpu_scale(ctx, dst);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return cpu_cpy(ctx, dst);
        case GGML_OP_ROPE:
            return cpu_rope(ctx, dst);
        default:
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported op %s on CPU, falling back to GPU\n", ggml_op_name(dst->op));
            return false;
    }
}
