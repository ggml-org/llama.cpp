//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "common.hpp"

#include "ccl-comm.hpp"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "unified-cache.hpp"

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#endif

static bool ggml_sycl_layout_ptr_stats_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_SYCL_LAYOUT_PTR_DEBUG");
        return env && std::string(env) == "1";
    }();
    return enabled;
}

static std::atomic<uint64_t> g_layout_ptr_host_cache_target_hit{0};
static std::atomic<uint64_t> g_layout_ptr_host_cache_aos_hit{0};
static std::atomic<uint64_t> g_layout_ptr_host_cache_layout_fallback{0};
static std::atomic<uint64_t> g_layout_ptr_host_cache_data_fallback{0};
static std::atomic<uint64_t> g_layout_ptr_host_cache_miss{0};

static std::mutex                                   g_sycl_host_alloc_mutex;
static std::unordered_map<void *, size_t>           g_sycl_host_alloc_sizes;

static std::mutex                                   g_opt_feature_registry_mutex;
static std::unordered_set<const optimize_feature *> g_opt_feature_registry;

bool ggml_sycl_is_optimize_feature_live(const optimize_feature * feature) {
    if (!feature) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_opt_feature_registry_mutex);
    return g_opt_feature_registry.find(feature) != g_opt_feature_registry.end();
}

void ggml_sycl_register_optimize_feature(optimize_feature * feature) {
    if (!feature) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_opt_feature_registry_mutex);
    g_opt_feature_registry.insert(feature);
}

void ggml_sycl_unregister_optimize_feature(optimize_feature * feature) {
    if (!feature) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_opt_feature_registry_mutex);
    g_opt_feature_registry.erase(feature);
}

void ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event event) {
    if (!ggml_sycl_layout_ptr_stats_enabled()) {
        return;
    }

    switch (event) {
        case ggml_sycl_layout_ptr_event::HOST_CACHE_TARGET_HIT:
            g_layout_ptr_host_cache_target_hit.fetch_add(1, std::memory_order_relaxed);
            break;
        case ggml_sycl_layout_ptr_event::HOST_CACHE_AOS_HIT:
            g_layout_ptr_host_cache_aos_hit.fetch_add(1, std::memory_order_relaxed);
            break;
        case ggml_sycl_layout_ptr_event::HOST_CACHE_LAYOUT_FALLBACK:
            g_layout_ptr_host_cache_layout_fallback.fetch_add(1, std::memory_order_relaxed);
            break;
        case ggml_sycl_layout_ptr_event::HOST_CACHE_DATA_FALLBACK:
            g_layout_ptr_host_cache_data_fallback.fetch_add(1, std::memory_order_relaxed);
            break;
        case ggml_sycl_layout_ptr_event::HOST_CACHE_MISS:
            g_layout_ptr_host_cache_miss.fetch_add(1, std::memory_order_relaxed);
            break;
    }
}

void ggml_sycl_layout_ptr_stats_dump() {
    if (!ggml_sycl_layout_ptr_stats_enabled()) {
        return;
    }

    fprintf(stderr,
            "[LAYOUT-PTR] host_cached target_hit=%llu aos_hit=%llu layout_fallback=%llu data_fallback=%llu miss=%llu\n",
            static_cast<unsigned long long>(g_layout_ptr_host_cache_target_hit.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_layout_ptr_host_cache_aos_hit.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_layout_ptr_host_cache_layout_fallback.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_layout_ptr_host_cache_data_fallback.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_layout_ptr_host_cache_miss.load(std::memory_order_relaxed)));
}

int get_current_device_id() {
    return dpct::dev_mgr::instance().current_device_id();
}

// Cached shared context and queues for TP mode
static sycl::context * g_tp_shared_context                       = nullptr;
static sycl::queue *   g_tp_shared_queues[GGML_SYCL_MAX_DEVICES] = { nullptr };
static std::mutex      g_tp_context_mutex;

// Initialize shared context and queues for TP mode
static void ggml_sycl_init_tp_shared_context() {
    if (g_tp_shared_context != nullptr) {
        return;  // Already initialized
    }
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return;
    }

    // In multi-process mode: only 1 device is locally visible per process
    int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;

    std::vector<sycl::device> tp_devices;
    for (int i = 0; i < num_local_devices; i++) {
        int          dev_id = g_sycl_tp_config.devices[i];
        sycl::device dev    = ggml_sycl_get_device(dev_id);
        tp_devices.push_back(dev);
    }
    g_tp_shared_context = new sycl::context(tp_devices);
    GGML_SYCL_DEBUG("SYCL TP: Created shared context for %d local devices (world_size=%d)\n", num_local_devices,
                    g_sycl_tp_config.world_size);

    // Create shared-context queues for each local TP device
    for (int i = 0; i < num_local_devices; i++) {
        int          dev_id        = g_sycl_tp_config.devices[i];
        sycl::device dev           = ggml_sycl_get_device(dev_id);
        g_tp_shared_queues[dev_id] = new sycl::queue(*g_tp_shared_context, dev, sycl::property::queue::in_order());
        GGML_SYCL_DEBUG("SYCL TP: Created shared-context queue for device %d at %p\n", dev_id,
                        (void *) g_tp_shared_queues[dev_id]);
    }
}

// Get shared context for TP mode
// Returns nullptr if not 
sycl::context * ggml_sycl_get_tp_context() {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_tp_context_mutex);

    // Initialize shared context if needed
    if (g_tp_shared_context == nullptr) {
        ggml_sycl_init_tp_shared_context();
    }

    return g_tp_shared_context;
}

// Get shared-context queue for a device 
// Returns nullptr if not  or device not part of TP
sycl::queue * ggml_sycl_get_tp_queue(int device) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_tp_context_mutex);

    // Initialize shared context if needed
    if (g_tp_shared_context == nullptr) {
        ggml_sycl_init_tp_shared_context();
    }

    // Check if device is part of local TP devices
    // In multi-process mode: only 1 device is locally visible
    int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
    for (int i = 0; i < num_local_devices; i++) {
        if (g_sycl_tp_config.devices[i] == device) {
            return g_tp_shared_queues[device];
        }
    }
    return nullptr;
}

// =============================================================================
// Device mapping helpers (logical -> actual dpct device id)
// =============================================================================

static std::array<int, GGML_SYCL_MAX_DEVICES> g_sycl_device_map = {};
static int g_sycl_device_map_count = -1;  // -1 = identity mapping

int ggml_sycl_map_device_id(int device) {
    if (g_sycl_device_map_count <= 0) {
        return device;
    }
    if (device < 0 || device >= g_sycl_device_map_count) {
        return device;
    }
    return g_sycl_device_map[device];
}

void ggml_sycl_set_device_map(const int * device_ids, int device_count) {
    if (!device_ids || device_count <= 0) {
        g_sycl_device_map_count = -1;
        return;
    }
    const int count = std::min(device_count, GGML_SYCL_MAX_DEVICES);
    for (int i = 0; i < count; ++i) {
        g_sycl_device_map[i] = device_ids[i];
    }
    g_sycl_device_map_count = count;
}

// ============================================================================
// TP Staging Cache: Stages mmap'd tensor data to device memory per-device
// Since Intel Arc lacks P2P, we duplicate to each device's local memory
// ============================================================================
#include <unordered_map>

struct StagedBuffer {
    void * ptrs[GGML_SYCL_MAX_DEVICES];  // Per-device pointers
    size_t size;
};

static std::unordered_map<const void *, StagedBuffer> g_tp_staging_cache;
static std::mutex                                     g_tp_staging_mutex;

// Runtime staging cache for single-GPU mode (non-weight data like positions, masks)
struct RuntimeStagedData {
    void * ptr;   // Pinned memory pointer
    size_t size;
};
static std::unordered_map<const void *, RuntimeStagedData> g_runtime_staging_cache;
static std::mutex                                          g_runtime_staging_mutex;

// Get or create a staged copy of mmap'd data for a specific device 
// Works in both TP mode and single-GPU mode (via host cache pinned memory)
void * ggml_sycl_get_staged_ptr_device(const void * src, size_t size, int device) {
    if (src == nullptr || size == 0) {
        return nullptr;
    }

    // Single-GPU mode: Stage through host cache's pinned memory pool
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        if (ggml_sycl::unified_cache_enabled()) {
            std::lock_guard<std::mutex> lock(g_runtime_staging_mutex);
            auto it = g_runtime_staging_cache.find(src);
            if (it != g_runtime_staging_cache.end() && it->second.size >= size) {
                return it->second.ptr;
            }
            if (auto * host = ggml_sycl::get_host_cache_for_device(device)) {
                void * pinned = host->allocate_pinned_runtime(size, 64);
                if (pinned) {
                    std::memcpy(pinned, src, size);
                    g_runtime_staging_cache[src] = { pinned, size };
                    return pinned;
                }
            }
        }
        return nullptr;
    }
    // Multi-process mode: No cross-device staging needed (each process has its own data)
    if (g_sycl_tp_config.is_multiprocess) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_tp_staging_mutex);

    // Check if already staged for this device
    auto it = g_tp_staging_cache.find(src);
    if (it != g_tp_staging_cache.end()) {
        if (it->second.size >= size && it->second.ptrs[device] != nullptr) {
            return it->second.ptrs[device];
        }
        // Size mismatch or device not staged - need to re-allocate
        if (it->second.size < size) {
            GGML_SYCL_DEBUG("[STAGING] Size mismatch for %p: cached=%zu, requested=%zu, reallocating\n", src,
                            it->second.size, size);
            // Free all staged pointers (only local devices - already guarded by multiprocess check above)
            int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
            for (int i = 0; i < num_local_devices; i++) {
                int dev_id = g_sycl_tp_config.devices[i];
                if (it->second.ptrs[dev_id] != nullptr && g_tp_shared_queues[dev_id] != nullptr) {
                    sycl::free(it->second.ptrs[dev_id], *g_tp_shared_queues[dev_id]);
                }
            }
            g_tp_staging_cache.erase(it);
            it = g_tp_staging_cache.end();
        }
    }

    // Ensure shared context is initialized
    {
        std::lock_guard<std::mutex> ctx_lock(g_tp_context_mutex);
        if (g_tp_shared_context == nullptr) {
            ggml_sycl_init_tp_shared_context();
        }
    }

    // Get or create staging entry
    StagedBuffer * entry;
    if (it == g_tp_staging_cache.end()) {
        g_tp_staging_cache[src] = {};
        entry                   = &g_tp_staging_cache[src];
        entry->size             = size;
        for (int i = 0; i < GGML_SYCL_MAX_DEVICES; i++) {
            entry->ptrs[i] = nullptr;
        }
    } else {
        entry = &it->second;
    }

    // Allocate on the requested device if not already done
    if (entry->ptrs[device] == nullptr) {
        sycl::queue * tp_queue = g_tp_shared_queues[device];
        if (tp_queue == nullptr) {
            GGML_LOG_ERROR("[STAGING] TP queue not initialized for device %d\n", device);
            return nullptr;
        }

        try {
            GGML_SYCL_DEBUG("[STAGING] Allocating %zu bytes on device %d using tp_queue=%p...\n", size, device,
                            (void *) tp_queue);
            // Allocate device memory on this specific device using shared-context queue
            ggml_sycl::unified_cache_add_runtime_bytes(device, size);
            void * staged = ggml_sycl_malloc_device(size, *tp_queue, "tp_staging_device");
            if (staged == nullptr) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device, size);
                GGML_LOG_ERROR("[STAGING] malloc_device returned nullptr for %zu bytes on device %d\n", size, device);
                return nullptr;
            }
            GGML_SYCL_DEBUG("[STAGING] Allocated at %p, copying from %p via host staging...\n", staged, src);

            // Two-stage copy: mmap'd -> host (pinned) -> device
            // The shared-context queue cannot access non-USM mmap'd memory directly
            // Use pinned host memory for staging
            void * host_staging = ggml_sycl_host_malloc(size);
            if (host_staging == nullptr) {
                GGML_LOG_ERROR("[STAGING] failed to allocate pinned host staging buffer for %zu bytes\n", size);
                ggml_sycl::unified_cache_sub_runtime_bytes(device, size);
                sycl::free(staged, *tp_queue);
                return nullptr;
            }
            GGML_SYCL_DEBUG("[STAGING] Host buffer at %p, doing std::memcpy...\n", host_staging);

            // Copy from mmap'd memory to heap (standard memcpy)
            std::memcpy(host_staging, src, size);
            GGML_SYCL_DEBUG("[STAGING] memcpy done, submitting SYCL memcpy to device...\n");

            // Copy from heap to device using shared-context queue
            // First wait for any pending operations on this queue
            tp_queue->wait();
            GGML_SYCL_DEBUG("[STAGING] Queue wait done, now memcpy...\n");
            tp_queue->memcpy(staged, host_staging, size).wait();
            GGML_SYCL_DEBUG("[STAGING] SYCL memcpy done\n");

            // Free the temporary host staging buffer
            ggml_sycl_host_free(host_staging);

            entry->ptrs[device] = staged;
            GGML_SYCL_DEBUG("[STAGING] Staged %zu bytes from %p to device %d: %p\n", size, src, device, staged);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[STAGING] Failed to allocate/copy %zu bytes on device %d: %s (code=%d)\n", size, device,
                           e.what(), static_cast<int>(e.code().value()));
            return nullptr;
        }
    }

    return entry->ptrs[device];
}

// Legacy function for backwards compatibility - returns device 0's staging
void * ggml_sycl_get_staged_ptr(const void * src, size_t size) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return nullptr;
    }
    return ggml_sycl_get_staged_ptr_device(src, size, g_sycl_tp_config.devices[0]);
}

// Clear all staging buffers (call at end of graph compute)
void ggml_sycl_clear_staging_cache() {
    std::lock_guard<std::mutex> lock(g_tp_staging_mutex);

    if (g_tp_staging_cache.empty()) {
        return;
    }

    GGML_SYCL_DEBUG("[STAGING] Clearing %zu staged buffers\n", g_tp_staging_cache.size());

    // Free per-device pointers using their respective queues
    // In multi-process mode: only 1 device is locally accessible, use local device count
    int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
    for (auto & entry : g_tp_staging_cache) {
        for (int i = 0; i < num_local_devices; i++) {
            int dev_id = g_sycl_tp_config.devices[i];
            if (entry.second.ptrs[dev_id] != nullptr && g_tp_shared_queues[dev_id] != nullptr) {
                ggml_sycl::unified_cache_sub_runtime_bytes(dev_id, entry.second.size);
                sycl::free(entry.second.ptrs[dev_id], *g_tp_shared_queues[dev_id]);
                entry.second.ptrs[dev_id] = nullptr;
            }
        }
    }
    g_tp_staging_cache.clear();
}

void * ggml_sycl_host_malloc(size_t size) try {
    if (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
        return nullptr;
    }

    // Check against probed host malloc limit (driver per-allocation limit)
    const size_t host_max_alloc = ggml_sycl_get_host_max_alloc_size();
    if (host_max_alloc > 0 && size > host_max_alloc) {
        GGML_LOG_WARN(
            "[SYCL] Refusing to allocate %.1f GB of pinned memory (driver limit: %.1f GB). "
            "Falling back to CPU memory with unified cache staging.\n",
            size / (1024.0 * 1024.0 * 1024.0), host_max_alloc / (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }

    void * ptr = nullptr;

    // For TP mode: use a shared context so memory is accessible from all devices
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        std::lock_guard<std::mutex> lock(g_tp_context_mutex);

        // Initialize shared context and queues if needed
        if (g_tp_shared_context == nullptr) {
            ggml_sycl_init_tp_shared_context();
        }

        // Allocate host memory accessible from all TP devices
        ggml_sycl::unified_cache_add_runtime_host_bytes(size);
        dpct::err0 err = CHECK_TRY_ERROR(ptr = sycl::malloc_host(size, *g_tp_shared_context));

        if (err != 0) {
            ggml_sycl::unified_cache_sub_runtime_host_bytes(size);
            GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of TP shared host memory\n", size / 1024.0 / 1024.0);
            return nullptr;
        }
        ggml_sycl_alloc_trace_record("host", size, "tp_shared_host");
        {
            std::lock_guard<std::mutex> guard(g_sycl_host_alloc_mutex);
            g_sycl_host_alloc_sizes[ptr] = size;
        }
        return ptr;
    }

    // Non-TP mode: use default queue for host malloc
    ggml_sycl::unified_cache_add_runtime_host_bytes(size);
    dpct::err0 err = CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, dpct::get_in_order_queue()));

    if (err != 0) {
        ggml_sycl::unified_cache_sub_runtime_host_bytes(size);
        // clear the error
        GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,
                       "syclGetErrorString is not supported");
        return nullptr;
    }
    ggml_sycl_alloc_trace_record("host", size, "host_malloc");
    {
        std::lock_guard<std::mutex> guard(g_sycl_host_alloc_mutex);
        g_sycl_host_alloc_sizes[ptr] = size;
    }
    return ptr;
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_host_free(void * ptr) try {
    if (!ptr) {
        return;
    }
    // Prefer shared TP context if it exists (pinned host memory may be bound to it).
    size_t alloc_size = 0;
    {
        std::lock_guard<std::mutex> guard(g_sycl_host_alloc_mutex);
        auto it = g_sycl_host_alloc_sizes.find(ptr);
        if (it != g_sycl_host_alloc_sizes.end()) {
            alloc_size = it->second;
            g_sycl_host_alloc_sizes.erase(it);
        }
    }
    if (alloc_size > 0) {
        ggml_sycl::unified_cache_sub_runtime_host_bytes(alloc_size);
    }
    if (g_tp_shared_context != nullptr) {
        sycl::free(ptr, *g_tp_shared_context);
        return;
    }
    // Fallback to default queue context.
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

bool gpu_has_xmx(sycl::device & dev) {
    return dev.has(sycl::aspect::ext_intel_matrix);
}

XMXCapabilities query_xmx_capabilities(sycl::device & dev) {
    XMXCapabilities caps;

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return caps;
    }
    caps.supported = true;

    // Query SLM size
    caps.slm_size = dev.get_info<sycl::info::device::local_mem_size>();

#if defined(SYCL_EXT_ONEAPI_MATRIX_VERSION) && SYCL_EXT_ONEAPI_MATRIX_VERSION >= 1
    using namespace sycl::ext::oneapi::experimental;

    try {
        auto combinations = dev.get_info<info::device::matrix_combinations>();

        for (const auto & combo : combinations) {
            // Find int8 configuration (for Q8_0)
            if (combo.atype == matrix_type::sint8 && combo.btype == matrix_type::sint8) {
                caps.supports_int8 = true;
                caps.M             = combo.msize;
                caps.N             = combo.nsize;
                caps.K             = combo.ksize;

                GGML_SYCL_DEBUG("[XMX] int8: M=%zu, N=%zu, K=%zu\n", caps.M, caps.N, caps.K);
            }

            if (combo.atype == matrix_type::fp16 && combo.btype == matrix_type::fp16) {
                caps.supports_fp16 = true;
            }
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG("[XMX] Query failed: %s\n", e.what());
    }
#else
    // Fallback: assume Intel Arc defaults
    caps.supports_int8 = true;
    caps.supports_fp16 = true;
    caps.M             = 8;
    caps.N             = 16;
    caps.K             = 32;
    GGML_SYCL_DEBUG("[XMX] Using default config: M=8, N=16, K=32\n");
#endif

    // Compute optimal tile counts based on SLM constraints
    bool slm_calculation_success = false;

    if (caps.M > 0 && caps.N > 0 && caps.K > 0 && caps.slm_size > 0) {
        // Fixed XMX dimensions (Intel Arc default)
        constexpr int XMX_M = 8;
        constexpr int XMX_N = 16;
        constexpr int XMX_K = 32;

        // SLM reservation for LUT (MXFP4->INT8 lookup table)
        constexpr size_t LUT_BYTES = 16;

        // SLM reservation for token tile (M × K × sizeof(int8))
        // Each work-group needs token data for XMX_M rows
        size_t token_tile_bytes = XMX_M * caps.K * sizeof(int8_t);

        // Remaining SLM budget after reserving LUT and tokens
        size_t slm_for_weights = caps.slm_size - LUT_BYTES - token_tile_bytes;

        // Calculate weight tile size for MXFP4 (4-bit quantization)
        // Per tile in N dimension: XMX_N * XMX_K / 2 bytes
        size_t weight_tile_bytes = caps.N * caps.K / 2;

        // Max tiles_n that fit in available SLM (conservative: 50% of remaining budget)
        // This ensures room for accumulation buffers and other SLM usage
        if (weight_tile_bytes > 0 && slm_for_weights > 0) {
            int max_tiles_from_slm  = static_cast<int>((slm_for_weights / 2) / weight_tile_bytes);
            caps.optimal_tiles_n    = std::max(1, std::min(4, max_tiles_from_slm));
            slm_calculation_success = true;

            GGML_LOG_INFO("[XMX] SLM-aware optimal_tiles_n=%d (SLM=%zuKB, weight_tile=%zuB, token_tile=%zuB)\n",
                          caps.optimal_tiles_n, caps.slm_size / 1024, weight_tile_bytes, token_tile_bytes);
        }
    }

    // Fallback to default calculation if SLM info unavailable or calculation failed
    if (!slm_calculation_success && caps.M > 0 && caps.N > 0) {
        caps.optimal_tiles_m = std::min(4, static_cast<int>(32 / caps.M));
        caps.optimal_tiles_n = std::min(4, static_cast<int>(64 / caps.N));
    }

    return caps;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size) {
    const int64_t max_range          = std::numeric_limits<int>::max();
    int64_t       sycl_down_blk_size = block_size;
    int64_t       global_range       = accumulate_block_num * sycl_down_blk_size;
    while (global_range > max_range) {
        sycl_down_blk_size /= 2;
        global_range = accumulate_block_num * sycl_down_blk_size;
    }
    return sycl_down_blk_size;
}

void retain_extra_gpu(ggml_tensor_extra_gpu * extra) {
    if (!extra) {
        return;
    }
    extra->refcount.fetch_add(1, std::memory_order_relaxed);
}

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    if (!extra) {
        return;
    }
    const int prev = extra->refcount.fetch_sub(1, std::memory_order_acq_rel);
    GGML_ASSERT(prev > 0);
    if (prev > 1) {
        return;
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            if (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        if (extra->data_device[i] != nullptr && streams.size() > 0) {
            ggml_sycl_set_device(i);
            if (extra->data_device_size[i] > 0) {
                ggml_sycl::unified_cache_sub_runtime_bytes(i, extra->data_device_size[i]);
            }
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(extra->data_device[i], *(streams[i]))));
            extra->data_device_size[i] = 0;
        }
        // Free XMX MXFP4 tiled cache (legacy - will be migrated to layout system)
        if (extra->xmx_mxfp4_tiled[i] != nullptr && streams.size() > 0) {
            const bool layout_alias = (extra->layout.mode == GGML_LAYOUT_XMX_TILED &&
                                       extra->layout.data_ptr == extra->xmx_mxfp4_tiled[i]);
            if (!layout_alias && extra->xmx_mxfp4_tiled_owned[i]) {
                ggml_sycl_set_device(i);
                if (extra->xmx_mxfp4_tiled_size > 0) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(i, extra->xmx_mxfp4_tiled_size);
                }
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(extra->xmx_mxfp4_tiled[i], *(streams[i]))));
            }
            extra->xmx_mxfp4_tiled[i] = nullptr;
            extra->xmx_mxfp4_tiled_owned[i] = false;
        }
        if (extra->xmx_mxfp4_tiled_aos_staging[i] != nullptr && streams.size() > 0) {
            ggml_sycl_set_device(i);
            if (extra->xmx_mxfp4_tiled_aos_staging_size[i] > 0) {
                ggml_sycl::unified_cache_sub_runtime_bytes(i, extra->xmx_mxfp4_tiled_aos_staging_size[i]);
            }
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(extra->xmx_mxfp4_tiled_aos_staging[i], *(streams[i]))));
            extra->xmx_mxfp4_tiled_aos_staging[i] = nullptr;
            extra->xmx_mxfp4_tiled_aos_staging_size[i] = 0;
        }

        if (extra->moe_expert_ptrs_device[i] != nullptr && streams.size() > 0) {
            ggml_sycl_set_device(i);
            if (extra->moe_expert_ptrs_size[i] > 0) {
                ggml_sycl::unified_cache_sub_runtime_bytes(i, extra->moe_expert_ptrs_size[i]);
            }
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(extra->moe_expert_ptrs_device[i], *(streams[i]))));
            extra->moe_expert_ptrs_device[i] = nullptr;
            extra->moe_expert_ptrs_size[i] = 0;
        }
        extra->moe_expert_ptrs_host[i].clear();

        if (extra->moe_expert_ptrs_compact_device[i] != nullptr && streams.size() > 0) {
            ggml_sycl_set_device(i);
            if (extra->moe_expert_ptrs_compact_capacity[i] > 0) {
                ggml_sycl::unified_cache_sub_runtime_bytes(i, extra->moe_expert_ptrs_compact_capacity[i]);
            }
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(extra->moe_expert_ptrs_compact_device[i], *(streams[i]))));
            extra->moe_expert_ptrs_compact_device[i] = nullptr;
            extra->moe_expert_ptrs_compact_size[i] = 0;
            extra->moe_expert_ptrs_compact_capacity[i] = 0;
        }
        if (extra->moe_expert_ptrs_missing_device[i] != nullptr && streams.size() > 0) {
            ggml_sycl_set_device(i);
            ggml_sycl::unified_cache_sub_runtime_bytes(i, sizeof(int));
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(extra->moe_expert_ptrs_missing_device[i], *(streams[i]))));
            extra->moe_expert_ptrs_missing_device[i] = nullptr;
        }
    }

    // Release unified layout-managed memory (handles SOA, COALESCED, XMX_TILED buffers)
    if (extra->layout.owns_memory && extra->layout.data_ptr && streams.size() > 0) {
        int dev = extra->layout.device_id;
        if (dev >= 0 && dev < static_cast<int>(streams.size())) {
            ggml_sycl_set_device(dev);
            ggml_sycl_release_layout(extra->layout, *(streams[dev]));
        }
    }

    ggml_sycl_unregister_optimize_feature(&extra->optimized_feature);
    delete extra;
}

// ============================================================================
// FFN Norm Cache for Tensor Parallelism
// ============================================================================
// The GGML backend scheduler may reuse the FFN norm buffer before TP can
// access it for device 1's computation. We cache FFN norm output immediately
// after the MUL operation to prevent stale data issues.

std::unordered_map<int, ffn_norm_cache_entry> g_tp_ffn_norm_cache;
std::mutex                                    g_tp_ffn_norm_cache_mutex;
int                                           g_tp_current_pass_id = 0;
bool                                          g_tp_enabled         = false;

void ggml_sycl_tp_cache_ffn_norm(int          layer,
                                 const void * data,
                                 int64_t      ne0,
                                 int64_t      ne1,
                                 size_t       size,
                                 queue_ptr    stream) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return;
    }

    // Multi-process mode: Skip caching - each process has only one device
    // and CCL handles cross-process communication
    if (g_sycl_tp_config.is_multiprocess) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_tp_ffn_norm_cache_mutex);

    auto                   it    = g_tp_ffn_norm_cache.find(layer);
    ffn_norm_cache_entry & entry = g_tp_ffn_norm_cache[layer];

    // Reallocate if size changed
    if (entry.size != size) {
        // DEBUG: Check L31 weight BEFORE any free/realloc
        {
            uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;

            struct {
                uint16_t d_bits;
                uint8_t  qs[16];
            } wblk;

            try {
                stream->memcpy(&wblk, (void *) l31_weight_addr, sizeof(wblk)).wait();
                uint16_t   d_raw = wblk.d_bits;
                sycl::half d_half;
                memcpy(&d_half, &d_raw, sizeof(sycl::half));
                float d_f = static_cast<float>(d_half);
                if (g_ggml_sycl_tp_debug && (d_f > 100.0f || std::isnan(d_f))) {
                    fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: BEFORE realloc, L31 weight CORRUPTED d=%f\n",
                            layer, d_f);
                } else if (g_ggml_sycl_tp_debug) {
                    static int before_dbg = 0;
                    if (before_dbg++ < 5) {
                        fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: BEFORE realloc, L31 weight OK d=%f\n", layer,
                                d_f);
                    }
                }
            } catch (...) {
                // Ignore errors if address is invalid
            }
        }

        if (entry.data != nullptr) {
            if (g_ggml_sycl_tp_debug) {
                fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: freeing old cache at %p\n", layer, entry.data);
            }
            ggml_sycl::unified_cache_sub_runtime_bytes(g_sycl_tp_config.devices[0], size);
            sycl::free(entry.data, *stream);
        }
        if (entry.data_dev1 != nullptr) {
            int dev1 = g_sycl_tp_config.devices[1];
            ggml_sycl_set_device(dev1);
            queue_ptr stream1 = &ggml_sycl_get_device(dev1).default_queue();
            ggml_sycl::unified_cache_sub_runtime_bytes(dev1, size);
            sycl::free(entry.data_dev1, *stream1);
            ggml_sycl_set_device(g_sycl_tp_config.devices[0]);
        }
        // Allocate new buffers
        ggml_sycl::unified_cache_add_runtime_bytes(g_sycl_tp_config.devices[0], size);
        entry.data = ggml_sycl_malloc_device(size, *stream, "tp_device_entry");
        if (!entry.data) {
            ggml_sycl::unified_cache_sub_runtime_bytes(g_sycl_tp_config.devices[0], size);
        }
        int dev1   = g_sycl_tp_config.devices[1];
        ggml_sycl_set_device(dev1);
        queue_ptr stream1 = &ggml_sycl_get_device(dev1).default_queue();
        ggml_sycl::unified_cache_add_runtime_bytes(dev1, size);
        entry.data_dev1   = ggml_sycl_malloc_device(size, *stream1, "tp_device_entry");
        if (!entry.data_dev1) {
            ggml_sycl::unified_cache_sub_runtime_bytes(dev1, size);
        }
        ggml_sycl_set_device(g_sycl_tp_config.devices[0]);
        if (!entry.data || !entry.data_dev1) {
            GGML_LOG_ERROR("SYCL TP: Failed to allocate FFN norm cache buffers for layer %d\n", layer);
            if (entry.data) {
                ggml_sycl::unified_cache_sub_runtime_bytes(g_sycl_tp_config.devices[0], size);
                sycl::free(entry.data, *stream);
                entry.data = nullptr;
            }
            if (entry.data_dev1) {
                ggml_sycl::unified_cache_sub_runtime_bytes(dev1, size);
                sycl::free(entry.data_dev1, *stream1);
                entry.data_dev1 = nullptr;
            }
            ggml_sycl_set_device(g_sycl_tp_config.devices[0]);
            entry.size = 0;
            return;
        }

        // DEBUG: Check if allocation overlaps with L31 FFN gate weight
        if (g_ggml_sycl_tp_debug && (layer == 31 || layer == 0)) {
            uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
            uintptr_t l31_weight_end  = l31_weight_addr + 16515072;  // shard size
            uintptr_t cache_start     = (uintptr_t) entry.data;
            uintptr_t cache_end       = cache_start + size;
            bool      overlap         = (cache_start < l31_weight_end && cache_end > l31_weight_addr);
            fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: alloc=%p size=%zu overlap_with_L31_weight=%d\n", layer,
                    entry.data, size, overlap);
            if (overlap) {
                fprintf(stderr, "TP DEBUG FFN_NORM_CACHE OVERLAP! cache=[%p,%p) weight=[0x%llx,0x%llx)\n",
                        (void *) cache_start, (void *) cache_end, (unsigned long long) l31_weight_addr,
                        (unsigned long long) l31_weight_end);
            }
        }

        // DEBUG: Check L31 weight AFTER realloc
        {
            uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;

            struct {
                uint16_t d_bits;
                uint8_t  qs[16];
            } wblk;

            try {
                stream->memcpy(&wblk, (void *) l31_weight_addr, sizeof(wblk)).wait();
                uint16_t   d_raw = wblk.d_bits;
                sycl::half d_half;
                memcpy(&d_half, &d_raw, sizeof(sycl::half));
                float d_f = static_cast<float>(d_half);
                if (g_ggml_sycl_tp_debug && (d_f > 100.0f || std::isnan(d_f))) {
                    fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: AFTER realloc, L31 weight CORRUPTED d=%f\n",
                            layer, d_f);
                } else if (g_ggml_sycl_tp_debug) {
                    static int after_dbg = 0;
                    if (after_dbg++ < 5) {
                        fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: AFTER realloc, L31 weight OK d=%f\n", layer,
                                d_f);
                    }
                }
            } catch (...) {
                // Ignore errors if address is invalid
            }
        }
    }

    // Copy current FFN norm output to cache
    stream->memcpy(entry.data, data, size).wait();

    // DEBUG: Check L31 weight AFTER cache copy
    if (g_ggml_sycl_tp_debug && (layer == 31 || layer == 0)) {
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;

        struct {
            uint16_t d_bits;
            uint8_t  qs[16];
        } wblk;

        try {
            stream->memcpy(&wblk, (void *) l31_weight_addr, sizeof(wblk)).wait();
            uint16_t   d_raw = wblk.d_bits;
            sycl::half d_half;
            memcpy(&d_half, &d_raw, sizeof(sycl::half));
            float d_f = static_cast<float>(d_half);
            if (d_f > 100.0f || std::isnan(d_f)) {
                fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: AFTER cache copy, L31 weight CORRUPTED d=%f\n",
                        layer, d_f);
            }
        } catch (...) {
        }
    }

    // Also copy to device 1's buffer (via host staging)
    void * host_buf = ggml_sycl_malloc_host_tracked_bytes(size, *stream, "tp_host_buf");
    stream->memcpy(host_buf, data, size).wait();
    int dev1 = g_sycl_tp_config.devices[1];
    ggml_sycl_set_device(dev1);
    queue_ptr stream1 = &ggml_sycl_get_device(dev1).default_queue();
    stream1->memcpy(entry.data_dev1, host_buf, size).wait();
    ggml_sycl_free_host_tracked_bytes(host_buf, size, *stream);
    ggml_sycl_set_device(g_sycl_tp_config.devices[0]);

    entry.ne0     = ne0;
    entry.ne1     = ne1;
    entry.size    = size;
    entry.pass_id = g_tp_current_pass_id;
}

void * ggml_sycl_tp_get_cached_ffn_norm(int layer, int device) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_tp_ffn_norm_cache_mutex);

    auto it = g_tp_ffn_norm_cache.find(layer);
    if (it == g_tp_ffn_norm_cache.end()) {
        return nullptr;
    }

    const ffn_norm_cache_entry & entry = it->second;

    // NOTE: We do NOT check pass_id for staleness anymore.
    // The cache is updated whenever device 0 runs the MUL for ffn_norm.
    // On generation passes, device 1's backend may be called instead of device 0's,
    // but the cached values from device 0's last run are still valid since the
    // FFN norm computation depends only on the current input token embedding,
    // which is correctly handled by the compute graph.
    //
    // The key insight: FFN norm values change per-pass because the input changes.
    // But if device 1 is running the pass, device 0 also runs the same pass
    // (just the SYCL scheduler may call device 1's backend first).
    // So when device 0's backend runs, it will update the cache.
    //
    // For now, we'll return the cached data regardless of pass_id.
    // This may cause issues if device 0's backend doesn't run at all on a pass,
    // but that would be a scheduler bug.

    // Return appropriate buffer for device
    int main_device = g_sycl_tp_config.devices[0];
    if (device == main_device) {
        return entry.data;
    } else {
        return entry.data_dev1;
    }
}

void ggml_sycl_tp_clear_ffn_norm_cache(int layer) {
    std::lock_guard<std::mutex> lock(g_tp_ffn_norm_cache_mutex);

    auto it = g_tp_ffn_norm_cache.find(layer);
    if (it != g_tp_ffn_norm_cache.end()) {
        // Free device memory (caller should ensure correct device is set)
        if (it->second.data != nullptr) {
            // Note: we don't free here to avoid device issues, just clear entry
            // Memory will be reused on next cache
        }
        g_tp_ffn_norm_cache.erase(it);
    }
}

void ggml_sycl_tp_new_pass() {
    g_tp_current_pass_id++;
}

// ============================================================================
// Global TP Config and Function Implementations
// ============================================================================

// Global TP config definition
ggml_sycl_tp_config g_sycl_tp_config = {};

// Shared reduce buffer for ALL_REDUCE
static float *    g_tp_shared_reduce_buf      = nullptr;
static size_t     g_tp_shared_reduce_buf_size = 0;
static std::mutex g_tp_shared_reduce_mutex;

// Persistent host buffers for CPU-based ALL_REDUCE (avoids per-call malloc/free)
static float * g_tp_host_buf0     = nullptr;  // Buffer for device 0 data
static float * g_tp_host_buf1     = nullptr;  // Buffer for device 1 data
static size_t  g_tp_host_buf_size = 0;

void ggml_sycl_tp_init(const int * device_ids, int num_devices) {
    if (num_devices < 1 || num_devices > GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("SYCL TP: Invalid number of devices: %d\n", num_devices);
        return;
    }

    g_sycl_tp_config.enabled    = true;
    g_sycl_tp_config.world_size = num_devices;
    g_sycl_tp_config.rank       = 0;  // Default rank
    for (int i = 0; i < num_devices; i++) {
        g_sycl_tp_config.devices[i] = device_ids[i];
    }

    // Check for multi-process mode (MPI)
    const char * pmi_rank = std::getenv("PMI_RANK");
    const char * pmi_size = std::getenv("PMI_SIZE");
    if (pmi_rank && pmi_size) {
        g_sycl_tp_config.is_multiprocess = true;
        g_sycl_tp_config.mpi_rank        = std::atoi(pmi_rank);
        g_sycl_tp_config.mpi_world_size  = std::atoi(pmi_size);
        g_sycl_tp_config.rank            = g_sycl_tp_config.mpi_rank;
        g_sycl_tp_config.world_size      = g_sycl_tp_config.mpi_world_size;
        GGML_LOG_INFO("SYCL TP: Multi-process mode enabled, rank=%d/%d\n", g_sycl_tp_config.mpi_rank,
                      g_sycl_tp_config.mpi_world_size);

        // Initialize oneCCL for multi-process communication
        // In multi-process mode, each process has only ONE device visible (level_zero:$RANK)
        // Get the queue for device 0 (the only locally visible device)
        int       local_device = device_ids[0];
        queue_ptr local_queue  = &(ggml_sycl_get_device(local_device).default_queue());
        ggml_sycl_ccl_init_multiprocess(g_sycl_tp_config.mpi_rank, g_sycl_tp_config.mpi_world_size, local_queue);
    }

    GGML_SYCL_DEBUG("SYCL TP: Initialized with %d devices\n", num_devices);

    // Pre-allocate quantized comm buffers if Flash Communication is enabled
    if (ggml_sycl_quant_allreduce_enabled()) {
        // Default initial size: 16M elements (covers most FFN/attention outputs)
        // This is ~16MB for INT8 buffers, ~64MB for FP32 result buffer
        ggml_sycl_tp_init_quant_comm_buffers(16 * 1024 * 1024);
    }
}

// =============================================================================
// Persistent FFN Compute Buffers for TP Mode
// =============================================================================

std::unordered_map<int, tp_ffn_compute_buffers> g_tp_ffn_buffers;
std::mutex                                      g_tp_ffn_buffers_mutex;

tp_host_staging_buffer g_tp_host_staging = { nullptr, 0, 0 };
std::mutex             g_tp_host_staging_mutex;

tp_ffn_compute_buffers * ggml_sycl_tp_ensure_ffn_buffers(int       layer,
                                                         int       device,
                                                         queue_ptr stream,
                                                         int64_t   K_full_padded,
                                                         int64_t   N_hidden_shard_padded,
                                                         int64_t   batch,
                                                         int64_t   N_out) {
    std::lock_guard<std::mutex> lock(g_tp_ffn_buffers_mutex);

    auto                     it   = g_tp_ffn_buffers.find(layer);
    tp_ffn_compute_buffers * bufs = nullptr;

    if (it == g_tp_ffn_buffers.end()) {
        // Initialize empty entry
        g_tp_ffn_buffers[layer] = {};
        bufs                    = &g_tp_ffn_buffers[layer];
    } else {
        bufs = &it->second;
    }

    // Check if we need to allocate or resize
    bool needs_alloc  = !bufs->allocated;
    bool needs_resize = bufs->allocated &&
                        (K_full_padded > bufs->K_full_padded || N_hidden_shard_padded > bufs->N_hidden_shard_padded ||
                         batch > bufs->batch_max || N_out > bufs->N_out);

    if (!needs_alloc && !needs_resize) {
        return bufs;  // Existing buffers are sufficient
    }

    // Free old buffers if resizing
    if (needs_resize && bufs->allocated) {
        const int runtime_device = bufs->device_id >= 0 ? bufs->device_id : device;
        if (bufs->input_q8_dev) {
            ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs->input_q8_size);
        }
        if (bufs->gate_out) {
            ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs->hidden_size);
        }
        if (bufs->up_out) {
            ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs->hidden_size);
        }
        if (bufs->hidden_out) {
            ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs->hidden_size);
        }
        if (bufs->hidden_q8_dev) {
            ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs->hidden_q8_size);
        }
        if (bufs->partial_out) {
            ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs->partial_size);
        }
        if (bufs->input_q8_dev) {
            sycl::free(bufs->input_q8_dev, *stream);
        }
        if (bufs->gate_out) {
            sycl::free(bufs->gate_out, *stream);
        }
        if (bufs->up_out) {
            sycl::free(bufs->up_out, *stream);
        }
        if (bufs->hidden_out) {
            sycl::free(bufs->hidden_out, *stream);
        }
        if (bufs->hidden_q8_dev) {
            sycl::free(bufs->hidden_q8_dev, *stream);
        }
        if (bufs->partial_out) {
            sycl::free(bufs->partial_out, *stream);
        }
        bufs->allocated = false;
    }

    // Use max of current and new dimensions (with headroom for batch)
    int64_t new_batch_max       = std::max(batch, bufs->batch_max) + 16;  // Headroom for varying batch sizes
    int64_t new_K_padded        = std::max(K_full_padded, bufs->K_full_padded);
    int64_t new_N_hidden_padded = std::max(N_hidden_shard_padded, bufs->N_hidden_shard_padded);
    int64_t new_N_out           = std::max(N_out, bufs->N_out);

    // Calculate buffer sizes
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    size_t input_q8_size  = new_batch_max * new_K_padded * q8_1_ts / q8_1_bs;
    size_t hidden_size    = new_N_hidden_padded * new_batch_max * sizeof(float);
    size_t hidden_q8_size = new_batch_max * new_N_hidden_padded * q8_1_ts / q8_1_bs;
    size_t partial_size   = new_N_out * new_batch_max * sizeof(float);
    size_t total_bytes    = input_q8_size + hidden_q8_size + partial_size + hidden_size * 3;

    // Allocate all buffers
    ggml_sycl::unified_cache_add_runtime_bytes(device, total_bytes);
    bufs->input_q8_dev  = (char *) ggml_sycl_malloc_device(input_q8_size, *stream, "ffn_buffers");
    bufs->gate_out      = (float *) ggml_sycl_malloc_device(hidden_size, *stream, "ffn_buffers");
    bufs->up_out        = (float *) ggml_sycl_malloc_device(hidden_size, *stream, "ffn_buffers");
    bufs->hidden_out    = (float *) ggml_sycl_malloc_device(hidden_size, *stream, "ffn_buffers");
    bufs->hidden_q8_dev = (char *) ggml_sycl_malloc_device(hidden_q8_size, *stream, "ffn_buffers");
    bufs->partial_out   = (float *) ggml_sycl_malloc_device(partial_size, *stream, "ffn_buffers");

    // Check allocation success
    if (!bufs->input_q8_dev || !bufs->gate_out || !bufs->up_out || !bufs->hidden_out || !bufs->hidden_q8_dev ||
        !bufs->partial_out) {
        GGML_LOG_ERROR("SYCL TP: Failed to allocate persistent FFN buffers for layer %d\n", layer);
        ggml_sycl::unified_cache_sub_runtime_bytes(device, total_bytes);
        // Partial cleanup
        if (bufs->input_q8_dev) {
            sycl::free(bufs->input_q8_dev, *stream);
        }
        if (bufs->gate_out) {
            sycl::free(bufs->gate_out, *stream);
        }
        if (bufs->up_out) {
            sycl::free(bufs->up_out, *stream);
        }
        if (bufs->hidden_out) {
            sycl::free(bufs->hidden_out, *stream);
        }
        if (bufs->hidden_q8_dev) {
            sycl::free(bufs->hidden_q8_dev, *stream);
        }
        if (bufs->partial_out) {
            sycl::free(bufs->partial_out, *stream);
        }
        *bufs = {};
        return nullptr;
    }

    // Record sizes and dimensions
    bufs->input_q8_size         = input_q8_size;
    bufs->hidden_size           = hidden_size;
    bufs->hidden_q8_size        = hidden_q8_size;
    bufs->partial_size          = partial_size;
    bufs->K_full_padded         = new_K_padded;
    bufs->N_hidden_shard_padded = new_N_hidden_padded;
    bufs->batch_max             = new_batch_max;
    bufs->N_out                 = new_N_out;
    bufs->allocated             = true;
    bufs->device_id             = device;

    GGML_SYCL_DEBUG("SYCL TP: Allocated persistent FFN buffers for layer %d (K=%lld, N_hidden=%lld, batch_max=%lld)\n",
                    layer, (long long) new_K_padded, (long long) new_N_hidden_padded, (long long) new_batch_max);

    return bufs;
}

void ggml_sycl_tp_free_ffn_buffers() {
    std::lock_guard<std::mutex> lock(g_tp_ffn_buffers_mutex);

    for (auto & [layer, bufs] : g_tp_ffn_buffers) {
        if (bufs.allocated) {
            const int runtime_device = bufs.device_id;
            if (bufs.input_q8_dev) {
                ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs.input_q8_size);
            }
            if (bufs.gate_out) {
                ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs.hidden_size);
            }
            if (bufs.up_out) {
                ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs.hidden_size);
            }
            if (bufs.hidden_out) {
                ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs.hidden_size);
            }
            if (bufs.hidden_q8_dev) {
                ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs.hidden_q8_size);
            }
            if (bufs.partial_out) {
                ggml_sycl::unified_cache_sub_runtime_bytes(runtime_device, bufs.partial_size);
            }
            // Use default queue for cleanup
            auto & q = dpct::get_in_order_queue();
            if (bufs.input_q8_dev) {
                sycl::free(bufs.input_q8_dev, q);
            }
            if (bufs.gate_out) {
                sycl::free(bufs.gate_out, q);
            }
            if (bufs.up_out) {
                sycl::free(bufs.up_out, q);
            }
            if (bufs.hidden_out) {
                sycl::free(bufs.hidden_out, q);
            }
            if (bufs.hidden_q8_dev) {
                sycl::free(bufs.hidden_q8_dev, q);
            }
            if (bufs.partial_out) {
                sycl::free(bufs.partial_out, q);
            }
        }
    }
    g_tp_ffn_buffers.clear();
    GGML_SYCL_DEBUG("SYCL TP: Freed all persistent FFN buffers\n");
}

float * ggml_sycl_tp_ensure_host_staging(size_t size, queue_ptr stream) {
    std::lock_guard<std::mutex> lock(g_tp_host_staging_mutex);

    if (g_tp_host_staging.capacity >= size && g_tp_host_staging.buf != nullptr) {
        g_tp_host_staging.size = size;
        return g_tp_host_staging.buf;
    }

    // Free old buffer
    if (g_tp_host_staging.buf != nullptr) {
        ggml_sycl_free_host_tracked_bytes(g_tp_host_staging.buf, g_tp_host_staging.capacity, *stream);
    }

    // Allocate with headroom
    size_t new_capacity        = size + (size / 4);  // 25% headroom
    g_tp_host_staging.buf      =
        static_cast<float *>(ggml_sycl_malloc_host_tracked_bytes(new_capacity, *stream, "tp_host_staging"));
    g_tp_host_staging.capacity = new_capacity;
    g_tp_host_staging.size     = size;

    GGML_SYCL_DEBUG("SYCL TP: Allocated persistent host staging buffer (%zu bytes)\n", new_capacity);
    return g_tp_host_staging.buf;
}

void ggml_sycl_tp_free_host_staging() {
    std::lock_guard<std::mutex> lock(g_tp_host_staging_mutex);

    if (g_tp_host_staging.buf != nullptr) {
        ggml_sycl_free_host_tracked_bytes(g_tp_host_staging.buf, g_tp_host_staging.capacity,
                                          dpct::get_in_order_queue());
        g_tp_host_staging.buf      = nullptr;
        g_tp_host_staging.capacity = 0;
        g_tp_host_staging.size     = 0;
    }
}

// =============================================================================
// Quantized Communication Buffers (Flash Communication)
// =============================================================================

static ggml_sycl_tp_quant_comm_buffers g_tp_quant_comm_bufs = {};
static std::mutex                      g_tp_quant_comm_mutex;

bool ggml_sycl_quant_allreduce_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_SYCL_QUANT_ALLREDUCE");
        // Enable by default for TP mode (33% bandwidth reduction)
        // Disable with GGML_SYCL_QUANT_ALLREDUCE=0
        enabled          = (env != nullptr) ? atoi(env) : 1;

        if (enabled) {
            GGML_LOG_INFO("SYCL TP: INT16 Quantized AllReduce enabled (33%% bandwidth reduction)\n");
        }
    }
    return enabled != 0;
}

// Get the minimum tensor size threshold for quantized allreduce
// Returns threshold from GGML_SYCL_QUANT_THRESHOLD env var, or default
static size_t get_quant_allreduce_threshold() {
    static size_t threshold   = 0;
    static bool   initialized = false;
    if (!initialized) {
        const char * env = getenv("GGML_SYCL_QUANT_THRESHOLD");
        // Default: 65536 elements (256KB FP32)
        // Benchmarks show quant overhead hurts small tensors (tg128: 8.1 -> 6.3 t/s)
        // but helps or is neutral for larger tensors (pp512: ~same performance)
        // Crossover is around 32K-64K elements
        threshold        = (env != nullptr) ? (size_t) atol(env) : 65536;
        initialized      = true;
        if (ggml_sycl_quant_allreduce_enabled()) {
            GGML_LOG_INFO("SYCL TP: Quant AllReduce threshold = %zu elements (%.1f KB FP32)\n", threshold,
                          (float) (threshold * sizeof(float)) / 1024.0f);
        }
    }
    return threshold;
}

bool ggml_sycl_should_use_quant_allreduce(size_t n_elements) {
    // Must be enabled globally
    if (!ggml_sycl_quant_allreduce_enabled()) {
        return false;
    }
    // Use FP32 for small tensors (lower overhead beats bandwidth savings)
    // Use INT16 quant for large tensors (bandwidth savings outweigh overhead)
    return n_elements >= get_quant_allreduce_threshold();
}

void ggml_sycl_tp_init_quant_comm_buffers(size_t initial_size) {
    std::lock_guard<std::mutex> lock(g_tp_quant_comm_mutex);

    if (g_tp_quant_comm_bufs.allocated) {
        return;
    }

    // Pre-allocate with 25% headroom
    size_t alloc_size = initial_size + (initial_size / 4);

    GGML_SYCL_DEBUG("SYCL TP: Pre-allocating INT16 quant comm buffers for %zu elements\n", alloc_size);

    // Allocate host buffers (pinned host memory)
    // INT16 = 2 bytes per element
    g_tp_quant_comm_bufs.host_q0     = (int16_t *) ggml_sycl_host_malloc(alloc_size * sizeof(int16_t));
    g_tp_quant_comm_bufs.host_q1     = (int16_t *) ggml_sycl_host_malloc(alloc_size * sizeof(int16_t));
    g_tp_quant_comm_bufs.host_result = (float *) ggml_sycl_host_malloc(alloc_size * sizeof(float));

    if (!g_tp_quant_comm_bufs.host_q0 || !g_tp_quant_comm_bufs.host_q1 || !g_tp_quant_comm_bufs.host_result) {
        GGML_LOG_ERROR("SYCL TP: Failed to allocate quant comm host buffers\n");
        if (g_tp_quant_comm_bufs.host_q0) {
            ggml_sycl_host_free(g_tp_quant_comm_bufs.host_q0);
        }
        if (g_tp_quant_comm_bufs.host_q1) {
            ggml_sycl_host_free(g_tp_quant_comm_bufs.host_q1);
        }
        if (g_tp_quant_comm_bufs.host_result) {
            ggml_sycl_host_free(g_tp_quant_comm_bufs.host_result);
        }
        g_tp_quant_comm_bufs.host_q0 = nullptr;
        g_tp_quant_comm_bufs.host_q1 = nullptr;
        g_tp_quant_comm_bufs.host_result = nullptr;
        return;
    }

    // Initialize all device buffer pointers to nullptr
    for (int i = 0; i < GGML_SYCL_MAX_DEVICES; i++) {
        g_tp_quant_comm_bufs.dev_q[i]      = nullptr;
        g_tp_quant_comm_bufs.dev_minmax[i] = nullptr;
    }

    // Allocate device buffers on each TP device
    for (int i = 0; i < g_sycl_tp_config.world_size && i < 2; i++) {
        int dev = g_sycl_tp_config.devices[i];
        ggml_sycl_set_device(dev);
        auto & q = dpct::get_in_order_queue();

        // INT16 = 2 bytes per element
        ggml_sycl::unified_cache_add_runtime_bytes(dev, alloc_size * sizeof(int16_t));
        g_tp_quant_comm_bufs.dev_q[i]      = ggml_sycl_malloc_device_t<int16_t>(alloc_size, q, "tp_quant_comm");
        if (!g_tp_quant_comm_bufs.dev_q[i]) {
            ggml_sycl::unified_cache_sub_runtime_bytes(dev, alloc_size * sizeof(int16_t));
        }
        ggml_sycl::unified_cache_add_runtime_bytes(dev, 2 * sizeof(float));
        g_tp_quant_comm_bufs.dev_minmax[i] = ggml_sycl_malloc_device_t<float>(2, q, "tp_quant_comm");
        if (!g_tp_quant_comm_bufs.dev_minmax[i]) {
            ggml_sycl::unified_cache_sub_runtime_bytes(dev, 2 * sizeof(float));
        }

        if (!g_tp_quant_comm_bufs.dev_q[i] || !g_tp_quant_comm_bufs.dev_minmax[i]) {
            GGML_LOG_ERROR("SYCL TP: Failed to allocate quant comm device buffers on device %d\n", dev);
            // Cleanup partial allocations
            if (g_tp_quant_comm_bufs.host_q0) {
                ggml_sycl_host_free(g_tp_quant_comm_bufs.host_q0);
            }
            if (g_tp_quant_comm_bufs.host_q1) {
                ggml_sycl_host_free(g_tp_quant_comm_bufs.host_q1);
            }
            if (g_tp_quant_comm_bufs.host_result) {
                ggml_sycl_host_free(g_tp_quant_comm_bufs.host_result);
            }
            for (int j = 0; j <= i; j++) {
                int dev_id = g_sycl_tp_config.devices[j];
                ggml_sycl_set_device(dev_id);
                auto & q_free = dpct::get_in_order_queue();
                if (g_tp_quant_comm_bufs.dev_q[j]) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(dev_id, alloc_size * sizeof(int16_t));
                    sycl::free(g_tp_quant_comm_bufs.dev_q[j], q_free);
                }
                if (g_tp_quant_comm_bufs.dev_minmax[j]) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(dev_id, 2 * sizeof(float));
                    sycl::free(g_tp_quant_comm_bufs.dev_minmax[j], q_free);
                }
            }
            g_tp_quant_comm_bufs = {};
            return;
        }
    }

    g_tp_quant_comm_bufs.capacity  = alloc_size;
    g_tp_quant_comm_bufs.allocated = true;

    GGML_SYCL_DEBUG("SYCL TP: Pre-allocated INT16 quant comm buffers: %zu elements (%zu MB INT16, %zu MB FP32)\n",
                    alloc_size, (alloc_size * sizeof(int16_t)) / (1024 * 1024),
                    (alloc_size * sizeof(float)) / (1024 * 1024));
}

void ggml_sycl_tp_ensure_quant_comm_buffers(size_t n_elements) {
    std::lock_guard<std::mutex> lock(g_tp_quant_comm_mutex);

    if (g_tp_quant_comm_bufs.allocated && g_tp_quant_comm_bufs.capacity >= n_elements) {
        return;  // Existing buffers are sufficient
    }

    // Need to resize - free old and allocate new
    if (g_tp_quant_comm_bufs.allocated) {
        GGML_SYCL_DEBUG("SYCL TP: Resizing quant comm buffers from %zu to %zu elements\n",
                        g_tp_quant_comm_bufs.capacity, n_elements);

        // Free host buffers
        if (g_tp_quant_comm_bufs.host_q0) {
            ggml_sycl_host_free(g_tp_quant_comm_bufs.host_q0);
        }
        if (g_tp_quant_comm_bufs.host_q1) {
            ggml_sycl_host_free(g_tp_quant_comm_bufs.host_q1);
        }
        if (g_tp_quant_comm_bufs.host_result) {
            ggml_sycl_host_free(g_tp_quant_comm_bufs.host_result);
        }

        // Free device buffers
        auto & q = dpct::get_in_order_queue();
        for (int i = 0; i < GGML_SYCL_MAX_DEVICES; i++) {
            const int dev_id = (i < g_sycl_tp_config.world_size) ? g_sycl_tp_config.devices[i] : i;
            if (g_tp_quant_comm_bufs.dev_q[i]) {
                ggml_sycl::unified_cache_sub_runtime_bytes(dev_id, g_tp_quant_comm_bufs.capacity * sizeof(int16_t));
                sycl::free(g_tp_quant_comm_bufs.dev_q[i], q);
            }
            if (g_tp_quant_comm_bufs.dev_minmax[i]) {
                ggml_sycl::unified_cache_sub_runtime_bytes(dev_id, 2 * sizeof(float));
                sycl::free(g_tp_quant_comm_bufs.dev_minmax[i], q);
            }
        }
        g_tp_quant_comm_bufs = {};
    }

    // Release lock and call init (which takes the lock again)
    // Note: We need to release/reacquire because init takes the lock
    // This is safe because we're single-threaded at this point
}

ggml_sycl_tp_quant_comm_buffers * ggml_sycl_tp_get_quant_comm_buffers() {
    return g_tp_quant_comm_bufs.allocated ? &g_tp_quant_comm_bufs : nullptr;
}

void ggml_sycl_tp_free_quant_comm_buffers() {
    std::lock_guard<std::mutex> lock(g_tp_quant_comm_mutex);

    if (!g_tp_quant_comm_bufs.allocated) {
        return;
    }

    GGML_SYCL_DEBUG("SYCL TP: Freeing quant comm buffers\n");

    // Free host buffers (pinned host memory)
    if (g_tp_quant_comm_bufs.host_q0) {
        ggml_sycl_host_free(g_tp_quant_comm_bufs.host_q0);
    }
    if (g_tp_quant_comm_bufs.host_q1) {
        ggml_sycl_host_free(g_tp_quant_comm_bufs.host_q1);
    }
    if (g_tp_quant_comm_bufs.host_result) {
        ggml_sycl_host_free(g_tp_quant_comm_bufs.host_result);
    }

    // Free device buffers
    auto & q = dpct::get_in_order_queue();
    for (int i = 0; i < GGML_SYCL_MAX_DEVICES; i++) {
        const int dev_id = (i < g_sycl_tp_config.world_size) ? g_sycl_tp_config.devices[i] : i;
        if (g_tp_quant_comm_bufs.dev_q[i]) {
            ggml_sycl::unified_cache_sub_runtime_bytes(dev_id, g_tp_quant_comm_bufs.capacity * sizeof(int16_t));
            sycl::free(g_tp_quant_comm_bufs.dev_q[i], q);
        }
        if (g_tp_quant_comm_bufs.dev_minmax[i]) {
            ggml_sycl::unified_cache_sub_runtime_bytes(dev_id, 2 * sizeof(float));
            sycl::free(g_tp_quant_comm_bufs.dev_minmax[i], q);
        }
    }

    g_tp_quant_comm_bufs = {};
}

void ggml_sycl_tp_free() {
    // NOTE: Do NOT free resources that need to persist across backend lifetimes!
    // The "fitting params to device memory" step creates temporary backends that
    // get freed before model loading. Resources like CCL and quant comm buffers
    // must persist for actual inference.
    //
    // NOT freed here (persist across backends):
    // - CCL resources (cleanup via atexit in ggml_sycl_ccl_init_multiprocess)
    // - Quant comm buffers (pre-allocated for TP allreduce performance)

    // Free persistent FFN buffers first (uses its own mutex)
    ggml_sycl_tp_free_ffn_buffers();
    ggml_sycl_tp_free_host_staging();
    // NOTE: Do NOT free quant comm buffers - they persist across backend lifetimes

    std::lock_guard<std::mutex> lock(g_tp_shared_reduce_mutex);
    if (g_tp_shared_reduce_buf != nullptr) {
        ggml_sycl::unified_cache_sub_runtime_bytes(dpct::dev_mgr::instance().current_device_id(),
                                                   g_tp_shared_reduce_buf_size);
        sycl::free(g_tp_shared_reduce_buf, dpct::get_in_order_queue());
        g_tp_shared_reduce_buf      = nullptr;
        g_tp_shared_reduce_buf_size = 0;
    }
    // Free persistent host buffers
    if (g_tp_host_buf0 != nullptr) {
        std::free(g_tp_host_buf0);
        g_tp_host_buf0 = nullptr;
    }
    if (g_tp_host_buf1 != nullptr) {
        std::free(g_tp_host_buf1);
        g_tp_host_buf1 = nullptr;
    }
    g_tp_host_buf_size = 0;

    // NOTE: Do NOT reset g_sycl_tp_config here!
    // The TP configuration (enabled, world_size, devices) must persist across
    // backend creations/destructions. The "fitting params to device memory" step
    // creates temporary backends that get freed before model loading, and if we
    // reset the config here, model loading will see world_size=1 instead of the
    // correct value, causing weight sharding to fail.
    // The TP config is set once during initialization and should remain valid
    // for the entire process lifetime.
}

int ggml_sycl_tp_world_size() {
    // For graph building, return 1 in multi-process mode
    // (each process builds the full graph, just with different data)
    if (g_sycl_tp_config.is_multiprocess) {
        return 1;
    }
    return g_sycl_tp_config.enabled ? g_sycl_tp_config.world_size : 1;
}

int ggml_sycl_tp_world_size_internal() {
    // Returns true world_size even in multi-process mode
    return g_sycl_tp_config.enabled ? g_sycl_tp_config.world_size : 1;
}

float * ggml_sycl_tp_ensure_shared_reduce_buffer(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_tp_shared_reduce_mutex);

    if (g_tp_shared_reduce_buf_size >= bytes && g_tp_shared_reduce_buf != nullptr) {
        return g_tp_shared_reduce_buf;
    }

    // Free old buffer
    if (g_tp_shared_reduce_buf != nullptr) {
        ggml_sycl::unified_cache_sub_runtime_bytes(dpct::dev_mgr::instance().current_device_id(),
                                                   g_tp_shared_reduce_buf_size);
        sycl::free(g_tp_shared_reduce_buf, dpct::get_in_order_queue());
    }

    // Allocate shared memory for zero-copy ALL_REDUCE
    ggml_sycl::unified_cache_add_runtime_bytes(dpct::dev_mgr::instance().current_device_id(), bytes);
    g_tp_shared_reduce_buf      = ggml_sycl_malloc_shared_t<float>(bytes / sizeof(float), dpct::get_in_order_queue(),
                                                              "tp_shared_reduce");
    g_tp_shared_reduce_buf_size = bytes;
    if (!g_tp_shared_reduce_buf) {
        ggml_sycl::unified_cache_sub_runtime_bytes(dpct::dev_mgr::instance().current_device_id(), bytes);
    }

    GGML_SYCL_DEBUG("SYCL TP: Allocated %zu bytes for shared reduce buffer\n", bytes);
    return g_tp_shared_reduce_buf;
}

void ggml_sycl_tp_get_host_reduce_buffers(size_t bytes, float ** buf0, float ** buf1) {
    std::lock_guard<std::mutex> lock(g_tp_shared_reduce_mutex);

    if (g_tp_host_buf_size >= bytes && g_tp_host_buf0 != nullptr && g_tp_host_buf1 != nullptr) {
        *buf0 = g_tp_host_buf0;
        *buf1 = g_tp_host_buf1;
        return;
    }

    // Free old buffers if they exist
    if (g_tp_host_buf0 != nullptr) {
        ggml_sycl_host_free(g_tp_host_buf0);
    }
    if (g_tp_host_buf1 != nullptr) {
        ggml_sycl_host_free(g_tp_host_buf1);
    }

    // Allocate new buffers (with some headroom to avoid frequent reallocs)
    size_t alloc_size  = bytes + (bytes / 4);  // 25% headroom
    g_tp_host_buf0     = (float *) ggml_sycl_host_malloc(alloc_size);
    g_tp_host_buf1     = (float *) ggml_sycl_host_malloc(alloc_size);
    g_tp_host_buf_size = alloc_size;

    if (g_tp_host_buf0 == nullptr || g_tp_host_buf1 == nullptr) {
        GGML_LOG_ERROR("SYCL TP: Failed to allocate %zu bytes for persistent host reduce buffers\n", alloc_size);
        if (g_tp_host_buf0) {
            ggml_sycl_host_free(g_tp_host_buf0);
        }
        if (g_tp_host_buf1) {
            ggml_sycl_host_free(g_tp_host_buf1);
        }
        g_tp_host_buf0 = nullptr;
        g_tp_host_buf1 = nullptr;
        g_tp_host_buf_size = 0;
        *buf0 = nullptr;
        *buf1 = nullptr;
        return;
    }

    GGML_SYCL_DEBUG("SYCL TP: Allocated %zu bytes for persistent host reduce buffers\n", alloc_size);

    *buf0 = g_tp_host_buf0;
    *buf1 = g_tp_host_buf1;
}

// ============================================================================
// Pipeline Parallelism (PP) Double-Buffered Transfer
// ============================================================================
// Eliminates per-transfer malloc/free overhead AND enables overlapping
// transfers using ping-pong double buffering.
//
// On Intel Arc (no P2P), dev2dev copies require host staging. We use:
// 1. sycl::malloc_host for pinned memory (faster DMA transfers)
// 2. Two buffers for double-buffering (overlap src->host with host->dst)
// 3. Event tracking to know when each buffer is safe to reuse

static void *                     g_pp_transfer_buf[2]   = { nullptr, nullptr };  // Double buffers
static size_t                     g_pp_transfer_buf_size = 0;
static std::mutex                 g_pp_transfer_mutex;
static int                        g_pp_current_buf  = 0;  // Which buffer to use next (0 or 1)
static std::optional<sycl::event> g_pp_pending_event[2];  // Pending events for each buffer

// Internal: allocate double buffers
static bool ggml_sycl_pp_alloc_buffers(size_t bytes) {
    // Free old buffers if they exist
    for (int i = 0; i < 2; i++) {
        if (g_pp_transfer_buf[i] != nullptr) {
            ggml_sycl_host_free(g_pp_transfer_buf[i]);
            g_pp_transfer_buf[i] = nullptr;
        }
        g_pp_pending_event[i].reset();
    }
    g_pp_transfer_buf_size = 0;

    // Allocate with 25% headroom to reduce reallocations
    size_t alloc_size = bytes + (bytes / 4);

    // Allocate pinned host memory for faster transfers
    g_pp_transfer_buf[0] = ggml_sycl_host_malloc(alloc_size);
    g_pp_transfer_buf[1] = ggml_sycl_host_malloc(alloc_size);
    if (g_pp_transfer_buf[0] == nullptr || g_pp_transfer_buf[1] == nullptr) {
        if (g_pp_transfer_buf[0]) {
            ggml_sycl_host_free(g_pp_transfer_buf[0]);
        }
        if (g_pp_transfer_buf[1]) {
            ggml_sycl_host_free(g_pp_transfer_buf[1]);
        }
        g_pp_transfer_buf[0] = g_pp_transfer_buf[1] = nullptr;
        GGML_LOG_ERROR("SYCL PP: Failed to allocate 2x %zu bytes pinned host buffers\n", alloc_size);
        return false;
    }

    g_pp_transfer_buf_size = alloc_size;
    GGML_SYCL_DEBUG("SYCL PP: Allocated 2x %zu bytes pinned host memory for double-buffered transfer\n", alloc_size);
    return true;
}

void * ggml_sycl_get_dev2dev_transfer_buffer(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_pp_transfer_mutex);

    // Allocate if needed
    if (g_pp_transfer_buf_size < bytes || g_pp_transfer_buf[0] == nullptr) {
        if (!ggml_sycl_pp_alloc_buffers(bytes)) {
            return nullptr;
        }
    }

    // Return the current buffer (simple single-buffer mode for compatibility)
    return g_pp_transfer_buf[0];
}

// Get buffer for double-buffered transfer, returns buffer index via out param
void * ggml_sycl_get_dev2dev_transfer_buffer_double(size_t bytes, int * buf_idx) {
    std::lock_guard<std::mutex> lock(g_pp_transfer_mutex);

    // Allocate if needed
    if (g_pp_transfer_buf_size < bytes || g_pp_transfer_buf[0] == nullptr) {
        if (!ggml_sycl_pp_alloc_buffers(bytes)) {
            *buf_idx = -1;
            return nullptr;
        }
    }

    // Get next buffer in ping-pong sequence
    int idx          = g_pp_current_buf;
    g_pp_current_buf = 1 - g_pp_current_buf;  // Toggle 0<->1

    // Wait for any pending operation on this buffer to complete
    if (g_pp_pending_event[idx].has_value()) {
        g_pp_pending_event[idx]->wait();
        g_pp_pending_event[idx].reset();
    }

    *buf_idx = idx;
    return g_pp_transfer_buf[idx];
}

// Record that a buffer has a pending transfer that must complete before reuse
void ggml_sycl_set_dev2dev_transfer_event(int buf_idx, sycl::event evt) {
    std::lock_guard<std::mutex> lock(g_pp_transfer_mutex);
    if (buf_idx >= 0 && buf_idx < 2) {
        g_pp_pending_event[buf_idx] = evt;
    }
}

// Wait for all pending transfers to complete
void ggml_sycl_wait_dev2dev_transfers() {
    std::lock_guard<std::mutex> lock(g_pp_transfer_mutex);
    for (int i = 0; i < 2; i++) {
        if (g_pp_pending_event[i].has_value()) {
            g_pp_pending_event[i]->wait();
            g_pp_pending_event[i].reset();
        }
    }
}

void ggml_sycl_free_dev2dev_transfer_buffer() {
    std::lock_guard<std::mutex> lock(g_pp_transfer_mutex);

    // Wait for any pending transfers
    for (int i = 0; i < 2; i++) {
        if (g_pp_pending_event[i].has_value()) {
            g_pp_pending_event[i]->wait();
            g_pp_pending_event[i].reset();
        }
    }

    // Free buffers
    for (int i = 0; i < 2; i++) {
        if (g_pp_transfer_buf[i] != nullptr) {
            ggml_sycl_host_free(g_pp_transfer_buf[i]);
            g_pp_transfer_buf[i] = nullptr;
        }
    }
    g_pp_transfer_buf_size = 0;
    g_pp_current_buf       = 0;
    GGML_SYCL_DEBUG("SYCL PP: Freed double-buffered transfer buffers\n");
}

int ggml_sycl_tp_get_rank(int device) {
    if (!g_sycl_tp_config.enabled) {
        return 0;
    }
    for (int i = 0; i < g_sycl_tp_config.world_size; i++) {
        if (g_sycl_tp_config.devices[i] == device) {
            return i;
        }
    }
    return 0;
}

bool ggml_sycl_tp_enabled() {
    return g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1;
}

void ggml_sycl_tp_get_slice(int64_t total_size, int rank, int world_size, int64_t * offset, int64_t * size) {
    int64_t slice_size = total_size / world_size;
    *offset            = rank * slice_size;
    *size              = (rank == world_size - 1) ? (total_size - *offset) : slice_size;
}

tp_layer_type ggml_sycl_tp_get_layer_type(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->extra == nullptr) {
        return tp_layer_type::TP_NONE;
    }

    auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
    if (extra->tp_type_cached) {
        return extra->tp_type;
    }

    // Determine type by name pattern
    tp_layer_type tp_type = tp_layer_type::TP_NONE;
    if (tensor->name) {
        // Column-parallel: output dimension is sharded (Q, K, V, gate, up)
        if (strstr(tensor->name, "attn_q") || strstr(tensor->name, "attn_k") || strstr(tensor->name, "attn_v") ||
            strstr(tensor->name, "ffn_gate") || strstr(tensor->name, "ffn_up")) {
            tp_type = tp_layer_type::TP_COLUMN_PARALLEL;
        }
        // Row-parallel: input dimension is sharded (O, down)
        else if (strstr(tensor->name, "attn_output") || strstr(tensor->name, "ffn_down")) {
            tp_type = tp_layer_type::TP_ROW_PARALLEL;
        }
    }

    extra->tp_type        = tp_type;
    extra->tp_type_cached = true;
    return tp_type;
}

bool ggml_sycl_tp_needs_allreduce(const ggml_tensor * tensor) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return false;
    }

    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);
    return (tp_type == tp_layer_type::TP_ROW_PARALLEL);
}

void ggml_sycl_tp_get_sharded_dims(const ggml_tensor * tensor,
                                   int                 rank,
                                   int                 world_size,
                                   int64_t *           local_ne0,
                                   int64_t *           local_ne1,
                                   int64_t *           offset_ne0,
                                   int64_t *           offset_ne1) {
    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);

    *local_ne0  = tensor->ne[0];
    *local_ne1  = tensor->ne[1];
    *offset_ne0 = 0;
    *offset_ne1 = 0;

    if (tp_type == tp_layer_type::TP_COLUMN_PARALLEL) {
        *local_ne1  = tensor->ne[1] / world_size;
        *offset_ne1 = rank * (*local_ne1);
    } else if (tp_type == tp_layer_type::TP_ROW_PARALLEL) {
        *local_ne0  = tensor->ne[0] / world_size;
        *offset_ne0 = rank * (*local_ne0);
    }
}

bool ggml_sycl_tp_should_shard(const ggml_tensor * tensor) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return false;
    }

    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);
    return (tp_type == tp_layer_type::TP_COLUMN_PARALLEL || tp_type == tp_layer_type::TP_ROW_PARALLEL);
}

void ggml_sycl_tp_copy_weight_shard(void *              dst_device,
                                    const void *        src_host,
                                    const ggml_tensor * tensor,
                                    int                 rank,
                                    int                 world_size,
                                    queue_ptr           stream) {
    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);

    int64_t ne0 = tensor->ne[0];
    int64_t ne1 = tensor->ne[1];

    if (tp_type == tp_layer_type::TP_COLUMN_PARALLEL) {
        // Shard ne1 dimension
        int64_t shard_ne1  = ne1 / world_size;
        int64_t offset_ne1 = rank * shard_ne1;

        size_t row_size   = ggml_row_size(tensor->type, ne0);
        size_t shard_size = row_size * shard_ne1;

        const char * src = static_cast<const char *>(src_host) + offset_ne1 * row_size;
        stream->memcpy(dst_device, src, shard_size).wait();
    } else if (tp_type == tp_layer_type::TP_ROW_PARALLEL) {
        // Shard ne0 dimension - more complex due to quantization blocks
        int64_t shard_ne0  = ne0 / world_size;
        int64_t offset_ne0 = rank * shard_ne0;

        size_t full_row_size  = ggml_row_size(tensor->type, ne0);
        size_t shard_row_size = ggml_row_size(tensor->type, shard_ne0);

        // Copy row by row
        char *       dst = static_cast<char *>(dst_device);
        const char * src = static_cast<const char *>(src_host);

        for (int64_t row = 0; row < ne1; row++) {
            size_t src_offset = row * full_row_size + ggml_row_size(tensor->type, offset_ne0);
            stream->memcpy(dst + row * shard_row_size, src + src_offset, shard_row_size).wait();
        }
    } else {
        // Full copy for non-sharded tensors
        size_t size = ggml_nbytes(tensor);
        stream->memcpy(dst_device, src_host, size).wait();
    }
}

size_t ggml_sycl_tp_get_shard_size(const ggml_tensor * tensor, int rank, int world_size) {
    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);

    int64_t ne0 = tensor->ne[0];
    int64_t ne1 = tensor->ne[1];
    int64_t ne2 = tensor->ne[2];
    int64_t ne3 = tensor->ne[3];

    if (tp_type == tp_layer_type::TP_COLUMN_PARALLEL) {
        ne1 = ne1 / world_size;
    } else if (tp_type == tp_layer_type::TP_ROW_PARALLEL) {
        ne0 = ne0 / world_size;
    }

    GGML_UNUSED(rank);
    return ggml_row_size(tensor->type, ne0) * ne1 * ne2 * ne3;
}

// ALL_REDUCE sum implementation
void ggml_sycl_all_reduce_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return;
    }

    // Multi-process mode: use CCL (ccl-comm.hpp provides stubs when CCL not available)
    if (g_sycl_tp_config.is_multiprocess && ggml_sycl_ccl_is_initialized()) {
        // The MUL_MAT result (partial sum) is in src[0], not dst
        // We need to allreduce src[0]->data and store result in dst->data
        ggml_tensor * src = dst->src[0];
        if (src == nullptr) {
            GGML_LOG_ERROR("SYCL CCL ALL_REDUCE: src[0] is null!\n");
            return;
        }

        float * src_data = static_cast<float *>(src->data);
        float * dst_data = static_cast<float *>(dst->data);
        size_t  count    = ggml_nelements(dst);

        // CCL allreduce: src_data -> dst_data with sum
        // Use the two-buffer overload: (send_buf, recv_buf, count, device)
        ggml_sycl_ccl_allreduce_sum_f32(src_data, dst_data, count, ctx.device);
        return;
    }

    // Single-process multi-device mode with quantized AllReduce:
    // The FFN/attention callbacks did the AllReduce using quantized_allreduce()
    // and wrote the combined result to src (node_24). But we still need to copy
    // from src to dst (attn_out-0) so downstream ops get the data.
    // (The quantized path writes to the MUL_MAT dst, not the ALL_REDUCE_SUM dst)
    if (ggml_sycl_quant_allreduce_enabled()) {
        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "ALL_REDUCE_SUM (quant): Still need to copy src->dst\n");
        }
        // Fall through to do the copy
    }

    // Single-process multi-device mode:
    // The MUL_MAT FALLBACK path already does ALL_REDUCE internally and writes
    // the result to dst->src[0] (the MUL_MAT output tensor).
    // We just need to COPY from src[0] to dst so downstream ops get the data.
    //
    // Graph structure:
    //   MUL_MAT -> dst->src[0] (e.g., "node_24")
    //   ALL_REDUCE_SUM -> dst (e.g., "attn_out-0")
    //   ADD reads from dst
    //
    // The FALLBACK wrote to dst->src[0], but ADD expects data in dst.

    ggml_tensor * src = dst->src[0];
    if (src == nullptr) {
        GGML_LOG_ERROR("SYCL TP ALL_REDUCE_SUM: src[0] is null!\n");
        return;
    }

    size_t dst_size    = ggml_nbytes(dst);
    int    main_device = ctx.device;

    // Get source data pointer (where MUL_MAT FALLBACK wrote the result)
    void * src_ptr = ggml_sycl_get_data_ptr(src, main_device);
    // Get destination data pointer (where downstream ops expect the data)
    void * dst_ptr = ggml_sycl_get_data_ptr(dst, main_device);

    if (g_ggml_sycl_tp_debug) {
        fprintf(stderr, "ALL_REDUCE_SUM: Copying from src=%s (%p) to dst=%s (%p), size=%zu\n",
                src->name ? src->name : "(null)", src_ptr, dst->name ? dst->name : "(null)", dst_ptr, dst_size);
    }

    // If src and dst point to same memory, nothing to do
    if (src_ptr == dst_ptr) {
        GGML_SYCL_DEBUG("ALL_REDUCE_SUM: src_ptr == dst_ptr, skipping copy\n");
        return;
    }

    // Copy from src to dst
    ggml_sycl_set_device(main_device);
    queue_ptr stream = &ggml_sycl_get_device(main_device).default_queue();
    stream->memcpy(dst_ptr, src_ptr, dst_size).wait();

    if (g_ggml_sycl_tp_debug) {
        // Verify the copy
        float verify[4];
        stream->memcpy(verify, dst_ptr, std::min(dst_size, 4 * sizeof(float))).wait();
        fprintf(stderr, "ALL_REDUCE_SUM: VERIFY dst[0..3]=[%.4f,%.4f,%.4f,%.4f]\n", verify[0], verify[1], verify[2],
                verify[3]);
    }

    GGML_UNUSED(ctx);
}

// ============================================================================
// Global PP Config and Function Implementations
// Pipeline Parallelism (vLLM-style layer split across devices)
// ============================================================================

// Global PP config definition
ggml_sycl_pp_config g_sycl_pp_config = {};

// Debug level for PP operations (set via GGML_SYCL_PP_DEBUG env var)
int g_ggml_sycl_pp_debug = 0;

// Static init flag
static bool g_pp_debug_initialized = false;

static void pp_init_debug() {
    if (!g_pp_debug_initialized) {
        const char * debug_env = std::getenv("GGML_SYCL_PP_DEBUG");
        if (debug_env) {
            g_ggml_sycl_pp_debug = std::atoi(debug_env);
        }
        g_pp_debug_initialized = true;
    }
}

#define PP_DEBUG(fmt, ...)                              \
    do {                                                \
        if (g_ggml_sycl_pp_debug >= 1)                  \
            GGML_LOG_DEBUG("[PP] " fmt, ##__VA_ARGS__); \
    } while (0)

#define PP_DEBUG_VERBOSE(fmt, ...)                        \
    do {                                                  \
        if (g_ggml_sycl_pp_debug >= 2)                    \
            GGML_LOG_DEBUG("[PP-V] " fmt, ##__VA_ARGS__); \
    } while (0)

void ggml_sycl_pp_init(const int * device_ids, int num_devices, int total_layers, const int * layers_per_stage) {
    pp_init_debug();

    if (num_devices < 1 || num_devices > GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("SYCL PP: Invalid number of devices: %d (max: %d)\n", num_devices, GGML_SYCL_MAX_DEVICES);
        return;
    }

    if (total_layers < 1 || total_layers > GGML_SYCL_PP_MAX_LAYERS) {
        GGML_LOG_ERROR("SYCL PP: Invalid number of layers: %d (max: %d)\n", total_layers, GGML_SYCL_PP_MAX_LAYERS);
        return;
    }

    // Clear any previous config
    ggml_sycl_pp_free();

    g_sycl_pp_config.enabled    = true;
    g_sycl_pp_config.num_stages = num_devices;

    // Copy device IDs
    for (int i = 0; i < num_devices; i++) {
        g_sycl_pp_config.devices[i] = device_ids[i];
    }

    // Calculate or copy layer distribution
    if (layers_per_stage != nullptr) {
        // Use provided distribution
        int total = 0;
        for (int i = 0; i < num_devices; i++) {
            g_sycl_pp_config.layers_per_stage[i] = layers_per_stage[i];
            total += layers_per_stage[i];
        }
        if (total != total_layers) {
            GGML_LOG_WARN("SYCL PP: layers_per_stage sum (%d) != total_layers (%d)\n", total, total_layers);
        }
    } else {
        // Distribute layers evenly
        int base_layers = total_layers / num_devices;
        int remainder   = total_layers % num_devices;

        for (int i = 0; i < num_devices; i++) {
            // Give extra layers to earlier stages (handles remainder)
            g_sycl_pp_config.layers_per_stage[i] = base_layers + (i < remainder ? 1 : 0);
        }
    }

    // Build layer-to-device lookup table
    int layer = 0;
    for (int stage = 0; stage < num_devices; stage++) {
        for (int l = 0; l < g_sycl_pp_config.layers_per_stage[stage]; l++) {
            if (layer < GGML_SYCL_PP_MAX_LAYERS) {
                g_sycl_pp_config.layer_to_device[layer] = g_sycl_pp_config.devices[stage];
                layer++;
            }
        }
    }

    // Log configuration
    PP_DEBUG("Initialized PP with %d stages, %d layers\n", num_devices, total_layers);
    for (int i = 0; i < num_devices; i++) {
        int start_layer, end_layer;
        ggml_sycl_pp_get_stage_layers(i, &start_layer, &end_layer);
        PP_DEBUG("  Stage %d: device %d, layers %d-%d (%d layers)\n", i, g_sycl_pp_config.devices[i], start_layer,
                 end_layer - 1, g_sycl_pp_config.layers_per_stage[i]);
    }
}

void ggml_sycl_pp_free() {
    // Free stage output buffers
    for (int i = 0; i < GGML_SYCL_MAX_DEVICES; i++) {
        if (g_sycl_pp_config.stage_output_buf[i] != nullptr) {
            ggml_sycl::unified_cache_sub_runtime_bytes(dpct::dev_mgr::instance().current_device_id(),
                                                       g_sycl_pp_config.stage_output_size);
            sycl::free(g_sycl_pp_config.stage_output_buf[i], dpct::get_in_order_queue());
            g_sycl_pp_config.stage_output_buf[i] = nullptr;
        }
    }

    // Reset config
    g_sycl_pp_config = {};
}

int ggml_sycl_pp_get_device_for_layer(int layer) {
    if (!g_sycl_pp_config.enabled) {
        return 0;  // Default device when PP disabled
    }

    if (layer < 0 || layer >= GGML_SYCL_PP_MAX_LAYERS) {
        GGML_LOG_WARN("SYCL PP: layer %d out of range\n", layer);
        return 0;
    }

    return g_sycl_pp_config.layer_to_device[layer];
}

void * ggml_sycl_pp_ensure_stage_buffer(int stage, size_t size) {
    if (stage < 0 || stage >= g_sycl_pp_config.num_stages) {
        GGML_LOG_ERROR("SYCL PP: Invalid stage %d\n", stage);
        return nullptr;
    }

    // Check if existing buffer is large enough
    if (g_sycl_pp_config.stage_output_buf[stage] != nullptr && g_sycl_pp_config.stage_output_size >= size) {
        return g_sycl_pp_config.stage_output_buf[stage];
    }

    // Free old buffer if exists
    if (g_sycl_pp_config.stage_output_buf[stage] != nullptr) {
        ggml_sycl::unified_cache_sub_runtime_bytes(dpct::dev_mgr::instance().current_device_id(),
                                                   g_sycl_pp_config.stage_output_size);
        sycl::free(g_sycl_pp_config.stage_output_buf[stage], dpct::get_in_order_queue());
    }

    // Allocate shared memory for inter-stage transfer (Intel Arc has no P2P)
    // Using malloc_shared allows both source and destination devices to access
    ggml_sycl::unified_cache_add_runtime_bytes(dpct::dev_mgr::instance().current_device_id(), size);
    g_sycl_pp_config.stage_output_buf[stage] =
        ggml_sycl_malloc_shared_t<char>(size, dpct::get_in_order_queue(), "pp_stage_output");
    if (!g_sycl_pp_config.stage_output_buf[stage]) {
        ggml_sycl::unified_cache_sub_runtime_bytes(dpct::dev_mgr::instance().current_device_id(), size);
    }
    g_sycl_pp_config.stage_output_size       = size;

    PP_DEBUG("Allocated stage %d buffer: %zu bytes (malloc_shared)\n", stage, size);

    return g_sycl_pp_config.stage_output_buf[stage];
}

sycl::event ggml_sycl_pp_stage_transfer(int          src_device,
                                        int          dst_device,
                                        const void * src,
                                        size_t       size,
                                        queue_ptr    src_queue,
                                        queue_ptr    dst_queue) {
    if (!g_sycl_pp_config.enabled) {
        return sycl::event();
    }

    // Find the stage index for src_device
    int src_stage = ggml_sycl_pp_get_stage_for_layer(0);  // Will find by device
    for (int i = 0; i < g_sycl_pp_config.num_stages; i++) {
        if (g_sycl_pp_config.devices[i] == src_device) {
            src_stage = i;
            break;
        }
    }

    // Ensure we have a stage buffer
    void * stage_buf = ggml_sycl_pp_ensure_stage_buffer(src_stage, size);
    if (stage_buf == nullptr) {
        GGML_LOG_ERROR("SYCL PP: Failed to allocate stage buffer\n");
        return sycl::event();
    }

    // Copy from source device to shared staging buffer
    sycl::event copy_event = src_queue->memcpy(stage_buf, src, size);

    // Record the completion event
    g_sycl_pp_config.stage_complete[src_stage] = copy_event;
    g_sycl_pp_config.total_stage_transfers++;

    PP_DEBUG_VERBOSE("Stage transfer: device %d -> staging (%zu bytes)\n", src_device, size);

    GGML_UNUSED(dst_device);
    GGML_UNUSED(dst_queue);
    return copy_event;
}

void ggml_sycl_pp_sync_stage(int stage) {
    if (!g_sycl_pp_config.enabled) {
        return;
    }

    if (stage < 0 || stage >= g_sycl_pp_config.num_stages) {
        return;
    }

    // Wait for stage completion event
    g_sycl_pp_config.stage_complete[stage].wait();
    g_sycl_pp_config.total_sync_waits++;

    PP_DEBUG_VERBOSE("Synced stage %d\n", stage);
}

void ggml_sycl_pp_sync_all() {
    if (!g_sycl_pp_config.enabled) {
        return;
    }

    for (int i = 0; i < g_sycl_pp_config.num_stages; i++) {
        g_sycl_pp_config.stage_complete[i].wait();
    }

    PP_DEBUG_VERBOSE("Synced all %d stages\n", g_sycl_pp_config.num_stages);
}

bool ggml_sycl_pp_enabled() {
    return g_sycl_pp_config.enabled && g_sycl_pp_config.num_stages > 1;
}

int ggml_sycl_pp_num_stages() {
    return g_sycl_pp_config.enabled ? g_sycl_pp_config.num_stages : 1;
}

void ggml_sycl_pp_get_stage_layers(int stage, int * start_layer, int * end_layer) {
    if (!g_sycl_pp_config.enabled || stage < 0 || stage >= g_sycl_pp_config.num_stages) {
        *start_layer = 0;
        *end_layer   = 0;
        return;
    }

    // Calculate start layer by summing previous stages
    int start = 0;
    for (int i = 0; i < stage; i++) {
        start += g_sycl_pp_config.layers_per_stage[i];
    }

    *start_layer = start;
    *end_layer   = start + g_sycl_pp_config.layers_per_stage[stage];
}

int ggml_sycl_pp_get_stage_for_layer(int layer) {
    if (!g_sycl_pp_config.enabled) {
        return 0;
    }

    int target_device = g_sycl_pp_config.layer_to_device[layer];

    for (int i = 0; i < g_sycl_pp_config.num_stages; i++) {
        if (g_sycl_pp_config.devices[i] == target_device) {
            return i;
        }
    }

    return 0;
}

void ggml_sycl_pp_set_chunked_prefill(int32_t chunk_size, bool enabled) {
    g_sycl_pp_config.chunk_size              = chunk_size;
    g_sycl_pp_config.chunked_prefill_enabled = enabled;

    PP_DEBUG("Chunked prefill: %s (chunk_size=%d)\n", enabled ? "enabled" : "disabled", chunk_size);
}

// Get the staging buffer for reading (after transfer is complete)
void * ggml_sycl_pp_get_stage_buffer(int stage) {
    if (stage < 0 || stage >= g_sycl_pp_config.num_stages) {
        return nullptr;
    }
    return g_sycl_pp_config.stage_output_buf[stage];
}

// Get statistics for debugging/profiling
void ggml_sycl_pp_get_stats(int64_t * transfers, int64_t * syncs) {
    if (transfers) {
        *transfers = g_sycl_pp_config.total_stage_transfers;
    }
    if (syncs) {
        *syncs = g_sycl_pp_config.total_sync_waits;
    }
}

void ggml_sycl_pp_reset_stats() {
    g_sycl_pp_config.total_stage_transfers = 0;
    g_sycl_pp_config.total_sync_waits      = 0;
}
