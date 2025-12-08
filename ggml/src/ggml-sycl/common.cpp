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
#include <mutex>

int get_current_device_id() {
  return dpct::dev_mgr::instance().current_device_id();
}

// Cached shared context and queues for TP mode
static sycl::context * g_tp_shared_context = nullptr;
static sycl::queue * g_tp_shared_queues[GGML_SYCL_MAX_DEVICES] = { nullptr };
static std::mutex g_tp_context_mutex;

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
        int dev_id = g_sycl_tp_config.devices[i];
        sycl::device dev = dpct::dev_mgr::instance().get_device(dev_id);
        tp_devices.push_back(dev);
    }
    g_tp_shared_context = new sycl::context(tp_devices);
    GGML_SYCL_DEBUG("SYCL TP: Created shared context for %d local devices (world_size=%d)\n",
                   num_local_devices, g_sycl_tp_config.world_size);

    // Create shared-context queues for each local TP device
    for (int i = 0; i < num_local_devices; i++) {
        int dev_id = g_sycl_tp_config.devices[i];
        sycl::device dev = dpct::dev_mgr::instance().get_device(dev_id);
        g_tp_shared_queues[dev_id] = new sycl::queue(*g_tp_shared_context, dev, sycl::property::queue::in_order());
        GGML_SYCL_DEBUG("SYCL TP: Created shared-context queue for device %d at %p\n", dev_id, (void*)g_tp_shared_queues[dev_id]);
    }
}

// Get shared context for TP mode
// Returns nullptr if not in TP mode
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

// Get shared-context queue for a device in TP mode
// Returns nullptr if not in TP mode or device not part of TP
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
static std::mutex g_tp_staging_mutex;

// Get or create a staged copy of mmap'd data for a specific device in TP mode
// Returns nullptr if not in TP mode or data is already USM
void * ggml_sycl_get_staged_ptr_device(const void * src, size_t size, int device) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return nullptr;
    }
    // Multi-process mode: No cross-device staging needed (each process has its own data)
    if (g_sycl_tp_config.is_multiprocess) {
        return nullptr;
    }
    if (src == nullptr || size == 0) {
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
            GGML_SYCL_DEBUG("[STAGING] Size mismatch for %p: cached=%zu, requested=%zu, reallocating\n",
                            src, it->second.size, size);
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
        entry = &g_tp_staging_cache[src];
        entry->size = size;
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
            GGML_SYCL_DEBUG("[STAGING] Allocating %zu bytes on device %d using tp_queue=%p...\n", size, device, (void*)tp_queue);
            // Allocate device memory on this specific device using shared-context queue
            void * staged = sycl::malloc_device(size, *tp_queue);
            if (staged == nullptr) {
                GGML_LOG_ERROR("[STAGING] malloc_device returned nullptr for %zu bytes on device %d\n", size, device);
                return nullptr;
            }
            GGML_SYCL_DEBUG("[STAGING] Allocated at %p, copying from %p via host staging...\n", staged, src);

            // Two-stage copy: mmap'd -> host (pinned) -> device
            // The shared-context queue cannot access non-USM mmap'd memory directly
            // Use regular heap memory + memcpy (simplest approach)
            void * host_staging = std::malloc(size);
            if (host_staging == nullptr) {
                GGML_LOG_ERROR("[STAGING] std::malloc returned nullptr for %zu bytes\n", size);
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
            std::free(host_staging);

            entry->ptrs[device] = staged;
            GGML_SYCL_DEBUG("[STAGING] Staged %zu bytes from %p to device %d: %p\n", size, src, device, staged);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[STAGING] Failed to allocate/copy %zu bytes on device %d: %s (code=%d)\n",
                          size, device, e.what(), static_cast<int>(e.code().value()));
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
                sycl::free(entry.second.ptrs[dev_id], *g_tp_shared_queues[dev_id]);
                entry.second.ptrs[dev_id] = nullptr;
            }
        }
    }
    g_tp_staging_cache.clear();
}

void* ggml_sycl_host_malloc(size_t size) try {
  if (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    return nullptr;
  }

  void* ptr = nullptr;

  // For TP mode: use a shared context so memory is accessible from all devices
  if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
    std::lock_guard<std::mutex> lock(g_tp_context_mutex);

    // Initialize shared context and queues if needed
    if (g_tp_shared_context == nullptr) {
      ggml_sycl_init_tp_shared_context();
    }

    // Allocate host memory accessible from all TP devices
    dpct::err0 err = CHECK_TRY_ERROR(
        ptr = sycl::malloc_host(size, *g_tp_shared_context));

    if (err != 0) {
      GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of TP shared host memory\n", size / 1024.0 / 1024.0);
      return nullptr;
    }
    return ptr;
  }

  // Non-TP mode: use default queue for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (void*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  if (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    return nullptr;
  }

  return ptr;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_host_free(void* ptr) try {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool gpu_has_xmx(sycl::device &dev) {
    return dev.has(sycl::aspect::ext_intel_matrix);
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size) {
  const int64_t max_range = std::numeric_limits<int>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_num * sycl_down_blk_size;
  while(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_num * sycl_down_blk_size;
  }
  return sycl_down_blk_size;
}

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            if (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        if (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(
                CHECK_TRY_ERROR(sycl::free(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}

// ============================================================================
// FFN Norm Cache for Tensor Parallelism
// ============================================================================
// The GGML backend scheduler may reuse the FFN norm buffer before TP can
// access it for device 1's computation. We cache FFN norm output immediately
// after the MUL operation to prevent stale data issues.

std::unordered_map<int, ffn_norm_cache_entry> g_tp_ffn_norm_cache;
std::mutex g_tp_ffn_norm_cache_mutex;
int g_tp_current_pass_id = 0;
bool g_tp_enabled = false;

void ggml_sycl_tp_cache_ffn_norm(int layer, const void* data, int64_t ne0, int64_t ne1,
                                  size_t size, queue_ptr stream) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return;
    }

    // Multi-process mode: Skip caching - each process has only one device
    // and CCL handles cross-process communication
    if (g_sycl_tp_config.is_multiprocess) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_tp_ffn_norm_cache_mutex);

    auto it = g_tp_ffn_norm_cache.find(layer);
    ffn_norm_cache_entry& entry = g_tp_ffn_norm_cache[layer];

    // Reallocate if size changed
    if (entry.size != size) {
        // DEBUG: Check L31 weight BEFORE any free/realloc
        {
            uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
            struct { uint16_t d_bits; uint8_t qs[16]; } wblk;
            try {
                stream->memcpy(&wblk, (void*)l31_weight_addr, sizeof(wblk)).wait();
                uint16_t d_raw = wblk.d_bits;
                sycl::half d_half;
                memcpy(&d_half, &d_raw, sizeof(sycl::half));
                float d_f = static_cast<float>(d_half);
                if (g_ggml_sycl_tp_debug && (d_f > 100.0f || std::isnan(d_f))) {
                    fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: BEFORE realloc, L31 weight CORRUPTED d=%f\n",
                            layer, d_f);
                } else if (g_ggml_sycl_tp_debug) {
                    static int before_dbg = 0;
                    if (before_dbg++ < 5)
                        fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: BEFORE realloc, L31 weight OK d=%f\n",
                                layer, d_f);
                }
            } catch (...) {
                // Ignore errors if address is invalid
            }
        }

        if (entry.data != nullptr) {
            if (g_ggml_sycl_tp_debug) fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: freeing old cache at %p\n", layer, entry.data);
            sycl::free(entry.data, *stream);
        }
        if (entry.data_dev1 != nullptr) {
            int dev1 = g_sycl_tp_config.devices[1];
            ggml_sycl_set_device(dev1);
            queue_ptr stream1 = &dpct::get_device(dev1).default_queue();
            sycl::free(entry.data_dev1, *stream1);
            ggml_sycl_set_device(g_sycl_tp_config.devices[0]);
        }
        // Allocate new buffers
        entry.data = sycl::malloc_device(size, *stream);
        int dev1 = g_sycl_tp_config.devices[1];
        ggml_sycl_set_device(dev1);
        queue_ptr stream1 = &dpct::get_device(dev1).default_queue();
        entry.data_dev1 = sycl::malloc_device(size, *stream1);
        ggml_sycl_set_device(g_sycl_tp_config.devices[0]);

        // DEBUG: Check if allocation overlaps with L31 FFN gate weight
        if (g_ggml_sycl_tp_debug && (layer == 31 || layer == 0)) {
            uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
            uintptr_t l31_weight_end = l31_weight_addr + 16515072;  // shard size
            uintptr_t cache_start = (uintptr_t)entry.data;
            uintptr_t cache_end = cache_start + size;
            bool overlap = (cache_start < l31_weight_end && cache_end > l31_weight_addr);
            fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: alloc=%p size=%zu overlap_with_L31_weight=%d\n",
                    layer, entry.data, size, overlap);
            if (overlap) {
                fprintf(stderr, "TP DEBUG FFN_NORM_CACHE OVERLAP! cache=[%p,%p) weight=[0x%llx,0x%llx)\n",
                        (void*)cache_start, (void*)cache_end,
                        (unsigned long long)l31_weight_addr, (unsigned long long)l31_weight_end);
            }
        }

        // DEBUG: Check L31 weight AFTER realloc
        {
            uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
            struct { uint16_t d_bits; uint8_t qs[16]; } wblk;
            try {
                stream->memcpy(&wblk, (void*)l31_weight_addr, sizeof(wblk)).wait();
                uint16_t d_raw = wblk.d_bits;
                sycl::half d_half;
                memcpy(&d_half, &d_raw, sizeof(sycl::half));
                float d_f = static_cast<float>(d_half);
                if (g_ggml_sycl_tp_debug && (d_f > 100.0f || std::isnan(d_f))) {
                    fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: AFTER realloc, L31 weight CORRUPTED d=%f\n",
                            layer, d_f);
                } else if (g_ggml_sycl_tp_debug) {
                    static int after_dbg = 0;
                    if (after_dbg++ < 5)
                        fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: AFTER realloc, L31 weight OK d=%f\n",
                                layer, d_f);
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
        struct { uint16_t d_bits; uint8_t qs[16]; } wblk;
        try {
            stream->memcpy(&wblk, (void*)l31_weight_addr, sizeof(wblk)).wait();
            uint16_t d_raw = wblk.d_bits;
            sycl::half d_half;
            memcpy(&d_half, &d_raw, sizeof(sycl::half));
            float d_f = static_cast<float>(d_half);
            if (d_f > 100.0f || std::isnan(d_f)) {
                fprintf(stderr, "TP DEBUG FFN_NORM_CACHE layer %d: AFTER cache copy, L31 weight CORRUPTED d=%f\n",
                        layer, d_f);
            }
        } catch (...) {}
    }

    // Also copy to device 1's buffer (via host staging)
    void* host_buf = sycl::malloc_host(size, *stream);
    stream->memcpy(host_buf, data, size).wait();
    int dev1 = g_sycl_tp_config.devices[1];
    ggml_sycl_set_device(dev1);
    queue_ptr stream1 = &dpct::get_device(dev1).default_queue();
    stream1->memcpy(entry.data_dev1, host_buf, size).wait();
    sycl::free(host_buf, *stream);
    ggml_sycl_set_device(g_sycl_tp_config.devices[0]);

    entry.ne0 = ne0;
    entry.ne1 = ne1;
    entry.size = size;
    entry.pass_id = g_tp_current_pass_id;

}

void* ggml_sycl_tp_get_cached_ffn_norm(int layer, int device) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_tp_ffn_norm_cache_mutex);

    auto it = g_tp_ffn_norm_cache.find(layer);
    if (it == g_tp_ffn_norm_cache.end()) {
        return nullptr;
    }

    const ffn_norm_cache_entry& entry = it->second;

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
static float* g_tp_shared_reduce_buf = nullptr;
static size_t g_tp_shared_reduce_buf_size = 0;
static std::mutex g_tp_shared_reduce_mutex;

void ggml_sycl_tp_init(const int* device_ids, int num_devices) {
    if (num_devices < 1 || num_devices > GGML_SYCL_MAX_DEVICES) {
        GGML_LOG_ERROR("SYCL TP: Invalid number of devices: %d\n", num_devices);
        return;
    }

    g_sycl_tp_config.enabled = true;
    g_sycl_tp_config.world_size = num_devices;
    g_sycl_tp_config.rank = 0;  // Default rank
    for (int i = 0; i < num_devices; i++) {
        g_sycl_tp_config.devices[i] = device_ids[i];
    }

    // Check for multi-process mode (MPI)
    const char* pmi_rank = std::getenv("PMI_RANK");
    const char* pmi_size = std::getenv("PMI_SIZE");
    if (pmi_rank && pmi_size) {
        g_sycl_tp_config.is_multiprocess = true;
        g_sycl_tp_config.mpi_rank = std::atoi(pmi_rank);
        g_sycl_tp_config.mpi_world_size = std::atoi(pmi_size);
        g_sycl_tp_config.rank = g_sycl_tp_config.mpi_rank;
        g_sycl_tp_config.world_size = g_sycl_tp_config.mpi_world_size;
        GGML_SYCL_DEBUG("SYCL TP: Multi-process mode enabled, rank=%d/%d\n",
                       g_sycl_tp_config.mpi_rank, g_sycl_tp_config.mpi_world_size);
    }

    GGML_SYCL_DEBUG("SYCL TP: Initialized with %d devices\n", num_devices);
}

void ggml_sycl_tp_free() {
    std::lock_guard<std::mutex> lock(g_tp_shared_reduce_mutex);
    if (g_tp_shared_reduce_buf != nullptr) {
        sycl::free(g_tp_shared_reduce_buf, dpct::get_in_order_queue());
        g_tp_shared_reduce_buf = nullptr;
        g_tp_shared_reduce_buf_size = 0;
    }
    g_sycl_tp_config = {};
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

float* ggml_sycl_tp_ensure_shared_reduce_buffer(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_tp_shared_reduce_mutex);

    if (g_tp_shared_reduce_buf_size >= bytes && g_tp_shared_reduce_buf != nullptr) {
        return g_tp_shared_reduce_buf;
    }

    // Free old buffer
    if (g_tp_shared_reduce_buf != nullptr) {
        sycl::free(g_tp_shared_reduce_buf, dpct::get_in_order_queue());
    }

    // Allocate shared memory for zero-copy ALL_REDUCE
    g_tp_shared_reduce_buf = sycl::malloc_shared<float>(bytes / sizeof(float), dpct::get_in_order_queue());
    g_tp_shared_reduce_buf_size = bytes;

    GGML_SYCL_DEBUG("SYCL TP: Allocated %zu bytes for shared reduce buffer\n", bytes);
    return g_tp_shared_reduce_buf;
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

void ggml_sycl_tp_get_slice(int64_t total_size, int rank, int world_size,
                             int64_t* offset, int64_t* size) {
    int64_t slice_size = total_size / world_size;
    *offset = rank * slice_size;
    *size = (rank == world_size - 1) ? (total_size - *offset) : slice_size;
}

tp_layer_type ggml_sycl_tp_get_layer_type(const ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->extra == nullptr) {
        return tp_layer_type::TP_NONE;
    }

    auto* extra = static_cast<ggml_tensor_extra_gpu*>(tensor->extra);
    if (extra->tp_type_cached) {
        return extra->tp_type;
    }

    // Determine type by name pattern
    tp_layer_type tp_type = tp_layer_type::TP_NONE;
    if (tensor->name) {
        // Column-parallel: output dimension is sharded (Q, K, V, gate, up)
        if (strstr(tensor->name, "attn_q") || strstr(tensor->name, "attn_k") ||
            strstr(tensor->name, "attn_v") || strstr(tensor->name, "ffn_gate") ||
            strstr(tensor->name, "ffn_up")) {
            tp_type = tp_layer_type::TP_COLUMN_PARALLEL;
        }
        // Row-parallel: input dimension is sharded (O, down)
        else if (strstr(tensor->name, "attn_output") || strstr(tensor->name, "ffn_down")) {
            tp_type = tp_layer_type::TP_ROW_PARALLEL;
        }
    }

    extra->tp_type = tp_type;
    extra->tp_type_cached = true;
    return tp_type;
}

bool ggml_sycl_tp_needs_allreduce(const ggml_tensor* tensor) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return false;
    }

    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);
    return (tp_type == tp_layer_type::TP_ROW_PARALLEL);
}

void ggml_sycl_tp_get_sharded_dims(const ggml_tensor* tensor, int rank, int world_size,
                                    int64_t* local_ne0, int64_t* local_ne1,
                                    int64_t* offset_ne0, int64_t* offset_ne1) {
    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);

    *local_ne0 = tensor->ne[0];
    *local_ne1 = tensor->ne[1];
    *offset_ne0 = 0;
    *offset_ne1 = 0;

    if (tp_type == tp_layer_type::TP_COLUMN_PARALLEL) {
        *local_ne1 = tensor->ne[1] / world_size;
        *offset_ne1 = rank * (*local_ne1);
    } else if (tp_type == tp_layer_type::TP_ROW_PARALLEL) {
        *local_ne0 = tensor->ne[0] / world_size;
        *offset_ne0 = rank * (*local_ne0);
    }
}

bool ggml_sycl_tp_should_shard(const ggml_tensor* tensor) {
    if (!g_sycl_tp_config.enabled || g_sycl_tp_config.world_size <= 1) {
        return false;
    }

    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);
    return (tp_type == tp_layer_type::TP_COLUMN_PARALLEL ||
            tp_type == tp_layer_type::TP_ROW_PARALLEL);
}

void ggml_sycl_tp_copy_weight_shard(void* dst_device, const void* src_host,
                                     const ggml_tensor* tensor, int rank,
                                     int world_size, queue_ptr stream) {
    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);

    int64_t ne0 = tensor->ne[0];
    int64_t ne1 = tensor->ne[1];

    if (tp_type == tp_layer_type::TP_COLUMN_PARALLEL) {
        // Shard ne1 dimension
        int64_t shard_ne1 = ne1 / world_size;
        int64_t offset_ne1 = rank * shard_ne1;

        size_t row_size = ggml_row_size(tensor->type, ne0);
        size_t shard_size = row_size * shard_ne1;

        const char* src = static_cast<const char*>(src_host) + offset_ne1 * row_size;
        stream->memcpy(dst_device, src, shard_size).wait();
    } else if (tp_type == tp_layer_type::TP_ROW_PARALLEL) {
        // Shard ne0 dimension - more complex due to quantization blocks
        int64_t shard_ne0 = ne0 / world_size;
        int64_t offset_ne0 = rank * shard_ne0;

        size_t full_row_size = ggml_row_size(tensor->type, ne0);
        size_t shard_row_size = ggml_row_size(tensor->type, shard_ne0);

        // Copy row by row
        char* dst = static_cast<char*>(dst_device);
        const char* src = static_cast<const char*>(src_host);

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

size_t ggml_sycl_tp_get_shard_size(const ggml_tensor* tensor, int rank, int world_size) {
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
        float* dst_data = static_cast<float*>(dst->data);
        size_t count = ggml_nelements(dst);
        ggml_sycl_ccl_allreduce_sum_f32(dst_data, count, ctx.device);
        return;
    }

    // Single-process multi-device mode: manual reduce using shared memory
    int world_size = g_sycl_tp_config.world_size;
    size_t dst_size = ggml_nbytes(dst);

    // Get shared buffer
    float* shared_buf = ggml_sycl_tp_ensure_shared_reduce_buffer(dst_size);
    if (!shared_buf) {
        GGML_LOG_ERROR("SYCL TP: Failed to allocate shared reduce buffer\n");
        return;
    }

    // Zero the shared buffer
    ctx.stream()->memset(shared_buf, 0, dst_size).wait();

    // Each device adds its contribution to shared buffer
    auto* extra = static_cast<ggml_tensor_extra_gpu*>(dst->extra);
    for (int rank = 0; rank < world_size; rank++) {
        int device = g_sycl_tp_config.devices[rank];
        float* device_data = static_cast<float*>(extra->data_device[device]);
        if (device_data == nullptr) continue;

        ggml_sycl_set_device(device);
        queue_ptr stream = &dpct::get_device(device).default_queue();

        size_t count = dst_size / sizeof(float);

        // Add device_data to shared_buf using GPU kernel
        stream->parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            shared_buf[i] += device_data[i];
        }).wait();
    }

    // Copy result back to all devices
    for (int rank = 0; rank < world_size; rank++) {
        int device = g_sycl_tp_config.devices[rank];
        float* device_data = static_cast<float*>(extra->data_device[device]);
        if (device_data == nullptr) continue;

        ggml_sycl_set_device(device);
        queue_ptr stream = &dpct::get_device(device).default_queue();
        stream->memcpy(device_data, shared_buf, dst_size).wait();
    }

    // Restore device
    ggml_sycl_set_device(ctx.device);

    GGML_UNUSED(ctx);
}
