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

#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <regex>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-sycl.h"
#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/convert.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/fattn.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/mmq.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set_rows_paged.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/sycl_hw.hpp"

#include <sycl/half_type.hpp>
#ifdef GGML_SYCL_MMQ_XMX
#    include "ggml-sycl/mmq_xmx.hpp"
#endif
#include "ggml-sycl/cont-batching.hpp"
#include "ggml-sycl/fused-ffn.hpp"
#include "ggml-sycl/fused-moe-esimd.hpp"
#include "ggml-sycl/fused-norm-gemm.hpp"
#include "ggml-sycl/gpu-sampler.hpp"
#include "ggml-sycl/mmvq.hpp"
#include "ggml-sycl/moe-sort.hpp"
#include "ggml-sycl/moe-xmx-fused.hpp"
#include "ggml-sycl/moe-xmx.hpp"
#include "ggml-sycl/quantized-comm.hpp"
#include "ggml-sycl/sycl-profiling.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"

static bool g_sycl_loaded                = false;
int         g_ggml_sycl_debug            = 0;
int         g_ggml_sycl_tp_debug         = 0;  // Tensor Parallelism debug output (GGML_SYCL_TP_DEBUG env var)
int         g_ggml_sycl_tp_async_ffn     = 0;  // Async FFN pipelining (DISABLED - causes hangs)
int         g_ggml_sycl_disable_graph    = 0;
int         g_ggml_sycl_disable_dnn      = 0;
int         g_ggml_sycl_prioritize_dmmv  = 0;
int         g_ggml_sycl_use_async_mem_op = 0;
int         g_ggml_sycl_gpu_reorder      = 0;           // Use GPU AoS->SoA reorder during upload (default off)
#ifdef GGML_SYCL_XMX_GEMM
int g_ggml_sycl_use_xmx_gemm  = 0;                      // Enable XMX-accelerated GEMM (experimental, 5-11x slower)
int g_ggml_sycl_xmx_threshold = 1024;                   // Max batch size for XMX (XMX faster for N < threshold)
#endif
thread_local bool g_ggml_sycl_graph_recording = false;  // True when SYCL graph is recording
reorder_mode      g_ggml_sycl_reorder_mode    = reorder_mode::SOA;  // Default to SoA (existing behavior)

// Maximum batch size for fused ESIMD MoE kernel.
// Larger batches (e.g., prefill with 512 tokens) fall back to host-side oneDNN batching
// which is ~7x faster for large batches (679 t/s vs 87 t/s on GPT-OSS 20B).
// The fused kernel excels at decode (single-token) but is slower than oneDNN for prefill.
constexpr int64_t GGML_SYCL_FUSED_MOE_MAX_BATCH = 32;

// Model load phase flag - when true, skip weight caching to avoid OOM during load
static std::atomic<bool> g_sycl_in_model_load{ false };

void ggml_backend_sycl_set_model_loading(bool loading) {
    g_sycl_in_model_load.store(loading, std::memory_order_release);
}

// Cross-device staging cache for multi-GPU (Intel Arc has no P2P)
// Key: (slot_idx << 32 | target_device) -> staged pointer
std::unordered_map<uint64_t, void *> g_expert_staging;
std::mutex                           g_expert_staging_mutex;

// Forward declaration for layer number extraction
static int extract_layer_number(const char * name);

// Forward declarations for device-side AoS -> SoA reorder and helpers
static inline void * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size);
static inline void   sycl_ext_free(dpct::queue_ptr stream, void * ptr);
static bool          reorder_aos_to_soa_device(const ggml_tensor * tensor,
                                               const void *        src_dev,
                                               void *              dst_dev,
                                               size_t              size,
                                               dpct::queue_ptr     stream);

static ggml_sycl_device_info ggml_sycl_init() {
    ggml_sycl_device_info info = {};

    info.device_count = dpct::dev_mgr::instance().device_count();
    if (info.device_count == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
    /* This is a bit misleading;  reserved for later */
    // #if defined(SYCL_USE_XMX)
    //     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
    // #else
    //     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
    // #endif
    for (int i = 0; i < info.device_count; ++i) {
        info.devices[i].vmm = 0;
        dpct::device_info prop;
        sycl::device      device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(prop, device)));

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc                   = 100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm                  = prop.get_max_compute_units();
        // Device-level capability: Intel GPUs support SoA weight reordering optimizations
        info.devices[i].supports_soa_reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
        info.devices[i].smpbo                = prop.get_local_mem_size();

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();

        // Query XMX (Intel matrix engine) capabilities
        info.devices[i].xmx_caps = query_xmx_capabilities(device);
        GGML_LOG_INFO("[SYCL] Device %d XMX: %s, M=%zu N=%zu K=%zu, SLM=%zuKB\n", i,
                      info.devices[i].xmx_caps.supported ? "yes" : "no", info.devices[i].xmx_caps.M,
                      info.devices[i].xmx_caps.N, info.devices[i].xmx_caps.K, info.devices[i].xmx_caps.slm_size / 1024);
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }
    return info;
}

const ggml_sycl_device_info & ggml_sycl_info() {
    static ggml_sycl_device_info info = ggml_sycl_init();
    return info;
}

static void print_device_detail(int id, sycl::device & device, std::string device_type) {
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type      = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name             = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name             = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size() / 1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(), name.c_str(),
                  version.c_str(), prop.get_max_compute_units(), prop.get_max_work_group_size(),
                  prop.get_max_sub_group_size(), global_mem_size,
                  device.get_info<sycl::info::device::driver_version>().c_str());
}

static void print_device_opt_feature(int device_count) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO("|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO("|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    for (int id = 0; id < device_count; ++id) {
        sycl::device      device       = dpct::dev_mgr::instance().get_device(id);
        std::string       backend_type = get_device_backend_and_type(device);
        int               type_id      = DeviceNums[backend_type]++;
        std::stringstream device_type;
        device_type << "[" << backend_type << ":" << std::to_string(type_id) << "]";
        std::string device_type_s = device_type.str();
        device_type_s             = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
        GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
                      ggml_sycl_info().devices[id].supports_soa_reorder ? "Y" : "N");
    }
}

void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int                           device_count = dpct::dev_mgr::instance().device_count();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_count);

    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |Max    |        |Max  |Global |                     |\n");
    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |compute|Max work|sub  |mem    |                     |\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|                                   "
        "Name|Version|units  |group   |group|size   |       Driver version|\n");
    GGML_LOG_INFO(
        "|--|-------------------|---------------------------------------|------"
        "-|-------|--------|-----|-------|---------------------|\n");

    for (int id = 0; id < device_count; ++id) {
        sycl::device      device       = dpct::dev_mgr::instance().get_device(id);
        std::string       backend_type = get_device_backend_and_type(device);
        int               type_id      = DeviceNums[backend_type]++;
        std::stringstream device_type;
        device_type << "[" << backend_type << ":" << std::to_string(type_id) << "]";
        print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_count);
}

static inline int get_sycl_env(const char * env_name, int default_val) {
    char * user_device_string = getenv(env_name);
    int    user_number        = default_val;

    unsigned n;
    if (user_device_string != NULL && sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (int) n;
    } else {
        user_number = default_val;
    }
    return user_number;
}

// Get reorder mode from environment variable GGML_SYCL_REORDER_MODE
// Valid values: "none", "soa", "coalesced" (case-sensitive)
// Default: "coalesced" - 35% faster tg128 vs none (42.59 vs 31.57 t/s on Arc A770)
static reorder_mode get_reorder_mode() {
    const char * mode = getenv("GGML_SYCL_REORDER_MODE");
    if (mode == nullptr) {
        return reorder_mode::COALESCED;  // Default: coalesced for best decode perf
    }
    if (strcmp(mode, "none") == 0) {
        return reorder_mode::NONE;
    }
    if (strcmp(mode, "soa") == 0) {
        return reorder_mode::SOA;
    }
    if (strcmp(mode, "coalesced") == 0) {
        return reorder_mode::COALESCED;
    }
    fprintf(stderr, "WARN: Unknown GGML_SYCL_REORDER_MODE '%s', using coalesced\n", mode);
    return reorder_mode::COALESCED;
}

static const char * reorder_mode_to_string(reorder_mode mode) {
    switch (mode) {
        case reorder_mode::NONE:
            return "none";
        case reorder_mode::SOA:
            return "soa";
        case reorder_mode::COALESCED:
            return "coalesced";
        default:
            return "unknown";
    }
}

static void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        g_ggml_sycl_debug           = get_sycl_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_tp_debug        = get_sycl_env("GGML_SYCL_TP_DEBUG", 0);
        g_ggml_sycl_disable_graph   = get_sycl_env("GGML_SYCL_DISABLE_GRAPH", 0);
        g_ggml_sycl_disable_dnn     = get_sycl_env("GGML_SYCL_DISABLE_DNN", 0);
        g_ggml_sycl_prioritize_dmmv = get_sycl_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
        g_ggml_sycl_gpu_reorder     = get_sycl_env("GGML_SYCL_GPU_REORDER", 0);
#ifdef GGML_SYCL_XMX_GEMM
        g_ggml_sycl_use_xmx_gemm  = get_sycl_env("GGML_SYCL_USE_XMX_GEMM", 0);
        g_ggml_sycl_xmx_threshold = get_sycl_env("GGML_SYCL_XMX_THRESHOLD", 64);
#endif
        g_ggml_sycl_reorder_mode = get_reorder_mode();
        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_TP_DEBUG: %d\n", g_ggml_sycl_tp_debug);
#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_GRAPH: %d\n", g_ggml_sycl_disable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_GRAPH: graph disabled by compile flag\n");
#endif
#if GGML_SYCL_DNNL
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_DNN: %d\n", g_ggml_sycl_disable_dnn);
#else
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_DNN: DNN disabled by compile flag\n");
#endif
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);
        GGML_LOG_INFO("  GGML_SYCL_GPU_REORDER: %d\n", g_ggml_sycl_gpu_reorder);
        GGML_LOG_INFO("  GGML_SYCL_REORDER_MODE: %s\n", reorder_mode_to_string(g_ggml_sycl_reorder_mode));
#ifdef GGML_SYCL_XMX_GEMM
        GGML_LOG_INFO("  GGML_SYCL_USE_XMX_GEMM: %d (experimental, 5-11x slower)\n", g_ggml_sycl_use_xmx_gemm);
        if (g_ggml_sycl_use_xmx_gemm) {
            GGML_LOG_INFO("  GGML_SYCL_XMX_THRESHOLD: %d (XMX for batch < %d)\n", g_ggml_sycl_xmx_threshold,
                          g_ggml_sycl_xmx_threshold);
        }
#endif
        GGML_LOG_INFO("Build with Macros:\n");
#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif
#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

        /* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/
        // Currently, we only use async malloc / free when graphs are enabled AND the graph is actually used.
        // async_mem_op is now controlled separately via GGML_SYCL_ASYNC_MEM environment variable,
        // defaulting to OFF to avoid issues with models that don't use SYCL graphs (like MoE models).
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        // Default to disabled - only enable if explicitly requested
        g_ggml_sycl_use_async_mem_op = get_sycl_env("GGML_SYCL_ASYNC_MEM", 0);
        if (g_ggml_sycl_use_async_mem_op) {
            for (unsigned int i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) {
                if (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    break;
                }
            }
        }
#endif
        if (CHECK_TRY_ERROR(g_all_sycl_device_count = dpct::dev_mgr::instance().device_count()) != 0) {
            initialized   = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        initialized   = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
inline void check_allow_gpu_index(const int device_index) {
    if (device_index >= ggml_sycl_info().device_count) {
        char error_buf[256];
        snprintf(error_buf, sizeof(error_buf), "%s error: device_index:%d is out of range: [0-%d]", __func__,
                 device_index, ggml_sycl_info().device_count - 1);
        GGML_LOG_ERROR("%s\n", error_buf);
        assert(false);
    }
}

GGML_API void ggml_backend_sycl_get_gpu_list(int * id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    for (int i = 0; i < max_len; i++) {
        id_list[i] = -1;
    }

    for (int i = 0; i < ggml_sycl_info().device_count; i++) {
        if (i >= max_len) {
            break;
        }
        id_list[i] = i;
    }
    return;
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int         device;
    void *      dev_ptr = nullptr;
    queue_ptr   stream;
    std::string name;
    bool        supports_soa_reorder;  // Device capability (not tensor state)
    // Track both tensor and extra so we can null tensor->extra on reset
    std::vector<std::pair<ggml_tensor *, ggml_tensor_extra_gpu *>> tensor_extras;

    // TP compute buffer support: per-device pointers
    // For TP compute buffers, we allocate on ALL TP devices and track base pointers here
    bool      is_tp_compute_buffer               = false;
    void *    tp_dev_ptrs[GGML_SYCL_MAX_DEVICES] = { nullptr };
    queue_ptr tp_streams[GGML_SYCL_MAX_DEVICES]  = { nullptr };

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream) :
        device(device),
        dev_ptr(dev_ptr),
        stream(stream),
        supports_soa_reorder(ggml_sycl_info().devices[device].supports_soa_reorder) {
        check_allow_gpu_index(device);
        name = (GGML_SYCL_NAME + std::to_string(device));
    }

    ~ggml_backend_sycl_buffer_context() {
        // Free TP compute buffer pointers
        if (is_tp_compute_buffer) {
            // In TP mode with shared host memory, all tp_dev_ptrs[] point to the SAME pointer
            // Only free it once via the primary device (dev_ptr)
            if (dev_ptr != nullptr && stream != nullptr) {
                ggml_sycl_set_device(device);
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(dev_ptr, *stream)));
            }
        } else if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(dev_ptr, *stream)));
        }

        // Release extra used by tensors and null tensor->extra to avoid dangling pointers
        for (auto & [tensor, extra] : tensor_extras) {
            if (tensor != nullptr) {
                tensor->extra = nullptr;
            }
            release_extra_gpu(extra);
        }
    }
};

GGML_API void ggml_backend_sycl_memcpy_d2h(const ggml_tensor * tensor, void * dst, size_t size) try {
    if (!tensor || !tensor->buffer || !dst) {
        GGML_ABORT("ggml_backend_sycl_memcpy_d2h: invalid arguments");
    }
    if (!ggml_backend_buffer_is_sycl(tensor->buffer)) {
        GGML_ABORT("ggml_backend_sycl_memcpy_d2h: non-SYCL buffer");
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) tensor->buffer->context;
    ggml_sycl_set_device(ctx->device);
    auto stream = ctx->stream ? ctx->stream : &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst, tensor->data, size).wait()));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

// ============================================================================
// Weight Streaming: Uses unified cache for all weight caching (dense + MoE)
// On first access: staging (mmap → host → device cache via unified cache)
// On subsequent access: fast D2D copy (device cache → tensor)
// ============================================================================

// Get or create cached device copy of mmap'd weight data
// Returns cached device pointer, or nullptr if caching not needed/failed
// key_ptr: stable identifier (tensor->data) that doesn't change across set_tensor calls
// src_ptr: source data pointer (may change if data is overwritten)
static void * get_or_cache_weight(const void *  key_ptr,
                                  const void *  src_ptr,
                                  size_t        size,
                                  int           device,
                                  sycl::queue * stream) {
    (void) device;  // Unified cache handles device selection

    // Check if weight streaming is enabled
    if (!ggml_sycl::unified_cache_enabled() || key_ptr == nullptr || src_ptr == nullptr || size == 0) {
        return nullptr;
    }

    // Skip caching during model load phase to avoid OOM on large models
    if (g_sycl_in_model_load.load(std::memory_order_acquire)) {
        return nullptr;
    }

    // Use unified cache for coordinated memory management
    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(*stream);
    if (!cache) {
        return nullptr;
    }

    // Try to cache with unified cache (handles eviction automatically)
    // Uses key_ptr (tensor->data) as stable key, src_ptr as data source
    // If src_ptr changed since last cache, data will be re-uploaded
    void * cached_ptr = cache->ensure_cached(key_ptr, src_ptr, size, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1,
                                             true);  // validate content for mutable weights

    if (cached_ptr) {
        GGML_SYCL_DEBUG("[UNIFIED-CACHE] Cached dense weight: key=%p src=%p, %zu bytes\n", key_ptr, src_ptr, size);
    }

    return cached_ptr;
}

bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

static void ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        // For TP compute buffers, view tensors also need extra->data_device[] set up
        if (ctx->is_tp_compute_buffer && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
            ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
            if (extra == nullptr) {
                extra         = new ggml_tensor_extra_gpu{};
                tensor->extra = extra;
                ctx->tensor_extras.push_back({ tensor, extra });
            }

            // Calculate offset of this VIEW tensor within the buffer
            ptrdiff_t offset = (char *) tensor->data - (char *) ctx->dev_ptr;

            // Set up data_device[] for each local TP device
            // In multi-process mode: only 1 device is locally visible
            int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
            for (int i = 0; i < num_local_devices; i++) {
                int dev_id = g_sycl_tp_config.devices[i];
                if (ctx->tp_dev_ptrs[dev_id] != nullptr) {
                    extra->data_device[dev_id] = (char *) ctx->tp_dev_ptrs[dev_id] + offset;
                    GGML_SYCL_DEBUG("SYCL TP: init_tensor (view) %s device %d: offset=%td, ptr=%p\n", tensor->name,
                                    dev_id, offset, extra->data_device[dev_id]);
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    // For TP compute buffers, set up extra->data_device[] for each TP device
    // This allows ggml_sycl_get_data_ptr() to resolve the correct per-device pointer
    if (ctx->is_tp_compute_buffer && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
        if (extra == nullptr) {
            extra         = new ggml_tensor_extra_gpu{};
            tensor->extra = extra;
            ctx->tensor_extras.push_back({ tensor, extra });
        }

        // Calculate offset of this tensor within the buffer
        ptrdiff_t offset = (char *) tensor->data - (char *) ctx->dev_ptr;

        // Set up data_device[] for each local TP device
        // In multi-process mode: only 1 device is locally visible
        int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
        for (int i = 0; i < num_local_devices; i++) {
            int dev_id = g_sycl_tp_config.devices[i];
            if (ctx->tp_dev_ptrs[dev_id] != nullptr) {
                extra->data_device[dev_id] = (char *) ctx->tp_dev_ptrs[dev_id] + offset;
                GGML_SYCL_DEBUG("SYCL TP: init_tensor %s device %d: offset=%td, ptr=%p\n", tensor->name, dev_id, offset,
                                extra->data_device[dev_id]);
            }
        }
    } else if ((tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q6_K ||
                tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_MXFP4) &&
               ggml_sycl_reorder_enabled()) {
        ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
        tensor->extra                 = extra;
        ctx->tensor_extras.push_back({ tensor, extra });  //used to release it when destroy ctx.
        GGML_SYCL_DEBUG("[SOA-DEBUG] init_tensor: %s type=%d allocated extra=%p reorder_mode=%d (total=%zu)\n",
                        tensor->name, tensor->type, (void *) extra, (int) extra->optimized_feature.get_reorder(),
                        ctx->tensor_extras.size());
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size   = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(
                ctx->stream->memset((char *) tensor->data + original_size, 0, padded_size - original_size).wait()));
        }
    }
    return GGML_STATUS_SUCCESS;
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

// Forward declarations for CPU-side SoA reorder functions (defined later in file)
static void reorder_q4_0_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows);
static bool reorder_q4_0_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows);
static void reorder_q8_0_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows);
static bool reorder_q8_0_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows);
static void reorder_q4_k_cpu(void * dst_soa, const void * src_aos, size_t nblocks);
static void reorder_q6_k_cpu(void * dst_soa, const void * src_aos, size_t nblocks);
static bool reorder_q6_k_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows);
static void reorder_mxfp4_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows);
static bool reorder_mxfp4_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows);

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *         tensor,
                                                const void *          data,
                                                size_t                offset,
                                                size_t                size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    ggml_sycl_set_device(ctx->device);
    // Use the buffer's stream for proper queue synchronization with compute operations
    auto stream = ctx->stream ? ctx->stream : &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));

    // Reset reorder flag when new data is written - data is now in AoS format
    // This is critical for correctness: if tensor was previously reordered to SoA,
    // the new AoS data would be misinterpreted as SoA without this reset
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    if (extra) {
        extra->optimized_feature.reset_reorder(tensor->name);
    }

    // Determine if we should do CPU-side SoA reordering during upload
    // This is faster than GPU-side reorder: no kernel launch, data already in cache
    // Check this FIRST before cache path, because cache stores AoS and we need SoA
    //
    // NOTE: We no longer exclude token_embd by name because:
    // 1. When tied weights are used, output tensors share the name "token_embd.weight"
    //    but need SoA layout for MUL_MAT operations
    // 2. GET_ROWS for Q4_0/Q8_0 already handles SoA layout via is_soa() check
    // 3. GET_ROWS for Q6_K is now supported on GPU (SoA and coalesced layouts)
    bool do_reorder = false;
    bool type_ok =
        (tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q4_K ||
         tensor->type == GGML_TYPE_Q6_K || tensor->type == GGML_TYPE_MXFP4);
    bool dims_ok     = tensor->ne[0] > 0 && tensor->ne[1] > 0;
    bool full_tensor = (offset == 0 && size == ggml_nbytes(tensor));

    if (type_ok && ggml_sycl_reorder_enabled() && ctx->supports_soa_reorder && dims_ok && full_tensor) {
        do_reorder = true;
    }

    // Debug: trace CPU reorder decision for MXFP4
    if (tensor->type == GGML_TYPE_MXFP4) {
        if (false) {
            fprintf(stderr,
                    "[SET_TENSOR-DEBUG] MXFP4: %s type_ok=%d reorder_enabled=%d soa_reorder=%d dims_ok=%d full=%d -> "
                    "do_cpu_reorder=%d\n",
                    tensor->name, type_ok, (int) ggml_sycl_reorder_enabled(), ctx->supports_soa_reorder, dims_ok,
                    full_tensor, do_reorder);
        }
    }

    // === SoA/Coalesced reorder (if needed) ===
    // GPU reorder avoids host-side work but is slower overall; CPU reorder stays as fallback.
    char *       reorder_buf      = nullptr;
    const void * src_for_upload   = data;
    reorder_mode cpu_reorder_mode = reorder_mode::SOA;  // Default to SoA, may be Coalesced

    // Check if coalesced layout is requested via global reorder mode
    const bool use_coalesced  = (g_ggml_sycl_reorder_mode == reorder_mode::COALESCED);
    const bool do_gpu_reorder = do_reorder && !use_coalesced && (g_ggml_sycl_gpu_reorder != 0);
    const bool do_cpu_reorder = do_reorder && !do_gpu_reorder;

    if (do_cpu_reorder) {
        reorder_buf         = (char *) malloc(size);
        const int64_t ncols = tensor->ne[0];
        // Use ggml_nrows to handle 3D tensors (e.g., MoE expert weights: [hidden, ffn, n_experts])
        const int64_t nrows = ggml_nrows(tensor);
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
                if (use_coalesced) {
                    if (reorder_q4_0_coalesced_cpu(reorder_buf, data, ncols, nrows)) {
                        cpu_reorder_mode = reorder_mode::COALESCED;
                        GGML_SYCL_DEBUG("[CPU-REORDER] Q4_0 AoS→Coalesced: %s ncols=%lld nrows=%lld\n", tensor->name,
                                        ncols, nrows);
                    } else {
                        // Fall back to SoA if coalesced fails (tile alignment)
                        reorder_q4_0_cpu(reorder_buf, data, ncols, nrows);
                        GGML_SYCL_DEBUG(
                            "[CPU-REORDER] Q4_0 coalesced failed, falling back to SoA: %s ncols=%lld nrows=%lld\n",
                            tensor->name, ncols, nrows);
                    }
                } else {
                    reorder_q4_0_cpu(reorder_buf, data, ncols, nrows);
                    GGML_SYCL_DEBUG("[CPU-REORDER] Q4_0 AoS→SoA: %s ncols=%lld nrows=%lld\n", tensor->name, ncols,
                                    nrows);
                }
                break;
            case GGML_TYPE_Q8_0:
                if (use_coalesced) {
                    if (reorder_q8_0_coalesced_cpu(reorder_buf, data, ncols, nrows)) {
                        cpu_reorder_mode = reorder_mode::COALESCED;
                        GGML_SYCL_DEBUG("[CPU-REORDER] Q8_0 AoS→Coalesced: %s ncols=%lld nrows=%lld\n", tensor->name,
                                        ncols, nrows);
                    } else {
                        // Fall back to SoA if coalesced fails (tile alignment)
                        reorder_q8_0_cpu(reorder_buf, data, ncols, nrows);
                        GGML_SYCL_DEBUG(
                            "[CPU-REORDER] Q8_0 coalesced failed, falling back to SoA: %s ncols=%lld nrows=%lld\n",
                            tensor->name, ncols, nrows);
                    }
                } else {
                    reorder_q8_0_cpu(reorder_buf, data, ncols, nrows);
                    GGML_SYCL_DEBUG("[CPU-REORDER] Q8_0 AoS→SoA: %s ncols=%lld nrows=%lld\n", tensor->name, ncols,
                                    nrows);
                }
                break;
            case GGML_TYPE_Q4_K:
                {
                    const size_t nblocks = size / sizeof(block_q4_K);
                    reorder_q4_k_cpu(reorder_buf, data, nblocks);
                    GGML_SYCL_DEBUG("[CPU-REORDER] Q4_K AoS→SoA: %s nblocks=%zu\n", tensor->name, nblocks);
                    break;
                }
            case GGML_TYPE_Q6_K:
                {
                    const size_t nblocks = size / sizeof(block_q6_K);
                    if (use_coalesced) {
                        if (reorder_q6_k_coalesced_cpu(reorder_buf, data, ncols, nrows)) {
                            cpu_reorder_mode = reorder_mode::COALESCED;
                            GGML_SYCL_DEBUG("[CPU-REORDER] Q6_K AoS→Coalesced: %s nblocks=%zu\n", tensor->name,
                                            nblocks);
                        } else {
                            // Fall back to SoA if coalesced fails (tile alignment)
                            reorder_q6_k_cpu(reorder_buf, data, nblocks);
                            GGML_SYCL_DEBUG(
                                "[CPU-REORDER] Q6_K coalesced failed, falling back to SoA: %s nblocks=%zu\n",
                                tensor->name, nblocks);
                        }
                    } else {
                        reorder_q6_k_cpu(reorder_buf, data, nblocks);
                        GGML_SYCL_DEBUG("[CPU-REORDER] Q6_K AoS→SoA: %s nblocks=%zu\n", tensor->name, nblocks);
                    }
                    break;
                }
            case GGML_TYPE_MXFP4:
                if (use_coalesced) {
                    if (reorder_mxfp4_coalesced_cpu(reorder_buf, data, ncols, nrows)) {
                        cpu_reorder_mode = reorder_mode::COALESCED;
                        GGML_SYCL_DEBUG("[CPU-REORDER] MXFP4 AoS→Coalesced: %s ncols=%lld nrows=%lld\n", tensor->name,
                                        ncols, nrows);
                    } else {
                        // Fall back to SoA if coalesced fails (tile alignment)
                        reorder_mxfp4_cpu(reorder_buf, data, ncols, nrows);
                        GGML_SYCL_DEBUG(
                            "[CPU-REORDER] MXFP4 coalesced failed, falling back to SoA: %s ncols=%lld nrows=%lld\n",
                            tensor->name, ncols, nrows);
                    }
                } else {
                    reorder_mxfp4_cpu(reorder_buf, data, ncols, nrows);
                    GGML_SYCL_DEBUG("[CPU-REORDER] MXFP4 AoS→SoA: %s ncols=%lld nrows=%lld\n", tensor->name, ncols,
                                    nrows);
                }
                break;
            default:
                // Shouldn't reach here due to type check above
                memcpy(reorder_buf, data, size);
                break;
        }
        src_for_upload = reorder_buf;
    }

    // === Weight streaming: cache (already reordered) weights ===
    // First access: staging (mmap → host reorder → device cache)
    // Subsequent: fast D2D copy (device cache → tensor)
    // Cache stores whatever we provide (AoS for GPU reorder, SoA for CPU reorder).
    static bool cache_disabled   = (std::getenv("GGML_SYCL_WEIGHT_CACHE_DISABLE") != nullptr);
    bool        is_weight_buffer = (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!cache_disabled && is_weight_buffer) {
        // Use tensor->data as stable key (doesn't change across set_tensor calls)
        // Use src_for_upload as source (may be reordered data)
        // Cache stores whatever we give it (AoS or SoA depending on reorder)
        void * cached_ptr = get_or_cache_weight(tensor->data, src_for_upload, size, ctx->device, stream);
        if (cached_ptr != nullptr) {
            if (do_gpu_reorder) {
                if (!reorder_aos_to_soa_device(tensor, cached_ptr, (char *) tensor->data + offset, size, stream)) {
                    GGML_ABORT("GPU AoS→SoA reorder failed for cached weight");
                }
            } else {
                // Fast path: D2D copy from cached device memory
                GGML_SYCL_DEBUG("[WEIGHT-CACHE] D2D copy: %zu bytes from %p to %p+%zu%s\n", size, cached_ptr,
                                tensor->data, offset, do_cpu_reorder ? " (SoA cached)" : "");
                SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, cached_ptr, size).wait()));
            }
            if (reorder_buf) {
                free(reorder_buf);
            }
            // Mark tensor as reordered if we did CPU or GPU reorder
            if ((do_cpu_reorder || do_gpu_reorder) && extra) {
                if (cpu_reorder_mode == reorder_mode::COALESCED) {
                    extra->optimized_feature.mark_coalesced_pretransformed(tensor->name);
                } else {
                    extra->optimized_feature.mark_soa_pretransformed(tensor->name);
                }
            } else if ((do_cpu_reorder || do_gpu_reorder) && !extra) {
                fprintf(stderr, "[SET_TENSOR-ERROR] reorder done but extra is NULL: %s\n", tensor->name);
            }
            return;
        }
    }

    // === Direct upload path (cache miss or cache disabled) ===
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device.
    // This function will be called during load model from disk.
    if (do_gpu_reorder) {
        void * tmp_buf = sycl_ext_malloc_device(stream, size);
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(tmp_buf, data, size)));
        if (!g_ggml_sycl_use_async_mem_op) {
            stream->wait_and_throw();
        }
        if (!reorder_aos_to_soa_device(tensor, tmp_buf, (char *) tensor->data + offset, size, stream)) {
            GGML_ABORT("GPU AoS→SoA reorder failed for upload");
        }
        sycl_ext_free(stream, tmp_buf);
    } else if (reorder_buf) {
        // Already have reordered data in reorder_buf
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, reorder_buf, size).wait()));
        free(reorder_buf);
    } else {
        // No reorder needed - use staging buffer for mmap workaround
        char * host_buf = (char *) malloc(size);
        memcpy(host_buf, data, size);
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, host_buf, size).wait()));
        free(host_buf);
    }
#else
    if (do_gpu_reorder) {
        void * tmp_buf = sycl_ext_malloc_device(stream, size);
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(tmp_buf, data, size)));
        if (!g_ggml_sycl_use_async_mem_op) {
            stream->wait_and_throw();
        }
        if (!reorder_aos_to_soa_device(tensor, tmp_buf, (char *) tensor->data + offset, size, stream)) {
            GGML_ABORT("GPU AoS→SoA reorder failed for upload");
        }
        sycl_ext_free(stream, tmp_buf);
    } else if (reorder_buf) {
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, reorder_buf, size).wait()));
        free(reorder_buf);
    } else {
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, data, size).wait()));
    }
#endif

    // Mark tensor as reordered if we did CPU-side reorder
    if ((do_cpu_reorder || do_gpu_reorder) && extra) {
        if (cpu_reorder_mode == reorder_mode::COALESCED) {
            extra->optimized_feature.mark_coalesced_pretransformed(tensor->name);
        } else {
            extra->optimized_feature.mark_soa_pretransformed(tensor->name);
        }
    } else if ((do_cpu_reorder || do_gpu_reorder) && !extra) {
        fprintf(stderr, "[SET_TENSOR-ERROR] reorder done but extra is NULL: %s\n", tensor->name);
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *   tensor,
                                                void *                data,
                                                size_t                offset,
                                                size_t                size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    // Use the buffer's stream for proper queue synchronization with compute operations
    // This fixes GPU speculative verification failures where tensor reads saw stale data
    auto & stream = ctx->stream ? *ctx->stream : dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(stream.memcpy(data, (const char *) tensor->data + offset, size).wait()));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

// Check if double-buffering is enabled (cached)
static bool g_pp_double_buffer_enabled = false;
static bool g_pp_double_buffer_checked = false;

static bool is_pp_double_buffer_enabled() {
    if (!g_pp_double_buffer_checked) {
        const char * env           = std::getenv("GGML_SYCL_PP_DOUBLE_BUFFER");
        g_pp_double_buffer_enabled = (env != nullptr && std::string(env) == "1");
        g_pp_double_buffer_checked = true;
        if (g_pp_double_buffer_enabled) {
            GGML_SYCL_DEBUG("SYCL PP: Double-buffering enabled\n");
        }
    }
    return g_pp_double_buffer_enabled;
}

static void dev2dev_memcpy(sycl::queue & q_dst,
                           sycl::queue & q_src,
                           void *        ptr_dst,
                           const void *  ptr_src,
                           size_t        size) {
    // Use persistent shared buffer to avoid per-transfer malloc/free overhead
    // This is a major optimization for pipeline parallelism (--split-mode layer)
    // where device-to-device transfers happen frequently between layers

    if (is_pp_double_buffer_enabled()) {
        // Double-buffered mode: use ping-pong buffers to allow overlapping transfers
        // When one buffer is being copied to destination, the other can receive new data
        int    buf_idx    = -1;
        void * shared_buf = ggml_sycl_get_dev2dev_transfer_buffer_double(size, &buf_idx);
        if (shared_buf == nullptr || buf_idx < 0) {
            // Fallback to single-buffer mode
            goto single_buffer;
        }

        // Copy src -> host buffer (wait for completion since we need data in buffer)
        q_src.memcpy(shared_buf, (const char *) ptr_src, size).wait();

        // Copy host buffer -> dst (don't wait - record event for next use of this buffer)
        sycl::event evt = q_dst.memcpy((char *) ptr_dst, shared_buf, size);
        ggml_sycl_set_dev2dev_transfer_event(buf_idx, evt);

        // Note: The next transfer will use the other buffer, and only wait if
        // that buffer's previous transfer isn't complete yet
        return;
    }

single_buffer:
    void * shared_buf = ggml_sycl_get_dev2dev_transfer_buffer(size);
    if (shared_buf == nullptr) {
        // Fallback to malloc if shared buffer allocation fails
        char * host_buf = (char *) malloc(size);
        q_src.memcpy(host_buf, (const char *) ptr_src, size).wait();
        q_dst.memcpy((char *) ptr_dst, host_buf, size).wait();
        free(host_buf);
        return;
    }

    // Use pinned host buffer: accessible from both host and all devices
    // Copy: src_device -> shared_buf -> dst_device
    q_src.memcpy(shared_buf, (const char *) ptr_src, size).wait();
    q_dst.memcpy((char *) ptr_dst, shared_buf, size).wait();
    // No free - buffer is persistent and reused
}

static bool ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *   src,
                                                ggml_tensor *         dst) try {
    bool is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *) src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *) dst->buffer->context;

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t    size       = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(*stream_dst, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#if 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (char *)dst->data, (const char *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(ctx->dev_ptr, value, buffer->size).wait()));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   uint8_t               value,
                                                   size_t                offset,
                                                   size_t                size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    if (size == 0) {
        return;  // Nothing to do
    }
    if (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    void * target_ptr = static_cast<char *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

static void ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    if (buffer == nullptr) {
        return;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    if (ctx != nullptr) {
        GGML_SYCL_DEBUG("[SOA-DEBUG] buffer_reset: freeing %zu extras (buffer=%p)\n", ctx->tensor_extras.size(),
                        (void *) buffer);
        for (auto & [tensor, extra] : ctx->tensor_extras) {
            GGML_SYCL_DEBUG("[SOA-DEBUG] buffer_reset: releasing extra=%p for tensor=%s\n", (void *) extra,
                            tensor ? tensor->name : "null");
            // Null tensor->extra BEFORE releasing to avoid dangling pointer
            if (tensor != nullptr) {
                tensor->extra = nullptr;
            }
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

static const ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// SYCL memory allocation type
enum ggml_sycl_mem_type {
    GGML_SYCL_MEM_DEVICE = 0,  // GPU device memory (fast, but device-local)
    GGML_SYCL_MEM_HOST   = 1,  // Pinned host memory (slower, but accessible from all devices)
    GGML_SYCL_MEM_SHARED = 2,  // Unified shared memory (auto-migrating)
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int                device;
    std::string        name;
    ggml_sycl_mem_type mem_type = GGML_SYCL_MEM_DEVICE;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *) buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                        size_t                     size) try {
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *) buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size                   = std::max(size, (size_t) 1);  // syclMalloc returns null for size 0

    void * dev_ptr;

    // Allocate memory based on buffer type's memory type setting
    switch (buft_ctx->mem_type) {
        case GGML_SYCL_MEM_HOST:
            // Pinned host memory - accessible from all devices (used for TP compute buffers)
            // In TP mode, use the shared context so memory is accessible from all TP devices
            if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
                sycl::queue * tp_queue = ggml_sycl_get_tp_queue(buft_ctx->device);
                if (tp_queue != nullptr) {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_host(size, tp_queue->get_context())));
                    GGML_SYCL_DEBUG("TP: Allocated %zu bytes HOST memory in shared context\n", size);
                } else {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_host(size, *stream)));
                }
            } else {
                SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_host(size, *stream)));
            }
            break;
        case GGML_SYCL_MEM_SHARED:
            // Unified shared memory - auto-migrating between host and device
            // In TP mode, use the shared context
            if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
                sycl::queue * tp_queue = ggml_sycl_get_tp_queue(buft_ctx->device);
                if (tp_queue != nullptr) {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_shared(size, *tp_queue)));
                    GGML_SYCL_DEBUG("TP: Allocated %zu bytes SHARED memory in shared context\n", size);
                } else {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_shared(size, *stream)));
                }
            } else {
                SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_shared(size, *stream)));
            }
            break;
        case GGML_SYCL_MEM_DEVICE:
        default:
            // GPU device memory - fastest but device-local
            // In TP mode, use the shared context so operations can access memory across devices
            if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
                sycl::queue * tp_queue = ggml_sycl_get_tp_queue(buft_ctx->device);
                if (tp_queue != nullptr) {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_device(size, *tp_queue)));
                    // DEBUG: Check if allocation overlaps with L31 FFN gate weight region
                    uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
                    uintptr_t alloc_start     = (uintptr_t) dev_ptr;
                    uintptr_t alloc_end       = alloc_start + size;
                    if (buft_ctx->device == 0 && alloc_start <= l31_weight_addr && alloc_end > l31_weight_addr) {
                        fprintf(stderr,
                                "TP DEBUG ALLOC OVERLAP! device=%d ptr=%p size=%zu overlaps L31 weight at 0x%llx\n",
                                buft_ctx->device, dev_ptr, size, (unsigned long long) l31_weight_addr);
                    }
                    GGML_SYCL_DEBUG("TP: Allocated %zu bytes DEVICE memory in shared context for device %d at %p\n",
                                    size, buft_ctx->device, dev_ptr);
                } else {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_device(size, *stream)));
                }
            } else {
                SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *) sycl::malloc_device(size, *stream)));
            }
            break;
    }

    if (!dev_ptr) {
        GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on device\n", __func__, size);
        return nullptr;
    }
    // In TP mode, use the shared-context queue for the buffer context
    queue_ptr ctx_stream = buft_ctx->stream;
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        sycl::queue * tp_queue = ggml_sycl_get_tp_queue(buft_ctx->device);
        if (tp_queue != nullptr) {
            ctx_stream = tp_queue;
        }
    }
    ggml_backend_sycl_buffer_context * ctx =
        new ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, ctx_stream);

    // In TP mode, allocate device memory on ALL TP devices for compute buffers
    // This allows each device to have its own copy of compute buffers
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1 && buft_ctx->mem_type == GGML_SYCL_MEM_DEVICE) {
        ctx->is_tp_compute_buffer          = true;
        ctx->tp_dev_ptrs[buft_ctx->device] = dev_ptr;  // Already allocated for main device
        ctx->tp_streams[buft_ctx->device]  = ctx_stream;

        // Allocate on other local TP devices (in multi-process mode, only 1 device is visible)
        int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
        for (int i = 0; i < num_local_devices; i++) {
            int dev_id = g_sycl_tp_config.devices[i];
            if (dev_id == buft_ctx->device) {
                continue;  // Skip main device (already done)
            }

            ggml_sycl_set_device(dev_id);
            sycl::queue * tp_queue = ggml_sycl_get_tp_queue(dev_id);
            if (tp_queue != nullptr) {
                void * ptr = nullptr;
                SYCL_CHECK(CHECK_TRY_ERROR(ptr = sycl::malloc_device(size, *tp_queue)));
                if (ptr != nullptr) {
                    ctx->tp_dev_ptrs[dev_id]  = ptr;
                    ctx->tp_streams[dev_id]   = tp_queue;
                    // DEBUG: Check if allocation overlaps with L31 FFN gate weight region
                    uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
                    uintptr_t alloc_start     = (uintptr_t) ptr;
                    uintptr_t alloc_end       = alloc_start + size;
                    if (dev_id == 0 && alloc_start <= l31_weight_addr && alloc_end > l31_weight_addr) {
                        fprintf(stderr,
                                "TP DEBUG ALLOC OVERLAP (compute)! device=%d ptr=%p size=%zu overlaps L31 weight at "
                                "0x%llx\n",
                                dev_id, ptr, size, (unsigned long long) l31_weight_addr);
                    }
                    GGML_SYCL_DEBUG("TP: Allocated compute buffer %zu bytes on device %d at %p\n", size, dev_id, ptr);
                } else {
                    GGML_LOG_ERROR("TP: Failed to allocate compute buffer on device %d\n", dev_id);
                }
            }
        }
        // Restore device context
        ggml_sycl_set_device(buft_ctx->device);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return dpct::get_current_device().get_max_mem_alloc_size();

    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                           const ggml_tensor *        tensor) {
    size_t  size = ggml_nbytes(tensor);
    int64_t ne0  = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    auto dev_count = ggml_backend_sycl_get_device_count();

    if (device >= dev_count or device < 0) {
        GGML_LOG_ERROR(
            "ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call "
            "ggml_backend_sycl_set_single_device()\n",
            device, dev_count - 1);
        GGML_ASSERT(device < dev_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto &    device_i                = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream                  = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */
                new ggml_backend_sycl_buffer_type_context{ i, GGML_SYCL_NAME + std::to_string(i), GGML_SYCL_MEM_DEVICE,
                                                          stream },
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

static ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    int device = ctx->device;
    if (device >= ggml_sycl_info().device_count or device < 0) {
        GGML_LOG_ERROR(
            "ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call "
            "ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count - 1);
        GGML_ASSERT(device < ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */
                new ggml_backend_sycl_buffer_type_context{ i, GGML_SYCL_NAME + std::to_string(i), GGML_SYCL_MEM_DEVICE,
                                                          ctx->stream(i, 0) },
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (tensor_split[i] < (i + 1 < ggml_sycl_info().device_count ? tensor_split[i + 1] : 1.0f)) {
            if (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            if (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ABORT("fatal error");
    }
}

static void get_row_split(int64_t *                                        row_low,
                          int64_t *                                        row_high,
                          const ggml_tensor *                              tensor,
                          const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split,
                          int                                              id) {
    const int64_t nrows    = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows * tensor_split[id];
    *row_low -= *row_low % rounding;
    if (id == ggml_sycl_info().device_count - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows * tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split * ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

struct ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() try {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    } catch (const sycl::exception & exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr>               streams;
};

static void ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *) 0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                   ggml_tensor *         tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr);  // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context *      ctx = (ggml_backend_sycl_split_buffer_context *) buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx =
        (ggml_backend_sycl_split_buffer_type_context *) buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    ctx->streams.push_back(&(dpct::get_current_device().default_queue()));

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t       size          = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if SYCL Buffer alloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char *          buf;
        /*
        DPCT1009:208: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *) sycl::malloc_device(size, *stream)));
        if (!buf) {
            char err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %lu Bytes of memory on device\n", __func__, size);
            throw std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(buf + original_size, 0, size - original_size).wait()));
        }

        extra->data_device[i] = buf;

        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(extra->events[i][is] = new sycl::event()));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                      ggml_tensor *         tensor,
                                                      const void *          data,
                                                      size_t                offset,
                                                      size_t                size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context *      ctx = (ggml_backend_sycl_split_buffer_context *) buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx =
        (ggml_backend_sycl_split_buffer_type_context *) buffer->buft->context;

    const int64_t           ne0   = tensor->ne[0];
    const size_t            nb1   = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split  = row_low * nb1;
        size_t       size          = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *) data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(extra->data_device[i], buf_host, original_size).wait()));
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                      const ggml_tensor *   tensor,
                                                      void *                data,
                                                      size_t                offset,
                                                      size_t                size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context *      ctx = (ggml_backend_sycl_split_buffer_context *) buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx =
        (ggml_backend_sycl_split_buffer_type_context *) buffer->buft->context;

    const int64_t           ne0   = tensor->ne[0];
    const size_t            nb1   = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split  = row_low * nb1;
        size_t       size          = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *) data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(buf_host, extra->data_device[i], original_size).wait()));
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ NULL,
};

// sycl split buffer type

static const char * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                              size_t                     size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = new ggml_backend_sycl_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                                 const ggml_tensor *        tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *) buft->context;

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");
    ggml_check_sycl();
    // FIXME: this is not thread safe
    static std::map<std::array<float, GGML_SYCL_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr ||
                    std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    if (it != buft_map.end()) {
        return &it->second;
    }

    struct ggml_backend_buffer_type buft{
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ new ggml_backend_sycl_split_buffer_type_context{ tensor_split_arr },
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    return &result.first->second;
}

//
// Tensor Parallelism Buffer Type
// This buffer type handles TP-sharded weights where each GPU holds its portion
// For true TP: allocates on ALL TP devices, each storing its rank's shard
//

// Forward declaration for FFN weight tracking (defined later)
static void store_ffn_weight_ref(const ggml_tensor * tensor);

struct ggml_backend_sycl_tp_buffer_context {
    ~ggml_backend_sycl_tp_buffer_context() {
        for (auto * extra : tensor_extras) {
            release_extra_gpu(extra);
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    int                                  main_device = 0;  // Primary device for this buffer
    int                                  world_size  = 1;  // Number of TP devices
    std::vector<int>                     devices;          // All TP device IDs
};

static const char * ggml_backend_sycl_tp_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_SYCL_NAME "_TP";
    GGML_UNUSED(buffer);
}

static void ggml_backend_sycl_tp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_sycl_tp_buffer_context *>(buffer->context);
    delete ctx;
}

static void * ggml_backend_sycl_tp_buffer_get_base(ggml_backend_buffer_t buffer) {
    return reinterpret_cast<void *>(0x2000);  // Dummy address, actual data in tensor extras
    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_sycl_tp_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx         = static_cast<ggml_backend_sycl_tp_buffer_context *>(buffer->context);
    int    main_device = ctx->main_device;
    int    world_size  = ctx->world_size;

    // DEBUG: Track which tensors enter init_tensor
    static int init_dbg = 0;
    bool       is_embd  = tensor->name[0] && strstr(tensor->name, "token_embd");
    if (g_ggml_sycl_tp_debug && (is_embd || (init_dbg++ < 5))) {
        fprintf(stderr, "TP DEBUG init_tensor ENTRY: tensor=%s, view_src=%s, buffer=%p\n", tensor->name,
                tensor->view_src ? tensor->view_src->name : "(none)", (void *) buffer);
    }

    // Check if this tensor is a view of another tensor in the same buffer
    // Views should share the parent's device allocations with an offset
    if (tensor->view_src != nullptr && tensor->view_src->buffer == buffer) {
        // This is a view - use parent's allocations with offset
        auto * parent_extra = static_cast<ggml_tensor_extra_gpu *>(tensor->view_src->extra);
        if (parent_extra != nullptr) {
            // Create extra for this view that points into parent's allocations
            auto * extra = new ggml_tensor_extra_gpu{};
            ctx->tensor_extras.push_back(extra);

            // Copy relevant fields from parent
            extra->tp_sharded     = parent_extra->tp_sharded;
            extra->tp_world_size  = parent_extra->tp_world_size;
            extra->tp_type        = parent_extra->tp_type;
            extra->tp_type_cached = true;

            // Find the root data owner (not an intermediate view)
            // and point to its optimized_feature so get_reorder() returns current state
            ggml_tensor_extra_gpu * root_extra = parent_extra;
            ggml_tensor *           root       = tensor->view_src;
            while (root->view_src != nullptr) {
                root = root->view_src;
                if (root->extra) {
                    root_extra = static_cast<ggml_tensor_extra_gpu *>(root->extra);
                }
            }
            extra->optimized_feature.set_data_owner(&root_extra->optimized_feature);

            // Calculate offset from parent to this view
            // tensor->data and tensor->view_src->data are both host pointers during graph build
            // The view offset is computed from the difference
            size_t view_offset =
                reinterpret_cast<char *>(tensor->data) - reinterpret_cast<char *>(tensor->view_src->data);

            // Set device pointers as parent + offset for each device
            // Use ctx->devices.size() not world_size - in multi-process mode we only have 1 local device
            for (int rank = 0; rank < (int) ctx->devices.size(); rank++) {
                int device = ctx->devices[rank];
                if (parent_extra->data_device[device] != nullptr) {
                    extra->data_device[device] =
                        reinterpret_cast<char *>(parent_extra->data_device[device]) + view_offset;
                }
            }

            // tensor->data now points to main device's view location
            tensor->data  = extra->data_device[main_device];
            tensor->extra = extra;

            GGML_SYCL_DEBUG("SYCL TP: view tensor %s uses parent %s + offset %zu\n", tensor->name,
                            tensor->view_src->name, view_offset);

            return GGML_STATUS_SUCCESS;
        }
    }

    auto * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    // Determine if this tensor should be sharded
    // Use ggml_sycl_tp_get_layer_type() which has caching - but tensor->extra isn't set yet
    // So we call it after setting extra, and it will cache the result
    extra->tp_type        = tp_layer_type::TP_NONE;
    extra->tp_type_cached = false;  // Will be computed and cached on first access
    tensor->extra         = extra;  // Set early so ggml_sycl_tp_get_layer_type can cache

    // Now get the TP type (this will compute and cache it)
    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);

    // DEBUG: Check why sharding might not happen
    static int shard_dbg = 0;
    if (g_ggml_sycl_tp_debug && shard_dbg++ < 20 && tensor->name[0] &&
        (strstr(tensor->name, "ffn_gate") || strstr(tensor->name, "attn_q"))) {
        fprintf(stderr, "TP DEBUG init_tensor %s: enabled=%d, world_size=%d, tp_type=%d (COL=%d,ROW=%d)\n",
                tensor->name, g_sycl_tp_config.enabled, world_size, (int) tp_type,
                (int) tp_layer_type::TP_COLUMN_PARALLEL, (int) tp_layer_type::TP_ROW_PARALLEL);
    }

    // Enable sharding when TP is active with multiple devices
    bool should_shard = g_sycl_tp_config.enabled && world_size > 1 &&
                        (tp_type == tp_layer_type::TP_COLUMN_PARALLEL || tp_type == tp_layer_type::TP_ROW_PARALLEL);

    // Check if we're in multi-process TP mode where tensor already has sharded dimensions
    bool is_multiprocess_tp = g_sycl_tp_config.is_multiprocess && ctx->devices.size() == 1;

    if (should_shard) {
        // TRUE DUAL-GPU TP: Allocate shards on ALL TP devices
        // Each device gets its rank's portion of the weight
        extra->tp_sharded    = true;
        extra->tp_world_size = world_size;

        // Store original (FULL, unshard) dimensions.
        // In both single-process and multi-process modes, the tensor is created with
        // SHARDED dimensions by the model layer code. We need to compute the TRUE
        // original dimensions by multiplying the sharded dimension back by world_size.
        //
        // Column-parallel: ne[1] is sharded (output dimension)
        // Row-parallel: ne[0] is sharded (input dimension)
        extra->tp_original_ne[0] = tensor->ne[0];
        extra->tp_original_ne[1] = tensor->ne[1];
        extra->tp_original_ne[2] = tensor->ne[2];
        extra->tp_original_ne[3] = tensor->ne[3];

        // Compute full (unsharded) dimensions for original storage
        if (tp_type == tp_layer_type::TP_COLUMN_PARALLEL) {
            extra->tp_original_ne[1] = tensor->ne[1] * world_size;
        } else if (tp_type == tp_layer_type::TP_ROW_PARALLEL) {
            extra->tp_original_ne[0] = tensor->ne[0] * world_size;
        }

        // Allocate on each TP device
        // In single-process mode, tensor->ne already has sharded dimensions from model layer
        // In multi-process mode, tensor->ne also has sharded dimensions from model loader
        // So we use tensor->ne directly as the shard size - no further division needed
        for (int rank = 0; rank < (int) ctx->devices.size(); rank++) {
            int device = ctx->devices[rank];

            // The tensor is already created with sharded dimensions by the model layer.
            // Each device gets an allocation of THIS size (the shard size).
            int64_t local_ne0  = tensor->ne[0];
            int64_t local_ne1  = tensor->ne[1];
            int64_t offset_ne0 = 0;
            int64_t offset_ne1 = 0;

            // For set_tensor later: compute offset for this rank's data extraction
            if (tp_type == tp_layer_type::TP_COLUMN_PARALLEL) {
                offset_ne1 = rank * tensor->ne[1];  // Each rank's starting column
            } else if (tp_type == tp_layer_type::TP_ROW_PARALLEL) {
                offset_ne0 = rank * tensor->ne[0];  // Each rank's starting row
            }

            // Calculate shard size
            size_t shard_size = ggml_row_size(tensor->type, local_ne0) * local_ne1 * tensor->ne[2] * tensor->ne[3];

            // For row-parallel on main device, allocate FULL size with zero padding
            // This allows MMVQ to use full tensor dimensions but compute only partial result
            // The zeros ensure contributions from the "other half" are zero
            bool   is_row_parallel_main = (tp_type == tp_layer_type::TP_ROW_PARALLEL && rank == 0);
            size_t alloc_size;
            if (is_row_parallel_main) {
                // Full size for main device row-parallel - use ORIGINAL (full) dimensions
                // tensor->ne is already sharded, but we need full size for MMVQ
                size_t full_row_size = ggml_row_size(tensor->type, extra->tp_original_ne[0]);
                alloc_size =
                    full_row_size * extra->tp_original_ne[1] * extra->tp_original_ne[2] * extra->tp_original_ne[3];
            } else {
                // Shard size for other cases
                alloc_size = shard_size;
            }

            // Pad for alignment
            size_t  padded_size     = alloc_size;
            int64_t ne0_for_padding = is_row_parallel_main ? extra->tp_original_ne[0] : local_ne0;
            if (ne0_for_padding % MATRIX_ROW_PADDING != 0) {
                padded_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0_for_padding % MATRIX_ROW_PADDING);
            }

            // Allocate on this device using shared-context queue for TP
            ggml_sycl_set_device(device);
            queue_ptr stream = ggml_sycl_get_tp_queue(device);
            if (stream == nullptr) {
                stream = &dpct::get_current_device().default_queue();
            }
            char * buf = static_cast<char *>(sycl::malloc_device(padded_size, *stream));
            if (!buf) {
                fprintf(stderr, "SYCL TP: Failed to allocate %zu bytes on device %d for tensor %s\n", padded_size,
                        device, tensor->name);
                return GGML_STATUS_ALLOC_FAILED;
            }

            // Zero-fill the entire buffer (important for row-parallel zero-padding)
            stream->memset(buf, 0, padded_size).wait();

            extra->data_device[device] = buf;

            // Store local dimensions for rank 0 (used for tensor->ne update)
            if (rank == 0) {
                extra->tp_local_ne[0]  = local_ne0;
                extra->tp_local_ne[1]  = local_ne1;
                extra->tp_local_ne[2]  = tensor->ne[2];
                extra->tp_local_ne[3]  = tensor->ne[3];
                extra->tp_offset_ne[0] = offset_ne0;
                extra->tp_offset_ne[1] = offset_ne1;
                extra->tp_rank         = rank;
            }
        }

        // NOTE: tensor->ne already has sharded dimensions from model layer creation.
        // The model layer creates tensors with {n_embd, tp_n_embd_head_k_x_n_head} etc.
        // We do NOT modify tensor->ne here - it's already correct for graph building.
        GGML_UNUSED(is_multiprocess_tp);

        // DEBUG: Verify sharding happened
        static int shard_verify_dbg = 0;
        if (g_ggml_sycl_tp_debug && shard_verify_dbg++ < 10 && tensor->name[0]) {
            fprintf(stderr, "TP DEBUG SHARD %s: orig=[%lld,%lld] -> shard=[%lld,%lld] (multiprocess=%d)\n",
                    tensor->name, (long long) extra->tp_original_ne[0], (long long) extra->tp_original_ne[1],
                    (long long) tensor->ne[0], (long long) tensor->ne[1], is_multiprocess_tp);
        }

        // tensor->data points to main device's shard
        tensor->data = extra->data_device[main_device];

    } else {
        // Non-sharded tensor: DUPLICATE on all TP devices for fast access
        // Intel Arc GPUs don't support P2P, and host memory is 32x slower for kernel access
        // So we duplicate non-sharded weights (layer norms, etc.) on each device
        extra->tp_sharded  = false;
        extra->tp_usm_host = false;

        // MULTI-PROCESS TP FIX: Even though local allocation isn't sharded (each process
        // has only 1 device), we need to track the TP type for TP layer tensors. This tells
        // MUL_MAT kernels that this weight produces partial results requiring ALL_REDUCE.
        // The tensor was created with sharded dimensions by the model loader based on
        // ggml_backend_sycl_get_tp_world_size() which returns the MPI world size.
        if (is_multiprocess_tp && tp_type != tp_layer_type::TP_NONE) {
            extra->tp_type           = tp_type;
            extra->tp_world_size     = g_sycl_tp_config.world_size;
            extra->tp_original_ne[0] = tensor->ne[0];
            extra->tp_original_ne[1] = tensor->ne[1];
            extra->tp_original_ne[2] = tensor->ne[2];
            extra->tp_original_ne[3] = tensor->ne[3];

            // Compute full (unsharded) dimensions for the original tensor
            if (tp_type == tp_layer_type::TP_COLUMN_PARALLEL) {
                extra->tp_original_ne[1] = tensor->ne[1] * g_sycl_tp_config.world_size;
            } else if (tp_type == tp_layer_type::TP_ROW_PARALLEL) {
                extra->tp_original_ne[0] = tensor->ne[0] * g_sycl_tp_config.world_size;
            }

            if (g_ggml_sycl_tp_debug) {
                static int mp_dbg = 0;
                if (mp_dbg++ < 10) {
                    fprintf(stderr,
                            "TP DEBUG MULTIPROCESS %s: tp_type=%d, world_size=%d, ne=[%lld,%lld], orig=[%lld,%lld]\n",
                            tensor->name, (int) tp_type, g_sycl_tp_config.world_size, (long long) tensor->ne[0],
                            (long long) tensor->ne[1], (long long) extra->tp_original_ne[0],
                            (long long) extra->tp_original_ne[1]);
                }
            }
        }

        size_t alloc_size = ggml_nbytes(tensor);

        // Pad for alignment
        size_t padded_size = alloc_size;
        if (tensor->ne[0] % MATRIX_ROW_PADDING != 0) {
            padded_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - tensor->ne[0] % MATRIX_ROW_PADDING);
        }

        // Allocate device memory on TP devices using shared-context queue
        // In multi-process mode: ctx->devices.size() == 1, world_size is MPI world size
        bool is_tok_embd = tensor->name[0] && strstr(tensor->name, "token_embd");
        for (int rank = 0; rank < (int) ctx->devices.size(); rank++) {
            int device = ctx->devices[rank];
            ggml_sycl_set_device(device);
            queue_ptr stream = ggml_sycl_get_tp_queue(device);
            if (stream == nullptr) {
                stream = &dpct::get_current_device().default_queue();
            }
            char * buf = static_cast<char *>(sycl::malloc_device(padded_size, *stream));
            if (!buf) {
                fprintf(stderr, "SYCL TP: Failed to allocate %zu bytes on device %d for tensor %s\n", padded_size,
                        device, tensor->name);
                return GGML_STATUS_ALLOC_FAILED;
            }

            // Zero padding
            stream->memset(buf, 0, padded_size).wait();

            // DEBUG: Track tensor allocation
            if (g_ggml_sycl_tp_debug && (is_tok_embd || (tensor->name[0] && strstr(tensor->name, "output_norm")))) {
                sycl::device q_dev = stream->get_device();
                fprintf(stderr, "TP DEBUG ALLOC %s: rank=%d, device=%d, queue_device='%s', buf=%p, size=%zu\n",
                        tensor->name, rank, device, q_dev.get_info<sycl::info::device::name>().c_str(), (void *) buf,
                        padded_size);
            }

            extra->data_device[device] = buf;
        }

        // tensor->data points to main device's copy
        tensor->data = extra->data_device[main_device];
    }

    // tensor->extra already set early in this function for caching
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_tp_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   const void *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto * ctx   = static_cast<ggml_backend_sycl_tp_buffer_context *>(buffer->context);
    auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);

    GGML_ASSERT(offset == 0);  // TP tensors must be set in full

    if (extra->tp_sharded) {
        // SHARDED TP: Copy each rank's shard to its device
        int world_size = ctx->world_size;

        // We need original dimensions for shard extraction
        // Temporarily restore them for the copy operation
        int64_t saved_ne[4] = { tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3] };
        tensor->ne[0]       = extra->tp_original_ne[0];
        tensor->ne[1]       = extra->tp_original_ne[1];
        tensor->ne[2]       = extra->tp_original_ne[2];
        tensor->ne[3]       = extra->tp_original_ne[3];

        if (g_sycl_tp_config.is_multiprocess) {
            // Multi-process mode: each process handles only its own rank
            // world_size is MPI world size, but we only have ONE device visible
            //
            // IMPORTANT: In multi-process mode, the model loader already extracts
            // this rank's shard from the file using strided reads. The 'data' parameter
            // contains ONLY this rank's portion, not the full tensor. So we do a
            // direct copy of 'size' bytes instead of calling ggml_sycl_tp_copy_weight_shard
            // (which would try to extract a shard from what it thinks is the full tensor).
            int device = ctx->devices[0];  // Only one device in multi-process mode
            ggml_sycl_set_device(device);
            queue_ptr stream = ggml_sycl_get_tp_queue(device);
            if (stream == nullptr) {
                stream = &dpct::get_current_device().default_queue();
            }

            // Direct copy - data is already this rank's shard
            stream->memcpy(extra->data_device[device], data, size).wait();

            // DEBUG: Print first bytes of column-parallel weights to verify correct loading
            if (tensor->name[0] && strstr(tensor->name, "blk.0.attn_q.weight")) {
                const uint8_t * src = static_cast<const uint8_t *>(data);
                fprintf(
                    stderr,
                    "[TP LOAD MP] tensor='%s' rank=%d size=%zu first_bytes=[%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x]\n",
                    tensor->name, g_sycl_tp_config.rank, size, src[0], src[1], src[2], src[3], src[4], src[5], src[6],
                    src[7]);
            }

            GGML_SYCL_DEBUG("SYCL TP MP: Direct copy %zu bytes to device %d for tensor %s\n", size, device,
                            tensor->name);
        } else {
            // Single-process multi-device mode: iterate all devices
            for (int rank = 0; rank < world_size; rank++) {
                int device = ctx->devices[rank];
                ggml_sycl_set_device(device);
                queue_ptr stream = ggml_sycl_get_tp_queue(device);
                if (stream == nullptr) {
                    stream = &dpct::get_current_device().default_queue();
                }

                // Copy this rank's shard to its device
                ggml_sycl_tp_copy_weight_shard(extra->data_device[device], data, tensor, rank, world_size, stream);
            }
        }

        // Restore sharded dimensions
        tensor->ne[0] = saved_ne[0];
        tensor->ne[1] = saved_ne[1];
        tensor->ne[2] = saved_ne[2];
        tensor->ne[3] = saved_ne[3];
    } else {
        // Non-sharded: DUPLICATE to local TP devices for fast kernel access
        // Copy same data to each device since we can't use P2P or host memory
        // In multi-process mode: ctx->devices.size() == 1, world_size is MPI world size
        bool is_tok_embd = tensor->name[0] && strstr(tensor->name, "token_embd");
        for (int rank = 0; rank < (int) ctx->devices.size(); rank++) {
            int device = ctx->devices[rank];
            ggml_sycl_set_device(device);
            queue_ptr stream = ggml_sycl_get_tp_queue(device);
            if (stream == nullptr) {
                stream = &dpct::get_current_device().default_queue();
            }
            // DEBUG: Track embedding table data copy
            if (g_ggml_sycl_tp_debug && is_tok_embd && rank == 0) {
                // Check SOURCE data before copy (mmapped file)
                struct {
                    uint16_t d_bits;
                    uint8_t  qs[16];
                } src_blk1, src_blk38, src_blk100;

                size_t blk_row_size = (4096 / 32) * 18;
                memcpy(&src_blk1, static_cast<const char *>(data) + 1 * blk_row_size, sizeof(src_blk1));
                memcpy(&src_blk38, static_cast<const char *>(data) + 38 * blk_row_size, sizeof(src_blk38));
                memcpy(&src_blk100, static_cast<const char *>(data) + 100 * blk_row_size, sizeof(src_blk100));
                sycl::half src1_d, src38_d, src100_d;
                memcpy(&src1_d, &src_blk1.d_bits, sizeof(sycl::half));
                memcpy(&src38_d, &src_blk38.d_bits, sizeof(sycl::half));
                memcpy(&src100_d, &src_blk100.d_bits, sizeof(sycl::half));
                fprintf(
                    stderr,
                    "TP DEBUG COPY SOURCE (host/mmap): tok1.d=%f (0x%04x), tok38.d=%f (0x%04x), tok100.d=%f (0x%04x)\n",
                    (float) src1_d, src_blk1.d_bits, (float) src38_d, src_blk38.d_bits, (float) src100_d,
                    src_blk100.d_bits);
                fprintf(stderr, "TP DEBUG COPY SOURCE: tok38.qs=0x%02x%02x, tok100.qs=0x%02x%02x\n", src_blk38.qs[0],
                        src_blk38.qs[1], src_blk100.qs[0], src_blk100.qs[1]);
                fprintf(stderr, "TP DEBUG COPY tok_embd: src=%p, dst=%p, size=%zu\n", data,
                        (void *) extra->data_device[device], size);
            }
            stream->memcpy(extra->data_device[device], data, size).wait();
            // DEBUG: Verify copy by reading back token 0, 1, 38, 100
            if (g_ggml_sycl_tp_debug && is_tok_embd) {
                struct {
                    sycl::half d;
                    uint8_t    qs[16];
                } blk0, blk1, blk38, blk100;

                size_t blk_row_size = (4096 / 32) * 18;  // 128 blocks * 18 bytes = 2304 bytes/row
                stream->memcpy(&blk0, extra->data_device[device], sizeof(blk0)).wait();
                stream->memcpy(&blk1, (char *) extra->data_device[device] + 1 * blk_row_size, sizeof(blk1)).wait();
                stream->memcpy(&blk38, (char *) extra->data_device[device] + 38 * blk_row_size, sizeof(blk38)).wait();
                stream->memcpy(&blk100, (char *) extra->data_device[device] + 100 * blk_row_size, sizeof(blk100))
                    .wait();
                fprintf(stderr, "TP DEBUG COPY VERIFY device=%d: tok0.d=%f, tok1.d=%f, tok38.d=%f, tok100.d=%f\n",
                        device, (float) blk0.d, (float) blk1.d, (float) blk38.d, (float) blk100.d);
                fprintf(stderr, "TP DEBUG COPY VERIFY device=%d: tok38.qs=0x%02x%02x, tok100.qs=0x%02x%02x\n", device,
                        blk38.qs[0], blk38.qs[1], blk100.qs[0], blk100.qs[1]);
            }
        }
    }

    // Store FFN weight reference for later computation on device 1
    store_ffn_weight_ref(tensor);

    GGML_UNUSED(buffer);
}

static void ggml_backend_sycl_tp_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   const ggml_tensor *   tensor,
                                                   void *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto * ctx    = static_cast<ggml_backend_sycl_tp_buffer_context *>(buffer->context);
    auto * extra  = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
    int    device = ctx->main_device;  // Get from main device

    ggml_sycl_set_device(device);
    queue_ptr stream = ggml_sycl_get_tp_queue(device);
    if (stream == nullptr) {
        stream = &dpct::get_current_device().default_queue();
    }

    if (extra->tp_sharded) {
        // For sharded tensors, we can only get rank 0's shard from main device
        size_t shard_size = ggml_sycl_tp_get_shard_size(tensor, 0, extra->tp_world_size);
        GGML_ASSERT(offset == 0 && size == shard_size);
        stream->memcpy(data, extra->data_device[device], shard_size).wait();
    } else {
        const char * src = static_cast<const char *>(extra->data_device[device]) + offset;
        stream->memcpy(data, src, size).wait();
    }

    GGML_UNUSED(buffer);
}

static bool ggml_backend_sycl_tp_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                   const ggml_tensor *   src,
                                                   ggml_tensor *         dst) {
    // For simplicity, don't support direct copy of TP tensors
    return false;
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

static void ggml_backend_sycl_tp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // Not commonly needed for TP buffers
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_tp_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_tp_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_tp_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_tp_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_sycl_tp_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_tp_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_tp_buffer_clear,
    /* .reset           = */ NULL,
};

// TP buffer type interface
struct ggml_backend_sycl_tp_buffer_type_context {
    int              main_device;
    int              world_size;
    std::vector<int> devices;  // All TP device IDs
};

static const char * ggml_backend_sycl_tp_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_TP";
    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_sycl_tp(ggml_backend_buffer_t buffer) {
    return buffer && buffer->buft && buffer->buft->iface.get_name == ggml_backend_sycl_tp_buffer_type_name;
}

static ggml_backend_buffer_t ggml_backend_sycl_tp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    auto * buft_ctx  = static_cast<ggml_backend_sycl_tp_buffer_type_context *>(buft->context);
    auto * ctx       = new ggml_backend_sycl_tp_buffer_context();
    ctx->main_device = buft_ctx->main_device;
    ctx->world_size  = buft_ctx->world_size;
    ctx->devices     = buft_ctx->devices;

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_tp_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_tp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_tp_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                              const ggml_tensor *        tensor) {
    // NOTE: Currently we DON'T shard weights because llama.cpp uses single-device compute.
    // Always return full tensor size.
    size_t size = ggml_nbytes(tensor);

    // Add padding
    if (tensor->ne[0] % MATRIX_ROW_PADDING != 0) {
        size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - tensor->ne[0] % MATRIX_ROW_PADDING);
    }

    return size;
    GGML_UNUSED(buft);
}

static bool ggml_backend_sycl_tp_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_tp_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_sycl_tp_buffer_type_name,
    /* .alloc_buffer   = */ ggml_backend_sycl_tp_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_sycl_tp_buffer_type_get_alignment,
    /* .get_max_size   = */ NULL,
    /* .get_alloc_size = */ ggml_backend_sycl_tp_buffer_type_get_alloc_size,
    /* .is_host        = */ ggml_backend_sycl_tp_buffer_type_is_host,
};

// Get the TP buffer type for a specific device (internal helper)
// This now includes ALL TP devices so the buffer can allocate on all of them
static ggml_backend_buffer_type_t get_tp_buffer_type_for_device(int main_device) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<int, ggml_backend_buffer_type> buft_map;

    auto it = buft_map.find(main_device);
    if (it != buft_map.end()) {
        return &it->second;
    }

    // Get all TP devices from the global config
    // NOTE: ggml_sycl_tp_world_size() returns 1 in multi-process mode (for graph building),
    // but ctx->world_size should reflect the actual MPI world size for TP tracking
    int              world_size = ggml_sycl_tp_world_size();
    std::vector<int> devices;

    // Check if TP should be active: either multiple local devices OR multi-process mode
    bool tp_active = g_sycl_tp_config.enabled && (world_size > 1 || g_sycl_tp_config.is_multiprocess);

    if (tp_active) {
        if (g_sycl_tp_config.is_multiprocess) {
            // Multi-process mode: each process has only ONE device (device 0)
            // Use MPI world size for TP tracking, but only one local device
            devices.push_back(0);                      // Always device 0 (restricted by ONEAPI_DEVICE_SELECTOR)
            world_size = g_sycl_tp_config.world_size;  // Use actual MPI world size
        } else {
            // Single-process multi-device mode: allocate on all devices
            for (int i = 0; i < world_size; i++) {
                devices.push_back(g_sycl_tp_config.devices[i]);
            }
        }
    } else {
        devices.push_back(main_device);
        world_size = 1;
    }

    auto * ctx       = new ggml_backend_sycl_tp_buffer_type_context();
    ctx->main_device = main_device;
    ctx->world_size  = world_size;
    ctx->devices     = devices;

    ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_sycl_tp_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), main_device),
        /* .context = */ ctx,
    };

    auto result = buft_map.emplace(main_device, buft);
    return &result.first->second;
}

// Initialize tensor parallelism and get buffer type
// This function matches the ggml_backend_tp_buffer_type_t signature
ggml_backend_buffer_type_t ggml_backend_sycl_tp_buffer_type(int n_devices, const int * device_ids) {
    // Initialize TP system if not already initialized
    static bool       tp_initialized = false;
    static std::mutex init_mutex;

    if (!tp_initialized) {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!tp_initialized) {
            // Initialize TP with the given devices
            if (n_devices > 0 && device_ids != nullptr) {
                ggml_sycl_tp_init(device_ids, n_devices);
            } else {
                // Auto-detect: use all available devices
                int device_count = ggml_backend_sycl_get_device_count();
                if (device_count > 1) {
                    std::vector<int> devices(device_count);
                    for (int i = 0; i < device_count; i++) {
                        devices[i] = i;
                    }
                    ggml_sycl_tp_init(devices.data(), device_count);
                }
            }
            tp_initialized = true;
        }
    }

    // Return the buffer type for the first device (main device)
    // The model loading code will use this for all tensor allocations
    int main_device = (n_devices > 0 && device_ids != nullptr) ? device_ids[0] : 0;
    return get_tp_buffer_type_for_device(main_device);
}

// Get the TP world size (number of devices in TP group, 1 if TP not enabled)
// This is called from llama layer to configure KV cache sharding and graph dimensions.
// When TP is enabled, graph will be built with sharded dimensions (n_head/world_size, etc.)
int ggml_backend_sycl_get_tp_world_size(void) {
    int ws = ggml_sycl_tp_world_size();

    // If TP not initialized yet, try to detect MPI environment early
    // This is needed because model loading queries world_size before buffer allocation triggers TP init
    if (ws <= 1) {
        // Check for Intel MPI (PMI) environment variables
        const char * pmi_size = std::getenv("PMI_SIZE");
        if (pmi_size) {
            int mpi_world_size = std::atoi(pmi_size);
            if (mpi_world_size > 1) {
                return mpi_world_size;
            }
        }
        // Check for Open MPI environment variables
        const char * ompi_size = std::getenv("OMPI_COMM_WORLD_SIZE");
        if (ompi_size) {
            int mpi_world_size = std::atoi(ompi_size);
            if (mpi_world_size > 1) {
                return mpi_world_size;
            }
        }
    }

    return ws > 0 ? ws : 1;
}

// Get the TP rank for this process
int ggml_backend_sycl_get_tp_rank(void) {
    // First check if TP is initialized
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.is_multiprocess) {
        return g_sycl_tp_config.rank;
    }

    // Try to detect from MPI environment
    const char * pmi_rank = std::getenv("PMI_RANK");
    if (pmi_rank) {
        return std::atoi(pmi_rank);
    }
    const char * ompi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
    if (ompi_rank) {
        return std::atoi(ompi_rank);
    }

    return 0;
}

// Check if running in multi-process TP mode
bool ggml_backend_sycl_is_multiprocess_tp(void) {
    // First check if TP is initialized
    if (g_sycl_tp_config.enabled) {
        return g_sycl_tp_config.is_multiprocess;
    }

    // Try to detect from MPI environment
    const char * pmi_size = std::getenv("PMI_SIZE");
    if (pmi_size && std::atoi(pmi_size) > 1) {
        return true;
    }
    const char * ompi_size = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (ompi_size && std::atoi(ompi_size) > 1) {
        return true;
    }

    return false;
}

// Get the byte offset for reading this rank's shard from GGUF file
// For column-parallel tensors (wq, wk, wv, ffn_gate, ffn_up): contiguous, returns offset
// For row-parallel tensors (wo, ffn_down): interleaved, returns 0 (requires special handling)
size_t ggml_backend_sycl_get_tp_data_offset(const char *    tensor_name,
                                            const int64_t * tensor_ne,
                                            enum ggml_type  tensor_type) {
    if (!ggml_backend_sycl_is_multiprocess_tp()) {
        return 0;
    }

    int world_size = ggml_backend_sycl_get_tp_world_size();
    int rank       = ggml_backend_sycl_get_tp_rank();

    if (world_size <= 1 || rank == 0) {
        return 0;  // Rank 0 always reads from start
    }

    // Determine TP layer type from tensor name
    // Column-parallel: wq, wk, wv, ffn_gate, ffn_up (split output dim ne[1])
    // Row-parallel: wo/attn_output, ffn_down (split input dim ne[0])
    bool is_column_parallel = false;
    bool is_row_parallel    = false;

    if (tensor_name) {
        // Column-parallel layers (output dimension split)
        if (strstr(tensor_name, "attn_q.weight") || strstr(tensor_name, "attn_k.weight") ||
            strstr(tensor_name, "attn_v.weight") || strstr(tensor_name, "ffn_gate.weight") ||
            strstr(tensor_name, "ffn_up.weight")) {
            is_column_parallel = true;
        }
        // Row-parallel layers (input dimension split)
        else if (strstr(tensor_name, "attn_output.weight") || strstr(tensor_name, "ffn_down.weight")) {
            is_row_parallel = true;
        }
    }

    if (!is_column_parallel && !is_row_parallel) {
        return 0;  // Not a TP tensor
    }

    int64_t ne0 = tensor_ne[0];
    int64_t ne1 = tensor_ne[1];

    if (is_column_parallel) {
        // Column-parallel: split ne[1] (output dimension)
        // Data is stored as rows of ne0 elements, we want rows [rank*ne1/world_size, ...]
        int64_t chunk_size = ne1 / world_size;
        int64_t remainder  = ne1 % world_size;
        int64_t offset_ne1;

        if (rank < remainder) {
            offset_ne1 = rank * (chunk_size + 1);
        } else {
            offset_ne1 = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size;
        }

        // Calculate byte offset: offset_ne1 rows * row_size
        size_t row_size = ggml_row_size(tensor_type, ne0);
        return offset_ne1 * row_size;
    } else if (is_row_parallel) {
        // Row-parallel: split ne[0] (input dimension)
        // Data is interleaved - each row has elements from all ranks
        // This requires special handling during copy, not just an offset
        // Return 0 and let the copy function handle it
        return 0;
    }

    return 0;
}

// ============================================================================
// Pipeline Parallelism (PP) API - vLLM-style layer split with chunked prefill
// ============================================================================

void ggml_backend_sycl_pp_init(const int * device_ids, int n_devices, int total_layers, const int * layers_per_stage) {
    ggml_sycl_pp_init(device_ids, n_devices, total_layers, layers_per_stage);
}

void ggml_backend_sycl_pp_free(void) {
    ggml_sycl_pp_free();
}

bool ggml_backend_sycl_pp_enabled(void) {
    return ggml_sycl_pp_enabled();
}

int ggml_backend_sycl_pp_num_stages(void) {
    return ggml_sycl_pp_num_stages();
}

int ggml_backend_sycl_pp_get_device_for_layer(int layer) {
    return ggml_sycl_pp_get_device_for_layer(layer);
}

void ggml_backend_sycl_pp_set_chunked_prefill(int32_t chunk_size, bool enabled) {
    ggml_sycl_pp_set_chunked_prefill(chunk_size, enabled);
}

// ===========================================================================
// GPU Sampling API Implementation
// ===========================================================================

// GPU sampler struct - holds backend context and sampler state
struct ggml_sycl_sampler {
    ggml_backend_t              backend;
    ggml_backend_sycl_context * sycl_ctx;
    ggml_sycl_sampler_state     state;
    int                         n_vocab;
};

ggml_sycl_sampler_t ggml_backend_sycl_sampler_create(ggml_backend_t backend, int n_vocab, uint32_t seed) try {
    GGML_ASSERT(ggml_backend_is_sycl(backend));

    ggml_sycl_sampler_t sampler = new ggml_sycl_sampler();
    sampler->backend            = backend;
    sampler->sycl_ctx           = (ggml_backend_sycl_context *) backend->context;
    sampler->n_vocab            = n_vocab;

    // Initialize sampler state
    ggml_sycl_sampler_init(sampler->state, n_vocab, seed);

    return sampler;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error creating GPU sampler: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_sampler_free(ggml_sycl_sampler_t sampler) try {
    if (sampler == nullptr) {
        return;
    }

    if (sampler->state.initialized) {
        sycl::queue & q = *sampler->sycl_ctx->stream();
        ggml_sycl_sampler_free(sampler->state, q);
    }

    delete sampler;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error freeing GPU sampler: %s\n", exc.what());
}

int32_t ggml_backend_sycl_sample_token(ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor, float temp) try {
    // Sample from index 0 (for single-token decode)
    return ggml_backend_sycl_sample_token_idx(sampler, logits_tensor, 0, temp);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in GPU sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t ggml_backend_sycl_sample_token_idx(ggml_sycl_sampler_t sampler,
                                           ggml_tensor *       logits_tensor,
                                           int                 idx,
                                           float               temp) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    // Get GPU pointer for logits
    float * logits_gpu = (float *) logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    // Get vocab size from tensor (ne[0] is vocab dimension)
    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    // Get number of batch entries (ne[1] is batch dimension)
    int n_batch = logits_tensor->ne[1];
    GGML_ASSERT(idx >= 0 && idx < n_batch);

    // Offset to the correct batch entry
    float * logits_at_idx = logits_gpu + (size_t) idx * n_vocab;

    // Set up config
    ggml_sycl_sampler_config config;
    config.temp   = temp;
    config.top_k  = 0;
    config.top_p  = 1.0f;
    config.min_p  = 0.0f;
    config.seed   = sampler->state.rng_state;  // Use current RNG state
    config.greedy = (temp == 0.0f);

    // Call GPU sampler with offset logits
    return ggml_sycl_sample_token(*sampler->sycl_ctx, logits_at_idx, config, sampler->state);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in GPU sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t ggml_backend_sycl_sample_token_full(ggml_sycl_sampler_t sampler,
                                            ggml_tensor *       logits_tensor,
                                            int                 idx,
                                            float               temp,
                                            int                 top_k,
                                            float               top_p,
                                            float               min_p) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    // Get the backend context from the tensor's buffer for proper synchronization
    ggml_backend_buffer_t buffer = logits_tensor->buffer;
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_sycl(buffer));

    // Get the context from the buffer's backend (the one that computed the logits)
    ggml_backend_sycl_buffer_context * buf_ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    GGML_ASSERT(buf_ctx != nullptr);

    // Sync the buffer's queue to ensure logits computation is complete
    // This is needed because the sampler might have a different queue
    sycl::queue & buf_queue = *buf_ctx->stream;
    buf_queue.wait();

    // Also sync the device's default queue (used by some copy operations)
    ggml_sycl_set_device(buf_ctx->device);
    dpct::dev_mgr::instance().get_device(buf_ctx->device).default_queue().wait();

    // Get GPU pointer for logits
    float * logits_gpu = (float *) logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    // Get vocab size from tensor (ne[0] is vocab dimension)
    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    // Get number of batch entries (ne[1] is batch dimension)
    int n_batch = logits_tensor->ne[1];
    GGML_ASSERT(idx >= 0 && idx < n_batch);

    // Offset to the correct batch entry
    float * logits_at_idx = logits_gpu + (size_t) idx * n_vocab;

    // Debug: read first few logits to verify data is valid
    sycl::queue & q = *sampler->sycl_ctx->stream();
    float         debug_logits[5];
    q.memcpy(debug_logits, logits_at_idx, 5 * sizeof(float)).wait();
    GGML_LOG_INFO(
        "[GPU SAMPLE DEBUG] idx=%d, logits_gpu=%p, logits_at_idx=%p, first 5 logits: %.2f %.2f %.2f %.2f %.2f\n", idx,
        (void *) logits_gpu, (void *) logits_at_idx, debug_logits[0], debug_logits[1], debug_logits[2], debug_logits[3],
        debug_logits[4]);

    // Set up config with full parameters
    ggml_sycl_sampler_config config;
    config.temp   = temp;
    config.top_k  = top_k;
    config.top_p  = top_p;
    config.min_p  = min_p;
    config.seed   = sampler->state.rng_state;
    config.greedy = (temp == 0.0f);

    // Call GPU sampler with offset logits
    int32_t sampled_token = ggml_sycl_sample_token(*sampler->sycl_ctx, logits_at_idx, config, sampler->state);
    GGML_LOG_INFO("[GPU SAMPLE DEBUG] sampled_token=%d (temp=%.2f, greedy=%d)\n", sampled_token, temp, config.greedy);
    return sampled_token;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in GPU sampling (full): %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_sample_token_async(ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor, float temp) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    float * logits_gpu = (float *) logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    ggml_sycl_sampler_config config;
    config.temp   = temp;
    config.top_k  = 0;
    config.top_p  = 1.0f;
    config.min_p  = 0.0f;
    config.seed   = sampler->state.rng_state;
    config.greedy = (temp == 0.0f);

    // Call async GPU sampler (doesn't wait for result)
    ggml_sycl_sample_token_async(*sampler->sycl_ctx, logits_gpu, config, sampler->state);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in async GPU sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t ggml_backend_sycl_sample_token_get(ggml_sycl_sampler_t sampler) try {
    GGML_ASSERT(sampler != nullptr);
    return ggml_sycl_sample_token_wait(*sampler->sycl_ctx, sampler->state);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error getting GPU sampling result: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

// ===========================================================================
// Multi-step GPU Sampling API
// ===========================================================================

void ggml_backend_sycl_sampler_reset_buffer(ggml_sycl_sampler_t sampler) try {
    GGML_ASSERT(sampler != nullptr);
    ggml_sycl_sampler_reset_buffer(sampler->state);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error resetting sampler buffer: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_sample_token_to_device(ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor, float temp) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    float * logits_gpu = (float *) logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    ggml_sycl_sampler_config config;
    config.temp   = temp;
    config.top_k  = 0;
    config.top_p  = 1.0f;
    config.min_p  = 0.0f;
    config.seed   = sampler->state.rng_state;
    config.greedy = (temp == 0.0f);

    return ggml_sycl_sample_token_to_buffer(*sampler->sycl_ctx, logits_gpu, config, sampler->state);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in sample_token_to_device: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_sample_token_to_device_full(ggml_sycl_sampler_t sampler,
                                                  ggml_tensor *       logits_tensor,
                                                  float               temp,
                                                  int                 top_k,
                                                  float               top_p,
                                                  float               min_p) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    float * logits_gpu = (float *) logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    ggml_sycl_sampler_config config;
    config.temp   = temp;
    config.top_k  = top_k;
    config.top_p  = top_p;
    config.min_p  = min_p;
    config.seed   = sampler->state.rng_state;
    config.greedy = (temp == 0.0f);

    return ggml_sycl_sample_token_to_buffer(*sampler->sycl_ctx, logits_gpu, config, sampler->state);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in sample_token_to_device_full: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t * ggml_backend_sycl_get_sampled_token_ptr(ggml_sycl_sampler_t sampler, int index) try {
    GGML_ASSERT(sampler != nullptr);
    return ggml_sycl_sampler_get_token_ptr(sampler->state, index);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error getting sampled token ptr: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t * ggml_backend_sycl_get_current_token_ptr(ggml_sycl_sampler_t sampler) try {
    GGML_ASSERT(sampler != nullptr);
    return ggml_sycl_sampler_get_current_token_ptr(sampler->state);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error getting current token ptr: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_get_sampled_tokens(ggml_sycl_sampler_t sampler, int32_t * tokens, int max_tokens) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(tokens != nullptr);
    return ggml_sycl_sampler_get_tokens(*sampler->sycl_ctx, sampler->state, tokens, max_tokens);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error getting sampled tokens: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_get_token_count(ggml_sycl_sampler_t sampler) try {
    GGML_ASSERT(sampler != nullptr);
    return sampler->state.token_count;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error getting token count: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_get_token_buffer_size(void) {
    return GPU_SAMPLER_TOKEN_BUFFER_SIZE;
}

// ===========================================================================
// Speculative Decoding Verification API Implementation
// ===========================================================================

int ggml_backend_sycl_verify_speculative(ggml_sycl_sampler_t sampler,
                                         ggml_tensor *       all_logits,
                                         const int32_t *     draft_tokens,
                                         int                 n_draft,
                                         int                 logits_offset) try {
    if (!sampler || !all_logits || !draft_tokens || n_draft <= 0) {
        return 0;
    }

    if (!sampler->sycl_ctx || !sampler->sycl_ctx->stream()) {
        return 0;
    }

    // Get logits data pointer from tensor - use tensor->data directly like sample_token_full
    // This ensures we use the same pointer as sampling functions
    const float * logits_data = (const float *) all_logits->data;
    if (!logits_data) {
        return 0;
    }

    const int n_vocab   = sampler->state.n_vocab;
    const int n_outputs = all_logits->ne[1];  // Number of output positions in batch

    // Validate that we have enough logits for the verification
    if (logits_offset + n_draft > n_outputs) {
        GGML_LOG_ERROR("SYCL verify_speculative: logits_offset (%d) + n_draft (%d) > n_outputs (%d)\n", logits_offset,
                       n_draft, n_outputs);
        return 0;
    }

    // Offset the logits pointer to start from the correct position
    const float * logits_at_offset = logits_data + (size_t) logits_offset * n_vocab;

    // Use the host wrapper which copies draft tokens to device
    return ggml_sycl_verify_speculative_host(*sampler->sycl_ctx, sampler->state, logits_at_offset, draft_tokens,
                                             n_draft, n_vocab);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error verifying speculative: %s\n", exc.what());
    return 0;
}

int ggml_backend_sycl_verify_speculative_with_tokens(ggml_sycl_sampler_t sampler,
                                                     ggml_tensor *       all_logits,
                                                     const int32_t *     draft_tokens,
                                                     int32_t *           sampled_tokens_out,
                                                     int                 n_draft,
                                                     int                 logits_offset) try {
    if (!sampler || !all_logits || !draft_tokens || !sampled_tokens_out || n_draft <= 0) {
        return 0;
    }

    if (!sampler->sycl_ctx || !sampler->sycl_ctx->stream()) {
        return 0;
    }

    // Get the backend context from the tensor's buffer - this is where the data actually lives
    ggml_backend_buffer_t buffer = all_logits->buffer;
    if (!buffer) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: tensor has no buffer\n");
        return 0;
    }

    if (!ggml_backend_buffer_is_sycl(buffer)) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: tensor buffer is not SYCL\n");
        return 0;
    }

    // Get the buffer's SYCL context - this is the actual device where data resides
    ggml_backend_sycl_buffer_context * buf_ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    if (!buf_ctx || !buf_ctx->stream) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: buffer has no valid context\n");
        return 0;
    }

    // Sync ALL possible queues to ensure data is visible:
    // 1. Buffer's stream (where buffer was created)
    // 2. Sampler's stream (our verification context)
    // 3. Device's default queue (catch-all)
    buf_ctx->stream->wait();
    sampler->sycl_ctx->stream()->wait();

    // Also sync the device's default queue
    ggml_sycl_set_device(buf_ctx->device);
    sycl::queue & default_q = dpct::dev_mgr::instance().get_device(buf_ctx->device).default_queue();
    default_q.wait();

    // Get logits data pointer from tensor
    const float * logits_data = (const float *) all_logits->data;
    if (!logits_data) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: tensor has no data\n");
        return 0;
    }

    const int n_vocab   = sampler->state.n_vocab;
    const int n_outputs = all_logits->ne[1];  // Number of output positions in batch

    // Validate that we have enough logits for the verification
    if (logits_offset + n_draft > n_outputs) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: logits_offset (%d) + n_draft (%d) > n_outputs (%d)\n",
                       logits_offset, n_draft, n_outputs);
        return 0;
    }

    // Offset the logits pointer to start from the correct position
    const float * logits_at_offset = logits_data + (size_t) logits_offset * n_vocab;

    // Use the extended function that also returns sampled tokens
    return ggml_sycl_verify_speculative_with_tokens(*sampler->sycl_ctx, sampler->state, logits_at_offset, draft_tokens,
                                                    sampled_tokens_out, n_draft, n_vocab);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error verifying speculative with tokens: %s\n", exc.what());
    return 0;
}

int ggml_backend_sycl_verify_speculative_from_ptr(ggml_sycl_sampler_t sampler,
                                                  const float *       gpu_logits,
                                                  int                 n_vocab,
                                                  int                 n_outputs,
                                                  const int32_t *     draft_tokens,
                                                  int32_t *           sampled_tokens_out,
                                                  int                 n_draft,
                                                  int                 logits_offset) try {
    if (sampler == nullptr || gpu_logits == nullptr || draft_tokens == nullptr) {
        GGML_LOG_ERROR("SYCL verify_speculative_from_ptr: null argument\n");
        return 0;
    }

    if (n_draft <= 0 || n_vocab <= 0) {
        GGML_LOG_ERROR("SYCL verify_speculative_from_ptr: invalid dimensions\n");
        return 0;
    }

    if (logits_offset + n_draft > n_outputs) {
        GGML_LOG_ERROR("SYCL verify_speculative_from_ptr: logits_offset (%d) + n_draft (%d) > n_outputs (%d)\n",
                       logits_offset, n_draft, n_outputs);
        return 0;
    }

    // Offset into the logits buffer
    const float * logits_at_offset = gpu_logits + logits_offset * n_vocab;

    // Use the internal function directly with GPU pointer
    return ggml_sycl_verify_speculative_with_tokens(*sampler->sycl_ctx, sampler->state, logits_at_offset, draft_tokens,
                                                    sampled_tokens_out, n_draft, n_vocab);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error verifying speculative from ptr: %s\n", exc.what());
    return 0;
}

int ggml_backend_sycl_verify_speculative_from_host(ggml_sycl_sampler_t sampler,
                                                   const float *       host_logits,
                                                   int                 n_vocab,
                                                   int                 n_outputs,
                                                   const int32_t *     draft_tokens,
                                                   int32_t *           sampled_tokens_out,
                                                   int                 n_draft,
                                                   int                 logits_offset) try {
    if (sampler == nullptr || host_logits == nullptr || draft_tokens == nullptr) {
        GGML_LOG_ERROR("SYCL verify_speculative_from_host: null argument\n");
        return 0;
    }

    if (n_draft <= 0 || n_vocab <= 0) {
        GGML_LOG_ERROR("SYCL verify_speculative_from_host: invalid dimensions\n");
        return 0;
    }

    if (logits_offset + n_draft > n_outputs) {
        GGML_LOG_ERROR("SYCL verify_speculative_from_host: logits_offset (%d) + n_draft (%d) > n_outputs (%d)\n",
                       logits_offset, n_draft, n_outputs);
        return 0;
    }

    // Copy logits from host to device
    sycl::queue & q           = *sampler->sycl_ctx->stream();
    size_t        logits_size = (size_t) n_outputs * n_vocab * sizeof(float);
    float *       gpu_logits  = sycl::malloc_device<float>(n_outputs * n_vocab, q);
    if (!gpu_logits) {
        GGML_LOG_ERROR("SYCL verify_speculative_from_host: failed to allocate device memory\n");
        return 0;
    }

    q.memcpy(gpu_logits, host_logits, logits_size).wait();

    // Offset into the logits buffer
    const float * logits_at_offset = gpu_logits + logits_offset * n_vocab;

    // Use the internal function
    int result = ggml_sycl_verify_speculative_with_tokens(*sampler->sycl_ctx, sampler->state, logits_at_offset,
                                                          draft_tokens, sampled_tokens_out, n_draft, n_vocab);

    // Cleanup
    sycl::free(gpu_logits, q);

    return result;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error verifying speculative from host: %s\n", exc.what());
    return 0;
}

// ===========================================================================
// Continuous Batching API Implementation (Multi-sequence GPU Sampling)
// ===========================================================================

// Multi-sequence sampler struct - holds backend context and multi-seq state
struct ggml_sycl_multi_seq_sampler_wrapper {
    ggml_backend_t              backend;
    ggml_backend_sycl_context * sycl_ctx;
    ggml_sycl_multi_seq_sampler state;
    uint32_t                    base_seed;
};

ggml_sycl_multi_seq_sampler_t ggml_backend_sycl_multi_seq_sampler_create(ggml_backend_t backend,
                                                                         int            max_seqs,
                                                                         int            n_vocab,
                                                                         uint32_t       seed) try {
    GGML_ASSERT(ggml_backend_is_sycl(backend));
    GGML_ASSERT(max_seqs > 0 && max_seqs <= CONT_BATCH_MAX_SEQS);
    GGML_ASSERT(n_vocab > 0);

    auto wrapper       = new ggml_sycl_multi_seq_sampler_wrapper();
    wrapper->backend   = backend;
    wrapper->sycl_ctx  = (ggml_backend_sycl_context *) backend->context;
    wrapper->base_seed = seed;

    // Initialize multi-sequence sampler state
    sycl::queue & q = *wrapper->sycl_ctx->stream();
    ggml_sycl_multi_seq_sampler_init(wrapper->state, q, max_seqs, n_vocab);

    return (ggml_sycl_multi_seq_sampler_t) wrapper;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error creating multi-seq sampler: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_multi_seq_sampler_free(ggml_sycl_multi_seq_sampler_t sampler) try {
    if (sampler == nullptr) {
        return;
    }

    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;
    if (wrapper->state.initialized) {
        sycl::queue & q = *wrapper->sycl_ctx->stream();
        ggml_sycl_multi_seq_sampler_free(wrapper->state, q);
    }

    delete wrapper;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error freeing multi-seq sampler: %s\n", exc.what());
}

bool ggml_backend_sycl_multi_seq_add(ggml_sycl_multi_seq_sampler_t sampler, int seq_id, float temp) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    if (seq_id < 0 || seq_id >= wrapper->state.max_seqs) {
        return false;
    }

    sycl::queue & q        = *wrapper->sycl_ctx->stream();
    // Generate unique seed for this sequence
    uint32_t      seq_seed = wrapper->base_seed + seq_id;
    int           slot     = ggml_sycl_multi_seq_add(wrapper->state, q, seq_id, temp, seq_seed);

    return slot >= 0;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error adding sequence: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

bool ggml_backend_sycl_multi_seq_remove(ggml_sycl_multi_seq_sampler_t sampler, int seq_id) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    // Check if sequence is active
    bool was_active = false;
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        if (wrapper->state.h_seq_active[i] && wrapper->state.h_seq_ids[i] == seq_id) {
            was_active = true;
            break;
        }
    }

    if (!was_active) {
        return false;
    }

    sycl::queue & q = *wrapper->sycl_ctx->stream();
    ggml_sycl_multi_seq_remove(wrapper->state, q, seq_id);
    return true;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error removing sequence: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_multi_seq_set_temp(ggml_sycl_multi_seq_sampler_t sampler, int seq_id, float temp) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    // Find slot for this sequence
    int slot = -1;
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        if (wrapper->state.h_seq_active[i] && wrapper->state.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        return;
    }

    sycl::queue & q = *wrapper->sycl_ctx->stream();
    q.memcpy(wrapper->state.temperatures + slot, &temp, sizeof(float)).wait();
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error setting temperature: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_multi_seq_set_params(ggml_sycl_multi_seq_sampler_t sampler,
                                            int                           seq_id,
                                            float                         temp,
                                            int                           top_k,
                                            float                         top_p,
                                            float                         min_p) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    // Find slot for this sequence
    int slot = -1;
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        if (wrapper->state.h_seq_active[i] && wrapper->state.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        return;
    }

    sycl::queue & q = *wrapper->sycl_ctx->stream();

    // Set greedy flag based on temperature
    uint8_t is_greedy = (temp <= 0.0f) ? 1 : 0;

    // Convert top_k to int32_t for device memory
    int32_t top_k_val = top_k;

    // Copy all params to device memory (async, then wait at end)
    q.memcpy(wrapper->state.temperatures + slot, &temp, sizeof(float));
    q.memcpy(wrapper->state.top_k_values + slot, &top_k_val, sizeof(int32_t));
    q.memcpy(wrapper->state.top_p_values + slot, &top_p, sizeof(float));
    q.memcpy(wrapper->state.min_p_values + slot, &min_p, sizeof(float));
    q.memcpy(wrapper->state.greedy_flags + slot, &is_greedy, sizeof(uint8_t));
    q.wait();
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error setting params: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_get_active_count(ggml_sycl_multi_seq_sampler_t sampler) {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;
    return wrapper->state.n_active;
}

int ggml_backend_sycl_multi_seq_sample(ggml_sycl_multi_seq_sampler_t sampler, float * batched_logits, bool greedy) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(batched_logits != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    if (wrapper->state.n_active == 0) {
        return 0;
    }

    sycl::queue & q = *wrapper->sycl_ctx->stream();
    ggml_sycl_multi_seq_sample(wrapper->state, q, batched_logits, greedy);

    return wrapper->state.n_active;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in multi-seq sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_sample_indexed(ggml_sycl_multi_seq_sampler_t sampler,
                                               float *                       logits_base,
                                               const int *                   seq_ids,
                                               const int *                   batch_indices,
                                               int                           n_seqs) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_base != nullptr);
    GGML_ASSERT(batch_indices != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    if (n_seqs == 0) {
        return 0;
    }

    sycl::queue & q = *wrapper->sycl_ctx->stream();

    // Copy batch indices to device
    int * d_batch_indices = sycl::malloc_device<int>(n_seqs, q);
    q.memcpy(d_batch_indices, batch_indices, n_seqs * sizeof(int)).wait();

    // Copy seq_ids to device for output indexing
    int * d_seq_ids = sycl::malloc_device<int>(n_seqs, q);
    if (seq_ids != nullptr) {
        q.memcpy(d_seq_ids, seq_ids, n_seqs * sizeof(int)).wait();
    } else {
        // Default: seq_id = input index
        std::vector<int> default_ids(n_seqs);
        for (int i = 0; i < n_seqs; i++) {
            default_ids[i] = i;
        }
        q.memcpy(d_seq_ids, default_ids.data(), n_seqs * sizeof(int)).wait();
    }

    // Update active sequences if seq_ids provided
    if (seq_ids != nullptr) {
        // Clear active flags
        std::fill(wrapper->state.h_seq_active.begin(), wrapper->state.h_seq_active.end(), 0);
        for (int i = 0; i < n_seqs; i++) {
            int seq_id = seq_ids[i];
            if (seq_id >= 0 && seq_id < wrapper->state.max_seqs) {
                wrapper->state.h_seq_active[seq_id] = 1;
            }
        }
        wrapper->state.n_active = n_seqs;

        // Copy active flags to device
        q.memcpy(wrapper->state.seq_active, wrapper->state.h_seq_active.data(), wrapper->state.max_seqs * sizeof(int))
            .wait();
    }

    // Call internal indexed sampling function
    ggml_sycl_multi_seq_sample_indexed(wrapper->state, q, logits_base, d_batch_indices, d_seq_ids, n_seqs);

    // Free temporary device memory
    sycl::free(d_batch_indices, q);
    sycl::free(d_seq_ids, q);

    return n_seqs;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in indexed multi-seq sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_get_tokens(ggml_sycl_multi_seq_sampler_t sampler,
                                           int32_t *                     tokens_out,
                                           int *                         seq_ids_out,
                                           int                           max_tokens) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(tokens_out != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    if (wrapper->state.n_active == 0) {
        return 0;
    }

    sycl::queue & q      = *wrapper->sycl_ctx->stream();
    int           n_copy = std::min(wrapper->state.n_active, max_tokens);

    // Copy sampled tokens from device
    std::vector<int32_t> all_tokens(wrapper->state.max_seqs);
    q.memcpy(all_tokens.data(), wrapper->state.sampled_tokens, wrapper->state.max_seqs * sizeof(int32_t)).wait();

    // Debug: print raw device array
    GGML_LOG_DEBUG("[get_tokens] raw sampled_tokens array (max_seqs=%d):\n", wrapper->state.max_seqs);
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        GGML_LOG_DEBUG("  sampled_tokens[%d] = %d, active=%d\n", i, all_tokens[i], wrapper->state.h_seq_active[i]);
    }

    // Copy only active sequences' tokens
    // When h_seq_active[i] is true, i IS the seq_id because indexed sampling
    // writes to sampled_tokens[seq_id]
    int idx = 0;
    for (int i = 0; i < wrapper->state.max_seqs && idx < n_copy; i++) {
        if (wrapper->state.h_seq_active[i]) {
            tokens_out[idx] = all_tokens[i];
            if (seq_ids_out != nullptr) {
                seq_ids_out[idx] = i;  // i IS the seq_id
            }
            GGML_LOG_DEBUG("  [get_tokens] collecting: out_idx=%d <- slot=%d, token=%d\n", idx, i, all_tokens[i]);
            idx++;
        }
    }

    return idx;
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error getting multi-seq tokens: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t * ggml_backend_sycl_multi_seq_get_token_ptr(ggml_sycl_multi_seq_sampler_t sampler, int seq_id) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    return ggml_sycl_multi_seq_get_current_token_ptr(wrapper->state, seq_id);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error getting multi-seq token ptr: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_get_active_seq_ids(ggml_sycl_multi_seq_sampler_t sampler,
                                                   int *                         seq_ids_out,
                                                   int                           max_seqs) {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(seq_ids_out != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    int idx = 0;
    for (int i = 0; i < wrapper->state.max_seqs && idx < max_seqs; i++) {
        if (wrapper->state.h_seq_active[i]) {
            seq_ids_out[idx++] = wrapper->state.h_seq_ids[i];
        }
    }

    return idx;
}

void ggml_backend_sycl_multi_seq_reset_buffer(ggml_sycl_multi_seq_sampler_t sampler, int seq_id) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    // Find slot for this sequence
    int slot = -1;
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        if (wrapper->state.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        return;
    }

    sycl::queue & q    = *wrapper->sycl_ctx->stream();
    int           zero = 0;
    q.memcpy(wrapper->state.write_indices + slot, &zero, sizeof(int));
    q.memcpy(wrapper->state.token_counts + slot, &zero, sizeof(int)).wait();
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error resetting multi-seq buffer: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_get_ring_tokens(ggml_sycl_multi_seq_sampler_t sampler,
                                                int                           seq_id,
                                                int32_t *                     tokens_out,
                                                int                           max_tokens) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(tokens_out != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper *) sampler;

    sycl::queue & q = *wrapper->sycl_ctx->stream();
    return ggml_sycl_multi_seq_get_tokens(wrapper->state, q, seq_id, tokens_out, max_tokens);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error getting ring tokens: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

// ===========================================================================
// Batched Logits Management
// ===========================================================================

// Thread-local storage for batch logits info (set by llama layer)
static thread_local struct {
    float * logits_ptr;  // Device pointer to [n_tokens, n_vocab] logits
    int     n_tokens;    // Number of tokens with logits
    int     n_vocab;     // Vocabulary size
    bool    valid;
} g_batch_logits_info = { nullptr, 0, 0, false };

// Called by llama layer after decode to set logits info
void ggml_backend_sycl_set_batch_logits_info(float * logits_device_ptr, int n_tokens, int n_vocab) {
    g_batch_logits_info.logits_ptr = logits_device_ptr;
    g_batch_logits_info.n_tokens   = n_tokens;
    g_batch_logits_info.n_vocab    = n_vocab;
    g_batch_logits_info.valid      = true;
}

void ggml_backend_sycl_clear_batch_logits_info(void) {
    g_batch_logits_info.valid = false;
}

float * ggml_backend_sycl_get_batch_logits_ptr(void * ctx, int batch_idx) {
    GGML_UNUSED(ctx);

    if (!g_batch_logits_info.valid) {
        return nullptr;
    }
    if (batch_idx < 0 || batch_idx >= g_batch_logits_info.n_tokens) {
        return nullptr;
    }

    return g_batch_logits_info.logits_ptr + batch_idx * g_batch_logits_info.n_vocab;
}

int ggml_backend_sycl_get_batch_logits_count(void * ctx) {
    GGML_UNUSED(ctx);

    if (!g_batch_logits_info.valid) {
        return 0;
    }
    return g_batch_logits_info.n_tokens;
}

// ===========================================================================
// KV Cache Synchronization (Barrier-based)
// ===========================================================================

void ggml_backend_sycl_submit_barrier(ggml_backend_t backend) {
    // Submit a barrier after graph execution
    // The barrier ensures all prior commands on the queue complete before subsequent ones
    // This is lighter-weight than full queue sync - it returns immediately after submission
    if (!ggml_backend_is_sycl(backend)) {
        return;
    }

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
    if (!sycl_ctx) {
        return;
    }

    try {
        // Submit barrier and store the event
        sycl_ctx->barrier_event       = sycl_ctx->stream()->ext_oneapi_submit_barrier();
        sycl_ctx->has_pending_barrier = true;
        GGML_SYCL_DEBUG("[SYCL-BARRIER] Submitted barrier after graph execution\n");
    } catch (const sycl::exception & exc) {
        GGML_LOG_ERROR("SYCL barrier submit failed: %s\n", exc.what());
    }
}

void ggml_backend_sycl_wait_barrier(ggml_backend_t backend) {
    // Wait for the previously submitted barrier to complete
    // Call this before the next ubatch's graph_compute
    if (!ggml_backend_is_sycl(backend)) {
        return;
    }

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
    if (!sycl_ctx || !sycl_ctx->has_pending_barrier) {
        return;
    }

    try {
        if (sycl_ctx->barrier_event.has_value()) {
            GGML_SYCL_DEBUG("[SYCL-BARRIER] Waiting on barrier event...\n");
            sycl_ctx->barrier_event->wait();
            GGML_SYCL_DEBUG("[SYCL-BARRIER] Barrier wait complete\n");
        }
        sycl_ctx->has_pending_barrier = false;
        sycl_ctx->barrier_event.reset();
    } catch (const sycl::exception & exc) {
        GGML_LOG_ERROR("SYCL barrier wait failed: %s\n", exc.what());
        sycl_ctx->has_pending_barrier = false;
        sycl_ctx->barrier_event.reset();
    }
}

// ===========================================================================
// Device Memory Utilities
// ===========================================================================

void ggml_backend_sycl_copy_device_to_tensor(void * src_device_ptr, ggml_tensor * tensor, size_t size) try {
    GGML_ASSERT(src_device_ptr != nullptr);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);

    // Get the SYCL context from the tensor's buffer
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(ctx != nullptr);

    // Set device and get queue
    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    // Device-to-device copy
    void * dst = tensor->data;
    stream.memcpy(dst, src_device_ptr, size).wait();
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in device-to-tensor copy: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_copy_tensor_to_buffer(ggml_backend_t        backend,
                                             ggml_tensor *         src_tensor,
                                             ggml_backend_buffer_t dst_buffer,
                                             size_t                dst_offset,
                                             size_t                size) try {
    GGML_ASSERT(backend != nullptr);
    GGML_ASSERT(src_tensor != nullptr);
    GGML_ASSERT(src_tensor->buffer != nullptr);
    GGML_ASSERT(dst_buffer != nullptr);

    // Get device from backend
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
    int                         device   = sycl_ctx->device;

    // Set device and get the CONTEXT's stream (same as compute graph)
    // CRITICAL: Must use context stream, not default_queue, to ensure synchronization
    // with the compute graph that filled the source tensor
    ggml_sycl_set_device(device);
    sycl::queue & stream = *sycl_ctx->stream(device, 0);

    // Get source pointer from tensor
    void * src = src_tensor->data;

    // Get destination pointer from buffer with offset
    void * dst_base = ggml_backend_buffer_get_base(dst_buffer);
    char * dst      = (char *) dst_base + dst_offset;

    // Device-to-device copy on the SAME queue as compute
    // The .wait() ensures this copy completes before we return
    stream.memcpy(dst, src, size).wait();
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in tensor-to-buffer copy: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void * ggml_backend_sycl_buffer_get_ptr(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return nullptr;
    }
    return ggml_backend_buffer_get_base(buffer);
}

void ggml_backend_sycl_buffer_get_async(ggml_backend_buffer_t buffer,
                                        const void *          src_ptr,
                                        void *                dst,
                                        size_t                offset,
                                        size_t                size) try {
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(src_ptr != nullptr);
    GGML_ASSERT(dst != nullptr);

    // Get the SYCL context from the buffer
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    GGML_ASSERT(ctx != nullptr);

    // Set device and get queue
    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    // Async device-to-host copy (NO wait - will complete on next synchronize)
    const char * src = (const char *) src_ptr + offset;
    stream.memcpy(dst, src, size);
    // Note: Do NOT call .wait() here - this is the async optimization!
    // The copy will complete when ggml_backend_sycl_synchronize is called
    GGML_SYCL_DEBUG("[SYCL] Async buffer get: %zu bytes from GPU to host (deferred)\n", size);
} catch (const sycl::exception & exc) {
    GGML_LOG_ERROR("SYCL error in async buffer get: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

// host buffer type

static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_sycl_host_free(buffer->context);
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                             size_t                     size) {
    void * ptr = ggml_sycl_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        // WARNING: This breaks lazy_moe! The returned buffer won't be SYCL-accessible.
        // This typically happens when trying to allocate too much pinned memory.
        GGML_LOG_WARN(
            "[SYCL] sycl::malloc_host(%.1f MB) failed, falling back to CPU buffer. "
            "lazy_moe will NOT work - expert weights won't be accessible from GPU.\n",
            size / (1024.0f * 1024.0f));
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type\n");
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_type_host = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_sycl_host_buffer_type_name,
                           /* .alloc_buffer     = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                           /* .get_max_size     = */ NULL,  // TODO: return device.maxBufferLength
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                           /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_sycl_buffer_type_host;
}

// Host compute buffer type - uses SYCL host memory with SYCL buffer interface
// Unlike the regular host buffer type (which uses CPU buffer interface),
// this buffer type is usable by SYCL kernels and is used for TP compute buffers
// to allow cross-device data sharing.
//
// IMPORTANT: For TP mode, all devices share the SAME buffer allocation.
// This allows non-sharded ops (running only on device 0) to write results
// that are immediately visible to device 1 for sharded ops.

static const char * ggml_backend_sycl_host_compute_buffer_type_name(ggml_backend_buffer_type_t buft) {
    static std::string name = GGML_SYCL_NAME "_HostCompute";
    return name.c_str();
    GGML_UNUSED(buft);
}

// Allocation function for compute buffers
// For TP mode: Allocate SINGLE malloc_host buffer accessible by ALL TP devices
// This is the Megatron-LM approach: activations in shared/host memory, weights sharded per device
// For non-TP mode: Uses regular device buffer allocation
static ggml_backend_buffer_t ggml_backend_sycl_host_compute_buffer_alloc(ggml_backend_buffer_type_t buft,
                                                                         size_t                     size) try {
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *) buft->context;

    // For TP mode, allocate SINGLE host buffer accessible by ALL devices
    // All devices read/write to the same memory - no need for data copies
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        int primary_device = buft_ctx->device;
        size               = std::max(size, (size_t) 1);

        GGML_SYCL_DEBUG("SYCL TP: Allocating SHARED compute buffer (malloc_shared) for %d TP devices (size=%zu)\n",
                        g_sycl_tp_config.world_size, size);

        // Get shared context for TP mode
        sycl::context * tp_context = ggml_sycl_get_tp_context();
        if (tp_context == nullptr) {
            GGML_LOG_ERROR("%s: TP shared context not initialized\n", __func__);
            return nullptr;
        }

        // Allocate HOST memory in the SHARED CONTEXT so all devices can access it
        // malloc_host allocates pinned host memory accessible from all devices in the context
        void * shared_ptr = nullptr;
        SYCL_CHECK(CHECK_TRY_ERROR(shared_ptr = (void *) sycl::malloc_host(size, *tp_context)));

        if (!shared_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %lu Bytes of host memory for TP\n", __func__, size);
            return nullptr;
        }
        GGML_SYCL_DEBUG("SYCL TP: Allocated HOST compute buffer (malloc_host): %p (size=%zu)\n", shared_ptr, size);

        // Get primary device's queue for buffer context
        ggml_sycl_set_device(primary_device);
        queue_ptr primary_stream = ggml_sycl_get_tp_queue(primary_device);
        if (primary_stream == nullptr) {
            auto & primary_dpct_dev = dpct::dev_mgr::instance().get_device(primary_device);
            primary_stream          = &(primary_dpct_dev.default_queue());
        }

        ggml_backend_sycl_buffer_context * ctx =
            new ggml_backend_sycl_buffer_context(primary_device, shared_ptr, primary_stream);
        ctx->is_tp_compute_buffer = true;

        // In multi-process mode, we only have ONE device visible per process
        // world_size is the MPI world size, not the number of local devices
        if (g_sycl_tp_config.is_multiprocess) {
            // Multi-process: each process has only one device (device 0 locally)
            int local_dev               = 0;  // Local device ID is always 0 in multi-process mode
            ctx->tp_dev_ptrs[local_dev] = shared_ptr;
            ctx->tp_streams[local_dev]  = ggml_sycl_get_tp_queue(local_dev);
            if (ctx->tp_streams[local_dev] == nullptr) {
                auto & dpct_dev            = dpct::dev_mgr::instance().get_device(local_dev);
                ctx->tp_streams[local_dev] = &(dpct_dev.default_queue());
            }
            GGML_SYCL_DEBUG("SYCL TP: Multi-process rank %d using local compute buffer: %p\n",
                            g_sycl_tp_config.mpi_rank, shared_ptr);
        } else {
            // Single-process multi-device: ALL devices share the SAME pointer
            for (int i = 0; i < g_sycl_tp_config.world_size; i++) {
                int dev_id               = g_sycl_tp_config.devices[i];
                ctx->tp_dev_ptrs[dev_id] = shared_ptr;  // Same pointer for all!
                ctx->tp_streams[dev_id]  = ggml_sycl_get_tp_queue(dev_id);
                if (ctx->tp_streams[dev_id] == nullptr) {
                    auto & dpct_dev         = dpct::dev_mgr::instance().get_device(dev_id);
                    ctx->tp_streams[dev_id] = &(dpct_dev.default_queue());
                }
                GGML_SYCL_DEBUG("SYCL TP: Device %d using shared compute buffer: %p\n", dev_id, shared_ptr);
            }
        }

        return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
    }

    // Non-TP mode: use regular allocation
    return ggml_backend_sycl_buffer_type_alloc_buffer(buft, size);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static const ggml_backend_buffer_type_i ggml_backend_sycl_host_compute_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_host_compute_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_host_compute_buffer_alloc,  // Per-device buffer for TP
    /* .get_alignment    = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,  // Not a CPU host buffer - it's SYCL host memory
};

ggml_backend_buffer_type_t ggml_backend_sycl_host_compute_buffer_type(int device) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    auto dev_count = ggml_backend_sycl_get_device_count();
    if (device >= dev_count || device < 0) {
        GGML_LOG_ERROR("ggml_backend_sycl_host_compute_buffer_type error: device_index:%d is out of range [0, %d]\n",
                       device, dev_count - 1);
        GGML_ASSERT(device < dev_count);
    }

    static struct ggml_backend_buffer_type ggml_backend_sycl_host_compute_buffer_types[GGML_SYCL_MAX_DEVICES];
    static bool                            initialized = false;

    if (!initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto &    device_i                             = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream                               = &(device_i.default_queue());
            // For TP mode: each device gets its own DEVICE memory compute buffer
            // This allows parallel execution with no cross-device memory issues
            ggml_backend_sycl_host_compute_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_host_compute_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */
                new ggml_backend_sycl_buffer_type_context{ i, GGML_SYCL_NAME "_Compute" + std::to_string(i),
                                                          GGML_SYCL_MEM_DEVICE, stream },
            };
        }
        initialized = true;
    }
    return &ggml_backend_sycl_host_compute_buffer_types[device];
}

// buffer pool for sycl (legacy)
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;

    int       device;
    queue_ptr qptr;

    struct ggml_sycl_buffer {
        void * ptr  = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t           pool_size                     = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, int device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        int    nnz      = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int    ibest     = -1;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                if (b.size > max_size) {
                    max_size = b.size;
                }
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest     = i;
                        if (!best_diff) {
                            void * ptr   = b.ptr;
                            *actual_size = b.size;
                            b.ptr        = nullptr;
                            b.size       = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_sycl_buffer & b   = buffer_pool[ibest];
            void *             ptr = b.ptr;
            *actual_size           = b.size;
            b.ptr                  = nullptr;
            b.size                 = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_device(look_ahead_size, *qptr)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            return nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                       (uint32_t) (max_size / 1024 / 1024), (uint32_t) (g_sycl_pool_size[id] / 1024 / 1024),
                       (uint32_t) (size / 1024 / 1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%lu, return %p\n", look_ahead_size, ptr);
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, *qptr)));
        pool_size -= size;
    }
};

struct ggml_sycl_pool_host : public ggml_sycl_pool {
    queue_ptr qptr;
    int       device;

    inline static int counter{ 0 };

    struct ggml_sycl_buffer {
        void * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    static constexpr int          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, int device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
        counter = 0;
    }

    void * alloc(size_t size, size_t * actual_size) override {
        if (counter == MAX_POOL_SIZE) {
            ggml_sycl_buffer b   = buffer_pool[0];
            void *           ptr = b.ptr;
            *actual_size         = b.size;
            counter              = 1;
            return ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];

        if (b.ptr == nullptr) {
            void * ptr;

            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *) sycl::malloc_host(size, *qptr)));
            if (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on host\n", __func__, size);
                return nullptr;
            }
            pool_size += size;
            *actual_size = size;
            counter      = counter + 1;
            return ptr;
        } else {
            ++counter;
            b.size = size;
            return b.ptr;
        }
    }

    void free(void * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise do nothing, pointers will be freed once the pool is deallocated.
        for (int i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                return;
            }
        }
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, int device) {
    // return pool for the host to speed up memory management
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_host(qptr, device));
}

ggml_backend_sycl_context::~ggml_backend_sycl_context() = default;

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, int device) {
    // TBD: NO VMM support
    // if (ggml_sycl_info().devices[device].vmm) {
    //     return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(device));
    // }
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}

void ggml_backend_sycl_context::init_kv_offload(const ggml_sycl::kv_offload_config & config) {
    if (kv_offload_mgr_) {
        GGML_LOG_DEBUG("kv_offload: already initialized\n");
        return;
    }

    if (config.gpu_kv_budget == 0) {
        GGML_LOG_WARN("kv_offload: invalid gpu_kv_budget (0)\n");
        return;
    }

    try {
        kv_offload_mgr_ = std::make_unique<ggml_sycl::kv_offload_manager>(*stream(), config);
        GGML_LOG_INFO("[KV-OFFLOAD] Initialized: threshold=%d, gpu_budget=%.1f MB, block_size=%d\n",
                      config.offload_threshold, config.gpu_kv_budget / (1024.0f * 1024.0f), config.block_size);
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("kv_offload: failed to initialize: %s\n", e.what());
        kv_offload_mgr_.reset();
    }
}

// TBD pool with virtual memory management
// struct ggml_sycl_pool_vmm : public ggml_sycl_pool

/// kernels
typedef void (*ggml_sycl_op_mul_mat_t)(ggml_backend_sycl_context & ctx,
                                       const ggml_tensor *         src0,
                                       const ggml_tensor *         src1,
                                       ggml_tensor *               dst,
                                       const char *                src0_dd_i,
                                       const float *               src1_ddf_i,
                                       const char *                src1_ddq_i,
                                       float *                     dst_dd_i,
                                       const int64_t               row_low,
                                       const int64_t               row_high,
                                       const int64_t               src1_ncols,
                                       const int64_t               src1_padded_row_size,
                                       const queue_ptr &           stream);

static void mul_mat_p021_f16_f32(const void * __restrict__ vx,
                                 const float * __restrict__ y,
                                 float * __restrict__ dst,
                                 const int                ncols_x,
                                 const int                nrows_x,
                                 const int                nchannels_x,
                                 const int                nchannels_y,
                                 const sycl::nd_item<3> & item_ct1) {
    const sycl::half * x = (const sycl::half *) vx;

    const int row_x     = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int channel   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int   ix = row_x * nchannels_x * ncols_x + channel_x * ncols_x + col_x;
        const float xi = sycl::vec<sycl::half, 1>(x[ix]).convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel * nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel * nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32(  // nc == non-contiguous
    const void * __restrict__ vx,
    const float * __restrict__ y,
    float * __restrict__ dst,
    const int                ncols_x,
    const int                nrows_x,
    const int                row_stride_x,
    const int                channel_stride_x,
    const int                channel_stride_y,
    const int                channel_x_divisor,
    const sycl::nd_item<3> & item_ct1) {
    const sycl::half * x = (const sycl::half *) vx;

    const int row_x     = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int channel   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel * nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x * channel_stride_x + row_x * row_stride_x + col_x;
        const int iy = channel * channel_stride_y + row_y;

        const float xi = sycl::vec<sycl::half, 1>(x[ix]).convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}

template <typename T> static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a     = b;
    b     = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void k_argsort_f32_i32(const float *            x,
                                              int *                    dst,
                                              const int                ncols,
                                              int                      ncols_pad,
                                              const int                tasks_per_thread,
                                              const sycl::nd_item<3> & item_ct1,
                                              uint8_t *                dpct_local) {
    // bitonic sort
    int col_index = item_ct1.get_local_id(2);
    int row       = item_ct1.get_group(1);

    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col >= ncols_pad) {
            return;
        }
    }

    const float * x_row   = x + row * ncols;
    auto          dst_row = (int *) dpct_local;

    // initialize indices
    for (int i = 0; i < tasks_per_thread; i++) {
        int col      = col_index * tasks_per_thread + i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < tasks_per_thread; i++) {
                int col = col_index * tasks_per_thread + i;
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC ? x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                                                             x_row[dst_row[col]] < x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (order == GGML_SORT_ORDER_ASC ? x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                                                             x_row[dst_row[col]] > x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

static void diag_mask_inf_f32(const float *            x,
                              float *                  dst,
                              const int                ncols,
                              const int                rows_per_channel,
                              const int                n_past,
                              const sycl::nd_item<3> & item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row * ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i]      = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void scale_f32(const float *            x,
                      float *                  dst,
                      const float              scale,
                      const float              bias,
                      const int                k,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}

template <typename Ti, typename To>
static void pool2d_nchw_kernel(const int                ih,
                               const int                iw,
                               const int                oh,
                               const int                ow,
                               const int                kh,
                               const int                kw,
                               const int                sh,
                               const int                sw,
                               const int                ph,
                               const int                pw,
                               const int                parallel_elements,
                               const Ti *               src,
                               To *                     dst,
                               const enum ggml_op_pool  op,
                               const sycl::nd_item<3> & item_ct1) {
    int idx = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (idx >= parallel_elements) {
        return;
    }

    const int  I_HW    = ih * iw;
    const int  O_HW    = oh * ow;
    const int  nc      = idx / O_HW;
    const int  cur_oh  = idx % O_HW / ow;
    const int  cur_ow  = idx % O_HW % ow;
    const Ti * i_ptr   = src + nc * I_HW;
    To *       o_ptr   = dst + nc * O_HW;
    const int  start_h = cur_oh * sh - ph;
    const int  bh      = sycl::max(0, start_h);
    const int  eh      = sycl::min(ih, start_h + kh);
    const int  start_w = cur_ow * sw - pw;
    const int  bw      = sycl::max(0, start_w);
    const int  ew      = sycl::min(iw, start_w + kw);

    To res = 0;

    switch (op) {
        case GGML_OP_POOL_AVG:
            res = 0;
            break;
        case GGML_OP_POOL_MAX:
            res = -FLT_MAX;
            break;
        default:
            res = (To) sycl::nan(uint32_t(0));
            break;
    }

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
#if DPCT_COMPATIBILITY_TEMP >= 350
            /*
                DPCT1098:106: The '*' expression is used instead of the __ldg
                call. These two expressions do not provide the exact same
                functionality. Check the generated code for potential precision
                and/or performance issues.
                */
            Ti cur = *(i_ptr + i * iw + j);
#else
            Ti cur = i_ptr[i * iw + j];
#endif
            switch (op) {
                case GGML_OP_POOL_AVG:
                    res += (cur / (kh * kw));
                    break;
                case GGML_OP_POOL_MAX:
                    res = sycl::max(res, (To) cur);
                    break;
                default:
                    res = (To) sycl::nan(uint32_t(0));
                    break;
            }
        }
    }
    o_ptr[cur_oh * ow + cur_ow] = res;
}

static void ggml_mul_mat_p021_f16_f32_sycl(const void *  vx,
                                           const float * y,
                                           float *       dst,
                                           const int     ncols_x,
                                           const int     nrows_x,
                                           const int     nchannels_x,
                                           const int     nchannels_y,
                                           queue_ptr     stream) {
    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x, nchannels_y, item_ct1);
                             });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(const void *  vx,
                                             const float * y,
                                             float *       dst,
                                             const int     ncols_x,
                                             const int     nrows_x,
                                             const int     row_stride_x,
                                             const int     nchannels_x,
                                             const int     nchannels_y,
                                             const int     channel_stride_x,
                                             const int     channel_stride_y,
                                             queue_ptr     stream) {
    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x, row_stride_x, channel_stride_x,
                                                        channel_stride_y, nchannels_y / nchannels_x, item_ct1);
                             });
    }
}

static void scale_f32_sycl(const float * x,
                           float *       dst,
                           const float   scale,
                           const float   bias,
                           const int     k,
                           queue_ptr     stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) { scale_f32(x, dst, scale, bias, k, item_ct1); });
}

static void sum_rows_f32_sycl(const float * x, float * dst, const int ncols, const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] { k_sum_rows_f32(x, dst, ncols, item_ct1); });
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_sycl(const float *   x,
                                 int *           dst,
                                 const int       ncols,
                                 const int       nrows,
                                 ggml_sort_order order,
                                 queue_ptr       stream,
                                 int             device) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    int nth            = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size) {
        nth *= 2;
    }
    if (nth > max_block_size) {
        nth = max_block_size;
    }

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t         shared_mem = ncols_pad * sizeof(int);
    GGML_ASSERT(shared_mem <= ggml_sycl_info().devices[device].smpbo);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                    x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                    dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                    x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                    dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

static void argmax_f32_i32_sycl(const float * x, int * dst, const int ncols, const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t         shared_mem = 256 * sizeof(float);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_data(sycl::range<1>(shared_mem / sizeof(float)), cgh);
        sycl::local_accessor<int, 1>   shared_indices(sycl::range<1>(shared_mem / sizeof(float)), cgh);

        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
            const int tid = item_ct1.get_local_id(2);
            const int row = item_ct1.get_global_id(1);

            float max_val = -INFINITY;
            int   max_idx = -1;

            for (int col = tid; col < ncols; col += 256) {
                float val = x[row * ncols + col];
                if (val > max_val) {
                    max_val = val;
                    max_idx = col;
                }
            }

            shared_data[tid]    = max_val;
            shared_indices[tid] = max_idx;
            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (int stride = 256 / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float val1 = shared_data[tid];
                    float val2 = shared_data[tid + stride];
                    if (val2 > val1) {
                        shared_data[tid]    = val2;
                        shared_indices[tid] = shared_indices[tid + stride];
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }

            if (tid == 0) {
                dst[row] = shared_indices[0];
            }
        });
    });
}

// TOP_K kernel for MoE expert gating
// For small K (2-8), uses register-based parallel reduction
#define SYCL_TOPK_BLOCK_SIZE 256
#define SYCL_TOPK_MAX_K      32

static void topk_f32_i32_sycl(const float * x,
                              int *         dst,
                              const int     ncols,
                              const int     nrows,
                              const int     k,
                              queue_ptr     stream) {
    GGML_ASSERT(k <= SYCL_TOPK_MAX_K);

    const sycl::range<3> block_dims(1, 1, SYCL_TOPK_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);

    // Each thread needs to store its local top-K values and indices
    // Then we reduce in shared memory
    // Shared memory layout: [BLOCK_SIZE * MAX_K] floats + [BLOCK_SIZE * MAX_K] ints
    const size_t shared_floats = SYCL_TOPK_BLOCK_SIZE * SYCL_TOPK_MAX_K;
    const size_t shared_ints   = SYCL_TOPK_BLOCK_SIZE * SYCL_TOPK_MAX_K;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_vals(sycl::range<1>(shared_floats), cgh);
        sycl::local_accessor<int, 1>   shared_idxs(sycl::range<1>(shared_ints), cgh);

        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
            const int tid        = item_ct1.get_local_id(2);
            const int row        = item_ct1.get_global_id(1);
            const int block_size = item_ct1.get_local_range(2);

            // Initialize local top-K with -inf
            float local_vals[SYCL_TOPK_MAX_K];
            int   local_idxs[SYCL_TOPK_MAX_K];
            for (int i = 0; i < k; i++) {
                local_vals[i] = -INFINITY;
                local_idxs[i] = -1;
            }

            // Each thread scans its portion and maintains local top-K
            // Using insertion sort since K is small
            for (int col = tid; col < ncols; col += block_size) {
                float val = x[row * ncols + col];

                // Check if this value should be in top-K
                if (val > local_vals[k - 1]) {
                    // Find insertion position
                    int pos = k - 1;
                    while (pos > 0 && val > local_vals[pos - 1]) {
                        pos--;
                    }
                    // Shift elements down
                    for (int j = k - 1; j > pos; j--) {
                        local_vals[j] = local_vals[j - 1];
                        local_idxs[j] = local_idxs[j - 1];
                    }
                    // Insert new value
                    local_vals[pos] = val;
                    local_idxs[pos] = col;
                }
            }

            // Store local top-K to shared memory
            for (int i = 0; i < k; i++) {
                shared_vals[tid * SYCL_TOPK_MAX_K + i] = local_vals[i];
                shared_idxs[tid * SYCL_TOPK_MAX_K + i] = local_idxs[i];
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);

            // Reduce: merge top-K lists from pairs of threads
            // Each iteration halves the number of active threads
            for (int stride = block_size / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    // Merge two sorted top-K lists into one
                    float merged_vals[SYCL_TOPK_MAX_K];
                    int   merged_idxs[SYCL_TOPK_MAX_K];

                    int i = 0, j = 0, m = 0;
                    while (m < k && (i < k || j < k)) {
                        float v1 = (i < k) ? shared_vals[tid * SYCL_TOPK_MAX_K + i] : -INFINITY;
                        float v2 = (j < k) ? shared_vals[(tid + stride) * SYCL_TOPK_MAX_K + j] : -INFINITY;

                        if (v1 >= v2) {
                            merged_vals[m] = v1;
                            merged_idxs[m] = shared_idxs[tid * SYCL_TOPK_MAX_K + i];
                            i++;
                        } else {
                            merged_vals[m] = v2;
                            merged_idxs[m] = shared_idxs[(tid + stride) * SYCL_TOPK_MAX_K + j];
                            j++;
                        }
                        m++;
                    }

                    // Write merged result back to shared memory
                    for (int idx = 0; idx < k; idx++) {
                        shared_vals[tid * SYCL_TOPK_MAX_K + idx] = merged_vals[idx];
                        shared_idxs[tid * SYCL_TOPK_MAX_K + idx] = merged_idxs[idx];
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }

            // Thread 0 writes the final top-K indices to output
            if (tid == 0) {
                for (int i = 0; i < k; i++) {
                    dst[row * k + i] = shared_idxs[i];
                }
            }
        });
    });
}

static void diag_mask_inf_f32_sycl(const float * x,
                                   float *       dst,
                                   const int     ncols_x,
                                   const int     nrows_x,
                                   const int     rows_per_channel,
                                   const int     n_past,
                                   queue_ptr     stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int            block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        diag_mask_inf_f32(x, dst, ncols_x, rows_per_channel, n_past, item_ct1);
    });
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *                     dst,
                                          const struct ggml_tensor * src,
                                          int64_t                    i3,
                                          int64_t                    i2,
                                          int64_t                    i1_low,
                                          int64_t                    i1_high,
                                          queue_ptr                  stream) try {
    dpct::memcpy_direction kind;
    char *                 src_ptr;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind    = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (char *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } else if (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (char *) src->data;
    } else if (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind                          = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int                     id;
        SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else if (ggml_backend_buffer_is_sycl_tp(src->buffer)) {
        // TP (Tensor Parallelism) buffer - similar to split buffer
        // Data is stored in device-specific locations within extra
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind                          = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int                     id;
        SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
        src_ptr = (char *) extra->data_device[id];
        GGML_SYCL_DEBUG("[CPY_TENSOR_2D] TP buffer: device=%d src_ptr=%p dst=%p (tensor=%s) stream_dev=%s\n", id,
                        (void *) src_ptr, dst, src->name,
                        stream->get_device().get_info<sycl::info::device::name>().c_str());
        // Sync before copy to detect any previous errors
        stream->wait();
    } else {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    char * dst_ptr = (char *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    const enum ggml_type type    = src->type;
    const int64_t        ts      = ggml_type_size(type);
    const int64_t        bs      = ggml_blck_size(type);
    int64_t              i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low * nb1 + i2 * nb2 + i3 * nb3;
    if (nb0 == ts && nb1 == ts * ne0 / bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1, kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1, ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1 * nb1);
            void *       rd = (void *) (dst_ptr + i1 * ts * ne0 / bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) {
                return r;
            }
        }
        return 0;
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

inline void ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context & ctx,
                                      const ggml_tensor *         src0,
                                      const ggml_tensor *         src1,
                                      ggml_tensor *               dst,
                                      const char *                src0_dd_i,
                                      const float *               src1_ddf_i,
                                      const char *                src1_ddq_i,
                                      float *                     dst_dd_i,
                                      const int64_t               row_low,
                                      const int64_t               row_high,
                                      const int64_t               src1_ncols,
                                      const int64_t               src1_padded_row_size,
                                      const queue_ptr &           stream) try {
    GGML_ASSERT(src0_dd_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));

    [[maybe_unused]] const int64_t ne0 = dst->ne[0];  // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    [[maybe_unused]] int           ldc = id == ctx.device ? ne0 : row_diff;  // used by MKL only

#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = false;
#endif
    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff * ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const sycl::half *) src0_dd_i : src0_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols * ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half * src1_ptr =
            src1->type == GGML_TYPE_F16 ? (const sycl::half *) src1->data + src1_padded_row_size : src1_as_f16.get();

#if GGML_SYCL_DNNL
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                  src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(), dst_dd_i,
                                  DnnlGemmWrapper::to_dt<float>(), stream);
#elif GGML_SYCL_HAS_ONEAPI_MATH
        {
            ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool(), row_diff * src1_ncols);

            const sycl::half alpha_f16 = 1.0f;
            const sycl::half beta_f16  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(
                dpct::gemm(*stream, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, row_diff,
                           src1_ncols, ne10, &alpha_f16, src0_ptr, dpct::library_data_t::real_half, ne00, src1_ptr,
                           dpct::library_data_t::real_half, ne10, &beta_f16, dst_f16.get(),
                           dpct::library_data_t::real_half, ldc, dpct::library_data_t::real_half)));
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting dst to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16, dst);
            to_fp32_sycl(dst_f16.get(), dst_dd_i, row_diff * src1_ncols, stream);
        }
#else
        static_assert(false, "Either GGML_SYCL_DNNL or GGML_SYCL_HAS_ONEAPI_MATH must be defined");
#endif
    } else {
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            // SoA reorder kernels require full tensor (they compute d_offset from k)
            const bool           src0_full_tensor = (row_diff == src0->ne[1]);
            const to_fp32_sycl_t to_fp32_sycl     = ggml_get_to_fp32_sycl(src0->type, dst, src0_full_tensor);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff * ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff * ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            // src1 is not reordered, full_tensor flag doesn't matter
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst, true);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols * ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols * ne10, stream);
        }
        const float * src0_ddf_i  = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

#if GGML_SYCL_DNNL
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i, DnnlGemmWrapper::to_dt<float>(),
                                  src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(), dst_dd_i,
                                  DnnlGemmWrapper::to_dt<float>(), stream);
#elif GGML_SYCL_HAS_ONEAPI_MATH
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(oneapi::math::blas::column_major::gemm(
                get_onemath_backend(*stream), oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
                row_diff, src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
        }
#else
        static_assert(false, "Either GGML_SYCL_DNNL or GGML_SYCL_HAS_ONEAPI_MATH must be defined");
#endif
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int32_t *   opts = (const int32_t *) dst->op_params;
    enum ggml_op_pool op   = static_cast<ggml_op_pool>(opts[0]);
    const int         k0   = opts[1];
    const int         k1   = opts[2];
    const int         s0   = opts[3];
    const int         s1   = opts[4];
    const int         p0   = opts[5];
    const int         p1   = opts[6];

    const int64_t IH = dst->src[0]->ne[1];
    const int64_t IW = dst->src[0]->ne[0];

    const int64_t N  = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int      parallel_elements = N * OC * OH * OW;
    const int      num_blocks        = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, num_blocks);
    main_stream->parallel_for(sycl::nd_range<3>(block_nums * sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE),
                                                sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) {
                                  pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0, parallel_elements, src0_dd,
                                                     dst_dd, op, item_ct1);
                              });
}

inline void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne = ggml_nelements(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    main_stream->parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> row) { dst_dd[row] /= ncols; });
}

inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *     dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *) dst_dd, ncols, nrows, order, main_stream, ctx.device);
}

inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *     dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *     dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    const int64_t k     = dst->ne[0];  // Output dimension 0 is K

    topk_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, k, main_stream);
}

inline void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne00   = dst->src[0]->ne[0];
    const int64_t ne01   = dst->src[0]->ne[1];
    const int     nrows0 = ggml_nrows(dst->src[0]);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

inline void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias, (float *) dst->op_params + 1, sizeof(float));

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

static void ggml_sycl_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        for (int id_other = 0; id_other < ggml_sycl_info().device_count; ++id_other) {
            if (i == id_other) {
                continue;
            }
            if (i != main_device && id_other != main_device) {
                continue;
            }

            // int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#endif  // NDEBUG

    peer_access_enabled = enable_peer_access;
}

template <template <int> typename quantize_f>
static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx,
                                 const ggml_tensor *         src0,
                                 const ggml_tensor *         src1,
                                 ggml_tensor *               dst,
                                 ggml_sycl_op_mul_mat_t      op) try {
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    // ne10 = K dimension = number of rows in Y matrix (each column has ne10 elements)
    // This is the padded row size, used for stride calculations in quantized buffers
    int64_t src1_padded_row_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char>  src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char>  src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *  src0_dd  = nullptr;
        float * src1_ddf = nullptr;  // float
        char *  src1_ddq = nullptr;  // q8_1
        float * dst_dd   = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int       used_devices = 0;
    queue_ptr main_stream  = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low = ne01 * tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high = ne01 * tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled =
        !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>, no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool dst_on_device  = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            // For TP buffers, use the device-specific data pointer
            if (ggml_backend_buffer_is_sycl_tp(src0->buffer)) {
                ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
                dev[i].src0_dd                = (char *) extra->data_device[i];
                GGML_SYCL_DEBUG("[MUL_MAT] TP buffer src0 device=%d ptr=%p (tensor=%s)\n", i, (void *) dev[i].src0_dd,
                                src0->name);
            } else {
                // Weight streaming: cache CPU weights to GPU on-demand via unified cache
                if (ggml_sycl::unified_cache_enabled() && src0->buffer && !ggml_backend_buffer_is_sycl(src0->buffer)) {
                    // src0 is on CPU (mmap'd) - stream to GPU cache
                    sycl::queue & stream     = *ctx.stream(i, 0);
                    // For mmap'd weights, use same pointer for key and source
                    // (mmap pointers are stable and content doesn't change)
                    void *        cached_ptr = ggml_sycl::get_unified_cache(stream)->ensure_cached(
                        src0->data, src0->data, ggml_nbytes(src0), ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                        extract_layer_number(src0->name), -1);
                    if (cached_ptr) {
                        dev[i].src0_dd = (char *) cached_ptr;
                        GGML_SYCL_DEBUG("[MUL_MAT] weight streaming: cached tensor=%s ptr=%p\n", src0->name,
                                        (void *) dev[i].src0_dd);
                    } else {
                        // Cache full - fall back to staging transfer
                        GGML_LOG_WARN("[MUL_MAT] unified cache full, fallback for tensor=%s\n", src0->name);
                        dev[i].src0_dd = (char *) ggml_sycl_get_data_ptr(src0, i);
                    }
                } else {
                    // Use ggml_sycl_get_data_ptr which handles staging mmap'd data for TP mode
                    dev[i].src0_dd = (char *) ggml_sycl_get_data_ptr(src0, i);
                    GGML_SYCL_DEBUG("[MUL_MAT] non-TP buffer src0 device=%d ptr=%p (tensor=%s)\n", i,
                                    (void *) dev[i].src0_dd, src0->name);
                }
            }
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
            GGML_SYCL_DEBUG("[MUL_MAT] NON-CONTIGUOUS src0 device=%d ptr=%p (tensor=%s, is_tp=%d)\n", i,
                            (void *) dev[i].src0_dd, src0->name, ggml_backend_buffer_is_sycl_tp(src0->buffer) ? 1 : 0);
        }

        if (src1_on_device && src1_is_contiguous) {
            // For TP compute buffers, use device-specific pointer
            dev[i].src1_ddf = (float *) ggml_sycl_get_data_ptr(src1, i);
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        // Debug: Sample src1_ddf content at MUL_MAT entry (before quantization)
        static bool mulmat_entry_debug_checked = false;
        static bool do_mulmat_entry_debug      = false;
        if (!mulmat_entry_debug_checked) {
            do_mulmat_entry_debug      = (getenv("GGML_SYCL_RMS_DEBUG") != nullptr);
            mulmat_entry_debug_checked = true;
        }
        if (do_mulmat_entry_debug && src1_on_device && src1_is_contiguous) {
            main_stream->wait();
            float sample[4];
            main_stream->memcpy(sample, dev[i].src1_ddf, 4 * sizeof(float)).wait();
            bool is_zeros = (sample[0] == 0.0f && sample[1] == 0.0f && sample[2] == 0.0f && sample[3] == 0.0f);
            fprintf(
                stderr,
                "[MUL_MAT_ENTRY] src1=%s src0=%s src1_ddf=%p first4=[%.4f, %.4f, %.4f, %.4f] is_zeros=%d is_soa=%d\n",
                src1->name, src0->name, (void *) dev[i].src1_ddf, sample[0], sample[1], sample[2], sample[3],
                is_zeros ? 1 : 0, src0_extra ? src0_extra->optimized_feature.is_soa() : 0);
        }

        if constexpr (quantize_enabled) {
            const size_t required_size = nrows1 * src1_padded_row_size * q8_1_ts / q8_1_bs;

            // Buffer aliasing debug - check if src1_ddf overlaps with any pre-allocated buffers
            static bool buffer_alias_debug_checked = false;
            static bool do_buffer_alias_debug      = false;
            if (!buffer_alias_debug_checked) {
                do_buffer_alias_debug      = (getenv("GGML_SYCL_BUFFER_ALIAS_DEBUG") != nullptr);
                buffer_alias_debug_checked = true;
            }

            if (do_buffer_alias_debug) {
                const uintptr_t src1_ddf_addr = reinterpret_cast<uintptr_t>(dev[i].src1_ddf);
                const size_t    src1_ddf_size = ggml_nelements(src1) * sizeof(float);
                const uintptr_t src1_ddf_end  = src1_ddf_addr + src1_ddf_size;

                // Print src1 tensor info to understand data source
                fprintf(stderr, "[BUFFER_ALIAS] src1='%s' src0='%s' is_view=%d src1_src=%s\n", src1->name, src0->name,
                        src1->view_src ? 1 : 0, src1->src[0] ? src1->src[0]->name : "(null)");

                fprintf(stderr, "[BUFFER_ALIAS] ne10=%lld src1_ddf=%p size=%zu end=%p is_soa=%d\n", (long long) ne10,
                        (void *) src1_ddf_addr, src1_ddf_size, (void *) src1_ddf_end,
                        src0_extra ? src0_extra->optimized_feature.is_soa() : 0);

                // Check overlap with pre-allocated mmvq_soa_buffers
                if (ctx.mmvq_soa_buffers.initialized) {
                    for (size_t buf_idx = 0; buf_idx < ctx.mmvq_soa_buffers.src1_ddq_buffers.size(); buf_idx++) {
                        const uintptr_t prealloc_addr =
                            reinterpret_cast<uintptr_t>(ctx.mmvq_soa_buffers.src1_ddq_buffers[buf_idx]);
                        const size_t    prealloc_size = ctx.mmvq_soa_buffers.src1_ddq_sizes[buf_idx];
                        const uintptr_t prealloc_end  = prealloc_addr + prealloc_size;

                        // Check for overlap: [A_start, A_end) overlaps [B_start, B_end) if A_start < B_end && B_start < A_end
                        bool overlaps = (src1_ddf_addr < prealloc_end) && (prealloc_addr < src1_ddf_end);
                        if (overlaps) {
                            fprintf(stderr,
                                    "[BUFFER_ALIAS] *** OVERLAP DETECTED *** src1_ddf [%p-%p] overlaps prealloc[%zu] "
                                    "[%p-%p]\n",
                                    (void *) src1_ddf_addr, (void *) src1_ddf_end, buf_idx, (void *) prealloc_addr,
                                    (void *) prealloc_end);
                        }
                    }
                }
            }

            // Check if we should use pre-allocated buffer for reordered MMVQ (both SOA and COALESCED).
            // CRITICAL: Use pre-allocated buffers for BOTH warmup AND recording.
            // During graph recording, kernel lambdas capture pointers by value.
            // If warmup uses pool (different pointer) and recording uses pre-alloc,
            // the graph replays with stale pointers -> garbage output.
            // NOTE: Both SOA and COALESCED modes use the same SoA-format quantization,
            // so we check is_reordered() (any reorder) not is_soa() (only SOA mode).
            const bool use_reordered_prealloc =
                src0_extra && src0_extra->optimized_feature.is_reordered() && ctx.mmvq_soa_buffers.initialized;

            if (use_reordered_prealloc) {
                // Use pre-allocated buffer for consistent pointer across warmup/recording/replay
                void * preallocated = ctx.mmvq_soa_buffers.get_next_buffer(required_size);
                if (preallocated) {
                    dev[i].src1_ddq = static_cast<char *>(preallocated);
                    GGML_SYCL_DEBUG("[MMVQ-SOA] Using pre-allocated buffer %d: %p (size=%zu, recording=%d)\n",
                                    ctx.mmvq_soa_buffers.current_buffer_idx - 1, preallocated, required_size,
                                    g_ggml_sycl_graph_recording ? 1 : 0);
                } else {
                    // Fallback to pool allocation if pre-allocated buffer not available
                    dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), required_size);
                    GGML_SYCL_DEBUG("[MMVQ-SOA] Pre-allocated buffer exhausted, using pool\n");
                }
            } else {
                // Standard pool allocation
                dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), required_size);
            }

            // Debug: check if src1_ddq overlaps with src1_ddf
            if (do_buffer_alias_debug) {
                const uintptr_t src1_ddq_addr = reinterpret_cast<uintptr_t>(dev[i].src1_ddq);
                const uintptr_t src1_ddq_end  = src1_ddq_addr + required_size;
                const uintptr_t src1_ddf_addr = reinterpret_cast<uintptr_t>(dev[i].src1_ddf);
                const size_t    src1_ddf_size = ggml_nelements(src1) * sizeof(float);
                const uintptr_t src1_ddf_end  = src1_ddf_addr + src1_ddf_size;

                bool ddq_ddf_overlap = (src1_ddq_addr < src1_ddf_end) && (src1_ddf_addr < src1_ddq_end);
                if (ddq_ddf_overlap) {
                    fprintf(stderr, "[BUFFER_ALIAS] *** DDQ/DDF OVERLAP *** src1_ddf [%p-%p] vs src1_ddq [%p-%p]\n",
                            (void *) src1_ddf_addr, (void *) src1_ddf_end, (void *) src1_ddq_addr,
                            (void *) src1_ddq_end);
                }

                fprintf(stderr, "[BUFFER_ALIAS] src1_ddq=%p size=%zu prealloc=%d\n", (void *) src1_ddq_addr,
                        required_size, use_reordered_prealloc ? 1 : 0);

                // Verify src1_ddf content is not all zeros (sync first to ensure data is ready)
                stream->wait();
                std::vector<float> first_vals(4);
                stream->memcpy(first_vals.data(), dev[i].src1_ddf, 4 * sizeof(float)).wait();
                float abs_sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    abs_sum += std::fabs(first_vals[k]);
                }
                bool is_zeros = (abs_sum == 0.0f);
                fprintf(stderr, "[BUFFER_ALIAS] src1_ddf content: first4=[%.4f, %.4f, %.4f, %.4f] is_zeros=%d\n",
                        first_vals[0], first_vals[1], first_vals[2], first_vals[3], is_zeros ? 1 : 0);
                if (is_zeros && ne10 == 4096) {
                    fprintf(stderr, "[BUFFER_ALIAS] *** WARNING: ne10=4096 has zeros - possible aliasing ***\n");
                }
            }

            // Zero padding blocks to prevent garbage in MMQ when ne10 < src1_padded_row_size
            // Bug: quantize_row_q8_1_sycl only fills ne10/QK8_1 blocks per row, but MMQ reads
            // src1_padded_row_size/QK8_1 blocks per row. The padding blocks must be zeroed.
            if (ne10 != src1_padded_row_size) {
                stream->memset(dev[i].src1_ddq, 0, required_size);
            }

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                // Debug: print pointers and device info before quantize
                GGML_SYCL_DEBUG("[QUANTIZE DEBUG] device=%d src1_ddf=%p src1_ddq=%p stream_device=%s\n", i,
                                (void *) dev[i].src1_ddf, (void *) dev[i].src1_ddq,
                                stream->get_device().get_info<sycl::info::device::name>().c_str());

                try {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1,
                                                       src1_padded_row_size, stream);
                } catch (const sycl::exception & exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        if (dst_on_device) {
            // For TP compute buffers, use device-specific pointer
            dev[i].dst_dd = (float *) ggml_sycl_get_data_ptr(dst, i);
            GGML_SYCL_DEBUG("[MUL_MAT] dst device=%d ptr=%p (tensor=%s)\n", i, (void *) dev[i].dst_dd, dst->name);
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low) * ne1 : ggml_nelements(dst);
            dev[i].dst_dd             = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(*src0_extra->events[ctx.device][0] = ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is         = split ? (src1_col_0 / src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool    src1_on_device = i == ctx.device;
            const bool    dst_on_device  = i == ctx.device;
            const int64_t row_diff       = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier({ *src0_extra->events[ctx.device][0] })));
            }

            for (int64_t i0 = 0; i0 < ne13 * ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0 * ne11 + src1_col_0) * src1_padded_row_size * q8_1_ts / q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char *  src0_dd_i  = dev[i].src0_dd + (i0 / i02_divisor) * (ne01 * ne00 * src0_ts) / src0_bs;
                float * src1_ddf_i = dev[i].src1_ddf + (i0 * ne11 + src1_col_0) * ne10;
                char *  src1_ddq_i = dev[i].src1_ddq + src1_ddq_i_offset;
                float * dst_dd_i   = dev[i].dst_dd + (i0 * ne1 + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low;  // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if constexpr (quantize_enabled) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_row_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } else {
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[ctx.device];
                            src1_ddf_i_source += (i0 * ne11 + src1_col_0) * ne10;

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(*stream, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * sizeof(float))));
                        }
                    }
                } else {
                    if (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } else {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    if constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        try {
                            quantize_row_q8_1_sycl<quantize_f>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                               src1_padded_row_size, stream);
                        } catch (const sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02 / i02_divisor, dev[i].row_low,
                                                       dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_row_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(
                    CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                                       dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_row_size, stream)));

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02 * nb2 + i03 * nb3);
                        GGML_ASSERT(dst->nb[1] == ne0 * sizeof(float));
                        dhf_dst_i += src1_col_0 * ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i, row_diff * sizeof(float),
                            row_diff * sizeof(float), src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02 * nb2 + i03 * nb3);
                        GGML_ASSERT(dst->nb[1] == ne0 * sizeof(float));
                        dhf_dst_i += src1_col_0 * ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i, src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(*src0_extra->events[i][is] = stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max         = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(ctx.stream()->ext_oneapi_submit_barrier({ *src0_extra->events[i][is] })));
            }
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

static void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

static void ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

static void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

static void ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

static void ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

static void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

// Check if a weight tensor is TP-sharded (has shards on multiple devices)
static bool is_tp_sharded_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->extra == nullptr) {
        return false;
    }
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(tensor->extra);
    return extra->tp_sharded && extra->tp_world_size > 1;
}

// Get the TP layer type from tensor extra
static tp_layer_type get_tp_layer_type(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->extra == nullptr) {
        return tp_layer_type::TP_NONE;
    }
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(tensor->extra);
    return extra->tp_type;
}

// Simple element-wise add kernel for ALL_REDUCE_SUM
// IMPORTANT: Use submit() to isolate kernel lambda scope from enclosing context.
static void ggml_sycl_add_f32(float * dst, const float * src, size_t n, queue_ptr stream) {
    stream
        ->submit([&](sycl::handler & cgh) {
            float *       dst_local = dst;
            const float * src_local = src;
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) { dst_local[idx] += src_local[idx]; });
        })
        .wait();
}

// TP mul_mat pre-check: Called BEFORE regular mul_mat
// For column-parallel: just returns false (regular mul_mat handles it)
// For row-parallel: returns false (regular mul_mat runs), then post-processing does ALL_REDUCE
static bool ggml_sycl_mul_mat_tp_pre(const ggml_tensor * src0) {
    if (!is_tp_sharded_tensor(src0)) {
        return false;
    }

    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    GGML_UNUSED(extra);
    return false;  // Always continue with regular mul_mat
}

// Global storage for column-parallel outputs on device 1
// Key: dst->data pointer (stable during graph execution)
// Value: device 1 buffer pointer
static std::unordered_map<void *, void *> g_tp_column_parallel_outputs;
static std::mutex                         g_tp_column_parallel_mutex;

// FFN input storage: stores the input to FFN column-parallel layers
// This is needed so that row-parallel (ffn_down) can compute device 1's full FFN path
// (struct ffn_input_storage defined in common.hpp)
std::unordered_map<int, ffn_input_storage> g_tp_ffn_inputs;  // Key: layer number
std::mutex                                 g_tp_ffn_input_mutex;

// Extract layer number from tensor name (e.g., "blk.0.ffn_gate" -> 0)
int ggml_sycl_tp_extract_layer_number(const char * name) {
    if (!name) {
        return -1;
    }
    const char * blk = strstr(name, "blk.");
    if (!blk) {
        return -1;
    }
    return atoi(blk + 4);
}

// Local alias for convenience
static int extract_layer_number(const char * name) {
    return ggml_sycl_tp_extract_layer_number(name);
}

// FFN weight storage: stores references to FFN weight tensors for device 1 computation
// (struct ffn_weight_refs defined in common.hpp)
std::unordered_map<int, ffn_weight_refs> g_tp_ffn_weights;  // Key: layer number
std::mutex                               g_tp_ffn_weight_mutex;

// Attention input storage: stores the input to attention column-parallel layers
// This is needed so that row-parallel (attn_output) can compute device 1's full attention path
// (struct attn_input_storage defined in common.hpp)
std::unordered_map<int, attn_input_storage> g_tp_attn_inputs;  // Key: layer number
std::mutex                                  g_tp_attn_input_mutex;

// Attention weight storage: stores references to attention weight tensors for device 1 computation
// (struct attn_weight_refs defined in common.hpp)
std::unordered_map<int, attn_weight_refs> g_tp_attn_weights;  // Key: layer number
std::mutex                                g_tp_attn_weight_mutex;

// Async FFN jobs: tracks in-flight FFN computations on device 1
// (struct tp_async_ffn_job defined in common.hpp)
std::unordered_map<int, tp_async_ffn_job> g_tp_async_ffn_jobs;  // Key: layer number
std::mutex                                g_tp_async_ffn_mutex;

// Async attention jobs: tracks in-flight attention computations on device 1
// (struct tp_async_attn_job defined in common.hpp)
std::unordered_map<int, tp_async_attn_job> g_tp_async_attn_jobs;  // Key: layer number
std::mutex                                 g_tp_async_attn_mutex;

// =============================================================================
// Thread-based pipelining for device 1 FFN
// =============================================================================
tp_device1_worker g_tp_device1_worker;
// Thread-based FFN pipelining - DISABLED due to SYCL/Level Zero MMVQ hang
// The MMVQ kernel hangs when called from a separate worker thread, even with
// a dedicated SYCL queue. This appears to be due to internal synchronization
// in the Level Zero driver or SYCL runtime that doesn't support concurrent
// kernel launches from multiple host threads on the same device.
// Note: Out-of-order queues were investigated and produce non-deterministic
// output on Intel Arc GPUs even with maximum synchronization, likely a driver bug.
int               g_ggml_sycl_tp_threaded_ffn = 0;  // DISABLED - causes hangs at MMVQ kernel

// Forward declaration of worker thread function
static void tp_device1_worker_thread_func();

// Worker thread function: runs FFN computations on device 1
static void tp_device1_worker_thread_func() {
    auto & w = g_tp_device1_worker;

    const int device = g_sycl_tp_config.devices[1];

    // Create a DEDICATED in-order queue for the worker thread
    // Using the shared TP queue causes hangs due to SYCL queue contention
    ggml_sycl_set_device(device);
    sycl::device         dev          = dpct::get_device(device);
    static sycl::queue * worker_queue = nullptr;
    if (!worker_queue) {
        worker_queue = new sycl::queue(dev, sycl::property_list{ sycl::property::queue::in_order() });
    }
    queue_ptr stream = worker_queue;

    if (!stream) {
        fprintf(stderr, "SYCL TP WORKER: Failed to create worker queue for device %d!\n", device);
        return;
    }

    // Create a minimal static context for MMVQ calls
    // MMVQ doesn't actually use the context, but needs it by reference
    static ggml_backend_sycl_context * worker_ctx = nullptr;
    if (!worker_ctx) {
        worker_ctx       = new ggml_backend_sycl_context(device);
        worker_ctx->name = "tp_worker";
    }
    ggml_backend_sycl_context & ctx = *worker_ctx;

    fprintf(stderr, "SYCL TP WORKER: Thread started on device %d, dedicated queue=%p\n", device, (void *) stream);

    while (true) {
        tp_ffn_work_item work;

        // Wait for work or shutdown
        {
            std::unique_lock<std::mutex> lock(w.work_mutex);
            w.work_cv.wait(
                lock, [] { return g_tp_device1_worker.shutdown.load() || !g_tp_device1_worker.work_queue.empty(); });

            if (w.shutdown.load() && w.work_queue.empty()) {
                fprintf(stderr, "SYCL TP WORKER: Shutdown requested, exiting\n");
                break;
            }

            work = std::move(w.work_queue.front());
            w.work_queue.pop();
        }

        // Process FFN work
        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Processing layer %d, K=%lld, batch=%lld\n", work.layer,
                    (long long) work.K_full, (long long) work.batch);
        }

        // Validate weight pointers
        if (!work.weights.gate || !work.weights.up || !work.weights.down) {
            fprintf(stderr, "SYCL TP WORKER: Null weight tensor for layer %d\n", work.layer);
            continue;
        }

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Weight pointers: gate=%p, up=%p, down=%p\n", (void *) work.weights.gate,
                    (void *) work.weights.up, (void *) work.weights.down);
            fprintf(stderr, "SYCL TP WORKER: gate->extra=%p, up->extra=%p, down->extra=%p\n", work.weights.gate->extra,
                    work.weights.up->extra, work.weights.down->extra);
        }

        // Get weight shards for device 1
        auto * gate_extra = static_cast<ggml_tensor_extra_gpu *>(work.weights.gate->extra);
        auto * up_extra   = static_cast<ggml_tensor_extra_gpu *>(work.weights.up->extra);
        auto * down_extra = static_cast<ggml_tensor_extra_gpu *>(work.weights.down->extra);

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Accessing data_device[%d]\n", device);
            fflush(stderr);
        }

        void * gate_weight_1 = gate_extra ? gate_extra->data_device[device] : nullptr;
        void * up_weight_1   = up_extra ? up_extra->data_device[device] : nullptr;
        void * down_weight_1 = down_extra ? down_extra->data_device[device] : nullptr;

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Weight device pointers: gate=%p, up=%p, down=%p\n", gate_weight_1,
                    up_weight_1, down_weight_1);
            fflush(stderr);
        }

        if (!gate_weight_1 || !up_weight_1 || !down_weight_1) {
            fprintf(stderr, "SYCL TP WORKER: Missing weight shards for layer %d\n", work.layer);
            continue;
        }

        // Get dimensions
        const int64_t N_hidden_shard = gate_extra->tp_local_ne[1];
        const int64_t N_out          = down_extra->tp_local_ne[1];

        // Allocate buffers on device 1
        const size_t  q8_1_ts               = sizeof(block_q8_1);
        const size_t  q8_1_bs               = QK8_1;
        const int64_t K_full_padded         = GGML_PAD(work.K_full, MATRIX_ROW_PADDING);
        const int64_t N_hidden_shard_padded = GGML_PAD(N_hidden_shard, MATRIX_ROW_PADDING);

        const size_t input_q8_size  = work.batch * K_full_padded * q8_1_ts / q8_1_bs;
        const size_t hidden_size    = N_hidden_shard * work.batch * sizeof(float);
        const size_t hidden_q8_size = work.batch * N_hidden_shard_padded * q8_1_ts / q8_1_bs;
        const size_t output_size    = N_out * work.batch * sizeof(float);

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Allocating buffers for layer %d (input_q8=%zu, hidden=%zu, output=%zu)\n",
                    work.layer, input_q8_size, hidden_size, output_size);
            fflush(stderr);
        }

        char *  input_q8_dev  = (char *) sycl::malloc_device(input_q8_size, *stream);
        float * gate_out      = (float *) sycl::malloc_device(hidden_size, *stream);
        float * up_out        = (float *) sycl::malloc_device(hidden_size, *stream);
        float * hidden_out    = (float *) sycl::malloc_device(hidden_size, *stream);
        char *  hidden_q8_dev = (char *) sycl::malloc_device(hidden_q8_size, *stream);
        float * partial_out   = (float *) sycl::malloc_device(output_size, *stream);
        float * result_buf    = (float *) ggml_sycl_host_malloc(output_size);

        if (g_ggml_sycl_tp_debug) {
            fprintf(
                stderr,
                "SYCL TP WORKER: Buffers allocated: input_q8=%p, gate=%p, up=%p, hidden=%p, partial=%p, result=%p\n",
                (void *) input_q8_dev, (void *) gate_out, (void *) up_out, (void *) hidden_out, (void *) partial_out,
                (void *) result_buf);
            fflush(stderr);
        }

        if (!input_q8_dev || !gate_out || !up_out || !hidden_out || !hidden_q8_dev || !partial_out || !result_buf) {
            fprintf(stderr, "SYCL TP WORKER: Buffer allocation failed for layer %d\n", work.layer);
            if (input_q8_dev) {
                sycl::free(input_q8_dev, *stream);
            }
            if (gate_out) {
                sycl::free(gate_out, *stream);
            }
            if (up_out) {
                sycl::free(up_out, *stream);
            }
            if (hidden_out) {
                sycl::free(hidden_out, *stream);
            }
            if (hidden_q8_dev) {
                sycl::free(hidden_q8_dev, *stream);
            }
            if (partial_out) {
                sycl::free(partial_out, *stream);
            }
            if (result_buf) {
                ggml_sycl_host_free(result_buf);
            }
            continue;
        }

        // Create fake dst tensors for MMVQ calls
        ggml_tensor fake_dst_hidden;
        memset(&fake_dst_hidden, 0, sizeof(fake_dst_hidden));
        fake_dst_hidden.ne[0] = N_hidden_shard;
        fake_dst_hidden.ne[1] = work.batch;

        ggml_tensor fake_dst_out;
        memset(&fake_dst_out, 0, sizeof(fake_dst_out));
        fake_dst_out.ne[0] = N_out;
        fake_dst_out.ne[1] = work.batch;

        // Step 1: Quantize input to Q8_1
        // Use SoA quantizer if weights are in SoA layout (reordered kernels expect SoA Y)
        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Step 1 - Quantizing input\n");
            fflush(stderr);
        }
        ggml_tensor_extra_gpu * gate_extra_worker = static_cast<ggml_tensor_extra_gpu *>(work.weights.gate->extra);
        const bool use_soa_input = gate_extra_worker && gate_extra_worker->optimized_feature.is_reordered();
        if (use_soa_input) {
            quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(work.input_dev1, input_q8_dev, work.K_full,
                                                                  work.batch, K_full_padded, stream);
        } else {
            quantize_row_q8_1_sycl<quantize_q8_1>(work.input_dev1, input_q8_dev, work.K_full, work.batch, K_full_padded,
                                                  stream);
        }
        stream->wait();  // Add wait to catch any async errors

        // Step 2-3: Gate and Up matmuls
        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Step 2 - Gate/Up memset\n");
            fflush(stderr);
        }
        stream->memset(gate_out, 0, hidden_size);
        stream->memset(up_out, 0, hidden_size);
        stream->wait();

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Step 3 - Gate matmul\n");
            fflush(stderr);
        }
        ggml_sycl_op_mul_mat_vec_q(ctx, work.weights.gate, nullptr, &fake_dst_hidden, (const char *) gate_weight_1,
                                   nullptr, input_q8_dev, gate_out, 0, N_hidden_shard, work.batch, K_full_padded,
                                   stream);

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Step 4 - Up matmul\n");
            fflush(stderr);
        }
        ggml_sycl_op_mul_mat_vec_q(ctx, work.weights.up, nullptr, &fake_dst_hidden, (const char *) up_weight_1, nullptr,
                                   input_q8_dev, up_out, 0, N_hidden_shard, work.batch, K_full_padded, stream);
        stream->wait();

        // Step 4: SiLU activation and multiply
        const int64_t n_elements = N_hidden_shard * work.batch;
        const int     block_size = 256;
        const int     num_blocks = (n_elements + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> item) {
            const int i = item.get_global_id(0);
            if (i < n_elements) {
                float g       = gate_out[i];
                float u       = up_out[i];
                float silu_g  = g / (1.0f + sycl::native::exp(-g));
                hidden_out[i] = silu_g * u;
            }
        });

        // Step 5: Quantize hidden for down matmul
        // Use SoA quantizer if down weight is in SoA layout
        ggml_tensor_extra_gpu * down_extra_worker = static_cast<ggml_tensor_extra_gpu *>(work.weights.down->extra);
        const bool use_soa_hidden = down_extra_worker && down_extra_worker->optimized_feature.is_reordered();
        if (use_soa_hidden) {
            quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(hidden_out, hidden_q8_dev, N_hidden_shard, work.batch,
                                                                  N_hidden_shard_padded, stream);
        } else {
            quantize_row_q8_1_sycl<quantize_q8_1>(hidden_out, hidden_q8_dev, N_hidden_shard, work.batch,
                                                  N_hidden_shard_padded, stream);
        }

        // Step 6: Down matmul
        stream->memset(partial_out, 0, output_size);
        stream->wait();
        ggml_sycl_op_mul_mat_vec_q(ctx, work.weights.down, nullptr, &fake_dst_out, (const char *) down_weight_1,
                                   nullptr, hidden_q8_dev, partial_out, 0, N_out, work.batch, N_hidden_shard_padded,
                                   stream);

        // Step 7: Copy result to host-pinned buffer
        stream->memcpy(result_buf, partial_out, output_size).wait();

        // Cleanup device buffers
        sycl::free(input_q8_dev, *stream);
        sycl::free(gate_out, *stream);
        sycl::free(up_out, *stream);
        sycl::free(hidden_out, *stream);
        sycl::free(hidden_q8_dev, *stream);
        sycl::free(partial_out, *stream);

        // Store result
        {
            std::lock_guard<std::mutex> lock(w.result_mutex);
            w.results[work.layer] = { work.layer, result_buf, N_out, work.batch, output_size, true };
        }
        w.result_cv.notify_all();

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Completed layer %d, result[0..3]=[%f,%f,%f,%f]\n", work.layer,
                    result_buf[0], result_buf[1], result_buf[2], result_buf[3]);
        }
    }
}

// Initialize worker thread
void ggml_sycl_tp_worker_init(void * /* ctx */) {
    auto & w = g_tp_device1_worker;

    if (w.initialized.load()) {
        return;  // Already initialized
    }

    if (!g_ggml_sycl_tp_threaded_ffn) {
        return;  // Thread-based pipelining disabled
    }

    // Note: We don't actually need the context anymore since we use
    // ggml_sycl_get_tp_queue() which is thread-safe
    w.ctx = nullptr;
    w.shutdown.store(false);

    // Set initialized BEFORE starting thread to avoid race
    w.initialized.store(true);
    w.worker_thread = std::thread(tp_device1_worker_thread_func);

    fprintf(stderr, "SYCL TP: Worker thread initialized\n");
}

// Shutdown worker thread
void ggml_sycl_tp_worker_shutdown() {
    auto & w = g_tp_device1_worker;

    if (!w.initialized.load()) {
        return;
    }

    // Signal shutdown
    {
        std::lock_guard<std::mutex> lock(w.work_mutex);
        w.shutdown.store(true);
    }
    w.work_cv.notify_all();

    // Wait for thread to finish
    if (w.worker_thread.joinable()) {
        w.worker_thread.join();
    }

    // Cleanup any remaining results
    {
        std::lock_guard<std::mutex> lock(w.result_mutex);
        for (auto & pair : w.results) {
            if (pair.second.result_buf) {
                ggml_sycl_host_free(pair.second.result_buf);
            }
        }
        w.results.clear();
    }

    w.initialized.store(false);
    fprintf(stderr, "SYCL TP: Worker thread shutdown complete\n");
}

// Submit FFN work to queue
void ggml_sycl_tp_submit_ffn_work(const tp_ffn_work_item & work) {
    auto & w = g_tp_device1_worker;

    if (!w.initialized.load()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(w.work_mutex);
        w.work_queue.push(work);
    }
    w.work_cv.notify_one();
}

// Get FFN result for a layer (optionally wait for it)
tp_ffn_result * ggml_sycl_tp_get_ffn_result(int layer, bool wait) {
    auto & w = g_tp_device1_worker;

    if (!w.initialized.load()) {
        return nullptr;
    }

    std::unique_lock<std::mutex> lock(w.result_mutex);

    if (wait) {
        // Wait until result is available or we detect it won't come
        w.result_cv.wait_for(lock, std::chrono::milliseconds(5000), [layer] {
            return g_tp_device1_worker.results.find(layer) != g_tp_device1_worker.results.end() &&
                   g_tp_device1_worker.results[layer].valid;
        });
    }

    auto it = w.results.find(layer);
    if (it != w.results.end() && it->second.valid) {
        return &it->second;
    }
    return nullptr;
}

// Release FFN result memory
void ggml_sycl_tp_release_ffn_result(int layer) {
    auto & w = g_tp_device1_worker;

    std::lock_guard<std::mutex> lock(w.result_mutex);
    auto                        it = w.results.find(layer);
    if (it != w.results.end()) {
        if (it->second.result_buf) {
            ggml_sycl_host_free(it->second.result_buf);
        }
        w.results.erase(it);
    }
}

// KV cache for device 1's attention heads (needed for token generation)
// Each layer stores K and V values for all processed positions
struct dev1_kv_cache_entry {
    float *   k_cache;      // [max_seq_len, n_heads_kv * head_dim]
    float *   v_cache;      // [max_seq_len, n_heads_kv * head_dim]
    int64_t   seq_pos;      // Current sequence position (next position to write)
    int64_t   max_seq_len;  // Allocated cache size
    int64_t   n_heads_kv;   // Number of KV heads per device
    int64_t   head_dim;     // Dimension per head
    queue_ptr stream;       // Device 1 stream for cache operations
};

std::unordered_map<int, dev1_kv_cache_entry> g_tp_dev1_kv_cache;  // Key: layer number
std::mutex                                   g_tp_dev1_kv_cache_mutex;

// Initialize or resize the KV cache for a given layer on device 1
// Called during prompt processing to set up cache with proper dimensions
static void init_dev1_kv_cache(int layer, int64_t max_seq_len, int64_t n_heads_kv, int64_t head_dim, queue_ptr stream) {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);

    auto it = g_tp_dev1_kv_cache.find(layer);
    if (it != g_tp_dev1_kv_cache.end()) {
        // Cache exists - check if reallocation needed
        if (it->second.max_seq_len >= max_seq_len) {
            return;  // Already big enough
        }
        // Free old cache
        if (it->second.k_cache) {
            sycl::free(it->second.k_cache, *stream);
        }
        if (it->second.v_cache) {
            sycl::free(it->second.v_cache, *stream);
        }
    }

    // Allocate new cache (oversized for typical generations)
    int64_t cache_max_seq = std::max(max_seq_len, (int64_t) 4096);
    size_t  cache_size    = cache_max_seq * n_heads_kv * head_dim * sizeof(float);

    dev1_kv_cache_entry entry;
    entry.k_cache     = (float *) sycl::malloc_device(cache_size, *stream);
    entry.v_cache     = (float *) sycl::malloc_device(cache_size, *stream);
    entry.seq_pos     = 0;
    entry.max_seq_len = cache_max_seq;
    entry.n_heads_kv  = n_heads_kv;
    entry.head_dim    = head_dim;
    entry.stream      = stream;

    if (!entry.k_cache || !entry.v_cache) {
        fprintf(stderr, "SYCL TP: WARNING - Failed to allocate KV cache for layer %d (size=%zu)\n", layer, cache_size);
        if (entry.k_cache) {
            sycl::free(entry.k_cache, *stream);
        }
        if (entry.v_cache) {
            sycl::free(entry.v_cache, *stream);
        }
        return;
    }

    g_tp_dev1_kv_cache[layer] = entry;
    static int log_count      = 0;
    if (g_ggml_sycl_tp_debug && log_count++ < 3) {
        fprintf(stderr, "SYCL TP: Allocated dev1 KV cache for layer %d: max_seq=%lld, n_kv_heads=%lld, head_dim=%lld\n",
                layer, (long long) cache_max_seq, (long long) n_heads_kv, (long long) head_dim);
    }
}

// Append new K and V values to the cache for a layer
// Called after K/V projection in the attention path
// k_new/v_new: [batch, n_heads_kv * head_dim] where batch is the new tokens to add
static void append_to_dev1_kv_cache(int           layer,
                                    const float * k_new,
                                    const float * v_new,
                                    int64_t       batch,
                                    queue_ptr     stream) {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);

    auto it = g_tp_dev1_kv_cache.find(layer);
    if (it == g_tp_dev1_kv_cache.end()) {
        static int warn = 0;
        if (warn++ < 5) {
            fprintf(stderr, "SYCL TP: WARNING - KV cache not initialized for layer %d\n", layer);
        }
        return;
    }

    auto & entry = it->second;

    // Check if we have room
    if (entry.seq_pos + batch > entry.max_seq_len) {
        fprintf(stderr, "SYCL TP: ERROR - KV cache overflow layer %d: pos=%lld + batch=%lld > max=%lld\n", layer,
                (long long) entry.seq_pos, (long long) batch, (long long) entry.max_seq_len);
        return;
    }

    // Copy new K and V to cache
    // Cache layout: [max_seq_len, n_heads_kv * head_dim]
    size_t kv_stride = entry.n_heads_kv * entry.head_dim;
    size_t copy_size = batch * kv_stride * sizeof(float);
    size_t offset    = entry.seq_pos * kv_stride * sizeof(float);

    stream->memcpy((char *) entry.k_cache + offset, k_new, copy_size);
    stream->memcpy((char *) entry.v_cache + offset, v_new, copy_size);
    stream->wait();

    entry.seq_pos += batch;

    static int dbg_count = 0;
    if (g_ggml_sycl_tp_debug && dbg_count++ < 3) {  // Reduced debug output
        fprintf(stderr, "TP DEBUG: Appended %lld tokens to layer %d KV cache, now at pos %lld\n", (long long) batch,
                layer, (long long) entry.seq_pos);
    }
}

// Get the current sequence length in the cache for a layer
static int64_t get_dev1_kv_cache_seq_len(int layer) {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);
    auto                        it = g_tp_dev1_kv_cache.find(layer);
    if (it == g_tp_dev1_kv_cache.end()) {
        return 0;
    }
    return it->second.seq_pos;
}

// Get cached K and V pointers for a layer
static bool get_dev1_kv_cache_ptrs(int layer, float ** k_cache, float ** v_cache, int64_t * seq_len) {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);
    auto                        it = g_tp_dev1_kv_cache.find(layer);
    if (it == g_tp_dev1_kv_cache.end()) {
        *k_cache = nullptr;
        *v_cache = nullptr;
        *seq_len = 0;
        return false;
    }
    *k_cache = it->second.k_cache;
    *v_cache = it->second.v_cache;
    *seq_len = it->second.seq_pos;
    return true;
}

// Reset all KV caches (call at start of new sequence)
static void reset_dev1_kv_cache() {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);
    for (auto & kv : g_tp_dev1_kv_cache) {
        kv.second.seq_pos = 0;
    }
    static int log_count = 0;
    if (g_ggml_sycl_tp_debug && (log_count++ < 3 || g_tp_dev1_kv_cache.size() > 0)) {
        fprintf(stderr, "SYCL TP: Reset dev1 KV cache for %zu layers\n", g_tp_dev1_kv_cache.size());
    }
}

// Free all KV caches
static void free_dev1_kv_cache() {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);
    for (auto & kv : g_tp_dev1_kv_cache) {
        if (kv.second.k_cache && kv.second.stream) {
            sycl::free(kv.second.k_cache, *kv.second.stream);
        }
        if (kv.second.v_cache && kv.second.stream) {
            sycl::free(kv.second.v_cache, *kv.second.stream);
        }
    }
    g_tp_dev1_kv_cache.clear();
    fprintf(stderr, "SYCL TP: Freed all dev1 KV caches\n");
}

// Store FFN and attention weight references (called during tensor loading)
static void store_ffn_weight_ref(const ggml_tensor * tensor) {
    if (!tensor || !tensor->name[0]) {
        return;
    }

    int layer = extract_layer_number(tensor->name);
    if (layer < 0) {
        return;
    }

    // Store FFN weight references
    if (strstr(tensor->name, "ffn_gate") || strstr(tensor->name, "ffn_up") || strstr(tensor->name, "ffn_down")) {
        std::lock_guard<std::mutex> lock(g_tp_ffn_weight_mutex);
        if (strstr(tensor->name, "ffn_gate")) {
            g_tp_ffn_weights[layer].gate = tensor;
        } else if (strstr(tensor->name, "ffn_up")) {
            g_tp_ffn_weights[layer].up = tensor;
        } else if (strstr(tensor->name, "ffn_down")) {
            g_tp_ffn_weights[layer].down = tensor;
        }
    }

    // Store attention weight references
    if (strstr(tensor->name, "attn_q") || strstr(tensor->name, "attn_k") || strstr(tensor->name, "attn_v") ||
        strstr(tensor->name, "attn_output")) {
        std::lock_guard<std::mutex> lock(g_tp_attn_weight_mutex);
        if (strstr(tensor->name, "attn_output")) {
            g_tp_attn_weights[layer].o = tensor;
        } else if (strstr(tensor->name, "attn_q")) {
            g_tp_attn_weights[layer].q = tensor;
        } else if (strstr(tensor->name, "attn_k")) {
            g_tp_attn_weights[layer].k = tensor;
        } else if (strstr(tensor->name, "attn_v")) {
            g_tp_attn_weights[layer].v = tensor;
        }
    }
}

// Clear old column-parallel outputs (call at appropriate times to free memory)
static void ggml_sycl_tp_clear_column_parallel_outputs(ggml_backend_sycl_context & ctx) {
    // Multi-process mode: No column-parallel outputs stored
    if (g_sycl_tp_config.is_multiprocess) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_tp_column_parallel_mutex);
    if (g_tp_column_parallel_outputs.empty()) {
        return;
    }

    // Free buffers on device 1
    int device = g_sycl_tp_config.devices[1];
    ggml_sycl_set_device(device);
    queue_ptr stream = ctx.stream(device, 0);

    for (auto & kv : g_tp_column_parallel_outputs) {
        if (kv.second != nullptr) {
            sycl::free(kv.second, *stream);
        }
    }
    g_tp_column_parallel_outputs.clear();
    ggml_sycl_set_device(ctx.device);
}

// Get stored column-parallel output for a tensor
static void * ggml_sycl_tp_get_column_parallel_output(const ggml_tensor * tensor) {
    std::lock_guard<std::mutex> lock(g_tp_column_parallel_mutex);
    auto                        it = g_tp_column_parallel_outputs.find(tensor->data);
    if (it != g_tp_column_parallel_outputs.end()) {
        return it->second;
    }
    return nullptr;
}

// TP column-parallel mul_mat post-processing
// For FFN layers: stores the input (src1) so row-parallel (ffn_down) can compute device 1's path
// For attention layers: stores the input (src1) so row-parallel (attn_output) can compute device 1's path
static void ggml_sycl_mul_mat_tp_column_parallel_post(ggml_backend_sycl_context &    ctx,
                                                      const ggml_tensor *            src0,
                                                      const ggml_tensor *            src1,
                                                      [[maybe_unused]] ggml_tensor * dst) {
    // Multi-GPU column-parallel: store input for row-parallel computation on device 1
    // This copies the input (src1) to device 1 for later use by row-parallel ops

    // Multi-process mode: Skip - each process handles its own computation
    // Cross-device input sharing is done via CCL, not direct device copying
    if (g_sycl_tp_config.is_multiprocess) {
        return;
    }

    if (!is_tp_sharded_tensor(src0)) {
        return;
    }

    const auto * src0_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (src0_extra->tp_type != tp_layer_type::TP_COLUMN_PARALLEL) {
        return;
    }

    const char * name = src0->name;
    if (!name) {
        return;
    }

    // Check if this is FFN or attention input storage
    bool is_ffn_gate = strstr(name, "ffn_gate") != nullptr;
    bool is_attn_q   = strstr(name, "attn_q") != nullptr && strstr(name, "attn_qkv") == nullptr;

    if (!is_ffn_gate && !is_attn_q) {
        return;  // Only store at ffn_gate and attn_q to avoid duplicate storage
    }

    int layer = extract_layer_number(name);

    // DEBUG: Track column-parallel entry during decode (batch=1)
    static int col_par_decode_dbg = 0;
    if (g_ggml_sycl_tp_debug && src1->ne[1] == 1 && is_ffn_gate && col_par_decode_dbg++ < 100) {
        fprintf(stderr, "TP DEBUG COL_PARALLEL FFN_GATE decode: layer=%d name=%s map_size_before=%zu\n", layer, name,
                g_tp_ffn_inputs.size());
    }

    // DEBUG: Check layer input during decode for all layers
    static int layer_input_dbg = 0;
    bool       is_debug_layer  = (layer >= 0 && layer <= 31);
    if (g_ggml_sycl_tp_debug && src1->ne[1] == 1 && is_debug_layer && is_attn_q && layer_input_dbg++ < 200) {
        queue_ptr     main_stream = ctx.stream();
        float         sample[8];
        // Use device-specific pointer for TP
        const float * src1_dd = static_cast<const float *>(ggml_sycl_get_data_ptr(src1, ctx.device));
        main_stream->memcpy(sample, src1_dd, 8 * sizeof(float)).wait();
        float sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += sample[i];
        }
        fprintf(stderr, "LAYER%d_INPUT decode: src1[0..7]=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] sum=%.4f\n", layer,
                sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sum);
    }
    if (layer < 0) {
        return;
    }

    const int main_device = ctx.device;
    const int device      = g_sycl_tp_config.devices[1];  // Device 1

    // Copy src1 (input) to device 1
    const int64_t K         = src1->ne[0];
    const int64_t batch     = src1->ne[1];
    const size_t  src1_size = batch * K * sizeof(float);

    // DEBUG: Print device 0's input values for comparison
    static int dev0_input_dbg = 0;
    bool       debug_input =
        (g_ggml_sycl_tp_debug && is_ffn_gate && layer < 3 && dev0_input_dbg++ < 10);  // Enable for first few layers
    if (debug_input) {
        queue_ptr main_stream = ctx.stream();
        void *    src1_ptr    = ggml_sycl_get_data_ptr(src1, main_device);
        float     sample[4];
        main_stream->memcpy(sample, src1_ptr, 4 * sizeof(float)).wait();
        fprintf(stderr, "TP DEBUG FFN CAPTURE layer %d batch=%lld: src1->data=%p src1_ptr=%p %s\n", layer,
                (long long) batch, src1->data, src1_ptr, (src1->data != src1_ptr) ? "MISMATCH!" : "");
        bool has_nan = std::isnan(sample[0]) || std::isnan(sample[1]) || std::isnan(sample[2]) || std::isnan(sample[3]);
        fprintf(stderr, "TP DEBUG FFN CAPTURE layer %d batch=%lld: device0_input[0..3]=[%f,%f,%f,%f] nan=%d\n", layer,
                (long long) batch, sample[0], sample[1], sample[2], sample[3], has_nan);
    }

    ggml_sycl_set_device(device);
    queue_ptr stream = ctx.stream(device, 0);

    // DEBUG: Check L31 weight BEFORE malloc_device for input staging
    static int staging_pre_malloc_dbg = 0;
    bool       check_staging = (g_ggml_sycl_tp_debug && layer == 31 && is_ffn_gate && staging_pre_malloc_dbg < 3);
    if (check_staging) {
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        queue_ptr dev0_stream     = ctx.stream(main_device, 0);
        uint8_t   weight_bytes[18];
        dev0_stream->memcpy(weight_bytes, (void *) l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float    d_f   = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING PRE-MALLOC: weight d=%f %s\n", d_f, (d_f > 100.0f) ? "CORRUPTED" : "OK");
    }

    // Allocate on device 1
    float * input_dev1 = (float *) sycl::malloc_device(src1_size, *stream);

    // DEBUG: Check L31 weight AFTER malloc_device and print allocated address
    if (check_staging) {
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        queue_ptr dev0_stream     = ctx.stream(main_device, 0);
        uint8_t   weight_bytes[18];
        dev0_stream->memcpy(weight_bytes, (void *) l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float    d_f   = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING POST-MALLOC: weight d=%f, input_dev1=%p %s\n", d_f, (void *) input_dev1,
                (d_f > 100.0f) ? "CORRUPTED" : "OK");
        staging_pre_malloc_dbg++;
    }

    if (!input_dev1) {
        GGML_LOG_ERROR("SYCL TP: ERROR - failed to allocate %s input on device %d\n", is_ffn_gate ? "FFN" : "attention",
                       device);
        ggml_sycl_set_device(main_device);
        return;
    }

    // Copy via host - OPTIMIZED: use persistent host staging buffer
    ggml_sycl_set_device(main_device);
    queue_ptr main_stream = ctx.stream();

    float * host_buf = ggml_sycl_tp_ensure_host_staging(src1_size, main_stream);
    if (!host_buf) {
        fprintf(stderr, "SYCL TP: ERROR - failed to get host staging buffer for %s input\n",
                is_ffn_gate ? "FFN" : "attention");
        sycl::free(input_dev1, *stream);
        return;
    }

    // DEBUG: Check after malloc_host
    static int staging_step_dbg = 0;
    bool       trace_staging    = (g_ggml_sycl_tp_debug && layer == 31 && is_ffn_gate && staging_step_dbg < 1);
    if (trace_staging) {
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        uint8_t   weight_bytes[18];
        main_stream->memcpy(weight_bytes, (void *) l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float    d_f   = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING STEP1 (after host malloc): weight d=%f %s\n", d_f,
                (d_f > 100.0f) ? "CORRUPTED" : "OK");
    }

    // For FFN gate: try to use cached FFN norm (prevents buffer aliasing issues)
    // The GGML scheduler may reuse the ffn_norm buffer before we can read it
    void * cached_ffn_norm = nullptr;
    if (is_ffn_gate) {
        cached_ffn_norm = ggml_sycl_tp_get_cached_ffn_norm(layer, main_device);
        if (cached_ffn_norm) {
            main_stream->memcpy(host_buf, cached_ffn_norm, src1_size).wait();
        } else {
            // IMPORTANT: Use device-specific pointer for TP mode!
            void * src1_ptr = ggml_sycl_get_data_ptr(src1, main_device);
            main_stream->memcpy(host_buf, src1_ptr, src1_size).wait();
        }
        // DEBUG: Check after memcpy to host_buf
        if (trace_staging) {
            uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
            uint8_t   weight_bytes[18];
            main_stream->memcpy(weight_bytes, (void *) l31_weight_addr, 18).wait();
            uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
            float    d_f   = ggml_fp16_to_fp32(d_raw);
            fprintf(stderr, "TP DEBUG L31 STAGING STEP2 (after host memcpy): weight d=%f %s\n", d_f,
                    (d_f > 100.0f) ? "CORRUPTED" : "OK");
        }
        // DEBUG: Check FFN input at storage time for batch=1 (disabled - TP working)
        static int ffn_store_dbg = 0;
        bool       debug_layer   = (batch == 1 && layer == 0 && ffn_store_dbg++ < 0);  // Disabled
        if (debug_layer) {
            float check[4];
            memcpy(check, host_buf, 4 * sizeof(float));
            bool has_nan = std::isnan(check[0]) || std::isnan(check[1]) || std::isnan(check[2]) || std::isnan(check[3]);
            fprintf(stderr, "TP DEBUG FFN STORE layer %d batch=1: src=%s, input[0..3]=[%f,%f,%f,%f] nan=%d\n", layer,
                    cached_ffn_norm ? "cached" : "src1", check[0], check[1], check[2], check[3], has_nan);
        }
    } else {
        // IMPORTANT: Use device-specific pointer for TP mode!
        void * src1_ptr = ggml_sycl_get_data_ptr(src1, main_device);
        main_stream->memcpy(host_buf, src1_ptr, src1_size).wait();
    }

    ggml_sycl_set_device(device);
    stream = ctx.stream(device, 0);
    stream->memcpy(input_dev1, host_buf, src1_size).wait();

    // DEBUG: Check after memcpy to device 1
    if (trace_staging) {
        ggml_sycl_set_device(main_device);
        queue_ptr dev0_stream     = ctx.stream(main_device, 0);
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        uint8_t   weight_bytes[18];
        dev0_stream->memcpy(weight_bytes, (void *) l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float    d_f   = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING STEP3 (after dev1 memcpy): weight d=%f %s\n", d_f,
                (d_f > 100.0f) ? "CORRUPTED" : "OK");
        ggml_sycl_set_device(device);
    }

    // OPTIMIZATION: Don't free host_buf - it's a persistent staging buffer
    // managed by ggml_sycl_tp_ensure_host_staging()
    // sycl::free(host_buf, *main_stream);

    // DEBUG: Check after host buffer free
    if (trace_staging) {
        ggml_sycl_set_device(main_device);
        queue_ptr dev0_stream     = ctx.stream(main_device, 0);
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        uint8_t   weight_bytes[18];
        dev0_stream->memcpy(weight_bytes, (void *) l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float    d_f   = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING STEP4 (after host free): weight d=%f %s\n", d_f,
                (d_f > 100.0f) ? "CORRUPTED" : "OK");
        staging_step_dbg++;
        ggml_sycl_set_device(device);
    }

    // Store for later use by row-parallel layer
    if (is_ffn_gate) {
        // Store the input
        {
            std::lock_guard<std::mutex> lock(g_tp_ffn_input_mutex);
            auto                        it = g_tp_ffn_inputs.find(layer);
            if (it != g_tp_ffn_inputs.end() && it->second.data != nullptr) {
                sycl::free(it->second.data, *stream);
            }
            g_tp_ffn_inputs[layer]   = { input_dev1, K, batch, src1_size };
            // DEBUG: Track FFN input storage during decode
            static int ffn_store_dbg = 0;
            if (g_ggml_sycl_tp_debug && batch == 1 && ffn_store_dbg++ < 100) {
                fprintf(stderr, "TP DEBUG FFN_INPUT_STORE decode: layer=%d ptr=%p map_size_after=%zu\n", layer,
                        (void *) input_dev1, g_tp_ffn_inputs.size());
            }
        }

        // PHASE 4 PIPELINING: Try to launch async FFN computation now
        // This allows device 1 to work while device 0 continues with other ops
        ffn_weight_refs weights = {};
        {
            std::lock_guard<std::mutex> lock(g_tp_ffn_weight_mutex);
            auto                        it = g_tp_ffn_weights.find(layer);
            if (it != g_tp_ffn_weights.end()) {
                weights = it->second;
            }
        }

        if (weights.gate && weights.up && weights.down) {
            // Try thread-based pipelining if enabled
            if (g_ggml_sycl_tp_threaded_ffn) {
                // Initialize worker thread if not already done
                if (!g_tp_device1_worker.initialized.load()) {
                    ggml_sycl_tp_worker_init(&ctx);
                }

                if (g_tp_device1_worker.initialized.load()) {
                    // Submit FFN work to worker thread
                    tp_ffn_work_item work = {};
                    work.layer            = layer;
                    work.input_dev1       = input_dev1;
                    work.K_full           = K;
                    work.batch            = batch;
                    work.weights          = weights;

                    if (g_ggml_sycl_tp_debug) {
                        static int thread_launch_count = 0;
                        if (thread_launch_count++ < 3) {
                            fprintf(stderr,
                                    "SYCL TP: Submitting FFN to worker thread for layer %d (K=%lld, batch=%lld)\n",
                                    layer, (long long) K, (long long) batch);
                        }
                    }
                    ggml_sycl_tp_submit_ffn_work(work);
                }
            }
            // Note: If threaded FFN is disabled or worker not init, fall back to sync path in row_parallel_post
        } else {
            // Weights not ready - will fall back to sync path in row_parallel_post
            if (g_ggml_sycl_tp_debug) {
                static int no_weights_count = 0;
                if (no_weights_count++ < 3) {
                    fprintf(stderr, "SYCL TP: No async FFN for layer %d (weights not ready: gate=%p, up=%p, down=%p)\n",
                            layer, (void *) weights.gate, (void *) weights.up, (void *) weights.down);
                }
            }
        }
    } else {  // is_attn_q
        std::lock_guard<std::mutex> lock(g_tp_attn_input_mutex);
        auto                        it = g_tp_attn_inputs.find(layer);
        if (it != g_tp_attn_inputs.end() && it->second.data != nullptr) {
            sycl::free(it->second.data, *stream);
        }
        g_tp_attn_inputs[layer] = { input_dev1, K, batch, src1_size };

        // DEBUG: Track attn input storage during decode
        static int attn_store_dbg = 0;
        if (g_ggml_sycl_tp_debug && batch == 1 && attn_store_dbg++ < 40) {
            fprintf(stderr, "TP DEBUG ATTN_Q_STORE decode: layer=%d data=%p map_size=%zu\n", layer, (void *) input_dev1,
                    g_tp_attn_inputs.size());
        }
    }

    ggml_sycl_set_device(main_device);
    return;  // Don't run the old code below

#if 0        // Full implementation for future reference
    if (!is_tp_sharded_tensor(src0)) {
        return;
    }

    const auto * src0_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (src0_extra->tp_type != tp_layer_type::TP_COLUMN_PARALLEL) {
        return;  // Only column-parallel needs this
    }

    const int world_size = src0_extra->tp_world_size;
    const int main_device = ctx.device;

    // Dimensions for column-parallel:
    // src0 (weight): [K, N_shard] where N_shard = N / world_size
    // src1 (input):  [K, batch]
    // dst (output):  [N_shard, batch]
    const int64_t K = src1->ne[0];
    const int64_t N_shard = dst->ne[0];
    const int64_t batch = dst->ne[1];
    const size_t dst_size = ggml_nbytes(dst);

    static int log_count = 0;
    if (log_count < 3) {
        fprintf(stderr, "SYCL TP column-parallel post: K=%ld, N_shard=%ld, batch=%ld, world_size=%d\n",
                (long)K, (long)N_shard, (long)batch, world_size);
        log_count++;
    }

    // For each other TP device (rank > 0), compute its column-parallel output
    for (int rank = 1; rank < world_size; rank++) {
        int device = g_sycl_tp_config.devices[rank];

        // Get this rank's weight shard
        void * weight_shard = src0_extra->data_device[device];
        if (weight_shard == nullptr) {
            fprintf(stderr, "SYCL TP: ERROR - no weight shard on device %d for rank %d\n", device, rank);
            continue;
        }

        // Q8_1 parameters for src1 quantization
        const size_t q8_1_ts = sizeof(block_q8_1);
        const size_t q8_1_bs = QK8_1;
        const int64_t K_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
        const size_t src1_float_size = batch * K * sizeof(float);
        const size_t src1_q8_size = batch * K_padded * q8_1_ts / q8_1_bs;

        ggml_sycl_set_device(device);
        queue_ptr stream = ctx.stream(device, 0);

        // Allocate buffers on target device
        float * src1_ddf_dev = (float *)sycl::malloc_device(src1_float_size, *stream);
        char * src1_ddq_dev = (char *)sycl::malloc_device(src1_q8_size, *stream);
        float * col_out = (float *)sycl::malloc_device(dst_size, *stream);

        if (!src1_ddf_dev || !src1_ddq_dev || !col_out) {
            fprintf(stderr, "SYCL TP: ERROR - failed to allocate temp buffers on device %d\n", device);
            if (src1_ddf_dev) sycl::free(src1_ddf_dev, *stream);
            if (src1_ddq_dev) sycl::free(src1_ddq_dev, *stream);
            if (col_out) sycl::free(col_out, *stream);
            ggml_sycl_set_device(main_device);
            continue;
        }

        // Copy src1 (full input) to device 1 - OPTIMIZED: use persistent staging buffer
        {
            ggml_sycl_set_device(main_device);
            queue_ptr main_stream = ctx.stream();

            float * host_buf = ggml_sycl_tp_ensure_host_staging(src1_float_size, main_stream);
            main_stream->memcpy(host_buf, src1->data, src1_float_size).wait();

            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);
            stream->memcpy(src1_ddf_dev, host_buf, src1_float_size).wait();

            // Don't free - using persistent staging buffer
        }

        // Quantize src1 to Q8_1 on target device
        // Use SoA quantizer if weight is reordered
        {
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);

            const bool use_soa_col_parallel = src0_extra && src0_extra->optimized_feature.is_reordered();
            if (use_soa_col_parallel) {
                quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(src1_ddf_dev, src1_ddq_dev,
                                                      K, batch, K_padded, stream);
            } else {
                quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_dev, src1_ddq_dev,
                                                      K, batch, K_padded, stream);
            }
            stream->wait();
        }

        // Call MMVQ kernel to compute column-parallel output
        {
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);

            stream->memset(col_out, 0, dst_size).wait();

            // Call MMVQ: weight_shard [K, N_shard] @ src1 [K, batch] -> col_out [N_shard, batch]
            ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
                                        (const char *)weight_shard,
                                        nullptr,
                                        src1_ddq_dev,
                                        col_out,
                                        0,           // row_low
                                        N_shard,     // row_high
                                        batch,       // src1_ncols
                                        K_padded,    // src1_padded_row_size
                                        stream);
            stream->wait();
        }

        // Store result in global map for retrieval by row-parallel
        // Key is dst->data pointer (identifies this tensor during graph execution)
        {
            std::lock_guard<std::mutex> lock(g_tp_column_parallel_mutex);
            // Free any existing buffer for this key
            auto it = g_tp_column_parallel_outputs.find(dst->data);
            if (it != g_tp_column_parallel_outputs.end() && it->second != nullptr) {
                sycl::free(it->second, *stream);
            }
            g_tp_column_parallel_outputs[dst->data] = col_out;
        }

        // Clean up temp buffers (but NOT col_out - it's stored)
        sycl::free(src1_ddf_dev, *stream);
        sycl::free(src1_ddq_dev, *stream);
    }

    ggml_sycl_set_device(main_device);
#endif       // End of disabled column-parallel implementation
}

// =============================================================================
// ASYNC FFN COMPUTATION FOR TENSOR PARALLELISM PIPELINING
// =============================================================================
//
// These functions launch FFN computation on device 1 asynchronously during
// column-parallel processing, allowing device 0 to continue with other work.
// The result is retrieved later at the ALL_REDUCE point.
//
// Timeline:
//   Column-parallel (gate):  [store input]---[launch async FFN on dev1]
//                                           |
//   Device 0 continues:      [up matmul]---[other ops]---[row-parallel down]
//                                                                    |
//   ALL_REDUCE:              [wait for async FFN]---[GPU add kernel]
//

// Launch async FFN computation on device 1 (returns immediately)
void ggml_sycl_tp_launch_async_ffn(ggml_backend_sycl_context & ctx,
                                   int                         layer,
                                   const float *               input_dev1,
                                   int64_t                     K_full,
                                   int64_t                     batch,
                                   const ffn_weight_refs &     weights) {
    // Multi-process mode: Each process handles its own FFN computation
    // No async device-to-device coordination needed
    if (g_sycl_tp_config.is_multiprocess) {
        return;
    }

    if (!weights.gate || !weights.up || !weights.down) {
        GGML_SYCL_DEBUG("SYCL TP ASYNC: Missing weight refs for layer %d\n", layer);
        return;
    }

    const int main_device = ctx.device;
    const int device      = g_sycl_tp_config.devices[1];

    ggml_sycl_set_device(device);
    queue_ptr stream = ctx.stream(device, 0);

    // Get weight shards for device 1
    auto * gate_extra = static_cast<ggml_tensor_extra_gpu *>(weights.gate->extra);
    auto * up_extra   = static_cast<ggml_tensor_extra_gpu *>(weights.up->extra);
    auto * down_extra = static_cast<ggml_tensor_extra_gpu *>(weights.down->extra);

    void * gate_weight_1 = gate_extra ? gate_extra->data_device[device] : nullptr;
    void * up_weight_1   = up_extra ? up_extra->data_device[device] : nullptr;
    void * down_weight_1 = down_extra ? down_extra->data_device[device] : nullptr;

    if (!gate_weight_1 || !up_weight_1 || !down_weight_1) {
        GGML_SYCL_DEBUG("SYCL TP ASYNC: Missing weight shards on device 1 for layer %d\n", layer);
        ggml_sycl_set_device(main_device);
        return;
    }

    // Get dimensions
    const int64_t N_hidden_shard = gate_extra->tp_local_ne[1];  // Sharded hidden dim
    const int64_t N_out          = down_extra->tp_local_ne[1];  // Output dimension

    // Allocate buffers on device 1
    const size_t  q8_1_ts               = sizeof(block_q8_1);
    const size_t  q8_1_bs               = QK8_1;
    const int64_t K_full_padded         = GGML_PAD(K_full, MATRIX_ROW_PADDING);
    const int64_t N_hidden_shard_padded = GGML_PAD(N_hidden_shard, MATRIX_ROW_PADDING);

    const size_t input_q8_size  = batch * K_full_padded * q8_1_ts / q8_1_bs;
    const size_t hidden_size    = N_hidden_shard * batch * sizeof(float);
    const size_t hidden_q8_size = batch * N_hidden_shard_padded * q8_1_ts / q8_1_bs;
    const size_t output_size    = N_out * batch * sizeof(float);

    char *  input_q8_dev  = (char *) sycl::malloc_device(input_q8_size, *stream);
    float * gate_out      = (float *) sycl::malloc_device(hidden_size, *stream);
    float * up_out        = (float *) sycl::malloc_device(hidden_size, *stream);
    float * hidden_out    = (float *) sycl::malloc_device(hidden_size, *stream);
    char *  hidden_q8_dev = (char *) sycl::malloc_device(hidden_q8_size, *stream);
    float * partial_out   = (float *) sycl::malloc_device(output_size, *stream);

    // Allocate DEDICATED result buffer for this async job (not shared!)
    // Each layer needs its own buffer to avoid races between concurrent async jobs
    float * result_buf = (float *) ggml_sycl_host_malloc(output_size);

    if (!input_q8_dev || !gate_out || !up_out || !hidden_out || !hidden_q8_dev || !partial_out || !result_buf) {
        GGML_SYCL_DEBUG("SYCL TP ASYNC: Buffer allocation failed for layer %d\n", layer);
        if (input_q8_dev) {
            sycl::free(input_q8_dev, *stream);
        }
        if (gate_out) {
            sycl::free(gate_out, *stream);
        }
        if (up_out) {
            sycl::free(up_out, *stream);
        }
        if (hidden_out) {
            sycl::free(hidden_out, *stream);
        }
        if (hidden_q8_dev) {
            sycl::free(hidden_q8_dev, *stream);
        }
        if (partial_out) {
            sycl::free(partial_out, *stream);
        }
        if (result_buf) {
            ggml_sycl_host_free(result_buf);
        }
        ggml_sycl_set_device(main_device);
        return;
    }

    // Create a fake dst tensor with correct dimensions for MMVQ calls
    ggml_tensor fake_dst_hidden;
    memset(&fake_dst_hidden, 0, sizeof(fake_dst_hidden));
    fake_dst_hidden.ne[0] = N_hidden_shard;
    fake_dst_hidden.ne[1] = batch;

    ggml_tensor fake_dst_out;
    memset(&fake_dst_out, 0, sizeof(fake_dst_out));
    fake_dst_out.ne[0] = N_out;
    fake_dst_out.ne[1] = batch;

    // Step 1: Quantize input to Q8_1
    // Use SoA quantizer if weights are in SoA layout (reordered kernels expect SoA Y)
    ggml_tensor_extra_gpu * gate_extra_async = static_cast<ggml_tensor_extra_gpu *>(weights.gate->extra);
    const bool use_soa_input_async           = gate_extra_async && gate_extra_async->optimized_feature.is_reordered();
    if (use_soa_input_async) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(input_dev1, input_q8_dev, K_full, batch, K_full_padded,
                                                              stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(input_dev1, input_q8_dev, K_full, batch, K_full_padded, stream);
    }

    // Step 2-3: Gate and Up matmuls (sequential on same queue)
    stream->memset(gate_out, 0, hidden_size);
    stream->memset(up_out, 0, hidden_size);
    stream->wait();

    ggml_sycl_op_mul_mat_vec_q(ctx, weights.gate, nullptr, &fake_dst_hidden, (const char *) gate_weight_1, nullptr,
                               input_q8_dev, gate_out, 0, N_hidden_shard, batch, K_full_padded, stream);

    ggml_sycl_op_mul_mat_vec_q(ctx, weights.up, nullptr, &fake_dst_hidden, (const char *) up_weight_1, nullptr,
                               input_q8_dev, up_out, 0, N_hidden_shard, batch, K_full_padded, stream);
    stream->wait();

    // Step 4: SiLU activation and multiply
    const int64_t n_elements = N_hidden_shard * batch;
    const int     block_size = 256;
    const int     num_blocks = (n_elements + block_size - 1) / block_size;
    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> item) {
        const int i = item.get_global_id(0);
        if (i < n_elements) {
            float g       = gate_out[i];
            float u       = up_out[i];
            float silu_g  = g / (1.0f + sycl::native::exp(-g));
            hidden_out[i] = silu_g * u;
        }
    });

    // Step 5: Quantize hidden for down matmul
    // Use SoA quantizer if down weight is in SoA layout
    ggml_tensor_extra_gpu * down_extra_async = static_cast<ggml_tensor_extra_gpu *>(weights.down->extra);
    const bool use_soa_hidden_async          = down_extra_async && down_extra_async->optimized_feature.is_reordered();
    if (use_soa_hidden_async) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(hidden_out, hidden_q8_dev, N_hidden_shard, batch,
                                                              N_hidden_shard_padded, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(hidden_out, hidden_q8_dev, N_hidden_shard, batch, N_hidden_shard_padded,
                                              stream);
    }

    // Step 6: Down matmul
    stream->memset(partial_out, 0, output_size);
    stream->wait();
    ggml_sycl_op_mul_mat_vec_q(ctx, weights.down, nullptr, &fake_dst_out, (const char *) down_weight_1, nullptr,
                               hidden_q8_dev, partial_out, 0, N_out, batch, N_hidden_shard_padded, stream);

    // Step 7: Copy result to shared buffer (this is the final operation)
    // Store the event so we can wait for it later
    sycl::event completion_event = stream->memcpy(result_buf, partial_out, output_size);

    // Clean up device 1 buffers (submit async free after computation)
    // Note: These frees depend on the computation completing
    stream->submit([&](sycl::handler & h) {
        h.depends_on(completion_event);
        h.host_task([=]() {
            // This runs after completion_event, safe to access pointers
        });
    });
    // Schedule actual frees
    sycl::free(input_q8_dev, *stream);
    sycl::free(gate_out, *stream);
    sycl::free(up_out, *stream);
    sycl::free(hidden_out, *stream);
    sycl::free(hidden_q8_dev, *stream);
    sycl::free(partial_out, *stream);

    // Store the job info
    {
        std::lock_guard<std::mutex> lock(g_tp_async_ffn_mutex);
        g_tp_async_ffn_jobs[layer] = { layer, completion_event, result_buf, N_out, batch, output_size, true };
    }

    if (g_ggml_sycl_tp_debug) {
        static int launch_dbg = 0;
        if (launch_dbg++ < 5) {
            fprintf(stderr, "SYCL TP ASYNC: Launched FFN layer %d, K=%lld, batch=%lld, N_hidden=%lld, N_out=%lld\n",
                    layer, (long long) K_full, (long long) batch, (long long) N_hidden_shard, (long long) N_out);
        }
    }

    ggml_sycl_set_device(main_device);
}

// Wait for and retrieve async FFN result
float * ggml_sycl_tp_wait_async_ffn(int layer, int64_t * out_ne0, int64_t * out_ne1, size_t * out_size) {
    // Quick early return if no async jobs have been launched (avoid mutex overhead)
    // This is critical for performance when async is disabled
    if (g_tp_async_ffn_jobs.empty()) {
        return nullptr;
    }

    tp_async_ffn_job job = {};

    {
        std::lock_guard<std::mutex> lock(g_tp_async_ffn_mutex);
        auto                        it = g_tp_async_ffn_jobs.find(layer);
        if (it == g_tp_async_ffn_jobs.end() || !it->second.valid) {
            return nullptr;
        }
        job              = it->second;
        it->second.valid = false;  // Mark as consumed
    }

    // Wait for computation to complete
    job.completion_event.wait();

    if (out_ne0) {
        *out_ne0 = job.ne0;
    }
    if (out_ne1) {
        *out_ne1 = job.ne1;
    }
    if (out_size) {
        *out_size = job.result_size;
    }

    if (g_ggml_sycl_tp_debug) {
        static int wait_dbg = 0;
        if (wait_dbg++ < 5) {
            fprintf(stderr, "SYCL TP ASYNC: Waited for FFN layer %d, result[0..3]=[%f,%f,%f,%f]\n", layer,
                    job.result_buf[0], job.result_buf[1], job.result_buf[2], job.result_buf[3]);
        }
    }

    return job.result_buf;
}

// Launch async attention computation on device 1
void ggml_sycl_tp_launch_async_attn(ggml_backend_sycl_context & ctx,
                                    int                         layer,
                                    const float *               input_dev1,
                                    int64_t                     K_full,
                                    int64_t                     batch,
                                    const attn_weight_refs &    weights) {
    // TODO: Implement async attention (similar to FFN but with Q/K/V/O path)
    // For now, attention will use the synchronous path
    (void) ctx;
    (void) layer;
    (void) input_dev1;
    (void) K_full;
    (void) batch;
    (void) weights;
}

// Wait for async attention result
float * ggml_sycl_tp_wait_async_attn(int layer, int64_t * out_ne0, int64_t * out_ne1, size_t * out_size) {
    // TODO: Implement async attention wait
    (void) layer;
    (void) out_ne0;
    (void) out_ne1;
    (void) out_size;
    return nullptr;
}

// TP row-parallel mul_mat post-processing: Compute on other TP devices and add results
// Called AFTER regular mul_mat completes on main device
//
// For row-parallel:
// - Weight [K, N] is split along K to [K/world_size, N] per device
// - Each device processes a different slice of src1 (the K dimension)
// - Outputs are summed (ALL_REDUCE_SUM)
//
// Main device already computed with its shard. This function:
// 1. For each other TP device: use its column-parallel output (or src1 slice), compute, add result
static void ggml_sycl_mul_mat_tp_row_parallel_post(ggml_backend_sycl_context & ctx,
                                                   const ggml_tensor *         src0,
                                                   const ggml_tensor *         src1,
                                                   ggml_tensor *               dst) {
    // Multi-process mode: Each process handles its own computation
    // ALL_REDUCE is done via CCL in the GGML_OP_ALL_REDUCE_SUM handler, not here
    if (g_sycl_tp_config.is_multiprocess) {
        return;
    }

    // Multi-GPU row-parallel: compute on device 1 and sum results
    // Device 0 already computed its partial result in the main mul_mat path

    if (!is_tp_sharded_tensor(src0)) {
        return;
    }

    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (extra->tp_type != tp_layer_type::TP_ROW_PARALLEL) {
        return;  // Only row-parallel needs this
    }

    const int world_size  = extra->tp_world_size;
    const int main_device = ctx.device;

    // Dimensions
    const int64_t ne00 = src0->ne[0];  // K_shard (sharded K dimension per device)
    const int64_t ne01 = src0->ne[1];  // N (output dimension)
    const int64_t ne10 = src1->ne[0];  // K (may be sharded if from column-parallel)
    const int64_t ne11 = src1->ne[1];  // batch/seq_len

    const int64_t K_shard = ne00;

    // Output size
    const size_t dst_nelems = ggml_nelements(dst);
    const size_t dst_size   = ggml_nbytes(dst);

    // Check if src1 already has sharded dimension (from column-parallel layer output)
    const bool src1_from_column_parallel = (ne10 == K_shard);

    if (src1_from_column_parallel) {
        // src1 came from column-parallel layer - check if this is ffn_down
        const char * name        = src0->name;
        int          layer       = extract_layer_number(name);
        bool         is_ffn_down = name && strstr(name, "ffn_down");

        if (is_ffn_down && layer >= 0) {
            // For FFN: check for async result first (Phase 4 pipelining)
            static int ffn_dbg_count = 0;
            if (g_ggml_sycl_tp_debug && ffn_dbg_count++ < 3) {
                fprintf(stderr, "TP DEBUG: FFN down layer %d, entering computation path\n", layer);
            }

            // PHASE 4 PIPELINING: Check if FFN result is available
            // Try thread-based pipelining first (new approach)
            float * async_result = nullptr;
            int64_t async_ne0 = 0, async_ne1 = 0;
            size_t  async_size = 0;

            if (g_ggml_sycl_tp_threaded_ffn && g_tp_device1_worker.initialized.load()) {
                // Check for thread-based result (wait with timeout)
                tp_ffn_result * thread_result = ggml_sycl_tp_get_ffn_result(layer, true);
                if (thread_result && thread_result->valid) {
                    async_result = thread_result->result_buf;
                    async_ne0    = thread_result->ne0;
                    async_ne1    = thread_result->ne1;
                    async_size   = thread_result->result_size;
                }
            } else if (g_ggml_sycl_tp_async_ffn) {
                // Fallback to old async FFN path (disabled)
                async_result = ggml_sycl_tp_wait_async_ffn(layer, &async_ne0, &async_ne1, &async_size);
            }

            if (async_result != nullptr) {
                // ASYNC PATH: Result already computed, just do ALL_REDUCE
                static int async_used_count = 0;
                if (g_ggml_sycl_tp_debug && async_used_count++ < 5) {
                    fprintf(stderr, "SYCL TP: Using ASYNC FFN result for layer %d (result[0..3]=[%f,%f,%f,%f])\n",
                            layer, async_result[0], async_result[1], async_result[2], async_result[3]);
                }

                // Do ALL_REDUCE: add async_result to dst using GPU kernel
                queue_ptr     main_stream  = ctx.stream();
                // Use ggml_sycl_get_data_ptr for correct device-specific pointer in TP mode
                float *       dst_ptr      = (float *) ggml_sycl_get_data_ptr(dst, main_device);
                const int64_t dst_elements = ne01 * ne11;

                // GPU kernel adds async_result (in shared memory) to dst
                // IMPORTANT: Use submit() to isolate kernel lambda scope from enclosing context.
                main_stream
                    ->submit([&](sycl::handler & cgh) {
                        float * dst_local   = dst_ptr;
                        float * async_local = async_result;
                        cgh.parallel_for(sycl::range<1>(dst_elements),
                                         [=](sycl::id<1> idx) { dst_local[idx] += async_local[idx]; });
                    })
                    .wait();

                // DEBUG: Verify final result
                if (g_ggml_sycl_tp_debug && async_used_count <= 3) {
                    float final_sample[4];
                    main_stream->memcpy(final_sample, dst_ptr, 4 * sizeof(float)).wait();
                    fprintf(stderr, "SYCL TP: ASYNC FFN layer %d FINAL[0..3]=[%f,%f,%f,%f]\n", layer, final_sample[0],
                            final_sample[1], final_sample[2], final_sample[3]);
                }

                // Free the async result buffer
                if (g_ggml_sycl_tp_threaded_ffn && g_tp_device1_worker.initialized.load()) {
                    // Thread-based: release via the proper function
                    ggml_sycl_tp_release_ffn_result(layer);
                } else {
                    // Old async path: free directly
                    ggml_sycl_host_free(async_result);
                }

                // Clear stored FFN input for this layer
                {
                    std::lock_guard<std::mutex> lock(g_tp_ffn_input_mutex);
                    auto                        it = g_tp_ffn_inputs.find(layer);
                    if (it != g_tp_ffn_inputs.end()) {
                        // The input buffer on device 1 needs to be freed
                        int device = g_sycl_tp_config.devices[1];
                        ggml_sycl_set_device(device);
                        queue_ptr dev1_stream = ctx.stream(device, 0);
                        if (it->second.data) {
                            sycl::free(it->second.data, *dev1_stream);
                        }
                        g_tp_ffn_inputs.erase(it);
                        ggml_sycl_set_device(main_device);
                    }
                }

                return;  // Done - used async result
            }

            // SYNC PATH: No async result, compute synchronously
            // DEBUG: Check L31 weight at START of FFN processing for layer 31 (limited)
            static int l31_start_dbg = 0;
            if (g_ggml_sycl_tp_debug && layer == 31 && l31_start_dbg++ < 3) {
                uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
                queue_ptr main_stream     = ctx.stream();

                struct {
                    uint16_t d_bits;
                    uint8_t  qs[16];
                } wblk;

                try {
                    main_stream->memcpy(&wblk, (void *) l31_weight_addr, sizeof(wblk)).wait();
                    uint16_t   d_raw = wblk.d_bits;
                    sycl::half d_half;
                    memcpy(&d_half, &d_raw, sizeof(sycl::half));
                    float d_f = static_cast<float>(d_half);
                    fprintf(stderr, "TP DEBUG FFN_START layer 31: L31 weight d=%f %s\n", d_f,
                            (d_f > 100.0f || std::isnan(d_f)) ? "CORRUPTED" : "OK");
                } catch (...) {
                }
            }
            ffn_input_storage ffn_input = {};
            {
                std::lock_guard<std::mutex> lock(g_tp_ffn_input_mutex);
                auto                        it = g_tp_ffn_inputs.find(layer);
                if (it != g_tp_ffn_inputs.end()) {
                    ffn_input = it->second;
                }
            }

            // DEBUG: Track FFN input lookup during decode (batch=1)
            static int ffn_lookup_dbg = 0;
            if (g_ggml_sycl_tp_debug && ne11 == 1 && ffn_lookup_dbg++ < 40) {
                fprintf(stderr, "TP DEBUG ROW_PARALLEL decode: layer=%d ffn_input.data=%p, map_size=%zu, name=%s\n",
                        layer, ffn_input.data, g_tp_ffn_inputs.size(), name);
            }

            if (ffn_input.data != nullptr) {
                static int ffn_found = 0;
                if (g_ggml_sycl_tp_debug && ffn_found++ < 3) {
                    fprintf(stderr, "TP DEBUG: FFN input found for layer %d, data=%p\n", layer, ffn_input.data);
                }
                // Get FFN weight references for this layer
                ffn_weight_refs weights = {};
                {
                    std::lock_guard<std::mutex> lock(g_tp_ffn_weight_mutex);
                    auto                        it = g_tp_ffn_weights.find(layer);
                    if (it != g_tp_ffn_weights.end()) {
                        weights = it->second;
                    }
                }

                // DEBUG: Check FFN weight lookup during decode (batch=1)
                static int ffn_weight_dbg = 0;
                if (g_ggml_sycl_tp_debug && ne11 == 1 && ffn_weight_dbg++ < 10) {
                    fprintf(stderr, "TP DEBUG ROW_PARALLEL decode FFN WEIGHTS: layer=%d gate=%p up=%p down=%p\n", layer,
                            (void *) weights.gate, (void *) weights.up, (void *) weights.down);
                }

                if (weights.gate && weights.up && weights.down) {
                    // Get device 1
                    int device = g_sycl_tp_config.devices[1];
                    ggml_sycl_set_device(device);
                    queue_ptr stream = ctx.stream(device, 0);

                    // Get weight shards for device 1
                    auto * gate_extra = static_cast<ggml_tensor_extra_gpu *>(weights.gate->extra);
                    auto * up_extra   = static_cast<ggml_tensor_extra_gpu *>(weights.up->extra);
                    auto * down_extra = static_cast<ggml_tensor_extra_gpu *>(weights.down->extra);

                    void * gate_weight_1 = gate_extra ? gate_extra->data_device[device] : nullptr;
                    void * up_weight_1   = up_extra ? up_extra->data_device[device] : nullptr;
                    void * down_weight_1 = down_extra ? down_extra->data_device[device] : nullptr;

                    // DEBUG: Compare device 0 and device 1 weight VALUES (not just pointers)
                    static int weight_dbg = 0;
                    if (g_ggml_sycl_tp_debug && weight_dbg++ < 3) {
                        void * gate_weight_0 = gate_extra ? gate_extra->data_device[main_device] : nullptr;
                        void * down_weight_0 = down_extra ? down_extra->data_device[main_device] : nullptr;
                        fprintf(stderr, "TP DEBUG FFN layer %d weights: gate_0=%p, gate_1=%p, down_0=%p, down_1=%p\n",
                                layer, gate_weight_0, gate_weight_1, down_weight_0, down_weight_1);

                        // Read first Q4_0 block from each device's gate weight to compare values
                        if (gate_weight_0 && gate_weight_1) {
                            uint8_t   block0[20], block1[20];  // Q4_0 block is 18 bytes
                            queue_ptr main_stream = ctx.stream();
                            queue_ptr dev1_stream = ctx.stream(device, 0);
                            main_stream->memcpy(block0, gate_weight_0, 20).wait();
                            dev1_stream->memcpy(block1, gate_weight_1, 20).wait();
                            fprintf(stderr,
                                    "TP DEBUG FFN layer %d gate[0:20]: dev0=[%02x,%02x,%02x,%02x...], "
                                    "dev1=[%02x,%02x,%02x,%02x...]\n",
                                    layer, block0[0], block0[1], block0[2], block0[3], block1[0], block1[1], block1[2],
                                    block1[3]);
                        }
                    }

                    if (!gate_weight_1 || !up_weight_1 || !down_weight_1) {
                        GGML_SYCL_DEBUG("SYCL TP: WARNING - missing weight shards on device 1 for layer %d\n", layer);
                    } else {
                        // Dimensions for FFN computation
                        // FFN input: [K, batch] where K = model dimension (full)
                        // Gate/Up weights: [K, N_hidden_shard] (column-parallel, output sharded)
                        // Down weight: [N_hidden_shard, N_out] (row-parallel, input sharded)
                        const int64_t K_full         = ffn_input.ne0;               // Full model dimension
                        const int64_t batch          = ffn_input.ne1;
                        const int64_t N_hidden_shard = gate_extra->tp_local_ne[1];  // Sharded hidden dim
                        const int64_t N_out          = ne01;  // Output dimension (same as device 0)

                        // Calculate padded dimensions
                        const int64_t K_full_padded         = GGML_PAD(K_full, MATRIX_ROW_PADDING);
                        const int64_t N_hidden_shard_padded = GGML_PAD(N_hidden_shard, MATRIX_ROW_PADDING);

                        // OPTIMIZATION: Use persistent buffers instead of malloc/free per call
                        // This eliminates ~535K memory allocations per 20-token inference
                        tp_ffn_compute_buffers * bufs = ggml_sycl_tp_ensure_ffn_buffers(
                            layer, device, stream, K_full_padded, N_hidden_shard_padded, batch, N_out);

                        if (!bufs) {
                            fprintf(stderr, "SYCL TP: ERROR - failed to get FFN buffers on device %d for layer %d\n",
                                    device, layer);
                        } else {
                            // Use pre-allocated persistent buffers
                            char *  input_q8_dev  = bufs->input_q8_dev;
                            float * gate_out      = bufs->gate_out;
                            float * up_out        = bufs->up_out;
                            float * hidden_out    = bufs->hidden_out;
                            char *  hidden_q8_dev = bufs->hidden_q8_dev;
                            float * partial_out   = bufs->partial_out;

                            // Calculate sizes for operations (using actual dimensions, not buffer capacity)
                            const size_t hidden_size = N_hidden_shard * batch * sizeof(float);
                            // DEBUG: Check FFN input values (always for batch=1 to debug NaN)
                            static int   ffn_in_dbg  = 0;
                            bool         do_debug =
                                (g_ggml_sycl_tp_debug && ffn_in_dbg++ < 3);  // Enable for first 3 to debug quant
                            if (do_debug) {
                                float in_sample[4];
                                stream->memcpy(in_sample, ffn_input.data, 4 * sizeof(float)).wait();
                                fprintf(
                                    stderr,
                                    "TP DEBUG FFN input layer %d: input[0..3]=[%f,%f,%f,%f], K_full=%lld, batch=%lld\n",
                                    layer, in_sample[0], in_sample[1], in_sample[2], in_sample[3], (long long) K_full,
                                    (long long) batch);
                            }

                            // Step 1: Quantize FFN input to Q8_1
                            // Use SoA quantizer if weights are in SoA layout
                            const bool use_soa_ffn_input = gate_extra && gate_extra->optimized_feature.is_reordered();
                            if (use_soa_ffn_input) {
                                quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
                                    (const float *) ffn_input.data, input_q8_dev, K_full, batch, K_full_padded, stream);
                            } else {
                                quantize_row_q8_1_sycl<quantize_q8_1>((const float *) ffn_input.data, input_q8_dev,
                                                                      K_full, batch, K_full_padded, stream);
                            }
                            // Note: No wait needed - same queue operations are serialized

                            // Create fake tensors with correct output dimensions for gate/up
                            // MMVQ uses dst->ne[0] for output stride, so we need correct dims
                            ggml_tensor fake_dst_hidden = *dst;
                            fake_dst_hidden.ne[0]       = N_hidden_shard;  // Output dimension for gate/up

                            // Steps 2-3: Gate and Up matmuls (run sequentially on same queue)
                            // Note: These could be parallelized with separate queues in Phase 4
                            stream->memset(gate_out, 0, hidden_size);
                            stream->memset(up_out, 0, hidden_size);
                            // Wait for memsets to complete before matmuls read from these buffers
                            stream->wait();

                            // Step 2: Gate matmul - input @ W_gate_1 -> gate_out
                            ggml_sycl_op_mul_mat_vec_q(ctx, weights.gate, src1, &fake_dst_hidden,
                                                       (const char *) gate_weight_1, nullptr, input_q8_dev, gate_out, 0,
                                                       N_hidden_shard, batch, K_full_padded, stream);

                            // Step 3: Up matmul - input @ W_up_1 -> up_out
                            ggml_sycl_op_mul_mat_vec_q(ctx, weights.up, src1, &fake_dst_hidden,
                                                       (const char *) up_weight_1, nullptr, input_q8_dev, up_out, 0,
                                                       N_hidden_shard, batch, K_full_padded, stream);
                            // Wait for both matmuls before SiLU
                            stream->wait();

                            // DEBUG: Check gate and up values and sums (disabled - TP working)
                            static int ffn_inter_dbg = 0;
                            bool       debug_this    = (ffn_inter_dbg++ < 0);  // Disabled
                            if (debug_this) {
                                float g_sample[4], u_sample[4];
                                stream->memcpy(g_sample, gate_out, 4 * sizeof(float)).wait();
                                stream->memcpy(u_sample, up_out, 4 * sizeof(float)).wait();
                                // Also compute sum of first 1024 elements
                                std::vector<float> gate_host(1024);
                                stream->memcpy(gate_host.data(), gate_out, 1024 * sizeof(float)).wait();
                                float gate_sum = 0, gate_max = -1e10, gate_min = 1e10;
                                for (int i = 0; i < 1024; i++) {
                                    gate_sum += gate_host[i];
                                    gate_max = std::max(gate_max, gate_host[i]);
                                    gate_min = std::min(gate_min, gate_host[i]);
                                }
                                fprintf(stderr,
                                        "TP DEBUG FFN layer %d batch=%lld: gate[0..3]=[%f,%f,%f,%f], gate_sum=%f, "
                                        "range=[%f,%f]\n",
                                        layer, (long long) batch, g_sample[0], g_sample[1], g_sample[2], g_sample[3],
                                        gate_sum, gate_min, gate_max);
                                fprintf(stderr, "TP DEBUG FFN layer %d batch=%lld: up[0..3]=[%f,%f,%f,%f]\n", layer,
                                        (long long) batch, u_sample[0], u_sample[1], u_sample[2], u_sample[3]);
                            }

                            // Step 4: SiLU activation on gate_out and multiply by up_out
                            // hidden_out[i] = silu(gate_out[i]) * up_out[i]
                            const int64_t n_elements = N_hidden_shard * batch;
                            const int     block_size = 256;
                            const int     num_blocks = (n_elements + block_size - 1) / block_size;
                            stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size),
                                                 [=](sycl::nd_item<1> item) {
                                                     const int i = item.get_global_id(0);
                                                     if (i < n_elements) {
                                                         float g       = gate_out[i];
                                                         float u       = up_out[i];
                                                         // SiLU: x * sigmoid(x)
                                                         float silu_g  = g / (1.0f + sycl::native::exp(-g));
                                                         hidden_out[i] = silu_g * u;
                                                     }
                                                 });
                            // No wait - quantization on same queue will serialize

                            // DEBUG: Check hidden_out values after SiLU (disabled - TP working)
                            static int hidden_dbg   = 0;
                            bool       debug_hidden = (hidden_dbg++ < 0);  // Disabled
                            if (debug_hidden) {
                                float h_sample[4];
                                stream->memcpy(h_sample, hidden_out, 4 * sizeof(float)).wait();
                                fprintf(stderr,
                                        "TP DEBUG FFN layer %d batch=%lld: hidden[0..3]=[%f,%f,%f,%f], "
                                        "N_hidden_shard=%lld\n",
                                        layer, (long long) batch, h_sample[0], h_sample[1], h_sample[2], h_sample[3],
                                        (long long) N_hidden_shard);
                            }

                            // Step 5: Quantize hidden_out to Q8_1 for down matmul
                            // DEBUG: Check hidden sum before quantization (disabled)
                            static int quant_dbg = 0;
                            if (quant_dbg++ < 0) {
                                std::vector<float> hidden_host(1024);
                                stream->memcpy(hidden_host.data(), hidden_out, 1024 * sizeof(float)).wait();
                                float hidden_sum = 0, hidden_max = -1e10, hidden_min = 1e10;
                                for (int i = 0; i < 1024; i++) {
                                    hidden_sum += hidden_host[i];
                                    hidden_max = std::max(hidden_max, hidden_host[i]);
                                    hidden_min = std::min(hidden_min, hidden_host[i]);
                                }
                                fprintf(stderr, "TP DEBUG FFN layer %d: hidden_sum=%f, hidden_range=[%f,%f]\n", layer,
                                        hidden_sum, hidden_min, hidden_max);
                            }

                            // Use SoA quantizer if down weights are reordered
                            const bool use_soa_hidden_ffn = down_extra && down_extra->optimized_feature.is_reordered();
                            if (use_soa_hidden_ffn) {
                                quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
                                    hidden_out, hidden_q8_dev, N_hidden_shard, batch, N_hidden_shard_padded, stream);
                            } else {
                                quantize_row_q8_1_sycl<quantize_q8_1>(hidden_out, hidden_q8_dev, N_hidden_shard, batch,
                                                                      N_hidden_shard_padded, stream);
                            }
                            // No wait - memset+matmul on same queue will serialize

                            // Step 6: Down matmul - hidden_out @ W_down_1 -> partial_out
                            // DEBUG: Check down matmul dimensions (disabled)
                            static int down_dbg = 0;
                            if (down_dbg++ < 0) {
                                fprintf(stderr,
                                        "TP DEBUG FFN DOWN layer %d: src0->ne=[%lld,%lld], N_out=%lld, "
                                        "N_hidden_shard=%lld, batch=%lld\n",
                                        layer, (long long) src0->ne[0], (long long) src0->ne[1], (long long) N_out,
                                        (long long) N_hidden_shard, (long long) batch);
                            }
                            stream->memset(partial_out, 0, dst_size);
                            // Wait for memset before matmul writes to same buffer
                            stream->wait();
                            ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst, (const char *) down_weight_1, nullptr,
                                                       hidden_q8_dev, partial_out, 0, N_out, batch,
                                                       N_hidden_shard_padded, stream);
                            // Wait before ALL_REDUCE reads partial_out
                            stream->wait();

                            // Step 7: ALL_REDUCE - add partial_out to dst on main device
                            // OPTIMIZED: Use malloc_shared buffer + GPU addition kernel
                            {
                                const size_t dst_elements = (size_t) (ne01 * ne11);

                                // Get shared buffer for ALL_REDUCE (accessible from both devices)
                                float * shared_buf = ggml_sycl_tp_ensure_shared_reduce_buffer(dst_size);

                                static int ffn_reduce_dbg = 0;
                                bool       debug_reduce   = (g_ggml_sycl_tp_debug && ffn_reduce_dbg++ < 150);

                                // WORKAROUND: OPTIMIZED path disabled due to UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
                                // when kernel on device 0 accesses malloc_shared buffer from device 1.
                                // TODO: Investigate cross-device kernel argument issue on Intel Arc.
                                // See TENSOR_PARALLELISM_PLAN.md for details.
                                if (false && shared_buf != nullptr) {
                                    // OPTIMIZED PATH: Use shared memory + GPU kernel
                                    // Step 7a: Copy device 1's partial result to shared buffer
                                    stream->memcpy(shared_buf, partial_out, dst_size).wait();

                                    // DEBUG: Check partial output values
                                    if (debug_reduce) {
                                        fprintf(stderr,
                                                "TP DEBUG FFN ALL_REDUCE layer %d batch=%lld: "
                                                "dev1_partial[0..3]=[%f,%f,%f,%f]\n",
                                                layer, (long long) batch, shared_buf[0], shared_buf[1], shared_buf[2],
                                                shared_buf[3]);
                                    }

                                    // Step 7b: Switch to main device and add using GPU kernel
                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    // Use ggml_sycl_get_data_ptr for correct device-specific pointer in TP mode
                                    float *   dst_ptr     = (float *) ggml_sycl_get_data_ptr(dst, main_device);

                                    // DEBUG: Check device 0's partial result before add
                                    if (debug_reduce) {
                                        float dev0_sample[4];
                                        main_stream->memcpy(dev0_sample, dst_ptr, 4 * sizeof(float)).wait();
                                        fprintf(stderr,
                                                "TP DEBUG FFN layer %d batch=%lld: dev0_partial[0..3]=[%f,%f,%f,%f]\n",
                                                layer, (long long) batch, dev0_sample[0], dev0_sample[1],
                                                dev0_sample[2], dev0_sample[3]);
                                    }

                                    // Step 7c: GPU kernel adds shared_buf to dst (dst += shared_buf)
                                    // shared_buf is malloc_shared so device 0 can read it directly
                                    // Use parallel_for_work_group for better compatibility with Intel Arc
                                    const size_t work_group_size = 256;
                                    const size_t num_groups = (dst_elements + work_group_size - 1) / work_group_size;
                                    main_stream
                                        ->parallel_for(sycl::nd_range<1>(num_groups * work_group_size, work_group_size),
                                                       [=](sycl::nd_item<1> item) {
                                                           size_t idx = item.get_global_id(0);
                                                           if (idx < dst_elements) {
                                                               dst_ptr[idx] += shared_buf[idx];
                                                           }
                                                       })
                                        .wait();

                                    // DEBUG: Verify result
                                    if (debug_reduce && layer < 2) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4 * sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG FFN layer %d: TOTAL[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                layer, total_sample[0], total_sample[1], total_sample[2],
                                                total_sample[3]);
                                    }
                                } else if (ggml_sycl_should_use_quant_allreduce(dst_elements)) {
                                    // QUANTIZED ALL_REDUCE: 33% bandwidth reduction (INT16)
                                    // Only used for large tensors where bandwidth savings outweigh kernel overhead
                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    float *   dst_ptr     = (float *) ggml_sycl_get_data_ptr(dst, main_device);

                                    quantized_allreduce(main_device, device, main_stream, stream, dst_ptr, partial_out,
                                                        dst_elements, debug_reduce);

                                    if (debug_reduce) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4 * sizeof(float)).wait();
                                        fprintf(stderr,
                                                "TP DEBUG FFN layer %d (QUANT): TOTAL[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
                                                layer, total_sample[0], total_sample[1], total_sample[2],
                                                total_sample[3]);
                                    }
                                } else {
                                    // Host-staged ALL_REDUCE (CPU addition) with persistent buffers
                                    // Uses pre-allocated host buffers to avoid per-call malloc/free overhead
                                    // OPTIMIZED: Reduced from 3 sync points to 2 by overlapping copies
                                    float *dev0_host, *host_buf;
                                    ggml_sycl_tp_get_host_reduce_buffers(dst_size, &dev0_host, &host_buf);

                                    // Start both copies in parallel (they're independent)
                                    auto ev_dev1 = stream->memcpy(host_buf, partial_out, dst_size);

                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    float *   dst_ptr     = (float *) ggml_sycl_get_data_ptr(dst, main_device);

                                    auto ev_dev0 = main_stream->memcpy(dev0_host, dst_ptr, dst_size);

                                    // Single sync point for both copies
                                    ev_dev1.wait();
                                    ev_dev0.wait();

                                    // DEBUG: Show partial results before addition
                                    if (debug_reduce) {
                                        fprintf(stderr,
                                                "TP DEBUG FFN layer %d batch=%lld: dev0[0..3]=[%.4f,%.4f,%.4f,%.4f] "
                                                "dev1[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
                                                layer, (long long) batch, dev0_host[0], dev0_host[1], dev0_host[2],
                                                dev0_host[3], host_buf[0], host_buf[1], host_buf[2], host_buf[3]);
                                    }

                                    for (size_t i = 0; i < dst_elements; i++) {
                                        dev0_host[i] += host_buf[i];
                                    }

                                    // DEBUG: Verify result
                                    if (debug_reduce) {
                                        fprintf(stderr, "TP DEBUG FFN layer %d: TOTAL[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
                                                layer, dev0_host[0], dev0_host[1], dev0_host[2], dev0_host[3]);
                                    }

                                    main_stream->memcpy(dst_ptr, dev0_host, dst_size).wait();
                                    // No free needed - buffers are persistent
                                }
                            }

                            // OPTIMIZATION: No cleanup needed - using persistent buffers
                            // Buffers are pre-allocated in ggml_sycl_tp_ensure_ffn_buffers()
                            // and freed only at program exit via ggml_sycl_tp_free_ffn_buffers()

                            // DEBUG: Check if L31 FFN processing corrupted the weight
                            if (layer == 31) {
                                ggml_sycl_set_device(main_device);
                                queue_ptr main_stream     = ctx.stream();
                                uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;

                                struct {
                                    uint16_t d_bits;
                                    uint8_t  qs[16];
                                } wblk;

                                try {
                                    main_stream->memcpy(&wblk, (void *) l31_weight_addr, sizeof(wblk)).wait();
                                    uint16_t   d_raw = wblk.d_bits;
                                    sycl::half d_half;
                                    memcpy(&d_half, &d_raw, sizeof(sycl::half));
                                    float d_f = static_cast<float>(d_half);
                                    if (g_ggml_sycl_tp_debug && (d_f > 100.0f || std::isnan(d_f))) {
                                        fprintf(stderr, "TP DEBUG FFN_CLEANUP layer 31: L31 weight CORRUPTED d=%f\n",
                                                d_f);
                                    } else if (g_ggml_sycl_tp_debug) {
                                        static int cleanup_dbg = 0;
                                        if (cleanup_dbg++ < 3) {
                                            fprintf(stderr, "TP DEBUG FFN_CLEANUP layer 31: L31 weight OK d=%f\n", d_f);
                                        }
                                    }
                                } catch (...) {
                                }
                            }
                        }
                    }

                    // Restore main device context
                    ggml_sycl_set_device(main_device);
                } else {
                    static int warn = 0;
                    if (warn++ < 3) {
                        fprintf(stderr,
                                "SYCL TP: WARNING - missing FFN weight refs for layer %d (gate=%p, up=%p, down=%p)\n",
                                layer, (void *) weights.gate, (void *) weights.up, (void *) weights.down);
                    }
                }

                // Clear stored FFN input for this layer (no longer needed)
                {
                    std::lock_guard<std::mutex> lock(g_tp_ffn_input_mutex);
                    auto                        it = g_tp_ffn_inputs.find(layer);
                    if (it != g_tp_ffn_inputs.end()) {
                        if (it->second.data) {
                            int device = g_sycl_tp_config.devices[1];
                            ggml_sycl_set_device(device);
                            queue_ptr stream = ctx.stream(device, 0);
                            sycl::free(it->second.data, *stream);
                            ggml_sycl_set_device(main_device);
                        }
                        g_tp_ffn_inputs.erase(it);
                    }
                }

                // Return - we've handled the ALL_REDUCE already
                return;
            } else {
                static int warn_count = 0;
                if (warn_count++ < 3) {
                    fprintf(stderr, "SYCL TP: WARNING - no stored FFN input for layer %d\n", layer);
                }
            }
        }

        // Check for attn_output row-parallel
        bool is_attn_output = name && strstr(name, "attn_output");

        if (is_attn_output && layer >= 0) {
            // Try to retrieve stored attention input
            static int attn_dbg_count = 0;
            if (g_ggml_sycl_tp_debug && attn_dbg_count++ < 40) {
                fprintf(stderr, "TP DEBUG: Attention output layer %d, entering computation path (batch=%lld)\n", layer,
                        (long long) ne11);
            }
            attn_input_storage attn_input = {};
            {
                std::lock_guard<std::mutex> lock(g_tp_attn_input_mutex);
                auto                        it = g_tp_attn_inputs.find(layer);
                if (it != g_tp_attn_inputs.end()) {
                    attn_input = it->second;
                }
            }

            // DEBUG: Track attention input lookup during decode (batch=1)
            static int attn_lookup_dbg = 0;
            if (g_ggml_sycl_tp_debug && ne11 == 1 && attn_lookup_dbg++ < 100) {
                fprintf(stderr, "TP DEBUG ATTN_OUTPUT decode: layer=%d attn_input.data=%p, map_size=%zu\n", layer,
                        attn_input.data, g_tp_attn_inputs.size());
            }

            if (attn_input.data != nullptr) {
                static int attn_found = 0;
                if (g_ggml_sycl_tp_debug && attn_found++ < 40) {
                    fprintf(stderr, "TP DEBUG: Attention input found for layer %d, data=%p\n", layer, attn_input.data);
                }
                // Get attention weight references
                attn_weight_refs attn_weights = {};
                {
                    std::lock_guard<std::mutex> lock(g_tp_attn_weight_mutex);
                    auto                        it = g_tp_attn_weights.find(layer);
                    if (it != g_tp_attn_weights.end()) {
                        attn_weights = it->second;
                    }
                }

                if (attn_weights.q && attn_weights.k && attn_weights.v && attn_weights.o) {
                    // Attention TP: compute device 1's contribution and ALL_REDUCE
                    // NOTE: Flash attention runs independently on each device's heads
                    // Then O projection is ALL_REDUCED

                    // Detect new sequence and reset KV cache if needed
                    // If layer 0 and batch > 1 (prompt processing) and cache already has content,
                    // this indicates a new prompt - reset all layer caches
                    if (layer == 0 && attn_input.ne1 > 1) {
                        int64_t cached_len = get_dev1_kv_cache_seq_len(0);
                        if (cached_len > 0) {
                            static int reset_count = 0;
                            if (g_ggml_sycl_tp_debug && reset_count++ < 5) {
                                fprintf(stderr,
                                        "TP DEBUG: New prompt detected (layer=0, batch=%ld, cached=%ld), resetting KV "
                                        "cache\n",
                                        (long) attn_input.ne1, (long) cached_len);
                            }
                            reset_dev1_kv_cache();
                        }
                    }

                    // Get device 1
                    int device = g_sycl_tp_config.devices[1];
                    ggml_sycl_set_device(device);
                    queue_ptr stream = ctx.stream(device, 0);

                    // Get weight shards for device 1
                    auto * q_extra = static_cast<ggml_tensor_extra_gpu *>(attn_weights.q->extra);
                    auto * k_extra = static_cast<ggml_tensor_extra_gpu *>(attn_weights.k->extra);
                    auto * v_extra = static_cast<ggml_tensor_extra_gpu *>(attn_weights.v->extra);
                    auto * o_extra = static_cast<ggml_tensor_extra_gpu *>(attn_weights.o->extra);

                    void * q_weight_1 = q_extra ? q_extra->data_device[device] : nullptr;
                    void * k_weight_1 = k_extra ? k_extra->data_device[device] : nullptr;
                    void * v_weight_1 = v_extra ? v_extra->data_device[device] : nullptr;
                    void * o_weight_1 = o_extra ? o_extra->data_device[device] : nullptr;

                    if (!q_weight_1 || !k_weight_1 || !v_weight_1 || !o_weight_1) {
                        static int warn = 0;
                        if (warn++ < 3) {
                            fprintf(stderr,
                                    "SYCL TP: WARNING - missing attention weight shards on device 1 for layer %d\n",
                                    layer);
                        }
                    } else {
                        // DEBUG: Compare O weight values on both devices
                        static int o_weight_dbg = 0;
                        if (g_ggml_sycl_tp_debug && o_weight_dbg++ < 3) {
                            void * o_weight_0 = o_extra ? o_extra->data_device[main_device] : nullptr;
                            fprintf(stderr, "TP DEBUG ATTN layer %d: O_weight_0=%p, O_weight_1=%p\n", layer, o_weight_0,
                                    o_weight_1);
                            // Read first few Q4_0 blocks and dequantize to check values
                            if (o_weight_0 && o_weight_1) {
                                // Q4_0 block: 2-byte scale (float16) + 16 bytes of quantized values (32 values, 4 bits each)
                                uint8_t   blocks0[36], blocks1[36];  // Read 2 blocks (18 bytes each)
                                queue_ptr main_stream = ctx.stream();
                                queue_ptr dev1_stream = ctx.stream(device, 0);
                                main_stream->memcpy(blocks0, o_weight_0, 36).wait();
                                dev1_stream->memcpy(blocks1, o_weight_1, 36).wait();

                                // Dequantize first block of each
                                uint16_t scale0_bits = blocks0[0] | (blocks0[1] << 8);
                                uint16_t scale1_bits = blocks1[0] | (blocks1[1] << 8);
                                // Convert float16 to float32 (simple approximation)
                                auto     f16_to_f32  = [](uint16_t h) -> float {
                                    uint32_t sign = (h >> 15) & 1;
                                    int32_t  exp  = (h >> 10) & 0x1F;
                                    uint32_t mant = h & 0x3FF;
                                    if (exp == 0) {
                                        return sign ? -0.0f : 0.0f;          // Zero/subnormal
                                    } else if (exp == 31) {
                                        return sign ? -INFINITY : INFINITY;  // Inf/NaN
                                    }
                                    float val = ldexpf(1.0f + mant / 1024.0f, exp - 15);
                                    return sign ? -val : val;
                                };
                                float scale0 = f16_to_f32(scale0_bits);
                                float scale1 = f16_to_f32(scale1_bits);

                                // Dequantize first 4 values from each block
                                auto dequant = [](float scale, uint8_t q) -> std::pair<float, float> {
                                    int q0 = q & 0xF;
                                    int q1 = (q >> 4) & 0xF;
                                    return { (q0 - 8) * scale, (q1 - 8) * scale };
                                };
                                auto [v00, v01] = dequant(scale0, blocks0[2]);
                                auto [v10, v11] = dequant(scale1, blocks1[2]);

                                fprintf(stderr,
                                        "TP DEBUG ATTN layer %d O weight: dev0 scale=%f, vals=[%f,%f,...], dev1 "
                                        "scale=%f, vals=[%f,%f,...]\n",
                                        layer, scale0, v00, v01, scale1, v10, v11);
                            }
                        }
                        // Dimensions for attention computation
                        // Input: [n_embd, batch] where n_embd = model dimension (full)
                        // Q weight: [n_embd, n_embd_q_shard] (column-parallel, output sharded by heads)
                        // K weight: [n_embd, n_embd_k_shard]
                        // V weight: [n_embd, n_embd_v_shard]
                        // O weight: [n_embd_q_shard, n_embd] (row-parallel, input sharded)
                        const int64_t n_embd         = attn_input.ne0;           // Full model dimension
                        const int64_t batch          = attn_input.ne1;           // Sequence length * batch
                        const int64_t n_embd_q_shard = q_extra->tp_local_ne[1];  // Q sharded output dim
                        const int64_t n_embd_k_shard = k_extra->tp_local_ne[1];  // K sharded output dim
                        const int64_t n_embd_v_shard = v_extra->tp_local_ne[1];  // V sharded output dim
                        const int64_t N_out          = ne01;                     // Output dimension (n_embd)

                        // Allocate buffers on device 1
                        const size_t  q8_1_ts       = sizeof(block_q8_1);
                        const size_t  q8_1_bs       = QK8_1;
                        const int64_t n_embd_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);

                        // Input quantization buffer
                        const size_t input_q8_size = batch * n_embd_padded * q8_1_ts / q8_1_bs;
                        char *       input_q8_dev  = (char *) sycl::malloc_device(input_q8_size, *stream);

                        // Q, K, V output buffers (float)
                        const size_t q_out_size = n_embd_q_shard * batch * sizeof(float);
                        const size_t k_out_size = n_embd_k_shard * batch * sizeof(float);
                        const size_t v_out_size = n_embd_v_shard * batch * sizeof(float);
                        float *      q_out      = (float *) sycl::malloc_device(q_out_size, *stream);
                        float *      k_out      = (float *) sycl::malloc_device(k_out_size, *stream);
                        float *      v_out      = (float *) sycl::malloc_device(v_out_size, *stream);

                        // Attention output buffer (same size as Q since it's the per-head output)
                        float * attn_out = (float *) sycl::malloc_device(q_out_size, *stream);

                        // For O projection, need to quantize attn_out
                        const int64_t n_embd_q_shard_padded = GGML_PAD(n_embd_q_shard, MATRIX_ROW_PADDING);
                        const size_t  attn_q8_size          = batch * n_embd_q_shard_padded * q8_1_ts / q8_1_bs;
                        char *        attn_q8_dev           = (char *) sycl::malloc_device(attn_q8_size, *stream);

                        // Output buffer for O projection (partial result)
                        float * partial_out = (float *) sycl::malloc_device(dst_size, *stream);

                        if (!input_q8_dev || !q_out || !k_out || !v_out || !attn_out || !attn_q8_dev || !partial_out) {
                            fprintf(stderr, "SYCL TP: ERROR - failed to allocate attention buffers on device %d\n",
                                    device);
                            if (input_q8_dev) {
                                sycl::free(input_q8_dev, *stream);
                            }
                            if (q_out) {
                                sycl::free(q_out, *stream);
                            }
                            if (k_out) {
                                sycl::free(k_out, *stream);
                            }
                            if (v_out) {
                                sycl::free(v_out, *stream);
                            }
                            if (attn_out) {
                                sycl::free(attn_out, *stream);
                            }
                            if (attn_q8_dev) {
                                sycl::free(attn_q8_dev, *stream);
                            }
                            if (partial_out) {
                                sycl::free(partial_out, *stream);
                            }
                        } else {
                            // Step 1: Quantize attention input to Q8_1
                            // Use SoA quantizer if Q weight is reordered (Q/K/V share same quantization format)
                            const bool use_soa_attn_input = q_extra && q_extra->optimized_feature.is_reordered();
                            if (use_soa_attn_input) {
                                quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) attn_input.data,
                                                                                      input_q8_dev, n_embd, batch,
                                                                                      n_embd_padded, stream);
                            } else {
                                quantize_row_q8_1_sycl<quantize_q8_1>((const float *) attn_input.data, input_q8_dev,
                                                                      n_embd, batch, n_embd_padded, stream);
                            }
                            stream->wait();

                            // Create temporary tensors with correct dimensions for MMVQ
                            // MMVQ uses dst->ne[0] for output stride, so we need correct dimensions
                            ggml_tensor fake_src1 = *src1;
                            fake_src1.ne[0]       = n_embd;  // Input dimension for Q/K/V projection

                            ggml_tensor fake_dst_q = *dst;
                            fake_dst_q.ne[0]       = n_embd_q_shard;  // Q output dimension

                            ggml_tensor fake_dst_k = *dst;
                            fake_dst_k.ne[0]       = n_embd_k_shard;  // K output dimension

                            ggml_tensor fake_dst_v = *dst;
                            fake_dst_v.ne[0]       = n_embd_v_shard;  // V output dimension

                            // Step 2: Q projection - input @ W_q1 -> q_out
                            stream->memset(q_out, 0, q_out_size);
                            ggml_sycl_op_mul_mat_vec_q(ctx, attn_weights.q, &fake_src1, &fake_dst_q,
                                                       (const char *) q_weight_1, nullptr, input_q8_dev, q_out, 0,
                                                       n_embd_q_shard, batch, n_embd_padded, stream);

                            // DEBUG: Capture Q values before RoPE for comparison with multi-process
                            static int q_before_rope_dbg = 0;
                            if (g_ggml_sycl_tp_debug && layer == 0 && q_before_rope_dbg++ < 3) {
                                float q_sample[8];
                                stream->memcpy(q_sample, q_out, 8 * sizeof(float)).wait();
                                int64_t rope_pos = get_dev1_kv_cache_seq_len(layer);
                                fprintf(stderr,
                                        "SP DEV1 L0: Q_before_rope[0..7]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f] "
                                        "rope_pos=%lld batch=%lld\n",
                                        q_sample[0], q_sample[1], q_sample[2], q_sample[3], q_sample[4], q_sample[5],
                                        q_sample[6], q_sample[7], (long long) rope_pos, (long long) batch);
                            }

                            // Step 3: K projection - input @ W_k1 -> k_out
                            stream->memset(k_out, 0, k_out_size);
                            ggml_sycl_op_mul_mat_vec_q(ctx, attn_weights.k, &fake_src1, &fake_dst_k,
                                                       (const char *) k_weight_1, nullptr, input_q8_dev, k_out, 0,
                                                       n_embd_k_shard, batch, n_embd_padded, stream);

                            // Step 4: V projection - input @ W_v1 -> v_out
                            stream->memset(v_out, 0, v_out_size);
                            ggml_sycl_op_mul_mat_vec_q(ctx, attn_weights.v, &fake_src1, &fake_dst_v,
                                                       (const char *) v_weight_1, nullptr, input_q8_dev, v_out, 0,
                                                       n_embd_v_shard, batch, n_embd_padded, stream);

                            // Step 4.5: Apply RoPE to Q and K
                            // Mistral uses NeoX-style RoPE with freq_base=10000.0, head_dim=128
                            const int64_t head_dim    = 128;                        // Mistral uses 128
                            const int64_t n_heads_q   = n_embd_q_shard / head_dim;  // 16 on each device
                            const int64_t n_heads_kv  = n_embd_k_shard / head_dim;  // 4 on each device
                            const float   freq_base   = 10000.0f;
                            const float   theta_scale = std::pow(freq_base, -2.0f / head_dim);

                            // Apply RoPE to Q (norm style: pairs adjacent elements)
                            // Q layout: [seq_len, n_heads_q * head_dim]
                            // NOTE: For token generation, position must account for cached sequence
                            int64_t q_cached_pos = get_dev1_kv_cache_seq_len(layer);
                            stream->parallel_for(sycl::range<3>(n_heads_q, batch, head_dim / 2), [=](sycl::id<3> idx) {
                                const int64_t h   = idx[0];  // Head index
                                const int64_t pos = idx[1];  // Position within current batch
                                const int64_t i0  = idx[2];  // Dimension pair index (0 to head_dim/2-1)

                                // Norm style: pair adjacent elements i0*2 and i0*2+1
                                const int64_t base_idx = pos * n_heads_q * head_dim + h * head_dim;
                                // Absolute position in sequence = cached tokens + current position
                                const int64_t abs_pos  = q_cached_pos + pos;

                                float theta     = abs_pos * std::pow(theta_scale, static_cast<float>(i0));
                                float cos_theta = sycl::cos(theta);
                                float sin_theta = sycl::sin(theta);

                                // Norm style: pairs (0,1), (2,3), (4,5), etc.
                                float x0 = q_out[base_idx + i0 * 2];
                                float x1 = q_out[base_idx + i0 * 2 + 1];

                                q_out[base_idx + i0 * 2]     = x0 * cos_theta - x1 * sin_theta;
                                q_out[base_idx + i0 * 2 + 1] = x0 * sin_theta + x1 * cos_theta;
                            });

                            // DEBUG: Capture Q values AFTER manual RoPE for single-process comparison
                            static int sp_post_rope_dbg = 0;
                            if (g_ggml_sycl_tp_debug && layer == 0 && sp_post_rope_dbg++ < 3) {
                                float q_after[8];
                                stream->memcpy(q_after, q_out, 8 * sizeof(float)).wait();
                                fprintf(stderr,
                                        "SP DEV1 Q_AFTER_ROPE layer 0: [%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f] "
                                        "abs_pos=%lld\n",
                                        q_after[0], q_after[1], q_after[2], q_after[3], q_after[4], q_after[5],
                                        q_after[6], q_after[7], (long long) q_cached_pos);
                            }

                            // Apply RoPE to K (norm style: pairs adjacent elements)
                            // K layout: [seq_len, n_heads_kv * head_dim]
                            // NOTE: For token generation (batch=1), the position needs to account for cached tokens
                            int64_t cached_pos = get_dev1_kv_cache_seq_len(layer);
                            stream
                                ->parallel_for(
                                    sycl::range<3>(n_heads_kv, batch, head_dim / 2),
                                    [=](sycl::id<3> idx) {
                                        const int64_t h   = idx[0];  // Head index
                                        const int64_t pos = idx[1];  // Position within current batch
                                        const int64_t i0  = idx[2];  // Dimension pair index (0 to head_dim/2-1)

                                        const int64_t base_idx = pos * n_heads_kv * head_dim + h * head_dim;
                                        // Absolute position in sequence = cached tokens + current position
                                        const int64_t abs_pos  = cached_pos + pos;

                                        float theta     = abs_pos * std::pow(theta_scale, static_cast<float>(i0));
                                        float cos_theta = sycl::cos(theta);
                                        float sin_theta = sycl::sin(theta);

                                        // Norm style: pairs (0,1), (2,3), (4,5), etc.
                                        float x0 = k_out[base_idx + i0 * 2];
                                        float x1 = k_out[base_idx + i0 * 2 + 1];

                                        k_out[base_idx + i0 * 2]     = x0 * cos_theta - x1 * sin_theta;
                                        k_out[base_idx + i0 * 2 + 1] = x0 * sin_theta + x1 * cos_theta;
                                    })
                                .wait();

                            // Step 4.6: Update KV cache with new K and V values (after RoPE)
                            // Initialize cache if needed (during first call / prompt processing)
                            init_dev1_kv_cache(layer, 4096, n_heads_kv, head_dim, stream);
                            // Append the new K and V values to cache
                            append_to_dev1_kv_cache(layer, k_out, v_out, batch, stream);

                            // Step 5: Multi-head attention computation WITH KV CACHE
                            // Q: [batch, n_heads_q * head_dim] - current queries (batch tokens)
                            // KV Cache: [full_seq_len, n_heads_kv * head_dim] - all previous + current tokens
                            // For GQA: gqa_ratio = n_heads_q / n_heads_kv Q heads share each KV head

                            const int64_t gqa_ratio      = n_heads_q / n_heads_kv;  // 4
                            const int64_t n_query_tokens = batch;                   // Number of new query tokens
                            const float   scale          = 1.0f / std::sqrt(static_cast<float>(head_dim));

                            // Get cached K/V for full sequence attention
                            float * k_cache    = nullptr;
                            float * v_cache    = nullptr;
                            int64_t kv_seq_len = 0;
                            get_dev1_kv_cache_ptrs(layer, &k_cache, &v_cache, &kv_seq_len);

                            // kv_seq_len now includes the tokens we just appended
                            // Query positions are: [kv_seq_len - n_query_tokens, kv_seq_len)
                            const int64_t q_start_pos = kv_seq_len - n_query_tokens;

                            // Debug output disabled for cleaner output
                            // static int kv_cache_dbg = 0;
                            // if (kv_cache_dbg++ < 3) {
                            //     fprintf(stderr, "TP DEBUG ATTN layer %d: n_query=%lld, kv_seq_len=%lld, q_start=%lld, k_cache=%p\n",
                            //             layer, (long long)n_query_tokens, (long long)kv_seq_len,
                            //             (long long)q_start_pos, (void*)k_cache);
                            // }

                            // Allocate attention scores buffer [n_heads_q, n_query_tokens, kv_seq_len]
                            const size_t scores_size = n_heads_q * n_query_tokens * kv_seq_len * sizeof(float);
                            float *      attn_scores = (float *) sycl::malloc_device(scores_size, *stream);

                            if (attn_scores && k_cache && v_cache && kv_seq_len > 0) {
                                // Compute attention scores: Q @ K^T / sqrt(head_dim) with GQA
                                // Q comes from q_out (current batch), K comes from cache (full sequence)
                                stream
                                    ->parallel_for(
                                        sycl::range<3>(n_heads_q, n_query_tokens, kv_seq_len),
                                        [=](sycl::id<3> idx) {
                                            const int64_t h       = idx[0];  // Q head index
                                            const int64_t q_local = idx[1];  // Query local position (0..batch-1)
                                            const int64_t k_pos   = idx[2];  // Key position (0..kv_seq_len-1)

                                            // Map Q head to KV head for GQA
                                            const int64_t kv_h = h / gqa_ratio;

                                            // Compute dot product Q_h[q_local] @ K_kv[k_pos]
                                            float score = 0.0f;
                                            for (int64_t d = 0; d < head_dim; d++) {
                                                // Q layout: [n_query_tokens, n_heads_q * head_dim]
                                                // K cache layout: [kv_seq_len, n_heads_kv * head_dim]
                                                const float q_val =
                                                    q_out[q_local * n_heads_q * head_dim + h * head_dim + d];
                                                const float k_val =
                                                    k_cache[k_pos * n_heads_kv * head_dim + kv_h * head_dim + d];
                                                score += q_val * k_val;
                                            }
                                            score *= scale;

                                            // Apply causal mask: query at absolute position (q_start_pos + q_local)
                                            // can only attend to key positions <= that absolute position
                                            const int64_t q_abs_pos = q_start_pos + q_local;
                                            if (k_pos > q_abs_pos) {
                                                score = -INFINITY;
                                            }

                                            attn_scores[h * n_query_tokens * kv_seq_len + q_local * kv_seq_len +
                                                        k_pos] = score;
                                        })
                                    .wait();

                                // Softmax over key dimension for each (head, query_pos)
                                stream
                                    ->parallel_for(
                                        sycl::range<2>(n_heads_q, n_query_tokens),
                                        [=](sycl::id<2> idx) {
                                            const int64_t h       = idx[0];
                                            const int64_t q_local = idx[1];
                                            const int64_t base = h * n_query_tokens * kv_seq_len + q_local * kv_seq_len;

                                            // Find max for numerical stability
                                            float max_val = -INFINITY;
                                            for (int64_t k = 0; k < kv_seq_len; k++) {
                                                max_val = sycl::fmax(max_val, attn_scores[base + k]);
                                            }

                                            // Compute exp and sum
                                            float sum = 0.0f;
                                            for (int64_t k = 0; k < kv_seq_len; k++) {
                                                const float exp_val   = sycl::exp(attn_scores[base + k] - max_val);
                                                attn_scores[base + k] = exp_val;
                                                sum += exp_val;
                                            }

                                            // Normalize
                                            const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                                            for (int64_t k = 0; k < kv_seq_len; k++) {
                                                attn_scores[base + k] *= inv_sum;
                                            }
                                        })
                                    .wait();

                                // Compute attention output: attn_probs @ V
                                // attn_out layout: [n_query_tokens, n_heads_q * head_dim]
                                // V comes from cache (full sequence)
                                stream
                                    ->parallel_for(
                                        sycl::range<3>(n_heads_q, n_query_tokens, head_dim),
                                        [=](sycl::id<3> idx) {
                                            const int64_t h       = idx[0];  // Q head index
                                            const int64_t q_local = idx[1];  // Output position (local)
                                            const int64_t d       = idx[2];  // Head dimension

                                            // Map Q head to KV head for GQA
                                            const int64_t kv_h = h / gqa_ratio;

                                            // Weighted sum of V values from cache
                                            float out_val = 0.0f;
                                            for (int64_t k_pos = 0; k_pos < kv_seq_len; k_pos++) {
                                                const float attn_weight = attn_scores[h * n_query_tokens * kv_seq_len +
                                                                                      q_local * kv_seq_len + k_pos];
                                                const float v_val =
                                                    v_cache[k_pos * n_heads_kv * head_dim + kv_h * head_dim + d];
                                                out_val += attn_weight * v_val;
                                            }

                                            attn_out[q_local * n_heads_q * head_dim + h * head_dim + d] = out_val;
                                        })
                                    .wait();

                                sycl::free(attn_scores, *stream);

                                // DEBUG: Check device 1's attention output
                                static int dev1_attn_dbg = 0;
                                if (g_ggml_sycl_tp_debug && dev1_attn_dbg++ < 3) {
                                    float attn_sample[4];
                                    stream->memcpy(attn_sample, attn_out, 4 * sizeof(float)).wait();
                                    fprintf(stderr, "TP DEBUG ATTN dev1 attn_out[0..3] layer %d: [%f, %f, %f, %f]\n",
                                            layer, attn_sample[0], attn_sample[1], attn_sample[2], attn_sample[3]);
                                }
                            } else {
                                // Fallback: just copy Q output if allocation fails
                                fprintf(stderr,
                                        "SYCL TP: WARNING - attention scores allocation failed, using fallback\n");
                                stream->memcpy(attn_out, q_out, q_out_size).wait();
                            }

                            // Step 6: Quantize attention output for O projection
                            // Use SoA quantizer if O weight is reordered
                            const bool use_soa_attn_output = o_extra && o_extra->optimized_feature.is_reordered();
                            if (use_soa_attn_output) {
                                quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
                                    attn_out, attn_q8_dev, n_embd_q_shard, batch, n_embd_q_shard_padded, stream);
                            } else {
                                quantize_row_q8_1_sycl<quantize_q8_1>(attn_out, attn_q8_dev, n_embd_q_shard, batch,
                                                                      n_embd_q_shard_padded, stream);
                            }
                            stream->wait();

                            // Step 7: O projection - attn_out @ W_o1 -> partial_out
                            stream->memset(partial_out, 0, dst_size);
                            ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst, (const char *) o_weight_1, nullptr,
                                                       attn_q8_dev, partial_out, 0, N_out, batch, n_embd_q_shard_padded,
                                                       stream);

                            // Step 8: ALL_REDUCE - add partial_out to dst on main device
                            // OPTIMIZED: Use malloc_shared buffer + GPU addition kernel
                            {
                                const size_t dst_elements = (size_t) (ne01 * ne11);

                                // Get shared buffer for ALL_REDUCE (accessible from both devices)
                                float * shared_buf = ggml_sycl_tp_ensure_shared_reduce_buffer(dst_size);

                                static int attn_reduce_dbg = 0;
                                bool       do_attn_dbg     = (g_ggml_sycl_tp_debug && attn_reduce_dbg++ < 3);

                                // WORKAROUND: OPTIMIZED path disabled due to UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
                                // when kernel on device 0 accesses malloc_shared buffer from device 1.
                                // TODO: Investigate cross-device kernel argument issue on Intel Arc.
                                // See TENSOR_PARALLELISM_PLAN.md for details.
                                if (false && shared_buf != nullptr) {
                                    // OPTIMIZED PATH: Use shared memory + GPU kernel
                                    // Step 8a: Copy device 1's partial result to shared buffer
                                    stream->memcpy(shared_buf, partial_out, dst_size).wait();

                                    // DEBUG: Check partial output values
                                    if (do_attn_dbg) {
                                        fprintf(stderr,
                                                "TP DEBUG ATTN ALL_REDUCE layer %d batch=%lld: "
                                                "partial_out[0..3]=[%f,%f,%f,%f]\n",
                                                layer, (long long) batch, shared_buf[0], shared_buf[1], shared_buf[2],
                                                shared_buf[3]);
                                    }

                                    // Step 8b: Switch to main device and add using GPU kernel
                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    // Use ggml_sycl_get_data_ptr for correct device-specific pointer in TP mode
                                    float *   dst_ptr     = (float *) ggml_sycl_get_data_ptr(dst, main_device);

                                    // DEBUG: Check device 0's partial result before add
                                    if (do_attn_dbg) {
                                        float dev0_sample[4];
                                        main_stream->memcpy(dev0_sample, dst_ptr, 4 * sizeof(float)).wait();
                                        fprintf(stderr,
                                                "TP DEBUG ATTN layer %d batch=%lld: dev0_out[0..3]=[%f,%f,%f,%f]\n",
                                                layer, (long long) batch, dev0_sample[0], dev0_sample[1],
                                                dev0_sample[2], dev0_sample[3]);
                                    }

                                    // Step 8c: GPU kernel adds shared_buf to dst (dst += shared_buf)
                                    // Use parallel_for with nd_range for better compatibility with Intel Arc
                                    const size_t work_group_size = 256;
                                    const size_t num_groups = (dst_elements + work_group_size - 1) / work_group_size;
                                    main_stream
                                        ->parallel_for(sycl::nd_range<1>(num_groups * work_group_size, work_group_size),
                                                       [=](sycl::nd_item<1> item) {
                                                           size_t idx = item.get_global_id(0);
                                                           if (idx < dst_elements) {
                                                               dst_ptr[idx] += shared_buf[idx];
                                                           }
                                                       })
                                        .wait();

                                    // DEBUG: Verify result
                                    if (do_attn_dbg) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4 * sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG ATTN after add: dst[0..3]=[%f,%f,%f,%f]\n",
                                                total_sample[0], total_sample[1], total_sample[2], total_sample[3]);
                                    }
                                } else if (ggml_sycl_should_use_quant_allreduce(dst_elements)) {
                                    // QUANTIZED ALL_REDUCE: 33% bandwidth reduction (INT16)
                                    // Only used for large tensors where bandwidth savings outweigh kernel overhead
                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    float *   dst_ptr     = (float *) ggml_sycl_get_data_ptr(dst, main_device);

                                    quantized_allreduce(main_device, device, main_stream, stream, dst_ptr, partial_out,
                                                        dst_elements, do_attn_dbg);

                                    // DEBUG: Always check first few ATTN quantized outputs
                                    static int quant_attn_dbg = 0;
                                    if (quant_attn_dbg++ < 6) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4 * sizeof(float)).wait();
                                        fprintf(stderr,
                                                "TP DEBUG ATTN (QUANT) layer %d: dst_tensor=%p dst='%s' dst_data=%p "
                                                "dst_ptr=%p dst[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                layer, (void *) dst, dst->name, dst->data, (void *) dst_ptr,
                                                total_sample[0], total_sample[1], total_sample[2], total_sample[3]);
                                        // Also check tensor->data
                                        if (dst->data != dst_ptr) {
                                            float tensor_sample[4];
                                            main_stream->memcpy(tensor_sample, dst->data, 4 * sizeof(float)).wait();
                                            fprintf(stderr,
                                                    "TP DEBUG ATTN (QUANT) layer %d: dst->data=%p "
                                                    "tensor_data[0..3]=[%.6f,%.6f,%.6f,%.6f] MISMATCH!\n",
                                                    layer, dst->data, tensor_sample[0], tensor_sample[1],
                                                    tensor_sample[2], tensor_sample[3]);
                                        }
                                    }
                                } else {
                                    // Host-staged ALL_REDUCE (CPU addition) with persistent buffers
                                    // OPTIMIZED: Reduced from 3 sync points to 2 by overlapping copies
                                    float *dev0_host, *host_buf;
                                    ggml_sycl_tp_get_host_reduce_buffers(dst_size, &dev0_host, &host_buf);

                                    // Start both copies in parallel (they're independent)
                                    auto ev_dev1 = stream->memcpy(host_buf, partial_out, dst_size);

                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    float *   dst_ptr     = (float *) ggml_sycl_get_data_ptr(dst, main_device);

                                    auto ev_dev0 = main_stream->memcpy(dev0_host, dst_ptr, dst_size);

                                    // Single sync point for both copies
                                    ev_dev1.wait();
                                    ev_dev0.wait();

                                    // DEBUG: Show partial results from both devices
                                    if (do_attn_dbg) {
                                        fprintf(stderr,
                                                "TP DEBUG ATTN NON-QUANT: dev0[0..3]=[%.6f,%.6f,%.6f,%.6f] "
                                                "dev1[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                dev0_host[0], dev0_host[1], dev0_host[2], dev0_host[3], host_buf[0],
                                                host_buf[1], host_buf[2], host_buf[3]);
                                    }

                                    for (size_t i = 0; i < dst_elements; i++) {
                                        dev0_host[i] += host_buf[i];
                                    }

                                    // DEBUG: Show result after addition
                                    if (do_attn_dbg) {
                                        fprintf(stderr, "TP DEBUG ATTN NON-QUANT: sum[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                dev0_host[0], dev0_host[1], dev0_host[2], dev0_host[3]);
                                    }

                                    main_stream->memcpy(dst_ptr, dev0_host, dst_size).wait();
                                    // No free needed - buffers are persistent

                                    // DEBUG: Also check non-quant path
                                    static int nonquant_attn_dbg = 0;
                                    if (g_ggml_sycl_tp_debug && nonquant_attn_dbg++ < 6) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4 * sizeof(float)).wait();
                                        fprintf(stderr,
                                                "TP DEBUG ATTN (NON-QUANT) layer %d: dst_tensor=%p dst='%s' "
                                                "dst_data=%p dst_ptr=%p dst[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                layer, (void *) dst, dst->name, dst->data, (void *) dst_ptr,
                                                total_sample[0], total_sample[1], total_sample[2], total_sample[3]);
                                    }
                                }
                            }

                            // Cleanup device 1 buffers
                            ggml_sycl_set_device(device);
                            stream = ctx.stream(device, 0);
                            sycl::free(input_q8_dev, *stream);
                            sycl::free(q_out, *stream);
                            sycl::free(k_out, *stream);
                            sycl::free(v_out, *stream);
                            sycl::free(attn_out, *stream);
                            sycl::free(attn_q8_dev, *stream);
                            sycl::free(partial_out, *stream);
                        }
                    }

                    // Restore main device context
                    ggml_sycl_set_device(main_device);
                } else {
                    static int warn = 0;
                    if (warn++ < 3) {
                        fprintf(stderr, "SYCL TP: WARNING - missing attention weight refs for layer %d\n", layer);
                    }
                }

                // Clear stored attention input for this layer (no longer needed)
                {
                    std::lock_guard<std::mutex> lock(g_tp_attn_input_mutex);
                    auto                        it = g_tp_attn_inputs.find(layer);
                    if (it != g_tp_attn_inputs.end()) {
                        if (it->second.data) {
                            int device = g_sycl_tp_config.devices[1];
                            ggml_sycl_set_device(device);
                            queue_ptr stream = ctx.stream(device, 0);
                            sycl::free(it->second.data, *stream);
                            ggml_sycl_set_device(main_device);
                        }
                        g_tp_attn_inputs.erase(it);
                    }
                }

                // Return - we've handled the attention computation
                return;
            } else {
                static int warn = 0;
                if (warn++ < 3) {
                    fprintf(stderr, "SYCL TP: WARNING - no stored attention input for layer %d\n", layer);
                }
            }
        }

        // For other row-parallel layers without stored input, use partial result
        return;
    }

    // src1 is NOT from column-parallel (full K dimension)
    // Need to slice src1 and process each slice on its respective device
    const int64_t K_full         = ne10;
    const size_t  q8_1_ts        = sizeof(block_q8_1);
    const size_t  q8_1_bs        = QK8_1;
    const int64_t K_shard_padded = GGML_PAD(K_shard, MATRIX_ROW_PADDING);

    // For each other TP device (rank > 0), compute partial result and add
    for (int rank = 1; rank < world_size; rank++) {
        int device = g_sycl_tp_config.devices[rank];

        // Get this rank's weight shard
        void * weight_shard = extra->data_device[device];
        if (weight_shard == nullptr) {
            fprintf(stderr, "SYCL TP: ERROR - no weight shard on device %d for rank %d\n", device, rank);
            continue;
        }

        // Calculate src1 float offset for this rank
        // Each rank processes K_shard elements starting at rank * K_shard
        const int64_t src1_k_offset = rank * K_shard;
        if (src1_k_offset + K_shard > K_full) {
            fprintf(stderr,
                    "SYCL TP: ERROR - src1 slice out of bounds for rank %d (offset=%ld, K_shard=%ld, K_full=%ld)\n",
                    rank, (long) src1_k_offset, (long) K_shard, (long) K_full);
            continue;
        }

        // src1 float slice size per batch element
        const size_t src1_float_slice_size = ne11 * K_shard * sizeof(float);
        const size_t src1_q8_size          = ne11 * K_shard_padded * q8_1_ts / q8_1_bs;

        ggml_sycl_set_device(device);
        queue_ptr stream = ctx.stream(device, 0);

        // Allocate buffers on target device
        float * src1_ddf_dev = (float *) sycl::malloc_device(src1_float_slice_size, *stream);
        char *  src1_ddq_dev = (char *) sycl::malloc_device(src1_q8_size, *stream);
        float * partial_out  = (float *) sycl::malloc_device(dst_size, *stream);

        if (!src1_ddf_dev || !src1_ddq_dev || !partial_out) {
            fprintf(stderr, "SYCL TP: ERROR - failed to allocate temp buffers on device %d\n", device);
            if (src1_ddf_dev) {
                sycl::free(src1_ddf_dev, *stream);
            }
            if (src1_ddq_dev) {
                sycl::free(src1_ddq_dev, *stream);
            }
            if (partial_out) {
                sycl::free(partial_out, *stream);
            }
            ggml_sycl_set_device(main_device);
            continue;
        }

        // Copy src1 float slice from main device to this device - OPTIMIZED: use persistent staging buffer
        {
            ggml_sycl_set_device(main_device);
            queue_ptr main_stream = ctx.stream();

            // src1 layout: [K_full, batch] with stride nb1 = K_full * sizeof(float)
            // We need elements [src1_k_offset, src1_k_offset + K_shard) for each batch
            float * host_buf = ggml_sycl_tp_ensure_host_staging(src1_float_slice_size, main_stream);

            const float * src1_data = (const float *) src1->data;
            for (int64_t b = 0; b < ne11; b++) {
                // Source: src1 at [src1_k_offset, b]
                const float * src_ptr = src1_data + b * K_full + src1_k_offset;
                float *       dst_ptr = host_buf + b * K_shard;
                main_stream->memcpy(dst_ptr, src_ptr, K_shard * sizeof(float)).wait();
            }

            // Copy to target device
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);
            stream->memcpy(src1_ddf_dev, host_buf, src1_float_slice_size).wait();

            // Don't free - using persistent staging buffer
        }

        // Quantize src1 float to Q8_1 on target device
        // Use SoA quantizer if weight is reordered
        {
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);

            const bool use_soa_row_parallel = extra && extra->optimized_feature.is_reordered();
            if (use_soa_row_parallel) {
                quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(src1_ddf_dev, src1_ddq_dev, K_shard, ne11,
                                                                      K_shard_padded, stream);
            } else {
                quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_dev, src1_ddq_dev, K_shard, ne11, K_shard_padded,
                                                      stream);
            }
            stream->wait();
        }

        // Call MMVQ kernel on this device
        {
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);

            // Zero output first
            stream->memset(partial_out, 0, dst_size);

            // Call MMVQ kernel
            ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
                                       (const char *) weight_shard,  // src0_dd_i
                                       nullptr,                      // src1_ddf_i (not needed)
                                       src1_ddq_dev,                 // src1_ddq_i
                                       partial_out,                  // dst_dd_i
                                       0,                            // row_low
                                       ne01,                         // row_high
                                       ne11,                         // src1_ncols
                                       K_shard_padded,               // src1_padded_row_size
                                       stream);
        }

        // Copy partial result back to main device and add - OPTIMIZED: use persistent staging buffer
        {
            ggml_sycl_set_device(main_device);
            queue_ptr main_stream = ctx.stream();

            float * host_buf = ggml_sycl_tp_ensure_host_staging(dst_size, main_stream);

            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);
            stream->memcpy(host_buf, partial_out, dst_size).wait();

            ggml_sycl_set_device(main_device);
            float * temp_add = (float *) sycl::malloc_device(dst_size, *main_stream);
            main_stream->memcpy(temp_add, host_buf, dst_size).wait();

            ggml_sycl_add_f32((float *) dst->data, temp_add, dst_nelems, main_stream);
            main_stream->wait();

            sycl::free(temp_add, *main_stream);
            // Don't free host_buf - using persistent staging buffer
        }

        // Free temp buffers
        ggml_sycl_set_device(device);
        stream = ctx.stream(device, 0);
        sycl::free(src1_ddf_dev, *stream);
        sycl::free(src1_ddq_dev, *stream);
        sycl::free(partial_out, *stream);
    }

    // Ensure we're back on main device
    ggml_sycl_set_device(main_device);
}

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx,
                                       const ggml_tensor *         src0,
                                       const ggml_tensor *         src1,
                                       ggml_tensor *               dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void *  src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx,
                                     const ggml_tensor *         src0,
                                     const ggml_tensor *         src1,
                                     ggml_tensor *               dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];
    const int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void *  src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x     = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);
    const int64_t channel_stride_y = nb11 / sizeof(float);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12,
                                     channel_stride_x, channel_stride_y, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half *       src0_as_f16,
                                   const sycl::half *       src1_as_f16,
                                   void *                   dst,
                                   const void **            ptrs_src,
                                   void **                  ptrs_dst,
                                   int64_t                  ne12,
                                   int64_t                  ne13,
                                   int64_t                  ne23,
                                   size_t                   nb02,
                                   size_t                   nb03,
                                   size_t                   nb12,
                                   size_t                   nb13,
                                   size_t                   nbd2,
                                   size_t                   nbd3,
                                   int64_t                  r2,
                                   int64_t                  r3,
                                   const sycl::nd_item<3> & item_ct1) {
    const int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const uint8_t * src0_bytes = reinterpret_cast<const uint8_t *>(src0_as_f16);
    const uint8_t * src1_bytes = reinterpret_cast<const uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx,
                                           const ggml_tensor *         src0,
                                           const ggml_tensor *         src1,
                                           ggml_tensor *               dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    const sycl::half * src0_f16 = static_cast<const sycl::half *>(src0->data);
    float *            dst_ddf  = static_cast<float *>(dst->data);

    const sycl::half * src1_f16       = static_cast<const sycl::half *>(src1->data);
    const size_t       type_size_src0 = ggml_type_size(src0->type);
    const size_t       type_size_src1 = ggml_type_size(src1->type);

    [[maybe_unused]] bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    [[maybe_unused]] bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    [[maybe_unused]] int64_t         s11 = nb11 / type_size_src1;
    [[maybe_unused]] int64_t         s12 = nb12 / type_size_src1;
    [[maybe_unused]] int64_t         s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    if (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");

        // iterate tensor dims and find the slowest moving dim and stride
        int    last_dim    = 0;
        int    last_str    = 0;
        size_t largest_str = 0;
        for (int i = 0; i < 4; i++) {
            // last stride is always the largest
            if (src1->nb[i] == largest_str) {
                if (src1->ne[last_dim] == 1) {
                    last_str = i;
                    last_dim = i;
                }
            }
            if (src1->nb[i] > largest_str) {
                largest_str = src1->nb[i];
                last_str    = i;
                last_dim    = i;
            }
        }
#if GGML_SYCL_DNNL
        // oneDNN handles strided data and does not need overhead of get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
#else
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_nc_sycl_t to_fp16_nc_sycl = get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = true;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    [[maybe_unused]] dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    [[maybe_unused]] dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    [[maybe_unused]] size_t nbd2 = dst->nb[2];
    [[maybe_unused]] size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    [[maybe_unused]] const void * alpha = &alpha_f32;
    [[maybe_unused]] const void * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1 / nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    [[maybe_unused]] const int64_t r2 = ne12 / ne02;
    [[maybe_unused]] const int64_t r3 = ne13 / ne03;

#if GGML_SYCL_DNNL
    // Use oneDNN for batch GEMM operations (primary path for Intel)
    {
        int64_t str_a0 = nb00 / type_size_src0;
        int64_t str_a1 = nb01 / type_size_src0;
        int64_t str_a2 = nb02 / type_size_src0;

        int64_t str_b0 = nb10 / type_size_src1;
        int64_t str_b1 = nb11 / type_size_src1;
        int64_t str_b2 = nb12 / type_size_src1;

        auto launch_gemm_for_batches = [&ctx, queue](const sycl::half * src0, const sycl::half * src1, float * dst,
                                                     int64_t a0, int64_t a1, int64_t batcha, int64_t /*b0*/, int64_t b1,
                                                     int64_t batchb, int64_t sa0, int64_t sa1, int64_t sa2, int64_t sb0,
                                                     int64_t sb1, int64_t sb2, int64_t sd2) {
            bool supported_broadcast = batchb == batcha ? true : batchb == 1 || batcha == 1 ? true : false;
            if (supported_broadcast) {
                DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0, DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                                      DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                                      DnnlGemmWrapper::to_dt<float>(), queue, batcha, batchb);
            } else {
                // iterate over batches from smaller set of matrices (matrix 0)
                int64_t batches0 = batcha;
                int64_t batches1 = batchb;

                if (batches0 > batches1) {
                    int64_t num_mul_mats = batches1;
                    int64_t sub_batch    = batches0 / num_mul_mats;
                    // src0 is batched and bigger, shift and multiply with src1
                    for (int64_t i0 = 0; i0 < num_mul_mats; i0++) {
                        const sycl::half * src0_shifted = src0 + (sa2 * i0 * sub_batch);
                        const sycl::half * src1_shifted = src1 + (sb2 * i0);
                        float *            dst_shifted  = dst + (sd2 * i0 * sub_batch);
                        DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sa0,
                                              sa1, sa2, src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1,
                                              sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(), queue, sub_batch, 1);
                    }
                } else {
                    int64_t num_mul_mats = batches0;
                    int64_t sub_batch    = batches1 / num_mul_mats;
                    // src1 is batched and bigger, shift and multiply with src0
                    for (int64_t i1 = 0; i1 < num_mul_mats; i1++) {
                        const sycl::half * src0_shifted = src0 + (sa2 * i1);
                        const sycl::half * src1_shifted = src1 + (sb2 * i1 * sub_batch);
                        float *            dst_shifted  = dst + (sd2 * i1 * sub_batch);
                        DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sa0,
                                              sa1, sa2, src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1,
                                              sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(), queue, 1, sub_batch);
                    }
                }
            }
        };

        const bool cont_batches_dim2_a = nb02 * ne02 == nb03;
        const bool cont_batches_dim2_b = nb12 * ne12 == nb13;
        const bool cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
        const bool cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
        if (cont_batches_dim2_a && cont_batches_dim2_b) {
            // A batch is considered contiguous if the dimension 2 is not strided
            int64_t batches0 = ne02 * ne03;
            int64_t batches1 = ne12 * ne13;
            launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0, ne10, ne11, batches1, str_a0,
                                    str_a1, str_a2, str_b0, str_b1, str_b2, nb2 / sizeof(float));
        } else if (cont_batches_dim3_a && cont_batches_dim3_b) {
            // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
            int64_t batches0 = ne02 * ne03;
            int64_t batches1 = ne12 * ne13;
            int64_t str_a3   = nb03 / type_size_src0;
            int64_t str_b3   = nb13 / type_size_src1;
            launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0, ne10, ne11, batches1, str_a0,
                                    str_a1, str_a3, str_b0, str_b1, str_b3, nb2 / sizeof(float));
        } else {
            for (int64_t b_a = 0; b_a < ne03; b_a++) {
                const sycl::half * src0_f16_shifted = src0_f16 + (nb03 * b_a / type_size_src0);
                const sycl::half * src1_f16_shifted = src1_f16 + (nb13 * b_a / type_size_src1);
                float *            dst_shifted      = dst_ddf + (nb3 * b_a / sizeof(float));
                int64_t            batches0         = ne02;
                int64_t            batches1         = ne12;
                launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted, ne00, ne01, batches0, ne10,
                                        ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1, str_b2,
                                        nb2 / sizeof(float));
            }
        }
    }
#elif GGML_SYCL_HAS_ONEAPI_MATH
    // Fallback to oneAPI Math (MKL/oneMath) for batch GEMM
    {
        if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            const int64_t sma = ne02 == 1 ? nb03 / nb00 : nb02 / nb00;
            const int64_t smb = ne12 == 1 ? s13 : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, ne01, ne11, ne10, alpha,
                src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma, src1_f16, dpct::library_data_t::real_half,
                s11, smb, beta, dst_ddf, mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } else {
            const int ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<const void *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<void *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<float>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                const void ** ptrs_src_get = ptrs_src.get();
                void **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * sizeof(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * sizeof(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23,
                                           nb02, nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, ne01, ne11, ne10, alpha,
                (const void **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (const void **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (void **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
#else
    static_assert(false,
                  "Either GGML_SYCL_DNNL or GGML_SYCL_HAS_ONEAPI_MATH must be defined for batch GEMM operations");
#endif
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

enum class mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
    MMQ          = 3,
};

inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // Temporarily enabled for debugging XMX vs MMQ comparison
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return !g_ggml_sycl_prioritize_dmmv;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_coalesced_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_coalesced_get_rows(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_coalesced_mmq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mmq(enum ggml_type type) {
    // MMQ SoA supports the same types as MMVQ SoA
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// CPU-side Q4_0 AoS → SoA reorder (performed during upload staging)
// This is faster than GPU-side reorder because:
// 1. No GPU kernel launch overhead
// 2. Data is already in L1/L2 cache from memcpy
// 3. Single pass through memory instead of extra GPU allocation + copy
// ============================================================================
static void reorder_q4_0_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK4_0;
    const size_t nblocks        = blocks_per_row * nrows;

    // AoS layout: [d:2][qs:16] per block = 18 bytes
    // SoA layout: all qs first (nblocks * 16), then all d (nblocks * 2)
    const uint8_t * aos    = (const uint8_t *) src_aos;
    uint8_t *       soa_qs = (uint8_t *) dst_soa;
    uint8_t *       soa_d  = soa_qs + nblocks * (QK4_0 / 2);  // d values start after all qs

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q4_0);

        // Copy qs (16 bytes at offset 2 in AoS block)
        memcpy(soa_qs + ib * (QK4_0 / 2), block_aos + sizeof(ggml_half), QK4_0 / 2);

        // Copy d (2 bytes at offset 0 in AoS block)
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
}

// Q4_0 Coalesced layout (direct AoS → Coalesced on CPU)
// AoS layout: [d:2][qs:16] per block = 18 bytes
// Coalesced layout: word-major within tiles
//   - For word w of block b in tile: offset = tile_base + w*stride + b*4
//   - Word plane stride = TILE_BLOCKS * 4 bytes
//   - Scales (d values) follow all quant tiles contiguously
// This matches the GPU DMMV/MMVQ coalesced kernel access pattern
static bool reorder_q4_0_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK   = QK4_0 / 2;        // 16 bytes of quants per block
    constexpr int WORDS_PER_BLOCK   = 4;                // 4 words of 4 bytes each = 16 bytes
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;  // bytes between word planes

    const size_t blocks_per_row = ncols / QK4_0;
    const size_t nblocks        = blocks_per_row * nrows;

    // Layout sizes
    const size_t row_quants_bytes   = ncols / 2;  // bytes per row of quants
    const size_t total_quants_bytes = nrows * row_quants_bytes;

    // AoS input
    const uint8_t * aos = (const uint8_t *) src_aos;

    // Coalesced output pointers
    uint8_t * coal_qs = (uint8_t *) dst_coalesced;
    uint8_t * coal_d  = coal_qs + total_quants_bytes;  // d values after all quants

    // Verify tile alignment
    if (blocks_per_row % TILE_BLOCKS != 0) {
        fprintf(stderr, "[CPU-REORDER] Q4_0 coalesced: blocks_per_row=%zu not divisible by %d, falling back to SoA\n",
                blocks_per_row, TILE_BLOCKS);
        reorder_q4_0_cpu(dst_coalesced, src_aos, ncols, nrows);
        return false;
    }

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q4_0);
        const uint8_t * src_qs    = block_aos + sizeof(ggml_half);  // qs at offset 2

        // Which row/column is this block?
        const size_t row           = ib / blocks_per_row;
        const size_t col_block     = ib % blocks_per_row;
        const size_t tile          = col_block / TILE_BLOCKS;
        const size_t block_in_tile = col_block % TILE_BLOCKS;

        // Base offset for this tile's quants in the output
        const size_t tile_base = row * row_quants_bytes + tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

        // Copy 16 bytes (4 words) of quants in word-major order
        for (int word = 0; word < WORDS_PER_BLOCK; word++) {
            const size_t word_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(coal_qs + word_offset, src_qs + word * 4, 4);
        }

        // Copy scale (d value) - scales are contiguous after all quants
        memcpy(coal_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
    return true;
}

// Q8_0 Coalesced layout (direct AoS → Coalesced on CPU)
// AoS layout: [d:2][qs:32] per block = 34 bytes
// Coalesced layout: word-major within tiles (8 words per block), scales after all quants
// This matches the GPU MMVQ/DMMV coalesced kernel access pattern
static bool reorder_q8_0_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK   = QK8_0;            // 32 bytes of quants per block
    constexpr int WORDS_PER_BLOCK   = 8;                // 8 words of 4 bytes each
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;  // bytes between word planes

    const size_t blocks_per_row = ncols / QK8_0;
    const size_t nblocks        = blocks_per_row * nrows;

    // Layout sizes
    const size_t row_quants_bytes   = ncols;  // bytes per row of quants
    const size_t total_quants_bytes = nrows * row_quants_bytes;

    // AoS input
    const uint8_t * aos = (const uint8_t *) src_aos;

    // Coalesced output pointers
    uint8_t * coal_qs = (uint8_t *) dst_coalesced;
    uint8_t * coal_d  = coal_qs + total_quants_bytes;  // d values after all quants

    // Verify tile alignment
    if (blocks_per_row % TILE_BLOCKS != 0) {
        fprintf(stderr, "[CPU-REORDER] Q8_0 coalesced: blocks_per_row=%zu not divisible by %d\n", blocks_per_row,
                TILE_BLOCKS);
        return false;
    }

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q8_0);
        const uint8_t * src_qs    = block_aos + sizeof(ggml_half);  // qs at offset 2

        // Which row/column is this block?
        const size_t row           = ib / blocks_per_row;
        const size_t col_block     = ib % blocks_per_row;
        const size_t tile          = col_block / TILE_BLOCKS;
        const size_t block_in_tile = col_block % TILE_BLOCKS;

        // Base offset for this tile's quants in the output
        const size_t tile_base = row * row_quants_bytes + tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

        // Copy 32 bytes (8 words) of quants in word-major order
        for (int word = 0; word < WORDS_PER_BLOCK; word++) {
            const size_t word_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(coal_qs + word_offset, src_qs + word * 4, 4);
        }

        // Copy scale (d value) - scales are contiguous after all quants
        memcpy(coal_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
    return true;
}

// MXFP4 Coalesced layout (direct AoS → Coalesced on CPU)
// AoS layout: [e:1][qs:16] per block = 17 bytes
// Coalesced layout: word-major within tiles (4 words per block), e values after all quants
// This matches the GPU MMVQ coalesced kernel access pattern
static bool reorder_mxfp4_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK   = QK_MXFP4 / 2;  // 16 bytes of quants per block
    constexpr int WORDS_PER_BLOCK   = 4;             // 4 words of 4 bytes each
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const size_t blocks_per_row = ncols / QK_MXFP4;
    const size_t nblocks        = blocks_per_row * nrows;

    // Layout sizes
    const size_t row_quants_bytes   = ncols / 2;  // bytes per row of quants
    const size_t total_quants_bytes = nrows * row_quants_bytes;

    // AoS input
    const uint8_t * aos = (const uint8_t *) src_aos;

    // Coalesced output pointers
    uint8_t * coal_qs = (uint8_t *) dst_coalesced;
    uint8_t * coal_e  = coal_qs + total_quants_bytes;  // exponent bytes after all quants

    // Verify tile alignment
    if (blocks_per_row % TILE_BLOCKS != 0) {
        fprintf(stderr, "[CPU-REORDER] MXFP4 coalesced: blocks_per_row=%zu not divisible by %d\n", blocks_per_row,
                TILE_BLOCKS);
        return false;
    }

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_mxfp4);
        const uint8_t * src_qs    = block_aos + sizeof(uint8_t);  // qs after exponent

        // Which row/column is this block?
        const size_t row           = ib / blocks_per_row;
        const size_t col_block     = ib % blocks_per_row;
        const size_t tile          = col_block / TILE_BLOCKS;
        const size_t block_in_tile = col_block % TILE_BLOCKS;

        // Base offset for this tile's quants in the output
        const size_t tile_base = row * row_quants_bytes + tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

        // Copy 16 bytes (4 words) of quants in word-major order
        for (int word = 0; word < WORDS_PER_BLOCK; word++) {
            const size_t word_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(coal_qs + word_offset, src_qs + word * 4, 4);
        }

        // Copy exponent (e value) - contiguous after all quants
        coal_e[ib] = block_aos[0];
    }
    return true;
}

// Q8_0: [d:2][qs:32] = 34 bytes per block
// SoA: [all qs][all d]
static void reorder_q8_0_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK8_0;
    const size_t nblocks        = blocks_per_row * nrows;

    const uint8_t * aos    = (const uint8_t *) src_aos;
    uint8_t *       soa_qs = (uint8_t *) dst_soa;
    uint8_t *       soa_d  = soa_qs + nblocks * QK8_0;  // d values start after all qs

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q8_0);

        // Copy qs (32 bytes at offset 2 in AoS block)
        memcpy(soa_qs + ib * QK8_0, block_aos + sizeof(ggml_half), QK8_0);

        // Copy d (2 bytes at offset 0 in AoS block)
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
}

// Q4_K: [dm:4][scales:12][qs:128] = 144 bytes per block (QK_K=256)
// SoA: [all qs][all scales][all dm]
static void reorder_q4_k_cpu(void * dst_soa, const void * src_aos, size_t nblocks) {
    const uint8_t * aos        = (const uint8_t *) src_aos;
    uint8_t *       soa_qs     = (uint8_t *) dst_soa;
    uint8_t *       soa_scales = soa_qs + nblocks * (QK_K / 2);
    uint8_t *       soa_dm     = soa_scales + nblocks * K_SCALE_SIZE;

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q4_K);

        // AoS layout: [dm:4][scales:12][qs:128]
        // Copy qs (128 bytes at offset 16 in AoS block)
        memcpy(soa_qs + ib * (QK_K / 2), block_aos + 4 + K_SCALE_SIZE, QK_K / 2);

        // Copy scales (12 bytes at offset 4 in AoS block)
        memcpy(soa_scales + ib * K_SCALE_SIZE, block_aos + 4, K_SCALE_SIZE);

        // Copy dm (4 bytes at offset 0 in AoS block)
        memcpy(soa_dm + ib * 4, block_aos, 4);
    }
}

// Q6_K: [ql:128][qh:64][scales:16][d:2] = 210 bytes per block (QK_K=256)
// SoA: [all ql][all qh][all scales][all d]
static void reorder_q6_k_cpu(void * dst_soa, const void * src_aos, size_t nblocks) {
    const uint8_t * aos        = (const uint8_t *) src_aos;
    uint8_t *       soa_ql     = (uint8_t *) dst_soa;
    uint8_t *       soa_qh     = soa_ql + nblocks * (QK_K / 2);
    uint8_t *       soa_scales = soa_qh + nblocks * (QK_K / 4);
    uint8_t *       soa_d      = soa_scales + nblocks * (QK_K / 16);

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q6_K);

        // AoS layout: [ql:128][qh:64][scales:16][d:2]
        // Copy ql (128 bytes at offset 0)
        memcpy(soa_ql + ib * (QK_K / 2), block_aos, QK_K / 2);

        // Copy qh (64 bytes at offset 128)
        memcpy(soa_qh + ib * (QK_K / 4), block_aos + (QK_K / 2), QK_K / 4);

        // Copy scales (16 bytes at offset 192)
        memcpy(soa_scales + ib * (QK_K / 16), block_aos + (QK_K / 2) + (QK_K / 4), QK_K / 16);

        // Copy d (2 bytes at offset 208)
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos + (QK_K / 2) + (QK_K / 4) + (QK_K / 16), sizeof(ggml_half));
    }
}

// Q6_K Coalesced layout (direct AoS → Coalesced on CPU)
// AoS layout: [ql:128][qh:64][scales:16][d:2] per block = 210 bytes
// Coalesced layout: word-major within tiles for each component
//   - Tile structure: [ql_word_major...][qh_word_major...][scales_word_major...]
//   - Word plane stride = TILE_BLOCKS * 4 bytes
//   - D values follow all quant tiles contiguously
// This matches the GPU DMMV/MMVQ coalesced kernel access pattern
static bool reorder_q6_k_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows) {
    // Variable tile decomposition: supports any block count via power-of-2 tiles
    // Each tile is the largest power-of-2 <= remaining blocks, max 32 (warp size)
    // Example: 56 blocks = 32 + 16 + 8 = 3 tiles

    const int    blocks_per_row = ncols / QK_K;
    const size_t nblocks        = (size_t) blocks_per_row * nrows;
    const int    num_tiles      = tile_count(blocks_per_row);

    // Compute row stride: sum of all tile sizes * bytes per block
    // Each tile stores: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) = 208 bytes per block
    size_t row_quants_bytes = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_quants_bytes += (size_t) ts * (128 + 64 + 16);
    }

    // AoS input
    const uint8_t * aos = (const uint8_t *) src_aos;

    // Coalesced layout: [row0_tiles][row1_tiles]...[all d values]
    // D values contiguous at end of all quant data
    uint8_t * coal_d = (uint8_t *) dst_coalesced + (size_t) nrows * row_quants_bytes;

    GGML_SYCL_DEBUG("[CPU-REORDER] Q6_K variable tile: blocks_per_row=%d, num_tiles=%d, row_stride=%zu\n",
                    blocks_per_row, num_tiles, row_quants_bytes);

    // Process each row
    for (int row = 0; row < nrows; row++) {
        uint8_t * row_dst   = (uint8_t *) dst_coalesced + row * row_quants_bytes;
        int       block_idx = 0;

        for (int tile = 0; tile < num_tiles; tile++) {
            const int tile_size         = tile_size_at(blocks_per_row, tile);
            const int word_plane_stride = tile_size * 4;  // Stride between word planes for this tile

            // Tile layout: [ql: tile_size * 128][qh: tile_size * 64][scales: tile_size * 16]
            uint8_t * tile_ql = row_dst;
            uint8_t * tile_qh = tile_ql + tile_size * 128;
            uint8_t * tile_sc = tile_qh + tile_size * 64;

            // Process each block in this tile
            for (int b = 0; b < tile_size; b++) {
                const size_t    global_block = row * blocks_per_row + block_idx + b;
                const uint8_t * block_aos    = aos + global_block * sizeof(block_q6_K);

                // === Process ql (128 bytes = 32 words of 4 bytes) ===
                const uint8_t * src_ql = block_aos;  // ql at offset 0
                for (int word = 0; word < 32; word++) {
                    memcpy(tile_ql + word * word_plane_stride + b * 4, src_ql + word * 4, 4);
                }

                // === Process qh (64 bytes = 16 words of 4 bytes) ===
                const uint8_t * src_qh = block_aos + 128;  // qh at offset 128
                for (int word = 0; word < 16; word++) {
                    memcpy(tile_qh + word * word_plane_stride + b * 4, src_qh + word * 4, 4);
                }

                // === Process scales (16 bytes = 4 words of 4 bytes) ===
                const uint8_t * src_sc = block_aos + 128 + 64;  // scales at offset 192
                for (int word = 0; word < 4; word++) {
                    memcpy(tile_sc + word * word_plane_stride + b * 4, src_sc + word * 4, 4);
                }

                // === Copy d (2 bytes) - contiguous at end of all quant data ===
                const uint8_t * src_d = block_aos + 128 + 64 + 16;  // d at offset 208
                memcpy(coal_d + global_block * sizeof(ggml_half), src_d, sizeof(ggml_half));
            }

            // Advance destination pointer to next tile
            row_dst = tile_sc + tile_size * 16;
            block_idx += tile_size;
        }
    }
    return true;  // Always succeeds with variable tiles
}

// MXFP4: [e:1][qs:16] = 17 bytes per block (QK_MXFP4=32)
// SoA: [all qs][all e]
static void reorder_mxfp4_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK_MXFP4;
    const size_t nblocks        = blocks_per_row * nrows;

    const uint8_t * aos    = (const uint8_t *) src_aos;
    uint8_t *       soa_qs = (uint8_t *) dst_soa;
    uint8_t *       soa_e  = soa_qs + nblocks * (QK_MXFP4 / 2);  // e values start after all qs

    for (size_t ib = 0; ib < nblocks; ib++) {
        const block_mxfp4 * block = (const block_mxfp4 *) (aos + ib * sizeof(block_mxfp4));

        // Copy qs array (16 bytes)
        memcpy(soa_qs + ib * (QK_MXFP4 / 2), block->qs, QK_MXFP4 / 2);

        // Copy e (scale) value (1 byte)
        soa_e[ib] = block->e;
    }
}

// Check if tensor is eligible for CPU-side SoA reorder during upload
static bool should_cpu_reorder(const ggml_tensor * tensor, const ggml_backend_sycl_buffer_context * ctx) {
    // Supported quantized types for CPU-side SoA reorder
    if (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q6_K && tensor->type != GGML_TYPE_MXFP4) {
        return false;
    }

    // Check if reordering is enabled
    if (!ggml_sycl_reorder_enabled()) {
        return false;
    }

    // Need tensor dimensions
    if (tensor->ne[0] == 0 || tensor->ne[1] == 0) {
        return false;
    }

    // NOTE: token_embd is no longer excluded because:
    // 1. When tied weights are used, output tensors share "token_embd.weight" name
    // 2. GET_ROWS for Q4_0/Q8_0 already handles SoA layout via is_soa() check
    // 3. User explicitly requested GPU loading for token_embd

    return ctx->supports_soa_reorder;
}

// Helper functions to unify device memory allocation for both async and sync paths
static inline void * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        return syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    return sycl::malloc(size, *stream, sycl::usm::alloc::device);
}

static inline void sycl_ext_free(dpct::queue_ptr stream, void * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        syclex::async_free(*stream, ptr);
        return;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    sycl::free(ptr, *stream);
}

static void reorder_qw_q4_0(uint8_t *       data_device,
                            const int       ncols,
                            const int       nrows,
                            size_t          size,
                            size_t          offset,
                            dpct::queue_ptr stream) {
    GGML_SYCL_KTRACE("reorder_qw_q4_0", " ncols=%d nrows=%d size=%zu offset=%zu nblocks=%zu", ncols, nrows, size,
                     offset, size / sizeof(block_q4_0));
    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q4_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q4_0) == 0));
    int    offset_blks   = offset / sizeof(block_q4_0);
    size_t nblocks       = size / sizeof(block_q4_0);
    auto   qs_ptr        = data_device + offset_blks * QK4_0 / 2;
    size_t d_byte_offset = ncols * nrows / 2;  // where d values start in SoA
    auto   d_ptr         = (sycl::half *) (qs_ptr + d_byte_offset) + offset_blks;

    auto reorder_event =
        stream->parallel_for(size / sizeof(block_q4_0), [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q4_0 * x  = (const block_q4_0 *) tmp_buf;
            const int          ib = i;

            for (int j = 0; j < QK4_0 / 2; j++) {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }

    sycl_ext_free(stream, tmp_buf);
}

static void reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q4_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q4_K) == 0);

    const int nblocks = size / sizeof(block_q4_K);

    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q4_K * x  = (const block_q4_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    sycl_ext_free(stream, tmp_buf);
}

static void reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q6_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q6_K) == 0);

    const int nblocks = size / sizeof(block_q6_K);

    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q6_K * x  = (const block_q6_K *) tmp_buf;
        const int          ib = i;

        const uint8_t * ql              = x[ib].ql;
        const uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        for (int j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    sycl_ext_free(stream, tmp_buf);
}

static bool reorder_aos_to_soa_device(const ggml_tensor * tensor,
                                      const void *        src_dev,
                                      void *              dst_dev,
                                      size_t              size,
                                      dpct::queue_ptr     stream) {
    if (!tensor || !src_dev || !dst_dev) {
        return false;
    }

    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = ggml_nrows(tensor);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            reorder_q4_0_aos_to_soa_sycl(src_dev, dst_dev, ncols, nrows, stream);
            break;
        case GGML_TYPE_Q8_0:
            reorder_q8_0_aos_to_soa_sycl(src_dev, dst_dev, ncols, nrows, stream);
            break;
        case GGML_TYPE_Q4_K:
            GGML_ASSERT(size % sizeof(block_q4_K) == 0);
            reorder_q4_k_aos_to_soa_sycl(src_dev, dst_dev, size / sizeof(block_q4_K), stream);
            break;
        case GGML_TYPE_Q6_K:
            GGML_ASSERT(size % sizeof(block_q6_K) == 0);
            reorder_q6_k_aos_to_soa_sycl(src_dev, dst_dev, size / sizeof(block_q6_K), stream);
            break;
        case GGML_TYPE_MXFP4:
            reorder_mxfp4_aos_to_soa_sycl(src_dev, dst_dev, ncols, nrows, stream);
            break;
        default:
            return false;
    }

    if (!g_ggml_sycl_use_async_mem_op) {
        stream->wait_and_throw();
    }
    return true;
}

// Q8_0: 32 int8 quants + fp16 scale = 34 bytes per block
// Reordered layout: [qs0..qsN] [d0..dN]
static void reorder_qw_q8_0(uint8_t *       data_device,
                            const int       ncols,
                            const int       nrows,
                            size_t          size,
                            size_t          offset,
                            dpct::queue_ptr stream) {
    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q8_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q8_0) == 0));
    int  offset_blks = offset / sizeof(block_q8_0);
    auto qs_ptr      = data_device + offset_blks * QK8_0;
    auto d_ptr       = (sycl::half *) (qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event =
        stream->parallel_for(size / sizeof(block_q8_0), [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q8_0 * x  = (const block_q8_0 *) tmp_buf;
            const int          ib = i;

            // Copy 32 int8 quants
            for (int j = 0; j < QK8_0; j++) {
                *(qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    sycl_ext_free(stream, tmp_buf);
}

// MXFP4: 16 packed bytes (32 4-bit elements) + 1 byte E8M0 exponent = 17 bytes per block
// Reordered layout: [qs0..qsN] [scale0..scaleN]
//
// Optimized for coalesced memory access:
// - Uses vectorized 4-byte writes (4 ints per block) instead of byte-by-byte
// - Tiled approach: work-group processes BLOCKS_PER_TILE blocks together
// - Adjacent threads write to adjacent memory addresses within a tile
static void reorder_qw_mxfp4(uint8_t *       data_device,
                             const int       ncols,
                             const int       nrows,
                             size_t          size,
                             size_t          offset,
                             dpct::queue_ptr stream) {
    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_mxfp4) == 0));
    GGML_ASSERT((offset % sizeof(block_mxfp4) == 0));

    const size_t num_blocks  = size / sizeof(block_mxfp4);
    const int    offset_blks = offset / sizeof(block_mxfp4);
    uint8_t *    qs_out      = data_device + offset_blks * (QK_MXFP4 / 2);
    uint8_t *    scale_out   = qs_out + (ncols / 2) * nrows + offset_blks;

    // Constants for tiled processing
    constexpr int BYTES_PER_BLOCK = QK_MXFP4 / 2;         // 16 bytes of qs per block
    constexpr int INTS_PER_BLOCK  = BYTES_PER_BLOCK / 4;  // 4 ints per block
    constexpr int BLOCKS_PER_TILE = WARP_SIZE;            // Process 32 blocks per work-group

    // Calculate grid dimensions
    // Each work-group has BLOCKS_PER_TILE threads, each handling one block
    const size_t num_tiles = (num_blocks + BLOCKS_PER_TILE - 1) / BLOCKS_PER_TILE;

    auto reorder_event = stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_tiles * BLOCKS_PER_TILE), sycl::range<1>(BLOCKS_PER_TILE)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_mxfp4 * src = (const block_mxfp4 *) tmp_buf;

            const size_t block_idx = item.get_global_id(0);
            if (block_idx >= num_blocks) {
                return;
            }

            // Use 4-byte vectorized writes for the 16-byte qs array
            // This reduces memory operations from 16 to 4 per block
            const uint32_t * src_qs = reinterpret_cast<const uint32_t *>(src[block_idx].qs);
            uint32_t *       dst_qs = reinterpret_cast<uint32_t *>(qs_out + block_idx * BYTES_PER_BLOCK);

// Vectorized write: 4 ints (16 bytes) per block
#pragma unroll
            for (int i = 0; i < INTS_PER_BLOCK; i++) {
                dst_qs[i] = src_qs[i];
            }

            // Single byte write for scale
            scale_out[block_idx] = src[block_idx].e;
        });

    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    sycl_ext_free(stream, tmp_buf);
}

// INTERNAL: Do NOT call directly - use reorder_tensor_to_soa() instead!
// This only transforms data, does NOT set the flag.
static void reorder_data_internal_(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t    ncols       = src0->ne[0];
    // Use ggml_nrows to handle tensors with >2 dimensions (e.g., MoE expert weights)
    // For 2D: nrows = ne[1], for 3D MoE: nrows = ne[1] * ne[2]
    size_t    nrows       = ggml_nrows(src0);
    size_t    size        = ggml_nbytes(src0);

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
            break;
        case GGML_TYPE_Q4_K:
            reorder_qw_q4_k(data_device, size, 0, stream);
            break;
        case GGML_TYPE_Q6_K:
            reorder_qw_q6_k(data_device, size, 0, stream);
            break;
        case GGML_TYPE_Q8_0:
            reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
            break;
        case GGML_TYPE_MXFP4:
            reorder_qw_mxfp4(data_device, ncols, nrows, size, 0, stream);
            break;
        default:
            GGML_ABORT("reorder_data_internal_() called with unsupported type");
            break;
    }
}

// =============================================================================
// UNIFIED REORDER FUNCTION - Single point of truth for SoA transformation
// This function ALWAYS does both:
//   1. Transform data from AoS to SoA layout (reorder_qw)
//   2. Set the reorder flag on the tensor (set_reorder)
// This ensures data and flag are ALWAYS in sync.
// Declared in common.hpp as friend of optimize_feature
// =============================================================================
bool reorder_tensor_to_soa(const ggml_tensor * tensor, dpct::queue_ptr stream, const char * caller) {
    if (!tensor || !tensor->extra) {
        fprintf(stderr, "[REORDER-UNIFIED] ERROR: tensor=%p extra=%p - cannot reorder\n", (void *) tensor,
                tensor ? (void *) tensor->extra : nullptr);
        return false;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);

    // Check if already reordered
    if (extra->optimized_feature.get_reorder() == reorder_mode::SOA) {
        // Already in SoA - no-op
        return true;
    }

    // Check if in unexpected state
    if (extra->optimized_feature.get_reorder() != reorder_mode::NONE) {
        fprintf(stderr, "[REORDER-UNIFIED] ERROR: tensor '%s' in mode %d, expected NONE(0) for SoA reorder\n",
                tensor->name, (int) extra->optimized_feature.get_reorder());
        return false;
    }

    // Check if type is supported
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            break;  // Supported
        default:
            fprintf(stderr, "[REORDER-UNIFIED] ERROR: tensor '%s' type %d not supported for SoA\n", tensor->name,
                    tensor->type);
            return false;
    }

    // DO THE REORDER - transform data from AoS to SoA
    reorder_data_internal_(tensor, stream);

    // SET THE FLAG - mark tensor as SoA (using private method via friend access)
    extra->optimized_feature.set_reorder_mode_(reorder_mode::SOA, tensor->name, caller);

    return true;
}

// Callback wrapper for expert cache SoA reordering (MXFP4)
// This matches the ggml_sycl::reorder_callback_fn signature
static void reorder_callback_mxfp4(uint8_t * data_device, int ncols, int nrows, size_t size, sycl::queue * stream) {
    reorder_qw_mxfp4(data_device, ncols, nrows, size, 0, stream);
}

// Check if tensor is eligible for reordering (regardless of batch size)
// Used by opt_for_reorder() to decide whether to reorder on first use
static bool can_reorder_tensor(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    bool result = ggml_sycl_reorder_enabled() &&  // reordering enabled via GGML_SYCL_REORDER_MODE
                  ctx.supports_soa_reorder &&     //device capability: GPU supports SoA optimization
                  (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_MUL_MAT_ID);  //MUL_MAT or MoE
    if (g_ggml_sycl_debug >= 2) {
        fprintf(stderr, "[REORDER-DBG] can_reorder: reorder_enabled=%d device_supports_soa=%d op=%d result=%d\n",
                (int) ggml_sycl_reorder_enabled(), (int) ctx.supports_soa_reorder, dst->op, (int) result);
    }
    return result;
}

// Check if we should USE the SoA kernel for this operation (requires batch=1)
// The tensor may already be reordered, but we only use SoA kernel for decode (batch=1)
static bool should_reorder_tensor(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    bool result = ggml_sycl_reorder_enabled() &&  // reordering enabled via GGML_SYCL_REORDER_MODE
                  ctx.supports_soa_reorder &&     //device capability: GPU supports SoA optimization
                  dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
                  dst->src[1]->ne[1] == 1 && dst->src[1]->ne[2] == 1 && dst->src[1]->ne[3] == 1;
    if (g_ggml_sycl_debug) {
        fprintf(stderr,
                "[REORDER-DBG] should_reorder: reorder_enabled=%d device_supports_soa=%d op=%d "
                "src1_dims=[%lld,%lld,%lld] result=%d\n",
                (int) ggml_sycl_reorder_enabled(), (int) ctx.supports_soa_reorder, dst->op,
                (long long) dst->src[1]->ne[1], (long long) dst->src[1]->ne[2], (long long) dst->src[1]->ne[3],
                (int) result);
    }
    return result;
}

// Check if a specific tensor needs reordering (not yet reordered)
static bool tensor_needs_reorder(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    if (!should_reorder_tensor(ctx, dst)) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    if (!src0 || !src0->extra) {
        return false;
    }
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    // If not yet reordered and supported type, reordering will happen
    if (!extra->optimized_feature.is_reordered()) {
        // Check if type supports reorder
        if (ggml_sycl_supports_reorder_mmvq(src0->type) || ggml_sycl_supports_reorder_dmmv(src0->type) ||
            ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
            return true;
        }
    }
    return false;
}

// Check if any MUL_MAT weight tensor needs reordering (used to decide if pre-reorder is needed)
// NOTE: We don't check activation dimensions (ne[1]) because we want to pre-reorder ALL weights
// during prompt phase, so decode phase graphs don't get blocked.
static bool graph_needs_reorder(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Skip if reordering is disabled or device doesn't support it
    if (!ggml_sycl_reorder_enabled() || !ctx.supports_soa_reorder) {
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        // Handle both MUL_MAT and MUL_MAT_ID (MoE expert weights)
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        const ggml_tensor * src0 = node->src[0];
        if (!src0 || !src0->extra) {
            continue;
        }
        const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
        // Check if weight tensor is NOT yet reordered and supports reorder
        if (!extra->optimized_feature.is_reordered() && extra->tp_type == tp_layer_type::TP_NONE &&
            (ggml_sycl_supports_reorder_mmvq(src0->type) || ggml_sycl_supports_reorder_dmmv(src0->type) ||
             ggml_sycl_supports_reorder_mul_mat_sycl(src0->type))) {
            // Note: MXFP4 MoE weights are also reordered now - fused kernel supports SoA layout
            if (g_ggml_sycl_debug) {
                fprintf(stderr, "[SYCL-GRAPH] needs_reorder: node %d '%s' src0='%s' type=%s\n", i, node->name,
                        src0->name, ggml_type_name(src0->type));
            }
            return true;
        }
    }
    return false;
}

// Convert reordered tensors to coalesced layout for better memory access patterns.
// Must be called AFTER reorder pass completes. Safe to call even if no tensors need conversion.
// Forward declaration for graph-level coalesced gating.
static bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
static bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

static void graph_convert_to_coalesced(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Check if coalesced mode is enabled via global reorder mode
    if (g_ggml_sycl_reorder_mode != reorder_mode::COALESCED) {
        return;
    }

    int coalesced_count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_MUL_MAT) {
            // Only MUL_MAT - MUL_MAT_ID uses _id kernels which don't support coalesced layout
            const ggml_tensor * src0 = node->src[0];
            const ggml_tensor * src1 = node->src[1];
            const bool          mmvq_eligible =
                ggml_sycl_supports_reorder_mmvq(src0->type) && can_use_mul_mat_vec_q(src0, src1, (ggml_tensor *) node);
            const bool dmmv_eligible = ggml_sycl_supports_reorder_dmmv(src0->type) &&
                                       can_use_dequantize_mul_mat_vec(src0, src1, (ggml_tensor *) node);

            // Only convert when an eligible kernel can consume coalesced layout.
            if (!mmvq_eligible && !dmmv_eligible) {
                continue;
            }

            // Avoid coalescing when DMMV is forced/preferred for types without coalesced DMMV kernels.
            const bool force_dmmv = std::getenv("GGML_SYCL_FORCE_DMMV") != nullptr;
            if (dmmv_eligible && (g_ggml_sycl_prioritize_dmmv || force_dmmv) &&
                !ggml_sycl_supports_coalesced_dmmv(src0->type)) {
                continue;
            }
            // Try each supported type
            if (ggml_sycl_convert_to_coalesced_q4_0(src0, ctx.stream())) {
                coalesced_count++;
            } else if (ggml_sycl_convert_to_coalesced_q8_0(src0, ctx.stream())) {
                coalesced_count++;
            } else if (ggml_sycl_convert_to_coalesced_q6_k(src0, ctx.stream())) {
                coalesced_count++;
            } else if (ggml_sycl_convert_to_coalesced_mxfp4(src0, ctx.stream())) {
                coalesced_count++;
            }
            continue;
        }

        if (node->op == GGML_OP_GET_ROWS) {
            const ggml_tensor * src0 = node->src[0];
            if (!ggml_sycl_supports_coalesced_get_rows(src0->type)) {
                continue;
            }
            if (ggml_sycl_convert_to_coalesced_q4_0(src0, ctx.stream())) {
                coalesced_count++;
            } else if (ggml_sycl_convert_to_coalesced_q8_0(src0, ctx.stream())) {
                coalesced_count++;
            } else if (ggml_sycl_convert_to_coalesced_q6_k(src0, ctx.stream())) {
                coalesced_count++;
            }
        }
    }
    if (coalesced_count > 0) {
        ctx.stream()->wait();
        GGML_SYCL_DEBUG("[SYCL-GRAPH] converted %d tensors to coalesced layout\n", coalesced_count);
    }
}

// Helper to parse layer ID from tensor name like "blk.5.ffn_gate_exps.weight"
static int parse_layer_id_from_name(const char * name) {
    if (!name) {
        return -1;
    }
    const char * blk_pos = strstr(name, "blk.");
    if (!blk_pos) {
        return -1;
    }
    return atoi(blk_pos + 4);  // Skip "blk."
}

// Pre-load all MoE experts into unified cache before graph recording
// This ensures stable cache pointers during graph execution
// Returns true if all experts were successfully pre-loaded and pinned
static bool graph_preload_moe_experts(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Unified cache handles all expert caching now
    if (!ggml_sycl::unified_cache_enabled()) {
        GGML_SYCL_DEBUG("[GRAPH-PRELOAD] Unified cache disabled, skipping preload\n");
        return true;  // No caching, but allow graphs to proceed
    }

    // First pass: find MoE nodes and count experts needed
    const ggml_tensor * first_moe_src0         = nullptr;
    size_t              total_experts_needed   = 0;
    int                 moe_nodes_found        = 0;
    int                 moe_nodes_skipped_sycl = 0;

    GGML_SYCL_DEBUG("[GRAPH-PRELOAD] Scanning graph with %d nodes for device %d\n", cgraph->n_nodes, ctx.device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        moe_nodes_found++;
        const ggml_tensor * src0 = node->src[0];
        if (!src0 || !src0->buffer) {
            continue;
        }

        bool is_sycl_buf = ggml_backend_buffer_is_sycl(src0->buffer);
        if (is_sycl_buf) {
            moe_nodes_skipped_sycl++;
            GGML_SYCL_DEBUG("[GRAPH-PRELOAD] Skipping MoE tensor '%s' (already on SYCL device)\n", src0->name);
            continue;  // Already on device
        }

        if (!first_moe_src0) {
            first_moe_src0 = src0;
        }
        total_experts_needed += static_cast<size_t>(src0->ne[2]);
    }

    GGML_SYCL_DEBUG("[GRAPH-PRELOAD] Found %d MoE nodes, %d skipped (on device), %zu experts need caching\n",
                    moe_nodes_found, moe_nodes_skipped_sycl, total_experts_needed);

    if (!first_moe_src0) {
        return true;  // No MoE nodes with mmap'd weights
    }

    // Set SoA reorder callback for MXFP4 (applied on cache miss)
    if (first_moe_src0->type == GGML_TYPE_MXFP4) {
        ggml_sycl::set_moe_reorder_callback(reorder_callback_mxfp4);
    }

    // Get unified cache and check if it has enough capacity
    sycl::queue *              stream = ctx.stream();
    ggml_sycl::unified_cache * cache  = ggml_sycl::get_unified_cache(*stream);
    if (!cache) {
        GGML_LOG_ERROR("[GRAPH-PRELOAD] Failed to get unified cache\n");
        return false;
    }

    // Check available memory vs needed
    const int64_t expert_size  = ggml_row_size(first_moe_src0->type, first_moe_src0->ne[0]) * first_moe_src0->ne[1];
    size_t        needed_bytes = total_experts_needed * static_cast<size_t>(expert_size);

    if (cache->available() < needed_bytes && cache->used() == 0) {
        // Fresh cache with insufficient budget - warn and disable graphs
        GGML_LOG_WARN(
            "[GRAPH-PRELOAD] Unified cache budget %.1f MB < %.1f MB needed for %zu experts, disabling graphs\n",
            cache->budget() / (1024.0f * 1024.0f), needed_bytes / (1024.0f * 1024.0f), total_experts_needed);
        return false;
    }

    int preload_count = 0;
    int pin_count     = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        const ggml_tensor * src0 = node->src[0];
        if (!src0 || !src0->buffer) {
            continue;
        }

        // Only process mmap'd/host buffers (lazy-moe case)
        bool is_sycl_buf = ggml_backend_buffer_is_sycl(src0->buffer);
        if (is_sycl_buf) {
            continue;  // Already on device, no caching needed
        }

        const int64_t n_experts = src0->ne[2];
        const int64_t expert_sz = ggml_row_size(src0->type, src0->ne[0]) * src0->ne[1];
        const char *  src0_data = static_cast<const char *>(src0->data);

        // Extract layer_id from tensor name (e.g., "blk.5.ffn_gate_exps")
        int layer_id = parse_layer_id_from_name(src0->name);
        if (layer_id < 0) {
            GGML_LOG_WARN("[GRAPH-PRELOAD] Could not parse layer_id from tensor '%s'\n", src0->name);
            continue;
        }

        GGML_SYCL_DEBUG("[GRAPH-PRELOAD] Pre-loading layer %d: %ld experts x %.1f MB\n", layer_id, (long) n_experts,
                        expert_sz / (1024.0f * 1024.0f));

        // Pre-load all experts for this layer via unified cache
        for (int64_t expert_id = 0; expert_id < n_experts; expert_id++) {
            const void * expert_ptr = src0_data + expert_id * expert_sz;

            void * cached_ptr;
            if (src0->type == GGML_TYPE_MXFP4) {
                cached_ptr = ggml_sycl::cache_moe_expert_with_reorder(
                    *stream, expert_ptr, static_cast<size_t>(expert_sz), layer_id, static_cast<int>(expert_id),
                    static_cast<int>(src0->ne[0]), static_cast<int>(src0->ne[1]));
            } else {
                cached_ptr = ggml_sycl::cache_moe_expert(*stream, expert_ptr, static_cast<size_t>(expert_sz), layer_id,
                                                         static_cast<int>(expert_id));
            }

            if (!cached_ptr) {
                GGML_LOG_ERROR("[GRAPH-PRELOAD] Failed to cache expert L%d:E%ld\n", layer_id, (long) expert_id);
                return false;
            }

            // Pin the entry to prevent eviction during graph execution
            ggml_sycl::pin_expert(expert_ptr);
            pin_count++;
            preload_count++;
        }
    }

    if (preload_count > 0) {
        GGML_LOG_INFO("[GRAPH-PRELOAD] Pre-loaded %d experts, pinned %d entries (%.1f%% hit rate)\n", preload_count,
                      pin_count, cache->hit_rate() * 100.0f);
    }

    return true;
}

// Unpin all expert cache slots after graph execution
static void graph_unpin_moe_experts([[maybe_unused]] ggml_backend_sycl_context * ctx = nullptr) {
    // Unified cache handles all expert caching
    ggml_sycl::unpin_all_experts();
    GGML_SYCL_DEBUG("[GRAPH-UNPIN] Unpinned unified cache entries\n");
}

// Pre-load all dense weights into unified cache before graph recording (weight streaming mode)
// This ensures stable cache pointers during graph execution
// Returns true if all weights were successfully pre-loaded and pinned
static bool graph_preload_weights(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Check if weight streaming is enabled via env var
    static bool weight_streaming_enabled = (std::getenv("GGML_SYCL_WEIGHT_STREAMING") != nullptr);

    // Skip entirely if streaming not enabled or unified cache disabled
    if (!weight_streaming_enabled || !ggml_sycl::unified_cache_enabled()) {
        return true;
    }

    // Collect all MUL_MAT nodes with CPU buffers that need streaming
    std::vector<const ggml_tensor *> cpu_weights;
    size_t                           total_weight_bytes = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        const ggml_tensor * src0 = node->src[0];
        if (!src0 || !src0->buffer) {
            continue;
        }

        // Only process CPU (mmap'd) buffers
        if (ggml_backend_buffer_is_sycl(src0->buffer)) {
            continue;
        }

        cpu_weights.push_back(src0);
        total_weight_bytes += ggml_nbytes(src0);
    }

    if (cpu_weights.empty()) {
        GGML_SYCL_DEBUG("[GRAPH-PRELOAD-WEIGHTS] No CPU MUL_MAT weights found in graph\n");
        return true;
    }

    GGML_SYCL_DEBUG("[GRAPH-PRELOAD-WEIGHTS] Found %zu CPU weights (%.1f MB)\n", cpu_weights.size(),
                    total_weight_bytes / (1024.0f * 1024.0f));

    // Get unified cache
    sycl::queue *              stream = ctx.stream();
    ggml_sycl::unified_cache * cache  = ggml_sycl::get_unified_cache(*stream);
    if (!cache) {
        GGML_LOG_WARN("[GRAPH-PRELOAD-WEIGHTS] Unified cache not available\n");
        return true;
    }

    // Check if we have enough space
    if (cache->available() < total_weight_bytes) {
        // Try to evict to make room
        cache->evict(total_weight_bytes - cache->available());
        if (cache->available() < total_weight_bytes) {
            GGML_LOG_WARN("[GRAPH-PRELOAD-WEIGHTS] Insufficient cache space: need %.1f MB, have %.1f MB\n",
                          total_weight_bytes / (1024.0f * 1024.0f), cache->available() / (1024.0f * 1024.0f));
            ctx.weight_streaming_graphs_disabled = true;
            return false;
        }
    }

    // Pre-load and pin all weights
    size_t loaded = 0;
    for (const ggml_tensor * weight : cpu_weights) {
        int    layer_id   = extract_layer_number(weight->name);
        // For mmap'd weights, use same pointer for key and source
        // (mmap pointers are stable and content doesn't change)
        void * cached_ptr = cache->ensure_cached(weight->data, weight->data, ggml_nbytes(weight),
                                                 ggml_sycl::cache_entry_type::DENSE_WEIGHT, layer_id, -1);

        if (cached_ptr) {
            cache->pin(weight->data);
            loaded++;
        } else {
            GGML_LOG_WARN("[GRAPH-PRELOAD-WEIGHTS] Failed to cache weight: %s\n", weight->name);
            ctx.weight_streaming_graphs_disabled = true;
            return false;
        }
    }

    GGML_SYCL_DEBUG("[GRAPH-PRELOAD-WEIGHTS] Pre-loaded and pinned %zu weights\n", loaded);
    return true;
}

// Unpin all dense weights after graph execution
static void graph_unpin_weights(ggml_backend_sycl_context * ctx) {
    // Unified cache handles unpinning via unpin_all()
    // Called during graph invalidation to allow eviction
    GGML_SYCL_DEBUG("[GRAPH-UNPIN-WEIGHTS] Unpin request (unified cache handles via unpin_all)\n");
}

// Pre-reorder ALL eligible weight tensors in the graph
// This ensures subsequent graph recordings don't get blocked by incremental reordering
// NOTE: We don't check should_reorder_tensor() because that requires ne[1]==1 (decode mode).
// We want to pre-reorder during prompt phase (ne[1]>1) to avoid blocking decode phase graphs.
static void graph_pre_reorder_all(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Skip if reordering is disabled or device doesn't support it
    if (!ggml_sycl_reorder_enabled() || !ctx.supports_soa_reorder) {
        return;
    }

    // Skip graph pre-reorder when weight streaming is enabled
    // With weight streaming, weights are copied to compute buffer on-demand.
    // Reordering the compute buffer copy would:
    // 1. Allocate extra GPU memory (OOM on large models)
    // 2. Only reorder the temporary copy (original mmap'd data stays unchanged)
    // 3. Be wasteful since reordering should happen during caching (Phase 3)
    static bool weight_streaming_enabled = (std::getenv("GGML_SYCL_WEIGHT_STREAMING") != nullptr);
    if (weight_streaming_enabled) {
        GGML_SYCL_DEBUG("[SYCL-GRAPH] weight streaming enabled, skipping pre-reorder\n");
        return;
    }

    int reorder_count    = 0;
    int moe_tensor_count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        // Handle both MUL_MAT and MUL_MAT_ID (MoE expert weights)
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        const ggml_tensor * src0 = node->src[0];
        if (!src0 || !src0->extra) {
            if (g_ggml_sycl_debug && node->op == GGML_OP_MUL_MAT_ID) {
                fprintf(stderr, "[SYCL-GRAPH] MUL_MAT_ID node %d: src0=%p, extra=%p\n", i, (void *) src0,
                        src0 ? src0->extra : nullptr);
            }
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT_ID) {
            moe_tensor_count++;
            if (g_ggml_sycl_debug && moe_tensor_count <= 3) {
                fprintf(stderr, "[SYCL-GRAPH] MUL_MAT_ID node %d: src0='%s' type=%s\n", i, src0->name,
                        ggml_type_name(src0->type));
            }
        }
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra->optimized_feature.is_reordered()) {
            if (g_ggml_sycl_debug >= 2 && node->op == GGML_OP_MUL_MAT_ID && moe_tensor_count <= 3) {
                fprintf(stderr, "[SYCL-GRAPH] MUL_MAT_ID src0='%s' - already reordered (mode=%d)\n", src0->name,
                        (int) extra->optimized_feature.get_reorder());
            }
            continue;  // Already reordered
        }
        // Skip CPU-resident tensors (weight streaming: weights on CPU are copied on-demand)
        // Reordering requires GPU-resident data; CPU tensors will be reordered after caching
        if (src0->buffer) {
            bool is_sycl = ggml_backend_buffer_is_sycl(src0->buffer);
            if (!is_sycl) {
                if (g_ggml_sycl_debug >= 2) {
                    fprintf(stderr, "[SYCL-GRAPH] src0='%s' - CPU buffer, skip reorder\n", src0->name);
                }
                continue;
            }
        } else {
            // No buffer means tensor data is not from a ggml buffer (e.g., view or direct allocation)
            // For weight streaming with mmap, weights may have buffer=NULL if they haven't been
            // transferred to a ggml buffer yet. Skip reordering for such tensors.
            if (g_ggml_sycl_debug >= 2) {
                fprintf(stderr, "[SYCL-GRAPH] src0='%s' - no buffer, skip reorder (data=%p)\n", src0->name, src0->data);
            }
            continue;
        }
        // Skip TP-sharded tensors
        if (extra->tp_type != tp_layer_type::TP_NONE) {
            if (g_ggml_sycl_debug >= 2 && node->op == GGML_OP_MUL_MAT_ID && moe_tensor_count <= 3) {
                fprintf(stderr, "[SYCL-GRAPH] MUL_MAT_ID src0='%s' - TP sharded\n", src0->name);
            }
            continue;
        }
        // Check if type supports reorder
        if (!ggml_sycl_supports_reorder_mmvq(src0->type) && !ggml_sycl_supports_reorder_dmmv(src0->type) &&
            !ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
            if (g_ggml_sycl_debug && node->op == GGML_OP_MUL_MAT_ID) {
                fprintf(stderr, "[SYCL-GRAPH] MUL_MAT_ID src0='%s' type=%s - not supported\n", src0->name,
                        ggml_type_name(src0->type));
            }
            continue;
        }
        if (g_ggml_sycl_debug >= 2 && node->op == GGML_OP_MUL_MAT_ID && moe_tensor_count <= 3) {
            fprintf(stderr, "[SYCL-GRAPH] MUL_MAT_ID src0='%s' type=%s - will reorder\n", src0->name,
                    ggml_type_name(src0->type));
        }
        // Perform reorder using unified function (does both transform + set flag)
        // Always do SoA here; coalesced conversion is handled later per-kernel eligibility.
        bool converted = reorder_tensor_to_soa(src0, ctx.stream(), "GRAPH_PRE_REORDER");
        if (converted) {
            if (g_ggml_sycl_debug && node->op == GGML_OP_MUL_MAT_ID) {
                fprintf(stderr, "[SYCL-GRAPH] SET reorder=%s: src0='%s' extra=%p reorder_mode=%d\n",
                        g_ggml_sycl_reorder_mode == reorder_mode::COALESCED ? "COALESCED" : "SOA", src0->name,
                        (void *) extra, (int) extra->optimized_feature.get_reorder());
            }
            reorder_count++;
        }
    }
    if (reorder_count > 0) {
        // Wait for all reorders to complete before proceeding
        ctx.stream()->wait();
        GGML_SYCL_DEBUG("[SYCL-GRAPH] pre-reordered %d tensors\n", reorder_count);
    }

    // Coalesced conversion is handled separately by graph_convert_to_coalesced()
}

static void opt_for_reorder(ggml_backend_sycl_context * ctx,
                            const ggml_tensor *         src0,
                            const ggml_tensor * /* src1 */,
                            ggml_tensor * dst,
                            mul_mat_algo  mm_algorithm) {
    // Skip reordering during SYCL graph recording - wait() calls are forbidden.
    // Tensors should be pre-reordered via graph_pre_reorder_all() before recording starts.
    if (g_ggml_sycl_graph_recording) {
        return;
    }

    // Use can_reorder_tensor() instead of should_reorder_tensor()
    // We want to reorder on FIRST USE regardless of batch size (prompt or decode).
    // The batch=1 check in should_reorder_tensor() only determines which kernel to use,
    // not whether to reorder. Reordering must happen before any SoA kernel can be used.
    if (!can_reorder_tensor(*ctx, dst)) {
        if (g_ggml_sycl_debug) {
            fprintf(stderr, "[REORDER-DBG] opt_for_reorder: can_reorder=false, skipping\n");
        }
        return;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra) {
        if (g_ggml_sycl_debug) {
            fprintf(stderr, "[REORDER-DBG] opt_for_reorder: extra=NULL - skipping\n");
        }
        return;  // Skip tensors without extra (e.g., permutations)
    }

    // Check if already reordered to a valid format
    // MMVQ/DMMV support both SOA and COALESCED modes
    const reorder_mode current_mode = extra->optimized_feature.get_reorder();
    const bool is_valid_reorder     = (current_mode == reorder_mode::SOA || current_mode == reorder_mode::COALESCED);
    if (g_ggml_sycl_debug) {
        fprintf(stderr, "[REORDER-DBG] opt_for_reorder: tensor='%s' extra=%p current_mode=%d is_valid=%d\n", src0->name,
                (void *) extra, (int) current_mode, (int) is_valid_reorder);
    }
    if (is_valid_reorder) {
        if (g_ggml_sycl_debug) {
            fprintf(stderr, "[REORDER-DBG] opt_for_reorder: already in valid mode %d - skipping\n", (int) current_mode);
            // Note: Data verification removed - incompatible with SYCL graph recording
        }
        if (current_mode == reorder_mode::SOA && g_ggml_sycl_reorder_mode == reorder_mode::COALESCED) {
            const bool allow_coalesced =
                (mm_algorithm == mul_mat_algo::MMVQ) ||
                (mm_algorithm == mul_mat_algo::MMQ && ggml_sycl_supports_coalesced_mmq(src0->type)) ||
                (mm_algorithm == mul_mat_algo::DMMV && ggml_sycl_supports_coalesced_dmmv(src0->type));
            if (allow_coalesced) {
                bool converted = false;
                switch (src0->type) {
                    case GGML_TYPE_Q4_0:
                        converted = ggml_sycl_convert_to_coalesced_q4_0(src0, ctx->stream());
                        break;
                    case GGML_TYPE_Q8_0:
                        converted = ggml_sycl_convert_to_coalesced_q8_0(src0, ctx->stream());
                        break;
                    case GGML_TYPE_Q6_K:
                        converted = ggml_sycl_convert_to_coalesced_q6_k(src0, ctx->stream());
                        break;
                    case GGML_TYPE_MXFP4:
                        converted = ggml_sycl_convert_to_coalesced_mxfp4(src0, ctx->stream());
                        break;
                    default:
                        break;
                }
                if (std::getenv("GGML_SYCL_MMQ_DEBUG")) {
                    static int coalesce_debug_valid = 0;
                    if (coalesce_debug_valid++ < 10) {
                        fprintf(stderr, "[REORDER-DBG] opt_for_reorder coalesced=%d type=%d (from SOA)\n",
                                converted ? 1 : 0, (int) src0->type);
                    }
                }
            }
        }
        return;  // Already in correct format (or coalesced conversion attempted)
    }

    // If reordered to an INVALID format, warn and skip (re-reordering not yet supported)
    // Valid formats are SOA (1) and COALESCED (2)
    if (extra->optimized_feature.is_reordered()) {
        static int reorder_mismatch_warn = 0;
        if (reorder_mismatch_warn++ < 5) {
            fprintf(stderr,
                    "[REORDER-WARN] opt_for_reorder: tensor '%s' is in unknown mode %d (expected SOA=1 or COALESCED=2) "
                    "- cannot use\n",
                    src0->name, (int) extra->optimized_feature.get_reorder());
        }
        return;  // Cannot use unknown format
    }

    // CRITICAL: Skip reorder for TP-sharded tensors - reorder corrupts memory in TP mode!
    // The reorder operation uses src0->data and ggml_nbytes which may not match the sharded layout
    if (extra->tp_type != tp_layer_type::TP_NONE) {
        static int tp_skip_log = 0;
        if (g_ggml_sycl_tp_debug && tp_skip_log++ < 5) {
            fprintf(stderr, "TP DEBUG: Skipping reorder for TP-sharded tensor %s (tp_type=%d)\n", src0->name,
                    (int) extra->tp_type);
        }
        return;  // Skip reorder for TP tensors
    }

    switch (mm_algorithm) {
        case mul_mat_algo::DMMV:
            if (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MMVQ:
            if (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MUL_MAT_SYCL:
            if (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MMQ:
            if (!ggml_sycl_supports_reorder_mmq(src0->type)) {
                return;
            }
            break;
    }

    static bool do_soa_layout_debug = std::getenv("GGML_SYCL_SOA_LAYOUT_DEBUG") != nullptr;
    static int  reorder_call_count  = 0;

    if (g_ggml_sycl_debug || do_soa_layout_debug) {
        fprintf(stderr, "[REORDER-DBG] opt_for_reorder: CALLING reorder_qw for src0=%s type=%d data=%p\n", src0->name,
                src0->type, src0->data);
    }

    // Debug: dump data BEFORE and AFTER reorder
    uint8_t before_bytes[36]  = { 0 };
    bool    do_layout_compare = do_soa_layout_debug;  // No limit - pipe to file
    if (do_layout_compare) {
        reorder_call_count++;
        ctx->stream()->wait();
        ctx->stream()->memcpy(before_bytes, src0->data, 36).wait();
        sycl::half d0_before, d1_before;
        memcpy(&d0_before, &before_bytes[0], 2);
        memcpy(&d1_before, &before_bytes[18], 2);
        fprintf(stderr, "[REORDER-DBG] BEFORE reorder_qw (tensor=%s):\n", src0->name);
        fprintf(stderr, "[REORDER-DBG]   Raw bytes: ");
        for (int i = 0; i < 36; i++) {
            fprintf(stderr, "%02x ", before_bytes[i]);
            if (i == 17) {
                fprintf(stderr, "| ");
            }
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "[REORDER-DBG]   IF AoS: d[0]=%.6f d[1]=%.6f\n", (float) d0_before, (float) d1_before);
    }

    // Use unified function (does both transform + set flag)
    const bool allow_coalesced = (mm_algorithm == mul_mat_algo::MMVQ) ||
                                 (mm_algorithm == mul_mat_algo::MMQ && ggml_sycl_supports_coalesced_mmq(src0->type)) ||
                                 (mm_algorithm == mul_mat_algo::DMMV && ggml_sycl_supports_coalesced_dmmv(src0->type));
    if (g_ggml_sycl_reorder_mode == reorder_mode::COALESCED && allow_coalesced) {
        convert_tensor_to_coalesced(src0, ctx->stream(), "OPT_FOR_REORDER");
        return;
    }

    reorder_tensor_to_soa(src0, ctx->stream(), "OPT_FOR_REORDER");

    // If coalesced mode is requested, convert immediately after SoA reorder.
    if (g_ggml_sycl_reorder_mode == reorder_mode::COALESCED) {
        const bool allow_coalesced =
            (mm_algorithm == mul_mat_algo::MMVQ) ||
            (mm_algorithm == mul_mat_algo::MMQ && ggml_sycl_supports_coalesced_mmq(src0->type)) ||
            (mm_algorithm == mul_mat_algo::DMMV && ggml_sycl_supports_coalesced_dmmv(src0->type));
        if (allow_coalesced) {
            bool converted = false;
            switch (src0->type) {
                case GGML_TYPE_Q4_0:
                    converted = ggml_sycl_convert_to_coalesced_q4_0(src0, ctx->stream());
                    break;
                case GGML_TYPE_Q8_0:
                    converted = ggml_sycl_convert_to_coalesced_q8_0(src0, ctx->stream());
                    break;
                case GGML_TYPE_Q6_K:
                    converted = ggml_sycl_convert_to_coalesced_q6_k(src0, ctx->stream());
                    break;
                case GGML_TYPE_MXFP4:
                    converted = ggml_sycl_convert_to_coalesced_mxfp4(src0, ctx->stream());
                    break;
                default:
                    break;
            }
            if (std::getenv("GGML_SYCL_MMQ_DEBUG")) {
                static int coalesce_debug = 0;
                if (coalesce_debug++ < 10) {
                    fprintf(stderr, "[REORDER-DBG] opt_for_reorder coalesced=%d type=%d\n", converted ? 1 : 0,
                            (int) src0->type);
                }
            }
        }
    }

    if (do_layout_compare) {
        ctx->stream()->wait();
        uint8_t after_bytes[36];
        ctx->stream()->memcpy(after_bytes, src0->data, 36).wait();
        sycl::half d0_after, d1_after;
        memcpy(&d0_after, &after_bytes[0], 2);
        memcpy(&d1_after, &after_bytes[18], 2);
        fprintf(stderr, "[REORDER-DBG] AFTER reorder_qw (tensor=%s):\n", src0->name);
        fprintf(stderr, "[REORDER-DBG]   Raw bytes: ");
        for (int i = 0; i < 36; i++) {
            fprintf(stderr, "%02x ", after_bytes[i]);
            if (i == 17) {
                fprintf(stderr, "| ");
            }
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "[REORDER-DBG]   IF AoS: d[0]=%.6f d[1]=%.6f (should be garbage if reordered)\n",
                (float) d0_after, (float) d1_after);

        // Read d values from expected SoA location
        int64_t    ncols    = src0->ne[0];
        int64_t    nrows    = src0->ne[1];
        int64_t    d_offset = nrows * ncols / 2;
        sycl::half soa_d[2];
        ctx->stream()->memcpy(soa_d, (const uint8_t *) src0->data + d_offset, 4).wait();
        fprintf(stderr, "[REORDER-DBG]   IF SoA: d[0]=%.6f d[1]=%.6f (at d_offset=%lld)\n", (float) soa_d[0],
                (float) soa_d[1], (long long) d_offset);

        bool bytes_changed = memcmp(after_bytes, before_bytes, 36) != 0;
        fprintf(stderr, "[REORDER-DBG]   *** DATA CHANGED: %s *** (call #%d)\n\n",
                bytes_changed ? "YES (reorder happened)" : "NO (BUG - reorder didn't change data!)",
                reorder_call_count);
        fflush(stderr);
    }

    if (g_ggml_sycl_debug) {
        fprintf(stderr, "[REORDER-DBG] opt_for_reorder: DONE reorder_qw, set reorder=SOA\n");
    }
}

// Pre-reorder all weight tensors that would be reordered during decode.
// This ensures consistent behavior from the first decode token and fixes
// non-determinism when llama graph reuse is enabled.
// Without this, the first decode token uses non-reordered kernels while
// subsequent tokens use reordered kernels, causing different results.
static void pre_reorder_all_tensors(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    // Skip if reordering is disabled or device doesn't support it
    if (!ggml_sycl_reorder_enabled() || !sycl_ctx->supports_soa_reorder) {
        return;
    }

    int reordered_count = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        ggml_tensor * src0 = node->src[0];  // weight tensor

        if (!src0 || !src0->extra) {
            continue;
        }

        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);

        if (extra->optimized_feature.is_reordered()) {
            continue;  // Already reordered
        }

        // Check if this type supports reordering (any algorithm)
        if (!ggml_sycl_supports_reorder_mmvq(src0->type) && !ggml_sycl_supports_reorder_dmmv(src0->type) &&
            !ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
            continue;
        }

        // Debug: dump data BEFORE and AFTER reorder
        static bool do_soa_layout_debug = std::getenv("GGML_SYCL_SOA_LAYOUT_DEBUG") != nullptr;
        uint8_t     before_bytes[36]    = { 0 };
        if (do_soa_layout_debug && reordered_count < 5) {
            sycl_ctx->stream()->wait();
            sycl_ctx->stream()->memcpy(before_bytes, src0->data, 36).wait();
            sycl::half d0_before;
            memcpy(&d0_before, &before_bytes[0], 2);
            fprintf(stderr, "[PRE-REORDER] BEFORE tensor=%s: bytes[0-5]=%02x %02x %02x %02x %02x %02x d[0]=%.6f\n",
                    src0->name, before_bytes[0], before_bytes[1], before_bytes[2], before_bytes[3], before_bytes[4],
                    before_bytes[5], (float) d0_before);
        }

        // Reorder using unified function (does both transform + set flag)
        // Always do SoA here; coalesced conversion is handled later per-kernel eligibility.
        bool converted = reorder_tensor_to_soa(src0, sycl_ctx->stream(), "PRE_REORDER_ALL");
        if (converted) {
            reordered_count++;
        }

        // Debug: verify data changed
        if (do_soa_layout_debug && reordered_count <= 5) {
            sycl_ctx->stream()->wait();
            uint8_t after_bytes[36];
            sycl_ctx->stream()->memcpy(after_bytes, src0->data, 36).wait();
            bool changed = memcmp(before_bytes, after_bytes, 36) != 0;

            // Read d values from expected SoA location
            int64_t    ncols    = src0->ne[0];
            int64_t    nrows    = src0->ne[1];
            int64_t    d_offset = nrows * ncols / 2;
            sycl::half soa_d[2];
            sycl_ctx->stream()->memcpy(soa_d, (const uint8_t *) src0->data + d_offset, 4).wait();

            fprintf(stderr, "[PRE-REORDER] AFTER tensor=%s: bytes[0-5]=%02x %02x %02x %02x %02x %02x\n", src0->name,
                    after_bytes[0], after_bytes[1], after_bytes[2], after_bytes[3], after_bytes[4], after_bytes[5]);
            fprintf(stderr, "[PRE-REORDER]   d_offset=%lld SoA_d[0]=%.6f SoA_d[1]=%.6f\n", (long long) d_offset,
                    (float) soa_d[0], (float) soa_d[1]);
            fprintf(stderr, "[PRE-REORDER]   *** DATA CHANGED: %s ***\n\n",
                    changed ? "YES (reorder worked)" : "NO (BUG!)");
            fflush(stderr);
        }
    }

    if (reordered_count > 0) {
        // Wait for all reordering to complete before proceeding
        sycl_ctx->stream()->wait();
        if (std::getenv("GGML_SYCL_SOA_LAYOUT_DEBUG")) {
            fprintf(stderr, "[PRE-REORDER] Completed reordering %d tensors\n", reordered_count);
        }
    }
}

static bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % GGML_SYCL_DMMV_X == 0 && src1->ne[1] == 1;
}

static bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // Q6_K MMQ kernel now works on Intel GPUs (WARP_SIZE=16) after fixing:
    // - Main loop stride (blocks_per_iter instead of blocks_per_warp)
    // - Y-tile allocation (QI6_K instead of WARP_SIZE)
    // - Y-tile indexing (no modulo wraparound)
    // - vec_dot Y-tile stride (QI6_K instead of WARP_SIZE)
    return ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx,
                              const ggml_tensor *         src0,
                              const ggml_tensor *         src1,
                              ggml_tensor *               dst) {
    GGML_SYCL_PROFILE_SCOPE_GEMM("mul_mat");
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // DEBUG: Check if TP sharded weights have correct dimensions
    if (is_tp_sharded_tensor(src0)) {
        static int tp_mm_dbg = 0;
        if (g_ggml_sycl_tp_debug && tp_mm_dbg++ < 10) {
            const auto * extra       = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
            const char * name        = src0->name;
            bool         is_attn_out = strstr(name, "attn_output") != nullptr;
            fprintf(stderr,
                    "TP DEBUG MUL_MAT %s: src0->ne=[%lld,%lld], tp_local_ne=[%lld,%lld], "
                    "src1->ne=[%lld,%lld,%lld,%lld], dst->ne=[%lld,%lld], device=%d\n",
                    name, (long long) src0->ne[0], (long long) src0->ne[1], (long long) extra->tp_local_ne[0],
                    (long long) extra->tp_local_ne[1], (long long) src1->ne[0], (long long) src1->ne[1],
                    (long long) src1->ne[2], (long long) src1->ne[3], (long long) dst->ne[0], (long long) dst->ne[1],
                    ctx.device);
            // For attn_output, also check src1 data values
            if (is_attn_out && ctx.device == 0) {
                queue_ptr stream = ctx.stream();
                float     sample[4];
                stream->memcpy(sample, src1->data, 4 * sizeof(float)).wait();
                fprintf(stderr, "TP DEBUG ATTN_OUT src1[0..3] = [%f, %f, %f, %f]\n", sample[0], sample[1], sample[2],
                        sample[3]);
            }
        }
    }

    // DEBUG: Check output.weight (lm_head) computation specifically
    // Make sure we match "output.weight" but NOT "attn_output.weight"
    // Debug controlled by GGML_SYCL_TP_DEBUG environment variable
    bool is_output_weight =
        src0->name[0] && strstr(src0->name, "output.weight") != nullptr && strstr(src0->name, "attn_output") == nullptr;
    static int output_dbg = 0;
    if (g_ggml_sycl_tp_debug && is_output_weight && output_dbg++ < 3) {
        const char * name   = src0->name;
        queue_ptr    stream = ctx.stream();
        int64_t      batch  = src1->ne[1];
        fprintf(stderr,
                "TP DEBUG LM_HEAD %s: src0->ne=[%lld,%lld], src1->ne=[%lld,%lld], dst->ne=[%lld,%lld], batch=%lld, "
                "device=%d\n",
                name, (long long) src0->ne[0], (long long) src0->ne[1], (long long) src1->ne[0],
                (long long) src1->ne[1], (long long) dst->ne[0], (long long) dst->ne[1], (long long) batch, ctx.device);
        // Check input (hidden state after output_norm)
        float input_sample[8];
        stream->memcpy(input_sample, src1->data, 8 * sizeof(float)).wait();
        float sum     = 0;
        bool  has_nan = false;
        for (int i = 0; i < 8; i++) {
            sum += input_sample[i];
            if (std::isnan(input_sample[i])) {
                has_nan = true;
            }
        }
        fprintf(stderr,
                "TP DEBUG LM_HEAD batch=%lld input [0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] sum=%.4f nan=%d\n",
                (long long) batch, input_sample[0], input_sample[1], input_sample[2], input_sample[3], input_sample[4],
                input_sample[5], input_sample[6], input_sample[7], sum, has_nan);

        // For batch>1 (prompt processing), also check position 1's hidden state (this determines next token)
        if (batch > 1) {
            float  pos1_sample[8];
            size_t offset = src1->ne[0] * sizeof(float);  // Skip to position 1
            stream->memcpy(pos1_sample, (char *) src1->data + offset, 8 * sizeof(float)).wait();
            float sum1     = 0;
            bool  has_nan1 = false;
            for (int i = 0; i < 8; i++) {
                sum1 += pos1_sample[i];
                if (std::isnan(pos1_sample[i])) {
                    has_nan1 = true;
                }
            }
            fprintf(
                stderr,
                "TP DEBUG LM_HEAD batch=%lld pos=1 [0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] sum=%.4f nan=%d\n",
                (long long) batch, pos1_sample[0], pos1_sample[1], pos1_sample[2], pos1_sample[3], pos1_sample[4],
                pos1_sample[5], pos1_sample[6], pos1_sample[7], sum1, has_nan1);
        }
    }

    // Check for TP-sharded weight tensor and log the operation
    ggml_sycl_mul_mat_tp_pre(src0);

    const bool split                  = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t    min_compute_capability = INT_MAX;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_sycl_info().device_count; ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_count ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } else {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);

    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    bool use_mul_mat_q =
        ggml_sycl_supports_mmq(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // Force specific kernel paths for isolated debugging
    // GGML_SYCL_FORCE_MMQ: Route all ops through MMQ (works for any batch size)
    // GGML_SYCL_FORCE_DMMV: Route batch=1 ops through DMMV (disables MMVQ)
    static bool force_mmq  = (getenv("GGML_SYCL_FORCE_MMQ") != nullptr);
    static bool force_dmmv = (getenv("GGML_SYCL_FORCE_DMMV") != nullptr);
    if (force_mmq && use_mul_mat_q) {
        use_dequantize_mul_mat_vec = false;
        use_mul_mat_vec_q          = false;
    }
    if (force_dmmv && use_dequantize_mul_mat_vec) {
        use_mul_mat_vec_q = false;  // DMMV takes priority over MMVQ for batch=1
    }

#ifdef GGML_SYCL_XMX_GEMM
    // XMX GEMM path (experimental, known to be 5-11x slower for quantized models)
    bool use_xmx_gemm = g_ggml_sycl_use_xmx_gemm ? true : false;
    if (use_xmx_gemm) {
        use_xmx_gemm = ggml_sycl_xmx_available() && ggml_sycl_xmx_supports_type(src0->type);
    }
    if (use_xmx_gemm) {
        // XMX requires K dimension to be divisible by XMX_K (32)
        const int64_t ncols_x = src0->ne[0];  // K dimension
        constexpr int XMX_K   = 32;
        use_xmx_gemm          = (ncols_x % XMX_K) == 0;
    }
    if (use_xmx_gemm) {
        int64_t batch = src1->ne[1];
        // XMX is beneficial for batch >= 1 and < threshold (DEBUG)
        use_xmx_gemm  = batch >= 1 && batch < g_ggml_sycl_xmx_threshold;
    }
#else
    bool use_xmx_gemm = false;
#endif

    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggerganov/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);

#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif  // SYCL_USE_XMX

    // DEBUG: Print path selection for Q4_0 (AFTER all checks)
    static int path_debug_count   = 0;
    static int batch1_debug_count = 0;
    if (src0->type == GGML_TYPE_Q4_0 && std::getenv("GGML_SYCL_MMQ_DEBUG")) {
        if (src1->ne[1] == 1 && batch1_debug_count < 5) {
            fprintf(stderr, "[MUL_MAT PATH] Q4_0 batch=1: dmmv=%d mmvq=%d mmq=%d xmx=%d\n", use_dequantize_mul_mat_vec,
                    use_mul_mat_vec_q, use_mul_mat_q, use_xmx_gemm);
            fflush(stderr);
            batch1_debug_count++;
        } else if (src1->ne[1] > 1 && path_debug_count < 5) {
            fprintf(stderr, "[MUL_MAT PATH] Q4_0: dmmv=%d mmvq=%d mmq=%d xmx=%d batch=%ld MAX_BATCH=%d\n",
                    use_dequantize_mul_mat_vec, use_mul_mat_vec_q, use_mul_mat_q, use_xmx_gemm, (long) src1->ne[1],
                    MMQ_MAX_BATCH_SIZE);
            fflush(stderr);
            path_debug_count++;
        }
    }

    // mmvq path is faster in the CUDA backend and on Intel with weight reordering enabled.
    // On Intel, MMVQ outperforms DMMV for Q4_0 decode (41.66 vs 31.42 t/s on Arc A770).
    if (!g_ggml_sycl_prioritize_dmmv &&
        (ctx.stream()->get_backend() == sycl::backend::ext_oneapi_cuda ||
         (should_reorder_tensor(ctx, dst) && ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
    }

    // DEBUG: Log path selection for FFN layers (only in TP mode)
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int   path_dbg[32]     = { 0 };
    static int   weight_check_dbg = 0;
    const char * mm_name          = src0->name;
    // Skip this debug block in multi-process mode (only one device per process)
    if (g_ggml_sycl_tp_debug && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1 &&
        !g_sycl_tp_config.is_multiprocess) {
        for (int l = 30; l <= 31; l++) {
            char gate_name[64];
            snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gate", l);
            if (strstr(mm_name, gate_name) && src1->ne[1] == 1 && path_dbg[l]++ < 2) {
                fprintf(stderr, "TP DEBUG PATH L%d FFN_GATE: use_dmmv=%d use_mmvq=%d use_mmq=%d use_xmx=%d\n", l,
                        use_dequantize_mul_mat_vec, use_mul_mat_vec_q, use_mul_mat_q, use_xmx_gemm);

                // DEBUG: Check weight data at START of mul_mat before any kernels
                if (l == 31 && weight_check_dbg++ < 3) {
                    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
                    if (extra) {
                        struct {
                            sycl::half d;
                            uint8_t    qs[16];
                        } wblk;

                        for (int dev = 0; dev < g_sycl_tp_config.world_size; dev++) {
                            int dev_id = g_sycl_tp_config.devices[dev];
                            if (extra->data_device[dev_id]) {
                                ggml_sycl_set_device(dev_id);
                                queue_ptr dev_stream = ctx.stream(dev_id, 0);
                                dev_stream->memcpy(&wblk, extra->data_device[dev_id], sizeof(wblk)).wait();
                                float d_f = (float) wblk.d;
                                int   v0  = (wblk.qs[0] & 0xF) - 8;
                                int   v1  = (wblk.qs[0] >> 4) - 8;
                                fprintf(stderr,
                                        "TP DEBUG L31 WEIGHT_CHECK device=%d: ptr=%p d=%f qs[0]=0x%02x deq=[%f,%f]\n",
                                        dev_id, extra->data_device[dev_id], d_f, wblk.qs[0], v0 * d_f, v1 * d_f);
                            }
                        }
                        ggml_sycl_set_device(ctx.device);  // Restore device
                    }
                }
            }
        }
    }

    if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        if (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            GGML_SYCL_KTRACE("mul_mat_f16_kq_single", " ne3=%lld", (long long) src0->ne[3]);
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } else {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            GGML_SYCL_KTRACE("mul_mat_f16_batched", " ne3=%lld", (long long) src0->ne[3]);
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) &&
               src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        GGML_SYCL_KTRACE("mul_mat_f16_kqv_single", " ne1=%lld", (long long) src1->ne[1]);
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
               src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        GGML_SYCL_KTRACE("mul_mat_f16_kqkv_multi", " batches=%lld", (long long) (src1->ne[2] * src1->ne[3]));
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        GGML_SYCL_KTRACE("mul_mat_dispatch_dmmv", " type=%d ne1=%lld", src0->type, (long long) src1->ne[1]);
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } else if (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.is_reordered()) {
            // Both SOA and COALESCED use the SoA quantization path - MMVQ handles coalesced internally
            GGML_SYCL_KTRACE("mul_mat_dispatch_mmvq_reordered", " type=%d ne1=%lld mode=%d", src0->type,
                             (long long) src1->ne[1], (int) extra->optimized_feature.get_reorder());
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } else {
            GGML_SYCL_KTRACE("mul_mat_dispatch_mmvq_aos", " type=%d ne1=%lld", src0->type, (long long) src1->ne[1]);
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
#if defined(GGML_SYCL_XMX_GEMM) && defined(GGML_SYCL_MMQ_XMX)
    } else if (use_xmx_gemm) {
        // XMX-accelerated quantized GEMM (experimental, known to be 5-11x slower)
        GGML_SYCL_KTRACE("mul_mat_dispatch_mmq_xmx", " type=%d ne1=%lld", src0->type, (long long) src1->ne[1]);
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q_xmx);
#endif
    } else if (use_mul_mat_q) {
        // Standard MMQ path (with optional ESIMD acceleration for Q4_0)
        // MMQ SoA kernels use SoA for both X (weights) and Y (activations)
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMQ);
        ggml_tensor_extra_gpu * mmq_extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);

        // DEBUG: SoA investigation - trace kernel dispatch decision
        if (g_ggml_sycl_debug) {
            static int mmq_dbg_count = 0;
            if (mmq_dbg_count++ < 10) {
                fprintf(stderr, "[MMQ-DISPATCH] src0=%s type=%d extra=%p reorder_mode=%d batch=%lld\n", src0->name,
                        src0->type, (void *) mmq_extra,
                        mmq_extra ? (int) mmq_extra->optimized_feature.get_reorder() : -1, (long long) src1->ne[1]);
            }
        }

        if (mmq_extra && mmq_extra->optimized_feature.is_reordered()) {
            GGML_SYCL_KTRACE("mul_mat_dispatch_mmq_reordered", " type=%d ne1=%lld mode=%d", src0->type,
                             (long long) src1->ne[1], (int) mmq_extra->optimized_feature.get_reorder());
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
        } else {
            GGML_SYCL_KTRACE("mul_mat_dispatch_mmq_aos", " type=%d ne1=%lld", src0->type, (long long) src1->ne[1]);
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
        }
    } else {
        GGML_SYCL_KTRACE("mul_mat_dispatch_onednn", " type=%d ne1=%lld", src0->type, (long long) src1->ne[1]);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }

    // DEBUG: Check FFN gate/up output for NaN (layers 0, 30, 31)
    const char * weight_name  = src0->name;
    static int   gate_dbg[32] = { 0 };
    static int   call_seq     = 0;

    // Check which layer this is
    int  layer   = -1;
    bool is_gate = false;
    for (int l = 0; l <= 31; l++) {
        char gate_name[64], up_name[64];
        snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gate", l);
        snprintf(up_name, sizeof(up_name), "blk.%d.ffn_up", l);
        if (strstr(weight_name, gate_name)) {
            layer   = l;
            is_gate = true;
            break;
        }
        if (strstr(weight_name, up_name)) {
            layer   = l;
            is_gate = false;
            break;
        }
    }

    // Debug layers 0, 30, 31 (for batch=1 and batch=2 at layer 31)
    bool batch1_debug = g_ggml_sycl_tp_debug && layer >= 0 && dst->ne[1] == 1 &&
                        (layer == 0 || layer == 30 || layer == 31) && gate_dbg[layer]++ < 3;
    static int l31_batch2   = 0;
    bool       batch2_debug = g_ggml_sycl_tp_debug && layer == 31 && dst->ne[1] == 2 && l31_batch2++ < 3;
    bool       should_debug = batch1_debug || batch2_debug;
    if (should_debug) {
        call_seq++;
        ctx.stream()->wait();
        float out_vals[8];
        ctx.stream()->memcpy(out_vals, dst->data, 8 * sizeof(float)).wait();
        bool has_nan = false;
        for (int i = 0; i < 8; i++) {
            if (std::isnan(out_vals[i])) {
                has_nan = true;
            }
        }
        // Also check input (src1)
        float in_vals[4];
        ctx.stream()->memcpy(in_vals, src1->data, 4 * sizeof(float)).wait();
        bool                    in_nan     = std::isnan(in_vals[0]) || std::isnan(in_vals[1]);
        int64_t                 batch      = dst->ne[1];
        // Check weight pointer and first few bytes
        bool                    is_tp_buf  = ggml_backend_buffer_is_sycl_tp(src0->buffer);
        ggml_tensor_extra_gpu * extra      = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        void *                  weight_ptr = extra ? extra->data_device[0] : src0->data;
        fprintf(stderr,
                "TP DEBUG #%d L%d %s b=%lld: in=[%.3f,%.3f,%.3f,%.3f] out=[%.3f,%.3f,...] in_nan=%d out_nan=%d\n",
                call_seq, layer, is_gate ? "GATE" : "UP", (long long) batch, in_vals[0], in_vals[1], in_vals[2],
                in_vals[3], out_vals[0], out_vals[1], in_nan, has_nan);
        fprintf(stderr, "TP DEBUG #%d L%d weight: ptr=%p ne=[%lld,%lld] is_tp=%d\n", call_seq, layer, weight_ptr,
                (long long) src0->ne[0], (long long) src0->ne[1], is_tp_buf);
    }

    // DEBUG: Capture FFN down output BEFORE TP post-processing (works for single GPU too)
    static int ffn_mm_dbg           = 0;
    bool       is_ffn_down_l0       = strstr(weight_name, "blk.0.ffn_down") != nullptr;
    // Also detect by dimensions: FFN down has shape [14336 or 7168, 4096] - K=hidden, N=model_dim
    bool       is_ffn_down_by_shape = (src0->ne[0] == 14336 || src0->ne[0] == 7168) && src0->ne[1] == 4096;
    if (g_ggml_sycl_tp_debug && ffn_mm_dbg++ < 20 && (is_ffn_down_l0 || (is_ffn_down_by_shape && ffn_mm_dbg < 3))) {
        ctx.stream()->wait();
        float out_vals[8];
        ctx.stream()->memcpy(out_vals, dst->data, 8 * sizeof(float)).wait();
        fprintf(stderr,
                "DEBUG FFN_DOWN_PRE_TP %s device=%d ne=[%lldx%lld] src0=[%lldx%lld] dst[0..7]=[%f, %f, %f, %f, %f, %f, "
                "%f, %f]\n",
                weight_name, ctx.device, (long long) dst->ne[0], (long long) dst->ne[1], (long long) src0->ne[0],
                (long long) src0->ne[1], out_vals[0], out_vals[1], out_vals[2], out_vals[3], out_vals[4], out_vals[5],
                out_vals[6], out_vals[7]);
        // Also check FFN input (src1) for NaN
        if (dst->ne[1] == 1) {  // Token generation
            float src1_vals[8];
            ctx.stream()->memcpy(src1_vals, src1->data, 8 * sizeof(float)).wait();
            fprintf(stderr, "DEBUG FFN_DOWN_INPUT batch=1 src1[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n", src1_vals[0],
                    src1_vals[1], src1_vals[2], src1_vals[3], src1_vals[4], src1_vals[5], src1_vals[6], src1_vals[7]);
        }
    }

    // Tensor Parallelism: Column-parallel post-processing
    // For column-parallel layers, compute on device 1 and store result for subsequent row-parallel
    ggml_sycl_mul_mat_tp_column_parallel_post(ctx, src0, src1, dst);

    // Tensor Parallelism: Row-parallel post-processing
    // For row-parallel layers, compute on other TP devices and add their results
    ggml_sycl_mul_mat_tp_row_parallel_post(ctx, src0, src1, dst);

    // DEBUG: Capture FFN down output AFTER TP post-processing
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int ffn_post_dbg = 0;
    if (g_ggml_sycl_tp_debug && is_ffn_down_l0 && ffn_post_dbg++ < 3) {
        ctx.stream()->wait();
        float out_vals[8];
        ctx.stream()->memcpy(out_vals, dst->data, 8 * sizeof(float)).wait();
        fprintf(stderr,
                "DEBUG FFN_DOWN_POST_TP %s device=%d ne=[%lldx%lld] dst[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                weight_name, ctx.device, (long long) dst->ne[0], (long long) dst->ne[1], out_vals[0], out_vals[1],
                out_vals[2], out_vals[3], out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
    }

    // Debug sync point to catch errors early in TP mode
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        try {
            ctx.stream()->wait();
            GGML_SYCL_DEBUG("[MUL_MAT] Kernel completed successfully on device %d\n", ctx.device);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[MUL_MAT] Kernel FAILED on device %d: %s (code=%d)\n", ctx.device, e.what(),
                           static_cast<int>(e.code().value()));
        }
    }
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

__dpct_inline__ static void k_copy_src1_to_contiguous(const char * __restrict__ src1_original,
                                                      char * __restrict__ src1_contiguous,
                                                      int * __restrict__ cur_src1_row,
                                                      mmid_row_mapping * __restrict__ row_mapping,
                                                      const char * __restrict ids,
                                                      int64_t                  i02,
                                                      size_t                   ids_nb1,
                                                      size_t                   ids_nb0,
                                                      int64_t                  ne11,
                                                      int64_t                  ne10,
                                                      size_t                   nb11,
                                                      size_t                   nb12,
                                                      const sycl::nd_item<3> & item_ct1,
                                                      int &                    src1_row) {
    int32_t iid1 = item_ct1.get_group(2);
    int32_t id   = item_ct1.get_group(1);

    const int32_t row_id_i = *(const int32_t *) (ids + iid1 * ids_nb1 + id * ids_nb0);

    if (row_id_i != i02) {
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    if (item_ct1.get_local_id(2) == 0) {
        src1_row              = dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(cur_src1_row, 1);
        row_mapping[src1_row] = { id, iid1 };
    }
    /*
    DPCT1065:194: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    const float * src1_row_original   = (const float *) (src1_original + i11 * nb11 + i12 * nb12);
    float *       src1_row_contiguous = (float *) (src1_contiguous + src1_row * nb11);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10; i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(char * __restrict__ dst_original,
                                                       const char * __restrict__ dst_contiguous,
                                                       const mmid_row_mapping * __restrict__ row_mapping,
                                                       int64_t                  ne0,
                                                       size_t                   nb1,
                                                       size_t                   nb2,
                                                       const sycl::nd_item<3> & item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *) (dst_contiguous + i * nb1);
    float *       dst_row_original   = (float *) (dst_original + i1 * nb1 + i2 * nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0; j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Debug helper for MoE comparison
static bool g_moe_debug_enabled    = false;
static int  g_moe_debug_call_count = 0;

static void init_moe_debug() {
    static bool initialized = false;
    if (!initialized) {
        g_moe_debug_enabled = (getenv("GGML_SYCL_MOE_DEBUG") != nullptr);
        initialized         = true;
    }
}

// Try fused MoE ESIMD kernel for batched prefill (ne12 > 1)
// Returns true if handled, false to fall back to other implementations
static bool ggml_sycl_mul_mat_id_fused(ggml_backend_sycl_context & ctx,
                                       const ggml_tensor *         src0,
                                       const ggml_tensor *         src1,
                                       const ggml_tensor *         ids,
                                       ggml_tensor *               dst) {
#if SYCL_ESIMD_MOE_AVAILABLE
    static bool fused_moe_disabled = (std::getenv("GGML_SYCL_DISABLE_FUSED_MOE") != nullptr);
    if (fused_moe_disabled) {
        return false;  // Disabled by environment variable
    }

    // Early batch size checks - avoid expensive GGML_TENSOR_BINARY_OP_LOCALS for common cases
    const int64_t batch_size = src1->ne[2];  // ne12 - number of tokens

    // Only use fused kernel for batched prefill (multiple tokens)
    // For single-token decode, MMVQ is faster due to higher parallelism
    if (batch_size <= 1) {
        return false;  // Use MMVQ for single-token decode
    }

    // For large batch sizes, host-side oneDNN batching is much faster (for non-graph mode)
    // See GGML_SYCL_FUSED_MOE_MAX_BATCH definition at file scope for details
    // During graph recording, we MUST use this path for large batches since host-side routing
    // requires synchronization that breaks graph recording.
    if (batch_size > GGML_SYCL_FUSED_MOE_MAX_BATCH && !g_ggml_sycl_graph_recording) {
        GGML_SYCL_DEBUG("[MoE FUSED] Batch %ld > %ld, using oneDNN batching\n", (long) batch_size,
                        (long) GGML_SYCL_FUSED_MOE_MAX_BATCH);
        return false;  // Fall back to host-side oneDNN batching
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    // Check for supported quantization types
    if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_MXFP4) {
        return false;
    }

    // Input must be F32
    if (src1->type != GGML_TYPE_F32) {
        return false;
    }

    const queue_ptr stream = ctx.stream();

    // Check if expert weights are in host memory
    // Fused kernel requires contiguous device memory for all experts
    // Fall back to MMVQ which has per-expert caching support
    {
        sycl::usm::alloc ptr_type = sycl::get_pointer_type(src0->data, stream->get_context());
        if (ptr_type == sycl::usm::alloc::unknown || ptr_type == sycl::usm::alloc::host) {
            GGML_SYCL_DEBUG("[MoE FUSED] Weights in host memory (type=%d), falling back to MMVQ for caching\n",
                            (int) ptr_type);
            return false;  // Let MMVQ handle with expert caching
        }
    }

    // Calculate parameters
    const int64_t num_experts = ne02;        // Number of experts
    const int64_t nrows       = ne01;        // Output rows per expert
    const int64_t ncols       = ne00;        // Hidden dimension (input size)
    const int64_t n_ids       = ids->ne[0];  // Expert selections per token
    const int64_t num_tokens  = ne12;        // Number of tokens (from src1)
    GGML_ASSERT(ne11 > 0 && "MoE input broadcast dimension ne11 must be positive");
    GGML_ASSERT((ne11 == 1 || ne11 == n_ids) && "MoE input broadcast dimension ne11 must be 1 or n_ids");

    // Strides
    const int64_t stride_expert = nb02;  // Bytes between experts

    GGML_SYCL_DEBUG(
        "[MoE FUSED] Attempting fused kernel: tokens=%ld, experts=%ld, nrows=%ld, ncols=%ld, ne11=%ld, type=%d\n",
        (long) num_tokens, (long) num_experts, (long) nrows, (long) ncols, (long) ne11, src0->type);
    GGML_SYCL_DEBUG("[MoE FUSED] Input strides: nb11=%ld, nb12=%ld, Output strides: nb1=%ld, nb2=%ld\n", (long) nb11,
                    (long) nb12, (long) nb1, (long) nb2);

    // Launch appropriate kernel based on quantization type
    if (src0->type == GGML_TYPE_Q8_0) {
        launch_fused_moe_q8_0(src0->data,                   // expert_weights
                              (const float *) src1->data,   // input
                              (const int32_t *) ids->data,  // expert_ids
                              (float *) dst->data,          // output
                              stride_expert, ncols, nrows, n_ids, num_tokens,
                              ne11,                         // src1 dimension 1 (for modulo wrapping)
                              ids->nb[0],                   // ids_nb0
                              ids->nb[1],                   // ids_nb1
                              nb11,                         // src1 stride for dim 1
                              nb12,                         // src1 stride for dim 2
                              nb1,                          // dst stride for dim 1
                              nb2,                          // dst stride for dim 2
                              *stream);
        GGML_SYCL_DEBUG("[MoE FUSED] Q8_0 kernel launched\n");
        return true;
    } else if (src0->type == GGML_TYPE_MXFP4) {
        // Check if weights are reordered to SoA layout (MXFP4 MoE only has SOA kernel)
        auto *     src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
        const bool use_soa    = src0_extra && src0_extra->optimized_feature.is_soa();

        // Debug: understand why use_soa might be false
        if (g_ggml_sycl_debug) {
            fprintf(stderr, "[MoE FUSED] MXFP4 check: src0='%s' extra=%p use_soa=%d reorder=%d\n", src0->name,
                    (void *) src0_extra, use_soa, src0_extra ? (int) src0_extra->optimized_feature.get_reorder() : -1);
        }

        if (use_soa) {
            // SoA layout: total_qs_size = (ncols / 2) * total_rows
            // From reorder_qw_mxfp4: scale_ptr = qs_ptr + (ncols / 2) * nrows
            const int64_t total_rows    = nrows * num_experts;
            const int64_t total_qs_size = (ncols / 2) * total_rows;

            launch_fused_moe_mxfp4_soa(src0->data, (const float *) src1->data, (const int32_t *) ids->data,
                                       (float *) dst->data, total_qs_size, ncols,
                                       nrows,  // nrows_per_expert
                                       n_ids, num_tokens, ne11, ids->nb[0], ids->nb[1], nb11, nb12, nb1, nb2, *stream);
            GGML_SYCL_DEBUG("[MoE FUSED] MXFP4 SoA kernel launched (tokens=%ld)\n", (long) num_tokens);

            // Debug: print first few output values after ESIMD kernel
            if (g_ggml_sycl_debug) {
                stream->wait();  // Ensure kernel completes
                std::vector<float> h_out(4);
                stream->memcpy(h_out.data(), dst->data, 4 * sizeof(float)).wait();
                fprintf(stderr, "[ESIMD MoE] final_output[0..3] = [%.6f, %.6f, %.6f, %.6f]\n", h_out[0], h_out[1],
                        h_out[2], h_out[3]);
            }
        } else {
            // Original AoS path - use persistent or parallel kernel
            // Use persistent kernel only for small batches (tg) where SLM caching helps
            // For large batches (pp), each token has different input so SLM caching doesn't help
            // Threshold: use persistent for num_tokens <= 8, otherwise use parallel kernel
            const bool use_persistent = use_persistent_moe_kernel() && (num_tokens <= 8);

            if (use_persistent) {
                launch_persistent_moe_mxfp4(src0->data, (const float *) src1->data, (const int32_t *) ids->data,
                                            (float *) dst->data, stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
                                            ids->nb[0], ids->nb[1], nb11, nb12, nb1, nb2, *stream);
                GGML_SYCL_DEBUG("[MoE FUSED] MXFP4 persistent kernel launched (tokens=%ld)\n", (long) num_tokens);
            } else {
                launch_fused_moe_mxfp4(src0->data, (const float *) src1->data, (const int32_t *) ids->data,
                                       (float *) dst->data, stride_expert, ncols, nrows, n_ids, num_tokens, ne11,
                                       ids->nb[0], ids->nb[1], nb11, nb12, nb1, nb2, *stream);
                GGML_SYCL_DEBUG("[MoE FUSED] MXFP4 parallel kernel launched (tokens=%ld)\n", (long) num_tokens);
            }
        }
        return true;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
#endif
    return false;
}

// XMX Sorted MoE path (experimental)
// Enabled via: GGML_SYCL_XMX_MOE=1
//
// Three-phase implementation:
// 1. GPU-side token sorting by expert (using moe-sort.hpp functions)
// 2. XMX GEMM per expert batch (using moe-xmx.hpp kernel)
// 3. Scatter results back to original order
static bool try_xmx_sorted_moe(ggml_backend_sycl_context & ctx,
                               const ggml_tensor *         src0,
                               const ggml_tensor *         src1,
                               const ggml_tensor *         ids,
                               ggml_tensor *               dst) {
#if SYCL_XMX_MOE_AVAILABLE
    static bool enabled = getenv("GGML_SYCL_XMX_MOE") != nullptr;
    if (!enabled) {
        return false;
    }

    static bool fused_enabled = getenv("GGML_SYCL_XMX_MOE_FUSED") != nullptr;
    // Enable XMX tile-aligned layout for MXFP4 MoE (requires XMX_MOE=1)
    static bool tiled_enabled = getenv("GGML_SYCL_XMX_MOE_TILED") != nullptr;

    // Get XMX capabilities from cached device info
    auto & caps = ggml_sycl_info().devices[ctx.device].xmx_caps;
    if (!caps.supported || !caps.supports_int8) {
        GGML_SYCL_DEBUG("[XMX MoE] XMX not supported or no int8 support, skipping\n");
        return false;
    }

    // Handle Q8_0 and MXFP4 quantized weights
    if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_MXFP4) {
        GGML_SYCL_DEBUG("[XMX MoE] Only Q8_0/MXFP4 supported, got type %d\n", src0->type);
        return false;
    }

    // Check weight memory layout - we support AoS, SoA, and Coalesced for Q8_0 and MXFP4
    ggml_tensor_extra_gpu * src0_extra   = (ggml_tensor_extra_gpu *) src0->extra;
    const bool              is_soa       = src0_extra && src0_extra->optimized_feature.is_soa();
    const bool              is_coalesced = src0_extra && src0_extra->optimized_feature.is_coalesced();
    const bool              is_aos       = !is_soa && !is_coalesced;  // Default AoS when no reordering

    GGML_SYCL_DEBUG("[XMX MoE] Weight layout: %s\n", is_coalesced ? "coalesced" : (is_soa ? "soa" : "aos"));

    // EARLY BAIL-OUT for graph recording compatibility
    // Q8_0 SoA and MXFP4 SoA paths are graph-compatible (use fused kernels with fixed grid size).
    // All other paths require host synchronization that breaks graph recording.
    const bool is_graph_compatible = is_soa && (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_MXFP4);
    if (g_ggml_sycl_graph_recording && !is_graph_compatible) {
        GGML_SYCL_DEBUG("[XMX MoE] Non-SoA or unsupported type during graph recording, falling back to ESIMD\n");
        return false;
    }

    // Check if expert weights are in host/mmap memory
    // If so, skip XMX path - let host-side dispatch populate the cache first
    // GPU kernels cannot access host memory and will hang
    if (src0->buffer && ggml_backend_buffer_is_host(src0->buffer)) {
        GGML_SYCL_DEBUG("[XMX MoE] Weights in host buffer, falling back to populate cache\n");
        return false;
    }

    sycl::queue * stream = ctx.stream();

    // Extract tensor dimensions
    // src0: [in_dim, out_dim, n_experts] - expert weights (Q8_0/MXFP4)
    // src1: [in_dim, ne11, n_tokens] - input tokens (F32), ne11 is broadcast dimension
    // ids: [n_ids, n_tokens] - expert assignments (I32)
    // dst: [out_dim, n_ids, n_tokens] - output (F32)
    const int64_t in_dim    = src0->ne[0];
    const int64_t out_dim   = src0->ne[1];
    const int64_t n_experts = src0->ne[2];
    const int64_t n_tokens  = src1->ne[2];  // FIX: was src1->ne[1], should be ne[2] like ESIMD path
    const int64_t ne11      = src1->ne[1];  // Broadcast dimension for input (1 or n_ids)
    const int64_t n_ids     = ids->ne[0];
    GGML_ASSERT(ne11 > 0 && "MoE input broadcast dimension ne11 must be positive");
    GGML_ASSERT((ne11 == 1 || ne11 == n_ids) && "MoE input broadcast dimension ne11 must be 1 or n_ids");
    const int64_t total_pairs  = n_tokens * n_ids;
    const int64_t n_input_rows = n_tokens * ne11;  // Total input rows accounting for broadcast

    GGML_SYCL_DEBUG("[XMX MoE] tokens=%ld, ids=%ld, experts=%ld, in_dim=%ld, out_dim=%ld\n", (long) n_tokens,
                    (long) n_ids, (long) n_experts, (long) in_dim, (long) out_dim);
    GGML_SYCL_DEBUG("[XMX MoE] src1 tensor: ne=[%ld,%ld,%ld,%ld] nb=[%ld,%ld,%ld,%ld]\n", (long) src1->ne[0],
                    (long) src1->ne[1], (long) src1->ne[2], (long) src1->ne[3], (long) src1->nb[0], (long) src1->nb[1],
                    (long) src1->nb[2], (long) src1->nb[3]);
    GGML_SYCL_DEBUG("[XMX MoE] ids tensor: ne=[%ld,%ld,%ld,%ld] nb=[%ld,%ld,%ld,%ld]\n", (long) ids->ne[0],
                    (long) ids->ne[1], (long) ids->ne[2], (long) ids->ne[3], (long) ids->nb[0], (long) ids->nb[1],
                    (long) ids->nb[2], (long) ids->nb[3]);

    // Allocate temporary buffers
    sycl::half *      tokens_sorted = sycl::malloc_device<sycl::half>(total_pairs * in_dim, *stream);
    MoETokenMapping * token_map     = sycl::malloc_device<MoETokenMapping>(total_pairs, *stream);
    // Allocate sorted_token_ids when fused/tiled paths may be used, or when graph recording
    // with MXFP4 SoA (fused path auto-enabled for graph compatibility)
    const bool        needs_sorted_token_ids =
        fused_enabled || tiled_enabled || (g_ggml_sycl_graph_recording && src0->type == GGML_TYPE_MXFP4 && is_soa);
    int32_t * sorted_token_ids = needs_sorted_token_ids ? sycl::malloc_device<int32_t>(total_pairs, *stream) : nullptr;
    int32_t * expert_counts    = sycl::malloc_device<int32_t>(n_experts, *stream);
    // Allocate n_experts+1 for fused path (needs cumulative offsets including total)
    int32_t * expert_offsets   = sycl::malloc_device<int32_t>(n_experts + 1, *stream);
    sycl::half * sorted_output = sycl::malloc_device<sycl::half>(total_pairs * out_dim, *stream);

    if (!tokens_sorted || !token_map || !expert_counts || !expert_offsets || !sorted_output) {
        GGML_SYCL_DEBUG("[XMX MoE] Failed to allocate temporary buffers\n");
        // Free any successfully allocated buffers
        if (tokens_sorted) {
            sycl::free(tokens_sorted, *stream);
        }
        if (token_map) {
            sycl::free(token_map, *stream);
        }
        if (sorted_token_ids) {
            sycl::free(sorted_token_ids, *stream);
        }
        if (expert_counts) {
            sycl::free(expert_counts, *stream);
        }
        if (expert_offsets) {
            sycl::free(expert_offsets, *stream);
        }
        if (sorted_output) {
            sycl::free(sorted_output, *stream);
        }
        return false;
    }
    if (needs_sorted_token_ids && !sorted_token_ids) {
        GGML_SYCL_DEBUG("[XMX MoE] Failed to allocate sorted_token_ids for fused path\n");
        sycl::free(tokens_sorted, *stream);
        sycl::free(token_map, *stream);
        sycl::free(expert_counts, *stream);
        sycl::free(expert_offsets, *stream);
        sycl::free(sorted_output, *stream);
        return false;
    }

    // Phase 1: Sort tokens by expert
    const int32_t * expert_ids = static_cast<const int32_t *>(ids->data);

    moe_count_tokens_per_expert<64>(expert_ids, expert_counts, n_tokens, n_ids, *stream);

    moe_compute_expert_offsets(expert_counts, expert_offsets, n_experts, *stream);

    // Copy offsets for atomic writes during sorting
    int32_t * expert_write_pos = sycl::malloc_device<int32_t>(n_experts, *stream);
    if (!expert_write_pos) {
        GGML_SYCL_DEBUG("[XMX MoE] Failed to allocate expert_write_pos\n");
        sycl::free(tokens_sorted, *stream);
        sycl::free(token_map, *stream);
        if (sorted_token_ids) {
            sycl::free(sorted_token_ids, *stream);
        }
        sycl::free(expert_counts, *stream);
        sycl::free(expert_offsets, *stream);
        sycl::free(sorted_output, *stream);
        return false;
    }
    // Make memcpy async for graph compatibility - sort will depend on this event
    sycl::event copy_write_pos_event = stream->memcpy(expert_write_pos, expert_offsets, n_experts * sizeof(int32_t));

    // Phase 1a: Convert F32 tokens to F16 for XMX processing
    // Handle 2D non-contiguous token layouts with broadcast dimension ne11
    // Input layout: [in_dim, ne11, n_tokens] with strides [nb0, nb1, nb2]
    const int64_t nb1        = src1->nb[1];  // Byte stride between id slots
    const int64_t nb2        = src1->nb[2];  // Byte stride between tokens
    const char *  tokens_f32 = static_cast<const char *>(src1->data);

    // Allocate for all input rows: n_tokens * ne11 (handles broadcast correctly)
    sycl::half * tokens_f16_input = sycl::malloc_device<sycl::half>(n_input_rows * in_dim, *stream);

    if (!tokens_f16_input) {
        GGML_SYCL_DEBUG("[XMX MoE] Failed to allocate tokens_f16_input\n");
        copy_write_pos_event.wait();  // Must wait before freeing
        sycl::free(tokens_sorted, *stream);
        sycl::free(token_map, *stream);
        if (sorted_token_ids) {
            sycl::free(sorted_token_ids, *stream);
        }
        sycl::free(expert_counts, *stream);
        sycl::free(expert_offsets, *stream);
        sycl::free(sorted_output, *stream);
        sycl::free(expert_write_pos, *stream);
        return false;
    }

    GGML_SYCL_DEBUG("[XMX MoE] ne11=%ld, n_input_rows=%ld, nb1=%ld, nb2=%ld\n", (long) ne11, (long) n_input_rows,
                    (long) nb1, (long) nb2);
    moe_convert_f32_to_f16(tokens_f32, tokens_f16_input, n_tokens, in_dim, ne11, nb1, nb2, *stream);

    // Phase 1b: Sort tokens by expert (with broadcast handling for ne11)
    // Use async version with dependency on memcpy for graph compatibility
    sycl::event sort_event =
        moe_sort_tokens_by_expert_async(tokens_f16_input, tokens_sorted, expert_ids, expert_write_pos, token_map,
                                        n_tokens, n_ids, ne11, in_dim, n_experts, *stream, copy_write_pos_event);

    // Phase 2: XMX GEMM per expert
    auto xmx_cfg = moe_xmx::MoEXMXConfig::from_capabilities(caps);

    // For Q8_0 SoA or MXFP4 SoA with graph-compatible fused path, we skip the host sync here.
    // For all other paths (AoS, Coalesced), we need to read counts/offsets.
    // Note: Graph recording bail-out for non-compatible paths happens at function entry.
    const bool use_graph_compatible_fused = is_soa && (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_MXFP4);

    // Read counts/offsets to host (only for non-graph-compatible paths)
    std::vector<int32_t> h_counts(n_experts), h_offsets(n_experts + 1);
    int64_t              actual_pairs = 0;

    if (!use_graph_compatible_fused) {
        stream->memcpy(h_counts.data(), expert_counts, n_experts * sizeof(int32_t)).wait();
        // Read n_experts+1 offsets to get the total valid pairs at the end
        stream->memcpy(h_offsets.data(), expert_offsets, (n_experts + 1) * sizeof(int32_t)).wait();
        // actual_pairs = sum of all expert counts = h_offsets[n_experts]
        // This is the number of VALID (token, expert) pairs, not total_pairs which includes empty slots
        actual_pairs = h_offsets[n_experts];
    }
    GGML_SYCL_DEBUG("[XMX MoE] actual_pairs=%ld vs total_pairs=%ld (savings: %ld)\n", (long) actual_pairs,
                    (long) total_pairs, (long) (total_pairs - actual_pairs));

    GGML_SYCL_DEBUG("[XMX MoE] Expert distribution: ");
    for (int64_t e = 0; e < n_experts; e++) {
        if (h_counts[e] > 0) {
            GGML_SYCL_DEBUG("e%ld=%d ", (long) e, h_counts[e]);
        }
    }
    GGML_SYCL_DEBUG("\n");

    // Allocate buffers for pre-quantized tokens (reused across experts)
    // Q8_0 quantization: int8 values + fp16 scales (one per QK8_0 elements)
    int64_t      max_batch    = total_pairs;  // Upper bound on tokens per expert
    int8_t *     q_tokens     = sycl::malloc_device<int8_t>(max_batch * in_dim, *stream);
    sycl::half * token_scales = sycl::malloc_device<sycl::half>(max_batch * (in_dim / QK8_0), *stream);

    if (!q_tokens || !token_scales) {
        GGML_SYCL_DEBUG("[XMX MoE] Failed to allocate q_tokens or token_scales\n");
        if (q_tokens) {
            sycl::free(q_tokens, *stream);
        }
        if (token_scales) {
            sycl::free(token_scales, *stream);
        }
        sycl::free(tokens_f16_input, *stream);
        sycl::free(tokens_sorted, *stream);
        sycl::free(token_map, *stream);
        if (sorted_token_ids) {
            sycl::free(sorted_token_ids, *stream);
        }
        sycl::free(expert_counts, *stream);
        sycl::free(expert_offsets, *stream);
        sycl::free(expert_write_pos, *stream);
        sycl::free(sorted_output, *stream);
        return false;
    }

    // Allocate buffer for Q8_0 expert scales (reused across experts)
    // Each expert has out_dim * (in_dim/QK8_0) scales
    // Not needed for SoA format - scales are already separate in memory
    int64_t      num_k_blocks     = in_dim / QK8_0;
    sycl::half * expert_scale_buf = nullptr;
    if (src0->type == GGML_TYPE_Q8_0 && !is_soa) {
        expert_scale_buf = sycl::malloc_device<sycl::half>(out_dim * num_k_blocks, *stream);
        if (!expert_scale_buf) {
            GGML_SYCL_DEBUG("[XMX MoE] Failed to allocate expert_scale_buf\n");
            sycl::free(q_tokens, *stream);
            sycl::free(token_scales, *stream);
            sycl::free(tokens_f16_input, *stream);
            sycl::free(tokens_sorted, *stream);
            sycl::free(token_map, *stream);
            if (sorted_token_ids) {
                sycl::free(sorted_token_ids, *stream);
            }
            sycl::free(expert_counts, *stream);
            sycl::free(expert_offsets, *stream);
            sycl::free(expert_write_pos, *stream);
            sycl::free(sorted_output, *stream);
            return false;
        }
    }

    // For SoA format, compute d_offset (byte offset from qs to scales)
    // SoA layout is FLAT across all experts:
    //   [all qs for all experts][all scales for all experts]
    // nblocks_per_expert = out_dim * (in_dim / 32)
    // total_nblocks = nblocks_per_expert * n_experts
    const int64_t nblocks_per_expert = out_dim * num_k_blocks;
    const int64_t total_nblocks      = nblocks_per_expert * n_experts;

    // For Q8_0: qs = 32 bytes/block, d = 2 bytes/block
    const int64_t soa_d_offset    = nblocks_per_expert * QK8_0;  // Per-expert offset (for non-flat SoA)
    const int64_t soa_total_qs_q8 = total_nblocks * QK8_0;       // Total qs region for flat SoA

    // For MXFP4: qs = 16 bytes/block, e = 1 byte/block
    constexpr int64_t MXFP4_QS_BYTES_PER_BLOCK = 16;                                   // QK_MXFP4 / 2
    const int64_t     soa_total_qs_mxfp4  = total_nblocks * MXFP4_QS_BYTES_PER_BLOCK;  // Total qs region for flat SoA
    const int64_t     mxfp4_qs_per_expert = nblocks_per_expert * MXFP4_QS_BYTES_PER_BLOCK;

    const char * layout_name = is_coalesced ? "Coalesced" : (is_soa ? "SoA" : "AoS");
    const char * quant_name  = (src0->type == GGML_TYPE_Q8_0) ? "Q8_0" : "MXFP4";
    GGML_SYCL_DEBUG("[XMX MoE] Using %s format for %s weights\n", layout_name, quant_name);

    // When is_soa or is_coalesced is true (from tensor metadata), src0->data
    // already points to device memory with the correct layout - no cache needed.
    // This matches how fused ESIMD uses src0->data directly.

    // Q8_0 stride per expert (bytes of qs data)
    const int64_t q8_qs_per_expert = nblocks_per_expert * QK8_0;

    // Try graph-compatible fused kernel path (single kernel for all experts)
    // Uses GPU-side work assignment via tile mapping - no host iteration needed
    // During graph recording, use fixed max grid size to avoid sync for total_tiles read.
    // Extra work-groups early-exit when their tile is out of range.
    if (src0->type == GGML_TYPE_Q8_0 && is_soa) {
        // Ensure tile mapping buffers are pre-allocated (enables graph recording)
        ctx.xmx_moe_buffers.allocate_tile_mapping(*stream);
        int32_t * expert_tile_offsets = ctx.xmx_moe_buffers.expert_tile_offsets;
        int32_t * total_tiles_dev     = ctx.xmx_moe_buffers.total_tiles;

        // Pre-quantize ALL sorted tokens for fused kernel
        // sort_event ensures sorted tokens are ready before quantization
        // (In-order queue: quantization will naturally wait for sort)
        moe_xmx::preprocess_tokens_q8(tokens_sorted, q_tokens, token_scales, total_pairs, in_dim, *stream);

        // Compute tile mapping on GPU (no host sync needed for this)
        // Each expert's token count is converted to tile count via ceil division
        constexpr int64_t TILE_M         = 32;  // XMX tile height (TILES_M=4 * XMX_M=8)
        // sort_event dependency ensures expert_counts are finalized before tile mapping
        sycl::event       tile_map_event = moe_compute_tile_mapping(expert_counts, expert_tile_offsets, total_tiles_dev,
                                                                    n_experts, TILE_M, *stream, sort_event);

        // Compute maximum possible tiles for fixed grid size (graph-compatible)
        // max_tiles = ceil(total_pairs / TILE_M) = worst case where all tokens go to one expert
        const int32_t max_tiles = static_cast<int32_t>((total_pairs + TILE_M - 1) / TILE_M);

        // During graph recording, use fixed max grid size to avoid host sync.
        // During normal execution, read actual total_tiles for efficiency.
        int32_t total_tiles_host = max_tiles;
        if (!g_ggml_sycl_graph_recording) {
            // Read actual tile count - one sync point for efficiency
            stream->memcpy(&total_tiles_host, total_tiles_dev, sizeof(int32_t), { tile_map_event }).wait();

            if (total_tiles_host == 0) {
                // No work - all experts have zero tokens
                GGML_SYCL_DEBUG("[XMX MoE] No tiles to process (all experts empty)\n");
                goto scatter_back;
            }
        }

        GGML_SYCL_DEBUG("[XMX MoE] Fused kernel: total_tiles=%d (max=%d, recording=%d) for %ld experts\n",
                        total_tiles_host, max_tiles, g_ggml_sycl_graph_recording ? 1 : 0, (long) n_experts);

        // SoA weight pointers
        const int8_t *     base_qs = static_cast<const int8_t *>(src0->data);
        const sycl::half * base_d =
            reinterpret_cast<const sycl::half *>(static_cast<const char *>(src0->data) + soa_total_qs_q8);

        // Single fused kernel launch - GPU-side work assignment via binary search
        // Each work-group finds its expert via expert_tile_offsets lookup.
        // Extra WGs (when total_tiles < max_tiles) early-exit in kernel.
        // Note: tile_map_event dependency ensures tile offsets are ready;
        // quantization completed earlier on in-order queue
        sycl::event gemm_event = moe_xmx::launch_fused_xmx_moe_q8_0_soa<4, 4>(
            *stream, tile_map_event, base_qs, base_d, q_tokens, token_scales, sorted_output, expert_offsets,
            expert_tile_offsets, total_tiles_host, n_experts, out_dim, in_dim, q8_qs_per_expert, xmx_cfg);

        // During graph recording, don't call wait() - use in-order queue semantics
        // During normal execution, wait before scatter-back
        if (!g_ggml_sycl_graph_recording) {
            gemm_event.wait();
        }

        // Set actual_pairs for scatter-back (use total_pairs as safe upper bound)
        // All sorted token entries are valid, so scatter will write correct results
        actual_pairs = total_pairs;

        GGML_SYCL_DEBUG("[MoE] Graph-compatible fused XMX path succeeded\n");
        goto scatter_back;
    }

    // Try XMX tile-aligned MXFP4 path first (if enabled)
    // This path uses pre-computed tiled layout for better XMX utilization
    // Track allocation failures to avoid repeated OOM
    static bool tiled_alloc_failed = false;

    GGML_SYCL_DEBUG("[MoE] Tiled path check: tiled_enabled=%d, src0->type=%d (MXFP4=%d), caps.supported=%d\n",
                    tiled_enabled ? 1 : 0, src0->type, GGML_TYPE_MXFP4, caps.supported ? 1 : 0);
    if (tiled_enabled && !tiled_alloc_failed && src0->type == GGML_TYPE_MXFP4 && caps.supported) {
        // Get or create tiled layout from tensor extra
        // Skip creation during graph recording (has .wait() calls) - fall through to SoA fused path
        if (src0_extra && src0_extra->xmx_mxfp4_tiled[ctx.device] == nullptr && !g_ggml_sycl_graph_recording) {
            // Create tiled layout on first use
            moe_xmx_fused::MXFPXMXConfig     cfg  = moe_xmx_fused::MXFPXMXConfig::from_device(ctx.device);
            moe_xmx_fused::MXFPXMXLayoutInfo info = moe_xmx_fused::MXFPXMXLayoutInfo::compute(out_dim, in_dim, cfg);

            void * tiled_buf = sycl::malloc_device(info.total_bytes * n_experts, *stream);
            if (!tiled_buf) {
                GGML_SYCL_DEBUG("[MoE] Failed to allocate tiled buffer (%zu bytes), disabling tiled path\n",
                                info.total_bytes * n_experts);
                tiled_alloc_failed = true;  // Disable further allocation attempts
            } else {
                // Validate layout and source data before conversion
                GGML_ASSERT((is_soa || is_coalesced) && "MXFP4 tiled path requires SoA/Coalesced layout");
                GGML_ASSERT(src0->data != nullptr && "MXFP4 source tensor data is null");

                // Validate source buffer bounds
                const size_t expected_qs_size = static_cast<size_t>(soa_total_qs_mxfp4);
                const size_t expected_e_size = static_cast<size_t>(n_experts) * static_cast<size_t>(nblocks_per_expert);
                const size_t total_expected  = expected_qs_size + expected_e_size;
                const size_t src0_actual_size = ggml_nbytes(src0);
                GGML_ASSERT(total_expected <= src0_actual_size && "MXFP4 SoA layout size exceeds tensor allocation");

                // Convert each expert's SoA weights to tiled layout
                // NOTE: src0->data is on device memory, reorder runs on host
                // So we need to copy device data to host, reorder, then copy back

                // 1. Allocate host buffers
                const size_t src_size         = total_expected;
                const size_t tiled_total_size = info.total_bytes * n_experts;
                uint8_t *    host_src         = sycl::malloc_host<uint8_t>(src_size, *stream);
                uint8_t *    host_tiled       = sycl::malloc_host<uint8_t>(tiled_total_size, *stream);

                if (!host_src || !host_tiled) {
                    GGML_SYCL_DEBUG("[MoE] Failed to allocate host buffers for reorder\n");
                    if (host_src) {
                        sycl::free(host_src, *stream);
                    }
                    if (host_tiled) {
                        sycl::free(host_tiled, *stream);
                    }
                    sycl::free(tiled_buf, *stream);
                    tiled_buf = nullptr;
                } else {
                    // 2. Copy device SoA data to host
                    stream->memcpy(host_src, src0->data, src_size).wait();

                    const uint8_t * base_qs = host_src;
                    const uint8_t * base_e  = base_qs + soa_total_qs_mxfp4;

                    // 3. Reorder on host
                    for (size_t e = 0; e < static_cast<size_t>(n_experts); e++) {
                        const size_t qs_offset    = e * static_cast<size_t>(mxfp4_qs_per_expert);
                        const size_t e_offset     = e * static_cast<size_t>(nblocks_per_expert);
                        const size_t tiled_offset = e * info.total_bytes;

                        const uint8_t * soa_qs    = base_qs + qs_offset;
                        const uint8_t * soa_e     = base_e + e_offset;
                        uint8_t *       tiled_out = host_tiled + tiled_offset;

                        moe_xmx_fused::reorder_mxfp4_to_xmx_layout(soa_qs, soa_e, tiled_out, info);
                    }

                    // 4. Copy tiled data to device
                    stream->memcpy(tiled_buf, host_tiled, tiled_total_size).wait();

                    // 5. Free host buffers
                    sycl::free(host_src, *stream);
                    sycl::free(host_tiled, *stream);
                }

                if (tiled_buf) {
                    GGML_SYCL_DEBUG("[MoE] Converted %d experts to tiled layout (%zu bytes per expert)\n", n_experts,
                                    info.total_bytes);

                    src0_extra->xmx_mxfp4_tiled[ctx.device] = tiled_buf;
                    src0_extra->xmx_mxfp4_tiled_size        = info.total_bytes * n_experts;
                }
            }
        }

        if (src0_extra && src0_extra->xmx_mxfp4_tiled[ctx.device]) {
            // Extract sorted_token_ids from token_map for tiled kernel - async with event
            sycl::event extract_event = stream->parallel_for(sycl::range<1>(total_pairs), [=](sycl::id<1> idx) {
                sorted_token_ids[idx] = token_map[idx].original_idx;
            });

            // Pre-quantize ALL sorted tokens for tiled kernel
            moe_xmx::preprocess_tokens_q8(tokens_sorted, q_tokens, token_scales, total_pairs, in_dim, *stream);

            moe_xmx_fused::MXFPXMXConfig     cfg  = moe_xmx_fused::MXFPXMXConfig::from_device(ctx.device);
            moe_xmx_fused::MXFPXMXLayoutInfo info = moe_xmx_fused::MXFPXMXLayoutInfo::compute(out_dim, in_dim, cfg);

            auto [ok, evt] = try_fused_xmx_moe_mxfp4_tiled(
                extract_event, static_cast<const uint8_t *>(src0_extra->xmx_mxfp4_tiled[ctx.device]), q_tokens,
                token_scales, sorted_token_ids, expert_offsets, sorted_output, static_cast<int>(total_pairs),
                static_cast<int>(n_experts), out_dim, in_dim,
                info.total_bytes,  // expert_tiled_stride
                ctx.device, *stream);
            if (ok) {
                GGML_SYCL_DEBUG("[MoE] XMX MXFP4 tiled path succeeded\n");
                // During graph recording, don't call wait() - use in-order queue semantics
                if (!g_ggml_sycl_graph_recording) {
                    evt.wait();
                }
                goto scatter_back;
            }
            GGML_SYCL_DEBUG("[MoE] XMX MXFP4 tiled path failed, falling back to SoA\n");
        }
    }

    // Try MXFP4 fused kernel path (single kernel for all experts)
    // Graph-compatible: uses event chaining instead of .wait() calls
    // Auto-enable during graph recording since this is the only graph-compatible MXFP4 path
    // (tiled path is skipped during graph recording due to .wait() in layout conversion)
    if ((fused_enabled || g_ggml_sycl_graph_recording) && src0->type == GGML_TYPE_MXFP4 && is_soa) {
        // Extract sorted_token_ids from token_map for fused kernel - async with event
        sycl::event extract_event = stream->parallel_for(
            sycl::range<1>(total_pairs), [=](sycl::id<1> idx) { sorted_token_ids[idx] = token_map[idx].original_idx; });

        // Pre-quantize ALL sorted tokens for fused kernel
        // Uses in-order queue semantics - will execute after extract_event completes
        moe_xmx::preprocess_tokens_q8(tokens_sorted, q_tokens, token_scales, total_pairs, in_dim, *stream);

        // MXFP4 SoA weight pointers
        const uint8_t * base_qs = static_cast<const uint8_t *>(src0->data);
        const uint8_t * base_e  = base_qs + soa_total_qs_mxfp4;

        auto [fused_ok, gemm_event] = try_fused_xmx_moe_mxfp4_soa(
            extract_event, base_qs, base_e, q_tokens, token_scales, sorted_token_ids, expert_offsets, sorted_output,
            static_cast<int>(total_pairs), static_cast<int>(n_experts), out_dim, in_dim, mxfp4_qs_per_expert,
            nblocks_per_expert, ctx.device, *stream);

        if (fused_ok) {
            GGML_SYCL_DEBUG("[MoE] Fused MXFP4 XMX path succeeded\n");
            // During graph recording, don't call wait() - use in-order queue semantics
            // During normal execution, wait before scatter-back
            if (!g_ggml_sycl_graph_recording) {
                gemm_event.wait();
            }
            goto scatter_back;
        }
        GGML_SYCL_DEBUG("[MoE] Fused MXFP4 XMX path failed, falling back to per-expert dispatch\n");
    }

    for (int64_t e = 0; e < n_experts; e++) {
        if (h_counts[e] == 0) {
            continue;
        }

        // Weight tensor: [in_dim, out_dim, n_experts]
        // Stride to expert e = e * (in_dim * out_dim) in elements
        const size_t bytes_per_expert = src0->nb[2];
        const void * expert_weights   = static_cast<const char *>(src0->data) + e * bytes_per_expert;

        // Pre-quantize fp16 tokens to int8 with per-block scales
        const sycl::half * expert_tokens = tokens_sorted + h_offsets[e] * in_dim;
        moe_xmx::preprocess_tokens_q8(expert_tokens, q_tokens, token_scales, h_counts[e], in_dim, *stream);

        // Dispatch to appropriate kernel based on quantization type and layout
        if (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0: qs = 32 bytes/block, d = 2 bytes/block (half)
            const int64_t q8_qs_per_expert = nblocks_per_expert * QK8_0;

            if (is_coalesced) {
                // Coalesced format: weights are tile-major reordered with separate qs and d arrays
                // Per-expert layout, use AoS-style stride
                const int8_t *     coalesced_qs = static_cast<const int8_t *>(expert_weights);
                const sycl::half * coalesced_d =
                    reinterpret_cast<const sycl::half *>(static_cast<const char *>(expert_weights) + soa_d_offset);

                moe_xmx::launch_xmx_moe_gemm_q8_0_coalesced<4, 4>(coalesced_qs, coalesced_d, q_tokens, token_scales,
                                                                  sorted_output + h_offsets[e] * out_dim, h_counts[e],
                                                                  out_dim, in_dim, xmx_cfg, *stream);
            } else if (is_soa) {
                // FLAT SoA format: [all qs for all experts][all d for all experts]
                // Expert e's data is at offset e * nblocks_per_expert within each region
                const int8_t *     base_qs = static_cast<const int8_t *>(src0->data);
                const sycl::half * base_d =
                    reinterpret_cast<const sycl::half *>(static_cast<const char *>(src0->data) + soa_total_qs_q8);

                // Calculate pointers for expert e
                const int8_t *     soa_qs = base_qs + e * q8_qs_per_expert;
                const sycl::half * soa_d  = base_d + e * nblocks_per_expert;

                moe_xmx::launch_xmx_moe_gemm_q8_0_soa<4, 4>(soa_qs, soa_d, q_tokens, token_scales,
                                                            sorted_output + h_offsets[e] * out_dim, h_counts[e],
                                                            out_dim, in_dim, xmx_cfg, *stream);
            } else {
                // AoS format: extract scales from interleaved Q8_0 blocks
                moe_xmx::extract_q8_0_scales(expert_weights, expert_scale_buf, out_dim, in_dim, *stream);

                moe_xmx::launch_xmx_moe_gemm_q8_0<4, 4>(expert_weights, expert_scale_buf, q_tokens, token_scales,
                                                        sorted_output + h_offsets[e] * out_dim, h_counts[e], out_dim,
                                                        in_dim, xmx_cfg, *stream);
            }
        } else if (src0->type == GGML_TYPE_MXFP4) {
            // MXFP4 layout: 32 4-bit values packed in 16 bytes + 1 byte E8M0 exponent = 17 bytes per block
            if (is_coalesced) {
                // Coalesced format: tile-major reordered with separate qs and e arrays
                // Per-expert layout, use AoS-style stride
                const uint8_t * coalesced_qs = static_cast<const uint8_t *>(expert_weights);
                const uint8_t * coalesced_e  = coalesced_qs + mxfp4_qs_per_expert;

                moe_xmx::launch_xmx_moe_gemm_mxfp4_coalesced<4, 4>(coalesced_qs, coalesced_e, q_tokens, token_scales,
                                                                   sorted_output + h_offsets[e] * out_dim, h_counts[e],
                                                                   out_dim, in_dim, xmx_cfg, *stream);
            } else if (is_soa) {
                // FLAT SoA format: [all qs for all experts][all e for all experts]
                // Expert e's data is at offset e * nblocks_per_expert within each region
                const uint8_t * base_qs = static_cast<const uint8_t *>(src0->data);
                const uint8_t * base_e  = base_qs + soa_total_qs_mxfp4;

                // Calculate pointers for expert e
                const uint8_t * soa_qs = base_qs + e * mxfp4_qs_per_expert;
                const uint8_t * soa_e  = base_e + e * nblocks_per_expert;

                if (e == 0 && g_ggml_sycl_debug) {
                    fprintf(stderr,
                            "[XMX MoE MXFP4 SoA] base=%p total_qs=%ld e_offset=%ld "
                            "soa_qs=%p soa_e=%p batch=%ld\n",
                            (void *) base_qs, (long) soa_total_qs_mxfp4, (long) (base_e - base_qs), (void *) soa_qs,
                            (void *) soa_e, (long) h_counts[e]);
                }

                moe_xmx::launch_xmx_moe_gemm_mxfp4_soa<4, 4>(soa_qs, soa_e, q_tokens, token_scales,
                                                             sorted_output + h_offsets[e] * out_dim, h_counts[e],
                                                             out_dim, in_dim, xmx_cfg, *stream);

                // Debug: print expert outputs for comparison with ESIMD
                if (g_ggml_sycl_debug && (e == 0 || e == 5 || e == 10 || e == 15 || e == 20 || e == 28)) {
                    stream->wait();  // Wait for this expert's kernel
                    std::vector<sycl::half> h_expert_out(4);
                    stream->memcpy(h_expert_out.data(), sorted_output + h_offsets[e] * out_dim, 4 * sizeof(sycl::half))
                        .wait();
                    fprintf(stderr, "[XMX MoE] Expert %ld sorted_output[0:3] = [%.6f, %.6f, %.6f, %.6f]\n", (long) e,
                            static_cast<float>(h_expert_out[0]), static_cast<float>(h_expert_out[1]),
                            static_cast<float>(h_expert_out[2]), static_cast<float>(h_expert_out[3]));
                }
            } else {
                // AoS format: interleaved [e:1][qs:16] = 17 bytes per block
                moe_xmx::launch_xmx_moe_gemm_mxfp4<4, 4>(expert_weights, q_tokens, token_scales,
                                                         sorted_output + h_offsets[e] * out_dim, h_counts[e], out_dim,
                                                         in_dim, xmx_cfg, *stream);
            }
        }
    }

    // Single sync after all expert kernels submitted (avoids per-kernel stalls)
    stream->wait();

scatter_back:
    // Free pre-quantization buffers
    sycl::free(q_tokens, *stream);
    sycl::free(token_scales, *stream);

    // Phase 3: Scatter results back to original positions with F16->F32 conversion
    // Use byte strides for proper output tensor layout (ESIMD-compatible)
    char *        final_output = static_cast<char *>(dst->data);
    const int64_t out_nb1      = dst->nb[1];  // Byte stride between id slots
    const int64_t out_nb2      = dst->nb[2];  // Byte stride between tokens

    // Debug: check output tensor layout
    GGML_SYCL_DEBUG("[XMX MoE] dst tensor: ne=[%ld,%ld,%ld,%ld] nb=[%ld,%ld,%ld,%ld]\n", (long) dst->ne[0],
                    (long) dst->ne[1], (long) dst->ne[2], (long) dst->ne[3], (long) dst->nb[0], (long) dst->nb[1],
                    (long) dst->nb[2], (long) dst->nb[3]);

    moe_scatter_results_f16_to_f32(sorted_output, final_output, token_map, actual_pairs, out_dim, n_ids, out_nb1,
                                   out_nb2, *stream);

    // Debug: print first few output values after scatter
    // Skip debug sync during graph recording to maintain graph compatibility
    if (g_ggml_sycl_debug && !g_ggml_sycl_graph_recording) {
        stream->wait();  // Ensure scatter completes
        std::vector<float> h_out(4);
        stream->memcpy(h_out.data(), final_output, 4 * sizeof(float)).wait();
        fprintf(stderr, "[XMX MoE] final_output[0..3] = [%.6f, %.6f, %.6f, %.6f]\n", h_out[0], h_out[1], h_out[2],
                h_out[3]);
    }

    // Free temporary buffers
    sycl::free(tokens_f16_input, *stream);
    if (expert_scale_buf) {
        sycl::free(expert_scale_buf, *stream);
    }
    sycl::free(tokens_sorted, *stream);
    sycl::free(token_map, *stream);
    if (sorted_token_ids) {
        sycl::free(sorted_token_ids, *stream);
    }
    sycl::free(expert_counts, *stream);
    sycl::free(expert_offsets, *stream);
    sycl::free(expert_write_pos, *stream);
    sycl::free(sorted_output, *stream);

    GGML_SYCL_DEBUG("[XMX MoE] XMX sorted kernel completed\n");
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    return false;
#endif
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    init_moe_debug();

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor * ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    // Try XMX sorted MoE path first when explicitly enabled (experimental)
    // This gives priority to the XMX kernel for testing/benchmarking
    static bool xmx_moe_enabled = (getenv("GGML_SYCL_XMX_MOE") != nullptr);
    if (xmx_moe_enabled && try_xmx_sorted_moe(ctx, src0, src1, ids, dst)) {
        GGML_SYCL_DEBUG("[MoE] XMX sorted dispatch successful for type %d\n", src0->type);
        return;
    }

    // Try fused MoE ESIMD kernel for batched prefill (ne12 > 1)
    // This is much faster than per-expert dispatch for prompt processing
    if (ggml_sycl_mul_mat_id_fused(ctx, src0, src1, ids, dst)) {
        GGML_SYCL_DEBUG("[MoE] Fused ESIMD dispatch successful for type %d\n", src0->type);
        return;
    }

    // Try XMX sorted MoE path as fallback (experimental, Q8_0/MXFP4)
    if (!xmx_moe_enabled && try_xmx_sorted_moe(ctx, src0, src1, ids, dst)) {
        GGML_SYCL_DEBUG("[MoE] XMX sorted dispatch successful for type %d\n", src0->type);
        return;
    }

    // Try GPU-side expert routing (MMVQ) - good for decode (ne12 == 1)
    // This avoids the host sync that blocks graph recording
    if (ggml_sycl_mul_mat_id_vec_q(ctx, src0, src1, ids, dst)) {
        GGML_SYCL_DEBUG("[MoE] GPU-side MMVQ dispatch successful for type %d\n", src0->type);
        return;
    }
    GGML_SYCL_DEBUG("[MoE] Falling back to host-side routing for type %d\n", src0->type);

    // Host-side routing requires synchronization which is incompatible with graph recording.
    // If we reach here during graph recording, we must abort because no graph-compatible
    // path was available (XMX sorted MoE, fused ESIMD, and MMVQ all returned false).
    if (g_ggml_sycl_graph_recording) {
        GGML_LOG_ERROR(
            "[MoE] No graph-compatible dispatch path for type=%d batch=%ld. "
            "Host-side routing requires sync incompatible with graph recording.\n",
            src0->type, (long) ne12);
        GGML_ABORT(
            "MoE operation requires host synchronization, incompatible with SYCL graph recording. "
            "Set GGML_SYCL_DISABLE_GRAPH=1 to disable graphs or use SoA layout.");
    }

    const queue_ptr stream = ctx.stream();

    const int64_t n_as  = ne02;
    const int64_t n_ids = ids->ne[0];

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char *      ids_dev = (const char *) ids->data;

    // Host sync - incompatible with SYCL graph recording
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    // Debug: print expert IDs and input values
    if (g_moe_debug_enabled) {
        g_moe_debug_call_count++;
        fprintf(stderr, "\n[MoE DEBUG] call=%d ne12=%ld n_as=%ld n_ids=%ld ids_ne1=%ld\n", g_moe_debug_call_count,
                (long) ne12, (long) n_as, (long) n_ids, (long) ids->ne[1]);
        fprintf(stderr, "[MoE DEBUG] ids tensor: ne=[%ld,%ld,%ld,%ld] nb=[%ld,%ld,%ld,%ld] op=%s view_src=%p\n",
                (long) ids->ne[0], (long) ids->ne[1], (long) ids->ne[2], (long) ids->ne[3], (long) ids->nb[0],
                (long) ids->nb[1], (long) ids->nb[2], (long) ids->nb[3], ggml_op_name(ids->op), (void *) ids->view_src);

        // Print expert IDs for first few tokens
        fprintf(stderr, "[MoE DEBUG] Expert IDs:\n");
        for (int64_t iid1 = 0; iid1 < std::min((int64_t) 4, ids->ne[1]); iid1++) {
            fprintf(stderr, "  token %ld: [", (long) iid1);
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t expert_id = *(const int32_t *) (ids_host.data() + iid1 * ids->nb[1] + id * ids->nb[0]);
                fprintf(stderr, "%d", expert_id);
                if (id < n_ids - 1) {
                    fprintf(stderr, ", ");
                }
            }
            fprintf(stderr, "]\n");
        }

        // Copy src1 to host and print first few values
        std::vector<float> src1_host(std::min((size_t) 16, (size_t) ggml_nelements(src1)));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(src1_host.data(), src1->data, src1_host.size() * sizeof(float))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[MoE DEBUG] src1 first values: [");
        for (size_t i = 0; i < src1_host.size(); i++) {
            fprintf(stderr, "%.4f", src1_host[i]);
            if (i < src1_host.size() - 1) {
                fprintf(stderr, ", ");
            }
        }
        fprintf(stderr, "]\n");
    }

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *) dst->data;

    // Check if expert weights need to be cached (mmap or host memory)
    // For lazy MoE: weights are in mmap'd memory and need per-expert caching to GPU
    bool         use_expert_cache = false;
    int          layer_id         = -1;
    const size_t expert_size      = nb02;  // Bytes per expert
    {
        // Check buffer metadata first - authoritative source of truth for memory location
        // ggml_backend_buffer_is_host() returns true for CPU/mmap buffers
        // This is more reliable than sycl::get_pointer_type() which may return incorrect
        // values (e.g., "shared") for mmap'd memory on Intel systems
        bool is_host_buffer = src0->buffer && ggml_backend_buffer_is_host(src0->buffer);

        // Also check ptr_type as fallback for edge cases
        sycl::usm::alloc ptr_type = sycl::get_pointer_type(src0->data, stream->get_context());

        // Use expert cache if: host buffer (mmap/CPU) OR non-device SYCL memory
        if (is_host_buffer || ptr_type != sycl::usm::alloc::device) {
            use_expert_cache = true;
            layer_id         = ggml_sycl_tp_extract_layer_number(src0->name);
            if (layer_id < 0) {
                layer_id = static_cast<int>(reinterpret_cast<uintptr_t>(src0->data) & 0x7FFFFFFF);
            }
            GGML_SYCL_DEBUG("[MoE] Using expert cache for %s (layer=%d, host_buf=%d, ptr_type=%d)\n", src0->name,
                            layer_id, is_host_buffer, (int) ptr_type);

            // Unified cache: set reorder callback for MXFP4 (applied on cache miss)
            if (src0->type == GGML_TYPE_MXFP4) {
                ggml_sycl::set_moe_reorder_callback(reorder_callback_mxfp4);
                GGML_SYCL_DEBUG("[UNIFIED-CACHE] Set MXFP4 reorder callback\n");
            }
            // Experts are cached on-demand in the loop below via unified cache API
        }
    }

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    if (ne12 == 1) {
        // Thread-local extra for reordered cached experts
        static thread_local ggml_tensor_extra_gpu cached_extra = {};

        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1 * ids->nb[1] + id * ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

                // Get expert data - from cache if using mmap/host, or directly if on device
                const void * expert_mmap_ptr = src0_original + i02 * nb02;

                if (use_expert_cache) {
                    // Unified cache: single cache for all weights (dense + MoE)
                    void * cached_ptr;
                    if (src0->type == GGML_TYPE_MXFP4) {
                        cached_ptr = ggml_sycl::cache_moe_expert_with_reorder(*stream, expert_mmap_ptr, expert_size,
                                                                              layer_id, i02, static_cast<int>(ne00),
                                                                              static_cast<int>(ne01));
                    } else {
                        cached_ptr = ggml_sycl::cache_moe_expert(*stream, expert_mmap_ptr, expert_size, layer_id, i02);
                    }

                    if (cached_ptr) {
                        src0_row.data = cached_ptr;

                        // Set up extra with reorder flag if SoA layout was applied
                        // MXFP4 uses cache_moe_expert_with_reorder() which applies SoA
                        if (src0->type == GGML_TYPE_MXFP4) {
                            cached_extra.optimized_feature = optimize_feature(reorder_mode::SOA);
                            cached_extra.tp_sharded        = false;
                            src0_row.extra                 = &cached_extra;
                        } else {
                            src0_row.extra = nullptr;
                        }
                    } else {
                        GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to cache expert L%d:E%d\n", layer_id, i02);
                        src0_row.data = src0_original + i02 * nb02;  // Fallback (will likely fail)
                    }
                } else {
                    src0_row.data = src0_original + i02 * nb02;
                }
                src1_row.data = src1_original + i11 * nb11 + i12 * nb12;
                dst_row.data  = dst_original + i1 * nb1 + i2 * nb2;

                ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        // Try fused MoE kernel for supported quantization types
        // This eliminates the gather/scatter overhead (96 kernels -> 1 kernel)
#if SYCL_ESIMD_MOE_AVAILABLE
        static bool fused_moe_disabled = (std::getenv("GGML_SYCL_DISABLE_FUSED_MOE") != nullptr);
        // Note: Fused kernel requires contiguous device memory for all experts
        // Cannot use when weights are in mmap (use_expert_cache is true)
        // For large batches, oneDNN GEMM is faster than fused ESIMD kernel
        const bool  use_fused_moe      = !fused_moe_disabled && fused_moe_esimd_available() &&
                                   !use_expert_cache &&                        // Cannot use with mmap weights
                                   (ne12 <= GGML_SYCL_FUSED_MOE_MAX_BATCH) &&  // Large batches use oneDNN
                                   (src0->type == GGML_TYPE_MXFP4) && src1->type == GGML_TYPE_F32;

        if (use_fused_moe) {
            const int64_t num_tokens = ids->ne[1];

            if (src0->type == GGML_TYPE_Q8_0) {
                launch_fused_moe_q8_0(src0->data,                   // expert_weights
                                      src1->data,                   // input (F32)
                                      (const int32_t *) ids->data,  // expert_ids (device)
                                      (float *) dst->data,          // output
                                      nb02,                         // stride_expert (bytes between experts)
                                      ne00,                         // ncols (hidden size)
                                      ne01,                         // nrows (output size per expert)
                                      n_ids,                        // n_ids (top_k)
                                      num_tokens,                   // num_tokens
                                      ne11,                         // ne11 (src1 dim 1)
                                      ids->nb[0],                   // ids_nb0
                                      ids->nb[1],                   // ids_nb1
                                      nb11,                         // in_nb11
                                      nb12,                         // in_nb12
                                      nb1,                          // out_nb1
                                      nb2,                          // out_nb2
                                      *stream);
            } else if (src0->type == GGML_TYPE_MXFP4) {
                launch_fused_moe_mxfp4(src0->data, (const float *) src1->data, (const int32_t *) ids->data,
                                       (float *) dst->data, nb02, ne00, ne01, n_ids, num_tokens, ne11, ids->nb[0],
                                       ids->nb[1], nb11, nb12, nb1, nb2, *stream);
            }
            return;
        }
#endif

        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float) * ggml_nelements(src1));
        ggml_sycl_pool_alloc<char> dst_contiguous(ctx.pool(), sizeof(float) * ggml_nelements(dst));

        src1_row.data = src1_contiguous.get();
        dst_row.data  = dst_contiguous.get();

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

                    if (row_id_i != i02) {
                        continue;
                    }

                    num_src1_rows++;
                }
            }

            if (num_src1_rows == 0) {
                continue;
            }

            ggml_sycl_pool_alloc<int>              dev_cur_src1_row(ctx.pool(), 1);
            ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), num_src1_rows);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dev_cur_src1_row.get(), 0, sizeof(int))));

            const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
            assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

            {
                sycl::range<3> block_dims(1, 1, std::min((unsigned int) ne10, max_work_group_size));
                sycl::range<3> grid_dims(1, n_ids, ids->ne[1]);
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<int, 0> src1_row_acc(cgh);

                    char * __restrict src1_contiguous_get             = src1_contiguous.get();
                    int * __restrict dev_cur_src1_row_get             = dev_cur_src1_row.get();
                    mmid_row_mapping * __restrict dev_row_mapping_get = dev_row_mapping.get();
                    size_t ids_nb_ct6                                 = ids->nb[1];
                    size_t ids_nb_ct7                                 = ids->nb[0];

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                            k_copy_src1_to_contiguous(src1_original, src1_contiguous_get, dev_cur_src1_row_get,
                                                      dev_row_mapping_get, ids_dev, i02, ids_nb_ct6, ids_nb_ct7, ne11,
                                                      ne10, nb11, nb12, item_ct1, src1_row_acc);
                        });
                });
            }

            // Get expert data - from cache if using mmap/host, or directly if on device
            // Thread-local extra for reordered cached experts (batched path)
            static thread_local ggml_tensor_extra_gpu cached_extra_batch    = {};
            const void *                              expert_mmap_ptr_batch = src0_original + i02 * nb02;

            if (use_expert_cache) {
                // Unified cache: single cache for all weights (dense + MoE)
                void * cached_ptr;
                if (src0->type == GGML_TYPE_MXFP4) {
                    cached_ptr =
                        ggml_sycl::cache_moe_expert_with_reorder(*stream, expert_mmap_ptr_batch, expert_size, layer_id,
                                                                 i02, static_cast<int>(ne00), static_cast<int>(ne01));
                } else {
                    cached_ptr =
                        ggml_sycl::cache_moe_expert(*stream, expert_mmap_ptr_batch, expert_size, layer_id, i02);
                }

                if (cached_ptr) {
                    src0_row.data = cached_ptr;

                    // Set up extra with reorder flag if SoA layout was applied
                    // MXFP4 uses cache_moe_expert_with_reorder() which applies SoA
                    if (src0->type == GGML_TYPE_MXFP4) {
                        cached_extra_batch.optimized_feature = optimize_feature(reorder_mode::SOA);
                        cached_extra_batch.tp_sharded        = false;
                        src0_row.extra                       = &cached_extra_batch;
                    } else {
                        src0_row.extra = nullptr;
                    }
                } else {
                    GGML_LOG_ERROR("[UNIFIED-CACHE] Failed to cache expert L%d:E%d (batch)\n", layer_id, i02);
                    src0_row.data = src0_original + i02 * nb02;  // Fallback
                }
            } else {
                src0_row.data = src0_original + i02 * nb02;
            }

            GGML_ASSERT(nb11 == sizeof(float) * ne10);
            GGML_ASSERT(nb1 == sizeof(float) * ne0);
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows * nb11;
            src1_row.nb[3] = num_src1_rows * nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows * nb1;
            dst_row.nb[3] = num_src1_rows * nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);

            {
                sycl::range<3> block_dims(1, 1, std::min((unsigned int) ne0, max_work_group_size));
                sycl::range<3> grid_dims(1, 1, num_src1_rows);
                stream->submit([&](sycl::handler & cgh) {
                    const char * __restrict dst_contiguous_get              = dst_contiguous.get();
                    const mmid_row_mapping * __restrict dev_row_mapping_get = dev_row_mapping.get();

                    cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         k_copy_dst_from_contiguous(dst_original, dst_contiguous_get,
                                                                    dev_row_mapping_get, ne0, nb1, nb2, item_ct1);
                                     });
                });
            }
        }
    }

    // Debug: print output values after MoE operation
    // Note: Skip during graph recording since .wait() is incompatible with command graphs
    if (g_moe_debug_enabled && !g_ggml_sycl_graph_recording) {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        std::vector<float> dst_host(std::min((size_t) 16, (size_t) ggml_nelements(dst)));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_host.data(), dst->data, dst_host.size() * sizeof(float))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[MoE DEBUG] dst first values: [");
        for (size_t i = 0; i < dst_host.size(); i++) {
            fprintf(stderr, "%.4f", dst_host[i]);
            if (i < dst_host.size() - 1) {
                fprintf(stderr, ", ");
            }
        }
        fprintf(stderr, "]\n");
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

static void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

static void ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

static void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

static void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

static void ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

static void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

static void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}

static void ggml_sycl_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_top_k(ctx, dst);
}

static void ggml_sycl_set_main_device(const int main_device) try {
    if (dpct::get_current_device_id() == static_cast<unsigned int>(main_device)) {
        return;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n", main_device, prop.get_name());
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

// Debug: Dump tensor values for NON-FA attention path comparison
#define NON_FA_DEBUG_DUMP 0
#if NON_FA_DEBUG_DUMP
#    include <cstring>

static void dump_non_fa_attention_tensor(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) {
    // Only dump for specific tensor names related to attention
    // Key tensors: "kq", "kq_soft_max", "kqv" for non-FA path
    // Also dump "fattn" for FA path comparison
    const char * name = dst->name;
    if (!name || name[0] == '\0') {
        return;
    }

    // Check if this is an attention-related tensor
    bool is_kq          = (strncmp(name, "kq", 2) == 0 && name[2] != 'v');  // kq but not kqv
    bool is_kq_soft_max = (strncmp(name, "kq_soft_max", 11) == 0);
    bool is_kqv         = (strncmp(name, "kqv", 3) == 0 && name[3] != '_');
    bool is_fattn       = (strncmp(name, "fattn", 5) == 0);

    if (!is_kq && !is_kq_soft_max && !is_kqv && !is_fattn) {
        return;
    }

    // Static call counter for each tensor type
    static int kq_count          = 0;
    static int kq_soft_max_count = 0;
    static int kqv_count         = 0;
    static int fattn_count       = 0;

    int call_num = 0;
    if (is_kq) {
        call_num = ++kq_count;
    } else if (is_kq_soft_max) {
        call_num = ++kq_soft_max_count;
    } else if (is_kqv) {
        call_num = ++kqv_count;
    } else if (is_fattn) {
        call_num = ++fattn_count;
    }

    // Only dump first 50 calls of each type
    if (call_num > 50) {
        return;
    }

    // Wait for computation to complete
    ctx.stream()->wait();

    // Get tensor info
    const int64_t ne0            = dst->ne[0];
    const int64_t ne1            = dst->ne[1];
    const int64_t ne2            = dst->ne[2];
    const int64_t ne3            = dst->ne[3];
    const size_t  total_elements = ne0 * ne1 * ne2 * ne3;

    // Only dump F32 tensors for now
    if (dst->type != GGML_TYPE_F32) {
        fprintf(stderr, "\n[NON-FA-DEBUG] %s call=%d type=%d (not F32, skipping dump)\n", name, call_num, dst->type);
        return;
    }

    fprintf(stderr, "\n[NON-FA-DEBUG] %s call=%d shape=[%lld,%lld,%lld,%lld] total=%zu\n", name, call_num,
            (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3, total_elements);

    // Copy data to host
    std::vector<float> host_data(total_elements);
    ctx.stream()->memcpy(host_data.data(), dst->data, total_elements * sizeof(float)).wait();

    // Print summary statistics
    float min_val = host_data[0], max_val = host_data[0], sum = 0;
    for (size_t i = 0; i < total_elements; i++) {
        float v = host_data[i];
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
        sum += v;
    }
    fprintf(stderr, "  Stats: min=%.6f max=%.6f mean=%.6f\n", min_val, max_val, sum / total_elements);

    // Print first few values for each dimension
    if (is_kqv || is_fattn) {
        // For attention output: [D][n_heads][n_queries][batch] or similar
        // Print first 8 values for first few heads
        fprintf(stderr, "  Output (first 8 values per head, first query):\n");
        for (int h = 0; h < std::min((int) ne2, 8); h++) {
            fprintf(stderr, "    h=%2d: [", h);
            for (int d = 0; d < std::min((int) ne0, 8); d++) {
                // Assuming layout [D][n_queries][n_heads][batch] -> index = d + ne0*(q + ne1*(h + ne2*b))
                size_t idx = d + ne0 * (0 + ne1 * h);  // first query
                fprintf(stderr, "%.6f%s", host_data[idx], d < 7 ? ", " : "");
            }
            fprintf(stderr, "]\n");
        }
        // Print a few higher heads
        for (int h : { 8, 16, 32, 63 }) {
            if (h < (int) ne2) {
                fprintf(stderr, "    h=%2d: [", h);
                for (int d = 0; d < std::min((int) ne0, 8); d++) {
                    size_t idx = d + ne0 * (0 + ne1 * h);
                    fprintf(stderr, "%.6f%s", host_data[idx], d < 7 ? ", " : "");
                }
                fprintf(stderr, "]\n");
            }
        }
    } else if (is_kq_soft_max) {
        // For attention weights: [n_kv][n_queries][n_heads][batch]
        // Print attention weights for first query of first few heads
        fprintf(stderr, "  Attention weights (first 16 KV positions, first query):\n");
        for (int h = 0; h < std::min((int) ne2, 4); h++) {
            fprintf(stderr, "    h=%2d: [", h);
            for (int kv = 0; kv < std::min((int) ne0, 16); kv++) {
                size_t idx = kv + ne0 * (0 + ne1 * h);
                fprintf(stderr, "%.4f%s", host_data[idx], kv < 15 ? ", " : "");
            }
            fprintf(stderr, "]\n");
        }
    }

    // Write full dump to file
    char filename[256];
    snprintf(filename, sizeof(filename), "/tmp/fa_debug/nonfa_%s_call%03d.txt",
             is_kq ? "kq" : (is_kq_soft_max ? "kq_softmax" : (is_kqv ? "kqv" : "fattn")), call_num);
    FILE * f = fopen(filename, "w");
    if (f) {
        fprintf(f, "# Tensor: %s call=%d shape=[%lld,%lld,%lld,%lld]\n", name, call_num, (long long) ne0,
                (long long) ne1, (long long) ne2, (long long) ne3);
        fprintf(f, "# Strides: nb=[%zu,%zu,%zu,%zu]\n", dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        fprintf(f, "\n=== DATA ===\n");

        // Write all data organized by dimensions
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    fprintf(f, "[b=%lld,h=%lld,q=%lld]: ", (long long) i3, (long long) i2, (long long) i1);
                    for (int64_t i0 = 0; i0 < ne0; i0++) {
                        size_t idx = i0 + ne0 * (i1 + ne1 * (i2 + ne2 * i3));
                        fprintf(f, "%.6f ", host_data[idx]);
                    }
                    fprintf(f, "\n");
                }
            }
        }

        fclose(f);
        fprintf(stderr, "  [Wrote full dump to %s]\n", filename);
    }
}
#endif

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    if (!g_sycl_loaded) {
        return false;
    }

    // Debug: trace operations in multi-process mode
    if (g_sycl_tp_config.is_multiprocess && g_ggml_sycl_tp_debug) {
        static int op_trace = 0;
        if (op_trace++ < 100) {
            fprintf(stderr, "[RANK %d] COMPUTE: op=%s tensor=%s ne=[%lld,%lld,%lld,%lld]\n", g_sycl_tp_config.mpi_rank,
                    ggml_op_name(dst->op), dst->name, (long long) dst->ne[0], (long long) dst->ne[1],
                    (long long) dst->ne[2], (long long) dst->ne[3]);
            fflush(stderr);
        }
    }

    // TP MODE: Skip ALL ops on secondary devices EXCEPT TP-sharded MUL_MAT
    // In Megatron-style TP, secondary devices ONLY compute their shard of
    // TP weight matmuls. ALL other ops (ADD, RMS_NORM, ROPE, etc.) run ONLY
    // on the main device to avoid both devices writing to the same compute buffer.
    // The main device's TP MUL_MAT handlers dispatch work to secondary devices
    // and perform ALL_REDUCE to sum the results.
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        int main_device = g_sycl_tp_config.devices[0];

        if (ctx.device != main_device) {
            // On secondary device - ONLY run MUL_MAT if src0 is TP-sharded
            bool is_tp_mul_mat = false;

            if (dst->op == GGML_OP_MUL_MAT && dst->src[0] != nullptr) {
                ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->src[0]->extra;
                if (extra && extra->tp_sharded) {
                    is_tp_mul_mat = true;
                }
            }

            // Skip ALL ops except TP MUL_MAT on secondary devices
            if (!is_tp_mul_mat) {
                static int skip_log = 0;
                if (skip_log++ < 5) {
                    GGML_SYCL_DEBUG("TP: Skipping op %s on device %d (not TP MUL_MAT)\n", ggml_op_name(dst->op),
                                    ctx.device);
                }
                return true;  // Op "succeeded" by being skipped
            }
        }
    }

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET_ROWS_PAGED:
            ggml_sycl_op_set_rows_paged(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_sycl_add(ctx, dst);
            break;
        case GGML_OP_ADD1:
            ggml_sycl_add1(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_sycl_count_equal(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_sycl_top_k(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_ALL_REDUCE_SUM:
            ggml_sycl_all_reduce_sum(ctx, dst);
            break;
        default:
            return false;
    }

#if NON_FA_DEBUG_DUMP
    // Dump attention-related tensors after computation
    dump_non_fa_attention_tensor(ctx, dst);
#endif

    return true;
} catch (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP " << ggml_op_name(dst->op) << std::endl;
    std::exit(1);
}

GGML_API void ggml_backend_sycl_get_device_description(int device, char * description, size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ggml_sycl_set_device(device);

    /*
    DPCT1009:218: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    /*
    DPCT1106:217: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(device).get_memory_info(*free, *total)));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;

    return sycl_ctx->name.c_str();
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;

    // Clean up tensor parallelism resources (including CCL) BEFORE destroying backend
    // This ensures CCL objects are destroyed before MPI finalization starts
    // The function is idempotent - safe to call multiple times
    ggml_sycl_tp_free();

    // Clean up pipeline parallelism persistent transfer buffer
    ggml_sycl_free_dev2dev_transfer_buffer();

    delete sycl_ctx;
    delete backend;
}

static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *  tensor,
                                               const void *   data,
                                               size_t         offset,
                                               size_t         size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
    ggml_backend_buffer_t       buf      = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    // Accept both regular SYCL buffer type and TP host compute buffer type
    GGML_ASSERT((buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) ||
                 buf->buft == ggml_backend_sycl_host_compute_buffer_type(sycl_ctx->device)) &&
                "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy((char *) tensor->data + offset, data, size)));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t      backend,
                                               const ggml_tensor * tensor,
                                               void *              data,
                                               size_t              offset,
                                               size_t              size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
    ggml_backend_buffer_t       buf      = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    // Accept both regular SYCL buffer type and TP host compute buffer type
    GGML_ASSERT((buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) ||
                 buf->buft == ggml_backend_sycl_host_compute_buffer_type(sycl_ctx->device)) &&
                "unsupported buffer type");

    // Use backend's stream for the async memcpy
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(data, (const char *) tensor->data + offset, size)));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
    bool is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(dst->data, src->data, ggml_nbytes(dst))));
        return true;
    }

    return false;
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
    const queue_ptr             stream   = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

// =============================================================================
// Helper functions for RMS_NORM + MUL + MUL_MAT fusion
// =============================================================================

// Helper to get MUL weight (gamma) from a MUL node
// MUL has two sources; return the one that isn't the RMSNorm output
static ggml_tensor * get_mul_weight(const ggml_tensor * mul, const ggml_tensor * rms_norm_out) {
    if (mul->src[0] == rms_norm_out) {
        return mul->src[1];
    }
    return mul->src[0];
}

// Check if RMS_NORM + MUL + MUL_MAT can be fused
static bool ggml_sycl_can_fuse_rmsnorm_mulmat(const ggml_tensor * rms_norm,
                                              const ggml_tensor * mul,
                                              const ggml_tensor * mulmat) {
    // Skip fusion for small batch sizes (token generation) where overhead hurts performance
    const int64_t nrows = rms_norm->src[0]->ne[1];
    if (nrows < 8) {
        return false;
    }

    // Check input types - input to RMS_NORM must be F32
    if (rms_norm->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    // Check intermediate types
    if (rms_norm->type != GGML_TYPE_F32) {
        return false;
    }
    if (mul->type != GGML_TYPE_F32) {
        return false;
    }
    // MUL_MAT output must be F32
    if (mulmat->type != GGML_TYPE_F32) {
        return false;
    }

    // Check GEMM weight type (support common quantized types)
    ggml_type w_type = mulmat->src[0]->type;
    if (w_type != GGML_TYPE_Q4_0 && w_type != GGML_TYPE_Q4_1 && w_type != GGML_TYPE_Q8_0 && w_type != GGML_TYPE_Q4_K &&
        w_type != GGML_TYPE_Q5_K && w_type != GGML_TYPE_Q6_K) {
        return false;
    }

    // Check single GPU (no split buffers) for simplicity
    if (ggml_backend_buffer_is_sycl_split(mulmat->src[0]->buffer)) {
        return false;
    }

    // Check dimensions are aligned for Q8_1 quantization
    int64_t ncols = rms_norm->src[0]->ne[0];
    if (ncols % QK8_1 != 0) {
        return false;
    }

    // Gamma dimensions must match
    const ggml_tensor * gamma = get_mul_weight(mul, rms_norm);
    if (gamma->ne[0] != ncols) {
        return false;
    }

    return true;
}

// Fused dispatch function: RMS_NORM + MUL + MUL_MAT
// Eliminates intermediate normalized tensor by fusing into quantization
static void ggml_sycl_mul_mat_with_rmsnorm(ggml_backend_sycl_context & ctx,
                                           const ggml_tensor *         x,      // Original input (pre-RMSNorm)
                                           const ggml_tensor *         gamma,  // RMSNorm weight (gamma)
                                           const ggml_tensor *         W,      // GEMM weight (quantized)
                                           ggml_tensor *               dst,    // Output
                                           float                       eps     // RMSNorm epsilon
) {
    const int64_t nrows        = x->ne[1];                                     // Batch size (M)
    const int64_t ncols        = x->ne[0];                                     // Hidden dim (K)
    // Must use MATRIX_ROW_PADDING (512) to match MMQ expectations, not QK8_1 (32)
    const int64_t ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);

    dpct::queue_ptr stream = ctx.stream();

    // Allocate temporary buffers
    ggml_sycl_pool_alloc<float> scales_buf(ctx.pool(), nrows);
    ggml_sycl_pool_alloc<char>  q8_buf(ctx.pool(), nrows * (ncols_padded / QK8_1) * sizeof(block_q8_1));

    // Get data pointers
    const float * x_dd     = (const float *) x->data;
    const float * gamma_dd = (const float *) gamma->data;

    // Fused RMSNorm + quantization
    // This eliminates the intermediate normalized tensor (~2MB for Mistral 7B at batch 128)
    fused_rmsnorm_quantize_q8_1_sycl(x_dd,              // Input (unnormalized)
                                     gamma_dd,          // Gamma (RMSNorm weight)
                                     q8_buf.get(),      // Q8_1 output
                                     scales_buf.get(),  // Temporary scales buffer
                                     nrows, ncols, ncols_padded, eps, stream, ctx.device);

    // Get the destination pointer
    float * dst_dd = (float *) dst->data;

    // Get weight pointer
    const char * W_dd = (const char *) W->data;

    // Call the MMQ kernel through the existing dispatch function
    // We pass:
    // - src0 = W (weights tensor for metadata: type, ne[0])
    // - src1 = x (input tensor for metadata: ne[0] for assertion)
    // - dst = dst (output tensor for metadata: ne[0] for stride)
    // - src1_ddq_i = q8_buf.get() (our fused quantized activations)

    // W->ne[0] = K (in_features), W->ne[1] = N (out_features for 2D)
    // But for higher-dim weights, use ggml_nrows(W) to get total output rows
    const int64_t nrows_W = ggml_nrows(W);  // out_features (N)

    // Call the MMQ dispatch function
    // Parameters: row range = [0, nrows_W), src1_ncols = nrows (batch size)
    ggml_sycl_op_mul_mat_q(ctx,
                           W,             // src0: weights (for metadata)
                           x,             // src1: original input (for ne[0] assertion - matches hidden dim)
                           dst,           // dst: output (for ne[0] stride)
                           W_dd,          // src0_dd_i: weights data
                           nullptr,       // src1_ddf_i: unused since we provide quantized data
                           q8_buf.get(),  // src1_ddq_i: our fused quantized activations
                           dst_dd,        // dst_dd_i: output data
                           0, nrows_W,    // row_low, row_high: full output range
                           nrows,         // src1_ncols: batch size
                           ncols_padded,  // src1_padded_row_size
                           stream);
}

// Helper function to check if fusion is valid based on types
static bool ggml_sycl_check_fusion_types(const ggml_cgraph * cgraph, int node_idx, int count) {
    // All tensors in the fusion chain must be F32
    for (int i = 0; i < count; i++) {
        if (node_idx + i >= cgraph->n_nodes) {
            return false;
        }
        const ggml_tensor * node = cgraph->nodes[node_idx + i];
        if (node->type != GGML_TYPE_F32) {
            return false;
        }
        // Input to RMS_NORM must also be F32
        if (i == 0 && node->src[0]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Per-projection fusion helper functions
// Find all MUL_MAT nodes that consume the given tensor
// =============================================================================
static std::vector<std::pair<int, ggml_tensor *>> find_mulmat_consumers(const ggml_cgraph * cgraph,
                                                                        const ggml_tensor * tensor,
                                                                        int                 start_idx) {
    std::vector<std::pair<int, ggml_tensor *>> consumers;
    for (int j = start_idx; j < cgraph->n_nodes; j++) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_MUL_MAT) {
            // MUL_MAT src[1] is the activation input
            if (node->src[1] == tensor) {
                consumers.push_back({ j, node });
            }
        }
    }
    return consumers;
}

// Check if all MUL_MAT consumers can be fused with RMS_NORM
static bool can_fuse_all_projections(const ggml_tensor *                                rms_norm,
                                     const ggml_tensor *                                mul,
                                     const std::vector<std::pair<int, ggml_tensor *>> & mulmat_consumers) {
    if (mulmat_consumers.empty()) {
        return false;
    }

    // DISABLED: Per-projection fusion causes numerical differences
    // TODO: Investigate root cause
    return false;

    // Skip fusion for small batches (token generation)
    // The fusion overhead only amortizes with larger batches during prompt processing
    const int64_t nrows = rms_norm->src[0]->ne[1];
    if (nrows < 8) {
        return false;
    }

    // Check input types
    if (rms_norm->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (rms_norm->type != GGML_TYPE_F32) {
        return false;
    }
    if (mul->type != GGML_TYPE_F32) {
        return false;
    }

    // Check dimensions are aligned for Q8_1 quantization
    int64_t ncols = rms_norm->src[0]->ne[0];
    if (ncols % QK8_1 != 0) {
        return false;
    }

    // Gamma dimensions must match
    const ggml_tensor * gamma = get_mul_weight(mul, rms_norm);
    if (gamma->ne[0] != ncols) {
        return false;
    }

    // Check all MUL_MAT consumers
    for (const auto & [idx, mulmat] : mulmat_consumers) {
        // MUL_MAT output must be F32
        if (mulmat->type != GGML_TYPE_F32) {
            return false;
        }

        // Check GEMM weight type
        ggml_type w_type = mulmat->src[0]->type;
        if (w_type != GGML_TYPE_Q4_0 && w_type != GGML_TYPE_Q4_1 && w_type != GGML_TYPE_Q8_0 &&
            w_type != GGML_TYPE_Q4_K && w_type != GGML_TYPE_Q5_K && w_type != GGML_TYPE_Q6_K) {
            return false;
        }

        // No split buffers
        if (ggml_backend_buffer_is_sycl_split(mulmat->src[0]->buffer)) {
            return false;
        }
    }

    return true;
}

// Execute per-projection fusion: fuse RMS_NORM into each MUL_MAT independently
static void execute_per_projection_fusion(ggml_backend_sycl_context & ctx,
                                          const ggml_tensor *         x,      // Original input (pre-RMSNorm)
                                          const ggml_tensor *         gamma,  // RMSNorm weight (gamma)
                                          float                       eps,    // RMSNorm epsilon
                                          const std::vector<std::pair<int, ggml_tensor *>> & mulmat_consumers) {
    const int64_t nrows        = x->ne[1];                                    // Batch size (M)
    const int64_t ncols        = x->ne[0];                                    // Hidden dim (K)
    // Must use MATRIX_ROW_PADDING (512) to match MMQ expectations, not QK8_1 (32)
    const int64_t ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);

    dpct::queue_ptr stream = ctx.stream();

    // Allocate temporary buffers for fused quantization
    // These are reused for each projection
    ggml_sycl_pool_alloc<float> scales_buf(ctx.pool(), nrows);
    ggml_sycl_pool_alloc<char>  q8_buf(ctx.pool(), nrows * (ncols_padded / QK8_1) * sizeof(block_q8_1));

    // Get data pointers
    const float * x_dd     = (const float *) x->data;
    const float * gamma_dd = (const float *) gamma->data;

    // Fused RMSNorm + quantization (done once, reused for all projections)
    fused_rmsnorm_quantize_q8_1_sycl(x_dd, gamma_dd, q8_buf.get(), scales_buf.get(), nrows, ncols, ncols_padded, eps,
                                     stream, ctx.device);

    // Execute each MUL_MAT with the pre-quantized activations
    for (const auto & [idx, mulmat] : mulmat_consumers) {
        const ggml_tensor * W      = mulmat->src[0];  // GEMM weights
        float *             dst_dd = (float *) mulmat->data;
        const char *        W_dd   = (const char *) W->data;

        // row_low/row_high refers to output rows (weight matrix rows)
        // W->ne[0] is K (input features), ggml_nrows(W) is N (output features)
        const int64_t nrows_W = ggml_nrows(W);  // out_features (N)

        // Call the MMQ dispatch function with our fused quantized activations
        ggml_sycl_op_mul_mat_q(ctx,
                               W,             // src0: weights (for metadata)
                               x,             // src1: original input (for ne[0] assertion)
                               mulmat,        // dst: output (for ne[0] stride)
                               W_dd,          // src0_dd_i: weights data
                               nullptr,       // src1_ddf_i: unused
                               q8_buf.get(),  // src1_ddq_i: our fused quantized activations
                               dst_dd,        // dst_dd_i: output data
                               0, nrows_W,    // row_low, row_high: full output range
                               nrows,         // src1_ncols: batch size
                               ncols_padded,  // src1_padded_row_size
                               stream);
    }
}

// =============================================================================
// FFN Fusion (gate_proj + up_proj + GLU) helper functions
// Detects pattern: MUL_MAT(gate) + MUL_MAT(up) -> GLU
// =============================================================================

// Find GLU node that consumes the given tensor as either src[0] (gate) or src[1] (up)
// Returns {glu_index, glu_node, is_gate} where is_gate=true if tensor is src[0]
struct glu_consumer_info {
    int           idx;
    ggml_tensor * glu;
    bool          is_gate;  // true if tensor is src[0] (gate), false if src[1] (up)
};

static glu_consumer_info find_glu_consumer(const ggml_cgraph * cgraph, const ggml_tensor * tensor, int start_idx) {
    for (int j = start_idx; j < cgraph->n_nodes; j++) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_GLU) {
            // For split GLU: src[0] = gate, src[1] = up
            if (node->src[0] == tensor) {
                return { j, node, true };  // tensor is gate
            }
            if (node->src[1] == tensor) {
                return { j, node, false };  // tensor is up
            }
        }
    }
    return { -1, nullptr, false };
}

// Find another MUL_MAT that shares the same input (src[1]) and feeds into the same GLU
static std::pair<int, ggml_tensor *> find_paired_mulmat(const ggml_cgraph * cgraph,
                                                        const ggml_tensor * mulmat,
                                                        const ggml_tensor * shared_input,
                                                        int                 start_idx,
                                                        int                 end_idx) {
    for (int j = start_idx; j < end_idx && j < cgraph->n_nodes; j++) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_MUL_MAT && node != mulmat) {
            // Check if this MUL_MAT shares the same input activation
            if (node->src[1] == shared_input) {
                return { j, node };
            }
        }
    }
    return { -1, nullptr };
}

// Check if FFN fusion can be applied for these two MUL_MATs
// Conditions:
// 1. Both are Q4_0 weights (for now)
// 2. Same input tensor (src[1])
// 3. Same weight dimensions
// 4. Small batch size (token generation, not prompt)
static bool can_fuse_ffn(const ggml_tensor * gate_mulmat, const ggml_tensor * up_mulmat) {
    static int can_fuse_debug = 0;
    bool       debug_this     = (can_fuse_debug++ < 5);

    // Check weight types - only Q4_0 supported for now
    if (gate_mulmat->src[0]->type != GGML_TYPE_Q4_0 || up_mulmat->src[0]->type != GGML_TYPE_Q4_0) {
        if (debug_this) {
            fprintf(stderr, "[FFN TRACE] can_fuse_ffn: FAIL type check (gate=%d, up=%d)\n",
                    (int) gate_mulmat->src[0]->type, (int) up_mulmat->src[0]->type);
        }
        return false;
    }

    // Check same input
    if (gate_mulmat->src[1] != up_mulmat->src[1]) {
        if (debug_this) {
            fprintf(stderr, "[FFN TRACE] can_fuse_ffn: FAIL same input check\n");
        }
        return false;
    }

    // Check weight dimensions match
    const ggml_tensor * W_gate = gate_mulmat->src[0];
    const ggml_tensor * W_up   = up_mulmat->src[0];
    if (W_gate->ne[0] != W_up->ne[0] || W_gate->ne[1] != W_up->ne[1]) {
        if (debug_this) {
            fprintf(stderr, "[FFN TRACE] can_fuse_ffn: FAIL dim check\n");
        }
        return false;
    }

    // Only fuse for small batches (token generation)
    // For large batches, separate kernels with better occupancy may be faster
    const ggml_tensor * input      = gate_mulmat->src[1];
    int                 batch_size = input->ne[1];
    if (batch_size > 8) {
        if (debug_this) {
            fprintf(stderr, "[FFN TRACE] can_fuse_ffn: FAIL batch size (%d > 8)\n", batch_size);
        }
        return false;
    }

    if (debug_this) {
        fprintf(stderr, "[FFN TRACE] can_fuse_ffn: PASS! batch=%d\n", batch_size);
    }
    return true;
}

// Enable to debug FFN fusion path vs baseline path
// #define GGML_SYCL_FFN_PATH_DEBUG

// Execute FFN fusion: gate_proj + up_proj + SwiGLU in single kernel
// This eliminates intermediate gate/up tensors and reduces kernel launches
static void execute_ffn_fusion(ggml_backend_sycl_context & ctx,
                               const ggml_tensor *         gate_mulmat,
                               const ggml_tensor *         up_mulmat,
                               ggml_tensor *               glu_dst) {
    // Get tensors
    const ggml_tensor * W_gate = gate_mulmat->src[0];  // Gate weights [K, N]
    const ggml_tensor * W_up   = up_mulmat->src[0];    // Up weights [K, N]
    const ggml_tensor * input  = gate_mulmat->src[1];  // Input [batch, K]

    // Dimensions
    const int64_t K     = W_gate->ne[0];  // Hidden size (e.g., 4096)
    const int64_t N     = W_gate->ne[1];  // Intermediate size (e.g., 14336)
    const int64_t batch = input->ne[1];   // Batch size (typically 1 for token gen)

    // Debug output with full tensor shapes
    GGML_SYCL_DEBUG("[FFN FUSION] K=%lld N=%lld batch=%lld gate_type=%d up_type=%d\n", (long long) K, (long long) N,
                    (long long) batch, (int) W_gate->type, (int) W_up->type);

    // Always print shapes when debugging FFN fusion issues
    static int ffn_debug_count = 0;
    if (ffn_debug_count++ < 3) {
        fprintf(stderr, "[FFN DEBUG] input ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%d\n",
                (long long) input->ne[0], (long long) input->ne[1], (long long) input->ne[2], (long long) input->ne[3],
                input->nb[0], input->nb[1], input->nb[2], input->nb[3], (int) input->type);
        fprintf(stderr, "[FFN DEBUG] dst ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%d\n",
                (long long) glu_dst->ne[0], (long long) glu_dst->ne[1], (long long) glu_dst->ne[2],
                (long long) glu_dst->ne[3], glu_dst->nb[0], glu_dst->nb[1], glu_dst->nb[2], glu_dst->nb[3],
                (int) glu_dst->type);
        fflush(stderr);
    }

    // Get stream and device
    int             device = ctx.device;
    dpct::queue_ptr stream = ctx.stream(device, 0);

    // Calculate padded size for Q8_1 alignment
    const int64_t K_padded = (K + QK8_1 - 1) / QK8_1 * QK8_1;
    const size_t  q8_size  = batch * K_padded * sizeof(block_q8_1) / QK8_1;

    // Allocate temporary buffer for quantized input
    ggml_sycl_pool_alloc<char> input_q8_alloc(ctx.pool(device), q8_size);
    void *                     input_q8 = input_q8_alloc.get();

    // Zero padding if needed
    if (K != K_padded) {
        stream->memset(input_q8, 0, q8_size);
    }

    // Quantize input to Q8_1
    const float * input_f32 = (const float *) input->data;

#if GGML_SYCL_FFN_PATH_DEBUG
    // DEBUG: Check input f32 values BEFORE quantization
    static int input_debug = 0;
    if (input_debug++ < 5) {
        // Wait for any prior operations on this input to complete
        stream->wait();
        float input_vals[8];
        stream->memcpy(input_vals, input_f32, 8 * sizeof(float)).wait();
        fprintf(stderr, "[FFN FUSION] input_f32 BEFORE quant [0:7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                input_vals[0], input_vals[1], input_vals[2], input_vals[3], input_vals[4], input_vals[5], input_vals[6],
                input_vals[7]);
        fprintf(stderr, "[FFN FUSION] input name=%s op=%s computed=%d\n", input->name ? input->name : "(null)",
                ggml_op_name(input->op), input->data != nullptr);
        fflush(stderr);
    }
#endif

    // Use SoA quantizer if gate weights are reordered
    ggml_tensor_extra_gpu * gate_extra_fusion = static_cast<ggml_tensor_extra_gpu *>(W_gate->extra);
    const bool use_soa_ffn_fusion = gate_extra_fusion && gate_extra_fusion->optimized_feature.is_reordered();
    if (use_soa_ffn_fusion) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(input_f32, input_q8, K, batch, K_padded, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(input_f32, input_q8, K, batch, K_padded, stream);
    }

    // Get weight and output pointers
    const void * W_gate_data = W_gate->data;
    const void * W_up_data   = W_up->data;
    float *      dst_data    = (float *) glu_dst->data;

#if GGML_SYCL_FFN_PATH_DEBUG
    static int addr_debug = 0;
    if (addr_debug++ < 5) {
        fprintf(stderr, "[FFN ADDR] glu_dst=%p glu_dst->data=%p dst_data=%p name=%s\n", (void *) glu_dst,
                (void *) glu_dst->data, (void *) dst_data, glu_dst->name ? glu_dst->name : "(null)");
        fflush(stderr);
    }
#endif

    // Select kernel variant based on environment variable
    // Multi-row kernel (default): 256 threads/WG, SLM caching, 64 rows/WG
    // Single-row kernel (fallback): 16 threads/WG, no SLM, 1 row/WG
    static bool use_single_row = (getenv("GGML_SYCL_FFN_SINGLE_ROW") != nullptr);

    if (use_single_row) {
        GGML_SYCL_DEBUG("[FFN FUSION] Using single-row kernel (16 threads/WG)\n");
        ggml_sycl_ffn::fused_ffn_gate_up_swiglu_sycl<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
            W_gate_data, W_up_data, input_q8, dst_data,
            K,      // ncols_in
            N,      // nrows_out
            batch,  // batch_size
            stream);
    } else {
        GGML_SYCL_DEBUG("[FFN FUSION] Using multi-row kernel (256 threads/WG, SLM caching)\n");
        ggml_sycl_ffn::fused_ffn_multirow_sycl<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
            W_gate_data, W_up_data, input_q8, dst_data,
            K,      // ncols_in
            N,      // nrows_out
            batch,  // batch_size
            stream);
    }

#if GGML_SYCL_FFN_PATH_DEBUG
    // DEBUG: Print fusion output after kernel completes
    static int fusion_debug_count = 0;
    if (fusion_debug_count++ < 5) {
        stream->wait();
        float dst_vals[8];
        stream->memcpy(dst_vals, dst_data, 8 * sizeof(float)).wait();
        fprintf(stderr, "[FFN FUSION] call %d: dst ne=[%lld,%lld] dst[0:7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                fusion_debug_count, (long long) N, (long long) batch, dst_vals[0], dst_vals[1], dst_vals[2],
                dst_vals[3], dst_vals[4], dst_vals[5], dst_vals[6], dst_vals[7]);

        // Also print some input values for comparison
        float input_vals[8];
        stream->memcpy(input_vals, input_f32, 8 * sizeof(float)).wait();
        fprintf(stderr, "[FFN FUSION] call %d: input[0:7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                fusion_debug_count, input_vals[0], input_vals[1], input_vals[2], input_vals[3], input_vals[4],
                input_vals[5], input_vals[6], input_vals[7]);
        fflush(stderr);
    }
#endif

    GGML_SYCL_DEBUG("[FFN FUSION] Kernel launched successfully\n");
}

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    GGML_SYCL_PROFILE_SCOPE_GRAPH("graph_compute");

    // Debug: trace graph compute entry
    if (g_sycl_tp_config.is_multiprocess && g_ggml_sycl_tp_debug) {
        fprintf(stderr, "[RANK %d] GRAPH_COMPUTE_IMPL: n_nodes=%d, device=%d\n", g_sycl_tp_config.mpi_rank,
                cgraph->n_nodes, sycl_ctx->device);
        fflush(stderr);
    }

    ggml_sycl_set_main_device(sycl_ctx->device);

    if (g_sycl_tp_config.is_multiprocess && g_ggml_sycl_tp_debug) {
        fprintf(stderr, "[RANK %d] GRAPH_COMPUTE_IMPL: after set_main_device\n", g_sycl_tp_config.mpi_rank);
        fflush(stderr);
    }

    // Increment pass ID for TP FFN norm cache (detects stale cached data)
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        ggml_sycl_tp_new_pass();
    }

    if (g_sycl_tp_config.is_multiprocess && g_ggml_sycl_tp_debug) {
        fprintf(stderr, "[RANK %d] GRAPH_COMPUTE_IMPL: after tp_new_pass\n", g_sycl_tp_config.mpi_rank);
        fflush(stderr);
    }

    static bool disable_fusion_env = (getenv("GGML_SYCL_DISABLE_FUSION") != nullptr);
    // Also disable fusion when TP is enabled - fused ops don't handle TP buffers
    bool        disable_fusion = disable_fusion_env || (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1);

    // Track nodes that have been executed via fusion (to skip later)
    std::unordered_set<const ggml_tensor *> fused_nodes;

    if (g_sycl_tp_config.is_multiprocess && g_ggml_sycl_tp_debug) {
        fprintf(stderr, "[RANK %d] GRAPH_COMPUTE_IMPL: starting node loop\n", g_sycl_tp_config.mpi_rank);
        fflush(stderr);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // Debug: trace each node in multi-process mode
        if (g_sycl_tp_config.is_multiprocess && g_ggml_sycl_tp_debug) {
            static int node_trace = 0;
            if (node_trace++ < 30) {
                fprintf(stderr, "[RANK %d] NODE %d: op=%s name=%s\n", g_sycl_tp_config.mpi_rank, i,
                        ggml_op_name(node->op), node->name);
                fflush(stderr);
            }
        }

        // Skip nodes already executed via fusion - no kernel work needed
        if (fused_nodes.count(node)) {
#if GGML_SYCL_FFN_PATH_DEBUG
            static int skip_debug = 0;
            if (skip_debug++ < 20) {
                fprintf(stderr, "[FFN SKIP] Skipping fused node %d: op=%s name=%s\n", i, ggml_op_name(node->op),
                        node->name ? node->name : "(null)");
                fflush(stderr);
            }
#endif
            continue;
        }

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
            node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
#ifndef NDEBUG
        // Allow both SYCL device buffers and SYCL host buffers
        // (host buffers are used for token indices in GET_ROWS, etc.)
        auto is_supported_buft = [&](ggml_backend_buffer_type_t buft) {
            return buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) ||
                   buft == ggml_backend_sycl_host_buffer_type();
        };
        if (!is_supported_buft(node->buffer->buft)) {
            fprintf(stderr,
                    "[BUFFER-DEBUG] Node %d (%s) op=%s has wrong buffer type: %s (expected SYCL or SYCL_Host)\n", i,
                    node->name ? node->name : "(null)", ggml_op_name(node->op),
                    node->buffer->buft->iface.get_name(node->buffer->buft));
        }
        assert(is_supported_buft(node->buffer->buft));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                if (!is_supported_buft(node->src[j]->buffer->buft)) {
                    fprintf(stderr,
                            "[BUFFER-DEBUG] Node %d (%s) op=%s src[%d] (%s) has wrong buffer type: %s (expected SYCL "
                            "or SYCL_Host)\n",
                            i, node->name ? node->name : "(null)", ggml_op_name(node->op), j,
                            node->src[j]->name ? node->src[j]->name : "(null)",
                            node->src[j]->buffer->buft->iface.get_name(node->src[j]->buffer->buft));
                }
                assert(is_supported_buft(node->src[j]->buffer->buft));
            }
        }
#endif

        if (!disable_fusion && node->op == GGML_OP_RMS_NORM) {
            // Try 3-way kernel fusion: RMS_NORM + MUL + ADD
            if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, { i + 2 }) &&
                ggml_sycl_check_fusion_types(cgraph, i, 3) && ggml_is_contiguous(cgraph->nodes[i + 2])) {
                ggml_tensor * mul_node = cgraph->nodes[i + 1];
                ggml_tensor * add_node = cgraph->nodes[i + 2];
                ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, node, mul_node, add_node);
                i += 2;  // Skip the MUL and ADD nodes
                continue;
            }

            // Try per-projection fusion: RMS_NORM + MUL -> multiple MUL_MATs
            // Unlike the standard 3-way fusion, this approach:
            // 1. Finds ALL MUL_MAT nodes that consume the normalized tensor
            // 2. Does the fused RMSNorm+quantization ONCE
            // 3. Reuses the quantized result for all projections (Q, K, V or gate, up)
            // This bypasses the ggml_can_fuse_subgraph limitation where shared intermediates fail.
            // Note: Only triggers for batch size >= 8 to avoid overhead during token generation
            if (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL) {
                ggml_tensor * mul_node = cgraph->nodes[i + 1];
                // Check if MUL uses RMS_NORM output
                if (mul_node->src[0] == node || mul_node->src[1] == node) {
                    // Find all MUL_MAT consumers of the MUL output
                    auto consumers = find_mulmat_consumers(cgraph, mul_node, i + 2);
                    if (can_fuse_all_projections(node, mul_node, consumers)) {
                        // Execute fusion: RMSNorm+quantize once, then all MUL_MATs
                        ggml_tensor * x     = node->src[0];
                        ggml_tensor * gamma = get_mul_weight(mul_node, node);
                        float         eps;
                        memcpy(&eps, node->op_params, sizeof(float));

                        execute_per_projection_fusion(*sycl_ctx, x, gamma, eps, consumers);

                        // Mark MUL and all MUL_MAT consumers as fused (to skip later)
                        fused_nodes.insert(mul_node);
                        for (const auto & [idx, mulmat] : consumers) {
                            fused_nodes.insert(mulmat);
                        }
                        continue;  // Skip RMS_NORM execution (already done in fusion)
                    }
                }
            }

            // Try 3-way kernel fusion: RMS_NORM + MUL + MUL_MAT (single consumer)
            // This must be checked BEFORE RMS_NORM + MUL, otherwise the simpler fusion matches first
            // Fuses normalization into quantization step, eliminating intermediate tensor
            // NOTE: This often fails because in transformers, the MUL output (normalized tensor)
            // is shared by multiple projections (Q, K, V), so the use-count check fails.
            // The per-projection fusion above handles those cases.
            if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT }, { i + 2 })) {
                ggml_tensor * mul_node    = cgraph->nodes[i + 1];
                ggml_tensor * mulmat_node = cgraph->nodes[i + 2];

                if (ggml_sycl_can_fuse_rmsnorm_mulmat(node, mul_node, mulmat_node)) {
                    // Get original input, gamma, and GEMM weights
                    ggml_tensor * x     = node->src[0];                    // Pre-RMSNorm input
                    ggml_tensor * gamma = get_mul_weight(mul_node, node);  // Gamma (norm weight)
                    ggml_tensor * W     = mulmat_node->src[0];             // GEMM weights

                    // Get epsilon from RMS_NORM op_params
                    float eps;
                    memcpy(&eps, node->op_params, sizeof(float));

                    // Call fused dispatch
                    ggml_sycl_mul_mat_with_rmsnorm(*sycl_ctx, x, gamma, W, mulmat_node, eps);

                    i += 2;  // Skip MUL and MUL_MAT nodes
                    continue;
                }
            }

            // Try 2-way kernel fusion: RMS_NORM + MUL
            if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, { i + 1 }) &&
                ggml_sycl_check_fusion_types(cgraph, i, 2)) {
                ggml_tensor * mul_node = cgraph->nodes[i + 1];
                ggml_sycl_op_rms_norm_fused(*sycl_ctx, node, mul_node);
                i++;  // Skip the MUL node
                continue;
            }
        }

        // Try 2-way kernel fusion: ADD + RMS_NORM
        // Pattern: residual + hidden_states -> RMS_NORM (common in transformer decoder blocks)
        // The kernel writes BOTH outputs: the ADD result (for other consumers like next residual)
        // AND the RMS_NORM result. This allows fusion even when ADD has multiple consumers.
        if (!disable_fusion && node->op == GGML_OP_ADD) {
            if (i + 1 < cgraph->n_nodes) {
                ggml_tensor * next = cgraph->nodes[i + 1];
                // Check: next op is RMS_NORM and it uses this ADD's output as input
                if (next->op == GGML_OP_RMS_NORM && next->src[0] == node &&
                    ggml_sycl_check_fusion_types(cgraph, i, 2) && ggml_is_contiguous(next)) {
                    ggml_sycl_op_add_rms_norm_fused(*sycl_ctx, node, next);
                    i++;  // Skip the RMS_NORM node
                    continue;
                }
            }
        }

        // Try 2-way kernel fusion: MUL + ADD
        // Pattern: x * scale + bias (common scale+bias pattern in normalization)
        // Fuses element-wise multiply and add into single kernel pass
        if (!disable_fusion && node->op == GGML_OP_MUL) {
            if (i + 1 < cgraph->n_nodes) {
                ggml_tensor * next = cgraph->nodes[i + 1];
                // Check: next op is ADD and uses this MUL's output
                if (next->op == GGML_OP_ADD && (next->src[0] == node || next->src[1] == node) &&
                    node->type == GGML_TYPE_F32 && next->type == GGML_TYPE_F32 && ggml_is_contiguous(node) &&
                    ggml_is_contiguous(next)) {
                    // Check that MUL output is only used by this ADD
                    bool mul_only_used_by_add = true;
                    for (int j = i + 2; j < cgraph->n_nodes && mul_only_used_by_add; j++) {
                        ggml_tensor * check = cgraph->nodes[j];
                        for (int s = 0; s < GGML_MAX_SRC && check->src[s]; s++) {
                            if (check->src[s] == node) {
                                mul_only_used_by_add = false;
                                break;
                            }
                        }
                    }
                    if (mul_only_used_by_add) {
                        ggml_sycl_op_mul_add_fused(*sycl_ctx, node, next);
                        i++;  // Skip the ADD node
                        continue;
                    }
                }
            }
        }

        // Try FFN fusion: MUL_MAT(gate) + MUL_MAT(up) + GLU
        // Pattern: Two MUL_MATs sharing same input, both feeding into same GLU node
        // Fuses gate_proj + up_proj + SwiGLU into single kernel pass
        // Control via GGML_SYCL_FFN_FUSION=1 environment variable (default: OFF)
        // Set GGML_SYCL_FFN_FUSION=1 to enable, =0 or unset to disable
        // NOTE: FFN fusion is incompatible with SYCL graphs because the fusion
        // executes outside the graph recording and doesn't replay during graph execution.
        // To use FFN fusion, also set GGML_SYCL_DISABLE_GRAPH=1
        static const char * ffn_env              = getenv("GGML_SYCL_FFN_FUSION");
        static bool         ffn_fusion_requested = (ffn_env != nullptr && atoi(ffn_env) != 0);
#ifdef GGML_SYCL_GRAPH
        // FFN fusion requires graphs to be disabled (otherwise fusion doesn't replay)
        static bool ffn_fusion_enabled      = ffn_fusion_requested && g_ggml_sycl_disable_graph;
        static bool ffn_graph_warning_shown = false;
        if (ffn_fusion_requested && !g_ggml_sycl_disable_graph && !ffn_graph_warning_shown) {
            fprintf(stderr, "[FFN FUSION] Warning: FFN fusion disabled because SYCL graphs are enabled.\n");
            fprintf(stderr, "[FFN FUSION] Set GGML_SYCL_DISABLE_GRAPH=1 to enable FFN fusion.\n");
            ffn_graph_warning_shown = true;
        }
#else
        static bool ffn_fusion_enabled = ffn_fusion_requested;
#endif
        static int ffn_debug_trace = 0;
        if (!disable_fusion && ffn_fusion_enabled && node->op == GGML_OP_MUL_MAT) {
            // Check if this MUL_MAT feeds into a GLU
            auto glu_info = find_glu_consumer(cgraph, node, i + 1);
            if (ffn_debug_trace++ < 5) {
                fprintf(stderr, "[FFN TRACE] MUL_MAT node %d (%s): glu_info.glu=%p\n", i, node->name,
                        (void *) glu_info.glu);
            }
            if (glu_info.glu != nullptr) {
                // Found GLU consumer, now find the paired MUL_MAT
                ggml_tensor *       glu       = glu_info.glu;
                const ggml_tensor * other_src = glu_info.is_gate ? glu->src[1] : glu->src[0];

                if (ffn_debug_trace < 10) {
                    fprintf(stderr, "[FFN TRACE] Found GLU idx=%d, is_gate=%d, other_src=%p (%s, op=%d)\n",
                            glu_info.idx, glu_info.is_gate, (void *) other_src, other_src ? other_src->name : "null",
                            other_src ? (int) other_src->op : -1);
                }

                // Check if other_src is from a MUL_MAT
                ggml_tensor * other_mulmat = nullptr;
                int           other_idx    = -1;
                for (int j = i + 1; j < glu_info.idx; j++) {
                    ggml_tensor * check = cgraph->nodes[j];
                    if (check == other_src && check->op == GGML_OP_MUL_MAT) {
                        other_mulmat = check;
                        other_idx    = j;
                        break;
                    }
                }

                if (ffn_debug_trace < 10) {
                    fprintf(stderr, "[FFN TRACE] Paired MUL_MAT search: other_mulmat=%p, other_idx=%d\n",
                            (void *) other_mulmat, other_idx);
                }

                // If we found a paired MUL_MAT, check if fusion is possible
                if (other_mulmat != nullptr) {
                    const ggml_tensor * gate_mm = glu_info.is_gate ? node : other_mulmat;
                    const ggml_tensor * up_mm   = glu_info.is_gate ? other_mulmat : node;

                    if (can_fuse_ffn(gate_mm, up_mm)) {
                        GGML_SYCL_DEBUG("[FFN FUSION] Detected pattern at node %d: gate_mm=%d, up_mm=%d, glu=%d\n", i,
                                        glu_info.is_gate ? i : other_idx, glu_info.is_gate ? other_idx : i,
                                        glu_info.idx);

                        // Execute fused FFN kernel
                        execute_ffn_fusion(*sycl_ctx, gate_mm, up_mm, glu);

                        // Mark up MUL_MAT and GLU as fused (to skip later)
                        fused_nodes.insert(other_mulmat);
                        fused_nodes.insert(glu);

                        // Skip this gate MUL_MAT (already computed in fusion)
                        continue;
                    }
                }
            }
        }

#if GGML_SYCL_FFN_PATH_DEBUG
        // Debug: check MUL_MAT operations after GLU (potential down_proj)
        static int mulmat_debug = 0;
        if (node->op == GGML_OP_MUL_MAT && node->src[1] && mulmat_debug++ < 40) {
            ggml_tensor * input = node->src[1];
            fprintf(stderr, "[MUL_MAT EXEC] node %d: %s src[1]=%s (op=%s) data=%p\n", i,
                    node->name ? node->name : "(null)", input->name ? input->name : "(null)", ggml_op_name(input->op),
                    (void *) input->data);
            // If input is a GLU (ffn_swiglu), print its values
            if (input->op == GGML_OP_GLU || (input->name && strstr(input->name, "swiglu"))) {
                dpct::queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
                stream->wait();
                float vals[8];
                stream->memcpy(vals, input->data, 8 * sizeof(float)).wait();
                fprintf(stderr, "[MUL_MAT EXEC] GLU input[0:7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", vals[0],
                        vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
            }
            fflush(stderr);
        }
#endif

        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    // DEBUG: Check L31 weight at END of graph compute (disabled - TP working correctly)
    static int end_pass_dbg = 0;
    if (end_pass_dbg++ < 0 && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        try {
            ggml_sycl_set_device(g_sycl_tp_config.devices[0]);
            queue_ptr stream = &dpct::get_current_device().default_queue();

            struct {
                uint16_t d_bits;
                uint8_t  qs[16];
            } wblk;

            stream->memcpy(&wblk, (void *) l31_weight_addr, sizeof(wblk)).wait();
            uint16_t   d_raw = wblk.d_bits;
            sycl::half d_half;
            memcpy(&d_half, &d_raw, sizeof(sycl::half));
            float d_f = static_cast<float>(d_half);
            fprintf(stderr, "TP DEBUG END_PASS: L31 weight d=%f %s\n", d_f,
                    (d_f > 100.0f || std::isnan(d_f)) ? "CORRUPTED" : "OK");
        } catch (...) {
            fprintf(stderr, "TP DEBUG END_PASS: L31 weight check failed\n");
        }
    }
}

#ifdef GGML_SYCL_GRAPH
static bool check_graph_compatibility(ggml_cgraph * cgraph) {
    // NOTE: Multi-device check removed (December 2024)
    // Each backend context has its own device and exec_graph, so graphs can be
    // created per-device. The scheduler calls graph_compute separately for each
    // device's split, so SYCL graphs work correctly for pipeline parallelism.
    // TP mode is disabled via separate check (g_sycl_tp_config.enabled).

    // First pass: count total experts needed for lazy-moe graph preload.
    // Each MUL_MAT_ID node with mmap'd weights contributes its experts to the total.
    // graph_preload_moe_experts will try to cache ALL of them, so we need enough slots.
    size_t total_experts_needed = 0;
    int    moe_node_count       = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        const ggml_tensor * src0 = cgraph->nodes[i]->src[0];
        if (!src0 || !src0->buffer) {
            continue;
        }
        bool is_sycl_buf = ggml_backend_buffer_is_sycl(src0->buffer);
        if (!is_sycl_buf) {
            total_experts_needed += static_cast<size_t>(src0->ne[2]);
            moe_node_count++;
        }
    }

    // Check if unified cache is enabled for lazy-moe
    // Graph preloading will handle the actual capacity check and caching
    if (total_experts_needed > 0) {
        if (!ggml_sycl::unified_cache_enabled()) {
            GGML_SYCL_DEBUG("[GRAPH-CHECK] lazy-moe: unified cache disabled, graphs allowed (%zu experts)\n",
                            total_experts_needed);
        } else {
            GGML_SYCL_DEBUG("[GRAPH-CHECK] lazy-moe: unified cache enabled, %zu experts from %d nodes\n",
                            total_experts_needed, moe_node_count);
        }
        // Continue - graph_preload_moe_experts() will do detailed capacity check
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_op node_op = cgraph->nodes[i]->op;
        switch (node_op) {
            default:
                break;
            case GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(node_op));
                return false;
            case GGML_OP_MUL_MAT_ID:
                {
                    // MoE MUL_MAT_ID graph compatibility:
                    // - ne12 in (1, 32]: fused MoE kernel (Q8_0, MXFP4) - no allocations
                    // - ne12 == 1 (decode): MMVQ kernel (Q4_0, Q8_0, MXFP4) - pre-allocated buffers
                    // - ne12 > 32: falls back to host-side oneDNN batching (graph-incompatible)
                    const ggml_tensor * node = cgraph->nodes[i];
                    const ggml_tensor * src0 = node->src[0];
                    const ggml_tensor * src1 = node->src[1];
                    const int64_t       ne12 = src1 ? src1->ne[2] : 1;

                    // Check if expert weights are in device memory
                    // Total expert capacity check was already done above for lazy-moe
                    bool is_sycl_buf = src0->buffer && ggml_backend_buffer_is_sycl(src0->buffer);
                    GGML_SYCL_DEBUG("[GRAPH-CHECK] MUL_MAT_ID: src0=%s buffer=%p is_sycl=%d\n", src0->name,
                                    (void *) src0->buffer, is_sycl_buf);
                    if (!src0->buffer) {
                        GGML_LOG_INFO("%s: WARNING: MUL_MAT_ID src0 has NULL buffer (tensor=%s)\n", __func__,
                                      src0->name);
                    }

                    bool graph_compatible = false;
                    // Uses GGML_SYCL_FUSED_MOE_MAX_BATCH defined at file scope
                    if (ggml_is_quantized(src0->type)) {
                        if (ne12 > 1 && ne12 <= GGML_SYCL_FUSED_MOE_MAX_BATCH) {
                            // Small batched prefill (2-32 tokens) - fused MoE kernel (Q8_0, MXFP4)
                            // These kernels don't do dynamic allocations
                            // Large batches (>32) fall back to host-side oneDNN batching which
                            // requires stream->wait() and is incompatible with graph recording
                            switch (src0->type) {
                                case GGML_TYPE_Q8_0:
                                case GGML_TYPE_MXFP4:
                                    graph_compatible = true;
                                    break;
                                default:
                                    break;
                            }
                        } else if (ne12 <= 1) {
                            // Decode (ne12 == 1) - MMVQ kernel with pre-allocated Q8_1 buffers
                            switch (src0->type) {
                                case GGML_TYPE_Q4_0:
                                case GGML_TYPE_Q8_0:
                                case GGML_TYPE_MXFP4:
                                    graph_compatible = true;
                                    break;
                                default:
                                    break;
                            }
                        }
                    }

                    if (!graph_compatible) {
                        static bool logged_once = false;
                        if (!logged_once) {
                            GGML_LOG_INFO("%s: disabling SYCL graphs for MUL_MAT_ID (type=%s, ne12=%ld)\n", __func__,
                                          ggml_type_name(src0->type), (long) ne12);
                            logged_once = true;
                        }
                        return false;
                    }
                    // Graph-compatible path available
                }
                break;
            case GGML_OP_MUL_MAT:
                // MUL_MAT is graph-compatible because we pre-reorder ALL tensors
                // before graph recording starts (via graph_pre_reorder_all).
                // No malloc/free calls happen during recording.
                // Note: MoE models now support graph recording via pre-allocated Q8_1 buffers
                // (ggml_sycl_moe_pre_allocate_buffers). The MUL_MAT_ID handler checks for
                // pre-allocated buffers during recording and falls back to pool allocation
                // when not recording.
                break;
        }
    }
    return true;
}

// Pre-allocate Q8_1 buffers for SoA MMVQ operations before graph recording.
// MUL_MAT with SoA reorder flag needs Q8_1 quantization buffers which cannot be
// allocated from pool during graph recording (pointer would change on replay).
static void ggml_sycl_mmvq_soa_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Skip if already initialized with sufficient buffers
    if (ctx.mmvq_soa_buffers.initialized) {
        ctx.mmvq_soa_buffers.reset_usage();
        return;
    }

    queue_ptr stream = ctx.stream();

    // Count MUL_MAT nodes with SoA reorder flag and find max dimensions
    int     soa_mmvq_count = 0;
    int64_t max_ne10       = 0;
    int64_t max_nrows      = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        const ggml_tensor * src0 = node->src[0];  // Weights
        const ggml_tensor * src1 = node->src[1];  // Activations

        // Check for any reorder optimization (count all reordered tensors)
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (!extra || !extra->optimized_feature.is_reordered()) {
            continue;
        }

        // Only count quantized types that go through MMVQ path
        if (!ggml_is_quantized(src0->type)) {
            continue;
        }

        // Check if this would use MMVQ (batch=1 decode phase)
        const int64_t ne11 = src1->ne[1];
        if (ne11 != 1) {
            continue;  // MMVQ only for batch=1
        }

        soa_mmvq_count++;

        // Track max dimensions
        const int64_t ne10  = src1->ne[0];
        const int64_t nrows = src1->ne[1] * src1->ne[2];

        if (ne10 > max_ne10) {
            max_ne10 = ne10;
        }
        if (nrows > max_nrows) {
            max_nrows = nrows;
        }
    }

    if (soa_mmvq_count == 0) {
        return;  // No SoA MMVQ operations
    }

    // Calculate buffer size (use max dimensions for all buffers)
    // Q8_1 format: 32 int8 quants + 2 half2 scales per block
    const int64_t ne10_padded   = GGML_PAD(max_ne10, MATRIX_ROW_PADDING);
    const int64_t q8_1_row_size = ne10_padded * sizeof(block_q8_1) / QK8_1;
    const size_t  buffer_size   = max_nrows * q8_1_row_size;

    GGML_SYCL_DEBUG("[MMVQ-SOA-GRAPH] Pre-allocating %d Q8_1 buffers, %zu bytes each (ne10=%lld, nrows=%lld)\n",
                    soa_mmvq_count, buffer_size, (long long) max_ne10, (long long) max_nrows);

    // Allocate buffers
    ctx.mmvq_soa_buffers.src1_ddq_buffers.resize(soa_mmvq_count);
    ctx.mmvq_soa_buffers.src1_ddq_sizes.resize(soa_mmvq_count);

    for (int i = 0; i < soa_mmvq_count; i++) {
        ctx.mmvq_soa_buffers.src1_ddq_buffers[i] = sycl::malloc_device(buffer_size, *stream);
        ctx.mmvq_soa_buffers.src1_ddq_sizes[i]   = buffer_size;

        if (!ctx.mmvq_soa_buffers.src1_ddq_buffers[i]) {
            GGML_LOG_ERROR("[MMVQ-SOA-GRAPH] Failed to allocate Q8_1 buffer %d\n", i);
            // Cleanup and abort
            for (int j = 0; j < i; j++) {
                sycl::free(ctx.mmvq_soa_buffers.src1_ddq_buffers[j], *stream);
            }
            ctx.mmvq_soa_buffers.src1_ddq_buffers.clear();
            ctx.mmvq_soa_buffers.src1_ddq_sizes.clear();
            return;
        }
    }

    ctx.mmvq_soa_buffers.max_ne10    = max_ne10;
    ctx.mmvq_soa_buffers.max_nrows   = max_nrows;
    ctx.mmvq_soa_buffers.initialized = true;
    ctx.mmvq_soa_buffers.reset_usage();

    // Wait for allocations to complete
    stream->wait();

    GGML_SYCL_DEBUG("[MMVQ-SOA-GRAPH] Pre-allocated %d buffers successfully\n", soa_mmvq_count);

    // Buffer aliasing debug - print all pre-allocated buffer addresses
    if (getenv("GGML_SYCL_BUFFER_ALIAS_DEBUG") != nullptr) {
        fprintf(stderr, "[BUFFER_ALIAS] Pre-allocated %d Q8_1 buffers:\n", soa_mmvq_count);
        for (int i = 0; i < soa_mmvq_count; i++) {
            const uintptr_t addr = reinterpret_cast<uintptr_t>(ctx.mmvq_soa_buffers.src1_ddq_buffers[i]);
            const size_t    size = ctx.mmvq_soa_buffers.src1_ddq_sizes[i];
            fprintf(stderr, "[BUFFER_ALIAS]   prealloc[%d]: %p - %p (size=%zu)\n", i, (void *) addr,
                    (void *) (addr + size), size);
        }
    }
}
#endif

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

    // Enable SoA reordering for optimized MMQ kernels
    pre_reorder_all_tensors(sycl_ctx, cgraph);

    // SoA reordering support when graphs are disabled.
    // This enables proper comparison between graph/non-graph paths for debugging.
    // Check if GGML_SYCL_SOA_PROMPT is set (used for both prompt and decode phases when enabled).
    static bool soa_always_enabled = (std::getenv("GGML_SYCL_SOA_PROMPT") != nullptr);
    if (soa_always_enabled && g_ggml_sycl_disable_graph) {
        // Detect if this is decode phase (single token) or prompt phase
        bool is_decode = false;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
                const ggml_tensor * src1 = cgraph->nodes[i]->src[1];
                if (src1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1) {
                    is_decode = true;
                }
                break;
            }
        }

        // For non-graph mode with SOA enabled, still do pre-reordering
        static bool first_reorder_log = true;
        if (first_reorder_log && (is_decode || soa_always_enabled)) {
            GGML_SYCL_DEBUG("[SYCL-SOA] SoA reordering enabled without graphs (GGML_SYCL_SOA_PROMPT=%d, phase=%s)\n",
                            soa_always_enabled ? 1 : 0, is_decode ? "decode" : "prompt");
            first_reorder_log = false;
        }

        // Reorder tensors for SoA layout
        graph_pre_reorder_all(*sycl_ctx, cgraph);
    }

#ifdef GGML_SYCL_GRAPH
    // Disable SYCL graph for TP mode - we need our handlers to run every pass for caching
    // Note: multi-GPU lazy-moe with global expert cache is now supported via pre-loading.
    // check_graph_compatibility() validates that cache can hold all layer experts.
    bool use_sycl_graph = !g_ggml_sycl_disable_graph && check_graph_compatibility(cgraph) &&
                          !(g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1);

    // Graph debug output (controlled by GGML_SYCL_DEBUG)
    static int graph_call_count = 0;
    graph_call_count++;
    if (g_ggml_sycl_debug) {
        fprintf(stderr, "[SYCL-GRAPH] call #%d: use_sycl_graph=%d, async_mem=%d, n_nodes=%d, has_exec_graph=%d\n",
                graph_call_count, use_sycl_graph, g_ggml_sycl_use_async_mem_op, cgraph->n_nodes,
                sycl_ctx->exec_graph ? 1 : 0);
        fflush(stderr);
    }

    // Check if graphs were disabled due to MoE preload failure (persists until model reload)
    if (sycl_ctx->moe_graphs_disabled) {
        GGML_SYCL_DEBUG("[SYCL-GRAPH] graphs disabled due to MoE preload failure\n");
        use_sycl_graph = false;
    }

    if (use_sycl_graph) {
        const bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        if (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        // Check if we're in decode phase (ne[1]==1 for MUL_MAT activations).
        // IMPORTANT: Skip SYCL graphs entirely during prompt phase because:
        // 1. Prompt phase uses non-reordered kernels
        // 2. Pre-reorder would corrupt prompt computation
        // 3. Decode phase uses reordered kernels after first decode reorders tensors
        bool is_decode_phase = false;
        bool is_prompt_phase = false;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
                const ggml_tensor * src1 = cgraph->nodes[i]->src[1];
                if (src1) {
                    if (src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1) {
                        is_decode_phase = true;
                    } else {
                        is_prompt_phase = true;
                    }
                }
                break;  // Only need to check first MUL_MAT
            }
        }
        // Original code with update disabled for now:
        // if (!sycl_ctx->exec_graph || !graph_update_support) {
        //     auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
        //                                              model_sycl_graph.finalize();
        //     sycl_ctx->exec_graph = std::make_unique<
        //         sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        // } else {
        //     try {
        //         sycl_ctx->exec_graph->update(model_sycl_graph);
        //         GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
        //     } catch (sycl::exception const & e) {
        //         GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
        //         auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
        //         sycl_ctx->exec_graph = std::make_unique<
        //             sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        //     }
        // }

        // Note: Prompt phase graph recording is now supported.
        // oneDNN primitives are cached during warmup (first inference) via DnnlPrimitiveCache
        // in gemm.hpp. During graph recording, cached primitives are reused (no JIT compilation),
        // making the execute() calls graph-compatible.
        //
        // Coalesced memory access optimization for MoE is applied during decode phase warm-up
        // via graph_convert_to_coalesced() below, not during prompt phase.

        // Minimum nodes to benefit from graph batching - skip tiny graphs
        constexpr int MIN_GRAPH_NODES = 10;
        if (cgraph->n_nodes < MIN_GRAPH_NODES) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] skipping - graph too small (%d < %d nodes)\n", cgraph->n_nodes,
                            MIN_GRAPH_NODES);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        // Check if cached graph matches current graph structure and phase.
        // Different n_nodes or different phase means we need to re-record.
        // Prompt and decode phases have same n_nodes but different matrix dimensions.
        // IMPORTANT: This must run BEFORE ALL pre-allocation checks below, because
        // pre-allocation checks use !exec_graph to determine if buffers are needed.
        if (sycl_ctx->exec_graph) {
            bool invalidate = false;
            if (sycl_ctx->exec_graph_n_nodes != cgraph->n_nodes) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] invalidating cache - n_nodes changed (%d -> %d)\n",
                                sycl_ctx->exec_graph_n_nodes, cgraph->n_nodes);
                invalidate                      = true;
                sycl_ctx->warmup_decode_n_nodes = 0;  // Force re-warmup for new topology
                sycl_ctx->warmup_prompt_n_nodes = 0;
            } else if (sycl_ctx->exec_graph_is_decode != is_decode_phase) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] invalidating cache - phase changed (%s -> %s)\n",
                                sycl_ctx->exec_graph_is_decode ? "decode" : "prompt",
                                is_decode_phase ? "decode" : "prompt");
                invalidate = true;
            }
            if (invalidate) {
                sycl_ctx->exec_graph.reset();
                sycl_ctx->exec_graph_n_nodes = 0;
                // Unpin expert cache slots when graph is invalidated
                // This allows slots to be evicted for the new graph recording
                graph_unpin_moe_experts(sycl_ctx);
                graph_unpin_weights(sycl_ctx);
            }
        }

        // Pre-reorder ALL tensors before graph recording (decode phase only by default).
        // This ensures we don't have incremental reordering blocking graph reuse.
        // NOTE: With GGML_SYCL_SOA_PROMPT=1, SoA reordering is also enabled for prompt phase.
        // This works because MMQ now has SoA-aware kernels for Q4_0 (and potentially other types).
        // Only run when exec_graph is null (before graph recording or after invalidation).
        static bool soa_prompt_enabled = (std::getenv("GGML_SYCL_SOA_PROMPT") != nullptr);
        const bool  should_reorder =
            (is_decode_phase || soa_prompt_enabled) && !sycl_ctx->exec_graph && graph_needs_reorder(*sycl_ctx, cgraph);
        if (should_reorder) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] pre-reordering all tensors before graph recording (%s phase)\n",
                            is_decode_phase ? "decode" : "prompt");
            graph_pre_reorder_all(*sycl_ctx, cgraph);
        }

        // Convert reordered tensors to coalesced layout for better memory access patterns.
        // This must run AFTER reorder (either from prompt phase via opt_for_reorder, or from
        // graph_pre_reorder_all above). Safe to call even if tensors are already converted.
        if (!sycl_ctx->exec_graph && (is_decode_phase || g_ggml_sycl_reorder_mode == reorder_mode::COALESCED)) {
            graph_convert_to_coalesced(*sycl_ctx, cgraph);
        }

        // Pre-allocate V2 partition attention buffers before graph recording.
        // This ensures V2 dispatch works during graph recording (malloc/free forbidden during recording).
        if (is_decode_phase && !sycl_ctx->exec_graph) {
            ggml_sycl_v2_pre_allocate_buffers(*sycl_ctx, cgraph);
        }

        // Pre-allocate MoE Q8_1 buffers before graph recording.
        // MUL_MAT_ID normally allocates these dynamically, which is incompatible with graph recording.
        if (is_decode_phase && !sycl_ctx->exec_graph) {
            ggml_sycl_moe_pre_allocate_buffers(*sycl_ctx, cgraph);
        }

        // Pre-allocate SoA MMVQ Q8_1 buffers before graph recording.
        // MUL_MAT with SoA reorder normally allocates from pool, which returns different pointers on replay.
        if (is_decode_phase && !sycl_ctx->exec_graph) {
            ggml_sycl_mmvq_soa_pre_allocate_buffers(*sycl_ctx, cgraph);
        }

        // Pre-load and pin all MoE experts before graph recording (lazy-moe with mmap'd weights).
        // This ensures stable cache slot pointers during graph execution.
        // Slots remain pinned until graph is invalidated.
        if (!sycl_ctx->exec_graph) {
            if (!graph_preload_moe_experts(*sycl_ctx, cgraph)) {
                GGML_LOG_WARN("[SYCL-GRAPH] Expert pre-load failed, disabling graphs for all splits\n");
                sycl_ctx->moe_graphs_disabled = true;  // Disable graphs for all subsequent splits
                graph_unpin_moe_experts(sycl_ctx);     // Clean up any partial pins
                ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
                return GGML_STATUS_SUCCESS;
            }
        }

        // Pre-load and pin all dense weights before graph recording (weight streaming mode).
        // This ensures stable cache slot pointers during graph execution.
        if (!sycl_ctx->exec_graph && !sycl_ctx->weight_streaming_graphs_disabled) {
            if (!graph_preload_weights(*sycl_ctx, cgraph)) {
                GGML_LOG_WARN("[SYCL-GRAPH] Weight pre-load failed, disabling graphs for weight streaming\n");
                graph_unpin_weights(sycl_ctx);  // Clean up any partial pins
                ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
                return GGML_STATUS_SUCCESS;
            }
        }

        // Reset MoE buffer usage counter at start of each graph execution.
        // This ensures buffers are reused in the same order each time.
        if (sycl_ctx->moe_buffers.initialized) {
            sycl_ctx->moe_buffers.reset_usage();
        }

        // Reset SoA MMVQ buffer usage counter at start of each graph execution.
        // This ensures buffers are reused in the same order each time.
        if (sycl_ctx->mmvq_soa_buffers.initialized) {
            sycl_ctx->mmvq_soa_buffers.reset_usage();
        }

        // Invalidate Q8_1 quantization cache at start of each graph execution.
        // Same src1 pointer may hold different data in a new forward pass.
        sycl_ctx->moe_q8_cache.invalidate();

        // Warmup pass: If this phase hasn't been warmed up, run without graph recording
        // to populate the oneDNN primitive cache. This avoids JIT compilation during
        // graph recording which is incompatible with SYCL command graphs.
        // Note: Prompt and decode phases have same n_nodes but different matrix dimensions,
        // so we track warmup per-phase to ensure primitives are cached for both.
        int & warmup_n_nodes = is_decode_phase ? sycl_ctx->warmup_decode_n_nodes : sycl_ctx->warmup_prompt_n_nodes;
        if (warmup_n_nodes != cgraph->n_nodes) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] warmup pass for %s phase (n_nodes=%d, warmed=%d)\n",
                            is_decode_phase ? "decode" : "prompt", cgraph->n_nodes, warmup_n_nodes);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            warmup_n_nodes = cgraph->n_nodes;
            GGML_SYCL_DEBUG("[SYCL-GRAPH] warmup complete for %s phase\n", is_decode_phase ? "decode" : "prompt");
            return GGML_STATUS_SUCCESS;
        }

        // If we already have an executable graph with matching structure, just execute it.
        // We don't need to re-record because the kernel arguments (buffer pointers)
        // are captured by reference in SYCL kernels, so updates happen automatically.
        if (sycl_ctx->exec_graph) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] execute existing graph...\n");
            sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
            GGML_SYCL_DEBUG("[SYCL-GRAPH] execute done\n");
        } else {
            // First time - record and finalize the graph
            GGML_SYCL_DEBUG("[SYCL-GRAPH-DEBUG] Creating command_graph for %d nodes...\n", cgraph->n_nodes);
            sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()),
                                                    { sycl_ex::property::graph::assume_buffer_outlives_graph{} });

#    if GGML_SYCL_DNNL
            // Pre-allocate oneDNN scratchpad pool before graph recording
            // This avoids memory allocation during recording which is incompatible with SYCL graphs
            size_t max_scratchpad = get_dnnl_primitive_cache().get_max_scratchpad_size();
            if (max_scratchpad > 0) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Pre-allocating scratchpad: %zu bytes\n", max_scratchpad);
                sycl_ctx->pre_allocate_scratchpad(max_scratchpad, sycl_ctx->stream());
            }
#    endif

            GGML_SYCL_DEBUG("[SYCL-GRAPH-DEBUG] begin_recording...\n");
            g_ggml_sycl_graph_recording = true;  // Mark recording state
            model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
            GGML_SYCL_DEBUG("[SYCL-GRAPH-DEBUG] calling compute_impl...\n");
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            GGML_SYCL_DEBUG("[SYCL-GRAPH-DEBUG] end_recording...\n");
            model_sycl_graph.end_recording();
            g_ggml_sycl_graph_recording = false;  // Clear recording state

            GGML_SYCL_DEBUG("[SYCL-GRAPH] finalize (new graph)...\n");
            auto exec_graph = model_sycl_graph.finalize();
            GGML_SYCL_DEBUG("[SYCL-GRAPH] finalize done, creating unique_ptr...\n");
            sycl_ctx->exec_graph =
                std::make_unique<sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            sycl_ctx->exec_graph_n_nodes   = cgraph->n_nodes;  // Track for cache validation
            sycl_ctx->exec_graph_is_decode = is_decode_phase;  // Track which phase this graph is for
            GGML_SYCL_DEBUG("[SYCL-GRAPH] unique_ptr created, cached n_nodes=%d, phase=%s\n", cgraph->n_nodes,
                            is_decode_phase ? "decode" : "prompt");

            GGML_SYCL_DEBUG("[SYCL-GRAPH] execute new graph...\n");
            sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
            GGML_SYCL_DEBUG("[SYCL-GRAPH] execute done\n");
        }
    } else
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;

    sycl::event * sycl_event = static_cast<sycl::event *>(event->context);

    const queue_ptr & stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event * sycl_event = static_cast<sycl::event *>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } else {
        GGML_ABORT("fatal error");
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,  // ggml_backend_sycl_cpy_tensor_async,
                                            // // TODO: update for the new
                                            // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d,
                              0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    return &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

int ggml_backend_sycl_get_device_count() {
    return ggml_sycl_info().device_count;
}

// backend device

struct ggml_backend_sycl_device_context {
    int         device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    ggml_sycl_set_device(ctx->device);
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).get_memory_info(*free, *total)));
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    return ggml_backend_sycl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    return ggml_backend_sycl_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_sycl_host_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                                           void *             ptr,
                                                                           size_t             size,
                                                                           size_t             max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *) dev->context;
    int                                device   = sycl_ctx->device;
    switch (op->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_ELU:
                    return true;
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
#if defined(GGML_SYCL_F16)
                    return ggml_is_contiguous(op->src[0]) && (op->type == op->src[0]->type);
#else
                    return ggml_is_contiguous(op->src[0]) &&
                           (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) &&
                           (op->type == op->src[0]->type);
#endif
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];

                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                ggml_type a_type = a->type;
                if (a_type == GGML_TYPE_IQ4_NL || a_type == GGML_TYPE_IQ4_XS || a_type == GGML_TYPE_IQ3_XXS ||
                    a_type == GGML_TYPE_IQ3_S || a_type == GGML_TYPE_IQ2_XXS || a_type == GGML_TYPE_IQ2_XS ||
                    a_type == GGML_TYPE_IQ2_S || a_type == GGML_TYPE_IQ1_S || a_type == GGML_TYPE_IQ1_M) {
                    if (b->ne[1] == 1 && ggml_nrows(b) > 1) {
                        return false;
                    }
                }
                ggml_type src0_type = op->src[0]->type;
                if (src0_type == GGML_TYPE_BF16) {
                    // TODO: support BF16 in mul_mat properly
                    // FIXME: keep a list of supported types to avoid breaking the backend when a new type is added
                    return false;
                }
                // Note: MXFP4 is supported via MMVQ kernel in mmvq.cpp for MUL_MAT_ID
                // TODO: The configuration below needs more work to be supported with oneDNN
                if (ggml_is_permuted(a) && !ggml_is_contiguous(a) && a->ne[2] > 1 && a->ne[3] > 1 &&
                    src0_type == GGML_TYPE_F16) {
                    return false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                if (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 && a->ne[0] > 128 &&
                    a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    return false;
                }
                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q6_K:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_SET:
            return (op->type == GGML_TYPE_F32) && (op->src[0] && op->src[1]) && (op->src[0]->type == GGML_TYPE_F32) &&
                   (op->src[1]->type == GGML_TYPE_F32);

        case GGML_OP_SET_ROWS:
            {
                return ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                         op->type == GGML_TYPE_F8_E4M3 ||  // FP8 KV cache support
                         op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q5_0 ||
                         op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_IQ4_NL) &&
                        (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32));
            }
            break;
        case GGML_OP_SET_ROWS_PAGED:
            {
                // Paged attention KV write with type conversion support
                // src[0] = source data, src[1] = indices, src[2] = block_table, src[3] = dst_orig
                if (op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr || op->src[3] == nullptr) {
                    return false;
                }
                // Source type must be F32 or F16
                const bool src_type_ok     = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                // Destination type (op->type which is view of dst_orig) must be F32 or F16
                const bool dst_type_ok     = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                // Indices can be I32 or I64
                const bool indices_type_ok = (op->src[1]->type == GGML_TYPE_I32 || op->src[1]->type == GGML_TYPE_I64);
                // Block table must be I32
                const bool block_table_type_ok = (op->src[2]->type == GGML_TYPE_I32);
                return src_type_ok && dst_type_ok && indices_type_ok && block_table_type_ok;
            }
            break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == src1_type && (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) &&
                    src0_type != GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                // F32 <-> I32 conversions
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                // BF16 conversions
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32;
            }
        case GGML_OP_CONCAT:
        case GGML_OP_DUP:
        case GGML_OP_ARGMAX:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ADD_ID:
        case GGML_OP_SUB:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_REPEAT:
            return true;
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_is_contiguous(op->src[0]) && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
#if defined(GGML_SYCL_F16)
            return ((op->type == GGML_TYPE_F32 || op->type == GGML_SYCL_F16) &&
                    (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_SYCL_F16) &&
                    (op->type == op->src[0]->type));
#else
            return (op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32) && (op->type == op->src[0]->type);
#endif
        case GGML_OP_NORM:
            return true;
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_RMS_NORM:
            return ((op->src[0]->ne[0] % WARP_SIZE) == 0);
        case GGML_OP_RMS_NORM_BACK:
            return ((op->src[0]->ne[0] % WARP_SIZE) == 0);
        case GGML_OP_SCALE:
            return true;
        case GGML_OP_CONT:
            return op->src[0]->type != GGML_TYPE_BF16;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK:
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
                return max_bias == 0.0f;
            }
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
            return true;
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32 && op->op_params[0] == GGML_SCALE_MODE_NEAREST &&
                   !(op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return op->src[0]->ne[0] * sizeof(int) <= ggml_sycl_info().devices[device].smpbo;
        case GGML_OP_TOP_K:
            // TOP_K uses shared memory: BLOCK_SIZE * MAX_K * (sizeof(float) + sizeof(int))
            // = 256 * 32 * 8 = 64KB (well within typical 128KB SLM limit)
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && op->ne[0] <= SYCL_TOPK_MAX_K;
        case GGML_OP_POOL_2D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_PAD:
            // TODO: add circular padding support for syscl, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
            return true;
        case GGML_OP_SSM_CONV:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ROLL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_sycl_flash_attn_ext_supported(op);
        case GGML_OP_ALL_REDUCE_SUM:
            // All-reduce sum is supported for F32 and F16
            return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Regular SYCL buffer type
    if (buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name) {
        ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *) buft->context;
        ggml_backend_sycl_device_context *      sycl_ctx = (ggml_backend_sycl_device_context *) dev->context;
        return buft_ctx->device == sycl_ctx->device;
    }

    // TP buffer type - check if the device is one of the TP devices
    if (buft->iface.get_name == ggml_backend_sycl_tp_buffer_type_name) {
        ggml_backend_sycl_tp_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_tp_buffer_type_context *) buft->context;
        ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *) dev->context;
        // Check if this device is in the list of TP devices
        for (int dev_id : buft_ctx->devices) {
            if (dev_id == sycl_ctx->device) {
                return true;
            }
        }
        return false;
    }

    // Host buffer type (CPU interface) - all SYCL devices can access host memory
    if (buft->iface.get_name == ggml_backend_sycl_host_buffer_type_name) {
        return true;
    }

    // Host compute buffer type (SYCL interface) - all SYCL devices can access host memory
    // This is used for TP compute buffers to allow cross-device data sharing
    if (buft->iface.get_name == ggml_backend_sycl_host_compute_buffer_type_name) {
        return true;
    }

    return false;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // Weight streaming mode: offload ALL ops to keep data flow on GPU
    // Without this, MUL_MAT runs on GPU but RMS_NORM/MUL/ADD stay on CPU = data mismatch
    static bool weight_streaming = (std::getenv("GGML_SYCL_WEIGHT_STREAMING") != nullptr);
    if (weight_streaming) {
        // Offload everything SYCL can handle - ensures consistent GPU-side execution
        return true;
    }

    // For MUL_MAT_ID with types that CPU can't handle (like MXFP4), always offload
    // This is critical for lazy MoE where weights are in mmap (CPU_Mapped buffer)
    // but require GPU kernels for computation
    if (op->op == GGML_OP_MUL_MAT_ID && op->src[0]) {
        ggml_type src_type = op->src[0]->type;
        // MXFP4 requires GPU (CPU doesn't support it)
        if (src_type == GGML_TYPE_MXFP4) {
            return true;
        }
    }

    // ADD_ID is used for MoE output scatter - must match MUL_MAT_ID's backend
    // If we offload MUL_MAT_ID, we must also offload ADD_ID to avoid
    // tensor buffer mismatches between SYCL and CPU backends
    if (op->op == GGML_OP_ADD_ID) {
        return true;
    }

    const int min_batch_size = 32;
    return get_op_batch_size(op) >= min_batch_size;
    GGML_UNUSED(dev);
}

static ggml_backend_event_t ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_SYCL_NO_PEER_COPY
    return nullptr;
#else
    sycl::event * event_ptr = new sycl::event();

    return new ggml_backend_event{
        /* .device = */ dev,
        /* .context = */ event_ptr,
    };
#endif
}

static void ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
    GGML_UNUSED(dev);
    if (event == nullptr) {
        return;
    }

    if (event->context != nullptr) {
        sycl::event * sycl_event = static_cast<sycl::event *>(event->context);
        delete sycl_event;
        event->context = nullptr;
    }

    delete event;
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
    GGML_UNUSED(dev);
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

    sycl::event * sycl_event = static_cast<sycl::event *>(event->context);
    SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

struct ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYCL_NAME;
}

static size_t ggml_backend_sycl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *) ggml_backend_sycl_split_buffer_type;
    }

    if (strcmp(name, "ggml_backend_tp_buffer_type") == 0) {
        return (void *) ggml_backend_sycl_tp_buffer_type;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_count,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};

// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    static ggml_backend_reg reg;
    static bool             initialized = false;

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();

                ggml_backend_dev_t dev =
                    new ggml_backend_device{ /* .iface       = */ ggml_backend_sycl_device_interface,
                                             /* .reg         = */ &reg,
                                             /* .context     = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_sycl_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_sycl_init(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    };

    ggml_backend_t sycl_backend =
        new ggml_backend{ /* .guid    = */ ggml_backend_sycl_guid(),
                          /* .iface   = */ ggml_backend_sycl_interface,
                          /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
                          /* .context = */ ctx };

    return sycl_backend;
}

// ============================================================================
// Flash Attention seq_ids host pointer support
// ============================================================================
// Thread-local storage for seq_ids host pointers
// These are set by llama layer before graph execution and used by fattn kernel
// The pointers must be valid SYCL_Host buffer memory (USM accessible from GPU)
struct ggml_sycl_seq_ids_cache {
    const int32_t * q_seq_ids  = nullptr;
    size_t          q_count    = 0;
    const int32_t * kv_seq_ids = nullptr;
    size_t          kv_count   = 0;
};

static thread_local ggml_sycl_seq_ids_cache g_sycl_seq_ids_cache;

void ggml_backend_sycl_set_seq_ids_host(const int32_t * q_seq_ids,
                                        size_t          q_count,
                                        const int32_t * kv_seq_ids,
                                        size_t          kv_count) {
    g_sycl_seq_ids_cache.q_seq_ids  = q_seq_ids;
    g_sycl_seq_ids_cache.q_count    = q_count;
    g_sycl_seq_ids_cache.kv_seq_ids = kv_seq_ids;
    g_sycl_seq_ids_cache.kv_count   = kv_count;
}

void ggml_backend_sycl_clear_seq_ids_host(void) {
    g_sycl_seq_ids_cache.q_seq_ids  = nullptr;
    g_sycl_seq_ids_cache.q_count    = 0;
    g_sycl_seq_ids_cache.kv_seq_ids = nullptr;
    g_sycl_seq_ids_cache.kv_count   = 0;
}

// Internal getter for fattn.cpp to access the cached host pointers
const int32_t * ggml_sycl_get_seq_ids_host_q(size_t * count) {
    if (count) {
        *count = g_sycl_seq_ids_cache.q_count;
    }
    return g_sycl_seq_ids_cache.q_seq_ids;
}

const int32_t * ggml_sycl_get_seq_ids_host_kv(size_t * count) {
    if (count) {
        *count = g_sycl_seq_ids_cache.kv_count;
    }
    return g_sycl_seq_ids_cache.kv_seq_ids;
}

// ==============================================================================
// Thread-local cache for pending device tokens (multi-step GPU decode)
// ==============================================================================

struct ggml_sycl_device_token_cache {
    void * token_ptr = nullptr;  // Device pointer to token(s)
    size_t n_tokens  = 0;        // Number of tokens
};

static thread_local ggml_sycl_device_token_cache g_sycl_device_token_cache;

void ggml_backend_sycl_set_pending_device_token(void * token_ptr, size_t n_tokens) {
    g_sycl_device_token_cache.token_ptr = token_ptr;
    g_sycl_device_token_cache.n_tokens  = n_tokens;
}

void ggml_backend_sycl_clear_pending_device_token(void) {
    g_sycl_device_token_cache.token_ptr = nullptr;
    g_sycl_device_token_cache.n_tokens  = 0;
}

// Internal getter for llama-graph.cpp to access the pending device token
void * ggml_sycl_get_pending_device_token(size_t * n_tokens) {
    if (n_tokens) {
        *n_tokens = g_sycl_device_token_cache.n_tokens;
    }
    return g_sycl_device_token_cache.token_ptr;
}

// Copy from device memory to host memory (synchronous)
// Used when tokens tensor is on CPU backend but we have device tokens
void ggml_sycl_copy_device_to_host(void * src_device, void * dst_host, size_t bytes) {
    // Use device 0's queue for the copy (the token buffer is on the sampler's device)
    // In multi-step decode, the sampler device is the same as the model device
    auto & q = dpct::dev_mgr::instance().get_device(0).default_queue();
    q.memcpy(dst_host, src_device, bytes).wait();
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
