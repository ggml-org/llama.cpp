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

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <unordered_set>
#include <unordered_map>
#include <mutex>

#include <sycl/sycl.hpp>
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#include <sycl/half_type.hpp>

#include "ggml-sycl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set_rows_paged.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/fattn.hpp"
#include "ggml-sycl/mmq.hpp"
#include "ggml-sycl/mmq_xmx.hpp"
#include "ggml-sycl/mmvq.hpp"
#include "ggml-sycl/fused-norm-gemm.hpp"
#include "ggml-sycl/fused-moe-esimd.hpp"
#include "ggml-sycl/gpu-sampler.hpp"
#include "ggml-sycl/cont-batching.hpp"
#include "ggml-sycl/quantized-comm.hpp"
#include "ggml.h"

static bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_tp_debug = 0;  // Tensor Parallelism debug output (GGML_SYCL_TP_DEBUG env var)
int g_ggml_sycl_tp_async_ffn = 0;  // Async FFN pipelining (DISABLED - causes hangs)
int g_ggml_sycl_disable_optimize = 0;
int g_ggml_sycl_disable_graph = 0;
int g_ggml_sycl_disable_dnn = 0;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_use_async_mem_op = 0;
int g_ggml_sycl_use_xmx_gemm = 0;  // Enable XMX-accelerated GEMM (experimental)
int g_ggml_sycl_xmx_threshold = 1024; // Max batch size for XMX (XMX faster for N < threshold)
thread_local bool g_ggml_sycl_graph_recording = false;  // True when SYCL graph is recording

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
        sycl::device device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units();
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
        info.devices[i].smpbo = prop.get_local_mem_size();

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
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

static void print_device_detail(int id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

static void print_device_opt_feature(int device_count) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    for (int id = 0; id < device_count; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
void ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    int device_count = dpct::dev_mgr::instance().device_count();
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
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      int type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_count);
}

static inline int get_sycl_env(const char *env_name, int default_val) {
    char *user_device_string = getenv(env_name);
    int user_number = default_val;

    unsigned n;
    if (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (int)n;
    } else {
        user_number = default_val;
    }
    return user_number;
}

static void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        g_ggml_sycl_debug = get_sycl_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_tp_debug = get_sycl_env("GGML_SYCL_TP_DEBUG", 0);
        g_ggml_sycl_disable_optimize = get_sycl_env("GGML_SYCL_DISABLE_OPT", 0);
        g_ggml_sycl_disable_graph = get_sycl_env("GGML_SYCL_DISABLE_GRAPH", 0);
        g_ggml_sycl_disable_dnn = get_sycl_env("GGML_SYCL_DISABLE_DNN", 0);
        g_ggml_sycl_prioritize_dmmv = get_sycl_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
        g_ggml_sycl_use_xmx_gemm = get_sycl_env("GGML_SYCL_USE_XMX_GEMM", 0);
        g_ggml_sycl_xmx_threshold = get_sycl_env("GGML_SYCL_XMX_THRESHOLD", 64);
        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_TP_DEBUG: %d\n", g_ggml_sycl_tp_debug);
        GGML_LOG_INFO("  GGML_SYCL_DISABLE_OPT: %d\n", g_ggml_sycl_disable_optimize);
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
        GGML_LOG_INFO("  GGML_SYCL_USE_XMX_GEMM: %d (experimental)\n", g_ggml_sycl_use_xmx_gemm);
        if (g_ggml_sycl_use_xmx_gemm) {
            GGML_LOG_INFO("  GGML_SYCL_XMX_THRESHOLD: %d (XMX for batch < %d)\n", g_ggml_sycl_xmx_threshold, g_ggml_sycl_xmx_threshold);
        }
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
        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                            dpct::dev_mgr::instance().device_count()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
inline void check_allow_gpu_index(const int device_index) {
  if (device_index >= ggml_sycl_info().device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_count - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    assert(false);
  }
}

GGML_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    for(int i=0;i<max_len;i++) id_list[i] = -1;

    for (int i=0;i< ggml_sycl_info().device_count;i++){
        if (i>=max_len) break;
        id_list[i] = i;
    }
    return;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;

    // TP compute buffer support: per-device pointers
    // For TP compute buffers, we allocate on ALL TP devices and track base pointers here
    bool is_tp_compute_buffer = false;
    void * tp_dev_ptrs[GGML_SYCL_MAX_DEVICES] = {nullptr};
    queue_ptr tp_streams[GGML_SYCL_MAX_DEVICES] = {nullptr};

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream) :
        device(device), dev_ptr(dev_ptr), stream(stream) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
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

        // Release extra used by tensors
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }
    }
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

static void
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        // For TP compute buffers, view tensors also need extra->data_device[] set up
        if (ctx->is_tp_compute_buffer && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
            ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;
            if (extra == nullptr) {
                extra = new ggml_tensor_extra_gpu{};
                tensor->extra = extra;
                ctx->tensor_extras.push_back(extra);
            }

            // Calculate offset of this VIEW tensor within the buffer
            ptrdiff_t offset = (char *)tensor->data - (char *)ctx->dev_ptr;

            // Set up data_device[] for each local TP device
            // In multi-process mode: only 1 device is locally visible
            int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
            for (int i = 0; i < num_local_devices; i++) {
                int dev_id = g_sycl_tp_config.devices[i];
                if (ctx->tp_dev_ptrs[dev_id] != nullptr) {
                    extra->data_device[dev_id] = (char *)ctx->tp_dev_ptrs[dev_id] + offset;
                    GGML_SYCL_DEBUG("SYCL TP: init_tensor (view) %s device %d: offset=%td, ptr=%p\n",
                                    tensor->name, dev_id, offset, extra->data_device[dev_id]);
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    // For TP compute buffers, set up extra->data_device[] for each TP device
    // This allows ggml_sycl_get_data_ptr() to resolve the correct per-device pointer
    if (ctx->is_tp_compute_buffer && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;
        if (extra == nullptr) {
            extra = new ggml_tensor_extra_gpu{};
            tensor->extra = extra;
            ctx->tensor_extras.push_back(extra);
        }

        // Calculate offset of this tensor within the buffer
        ptrdiff_t offset = (char *)tensor->data - (char *)ctx->dev_ptr;

        // Set up data_device[] for each local TP device
        // In multi-process mode: only 1 device is locally visible
        int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
        for (int i = 0; i < num_local_devices; i++) {
            int dev_id = g_sycl_tp_config.devices[i];
            if (ctx->tp_dev_ptrs[dev_id] != nullptr) {
                extra->data_device[dev_id] = (char *)ctx->tp_dev_ptrs[dev_id] + offset;
                GGML_SYCL_DEBUG("SYCL TP: init_tensor %s device %d: offset=%td, ptr=%p\n",
                                tensor->name, dev_id, offset, extra->data_device[dev_id]);
            }
        }
    } else if ((tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q6_K) &&
        !g_ggml_sycl_disable_optimize) {
        ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
        tensor->extra                 = extra;
        ctx->tensor_extras.push_back(extra);  //used to release it when destroy ctx.
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (char *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    char * host_buf = (char *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, host_buf, size).wait()));
    free(host_buf);
#else
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((char *) tensor->data + offset, data, size).wait()));
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (const char *)tensor->data + offset, size)
            .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Check if double-buffering is enabled (cached)
static bool g_pp_double_buffer_enabled = false;
static bool g_pp_double_buffer_checked = false;

static bool is_pp_double_buffer_enabled() {
    if (!g_pp_double_buffer_checked) {
        const char* env = std::getenv("GGML_SYCL_PP_DOUBLE_BUFFER");
        g_pp_double_buffer_enabled = (env != nullptr && std::string(env) == "1");
        g_pp_double_buffer_checked = true;
        if (g_pp_double_buffer_enabled) {
            GGML_SYCL_DEBUG("SYCL PP: Double-buffering enabled\n");
        }
    }
    return g_pp_double_buffer_enabled;
}

static void dev2dev_memcpy(sycl::queue &q_dst, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {
    // Use persistent shared buffer to avoid per-transfer malloc/free overhead
    // This is a major optimization for pipeline parallelism (--split-mode layer)
    // where device-to-device transfers happen frequently between layers

    if (is_pp_double_buffer_enabled()) {
        // Double-buffered mode: use ping-pong buffers to allow overlapping transfers
        // When one buffer is being copied to destination, the other can receive new data
        int buf_idx = -1;
        void* shared_buf = ggml_sycl_get_dev2dev_transfer_buffer_double(size, &buf_idx);
        if (shared_buf == nullptr || buf_idx < 0) {
            // Fallback to single-buffer mode
            goto single_buffer;
        }

        // Copy src -> host buffer (wait for completion since we need data in buffer)
        q_src.memcpy(shared_buf, (const char *)ptr_src, size).wait();

        // Copy host buffer -> dst (don't wait - record event for next use of this buffer)
        sycl::event evt = q_dst.memcpy((char *)ptr_dst, shared_buf, size);
        ggml_sycl_set_dev2dev_transfer_event(buf_idx, evt);

        // Note: The next transfer will use the other buffer, and only wait if
        // that buffer's previous transfer isn't complete yet
        return;
    }

single_buffer:
    void* shared_buf = ggml_sycl_get_dev2dev_transfer_buffer(size);
    if (shared_buf == nullptr) {
        // Fallback to malloc if shared buffer allocation fails
        char *host_buf = (char *)malloc(size);
        q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
        q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
        free(host_buf);
        return;
    }

    // Use pinned host buffer: accessible from both host and all devices
    // Copy: src_device -> shared_buf -> dst_device
    q_src.memcpy(shared_buf, (const char *)ptr_src, size).wait();
    q_dst.memcpy((char *)ptr_dst, shared_buf, size).wait();
    // No free - buffer is persistent and reused
}

static bool
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst) try {
    bool is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

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

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    SYCL_CHECK(CHECK_TRY_ERROR((*stream)
                                    .memset(ctx->dev_ptr, value, buffer->size)
                                    .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
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
        for (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
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
    int device;
    std::string name;
    ggml_sycl_mem_type mem_type = GGML_SYCL_MEM_DEVICE;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0

    void * dev_ptr;

    // Allocate memory based on buffer type's memory type setting
    switch (buft_ctx->mem_type) {
        case GGML_SYCL_MEM_HOST:
            // Pinned host memory - accessible from all devices (used for TP compute buffers)
            // In TP mode, use the shared context so memory is accessible from all TP devices
            if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
                sycl::queue * tp_queue = ggml_sycl_get_tp_queue(buft_ctx->device);
                if (tp_queue != nullptr) {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_host(size, tp_queue->get_context())));
                    GGML_SYCL_DEBUG("TP: Allocated %zu bytes HOST memory in shared context\n", size);
                } else {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_host(size, *stream)));
                }
            } else {
                SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_host(size, *stream)));
            }
            break;
        case GGML_SYCL_MEM_SHARED:
            // Unified shared memory - auto-migrating between host and device
            // In TP mode, use the shared context
            if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
                sycl::queue * tp_queue = ggml_sycl_get_tp_queue(buft_ctx->device);
                if (tp_queue != nullptr) {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_shared(size, *tp_queue)));
                    GGML_SYCL_DEBUG("TP: Allocated %zu bytes SHARED memory in shared context\n", size);
                } else {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_shared(size, *stream)));
                }
            } else {
                SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_shared(size, *stream)));
            }
            break;
        case GGML_SYCL_MEM_DEVICE:
        default:
            // GPU device memory - fastest but device-local
            // In TP mode, use the shared context so operations can access memory across devices
            if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
                sycl::queue * tp_queue = ggml_sycl_get_tp_queue(buft_ctx->device);
                if (tp_queue != nullptr) {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_device(size, *tp_queue)));
                    // DEBUG: Check if allocation overlaps with L31 FFN gate weight region
                    uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
                    uintptr_t alloc_start = (uintptr_t)dev_ptr;
                    uintptr_t alloc_end = alloc_start + size;
                    if (buft_ctx->device == 0 && alloc_start <= l31_weight_addr && alloc_end > l31_weight_addr) {
                        fprintf(stderr, "TP DEBUG ALLOC OVERLAP! device=%d ptr=%p size=%zu overlaps L31 weight at 0x%llx\n",
                                buft_ctx->device, dev_ptr, size, (unsigned long long)l31_weight_addr);
                    }
                    GGML_SYCL_DEBUG("TP: Allocated %zu bytes DEVICE memory in shared context for device %d at %p\n", size, buft_ctx->device, dev_ptr);
                } else {
                    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_device(size, *stream)));
                }
            } else {
                SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_device(size, *stream)));
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
    ggml_backend_sycl_buffer_context * ctx = new ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, ctx_stream);

    // In TP mode, allocate device memory on ALL TP devices for compute buffers
    // This allows each device to have its own copy of compute buffers
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1 &&
        buft_ctx->mem_type == GGML_SYCL_MEM_DEVICE) {
        ctx->is_tp_compute_buffer = true;
        ctx->tp_dev_ptrs[buft_ctx->device] = dev_ptr;  // Already allocated for main device
        ctx->tp_streams[buft_ctx->device] = ctx_stream;

        // Allocate on other local TP devices (in multi-process mode, only 1 device is visible)
        int num_local_devices = g_sycl_tp_config.is_multiprocess ? 1 : g_sycl_tp_config.world_size;
        for (int i = 0; i < num_local_devices; i++) {
            int dev_id = g_sycl_tp_config.devices[i];
            if (dev_id == buft_ctx->device) continue;  // Skip main device (already done)

            ggml_sycl_set_device(dev_id);
            sycl::queue * tp_queue = ggml_sycl_get_tp_queue(dev_id);
            if (tp_queue != nullptr) {
                void * ptr = nullptr;
                SYCL_CHECK(CHECK_TRY_ERROR(ptr = sycl::malloc_device(size, *tp_queue)));
                if (ptr != nullptr) {
                    ctx->tp_dev_ptrs[dev_id] = ptr;
                    ctx->tp_streams[dev_id] = tp_queue;
                    // DEBUG: Check if allocation overlaps with L31 FFN gate weight region
                    uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
                    uintptr_t alloc_start = (uintptr_t)ptr;
                    uintptr_t alloc_end = alloc_start + size;
                    if (dev_id == 0 && alloc_start <= l31_weight_addr && alloc_end > l31_weight_addr) {
                        fprintf(stderr, "TP DEBUG ALLOC OVERLAP (compute)! device=%d ptr=%p size=%zu overlaps L31 weight at 0x%llx\n",
                                dev_id, ptr, size, (unsigned long long)l31_weight_addr);
                    }
                    GGML_SYCL_DEBUG("TP: Allocated compute buffer %zu bytes on device %d at %p\n",
                                    size, dev_id, ptr);
                } else {
                    GGML_LOG_ERROR("TP: Failed to allocate compute buffer on device %d\n", dev_id);
                }
            }
        }
        // Restore device context
        ggml_sycl_set_device(buft_ctx->device);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
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

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

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
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_count = ggml_backend_sycl_get_device_count();

    if (device>=dev_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_count-1);
        GGML_ASSERT(device<dev_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), GGML_SYCL_MEM_DEVICE, stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }
    return &ggml_backend_sycl_buffer_types[device];
}

static ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    int device = ctx->device;
    if (device>=ggml_sycl_info().device_count or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_count-1);
        GGML_ASSERT(device<ggml_sycl_info().device_count);
    }
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < ggml_sycl_info().device_count; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), GGML_SYCL_MEM_DEVICE, ctx->stream(i, 0)},
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

    switch(type) {
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

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;
    if (id == ggml_sycl_info().device_count - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

struct ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() try {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

static void ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

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

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if SYCL Buffer alloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char * buf;
        /*
        DPCT1009:208: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)sycl::malloc_device(
                                        size, *stream)));
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
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_size, 0, size - original_size)
                    .wait()));
        }

        extra->data_device[i] = buf;

        for (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = new sycl::event()));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, const void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          const ggml_tensor *tensor, void *data,
                                          size_t offset, size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
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

static ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
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

static size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

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
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");
    ggml_check_sycl();
    // FIXME: this is not thread safe
    static std::map<std::array<float, GGML_SYCL_MAX_DEVICES>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](float x) { return x == 0.0f; });
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

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ new ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
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
        for (auto* extra : tensor_extras) {
            release_extra_gpu(extra);
        }
    }
    std::vector<ggml_tensor_extra_gpu*> tensor_extras;
    int main_device = 0;     // Primary device for this buffer
    int world_size = 1;      // Number of TP devices
    std::vector<int> devices; // All TP device IDs
};

static const char * ggml_backend_sycl_tp_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_SYCL_NAME "_TP";
    GGML_UNUSED(buffer);
}

static void ggml_backend_sycl_tp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto* ctx = static_cast<ggml_backend_sycl_tp_buffer_context*>(buffer->context);
    delete ctx;
}

static void* ggml_backend_sycl_tp_buffer_get_base(ggml_backend_buffer_t buffer) {
    return reinterpret_cast<void*>(0x2000);  // Dummy address, actual data in tensor extras
    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_sycl_tp_buffer_init_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor) {

    auto* ctx = static_cast<ggml_backend_sycl_tp_buffer_context*>(buffer->context);
    int main_device = ctx->main_device;
    int world_size = ctx->world_size;

    // DEBUG: Track which tensors enter init_tensor
    static int init_dbg = 0;
    bool is_embd = tensor->name && strstr(tensor->name, "token_embd");
    if (g_ggml_sycl_tp_debug && (is_embd || (init_dbg++ < 5))) {
        fprintf(stderr, "TP DEBUG init_tensor ENTRY: tensor=%s, view_src=%s, buffer=%p\n",
                tensor->name ? tensor->name : "(null)",
                tensor->view_src ? (tensor->view_src->name ? tensor->view_src->name : "(view)") : "(none)",
                (void*)buffer);
    }

    // Check if this tensor is a view of another tensor in the same buffer
    // Views should share the parent's device allocations with an offset
    if (tensor->view_src != nullptr && tensor->view_src->buffer == buffer) {
        // This is a view - use parent's allocations with offset
        auto* parent_extra = static_cast<ggml_tensor_extra_gpu*>(tensor->view_src->extra);
        if (parent_extra != nullptr) {
            // Create extra for this view that points into parent's allocations
            auto* extra = new ggml_tensor_extra_gpu{};
            ctx->tensor_extras.push_back(extra);

            // Copy relevant fields from parent
            extra->tp_sharded = parent_extra->tp_sharded;
            extra->tp_world_size = parent_extra->tp_world_size;
            extra->tp_type = parent_extra->tp_type;
            extra->tp_type_cached = true;

            // Calculate offset from parent to this view
            // tensor->data and tensor->view_src->data are both host pointers during graph build
            // The view offset is computed from the difference
            size_t view_offset = reinterpret_cast<char*>(tensor->data) - reinterpret_cast<char*>(tensor->view_src->data);

            // Set device pointers as parent + offset for each device
            // Use ctx->devices.size() not world_size - in multi-process mode we only have 1 local device
            for (int rank = 0; rank < (int)ctx->devices.size(); rank++) {
                int device = ctx->devices[rank];
                if (parent_extra->data_device[device] != nullptr) {
                    extra->data_device[device] = reinterpret_cast<char*>(parent_extra->data_device[device]) + view_offset;
                }
            }

            // tensor->data now points to main device's view location
            tensor->data = extra->data_device[main_device];
            tensor->extra = extra;

            GGML_SYCL_DEBUG("SYCL TP: view tensor %s uses parent %s + offset %zu\n",
                           tensor->name ? tensor->name : "(null)",
                           tensor->view_src->name ? tensor->view_src->name : "(null)",
                           view_offset);

            return GGML_STATUS_SUCCESS;
        }
    }

    auto* extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    // Determine if this tensor should be sharded
    // Use ggml_sycl_tp_get_layer_type() which has caching - but tensor->extra isn't set yet
    // So we call it after setting extra, and it will cache the result
    extra->tp_type = tp_layer_type::TP_NONE;
    extra->tp_type_cached = false;  // Will be computed and cached on first access
    tensor->extra = extra;  // Set early so ggml_sycl_tp_get_layer_type can cache

    // Now get the TP type (this will compute and cache it)
    tp_layer_type tp_type = ggml_sycl_tp_get_layer_type(tensor);

    // DEBUG: Check why sharding might not happen
    static int shard_dbg = 0;
    if (g_ggml_sycl_tp_debug && shard_dbg++ < 20 && tensor->name && (strstr(tensor->name, "ffn_gate") || strstr(tensor->name, "attn_q"))) {
        fprintf(stderr, "TP DEBUG init_tensor %s: enabled=%d, world_size=%d, tp_type=%d (COL=%d,ROW=%d)\n",
                tensor->name, g_sycl_tp_config.enabled, world_size, (int)tp_type,
                (int)tp_layer_type::TP_COLUMN_PARALLEL, (int)tp_layer_type::TP_ROW_PARALLEL);
    }

    // Enable sharding when TP is active with multiple devices
    bool should_shard = g_sycl_tp_config.enabled && world_size > 1 &&
                        (tp_type == tp_layer_type::TP_COLUMN_PARALLEL ||
                         tp_type == tp_layer_type::TP_ROW_PARALLEL);

    // Check if we're in multi-process TP mode where tensor already has sharded dimensions
    bool is_multiprocess_tp = g_sycl_tp_config.is_multiprocess && ctx->devices.size() == 1;

    if (should_shard) {
        // TRUE DUAL-GPU TP: Allocate shards on ALL TP devices
        // Each device gets its rank's portion of the weight
        extra->tp_sharded = true;
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
        for (int rank = 0; rank < (int)ctx->devices.size(); rank++) {
            int device = ctx->devices[rank];

            // The tensor is already created with sharded dimensions by the model layer.
            // Each device gets an allocation of THIS size (the shard size).
            int64_t local_ne0 = tensor->ne[0];
            int64_t local_ne1 = tensor->ne[1];
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
            bool is_row_parallel_main = (tp_type == tp_layer_type::TP_ROW_PARALLEL && rank == 0);
            size_t alloc_size;
            if (is_row_parallel_main) {
                // Full size for main device row-parallel - use ORIGINAL (full) dimensions
                // tensor->ne is already sharded, but we need full size for MMVQ
                size_t full_row_size = ggml_row_size(tensor->type, extra->tp_original_ne[0]);
                alloc_size = full_row_size * extra->tp_original_ne[1] * extra->tp_original_ne[2] * extra->tp_original_ne[3];
            } else {
                // Shard size for other cases
                alloc_size = shard_size;
            }

            // Pad for alignment
            size_t padded_size = alloc_size;
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
            char* buf = static_cast<char*>(sycl::malloc_device(padded_size, *stream));
            if (!buf) {
                fprintf(stderr, "SYCL TP: Failed to allocate %zu bytes on device %d for tensor %s\n",
                        padded_size, device, tensor->name ? tensor->name : "(null)");
                return GGML_STATUS_ALLOC_FAILED;
            }

            // Zero-fill the entire buffer (important for row-parallel zero-padding)
            stream->memset(buf, 0, padded_size).wait();

            extra->data_device[device] = buf;

            // Store local dimensions for rank 0 (used for tensor->ne update)
            if (rank == 0) {
                extra->tp_local_ne[0] = local_ne0;
                extra->tp_local_ne[1] = local_ne1;
                extra->tp_local_ne[2] = tensor->ne[2];
                extra->tp_local_ne[3] = tensor->ne[3];
                extra->tp_offset_ne[0] = offset_ne0;
                extra->tp_offset_ne[1] = offset_ne1;
                extra->tp_rank = rank;
            }

        }

        // NOTE: tensor->ne already has sharded dimensions from model layer creation.
        // The model layer creates tensors with {n_embd, tp_n_embd_head_k_x_n_head} etc.
        // We do NOT modify tensor->ne here - it's already correct for graph building.
        GGML_UNUSED(is_multiprocess_tp);

        // DEBUG: Verify sharding happened
        static int shard_verify_dbg = 0;
        if (g_ggml_sycl_tp_debug && shard_verify_dbg++ < 10 && tensor->name) {
            fprintf(stderr, "TP DEBUG SHARD %s: orig=[%lld,%lld] -> shard=[%lld,%lld] (multiprocess=%d)\n",
                    tensor->name,
                    (long long)extra->tp_original_ne[0], (long long)extra->tp_original_ne[1],
                    (long long)tensor->ne[0], (long long)tensor->ne[1],
                    is_multiprocess_tp);
        }

        // tensor->data points to main device's shard
        tensor->data = extra->data_device[main_device];

    } else {
        // Non-sharded tensor: DUPLICATE on all TP devices for fast access
        // Intel Arc GPUs don't support P2P, and host memory is 32x slower for kernel access
        // So we duplicate non-sharded weights (layer norms, etc.) on each device
        extra->tp_sharded = false;
        extra->tp_usm_host = false;
        size_t alloc_size = ggml_nbytes(tensor);

        // Pad for alignment
        size_t padded_size = alloc_size;
        if (tensor->ne[0] % MATRIX_ROW_PADDING != 0) {
            padded_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - tensor->ne[0] % MATRIX_ROW_PADDING);
        }

        // Allocate device memory on TP devices using shared-context queue
        // In multi-process mode: ctx->devices.size() == 1, world_size is MPI world size
        bool is_tok_embd = tensor->name && strstr(tensor->name, "token_embd");
        for (int rank = 0; rank < (int)ctx->devices.size(); rank++) {
            int device = ctx->devices[rank];
            ggml_sycl_set_device(device);
            queue_ptr stream = ggml_sycl_get_tp_queue(device);
            if (stream == nullptr) {
                stream = &dpct::get_current_device().default_queue();
            }
            char* buf = static_cast<char*>(sycl::malloc_device(padded_size, *stream));
            if (!buf) {
                fprintf(stderr, "SYCL TP: Failed to allocate %zu bytes on device %d for tensor %s\n",
                        padded_size, device, tensor->name ? tensor->name : "(null)");
                return GGML_STATUS_ALLOC_FAILED;
            }

            // Zero padding
            stream->memset(buf, 0, padded_size).wait();

            // DEBUG: Track tensor allocation
            if (g_ggml_sycl_tp_debug && (is_tok_embd || (tensor->name && strstr(tensor->name, "output_norm")))) {
                sycl::device q_dev = stream->get_device();
                fprintf(stderr, "TP DEBUG ALLOC %s: rank=%d, device=%d, queue_device='%s', buf=%p, size=%zu\n",
                        tensor->name ? tensor->name : "(null)", rank, device,
                        q_dev.get_info<sycl::info::device::name>().c_str(), (void*)buf, padded_size);
            }

            extra->data_device[device] = buf;
        }

        // tensor->data points to main device's copy
        tensor->data = extra->data_device[main_device];
    }

    // tensor->extra already set early in this function for caching
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_tp_buffer_set_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor,
    const void* data, size_t offset, size_t size) {

    auto* ctx = static_cast<ggml_backend_sycl_tp_buffer_context*>(buffer->context);
    auto* extra = static_cast<ggml_tensor_extra_gpu*>(tensor->extra);

    GGML_ASSERT(offset == 0);  // TP tensors must be set in full

    if (extra->tp_sharded) {
        // SHARDED TP: Copy each rank's shard to its device
        int world_size = ctx->world_size;

        // We need original dimensions for shard extraction
        // Temporarily restore them for the copy operation
        int64_t saved_ne[4] = {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
        tensor->ne[0] = extra->tp_original_ne[0];
        tensor->ne[1] = extra->tp_original_ne[1];
        tensor->ne[2] = extra->tp_original_ne[2];
        tensor->ne[3] = extra->tp_original_ne[3];

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
            if (tensor->name && strstr(tensor->name, "blk.0.attn_q.weight")) {
                const uint8_t* src = static_cast<const uint8_t*>(data);
                fprintf(stderr, "[TP LOAD MP] tensor='%s' rank=%d size=%zu first_bytes=[%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x]\n",
                        tensor->name, g_sycl_tp_config.rank, size,
                        src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]);
            }

            GGML_SYCL_DEBUG("SYCL TP MP: Direct copy %zu bytes to device %d for tensor %s\n",
                            size, device, tensor->name ? tensor->name : "(null)");
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
                ggml_sycl_tp_copy_weight_shard(extra->data_device[device], data, tensor,
                                                rank, world_size, stream);
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
        bool is_tok_embd = tensor->name && strstr(tensor->name, "token_embd");
        for (int rank = 0; rank < (int)ctx->devices.size(); rank++) {
            int device = ctx->devices[rank];
            ggml_sycl_set_device(device);
            queue_ptr stream = ggml_sycl_get_tp_queue(device);
            if (stream == nullptr) {
                stream = &dpct::get_current_device().default_queue();
            }
            // DEBUG: Track embedding table data copy
            if (g_ggml_sycl_tp_debug && is_tok_embd && rank == 0) {
                // Check SOURCE data before copy (mmapped file)
                struct { uint16_t d_bits; uint8_t qs[16]; } src_blk1, src_blk38, src_blk100;
                size_t blk_row_size = (4096/32) * 18;
                memcpy(&src_blk1, (char*)data + 1 * blk_row_size, sizeof(src_blk1));
                memcpy(&src_blk38, (char*)data + 38 * blk_row_size, sizeof(src_blk38));
                memcpy(&src_blk100, (char*)data + 100 * blk_row_size, sizeof(src_blk100));
                sycl::half src1_d, src38_d, src100_d;
                memcpy(&src1_d, &src_blk1.d_bits, sizeof(sycl::half));
                memcpy(&src38_d, &src_blk38.d_bits, sizeof(sycl::half));
                memcpy(&src100_d, &src_blk100.d_bits, sizeof(sycl::half));
                fprintf(stderr, "TP DEBUG COPY SOURCE (host/mmap): tok1.d=%f (0x%04x), tok38.d=%f (0x%04x), tok100.d=%f (0x%04x)\n",
                        (float)src1_d, src_blk1.d_bits, (float)src38_d, src_blk38.d_bits, (float)src100_d, src_blk100.d_bits);
                fprintf(stderr, "TP DEBUG COPY SOURCE: tok38.qs=0x%02x%02x, tok100.qs=0x%02x%02x\n",
                        src_blk38.qs[0], src_blk38.qs[1], src_blk100.qs[0], src_blk100.qs[1]);
                fprintf(stderr, "TP DEBUG COPY tok_embd: src=%p, dst=%p, size=%zu\n",
                        data, (void*)extra->data_device[device], size);
            }
            stream->memcpy(extra->data_device[device], data, size).wait();
            // DEBUG: Verify copy by reading back token 0, 1, 38, 100
            if (g_ggml_sycl_tp_debug && is_tok_embd) {
                struct { sycl::half d; uint8_t qs[16]; } blk0, blk1, blk38, blk100;
                size_t blk_row_size = (4096/32) * 18;  // 128 blocks * 18 bytes = 2304 bytes/row
                stream->memcpy(&blk0, extra->data_device[device], sizeof(blk0)).wait();
                stream->memcpy(&blk1, (char*)extra->data_device[device] + 1 * blk_row_size, sizeof(blk1)).wait();
                stream->memcpy(&blk38, (char*)extra->data_device[device] + 38 * blk_row_size, sizeof(blk38)).wait();
                stream->memcpy(&blk100, (char*)extra->data_device[device] + 100 * blk_row_size, sizeof(blk100)).wait();
                fprintf(stderr, "TP DEBUG COPY VERIFY device=%d: tok0.d=%f, tok1.d=%f, tok38.d=%f, tok100.d=%f\n",
                        device, (float)blk0.d, (float)blk1.d, (float)blk38.d, (float)blk100.d);
                fprintf(stderr, "TP DEBUG COPY VERIFY device=%d: tok38.qs=0x%02x%02x, tok100.qs=0x%02x%02x\n",
                        device, blk38.qs[0], blk38.qs[1], blk100.qs[0], blk100.qs[1]);
            }
        }
    }

    // Store FFN weight reference for later computation on device 1
    store_ffn_weight_ref(tensor);

    GGML_UNUSED(buffer);
}

static void ggml_backend_sycl_tp_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* tensor,
    void* data, size_t offset, size_t size) {

    auto* ctx = static_cast<ggml_backend_sycl_tp_buffer_context*>(buffer->context);
    auto* extra = static_cast<ggml_tensor_extra_gpu*>(tensor->extra);
    int device = ctx->main_device;  // Get from main device

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
        const char* src = static_cast<const char*>(extra->data_device[device]) + offset;
        stream->memcpy(data, src, size).wait();
    }

    GGML_UNUSED(buffer);
}

static bool ggml_backend_sycl_tp_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
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
    int main_device;
    int world_size;
    std::vector<int> devices;  // All TP device IDs
};

static const char* ggml_backend_sycl_tp_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_TP";
    GGML_UNUSED(buft);
}

static bool ggml_backend_buffer_is_sycl_tp(ggml_backend_buffer_t buffer) {
    return buffer && buffer->buft && buffer->buft->iface.get_name == ggml_backend_sycl_tp_buffer_type_name;
}

static ggml_backend_buffer_t ggml_backend_sycl_tp_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {

    auto* buft_ctx = static_cast<ggml_backend_sycl_tp_buffer_type_context*>(buft->context);
    auto* ctx = new ggml_backend_sycl_tp_buffer_context();
    ctx->main_device = buft_ctx->main_device;
    ctx->world_size = buft_ctx->world_size;
    ctx->devices = buft_ctx->devices;

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_tp_buffer_interface, ctx, size);
}

static size_t ggml_backend_sycl_tp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_tp_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {

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
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<int, ggml_backend_buffer_type> buft_map;

    auto it = buft_map.find(main_device);
    if (it != buft_map.end()) {
        return &it->second;
    }

    // Get all TP devices from the global config
    int world_size = ggml_sycl_tp_world_size();
    std::vector<int> devices;

    if (g_sycl_tp_config.enabled && world_size > 1) {
        if (g_sycl_tp_config.is_multiprocess) {
            // Multi-process mode: each process has only ONE device (device 0)
            // world_size reflects MPI processes, but we only allocate on our device
            devices.push_back(0);  // Always device 0 (restricted by ONEAPI_DEVICE_SELECTOR)
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

    auto* ctx = new ggml_backend_sycl_tp_buffer_type_context();
    ctx->main_device = main_device;
    ctx->world_size = world_size;
    ctx->devices = devices;

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
ggml_backend_buffer_type_t ggml_backend_sycl_tp_buffer_type(int n_devices, const int* device_ids) {
    // Initialize TP system if not already initialized
    static bool tp_initialized = false;
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
        const char* pmi_size = std::getenv("PMI_SIZE");
        if (pmi_size) {
            int mpi_world_size = std::atoi(pmi_size);
            if (mpi_world_size > 1) {
                return mpi_world_size;
            }
        }
        // Check for Open MPI environment variables
        const char* ompi_size = std::getenv("OMPI_COMM_WORLD_SIZE");
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
    const char* pmi_rank = std::getenv("PMI_RANK");
    if (pmi_rank) {
        return std::atoi(pmi_rank);
    }
    const char* ompi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
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
    const char* pmi_size = std::getenv("PMI_SIZE");
    if (pmi_size && std::atoi(pmi_size) > 1) {
        return true;
    }
    const char* ompi_size = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (ompi_size && std::atoi(ompi_size) > 1) {
        return true;
    }

    return false;
}

// Get the byte offset for reading this rank's shard from GGUF file
// For column-parallel tensors (wq, wk, wv, ffn_gate, ffn_up): contiguous, returns offset
// For row-parallel tensors (wo, ffn_down): interleaved, returns 0 (requires special handling)
size_t ggml_backend_sycl_get_tp_data_offset(
    const char * tensor_name,
    const int64_t * tensor_ne,
    enum ggml_type tensor_type) {

    if (!ggml_backend_sycl_is_multiprocess_tp()) {
        return 0;
    }

    int world_size = ggml_backend_sycl_get_tp_world_size();
    int rank = ggml_backend_sycl_get_tp_rank();

    if (world_size <= 1 || rank == 0) {
        return 0;  // Rank 0 always reads from start
    }

    // Determine TP layer type from tensor name
    // Column-parallel: wq, wk, wv, ffn_gate, ffn_up (split output dim ne[1])
    // Row-parallel: wo/attn_output, ffn_down (split input dim ne[0])
    bool is_column_parallel = false;
    bool is_row_parallel = false;

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
        int64_t remainder = ne1 % world_size;
        int64_t offset_ne1;

        if (rank < remainder) {
            offset_ne1 = rank * (chunk_size + 1);
        } else {
            offset_ne1 = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size;
        }

        // Calculate byte offset: offset_ne1 rows * row_size
        size_t row_size = ggml_row_size(tensor_type, ne0);
        return offset_ne1 * row_size;
    }
    else if (is_row_parallel) {
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

void ggml_backend_sycl_pp_init(
    const int * device_ids, int n_devices,
    int total_layers, const int * layers_per_stage) {
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
    ggml_backend_t backend;
    ggml_backend_sycl_context* sycl_ctx;
    ggml_sycl_sampler_state state;
    int n_vocab;
};

ggml_sycl_sampler_t ggml_backend_sycl_sampler_create(
    ggml_backend_t backend, int n_vocab, uint32_t seed
) try {
    GGML_ASSERT(ggml_backend_is_sycl(backend));

    ggml_sycl_sampler_t sampler = new ggml_sycl_sampler();
    sampler->backend = backend;
    sampler->sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    sampler->n_vocab = n_vocab;

    // Initialize sampler state
    ggml_sycl_sampler_init(sampler->state, n_vocab, seed);

    return sampler;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error creating GPU sampler: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_sampler_free(ggml_sycl_sampler_t sampler) try {
    if (sampler == nullptr) return;

    if (sampler->state.initialized) {
        sycl::queue& q = *sampler->sycl_ctx->stream();
        ggml_sycl_sampler_free(sampler->state, q);
    }

    delete sampler;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error freeing GPU sampler: %s\n", exc.what());
}

int32_t ggml_backend_sycl_sample_token(
    ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor, float temp
) try {
    // Sample from index 0 (for single-token decode)
    return ggml_backend_sycl_sample_token_idx(sampler, logits_tensor, 0, temp);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in GPU sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t ggml_backend_sycl_sample_token_idx(
    ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor, int idx, float temp
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    // Get GPU pointer for logits
    float* logits_gpu = (float*)logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    // Get vocab size from tensor (ne[0] is vocab dimension)
    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    // Get number of batch entries (ne[1] is batch dimension)
    int n_batch = logits_tensor->ne[1];
    GGML_ASSERT(idx >= 0 && idx < n_batch);

    // Offset to the correct batch entry
    float* logits_at_idx = logits_gpu + (size_t)idx * n_vocab;

    // Set up config
    ggml_sycl_sampler_config config;
    config.temp = temp;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.seed = sampler->state.rng_state;  // Use current RNG state
    config.greedy = (temp == 0.0f);

    // Call GPU sampler with offset logits
    return ggml_sycl_sample_token(*sampler->sycl_ctx, logits_at_idx, config, sampler->state);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in GPU sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t ggml_backend_sycl_sample_token_full(
    ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor, int idx,
    float temp, int top_k, float top_p, float min_p
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    // Get the backend context from the tensor's buffer for proper synchronization
    ggml_backend_buffer_t buffer = logits_tensor->buffer;
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_sycl(buffer));

    // Get the context from the buffer's backend (the one that computed the logits)
    ggml_backend_sycl_buffer_context * buf_ctx =
        (ggml_backend_sycl_buffer_context *)buffer->context;
    GGML_ASSERT(buf_ctx != nullptr);

    // Sync the buffer's queue to ensure logits computation is complete
    // This is needed because the sampler might have a different queue
    sycl::queue& buf_queue = *buf_ctx->stream;
    buf_queue.wait();

    // Also sync the device's default queue (used by some copy operations)
    ggml_sycl_set_device(buf_ctx->device);
    dpct::dev_mgr::instance().get_device(buf_ctx->device).default_queue().wait();

    // Get GPU pointer for logits
    float* logits_gpu = (float*)logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    // Get vocab size from tensor (ne[0] is vocab dimension)
    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    // Get number of batch entries (ne[1] is batch dimension)
    int n_batch = logits_tensor->ne[1];
    GGML_ASSERT(idx >= 0 && idx < n_batch);

    // Offset to the correct batch entry
    float* logits_at_idx = logits_gpu + (size_t)idx * n_vocab;

    // Debug: read first few logits to verify data is valid
    sycl::queue& q = *sampler->sycl_ctx->stream();
    float debug_logits[5];
    q.memcpy(debug_logits, logits_at_idx, 5 * sizeof(float)).wait();
    GGML_LOG_INFO("[GPU SAMPLE DEBUG] idx=%d, logits_gpu=%p, logits_at_idx=%p, first 5 logits: %.2f %.2f %.2f %.2f %.2f\n",
                  idx, (void*)logits_gpu, (void*)logits_at_idx,
                  debug_logits[0], debug_logits[1], debug_logits[2], debug_logits[3], debug_logits[4]);

    // Set up config with full parameters
    ggml_sycl_sampler_config config;
    config.temp = temp;
    config.top_k = top_k;
    config.top_p = top_p;
    config.min_p = min_p;
    config.seed = sampler->state.rng_state;
    config.greedy = (temp == 0.0f);

    // Call GPU sampler with offset logits
    int32_t sampled_token = ggml_sycl_sample_token(*sampler->sycl_ctx, logits_at_idx, config, sampler->state);
    GGML_LOG_INFO("[GPU SAMPLE DEBUG] sampled_token=%d (temp=%.2f, greedy=%d)\n", sampled_token, temp, config.greedy);
    return sampled_token;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in GPU sampling (full): %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_sample_token_async(
    ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor, float temp
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    float* logits_gpu = (float*)logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    ggml_sycl_sampler_config config;
    config.temp = temp;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.seed = sampler->state.rng_state;
    config.greedy = (temp == 0.0f);

    // Call async GPU sampler (doesn't wait for result)
    ggml_sycl_sample_token_async(*sampler->sycl_ctx, logits_gpu, config, sampler->state);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in async GPU sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t ggml_backend_sycl_sample_token_get(ggml_sycl_sampler_t sampler) try {
    GGML_ASSERT(sampler != nullptr);
    return ggml_sycl_sample_token_wait(*sampler->sycl_ctx, sampler->state);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error getting GPU sampling result: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

// ===========================================================================
// Multi-step GPU Sampling API
// ===========================================================================

void ggml_backend_sycl_sampler_reset_buffer(ggml_sycl_sampler_t sampler) try {
    GGML_ASSERT(sampler != nullptr);
    ggml_sycl_sampler_reset_buffer(sampler->state);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error resetting sampler buffer: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_sample_token_to_device(
    ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor, float temp
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    float* logits_gpu = (float*)logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    ggml_sycl_sampler_config config;
    config.temp = temp;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.seed = sampler->state.rng_state;
    config.greedy = (temp == 0.0f);

    return ggml_sycl_sample_token_to_buffer(*sampler->sycl_ctx, logits_gpu, config, sampler->state);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in sample_token_to_device: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_sample_token_to_device_full(
    ggml_sycl_sampler_t sampler, ggml_tensor * logits_tensor,
    float temp, int top_k, float top_p, float min_p
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_tensor != nullptr);
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);

    float* logits_gpu = (float*)logits_tensor->data;
    GGML_ASSERT(logits_gpu != nullptr);

    int n_vocab = logits_tensor->ne[0];
    GGML_ASSERT(n_vocab == sampler->n_vocab);

    ggml_sycl_sampler_config config;
    config.temp = temp;
    config.top_k = top_k;
    config.top_p = top_p;
    config.min_p = min_p;
    config.seed = sampler->state.rng_state;
    config.greedy = (temp == 0.0f);

    return ggml_sycl_sample_token_to_buffer(*sampler->sycl_ctx, logits_gpu, config, sampler->state);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in sample_token_to_device_full: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t * ggml_backend_sycl_get_sampled_token_ptr(
    ggml_sycl_sampler_t sampler, int index
) try {
    GGML_ASSERT(sampler != nullptr);
    return ggml_sycl_sampler_get_token_ptr(sampler->state, index);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error getting sampled token ptr: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t * ggml_backend_sycl_get_current_token_ptr(ggml_sycl_sampler_t sampler) try {
    GGML_ASSERT(sampler != nullptr);
    return ggml_sycl_sampler_get_current_token_ptr(sampler->state);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error getting current token ptr: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_get_sampled_tokens(
    ggml_sycl_sampler_t sampler, int32_t * tokens, int max_tokens
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(tokens != nullptr);
    return ggml_sycl_sampler_get_tokens(*sampler->sycl_ctx, sampler->state, tokens, max_tokens);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error getting sampled tokens: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_get_token_count(ggml_sycl_sampler_t sampler) try {
    GGML_ASSERT(sampler != nullptr);
    return sampler->state.token_count;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error getting token count: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_get_token_buffer_size(void) {
    return GPU_SAMPLER_TOKEN_BUFFER_SIZE;
}

// ===========================================================================
// Speculative Decoding Verification API Implementation
// ===========================================================================

int ggml_backend_sycl_verify_speculative(
    ggml_sycl_sampler_t sampler,
    ggml_tensor * all_logits,
    const int32_t * draft_tokens,
    int n_draft,
    int logits_offset
) try {
    if (!sampler || !all_logits || !draft_tokens || n_draft <= 0) {
        return 0;
    }

    if (!sampler->sycl_ctx || !sampler->sycl_ctx->stream()) {
        return 0;
    }

    // Get logits data pointer from tensor - use tensor->data directly like sample_token_full
    // This ensures we use the same pointer as sampling functions
    const float * logits_data = (const float *)all_logits->data;
    if (!logits_data) {
        return 0;
    }

    const int n_vocab = sampler->state.n_vocab;
    const int n_outputs = all_logits->ne[1];  // Number of output positions in batch

    // Validate that we have enough logits for the verification
    if (logits_offset + n_draft > n_outputs) {
        GGML_LOG_ERROR("SYCL verify_speculative: logits_offset (%d) + n_draft (%d) > n_outputs (%d)\n",
                       logits_offset, n_draft, n_outputs);
        return 0;
    }

    // Offset the logits pointer to start from the correct position
    const float * logits_at_offset = logits_data + (size_t)logits_offset * n_vocab;

    // Use the host wrapper which copies draft tokens to device
    return ggml_sycl_verify_speculative_host(
        *sampler->sycl_ctx, sampler->state, logits_at_offset, draft_tokens, n_draft, n_vocab);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error verifying speculative: %s\n", exc.what());
    return 0;
}

int ggml_backend_sycl_verify_speculative_with_tokens(
    ggml_sycl_sampler_t sampler,
    ggml_tensor * all_logits,
    const int32_t * draft_tokens,
    int32_t * sampled_tokens_out,
    int n_draft,
    int logits_offset
) try {
    if (!sampler || !all_logits || !draft_tokens || !sampled_tokens_out || n_draft <= 0) {
        return 0;
    }

    if (!sampler->sycl_ctx || !sampler->sycl_ctx->stream()) {
        return 0;
    }

    // Get the backend context from the tensor's buffer for proper synchronization
    ggml_backend_buffer_t buffer = all_logits->buffer;
    if (!buffer) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: tensor has no buffer\n");
        return 0;
    }

    if (!ggml_backend_buffer_is_sycl(buffer)) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: tensor buffer is not SYCL\n");
        return 0;
    }

    // Get the context from the buffer's backend (the one that computed the logits)
    ggml_backend_sycl_buffer_context * buf_ctx =
        (ggml_backend_sycl_buffer_context *)buffer->context;
    if (!buf_ctx) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: buffer has no context\n");
        return 0;
    }

    // Sync the buffer's queue to ensure logits computation is complete
    // This is needed because the sampler might have a different queue
    sycl::queue& buf_queue = *buf_ctx->stream;
    buf_queue.wait();

    // Also sync the device's default queue (used by some copy operations)
    ggml_sycl_set_device(buf_ctx->device);
    dpct::dev_mgr::instance().get_device(buf_ctx->device).default_queue().wait();

    // Get logits data pointer from tensor
    const float * logits_data = (const float *)all_logits->data;
    if (!logits_data) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: tensor has no data\n");
        return 0;
    }

    const int n_vocab = sampler->state.n_vocab;
    const int n_outputs = all_logits->ne[1];  // Number of output positions in batch
    const int tensor_vocab = all_logits->ne[0];  // Vocabulary size from tensor

    // Debug: show actual values
    GGML_LOG_INFO("SYCL verify: n_vocab=%d, tensor_ne0=%d, n_outputs=%d, n_draft=%d\n",
                   n_vocab, tensor_vocab, n_outputs, n_draft);
    GGML_LOG_INFO("SYCL verify: sampler device=%d, buffer device=%d, logits_data=%p, nbytes=%zu\n",
                   sampler->sycl_ctx->device, buf_ctx->device, (void*)logits_data, all_logits->nb[1] * n_outputs);

    // Debug: Check buffer base vs tensor data - they might be different!
    void * buffer_base = ggml_backend_buffer_get_base(buffer);
    size_t buffer_size = ggml_backend_buffer_get_size(buffer);
    GGML_LOG_INFO("SYCL verify: buffer_base=%p, buffer_size=%zu, tensor_data=%p\n",
                   buffer_base, buffer_size, (void*)logits_data);

    // Check if data pointer is within buffer range
    ptrdiff_t offset = (const char*)logits_data - (const char*)buffer_base;
    GGML_LOG_INFO("SYCL verify: data offset from buffer_base=%td (should be positive and < %zu)\n",
                   offset, buffer_size);

    // Validate that we have enough logits for the verification
    if (logits_offset + n_draft > n_outputs) {
        GGML_LOG_ERROR("SYCL verify_speculative_with_tokens: logits_offset (%d) + n_draft (%d) > n_outputs (%d)\n",
                       logits_offset, n_draft, n_outputs);
        return 0;
    }

    // Offset the logits pointer to start from the correct position
    const float * logits_at_offset = logits_data + (size_t)logits_offset * n_vocab;

    // Debug: verify data is accessible via buffer's queue
    float debug_logits[5];
    buf_queue.memcpy(debug_logits, logits_at_offset, 5 * sizeof(float)).wait();
    GGML_LOG_INFO("SYCL verify via buf_queue: first 5 logits = [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
                   debug_logits[0], debug_logits[1], debug_logits[2], debug_logits[3], debug_logits[4]);

    // Use the extended function that also returns sampled tokens
    return ggml_sycl_verify_speculative_with_tokens(
        *sampler->sycl_ctx, sampler->state, logits_at_offset, draft_tokens,
        sampled_tokens_out, n_draft, n_vocab);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error verifying speculative with tokens: %s\n", exc.what());
    return 0;
}

// ===========================================================================
// Continuous Batching API Implementation (Multi-sequence GPU Sampling)
// ===========================================================================

// Multi-sequence sampler struct - holds backend context and multi-seq state
struct ggml_sycl_multi_seq_sampler_wrapper {
    ggml_backend_t backend;
    ggml_backend_sycl_context* sycl_ctx;
    ggml_sycl_multi_seq_sampler state;
    uint32_t base_seed;
};

ggml_sycl_multi_seq_sampler_t ggml_backend_sycl_multi_seq_sampler_create(
    ggml_backend_t backend, int max_seqs, int n_vocab, uint32_t seed
) try {
    GGML_ASSERT(ggml_backend_is_sycl(backend));
    GGML_ASSERT(max_seqs > 0 && max_seqs <= CONT_BATCH_MAX_SEQS);
    GGML_ASSERT(n_vocab > 0);

    auto wrapper = new ggml_sycl_multi_seq_sampler_wrapper();
    wrapper->backend = backend;
    wrapper->sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    wrapper->base_seed = seed;

    // Initialize multi-sequence sampler state
    sycl::queue& q = *wrapper->sycl_ctx->stream();
    ggml_sycl_multi_seq_sampler_init(wrapper->state, q, max_seqs, n_vocab);

    return (ggml_sycl_multi_seq_sampler_t)wrapper;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error creating multi-seq sampler: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_multi_seq_sampler_free(
    ggml_sycl_multi_seq_sampler_t sampler
) try {
    if (sampler == nullptr) return;

    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;
    if (wrapper->state.initialized) {
        sycl::queue& q = *wrapper->sycl_ctx->stream();
        ggml_sycl_multi_seq_sampler_free(wrapper->state, q);
    }

    delete wrapper;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error freeing multi-seq sampler: %s\n", exc.what());
}

bool ggml_backend_sycl_multi_seq_add(
    ggml_sycl_multi_seq_sampler_t sampler, int seq_id, float temp
) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    if (seq_id < 0 || seq_id >= wrapper->state.max_seqs) return false;

    sycl::queue& q = *wrapper->sycl_ctx->stream();
    // Generate unique seed for this sequence
    uint32_t seq_seed = wrapper->base_seed + seq_id;
    int slot = ggml_sycl_multi_seq_add(wrapper->state, q, seq_id, temp, seq_seed);

    return slot >= 0;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error adding sequence: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

bool ggml_backend_sycl_multi_seq_remove(
    ggml_sycl_multi_seq_sampler_t sampler, int seq_id
) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    // Check if sequence is active
    bool was_active = false;
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        if (wrapper->state.h_seq_active[i] && wrapper->state.h_seq_ids[i] == seq_id) {
            was_active = true;
            break;
        }
    }

    if (!was_active) return false;

    sycl::queue& q = *wrapper->sycl_ctx->stream();
    ggml_sycl_multi_seq_remove(wrapper->state, q, seq_id);
    return true;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error removing sequence: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_multi_seq_set_temp(
    ggml_sycl_multi_seq_sampler_t sampler, int seq_id, float temp
) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    // Find slot for this sequence
    int slot = -1;
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        if (wrapper->state.h_seq_active[i] && wrapper->state.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) return;

    sycl::queue& q = *wrapper->sycl_ctx->stream();
    q.memcpy(wrapper->state.temperatures + slot, &temp, sizeof(float)).wait();
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error setting temperature: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_multi_seq_set_params(
    ggml_sycl_multi_seq_sampler_t sampler, int seq_id,
    float temp, int top_k, float top_p, float min_p
) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    // Find slot for this sequence
    int slot = -1;
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        if (wrapper->state.h_seq_active[i] && wrapper->state.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) return;

    sycl::queue& q = *wrapper->sycl_ctx->stream();

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
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error setting params: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_get_active_count(
    ggml_sycl_multi_seq_sampler_t sampler
) {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;
    return wrapper->state.n_active;
}

int ggml_backend_sycl_multi_seq_sample(
    ggml_sycl_multi_seq_sampler_t sampler,
    float * batched_logits,
    bool greedy
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(batched_logits != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    if (wrapper->state.n_active == 0) return 0;

    sycl::queue& q = *wrapper->sycl_ctx->stream();
    ggml_sycl_multi_seq_sample(wrapper->state, q, batched_logits, greedy);

    return wrapper->state.n_active;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in multi-seq sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_sample_indexed(
    ggml_sycl_multi_seq_sampler_t sampler,
    float * logits_base,
    const int * seq_ids,
    const int * batch_indices,
    int n_seqs
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(logits_base != nullptr);
    GGML_ASSERT(batch_indices != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    if (n_seqs == 0) return 0;

    sycl::queue& q = *wrapper->sycl_ctx->stream();

    // Copy batch indices to device
    int* d_batch_indices = sycl::malloc_device<int>(n_seqs, q);
    q.memcpy(d_batch_indices, batch_indices, n_seqs * sizeof(int)).wait();

    // Copy seq_ids to device for output indexing
    int* d_seq_ids = sycl::malloc_device<int>(n_seqs, q);
    if (seq_ids != nullptr) {
        q.memcpy(d_seq_ids, seq_ids, n_seqs * sizeof(int)).wait();
    } else {
        // Default: seq_id = input index
        std::vector<int> default_ids(n_seqs);
        for (int i = 0; i < n_seqs; i++) default_ids[i] = i;
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
        q.memcpy(wrapper->state.seq_active, wrapper->state.h_seq_active.data(),
                 wrapper->state.max_seqs * sizeof(int)).wait();
    }

    // Call internal indexed sampling function
    ggml_sycl_multi_seq_sample_indexed(wrapper->state, q, logits_base, d_batch_indices, d_seq_ids, n_seqs);

    // Free temporary device memory
    sycl::free(d_batch_indices, q);
    sycl::free(d_seq_ids, q);

    return n_seqs;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in indexed multi-seq sampling: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_get_tokens(
    ggml_sycl_multi_seq_sampler_t sampler,
    int32_t * tokens_out,
    int * seq_ids_out,
    int max_tokens
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(tokens_out != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    if (wrapper->state.n_active == 0) return 0;

    sycl::queue& q = *wrapper->sycl_ctx->stream();
    int n_copy = std::min(wrapper->state.n_active, max_tokens);

    // Copy sampled tokens from device
    std::vector<int32_t> all_tokens(wrapper->state.max_seqs);
    q.memcpy(all_tokens.data(), wrapper->state.sampled_tokens,
             wrapper->state.max_seqs * sizeof(int32_t)).wait();

    // Debug: print raw device array
    GGML_LOG_DEBUG("[get_tokens] raw sampled_tokens array (max_seqs=%d):\n", wrapper->state.max_seqs);
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        GGML_LOG_DEBUG("  sampled_tokens[%d] = %d, active=%d\n",
                       i, all_tokens[i], wrapper->state.h_seq_active[i]);
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
            GGML_LOG_DEBUG("  [get_tokens] collecting: out_idx=%d <- slot=%d, token=%d\n",
                           idx, i, all_tokens[i]);
            idx++;
        }
    }

    return idx;
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error getting multi-seq tokens: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int32_t * ggml_backend_sycl_multi_seq_get_token_ptr(
    ggml_sycl_multi_seq_sampler_t sampler, int seq_id
) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    return ggml_sycl_multi_seq_get_current_token_ptr(wrapper->state, seq_id);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error getting multi-seq token ptr: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_get_active_seq_ids(
    ggml_sycl_multi_seq_sampler_t sampler,
    int * seq_ids_out,
    int max_seqs
) {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(seq_ids_out != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    int idx = 0;
    for (int i = 0; i < wrapper->state.max_seqs && idx < max_seqs; i++) {
        if (wrapper->state.h_seq_active[i]) {
            seq_ids_out[idx++] = wrapper->state.h_seq_ids[i];
        }
    }

    return idx;
}

void ggml_backend_sycl_multi_seq_reset_buffer(
    ggml_sycl_multi_seq_sampler_t sampler, int seq_id
) try {
    GGML_ASSERT(sampler != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    // Find slot for this sequence
    int slot = -1;
    for (int i = 0; i < wrapper->state.max_seqs; i++) {
        if (wrapper->state.h_seq_ids[i] == seq_id) {
            slot = i;
            break;
        }
    }

    if (slot < 0) return;

    sycl::queue& q = *wrapper->sycl_ctx->stream();
    int zero = 0;
    q.memcpy(wrapper->state.write_indices + slot, &zero, sizeof(int));
    q.memcpy(wrapper->state.token_counts + slot, &zero, sizeof(int)).wait();
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error resetting multi-seq buffer: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

int ggml_backend_sycl_multi_seq_get_ring_tokens(
    ggml_sycl_multi_seq_sampler_t sampler,
    int seq_id,
    int32_t * tokens_out,
    int max_tokens
) try {
    GGML_ASSERT(sampler != nullptr);
    GGML_ASSERT(tokens_out != nullptr);
    auto wrapper = (ggml_sycl_multi_seq_sampler_wrapper*)sampler;

    sycl::queue& q = *wrapper->sycl_ctx->stream();
    return ggml_sycl_multi_seq_get_tokens(wrapper->state, q, seq_id, tokens_out, max_tokens);
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error getting ring tokens: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

// ===========================================================================
// Batched Logits Management
// ===========================================================================

// Thread-local storage for batch logits info (set by llama layer)
static thread_local struct {
    float* logits_ptr;      // Device pointer to [n_tokens, n_vocab] logits
    int n_tokens;           // Number of tokens with logits
    int n_vocab;            // Vocabulary size
    bool valid;
} g_batch_logits_info = { nullptr, 0, 0, false };

// Called by llama layer after decode to set logits info
void ggml_backend_sycl_set_batch_logits_info(
    float* logits_device_ptr, int n_tokens, int n_vocab
) {
    g_batch_logits_info.logits_ptr = logits_device_ptr;
    g_batch_logits_info.n_tokens = n_tokens;
    g_batch_logits_info.n_vocab = n_vocab;
    g_batch_logits_info.valid = true;
}

void ggml_backend_sycl_clear_batch_logits_info(void) {
    g_batch_logits_info.valid = false;
}

float * ggml_backend_sycl_get_batch_logits_ptr(void * ctx, int batch_idx) {
    GGML_UNUSED(ctx);

    if (!g_batch_logits_info.valid) return nullptr;
    if (batch_idx < 0 || batch_idx >= g_batch_logits_info.n_tokens) return nullptr;

    return g_batch_logits_info.logits_ptr + batch_idx * g_batch_logits_info.n_vocab;
}

int ggml_backend_sycl_get_batch_logits_count(void * ctx) {
    GGML_UNUSED(ctx);

    if (!g_batch_logits_info.valid) return 0;
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

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    if (!sycl_ctx) return;

    try {
        // Submit barrier and store the event
        sycl_ctx->barrier_event = sycl_ctx->stream()->ext_oneapi_submit_barrier();
        sycl_ctx->has_pending_barrier = true;
        GGML_SYCL_DEBUG("[SYCL-BARRIER] Submitted barrier after graph execution\n");
    } catch (sycl::exception const& exc) {
        GGML_LOG_ERROR("SYCL barrier submit failed: %s\n", exc.what());
    }
}

void ggml_backend_sycl_wait_barrier(ggml_backend_t backend) {
    // Wait for the previously submitted barrier to complete
    // Call this before the next ubatch's graph_compute
    if (!ggml_backend_is_sycl(backend)) {
        return;
    }

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    if (!sycl_ctx || !sycl_ctx->has_pending_barrier) return;

    try {
        if (sycl_ctx->barrier_event.has_value()) {
            GGML_SYCL_DEBUG("[SYCL-BARRIER] Waiting on barrier event...\n");
            sycl_ctx->barrier_event->wait();
            GGML_SYCL_DEBUG("[SYCL-BARRIER] Barrier wait complete\n");
        }
        sycl_ctx->has_pending_barrier = false;
        sycl_ctx->barrier_event.reset();
    } catch (sycl::exception const& exc) {
        GGML_LOG_ERROR("SYCL barrier wait failed: %s\n", exc.what());
        sycl_ctx->has_pending_barrier = false;
        sycl_ctx->barrier_event.reset();
    }
}

// ===========================================================================
// Device Memory Utilities
// ===========================================================================

void ggml_backend_sycl_copy_device_to_tensor(
    void * src_device_ptr,
    ggml_tensor * tensor,
    size_t size
) try {
    GGML_ASSERT(src_device_ptr != nullptr);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);

    // Get the SYCL context from the tensor's buffer
    ggml_backend_sycl_buffer_context * ctx =
        (ggml_backend_sycl_buffer_context *)tensor->buffer->context;
    GGML_ASSERT(ctx != nullptr);

    // Set device and get queue
    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    // Device-to-device copy
    void* dst = tensor->data;
    stream.memcpy(dst, src_device_ptr, size).wait();
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in device-to-tensor copy: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void ggml_backend_sycl_copy_tensor_to_buffer(
    ggml_backend_t backend,
    ggml_tensor * src_tensor,
    ggml_backend_buffer_t dst_buffer,
    size_t dst_offset,
    size_t size
) try {
    GGML_ASSERT(backend != nullptr);
    GGML_ASSERT(src_tensor != nullptr);
    GGML_ASSERT(src_tensor->buffer != nullptr);
    GGML_ASSERT(dst_buffer != nullptr);

    // Get device from backend
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    int device = sycl_ctx->device;

    // Set device and get queue
    ggml_sycl_set_device(device);
    auto stream = dpct::dev_mgr::instance().get_device(device).default_queue();

    // Get source pointer from tensor
    void * src = src_tensor->data;

    // Get destination pointer from buffer with offset
    void * dst_base = ggml_backend_buffer_get_base(dst_buffer);
    char * dst = (char *)dst_base + dst_offset;

    // Device-to-device copy
    stream.memcpy(dst, src, size).wait();
}
catch (sycl::exception const &exc) {
    GGML_LOG_ERROR("SYCL error in tensor-to-buffer copy: %s\n", exc.what());
    GGML_ABORT("SYCL exception");
}

void * ggml_backend_sycl_buffer_get_ptr(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return nullptr;
    }
    return ggml_backend_buffer_get_base(buffer);
}

void ggml_backend_sycl_buffer_get_async(
    ggml_backend_buffer_t buffer,
    const void * src_ptr,
    void * dst,
    size_t offset,
    size_t size
) try {
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(src_ptr != nullptr);
    GGML_ASSERT(dst != nullptr);

    // Get the SYCL context from the buffer
    ggml_backend_sycl_buffer_context * ctx =
        (ggml_backend_sycl_buffer_context *)buffer->context;
    GGML_ASSERT(ctx != nullptr);

    // Set device and get queue
    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    // Async device-to-host copy (NO wait - will complete on next synchronize)
    const char * src = (const char *)src_ptr + offset;
    stream.memcpy(dst, src, size);
    // Note: Do NOT call .wait() here - this is the async optimization!
    // The copy will complete when ggml_backend_sycl_synchronize is called
    GGML_SYCL_DEBUG("[SYCL] Async buffer get: %zu bytes from GPU to host (deferred)\n", size);
}
catch (sycl::exception const &exc) {
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

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_sycl_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type\n");
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_sycl_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // TODO: return device.maxBufferLength
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
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
static ggml_backend_buffer_t
ggml_backend_sycl_host_compute_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) try {
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    // For TP mode, allocate SINGLE host buffer accessible by ALL devices
    // All devices read/write to the same memory - no need for data copies
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        int primary_device = buft_ctx->device;
        size = std::max(size, (size_t)1);

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
        SYCL_CHECK(CHECK_TRY_ERROR(shared_ptr = (void *)sycl::malloc_host(size, *tp_context)));

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
            primary_stream = &(primary_dpct_dev.default_queue());
        }

        ggml_backend_sycl_buffer_context * ctx = new ggml_backend_sycl_buffer_context(primary_device, shared_ptr, primary_stream);
        ctx->is_tp_compute_buffer = true;

        // In multi-process mode, we only have ONE device visible per process
        // world_size is the MPI world size, not the number of local devices
        if (g_sycl_tp_config.is_multiprocess) {
            // Multi-process: each process has only one device (device 0 locally)
            int local_dev = 0;  // Local device ID is always 0 in multi-process mode
            ctx->tp_dev_ptrs[local_dev] = shared_ptr;
            ctx->tp_streams[local_dev] = ggml_sycl_get_tp_queue(local_dev);
            if (ctx->tp_streams[local_dev] == nullptr) {
                auto & dpct_dev = dpct::dev_mgr::instance().get_device(local_dev);
                ctx->tp_streams[local_dev] = &(dpct_dev.default_queue());
            }
            GGML_SYCL_DEBUG("SYCL TP: Multi-process rank %d using local compute buffer: %p\n",
                            g_sycl_tp_config.mpi_rank, shared_ptr);
        } else {
            // Single-process multi-device: ALL devices share the SAME pointer
            for (int i = 0; i < g_sycl_tp_config.world_size; i++) {
                int dev_id = g_sycl_tp_config.devices[i];
                ctx->tp_dev_ptrs[dev_id] = shared_ptr;  // Same pointer for all!
                ctx->tp_streams[dev_id] = ggml_sycl_get_tp_queue(dev_id);
                if (ctx->tp_streams[dev_id] == nullptr) {
                    auto & dpct_dev = dpct::dev_mgr::instance().get_device(dev_id);
                    ctx->tp_streams[dev_id] = &(dpct_dev.default_queue());
                }
                GGML_SYCL_DEBUG("SYCL TP: Device %d using shared compute buffer: %p\n", dev_id, shared_ptr);
            }
        }

        return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
    }

    // Non-TP mode: use regular allocation
    return ggml_backend_sycl_buffer_type_alloc_buffer(buft, size);
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
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
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    auto dev_count = ggml_backend_sycl_get_device_count();
    if (device >= dev_count || device < 0) {
        GGML_LOG_ERROR("ggml_backend_sycl_host_compute_buffer_type error: device_index:%d is out of range [0, %d]\n",
            device, dev_count - 1);
        GGML_ASSERT(device < dev_count);
    }

    static struct ggml_backend_buffer_type ggml_backend_sycl_host_compute_buffer_types[GGML_SYCL_MAX_DEVICES];
    static bool initialized = false;

    if (!initialized) {
        for (int i = 0; i < dev_count; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            // For TP mode: each device gets its own DEVICE memory compute buffer
            // This allows parallel execution with no cross-device memory issues
            ggml_backend_sycl_host_compute_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_host_compute_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME "_Compute" + std::to_string(i), GGML_SYCL_MEM_DEVICE, stream},
            };
        }
        initialized = true;
    }
    return &ggml_backend_sycl_host_compute_buffer_types[device];
}

// buffer pool for sycl (legacy)
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;

    int device;
    queue_ptr qptr;
    struct ggml_sycl_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

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
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(
            CHECK_TRY_ERROR(ptr = (void *)sycl::malloc_device(
                                look_ahead_size, *qptr)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            return nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%lu, return %p\n", look_ahead_size, ptr);
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
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
            ggml_sycl_buffer b               = buffer_pool[0];
            void *           ptr             = b.ptr;
            *actual_size                     = b.size;
            counter                          = 1;
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

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, int device) {
    // TBD: NO VMM support
    // if (ggml_sycl_info().devices[device].vmm) {
    //     return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(device));
    // }
   return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}

// TBD pool with virtual memory management
// struct ggml_sycl_pool_vmm : public ggml_sycl_pool

/// kernels
typedef void (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream);



static void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x,const int channel_stride_y, const int channel_x_divisor,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel * channel_stride_y + row_y;

        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols,
                           const sycl::nd_item<3> &item_ct1) {
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


template<typename T>
static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void
k_argsort_f32_i32(const float *x, int *dst, const int ncols, int ncols_pad,
                  const int tasks_per_thread, const sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    int col_index =  item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    for (int i = 0; i < tasks_per_thread; i++) {
        int col = col_index * tasks_per_thread + i;
        if (col >= ncols_pad) {
            return;
        }
    }

    const float * x_row = x + row * ncols;
    auto dst_row = (int *)dpct_local;

    // initialize indices
    for (int i=0;i<tasks_per_thread;i++){
        int col = col_index*tasks_per_thread+i;
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
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] < x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] >
                                        x_row[dst_row[ixj]]))) {
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

static void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past,
                              const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void scale_f32(const float * x, float * dst, const float scale, const float bias, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}


template <typename Ti, typename To>
static  void pool2d_nchw_kernel(
        const int ih, const int iw, const int oh, const int ow,
        const int kh, const int kw, const int sh, const int sw,
        const int ph, const int pw, const int parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> &item_ct1) {
        int idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if (idx >= parallel_elements) {
            return;
        }

        const int I_HW = ih * iw;
        const int O_HW = oh * ow;
        const int nc = idx / O_HW;
        const int cur_oh = idx % O_HW / ow;
        const int cur_ow = idx % O_HW % ow;
        const Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        const int start_h = cur_oh * sh - ph;
        const int bh = sycl::max(0, start_h);
        const int eh = sycl::min(ih, start_h + kh);
        const int start_w = cur_ow * sw - pw;
        const int bw = sycl::max(0, start_w);
        const int ew = sycl::min(iw, start_w + kw);

        To res = 0;

        switch (op) {
            case GGML_OP_POOL_AVG: res = 0; break;
            case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
            default:
                res      = (To) sycl::nan(uint32_t(0));
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
                    case GGML_OP_POOL_AVG: res += (cur / (kh * kw)); break;
                    case GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); break;
                    default:
                        res = (To) sycl::nan(uint32_t(0));
                        break;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}


static void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y,
                                           float *dst, const int ncols_x,
                                           const int nrows_x,
                                           const int nchannels_x,
                                           const int nchannels_y,
                                           queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(
    const void *vx, const float *y, float *dst, const int ncols_x,
    const int nrows_x, const int row_stride_x, const int nchannels_x,
    const int nchannels_y, const int channel_stride_x, const int channel_stride_y, queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



static void scale_f32_sycl(const float *x, float *dst, const float scale, const float bias,
                           const int k, queue_ptr stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


static void sum_rows_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream, int device) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    int nth = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    if (nth > max_block_size)
        nth = max_block_size;

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);
    GGML_ASSERT(shared_mem<=ggml_sycl_info().devices[device].smpbo);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

static void argmax_f32_i32_sycl(const float *x, int *dst, const int ncols,
                               const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t shared_mem = 256 * sizeof(float);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_data(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);
        sycl::local_accessor<int, 1> shared_indices(
            sycl::range<1>(shared_mem/sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                const int tid = item_ct1.get_local_id(2);
                const int row = item_ct1.get_global_id(1);

                float max_val = -INFINITY;
                int max_idx = -1;

                for (int col = tid; col < ncols; col += 256) {
                    float val = x[row * ncols + col];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                for (int stride = 256/2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        float val1 = shared_data[tid];
                        float val2 = shared_data[tid + stride];
                        if (val2 > val1) {
                            shared_data[tid] = val2;
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
static void diag_mask_inf_f32_sycl(const float *x, float *dst,
                                   const int ncols_x, const int nrows_x,
                                   const int rows_per_channel, const int n_past,
                                   queue_ptr stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst,
                                          const struct ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          queue_ptr stream) try {

    dpct::memcpy_direction kind;
    char * src_ptr;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
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
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else if (ggml_backend_buffer_is_sycl_tp(src->buffer)) {
        // TP (Tensor Parallelism) buffer - similar to split buffer
        // Data is stored in device-specific locations within extra
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        src_ptr = (char *) extra->data_device[id];
        GGML_SYCL_DEBUG("[CPY_TENSOR_2D] TP buffer: device=%d src_ptr=%p dst=%p (tensor=%s) stream_dev=%s\n",
                        id, (void*)src_ptr, dst, src->name,
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
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) return r;
        }
        return 0;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr &stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    const int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

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
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1->data + src1_padded_row_size
                                         : src1_as_f16.get();

#if GGML_SYCL_DNNL
        if (!g_ggml_sycl_disable_dnn) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        else
#endif
        {
            ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool(), row_diff * src1_ncols);

            const sycl::half alpha_f16 = 1.0f;
            const sycl::half beta_f16  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::math::transpose::trans,
                oneapi::math::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha_f16, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta_f16,
                dst_f16.get(), dpct::library_data_t::real_half, ldc,
                dpct::library_data_t::real_half)));
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting dst to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16, dst);
            to_fp32_sycl(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
        }
    } else {
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

#if GGML_SYCL_DNNL
        if (!g_ggml_sycl_disable_dnn) {
            DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                      DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        else
#endif
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(oneapi::math::blas::column_major::gemm(
                get_onemath_backend(*stream), oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, row_diff,
                src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    const int64_t IH = dst->src[0]->ne[1];
    const int64_t IW = dst->src[0]->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const int parallel_elements = N * OC * OH * OW;
    const int num_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, num_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               item_ct1);
        });
}

inline void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne = ggml_nelements(dst->src[0]);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
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

    main_stream->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device);
}

inline void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

inline void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t ne01 = dst->src[0]->ne[1];
    const int nrows0 = ggml_nrows(dst->src[0]);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

inline void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

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
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

template <template <int> typename quantize_f>
static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) try {

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

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char> src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char> src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *src0_dd = nullptr;
        float *src1_ddf = nullptr; // float
        char *src1_ddq = nullptr;  // q8_1
        float *dst_dd = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low  = ne01*tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high  = ne01*tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            // For TP buffers, use the device-specific data pointer
            if (ggml_backend_buffer_is_sycl_tp(src0->buffer)) {
                ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
                dev[i].src0_dd = (char *) extra->data_device[i];
                GGML_SYCL_DEBUG("[MUL_MAT] TP buffer src0 device=%d ptr=%p (tensor=%s)\n",
                                i, (void*)dev[i].src0_dd, src0->name);
            } else {
                // Use ggml_sycl_get_data_ptr which handles staging mmap'd data for TP mode
                dev[i].src0_dd = (char *) ggml_sycl_get_data_ptr(src0, i);
                GGML_SYCL_DEBUG("[MUL_MAT] non-TP buffer src0 device=%d ptr=%p (tensor=%s)\n",
                                i, (void*)dev[i].src0_dd, src0->name);
            }
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
            GGML_SYCL_DEBUG("[MUL_MAT] NON-CONTIGUOUS src0 device=%d ptr=%p (tensor=%s, is_tp=%d)\n",
                            i, (void*)dev[i].src0_dd, src0->name,
                            ggml_backend_buffer_is_sycl_tp(src0->buffer) ? 1 : 0);
        }

        if (src1_on_device && src1_is_contiguous) {
            // For TP compute buffers, use device-specific pointer
            dev[i].src1_ddf = (float *) ggml_sycl_get_data_ptr(src1, i);
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            // Zero padding blocks to prevent garbage in MMQ when ne10 < src1_padded_col_size
            // Bug: quantize_row_q8_1_sycl only fills ne10/QK8_1 blocks per row, but MMQ reads
            // src1_padded_col_size/QK8_1 blocks per row. The padding blocks must be zeroed.
            if (ne10 != src1_padded_col_size) {
                stream->memset(dev[i].src1_ddq, 0, nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);
            }

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                // Debug: print pointers and device info before quantize
                GGML_SYCL_DEBUG("[QUANTIZE DEBUG] device=%d src1_ddf=%p src1_ddq=%p stream_device=%s\n",
                                i, (void*)dev[i].src1_ddf, (void*)dev[i].src1_ddq,
                                stream->get_device().get_info<sycl::info::device::name>().c_str());
                try {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } catch (sycl::exception const &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        if (dst_on_device) {
            // For TP compute buffers, use device-specific pointer
            dev[i].dst_dd = (float *) ggml_sycl_get_data_ptr(dst, i);
            GGML_SYCL_DEBUG("[MUL_MAT] dst device=%d ptr=%p (tensor=%s)\n",
                            i, (void*)dev[i].dst_dd, dst->name);
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool src1_on_device = i == ctx.device;
            const bool  dst_on_device = i == ctx.device;
            const int64_t row_diff = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = dev[i].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if constexpr (quantize_enabled) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
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
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } catch (const sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[i].row_low, dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i,
                            row_diff * sizeof(float), row_diff * sizeof(float),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
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
    stream->submit([&](sycl::handler& cgh) {
        float* dst_local = dst;
        const float* src_local = src;
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            dst_local[idx] += src_local[idx];
        });
    }).wait();
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
static std::unordered_map<void*, void*> g_tp_column_parallel_outputs;
static std::mutex g_tp_column_parallel_mutex;

// FFN input storage: stores the input to FFN column-parallel layers
// This is needed so that row-parallel (ffn_down) can compute device 1's full FFN path
// (struct ffn_input_storage defined in common.hpp)
std::unordered_map<int, ffn_input_storage> g_tp_ffn_inputs;  // Key: layer number
std::mutex g_tp_ffn_input_mutex;

// Extract layer number from tensor name (e.g., "blk.0.ffn_gate" -> 0)
int ggml_sycl_tp_extract_layer_number(const char * name) {
    if (!name) return -1;
    const char * blk = strstr(name, "blk.");
    if (!blk) return -1;
    return atoi(blk + 4);
}

// Local alias for convenience
static int extract_layer_number(const char * name) {
    return ggml_sycl_tp_extract_layer_number(name);
}

// FFN weight storage: stores references to FFN weight tensors for device 1 computation
// (struct ffn_weight_refs defined in common.hpp)
std::unordered_map<int, ffn_weight_refs> g_tp_ffn_weights;  // Key: layer number
std::mutex g_tp_ffn_weight_mutex;

// Attention input storage: stores the input to attention column-parallel layers
// This is needed so that row-parallel (attn_output) can compute device 1's full attention path
// (struct attn_input_storage defined in common.hpp)
std::unordered_map<int, attn_input_storage> g_tp_attn_inputs;  // Key: layer number
std::mutex g_tp_attn_input_mutex;

// Attention weight storage: stores references to attention weight tensors for device 1 computation
// (struct attn_weight_refs defined in common.hpp)
std::unordered_map<int, attn_weight_refs> g_tp_attn_weights;  // Key: layer number
std::mutex g_tp_attn_weight_mutex;

// Async FFN jobs: tracks in-flight FFN computations on device 1
// (struct tp_async_ffn_job defined in common.hpp)
std::unordered_map<int, tp_async_ffn_job> g_tp_async_ffn_jobs;  // Key: layer number
std::mutex g_tp_async_ffn_mutex;

// Async attention jobs: tracks in-flight attention computations on device 1
// (struct tp_async_attn_job defined in common.hpp)
std::unordered_map<int, tp_async_attn_job> g_tp_async_attn_jobs;  // Key: layer number
std::mutex g_tp_async_attn_mutex;

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
int g_ggml_sycl_tp_threaded_ffn = 0;  // DISABLED - causes hangs at MMVQ kernel

// Forward declaration of worker thread function
static void tp_device1_worker_thread_func();

// Worker thread function: runs FFN computations on device 1
static void tp_device1_worker_thread_func() {
    auto & w = g_tp_device1_worker;

    const int device = g_sycl_tp_config.devices[1];

    // Create a DEDICATED in-order queue for the worker thread
    // Using the shared TP queue causes hangs due to SYCL queue contention
    ggml_sycl_set_device(device);
    sycl::device dev = dpct::get_device(device);
    static sycl::queue * worker_queue = nullptr;
    if (!worker_queue) {
        worker_queue = new sycl::queue(dev, sycl::property_list{sycl::property::queue::in_order()});
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
        worker_ctx = new ggml_backend_sycl_context(device);
        worker_ctx->name = "tp_worker";
    }
    ggml_backend_sycl_context & ctx = *worker_ctx;

    fprintf(stderr, "SYCL TP WORKER: Thread started on device %d, dedicated queue=%p\n", device, (void*)stream);

    while (true) {
        tp_ffn_work_item work;

        // Wait for work or shutdown
        {
            std::unique_lock<std::mutex> lock(w.work_mutex);
            w.work_cv.wait(lock, [&w] {
                return w.shutdown.load() || !w.work_queue.empty();
            });

            if (w.shutdown.load() && w.work_queue.empty()) {
                fprintf(stderr, "SYCL TP WORKER: Shutdown requested, exiting\n");
                break;
            }

            work = std::move(w.work_queue.front());
            w.work_queue.pop();
        }

        // Process FFN work
        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Processing layer %d, K=%lld, batch=%lld\n",
                    work.layer, (long long)work.K_full, (long long)work.batch);
        }

        // Validate weight pointers
        if (!work.weights.gate || !work.weights.up || !work.weights.down) {
            fprintf(stderr, "SYCL TP WORKER: Null weight tensor for layer %d\n", work.layer);
            continue;
        }

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Weight pointers: gate=%p, up=%p, down=%p\n",
                    (void*)work.weights.gate, (void*)work.weights.up, (void*)work.weights.down);
            fprintf(stderr, "SYCL TP WORKER: gate->extra=%p, up->extra=%p, down->extra=%p\n",
                    work.weights.gate->extra, work.weights.up->extra, work.weights.down->extra);
        }

        // Get weight shards for device 1
        auto * gate_extra = static_cast<ggml_tensor_extra_gpu *>(work.weights.gate->extra);
        auto * up_extra = static_cast<ggml_tensor_extra_gpu *>(work.weights.up->extra);
        auto * down_extra = static_cast<ggml_tensor_extra_gpu *>(work.weights.down->extra);

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Accessing data_device[%d]\n", device);
            fflush(stderr);
        }

        void * gate_weight_1 = gate_extra ? gate_extra->data_device[device] : nullptr;
        void * up_weight_1 = up_extra ? up_extra->data_device[device] : nullptr;
        void * down_weight_1 = down_extra ? down_extra->data_device[device] : nullptr;

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Weight device pointers: gate=%p, up=%p, down=%p\n",
                    gate_weight_1, up_weight_1, down_weight_1);
            fflush(stderr);
        }

        if (!gate_weight_1 || !up_weight_1 || !down_weight_1) {
            fprintf(stderr, "SYCL TP WORKER: Missing weight shards for layer %d\n", work.layer);
            continue;
        }

        // Get dimensions
        const int64_t N_hidden_shard = gate_extra->tp_local_ne[1];
        const int64_t N_out = down_extra->tp_local_ne[1];

        // Allocate buffers on device 1
        const size_t q8_1_ts = sizeof(block_q8_1);
        const size_t q8_1_bs = QK8_1;
        const int64_t K_full_padded = GGML_PAD(work.K_full, MATRIX_ROW_PADDING);
        const int64_t N_hidden_shard_padded = GGML_PAD(N_hidden_shard, MATRIX_ROW_PADDING);

        const size_t input_q8_size = work.batch * K_full_padded * q8_1_ts / q8_1_bs;
        const size_t hidden_size = N_hidden_shard * work.batch * sizeof(float);
        const size_t hidden_q8_size = work.batch * N_hidden_shard_padded * q8_1_ts / q8_1_bs;
        const size_t output_size = N_out * work.batch * sizeof(float);

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Allocating buffers for layer %d (input_q8=%zu, hidden=%zu, output=%zu)\n",
                    work.layer, input_q8_size, hidden_size, output_size);
            fflush(stderr);
        }

        char * input_q8_dev = (char *)sycl::malloc_device(input_q8_size, *stream);
        float * gate_out = (float *)sycl::malloc_device(hidden_size, *stream);
        float * up_out = (float *)sycl::malloc_device(hidden_size, *stream);
        float * hidden_out = (float *)sycl::malloc_device(hidden_size, *stream);
        char * hidden_q8_dev = (char *)sycl::malloc_device(hidden_q8_size, *stream);
        float * partial_out = (float *)sycl::malloc_device(output_size, *stream);
        float * result_buf = (float *)ggml_sycl_host_malloc(output_size);

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Buffers allocated: input_q8=%p, gate=%p, up=%p, hidden=%p, partial=%p, result=%p\n",
                    (void*)input_q8_dev, (void*)gate_out, (void*)up_out, (void*)hidden_out, (void*)partial_out, (void*)result_buf);
            fflush(stderr);
        }

        if (!input_q8_dev || !gate_out || !up_out || !hidden_out || !hidden_q8_dev || !partial_out || !result_buf) {
            fprintf(stderr, "SYCL TP WORKER: Buffer allocation failed for layer %d\n", work.layer);
            if (input_q8_dev) sycl::free(input_q8_dev, *stream);
            if (gate_out) sycl::free(gate_out, *stream);
            if (up_out) sycl::free(up_out, *stream);
            if (hidden_out) sycl::free(hidden_out, *stream);
            if (hidden_q8_dev) sycl::free(hidden_q8_dev, *stream);
            if (partial_out) sycl::free(partial_out, *stream);
            if (result_buf) ggml_sycl_host_free(result_buf);
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
        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Step 1 - Quantizing input\n");
            fflush(stderr);
        }
        quantize_row_q8_1_sycl<quantize_q8_1>(
            work.input_dev1, input_q8_dev,
            work.K_full, work.batch, K_full_padded, stream);
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
        ggml_sycl_op_mul_mat_vec_q(ctx, work.weights.gate, nullptr, &fake_dst_hidden,
            (const char *)gate_weight_1, nullptr, input_q8_dev,
            gate_out, 0, N_hidden_shard, work.batch, K_full_padded, stream);

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Step 4 - Up matmul\n");
            fflush(stderr);
        }
        ggml_sycl_op_mul_mat_vec_q(ctx, work.weights.up, nullptr, &fake_dst_hidden,
            (const char *)up_weight_1, nullptr, input_q8_dev,
            up_out, 0, N_hidden_shard, work.batch, K_full_padded, stream);
        stream->wait();

        // Step 4: SiLU activation and multiply
        const int64_t n_elements = N_hidden_shard * work.batch;
        const int block_size = 256;
        const int num_blocks = (n_elements + block_size - 1) / block_size;
        stream->parallel_for(
            sycl::nd_range<1>(num_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                const int i = item.get_global_id(0);
                if (i < n_elements) {
                    float g = gate_out[i];
                    float u = up_out[i];
                    float silu_g = g / (1.0f + sycl::native::exp(-g));
                    hidden_out[i] = silu_g * u;
                }
            });

        // Step 5: Quantize hidden for down matmul
        quantize_row_q8_1_sycl<quantize_q8_1>(
            hidden_out, hidden_q8_dev,
            N_hidden_shard, work.batch, N_hidden_shard_padded, stream);

        // Step 6: Down matmul
        stream->memset(partial_out, 0, output_size);
        stream->wait();
        ggml_sycl_op_mul_mat_vec_q(ctx, work.weights.down, nullptr, &fake_dst_out,
            (const char *)down_weight_1, nullptr, hidden_q8_dev,
            partial_out, 0, N_out, work.batch, N_hidden_shard_padded, stream);

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
            w.results[work.layer] = {
                work.layer,
                result_buf,
                N_out,
                work.batch,
                output_size,
                true
            };
        }
        w.result_cv.notify_all();

        if (g_ggml_sycl_tp_debug) {
            fprintf(stderr, "SYCL TP WORKER: Completed layer %d, result[0..3]=[%f,%f,%f,%f]\n",
                    work.layer, result_buf[0], result_buf[1], result_buf[2], result_buf[3]);
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
        w.result_cv.wait_for(lock, std::chrono::milliseconds(5000), [&w, layer] {
            return w.results.find(layer) != w.results.end() && w.results[layer].valid;
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
    auto it = w.results.find(layer);
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
    float * k_cache;      // [max_seq_len, n_heads_kv * head_dim]
    float * v_cache;      // [max_seq_len, n_heads_kv * head_dim]
    int64_t seq_pos;      // Current sequence position (next position to write)
    int64_t max_seq_len;  // Allocated cache size
    int64_t n_heads_kv;   // Number of KV heads per device
    int64_t head_dim;     // Dimension per head
    queue_ptr stream;     // Device 1 stream for cache operations
};
std::unordered_map<int, dev1_kv_cache_entry> g_tp_dev1_kv_cache;  // Key: layer number
std::mutex g_tp_dev1_kv_cache_mutex;
static bool g_tp_dev1_kv_cache_initialized = false;

// Initialize or resize the KV cache for a given layer on device 1
// Called during prompt processing to set up cache with proper dimensions
static void init_dev1_kv_cache(int layer, int64_t max_seq_len, int64_t n_heads_kv,
                                int64_t head_dim, queue_ptr stream) {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);

    auto it = g_tp_dev1_kv_cache.find(layer);
    if (it != g_tp_dev1_kv_cache.end()) {
        // Cache exists - check if reallocation needed
        if (it->second.max_seq_len >= max_seq_len) {
            return;  // Already big enough
        }
        // Free old cache
        if (it->second.k_cache) sycl::free(it->second.k_cache, *stream);
        if (it->second.v_cache) sycl::free(it->second.v_cache, *stream);
    }

    // Allocate new cache (oversized for typical generations)
    int64_t cache_max_seq = std::max(max_seq_len, (int64_t)4096);
    size_t cache_size = cache_max_seq * n_heads_kv * head_dim * sizeof(float);

    dev1_kv_cache_entry entry;
    entry.k_cache = (float *)sycl::malloc_device(cache_size, *stream);
    entry.v_cache = (float *)sycl::malloc_device(cache_size, *stream);
    entry.seq_pos = 0;
    entry.max_seq_len = cache_max_seq;
    entry.n_heads_kv = n_heads_kv;
    entry.head_dim = head_dim;
    entry.stream = stream;

    if (!entry.k_cache || !entry.v_cache) {
        fprintf(stderr, "SYCL TP: WARNING - Failed to allocate KV cache for layer %d (size=%zu)\n",
                layer, cache_size);
        if (entry.k_cache) sycl::free(entry.k_cache, *stream);
        if (entry.v_cache) sycl::free(entry.v_cache, *stream);
        return;
    }

    g_tp_dev1_kv_cache[layer] = entry;
    static int log_count = 0;
    if (g_ggml_sycl_tp_debug && log_count++ < 3) {
        fprintf(stderr, "SYCL TP: Allocated dev1 KV cache for layer %d: max_seq=%lld, n_kv_heads=%lld, head_dim=%lld\n",
                layer, (long long)cache_max_seq, (long long)n_heads_kv, (long long)head_dim);
    }
}

// Append new K and V values to the cache for a layer
// Called after K/V projection in the attention path
// k_new/v_new: [batch, n_heads_kv * head_dim] where batch is the new tokens to add
static void append_to_dev1_kv_cache(int layer, const float * k_new, const float * v_new,
                                     int64_t batch, queue_ptr stream) {
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
        fprintf(stderr, "SYCL TP: ERROR - KV cache overflow layer %d: pos=%lld + batch=%lld > max=%lld\n",
                layer, (long long)entry.seq_pos, (long long)batch, (long long)entry.max_seq_len);
        return;
    }

    // Copy new K and V to cache
    // Cache layout: [max_seq_len, n_heads_kv * head_dim]
    size_t kv_stride = entry.n_heads_kv * entry.head_dim;
    size_t copy_size = batch * kv_stride * sizeof(float);
    size_t offset = entry.seq_pos * kv_stride * sizeof(float);

    stream->memcpy((char *)entry.k_cache + offset, k_new, copy_size);
    stream->memcpy((char *)entry.v_cache + offset, v_new, copy_size);
    stream->wait();

    entry.seq_pos += batch;

    static int dbg_count = 0;
    if (g_ggml_sycl_tp_debug && dbg_count++ < 3) {  // Reduced debug output
        fprintf(stderr, "TP DEBUG: Appended %lld tokens to layer %d KV cache, now at pos %lld\n",
                (long long)batch, layer, (long long)entry.seq_pos);
    }
}

// Get the current sequence length in the cache for a layer
static int64_t get_dev1_kv_cache_seq_len(int layer) {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);
    auto it = g_tp_dev1_kv_cache.find(layer);
    if (it == g_tp_dev1_kv_cache.end()) {
        return 0;
    }
    return it->second.seq_pos;
}

// Get cached K and V pointers for a layer
static bool get_dev1_kv_cache_ptrs(int layer, float ** k_cache, float ** v_cache, int64_t * seq_len) {
    std::lock_guard<std::mutex> lock(g_tp_dev1_kv_cache_mutex);
    auto it = g_tp_dev1_kv_cache.find(layer);
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
    if (!tensor || !tensor->name) return;

    int layer = extract_layer_number(tensor->name);
    if (layer < 0) return;

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
    if (strstr(tensor->name, "attn_q") || strstr(tensor->name, "attn_k") ||
        strstr(tensor->name, "attn_v") || strstr(tensor->name, "attn_output")) {
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
    if (g_tp_column_parallel_outputs.empty()) return;

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
    auto it = g_tp_column_parallel_outputs.find(tensor->data);
    if (it != g_tp_column_parallel_outputs.end()) {
        return it->second;
    }
    return nullptr;
}

// TP column-parallel mul_mat post-processing
// For FFN layers: stores the input (src1) so row-parallel (ffn_down) can compute device 1's path
// For attention layers: stores the input (src1) so row-parallel (attn_output) can compute device 1's path
static void ggml_sycl_mul_mat_tp_column_parallel_post(ggml_backend_sycl_context & ctx,
                                                        const ggml_tensor * src0,
                                                        const ggml_tensor * src1,
                                                        ggml_tensor * dst) {
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
    bool is_attn_q = strstr(name, "attn_q") != nullptr && strstr(name, "attn_qkv") == nullptr;

    if (!is_ffn_gate && !is_attn_q) {
        return;  // Only store at ffn_gate and attn_q to avoid duplicate storage
    }

    int layer = extract_layer_number(name);

    // DEBUG: Track column-parallel entry during decode (batch=1)
    static int col_par_decode_dbg = 0;
    if (g_ggml_sycl_tp_debug && src1->ne[1] == 1 && is_ffn_gate && col_par_decode_dbg++ < 100) {
        fprintf(stderr, "TP DEBUG COL_PARALLEL FFN_GATE decode: layer=%d name=%s map_size_before=%zu\n",
                layer, name, g_tp_ffn_inputs.size());
    }

    // DEBUG: Check layer input during decode for all layers
    static int layer_input_dbg = 0;
    bool is_debug_layer = (layer >= 0 && layer <= 31);
    if (g_ggml_sycl_tp_debug && src1->ne[1] == 1 && is_debug_layer && is_attn_q && layer_input_dbg++ < 200) {
        queue_ptr main_stream = ctx.stream();
        float sample[8];
        // Use device-specific pointer for TP
        const float * src1_dd = static_cast<const float *>(ggml_sycl_get_data_ptr(src1, ctx.device));
        main_stream->memcpy(sample, src1_dd, 8*sizeof(float)).wait();
        float sum = 0;
        for (int i = 0; i < 8; i++) sum += sample[i];
        fprintf(stderr, "LAYER%d_INPUT decode: src1[0..7]=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] sum=%.4f\n",
                layer, sample[0], sample[1], sample[2], sample[3],
                sample[4], sample[5], sample[6], sample[7], sum);
    }
    if (layer < 0) {
        return;
    }

    const int main_device = ctx.device;
    const int device = g_sycl_tp_config.devices[1];  // Device 1

    // Copy src1 (input) to device 1
    const int64_t K = src1->ne[0];
    const int64_t batch = src1->ne[1];
    const size_t src1_size = batch * K * sizeof(float);

    // DEBUG: Print device 0's input values for comparison
    static int dev0_input_dbg = 0;
    bool debug_input = (g_ggml_sycl_tp_debug && is_ffn_gate && layer < 3 && dev0_input_dbg++ < 10);  // Enable for first few layers
    if (debug_input) {
        queue_ptr main_stream = ctx.stream();
        void * src1_ptr = ggml_sycl_get_data_ptr(src1, main_device);
        float sample[4];
        main_stream->memcpy(sample, src1_ptr, 4*sizeof(float)).wait();
        fprintf(stderr, "TP DEBUG FFN CAPTURE layer %d batch=%lld: src1->data=%p src1_ptr=%p %s\n",
                layer, (long long)batch, src1->data, src1_ptr, (src1->data != src1_ptr) ? "MISMATCH!" : "");
        bool has_nan = std::isnan(sample[0]) || std::isnan(sample[1]) || std::isnan(sample[2]) || std::isnan(sample[3]);
        fprintf(stderr, "TP DEBUG FFN CAPTURE layer %d batch=%lld: device0_input[0..3]=[%f,%f,%f,%f] nan=%d\n",
                layer, (long long)batch, sample[0], sample[1], sample[2], sample[3], has_nan);
    }

    ggml_sycl_set_device(device);
    queue_ptr stream = ctx.stream(device, 0);

    // DEBUG: Check L31 weight BEFORE malloc_device for input staging
    static int staging_pre_malloc_dbg = 0;
    bool check_staging = (g_ggml_sycl_tp_debug && layer == 31 && is_ffn_gate && staging_pre_malloc_dbg < 3);
    if (check_staging) {
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        queue_ptr dev0_stream = ctx.stream(main_device, 0);
        uint8_t weight_bytes[18];
        dev0_stream->memcpy(weight_bytes, (void*)l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float d_f = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING PRE-MALLOC: weight d=%f %s\n",
                d_f, (d_f > 100.0f) ? "CORRUPTED" : "OK");
    }

    // Allocate on device 1
    float * input_dev1 = (float *)sycl::malloc_device(src1_size, *stream);

    // DEBUG: Check L31 weight AFTER malloc_device and print allocated address
    if (check_staging) {
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        queue_ptr dev0_stream = ctx.stream(main_device, 0);
        uint8_t weight_bytes[18];
        dev0_stream->memcpy(weight_bytes, (void*)l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float d_f = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING POST-MALLOC: weight d=%f, input_dev1=%p %s\n",
                d_f, (void*)input_dev1, (d_f > 100.0f) ? "CORRUPTED" : "OK");
        staging_pre_malloc_dbg++;
    }

    if (!input_dev1) {
        GGML_LOG_ERROR("SYCL TP: ERROR - failed to allocate %s input on device %d\n",
                is_ffn_gate ? "FFN" : "attention", device);
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
    bool trace_staging = (g_ggml_sycl_tp_debug && layer == 31 && is_ffn_gate && staging_step_dbg < 1);
    if (trace_staging) {
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        uint8_t weight_bytes[18];
        main_stream->memcpy(weight_bytes, (void*)l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float d_f = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING STEP1 (after host malloc): weight d=%f %s\n",
                d_f, (d_f > 100.0f) ? "CORRUPTED" : "OK");
    }

    // For FFN gate: try to use cached FFN norm (prevents buffer aliasing issues)
    // The GGML scheduler may reuse the ffn_norm buffer before we can read it
    void* cached_ffn_norm = nullptr;
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
            uint8_t weight_bytes[18];
            main_stream->memcpy(weight_bytes, (void*)l31_weight_addr, 18).wait();
            uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
            float d_f = ggml_fp16_to_fp32(d_raw);
            fprintf(stderr, "TP DEBUG L31 STAGING STEP2 (after host memcpy): weight d=%f %s\n",
                    d_f, (d_f > 100.0f) ? "CORRUPTED" : "OK");
        }
        // DEBUG: Check FFN input at storage time for batch=1 (disabled - TP working)
        static int ffn_store_dbg = 0;
        bool debug_layer = (batch == 1 && layer == 0 && ffn_store_dbg++ < 0);  // Disabled
        if (debug_layer) {
            float check[4];
            memcpy(check, host_buf, 4*sizeof(float));
            bool has_nan = std::isnan(check[0]) || std::isnan(check[1]) || std::isnan(check[2]) || std::isnan(check[3]);
            fprintf(stderr, "TP DEBUG FFN STORE layer %d batch=1: src=%s, input[0..3]=[%f,%f,%f,%f] nan=%d\n",
                    layer, cached_ffn_norm ? "cached" : "src1", check[0], check[1], check[2], check[3], has_nan);
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
        queue_ptr dev0_stream = ctx.stream(main_device, 0);
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        uint8_t weight_bytes[18];
        dev0_stream->memcpy(weight_bytes, (void*)l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float d_f = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING STEP3 (after dev1 memcpy): weight d=%f %s\n",
                d_f, (d_f > 100.0f) ? "CORRUPTED" : "OK");
        ggml_sycl_set_device(device);
    }

    // OPTIMIZATION: Don't free host_buf - it's a persistent staging buffer
    // managed by ggml_sycl_tp_ensure_host_staging()
    // sycl::free(host_buf, *main_stream);

    // DEBUG: Check after host buffer free
    if (trace_staging) {
        ggml_sycl_set_device(main_device);
        queue_ptr dev0_stream = ctx.stream(main_device, 0);
        uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
        uint8_t weight_bytes[18];
        dev0_stream->memcpy(weight_bytes, (void*)l31_weight_addr, 18).wait();
        uint16_t d_raw = weight_bytes[0] | (weight_bytes[1] << 8);
        float d_f = ggml_fp16_to_fp32(d_raw);
        fprintf(stderr, "TP DEBUG L31 STAGING STEP4 (after host free): weight d=%f %s\n",
                d_f, (d_f > 100.0f) ? "CORRUPTED" : "OK");
        staging_step_dbg++;
        ggml_sycl_set_device(device);
    }

    // Store for later use by row-parallel layer
    if (is_ffn_gate) {
        // Store the input
        {
            std::lock_guard<std::mutex> lock(g_tp_ffn_input_mutex);
            auto it = g_tp_ffn_inputs.find(layer);
            if (it != g_tp_ffn_inputs.end() && it->second.data != nullptr) {
                sycl::free(it->second.data, *stream);
            }
            g_tp_ffn_inputs[layer] = {input_dev1, K, batch, src1_size};
            // DEBUG: Track FFN input storage during decode
            static int ffn_store_dbg = 0;
            if (g_ggml_sycl_tp_debug && batch == 1 && ffn_store_dbg++ < 100) {
                fprintf(stderr, "TP DEBUG FFN_INPUT_STORE decode: layer=%d ptr=%p map_size_after=%zu\n",
                        layer, (void*)input_dev1, g_tp_ffn_inputs.size());
            }
        }

        // PHASE 4 PIPELINING: Try to launch async FFN computation now
        // This allows device 1 to work while device 0 continues with other ops
        ffn_weight_refs weights = {};
        {
            std::lock_guard<std::mutex> lock(g_tp_ffn_weight_mutex);
            auto it = g_tp_ffn_weights.find(layer);
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
                    work.layer = layer;
                    work.input_dev1 = input_dev1;
                    work.K_full = K;
                    work.batch = batch;
                    work.weights = weights;

                    if (g_ggml_sycl_tp_debug) {
                        static int thread_launch_count = 0;
                        if (thread_launch_count++ < 3) {
                            fprintf(stderr, "SYCL TP: Submitting FFN to worker thread for layer %d (K=%lld, batch=%lld)\n",
                                    layer, (long long)K, (long long)batch);
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
                            layer, (void*)weights.gate, (void*)weights.up, (void*)weights.down);
                }
            }
        }
    } else {  // is_attn_q
        std::lock_guard<std::mutex> lock(g_tp_attn_input_mutex);
        auto it = g_tp_attn_inputs.find(layer);
        if (it != g_tp_attn_inputs.end() && it->second.data != nullptr) {
            sycl::free(it->second.data, *stream);
        }
        g_tp_attn_inputs[layer] = {input_dev1, K, batch, src1_size};

        // DEBUG: Track attn input storage during decode
        static int attn_store_dbg = 0;
        if (g_ggml_sycl_tp_debug && batch == 1 && attn_store_dbg++ < 40) {
            fprintf(stderr, "TP DEBUG ATTN_Q_STORE decode: layer=%d data=%p map_size=%zu\n",
                    layer, (void*)input_dev1, g_tp_attn_inputs.size());
        }
    }

    ggml_sycl_set_device(main_device);
    return;  // Don't run the old code below

#if 0  // Full implementation for future reference
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
        {
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);

            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_dev, src1_ddq_dev,
                                                  K, batch, K_padded, stream);
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
#endif  // End of disabled column-parallel implementation
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
void ggml_sycl_tp_launch_async_ffn(
    ggml_backend_sycl_context & ctx,
    int layer,
    const float * input_dev1,
    int64_t K_full,
    int64_t batch,
    const ffn_weight_refs & weights
) {
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
    const int device = g_sycl_tp_config.devices[1];

    ggml_sycl_set_device(device);
    queue_ptr stream = ctx.stream(device, 0);

    // Get weight shards for device 1
    auto * gate_extra = static_cast<ggml_tensor_extra_gpu *>(weights.gate->extra);
    auto * up_extra = static_cast<ggml_tensor_extra_gpu *>(weights.up->extra);
    auto * down_extra = static_cast<ggml_tensor_extra_gpu *>(weights.down->extra);

    void * gate_weight_1 = gate_extra ? gate_extra->data_device[device] : nullptr;
    void * up_weight_1 = up_extra ? up_extra->data_device[device] : nullptr;
    void * down_weight_1 = down_extra ? down_extra->data_device[device] : nullptr;

    if (!gate_weight_1 || !up_weight_1 || !down_weight_1) {
        GGML_SYCL_DEBUG("SYCL TP ASYNC: Missing weight shards on device 1 for layer %d\n", layer);
        ggml_sycl_set_device(main_device);
        return;
    }

    // Get dimensions
    const int64_t N_hidden_shard = gate_extra->tp_local_ne[1];  // Sharded hidden dim
    const int64_t N_out = down_extra->tp_local_ne[1];          // Output dimension

    // Allocate buffers on device 1
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
    const int64_t K_full_padded = GGML_PAD(K_full, MATRIX_ROW_PADDING);
    const int64_t N_hidden_shard_padded = GGML_PAD(N_hidden_shard, MATRIX_ROW_PADDING);

    const size_t input_q8_size = batch * K_full_padded * q8_1_ts / q8_1_bs;
    const size_t hidden_size = N_hidden_shard * batch * sizeof(float);
    const size_t hidden_q8_size = batch * N_hidden_shard_padded * q8_1_ts / q8_1_bs;
    const size_t output_size = N_out * batch * sizeof(float);

    char * input_q8_dev = (char *)sycl::malloc_device(input_q8_size, *stream);
    float * gate_out = (float *)sycl::malloc_device(hidden_size, *stream);
    float * up_out = (float *)sycl::malloc_device(hidden_size, *stream);
    float * hidden_out = (float *)sycl::malloc_device(hidden_size, *stream);
    char * hidden_q8_dev = (char *)sycl::malloc_device(hidden_q8_size, *stream);
    float * partial_out = (float *)sycl::malloc_device(output_size, *stream);

    // Allocate DEDICATED result buffer for this async job (not shared!)
    // Each layer needs its own buffer to avoid races between concurrent async jobs
    float * result_buf = (float *)ggml_sycl_host_malloc(output_size);

    if (!input_q8_dev || !gate_out || !up_out || !hidden_out || !hidden_q8_dev || !partial_out || !result_buf) {
        GGML_SYCL_DEBUG("SYCL TP ASYNC: Buffer allocation failed for layer %d\n", layer);
        if (input_q8_dev) sycl::free(input_q8_dev, *stream);
        if (gate_out) sycl::free(gate_out, *stream);
        if (up_out) sycl::free(up_out, *stream);
        if (hidden_out) sycl::free(hidden_out, *stream);
        if (hidden_q8_dev) sycl::free(hidden_q8_dev, *stream);
        if (partial_out) sycl::free(partial_out, *stream);
        if (result_buf) ggml_sycl_host_free(result_buf);
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
    quantize_row_q8_1_sycl<quantize_q8_1>(
        input_dev1, input_q8_dev,
        K_full, batch, K_full_padded, stream);

    // Step 2-3: Gate and Up matmuls (sequential on same queue)
    stream->memset(gate_out, 0, hidden_size);
    stream->memset(up_out, 0, hidden_size);
    stream->wait();

    ggml_sycl_op_mul_mat_vec_q(ctx, weights.gate, nullptr, &fake_dst_hidden,
        (const char *)gate_weight_1, nullptr, input_q8_dev,
        gate_out, 0, N_hidden_shard, batch, K_full_padded, stream);

    ggml_sycl_op_mul_mat_vec_q(ctx, weights.up, nullptr, &fake_dst_hidden,
        (const char *)up_weight_1, nullptr, input_q8_dev,
        up_out, 0, N_hidden_shard, batch, K_full_padded, stream);
    stream->wait();

    // Step 4: SiLU activation and multiply
    const int64_t n_elements = N_hidden_shard * batch;
    const int block_size = 256;
    const int num_blocks = (n_elements + block_size - 1) / block_size;
    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * block_size, block_size),
        [=](sycl::nd_item<1> item) {
            const int i = item.get_global_id(0);
            if (i < n_elements) {
                float g = gate_out[i];
                float u = up_out[i];
                float silu_g = g / (1.0f + sycl::native::exp(-g));
                hidden_out[i] = silu_g * u;
            }
        });

    // Step 5: Quantize hidden for down matmul
    quantize_row_q8_1_sycl<quantize_q8_1>(
        hidden_out, hidden_q8_dev,
        N_hidden_shard, batch, N_hidden_shard_padded, stream);

    // Step 6: Down matmul
    stream->memset(partial_out, 0, output_size);
    stream->wait();
    ggml_sycl_op_mul_mat_vec_q(ctx, weights.down, nullptr, &fake_dst_out,
        (const char *)down_weight_1, nullptr, hidden_q8_dev,
        partial_out, 0, N_out, batch, N_hidden_shard_padded, stream);

    // Step 7: Copy result to shared buffer (this is the final operation)
    // Store the event so we can wait for it later
    sycl::event completion_event = stream->memcpy(result_buf, partial_out, output_size);

    // Clean up device 1 buffers (submit async free after computation)
    // Note: These frees depend on the computation completing
    stream->submit([&](sycl::handler& h) {
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
        g_tp_async_ffn_jobs[layer] = {
            layer,
            completion_event,
            result_buf,
            N_out,
            batch,
            output_size,
            true
        };
    }

    if (g_ggml_sycl_tp_debug) {
        static int launch_dbg = 0;
        if (launch_dbg++ < 5) {
            fprintf(stderr, "SYCL TP ASYNC: Launched FFN layer %d, K=%lld, batch=%lld, N_hidden=%lld, N_out=%lld\n",
                    layer, (long long)K_full, (long long)batch, (long long)N_hidden_shard, (long long)N_out);
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
        auto it = g_tp_async_ffn_jobs.find(layer);
        if (it == g_tp_async_ffn_jobs.end() || !it->second.valid) {
            return nullptr;
        }
        job = it->second;
        it->second.valid = false;  // Mark as consumed
    }

    // Wait for computation to complete
    job.completion_event.wait();

    if (out_ne0) *out_ne0 = job.ne0;
    if (out_ne1) *out_ne1 = job.ne1;
    if (out_size) *out_size = job.result_size;

    if (g_ggml_sycl_tp_debug) {
        static int wait_dbg = 0;
        if (wait_dbg++ < 5) {
            fprintf(stderr, "SYCL TP ASYNC: Waited for FFN layer %d, result[0..3]=[%f,%f,%f,%f]\n",
                    layer, job.result_buf[0], job.result_buf[1], job.result_buf[2], job.result_buf[3]);
        }
    }

    return job.result_buf;
}

// Launch async attention computation on device 1
void ggml_sycl_tp_launch_async_attn(
    ggml_backend_sycl_context & ctx,
    int layer,
    const float * input_dev1,
    int64_t K_full,
    int64_t batch,
    const attn_weight_refs & weights
) {
    // TODO: Implement async attention (similar to FFN but with Q/K/V/O path)
    // For now, attention will use the synchronous path
    (void)ctx; (void)layer; (void)input_dev1; (void)K_full; (void)batch; (void)weights;
}

// Wait for async attention result
float * ggml_sycl_tp_wait_async_attn(int layer, int64_t * out_ne0, int64_t * out_ne1, size_t * out_size) {
    // TODO: Implement async attention wait
    (void)layer; (void)out_ne0; (void)out_ne1; (void)out_size;
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
                                                    const ggml_tensor * src0,
                                                    const ggml_tensor * src1,
                                                    ggml_tensor * dst) {
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

    const int world_size = extra->tp_world_size;
    const int main_device = ctx.device;

    // Dimensions
    const int64_t ne00 = src0->ne[0];  // K_shard (sharded K dimension per device)
    const int64_t ne01 = src0->ne[1];  // N (output dimension)
    const int64_t ne10 = src1->ne[0];  // K (may be sharded if from column-parallel)
    const int64_t ne11 = src1->ne[1];  // batch/seq_len

    const int64_t K_shard = ne00;

    // Output size
    const size_t dst_nelems = ggml_nelements(dst);
    const size_t dst_size = ggml_nbytes(dst);

    // Check if src1 already has sharded dimension (from column-parallel layer output)
    const bool src1_from_column_parallel = (ne10 == K_shard);

    if (src1_from_column_parallel) {
        // src1 came from column-parallel layer - check if this is ffn_down
        const char * name = src0->name;
        int layer = extract_layer_number(name);
        bool is_ffn_down = name && strstr(name, "ffn_down");

        if (is_ffn_down && layer >= 0) {
            // For FFN: check for async result first (Phase 4 pipelining)
            static int ffn_dbg_count = 0;
            if (g_ggml_sycl_tp_debug && ffn_dbg_count++ < 3) fprintf(stderr, "TP DEBUG: FFN down layer %d, entering computation path\n", layer);

            // PHASE 4 PIPELINING: Check if FFN result is available
            // Try thread-based pipelining first (new approach)
            float * async_result = nullptr;
            int64_t async_ne0 = 0, async_ne1 = 0;
            size_t async_size = 0;

            if (g_ggml_sycl_tp_threaded_ffn && g_tp_device1_worker.initialized.load()) {
                // Check for thread-based result (wait with timeout)
                tp_ffn_result * thread_result = ggml_sycl_tp_get_ffn_result(layer, true);
                if (thread_result && thread_result->valid) {
                    async_result = thread_result->result_buf;
                    async_ne0 = thread_result->ne0;
                    async_ne1 = thread_result->ne1;
                    async_size = thread_result->result_size;
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
                queue_ptr main_stream = ctx.stream();
                // Use ggml_sycl_get_data_ptr for correct device-specific pointer in TP mode
                float * dst_ptr = (float *)ggml_sycl_get_data_ptr(dst, main_device);
                const int64_t dst_elements = ne01 * ne11;

                // GPU kernel adds async_result (in shared memory) to dst
                // IMPORTANT: Use submit() to isolate kernel lambda scope from enclosing context.
                main_stream->submit([&](sycl::handler& cgh) {
                    float* dst_local = dst_ptr;
                    float* async_local = async_result;
                    cgh.parallel_for(sycl::range<1>(dst_elements), [=](sycl::id<1> idx) {
                        dst_local[idx] += async_local[idx];
                    });
                }).wait();

                // DEBUG: Verify final result
                if (g_ggml_sycl_tp_debug && async_used_count <= 3) {
                    float final_sample[4];
                    main_stream->memcpy(final_sample, dst_ptr, 4*sizeof(float)).wait();
                    fprintf(stderr, "SYCL TP: ASYNC FFN layer %d FINAL[0..3]=[%f,%f,%f,%f]\n",
                            layer, final_sample[0], final_sample[1], final_sample[2], final_sample[3]);
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
                    auto it = g_tp_ffn_inputs.find(layer);
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
                queue_ptr main_stream = ctx.stream();
                struct { uint16_t d_bits; uint8_t qs[16]; } wblk;
                try {
                    main_stream->memcpy(&wblk, (void*)l31_weight_addr, sizeof(wblk)).wait();
                    uint16_t d_raw = wblk.d_bits;
                    sycl::half d_half;
                    memcpy(&d_half, &d_raw, sizeof(sycl::half));
                    float d_f = static_cast<float>(d_half);
                    fprintf(stderr, "TP DEBUG FFN_START layer 31: L31 weight d=%f %s\n",
                            d_f, (d_f > 100.0f || std::isnan(d_f)) ? "CORRUPTED" : "OK");
                } catch (...) {}
            }
            ffn_input_storage ffn_input = {};
            {
                std::lock_guard<std::mutex> lock(g_tp_ffn_input_mutex);
                auto it = g_tp_ffn_inputs.find(layer);
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
                if (g_ggml_sycl_tp_debug && ffn_found++ < 3) fprintf(stderr, "TP DEBUG: FFN input found for layer %d, data=%p\n", layer, ffn_input.data);
                // Get FFN weight references for this layer
                ffn_weight_refs weights = {};
                {
                    std::lock_guard<std::mutex> lock(g_tp_ffn_weight_mutex);
                    auto it = g_tp_ffn_weights.find(layer);
                    if (it != g_tp_ffn_weights.end()) {
                        weights = it->second;
                    }
                }

                // DEBUG: Check FFN weight lookup during decode (batch=1)
                static int ffn_weight_dbg = 0;
                if (g_ggml_sycl_tp_debug && ne11 == 1 && ffn_weight_dbg++ < 10) {
                    fprintf(stderr, "TP DEBUG ROW_PARALLEL decode FFN WEIGHTS: layer=%d gate=%p up=%p down=%p\n",
                            layer, (void*)weights.gate, (void*)weights.up, (void*)weights.down);
                }

                if (weights.gate && weights.up && weights.down) {
                    // Get device 1
                    int device = g_sycl_tp_config.devices[1];
                    ggml_sycl_set_device(device);
                    queue_ptr stream = ctx.stream(device, 0);

                    // Get weight shards for device 1
                    auto * gate_extra = static_cast<ggml_tensor_extra_gpu *>(weights.gate->extra);
                    auto * up_extra = static_cast<ggml_tensor_extra_gpu *>(weights.up->extra);
                    auto * down_extra = static_cast<ggml_tensor_extra_gpu *>(weights.down->extra);

                    void * gate_weight_1 = gate_extra ? gate_extra->data_device[device] : nullptr;
                    void * up_weight_1 = up_extra ? up_extra->data_device[device] : nullptr;
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
                            uint8_t block0[20], block1[20];  // Q4_0 block is 18 bytes
                            queue_ptr main_stream = ctx.stream();
                            queue_ptr dev1_stream = ctx.stream(device, 0);
                            main_stream->memcpy(block0, gate_weight_0, 20).wait();
                            dev1_stream->memcpy(block1, gate_weight_1, 20).wait();
                            fprintf(stderr, "TP DEBUG FFN layer %d gate[0:20]: dev0=[%02x,%02x,%02x,%02x...], dev1=[%02x,%02x,%02x,%02x...]\n",
                                    layer, block0[0], block0[1], block0[2], block0[3],
                                    block1[0], block1[1], block1[2], block1[3]);
                        }
                    }

                    if (!gate_weight_1 || !up_weight_1 || !down_weight_1) {
                        GGML_SYCL_DEBUG("SYCL TP: WARNING - missing weight shards on device 1 for layer %d\n", layer);
                    } else {
                        // Dimensions for FFN computation
                        // FFN input: [K, batch] where K = model dimension (full)
                        // Gate/Up weights: [K, N_hidden_shard] (column-parallel, output sharded)
                        // Down weight: [N_hidden_shard, N_out] (row-parallel, input sharded)
                        const int64_t K_full = ffn_input.ne0;  // Full model dimension
                        const int64_t batch = ffn_input.ne1;
                        const int64_t N_hidden_shard = gate_extra->tp_local_ne[1];  // Sharded hidden dim
                        const int64_t N_out = ne01;  // Output dimension (same as device 0)

                        // Calculate padded dimensions
                        const int64_t K_full_padded = GGML_PAD(K_full, MATRIX_ROW_PADDING);
                        const int64_t N_hidden_shard_padded = GGML_PAD(N_hidden_shard, MATRIX_ROW_PADDING);

                        // OPTIMIZATION: Use persistent buffers instead of malloc/free per call
                        // This eliminates ~535K memory allocations per 20-token inference
                        tp_ffn_compute_buffers * bufs = ggml_sycl_tp_ensure_ffn_buffers(
                            layer, device, stream,
                            K_full_padded, N_hidden_shard_padded, batch, N_out);

                        if (!bufs) {
                            fprintf(stderr, "SYCL TP: ERROR - failed to get FFN buffers on device %d for layer %d\n", device, layer);
                        } else {
                            // Use pre-allocated persistent buffers
                            char * input_q8_dev = bufs->input_q8_dev;
                            float * gate_out = bufs->gate_out;
                            float * up_out = bufs->up_out;
                            float * hidden_out = bufs->hidden_out;
                            char * hidden_q8_dev = bufs->hidden_q8_dev;
                            float * partial_out = bufs->partial_out;

                            // Calculate sizes for operations (using actual dimensions, not buffer capacity)
                            const size_t hidden_size = N_hidden_shard * batch * sizeof(float);
                            // DEBUG: Check FFN input values (always for batch=1 to debug NaN)
                            static int ffn_in_dbg = 0;
                            bool do_debug = (g_ggml_sycl_tp_debug && ffn_in_dbg++ < 3);  // Enable for first 3 to debug quant
                            if (do_debug) {
                                float in_sample[4];
                                stream->memcpy(in_sample, ffn_input.data, 4*sizeof(float)).wait();
                                fprintf(stderr, "TP DEBUG FFN input layer %d: input[0..3]=[%f,%f,%f,%f], K_full=%lld, batch=%lld\n",
                                        layer, in_sample[0], in_sample[1], in_sample[2], in_sample[3],
                                        (long long)K_full, (long long)batch);
                            }

                            // Step 1: Quantize FFN input to Q8_1
                            quantize_row_q8_1_sycl<quantize_q8_1>(
                                (const float *)ffn_input.data, input_q8_dev,
                                K_full, batch, K_full_padded, stream);
                            // Note: No wait needed - same queue operations are serialized

                            // Create fake tensors with correct output dimensions for gate/up
                            // MMVQ uses dst->ne[0] for output stride, so we need correct dims
                            ggml_tensor fake_dst_hidden = *dst;
                            fake_dst_hidden.ne[0] = N_hidden_shard;  // Output dimension for gate/up

                            // Steps 2-3: Gate and Up matmuls (run sequentially on same queue)
                            // Note: These could be parallelized with separate queues in Phase 4
                            stream->memset(gate_out, 0, hidden_size);
                            stream->memset(up_out, 0, hidden_size);
                            // Wait for memsets to complete before matmuls read from these buffers
                            stream->wait();

                            // Step 2: Gate matmul - input @ W_gate_1 -> gate_out
                            ggml_sycl_op_mul_mat_vec_q(ctx, weights.gate, src1, &fake_dst_hidden,
                                (const char *)gate_weight_1, nullptr, input_q8_dev,
                                gate_out, 0, N_hidden_shard, batch, K_full_padded, stream);

                            // Step 3: Up matmul - input @ W_up_1 -> up_out
                            ggml_sycl_op_mul_mat_vec_q(ctx, weights.up, src1, &fake_dst_hidden,
                                (const char *)up_weight_1, nullptr, input_q8_dev,
                                up_out, 0, N_hidden_shard, batch, K_full_padded, stream);
                            // Wait for both matmuls before SiLU
                            stream->wait();

                            // DEBUG: Check gate and up values and sums (disabled - TP working)
                            static int ffn_inter_dbg = 0;
                            bool debug_this = (ffn_inter_dbg++ < 0);  // Disabled
                            if (debug_this) {
                                float g_sample[4], u_sample[4];
                                stream->memcpy(g_sample, gate_out, 4*sizeof(float)).wait();
                                stream->memcpy(u_sample, up_out, 4*sizeof(float)).wait();
                                // Also compute sum of first 1024 elements
                                std::vector<float> gate_host(1024);
                                stream->memcpy(gate_host.data(), gate_out, 1024*sizeof(float)).wait();
                                float gate_sum = 0, gate_max = -1e10, gate_min = 1e10;
                                for (int i = 0; i < 1024; i++) {
                                    gate_sum += gate_host[i];
                                    gate_max = std::max(gate_max, gate_host[i]);
                                    gate_min = std::min(gate_min, gate_host[i]);
                                }
                                fprintf(stderr, "TP DEBUG FFN layer %d batch=%lld: gate[0..3]=[%f,%f,%f,%f], gate_sum=%f, range=[%f,%f]\n",
                                        layer, (long long)batch, g_sample[0], g_sample[1], g_sample[2], g_sample[3], gate_sum, gate_min, gate_max);
                                fprintf(stderr, "TP DEBUG FFN layer %d batch=%lld: up[0..3]=[%f,%f,%f,%f]\n",
                                        layer, (long long)batch, u_sample[0], u_sample[1], u_sample[2], u_sample[3]);
                            }

                            // Step 4: SiLU activation on gate_out and multiply by up_out
                            // hidden_out[i] = silu(gate_out[i]) * up_out[i]
                            const int64_t n_elements = N_hidden_shard * batch;
                            const int block_size = 256;
                            const int num_blocks = (n_elements + block_size - 1) / block_size;
                            stream->parallel_for(
                                sycl::nd_range<1>(num_blocks * block_size, block_size),
                                [=](sycl::nd_item<1> item) {
                                    const int i = item.get_global_id(0);
                                    if (i < n_elements) {
                                        float g = gate_out[i];
                                        float u = up_out[i];
                                        // SiLU: x * sigmoid(x)
                                        float silu_g = g / (1.0f + sycl::native::exp(-g));
                                        hidden_out[i] = silu_g * u;
                                    }
                                });
                            // No wait - quantization on same queue will serialize

                            // DEBUG: Check hidden_out values after SiLU (disabled - TP working)
                            static int hidden_dbg = 0;
                            bool debug_hidden = (hidden_dbg++ < 0);  // Disabled
                            if (debug_hidden) {
                                float h_sample[4];
                                stream->memcpy(h_sample, hidden_out, 4*sizeof(float)).wait();
                                fprintf(stderr, "TP DEBUG FFN layer %d batch=%lld: hidden[0..3]=[%f,%f,%f,%f], N_hidden_shard=%lld\n",
                                        layer, (long long)batch, h_sample[0], h_sample[1], h_sample[2], h_sample[3], (long long)N_hidden_shard);
                            }

                            // Step 5: Quantize hidden_out to Q8_1 for down matmul
                            // DEBUG: Check hidden sum before quantization (disabled)
                            static int quant_dbg = 0;
                            if (quant_dbg++ < 0) {
                                std::vector<float> hidden_host(1024);
                                stream->memcpy(hidden_host.data(), hidden_out, 1024*sizeof(float)).wait();
                                float hidden_sum = 0, hidden_max = -1e10, hidden_min = 1e10;
                                for (int i = 0; i < 1024; i++) {
                                    hidden_sum += hidden_host[i];
                                    hidden_max = std::max(hidden_max, hidden_host[i]);
                                    hidden_min = std::min(hidden_min, hidden_host[i]);
                                }
                                fprintf(stderr, "TP DEBUG FFN layer %d: hidden_sum=%f, hidden_range=[%f,%f]\n",
                                        layer, hidden_sum, hidden_min, hidden_max);
                            }

                            quantize_row_q8_1_sycl<quantize_q8_1>(
                                hidden_out, hidden_q8_dev,
                                N_hidden_shard, batch, N_hidden_shard_padded, stream);
                            // No wait - memset+matmul on same queue will serialize

                            // Step 6: Down matmul - hidden_out @ W_down_1 -> partial_out
                            // DEBUG: Check down matmul dimensions (disabled)
                            static int down_dbg = 0;
                            if (down_dbg++ < 0) {
                                fprintf(stderr, "TP DEBUG FFN DOWN layer %d: src0->ne=[%lld,%lld], N_out=%lld, N_hidden_shard=%lld, batch=%lld\n",
                                        layer, (long long)src0->ne[0], (long long)src0->ne[1],
                                        (long long)N_out, (long long)N_hidden_shard, (long long)batch);
                            }
                            stream->memset(partial_out, 0, dst_size);
                            // Wait for memset before matmul writes to same buffer
                            stream->wait();
                            ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
                                (const char *)down_weight_1, nullptr, hidden_q8_dev,
                                partial_out, 0, N_out, batch, N_hidden_shard_padded, stream);
                            // Wait before ALL_REDUCE reads partial_out
                            stream->wait();

                            // Step 7: ALL_REDUCE - add partial_out to dst on main device
                            // OPTIMIZED: Use malloc_shared buffer + GPU addition kernel
                            {
                                const size_t dst_elements = (size_t)(ne01 * ne11);

                                // Get shared buffer for ALL_REDUCE (accessible from both devices)
                                float * shared_buf = ggml_sycl_tp_ensure_shared_reduce_buffer(dst_size);

                                static int ffn_reduce_dbg = 0;
                                bool debug_reduce = (g_ggml_sycl_tp_debug && ffn_reduce_dbg++ < 150);

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
                                        fprintf(stderr, "TP DEBUG FFN ALL_REDUCE layer %d batch=%lld: dev1_partial[0..3]=[%f,%f,%f,%f]\n",
                                                layer, (long long)batch, shared_buf[0], shared_buf[1], shared_buf[2], shared_buf[3]);
                                    }

                                    // Step 7b: Switch to main device and add using GPU kernel
                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    // Use ggml_sycl_get_data_ptr for correct device-specific pointer in TP mode
                                    float * dst_ptr = (float *)ggml_sycl_get_data_ptr(dst, main_device);

                                    // DEBUG: Check device 0's partial result before add
                                    if (debug_reduce) {
                                        float dev0_sample[4];
                                        main_stream->memcpy(dev0_sample, dst_ptr, 4*sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG FFN layer %d batch=%lld: dev0_partial[0..3]=[%f,%f,%f,%f]\n",
                                                layer, (long long)batch, dev0_sample[0], dev0_sample[1], dev0_sample[2], dev0_sample[3]);
                                    }

                                    // Step 7c: GPU kernel adds shared_buf to dst (dst += shared_buf)
                                    // shared_buf is malloc_shared so device 0 can read it directly
                                    // Use parallel_for_work_group for better compatibility with Intel Arc
                                    const size_t work_group_size = 256;
                                    const size_t num_groups = (dst_elements + work_group_size - 1) / work_group_size;
                                    main_stream->parallel_for(
                                        sycl::nd_range<1>(num_groups * work_group_size, work_group_size),
                                        [=](sycl::nd_item<1> item) {
                                            size_t idx = item.get_global_id(0);
                                            if (idx < dst_elements) {
                                                dst_ptr[idx] += shared_buf[idx];
                                            }
                                        }
                                    ).wait();

                                    // DEBUG: Verify result
                                    if (debug_reduce && layer < 2) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4*sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG FFN layer %d: TOTAL[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                layer, total_sample[0], total_sample[1], total_sample[2], total_sample[3]);
                                    }
                                } else if (ggml_sycl_quant_allreduce_enabled()) {
                                    // QUANTIZED ALL_REDUCE: 50% bandwidth reduction
                                    // Uses GPU-side INT8 quantization to reduce transfer size
                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    float * dst_ptr = (float *)ggml_sycl_get_data_ptr(dst, main_device);

                                    quantized_allreduce(
                                        main_device, device,
                                        main_stream, stream,
                                        dst_ptr, partial_out,
                                        dst_elements,
                                        debug_reduce
                                    );

                                    if (debug_reduce) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4*sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG FFN layer %d (QUANT): TOTAL[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
                                                layer, total_sample[0], total_sample[1], total_sample[2], total_sample[3]);
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
                                    float * dst_ptr = (float *)ggml_sycl_get_data_ptr(dst, main_device);

                                    auto ev_dev0 = main_stream->memcpy(dev0_host, dst_ptr, dst_size);

                                    // Single sync point for both copies
                                    ev_dev1.wait();
                                    ev_dev0.wait();

                                    // DEBUG: Show partial results before addition
                                    if (debug_reduce) {
                                        fprintf(stderr, "TP DEBUG FFN layer %d batch=%lld: dev0[0..3]=[%.4f,%.4f,%.4f,%.4f] dev1[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
                                                layer, (long long)batch, dev0_host[0], dev0_host[1], dev0_host[2], dev0_host[3],
                                                host_buf[0], host_buf[1], host_buf[2], host_buf[3]);
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
                                queue_ptr main_stream = ctx.stream();
                                uintptr_t l31_weight_addr = 0xffffd5575d400000ULL;
                                struct { uint16_t d_bits; uint8_t qs[16]; } wblk;
                                try {
                                    main_stream->memcpy(&wblk, (void*)l31_weight_addr, sizeof(wblk)).wait();
                                    uint16_t d_raw = wblk.d_bits;
                                    sycl::half d_half;
                                    memcpy(&d_half, &d_raw, sizeof(sycl::half));
                                    float d_f = static_cast<float>(d_half);
                                    if (g_ggml_sycl_tp_debug && (d_f > 100.0f || std::isnan(d_f))) {
                                        fprintf(stderr, "TP DEBUG FFN_CLEANUP layer 31: L31 weight CORRUPTED d=%f\n", d_f);
                                    } else if (g_ggml_sycl_tp_debug) {
                                        static int cleanup_dbg = 0;
                                        if (cleanup_dbg++ < 3)
                                            fprintf(stderr, "TP DEBUG FFN_CLEANUP layer 31: L31 weight OK d=%f\n", d_f);
                                    }
                                } catch (...) {}
                            }
                        }
                    }

                    // Restore main device context
                    ggml_sycl_set_device(main_device);
                } else {
                    static int warn = 0;
                    if (warn++ < 3) {
                        fprintf(stderr, "SYCL TP: WARNING - missing FFN weight refs for layer %d (gate=%p, up=%p, down=%p)\n",
                                layer, (void*)weights.gate, (void*)weights.up, (void*)weights.down);
                    }
                }

                // Clear stored FFN input for this layer (no longer needed)
                {
                    std::lock_guard<std::mutex> lock(g_tp_ffn_input_mutex);
                    auto it = g_tp_ffn_inputs.find(layer);
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
            if (g_ggml_sycl_tp_debug && attn_dbg_count++ < 40) fprintf(stderr, "TP DEBUG: Attention output layer %d, entering computation path (batch=%lld)\n", layer, (long long)ne11);
            attn_input_storage attn_input = {};
            {
                std::lock_guard<std::mutex> lock(g_tp_attn_input_mutex);
                auto it = g_tp_attn_inputs.find(layer);
                if (it != g_tp_attn_inputs.end()) {
                    attn_input = it->second;
                }
            }

            // DEBUG: Track attention input lookup during decode (batch=1)
            static int attn_lookup_dbg = 0;
            if (g_ggml_sycl_tp_debug && ne11 == 1 && attn_lookup_dbg++ < 100) {
                fprintf(stderr, "TP DEBUG ATTN_OUTPUT decode: layer=%d attn_input.data=%p, map_size=%zu\n",
                        layer, attn_input.data, g_tp_attn_inputs.size());
            }

            if (attn_input.data != nullptr) {
                static int attn_found = 0;
                if (g_ggml_sycl_tp_debug && attn_found++ < 40) fprintf(stderr, "TP DEBUG: Attention input found for layer %d, data=%p\n", layer, attn_input.data);
                // Get attention weight references
                attn_weight_refs attn_weights = {};
                {
                    std::lock_guard<std::mutex> lock(g_tp_attn_weight_mutex);
                    auto it = g_tp_attn_weights.find(layer);
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
                                fprintf(stderr, "TP DEBUG: New prompt detected (layer=0, batch=%ld, cached=%ld), resetting KV cache\n",
                                        (long)attn_input.ne1, (long)cached_len);
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
                            fprintf(stderr, "SYCL TP: WARNING - missing attention weight shards on device 1 for layer %d\n", layer);
                        }
                    } else {
                        // DEBUG: Compare O weight values on both devices
                        static int o_weight_dbg = 0;
                        if (g_ggml_sycl_tp_debug && o_weight_dbg++ < 3) {
                            void * o_weight_0 = o_extra ? o_extra->data_device[main_device] : nullptr;
                            fprintf(stderr, "TP DEBUG ATTN layer %d: O_weight_0=%p, O_weight_1=%p\n",
                                    layer, o_weight_0, o_weight_1);
                            // Read first few Q4_0 blocks and dequantize to check values
                            if (o_weight_0 && o_weight_1) {
                                // Q4_0 block: 2-byte scale (float16) + 16 bytes of quantized values (32 values, 4 bits each)
                                uint8_t blocks0[36], blocks1[36];  // Read 2 blocks (18 bytes each)
                                queue_ptr main_stream = ctx.stream();
                                queue_ptr dev1_stream = ctx.stream(device, 0);
                                main_stream->memcpy(blocks0, o_weight_0, 36).wait();
                                dev1_stream->memcpy(blocks1, o_weight_1, 36).wait();

                                // Dequantize first block of each
                                uint16_t scale0_bits = blocks0[0] | (blocks0[1] << 8);
                                uint16_t scale1_bits = blocks1[0] | (blocks1[1] << 8);
                                // Convert float16 to float32 (simple approximation)
                                auto f16_to_f32 = [](uint16_t h) -> float {
                                    uint32_t sign = (h >> 15) & 1;
                                    int32_t exp = (h >> 10) & 0x1F;
                                    uint32_t mant = h & 0x3FF;
                                    if (exp == 0) {
                                        return sign ? -0.0f : 0.0f;  // Zero/subnormal
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
                                    return {(q0 - 8) * scale, (q1 - 8) * scale};
                                };
                                auto [v00, v01] = dequant(scale0, blocks0[2]);
                                auto [v10, v11] = dequant(scale1, blocks1[2]);

                                fprintf(stderr, "TP DEBUG ATTN layer %d O weight: dev0 scale=%f, vals=[%f,%f,...], dev1 scale=%f, vals=[%f,%f,...]\n",
                                        layer, scale0, v00, v01, scale1, v10, v11);
                            }
                        }
                        // Dimensions for attention computation
                        // Input: [n_embd, batch] where n_embd = model dimension (full)
                        // Q weight: [n_embd, n_embd_q_shard] (column-parallel, output sharded by heads)
                        // K weight: [n_embd, n_embd_k_shard]
                        // V weight: [n_embd, n_embd_v_shard]
                        // O weight: [n_embd_q_shard, n_embd] (row-parallel, input sharded)
                        const int64_t n_embd = attn_input.ne0;  // Full model dimension
                        const int64_t batch = attn_input.ne1;   // Sequence length * batch
                        const int64_t n_embd_q_shard = q_extra->tp_local_ne[1];  // Q sharded output dim
                        const int64_t n_embd_k_shard = k_extra->tp_local_ne[1];  // K sharded output dim
                        const int64_t n_embd_v_shard = v_extra->tp_local_ne[1];  // V sharded output dim
                        const int64_t N_out = ne01;  // Output dimension (n_embd)

                        // Allocate buffers on device 1
                        const size_t q8_1_ts = sizeof(block_q8_1);
                        const size_t q8_1_bs = QK8_1;
                        const int64_t n_embd_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);

                        // Input quantization buffer
                        const size_t input_q8_size = batch * n_embd_padded * q8_1_ts / q8_1_bs;
                        char * input_q8_dev = (char *)sycl::malloc_device(input_q8_size, *stream);

                        // Q, K, V output buffers (float)
                        const size_t q_out_size = n_embd_q_shard * batch * sizeof(float);
                        const size_t k_out_size = n_embd_k_shard * batch * sizeof(float);
                        const size_t v_out_size = n_embd_v_shard * batch * sizeof(float);
                        float * q_out = (float *)sycl::malloc_device(q_out_size, *stream);
                        float * k_out = (float *)sycl::malloc_device(k_out_size, *stream);
                        float * v_out = (float *)sycl::malloc_device(v_out_size, *stream);

                        // Attention output buffer (same size as Q since it's the per-head output)
                        float * attn_out = (float *)sycl::malloc_device(q_out_size, *stream);

                        // For O projection, need to quantize attn_out
                        const int64_t n_embd_q_shard_padded = GGML_PAD(n_embd_q_shard, MATRIX_ROW_PADDING);
                        const size_t attn_q8_size = batch * n_embd_q_shard_padded * q8_1_ts / q8_1_bs;
                        char * attn_q8_dev = (char *)sycl::malloc_device(attn_q8_size, *stream);

                        // Output buffer for O projection (partial result)
                        float * partial_out = (float *)sycl::malloc_device(dst_size, *stream);

                        if (!input_q8_dev || !q_out || !k_out || !v_out || !attn_out || !attn_q8_dev || !partial_out) {
                            fprintf(stderr, "SYCL TP: ERROR - failed to allocate attention buffers on device %d\n", device);
                            if (input_q8_dev) sycl::free(input_q8_dev, *stream);
                            if (q_out) sycl::free(q_out, *stream);
                            if (k_out) sycl::free(k_out, *stream);
                            if (v_out) sycl::free(v_out, *stream);
                            if (attn_out) sycl::free(attn_out, *stream);
                            if (attn_q8_dev) sycl::free(attn_q8_dev, *stream);
                            if (partial_out) sycl::free(partial_out, *stream);
                        } else {
                            // Step 1: Quantize attention input to Q8_1
                            quantize_row_q8_1_sycl<quantize_q8_1>(
                                (const float *)attn_input.data, input_q8_dev,
                                n_embd, batch, n_embd_padded, stream);
                            stream->wait();

                            // Create temporary tensors with correct dimensions for MMVQ
                            // MMVQ uses dst->ne[0] for output stride, so we need correct dimensions
                            ggml_tensor fake_src1 = *src1;
                            fake_src1.ne[0] = n_embd;  // Input dimension for Q/K/V projection

                            ggml_tensor fake_dst_q = *dst;
                            fake_dst_q.ne[0] = n_embd_q_shard;  // Q output dimension

                            ggml_tensor fake_dst_k = *dst;
                            fake_dst_k.ne[0] = n_embd_k_shard;  // K output dimension

                            ggml_tensor fake_dst_v = *dst;
                            fake_dst_v.ne[0] = n_embd_v_shard;  // V output dimension

                            // Step 2: Q projection - input @ W_q1 -> q_out
                            stream->memset(q_out, 0, q_out_size).wait();
                            ggml_sycl_op_mul_mat_vec_q(ctx, attn_weights.q, &fake_src1, &fake_dst_q,
                                (const char *)q_weight_1, nullptr, input_q8_dev,
                                q_out, 0, n_embd_q_shard, batch, n_embd_padded, stream);
                            stream->wait();

                            // DEBUG: Capture Q values before RoPE for comparison with multi-process
                            static int q_before_rope_dbg = 0;
                            if (g_ggml_sycl_tp_debug && layer == 0 && q_before_rope_dbg++ < 3) {
                                float q_sample[8];
                                stream->memcpy(q_sample, q_out, 8*sizeof(float)).wait();
                                int64_t rope_pos = get_dev1_kv_cache_seq_len(layer);
                                fprintf(stderr, "SP DEV1 L0: Q_before_rope[0..7]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f] rope_pos=%lld batch=%lld\n",
                                        q_sample[0], q_sample[1], q_sample[2], q_sample[3],
                                        q_sample[4], q_sample[5], q_sample[6], q_sample[7],
                                        (long long)rope_pos, (long long)batch);
                            }

                            // Step 3: K projection - input @ W_k1 -> k_out
                            stream->memset(k_out, 0, k_out_size).wait();
                            ggml_sycl_op_mul_mat_vec_q(ctx, attn_weights.k, &fake_src1, &fake_dst_k,
                                (const char *)k_weight_1, nullptr, input_q8_dev,
                                k_out, 0, n_embd_k_shard, batch, n_embd_padded, stream);
                            stream->wait();

                            // Step 4: V projection - input @ W_v1 -> v_out
                            stream->memset(v_out, 0, v_out_size).wait();
                            ggml_sycl_op_mul_mat_vec_q(ctx, attn_weights.v, &fake_src1, &fake_dst_v,
                                (const char *)v_weight_1, nullptr, input_q8_dev,
                                v_out, 0, n_embd_v_shard, batch, n_embd_padded, stream);
                            stream->wait();

                            // Step 4.5: Apply RoPE to Q and K
                            // Mistral uses NeoX-style RoPE with freq_base=10000.0, head_dim=128
                            const int64_t head_dim = 128;  // Mistral uses 128
                            const int64_t n_heads_q = n_embd_q_shard / head_dim;  // 16 on each device
                            const int64_t n_heads_kv = n_embd_k_shard / head_dim; // 4 on each device
                            const float freq_base = 10000.0f;
                            const float theta_scale = std::pow(freq_base, -2.0f / head_dim);

                            // Apply RoPE to Q (norm style: pairs adjacent elements)
                            // Q layout: [seq_len, n_heads_q * head_dim]
                            // NOTE: For token generation, position must account for cached sequence
                            int64_t q_cached_pos = get_dev1_kv_cache_seq_len(layer);
                            stream->parallel_for(
                                sycl::range<3>(n_heads_q, batch, head_dim / 2),
                                [=](sycl::id<3> idx) {
                                    const int64_t h = idx[0];      // Head index
                                    const int64_t pos = idx[1];    // Position within current batch
                                    const int64_t i0 = idx[2];     // Dimension pair index (0 to head_dim/2-1)

                                    // Norm style: pair adjacent elements i0*2 and i0*2+1
                                    const int64_t base_idx = pos * n_heads_q * head_dim + h * head_dim;
                                    // Absolute position in sequence = cached tokens + current position
                                    const int64_t abs_pos = q_cached_pos + pos;

                                    float theta = abs_pos * std::pow(theta_scale, static_cast<float>(i0));
                                    float cos_theta = sycl::cos(theta);
                                    float sin_theta = sycl::sin(theta);

                                    // Norm style: pairs (0,1), (2,3), (4,5), etc.
                                    float x0 = q_out[base_idx + i0 * 2];
                                    float x1 = q_out[base_idx + i0 * 2 + 1];

                                    q_out[base_idx + i0 * 2] = x0 * cos_theta - x1 * sin_theta;
                                    q_out[base_idx + i0 * 2 + 1] = x0 * sin_theta + x1 * cos_theta;
                                }).wait();

                            // DEBUG: Capture Q values AFTER manual RoPE for single-process comparison
                            static int sp_post_rope_dbg = 0;
                            if (g_ggml_sycl_tp_debug && layer == 0 && sp_post_rope_dbg++ < 3) {
                                float q_after[8];
                                stream->memcpy(q_after, q_out, 8*sizeof(float)).wait();
                                fprintf(stderr, "SP DEV1 Q_AFTER_ROPE layer 0: [%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f] abs_pos=%lld\n",
                                        q_after[0], q_after[1], q_after[2], q_after[3],
                                        q_after[4], q_after[5], q_after[6], q_after[7],
                                        (long long)q_cached_pos);
                            }

                            // Apply RoPE to K (norm style: pairs adjacent elements)
                            // K layout: [seq_len, n_heads_kv * head_dim]
                            // NOTE: For token generation (batch=1), the position needs to account for cached tokens
                            int64_t cached_pos = get_dev1_kv_cache_seq_len(layer);
                            stream->parallel_for(
                                sycl::range<3>(n_heads_kv, batch, head_dim / 2),
                                [=](sycl::id<3> idx) {
                                    const int64_t h = idx[0];      // Head index
                                    const int64_t pos = idx[1];    // Position within current batch
                                    const int64_t i0 = idx[2];     // Dimension pair index (0 to head_dim/2-1)

                                    const int64_t base_idx = pos * n_heads_kv * head_dim + h * head_dim;
                                    // Absolute position in sequence = cached tokens + current position
                                    const int64_t abs_pos = cached_pos + pos;

                                    float theta = abs_pos * std::pow(theta_scale, static_cast<float>(i0));
                                    float cos_theta = sycl::cos(theta);
                                    float sin_theta = sycl::sin(theta);

                                    // Norm style: pairs (0,1), (2,3), (4,5), etc.
                                    float x0 = k_out[base_idx + i0 * 2];
                                    float x1 = k_out[base_idx + i0 * 2 + 1];

                                    k_out[base_idx + i0 * 2] = x0 * cos_theta - x1 * sin_theta;
                                    k_out[base_idx + i0 * 2 + 1] = x0 * sin_theta + x1 * cos_theta;
                                }).wait();

                            // Step 4.6: Update KV cache with new K and V values (after RoPE)
                            // Initialize cache if needed (during first call / prompt processing)
                            init_dev1_kv_cache(layer, 4096, n_heads_kv, head_dim, stream);
                            // Append the new K and V values to cache
                            append_to_dev1_kv_cache(layer, k_out, v_out, batch, stream);

                            // Step 5: Multi-head attention computation WITH KV CACHE
                            // Q: [batch, n_heads_q * head_dim] - current queries (batch tokens)
                            // KV Cache: [full_seq_len, n_heads_kv * head_dim] - all previous + current tokens
                            // For GQA: gqa_ratio = n_heads_q / n_heads_kv Q heads share each KV head

                            const int64_t gqa_ratio = n_heads_q / n_heads_kv;     // 4
                            const int64_t n_query_tokens = batch;  // Number of new query tokens
                            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

                            // Get cached K/V for full sequence attention
                            float * k_cache = nullptr;
                            float * v_cache = nullptr;
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
                            float * attn_scores = (float *)sycl::malloc_device(scores_size, *stream);

                            if (attn_scores && k_cache && v_cache && kv_seq_len > 0) {
                                // Compute attention scores: Q @ K^T / sqrt(head_dim) with GQA
                                // Q comes from q_out (current batch), K comes from cache (full sequence)
                                stream->parallel_for(
                                    sycl::range<3>(n_heads_q, n_query_tokens, kv_seq_len),
                                    [=](sycl::id<3> idx) {
                                        const int64_t h = idx[0];       // Q head index
                                        const int64_t q_local = idx[1]; // Query local position (0..batch-1)
                                        const int64_t k_pos = idx[2];   // Key position (0..kv_seq_len-1)

                                        // Map Q head to KV head for GQA
                                        const int64_t kv_h = h / gqa_ratio;

                                        // Compute dot product Q_h[q_local] @ K_kv[k_pos]
                                        float score = 0.0f;
                                        for (int64_t d = 0; d < head_dim; d++) {
                                            // Q layout: [n_query_tokens, n_heads_q * head_dim]
                                            // K cache layout: [kv_seq_len, n_heads_kv * head_dim]
                                            const float q_val = q_out[q_local * n_heads_q * head_dim + h * head_dim + d];
                                            const float k_val = k_cache[k_pos * n_heads_kv * head_dim + kv_h * head_dim + d];
                                            score += q_val * k_val;
                                        }
                                        score *= scale;

                                        // Apply causal mask: query at absolute position (q_start_pos + q_local)
                                        // can only attend to key positions <= that absolute position
                                        const int64_t q_abs_pos = q_start_pos + q_local;
                                        if (k_pos > q_abs_pos) {
                                            score = -INFINITY;
                                        }

                                        attn_scores[h * n_query_tokens * kv_seq_len + q_local * kv_seq_len + k_pos] = score;
                                    }).wait();

                                // Softmax over key dimension for each (head, query_pos)
                                stream->parallel_for(
                                    sycl::range<2>(n_heads_q, n_query_tokens),
                                    [=](sycl::id<2> idx) {
                                        const int64_t h = idx[0];
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
                                            const float exp_val = sycl::exp(attn_scores[base + k] - max_val);
                                            attn_scores[base + k] = exp_val;
                                            sum += exp_val;
                                        }

                                        // Normalize
                                        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                                        for (int64_t k = 0; k < kv_seq_len; k++) {
                                            attn_scores[base + k] *= inv_sum;
                                        }
                                    }).wait();

                                // Compute attention output: attn_probs @ V
                                // attn_out layout: [n_query_tokens, n_heads_q * head_dim]
                                // V comes from cache (full sequence)
                                stream->parallel_for(
                                    sycl::range<3>(n_heads_q, n_query_tokens, head_dim),
                                    [=](sycl::id<3> idx) {
                                        const int64_t h = idx[0];       // Q head index
                                        const int64_t q_local = idx[1]; // Output position (local)
                                        const int64_t d = idx[2];       // Head dimension

                                        // Map Q head to KV head for GQA
                                        const int64_t kv_h = h / gqa_ratio;

                                        // Weighted sum of V values from cache
                                        float out_val = 0.0f;
                                        for (int64_t k_pos = 0; k_pos < kv_seq_len; k_pos++) {
                                            const float attn_weight = attn_scores[h * n_query_tokens * kv_seq_len + q_local * kv_seq_len + k_pos];
                                            const float v_val = v_cache[k_pos * n_heads_kv * head_dim + kv_h * head_dim + d];
                                            out_val += attn_weight * v_val;
                                        }

                                        attn_out[q_local * n_heads_q * head_dim + h * head_dim + d] = out_val;
                                    }).wait();

                                sycl::free(attn_scores, *stream);

                                // DEBUG: Check device 1's attention output
                                static int dev1_attn_dbg = 0;
                                if (g_ggml_sycl_tp_debug && dev1_attn_dbg++ < 3) {
                                    float attn_sample[4];
                                    stream->memcpy(attn_sample, attn_out, 4*sizeof(float)).wait();
                                    fprintf(stderr, "TP DEBUG ATTN dev1 attn_out[0..3] layer %d: [%f, %f, %f, %f]\n",
                                            layer, attn_sample[0], attn_sample[1], attn_sample[2], attn_sample[3]);
                                }
                            } else {
                                // Fallback: just copy Q output if allocation fails
                                fprintf(stderr, "SYCL TP: WARNING - attention scores allocation failed, using fallback\n");
                                stream->memcpy(attn_out, q_out, q_out_size).wait();
                            }

                            // Step 6: Quantize attention output for O projection
                            quantize_row_q8_1_sycl<quantize_q8_1>(
                                attn_out, attn_q8_dev,
                                n_embd_q_shard, batch, n_embd_q_shard_padded, stream);
                            stream->wait();

                            // Step 7: O projection - attn_out @ W_o1 -> partial_out
                            stream->memset(partial_out, 0, dst_size).wait();
                            ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
                                (const char *)o_weight_1, nullptr, attn_q8_dev,
                                partial_out, 0, N_out, batch, n_embd_q_shard_padded, stream);
                            stream->wait();

                            // Step 8: ALL_REDUCE - add partial_out to dst on main device
                            // OPTIMIZED: Use malloc_shared buffer + GPU addition kernel
                            {
                                const size_t dst_elements = (size_t)(ne01 * ne11);

                                // Get shared buffer for ALL_REDUCE (accessible from both devices)
                                float * shared_buf = ggml_sycl_tp_ensure_shared_reduce_buffer(dst_size);

                                static int attn_reduce_dbg = 0;
                                bool do_attn_dbg = (g_ggml_sycl_tp_debug && attn_reduce_dbg++ < 3);

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
                                        fprintf(stderr, "TP DEBUG ATTN ALL_REDUCE layer %d batch=%lld: partial_out[0..3]=[%f,%f,%f,%f]\n",
                                                layer, (long long)batch, shared_buf[0], shared_buf[1], shared_buf[2], shared_buf[3]);
                                    }

                                    // Step 8b: Switch to main device and add using GPU kernel
                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    // Use ggml_sycl_get_data_ptr for correct device-specific pointer in TP mode
                                    float * dst_ptr = (float *)ggml_sycl_get_data_ptr(dst, main_device);

                                    // DEBUG: Check device 0's partial result before add
                                    if (do_attn_dbg) {
                                        float dev0_sample[4];
                                        main_stream->memcpy(dev0_sample, dst_ptr, 4*sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG ATTN layer %d batch=%lld: dev0_out[0..3]=[%f,%f,%f,%f]\n",
                                                layer, (long long)batch, dev0_sample[0], dev0_sample[1], dev0_sample[2], dev0_sample[3]);
                                    }

                                    // Step 8c: GPU kernel adds shared_buf to dst (dst += shared_buf)
                                    // Use parallel_for with nd_range for better compatibility with Intel Arc
                                    const size_t work_group_size = 256;
                                    const size_t num_groups = (dst_elements + work_group_size - 1) / work_group_size;
                                    main_stream->parallel_for(
                                        sycl::nd_range<1>(num_groups * work_group_size, work_group_size),
                                        [=](sycl::nd_item<1> item) {
                                            size_t idx = item.get_global_id(0);
                                            if (idx < dst_elements) {
                                                dst_ptr[idx] += shared_buf[idx];
                                            }
                                        }
                                    ).wait();

                                    // DEBUG: Verify result
                                    if (do_attn_dbg) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4*sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG ATTN after add: dst[0..3]=[%f,%f,%f,%f]\n",
                                                total_sample[0], total_sample[1], total_sample[2], total_sample[3]);
                                    }
                                } else if (ggml_sycl_quant_allreduce_enabled()) {
                                    // QUANTIZED ALL_REDUCE: 50% bandwidth reduction
                                    ggml_sycl_set_device(main_device);
                                    queue_ptr main_stream = ctx.stream();
                                    float * dst_ptr = (float *)ggml_sycl_get_data_ptr(dst, main_device);

                                    quantized_allreduce(
                                        main_device, device,
                                        main_stream, stream,
                                        dst_ptr, partial_out,
                                        dst_elements,
                                        do_attn_dbg
                                    );

                                    // DEBUG: Always check first few ATTN quantized outputs
                                    static int quant_attn_dbg = 0;
                                    if (quant_attn_dbg++ < 6) {
                                        float total_sample[4];
                                        main_stream->memcpy(total_sample, dst_ptr, 4*sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG ATTN (QUANT) layer %d: dst_tensor=%p dst='%s' dst_data=%p dst_ptr=%p dst[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                layer, (void*)dst, dst->name ? dst->name : "(null)", dst->data, (void*)dst_ptr, total_sample[0], total_sample[1], total_sample[2], total_sample[3]);
                                        // Also check tensor->data
                                        if (dst->data != dst_ptr) {
                                            float tensor_sample[4];
                                            main_stream->memcpy(tensor_sample, dst->data, 4*sizeof(float)).wait();
                                            fprintf(stderr, "TP DEBUG ATTN (QUANT) layer %d: dst->data=%p tensor_data[0..3]=[%.6f,%.6f,%.6f,%.6f] MISMATCH!\n",
                                                    layer, dst->data, tensor_sample[0], tensor_sample[1], tensor_sample[2], tensor_sample[3]);
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
                                    float * dst_ptr = (float *)ggml_sycl_get_data_ptr(dst, main_device);

                                    auto ev_dev0 = main_stream->memcpy(dev0_host, dst_ptr, dst_size);

                                    // Single sync point for both copies
                                    ev_dev1.wait();
                                    ev_dev0.wait();

                                    // DEBUG: Show partial results from both devices
                                    if (do_attn_dbg) {
                                        fprintf(stderr, "TP DEBUG ATTN NON-QUANT: dev0[0..3]=[%.6f,%.6f,%.6f,%.6f] dev1[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                dev0_host[0], dev0_host[1], dev0_host[2], dev0_host[3],
                                                host_buf[0], host_buf[1], host_buf[2], host_buf[3]);
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
                                        main_stream->memcpy(total_sample, dst_ptr, 4*sizeof(float)).wait();
                                        fprintf(stderr, "TP DEBUG ATTN (NON-QUANT) layer %d: dst_tensor=%p dst='%s' dst_data=%p dst_ptr=%p dst[0..3]=[%.6f,%.6f,%.6f,%.6f]\n",
                                                layer, (void*)dst, dst->name ? dst->name : "(null)", dst->data, (void*)dst_ptr, total_sample[0], total_sample[1], total_sample[2], total_sample[3]);
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
                    auto it = g_tp_attn_inputs.find(layer);
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
    const int64_t K_full = ne10;
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
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
            fprintf(stderr, "SYCL TP: ERROR - src1 slice out of bounds for rank %d (offset=%ld, K_shard=%ld, K_full=%ld)\n",
                    rank, (long)src1_k_offset, (long)K_shard, (long)K_full);
            continue;
        }

        // src1 float slice size per batch element
        const size_t src1_float_slice_size = ne11 * K_shard * sizeof(float);
        const size_t src1_q8_size = ne11 * K_shard_padded * q8_1_ts / q8_1_bs;

        ggml_sycl_set_device(device);
        queue_ptr stream = ctx.stream(device, 0);

        // Allocate buffers on target device
        float * src1_ddf_dev = (float *)sycl::malloc_device(src1_float_slice_size, *stream);
        char * src1_ddq_dev = (char *)sycl::malloc_device(src1_q8_size, *stream);
        float * partial_out = (float *)sycl::malloc_device(dst_size, *stream);

        if (!src1_ddf_dev || !src1_ddq_dev || !partial_out) {
            fprintf(stderr, "SYCL TP: ERROR - failed to allocate temp buffers on device %d\n", device);
            if (src1_ddf_dev) sycl::free(src1_ddf_dev, *stream);
            if (src1_ddq_dev) sycl::free(src1_ddq_dev, *stream);
            if (partial_out) sycl::free(partial_out, *stream);
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

            const float * src1_data = (const float *)src1->data;
            for (int64_t b = 0; b < ne11; b++) {
                // Source: src1 at [src1_k_offset, b]
                const float * src_ptr = src1_data + b * K_full + src1_k_offset;
                float * dst_ptr = host_buf + b * K_shard;
                main_stream->memcpy(dst_ptr, src_ptr, K_shard * sizeof(float)).wait();
            }

            // Copy to target device
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);
            stream->memcpy(src1_ddf_dev, host_buf, src1_float_slice_size).wait();

            // Don't free - using persistent staging buffer
        }

        // Quantize src1 float to Q8_1 on target device
        {
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);

            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_dev, src1_ddq_dev,
                                                  K_shard, ne11, K_shard_padded, stream);
            stream->wait();
        }

        // Call MMVQ kernel on this device
        {
            ggml_sycl_set_device(device);
            stream = ctx.stream(device, 0);

            // Zero output first
            stream->memset(partial_out, 0, dst_size).wait();

            // Call MMVQ kernel
            ggml_sycl_op_mul_mat_vec_q(ctx, src0, src1, dst,
                                        (const char *)weight_shard,  // src0_dd_i
                                        nullptr,                      // src1_ddf_i (not needed)
                                        src1_ddq_dev,                 // src1_ddq_i
                                        partial_out,                  // dst_dd_i
                                        0,                            // row_low
                                        ne01,                         // row_high
                                        ne11,                         // src1_ncols
                                        K_shard_padded,               // src1_padded_row_size
                                        stream);
            stream->wait();
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
            float * temp_add = (float *)sycl::malloc_device(dst_size, *main_stream);
            main_stream->memcpy(temp_add, host_buf, dst_size).wait();

            ggml_sycl_add_f32((float *)dst->data, temp_add, dst_nelems, main_stream);
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

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1,
                                     ggml_tensor *dst) try {
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

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);
    const int64_t channel_stride_y = nb11 / sizeof(float);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half * src0_as_f16, const sycl::half * src1_as_f16, void * dst,
                                   const void ** ptrs_src, void ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, const sycl::nd_item<3> & item_ct1) {
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

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) try {
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

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    if (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

        // iterate tensor dims and find the slowest moving dim and stride
        int last_dim=0;
        int last_str=0;
        size_t largest_str=0;
        for(int i = 0; i< 4; i++){
            // last stride is always the largest
            if(src1->nb[i] == largest_str){
                if(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            if(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
#if GGML_SYCL_DNNL
        // oneDNN handles strided data and does not need overhead of get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# else
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

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f32;
    const void * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_disable_dnn) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](const sycl::half *src0,
                                                const sycl::half *src1, float *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                bool supported_broadcast = batchb == batcha ? true
                        : batchb == 1 || batcha == 1        ? true
                                                            : false;
                if (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<float>(), queue, batcha, batchb);
                } else {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    if (batches0 > batches1) {
                        int64_t num_mul_mats = batches1;
                        int64_t sub_batch = batches0 / num_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        for (int64_t i0 = 0; i0 < num_mul_mats; i0++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            const sycl::half *src1_shifted = src1 + (sb2 * i0);
                            float *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, sub_batch, 1);
                        }
                    } else {
                        int64_t num_mul_mats = batches0;
                        int64_t sub_batch = batches1 / num_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        for (int64_t i1 = 0; i1 < num_mul_mats; i1++) {
                            const sycl::half *src0_shifted = src0 + (sa2 * i1);
                            const sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            float *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(),
                                    queue, 1, sub_batch);
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
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / sizeof(float));
            } else if (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / sizeof(float));
            } else {
                for (int64_t b_a = 0; b_a < ne03; b_a++) {
                    const sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    const sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    float *dst_shifted = dst_ddf + (nb3 * b_a / sizeof(float));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / sizeof(float));
                }
            }

    }
    else
#endif
    {
        if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            const int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::math::transpose::trans,
                                                        oneapi::math::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
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
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, ne01, ne11, ne10, alpha,
                (const void **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (const void **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (void **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

enum class mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
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

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
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

static void reorder_qw_q4_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q4_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q4_0) == 0));
    int offset_blks = offset / sizeof(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q4_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q4_0* x = (const block_q4_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK4_0/2; j ++)
            {
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

static void reorder_qw(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

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
        default:
            GGML_ABORT("reorder_qw() called with unsupported type");
            break;
    }
}

static bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    return !g_ggml_sycl_disable_optimize && //allow optimize, controlled by $GGML_SYCL_DISABLE_OPT
            ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
            dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
            dst->src[1]->ne[1]==1 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

// Check if a specific tensor needs reordering (not yet reordered)
static bool tensor_needs_reorder(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    if (!should_reorder_tensor(ctx, dst)) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    if (!src0 || !src0->extra) {
        return false;
    }
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    // If not yet reordered and supported type, reordering will happen
    if (!extra->optimized_feature.reorder) {
        // Check if type supports reorder
        if (ggml_sycl_supports_reorder_mmvq(src0->type) ||
            ggml_sycl_supports_reorder_dmmv(src0->type) ||
            ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
            return true;
        }
    }
    return false;
}

// Check if any MUL_MAT weight tensor needs reordering (used to decide if pre-reorder is needed)
// NOTE: We don't check activation dimensions (ne[1]) because we want to pre-reorder ALL weights
// during prompt phase, so decode phase graphs don't get blocked.
static bool graph_needs_reorder(ggml_backend_sycl_context& ctx, ggml_cgraph * cgraph) {
    // Skip if optimize is disabled globally
    if (g_ggml_sycl_disable_optimize || !ctx.opt_feature.reorder) {
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        const ggml_tensor * src0 = node->src[0];
        if (!src0 || !src0->extra) {
            continue;
        }
        const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
        // Check if weight tensor is NOT yet reordered and supports reorder
        if (!extra->optimized_feature.reorder &&
            extra->tp_type == tp_layer_type::TP_NONE &&
            (ggml_sycl_supports_reorder_mmvq(src0->type) ||
             ggml_sycl_supports_reorder_dmmv(src0->type) ||
             ggml_sycl_supports_reorder_mul_mat_sycl(src0->type))) {
            if (g_ggml_sycl_debug) {
                fprintf(stderr, "[SYCL-GRAPH] needs_reorder: node %d '%s' src0='%s' type=%s\n",
                        i, node->name ? node->name : "?",
                        src0->name ? src0->name : "?",
                        ggml_type_name(src0->type));
            }
            return true;
        }
    }
    return false;
}

// Pre-reorder ALL eligible weight tensors in the graph
// This ensures subsequent graph recordings don't get blocked by incremental reordering
// NOTE: We don't check should_reorder_tensor() because that requires ne[1]==1 (decode mode).
// We want to pre-reorder during prompt phase (ne[1]>1) to avoid blocking decode phase graphs.
static void graph_pre_reorder_all(ggml_backend_sycl_context& ctx, ggml_cgraph * cgraph) {
    // Skip if optimize is disabled globally
    if (g_ggml_sycl_disable_optimize || !ctx.opt_feature.reorder) {
        return;
    }

    int reorder_count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        const ggml_tensor * src0 = node->src[0];
        if (!src0 || !src0->extra) {
            continue;
        }
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra->optimized_feature.reorder) {
            continue;  // Already reordered
        }
        // Skip TP-sharded tensors
        if (extra->tp_type != tp_layer_type::TP_NONE) {
            continue;
        }
        // Check if type supports reorder
        if (!ggml_sycl_supports_reorder_mmvq(src0->type) &&
            !ggml_sycl_supports_reorder_dmmv(src0->type) &&
            !ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
            continue;
        }
        // Perform reorder
        reorder_qw(src0, ctx.stream());
        extra->optimized_feature.reorder = true;
        reorder_count++;
    }
    if (reorder_count > 0) {
        // Wait for all reorders to complete before proceeding
        ctx.stream()->wait();
        GGML_SYCL_DEBUG("[SYCL-GRAPH] pre-reordered %d tensors\n", reorder_count);
    }
}

static void opt_for_reorder(ggml_backend_sycl_context * ctx, const ggml_tensor * src0, const ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    if (!should_reorder_tensor(*ctx, dst)) {
        return;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;  // Skip permutations and already reordered tensors
    }

    // CRITICAL: Skip reorder for TP-sharded tensors - reorder corrupts memory in TP mode!
    // The reorder operation uses src0->data and ggml_nbytes which may not match the sharded layout
    if (extra->tp_type != tp_layer_type::TP_NONE) {
        static int tp_skip_log = 0;
        if (g_ggml_sycl_tp_debug && tp_skip_log++ < 5) {
            fprintf(stderr, "TP DEBUG: Skipping reorder for TP-sharded tensor %s (tp_type=%d)\n",
                    src0->name ? src0->name : "?", (int)extra->tp_type);
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
    }

    reorder_qw(src0, ctx->stream());
    extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
}

// Pre-reorder all weight tensors that would be reordered during decode.
// This ensures consistent behavior from the first decode token and fixes
// non-determinism when llama graph reuse is enabled.
// Without this, the first decode token uses non-reordered kernels while
// subsequent tokens use reordered kernels, causing different results.
static void pre_reorder_all_tensors(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    if (g_ggml_sycl_disable_optimize || !sycl_ctx->opt_feature.reorder) {
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

        if (extra->optimized_feature.reorder) {
            continue;  // Already reordered
        }

        // Check if this type supports reordering (any algorithm)
        if (!ggml_sycl_supports_reorder_mmvq(src0->type) &&
            !ggml_sycl_supports_reorder_dmmv(src0->type) &&
            !ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
            continue;
        }

        // Reorder the weight tensor
        reorder_qw(src0, sycl_ctx->stream());
        extra->optimized_feature.reorder = true;
        reordered_count++;
    }

    if (reordered_count > 0) {
        // Wait for all reordering to complete before proceeding
        sycl_ctx->stream()->wait();
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

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // DEBUG: Check if TP sharded weights have correct dimensions
    if (is_tp_sharded_tensor(src0)) {
        static int tp_mm_dbg = 0;
        if (g_ggml_sycl_tp_debug && tp_mm_dbg++ < 10) {
            const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
            const char * name = src0->name ? src0->name : "?";
            bool is_attn_out = strstr(name, "attn_output") != nullptr;
            fprintf(stderr, "TP DEBUG MUL_MAT %s: src0->ne=[%lld,%lld], tp_local_ne=[%lld,%lld], src1->ne=[%lld,%lld,%lld,%lld], dst->ne=[%lld,%lld], device=%d\n",
                    name,
                    (long long)src0->ne[0], (long long)src0->ne[1],
                    (long long)extra->tp_local_ne[0], (long long)extra->tp_local_ne[1],
                    (long long)src1->ne[0], (long long)src1->ne[1], (long long)src1->ne[2], (long long)src1->ne[3],
                    (long long)dst->ne[0], (long long)dst->ne[1],
                    ctx.device);
            // For attn_output, also check src1 data values
            if (is_attn_out && ctx.device == 0) {
                queue_ptr stream = ctx.stream();
                float sample[4];
                stream->memcpy(sample, src1->data, 4*sizeof(float)).wait();
                fprintf(stderr, "TP DEBUG ATTN_OUT src1[0..3] = [%f, %f, %f, %f]\n",
                        sample[0], sample[1], sample[2], sample[3]);
            }
        }
    }

    // DEBUG: Check output.weight (lm_head) computation specifically
    // Make sure we match "output.weight" but NOT "attn_output.weight"
    // Debug controlled by GGML_SYCL_TP_DEBUG environment variable
    bool is_output_weight = src0->name &&
                            strstr(src0->name, "output.weight") != nullptr &&
                            strstr(src0->name, "attn_output") == nullptr;
    static int output_dbg = 0;
    if (g_ggml_sycl_tp_debug && is_output_weight && output_dbg++ < 3) {
        const char * name = src0->name ? src0->name : "?";
        queue_ptr stream = ctx.stream();
        int64_t batch = src1->ne[1];
        fprintf(stderr, "TP DEBUG LM_HEAD %s: src0->ne=[%lld,%lld], src1->ne=[%lld,%lld], dst->ne=[%lld,%lld], batch=%lld, device=%d\n",
                name,
                (long long)src0->ne[0], (long long)src0->ne[1],
                (long long)src1->ne[0], (long long)src1->ne[1],
                (long long)dst->ne[0], (long long)dst->ne[1],
                (long long)batch, ctx.device);
        // Check input (hidden state after output_norm)
        float input_sample[8];
        stream->memcpy(input_sample, src1->data, 8*sizeof(float)).wait();
        float sum = 0;
        bool has_nan = false;
        for (int i = 0; i < 8; i++) {
            sum += input_sample[i];
            if (std::isnan(input_sample[i])) has_nan = true;
        }
        fprintf(stderr, "TP DEBUG LM_HEAD batch=%lld input [0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] sum=%.4f nan=%d\n",
                (long long)batch, input_sample[0], input_sample[1], input_sample[2], input_sample[3],
                input_sample[4], input_sample[5], input_sample[6], input_sample[7],
                sum, has_nan);

        // For batch>1 (prompt processing), also check position 1's hidden state (this determines next token)
        if (batch > 1) {
            float pos1_sample[8];
            size_t offset = src1->ne[0] * sizeof(float);  // Skip to position 1
            stream->memcpy(pos1_sample, (char*)src1->data + offset, 8*sizeof(float)).wait();
            float sum1 = 0;
            bool has_nan1 = false;
            for (int i = 0; i < 8; i++) {
                sum1 += pos1_sample[i];
                if (std::isnan(pos1_sample[i])) has_nan1 = true;
            }
            fprintf(stderr, "TP DEBUG LM_HEAD batch=%lld pos=1 [0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f] sum=%.4f nan=%d\n",
                    (long long)batch, pos1_sample[0], pos1_sample[1], pos1_sample[2], pos1_sample[3],
                    pos1_sample[4], pos1_sample[5], pos1_sample[6], pos1_sample[7],
                    sum1, has_nan1);
        }
    }

    // Check for TP-sharded weight tensor and log the operation
    ggml_sycl_mul_mat_tp_pre(src0);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

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

    bool use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // XMX GEMM path
    bool use_xmx_gemm = g_ggml_sycl_use_xmx_gemm ? true : false;
    if (use_xmx_gemm) {
        use_xmx_gemm = ggml_sycl_xmx_available() && ggml_sycl_xmx_supports_type(src0->type);
    }
    if (use_xmx_gemm) {
        int64_t batch = src1->ne[1];
        // XMX is beneficial for batch >= 8 and < threshold
        use_xmx_gemm = batch >= 8 && batch < g_ggml_sycl_xmx_threshold;
    }


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggerganov/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // SYCL_USE_XMX

    // mmvq path is faster in the CUDA backend.
    if (!g_ggml_sycl_prioritize_dmmv && (ctx.stream()->get_backend() == sycl::backend::ext_oneapi_cuda
        // Dispatch becomes obscure with the reorder, MMVQ when the reorder optimization
        // is enabled takes precedence over DMMV, the current if-else implementation
        // requires disabling DMMV if both conditions are met
        || (should_reorder_tensor(ctx, dst) && ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
    }

    // DEBUG: Log path selection for FFN layers (only in TP mode)
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    static int path_dbg[32] = {0};
    static int weight_check_dbg = 0;
    const char * mm_name = src0->name ? src0->name : "?";
    // Skip this debug block in multi-process mode (only one device per process)
    if (g_ggml_sycl_tp_debug && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1 &&
        !g_sycl_tp_config.is_multiprocess) {
        for (int l = 30; l <= 31; l++) {
            char gate_name[64];
            snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gate", l);
            if (strstr(mm_name, gate_name) && src1->ne[1] == 1 && path_dbg[l]++ < 2) {
                fprintf(stderr, "TP DEBUG PATH L%d FFN_GATE: use_dmmv=%d use_mmvq=%d use_mmq=%d use_xmx=%d\n",
                        l, use_dequantize_mul_mat_vec, use_mul_mat_vec_q, use_mul_mat_q, use_xmx_gemm);

                // DEBUG: Check weight data at START of mul_mat before any kernels
                if (l == 31 && weight_check_dbg++ < 3) {
                    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
                    if (extra) {
                        struct { sycl::half d; uint8_t qs[16]; } wblk;
                        for (int dev = 0; dev < g_sycl_tp_config.world_size; dev++) {
                            int dev_id = g_sycl_tp_config.devices[dev];
                            if (extra->data_device[dev_id]) {
                                ggml_sycl_set_device(dev_id);
                                queue_ptr dev_stream = ctx.stream(dev_id, 0);
                                dev_stream->memcpy(&wblk, extra->data_device[dev_id], sizeof(wblk)).wait();
                                float d_f = (float)wblk.d;
                                int v0 = (wblk.qs[0] & 0xF) - 8;
                                int v1 = (wblk.qs[0] >> 4) - 8;
                                fprintf(stderr, "TP DEBUG L31 WEIGHT_CHECK device=%d: ptr=%p d=%f qs[0]=0x%02x deq=[%f,%f]\n",
                                        dev_id, extra->data_device[dev_id], d_f, wblk.qs[0], v0*d_f, v1*d_f);
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
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } else {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } else if (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } else {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } else if (use_xmx_gemm) {
        // XMX-accelerated quantized GEMM
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q_xmx);
    } else if (use_mul_mat_q) {
        // Standard MMQ path
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }

    // DEBUG: Check FFN gate/up output for NaN (layers 0, 30, 31)
    const char * weight_name = src0->name ? src0->name : "?";
    static int gate_dbg[32] = {0};
    static int call_seq = 0;

    // Check which layer this is
    int layer = -1;
    bool is_gate = false;
    for (int l = 0; l <= 31; l++) {
        char gate_name[64], up_name[64];
        snprintf(gate_name, sizeof(gate_name), "blk.%d.ffn_gate", l);
        snprintf(up_name, sizeof(up_name), "blk.%d.ffn_up", l);
        if (strstr(weight_name, gate_name)) { layer = l; is_gate = true; break; }
        if (strstr(weight_name, up_name)) { layer = l; is_gate = false; break; }
    }

    // Debug layers 0, 30, 31 (for batch=1 and batch=2 at layer 31)
    bool batch1_debug = g_ggml_sycl_tp_debug && layer >= 0 && dst->ne[1] == 1 &&
                        (layer == 0 || layer == 30 || layer == 31) &&
                        gate_dbg[layer]++ < 3;
    static int l31_batch2 = 0;
    bool batch2_debug = g_ggml_sycl_tp_debug && layer == 31 && dst->ne[1] == 2 && l31_batch2++ < 3;
    bool should_debug = batch1_debug || batch2_debug;
    if (should_debug) {
        call_seq++;
        ctx.stream()->wait();
        float out_vals[8];
        ctx.stream()->memcpy(out_vals, dst->data, 8*sizeof(float)).wait();
        bool has_nan = false;
        for (int i = 0; i < 8; i++) if (std::isnan(out_vals[i])) has_nan = true;
        // Also check input (src1)
        float in_vals[4];
        ctx.stream()->memcpy(in_vals, src1->data, 4*sizeof(float)).wait();
        bool in_nan = std::isnan(in_vals[0]) || std::isnan(in_vals[1]);
        int64_t batch = dst->ne[1];
        // Check weight pointer and first few bytes
        bool is_tp_buf = ggml_backend_buffer_is_sycl_tp(src0->buffer);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        void * weight_ptr = extra ? extra->data_device[0] : src0->data;
        fprintf(stderr, "TP DEBUG #%d L%d %s b=%lld: in=[%.3f,%.3f,%.3f,%.3f] out=[%.3f,%.3f,...] in_nan=%d out_nan=%d\n",
                call_seq, layer, is_gate ? "GATE" : "UP", (long long)batch,
                in_vals[0], in_vals[1], in_vals[2], in_vals[3],
                out_vals[0], out_vals[1], in_nan, has_nan);
        fprintf(stderr, "TP DEBUG #%d L%d weight: ptr=%p ne=[%lld,%lld] is_tp=%d\n",
                call_seq, layer, weight_ptr, (long long)src0->ne[0], (long long)src0->ne[1], is_tp_buf);
    }

    // DEBUG: Capture FFN down output BEFORE TP post-processing (works for single GPU too)
    static int ffn_mm_dbg = 0;
    bool is_ffn_down_l0 = strstr(weight_name, "blk.0.ffn_down") != nullptr;
    // Also detect by dimensions: FFN down has shape [14336 or 7168, 4096] - K=hidden, N=model_dim
    bool is_ffn_down_by_shape = (src0->ne[0] == 14336 || src0->ne[0] == 7168) && src0->ne[1] == 4096;
    if (g_ggml_sycl_tp_debug && ffn_mm_dbg++ < 20 && (is_ffn_down_l0 || (is_ffn_down_by_shape && ffn_mm_dbg < 3))) {
        ctx.stream()->wait();
        float out_vals[8];
        ctx.stream()->memcpy(out_vals, dst->data, 8*sizeof(float)).wait();
        fprintf(stderr, "DEBUG FFN_DOWN_PRE_TP %s device=%d ne=[%lldx%lld] src0=[%lldx%lld] dst[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                weight_name, ctx.device, (long long)dst->ne[0], (long long)dst->ne[1],
                (long long)src0->ne[0], (long long)src0->ne[1],
                out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
        // Also check FFN input (src1) for NaN
        if (dst->ne[1] == 1) {  // Token generation
            float src1_vals[8];
            ctx.stream()->memcpy(src1_vals, src1->data, 8*sizeof(float)).wait();
            fprintf(stderr, "DEBUG FFN_DOWN_INPUT batch=1 src1[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                    src1_vals[0], src1_vals[1], src1_vals[2], src1_vals[3],
                    src1_vals[4], src1_vals[5], src1_vals[6], src1_vals[7]);
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
        ctx.stream()->memcpy(out_vals, dst->data, 8*sizeof(float)).wait();
        fprintf(stderr, "DEBUG FFN_DOWN_POST_TP %s device=%d ne=[%lldx%lld] dst[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                weight_name, ctx.device, (long long)dst->ne[0], (long long)dst->ne[1],
                out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
    }

    // Debug sync point to catch errors early in TP mode
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1) {
        try {
            ctx.stream()->wait();
            GGML_SYCL_DEBUG("[MUL_MAT] Kernel completed successfully on device %d\n", ctx.device);
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[MUL_MAT] Kernel FAILED on device %d: %s (code=%d)\n",
                          ctx.device, e.what(), static_cast<int>(e.code().value()));
        }
    }
}


struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

__dpct_inline__ static void k_copy_src1_to_contiguous(
    const char *__restrict__ src1_original, char *__restrict__ src1_contiguous,
    int *__restrict__ cur_src1_row, mmid_row_mapping *__restrict__ row_mapping,
    const char *__restrict ids, int64_t i02, size_t ids_nb1, size_t ids_nb0,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    const sycl::nd_item<3> &item_ct1, int &src1_row) {
    int32_t iid1 = item_ct1.get_group(2);
    int32_t id = item_ct1.get_group(1);

    const int32_t row_id_i = *(const int32_t *) (ids + iid1*ids_nb1 + id*ids_nb0);

    if (row_id_i != i02) {
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    if (item_ct1.get_local_id(2) == 0) {
        src1_row =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                cur_src1_row, 1);
        row_mapping[src1_row] = {id, iid1};
    }
    /*
    DPCT1065:194: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    const float * src1_row_original = (const float *)(src1_original + i11*nb11 + i12*nb12);
    float * src1_row_contiguous = (float *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(
    char *__restrict__ dst_original, const char *__restrict__ dst_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Debug helper for MoE comparison
static bool g_moe_debug_enabled = false;
static int g_moe_debug_call_count = 0;

static void init_moe_debug() {
    static bool initialized = false;
    if (!initialized) {
        g_moe_debug_enabled = (getenv("GGML_SYCL_MOE_DEBUG") != nullptr);
        initialized = true;
    }
}

// Try fused MoE ESIMD kernel for batched prefill (ne12 > 1)
// Returns true if handled, false to fall back to other implementations
static bool ggml_sycl_mul_mat_id_fused(ggml_backend_sycl_context & ctx,
                                        const ggml_tensor *src0,
                                        const ggml_tensor *src1,
                                        const ggml_tensor *ids,
                                        ggml_tensor *dst) {
#if SYCL_ESIMD_MOE_AVAILABLE
    static bool fused_moe_disabled = (std::getenv("GGML_SYCL_DISABLE_FUSED_MOE") != nullptr);
    if (fused_moe_disabled) {
        return false;  // Disabled by environment variable
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    // Only use fused kernel for batched prefill (multiple tokens)
    if (ne12 <= 1) {
        return false;  // Use MMVQ for single-token decode
    }

    // Check for supported quantization types
    if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_MXFP4) {
        return false;
    }

    // Input must be F32
    if (src1->type != GGML_TYPE_F32) {
        return false;
    }

    const queue_ptr stream = ctx.stream();

    // Calculate parameters
    const int64_t num_experts = ne02;          // Number of experts
    const int64_t nrows = ne01;                // Output rows per expert
    const int64_t ncols = ne00;                // Hidden dimension (input size)
    const int64_t n_ids = ids->ne[0];          // Expert selections per token
    const int64_t num_tokens = ne12;           // Number of tokens (from src1)

    // Strides
    const int64_t stride_expert = nb02;        // Bytes between experts

    GGML_SYCL_DEBUG("[MoE FUSED] Attempting fused kernel: tokens=%ld, experts=%ld, nrows=%ld, ncols=%ld, ne11=%ld, type=%d\n",
                    (long)num_tokens, (long)num_experts, (long)nrows, (long)ncols, (long)ne11, src0->type);
    GGML_SYCL_DEBUG("[MoE FUSED] Input strides: nb11=%ld, nb12=%ld, Output strides: nb1=%ld, nb2=%ld\n",
                    (long)nb11, (long)nb12, (long)nb1, (long)nb2);

    // Launch appropriate kernel based on quantization type
    if (src0->type == GGML_TYPE_Q8_0) {
        launch_fused_moe_q8_0(
            src0->data,                        // expert_weights
            (const float *)src1->data,         // input
            (const int32_t *)ids->data,        // expert_ids
            (float *)dst->data,                // output
            stride_expert,
            ncols,
            nrows,
            n_ids,
            num_tokens,
            ne11,                              // src1 dimension 1 (for modulo wrapping)
            ids->nb[0],                        // ids_nb0
            ids->nb[1],                        // ids_nb1
            nb11,                              // src1 stride for dim 1
            nb12,                              // src1 stride for dim 2
            nb1,                               // dst stride for dim 1
            nb2,                               // dst stride for dim 2
            *stream
        );
        GGML_SYCL_DEBUG("[MoE FUSED] Q8_0 kernel launched\n");
        return true;
    } else if (src0->type == GGML_TYPE_MXFP4) {
        launch_fused_moe_mxfp4(
            src0->data,
            (const float *)src1->data,
            (const int32_t *)ids->data,
            (float *)dst->data,
            stride_expert,
            ncols,
            nrows,
            n_ids,
            num_tokens,
            ne11,
            ids->nb[0],
            ids->nb[1],
            nb11,
            nb12,
            nb1,
            nb2,
            *stream
        );
        GGML_SYCL_DEBUG("[MoE FUSED] MXFP4 kernel launched\n");
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

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    init_moe_debug();

    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    // Try fused MoE ESIMD kernel first for batched prefill (ne12 > 1)
    // This is much faster than per-expert dispatch for prompt processing
    if (ggml_sycl_mul_mat_id_fused(ctx, src0, src1, ids, dst)) {
        GGML_SYCL_DEBUG("[MoE] Fused ESIMD dispatch successful for type %d\n", src0->type);
        return;
    }

    // Try GPU-side expert routing (MMVQ) - good for decode (ne12 == 1)
    // This avoids the host sync that blocks graph recording
    if (ggml_sycl_mul_mat_id_vec_q(ctx, src0, src1, ids, dst)) {
        GGML_SYCL_DEBUG("[MoE] GPU-side MMVQ dispatch successful for type %d\n", src0->type);
        return;
    }
    GGML_SYCL_DEBUG("[MoE] Falling back to host-side routing for type %d\n", src0->type);

    const queue_ptr stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;

    // Host sync - incompatible with SYCL graph recording
    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    // Debug: print expert IDs and input values
    if (g_moe_debug_enabled) {
        g_moe_debug_call_count++;
        fprintf(stderr, "\n[MoE DEBUG] call=%d ne12=%ld n_as=%ld n_ids=%ld ids_ne1=%ld\n",
                g_moe_debug_call_count, (long)ne12, (long)n_as, (long)n_ids, (long)ids->ne[1]);
        fprintf(stderr, "[MoE DEBUG] ids tensor: ne=[%ld,%ld,%ld,%ld] nb=[%ld,%ld,%ld,%ld] op=%s view_src=%p\n",
                (long)ids->ne[0], (long)ids->ne[1], (long)ids->ne[2], (long)ids->ne[3],
                (long)ids->nb[0], (long)ids->nb[1], (long)ids->nb[2], (long)ids->nb[3],
                ggml_op_name(ids->op), (void*)ids->view_src);

        // Print expert IDs for first few tokens
        fprintf(stderr, "[MoE DEBUG] Expert IDs:\n");
        for (int64_t iid1 = 0; iid1 < std::min((int64_t)4, ids->ne[1]); iid1++) {
            fprintf(stderr, "  token %ld: [", (long)iid1);
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t expert_id = *(const int32_t *)(ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                fprintf(stderr, "%d", expert_id);
                if (id < n_ids - 1) fprintf(stderr, ", ");
            }
            fprintf(stderr, "]\n");
        }

        // Copy src1 to host and print first few values
        std::vector<float> src1_host(std::min((size_t)16, (size_t)ggml_nelements(src1)));
        SYCL_CHECK(CHECK_TRY_ERROR(
            stream->memcpy(src1_host.data(), src1->data, src1_host.size() * sizeof(float))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[MoE DEBUG] src1 first values: [");
        for (size_t i = 0; i < src1_host.size(); i++) {
            fprintf(stderr, "%.4f", src1_host[i]);
            if (i < src1_host.size() - 1) fprintf(stderr, ", ");
        }
        fprintf(stderr, "]\n");
    }

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char *src0_original = (char *)src0->data;
    char *src1_original = (char *)src1->data;
    char *dst_original = (char *)dst->data;

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
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        // Try fused MoE kernel for supported quantization types
        // This eliminates the gather/scatter overhead (96 kernels -> 1 kernel)
#if SYCL_ESIMD_MOE_AVAILABLE
        static bool fused_moe_disabled = (std::getenv("GGML_SYCL_DISABLE_FUSED_MOE") != nullptr);
        const bool use_fused_moe = !fused_moe_disabled && fused_moe_esimd_available() &&  // Enabled for MXFP4
                                   (src0->type == GGML_TYPE_MXFP4) &&  // Only MXFP4 - Q8_0 models also use MXFP4 for MoE
                                   src1->type == GGML_TYPE_F32;

        if (use_fused_moe) {
            const int64_t num_tokens = ids->ne[1];

            if (src0->type == GGML_TYPE_Q8_0) {
                launch_fused_moe_q8_0(
                    src0->data,                    // expert_weights
                    src1->data,                    // input (F32)
                    (const int32_t *)ids->data,    // expert_ids (device)
                    (float *)dst->data,            // output
                    nb02,                          // stride_expert (bytes between experts)
                    ne00,                          // ncols (hidden size)
                    ne01,                          // nrows (output size per expert)
                    n_ids,                         // n_ids (top_k)
                    num_tokens,                    // num_tokens
                    ne11,                          // ne11 (src1 dim 1)
                    ids->nb[0],                    // ids_nb0
                    ids->nb[1],                    // ids_nb1
                    nb11,                          // in_nb11
                    nb12,                          // in_nb12
                    nb1,                           // out_nb1
                    nb2,                           // out_nb2
                    *stream
                );
            } else if (src0->type == GGML_TYPE_MXFP4) {
                launch_fused_moe_mxfp4(
                    src0->data,
                    (const float *)src1->data,
                    (const int32_t *)ids->data,
                    (float *)dst->data,
                    nb02,
                    ne00,
                    ne01,
                    n_ids,
                    num_tokens,
                    ne11,
                    ids->nb[0],
                    ids->nb[1],
                    nb11,
                    nb12,
                    nb1,
                    nb2,
                    *stream
                );
            }
            return;
        }
#endif

        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);

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


            ggml_sycl_pool_alloc<int> dev_cur_src1_row(ctx.pool(), 1);
            ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), num_src1_rows);
            SYCL_CHECK(CHECK_TRY_ERROR(
                stream->memset(dev_cur_src1_row.get(), 0, sizeof(int))));

            const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
            assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

            {
                sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne10, max_work_group_size));
                sycl::range<3> grid_dims(1, n_ids, ids->ne[1]);
                stream->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<int, 0> src1_row_acc(cgh);

                    char *__restrict src1_contiguous_get =
                        src1_contiguous.get();
                    int *__restrict dev_cur_src1_row_get =
                        dev_cur_src1_row.get();
                    mmid_row_mapping *__restrict dev_row_mapping_get =
                        dev_row_mapping.get();
                    size_t ids_nb_ct6 = ids->nb[1];
                    size_t ids_nb_ct7 = ids->nb[0];

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_copy_src1_to_contiguous(
                                src1_original, src1_contiguous_get,
                                dev_cur_src1_row_get,
                                dev_row_mapping_get, ids_dev, i02,
                                ids_nb_ct6, ids_nb_ct7, ne11, ne10, nb11, nb12,
                                item_ct1, src1_row_acc);
                        });
                });
            }

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);

            {
                sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
                sycl::range<3> grid_dims(1, 1, num_src1_rows);
                stream->submit([&](sycl::handler &cgh) {
                    const char *__restrict dst_contiguous_get =
                        dst_contiguous.get();
                    const mmid_row_mapping *__restrict dev_row_mapping_get =
                        dev_row_mapping.get();

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_copy_dst_from_contiguous(dst_original,
                                                       dst_contiguous_get,
                                                       dev_row_mapping_get,
                                                       ne0, nb1, nb2, item_ct1);
                        });
                });
            }
        }

    }

    // Debug: print output values after MoE operation
    if (g_moe_debug_enabled) {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        std::vector<float> dst_host(std::min((size_t)16, (size_t)ggml_nelements(dst)));
        SYCL_CHECK(CHECK_TRY_ERROR(
            stream->memcpy(dst_host.data(), dst->data, dst_host.size() * sizeof(float))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[MoE DEBUG] dst first values: [");
        for (size_t i = 0; i < dst_host.size(); i++) {
            fprintf(stderr, "%.4f", dst_host[i]);
            if (i < dst_host.size() - 1) fprintf(stderr, ", ");
        }
        fprintf(stderr, "]\n");
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
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


static void ggml_sycl_set_main_device(const int main_device) try {
    if (dpct::get_current_device_id() == static_cast<unsigned int> (main_device)) {
        return;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Debug: Dump tensor values for NON-FA attention path comparison
#define NON_FA_DEBUG_DUMP 0
#if NON_FA_DEBUG_DUMP
#include <cstring>
static void dump_non_fa_attention_tensor(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) {
    // Only dump for specific tensor names related to attention
    // Key tensors: "kq", "kq_soft_max", "kqv" for non-FA path
    // Also dump "fattn" for FA path comparison
    const char* name = dst->name;
    if (!name || name[0] == '\0') return;

    // Check if this is an attention-related tensor
    bool is_kq = (strncmp(name, "kq", 2) == 0 && name[2] != 'v');  // kq but not kqv
    bool is_kq_soft_max = (strncmp(name, "kq_soft_max", 11) == 0);
    bool is_kqv = (strncmp(name, "kqv", 3) == 0 && name[3] != '_');
    bool is_fattn = (strncmp(name, "fattn", 5) == 0);

    if (!is_kq && !is_kq_soft_max && !is_kqv && !is_fattn) return;

    // Static call counter for each tensor type
    static int kq_count = 0;
    static int kq_soft_max_count = 0;
    static int kqv_count = 0;
    static int fattn_count = 0;

    int call_num = 0;
    if (is_kq) call_num = ++kq_count;
    else if (is_kq_soft_max) call_num = ++kq_soft_max_count;
    else if (is_kqv) call_num = ++kqv_count;
    else if (is_fattn) call_num = ++fattn_count;

    // Only dump first 50 calls of each type
    if (call_num > 50) return;

    // Wait for computation to complete
    ctx.stream()->wait();

    // Get tensor info
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    const size_t total_elements = ne0 * ne1 * ne2 * ne3;

    // Only dump F32 tensors for now
    if (dst->type != GGML_TYPE_F32) {
        fprintf(stderr, "\n[NON-FA-DEBUG] %s call=%d type=%d (not F32, skipping dump)\n",
                name, call_num, dst->type);
        return;
    }

    fprintf(stderr, "\n[NON-FA-DEBUG] %s call=%d shape=[%lld,%lld,%lld,%lld] total=%zu\n",
            name, call_num, (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3, total_elements);

    // Copy data to host
    std::vector<float> host_data(total_elements);
    ctx.stream()->memcpy(host_data.data(), dst->data, total_elements * sizeof(float)).wait();

    // Print summary statistics
    float min_val = host_data[0], max_val = host_data[0], sum = 0;
    for (size_t i = 0; i < total_elements; i++) {
        float v = host_data[i];
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }
    fprintf(stderr, "  Stats: min=%.6f max=%.6f mean=%.6f\n", min_val, max_val, sum / total_elements);

    // Print first few values for each dimension
    if (is_kqv || is_fattn) {
        // For attention output: [D][n_heads][n_queries][batch] or similar
        // Print first 8 values for first few heads
        fprintf(stderr, "  Output (first 8 values per head, first query):\n");
        for (int h = 0; h < std::min((int)ne2, 8); h++) {
            fprintf(stderr, "    h=%2d: [", h);
            for (int d = 0; d < std::min((int)ne0, 8); d++) {
                // Assuming layout [D][n_queries][n_heads][batch] -> index = d + ne0*(q + ne1*(h + ne2*b))
                size_t idx = d + ne0 * (0 + ne1 * h);  // first query
                fprintf(stderr, "%.6f%s", host_data[idx], d < 7 ? ", " : "");
            }
            fprintf(stderr, "]\n");
        }
        // Print a few higher heads
        for (int h : {8, 16, 32, 63}) {
            if (h < (int)ne2) {
                fprintf(stderr, "    h=%2d: [", h);
                for (int d = 0; d < std::min((int)ne0, 8); d++) {
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
        for (int h = 0; h < std::min((int)ne2, 4); h++) {
            fprintf(stderr, "    h=%2d: [", h);
            for (int kv = 0; kv < std::min((int)ne0, 16); kv++) {
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
    FILE* f = fopen(filename, "w");
    if (f) {
        fprintf(f, "# Tensor: %s call=%d shape=[%lld,%lld,%lld,%lld]\n",
                name, call_num, (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3);
        fprintf(f, "# Strides: nb=[%zu,%zu,%zu,%zu]\n", dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        fprintf(f, "\n=== DATA ===\n");

        // Write all data organized by dimensions
        for (int64_t i3 = 0; i3 < ne3; i3++) {
            for (int64_t i2 = 0; i2 < ne2; i2++) {
                for (int64_t i1 = 0; i1 < ne1; i1++) {
                    fprintf(f, "[b=%lld,h=%lld,q=%lld]: ", (long long)i3, (long long)i2, (long long)i1);
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
    if (!g_sycl_loaded) return false;

    // Debug: trace operations in multi-process mode
    if (g_sycl_tp_config.is_multiprocess && g_ggml_sycl_tp_debug) {
        static int op_trace = 0;
        if (op_trace++ < 100) {
            fprintf(stderr, "[RANK %d] COMPUTE: op=%s tensor=%s ne=[%lld,%lld,%lld,%lld]\n",
                    g_sycl_tp_config.mpi_rank, ggml_op_name(dst->op),
                    dst->name ? dst->name : "(null)",
                    (long long)dst->ne[0], (long long)dst->ne[1],
                    (long long)dst->ne[2], (long long)dst->ne[3]);
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
                ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)dst->src[0]->extra;
                if (extra && extra->tp_sharded) {
                    is_tp_mul_mat = true;
                }
            }

            // Skip ALL ops except TP MUL_MAT on secondary devices
            if (!is_tp_mul_mat) {
                static int skip_log = 0;
                if (skip_log++ < 5) {
                    GGML_SYCL_DEBUG("TP: Skipping op %s on device %d (not TP MUL_MAT)\n",
                                   ggml_op_name(dst->op), ctx.device);
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
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
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
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
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
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API void ggml_backend_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_backend_sycl_get_device_memory(int device, size_t *free,
                                                   size_t *total) try {
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
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::dev_mgr::instance().get_device(device).get_memory_info(*free, *total)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    return sycl_ctx->name.c_str();
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

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
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    // Accept both regular SYCL buffer type and TP host compute buffer type
    GGML_ASSERT((buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) ||
                 buf->buft == ggml_backend_sycl_host_compute_buffer_type(sycl_ctx->device)) &&
                "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    // Accept both regular SYCL buffer type and TP host compute buffer type
    GGML_ASSERT((buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) ||
                 buf->buft == ggml_backend_sycl_host_compute_buffer_type(sycl_ctx->device)) &&
                "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *src,
                                               ggml_tensor *dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
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
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        return true;
    }

    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
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
static bool ggml_sycl_can_fuse_rmsnorm_mulmat(
    const ggml_tensor * rms_norm,
    const ggml_tensor * mul,
    const ggml_tensor * mulmat
) {
    // Skip fusion for small batch sizes (token generation) where overhead hurts performance
    const int64_t nrows = rms_norm->src[0]->ne[1];
    if (nrows < 8) return false;

    // Check input types - input to RMS_NORM must be F32
    if (rms_norm->src[0]->type != GGML_TYPE_F32) return false;
    // Check intermediate types
    if (rms_norm->type != GGML_TYPE_F32) return false;
    if (mul->type != GGML_TYPE_F32) return false;
    // MUL_MAT output must be F32
    if (mulmat->type != GGML_TYPE_F32) return false;

    // Check GEMM weight type (support common quantized types)
    ggml_type w_type = mulmat->src[0]->type;
    if (w_type != GGML_TYPE_Q4_0 && w_type != GGML_TYPE_Q4_1 &&
        w_type != GGML_TYPE_Q8_0 && w_type != GGML_TYPE_Q4_K &&
        w_type != GGML_TYPE_Q5_K && w_type != GGML_TYPE_Q6_K) {
        return false;
    }

    // Check single GPU (no split buffers) for simplicity
    if (ggml_backend_buffer_is_sycl_split(mulmat->src[0]->buffer)) {
        return false;
    }

    // Check dimensions are aligned for Q8_1 quantization
    int64_t ncols = rms_norm->src[0]->ne[0];
    if (ncols % QK8_1 != 0) return false;

    // Gamma dimensions must match
    const ggml_tensor * gamma = get_mul_weight(mul, rms_norm);
    if (gamma->ne[0] != ncols) return false;

    return true;
}

// Fused dispatch function: RMS_NORM + MUL + MUL_MAT
// Eliminates intermediate normalized tensor by fusing into quantization
static void ggml_sycl_mul_mat_with_rmsnorm(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * x,           // Original input (pre-RMSNorm)
    const ggml_tensor * gamma,       // RMSNorm weight (gamma)
    const ggml_tensor * W,           // GEMM weight (quantized)
    ggml_tensor * dst,               // Output
    float eps                        // RMSNorm epsilon
) {
    const int64_t nrows = x->ne[1];          // Batch size (M)
    const int64_t ncols = x->ne[0];          // Hidden dim (K)
    // Must use MATRIX_ROW_PADDING (512) to match MMQ expectations, not QK8_1 (32)
    const int64_t ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);

    dpct::queue_ptr stream = ctx.stream();

    // Allocate temporary buffers
    ggml_sycl_pool_alloc<float> scales_buf(ctx.pool(), nrows);
    ggml_sycl_pool_alloc<char> q8_buf(ctx.pool(),
        nrows * (ncols_padded / QK8_1) * sizeof(block_q8_1));

    // Get data pointers
    const float * x_dd = (const float *)x->data;
    const float * gamma_dd = (const float *)gamma->data;

    // Fused RMSNorm + quantization
    // This eliminates the intermediate normalized tensor (~2MB for Mistral 7B at batch 128)
    fused_rmsnorm_quantize_q8_1_sycl(
        x_dd,                    // Input (unnormalized)
        gamma_dd,                // Gamma (RMSNorm weight)
        q8_buf.get(),            // Q8_1 output
        scales_buf.get(),        // Temporary scales buffer
        nrows,
        ncols,
        ncols_padded,
        eps,
        stream,
        ctx.device
    );

    // Get the destination pointer
    float * dst_dd = (float *)dst->data;

    // Get weight pointer
    const char * W_dd = (const char *)W->data;

    // Call the MMQ kernel through the existing dispatch function
    // We pass:
    // - src0 = W (weights tensor for metadata: type, ne[0])
    // - src1 = x (input tensor for metadata: ne[0] for assertion)
    // - dst = dst (output tensor for metadata: ne[0] for stride)
    // - src1_ddq_i = q8_buf.get() (our fused quantized activations)

    // W->ne[0] = K (in_features), W->ne[1] = N (out_features for 2D)
    // But for higher-dim weights, use ggml_nrows(W) to get total output rows
    const int64_t nrows_W = ggml_nrows(W);   // out_features (N)

    // Call the MMQ dispatch function
    // Parameters: row range = [0, nrows_W), src1_ncols = nrows (batch size)
    ggml_sycl_op_mul_mat_q(
        ctx,
        W,                    // src0: weights (for metadata)
        x,                    // src1: original input (for ne[0] assertion - matches hidden dim)
        dst,                  // dst: output (for ne[0] stride)
        W_dd,                 // src0_dd_i: weights data
        nullptr,              // src1_ddf_i: unused since we provide quantized data
        q8_buf.get(),         // src1_ddq_i: our fused quantized activations
        dst_dd,               // dst_dd_i: output data
        0, nrows_W,           // row_low, row_high: full output range
        nrows,                // src1_ncols: batch size
        ncols_padded,         // src1_padded_row_size
        stream
    );
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
static std::vector<std::pair<int, ggml_tensor*>> find_mulmat_consumers(
    const ggml_cgraph * cgraph,
    const ggml_tensor * tensor,
    int start_idx
) {
    std::vector<std::pair<int, ggml_tensor*>> consumers;
    for (int j = start_idx; j < cgraph->n_nodes; j++) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_MUL_MAT) {
            // MUL_MAT src[1] is the activation input
            if (node->src[1] == tensor) {
                consumers.push_back({j, node});
            }
        }
    }
    return consumers;
}

// Check if all MUL_MAT consumers can be fused with RMS_NORM
static bool can_fuse_all_projections(
    const ggml_tensor * rms_norm,
    const ggml_tensor * mul,
    const std::vector<std::pair<int, ggml_tensor*>> & mulmat_consumers
) {
    if (mulmat_consumers.empty()) return false;

    // DISABLED: Per-projection fusion causes numerical differences
    // TODO: Investigate root cause
    return false;

    // Skip fusion for small batches (token generation)
    // The fusion overhead only amortizes with larger batches during prompt processing
    const int64_t nrows = rms_norm->src[0]->ne[1];
    if (nrows < 8) return false;

    // Check input types
    if (rms_norm->src[0]->type != GGML_TYPE_F32) return false;
    if (rms_norm->type != GGML_TYPE_F32) return false;
    if (mul->type != GGML_TYPE_F32) return false;

    // Check dimensions are aligned for Q8_1 quantization
    int64_t ncols = rms_norm->src[0]->ne[0];
    if (ncols % QK8_1 != 0) return false;

    // Gamma dimensions must match
    const ggml_tensor * gamma = get_mul_weight(mul, rms_norm);
    if (gamma->ne[0] != ncols) return false;

    // Check all MUL_MAT consumers
    for (const auto & [idx, mulmat] : mulmat_consumers) {
        // MUL_MAT output must be F32
        if (mulmat->type != GGML_TYPE_F32) return false;

        // Check GEMM weight type
        ggml_type w_type = mulmat->src[0]->type;
        if (w_type != GGML_TYPE_Q4_0 && w_type != GGML_TYPE_Q4_1 &&
            w_type != GGML_TYPE_Q8_0 && w_type != GGML_TYPE_Q4_K &&
            w_type != GGML_TYPE_Q5_K && w_type != GGML_TYPE_Q6_K) {
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
static void execute_per_projection_fusion(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * x,           // Original input (pre-RMSNorm)
    const ggml_tensor * gamma,       // RMSNorm weight (gamma)
    float eps,                       // RMSNorm epsilon
    const std::vector<std::pair<int, ggml_tensor*>> & mulmat_consumers
) {
    const int64_t nrows = x->ne[1];          // Batch size (M)
    const int64_t ncols = x->ne[0];          // Hidden dim (K)
    // Must use MATRIX_ROW_PADDING (512) to match MMQ expectations, not QK8_1 (32)
    const int64_t ncols_padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);

    dpct::queue_ptr stream = ctx.stream();

    // Allocate temporary buffers for fused quantization
    // These are reused for each projection
    ggml_sycl_pool_alloc<float> scales_buf(ctx.pool(), nrows);
    ggml_sycl_pool_alloc<char> q8_buf(ctx.pool(),
        nrows * (ncols_padded / QK8_1) * sizeof(block_q8_1));

    // Get data pointers
    const float * x_dd = (const float *)x->data;
    const float * gamma_dd = (const float *)gamma->data;

    // Fused RMSNorm + quantization (done once, reused for all projections)
    fused_rmsnorm_quantize_q8_1_sycl(
        x_dd,
        gamma_dd,
        q8_buf.get(),
        scales_buf.get(),
        nrows,
        ncols,
        ncols_padded,
        eps,
        stream,
        ctx.device
    );

    // Execute each MUL_MAT with the pre-quantized activations
    for (const auto & [idx, mulmat] : mulmat_consumers) {
        const ggml_tensor * W = mulmat->src[0];  // GEMM weights
        float * dst_dd = (float *)mulmat->data;
        const char * W_dd = (const char *)W->data;

        // row_low/row_high refers to output rows (weight matrix rows)
        // W->ne[0] is K (input features), ggml_nrows(W) is N (output features)
        const int64_t nrows_W = ggml_nrows(W);   // out_features (N)

        // Call the MMQ dispatch function with our fused quantized activations
        ggml_sycl_op_mul_mat_q(
            ctx,
            W,                    // src0: weights (for metadata)
            x,                    // src1: original input (for ne[0] assertion)
            mulmat,               // dst: output (for ne[0] stride)
            W_dd,                 // src0_dd_i: weights data
            nullptr,              // src1_ddf_i: unused
            q8_buf.get(),         // src1_ddq_i: our fused quantized activations
            dst_dd,               // dst_dd_i: output data
            0, nrows_W,           // row_low, row_high: full output range
            nrows,                // src1_ncols: batch size
            ncols_padded,         // src1_padded_row_size
            stream
        );
    }
}

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    // Debug: trace graph compute entry
    if (g_sycl_tp_config.is_multiprocess && g_ggml_sycl_tp_debug) {
        fprintf(stderr, "[RANK %d] GRAPH_COMPUTE_IMPL: n_nodes=%d, device=%d\n",
                g_sycl_tp_config.mpi_rank, cgraph->n_nodes, sycl_ctx->device);
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
    bool disable_fusion = disable_fusion_env || (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1);

    // Track nodes that have been executed via fusion (to skip later)
    std::unordered_set<const ggml_tensor*> fused_nodes;

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
                fprintf(stderr, "[RANK %d] NODE %d: op=%s name=%s\n",
                        g_sycl_tp_config.mpi_rank, i, ggml_op_name(node->op),
                        node->name ? node->name : "(null)");
                fflush(stderr);
            }
        }

        // Skip nodes already executed via fusion - no kernel work needed
        if (fused_nodes.count(node)) {
            continue;
        }

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif

        if (!disable_fusion && node->op == GGML_OP_RMS_NORM) {
            // Try 3-way kernel fusion: RMS_NORM + MUL + ADD
            if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, { i + 2 }) &&
                ggml_sycl_check_fusion_types(cgraph, i, 3) &&
                ggml_is_contiguous(cgraph->nodes[i + 2])) {
                ggml_tensor * mul_node = cgraph->nodes[i + 1];
                ggml_tensor * add_node = cgraph->nodes[i + 2];
                ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, node, mul_node, add_node);
                i += 2; // Skip the MUL and ADD nodes
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
                        ggml_tensor * x = node->src[0];
                        ggml_tensor * gamma = get_mul_weight(mul_node, node);
                        float eps;
                        memcpy(&eps, node->op_params, sizeof(float));

                        execute_per_projection_fusion(*sycl_ctx, x, gamma, eps, consumers);

                        // Mark MUL and all MUL_MAT consumers as fused (to skip later)
                        fused_nodes.insert(mul_node);
                        for (const auto & [idx, mulmat] : consumers) {
                            fused_nodes.insert(mulmat);
                        }
                        continue; // Skip RMS_NORM execution (already done in fusion)
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
                ggml_tensor * mul_node = cgraph->nodes[i + 1];
                ggml_tensor * mulmat_node = cgraph->nodes[i + 2];

                if (ggml_sycl_can_fuse_rmsnorm_mulmat(node, mul_node, mulmat_node)) {
                    // Get original input, gamma, and GEMM weights
                    ggml_tensor * x = node->src[0];              // Pre-RMSNorm input
                    ggml_tensor * gamma = get_mul_weight(mul_node, node);  // Gamma (norm weight)
                    ggml_tensor * W = mulmat_node->src[0];        // GEMM weights

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
                i++; // Skip the MUL node
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
                if (next->op == GGML_OP_RMS_NORM &&
                    next->src[0] == node &&
                    ggml_sycl_check_fusion_types(cgraph, i, 2) &&
                    ggml_is_contiguous(next)) {
                    ggml_sycl_op_add_rms_norm_fused(*sycl_ctx, node, next);
                    i++; // Skip the RMS_NORM node
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
                if (next->op == GGML_OP_ADD &&
                    (next->src[0] == node || next->src[1] == node) &&
                    node->type == GGML_TYPE_F32 &&
                    next->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(node) &&
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
                        i++; // Skip the ADD node
                        continue;
                    }
                }
            }
        }

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
            struct { uint16_t d_bits; uint8_t qs[16]; } wblk;
            stream->memcpy(&wblk, (void*)l31_weight_addr, sizeof(wblk)).wait();
            uint16_t d_raw = wblk.d_bits;
            sycl::half d_half;
            memcpy(&d_half, &d_raw, sizeof(sycl::half));
            float d_f = static_cast<float>(d_half);
            fprintf(stderr, "TP DEBUG END_PASS: L31 weight d=%f %s\n",
                    d_f, (d_f > 100.0f || std::isnan(d_f)) ? "CORRUPTED" : "OK");
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
                    // Check if GPU-side dispatch is available (no host sync needed)
                    // GPU dispatch uses kernel that reads expert IDs directly on GPU
                    // Note: GPU dispatch supports ne12 > 1, but SYCL graphs only work with ne12 == 1
                    // because the graph topology changes between prefill and decode
                    const ggml_tensor * node = cgraph->nodes[i];
                    const ggml_tensor * src0 = node->src[0];
                    const ggml_tensor * src1 = node->src[1];
                    const int64_t ne12 = src1->ne[2];

                    bool gpu_dispatch_available = false;
                    if (ne12 == 1 && ggml_is_quantized(src0->type)) {
                        switch (src0->type) {
                            case GGML_TYPE_Q4_0:
                            case GGML_TYPE_Q8_0:
                            case GGML_TYPE_MXFP4:
                                gpu_dispatch_available = true;
                                break;
                            default:
                                break;
                        }
                    }

                    if (!gpu_dispatch_available) {
                        // Fall back to host dispatch or GPU dispatch without graph recording
                        // Only log once to avoid spamming the console
                        static bool logged_once = false;
                        if (!logged_once) {
                            GGML_LOG_INFO("%s: disabling SYCL graphs for MUL_MAT_ID (type %s, ne12=%" PRId64 ")\n",
                                          __func__, ggml_type_name(src0->type), ne12);
                            logged_once = true;
                        }
                        return false;
                    }
                    // GPU dispatch available with ne12==1 - MUL_MAT_ID is graph-compatible
                }
                break;
            case GGML_OP_MUL_MAT:
                // MUL_MAT is graph-compatible because we pre-reorder ALL tensors
                // before graph recording starts (via graph_pre_reorder_all).
                // No malloc/free calls happen during recording.
                // Note: MoE models still can't use graphs because MUL_MAT_ID's GPU dispatch
                // path (ggml_sycl_mul_mat_id_vec_q) uses ggml_sycl_pool_alloc for Q8_1 quantization
                // buffers, which is incompatible with SYCL graph recording.
                // TODO: Pre-allocate Q8_1 buffers for MoE to enable graph support.
                if (!g_ggml_sycl_use_async_mem_op) {
                    // Without async mem, only allow graphs for simple (non-MoE) models
                    // MoE models have pool allocations in MUL_MAT_ID that break graph recording
                    for (int j = 0; j < cgraph->n_nodes; j++) {
                        if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID) {
                            return false;  // MoE model detected - skip graphs
                        }
                    }
                }
                break;
        }
    }
    return true;
}
#endif

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

    // Pre-reorder disabled for now - Q8_0 doesn't use reordering anyway
    // pre_reorder_all_tensors(sycl_ctx, cgraph);

#ifdef GGML_SYCL_GRAPH
    // Disable SYCL graph for TP mode - we need our handlers to run every pass for caching
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

        // Skip SYCL graphs during prompt phase - use non-graph compute
        if (is_prompt_phase) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] skipping graph - prompt phase (ne[1]>1)\n");
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        // Pre-reorder ALL tensors before decode graph recording.
        // This ensures we don't have incremental reordering blocking graph reuse.
        if (is_decode_phase && graph_needs_reorder(*sycl_ctx, cgraph)) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] pre-reordering all tensors before graph recording (decode phase)\n");
            graph_pre_reorder_all(*sycl_ctx, cgraph);
        }

        // Pre-allocate V2 partition attention buffers before graph recording.
        // This ensures V2 dispatch works during graph recording (malloc/free forbidden during recording).
        if (is_decode_phase && !sycl_ctx->exec_graph) {
            ggml_sycl_v2_pre_allocate_buffers(*sycl_ctx, cgraph);
        }

        // Minimum nodes to benefit from graph batching - skip tiny graphs
        constexpr int MIN_GRAPH_NODES = 10;
        if (cgraph->n_nodes < MIN_GRAPH_NODES) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] skipping - graph too small (%d < %d nodes)\n",
                           cgraph->n_nodes, MIN_GRAPH_NODES);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }

        // Check if cached graph matches current graph structure.
        // Different n_nodes means different graph topology - must re-record.
        if (sycl_ctx->exec_graph && sycl_ctx->exec_graph_n_nodes != cgraph->n_nodes) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] invalidating cache - n_nodes changed (%d -> %d)\n",
                           sycl_ctx->exec_graph_n_nodes, cgraph->n_nodes);
            sycl_ctx->exec_graph.reset();
            sycl_ctx->exec_graph_n_nodes = 0;
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
            sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

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
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            sycl_ctx->exec_graph_n_nodes = cgraph->n_nodes;  // Track for cache validation
            GGML_SYCL_DEBUG("[SYCL-GRAPH] unique_ptr created, cached n_nodes=%d\n", cgraph->n_nodes);

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

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
try
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    const queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } else
        GGML_ABORT("fatal error");
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL, // ggml_backend_sycl_cpy_tensor_async,
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
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
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
    int device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ggml_sycl_set_device(ctx->device);
    SYCL_CHECK(CHECK_TRY_ERROR(
    dpct::dev_mgr::instance().get_device(ctx->device).get_memory_info(*free, *total)));
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
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_sycl_host_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    int device = sycl_ctx->device;
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
#if defined (GGML_SYCL_F16)
                    return ggml_is_contiguous(op->src[0]) && (op->type == op->src[0]->type);
#else
                    return ggml_is_contiguous(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) && (op->type == op->src[0]->type);
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
                if (a_type == GGML_TYPE_IQ4_NL  || a_type == GGML_TYPE_IQ4_XS ||
                    a_type == GGML_TYPE_IQ3_XXS || a_type == GGML_TYPE_IQ3_S  ||
                    a_type == GGML_TYPE_IQ2_XXS || a_type == GGML_TYPE_IQ2_XS || a_type == GGML_TYPE_IQ2_S ||
                    a_type == GGML_TYPE_IQ1_S || a_type == GGML_TYPE_IQ1_M
                    ) {
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
                if (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  return false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                if (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    return false;
                }
                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
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
                        return true;
                    default:
                        return false;
                }
            }
         case GGML_OP_SET:
               return (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
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
                if (op->src[0] == nullptr || op->src[1] == nullptr ||
                    op->src[2] == nullptr || op->src[3] == nullptr) {
                    return false;
                }
                // Source type must be F32 or F16
                const bool src_type_ok = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                // Destination type (op->type which is view of dst_orig) must be F32 or F16
                const bool dst_type_ok = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
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
                if (src0_type == src1_type && (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) && src0_type != GGML_TYPE_BF16) {
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
                if(src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_Q4_1) {
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
            return ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
#if defined (GGML_SYCL_F16)
            return ((op->type == GGML_TYPE_F32 || op->type == GGML_SYCL_F16) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_SYCL_F16) && (op->type == op->src[0]->type));
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
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
            return true;
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32 && op->op_params[0] == GGML_SCALE_MODE_NEAREST && !(op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return op->src[0]->ne[0] * sizeof(int) <=
                   ggml_sycl_info().devices[device].smpbo;
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
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
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
        ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
        ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
        return buft_ctx->device == sycl_ctx->device;
    }

    // TP buffer type - check if the device is one of the TP devices
    if (buft->iface.get_name == ggml_backend_sycl_tp_buffer_type_name) {
        ggml_backend_sycl_tp_buffer_type_context * buft_ctx = (ggml_backend_sycl_tp_buffer_type_context *)buft->context;
        ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
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
    const int min_batch_size = 32;
    return get_op_batch_size(op) >= min_batch_size;
    GGML_UNUSED(dev);
}

static ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    return nullptr;
#else
  sycl::event *event_ptr = new sycl::event();

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
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
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
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }

    if (strcmp(name, "ggml_backend_tp_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_tp_buffer_type;
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
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
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

    ggml_backend_t sycl_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    return sycl_backend;
}

// ============================================================================
// Flash Attention seq_ids host pointer support
// ============================================================================
// Thread-local storage for seq_ids host pointers
// These are set by llama layer before graph execution and used by fattn kernel
// The pointers must be valid SYCL_Host buffer memory (USM accessible from GPU)
struct ggml_sycl_seq_ids_cache {
    const int32_t * q_seq_ids = nullptr;
    size_t q_count = 0;
    const int32_t * kv_seq_ids = nullptr;
    size_t kv_count = 0;
};

static thread_local ggml_sycl_seq_ids_cache g_sycl_seq_ids_cache;

void ggml_backend_sycl_set_seq_ids_host(
    const int32_t * q_seq_ids, size_t q_count,
    const int32_t * kv_seq_ids, size_t kv_count) {
    g_sycl_seq_ids_cache.q_seq_ids = q_seq_ids;
    g_sycl_seq_ids_cache.q_count = q_count;
    g_sycl_seq_ids_cache.kv_seq_ids = kv_seq_ids;
    g_sycl_seq_ids_cache.kv_count = kv_count;
}

void ggml_backend_sycl_clear_seq_ids_host(void) {
    g_sycl_seq_ids_cache.q_seq_ids = nullptr;
    g_sycl_seq_ids_cache.q_count = 0;
    g_sycl_seq_ids_cache.kv_seq_ids = nullptr;
    g_sycl_seq_ids_cache.kv_count = 0;
}

// Internal getter for fattn.cpp to access the cached host pointers
const int32_t * ggml_sycl_get_seq_ids_host_q(size_t * count) {
    if (count) *count = g_sycl_seq_ids_cache.q_count;
    return g_sycl_seq_ids_cache.q_seq_ids;
}

const int32_t * ggml_sycl_get_seq_ids_host_kv(size_t * count) {
    if (count) *count = g_sycl_seq_ids_cache.kv_count;
    return g_sycl_seq_ids_cache.kv_seq_ids;
}

// ==============================================================================
// Thread-local cache for pending device tokens (multi-step GPU decode)
// ==============================================================================

struct ggml_sycl_device_token_cache {
    void * token_ptr = nullptr;  // Device pointer to token(s)
    size_t n_tokens = 0;         // Number of tokens
};

static thread_local ggml_sycl_device_token_cache g_sycl_device_token_cache;

void ggml_backend_sycl_set_pending_device_token(void * token_ptr, size_t n_tokens) {
    g_sycl_device_token_cache.token_ptr = token_ptr;
    g_sycl_device_token_cache.n_tokens = n_tokens;
}

void ggml_backend_sycl_clear_pending_device_token(void) {
    g_sycl_device_token_cache.token_ptr = nullptr;
    g_sycl_device_token_cache.n_tokens = 0;
}

// Internal getter for llama-graph.cpp to access the pending device token
void * ggml_sycl_get_pending_device_token(size_t * n_tokens) {
    if (n_tokens) *n_tokens = g_sycl_device_token_cache.n_tokens;
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
