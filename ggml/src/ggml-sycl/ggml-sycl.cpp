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
#include <array>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <functional>
#include <limits>
#include <optional>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <string_view>

#include <sycl/sycl.hpp>
#if __has_include(<unified-runtime/ur_api.h>)
#include <unified-runtime/ur_api.h>
#define GGML_SYCL_HAS_UR_API 1
#else
#define GGML_SYCL_HAS_UR_API 0
#endif
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#endif
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#if SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#if defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif
#include <sycl/half_type.hpp>

#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#ifdef GGML_SYCL_TESTING
#include "ggml-sycl/ggml-sycl-test.h"
#endif

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/fwht.hpp"
#include "ggml-sycl/fused-gemm.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/mem.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/dsv4-hc.hpp"
#include "ggml-sycl/lightning-indexer.hpp"
#include "ggml-sycl/conv2d.hpp"
#include "ggml-sycl/conv2d-dw.hpp"
#include "ggml-sycl/conv2d-transpose.hpp"
#include "ggml-sycl/turbo-wht.hpp"
#include "ggml-sycl/xe-kmd.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/ssm_scan.hpp"
#include "ggml-sycl/fill.hpp"
#include "ggml-sycl/cumsum.hpp"
#include "ggml-sycl/diag.hpp"
#include "ggml-sycl/opt-step.hpp"
#include "ggml-sycl/solve_tri.hpp"
#include "ggml-sycl/gated_delta_net.hpp"
#include "ggml-sycl/pool.hpp"
#include "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000
#include "ggml-sycl/turbo-wht.hpp"

static bool g_sycl_loaded = false;
int g_ggml_sycl_debug = 0;
int g_ggml_sycl_dev_debug = 0;
int g_ggml_sycl_enable_optimize = 1;
int g_ggml_sycl_enable_graph = 0;
int g_ggml_sycl_enable_dnn = 1;
int g_ggml_sycl_fa_onednn = 1;
int g_ggml_sycl_fa_onednn_max_kv = 0;
int g_ggml_sycl_enable_mkl_fa = 1;
int g_ggml_sycl_memtrace = 0;
int g_ggml_sycl_memtrace_step = 64;
int g_ggml_sycl_enable_vmm = 1;
int g_ggml_sycl_enable_fusion = 1;
int g_ggml_sycl_enable_esimd = 1;
int g_ggml_sycl_mmvq_wide = 1;
int g_ggml_sycl_prioritize_dmmv = 0;
int g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
int g_ggml_sycl_use_async_mem_op = 0;
int g_ggml_sycl_use_async_mem_op_requested = 1;
int g_ggml_sycl_use_level_zero_api = 0;
int g_ggml_sycl_enable_flash_attention = 1;
int g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
int g_ggml_sycl_usm_system = 0;
int g_ggml_sycl_enable_host_pinned_mem = 1;
int g_ggml_sycl_host_pinned_mem_2g = 0;
int g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
int g_ggml_sycl_enable_sparse_fa = 0;
int g_ggml_sycl_debug_sparse_fa = 0;
int g_ggml_sycl_sparse_fa_margin = 256;

int g_ggml_sycl_fa_force_vec_standard = 0;
int g_ggml_sycl_fa_q8_gqa_tile = 0;
int g_ggml_sycl_graph_eviction_timeout = 10;

// Resident work-groups per Xe-core, consumed by launch_fattn as max_blocks_per_sm.
//
// This was previously derived as max_work_group_size / max_compute_units, which
// divides a work-item count by an execution-unit count. On an Arc A770 reporting
// 512 compute units and a 1024 max work-group size that yields 2, so
// blocks_per_wave = nsm * max_wg_per_cu = 64 and flash-attention decode was
// measured launching 64 work-groups of 128 work-items - about 12.5% of the
// device's ~65536 resident work-items - at every depth and on every route.
//
// Xe-HPG has 16 XVE per Xe-core and 8 hardware threads per XVE, so 128 hardware
// threads per Xe-core. A 128-work-item work-group at SIMD16 occupies 8 of them,
// giving 16 resident work-groups, which the P5.3 IGC inventory supports for this
// kernel family (GRF 128, zero spill).
//
// Measured on A770 with Mistral-7B Q4_K_M, q8_0 KV, tg32 tokens/s, versus the
// previous effective value of 2:
//
//   depth |  wg=2 |  wg=4 |  wg=8 | wg=16 | wg=32
//       0 | 26.63 |     - | 26.80 | 26.68 |     -
//    4096 | 18.93 | 21.39 | 22.51 | 22.62 | 22.89
//    8192 | 14.81 | 17.84 | 19.85 | 20.74 | 20.46
//   16384 | 10.22 | 13.46 | 15.90 | 17.49 | 16.46
//
// 16 wins both deep cells and 32 regresses there, so the hardware ceiling is the
// right default rather than a clamped fraction of it. Depth 0 is unaffected
// because parallel_blocks is clamped by ntiles_KQ at shallow depth.
static bool g_ggml_sycl_max_wg_per_cu_explicit = false;

bool ggml_sycl_max_wg_per_cu_is_explicit() {
    return g_ggml_sycl_max_wg_per_cu_explicit;
}

static int ggml_sycl_max_wg_per_cu() {
    static const int value = [] {
        const int fallback = 16;
        const char * env = getenv("GGML_SYCL_MAX_WG_PER_CU");
        if (env == nullptr || env[0] == '\0') {
            return fallback;
        }
        char * end = nullptr;
        const long parsed = strtol(env, &end, 10);
        if (end == env || *end != '\0' || parsed < 1 || parsed > 1024) {
            GGML_LOG_WARN(
                "ggml_sycl_max_wg_per_cu: ignoring invalid GGML_SYCL_MAX_WG_PER_CU=\"%s\", using %d\n",
                env, fallback);
            return fallback;
        }
        g_ggml_sycl_max_wg_per_cu_explicit = true;
        return (int) parsed;
    }();
    return value;
}

#if GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
static bool ggml_sycl_dnnl_detect_optimized_gemm(int device) {
    using dt = dnnl::memory::data_type;

    try {
        const dnnl::memory::dims dims    = { 1, 64, 64 };
        const dnnl::memory::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        const auto & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        const auto   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        const auto   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f32, strides), attr);

        const std::string impl = pd.impl_info_str();
        if (impl.find("ref") == std::string::npos) {
            return true;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } catch (const std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    return false;
}
#endif

static ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    int discovered_devices = 0;
    try {
        discovered_devices = dpct::dev_mgr::instance().device_count();
    } catch (sycl::exception const &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        discovered_devices = 0;
    }
    if (discovered_devices == 0) {
        GGML_LOG_WARN("%s: no SYCL devices discovered by the runtime\n", GGML_SYCL_NAME);
        return info;
    }

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    for (int i = 0; i < discovered_devices; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        if (!device.is_gpu()) {
            continue;
        }

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

        GGML_ASSERT(info.device_count < GGML_SYCL_MAX_DEVICES);

        const int gpu_index = info.device_count++;

#if !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[gpu_index].vmm = 0;
#else
        info.devices[gpu_index].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        if (info.devices[gpu_index].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            const size_t physical_page = 2ull << 20; // 2 MiB
            info.devices[gpu_index].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#endif

        info.default_tensor_split[gpu_index] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[gpu_index].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[gpu_index].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[gpu_index].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#if GGML_SYCL_DNNL
        // i is the SYCL device ordinal; gpu_index is the registry slot they land in
        info.devices[gpu_index].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#endif
        info.devices[gpu_index].smpbo = prop.get_local_mem_size();
        info.devices[gpu_index].warp_size = WARP_SIZE;
        info.devices[gpu_index].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

#ifdef GGML_SYCL_GRAPH
        info.devices[gpu_index].graph_support        = device.has(sycl::aspect::ext_oneapi_limited_graph);
        info.devices[gpu_index].graph_update_support = device.has(sycl::aspect::ext_oneapi_graph);
#endif

        info.max_work_group_sizes[gpu_index] = prop.get_max_work_group_size();
        info.devices[gpu_index].max_wg_per_cu = ggml_sycl_max_wg_per_cu();
        info.devices[gpu_index].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        if (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = false;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        if (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            if (r == ZE_RESULT_SUCCESS) {
                info.devices[gpu_index].l0_device_type_valid = true;
                info.devices[gpu_index].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    if (info.device_count == 0) {
        GGML_LOG_WARN("%s: no usable SYCL GPU devices found; leaving SYCL backend disabled\n", GGML_SYCL_NAME);
        return info;
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    // Large buffers can be allocated before ggml_check_sycl() initializes other
    // g_ggml_sycl_enable_* globals, so initialize this one as early as we can.
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#else
    g_ggml_sycl_use_level_zero_api = 0;
#endif

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
    int device_count = ggml_sycl_info().device_count;
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

static const char* dev2dev_int2str(int dev2dev) {
    if (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        return "SYCL API";
    } else if (dev2dev == DEV2DEV_MEMCPY_L0) {
        return "Level Zero API";
    } else if (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        return "Host Forward";
    } else {
        return "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static ze_result_t init_zes() {
    ze_result_t res = zesInit(0);
    if (res != ZE_RESULT_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (int) res);
    }
    return res;
}

ze_result_t get_zes_init_res() {
    static ze_result_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (int) zes_init_res);
    return zes_init_res;
}
#endif

void initialize_sycl_begining() {
    ggml_sycl_xe_kmd_flush_log();
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#endif
}

// Defined below; forward-declared so the startup env report can read the
// same cached flag ggml_sycl_try_fuse_ffn_swiglu() gates on, instead of a
// second independent ggml_sycl_get_env() call that could drift out of sync.
static bool ggml_sycl_ffn_fusion_enabled();

static void ggml_check_sycl() try {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    static bool initialized = false;

    if (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_dev_debug = ggml_sycl_get_env("GGML_SYCL_DEV_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_graph = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_step = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
        g_ggml_sycl_enable_vmm = ggml_sycl_get_env("GGML_SYCL_ENABLE_VMM", 1);
        g_ggml_sycl_enable_fusion = ggml_sycl_get_env("GGML_SYCL_ENABLE_FUSION", 1);
        g_ggml_sycl_enable_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_ESIMD", 1);
        g_ggml_sycl_mmvq_wide = ggml_sycl_get_env("GGML_SYCL_MMVQ_WIDE", 1);
        g_ggml_sycl_prioritize_dmmv = ggml_sycl_get_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
        g_ggml_sycl_xmx_gather_types = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_TYPES", GGML_SYCL_XMX_GATHER_TYPES_DEFAULT);

        // g_ggml_sycl_use_level_zero_api is initialized early in ggml_sycl_init()
        // (before this function may run), since large buffers can be allocated
        // ahead of ggml_check_sycl(). Do not overwrite it here.
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);
        if (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#else
        g_ggml_sycl_enable_flash_attention = 0;
#endif
        g_ggml_sycl_fa_force_vec_standard = ggml_sycl_get_env("GGML_SYCL_FA_FORCE_VEC_STANDARD", 0);
        g_ggml_sycl_fa_q8_gqa_tile = ggml_sycl_get_env("GGML_SYCL_FA_Q8_GQA_TILE", 0);
        g_ggml_sycl_graph_eviction_timeout = ggml_sycl_get_env("GGML_SYCL_GRAPH_EVICTION_TIMEOUT", 10);
        if (g_ggml_sycl_graph_eviction_timeout <= 0) {
            // A timeout <= 0 makes the sweep in ggml_backend_sycl_context::sycl_graph() run on
            // every call and evict the entry being looked up before it is used, so no graph
            // ever survives long enough to replay.
            GGML_LOG_WARN("%s: ignoring invalid GGML_SYCL_GRAPH_EVICTION_TIMEOUT=%d, using 10\n",
                          __func__, g_ggml_sycl_graph_eviction_timeout);
            g_ggml_sycl_graph_eviction_timeout = 10;
        }

        g_ggml_sycl_usm_system = ggml_sycl_get_env("GGML_SYCL_USM_SYSTEM", 0);

        g_ggml_sycl_usm_system = ggml_sycl_get_env("GGML_SYCL_USM_SYSTEM", 0);
        g_ggml_sycl_enable_host_pinned_mem =
            ggml_sycl_get_env("GGML_SYCL_ENABLE_HOST_PINNED_MEM", 1);

        g_ggml_sycl_host_pinned_mem_2g =
            ggml_sycl_get_env("GGML_SYCL_HOST_PINNED_MEM_2G", 0) & g_ggml_sycl_enable_host_pinned_mem;

        g_ggml_sycl_enable_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA", 0);
        g_ggml_sycl_debug_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_DEBUG", 0);
        g_ggml_sycl_sparse_fa_margin = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_MARGIN", 256);

        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");

        GGML_LOG_INFO("Build with Macros:\n");
#if GGML_SYCL_DNNL
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#if defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#if defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
        GGML_LOG_INFO("  GGML_SYCL_GRAPH_EVICTION_TIMEOUT: %d\n", g_ggml_sycl_graph_eviction_timeout);
#else
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

#if defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#if defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#if defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#else
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#else
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#endif

#if GGML_SYCL_DNNL
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_FORCE_VEC_STANDARD: %d\n",
            g_ggml_sycl_fa_force_vec_standard);
        GGML_LOG_INFO("  GGML_SYCL_FA_LARGE_GRF: %d (effective mode; 256-GRF FA variants compiled: %d)\n",
                      ggml_sycl_fa_large_grf_mode(), (int) ggml_sycl_fa_large_grf_variants());
        GGML_LOG_INFO("  GGML_SYCL_FA_Q8_GQA_TILE: %d\n",
            g_ggml_sycl_fa_q8_gqa_tile);
        GGML_LOG_INFO("  GGML_SYCL_FFN_FUSION: %d\n", ggml_sycl_ffn_fusion_enabled());

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);

#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);

        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#else
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_USM_SYSTEM: %d\n", g_ggml_sycl_usm_system);

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_REQUESTED: %s\n", GGML_SYCL_XMX_GATHER_REQUESTED);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_EFFECTIVE: %s\n", GGML_SYCL_XMX_GATHER_EFFECTIVE);
#ifndef GGML_SYCL_NO_XMX_GATHER
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
#else
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: XMX gather GEMMs disabled by compile flag\n");
#endif

#if defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);

#if defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#else
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#else
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_USM_SYSTEM: %d\n", g_ggml_sycl_usm_system);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_HOST_PINNED_MEM: %d\n", g_ggml_sycl_enable_host_pinned_mem);
        GGML_LOG_INFO("  GGML_SYCL_HOST_PINNED_MEM_2G: %d\n", g_ggml_sycl_host_pinned_mem_2g);

        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA: %d\n", g_ggml_sycl_enable_sparse_fa);
        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_DEBUG: %d\n", g_ggml_sycl_debug_sparse_fa);
        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_MARGIN: %d\n", g_ggml_sycl_sparse_fa_margin);

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/
        // Async USM allocation/free is also useful outside the graph path: it avoids the host waits in the reorder
        // staging path while preserving queue ordering semantics. Graph support still depends on the extension being
        // available, but it no longer needs to control the non-graph fast path.
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
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
                            ggml_sycl_info().device_count) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);

        if (g_all_sycl_device_count == 0) {
            GGML_LOG_WARN("%s: no usable SYCL GPU devices found; CPU fallback remains available\n", __func__);
        }

        initialized = true;
        g_sycl_loaded = g_all_sycl_device_count > 0;
        if (g_sycl_loaded) {
            ggml_backend_sycl_print_sycl_devices();
        }
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

inline void free_aligned_mem_host(void * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#else
    free(memblock);
#endif
}

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    bool is_usm_system;

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream, bool is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    ~ggml_backend_sycl_buffer_context() {
        if (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            if (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            else
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);
struct ggml_backend_sycl_device_context;
static ggml_backend_sycl_device_context * ggml_backend_sycl_device_context_from_device(ggml_backend_dev_t dev);
static ggml_backend_sycl_device_context * ggml_backend_sycl_device_context_from_backend(ggml_backend_t backend);
static void ggml_backend_sycl_record_failed_status(ggml_backend_sycl_device_context * dev_ctx);
static void ggml_backend_sycl_record_failed_exception(ggml_backend_sycl_device_context * dev_ctx, const sycl::exception & exc);
#ifdef GGML_SYCL_TESTING
static void ggml_backend_sycl_test_maybe_record_sync_failure(ggml_backend_sycl_device_context * dev_ctx);
#endif


static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
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
        return GGML_STATUS_SUCCESS;
    }

    if (g_ggml_sycl_enable_optimize) {
        // Quantized tensors carry backend-owned layout/reorder metadata.
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:{
                ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                break;
            }
            default:
                break;
        }
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

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_is_l0_discrete_gpu(int device) {
    return ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

// Copy size bytes from ptr_src to ptr_dst through a temporary host buffer, waiting
// for both transfers to finish. q_src and q_dst access the source and destination.
// Aborts if host allocation fails; exceptions from SYCL copies or waits propagate.
static void memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, void *ptr_dst,
                         const void *ptr_src, size_t size) {
    char *host_buf = (char *)malloc(size);
    if (host_buf == nullptr) {
        GGML_ABORT("memcpy_host_forward: failed to allocate host buffer of size %zu\n", size);
    }
    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
    free(host_buf);
}

// Copy size bytes between devices, waiting for completion. The configured copy
// mode selects Level Zero, SYCL peer access, or host staging. Unavailable direct
// paths and Level Zero command-list creation or copy failures use host staging.
// SYCL peer-copy failures and host allocation failures abort; other SYCL
// exceptions propagate.
static void dev2dev_memcpy(int device_dst, sycl::queue &q_dst, int device_src, sycl::queue &q_src, void *ptr_dst,
                    const void *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        const bool l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        if (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            if (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                if (r == ZE_RESULT_SUCCESS) {
                    return;
                }
            }
        }
    }
#endif

    if (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        if (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            return;
        }
    }

    // Host-staged copy
    if(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } else {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
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
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->data, src->data, size);

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

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    for (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<char*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
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
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

static const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool check_usm_system(int device, size_t size) {
    bool use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    if (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = false;
    }

    return use_usm_system;
}

inline void * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    const queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    bool use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    void * dev_ptr;
    if (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (void *)aligned_malloc_host(alignment, aligned_size);
        if (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            return nullptr;
        }
    } else {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        if (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          return nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

bool is_bmg_g31_arch(int device) {
    return ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

static size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    size_t max_alloc_size = dpct::get_current_device().get_max_mem_alloc_size();
    if (g_ggml_sycl_host_pinned_mem_2g) {
        return std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
    } else {
        ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
        int device = ctx->device;
        if(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_size = max_alloc_size*0.6;
        }
        return max_alloc_size;

    }
    GGML_UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_size(tensor)
        : ggml_nbytes(tensor);
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
    /* .get_name            = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
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
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
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
                /* .context  = */ new ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
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
        case GGML_TYPE_Q1_0:
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

        ggml_sycl_set_device(i);
        const queue_ptr stream = ctx->streams[i];
        char * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        if (!buf) {
            char err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
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
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
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
    return SYCL_BUFFER_ALIGNMENT;
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
    /* .get_name            = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(int main_device, const float * tensor_split) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_UNUSED(main_device);

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

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

// host buffer type

struct ggml_backend_sycl_device_context {
    int device;
    std::string name;
    std::string description;
    int op_offload_min_batch_size;
    // first failure since the last ggml_backend_sycl_consume_last_failure();
    // every graph submission, synchronize and event wait records into it
    std::atomic<int> pending_status = GGML_STATUS_SUCCESS;
    std::atomic<int> pending_cause  = GGML_SYCL_FAILURE_CAUSE_NONE;
    std::atomic<int> pending_raw_code = 0;
#ifdef GGML_SYCL_TESTING
    std::atomic<bool> test_sync_failure_once = false;
#endif
};

static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

static int ggml_backend_sycl_host_buffer_type_device(ggml_backend_buffer_type_t buft) {
    return static_cast<const ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
static void * ggml_backend_sycl_host_malloc(int device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    void * ptr = nullptr;
    try {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } catch (...) {
        ptr = nullptr;
    }
    if (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    return ptr;
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer->context == nullptr) {
        return;
    }
    if (g_ggml_sycl_enable_host_pinned_mem) {
        const int device = ggml_backend_sycl_host_buffer_type_device(buffer->buft);
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(buffer->context, q)));
    } else {
        free_aligned_mem_host((void *) buffer->context);
    }
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_buffer_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
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

static size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    if (g_ggml_sycl_enable_host_pinned_mem) {
        const int device = ggml_backend_sycl_host_buffer_type_device(buft);
        size_t max_alloc_size = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_size();
        if (g_ggml_sycl_host_pinned_mem_2g) {
            return std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
        } else {
            return max_alloc_size;
        }
    } else {
        return SIZE_MAX;
    }
}

static ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type_for_device(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    static std::vector<ggml_backend_buffer_type> buffer_types_host = [] {
        std::vector<ggml_backend_buffer_type> bufts(ggml_backend_sycl_get_device_count());
        for (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_buffer_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_buffer_type_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ nullptr,
            };
        }
        return bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (int) buffer_types_host.size());

    return &buffer_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    return ggml_backend_sycl_host_buffer_type_for_device(0);
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
#ifdef DEBUG_SYCL_POOL
        int    n_cached    = 0;
        size_t bytes_cached = 0;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        const auto slots = format_slots_in_alloc_order();
        if (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() const {
        std::string line;
        char buf[32];
        bool first = true;
        for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            if (buffer_pool[i].ptr == nullptr) {
                continue;
            }
            if (!first) {
                line += '/';
            }
            first = false;
            snprintf(buf, sizeof(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            line += buf;
        }
        return line;
    }
#endif

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

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (void *)ggml_sycl_malloc_device(look_ahead_size, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        if (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            return nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
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
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#if defined(GGML_SYCL_SUPPORT_VMM)
struct ggml_sycl_pool_vmm : public ggml_sycl_pool {
    static const size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    struct mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        void * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, int device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        if (pool_addr == 0) {
            return;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        for (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            if (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_size / (1024 * 1024), pool_size / (1024 * 1024),
                              (pool_size + reserve_size) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            void * map_ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_size, reserve_size,
                                    sycl::ext::oneapi::experimental::address_access_mode::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_size += reserve_size;
            ggml_sycl_memtrace_add(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_size);

#ifdef DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (unsigned long long) (pool_size/1024/1024),
                          (unsigned long long) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = reinterpret_cast<void *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<void *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

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
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
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
#if defined(GGML_SYCL_SUPPORT_VMM)
    if (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}


std::unique_ptr<ggml_sycl_fattn_kv_buffers> ggml_backend_sycl_context::new_fattn_kv_buffers(queue_ptr qptr, int device) {
    return std::unique_ptr<ggml_sycl_fattn_kv_buffers>(new ggml_sycl_fattn_kv_buffers(qptr, device));
}

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

static void init_argsort_indices_padded(
        int * idx,
        const int nrows,
        const int ncols_pad,
        const sycl::nd_item<1> & item_ct1) {
    const size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    idx[gid] = (int) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
static void argsort_f32_i32_global_pass(const float *            x,
                                        int *                    idx,
                                        const int                ncols,
                                        const int                nrows,
                                        const int                ncols_pad,
                                        const int                j,
                                        const int                k,
                                        const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols_pad;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols_pad);
    const int col = (int) (gid % (size_t) ncols_pad);
    const int ixj = col ^ j;

    if (ixj <= col || ixj >= ncols_pad) {
        return;
    }

    const size_t base  = (size_t) row * (size_t) ncols_pad;
    const size_t pos_a = base + (size_t) col;
    const size_t pos_b = base + (size_t) ixj;

    const int a = idx[pos_a];
    const int b = idx[pos_b];

    bool do_swap = false;

    if ((col & k) == 0) {
        if (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    } else {
        if (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = true;
        }
    }

    if (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

static void copy_argsort_indices_unpadded(const int *              idx_padded,
                                          int *                    dst,
                                          const int                nrows,
                                          const int                ncols,
                                          const int                ncols_pad,
                                          const sycl::nd_item<1> & item_ct1) {
    const size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    const size_t total = (size_t) nrows * (size_t) ncols;

    if (gid >= total) {
        return;
    }

    const int row = (int) (gid / (size_t) ncols);
    const int col = (int) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 queue_ptr stream, int device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);
    const size_t shared_mem = (size_t) ncols_pad * sizeof(int);
    const size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    if (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<int> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        int *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        const size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        const size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        for (int k = 2; k <= ncols_pad; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                if (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } else if (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } else {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        const size_t total   = (size_t) nrows * (size_t) ncols;
        const size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        return;
    }

    int nth = 1;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    if (nth > max_block_size)
        nth = max_block_size;

    const int tasks_per_thread = ncols_pad / nth;

    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, nrows, 1);

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

// Scan and block merge, shared by every launch shape below so a partitioned row uses the
// same insertion order as an unpartitioned one.
//
// src_map  != nullptr: report src_map[col] instead of col, so a merge pass can carry the
//                      original column index through.
// out_vals != nullptr: also emit the k winning values, for a later merge pass.
// swap01:              emit in the output order the single-pass path uses.
static void top_k_scan_merge_f32(
    const float *   src_vals,
    const int32_t * src_map,
    const int       begin,
    const int       end,
    const int       k,
    const int       block_size,
    float *         shared_vals,
    int *           shared_idx,
    float *         out_vals,
    int32_t *       out_idx,
    const bool      swap01,
    const sycl::nd_item<1> & item_ct1
) {
    const int tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    float * lv = shared_vals + tid;
    int * li = shared_idx + tid;

    for (int i = 0; i < k; i++) {
        lv[i * block_size] = -FLT_MAX;
        li[i * block_size] = -1;
    }

    // An empty slot (li == -1) is always taken regardless of value: comparing
    // against a -FLT_MAX sentinel instead left the slot at -1 for a genuine
    // -INFINITY score (e.g. a masked cell from a sparse-attention indexer),
    // and the CPU ggml_set_rows consuming that -1 index aborted its range
    // check. The (k-1)-th slot is read fresh each time rather than cached in
    // a register so an unfilled tail is still detected.
    for (int col = begin + tid; col < end; col += block_size) {
        float val = src_vals[col];

        if (li[(k - 1) * block_size] < 0 || val > lv[(k - 1) * block_size]) {
            int pos = k - 1;
            while (pos > 0 && val > lv[(pos - 1) * block_size]) {
                pos--;
            }

            for (int i = k - 1; i > pos; i--) {
                lv[i * block_size] = lv[(i - 1) * block_size];
                li[i * block_size] = li[(i - 1) * block_size];
            }
            lv[pos * block_size] = val;
            li[pos * block_size] = src_map ? src_map[col] : col;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (tid != 0) {
        return;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    float * fv = shared_vals + (size_t) k * block_size;
    int * fi = shared_idx + (size_t) k * block_size;

    for (int i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    for (int t = 0; t < block_size; t++) {
        for (int i = 0; i < k; i++) {
            int idx = shared_idx[i * block_size + t];

            if (idx < 0) {
                // Lane t saw fewer than k columns; its list is packed at the front,
                // so nothing further from this lane is usable.
                break;
            }

            float val = shared_vals[i * block_size + t];

            if (fi[k - 1] < 0 || val > fv[k - 1]) {
                // An open final slot (fi[k-1] < 0) is always taken, even for a
                // genuine -INFINITY score (see the scan-phase comment above);
                // otherwise the lane's list is sorted descending, so once an
                // entry loses to a full final k-th best every later entry from
                // this lane loses too, and can be skipped.
                int pos = k - 1;
                while (pos > 0 && val > fv[pos - 1]) {
                    pos--;
                }

                for (int j = k - 1; j > pos; j--) {
                    fv[j] = fv[j - 1];
                    fi[j] = fi[j - 1];
                }
                fv[pos] = val;
                fi[pos] = idx;
            } else {
                break;
            }
        }
    }

    if (out_vals) {
        for (int i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    for (int i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    if (swap01 && k > 1) {
        int32_t temp = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = temp;
    }
}

static void top_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    const float * src,
    int32_t * dst_indices,
    const int64_t ncols,
    const int64_t nrows,
    const int k,
    dpct::queue_ptr main_stream
) {
    // A row is scanned by exactly one work-group, so a vocabulary-sized row leaves the
    // rest of the device idle. What the scan is short of is memory requests in flight,
    // not bandwidth or per-request latency, so lanes in flight is the lever: split the
    // row across independent work-groups, have each emit its partition's top-k, and
    // merge those nsplit*k candidates in a second launch.
    //
    // split_block trades parallelism against SLM residency. Its cost is
    // (split_block + 1) * k * 8 bytes of SLM per group, so at the k <= 32 ceiling 128
    // lanes need about 33 KB, which leaves a single resident group per Xe-core. Revisit
    // if the supported k ever grows.
    constexpr int split_block = 128;
    constexpr int max_splits  = 128;
    constexpr int min_cols    = 8192;

    int nsplit = 1;
    if (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        const int64_t want = ncols / split_block;
        nsplit = (int) (want > max_splits ? max_splits : want);
    }

    if (nsplit > 1) {
        const int nchunk = (int) ((ncols + nsplit - 1) / nsplit);
        const size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<float>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        float *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        const sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int grp = item_ct1.get_group(0);
                    const int row = grp / nsplit;
                    const int part = grp % nsplit;

                    const int begin = part * nchunk;
                    int end = begin + nchunk;
                    if (end > (int) ncols) {
                        end = (int) ncols;
                    }

                    top_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, false, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    const int row = item_ct1.get_group(0);
                    const size_t off = (size_t) row * nsplit * k;

                    top_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
                });
        });

        return;
    }

    const int block_size = 128;

    const sycl::range<1> block_dims(block_size);
    const sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_vals(sycl::range<1>((block_size + 1) * k), cgh);
        sycl::local_accessor<int, 1> shared_idx(sycl::range<1>((block_size + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                const int row = item_ct1.get_group(0);

                if (row >= nrows) return;

                top_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (int) ncols, k, block_size,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
            });
    });
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

#if GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    if (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } else {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*sizeof(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        return;
    }
#endif

    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, src1);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1->data + src1_padded_row_size
                                         : src1_as_f16.get();

        // dequantize inside the GEMM instead of writing the f16 weights out and reading them back
        if (src0->type != GGML_TYPE_F16 &&
            ggml_sycl_fused_dequant_gemm_f16(src0->type, src0_dd_i, src1_ptr, dst_dd_i, row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
            return;
        }

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, src0);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#if GGML_SYCL_DNNL
        if (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        else
#endif
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
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

        {
#if GGML_SYCL_DNNL
            const int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            const bool use_mkl_direct = gemm_flops < 256 * 256 * 256;
            if (g_ggml_sycl_enable_dnn && !use_mkl_direct && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<float>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
            }
            else
#endif
            {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
            }
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
                         main_stream, ctx.device, ctx.pool());
}

static void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    const int k = dst->ne[0];
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    if (k <= SYCL_TOP_K_SCAN_MERGE_MAX_K) {
        top_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    } else {
        ggml_sycl_top_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    }
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

static void tri_f32_sycl(
    const float * src,
    float * dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    const size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        const int64_t idx = (int64_t) tid[0];

        const int64_t i0 = idx % ne0;
        const int64_t t1 = idx / ne0;
        const int64_t i1 = t1 % ne1;

        bool keep = false;
        switch (ttype) {
            case GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); break;
            case GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); break;
            case GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); break;
            case GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); break;
            default: keep = false; break;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

static void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(src0->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
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

static void ggml_sycl_set_peer_access(const int n_tokens, [[maybe_unused]] int main_device) {
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
            dev[i].src0_dd = (char *) src0->data;
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (float *) src1->data;
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
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
            dev[i].dst_dd = (float *) dst->data;
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
                float * src1_ddf_i = (float *) ((char *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type));
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
                            const char * src1_ddf_i_source = (const char *) src1_extra->data_device[ctx.device] +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_size(src1->type))));
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

static void ggml_sycl_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows_back(ctx, dst);
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
#if GGML_SYCL_DNNL
    const size_t       type_size_src0 = ggml_type_size(src0->type);
#endif
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

#if GGML_SYCL_DNNL
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
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, src1);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# else
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type, src1);
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
    if (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
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
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
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
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
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
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    return false;
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return !g_ggml_sycl_prioritize_dmmv;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
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

inline bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
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

static bool ggml_sycl_supports_reorder_esimd(enum ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
#else
    GGML_UNUSED(type);
    return false;
#endif
}

static bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
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
        case GGML_TYPE_TQ3_1S:
        case GGML_TYPE_TQ4_1S:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
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
        void * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        return ptr;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    return ggml_sycl_malloc_device(size, *stream);
}

static inline void sycl_ext_free(dpct::queue_ptr stream, void * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        return;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
struct sycl_reorder_temp_buffer {
    void *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        if (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            if (ptr) {
                host_fallback = true;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        if (!ptr) {
            return;
        }
        if (host_fallback) {
            sycl::free(ptr, *stream);
        } else {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator bool() const { return ptr != nullptr; }

    sycl_reorder_temp_buffer(const sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(const sycl_reorder_temp_buffer &) = delete;

private:
    bool host_fallback = false;
};

static bool reorder_qw_q4_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

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
    return true;
}

static bool reorder_qw_q8_0(uint8_t * data_device, const int ncols, const int nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q8_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q8_0) == 0));
    int offset_blks = offset / sizeof(block_q8_0);
    auto qs_ptr = data_device + offset_blks * QK8_0;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / sizeof(block_q8_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q8_0* x = (const block_q8_0*)tmp_buf;
            const int ib = i;

            for (int j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q4_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q4_K) == 0);

    const int nblocks = size / sizeof(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

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
    return true;
}

// Reorder each expert slice into a self-contained SoA layout.
static bool reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q4_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q4_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q4_K * x    = (const block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

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
    return true;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
static bool reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q5_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q5_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q5_K * x    = (const block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
static bool reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % sizeof(block_q6_K) == 0);
    const int    blocks_per_expert = (int) (expert_bytes / sizeof(block_q6_K));
    const size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    const int total_blocks = blocks_per_expert * (int) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        const int          gb   = gb_;
        const int          e    = gb / blocks_per_expert;
        const int          ib   = gb % blocks_per_expert;
        const block_q6_K * x    = (const block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * ql_ptr     = base;
        auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        for (int j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q2_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q2_K) == 0);

    const int nblocks = size / sizeof(block_q2_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *        qs_ptr     = data_device;
    auto *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q2_K * x  = (const block_q2_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q3_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q3_K) == 0);

    const int nblocks = size / sizeof(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       qs_ptr     = data_device;
    auto *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    auto *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q3_K * x  = (const block_q3_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        for (int j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q5_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q5_K) == 0);

    const int nblocks = size / sizeof(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    auto * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q5_K * x  = (const block_q5_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    return true;
}

static bool reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q6_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q6_K) == 0);

    const int nblocks = size / sizeof(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        return false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

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
    return true;
}

static bool reorder_qw(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    if (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        switch (src0->type) {
            case GGML_TYPE_Q4_K:
                return reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q5_K:
                return reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            case GGML_TYPE_Q6_K:
                return reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            default:
                return false;
        }
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            return reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q8_0:
            return reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        case GGML_TYPE_Q2_K:
            return reorder_qw_q2_k(data_device, size, 0, stream);
        case GGML_TYPE_Q3_K:
            return reorder_qw_q3_k(data_device, size, 0, stream);
        case GGML_TYPE_Q4_K:
            return reorder_qw_q4_k(data_device, size, 0, stream);
        case GGML_TYPE_Q5_K:
            return reorder_qw_q5_k(data_device, size, 0, stream);
        case GGML_TYPE_Q6_K:
            return reorder_qw_q6_k(data_device, size, 0, stream);
        default:
            return false;
    }
}

static bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst) {
    return g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
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

    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
static void opt_for_reorder_id(ggml_backend_sycl_context * ctx, const ggml_tensor * src0) {
    if (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        return;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K) {
        return;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;
    }
    if (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;
    }
}


static bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    const int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    return ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1 &&
           !ggml_type_is_turbo(src0->type);
}

static bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE && !ggml_type_is_turbo(src0->type);
}

static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    if (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        return;
    }

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


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    if (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        bool use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        if (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
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
    } else if (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
static bool ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, const ggml_tensor * wu,
                                             const ggml_tensor * wg, const ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream = ctx.stream();
    const int src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_q8_1>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->data, wu->data,
                                             src1_ddq, (float *) glu->data, (int) ne00, (int) wu->ne[1],
                                             (int) ne11,
                                             /*stride_col_y=*/src1_padded_cols / QK8_1,
                                             /*stride_col_dst=*/(int) glu->ne[0], stream);
}

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
static bool ggml_sycl_mul_mat_glu_mmvq_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_sycl_can_fuse(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        return false;
    }

    ggml_tensor *       glu  = cgraph->nodes[node_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    const ggml_tensor * wu   = up->src[0];
    const ggml_tensor * wg   = gate->src[0];
    const ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    if (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        return false;
    }

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    if (g_ggml_sycl_prioritize_dmmv) {
        return false;
    }

    // quant pairs the reorder kernel cannot serve (mixed gate/up types) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    if (wg->type != wu->type || (wu->type != GGML_TYPE_Q4_K && wu->type != GGML_TYPE_Q5_K)) {
        return ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, up, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    if (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        return false;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    const auto * extra_u = static_cast<const ggml_tensor_extra_gpu *>(wu->extra);
    const auto * extra_g = static_cast<const ggml_tensor_extra_gpu *>(wg->extra);
    if (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        return false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    const int64_t ne00 = wu->ne[0];
    const int64_t ne11 = act->ne[1];

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((const float *) act->data, src1_ddq, (int) ne00, (int) ne11,
                                                          src1_padded_cols, stream);

    return ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (float *) glu->data, (int) ne00, (int) wu->ne[1], (int) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (int) sizeof(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(int) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
static int ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * node = cgraph->nodes[node_idx];
    if (ggml_sycl_info().device_count != 1 || node->type != GGML_TYPE_F32 ||
        node->src[0]->type != GGML_TYPE_F32 || node->src[0]->ne[0] >= 1024) {
        return 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    int           count = 0;
    int           last  = node_idx;
    float         eps0;
    memcpy(&eps0, node->op_params, sizeof(float));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * ab = (const char *) a->data;
        const char * bb = (const char *) b->data;
        return ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    for (int j = node_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        if (ggml_is_empty(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;  // not a launch; cannot break a run of adjacent norms
        }
        if (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, node) ||
            !ggml_are_same_shape(nj->src[0], node->src[0])) {
            break;  // any other launch ends the run
        }
        bool same_nb = true;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (nj->nb[d] != node->nb[d] || nj->src[0]->nb[d] != node->src[0]->nb[d]) {
                same_nb = false;
                break;
            }
        }
        if (!same_nb) {
            break;  // one nb[] stride set is shared by the whole batch
        }
        float epsj;
        memcpy(&epsj, nj->op_params, sizeof(float));
        if (epsj != eps0) {
            break;  // eps mismatch ends the run
        }
        bool indep = true;
        for (int k = 0; k < count; ++k) {
            if (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = false;
                break;
            }
        }
        if (!indep) {
            break;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        last           = j;
    }
    if (count < 2) {
        return 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    return last - node_idx;
}

__dpct_inline__ static void k_copy_src1_to_contiguous(
    const char *__restrict__ src1_original, char *__restrict__ src1_contiguous,
    const mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    const sycl::nd_item<3> &item_ct1) {
    const int32_t src1_row = item_ct1.get_group(2);

    const int32_t iid1 = row_mapping[src1_row].i2;
    const int32_t id   = row_mapping[src1_row].i1;

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

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

__dpct_inline__ static void k_zero_dst_rows(
    char *__restrict__ dst_original,
    const mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, const sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = 0.0f;
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
static bool ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)
{
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    if (ne12 != 1) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) return false;
    if (!ggml_is_contiguous(src1)) return false;

    const int64_t n_ids_per_group = ids->ne[0];
    if (ids->ne[1] != 1) return false;
    if (ne11 != 1 && ne11 != n_ids_per_group) return false;

    const queue_ptr stream           = ctx.stream();
    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
    const int       n_experts_used   = (int) n_ids_per_group;
    const int       nrows            = (int) src0->ne[1];

    // Lazily reorder the (Q4_K) expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    const ggml_tensor_extra_gpu * src0_extra =
        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
    char * src1_ddq = src1_q8_alloc.get();
    if (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) ne11,
            src1_padded_cols, stream);
    } else {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (const float *) src1->data, src1_ddq, (int) ne10, (int) ne11,
            src1_padded_cols, stream);
    }

    const size_t bytes_per_qrow = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
    const size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    if (use_reorder) {
        return ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
            (float *) dst->data, (int) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    return ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
        (float *) dst->data, (int) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
static void mmid_counting_sort_rows(
        const ggml_tensor * ids, const char * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src,
        std::vector<mmid_row_mapping> & skipped_row_dst) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i == -1 || (row_id_i >= 0 && row_id_i < n_as));
            if (row_id_i == -1) {
                continue;
            }
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    for (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    skipped_row_dst.clear();
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i == -1 || (row_id_i >= 0 && row_id_i < n_as));

            // a skipped slot has no expert, its dst row is zeroed instead
            if (row_id_i == -1) {
                skipped_row_dst.push_back({(int32_t) id, (int32_t) iid1});
                continue;
            }

            // find and validate the next free row for a given expert (row_id_i)
            const int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
    routed_row_src.resize(expert_row_offsets[n_as]);
}

static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) try {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    const ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    if (ne12 == 1) {
        if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            return;
        }
    }

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host, ctx.mmid_skipped_row_host and ctx.mmid_tile_schedule_host
    // are drained before we refill them: the routed-row and skipped-row copies and the grouped GEMM's
    // tile schedule copy are all async, so removing this wait would let the next node overwrite a
    // buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

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
                GGML_ASSERT(i02 == -1 || (i02 >= 0 && i02 < n_as));

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

            if (i02 == -1) { // skipped slot
                SYCL_CHECK(CHECK_TRY_ERROR(
                        stream->memset(dst_original + i1*nb1 + i2*nb2, 0, ne0*sizeof(float))));
                continue;
            }

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        const int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne0);

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_row_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_row_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;
        // the slot/token pairs of the skipped slots, whose dst rows are zeroed. Lives in ctx like
        // routed_row_src: the async memcpy below reads it, and the wait at the top of this function
        // drains that before the next node reuses it, so no extra wait is needed here
        std::vector<mmid_row_mapping> & skipped_row_dst = ctx.mmid_skipped_row_host;

        mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src, skipped_row_dst);

        const int64_t n_valid_rows = (int64_t) routed_row_src.size();

        ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_rows);
        if (n_valid_rows > 0) { // every slot skipped leaves nothing to copy
            SYCL_CHECK(CHECK_TRY_ERROR(
                    stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), n_valid_rows*sizeof(mmid_row_mapping))));
        }

        const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        if (!skipped_row_dst.empty()) {
            ggml_sycl_pool_alloc<mmid_row_mapping> dev_skipped(ctx.pool(), skipped_row_dst.size());
            SYCL_CHECK(CHECK_TRY_ERROR(
                    stream->memcpy(dev_skipped.get(), skipped_row_dst.data(), skipped_row_dst.size()*sizeof(mmid_row_mapping))));

            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, skipped_row_dst.size());
            stream->submit([&](sycl::handler &cgh) {
                mmid_row_mapping *__restrict dev_skipped_get = dev_skipped.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_zero_dst_rows(dst_original, dev_skipped_get, ne0, nb1, nb2, item_ct1);
                    });
            });
            // dev_skipped returns to the pool at scope exit; a later allocation reuses it only
            // through the same in-order stream, after this kernel
        }

        if (n_valid_rows > 0) {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_valid_rows);
            stream->submit([&](sycl::handler &cgh) {
                char *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_src1_to_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_row_mapping_get,
                            ne11, ne10, nb11, nb12,
                            item_ct1);
                    });
            });
        }

        bool grouped = false;
        if (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && dst->op_params[0] == GGML_PREC_DEFAULT &&
            // ggml_prec_set_src(dst, GGML_PREC_F32, 1) stores its requirement in op_params[3]
            // (ggml_set_op_params_i32(a, 2 + idx, prec) with idx=1), not op_params[0]. Nodes
            // that request it (e.g. Mistral MoE's expert-down projection in llama-graph.cpp,
            // to avoid overflow past +-65504) must not take this path: grouped_gemm_pack_b
            // casts every activation to sycl::half.
            dst->op_params[3] != GGML_PREC_F32 &&
            nb11 == sizeof(float)*ne10 && nb1 == sizeof(float)*ne0) {
            grouped = ggml_sycl_grouped_dequant_gemm_f16(src0->type, src0_original, nb02,
                                                         (const float *) src1_contiguous.get(), (float *) dst_contiguous.get(),
                                                         // skipped slots (-1 ids) own no slice, so the
                                                         // slices cover n_valid_rows, not n_routed_rows
                                                         expert_row_offsets.data(), n_as, ne01, ne10, n_valid_rows,
                                                         ctx.mmid_tile_schedule_host, ctx.pool(), stream);
        }

        for (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            const int64_t num_src1_rows = expert_row_counts[i02];

            if (num_src1_rows == 0) {
                continue;
            }

            const int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == sizeof(float)*ne10);
            GGML_ASSERT(nb1 == sizeof(float)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        if (n_valid_rows > 0) {
            sycl::range<3> block_dims(1, 1, std::min((unsigned int)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_valid_rows);
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

static void ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

static void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

static void ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

static void ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

static void ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
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

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, ggml_backend_sycl_device_context * dev_ctx, struct ggml_tensor * dst) try {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    if (!g_sycl_loaded) return false;

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
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
        case GGML_OP_GET_ROWS_BACK:
            ggml_sycl_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
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
                case GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
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
                case GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
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
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
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
        case GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
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
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
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
            ggml_sycl_op_top_k(ctx, dst);
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
        case GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            break;
        case GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
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
        case GGML_OP_TURBO_WHT:
            ggml_sycl_op_turbo_wht(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
} catch (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    ggml_backend_sycl_record_failed_exception(dev_ctx, e);
    return false;
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

bool sycl_get_mem_info(int device, size_t * free, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    MemoryAPIType mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_RESULT_SUCCESS ?
        (MemoryAPIType) g_ggml_sycl_get_mem_api : MemoryAPIType::MEMORY_API_TYPE_SYCL;
#else
    mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;
#endif
    bool res = get_memory_size(dpct::dev_mgr::instance().get_device(device),
        *free, *total, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *total, *free);
    return res;
}

void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    if (!sycl_get_mem_info(device, free, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *free, *total);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
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

    if (sycl_ctx->private_queue) {
        try {
            sycl_ctx->private_queue->wait_and_throw();
        } catch (sycl::exception const & exc) {
            GGML_LOG_ERROR("%s: SYCL private queue wait failed: %s\n", __func__, exc.what());
        }
    }
    delete sycl_ctx;
    delete backend;
}

static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    try {
        GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (stream)->memcpy((char *)tensor->data + offset, data, size)));
    } catch (sycl::exception const & exc) {
        ggml_backend_sycl_record_failed_exception(dev_ctx, exc);
        GGML_LOG_ERROR("%s: SYCL set_tensor_async failed: %s\n", __func__, exc.what());
    }
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    try {
        GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            data, (const char *)tensor->data + offset, size)));
    } catch (sycl::exception const & exc) {
        ggml_backend_sycl_record_failed_exception(dev_ctx, exc);
        GGML_LOG_ERROR("%s: SYCL get_tensor_async failed: %s\n", __func__, exc.what());
    }
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *src,
                                               ggml_tensor *dst) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);
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
        try {
            const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
            SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
                dst->data, src->data, ggml_nbytes(dst))));
            return true;
        } catch (sycl::exception const & exc) {
            ggml_backend_sycl_record_failed_exception(dev_ctx, exc);
            GGML_LOG_ERROR("%s: SYCL cpy_tensor_async failed: %s\n", __func__, exc.what());
            return false;
        }
    }

    return false;
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);
 
    try {
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
#ifdef GGML_SYCL_TESTING
        ggml_backend_sycl_test_maybe_record_sync_failure(dev_ctx);
#endif
    } catch (sycl::exception const & exc) {
        ggml_backend_sycl_record_failed_exception(dev_ctx, exc);
        GGML_LOG_ERROR("%s: SYCL synchronize failed: %s\n", __func__, exc.what());
    }
}

static bool ggml_sycl_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
static int ggml_sycl_try_gdn_cache_fusion(const ggml_cgraph * cgraph, int node_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }

    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_sycl_try_fuse_ffn_swiglu(
    ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);

static ggml_status ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_backend_sycl_device_context * dev_ctx, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_sycl_is_view_or_noop(node)) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const int nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        if (nodes_to_skip != 0) {
            i += nodes_to_skip;
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
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            const int gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            if (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, node, fused_state_cpy);
                i += gdn_nodes_to_skip;
                continue;
            }
        }
        // Fused dense FFN consumes this MUL_MAT, the next MUL_MAT and the GLU.
        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_try_fuse_ffn_swiglu(*sycl_ctx, cgraph, i)) {
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        if (node->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }
        if (node->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(node) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1]);
            i++;
            continue;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        if (node->op == GGML_OP_L2_NORM) {
            const int l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            if (l2_batch_skip > 0) {
                i += l2_batch_skip;
                continue;
            }
        }

        if (node->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            continue;
        }

        if (node->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, node, nullptr, cgraph->nodes[i + 1]);
            i++;
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_mmvq_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            continue;
        }

        bool ok = ggml_sycl_compute_forward(*sycl_ctx, dev_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op failed or unsupported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static bool ggml_sycl_rope_fusion_profile_enabled() {
    const char * value = getenv("GGML_SYCL_ROPE_FUSION_PROFILE");
    return value != nullptr && strcmp(value, "1") == 0;
}

struct ggml_sycl_rope_fusion_profile {
    std::atomic<uint64_t> graph_calls             = 0;
    std::atomic<uint64_t> rope_nodes              = 0;
    std::atomic<uint64_t> structural_matches      = 0;
    std::atomic<uint64_t> eligible_matches        = 0;
    std::atomic<uint64_t> decode_eligible_matches = 0;
    std::atomic<uint64_t> batched_eligible_matches = 0;
    std::atomic<uint64_t> eligible_index_rows     = 0;
    std::atomic<uint64_t> reject_ne3              = 0;
    std::atomic<uint64_t> reject_set_rows_type    = 0;
    std::atomic<uint64_t> reject_index_type       = 0;
    std::atomic<uint64_t> reject_view_shape       = 0;
    std::atomic<uint64_t> reject_rope_mode        = 0;

    ~ggml_sycl_rope_fusion_profile() {
        if (!ggml_sycl_rope_fusion_profile_enabled()) {
            return;
        }
        fprintf(
            stderr,
            "GGML_SYCL_ROPE_FUSION_PROFILE: graph_calls=%" PRIu64 " rope_nodes=%" PRIu64
            " structural_matches=%" PRIu64 " eligible_matches=%" PRIu64
            " decode_eligible_matches=%" PRIu64 " batched_eligible_matches=%" PRIu64
            " eligible_index_rows=%" PRIu64 " reject_ne3=%" PRIu64
            " reject_set_rows_type=%" PRIu64 " reject_index_type=%" PRIu64
            " reject_view_shape=%" PRIu64 " reject_rope_mode=%" PRIu64 "\n",
            graph_calls.load(), rope_nodes.load(), structural_matches.load(), eligible_matches.load(),
            decode_eligible_matches.load(), batched_eligible_matches.load(), eligible_index_rows.load(),
            reject_ne3.load(), reject_set_rows_type.load(), reject_index_type.load(),
            reject_view_shape.load(), reject_rope_mode.load());
    }
};

static ggml_sycl_rope_fusion_profile & ggml_sycl_rope_fusion_profile_data() {
    static ggml_sycl_rope_fusion_profile profile;
    return profile;
}

static bool ggml_sycl_rope_view_set_rows_eligible(
        const ggml_cgraph * cgraph, int node_idx, ggml_sycl_rope_fusion_profile & profile) {
    const ggml_tensor * rope     = cgraph->nodes[node_idx + 0];
    const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
    const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

    if (rope->src[0]->ne[3] != 1) {
        profile.reject_ne3.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        profile.reject_set_rows_type.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        profile.reject_index_type.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        profile.reject_view_shape.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_MROPE) {
        profile.reject_rope_mode.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

// FFN fusion eligibility profiler.
//
// Upstream CUDA fuses the dense feed-forward block as
// (ffn_up MUL_MAT, ffn_gate MUL_MAT, GLU) into a single kernel, optionally
// absorbing bias and scale. The fork's build_ffn emits exactly that shape for
// LLM_FFN_SILU with LLM_FFN_PAR: two MUL_MAT nodes feeding ggml_swiglu_split
// (src/llama-graph.cpp:1336). SYCL currently fuses only the MoE MUL_MAT_ID path.
//
// This counts how many production sites would actually qualify before any kernel
// is written. The ROPE fusion effort proved the value of measuring first: it was
// structurally matched but 0% production-eligible because SET_ROWS destinations
// were q8_0.
static bool ggml_sycl_ffn_fusion_profile_enabled() {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_FFN_FUSION_PROFILE", 0) != 0;
    return enabled;
}

struct ggml_sycl_ffn_fusion_profile {
    std::atomic<uint64_t> graph_calls              = 0;
    std::atomic<uint64_t> glu_nodes                = 0;
    std::atomic<uint64_t> structural_matches       = 0;
    std::atomic<uint64_t> eligible_matches         = 0;
    std::atomic<uint64_t> decode_eligible_matches  = 0;
    std::atomic<uint64_t> batched_eligible_matches = 0;
    std::atomic<uint64_t> reject_not_mul_mat       = 0;
    std::atomic<uint64_t> reject_src_mismatch      = 0;
    std::atomic<uint64_t> reject_glu_op            = 0;
    std::atomic<uint64_t> reject_type              = 0;
    std::atomic<uint64_t> reject_shape             = 0;
    std::atomic<uint64_t> reject_non_contig        = 0;

    ~ggml_sycl_ffn_fusion_profile() {
        if (!ggml_sycl_ffn_fusion_profile_enabled()) {
            return;
        }
        fprintf(
            stderr,
            "GGML_SYCL_FFN_FUSION_PROFILE: graph_calls=%" PRIu64 " glu_nodes=%" PRIu64
            " structural_matches=%" PRIu64 " eligible_matches=%" PRIu64
            " decode_eligible_matches=%" PRIu64 " batched_eligible_matches=%" PRIu64
            " reject_not_mul_mat=%" PRIu64 " reject_src_mismatch=%" PRIu64
            " reject_glu_op=%" PRIu64 " reject_type=%" PRIu64
            " reject_shape=%" PRIu64 " reject_non_contig=%" PRIu64 "\n",
            graph_calls.load(), glu_nodes.load(), structural_matches.load(),
            eligible_matches.load(), decode_eligible_matches.load(),
            batched_eligible_matches.load(), reject_not_mul_mat.load(),
            reject_src_mismatch.load(), reject_glu_op.load(), reject_type.load(),
            reject_shape.load(), reject_non_contig.load());
    }
};

static ggml_sycl_ffn_fusion_profile & ggml_sycl_ffn_fusion_profile_data() {
    static ggml_sycl_ffn_fusion_profile profile;
    return profile;
}

// Mirrors the upstream CUDA ggml_cuda_should_fuse_mul_mat predicate: both
// producers must be MUL_MAT feeding one GLU, sharing the same activation input,
// with matching quantized weight types and contiguous, shape-identical outputs.
static bool ggml_sycl_ffn_fusion_eligible(
    const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu,
    ggml_sycl_ffn_fusion_profile & profile) {
    if (up->op != GGML_OP_MUL_MAT || gate->op != GGML_OP_MUL_MAT) {
        profile.reject_not_mul_mat.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // Both projections must consume the same normalized activation, otherwise the
    // fused kernel cannot share its load of the input.
    if (up->src[1] != gate->src[1]) {
        profile.reject_src_mismatch.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        profile.reject_glu_op.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (up->src[0]->type != gate->src[0]->type || up->type != gate->type) {
        profile.reject_type.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!ggml_are_same_shape(up, gate)) {
        profile.reject_shape.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!ggml_is_contiguous(up) || !ggml_is_contiguous(gate)) {
        profile.reject_non_contig.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

static void ggml_sycl_profile_ffn_fusion(const ggml_cgraph * cgraph) {
    if (!ggml_sycl_ffn_fusion_profile_enabled()) {
        return;
    }

    ggml_sycl_ffn_fusion_profile & profile = ggml_sycl_ffn_fusion_profile_data();
    profile.graph_calls.fetch_add(1, std::memory_order_relaxed);
    for (int node_idx = 0; node_idx < cgraph->n_nodes; ++node_idx) {
        if (cgraph->nodes[node_idx]->op == GGML_OP_GLU) {
            profile.glu_nodes.fetch_add(1, std::memory_order_relaxed);
        }
        if (node_idx + 2 >= cgraph->n_nodes ||
            !ggml_can_fuse_subgraph(
                cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU },
                { node_idx + 2 })) {
            continue;
        }
        const ggml_tensor * first  = cgraph->nodes[node_idx];
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu    = cgraph->nodes[node_idx + 2];
        // ggml_swiglu_split(gate, up) passes gate as src[0] and up as src[1]; the
        // graph may emit either producer first, so accept both orderings.
        const bool ordered = glu->src[0] == first && glu->src[1] == second;
        const bool swapped = glu->src[0] == second && glu->src[1] == first;
        if (!ordered && !swapped) {
            continue;
        }
        profile.structural_matches.fetch_add(1, std::memory_order_relaxed);

        const ggml_tensor * gate = ordered ? first : second;
        const ggml_tensor * up   = ordered ? second : first;
        if (!ggml_sycl_ffn_fusion_eligible(up, gate, glu, profile)) {
            continue;
        }
        profile.eligible_matches.fetch_add(1, std::memory_order_relaxed);
        if (up->ne[1] == 1) {
            profile.decode_eligible_matches.fetch_add(1, std::memory_order_relaxed);
        } else {
            profile.batched_eligible_matches.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// Fuses (ffn_gate MUL_MAT, ffn_up MUL_MAT, GLU) into one kernel for
// single-token dense Q4_K decode. Default ON as of the P6.2 re-campaign
// against this fixed kernel (PR #38 carries the promotion evidence; the
// docs/research/sycl/ffn-fusion-campaign-2026-07-26/ directory predates the fix
// and is historical only). Set GGML_SYCL_FFN_FUSION=0 to opt out.
static bool ggml_sycl_ffn_fusion_enabled() {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_FFN_FUSION", 1) != 0;
    return enabled;
}

// Attempt to execute (ffn_gate MUL_MAT, ffn_up MUL_MAT, GLU) at node_idx as a
// single fused kernel. Returns true if the three nodes were consumed.
//
// ggml_can_fuse_subgraph with {node_idx + 2} as the only declared output is what
// makes this safe: it establishes that the two MUL_MAT results feed nothing
// outside the subgraph, so skipping their individual writes cannot lose data.
static bool ggml_sycl_try_fuse_ffn_swiglu(
        ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_sycl_ffn_fusion_enabled()) {
        return false;
    }
    if (node_idx + 2 >= cgraph->n_nodes ||
        !ggml_can_fuse_subgraph(
            cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, { node_idx + 2 })) {
        return false;
    }

    ggml_tensor * first  = cgraph->nodes[node_idx];
    ggml_tensor * second = cgraph->nodes[node_idx + 1];
    ggml_tensor * glu    = cgraph->nodes[node_idx + 2];

    // ggml_swiglu_split(gate, up) passes gate as src[0] and up as src[1]; accept
    // either emission order for the two producers.
    const bool ordered = glu->src[0] == first && glu->src[1] == second;
    const bool swapped = glu->src[0] == second && glu->src[1] == first;
    if (!ordered && !swapped) {
        return false;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return false;
    }

    ggml_tensor * gate = ordered ? first : second;
    ggml_tensor * up   = ordered ? second : first;

    if (gate->src[1] != up->src[1]) {
        return false;
    }
    if (gate->src[0]->type != up->src[0]->type || gate->type != up->type) {
        return false;
    }
    if (!ggml_are_same_shape(gate, up) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up)) {
        return false;
    }

    return ggml_sycl_mul_mat_vec_q_fused_swiglu(ctx, gate->src[0], up->src[0], glu);
}

static void ggml_sycl_profile_rope_fusion(const ggml_cgraph * cgraph) {
    if (!ggml_sycl_rope_fusion_profile_enabled()) {
        return;
    }

    ggml_sycl_rope_fusion_profile & profile = ggml_sycl_rope_fusion_profile_data();
    profile.graph_calls.fetch_add(1, std::memory_order_relaxed);
    for (int node_idx = 0; node_idx < cgraph->n_nodes; ++node_idx) {
        if (cgraph->nodes[node_idx]->op == GGML_OP_ROPE) {
            profile.rope_nodes.fetch_add(1, std::memory_order_relaxed);
        }
        if (node_idx + 2 >= cgraph->n_nodes ||
            !ggml_can_fuse_subgraph(
                cgraph, node_idx, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { node_idx + 2 }) ||
            !ggml_check_edges(cgraph, node_idx, { { 1, 0, 0 }, { 2, 0, 1 } })) {
            continue;
        }
        profile.structural_matches.fetch_add(1, std::memory_order_relaxed);
        if (!ggml_sycl_rope_view_set_rows_eligible(cgraph, node_idx, profile)) {
            continue;
        }
        const uint64_t index_rows = ggml_nelements(cgraph->nodes[node_idx + 2]->src[1]);
        profile.eligible_matches.fetch_add(1, std::memory_order_relaxed);
        profile.eligible_index_rows.fetch_add(index_rows, std::memory_order_relaxed);
        if (index_rows == 1) {
            profile.decode_eligible_matches.fetch_add(1, std::memory_order_relaxed);
        } else {
            profile.batched_eligible_matches.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

static bool ggml_sycl_graph_profile_enabled() {
    const char * value = getenv("GGML_SYCL_GRAPH_PROFILE");
    return value != nullptr && strcmp(value, "1") == 0;
}

struct ggml_sycl_graph_profile {
    std::atomic<uint64_t> graph_calls       = 0;
    std::atomic<uint64_t> replay_calls      = 0;
    std::atomic<uint64_t> gen_rerecords     = 0;
    std::atomic<uint64_t> direct_calls      = 0;
    std::atomic<uint64_t> nodes             = 0;
    std::atomic<uint64_t> record_us         = 0;
    std::atomic<uint64_t> finalize_calls    = 0;
    std::atomic<uint64_t> finalize_us       = 0;
    std::atomic<uint64_t> update_calls      = 0;
    std::atomic<uint64_t> update_fallbacks  = 0;
    std::atomic<uint64_t> update_us         = 0;
    std::atomic<uint64_t> submit_us         = 0;
    std::atomic<uint64_t> wait_us           = 0;
    std::atomic<uint64_t> direct_enqueue_us = 0;

    ~ggml_sycl_graph_profile() {
        if (!ggml_sycl_graph_profile_enabled()) {
            return;
        }
        fprintf(
            stderr,
            "GGML_SYCL_GRAPH_PROFILE: graph_calls=%" PRIu64 " replay_calls=%" PRIu64
            " gen_rerecords=%" PRIu64 " direct_calls=%" PRIu64
            " nodes=%" PRIu64 " record_us=%" PRIu64
            " finalize_calls=%" PRIu64 " finalize_us=%" PRIu64
            " update_calls=%" PRIu64 " update_fallbacks=%" PRIu64
            " update_us=%" PRIu64 " submit_us=%" PRIu64
            " wait_us=%" PRIu64 " direct_enqueue_us=%" PRIu64 "\n",
            graph_calls.load(), replay_calls.load(), gen_rerecords.load(), direct_calls.load(), nodes.load(),
            record_us.load(), finalize_calls.load(), finalize_us.load(), update_calls.load(),
            update_fallbacks.load(), update_us.load(), submit_us.load(), wait_us.load(),
            direct_enqueue_us.load());
    }
};

static ggml_sycl_graph_profile & ggml_sycl_graph_profile_data() {
    static ggml_sycl_graph_profile profile;
    return profile;
}

static uint64_t ggml_sycl_elapsed_us(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
}

#ifdef GGML_SYCL_GRAPH
// Reports if ggml_sycl_mul_mat() gives this node to oneMKL or oneDNN. Both libraries chain the
// submission on events made before recording started, which SYCL graphs do not allow.
// Mirrors ggml_sycl_op_fwht()'s own preconditions (fwht.cpp) without launching a kernel, so the
// graph-compatibility scan can tell whether a Hadamard-hinted MUL_MAT will actually take that
// path or fall through to the ordinary dispatch below it.
static bool ggml_sycl_fwht_supported_shape(const ggml_tensor * src, const ggml_tensor * dst) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    switch (src->ne[0]) {
        case 64: case 128: case 256: case 512:
        case 384: case 768: case 640: case 1280:
            return true;
        default:
            return false;
    }
}

static bool mul_mat_uses_library_gemm(ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // ggml_sycl_mul_mat() only takes the WHT-hint early-out (never a library) when
    // ggml_sycl_op_fwht() actually handles the shape; otherwise it falls through to the same
    // dispatch evaluated below, which for F32 rotation weights can reach oneMKL.
    if (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_fwht_supported_shape(src1, dst)) {
        return false;
    }

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);

    // the branch order below follows the dispatch in ggml_sycl_mul_mat()
    if (!split && src0->type == GGML_TYPE_F16) {
        if (ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
            return !(src0->ne[3] == 1 && src1->ne[3] == 1);
        }
        if (!ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
            return false;
        }
        if (!ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
            return true;
        }
    }

    // ggml_sycl_op_mul_mat_sycl() is the last resort of the dispatch and the only branch that can
    // reach a library GEMM (oneMKL for the f16/quantized path, oneDNN for the bf16 fast path). The
    // reorder decision ggml_sycl_mul_mat() makes between DMMV and MMVQ is left out here: neither
    // dequantize_mul_mat_vec nor mul_mat_vec_q uses a library.
    if (can_use_dequantize_mul_mat_vec(src0, src1, dst) || can_use_mul_mat_vec_q(src0, src1, dst)) {
        return false;
    }

    bool use_mul_mat_q = ggml_sycl_supports_mmq(src0->type) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    // mmq needs the __dp4a instruction; workaround in
    // https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // SYCL_USE_XMX
    return !use_mul_mat_q;
}

// Return whether cgraph passes the backend's graph-capture eligibility checks.
// Rejects multiple devices, contiguous dim-3 CONCAT, non-fused MUL_MAT_ID, matrix multiplication
// paths that require unavailable async memory operations, and any node that dispatches to a
// oneMKL/oneDNN library GEMM (those chain on pre-recording events, which graphs disallow).
// Device graph support is checked separately by the caller.
static bool check_graph_compatibility(ggml_backend_sycl_context * ctx, ggml_cgraph * cgraph) {
    if (ggml_sycl_info().device_count > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_DEBUG("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // skip the nodes that ggml_backend_sycl_graph_compute_impl() does not run
        if (ggml_sycl_is_view_or_noop(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const ggml_op node_op      = node->op;
        bool          uses_library = false;

        switch (node_op) {
            default:
                break;
            case GGML_OP_CONCAT: {
                // dim==3 contiguous concat uses blocking memcpy; other dims use async GPU kernels.
                const int32_t dim = ((const int32_t *) node->op_params)[0];
                const ggml_tensor * src0 = node->src[0];
                const ggml_tensor * src1 = node->src[1];
                if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
                    GGML_LOG_DEBUG("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                                   ggml_op_name(node_op));
                    return false;
                }
                break;
            }
            case GGML_OP_MUL_MAT_ID: {
                // The non-fused path (ne12 > 1, or a ne12==1 node that fails any fused-path
                // precondition) blocks on host in ggml_sycl_mul_mat_id(); the fused single-token
                // decode path in ggml_sycl_mul_mat_id_mmvq_fused() runs entirely on GPU. Mirror
                // every bail-out check in that function here, since a node that dispatch would
                // fall through to the blocking path on is not graph-capturable even if it
                // superficially looks like a decode-shaped MUL_MAT_ID.
                // opt_for_reorder_id() (used inside the fused path) uses sycl_reorder_temp_buffer
                // (USM malloc/free) only for its supported types (Q4_K/Q5_K/Q6_K), which is the
                // only case that needs async USM allocation to be graph-capturable; for any other
                // type it returns immediately without touching USM, so gating on the type here
                // avoids rejecting graph capture for MoE models (MXFP4, Q4_0, Q8_0, IQ*, ...) that
                // never exercise this allocation at all.
                const ggml_tensor * mmid_dst  = node;
                const ggml_tensor * mmid_src0 = mmid_dst->src[0];
                const ggml_tensor * mmid_src1 = mmid_dst->src[1];
                const ggml_tensor * mmid_ids  = mmid_dst->src[2];
                // src0's optimized_feature.reorder flag could in principle be poisoned by the
                // *generic* opt_for_reorder() (which supports a wider type set than the
                // ID-specific reorder dispatcher) if a dense MUL_MAT ever ran against the same
                // extra. That requires ggml_can_mul_mat()'s t1->ne[2] % t0->ne[2] == 0 to hold
                // with should_reorder_tensor()'s src1->ne[2] == 1, which is only satisfiable when
                // t0->ne[2] (n_expert) == 1 - and reorder_qw() itself only supports Q4_K/Q5_K/Q6_K
                // once ne[2] > 1, so the wider-type poisoning path never reaches a real (n_expert
                // > 1) multi-expert tensor. Requiring ne[2] > 1 here closes the theoretical n_expert
                // == 1 edge case for free; a single-expert MUL_MAT_ID isn't worth graph-capturing
                // anyway.
                const bool fused_dispatchable =
                    ggml_sycl_mul_mat_vec_q_id_supports_type(mmid_src0->type) &&
                    mmid_src0->ne[2] > 1 &&
                    mmid_src1->ne[2] == 1 &&
                    mmid_src1->type == GGML_TYPE_F32 && mmid_dst->type == GGML_TYPE_F32 &&
                    mmid_src1->ne[0] == mmid_src0->ne[0] && mmid_src1->ne[0] % QK8_1 == 0 &&
                    ggml_is_contiguous(mmid_src1) &&
                    mmid_ids->ne[1] == 1 &&
                    (mmid_src1->ne[1] == 1 || mmid_src1->ne[1] == mmid_ids->ne[0]);
                const bool id_reorder_needs_async_alloc =
                    g_ggml_sycl_enable_optimize &&
                    (mmid_src0->type == GGML_TYPE_Q4_K || mmid_src0->type == GGML_TYPE_Q5_K ||
                     mmid_src0->type == GGML_TYPE_Q6_K);
                if (!fused_dispatchable || (id_reorder_needs_async_alloc && !g_ggml_sycl_use_async_mem_op)) {
                    GGML_LOG_DEBUG("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                                   ggml_op_name(node_op));
                    return false;
                }
                break;
            }
            case GGML_OP_OUT_PROD:
            case GGML_OP_CONV_3D:
            case GGML_OP_SOLVE_TRI:
                // these ops always call a oneMKL routine
                uses_library = true;
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                // Sparse FA (opt-in, GGML_SYCL_SPARSE_FA=1) is tried before the dense kernels
                // and, like the library routes below, ends with a blocking stream->wait().
                uses_library = ggml_sycl_flash_attn_ext_uses_library(ctx->device, node) ||
                               ggml_sycl_flash_attn_ext_sparse_applicable(node);
                break;
            case GGML_OP_SET_ROWS:
                // The K-quant/IQ block-structured types (anything reorder_qw() and the fast
                // per-type set_rows_sycl_q<>() kernels don't cover) fall back to a host-side
                // quantization path in set_rows.cpp that reads the tensors back with a blocking
                // stream->wait() per row. None of these are valid --cache-type-k/-v choices
                // (see kv_cache_types in common/arg.cpp) so this is not known to be reachable
                // today, but nothing enforces that invariant here, so gate on it directly.
                switch (node->type) {
                    case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M: case GGML_TYPE_IQ4_XS:
                        uses_library = true;
                        break;
                    default:
                        break;
                }
                break;
            case GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                if (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_DEBUG(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(node_op));
                    return false;
                }
                uses_library = mul_mat_uses_library_gemm(node);
                break;
        }

        if (uses_library) {
            // covers oneMKL/oneDNN GEMM routes, sparse FA, and the SET_ROWS host fallback -
            // all end in a blocking call a SYCL graph cannot capture
            GGML_LOG_DEBUG("%s: disabling SYCL graphs due to node type %s reaching a blocking call\n", __func__,
                           ggml_op_name(node_op));
            return false;
        }
    }
    return true;
}

// Detects weights that ggml_backend_sycl_graph_compute_impl() will reorder in place on this
// call. Recording a reorder that mutates a weight buffer during capture hangs on icpx <= the
// version fixed by https://github.com/intel/llvm/pull/21170 (see the compile-time gate below).
static bool graph_needs_reorder(ggml_backend_sycl_context * ctx, const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (ggml_sycl_is_view_or_noop(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const ggml_tensor * src0 = node->src[0];
        if (src0 == nullptr) {
            continue;
        }

        bool reorder_eligible = false;
        if (node->op == GGML_OP_MUL_MAT) {
            // should_reorder_tensor() does not itself gate on src0->type: without this check,
            // a decode-shaped MUL_MAT of a type reorder_qw() never handles (F16, IQ*, MXFP4, ...)
            // reads as "reorder eligible" here, but its extra->optimized_feature.reorder flag can
            // never become true, so graph_needs_reorder() would return true forever and graphs
            // would never enable for that model on this path.
            //
            // opt_for_reorder() picks its type predicate from the mul_mat_algo the *runtime*
            // dispatch in ggml_sycl_mul_mat() selects (DMMV, MMVQ, or the MUL_MAT_SYCL/library
            // last resort), which this static scan cannot know ahead of time. Union all three:
            // a false positive here only makes graphs stay off a call longer, a false negative
            // lets an actual reorder happen mid-recording - the hang this whole gate exists to
            // prevent.
            reorder_eligible = should_reorder_tensor(*ctx, node) &&
                                (ggml_sycl_supports_reorder_dmmv(src0->type) ||
                                 ggml_sycl_supports_reorder_mmvq(src0->type) ||
                                 ggml_sycl_supports_reorder_mul_mat_sycl(src0->type));
        } else if (node->op == GGML_OP_MUL_MAT_ID &&
                   (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K)) {
            // mirrors the id_reorder_needs_async_alloc gate in check_graph_compatibility()
            reorder_eligible = g_ggml_sycl_enable_optimize;
        }
        if (!reorder_eligible) {
            continue;
        }

        const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
        if (extra != nullptr && !extra->optimized_feature.reorder) {
            GGML_LOG_DEBUG("%s: disabling SYCL graphs due to needing reorder\n", __func__);
            return true;
        }
    }

    return false;
}

static const void * ggml_sycl_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

// Detects whether the cgraph's nodes changed since the last call that used this ggml_sycl_graph
// (by uid, then by a full node/src comparison). A change means the previously recorded graph's
// baked-in pointers/shapes are stale and it must be re-recorded (or re-finalized) before replay.
static bool ggml_sycl_graph_update_required(ggml_sycl_graph * graph, ggml_cgraph * cgraph) {
    bool res = false;

    if (cgraph->uid != 0 && cgraph->uid == graph->uid) {
        GGML_SYCL_DEBUG("[SYCL-GRAPH] graph id %" PRIu64 " reused\n", cgraph->uid);
        GGML_ASSERT((int) graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    if ((int) graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_sycl_graph::node_properties prop = {};
        prop.node_data = cgraph->nodes[i]->data;
        prop.node_type = cgraph->nodes[i]->type;
        prop.node_op   = cgraph->nodes[i]->op;
        memcpy(prop.node_ne, cgraph->nodes[i]->ne, sizeof(prop.node_ne));
        memcpy(prop.node_nb, cgraph->nodes[i]->nb, sizeof(prop.node_nb));
        memcpy(prop.node_op_params, cgraph->nodes[i]->op_params, sizeof(prop.node_op_params));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

#endif

// Runs cgraph without a SYCL graph, recording the direct-call profile counters. Shared by the
// graph-incompatible fallback and by GGML_SYCL_GRAPH's own not-yet-warmed-up/graph-unsupported
// paths so both stay in sync.
static ggml_status ggml_sycl_graph_direct_compute(
        ggml_backend_sycl_context * sycl_ctx, ggml_backend_sycl_device_context * dev_ctx,
        ggml_cgraph * cgraph, ggml_sycl_graph_profile * graph_profile) {
    const auto direct_start = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_sycl_graph_compute_impl(sycl_ctx, dev_ctx, cgraph);
    if (graph_profile != nullptr) {
        graph_profile->direct_calls.fetch_add(1, std::memory_order_relaxed);
        graph_profile->direct_enqueue_us.fetch_add(
            ggml_sycl_elapsed_us(direct_start), std::memory_order_relaxed);
    }
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_sycl_record_failed_status(dev_ctx);
    }
    return status;
}

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);
    // The pending failure is sticky until consumed. The scheduler synchronizes
    // this backend between splits without consuming, so clearing here would
    // drop a failure that surfaced in such a sync before the next consume.
    ggml_sycl_profile_rope_fusion(cgraph);
    ggml_sycl_profile_ffn_fusion(cgraph);
    const bool graph_profile_enabled = ggml_sycl_graph_profile_enabled();
    ggml_sycl_graph_profile * graph_profile =
        graph_profile_enabled ? &ggml_sycl_graph_profile_data() : nullptr;

    // set the device on every call: a graph replay below does not go through
    // ggml_backend_sycl_graph_compute_impl(), which used to do this itself
    ggml_sycl_set_main_device(sycl_ctx->device);

#ifdef GGML_SYCL_GRAPH
    bool              use_sycl_graph = false;
    ggml_sycl_graph * graph          = nullptr;

    if (g_ggml_sycl_enable_graph) {
        if (!ggml_sycl_info().devices[sycl_ctx->device].graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
        } else {
            graph = sycl_ctx->sycl_graph(ggml_sycl_graph_get_key(cgraph));

            // The nodes of a cgraph do not change while its uid does not, so the compatibility
            // scan runs again only when the uid changes. A uid of 0 means it is not set.
            if (cgraph->uid == 0 || cgraph->uid != graph->compatible_uid) {
                graph->compatible_uid = cgraph->uid;
                graph->compatible     = check_graph_compatibility(sycl_ctx, cgraph);
// SYCL async memory allocation extensions are available but lead to a hang when reordering
// is needed in versions prior to this fix https://github.com/intel/llvm/pull/21170
#if !defined(__INTEL_LLVM_COMPILER) || __INTEL_LLVM_COMPILER <= 20250303
                graph->needs_reorder = true;
#else
                graph->needs_reorder = false;
#endif
            }

            // an eager call applies the reorder, so once needs_reorder is confirmed false for
            // this uid it stays false: every weight it names is now reordered, and reorder is
            // one-way.
            if (graph->needs_reorder) {
                graph->needs_reorder = graph_needs_reorder(sycl_ctx, cgraph);
            }
            use_sycl_graph = graph->compatible && !graph->needs_reorder;
        }
    }
    if (use_sycl_graph) {
        const bool properties_changed = ggml_sycl_graph_update_required(graph, cgraph);
        bool replay_graph = false;
        bool sycl_graph_update_required = false;

        if (!graph->warmup_complete) {
            // Warmup: need at least 2 calls with no property change on the 2nd call.
            if (!properties_changed) {
                graph->warmup_complete = true;
                replay_graph = true;
                sycl_graph_update_required = true;
                GGML_SYCL_DEBUG("[SYCL-GRAPH] warmup complete\n");
            }
        } else {
            // Post-warmup: replay graph while properties remain stable.
            if (properties_changed) {
                graph->warmup_complete = false;
                GGML_SYCL_DEBUG("[SYCL-GRAPH] warmup reset\n");
            } else {
                replay_graph = true;
                sycl_graph_update_required = graph->exec_graph == nullptr;
            }
        }

        if (!replay_graph) {
            return ggml_sycl_graph_direct_compute(sycl_ctx, dev_ctx, cgraph, graph_profile);
        }

        // The FA K/V F16 scratch (fattn_buffers()) is shared by every graph in this context's
        // map, and grows (freeing the old allocation) not only here but also on any eager,
        // non-recorded flash-attention call - the warmup call, a post-reset eager call, or the
        // plain non-graph path below. A cached graph whose recording predates the buffer's
        // current generation has a freed pointer baked into it, so force a re-record instead of
        // replaying it.
        if (!sycl_graph_update_required && graph->fattn_generation != sycl_ctx->fattn_buffers().generation()) {
            sycl_graph_update_required = true;
            if (graph_profile != nullptr) {
                graph_profile->gen_rerecords.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (sycl_graph_update_required) {
            const bool graph_update_support = ggml_sycl_info().devices[sycl_ctx->device].graph_update_support;

            // No explicit pre-growth of fattn_buffers() here (unlike the direct port of this
            // code): every path that reaches this block either just ran this exact cgraph eagerly
            // (the warmup-completion call, which sized the buffers for these exact shapes via the
            // real kernels' own need_f16 logic) or is only here because *another* graph's growth
            // bumped the generation counter, which means the buffers are already at least as big
            // as what this graph itself has ever needed (ensure_half() never shrinks). The
            // before/after generation check below is a backstop for that invariant, not the
            // primary defense: growing mid-capture would call qptr->wait() on a recording queue,
            // which SYCL_CHECK turns into a hard abort, so this can only catch the one subcase
            // that doesn't wait - a first-ever allocation with no prior buffer to free.
            const uint64_t fattn_gen_before = sycl_ctx->fattn_buffers().generation();

            sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

            const auto record_start = std::chrono::steady_clock::now();
            model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
            ggml_status graph_status;
            {
                // RAII: guarantees graph_recording resets to false even if
                // compute_impl or end_recording throws, so a mid-capture exception
                // cannot leave the flag stuck true and permanently suppress the FA
                // profile on this context.
                // RAII: also ends the recording itself on unwind, not just the flag. Without
                // this, an exception thrown out of compute_impl (a node reached a blocking
                // call the compatibility scan didn't anticipate) would leave the underlying
                // SYCL queue stuck in the recording state for the rest of the process.
                struct graph_recording_guard {
                    bool &                 recording_flag;
                    std::function<void()>  end_recording;
                    bool                   ended = false;
                    graph_recording_guard(bool & f, std::function<void()> end_recording_)
                        : recording_flag(f), end_recording(std::move(end_recording_)) {
                        recording_flag = true;
                    }
                    void end() {
                        end_recording();
                        ended = true;
                    }
                    ~graph_recording_guard() {
                        if (!ended) {
                            try {
                                end_recording();
                            } catch (...) {
                                // best-effort during unwind; nothing more can be done here
                            }
                        }
                        recording_flag = false;
                    }
                } recording_guard(sycl_ctx->graph_recording,
                                  [&model_sycl_graph]() { model_sycl_graph.end_recording(); });
                graph_status = ggml_backend_sycl_graph_compute_impl(sycl_ctx, dev_ctx, cgraph);
                recording_guard.end();
            }
            if (graph_profile != nullptr) {
                graph_profile->graph_calls.fetch_add(1, std::memory_order_relaxed);
                graph_profile->nodes.fetch_add(cgraph->n_nodes, std::memory_order_relaxed);
                graph_profile->record_us.fetch_add(
                    ggml_sycl_elapsed_us(record_start), std::memory_order_relaxed);
            }
            const uint64_t fattn_gen_after = sycl_ctx->fattn_buffers().generation();
            if (graph_status == GGML_STATUS_SUCCESS && fattn_gen_after != fattn_gen_before) {
                // The invariant above didn't hold: this recording may have baked in a pointer
                // that a concurrent growth elsewhere has already freed. Discard it rather than
                // finalize/replay it.
                GGML_LOG_DEBUG("%s: discarding a recording made during FA scratch growth\n", __func__);
                graph_status = GGML_STATUS_FAILED;
            }
            if (graph_status != GGML_STATUS_SUCCESS) {
                // Whatever exec_graph currently holds (nullptr, or a graph recorded for the old
                // properties/pointers that triggered this re-record) is not safe to replay next
                // call: force a full re-record from a clean warmup.
                graph->warmup_complete = false;
                graph->exec_graph.reset();
                ggml_backend_sycl_record_failed_status(dev_ctx);
                return graph_status;
            }
            graph->fattn_generation = fattn_gen_after;

            const auto finalize_update_start = std::chrono::steady_clock::now();
            if (!graph->exec_graph || !graph_update_support) {
                auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                         model_sycl_graph.finalize();
                graph->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
                if (graph_profile != nullptr) {
                    graph_profile->finalize_calls.fetch_add(1, std::memory_order_relaxed);
                    graph_profile->finalize_us.fetch_add(
                        ggml_sycl_elapsed_us(finalize_update_start), std::memory_order_relaxed);
                }
            } else {
                try {
                    graph->exec_graph->update(model_sycl_graph);
                    GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
                    if (graph_profile != nullptr) {
                        graph_profile->update_calls.fetch_add(1, std::memory_order_relaxed);
                        graph_profile->update_us.fetch_add(
                            ggml_sycl_elapsed_us(finalize_update_start), std::memory_order_relaxed);
                    }
                } catch (sycl::exception const & e) {
                    // the executable graph cannot be updated due to violated constraints,
                    // so finalize the recorded graph again
                    GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                    auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                    graph->exec_graph = std::make_unique<
                        sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
                    if (graph_profile != nullptr) {
                        graph_profile->update_calls.fetch_add(1, std::memory_order_relaxed);
                        graph_profile->update_fallbacks.fetch_add(1, std::memory_order_relaxed);
                        graph_profile->update_us.fetch_add(
                            ggml_sycl_elapsed_us(finalize_update_start), std::memory_order_relaxed);
                    }
                }
            }
        } else if (graph_profile != nullptr) {
            graph_profile->replay_calls.fetch_add(1, std::memory_order_relaxed);
        }

        const auto submit_start = std::chrono::steady_clock::now();
        try {
            sycl_ctx->stream()->ext_oneapi_graph(*(graph->exec_graph));
            if (graph_profile != nullptr) {
                graph_profile->submit_us.fetch_add(
                    ggml_sycl_elapsed_us(submit_start), std::memory_order_relaxed);
            }
            const auto wait_start = std::chrono::steady_clock::now();
            sycl_ctx->stream()->wait();
            if (graph_profile != nullptr) {
                graph_profile->wait_us.fetch_add(
                    ggml_sycl_elapsed_us(wait_start), std::memory_order_relaxed);
            }
        } catch (sycl::exception const & e) {
            GGML_LOG_ERROR("%s: SYCL graph launch/wait failed: %s\n", __func__, e.what());
            // A failed launch/wait leaves no guarantee this exec_graph is still valid to
            // resubmit; force a full re-record next time rather than retrying it as-is.
            graph->warmup_complete = false;
            graph->exec_graph.reset();
            ggml_backend_sycl_record_failed_exception(dev_ctx, e);
            return GGML_STATUS_FAILED;
        }
    } else
#endif
    {
        const ggml_status status = ggml_sycl_graph_direct_compute(sycl_ctx, dev_ctx, cgraph, graph_profile);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);
 
    try {
        sycl::event * sycl_event = static_cast<sycl::event *>(event->context);
        const queue_ptr & stream = sycl_ctx->stream(sycl_ctx->device, 0);
        // Record the current state of the queue
        SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
    } catch (sycl::exception const & exc) {
        ggml_backend_sycl_record_failed_exception(dev_ctx, exc);
        GGML_LOG_ERROR("%s: SYCL event record failed: %s\n", __func__, exc.what());
    }
}
 
static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);

    try {
        sycl::event * sycl_event = static_cast<sycl::event *>(event->context);

        if (ggml_backend_is_sycl(backend) && event->device == backend->device) {
            // same device: make the queue wait on the device instead of blocking the host, so
            // a private-stream backend's copy can overlap this queue's work (the equivalent of
            // cudaStreamWaitEvent)
            ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) backend->context;
            const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier({ *sycl_event })));
        } else if (ggml_backend_is_sycl(backend)) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
        } else {
            GGML_ABORT("fatal error");
        }
    } catch (sycl::exception const & exc) {
        ggml_backend_sycl_record_failed_exception(dev_ctx, exc);
        GGML_LOG_ERROR("%s: SYCL event wait failed: %s\n", __func__, exc.what());
    }
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
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

int ggml_backend_sycl_get_device_count(void) {
    return ggml_sycl_info().device_count;
}


// backend device

static ggml_backend_sycl_device_context * ggml_backend_sycl_device_context_from_device(ggml_backend_dev_t dev) {
    return dev ? static_cast<ggml_backend_sycl_device_context *>(dev->context) : nullptr;
}
 
static ggml_backend_sycl_device_context * ggml_backend_sycl_device_context_from_backend(ggml_backend_t backend) {
    if (backend == nullptr || !ggml_backend_is_sycl(backend)) {
        return nullptr;
    }
 
    return ggml_backend_sycl_device_context_from_device(backend->device);
}
 
static ggml_backend_sycl_failure_cause ggml_backend_sycl_classify_exception(const sycl::exception & exc) {
    const int raw_code = exc.code().value();
    const std::string_view category_name(exc.code().category().name());
    const bool is_ur_category =
        category_name == "ur_result_t" ||
        category_name == "unified-runtime" ||
        category_name == "unified_runtime";

#if GGML_SYCL_HAS_UR_API
    if (is_ur_category && (raw_code == UR_RESULT_ERROR_DEVICE_LOST || raw_code == UR_RESULT_ERROR_DEVICE_REQUIRES_RESET)) {
        return GGML_SYCL_FAILURE_CAUSE_DEVICE_LOST;
    }
#else
    GGML_UNUSED(raw_code);
    GGML_UNUSED(is_ur_category);
#endif

    return GGML_SYCL_FAILURE_CAUSE_OTHER;
}
 
static void ggml_backend_sycl_record_failed_status(ggml_backend_sycl_device_context * dev_ctx) {
    if (dev_ctx != nullptr && dev_ctx->pending_status.load() == GGML_STATUS_SUCCESS) {
        dev_ctx->pending_status.store(GGML_STATUS_FAILED);
        dev_ctx->pending_cause.store(GGML_SYCL_FAILURE_CAUSE_OTHER);
        dev_ctx->pending_raw_code.store(0);
    }
}
#ifdef GGML_SYCL_TESTING
static void ggml_backend_sycl_test_maybe_record_sync_failure(ggml_backend_sycl_device_context * dev_ctx) {
    if (dev_ctx != nullptr && dev_ctx->test_sync_failure_once.exchange(false)) {
        ggml_backend_sycl_record_failed_status(dev_ctx);
    }
}

bool ggml_backend_sycl_test_inject_sync_failure_once(ggml_backend_t backend) {
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);
    if (dev_ctx == nullptr) {
        return false;
    }

    dev_ctx->test_sync_failure_once.store(true);
    return true;
}
#endif
 
static void ggml_backend_sycl_record_failed_exception(ggml_backend_sycl_device_context * dev_ctx, const sycl::exception & exc) {
    if (dev_ctx != nullptr) {
        dev_ctx->pending_status.store(GGML_STATUS_FAILED);
        dev_ctx->pending_cause.store(ggml_backend_sycl_classify_exception(exc));
        dev_ctx->pending_raw_code.store(exc.code().value());
    }
}
 
extern "C" ggml_backend_sycl_failure ggml_backend_sycl_consume_last_failure(ggml_backend_t backend) {
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_backend(backend);
    if (dev_ctx == nullptr) {
        return {
            GGML_STATUS_SUCCESS,
            GGML_SYCL_FAILURE_CAUSE_NONE,
            0,
        };
    }
 
    ggml_backend_sycl_failure out = {
        static_cast<ggml_status>(dev_ctx->pending_status.exchange(GGML_STATUS_SUCCESS)),
        static_cast<ggml_backend_sycl_failure_cause>(dev_ctx->pending_cause.exchange(GGML_SYCL_FAILURE_CAUSE_NONE)),
        dev_ctx->pending_raw_code.exchange(0),
    };
    return out;
}
 
extern "C" ggml_status ggml_backend_sycl_consume_last_status(ggml_backend_t backend) {
    return ggml_backend_sycl_consume_last_failure(backend).status;
}

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    if (!sycl_get_mem_info(ctx->device, free, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *total, *free);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *free, *total);
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    const sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    if (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
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
        /* .mmap_support          = */ true,
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
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    return ggml_backend_sycl_host_buffer_type_for_device(ctx->device);
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
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
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
            return true;
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
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_CEIL:
                    return true;
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
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
                case GGML_GLU_OP_SWIGLU_CLAMP:
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

                // Q8_CR/Q5_CR/Q6_CR (ConvRot-rotated) have no SYCL dequant/vec_dot kernels yet, CPU-only.
                if (a->type == GGML_TYPE_Q8_CR || a->type == GGML_TYPE_Q5_CR || a->type == GGML_TYPE_Q6_CR) {
                    return false;
                }

                if (a->type == GGML_TYPE_TQ3_1S || a->type == GGML_TYPE_TQ4_1S) {
                    return true;
                }

                if (a->ne[3] != b->ne[3]) {
                    return false;
                }

                ggml_type src0_type = op->src[0]->type;

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

                if (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    return false;
                }

                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_I32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            // return true;
             return op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         case GGML_OP_SET:
               return (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->nb[0] != ggml_type_size(op->src[0]->type)) {
                    return false;
                }
                if (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    return false;
                }
                if (ggml_type_is_turbo(op->type) && op->src[0]->ne[0] % 128 != 0) {
                    return false;
                }

                auto res = ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                         op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q5_0 ||
                         op->type == GGML_TYPE_Q1_0 || op->type == GGML_TYPE_Q2_0 ||
                         op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_IQ4_NL ||
                         op->type == GGML_TYPE_MXFP4 || op->type == GGML_TYPE_NVFP4 ||
                         op->type == GGML_TYPE_Q2_K || op->type == GGML_TYPE_Q3_K || op->type == GGML_TYPE_Q4_K ||
                         op->type == GGML_TYPE_Q5_K || op->type == GGML_TYPE_Q6_K ||
                         op->type == GGML_TYPE_IQ2_XXS || op->type == GGML_TYPE_IQ2_XS || op->type == GGML_TYPE_IQ2_S ||
                         op->type == GGML_TYPE_IQ3_XXS || op->type == GGML_TYPE_IQ3_S ||
                         op->type == GGML_TYPE_IQ1_S || op->type == GGML_TYPE_IQ1_M || op->type == GGML_TYPE_IQ4_XS ||
                         // TQ3_1S/TQ4_1S are weight-only types: ggml_sycl_op_set_rows has no kernel for them
                         ggml_type_is_turbo(op->type)) &&
                        (op->src[0]->type == GGML_TYPE_F32 ||
                         // an f16 source only feeds the float destinations; the quantizers read float.
                         // Declining it here is not free: SET_ROWS returns a view of its destination,
                         // so a fallback backend would write into this device's memory.
                         (op->src[0]->type == GGML_TYPE_F16 &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16))) &&
                        (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32));
                return res;
            }
            break;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_LIGHTNING_INDEXER:
            return op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                if (src0_type == GGML_TYPE_F16) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_BF16) {
                    if (src1_type == GGML_TYPE_Q4_0 || //big error in ut
                        src1_type == GGML_TYPE_Q4_1 || //big error in ut
                        src1_type == GGML_TYPE_Q8_0 || //big error in ut
                        src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_F32) {
                    if (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        return false;
                    }
                }

                if (src1_type == GGML_TYPE_F32) {
                    if (src0_type == GGML_TYPE_Q1_0 ||
                        src0_type == GGML_TYPE_NVFP4 ||
                        src0_type == GGML_TYPE_Q2_K ||
                        src0_type == GGML_TYPE_Q3_K ||
                        src0_type == GGML_TYPE_Q4_K ||
                        src0_type == GGML_TYPE_Q5_K ||
                        src0_type == GGML_TYPE_Q6_K ||
                        src0_type == GGML_TYPE_IQ2_XXS ||
                        src0_type == GGML_TYPE_IQ2_XS ||
                        src0_type == GGML_TYPE_IQ2_S ||
                        src0_type == GGML_TYPE_IQ3_XXS ||
                        src0_type == GGML_TYPE_IQ1_S ||
                        src0_type == GGML_TYPE_IQ1_M ||
                        src0_type == GGML_TYPE_IQ3_S ||
                        src0_type == GGML_TYPE_IQ4_NL ||
                        src0_type == GGML_TYPE_IQ4_XS
                    ) {
                        return false;
                    }
                }

                if (src0_type == src1_type) {
                    if (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        return false;
                    }
                }

                if (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    return false;
                }

                return true;
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
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SCALE:
            return true;
        case GGML_OP_CONT:
            return true;
        case GGML_OP_TRI:
            {
                const ggml_tensor * src0 = op->src[0];
                return src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
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
        case GGML_OP_ROPE_BACK:
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_UPSCALE:
            return true;
        case GGML_OP_COL2IM_1D:
            return ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        case GGML_OP_CONV_3D:
            return op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return true;
        case GGML_OP_TOP_K: {
            const ggml_tensor * src0 = op->src[0];
            const int k = op->ne[0];
            return src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_PAD:
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return true;
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return ggml_get_op_params_i32(op, 1) == 0;
        case GGML_OP_SSM_CONV:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ROLL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_SSM_SCAN:
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } else {
                // TODO Mamba-1 not yet ported to SYCL
                return false;
            }
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_DIAG:
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            return true;
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_sycl_flash_attn_ext_supported(device, op);
        case GGML_OP_TURBO_WHT:
            {
                const ggml_tensor * scale = op->src[1];
                int group_size = 0;
                memcpy(&group_size, op->op_params + sizeof(int), sizeof(int));
                return op->src[0]->type == GGML_TYPE_F32
                    && op->type == GGML_TYPE_F32
                    && ggml_is_contiguous(op->src[0])
                    && ggml_is_contiguous(op)
                    && (scale == nullptr || (scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale)))
                    && (group_size == 32 || group_size == 64 || group_size == 128)
                    && op->src[0]->ne[0] % group_size == 0;
            }
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    bool res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    return res;
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return buft_ctx->device == sycl_ctx->device;
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
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
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


static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * dev_ctx = ggml_backend_sycl_device_context_from_device(dev);
 
    try {
        sycl::event * sycl_event = static_cast<sycl::event *>(event->context);
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } catch (sycl::exception const & exc) {
        ggml_backend_sycl_record_failed_exception(dev_ctx, exc);
        GGML_LOG_ERROR("%s: SYCL device event synchronize failed: %s\n", __func__, exc.what());
    }
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

// ==========================================================================
// Tensor parallelism (--split-mode tensor) for the SYCL backend.
//
// The meta-backend invokes these three entry points via get_proc_address:
//   * ggml_backend_sycl_comm_init             - one-time per-graph setup
//   * ggml_backend_sycl_comm_allreduce_tensor - per-allreduce step
//   * ggml_backend_sycl_comm_free             - tear-down
//
// For N=2 (dual-GPU), this is a degenerate ring allreduce with dual paths
// chosen by tensor size:
//
//   * Small (nelem < 32K): FP32 direct memcpy + per-device ADD
//     kernel. The kernel depends_on() its corresponding memcpy event
//     so it doesn't read partial data. Both devices run in parallel.
//
//   * Large (nelem >= 32K): BF16-compressed. Each device compresses
//     its FP32 partial to BF16 locally, cross-device memcpys
//     to the peer (half the PCI bandwidth), where it is decompressed
//     and added into the local FP32 partial. 6 SYCL submissions per
//     allreduce (2 compress + 2 memcpy + 2 decompress-add) vs the
//     4 for the small path, but the bandwidth saving > 6 GB/s PCIe x 2
//     dominates for larger tensors.
//
// Storage: A persistent uint8_t buffer per device, sized to
// 4 * nelem bytes. Both paths reinterpret the same bytes (small path
// as nelem floats; large path as outbox + inbox = 2*nelem uint16_t
// each, using the full 4*nelem byte budget either way). Single
// alloc+free per device keeps the SYCL pool's strict-LIFO invariant
// trivial.
//
// For non-(N=2 FP32 contiguous) cases, comm_init or comm_allreduce_tensor
// returns null/false, causing the meta-backend to use its generic
// butterfly all-reduce fallback.
// ==========================================================================

struct ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    uint8_t * host_out[2] = { nullptr, nullptr };
    uint8_t * host_in[2]  = { nullptr, nullptr };
    size_t    host_bytes  = 0;

    void free_host() {
        for (int i = 0; i < 2; ++i) {
            const sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            if (host_out[i]) { sycl::free(host_out[i], sctx); host_out[i] = nullptr; }
            if (host_in[i])  { sycl::free(host_in[i],  sctx); host_in[i]  = nullptr; }
        }
        host_bytes = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
static void ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    void * dst0, const void * src1, void * dst1, const void * src0,
                                    size_t nbytes) {
    if (comm->host_bytes < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_out[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_in[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_out[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_in[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_out[0] && comm->host_in[0] && comm->host_out[1] && comm->host_in[1]);
        comm->host_bytes = nbytes;
    }

    sycl::event e0 = q0->memcpy(comm->host_out[0], src0, nbytes);
    sycl::event e1 = q1->memcpy(comm->host_out[1], src1, nbytes);
    e0.wait();
    e1.wait();
    std::memcpy(comm->host_in[0], comm->host_out[1], nbytes);
    std::memcpy(comm->host_in[1], comm->host_out[0], nbytes);
    q0->memcpy(dst0, comm->host_in[0], nbytes);
    q1->memcpy(dst1, comm->host_in[1], nbytes);
}

void * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) try {
    for (size_t i = 0; i < n_backends; ++i) {
        if (!ggml_backend_is_sycl(backends[i])) {
            return nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    if (n_backends != 2) {
        return nullptr;
    }

    auto * ctx = new ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    auto * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    auto * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    return ctx;
}
catch (const sycl::exception &) { return nullptr; }
catch (...)                     { return nullptr; }

void ggml_backend_sycl_comm_free(void * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    if (comm_ctx == nullptr) {
        return;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    if (comm_ctx->backends.size() == 2) {
        auto * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        auto * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        try {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } catch (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

bool ggml_backend_sycl_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) try {
    if (comm_ctx_v == nullptr) {
        return false;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    if (n_backends != 2) {
        return false;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    if (tensors[0]->type != tensors[1]->type) {
        return false;
    }
    if (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        return false;
    }
    if (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        return false;
    }

    const int64_t nelem  = ggml_nelements(tensors[0]);
    const size_t  nbytes = ggml_nbytes(tensors[0]);
    if (nelem == 0) {
        return true;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    if (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    if (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->data, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((float) f16_out0[i] + (float) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((float) f16_out1[i] + (float) f16_tmp1[i]);
            });
        });
        return true;
    }

    float * out0 = (float *) tensors[0]->data;
    float * out1 = (float *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    static constexpr int64_t BF16_THRESHOLD = 32768;

    if (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        float * tmp0 = (float *) buf0;
        float * tmp1 = (float *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->data, tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out0[i] += tmp0[i];
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out1[i] += tmp1[i];
            });
        });
        return true;
    }

    // BF16 large path: 6 SYCL submissions per allreduce, but the
    // cross-device memcpy is HALF the bytes. Pure bit-shift
    // conversion (no rounding) — matches ggml's truncating fp32->bf16.
    uint16_t * outbox0 = (uint16_t *) buf0;
    uint16_t * inbox0  = outbox0 + nelem;
    uint16_t * outbox1 = (uint16_t *) buf1;
    uint16_t * inbox1  = outbox1 + nelem;

    // Phase A: compress each device's local partial in parallel.
    q0->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox0[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    const size_t bf16_bytes = nelem * sizeof(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<float>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<float>(((uint32_t) inbox1[i]) << 16);
        });
    });

    return true;
}
catch (const sycl::exception &) { return false; }
catch (...)                     { return false; }

// Backend whose copies run on a private in-order queue instead of the device-wide default
// queue. Probed on Arc A770 (oneAPI 2026.1, compute-runtime 26.35): with the Level Zero v1
// adapter a memcpy on a second queue runs on the copy engine concurrently with kernels on the
// default queue. The v2 adapter and a disabled copy engine both serialize the two queues, so
// those configurations get no private-stream backend (NULL: callers fall back or disable).
static ggml_backend_t ggml_backend_sycl_init_private_stream(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * dev_ctx = (ggml_backend_sycl_device_context *) dev->context;
    try {
        sycl::queue & dq = dpct::get_device(dev_ctx->device).default_queue();
        const std::string platform = dq.get_device().get_platform().get_info<sycl::info::platform::name>();
        // runtime (not ggml) variables with string values: "0" disables, while "lower:upper"
        // selects an engine range, so an integer parse would misread e.g. "0:0"
        auto env_is_zero = [](const char * name) { return ggml_sycl_env_is_zero(getenv(name)); };
        const char * reason = nullptr;
        if (dq.get_backend() != sycl::backend::ext_oneapi_level_zero) {
            reason = "not a Level Zero device";
        } else if (platform.find("Level-Zero V2") != std::string::npos) {
            reason = "the Level Zero v2 adapter serializes queues (unset SYCL_UR_USE_LEVEL_ZERO_V2)";
        } else if (env_is_zero("UR_L0_USE_COPY_ENGINE") || env_is_zero("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE") ||
                   env_is_zero("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE") ||
                   env_is_zero("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE")) {
            reason = ggml_sycl_xe_kmd_defaults_applied()
                         ? "copies are routed to the compute engine by the xe KMD default (DG2 blitter "
                           "failure, docs/backend/SYCL.md); --prefetch-experts-slots needs "
                           "UR_L0_USE_COPY_ENGINE=1, which brings that failure back"
                         : "copies are routed to the compute engine";
        }
        if (reason != nullptr) {
            GGML_LOG_WARN("%s: no private stream on SYCL%d: %s\n", __func__, dev_ctx->device, reason);
            return nullptr;
        }

        sycl::queue pq(dq.get_context(), dq.get_device(), dpct::exception_handler,
                       sycl::property_list{ sycl::property::queue::in_order() });
        ggml_backend_t backend = ggml_backend_sycl_init(dev_ctx->device);
        if (backend == nullptr) {
            return nullptr;
        }
        ((ggml_backend_sycl_context *) backend->context)->use_private_queue(std::move(pq));
        return backend;
    } catch (sycl::exception const & exc) {
        GGML_LOG_ERROR("%s: failed to create a private queue: %s\n", __func__, exc.what());
        return nullptr;
    }
}

// With one SYCL device, every backend op, the buffer set/get/copy calls, the host allocator and
// the MoE cache submit to that device's one in-order default queue (dpct in_order_queue); the
// only other queue, the private stream above, is never handed to the scheduler as a split
// backend. So async work runs in issue order, which lets the scheduler skip its host sync before
// a host-to-device split input copy (see ggml_backend_async_is_stream_ordered_t).
// With several devices, tensor-split MUL_MAT submits to other devices' queues (ctx.stream(i, is))
// and only joins them back onto the main queue with barriers at the end of the op. Stream order
// would then rest on every such path always joining before it returns, which is not audited,
// so multi-device configurations keep the scheduler's host sync.
static bool ggml_backend_sycl_async_is_stream_ordered(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_sycl_info().device_count == 1;
}

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_init_private_stream") == 0) {
        return (void *)ggml_backend_sycl_init_private_stream;
    }
    if (strcmp(name, "ggml_backend_async_is_stream_ordered") == 0) {
        return (void *)ggml_backend_sycl_async_is_stream_ordered;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_sycl_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_sycl_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_sycl_comm_allreduce_tensor;
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

extern "C" void ggml_sycl_moe_cache_register(void * reg);

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

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
            ggml_sycl_moe_cache_register(&reg);
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

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
