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
#include <mutex>
#include <unordered_map>
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#endif

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

int get_current_device_id() {
  return dpct::dev_mgr::instance().current_device_id();
}

void* ggml_sycl_host_malloc(size_t size) try {
  if (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    return nullptr;
  }

  void* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
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

int ggml_sycl_get_env(const char *env_name, int default_val) {
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

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    return g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Device bytes allocated through ggml_sycl_malloc_device, per device and per pointer
static std::mutex g_ggml_sycl_dev_alloc_mutex;
static std::unordered_map<void *, std::pair<int, size_t>> g_ggml_sycl_dev_allocs;
static size_t g_ggml_sycl_dev_alloc_bytes[GGML_SYCL_MAX_DEVICES] = {};
// free memory reported when we had nothing allocated (0 = not taken yet)
static size_t g_ggml_sycl_dev_free_baseline[GGML_SYCL_MAX_DEVICES] = {};

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
static void * ggml_sycl_malloc_device_impl(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (ggml_sycl_use_level_zero_device_alloc(q)) {
        void *ptr = nullptr;
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME
        ze_relaxed_allocation_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC,
            nullptr,
            ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#else
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        if (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            return ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        return nullptr;
    }
#endif
    void * ptr = sycl::malloc_device(size, q);
    if (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        return nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    return ptr;
}

// caller holds g_ggml_sycl_dev_alloc_mutex
static bool ggml_sycl_free_mem_enough(int device, size_t size, ggml_sycl_mem_type type) {
    const size_t allocated = g_ggml_sycl_dev_alloc_bytes[device];

    // TODO: remove once zeMemAllocDevice refuses allocations beyond VRAM on BMG
    // (https://github.com/intel/compute-runtime/issues/1047). Only model/split buffers are checked.
    const gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    const bool is_bmg   = arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
    const bool check    = g_ggml_sycl_bmg_vram_check && is_bmg && type == GGML_SYCL_MEM_BUFFER;
    const bool trace    = type == GGML_SYCL_MEM_BUFFER && ggml_sycl_memtrace_enabled();
    size_t free_mem = 0, total_mem = 0;
    if (check || trace) {
        sycl_get_mem_info(device, &free_mem, &total_mem);
    }
    if (trace) {
        ggml_sycl_memtrace_report_device("before buffer alloc", device, free_mem, total_mem);
    }
    if (check) {
        size_t & baseline = g_ggml_sycl_dev_free_baseline[device];
        if (allocated == 0) {
            baseline = free_mem;
        }
        // the drop in free memory since the baseline is ours once written, the rest of ours is not counted yet
        const size_t written   = baseline > free_mem ? baseline - free_mem : 0;
        const size_t unwritten = baseline && allocated > written ? allocated - written : 0;
        const size_t usable    = free_mem > unwritten ? free_mem - unwritten : 0;
        if (allocated + size > total_mem || size > usable) {
            GGML_LOG_ERROR("%s: SYCL%d: cannot allocate %zu MiB: %zu MiB already allocated, %zu MiB usable "
                           "(%zu MiB free of %zu MiB), set GGML_SYCL_BMG_VRAM_CHECK=0 to disable this check\n",
                           __func__, device, size >> 20, allocated >> 20, usable >> 20, free_mem >> 20, total_mem >> 20);
            return false;
        }
    }
    return true;
}

static void ggml_sycl_dev_alloc_record(void * ptr, int device, size_t size) {
    g_ggml_sycl_dev_allocs[ptr] = { device, size };
    g_ggml_sycl_dev_alloc_bytes[device] += size;
}

void * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
    const int device = (int) dpct::dev_mgr::instance().get_device_id(q.get_device());
    GGML_ASSERT(device >= 0 && device < GGML_SYCL_MAX_DEVICES);

    std::lock_guard<std::mutex> lock(g_ggml_sycl_dev_alloc_mutex);
    if (!ggml_sycl_free_mem_enough(device, size, type)) {
        ggml_sycl_memtrace_fail(type, size);
        return nullptr;
    }

    void * ptr = ggml_sycl_malloc_device_impl(size, q, type);
    if (ptr) {
        ggml_sycl_dev_alloc_record(ptr, device, size);
    }
    return ptr;
}

void ggml_sycl_free_device(void *ptr, sycl::queue &q) {
    if (!ptr) return;
    {
        std::lock_guard<std::mutex> lock(g_ggml_sycl_dev_alloc_mutex);
        auto it = g_ggml_sycl_dev_allocs.find(ptr);
        if (it != g_ggml_sycl_dev_allocs.end()) {
            const int device = it->second.first;
            g_ggml_sycl_dev_alloc_bytes[device] -= it->second.second;
            if (g_ggml_sycl_dev_alloc_bytes[device] == 0) {
                g_ggml_sycl_dev_free_baseline[device] = 0;
            }
            g_ggml_sycl_dev_allocs.erase(it);
        }
    }
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    if (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        return;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, q)));
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
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
