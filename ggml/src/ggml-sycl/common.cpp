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
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
#include <sycl/backend.hpp>
#include <level_zero/ze_api.h>
#endif

#include <cstdio>
#include <cstdlib>

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

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver.
// sycl::malloc_device creates a 1:1 host memory mirror of every VRAM allocation
// via xe_gem_prime_export, consuming system RAM equal to VRAM allocated.
// zeMemAllocDevice uses the SVM/P2P path with no host staging.
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
static bool ggml_sycl_queue_supports_level_zero(sycl::queue &q) {
    const char * env = std::getenv("GGML_SYCL_ENABLE_LEVEL_ZERO");
    unsigned int enabled = 1;
    if (env && sscanf(env, " %u", &enabled) != 1) {
        enabled = 1;
    }

    return enabled && q.get_device().is_gpu() && q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

void * ggml_sycl_malloc_device(size_t size, sycl::queue &q) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
    if (ggml_sycl_queue_supports_level_zero(q)) {
        void *ptr = nullptr;
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, 64, ze_dev, &ptr);
        if (r == ZE_RESULT_SUCCESS && ptr) {
            return ptr;
        }
        return nullptr;
    }
#endif
    return sycl::malloc_device(size, q);
}

void ggml_sycl_free_device(void *ptr, sycl::queue &q) {
    if (!ptr) return;
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO
    if (ggml_sycl_queue_supports_level_zero(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        return;
    }
#endif
    sycl::free(ptr, q);
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
