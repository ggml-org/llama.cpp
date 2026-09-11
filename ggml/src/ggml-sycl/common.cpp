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

#if GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem, once per device
bool ggml_sycl_dnnl_has_optimized_gemm(queue_ptr q) {
    enum class gemm_impl : uint8_t {
        UNKNOWN   = 0,
        OPTIMIZED = 1,
        REFERENCE = 2,
    };

    static gemm_impl cache[GGML_SYCL_MAX_DEVICES] = {};

    // with a split model the queue can run on another device than ctx.device
    const int device = dpct::dev_mgr::instance().get_device_id(q->get_device());

    GGML_ASSERT(device >= 0 && device < GGML_SYCL_MAX_DEVICES);

    if (cache[device] == gemm_impl::UNKNOWN) {
        using dt = dnnl::memory::data_type;

        cache[device] = gemm_impl::REFERENCE;
        try {
            const dnnl::memory::dims dims    = { 1, 64, 64 };
            const dnnl::memory::dims strides = { 64 * 64, 64, 1 };

            dnnl::primitive_attr attr;
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

            const auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
            const auto pd  = dnnl::matmul::primitive_desc(eng,
                                                          dnnl::memory::desc(dims, dt::f16, strides),
                                                          dnnl::memory::desc(dims, dt::f16, strides),
                                                          dnnl::memory::desc(dims, dt::f32, strides), attr);

            const std::string impl = pd.impl_info_str();
            if (impl.find("ref") == std::string::npos) {
                cache[device] = gemm_impl::OPTIMIZED;
            } else {
                GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                              __func__, device, impl.c_str());
            }
        } catch (const std::exception & e) {
            GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                          __func__, device, e.what());
        }
    }

    return cache[device] == gemm_impl::OPTIMIZED;
}
#endif

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

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
void * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
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
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, 64, ze_dev, &ptr);
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

void ggml_sycl_free_device(void *ptr, sycl::queue &q) {
    if (!ptr) return;
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
