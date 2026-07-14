//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "common.hpp"

sycl::half * ggml_sycl_fattn_kv_buffers::kv_buffer::ensure_half(size_t n_elems) {
    const size_t need_bytes = n_elems * sizeof(sycl::half);

    if (capacity >= need_bytes) {
        return ptr;
    }

    if (ptr) {
        SYCL_CHECK(CHECK_TRY_ERROR(qptr->wait()));
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, *qptr)));
        ptr = nullptr;
        capacity = 0;
    }

    size_t cap = 0;
    while (cap < need_bytes) {
        cap += CHUNK_SIZE;
    }

    void * dev_ptr;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dev_ptr = sycl::malloc_device(
                        cap, *qptr)));

    if (!dev_ptr) {
        GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on device\n", __func__, cap);
        GGML_ABORT("fattn buffer alloc failed");
    }

    ptr = static_cast<sycl::half *>(dev_ptr);
    capacity = cap;
    return ptr;
}

ggml_sycl_fattn_kv_buffers::kv_buffer::~kv_buffer() {
#ifdef DEBUG_SYCL_POOL
    GGML_LOG_INFO("ggml_sycl_fattn_kv_buffer[%d]: %.2f MiB\n", device, capacity / 1024.0 / 1024.0);
#endif
    if (ptr) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, *qptr)));
    }
}
