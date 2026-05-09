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

#ifndef GGML_SYCL_FATTN_BUFFERS_HPP
#define GGML_SYCL_FATTN_BUFFERS_HPP

#include <sycl/sycl.hpp>

typedef sycl::queue *queue_ptr;

struct ggml_sycl_fattn_kv_buffers {
    // buffers grow in chunks of this size
    static constexpr size_t CHUNK_SIZE = 16ull << 20; // 16 MiB

    struct kv_buffer {
        kv_buffer(queue_ptr qptr_, int device_) : qptr(qptr_), device(device_) {}
        ~kv_buffer();

        kv_buffer(const kv_buffer &) = delete;
        kv_buffer & operator=(const kv_buffer &) = delete;

        sycl::half * ensure_half(size_t n_elems);

    private:
        sycl::half * ptr      = nullptr;
        size_t       capacity = 0;
        queue_ptr    qptr     = nullptr;
        [[maybe_unused]] int device = 0;
    };

    kv_buffer K;
    kv_buffer V;

    ggml_sycl_fattn_kv_buffers(queue_ptr qptr, int device) : K(qptr, device), V(qptr, device) {}

    ggml_sycl_fattn_kv_buffers(const ggml_sycl_fattn_kv_buffers &) = delete;
    ggml_sycl_fattn_kv_buffers & operator=(const ggml_sycl_fattn_kv_buffers &) = delete;
};

/**
 * Imitates `ggml_sycl_pool_alloc` to keep the code calling alloc unchanged.
 */
struct ggml_sycl_fattn_alloc {
    ggml_sycl_fattn_kv_buffers::kv_buffer & buf;
    sycl::half *                         ptr = nullptr;

    explicit ggml_sycl_fattn_alloc(ggml_sycl_fattn_kv_buffers::kv_buffer & buf_) : buf(buf_) {}

    sycl::half * alloc(size_t n_elems) {
        ptr = buf.ensure_half(n_elems);
        return ptr;
    }
};
#endif
