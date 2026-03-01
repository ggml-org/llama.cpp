//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

// PinnedBufferPool implementation.
// Extracted from expert-cache.cpp during ExpertCache removal (Task 7).
// Provides ring-buffered host-pinned staging buffers for CPU expert dispatch.

#include "pinned-buffer-pool.hpp"

#include "common.hpp"
#include "unified-cache.hpp"

#include <cassert>

namespace ggml_sycl {

PinnedBufferPool::~PinnedBufferPool() {
    shutdown();
}

void PinnedBufferPool::init(sycl::queue & q, int device_id, size_t max_experts,
                             size_t act_dim, size_t out_dim) {
    if (is_initialized()) {
        return;
    }

    device_id_   = device_id;
    act_stride_  = act_dim;
    out_stride_  = out_dim;
    max_experts_ = max_experts;

    const size_t act_bytes = max_experts * act_dim * sizeof(float);
    const size_t out_bytes = max_experts * out_dim * sizeof(float);

    // Allocate activation pool via unified_alloc with pinned host constraint
    alloc_request req_act;
    req_act.queue                               = &q;
    req_act.device                              = device_id;
    req_act.size                                = act_bytes;
    req_act.intent.role                         = alloc_role::EXPERT_STAGING;
    req_act.intent.category                     = runtime_category::EXPERT_CACHE;
    req_act.intent.cohort_id                    = "moe_act_pool";
    req_act.intent.constraints.must_host_pinned = true;

    if (!unified_alloc(req_act, &act_alloc_)) {
        GGML_LOG_WARN("[MOE-POOL] Failed to allocate activation pool (%zu bytes)\n", act_bytes);
        return;
    }
    act_pool_ = static_cast<float *>(act_alloc_.ptr);

    // Allocate output pool
    alloc_request req_out    = req_act;
    req_out.size             = out_bytes;
    req_out.intent.cohort_id = "moe_out_pool";

    if (!unified_alloc(req_out, &out_alloc_)) {
        GGML_LOG_WARN("[MOE-POOL] Failed to allocate output pool (%zu bytes)\n", out_bytes);
        unified_free(act_alloc_);
        act_alloc_ = {};
        act_pool_  = nullptr;
        return;
    }
    out_pool_ = static_cast<float *>(out_alloc_.ptr);

    GGML_LOG_INFO("[MOE-POOL] Pinned buffer pool: act=%zu KB, out=%zu KB, max_experts=%zu\n",
                  act_bytes / 1024, out_bytes / 1024, max_experts);
}

void PinnedBufferPool::shutdown() {
    if (act_alloc_.ptr) {
        unified_free(act_alloc_);
        act_alloc_ = {};
        act_pool_  = nullptr;
    }
    if (out_alloc_.ptr) {
        unified_free(out_alloc_);
        out_alloc_ = {};
        out_pool_  = nullptr;
    }
}

PinnedBufferPool::BufferPair PinnedBufferPool::acquire(size_t n_experts) {
    GGML_ASSERT(n_experts <= max_experts_ && "Expert count exceeds pool capacity");
    GGML_ASSERT(act_pool_ && out_pool_ && "Pool not initialized");
    return { act_pool_, out_pool_ };
}

void PinnedBufferPool::release(BufferPair) {
    // No-op: CPU vec_dot kernels write every output element that the scatter
    // loop reads back (n_cpu * N floats), so zeroing is unnecessary.
    // Stale data in unused pool slots is never accessed.
}

}  // namespace ggml_sycl
