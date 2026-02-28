//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "cpu-dispatch.hpp"
#include "unified-cache.hpp"

namespace ggml_sycl {

// Persistent thread pool for CPU expert computation.
// Replaces per-call std::async with pre-spawned workers and ring-buffered
// staging slots to avoid per-token allocation overhead.
//
// Thread count default: hardware_concurrency - 2 (reserve for GPU driver).
// Override: GGML_SYCL_CPU_EXPERT_THREADS=N
//
// Ring buffer provides RING_SLOTS pre-allocated staging slots (pinned host
// memory via unified_alloc) for activation inputs and output buffers.
class CpuExpertPool {
public:
    CpuExpertPool() = default;
    ~CpuExpertPool();

    CpuExpertPool(const CpuExpertPool &)             = delete;
    CpuExpertPool & operator=(const CpuExpertPool &) = delete;
    CpuExpertPool(CpuExpertPool &&)                  = delete;
    CpuExpertPool & operator=(CpuExpertPool &&)      = delete;

    // Initialize the pool. Must be called once (e.g. from moe_hybrid_init_once).
    //   n_threads:    worker thread count (0 = auto: hardware_concurrency - 2)
    //   max_experts:  max expert dispatches per MUL_MAT_ID (for ring sizing)
    //   act_dim:      activation vector dimension (floats)
    //   out_dim:      output vector dimension (floats)
    //   q:            SYCL queue for pinned host allocation
    void init(int n_threads, size_t max_experts, size_t act_dim, size_t out_dim,
              sycl::queue & q);

    // Shut down all workers and free ring buffer memory.
    void shutdown();

    // Submit a batch of CPU expert tasks. Returns a future for synchronization.
    // The caller must ensure tasks remain valid until the future completes.
    std::future<void> submit_batch(const cpu_expert_task * tasks, int n_tasks);

    // Ring buffer: acquire a staging slot for up to max_experts_ experts.
    struct StagingSlot {
        float * act     = nullptr;  // Activation input buffer (pinned host)
        float * out     = nullptr;  // Output buffer (pinned host)
        int     slot_id = -1;
    };
    StagingSlot acquire_staging();
    void        release_staging(int slot_id);

    bool is_active() const { return active_.load(std::memory_order_acquire); }

private:
    void worker_thread();

    std::vector<std::thread>              threads_;
    std::queue<std::function<void()>>     work_queue_;
    std::mutex                            mutex_;
    std::condition_variable               cv_;
    std::atomic<bool>                     active_{false};
    std::atomic<bool>                     shutting_down_{false};

    // Ring buffer for staging memory
    static constexpr int RING_SLOTS = 4;
    struct RingEntry {
        float * act    = nullptr;
        float * out    = nullptr;
        bool    in_use = false;
    };
    RingEntry    ring_[RING_SLOTS] = {};
    std::mutex   ring_mutex_;
    size_t       act_stride_   = 0;  // floats per expert activation
    size_t       out_stride_   = 0;  // floats per expert output
    size_t       max_experts_  = 0;
    alloc_handle ring_alloc_;        // Single unified_alloc for all ring buffers
};

}  // namespace ggml_sycl
