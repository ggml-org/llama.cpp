//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "cpu-expert-pool.hpp"

#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace ggml_sycl {

// ---------------------------------------------------------------------------
// Env var: GGML_SYCL_CPU_EXPERT_THREADS=N overrides thread count.
// ---------------------------------------------------------------------------
static int get_cpu_expert_thread_count() {
    const char * env = std::getenv("GGML_SYCL_CPU_EXPERT_THREADS");
    if (env) {
        int n = std::atoi(env);
        if (n > 0) {
            return n;
        }
    }
    // Default: hardware_concurrency - 2 (reserve for GPU driver threads)
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, hw - 2);
}

// ---------------------------------------------------------------------------
// CpuExpertPool implementation
// ---------------------------------------------------------------------------

void CpuExpertPool::init(int n_threads, size_t max_experts,
                          size_t act_dim, size_t out_dim,
                          sycl::queue & q) {
    if (active_.load(std::memory_order_acquire)) {
        GGML_LOG_WARN("[CPU-EXPERT-POOL] init() called on already-active pool, ignoring\n");
        return;
    }

    max_experts_ = max_experts;
    act_stride_  = act_dim;
    out_stride_  = out_dim;

    if (n_threads <= 0) {
        n_threads = get_cpu_expert_thread_count();
    }

    // Allocate ring buffer via unified cache (pinned host memory)
    const size_t act_bytes_per_slot = max_experts * act_dim * sizeof(float);
    const size_t out_bytes_per_slot = max_experts * out_dim * sizeof(float);
    const size_t per_slot           = act_bytes_per_slot + out_bytes_per_slot;
    const size_t total              = per_slot * RING_SLOTS;

    if (total > 0) {
        alloc_request req;
        req.queue  = &q;
        req.device = -1;  // Host allocation
        req.size   = total;
        req.intent = { alloc_role::EXPERT_STAGING,
                       runtime_category::EXPERT_CACHE, "cpu_expert_ring",
                       { .must_host_pinned = true } };

        if (!unified_alloc(req, &ring_alloc_)) {
            GGML_LOG_WARN("[CPU-EXPERT-POOL] Failed to allocate ring buffer "
                          "(%zu bytes), falling back to std::async\n", total);
            return;
        }

        char * base = static_cast<char *>(ring_alloc_.ptr);
        for (int i = 0; i < RING_SLOTS; i++) {
            ring_[i].act    = reinterpret_cast<float *>(
                base + static_cast<size_t>(i) * per_slot);
            ring_[i].out    = reinterpret_cast<float *>(
                base + static_cast<size_t>(i) * per_slot + act_bytes_per_slot);
            ring_[i].in_use = false;
        }
    }

    // Spawn worker threads
    shutting_down_.store(false, std::memory_order_release);
    active_.store(true, std::memory_order_release);
    threads_.reserve(n_threads);
    for (int i = 0; i < n_threads; i++) {
        threads_.emplace_back(&CpuExpertPool::worker_thread, this);
    }

    GGML_LOG_INFO("[CPU-EXPERT-POOL] Initialized: %d threads, %d ring slots, "
                  "%.1f KB/slot (act=%.1f KB, out=%.1f KB)\n",
                  n_threads, RING_SLOTS,
                  per_slot / 1024.0,
                  act_bytes_per_slot / 1024.0,
                  out_bytes_per_slot / 1024.0);
}

void CpuExpertPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return shutting_down_.load(std::memory_order_acquire)
                       || !work_queue_.empty();
            });
            if (shutting_down_.load(std::memory_order_acquire)
                && work_queue_.empty()) {
                return;
            }
            task = std::move(work_queue_.front());
            work_queue_.pop();
        }
        task();
    }
}

std::future<void> CpuExpertPool::submit_batch(const cpu_expert_task * tasks,
                                               int n_tasks) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future  = promise->get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        work_queue_.push([tasks, n_tasks, promise]() {
            ggml_sycl_cpu_expert_mul_mat_batched(tasks, n_tasks);
            promise->set_value();
        });
    }
    cv_.notify_one();
    return future;
}

CpuExpertPool::StagingSlot CpuExpertPool::acquire_staging() {
    std::lock_guard<std::mutex> lock(ring_mutex_);
    for (int i = 0; i < RING_SLOTS; i++) {
        if (!ring_[i].in_use) {
            ring_[i].in_use = true;
            // Zero the output buffer for clean accumulation
            if (ring_[i].out && max_experts_ > 0 && out_stride_ > 0) {
                std::memset(ring_[i].out, 0,
                            max_experts_ * out_stride_ * sizeof(float));
            }
            return { ring_[i].act, ring_[i].out, i };
        }
    }
    // All slots in use — caller should fall back to external allocation
    return { nullptr, nullptr, -1 };
}

void CpuExpertPool::release_staging(int slot_id) {
    if (slot_id < 0 || slot_id >= RING_SLOTS) {
        return;
    }
    std::lock_guard<std::mutex> lock(ring_mutex_);
    ring_[slot_id].in_use = false;
}

void CpuExpertPool::shutdown() {
    if (!active_.load(std::memory_order_acquire)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutting_down_.store(true, std::memory_order_release);
    }
    cv_.notify_all();

    for (auto & t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    threads_.clear();
    active_.store(false, std::memory_order_release);

    if (ring_alloc_.ptr) {
        unified_free(ring_alloc_);
        ring_alloc_ = {};
    }

    // Reset ring entries
    for (int i = 0; i < RING_SLOTS; i++) {
        ring_[i] = {};
    }

    GGML_LOG_INFO("[CPU-EXPERT-POOL] Shut down\n");
}

CpuExpertPool::~CpuExpertPool() {
    shutdown();
}

}  // namespace ggml_sycl
