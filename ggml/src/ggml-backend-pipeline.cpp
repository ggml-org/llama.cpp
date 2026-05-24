#include "ggml-backend-pipeline.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstring>
#include <algorithm>

// NOTE: This file is part of ggml-base and must NOT depend on CUDA.
// TMA integration is done at the llama-context level (Task 4) where
// both CPU and CUDA backends are linked.

#define GGML_PIPELINE_MAX_EVENTS 2

struct ggml_backend_sched_pipelined {
    ggml_backend_sched_t base;
    int depth;
    int split_size;

    // CPU threadpools (dual CCD pairs)
    ggml_threadpool_params tp_params[2];
    ggml_threadpool_t cpu_tp[2];
    int num_tp;
    int active_pool;
    ggml_backend_t cpu_backend;

    // GPU
    ggml_backend_t gpu_backend;
    ggml_backend_event_t stage_events[GGML_PIPELINE_MAX_EVENTS];  // ping-pong
    ggml_backend_dev_t gpu_device;

    bool initialized;
};

ggml_backend_sched_pipelined_t ggml_backend_sched_pipelined_init(
    ggml_backend_sched_t base,
    int depth,
    int split_size,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll,
    ggml_backend_t gpu_backend) {

    auto * sched = new ggml_backend_sched_pipelined();
    sched->base = base;
    sched->depth = depth;
    sched->split_size = split_size;
    sched->cpu_backend = nullptr;
    sched->gpu_backend = gpu_backend;
    sched->num_tp = 0;
    sched->active_pool = 0;
    sched->gpu_device = nullptr;
    sched->stage_events[0] = nullptr;
    sched->stage_events[1] = nullptr;
    sched->initialized = false;

    // Initialize dual threadpools from CCD pairs
    int threads_per_pair = n_threads / 2;
    sched->num_tp = ggml_cpu_init_dual_threadpool(sched->tp_params, 2,
                                                   threads_per_pair, prio, poll);

    for (int i = 0; i < sched->num_tp; i++) {
        sched->cpu_tp[i] = ggml_threadpool_new(&sched->tp_params[i]);
    }

    // Fallback: if dual pool fails, use single pool
    if (sched->num_tp < 2) {
        sched->num_tp = 1;
        memset(&sched->tp_params[0], 0, sizeof(ggml_threadpool_params));
        sched->tp_params[0].n_threads = n_threads;
        sched->tp_params[0].prio = prio;
        sched->tp_params[0].poll = poll;
        sched->cpu_tp[0] = ggml_threadpool_new(&sched->tp_params[0]);
    }

    // Initialize backend events for synchronization
    if (gpu_backend) {
        sched->gpu_device = ggml_backend_get_device(gpu_backend);
        if (sched->gpu_device) {
            for (int i = 0; i < GGML_PIPELINE_MAX_EVENTS; i++) {
                sched->stage_events[i] = ggml_backend_event_new(sched->gpu_device);
            }
        }
    }

    sched->initialized = true;
    return sched;
}

enum ggml_status ggml_backend_sched_pipelined_compute(
    ggml_backend_sched_pipelined_t sched,
    struct ggml_cgraph * gf) {

    if (!sched || !sched->initialized) {
        return ggml_backend_sched_graph_compute_async(sched->base, gf);
    }

    // Allocate the graph (creates splits)
    if (!ggml_backend_sched_alloc_graph(sched->base, gf)) {
        return GGML_STATUS_FAILED;
    }

    // Get CPU backend from scheduler
    if (!sched->cpu_backend) {
        sched->cpu_backend = ggml_backend_sched_get_backend(sched->base, 0);
    }

    int n_splits = ggml_backend_sched_get_n_splits(sched->base);
    if (n_splits <= 1) {
        // No benefit to pipelining with 0-1 splits
        return ggml_backend_sched_graph_compute_async(sched->base, gf);
    }

    // Pipeline execution with threadpool rotation.
    // The dual CCD pools provide NUMA-local execution per split.
    // Full async TMA integration requires access to the scheduler's
    // internal split data (private to ggml-backend.cpp) and will be
    // implemented once the internal API is available.

    // Set threadpool for CPU backend using the direct API
    ggml_backend_cpu_set_threadpool(sched->cpu_backend, sched->cpu_tp[sched->active_pool]);

    enum ggml_status status = ggml_backend_sched_graph_compute_async(sched->base, gf);

    // Rotate pool for next call
    if (sched->num_tp >= 2) {
        sched->active_pool = 1 - sched->active_pool;
    }

    return status;
}

void ggml_backend_sched_pipelined_synchronize(ggml_backend_sched_pipelined_t sched) {
    if (!sched) return;
    ggml_backend_sched_synchronize(sched->base);
}

void ggml_backend_sched_pipelined_free(ggml_backend_sched_pipelined_t sched) {
    if (!sched) return;

    if (sched->gpu_device) {
        for (int i = 0; i < GGML_PIPELINE_MAX_EVENTS; i++) {
            if (sched->stage_events[i]) {
                ggml_backend_event_free(sched->stage_events[i]);
            }
        }
    }

    for (int i = 0; i < sched->num_tp; i++) {
        if (sched->cpu_tp[i]) {
            ggml_threadpool_free(sched->cpu_tp[i]);
        }
    }

    delete sched;
}
