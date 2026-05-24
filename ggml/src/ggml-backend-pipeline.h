#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pipelined backend scheduler for hybrid CPU+GPU prefill.
// Wraps an existing ggml_backend_sched and executes its splits
// in a depth-3 pipeline: CPU compute -> TMA transfer -> GPU compute.

typedef struct ggml_backend_sched_pipelined * ggml_backend_sched_pipelined_t;

ggml_backend_sched_pipelined_t ggml_backend_sched_pipelined_init(
    ggml_backend_sched_t base,
    int depth,
    int split_size,
    int n_threads,
    enum ggml_sched_priority prio,
    uint32_t poll,
    ggml_backend_t gpu_backend);

enum ggml_status ggml_backend_sched_pipelined_compute(
    ggml_backend_sched_pipelined_t sched,
    struct ggml_cgraph * gf);

void ggml_backend_sched_pipelined_synchronize(ggml_backend_sched_pipelined_t sched);

void ggml_backend_sched_pipelined_free(ggml_backend_sched_pipelined_t sched);

#ifdef __cplusplus
}
#endif
