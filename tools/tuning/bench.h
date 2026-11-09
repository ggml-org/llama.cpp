#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdint>
#include <functional>
#include <vector>

// A prebuilt graph replicated to amortize dispatch and synchronization overhead.
struct perf_cell {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_cgraph *           gf     = nullptr;
    int                     n_runs = 0;
};

using build_graph_fn  = std::function<ggml_tensor *(ggml_context *)>;
using init_tensors_fn = std::function<void(ggml_context *)>;
using op_flops_fn     = std::function<uint64_t(ggml_tensor *)>;

perf_cell build_perf_cell(ggml_backend_t          backend,
                          const build_graph_fn &  build,
                          const init_tensors_fn & init,
                          const op_flops_fn &     flops);

double time_cell_median(ggml_backend_t backend, const perf_cell & cell, int reps);

struct cooldown_opts {
    bool   enabled   = true;
    double drift     = 0.10;  // anchor drift that triggers a cooldown
    double eps       = 0.03;  // anchor tolerance to call the GPU cool again
    int    max_wait  = 120;  // seconds of cooling per cell before giving up
    int    max_retry = 2;    // re-measure rounds per cell before giving up
};

using set_candidate_fn   = std::function<void(int)>;
using clear_candidate_fn = std::function<void()>;

struct cell_result {
    std::vector<double> t;
    bool                trusted    = true;
    double              anchor_min = 0.0;
    double              anchor_max = 0.0;
};

// Times candidates in order while using baseline_cand as a thermal-drift anchor.
cell_result measure_cell(ggml_backend_t             backend,
                         const perf_cell &          cell,
                         int                        reps,
                         const std::vector<int> &   order,
                         const set_candidate_fn &   set_cand,
                         const clear_candidate_fn & clear_cand,
                         int                        baseline_cand,
                         const cooldown_opts &      cool,
                         const char *               cell_label);
