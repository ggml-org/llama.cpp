#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <functional>
#include <vector>

// one prebuilt op graph, replicated n_runs times so a single graph_compute amortizes
// dispatch/sync overhead. reused across candidates: an override only changes which
// pipeline is picked at encode time, so the (large) input tensors stay allocated.
struct perf_cell {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_cgraph *           gf     = nullptr;
    int                     n_runs = 0;
    bool                    ok     = false;
};

// builds the op graph for one shape. returns the output tensor, or null if unsupported.
using build_graph_fn = std::function<ggml_tensor *(ggml_context *)>;
// fills the allocated tensors of ctx with input data
using init_tensors_fn = std::function<void(ggml_context *)>;
// flops of one op instance, used to size n_runs
using op_flops_fn = std::function<uint64_t(ggml_tensor *)>;

perf_cell build_perf_cell(ggml_backend_t backend, const build_graph_fn & build,
                          const init_tensors_fn & init, const op_flops_fn & flops);

// median per-op time (us) over the prebuilt cell for whatever config is currently set
double time_cell_median(ggml_backend_t backend, const perf_cell & cell, int reps);

struct cooldown_opts {
    bool   enabled   = true;
    double drift     = 0.10;  // anchor drift that triggers a cooldown
    double eps       = 0.03;  // anchor tolerance to call the GPU cool again
    int    max_wait  = 120;   // seconds of cooling per cell before giving up
    int    max_retry = 2;     // re-measure rounds per cell before giving up
};

// applies candidate i (an index into the tuner's own candidate list)
using set_candidate_fn = std::function<void(int)>;
// undoes the last set_candidate
using clear_candidate_fn = std::function<void()>;

// giving up on a cell returns early with trusted == false, so a trusted cell is one every
// candidate of was measured; callers may still see a non-positive t[] from a failed measure.
struct cell_result {
    std::vector<double> t;               // time (us) per candidate index, <= 0 if not measured
    bool                trusted = true;  // false -> caller must drop this cell
    double              anchor_min = 0.0;
    double              anchor_max = 0.0;
    int                 n_cooldowns = 0;
    int                 n_remeasures = 0;
};

// times every candidate over the prebuilt cell, re-measuring a periodic baseline anchor
// to watch for thermal drift. order[] gives the (shuffled) visiting order; baseline_cand is
// the candidate the anchor forces, so drift is measured against a config the tuner controls.
cell_result measure_cell(ggml_backend_t backend, const perf_cell & cell, int reps,
                         int n_cands, const std::vector<int> & order,
                         const set_candidate_fn & set_cand,
                         const clear_candidate_fn & clear_cand,
                         int baseline_cand, const cooldown_opts & cool,
                         const char * cell_label);
