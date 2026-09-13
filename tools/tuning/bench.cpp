#include "bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <utility>

perf_cell build_perf_cell(ggml_backend_t          backend,
                          const build_graph_fn &  build,
                          const init_tensors_fn & init,
                          const op_flops_fn &     flops) {
    perf_cell cell;

    const size_t graph_nodes = 1024;

    ggml_init_params params = {
        /* .mem_size  = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(graph_nodes, false),
        /* .mem_base  = */ NULL,
        /* .no_alloc  = */ true,
    };

    cell.ctx.reset(ggml_init(params));
    GGML_ASSERT(cell.ctx);

    ggml_tensor * out = build(cell.ctx.get());
    if (!out || !ggml_backend_supports_op(backend, out)) {
        return cell;
    }

    cell.buf.reset(ggml_backend_alloc_ctx_tensors(cell.ctx.get(), backend));
    if (!cell.buf) {
        return cell;
    }

    init(cell.ctx.get());

    cell.gf = ggml_new_graph_custom(cell.ctx.get(), graph_nodes, false);
    ggml_build_forward_expand(cell.gf, out);

    // replicate the op to amortize overhead (target ~50 GFLOP/compute, capped to bound graph size)
    cell.n_runs            = 1;
    const uint64_t n_flops = flops(out);
    if (n_flops > 0) {
        const uint64_t target_flops = 50ULL * 1000 * 1000 * 1000;
        const int      cap          = 512;
        const int      by_flops     = (int) std::min<int64_t>(cap, (int64_t) (target_flops / n_flops));
        cell.n_runs =
            std::max(1, std::min<int>(by_flops, (int) (ggml_graph_size(cell.gf) - ggml_graph_n_nodes(cell.gf))));
    }
    for (int i = 1; i < cell.n_runs; ++i) {
        ggml_graph_add_node(cell.gf, out);
    }

    return cell;
}

double time_cell_median(ggml_backend_t backend, const perf_cell & cell, int reps) {
    if (cell.gf == nullptr) {
        return -1.0;
    }

    ggml_backend_graph_compute(backend, cell.gf);  // warmup (compiles the pipeline for this config)
    ggml_backend_synchronize(backend);

    std::vector<double> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, cell.gf);
        ggml_backend_synchronize(backend);
        samples.push_back((double) (ggml_time_us() - t0));
    }
    std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());

    return samples[samples.size() / 2] / cell.n_runs;
}

static double measure_one(ggml_backend_t             backend,
                          const perf_cell &          cell,
                          int                        reps,
                          const set_candidate_fn &   set_cand,
                          const clear_candidate_fn & clear_cand,
                          int                        cand) {
    set_cand(cand);
    const double t = time_cell_median(backend, cell, reps);
    clear_cand();

    return t;
}

// waits for the anchor to come back within eps of anchor_ref, with exponential backoff.
// returns the converged anchor, or -1 if it never converged within max_wait.
static double cool_until_steady(ggml_backend_t             backend,
                                const perf_cell &          cell,
                                int                        reps,
                                const set_candidate_fn &   set_cand,
                                const clear_candidate_fn & clear_cand,
                                int                        baseline_cand,
                                double &                   anchor_ref,
                                const cooldown_opts &      cool,
                                const char *               cell_label) {
    int total_wait = 0;

    for (int sleep_s = 2; total_wait < cool.max_wait; sleep_s = std::min(sleep_s * 2, 32)) {
        const int this_wait = std::min(sleep_s, cool.max_wait - total_wait);

        fprintf(stderr, "# COOL sleeping %ds (%ds/%ds) %s\n", this_wait, total_wait + this_wait, cool.max_wait,
                cell_label);
        std::this_thread::sleep_for(std::chrono::seconds(this_wait));
        total_wait += this_wait;

        const double a = measure_one(backend, cell, reps, set_cand, clear_cand, baseline_cand);
        if (a <= 0.0) {
            continue;
        }

        // a faster anchor means the machine got cooler than anything seen so far: adopt it
        if (a < anchor_ref) {
            anchor_ref = a;
        }

        if (a <= anchor_ref * (1.0 + cool.eps)) {
            fprintf(stderr, "# COOL steady after %ds %s\n", total_wait, cell_label);
            return a;
        }
    }

    fprintf(stderr, "# COOL gave up after %ds %s\n", total_wait, cell_label);

    return -1.0;
}

cell_result measure_cell(ggml_backend_t             backend,
                         const perf_cell &          cell,
                         int                        reps,
                         const std::vector<int> &   order,
                         const set_candidate_fn &   set_cand,
                         const clear_candidate_fn & clear_cand,
                         int                        baseline_cand,
                         const cooldown_opts &      cool,
                         const char *               cell_label) {
    cell_result res;
    res.t.assign(order.size(), 0.0);

    double anchor_ref = 0.0;

    // anchors accepted as clean, as (value, position in order[]). the dirty window starts
    // at the position of the last anchor still within eps of anchor_ref, so a downward
    // drift (anchor_ref dropping) naturally widens the window to the whole cell.
    std::vector<std::pair<double, size_t>> anchors;

    auto window_start = [&]() -> size_t {
        for (size_t i = anchors.size(); i-- > 0;) {
            if (anchors[i].first <= anchor_ref * (1.0 + cool.eps)) {
                return anchors[i].second;
            }
        }
        return 0;  // no clean anchor left -> the whole cell is suspect
    };

    int retries_left = cool.max_retry;

    for (size_t i = 0; i < order.size(); ++i) {
        res.t[order[i]] = measure_one(backend, cell, reps, set_cand, clear_cand, order[i]);

        if (i % 4 != 0) {
            continue;
        }

        const double a = measure_one(backend, cell, reps, set_cand, clear_cand, baseline_cand);
        if (a <= 0.0) {
            continue;
        }

        res.anchor_min = res.anchor_min > 0.0 ? std::min(res.anchor_min, a) : a;
        res.anchor_max = std::max(res.anchor_max, a);

        if (anchor_ref == 0.0) {
            anchor_ref = a;
            anchors.push_back({ a, i });
            continue;
        }

        const double drift = std::fabs(a - anchor_ref) / anchor_ref;

        // a cooler anchor than any so far becomes the reference: whatever was measured
        // before it was measured on a hotter machine
        if (a < anchor_ref) {
            anchor_ref = a;
        }

        if (drift <= cool.drift) {
            anchors.push_back({ a, i });
            continue;
        }

        fprintf(stderr, "# WARN throttling? anchor drift %.1f%% %s\n", 100.0 * drift, cell_label);

        if (!cool.enabled) {
            anchors.push_back({ a, i });
            continue;
        }

        if (retries_left <= 0) {
            fprintf(stderr, "# DIRTY retries exhausted %s\n", cell_label);
            res.trusted = false;
            return res;
        }

        const size_t dirty_from = window_start();

        const double a_cool =
            cool_until_steady(backend, cell, reps, set_cand, clear_cand, baseline_cand, anchor_ref, cool, cell_label);
        if (a_cool <= 0.0) {
            res.trusted = false;
            return res;
        }

        // the converged anchor is the only clean one now; re-measure the dirty window from it
        anchors.clear();
        anchors.push_back({ a_cool, dirty_from });

        retries_left--;

        fprintf(stderr, "# REDO candidates %zu..%zu %s\n", dirty_from, i, cell_label);
        for (size_t j = dirty_from; j <= i; ++j) {
            res.t[order[j]] = measure_one(backend, cell, reps, set_cand, clear_cand, order[j]);
        }
    }

    return res;
}
