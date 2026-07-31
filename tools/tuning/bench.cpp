#include "bench.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

perf_cell build_perf_cell(ggml_backend_t backend, const build_graph_fn & build,
                          const init_tensors_fn & init, const op_flops_fn & flops) {
    perf_cell cell;

    const size_t graph_nodes = 1024;

    ggml_init_params params = {
        /* .mem_size  = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
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
    if (cell.buf == NULL) {
        return cell;
    }

    init(cell.ctx.get());

    cell.gf = ggml_new_graph_custom(cell.ctx.get(), graph_nodes, false);
    ggml_build_forward_expand(cell.gf, out);

    // replicate the op to amortize overhead (target ~50 GFLOP/compute, capped to bound graph size)
    cell.n_runs = 1;
    if (flops(out) > 0) {
        const uint64_t target_flops = 50ULL * 1000 * 1000 * 1000;
        const int      cap          = 512;
        const int      by_flops     = (int) std::min<int64_t>(cap, (int64_t) (target_flops / flops(out)));
        cell.n_runs = std::max(1, std::min<int>(by_flops, (int) (ggml_graph_size(cell.gf) - ggml_graph_n_nodes(cell.gf))));
    }
    for (int i = 1; i < cell.n_runs; ++i) {
        ggml_graph_add_node(cell.gf, out);
    }

    cell.ok = true;

    return cell;
}

double time_cell_median(ggml_backend_t backend, const perf_cell & cell, int reps) {
    if (!cell.ok) {
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
    std::nth_element(samples.begin(), samples.begin() + samples.size()/2, samples.end());

    return samples[samples.size()/2] / cell.n_runs;
}

cell_result measure_cell(ggml_backend_t backend, const perf_cell & cell, int reps,
                         int n_cands, const std::vector<int> & order,
                         const set_candidate_fn & set_cand,
                         const clear_candidate_fn & clear_cand,
                         int baseline_cand, const cooldown_opts & cool,
                         const char * cell_label) {
    cell_result res;
    res.t.assign(n_cands, 0.0);

    double anchor_ref = 0.0;

    for (size_t i = 0; i < order.size(); ++i) {
        set_cand(order[i]);
        res.t[order[i]] = time_cell_median(backend, cell, reps);
        clear_cand();

        if (i % 4 != 0) {
            continue;
        }

        // re-measure the baseline config as an anchor: same config every time, so any
        // change is the machine, not the kernel
        set_cand(baseline_cand);
        const double a = time_cell_median(backend, cell, reps);
        clear_cand();

        if (a <= 0.0) {
            continue;
        }

        res.anchor_min = res.anchor_min > 0.0 ? std::min(res.anchor_min, a) : a;
        res.anchor_max = std::max(res.anchor_max, a);

        if (anchor_ref > 0.0) {
            const double drift = std::fabs(a - anchor_ref) / anchor_ref;
            if (drift > cool.drift) {
                fprintf(stderr, "# WARN throttling? anchor drift %.1f%% %s\n", 100.0*drift, cell_label);
            }
        }

        anchor_ref = anchor_ref > 0.0 ? std::min(anchor_ref, a) : a;
    }

    return res;
}
