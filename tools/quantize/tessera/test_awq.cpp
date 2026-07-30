#include "tessera-awq.h"

#include <cassert>
#include <cmath>
#include <cstdio>

static ts_awq_score test_eval(const ts_awq_candidate * cand,
                               const ts_awq_layer * layer, void * ctx) {
    (void)layer;
    (void)ctx;
    ts_awq_score s;
    float d = cand->alpha - 0.5f;
    s.mse = d * d;
    s.relative_frob = 0.0f;
    s.heldout_mse = 0.0f;
    s.composite = -(d * d);
    return s;
}

int main() {
    float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    ts_awq_layer layer;
    layer.name = "test_layer";
    layer.family = "ffn";
    layer.weights = weights;
    layer.act_scales = nullptr;
    layer.calib_X = nullptr;
    layer.ref_output = nullptr;
    layer.imatrix = nullptr;
    layer.out_dim = 4;
    layer.in_dim = 4;
    layer.n_tokens = 0;
    layer.kurtosis = 3.0f;
    layer.eff_rank = 0.5f;

    ts_awq_evolve_params params;
    params.population = 8;
    params.generations = 20;
    params.islands = 2;
    params.migration_interval = 10;
    params.mutation_sigma = 0.1f;
    params.crossover_rate = 0.7f;
    params.heldout_weight = 2.0f;
    params.seed = 42;
    params.verbose = false;

    // Run 1: convergence
    ts_awq_evolve_result r1;
    int rc = ts_awq_evolve(&layer, test_eval, nullptr, &params, &r1);
    assert(rc == 0);
    printf("run1: alpha=%.4f composite=%.6f evals=%lld\n",
           r1.best.alpha, r1.best_score.composite, (long long)r1.evaluations);
    assert(fabsf(r1.best.alpha - 0.5f) < 0.15f);

    // Run 2: determinism
    ts_awq_evolve_result r2;
    rc = ts_awq_evolve(&layer, test_eval, nullptr, &params, &r2);
    assert(rc == 0);
    assert(r1.best.alpha == r2.best.alpha);
    assert(r1.best.clip == r2.best.clip);
    assert(r1.best.lrq_rank_frac == r2.best.lrq_rank_frac);
    assert(r1.best.rotation_lr == r2.best.rotation_lr);
    assert(r1.best_score.composite == r2.best_score.composite);
    assert(r1.evaluations == r2.evaluations);

    // Archive sanity
    assert(!r1.archive.empty());

    // JSON sanity
    std::string json = ts_awq_candidate_json(&r1.best);
    assert(json.find("\"alpha\"") != std::string::npos);

    printf("PASS\n");
    return 0;
}
