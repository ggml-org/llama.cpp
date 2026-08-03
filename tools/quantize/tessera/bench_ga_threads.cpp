//
// bench_ga_threads.cpp
//
// Verifies ts_awq_evolve_all produces identical results whether run serial
// (n_threads=1) or parallel (n_threads>1), and times the wall-clock speedup
// on a multi-layer synthetic workload. Each layer's GA is independent and
// seeded by params.seed + layer_index, so the parallel path must match the
// serial path bit-for-bit on every layer's best alpha / best composite.
//

#include "tessera-awq.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static uint32_t g_st = 0xdeadbeefu;
static float nxt() {
    g_st = 1103515245u * g_st + 12345u;
    float u1 = (float)((g_st >> 8) + 1) / 16777216.0f;
    g_st = 1103515245u * g_st + 12345u;
    float u2 = (float)((g_st >> 8) + 1) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

int main(int argc, char ** argv) {
    int n_layers   = (argc > 1) ? std::atoi(argv[1]) : 32;
    int64_t in_dim  = (argc > 2) ? std::atoll(argv[2]) : 512;
    int64_t out_dim = (argc > 3) ? std::atoll(argv[3]) : 512;
    int n_threads  = (argc > 4) ? std::atoi(argv[4]) : (int)std::thread::hardware_concurrency();

    // Build n_layers independent layers with second_moment so the Python
    // parity fitness path (ts_awq_evaluate_layer) runs - that is the path
    // whose matmul we accelerated.
    std::vector<std::vector<float>> wbuf(n_layers), smbuf(n_layers);
    std::vector<ts_awq_layer> layers(n_layers);
    for (int l = 0; l < n_layers; l++) {
        wbuf[l].assign((size_t)out_dim * in_dim, 0.0f);
        smbuf[l].assign((size_t)in_dim, 0.0f);
        for (auto & v : wbuf[l])  v = nxt() * 0.1f;
        for (auto & v : smbuf[l]) v = 0.5f + nxt() * 0.5f;
        ts_awq_layer & L = layers[l];
        L.name        = "L" + std::to_string(l);
        L.family      = "ffn";
        L.weights     = wbuf[l].data();
        L.second_moment = smbuf[l].data();
        L.fourth_moment = nullptr;
        L.max_abs       = nullptr;
        L.train_activations = nullptr;
        L.heldout_activations = nullptr;
        L.out_dim = out_dim;
        L.in_dim  = in_dim;
        L.n_tokens = 0;
        L.n_tokens_h = 0;
    }

    ts_awq_evolve_params params = {};
    params.population         = 8;
    params.generations        = 8;
    params.islands            = 2;
    params.migration_interval = 10;
    params.mutation_sigma     = 0.1f;
    params.crossover_rate     = 0.7f;
    params.heldout_weight     = 2.0f;
    params.seed               = 42;
    params.verbose            = false;

    auto run = [&](int nt, std::vector<ts_awq_evolve_result> & res) {
        ts_awq_evolve_params p = params;
        p.n_threads = nt;
        res.clear();
        auto t0 = std::chrono::steady_clock::now();
        int rc = ts_awq_evolve_all(layers.data(), (int64_t)layers.size(),
                                   ts_awq_default_eval, nullptr, &p, &res);
        double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        printf("n_threads=%2d rc=%d time=%.3fs (n_layers=%d)\n",
               nt, rc, secs, n_layers);
        return secs;
    };

    std::vector<ts_awq_evolve_result> r_ser, r_par;
    double t_ser = run(1, r_ser);
    double t_par = run(n_threads, r_par);

    // Determinism: compare best alpha + composite on every layer.
    int mism = 0;
    for (int l = 0; l < n_layers; l++) {
        float a_s = r_ser[l].best.genes.alpha;
        float a_p = r_par[l].best.genes.alpha;
        float c_s = r_ser[l].best_score.composite;
        float c_p = r_par[l].best_score.composite;
        if (a_s != a_p || c_s != c_p) {
            mism++;
            if (mism <= 3) {
                printf("  MISMATCH layer %d: alpha %.6f vs %.6f ; comp %.6e vs %.6e\n",
                       l, a_s, a_p, c_s, c_p);
            }
        }
    }
    printf("determinism: %d/%d layers match exactly\n",
           n_layers - mism, n_layers);
    printf("speedup: %.2fx (serial %.3fs -> parallel %.3fs with %d threads)\n",
           t_ser / t_par, t_ser, t_par, n_threads);
    return mism == 0 ? 0 : 1;
}
