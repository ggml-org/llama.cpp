//
// bench_fitness_matmul.cpp
//
// Microbenchmark for ts_awq_relative_output_error on a realistic-sized
// synthetic layer (in_dim=4096, out_dim=4096, n_tokens=256). Used to measure
// the BLAS acceleration of the per-tensor GA fitness hot path before/after
// the cblas_sgemm change.
//
// Build (standalone, mirrors test_all.sh):
//   clang++ -std=c++17 -O2 tools/quantize/tessera/bench_fitness_matmul.cpp \
//     tools/quantize/tessera/tessera-awq.cpp \
//     tools/quantize/tessera/tessera-awq-fitness.cpp \
//     tools/quantize/tessera/tessera-policy.cpp \
//     -I vendor -framework Accelerate -o /tmp/bench_fitness
//
// The weights/activations are filled deterministically so the two matmuls
// exercise a realistic numeric range (and so before/after compare identical
// inputs). The reported fitness is also printed so a parity drift can be
// spotted in the timing run.
//

#include "tessera-awq-fitness.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Simple deterministic LCG so the bench is reproducible across runs and
// does not depend on the std library distribution machinery.
static uint32_t lcg_state = 0x1234abcdu;
static float next_randn_fallback() {
    // Box-Muller on two uniforms from the LCG. Good enough for a benchmark
    // (we only need a realistic numeric spread, not statistical purity).
    lcg_state = 1103515245u * lcg_state + 12345u;
    float u1 = (float)((lcg_state >> 8) + 1) / 16777216.0f;
    lcg_state = 1103515245u * lcg_state + 12345u;
    float u2 = (float)((lcg_state >> 8) + 1) / 16777216.0f;
    float r = sqrtf(-2.0f * logf(u1));
    return r * cosf(6.28318530718f * u2);
}

int main(int argc, char ** argv) {
    int64_t in_dim   = (argc > 1) ? std::atoll(argv[1]) : 4096;
    int64_t out_dim  = (argc > 2) ? std::atoll(argv[2]) : 4096;
    int64_t n_tokens = (argc > 3) ? std::atoll(argv[3]) : 256;
    int     reps     = (argc > 4) ? std::atoi(argv[4])  : 5;

    std::vector<float> W((size_t)out_dim * in_dim);
    std::vector<float> R((size_t)out_dim * in_dim);  // reconstructed
    std::vector<float> A((size_t)n_tokens * in_dim);
    for (auto & v : W) v = next_randn_fallback() * 0.1f;
    for (auto & v : R) v = next_randn_fallback() * 0.1f;
    for (auto & v : A) v = next_randn_fallback();

    // reference = A @ W^T is not provided so the function computes it
    // internally (that is the hot path we want to time).
    double fitness = 0.0;
    double best = 1e30, worst = 0.0;
    for (int it = 0; it < reps; it++) {
        auto t0 = std::chrono::steady_clock::now();
        fitness = ts_awq_relative_output_error(
            A.data(), W.data(), R.data(), n_tokens, in_dim, out_dim, nullptr);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        if (secs < best) best = secs;
        if (secs > worst) worst = secs;
    }
    double flops = 2.0 * (double)n_tokens * (double)out_dim * (double)in_dim;
    printf("size: n_tokens=%lld in_dim=%lld out_dim=%lld reps=%d\n",
           (long long)n_tokens, (long long)in_dim, (long long)out_dim, reps);
    printf("fitness=%.8e\n", fitness);
    // Two matmuls (ref + approx) per call: total work = 2 * flops.
    printf("best=%.4fs worst=%.4fs (2 matmuls, %.3f GFLOP total)\n",
           best, worst, 2.0 * flops / 1e9);
    printf("throughput=%.2f GFLOP/s (at best)\n", (2.0 * flops / 1e9) / best);
    return 0;
}
