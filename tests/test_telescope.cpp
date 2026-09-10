// Standalone test: llama-rollback-telescope CPU reference vs independent replay.
//
// Build (Windows, nvcc as host compiler or any C++17 compiler):
//   nvcc -O2 -std=c++17 -I src test_telescope.cpp src/llama-rollback-telescope.cpp
// or with clang/g++:
//   c++ -O2 -std=c++17 -I src -fopenmp test_telescope.cpp src/llama-rollback-telescope.cpp
//
// Verifies: (1) rollback succeeds within the window with zero fallbacks,
// (2) the forward-telescoped state matches an independent replay of the exact
// per-step map S <- gamma*S - beta*k*(k^T S) + beta*k*v^T (rel-err 0 in fp32).
#include "llama-rollback-telescope.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace llama;

static double rel_err(const std::vector<float> & a, const std::vector<float> & b) {
    double na = 0, nb = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
        d  += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
    }
    return std::sqrt(d / std::max(nb, 1e-30));
}

int main() {
    const uint32_t L = 2, H = 4, d = 32;
    const uint32_t window = 8, slots = 5, ring_cap = window + 1;
    const uint32_t total = 24, p0 = 20;

    telescope_config cfg;
    cfg.window = window; cfg.coverage = window * slots; cfg.slots = slots;
    cfg.L = L; cfg.L_meta = L; cfg.H = H; cfg.d = d; cfg.ring_cap = ring_cap;
    telescope_rollback tel(cfg);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // initial state S0 (kept for independent reference)
    std::vector<float> S0((size_t) L * H * d * d);
    for (auto & x : S0) x = nd(rng) * 0.1f;

    std::vector<std::vector<float>> k_hist(total), v_hist(total), b_hist(total), g_hist(total);

    // forward pass with capture/anchor
    std::vector<float> S = S0;
    for (uint32_t t = 0; t < total; ++t) {
        std::vector<float> k((size_t) L * H * d), v((size_t) L * H * d), b((size_t) L * H), g((size_t) L * H);
        for (uint32_t l = 0; l < L; ++l) for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t c = 0; c < d; ++c) {
                k[(size_t) l * H * d + h * d + c] = nd(rng) * 0.1f;
                v[(size_t) l * H * d + h * d + c] = nd(rng) * 0.1f;
            }
            b[(size_t) l * H + h] = 0.3f + 0.6f * ud(rng);
            // gamma mix: 90% long-memory (0.9-1.0), 10% strong decay (1e-3-0.5)
            g[(size_t) l * H + h] = ud(rng) < 0.9f ? (0.9f + 0.1f * ud(rng)) : (1e-3f + 0.5f * ud(rng));
        }
        k_hist[t] = k; v_hist[t] = v; b_hist[t] = b; g_hist[t] = g;
        tel.capture(t, k.data(), v.data(), b.data(), g.data());

        // forward step (fp32, same formula as telescope)
        for (uint32_t l = 0; l < L; ++l) for (uint32_t h = 0; h < H; ++h) {
            float * Slh = S.data() + ((size_t) l * H + h) * d * d;
            const float * kk = k.data() + (size_t) l * H * d + h * d;
            const float * vv = v.data() + (size_t) l * H * d + h * d;
            const float gg = g[(size_t) l * H + h], bb = b[(size_t) l * H + h];
            for (uint32_t c = 0; c < d; ++c) {
                double w = 0;
                for (uint32_t r = 0; r < d; ++r) w += (double) kk[r] * (double) Slh[(size_t) r * d + c];
                for (uint32_t r = 0; r < d; ++r)
                    Slh[(size_t) r * d + c] = (float)((double) gg * Slh[(size_t) r * d + c]
                        - (double) bb * w * kk[r] + (double) bb * vv[c] * kk[r]);
            }
        }
        // anchor semantics: S_after_pos (state after processing `t`)
        if (t % window == 0) tel.place_anchor(t, S.data());
    }

    // telescope rollback to p0
    std::vector<float> S_out((size_t) L * H * d * d);
    auto res = tel.rollback(p0, S_out.data());
    printf("rollback: ok=%d anchor=%u fallback=%llu\n", res.ok, res.anchor,
           (unsigned long long) res.fallback_count);
    if (!res.ok) { printf("FAIL: rollback not ok\n"); return 1; }
    if (res.fallback_count != 0) { printf("FAIL: unexpected fallback\n"); return 1; }

    // independent reference: forward S0 -> p0 using hist
    std::vector<float> S_ref = S0;
    for (uint32_t t = 0; t < p0; ++t) {
        const auto & k = k_hist[t], & v = v_hist[t], & b = b_hist[t], & g = g_hist[t];
        for (uint32_t l = 0; l < L; ++l) for (uint32_t h = 0; h < H; ++h) {
            float * Slh = S_ref.data() + ((size_t) l * H + h) * d * d;
            const float * kk = k.data() + (size_t) l * H * d + h * d;
            const float * vv = v.data() + (size_t) l * H * d + h * d;
            const float gg = g[(size_t) l * H + h], bb = b[(size_t) l * H + h];
            for (uint32_t c = 0; c < d; ++c) {
                double w = 0;
                for (uint32_t r = 0; r < d; ++r) w += (double) kk[r] * (double) Slh[(size_t) r * d + c];
                for (uint32_t r = 0; r < d; ++r)
                    Slh[(size_t) r * d + c] = (float)((double) gg * Slh[(size_t) r * d + c]
                        - (double) bb * w * kk[r] + (double) bb * vv[c] * kk[r]);
            }
        }
    }

    const double err = rel_err(S_out, S_ref);
    printf("telescope vs independent replay rel-err = %.3e (tolerance 1e-4)\n", err);
    if (err > 1e-4) { printf("FAIL: replay mismatch\n"); return 1; }

    printf("resident_bytes = %.2f MiB\n", tel.resident_bytes() / 1048576.0);
    printf("ALL PASS\n");
    return 0;
}
