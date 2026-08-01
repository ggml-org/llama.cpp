#include "tessera-flrq.h"
#include "tessera-linalg.h"

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define TS_HAS_CBLAS 1
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#define TS_HAS_CBLAS 1
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>

// --- FLRQ (Factored Low-Rank Quantization, Jan 2026) ---
//
// Mirrors tools/tessera/per_tensor_calibrate.py (flrq_sketch, flrq_bcl,
// flrq_select_rank). The BLC step is the deterministic numerical core and is
// what the Python parity test pins; the sketch step uses a C++ RNG (xoshiro
// Box-Muller) so it is reproducible but not bit-identical to numpy's PCG64,
// hence the parity test loads the basis from the fixture instead.

namespace {

// xoshiro128** PRNG -- a small, fast, well-mixed generator seeded from a
// uint32 (splitmix64 expands the seed). Used only for the calibration-free
// Gaussian sketch; the parity-pinned BLC step takes the basis as input.
struct ts_rng {
    uint32_t s[4];
    explicit ts_rng(uint32_t seed) {
        // splitmix64 to expand the 32-bit seed into four 64-bit words, then
        // truncate to the 32-bit state xoshiro128** uses.
        uint64_t z = seed;
        for (int i = 0; i < 4; i++) {
            z = (z + 0x9E3779B97F4A7C15ULL);
            uint64_t t = z;
            t = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
            t = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
            t = t ^ (t >> 31);
            s[i] = (uint32_t)t;
        }
    }
    uint32_t next() {
        const uint32_t result = (uint32_t)(s[1] * 5u) << 7 | ((s[1] * 5u) >> 25);
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 11) | (s[3] >> 21);
        return result;
    }
    double u01() {
        return (double)next() * (1.0 / 4294967296.0);
    }
    double gaussian() {
        // Box-Muller (matches numpy's standard_normal to first order; not
        // bit-identical because numpy uses a ziggurat on PCG64).
        double u1, u2;
        do { u1 = u01(); } while (u1 <= 0.0);
        u2 = u01();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.141592653589793 * u2);
    }
};

// C(m x n) = A(m x k) @ B(k x n)
void ts_matmul(const float * A, const float * B, float * C,
               int64_t m, int64_t k, int64_t n) {
    if (m <= 0 || n <= 0) return;
#if defined(TS_HAS_CBLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                1.0f, A, (int)k, B, (int)n, 0.0f, C, (int)n);
#else
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float s = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                s += A[i*k + p] * B[p*n + j];
            }
            C[i*n + j] = s;
        }
    }
#endif
}

// C(n x k) = A^T @ B, where A is (m x n) and B is (m x k)
void ts_matmul_atb(const float * A, const float * B, float * C,
                   int64_t m, int64_t n, int64_t k) {
    if (n <= 0 || k <= 0) return;
#if defined(TS_HAS_CBLAS)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)n, (int)k, (int)m,
                1.0f, A, (int)n, B, (int)k, 0.0f, C, (int)k);
#else
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < k; j++) {
            float s = 0.0f;
            for (int64_t r = 0; r < m; r++) {
                s += A[r*n + i] * B[r*k + j];
            }
            C[i*k + j] = s;
        }
    }
#endif
}

// Round-half-to-even (banker's rounding), matching numpy.round on the
// residual quantiser. Promotion to double matches Python's float32 scalar
// promotion, which is what np.round sees after the .astype(np.int8) cast.
float ts_round_half_even(float x) {
    // numpy rounds to the nearest even integer on ties; uses double internally.
    double v = (double)x;
    double r = std::nearbyint(v);
    return (float)r;
}

}  // namespace

// ---------------------------------------------------------------------------
// R1-Sketch: Y = W @ Omega (Omega Gaussian), then U_basis = top-r left
// singular vectors of Y. The left singular vectors of Y are the eigenvectors
// of M = Y Y^T (K x K), so we use ts_linalg_sym_eig for an exact (LAPACK-grade)
// decomposition -- this is what makes the basis usable for parity.
// ---------------------------------------------------------------------------
int ts_flrq_sketch_run(const float * weight, int64_t K, int64_t N,
                       int64_t n_projections, uint32_t seed,
                       int64_t target_rank, ts_flrq_sketch * out) {
    if (n_projections < 1) {
        n_projections = TS_FLRQ_DEFAULT_PROJECTIONS;
    }
    int64_t r = target_rank > 0 ? target_rank : std::min({K, N, (int64_t)16});
    r = std::max((int64_t)1, std::min({r, K, N}));
    int64_t total_width = std::min(N, n_projections * r);

    ts_rng rng(seed ? seed : 1u);
    std::vector<float> omega((size_t)N * total_width);
    for (size_t i = 0; i < omega.size(); i++) {
        omega[i] = (float)rng.gaussian();
    }

    std::vector<float> Y((size_t)K * total_width);
    ts_matmul(weight, omega.data(), Y.data(), K, N, total_width);

    // M = Y @ Y^T  (K x K). Left singular vectors of Y = eigenvectors of M.
    // Build Y^T explicitly (total_width x K) so ts_matmul gives the K x K gram.
    std::vector<float> Yt((size_t)total_width * K);
    for (int64_t a = 0; a < K; a++) {
        for (int64_t b = 0; b < total_width; b++) {
            Yt[b*K + a] = Y[a*total_width + b];
        }
    }
    std::vector<float> M((size_t)K * K);
    ts_matmul(Y.data(), Yt.data(), M.data(), K, total_width, K);

    std::vector<float> eigvals(K), eigvecs(K * K);
    ts_linalg_sym_eig(M.data(), eigvals.data(), eigvecs.data(), K);

    // U_basis = first r eigenvectors (eigvecs column j); sigma = sqrt(max(eigval,0)).
    out->U_basis.assign((size_t)K * r, 0.0f);
    out->sigma.assign(r, 0.0f);
    for (int64_t j = 0; j < r; j++) {
        float lam = eigvals[j] > 0.0f ? eigvals[j] : 0.0f;
        out->sigma[j] = std::sqrt(lam);
        for (int64_t i = 0; i < K; i++) {
            out->U_basis[i*r + j] = eigvecs[i*K + j];
        }
    }
    out->Y = std::move(Y);
    out->total_width = total_width;
    out->r = r;
    return 0;
}

// ---------------------------------------------------------------------------
// BLC: Best Low-Rank Approximation under Clipping. Iterates
//   1. V = U^T @ W
//   2. per iter: update (scale, clip), quantise residual, V = U^T @ (W - R_q)
// Matches flrq_bcl in per_tensor_calibrate.py exactly.
// ---------------------------------------------------------------------------
int ts_flrq_bcl(const float * weight, int64_t K, int64_t N,
                const float * U_basis, int64_t r,
                int64_t n_iters, int qbits, ts_flrq_bcl_result * out) {
    if (qbits < 1 || qbits > 12) {
        return -1;
    }
    if (n_iters < 1) {
        n_iters = TS_FLRQ_DEFAULT_BLC_ITERS;
    }
    const float qmax = (float)((1 << (qbits - 1)) - 1);  // e.g. 7 for qbits=4

    out->U.assign(U_basis, U_basis + (size_t)K * r);

    // V = U^T @ W   (r x N)
    std::vector<float> V((size_t)r * N);
    ts_matmul_atb(U_basis, weight, V.data(), K, r, N);
    out->V = V;

    // residual = W - U @ V  (K x N)
    std::vector<float> low_rank((size_t)K * N);
    ts_matmul(U_basis, V.data(), low_rank.data(), K, r, N);
    std::vector<float> residual((size_t)K * N);
    for (size_t i = 0; i < residual.size(); i++) {
        residual[i] = weight[i] - low_rank[i];
    }
    std::vector<float> residual_q = residual;

    float scale = 1.0f, clip = 1.0f;
    for (int64_t it = 0; it < n_iters; it++) {
        // 2a. Update scale/clip. clip = 1.0 for qbits >= 3, else 0.95 slack.
        float max_abs = 0.0f;
        for (size_t i = 0; i < residual.size(); i++) {
            float a = std::fabs(residual[i]);
            if (a > max_abs) {
                max_abs = a;
            }
        }
        if (max_abs <= 0.0f) {
            scale = 1.0f;
            clip = 1.0f;
        } else {
            clip = (qbits >= 3) ? 1.0f : 0.95f;
            scale = (qmax * clip) / max_abs;
        }
        // 2b. Quantise residual under (scale, clip) and dequantise.
        // np.round is round-half-to-even; ts_round_half_even matches it.
        float inv_scale = scale > 1e-12f ? 1.0f / scale : 0.0f;
        for (size_t i = 0; i < residual.size(); i++) {
            float rs = residual[i] * scale;
            if (rs > qmax)  rs = qmax;
            if (rs < -qmax) rs = -qmax;
            float qi = ts_round_half_even(rs);
            residual_q[i] = qi * inv_scale;
        }
        // 2c. V = U^T @ (W - residual_q), then recompute residual.
        std::vector<float> target((size_t)K * N);
        for (size_t i = 0; i < target.size(); i++) {
            target[i] = weight[i] - residual_q[i];
        }
        ts_matmul_atb(U_basis, target.data(), V.data(), K, r, N);
        ts_matmul(U_basis, V.data(), low_rank.data(), K, r, N);
        for (size_t i = 0; i < residual.size(); i++) {
            residual[i] = weight[i] - low_rank[i];
        }
    }

    // Final reconstruction MSE: mean((W - (U@V + residual_q))^2).
    float mse = 0.0f;
    for (int64_t i = 0; i < K * N; i++) {
        float recon = low_rank[i] + residual_q[i];
        float d = weight[i] - recon;
        mse += d * d;
    }
    mse /= (float)(K * N);

    out->V = std::move(V);
    out->residual = std::move(residual);
    out->residual_q = std::move(residual_q);
    out->residual_scale = scale;
    out->residual_clip = clip;
    out->reconstruction_mse = mse;
    return 0;
}

// ---------------------------------------------------------------------------
// End-to-end driver: rank sweep + final decomposition at the chosen rank.
// ---------------------------------------------------------------------------
int ts_train_flrq(const float * weight, int64_t K, int64_t N,
                  const ts_flrq_params * params,
                  ts_flrq_bcl_result * out, int64_t * result_rank) {
    int64_t n_proj  = params->n_projections > 0 ? params->n_projections : TS_FLRQ_DEFAULT_PROJECTIONS;
    uint32_t seed   = params->seed ? params->seed : 1u;
    int64_t iters   = params->blc_iters > 0 ? params->blc_iters : TS_FLRQ_DEFAULT_BLC_ITERS;
    int64_t qbits   = params->qbits > 0 ? params->qbits : TS_FLRQ_DEFAULT_QBITS;
    float thresh    = params->mse_threshold > 0.0f ? params->mse_threshold : TS_FLRQ_DEFAULT_MSE_THRESHOLD;

    std::vector<int64_t> candidates;
    for (int i = 0; i < 8; i++) {
        if (params->rank_candidates[i] <= 0) {
            break;
        }
        candidates.push_back(params->rank_candidates[i]);
    }
    if (candidates.empty()) {
        const int64_t def[] = {4, 8, 16, 32, 64};
        for (int64_t v : def) {
            candidates.push_back(v);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    // Clamp candidates to the feasible rank range.
    std::vector<int64_t> feasible;
    for (int64_t r : candidates) {
        if (r >= 1 && r <= std::min(K, N)) {
            feasible.push_back(r);
        }
    }
    if (feasible.empty()) {
        feasible.push_back(std::max((int64_t)1, std::min(K, N)));
    }

    // ||W||_F^2  (with 1e-12 floor, matching Python).
    double w_fro2 = 0.0;
    for (int64_t i = 0; i < K * N; i++) {
        w_fro2 += (double)weight[i] * weight[i];
    }
    w_fro2 += 1e-12;

    int64_t chosen = feasible.back();
    ts_flrq_bcl_result best;
    bool have_best = false;
    for (int64_t r : feasible) {
        ts_flrq_sketch sk;
        if (ts_flrq_sketch_run(weight, K, N, n_proj, seed + (uint32_t)r, r, &sk) != 0) {
            continue;
        }
        ts_flrq_bcl_result res;
        if (ts_flrq_bcl(weight, K, N, sk.U_basis.data(), r, iters, qbits, &res) != 0) {
            continue;
        }
        // Relative reconstruction MSE: ||W - (U@V + R_q)||_F^2 / ||W||_F^2.
        double rel = (double)res.reconstruction_mse * (double)(K * N) / w_fro2;
        bool clears = rel <= (double)thresh;
        if (clears && (!have_best || r < chosen)) {
            chosen = r;
            best = std::move(res);
            have_best = true;
            break;  // smallest clearing rank wins; feasible is sorted ascending
        }
        if (!have_best) {
            best = std::move(res);
            have_best = true;
        }
    }

    // Recompute at chosen rank with the canonical seed so the returned factors
    // are exactly the ones that produced the recorded MSE (matches train_flrq).
    {
        ts_flrq_sketch sk;
        if (ts_flrq_sketch_run(weight, K, N, n_proj, seed + (uint32_t)chosen, chosen, &sk) != 0) {
            *result_rank = -1;
            return -1;
        }
        ts_flrq_bcl_result res;
        if (ts_flrq_bcl(weight, K, N, sk.U_basis.data(), chosen, iters, qbits, &res) != 0) {
            *result_rank = -1;
            return -1;
        }
        *out = std::move(res);
    }
    *result_rank = chosen;
    return 0;
}
