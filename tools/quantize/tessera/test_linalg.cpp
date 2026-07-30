//
// test_linalg.cpp
//
// Smoke tests for tessera-linalg.cpp. Prints PASS/FAIL per test,
// returns 0 only if all pass.
//

#include "tessera-linalg.h"

#include <cmath>
#include <cstdio>
#include <vector>

static bool test_qr() {
    const int64_t m = 4, n = 3;
    const float A[m*n] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 10.0f,
        2.0f, 0.0f, 1.0f,
    };
    std::vector<float> Q(m*n), R(n*n);
    ts_linalg_householder_qr(A, Q.data(), R.data(), m, n);

    // Q^T Q ~= I_n
    float err_orth = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            float s = 0.0f;
            for (int64_t r = 0; r < m; r++) s += Q[r*n + i] * Q[r*n + j];
            float target = (i == j) ? 1.0f : 0.0f;
            err_orth = fmaxf(err_orth, fabsf(s - target));
        }
    }

    // Q R ~= A
    float err_recon = 0.0f;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float s = 0.0f;
            for (int64_t p = 0; p < n; p++) s += Q[i*n + p] * R[p*n + j];
            err_recon = fmaxf(err_recon, fabsf(s - A[i*n + j]));
        }
    }

    printf("  qr: orth=%.2e recon=%.2e\n", err_orth, err_recon);
    return err_orth < 1e-4f && err_recon < 1e-4f;
}

static bool test_stiefel_project() {
    const int64_t n = 4;
    std::vector<float> R(n*n), G(n*n), P(n*n);
    ts_linalg_random_orthogonal(R.data(), n, 1234);
    // arbitrary ambient gradient
    for (int64_t i = 0; i < n*n; i++) {
        G[i] = 0.3f * (float)(i % 5) - 0.1f * (float)(i % 3);
    }
    ts_linalg_stiefel_project(G.data(), R.data(), P.data(), n, n);

    // R^T P should be skew-symmetric: M + M^T ~= 0
    float err = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            float mij = 0.0f, mji = 0.0f;
            for (int64_t r = 0; r < n; r++) {
                mij += R[r*n + i] * P[r*n + j];
                mji += R[r*n + j] * P[r*n + i];
            }
            err = fmaxf(err, fabsf(mij + mji));
        }
    }

    printf("  stiefel_project: skew_err=%.2e\n", err);
    return err < 1e-4f;
}

static bool test_svd_topk() {
    const int64_t m = 3, n = 3, k = 3;
    const float A[m*n] = {
        3.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    std::vector<float> U(m*k), S(k), V(n*k);
    ts_linalg_svd_topk(A, U.data(), S.data(), V.data(), m, n, k, 30, 7);

    printf("  svd_topk: S=[%.4f, %.4f, %.4f]\n", S[0], S[1], S[2]);
    return fabsf(S[0] - 3.0f) < 1e-2f;
}

static bool test_gram_schmidt() {
    const int64_t k = 3, n = 4;
    float V[k*n] = {
        1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 1.0f,
    };
    ts_linalg_gram_schmidt(V, k, n);

    // V V^T ~= I_k
    float err = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = 0; j < k; j++) {
            float s = 0.0f;
            for (int64_t c = 0; c < n; c++) s += V[i*n + c] * V[j*n + c];
            float target = (i == j) ? 1.0f : 0.0f;
            err = fmaxf(err, fabsf(s - target));
        }
    }

    printf("  gram_schmidt: orth_err=%.2e\n", err);
    return err < 1e-4f;
}

int main() {
    struct { const char * name; bool (*fn)(); } tests[] = {
        { "qr",             test_qr },
        { "stiefel_project", test_stiefel_project },
        { "svd_topk",        test_svd_topk },
        { "gram_schmidt",    test_gram_schmidt },
    };

    bool all = true;
    for (auto & t : tests) {
        bool ok = t.fn();
        printf("[%s] %s\n", ok ? "PASS" : "FAIL", t.name);
        all = all && ok;
    }
    return all ? 0 : 1;
}
