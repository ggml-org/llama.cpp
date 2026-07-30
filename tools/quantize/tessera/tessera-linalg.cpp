//
// tessera-linalg.cpp
//
// Linalg primitives for FLRQ (sketch + power iteration) and
// DartQuant (Householder QR + Stiefel manifold optimization).
// Row-major float, caller-provided output buffers.
//

#include "tessera-linalg.h"

#include <cmath>
#include <vector>

static const float TS_PI = 3.141592653589793f;

// ---------------------------------------------------------------------------
// PRNG + small helpers
// ---------------------------------------------------------------------------

static uint32_t ts_xorshift32(uint32_t * state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float ts_rand_gaussian(uint32_t * state) {
    float u1;
    do {
        u1 = (ts_xorshift32(state) + 1u) * (1.0f / 4294967296.0f);
    } while (u1 <= 0.0f);
    float u2 = ts_xorshift32(state) * (1.0f / 4294967296.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * TS_PI * u2);
}

static void ts_fill_gaussian(float * buf, int64_t count, uint32_t * state) {
    for (int64_t i = 0; i < count; i++) {
        buf[i] = ts_rand_gaussian(state);
    }
}

// C(m x n) = A(m x k) @ B(k x n)
static void ts_matmul(const float * A, const float * B, float * C,
                      int64_t m, int64_t k, int64_t n) {
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float s = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                s += A[i*k + p] * B[p*n + j];
            }
            C[i*n + j] = s;
        }
    }
}

// C(n x k) = A^T @ B, where A is (m x n) and B is (m x k)
static void ts_matmul_atb(const float * A, const float * B, float * C,
                          int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < k; j++) {
            float s = 0.0f;
            for (int64_t r = 0; r < m; r++) {
                s += A[r*n + i] * B[r*k + j];
            }
            C[i*k + j] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Householder QR
// ---------------------------------------------------------------------------

void ts_linalg_householder_qr(const float * A, float * Q, float * R,
                              int64_t m, int64_t n) {
    std::vector<float> Rw(A, A + m*n);        // working copy of A
    std::vector<float> V(m*n, 0.0f);          // householder vectors, col j in V[:, j]
    std::vector<float> beta(n, 0.0f);

    for (int64_t j = 0; j < n; j++) {
        float norm_x = 0.0f;
        for (int64_t i = j; i < m; i++) {
            float t = Rw[i*n + j];
            norm_x += t*t;
        }
        norm_x = sqrtf(norm_x);
        if (norm_x < 1e-12f) {
            continue;
        }
        float x0 = Rw[j*n + j];
        float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
        float u1 = x0 + sign * norm_x;
        if (fabsf(u1) < 1e-12f) {
            continue;
        }
        V[j*n + j] = 1.0f;
        for (int64_t i = j + 1; i < m; i++) {
            V[i*n + j] = Rw[i*n + j] / u1;
        }
        beta[j] = sign * u1 / norm_x;

        // R[j:, p] -= beta * v * (v^T R[j:, p])
        for (int64_t p = 0; p < n; p++) {
            float s = 0.0f;
            for (int64_t i = j; i < m; i++) {
                s += V[i*n + j] * Rw[i*n + p];
            }
            s *= beta[j];
            for (int64_t i = j; i < m; i++) {
                Rw[i*n + p] -= s * V[i*n + j];
            }
        }
    }

    // R is the upper-triangular top n x n block
    for (int64_t i = 0; i < n; i++) {
        for (int64_t p = 0; p < n; p++) {
            R[i*n + p] = (p >= i) ? Rw[i*n + p] : 0.0f;
        }
    }

    // thin Q = H_0 H_1 ... H_{n-1} E; apply reflections in reverse to E
    for (int64_t i = 0; i < m; i++) {
        for (int64_t c = 0; c < n; c++) {
            Q[i*n + c] = (i == c) ? 1.0f : 0.0f;
        }
    }
    std::vector<float> vtX(n);
    for (int64_t j = n - 1; j >= 0; j--) {
        if (beta[j] == 0.0f) {
            continue;
        }
        for (int64_t c = 0; c < n; c++) {
            float s = 0.0f;
            for (int64_t i = j; i < m; i++) {
                s += V[i*n + j] * Q[i*n + c];
            }
            vtX[c] = s * beta[j];
        }
        for (int64_t c = 0; c < n; c++) {
            for (int64_t i = j; i < m; i++) {
                Q[i*n + c] -= vtX[c] * V[i*n + j];
            }
        }
    }
}

void ts_linalg_qr_retract(const float * M, float * Q, int64_t m, int64_t n) {
    std::vector<float> R(n*n);
    ts_linalg_householder_qr(M, Q, R.data(), m, n);
    // sign-canonicalize so the R diagonal is non-negative
    for (int64_t j = 0; j < n; j++) {
        if (R[j*n + j] < 0.0f) {
            for (int64_t i = 0; i < m; i++) {
                Q[i*n + j] = -Q[i*n + j];
            }
        }
    }
}

void ts_linalg_random_orthogonal(float * R, int64_t n, uint32_t seed) {
    uint32_t st = seed ? seed : 1u;
    std::vector<float> G(n*n);
    ts_fill_gaussian(G.data(), n*n, &st);
    ts_linalg_qr_retract(G.data(), R, n, n);
}

// ---------------------------------------------------------------------------
// Stiefel manifold primitives
// ---------------------------------------------------------------------------

void ts_linalg_stiefel_project(const float * G, const float * R,
                               float * P, int64_t m, int64_t n) {
    std::vector<float> M(n*n);   // M = R^T @ G
    ts_matmul_atb(R, G, M.data(), m, n, n);
    // sym(M) = (M + M^T) / 2
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            float s = 0.5f * (M[i*n + j] + M[j*n + i]);
            M[i*n + j] = s;
            M[j*n + i] = s;
        }
    }
    // P = G - R @ sym(M)
    for (int64_t r = 0; r < m; r++) {
        for (int64_t j = 0; j < n; j++) {
            float s = 0.0f;
            for (int64_t i = 0; i < n; i++) {
                s += R[r*n + i] * M[i*n + j];
            }
            P[r*n + j] = G[r*n + j] - s;
        }
    }
}

void ts_linalg_qr_orth_step(float * R, const float * G, float lr,
                            int64_t m, int64_t n) {
    std::vector<float> P(m*n);
    ts_linalg_stiefel_project(G, R, P.data(), m, n);
    std::vector<float> M(m*n);
    for (int64_t i = 0; i < m*n; i++) {
        M[i] = R[i] + lr * P[i];
    }
    ts_linalg_qr_retract(M.data(), R, m, n);
}

// ---------------------------------------------------------------------------
// Randomized SVD + sketch
// ---------------------------------------------------------------------------

void ts_linalg_svd_topk(const float * A, float * U, float * S, float * V,
                        int64_t m, int64_t n, int64_t k,
                        int64_t n_iters, uint32_t seed) {
    if (k < 1) k = 1;
    if (k > n) k = n;
    if (k > m) k = m;

    uint32_t st = seed ? seed : 1u;
    std::vector<float> Omega(n*k);
    ts_fill_gaussian(Omega.data(), n*k, &st);

    std::vector<float> Y(m*k);                 // Y = A @ Omega
    ts_matmul(A, Omega.data(), Y.data(), m, n, k);

    std::vector<float> Q(m*k), Rt(k*k);
    ts_linalg_householder_qr(Y.data(), Q.data(), Rt.data(), m, k);

    std::vector<float> Z(n*k);                 // Z = A^T @ Q
    ts_matmul_atb(A, Q.data(), Z.data(), m, n, k);

    std::vector<float> Qrow(n*k), Rt2(k*k);    // Qrow = orth(Z), n x k
    ts_linalg_householder_qr(Z.data(), Qrow.data(), Rt2.data(), n, k);

    std::vector<float> AtA(n*n);               // A^T A
    ts_matmul_atb(A, A, AtA.data(), m, n, n);

    std::vector<float> Z2(n*k);
    int64_t iters = n_iters > 1 ? n_iters : 1;
    for (int64_t it = 0; it < iters; it++) {
        ts_matmul(AtA.data(), Qrow.data(), Z2.data(), n, n, k);
        ts_linalg_householder_qr(Z2.data(), Qrow.data(), Rt2.data(), n, k);
    }

    std::vector<float> B(m*k);                 // B = A @ Qrow
    ts_matmul(A, Qrow.data(), B.data(), m, n, k);

    std::vector<float> BtB(k*k);               // B^T B
    ts_matmul_atb(B.data(), B.data(), BtB.data(), m, k, k);

    // power iteration on B^T B with Gram-Schmidt deflation
    std::vector<float> Vcols(k*k);             // row j = j-th right vector (len k)
    std::vector<float> v(k), nv(k), Bv(m);
    uint32_t st2 = (seed + 1u) ? (seed + 1u) : 1u;
    for (int64_t j = 0; j < k; j++) {
        ts_fill_gaussian(v.data(), k, &st2);
        float nrm = 0.0f;
        for (int64_t i = 0; i < k; i++) nrm += v[i]*v[i];
        nrm = sqrtf(nrm);
        if (nrm < 1e-12f) {
            v[0] = 1.0f;
        } else {
            float inv = 1.0f / nrm;
            for (int64_t i = 0; i < k; i++) v[i] *= inv;
        }
        float sigma = 0.0f;
        int64_t piters = n_iters > 10 ? n_iters : 10;
        for (int64_t it = 0; it < piters; it++) {
            for (int64_t r = 0; r < k; r++) {
                float s = 0.0f;
                for (int64_t c = 0; c < k; c++) s += BtB[r*k + c] * v[c];
                nv[r] = s;
            }
            nrm = 0.0f;
            for (int64_t i = 0; i < k; i++) nrm += nv[i]*nv[i];
            nrm = sqrtf(nrm);
            if (nrm < 1e-12f) {
                break;
            }
            float inv = 1.0f / nrm;
            for (int64_t i = 0; i < k; i++) v[i] = nv[i] * inv;
            sigma = sqrtf(nrm);
        }
        // orthogonalize against previously found vectors
        for (int64_t p = 0; p < j; p++) {
            float proj = 0.0f;
            for (int64_t i = 0; i < k; i++) proj += v[i] * Vcols[p*k + i];
            for (int64_t i = 0; i < k; i++) v[i] -= proj * Vcols[p*k + i];
        }
        nrm = 0.0f;
        for (int64_t i = 0; i < k; i++) nrm += v[i]*v[i];
        nrm = sqrtf(nrm);
        if (nrm < 1e-12f) {
            // degenerate direction: synthesise e_j and re-orthogonalize
            for (int64_t i = 0; i < k; i++) v[i] = 0.0f;
            v[j] = 1.0f;
            for (int64_t p = 0; p < j; p++) {
                float proj = 0.0f;
                for (int64_t i = 0; i < k; i++) proj += v[i] * Vcols[p*k + i];
                for (int64_t i = 0; i < k; i++) v[i] -= proj * Vcols[p*k + i];
            }
            nrm = 0.0f;
            for (int64_t i = 0; i < k; i++) nrm += v[i]*v[i];
            nrm = sqrtf(nrm);
            if (nrm < 1e-12f) {
                for (int64_t i = 0; i < k; i++) v[i] = 0.0f;
                v[j] = 1.0f;
                nrm = 1.0f;
            }
            float inv = 1.0f / nrm;
            for (int64_t i = 0; i < k; i++) v[i] *= inv;
            sigma = 0.0f;
        } else {
            float inv = 1.0f / nrm;
            for (int64_t i = 0; i < k; i++) v[i] *= inv;
            // sigma = ||B v||
            for (int64_t r = 0; r < m; r++) {
                float s = 0.0f;
                for (int64_t c = 0; c < k; c++) s += B[r*k + c] * v[c];
                Bv[r] = s;
            }
            sigma = 0.0f;
            for (int64_t i = 0; i < m; i++) sigma += Bv[i]*Bv[i];
            sigma = sqrtf(sigma);
        }
        for (int64_t i = 0; i < k; i++) Vcols[j*k + i] = v[i];
        S[j] = sigma;
    }

    // V = Qrow @ Vcols^T  ->  V[i,j] = sum_a Qrow[i,a] * Vcols[j,a]
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < k; j++) {
            float s = 0.0f;
            for (int64_t a = 0; a < k; a++) s += Qrow[i*k + a] * Vcols[j*k + a];
            V[i*k + j] = s;
        }
    }
    // U[:, j] = B @ v_j / sigma_j
    for (int64_t j = 0; j < k; j++) {
        for (int64_t r = 0; r < m; r++) {
            float s = 0.0f;
            for (int64_t c = 0; c < k; c++) s += B[r*k + c] * Vcols[j*k + c];
            Bv[r] = s;
        }
        float inv = (S[j] > 1e-12f) ? 1.0f / S[j] : 0.0f;
        for (int64_t r = 0; r < m; r++) U[r*k + j] = Bv[r] * inv;
    }
}

void ts_linalg_sketch(const float * A, float * sketch,
                      int64_t m, int64_t n, int64_t k, uint32_t seed) {
    uint32_t st = seed ? seed : 1u;
    std::vector<float> Omega(n*k);
    ts_fill_gaussian(Omega.data(), n*k, &st);
    ts_matmul(A, Omega.data(), sketch, m, n, k);
}

// ---------------------------------------------------------------------------
// Gram-Schmidt
// ---------------------------------------------------------------------------

void ts_linalg_gram_schmidt(float * V, int64_t k, int64_t n) {
    for (int64_t j = 0; j < k; j++) {
        float * vj = V + j*n;
        for (int64_t p = 0; p < j; p++) {
            const float * vp = V + p*n;
            float proj = 0.0f;
            for (int64_t i = 0; i < n; i++) proj += vj[i] * vp[i];
            for (int64_t i = 0; i < n; i++) vj[i] -= proj * vp[i];
        }
        float nrm = 0.0f;
        for (int64_t i = 0; i < n; i++) nrm += vj[i]*vj[i];
        nrm = sqrtf(nrm);
        if (nrm > 1e-12f) {
            float inv = 1.0f / nrm;
            for (int64_t i = 0; i < n; i++) vj[i] *= inv;
        }
    }
}
