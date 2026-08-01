//
// tessera-linalg.cpp
//
// Linalg primitives for FLRQ (sketch + power iteration) and
// DartQuant (Householder QR + Stiefel manifold optimization).
// Row-major float, caller-provided output buffers.
//

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
static void ts_matmul_atb(const float * A, const float * B, float * C,
                          int64_t m, int64_t n, int64_t k) {
    if (n <= 0 || k <= 0) return;
#if defined(TS_HAS_CBLAS)
    // A is (m x n) row-major; A^T is (n x m). B is (m x k) row-major.
    // C = A^T @ B, (n x k). op(A)=Trans, op(B)=NoTrans.
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
#if defined(TS_HAS_CBLAS)
    // ssyrk: C = alpha * A^T A + beta * C. Row-major lower triangle first.
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans,
                (int)n, (int)m, 1.0f, A, (int)n, 0.0f, AtA.data(), (int)n);
    // copy upper to lower for full matrix (power iteration needs full AtA)
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = i + 1; j < n; j++)
            AtA[j*n + i] = AtA[i*n + j];
#else
    ts_matmul_atb(A, A, AtA.data(), m, n, n);
#endif

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

// ---------------------------------------------------------------------------
// tessera-flrq helper: symmetric eigendecomposition via cyclic Jacobi.
// Self-contained; only touched by the FLRQ sketch-basis path. Computes the
// eigendecomposition of the small (K x K) sketch gram matrix Y Y^T so the
// FLRQ basis matches numpy.linalg.svd(Y) (whose left singular vectors are the
// eigenvectors of Y Y^T). Sorts eigenpairs descending by eigenvalue.
// ---------------------------------------------------------------------------

void ts_linalg_sym_eig(const float * A, float * eigvals, float * eigvecs, int64_t n) {
    // Work on a symmetric copy (force exact symmetry to absorb input noise).
    std::vector<float> M(n * n);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            float v = 0.5f * (A[i*n + j] + A[j*n + i]);
            M[i*n + j] = v;
            M[j*n + i] = v;
        }
    }
    // V starts as identity.
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            eigvecs[i*n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    const int64_t max_sweeps = 80;
    for (int64_t sweep = 0; sweep < max_sweeps; sweep++) {
        // off-diagonal Frobenius norm.
        float off = 0.0f;
        for (int64_t p = 0; p < n; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                off += M[p*n + q] * M[p*n + q];
            }
        }
        if (off <= 1e-30f) {
            break;
        }
        // Cyclic sweep over all (p, q) pairs above the diagonal.
        for (int64_t p = 0; p < n - 1; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                float apq = M[p*n + q];
                if (fabsf(apq) < 1e-30f) {
                    continue;
                }
                float app = M[p*n + p];
                float aqq = M[q*n + q];
                // Rotation angle: tau = (aqq - app) / (2 apq); t = sign/(|tau|+sqrt(1+tau^2)).
                float tau = (aqq - app) / (2.0f * apq);
                float t;
                if (tau >= 0.0f) {
                    t = 1.0f / (tau + sqrtf(1.0f + tau*tau));
                } else {
                    t = -1.0f / (-tau + sqrtf(1.0f + tau*tau));
                }
                float c = 1.0f / sqrtf(1.0f + t*t);
                float s = t * c;
                // Apply rotation to columns/rows p, q of M.
                for (int64_t i = 0; i < n; i++) {
                    float mip = M[i*n + p];
                    float miq = M[i*n + q];
                    M[i*n + p] = c*mip - s*miq;
                    M[i*n + q] = s*mip + c*miq;
                }
                for (int64_t j = 0; j < n; j++) {
                    float mpj = M[p*n + j];
                    float mqj = M[q*n + j];
                    M[p*n + j] = c*mpj - s*mqj;
                    M[q*n + j] = s*mpj + c*mqj;
                }
                // Accumulate eigenvectors: V = V @ R.
                for (int64_t i = 0; i < n; i++) {
                    float vip = eigvecs[i*n + p];
                    float viq = eigvecs[i*n + q];
                    eigvecs[i*n + p] = c*vip - s*viq;
                    eigvecs[i*n + q] = s*vip + c*viq;
                }
            }
        }
    }

    // Collect eigenvalues (diagonal) and sort descending (selection sort; n is small).
    for (int64_t i = 0; i < n; i++) {
        eigvals[i] = M[i*n + i];
    }
    std::vector<int64_t> order(n);
    for (int64_t i = 0; i < n; i++) {
        order[i] = i;
    }
    for (int64_t i = 0; i < n; i++) {
        int64_t best = i;
        for (int64_t j = i + 1; j < n; j++) {
            if (eigvals[order[j]] > eigvals[order[best]]) {
                best = j;
            }
        }
        int64_t tmp = order[i];
        order[i] = order[best];
        order[best] = tmp;
    }
    std::vector<float> ev_sorted(n * n);
    std::vector<float> lam_sorted(n);
    for (int64_t j = 0; j < n; j++) {
        lam_sorted[j] = eigvals[order[j]];
        for (int64_t i = 0; i < n; i++) {
            ev_sorted[i*n + j] = eigvecs[i*n + order[j]];
        }
    }
    for (int64_t i = 0; i < n; i++) {
        eigvals[i] = lam_sorted[i];
        for (int64_t j = 0; j < n; j++) {
            eigvecs[i*n + j] = ev_sorted[i*n + j];
        }
    }
}
