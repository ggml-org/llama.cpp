#pragma once

//
// tessera-linalg.h
//
// Linalg primitives for FLRQ (sketch + power iteration) and
// DartQuant (Householder QR + Stiefel manifold optimization).
// All matrices are row-major float, dimensions passed explicitly.
//

#include <cstdint>
#include <cstddef>

// QR decomposition via Householder reflections.
// A is (m x n), m >= n. Q is (m x n) thin factor, R is (n x n) upper.
void ts_linalg_householder_qr(const float * A, float * Q, float * R,
                              int64_t m, int64_t n);

// QR retract: project arbitrary (m x n) matrix onto St(m,n) via QR.
// M is (m x n), output Q is (m x n) with orthonormal columns.
void ts_linalg_qr_retract(const float * M, float * Q, int64_t m, int64_t n);

// Random orthogonal matrix (n x n) via QR of Gaussian. Deterministic given seed.
void ts_linalg_random_orthogonal(float * R, int64_t n, uint32_t seed);

// Stiefel projection: project gradient G onto tangent space at R.
// G, R are (m x n) on St(m,n). Output P is (m x n).
void ts_linalg_stiefel_project(const float * G, const float * R,
                               float * P, int64_t m, int64_t n);

// QR-Orth step: R_new = qr_retract(R + lr * G_projected).
// In-place on R.
void ts_linalg_qr_orth_step(float * R, const float * G, float lr,
                            int64_t m, int64_t n);

// Power iteration for top-k singular vectors.
// A is (m x n). U is (m x k), S is (k,), V is (n x k).
// n_iters: number of power steps. seed for random init.
void ts_linalg_svd_topk(const float * A, float * U, float * S, float * V,
                        int64_t m, int64_t n, int64_t k,
                        int64_t n_iters, uint32_t seed);

// Randomized sketch: S = A @ Omega where Omega is (n x k) Gaussian.
// Output sketch is (m x k).
void ts_linalg_sketch(const float * A, float * sketch,
                      int64_t m, int64_t n, int64_t k, uint32_t seed);

// Gram-Schmidt orthonormalization in-place. V is (k x n) row vectors.
void ts_linalg_gram_schmidt(float * V, int64_t k, int64_t n);

// --- tessera-flrq helper (B3 FLRQ port) ---
// Symmetric eigendecomposition via cyclic Jacobi rotations. A is a symmetric
// (n x n) row-major matrix (only the upper triangle is read). On output
// `eigvals` holds the eigenvalues sorted DESCENDING and `eigvecs` is (n x n)
// with column j the eigenvector for eigvals[j]. Matches LAPACK dsyev to
// float precision (signs aside), which is what the FLRQ sketch-basis parity
// test requires. Used only on the small K x K sketch gram matrix.
void ts_linalg_sym_eig(const float * A, float * eigvals, float * eigvecs, int64_t n);
