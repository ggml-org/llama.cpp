#!/usr/bin/env python3
"""DartQuant QR / Stiefel-manifold helpers.

Pure-Python implementations of the small dense linear-algebra kernels that
DartQuant (NeurIPS 2025) needs. The training loop in
``per_tensor_calibrate.py`` stays free of any third-party linalg dependency
beyond numpy; this module factors out the linear-algebra primitives so the
training loop is easy to read.

What lives here:

* ``householder_qr``        : Householder reflection-based QR factorization.
* ``qr_retract``            : Q of ``np.linalg.qr``-equivalent, used to pull a
                              perturbed matrix back to the orthogonal group.
* ``random_orthogonal``     : R initialization (Haar-uniform-ish).
* ``stiefel_project``       : projection of an ambient gradient onto the
                              tangent space of the Stiefel manifold at ``R``.
* ``qr_orth_step``          : one step of the QR-Orth optimizer (Absil et al.
                              2008 retraction); replaces Cayley SGD.

Notation: ``R`` is square ``(K, K)`` and lives on ``O(K) = {R : R^T R = I}``.
A point in the tangent space at ``R`` is any matrix ``G`` that satisfies
``R^T G + G^T R = 0`` (skew-symmetric in the rotated frame).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Householder QR
# ---------------------------------------------------------------------------


def householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR factorization via Householder reflections.

    Pure-Python implementation operating on numpy float32 / float64 arrays.
    Returns ``(Q, R)`` such that ``A = Q @ R``, ``Q`` is square ``(m, m)``
    and ``R`` is upper-triangular ``(m, n)``. The convention is
    ``R[j, j] >= 0`` (the sign is folded into ``Q`` so ``R`` is canonical).

    Cost: ``O(m * n * min(m, n))``. For ``(K, K)`` square matrices the cost
    is ``K^3 / 3`` flops plus a final ``O(m^2 * n)`` for the ``Q`` assembly.
    The Householder application is vectorised over the trailing dimension
    so even at ``K = 1024`` the routine completes in a fraction of a second.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"householder_qr expects a 2-D matrix, got {A.shape}")
    m, n = A.shape
    k = min(m, n)
    R = A.astype(np.float64, copy=True)
    Q = np.eye(m, dtype=np.float64)
    for j in range(k):
        # Householder vector for column j of the trailing submatrix.
        x = R[j:, j]
        norm_x = float(np.linalg.norm(x))
        if norm_x < 1.0e-12:
            # Column is already zero below the diagonal; skip.
            continue
        sign = 1.0 if x[0] >= 0.0 else -1.0
        u1 = x[0] + sign * norm_x
        if abs(u1) < 1.0e-12:
            # x is already a unit vector along the first axis; the
            # reflection is the identity and we can skip the apply.
            continue
        v = x / u1
        v[0] = 1.0
        beta = sign * u1 / norm_x  # tau = 2 / (v^T v); folded form
        # Apply the reflection from the left: R[j:, :] -= beta * v * (v^T R[j:, :])
        vT_R = v @ R[j:, :]
        R[j:, :] -= beta * np.outer(v, vT_R)
        # Apply the same reflection to Q from the right: Q[:, j:] -= beta * (Q[:, j:] v) v^T
        Qv = Q[:, j:] @ v
        Q[:, j:] -= beta * np.outer(Qv, v)
    return Q.astype(A.dtype, copy=False), np.triu(R[:k, :k] if m >= n else R[:m, :k])


def qr_retract(M: np.ndarray) -> np.ndarray:
    """Q-factor of ``M`` (i.e. the orthogonal retraction to ``O(K)``).

    Convenience wrapper used by the QR-Orth optimizer. Implemented via
    ``np.linalg.qr`` (LAPACK) for speed; the Householder routine above is
    kept around for cross-validation and for small matrices where the
    python-loop overhead is negligible.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"qr_retract expects a square matrix, got {M.shape}")
    Q, _ = np.linalg.qr(M)
    # Sign canonicalisation: make the diagonal of R positive so Q is
    # unique up to the sign of the columns. The R-diagonal of the QR
    # factorisation of ``M`` is what we want.
    _, R = np.linalg.qr(M)
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0.0] = 1.0
    return (Q * diag_sign[None, :]).astype(M.dtype, copy=False)


# ---------------------------------------------------------------------------
# Stiefel manifold primitives
# ---------------------------------------------------------------------------


def random_orthogonal(K: int, seed: int = 0) -> np.ndarray:
    """Initialise an orthogonal matrix of shape ``(K, K)``.

    Uses QR of a Gaussian draw; sign-canonicalised so the diagonal of the
    R-factor is non-negative. This is the standard Haar-uniform proxy for
    ``O(K)``.
    """
    rng = np.random.default_rng(seed)
    G = rng.normal(loc=0.0, scale=1.0, size=(K, K)).astype(np.float64)
    Q, R = np.linalg.qr(G)
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0.0] = 1.0
    return (Q * diag_sign[None, :]).astype(np.float32)


def stiefel_project(G: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Project ambient gradient ``G`` onto the tangent space of ``O(K)`` at ``R``.

    For ``R`` in the Stiefel manifold (the orthogonal group), the tangent
    space at ``R`` consists of matrices ``Z`` satisfying ``R^T Z`` being
    skew-symmetric. The projection is::

        proj_t(G, R) = G - R @ sym(R^T @ G)

    where ``sym(M) = (M + M^T) / 2``. The result is in the tangent space
    of ``O(K)`` at ``R`` and is the canonical Riemannian gradient used by
    the QR-Orth optimiser.
    """
    G = np.asarray(G, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if G.shape != R.shape:
        raise ValueError(f"shapes differ: G={G.shape} R={R.shape}")
    M = R.T @ G
    sym = 0.5 * (M + M.T)
    return (G - R @ sym).astype(G.dtype, copy=False)


def qr_orth_step(R: np.ndarray, G: np.ndarray, lr: float) -> np.ndarray:
    """One QR-Orth optimisation step on the Stiefel manifold.

    Combines tangent-space projection with a QR retraction::

        M       = R - lr * proj_tangent(G, R)
        R_next  = qf(M)              # Q-factor of M

    ``lr`` is the step size; the manifold constraint is preserved exactly
    by the retraction. The gradient ``G`` is the ambient gradient
    (i.e. ``d loss / d R`` ignoring the manifold constraint). The step is
    first-order in ``lr``; convergence typically requires a small step
    (e.g. ``lr <= 1e-2`` for FP32 weights around unit scale).
    """
    R = np.asarray(R, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    tangent = stiefel_project(G, R)
    M = R - lr * tangent
    return qr_retract(M).astype(R.dtype, copy=False)


# ---------------------------------------------------------------------------
# Sanity tests (run as ``python3 -m tools.tessera._dartquant_linalg``)
# ---------------------------------------------------------------------------


def _selftest() -> None:
    K = 32
    rng = np.random.default_rng(0)
    R0 = random_orthogonal(K, seed=0)
    # Orthogonality check.
    err_orth = float(np.max(np.abs(R0.T @ R0 - np.eye(K))))
    assert err_orth < 1.0e-5, f"R0 not orthogonal: {err_orth}"
    # Round-trip check on Householder QR.
    A = rng.normal(size=(K, K))
    Q, R = householder_qr(A)
    err_recon = float(np.max(np.abs(Q @ R - A)))
    err_orth_q = float(np.max(np.abs(Q.T @ Q - np.eye(K))))
    assert err_recon < 1.0e-4, f"householder QR reconstruction error: {err_recon}"
    assert err_orth_q < 1.0e-5, f"householder Q not orthogonal: {err_orth_q}"
    # Stiefel projection should land in the tangent space.
    G = rng.normal(size=(K, K))
    Z = stiefel_project(G, R0)
    skew = R0.T @ Z
    skew_err = float(np.max(np.abs(skew + skew.T)))
    assert skew_err < 1.0e-5, f"projected G is not tangent: {skew_err}"
    # QR-Orth step preserves orthogonality.
    R1 = qr_orth_step(R0, G, lr=1.0e-2)
    err_orth_1 = float(np.max(np.abs(R1.T @ R1 - np.eye(K))))
    assert err_orth_1 < 1.0e-5, f"qr_orth_step broke orthogonality: {err_orth_1}"
    print(
        f"selftest: orth={err_orth:.2e} recon={err_recon:.2e} "
        f"qorth={err_orth_q:.2e} tangent_skew={skew_err:.2e} "
        f"step_orth={err_orth_1:.2e}"
    )


if __name__ == "__main__":
    _selftest()
