#!/usr/bin/env python3
"""Pure-Python linear algebra primitives for FLRQ.

The R1-Sketch step in FLRQ (Jan 2026) needs:

- Gaussian random matrix generation (calibration-free sketch)
- Matrix-vector and matrix-matrix products
- QR decomposition (modified Gram-Schmidt)
- Top-k singular value decomposition via block power iteration

This module is deliberately stdlib-only (no numpy, no scipy, no torch)
so the FLRQ sketch primitive can run in environments where the larger
numerical stack is unavailable, and so the sketch step has a small,
auditable reference implementation.  All matrices are 2-D lists of
floats; all vectors are lists of floats.

The functions are tuned for the FLRQ use case: weight matrices up to
roughly 8192 x 8192, sketch width in the dozens, target rank in
{4, 8, 16, 32, 64}.  For larger workloads the numpy path in
``per_tensor_calibrate.py`` is dramatically faster; this module exists
to document the algorithm, to act as a regression oracle, and to keep
the FLRQ sketch step callable from a stdlib-only process.

Conventions:

- ``A`` is a list of ``m`` rows, each a list of ``n`` floats.  ``A[i][j]``
  is the (i, j) element.
- ``x`` is a list of ``n`` floats.  ``x[i]`` is the i-th element.
- All returned matrices follow the same row-major convention.
- Random state is always taken from an explicit ``seed`` so callers get
  reproducible sketches; we never touch ``random`` module state.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def eye(n: int) -> list[list[float]]:
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def random_gaussian(rows: int, cols: int, seed: int) -> list[list[float]]:
    """Gaussian random matrix.  Mean 0, variance 1.  Deterministic in seed."""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------------------
# Element-wise and reductions
# ---------------------------------------------------------------------------


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dot: length mismatch {len(a)} vs {len(b)}")
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def vec_norm(x: Sequence[float]) -> float:
    return math.sqrt(max(dot(x, x), 0.0))


def mat_norm(A: Sequence[Sequence[float]]) -> float:
    s = 0.0
    for row in A:
        for v in row:
            s += v * v
    return math.sqrt(s)


def transpose(A: Sequence[Sequence[float]]) -> list[list[float]]:
    if not A:
        return []
    rows = len(A)
    cols = len(A[0])
    out = zeros(cols, rows)
    for i in range(rows):
        row = A[i]
        for j in range(cols):
            out[j][i] = row[j]
    return out


def scale(A: Sequence[Sequence[float]], c: float) -> list[list[float]]:
    return [[v * c for v in row] for row in A]


def subtract(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(A) != len(B) or (A and len(A[0]) != len(B[0])):
        raise ValueError(
            f"subtract: shape mismatch {len(A)}x{len(A[0]) if A else 0} vs {len(B)}x{len(B[0]) if B else 0}"
        )
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mean_rows(A: Sequence[Sequence[float]]) -> list[float]:
    """Column-wise mean over rows.  Returns a list with len(A[0]) entries."""
    if not A:
        return []
    cols = len(A[0])
    out = [0.0] * cols
    for row in A:
        if len(row) != cols:
            raise ValueError("mean_rows: rows have inconsistent length")
        for j in range(cols):
            out[j] += row[j]
    inv = 1.0 / float(len(A))
    return [v * inv for v in out]


# ---------------------------------------------------------------------------
# Matrix products
# ---------------------------------------------------------------------------


def matvec(A: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
    """y = A @ x.  A is m x n, x is length n, y is length m."""
    if not A:
        return []
    n = len(A[0])
    if len(x) != n:
        raise ValueError(f"matvec: A has {n} cols, x has {len(x)}")
    out = [0.0] * len(A)
    for i, row in enumerate(A):
        s = 0.0
        for j in range(n):
            s += row[j] * x[j]
        out[i] = s
    return out


def matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> list[list[float]]:
    """C = A @ B.  A is m x k, B is k x n, C is m x n.

    Naive triple loop.  The FLRQ sketch path keeps ``k`` small (the
    number of random projections) so a blocked BLAS is not necessary.
    """
    if not A or not B:
        return []
    m = len(A)
    k = len(A[0])
    if len(B) != k:
        raise ValueError(f"matmul: A has {k} cols, B has {len(B)} rows")
    n = len(B[0])
    out = zeros(m, n)
    B_cols: list[list[float]] = transpose(B)
    for i in range(m):
        row = A[i]
        out_row = out[i]
        for j in range(n):
            col = B_cols[j]
            s = 0.0
            for p in range(k):
                s += row[p] * col[p]
            out_row[j] = s
    return out


# ---------------------------------------------------------------------------
# QR and SVD
# ---------------------------------------------------------------------------


def gram_schmidt(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Modified Gram-Schmidt orthonormalisation.

    Returns a list of orthonormal vectors with the same length as the
    input.  Input vectors that collapse to zero are skipped (the
    output may therefore be shorter than the input); callers that need
    a fixed output length should pad or retry.
    """
    out: list[list[float]] = []
    for v in vectors:
        u = list(v)
        for q in out:
            proj = dot(u, q)
            for i in range(len(u)):
                u[i] -= proj * q[i]
        n = vec_norm(u)
        if n > 1e-12:
            inv = 1.0 / n
            for i in range(len(u)):
                u[i] *= inv
            out.append(u)
    return out


def qr(A: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """QR via modified Gram-Schmidt.  A is m x n with m >= n; returns Q (m x n) and R (n x n)."""
    if not A:
        return [], []
    m = len(A)
    n = len(A[0])
    if m < n:
        raise ValueError(f"qr: needs m >= n, got {m} x {n}")
    Q: list[list[list[float]]] = []
    R = zeros(n, n)
    for j in range(n):
        v = [A[i][j] for i in range(m)]
        for k, q in enumerate(Q):
            r_kj = dot(q, v)
            R[k][j] = r_kj
            for i in range(m):
                v[i] -= r_kj * q[i]
        r_jj = vec_norm(v)
        if r_jj < 1e-12:
            # Linearly dependent column.  Pad with a unit vector orthogonal
            # to the existing Q so the caller still gets an n x n R.  In
            # practice this only happens when A is rank-deficient; FLRQ
            # sketches are full-rank with overwhelming probability.
            v = [0.0] * m
            v[j] = 1.0
            for q in Q:
                proj = dot(q, v)
                for i in range(m):
                    v[i] -= proj * q[i]
            r_jj = vec_norm(v)
            if r_jj < 1e-12:
                v = [0.0] * m
                v[j % m] = 1.0
                r_jj = 1.0
        inv = 1.0 / r_jj
        for i in range(m):
            v[i] *= inv
        R[j][j] = r_jj
        Q.append(v)
    # Stack the Q column-vectors into a matrix.
    Qmat = zeros(m, n)
    for j, q in enumerate(Q):
        for i in range(m):
            Qmat[i][j] = q[i]
    return Qmat, R


def power_iteration(
    A: Sequence[Sequence[float]],
    n_iters: int,
    seed: int,
) -> tuple[list[float], float]:
    """Top right singular vector of A via power iteration on A^T A.

    Returns (v, sigma) where A v = sigma u for some unit u, and v is
    the corresponding right singular vector.  ``n_iters`` should be at
    least 20 for FLRQ accuracy; the helper is the building block of
    ``svd_topk`` and is not normally called directly.
    """
    if not A:
        return [], 0.0
    n = len(A[0])
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(n)]
    nrm = vec_norm(v)
    if nrm < 1e-12:
        v[0] = 1.0
    else:
        inv = 1.0 / nrm
        v = [x * inv for x in v]
    AtA = matmul(transpose(A), A)
    sigma = 0.0
    for _ in range(n_iters):
        v = matvec(AtA, v)
        nrm = vec_norm(v)
        if nrm < 1e-12:
            return v, 0.0
        inv = 1.0 / nrm
        v = [x * inv for x in v]
        sigma = math.sqrt(nrm)
    return v, sigma


def svd_topk(
    A: Sequence[Sequence[float]],
    k: int,
    n_iters: int,
    seed: int,
) -> tuple[list[list[float]], list[float]]:
    """Top-k right singular vectors of A via randomised block power iteration.

    Returns (V, sigma) where V is n x k with orthonormal columns and
    ``sigma`` is the list of the top-k singular values.  The algorithm
    follows Halko et al. (2011), section 4.1: a single pass of QB
    iteration is enough for FLRQ because the sketch is already a
    low-variance estimator.

    Sketch: ``Y = A @ Omega`` with ``Omega`` Gaussian of shape n x k.
    QB loop: ``Q = orth((A^T A) @ Q)`` keeps ``Q`` in the row space
    (n x k) so each iteration multiplies by ``A^T A`` (n x n).  The
    final right singular vectors are ``V = Q @ V_B`` where ``V_B`` are
    the right singular vectors of the small matrix ``B = A @ Q``
    (m x k); the singular values of ``B`` equal those of ``A`` in the
    range of the sketch.
    """
    if not A:
        return [], []
    m = len(A)
    n = len(A[0])
    k = max(1, min(k, n, m))
    Omega = random_gaussian(n, k, seed)
    Y = matmul(A, Omega)  # m x k
    Q, _ = qr(Y)  # m x k, orth cols in range(A)
    # Project Q into the row space: Z = A^T @ Q (n x k), then orth.
    Z = matmul(transpose(A), Q)  # n x k
    Qrow, _ = qr(Z)  # n x k, orth cols in row space of A
    AtA = matmul(transpose(A), A)  # n x n
    for _ in range(max(1, n_iters)):
        Z = matmul(AtA, Qrow)  # n x k
        Qrow, _ = qr(Z)  # n x k
    # Small SVD: B = A @ Qrow (m x k).  Right singular vectors of B
    # give us the V_B mapping back to A's right singular vectors.
    B = matmul(A, Qrow)  # m x k
    # Power iteration on B^T B to recover the right singular vectors.
    BtB = matmul(transpose(B), B)  # k x k
    Vcols: list[list[float]] = []
    sigmas: list[float] = []
    rng = random.Random(seed + 1)
    for j in range(k):
        v = [rng.gauss(0.0, 1.0) for _ in range(k)]
        nrm = vec_norm(v)
        if nrm < 1e-12:
            v[0] = 1.0
        else:
            inv = 1.0 / nrm
            v = [x * inv for x in v]
        sigma = 0.0
        for _ in range(max(10, n_iters)):
            v = matvec(BtB, v)
            nrm = vec_norm(v)
            if nrm < 1e-12:
                break
            inv = 1.0 / nrm
            v = [x * inv for x in v]
            sigma = math.sqrt(nrm)
        # Orthogonalise against previously found vectors.
        for prev in Vcols:
            proj = dot(v, prev)
            for i in range(k):
                v[i] -= proj * prev[i]
        nrm = vec_norm(v)
        if nrm < 1e-12:
            # Degenerate direction.  Synthesise an e_j and re-orthogonalise.
            v = [0.0] * k
            v[j] = 1.0
            for prev in Vcols:
                proj = dot(v, prev)
                for i in range(k):
                    v[i] -= proj * prev[i]
            nrm = vec_norm(v)
            if nrm < 1e-12:
                v = [0.0] * k
                v[j] = 1.0
                nrm = 1.0
            inv = 1.0 / nrm
            v = [x * inv for x in v]
            sigma = 0.0
        else:
            inv = 1.0 / nrm
            v = [x * inv for x in v]
            # Recompute sigma against the orthogonalised v: ||B v||.
            Bv = matvec(B, v)
            sigma = vec_norm(Bv)
        Vcols.append(v)
        sigmas.append(float(sigma))
    # Map the small-basis right singular vectors back to A: V = Qrow @ V_B.
    V_small = transpose(Vcols)  # k x k
    V = matmul(Qrow, V_small)  # n x k
    return V, sigmas
