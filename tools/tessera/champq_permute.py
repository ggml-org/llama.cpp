#!/usr/bin/env python3
"""CHAMP-Q channel permutation for the Tile640 quantizer.

CHAMP-Q permutes the input channels of a weight matrix so the most
sensitive channels (largest activation magnitude) are grouped together,
then runs the existing AWQ / ternary quantization on the permuted
weight. After quantization, the inverse permutation is applied to the
output so the runtime sees a normal Tile640 tensor (same channel order
as the source). The permutation is computed at calibration time, so the
runtime cost is zero. The output GGUF is bit-compatible with the
non-CHAMP-Q path.

This is the "simple" L2-norm rank permutation. A learned per-layer
permutation via LBFGS that minimizes the BF16-vs-quantized cross-entropy
is future work and is intentionally out of scope here.

The default integration in tools/tile640/quantize_v3.py uses Option A
(see PROJECT-STATUS.md / runtime-aware-pipeline notes): the encoded
Tile640 components are decoded back to a dense F32 weight, the input
dimension is permuted back to the original order, and the un-permuted
weight is re-quantized. The output GGUF is therefore in original channel
order and is interchangeable with the non-CHAMP-Q output. The cost is
one extra quantization pass per tensor; the benefit (when it materialises
on real activations) is a lower per-row ternary error.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Tile640 constants (must match ggml C++ + quantize_v3.py).
TILE640_PAGE_SIZE = 640
TILE640_LANE_SIZE = 20
TILE640_LANES_PER_PAGE = 32

# JSON schema for the on-disk policy. Versioned so future
# permutations (e.g. a learned per-layer LBFGS) can extend it.
SCHEMA = "llama.tessera.champq-permute.v1"


# ---------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------


def compute_champq_permutation(
    arr: np.ndarray,
    act_scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a permutation of the input-channel axis that puts the most
    active channels first.

    Args:
        arr: weight matrix with shape (..., in_dim). The last axis is
            the input-channel axis that will be permuted.
        act_scales: optional per-input-channel activation magnitude with
            shape (in_dim,). When provided, the permutation is sorted by
            the activation observer. When absent, the permutation is
            sorted by the per-channel L2 norm of the weight (a cheap
            proxy for "channel importance" that requires no calibration
            data).

    Returns:
        1-D np.ndarray of length in_dim, dtype int64, a permutation of
        [0, in_dim). Indices are ordered from largest magnitude to
        smallest.
    """
    if arr.ndim < 2:
        raise ValueError(
            f"compute_champq_permutation: arr must be at least 2-D, got {arr.ndim}-D"
        )
    in_dim = arr.shape[-1]
    if act_scales is not None:
        if act_scales.shape != (in_dim,):
            raise ValueError(
                f"act_scales shape {act_scales.shape} does not match in_dim {in_dim}"
            )
        magnitudes = np.asarray(act_scales, dtype=np.float64)
    else:
        flat = arr.reshape(-1, in_dim).astype(np.float64)
        magnitudes = np.linalg.norm(flat, axis=0)
    # argsort descending: largest magnitudes first
    return np.argsort(-magnitudes, kind="stable").astype(np.int64)


def apply_champq_permutation(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply the input-channel permutation to a 2-D or 3-D weight array.

    Args:
        arr: weight matrix with shape (..., in_dim). The last axis is
            permuted.
        perm: 1-D int array, length in_dim, a permutation of [0, in_dim).

    Returns:
        A new array with the same shape as arr, with the last axis
        permuted by `perm`.
    """
    if arr.shape[-1] != perm.shape[0]:
        raise ValueError(
            f"apply_champq_permutation: perm length {perm.shape[0]} does not match "
            f"in_dim {arr.shape[-1]}"
        )
    return np.ascontiguousarray(arr[..., perm])


def invert_champq_permutation(perm: np.ndarray) -> np.ndarray:
    """Return the inverse permutation such that
    apply_champq_permutation(apply_champq_permutation(arr, perm),
    inverse(perm)) == arr."""
    perm = np.asarray(perm, dtype=np.int64)
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(perm.size, dtype=perm.dtype)
    return inverse


# ---------------------------------------------------------------------------
# Extended permutation modes (random / GA / LBFGS)
# ---------------------------------------------------------------------------
#
# The "simple" L2-norm rank permutation above is the calibration-time
# default. The modes below are alternative strategies that may pick a
# permutation with different properties. They share the same return
# contract: a 1-D np.ndarray of length in_dim, dtype int64, a
# permutation of [0, in_dim).
#
# The LBFGS mode is the "smart" version. It parameterises a
# doubly-stochastic matrix M (Sinkhorn-normalised), uses LBFGS to
# minimise a smoothness proxy over the relaxed problem, and projects
# back to a permutation via the Hungarian algorithm (or the greedy
# row-max trick if Hungarian is too expensive for the chosen K).
#
# All three modes are pure-numpy + stdlib. No torch, no scipy. The
# LBFGS step itself is in tools/tessera/_champq_lbfgs.py.


def random_permutation(in_dim: int, seed: int = 0) -> np.ndarray:
    """Return a uniformly random permutation of [0, in_dim)."""
    if in_dim <= 0:
        raise ValueError(f"in_dim must be positive, got {in_dim}")
    rng = np.random.default_rng(seed)
    return rng.permutation(in_dim).astype(np.int64)


def ga_permutation(
    weight: np.ndarray,
    act_scales: Optional[np.ndarray] = None,
    population: int = 16,
    generations: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Learn a permutation via a small genetic algorithm that minimises
    the smoothness proxy.

    Population is initialised with the L2-rank permutation (cheap
    heuristic) plus ``population - 1`` random permutations. The
    fitness is the negative smoothness proxy (lower smoothness is
    better). Selection is a 3-tournament; crossover is order
    crossover (OX); mutation is a pair-swap on a random position.

    Args:
        weight: weight matrix with shape (..., in_dim). The last
            axis is the permutation target.
        act_scales: optional per-input-channel activation magnitude
            with shape (in_dim,). When provided, the smoothness
            proxy is weighted by ``act_scales``; when absent, the
            proxy is unweighted.
        population: number of individuals in the GA population.
        generations: number of generations.
        seed: RNG seed.

    Returns:
        1-D np.ndarray of length in_dim, dtype int64. The best
        permutation found by the GA.
    """
    if weight.ndim < 2:
        raise ValueError(
            f"ga_permutation: weight must be at least 2-D, got {weight.ndim}-D"
        )
    in_dim = weight.shape[-1]
    rng = np.random.default_rng(seed)

    # Initial population: the L2-rank heuristic + random fill.
    pop: List[np.ndarray] = [compute_champq_permutation(weight, act_scales)]
    while len(pop) < max(population, 1):
        pop.append(rng.permutation(in_dim).astype(np.int64))

    def fitness(perm: np.ndarray) -> float:
        return smoothness_proxy(weight, perm, act_scales)

    # Evaluate initial population.
    scores = np.array([fitness(p) for p in pop], dtype=np.float64)
    best_idx = int(np.argmin(scores))
    best_perm = pop[best_idx].copy()
    best_score = float(scores[best_idx])

    for gen in range(generations):
        # Tournament selection: pick 3, return the best.
        def tournament() -> np.ndarray:
            idxs = rng.integers(0, len(pop), size=3)
            j = int(np.argmin(scores[idxs]))
            return pop[idxs[j]].copy()

        # Order crossover (OX) on two parents.
        def ox_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            i, j = sorted(rng.integers(0, in_dim, size=2).tolist())
            child = -np.ones(in_dim, dtype=np.int64)
            child[i:j] = p1[i:j]
            # Fill the rest in the order of p2, skipping p1[i:j].
            p2_set = set(p1[i:j].tolist())
            fill = [x for x in p2.tolist() if x not in p2_set]
            # Walk through positions outside [i, j) and place fill[i].
            out_pos = [k for k in range(in_dim) if k < i or k >= j]
            for k, v in zip(out_pos, fill):
                child[k] = v
            return child

        # Pair-swap mutation.
        def swap_mutate(perm: np.ndarray) -> np.ndarray:
            a, b = rng.integers(0, in_dim, size=2).tolist()
            out = perm.copy()
            out[a], out[b] = out[b], out[a]
            return out

        # Build next generation with elitism (keep the best).
        new_pop: List[np.ndarray] = [best_perm.copy()]
        new_scores: List[float] = [best_score]
        while len(new_pop) < len(pop):
            p1 = tournament()
            p2 = tournament()
            child = ox_crossover(p1, p2)
            if rng.random() < 0.2:
                child = swap_mutate(child)
            new_pop.append(child)
            new_scores.append(fitness(child))
        pop = new_pop
        scores = np.array(new_scores, dtype=np.float64)
        best_idx = int(np.argmin(scores))
        if scores[best_idx] < best_score:
            best_perm = pop[best_idx].copy()
            best_score = float(scores[best_idx])

    return best_perm


# ---------------------------------------------------------------------------
# Smoothness proxy and continuous relaxation
# ---------------------------------------------------------------------------


def smoothness_proxy(
    weight: np.ndarray,
    perm: np.ndarray,
    act_scales: Optional[np.ndarray] = None,
) -> float:
    """Sum of squared second-differences of the permuted weight rows,
    optionally weighted by per-input-channel activation magnitude.

    Lower is smoother. A perfectly smooth (linear ramp) row has
    second-difference zero and contributes nothing; a row with sharp
    channel-to-channel swings contributes heavily.

    Args:
        weight: weight matrix with shape (..., in_dim). The last
            axis is permuted.
        perm: 1-D int array of length in_dim, a permutation.
        act_scales: optional per-input-channel weight of shape
            (in_dim,). When provided, the squared second-difference
            at column ``c`` is multiplied by ``act_scales[c]``; when
            absent, the proxy is unweighted (a row-only smoothness
            measure).

    Returns:
        A single float, the proxy value. Zero on a constant row.
    """
    if weight.shape[-1] != perm.shape[0]:
        raise ValueError(
            f"smoothness_proxy: perm length {perm.shape[0]} does not match "
            f"in_dim {weight.shape[-1]}"
        )
    w_perm = apply_champq_permutation(weight, perm)
    # Second difference along the input axis. Boundary (first and
    # last column) contribute nothing because we cannot form a
    # 3-point stencil there.
    d2 = w_perm[..., :-2] - 2.0 * w_perm[..., 1:-1] + w_perm[..., 2:]
    s2 = d2 * d2  # (..., K-2)
    if act_scales is not None:
        if act_scales.shape != (weight.shape[-1],):
            raise ValueError(
                f"act_scales shape {act_scales.shape} does not match in_dim {weight.shape[-1]}"
            )
        # X_hat[c] weights the squared second-difference at column
        # c (the centre of the 3-point stencil). The d2 axis
        # corresponds to c in [1, K-2].
        s2 = s2 * act_scales[..., 1:-1]
    return float(np.sum(s2))


def sinkhorn_project(M: np.ndarray, n_iters: int = 25, eps: float = 1.0e-12) -> np.ndarray:
    """Project a non-negative matrix to the doubly-stochastic manifold
    by alternating row and column normalisation.

    The standard Sinkhorn-Knopp algorithm. For each iteration, divide
    each row by its sum (so the rows sum to 1), then divide each
    column by its sum (so the columns sum to 1). Converges geometrically
    at a rate that depends on the spectral gap of ``log M``.

    Args:
        M: non-negative (n, n) matrix.
        n_iters: number of alternating normalisations. 20-50 is
            typically enough for a permutation-friendly M.
        eps: small constant to avoid division by zero.

    Returns:
        A (n, n) numpy array with non-negative entries and
        row/column sums all close to 1.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"sinkhorn_project: expected square matrix, got {M.shape}")
    if n_iters <= 0:
        return M
    out = np.maximum(M, eps)
    for _ in range(int(n_iters)):
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > eps, row_sum, 1.0)
        out = out / row_sum
        col_sum = out.sum(axis=0, keepdims=True)
        col_sum = np.where(col_sum > eps, col_sum, 1.0)
        out = out / col_sum
    return out


def hungarian(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost assignment via the Kuhn-Munkres algorithm.

    Pure numpy + Python loops. O(n^3) time, O(n^2) memory. The
    implementation is the standard ``u, v, p, way`` formulation
    (often called the Jonker-Volgenant variant) translated to
    numpy; the per-row inner loop is a Python for-loop, which is
    fine for K up to a few thousand.

    Args:
        cost: (n, n) cost matrix. Lower cost = more preferred.

    Returns:
        A 1-D np.ndarray of length n, dtype int64. ``assignment[i]``
        is the column assigned to row ``i``. The assignment is a
        permutation of [0, n).
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"hungarian: expected square matrix, got {cost.shape}")
    n = cost.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    INF = float("inf")
    # 1-indexed working arrays of size n+1. Index 0 is a sentinel
    # (the unmatched / null column). Using 1-indexed arrays avoids
    # a special case for column 0 inside the inner loop.
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)  # p[j] = row currently matched to column j
    way = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = INF
            j1 = 0
            # Inner loop: relax edges from row i0 to every free column.
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if int(p[j0]) == 0:
                break
        # Augmenting: walk back the path and update the matching.
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] = i means column j is matched to row i. Invert to get
    # assignment[i] = j.
    assignment = np.zeros(n, dtype=np.int64)
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[int(p[j]) - 1] = j - 1
    return assignment


def greedy_assignment(M: np.ndarray) -> np.ndarray:
    """Greedy row-max assignment. O(n^2) but suboptimal. Used as a
    fallback when ``hungarian`` is too slow for very large K.

    The greedy algorithm walks rows in order and assigns each row
    to the highest-weight unused column. Does not minimise the
    total cost; the optimality gap is small in practice when M is
    close to a permutation matrix (e.g. after Sinkhorn).
    """
    n = M.shape[0]
    used = np.zeros(n, dtype=bool)
    # Sort all (row, col) pairs by descending M value and walk them.
    flat = M.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    assignment = np.full(n, -1, dtype=np.int64)
    for idx in order:
        r, c = divmod(int(idx), n)
        if not used[c] and assignment[r] == -1:
            assignment[r] = c
            used[c] = True
        if np.all(used):
            break
    if np.any(assignment == -1):
        # Fill any leftover rows with the remaining columns in
        # ascending order. Should only happen if M has a row with
        # all-zero entries, which Sinkhorn avoids.
        leftover = [c for c in range(n) if not used[c]]
        for i in range(n):
            if assignment[i] == -1:
                assignment[i] = leftover.pop(0)
    return assignment


def project_to_permutation(M: np.ndarray, mode: str = "hungarian") -> np.ndarray:
    """Project a non-negative matrix to a permutation.

    ``mode = "hungarian"`` uses the O(n^3) Kuhn-Munkres solver to
    minimise the total cost. ``mode = "greedy"`` uses the O(n^2)
    greedy row-max assignment, which is faster but suboptimal. For
    K up to a few thousand, Hungarian is fine; beyond that, fall
    back to greedy.
    """
    if mode == "hungarian":
        return hungarian(-np.asarray(M, dtype=np.float64))
    if mode == "greedy":
        return greedy_assignment(np.asarray(M, dtype=np.float64))
    raise ValueError(f"unknown projection mode {mode!r}")


# ---------------------------------------------------------------------------
# LBFGS permutation: continuous relaxation of the smoothness objective
# ---------------------------------------------------------------------------


def smoothness_loss_grad(
    M: np.ndarray,
    weight: np.ndarray,
    act_scales: Optional[np.ndarray] = None,
    binariness: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Smoothness loss and gradient for the continuous relaxation.

    The continuous relaxation treats the permutation as a doubly-
    stochastic matrix ``M`` (shape (K, K)). The relaxed weight is
    ``W_perm = W @ M`` (shape (out_dim, K)). The smoothness loss is
    the same as the discrete proxy applied to ``W_perm``:

        L(M) = sum_{r, c=1..K-2} (W_perm[r, c-1] - 2*W_perm[r, c]
                                    + W_perm[r, c+1])^2 * X_hat[c]

    with the boundary contribution zeroed (no 3-point stencil at
    the edges).

    The gradient is closed-form. Using ``W_perm = W @ M`` and
    treating M as the variable, ``d W_perm[r, c] / d M[i, k] = W[r, i]
    * delta(c, k)``. Let ``g[r, c] = d L / d W_perm[r, c]``; then

        d L / d M[i, k] = sum_r g[r, k] * W[r, i] = (W.T @ g)[i, k]

    The closed-form gradient makes the LBFGS step cheap (one
    matmul per step in addition to the W @ M matmul).

    Binariness penalty: when ``binariness > 0``, add

        lambda * sum_{i, j} M[i, j] * (1 - M[i, j])

    to the loss. This is zero on a permutation matrix (M in {0, 1})
    and positive on fractional entries, so it pushes M toward the
    corners of the unit hypercube. Without this penalty, the
    smoothness loss is over-relaxed: fractional M can interpolate
    the second-difference stencil and achieve a much lower loss
    than any actual permutation. The penalty balances the two
    objectives; ``binariness = 1.0`` is a reasonable starting point
    on K ~ 64-1024 weights and may need tuning for other K.

    Args:
        M: (K, K) non-negative matrix, the current iterate. The
            caller is expected to keep M on the doubly-stochastic
            manifold via Sinkhorn projection between steps.
        weight: (out_dim, K) weight matrix.
        act_scales: optional (K,) per-input-channel weight. When
            None, the loss is unweighted.
        binariness: weight of the binariness penalty. 0 disables it.

    Returns:
        (loss, grad_M) where ``loss`` is a Python float and
        ``grad_M`` is a numpy array of shape (K, K).
    """
    M = np.asarray(M, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if M.ndim != 2 or M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"smoothness_loss_grad: M must be square, got {M.shape}")
    if M.shape[1] != weight.shape[-1]:
        raise ValueError(
            f"smoothness_loss_grad: M shape {M.shape} incompatible with weight {weight.shape}"
        )
    w_perm = weight @ M  # (out_dim, K)
    in_dim = w_perm.shape[1]
    if in_dim < 3:
        # Trivial case: no interior columns, smoothness is zero.
        return 0.0, np.zeros_like(M)

    out_dim, K = w_perm.shape
    d2 = w_perm[:, :-2] - 2.0 * w_perm[:, 1:-1] + w_perm[:, 2:]  # (out_dim, K-2)
    s2 = d2 * d2
    if act_scales is not None:
        if act_scales.shape != (in_dim,):
            raise ValueError(
                f"act_scales shape {act_scales.shape} does not match in_dim {in_dim}"
            )
        s2 = s2 * act_scales[1:-1]  # X_hat[c] weights the stencil at column c
    loss = float(np.sum(s2))

    # Gradient: g[r, c] = d L / d W_perm[r, c]. Boundary c=0, c=K-1
    # contributes nothing. For c in [1, K-2], the three d2 entries
    # that depend on W_perm[r, c] are:
    #   d2[r, c-2] weighted by X_hat[c-1], derivative w.r.t. W_perm[r, c] is +1
    #   d2[r, c-1] weighted by X_hat[c],   derivative is -2
    #   d2[r, c]   weighted by X_hat[c+1], derivative is +1
    # So g[r, c] = 2 * (X_hat[c-1] * d2[r, c-2] - 2*X_hat[c]*d2[r, c-1] + X_hat[c+1]*d2[r, c]).
    #
    # Implementation: weight d2 by X_hat first (h = d2 * X_hat[1:-1]
    # element-wise), then pad with zeros at both ends and apply the
    # 3-point stencil once. The padding handles the one-sided
    # stencil at c=1 and c=K-2 correctly (the missing neighbour is
    # zero, not a repeat of the boundary value).
    if act_scales is not None:
        h = d2 * act_scales[1:-1]
    else:
        h = d2
    h_pad = np.zeros((out_dim, K), dtype=w_perm.dtype)
    h_pad[:, 1:K - 1] = h
    g = np.zeros_like(w_perm)
    g[:, 1:K - 1] = 2.0 * (h_pad[:, :-2] - 2.0 * h_pad[:, 1:-1] + h_pad[:, 2:])

    grad_M = weight.T @ g  # (K, K)

    if binariness > 0.0:
        # Binariness penalty: lambda * sum M * (1 - M). Gradient is
        # lambda * (1 - 2*M). At M = 0 or 1 the gradient is +/- lambda,
        # so the LBFGS sees a constant push toward the corners.
        loss = loss + float(binariness) * float(np.sum(M * (1.0 - M)))
        grad_M = grad_M + float(binariness) * (1.0 - 2.0 * M)

    return loss, grad_M


def lbfgs_permutation(
    weight: np.ndarray,
    act_scales: Optional[np.ndarray] = None,
    n_iters: int = 30,
    lr: float = 1.0e-1,
    sinkhorn_iters: int = 20,
    history: int = 8,
    seed: int = 0,
    projection: str = "hungarian",
    init: str = "l2rank",
    binariness: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    """Learn an input-channel permutation via LBFGS on the continuous
    relaxation of the smoothness proxy.

    Algorithm:

    1. Initialise ``M`` (the doubly-stochastic iterate). The
       default is to start from the L2-rank permutation embedded
       as a permutation matrix; ``init = "random"`` starts from
       a uniform doubly-stochastic initialisation (Sinkhorn on a
       random non-negative matrix); ``init = "identity"`` starts
       from the identity permutation.
    2. Repeat for ``n_iters`` iterations:
       a. Compute the smoothness loss and closed-form gradient
          via ``smoothness_loss_grad``.
       b. Take one LBFGS step in M-space (see
          ``tools/tessera/_champq_lbfgs.py``).
       c. Project the new M back to the doubly-stochastic manifold
          via Sinkhorn normalisation.
    3. Project the final M to a permutation matrix via
       Hungarian assignment (or greedy row-max when ``projection
       = "greedy"``).

    Args:
        weight: weight matrix with shape (out_dim, in_dim).
        act_scales: optional per-input-channel weight of shape
            (in_dim,). When None, the smoothness proxy is
            unweighted.
        n_iters: number of LBFGS iterations.
        lr: step size for the LBFGS line search starting alpha.
            The line search shrinks automatically; lr is the
            initial value, not a hard step size.
        sinkhorn_iters: number of Sinkhorn alternating
            normalisations applied after each LBFGS step.
        history: LBFGS history length.
        seed: RNG seed (used for ``init = "random"``).
        projection: ``"hungarian"`` (default) or ``"greedy"``.
        init: ``"l2rank"`` (default), ``"random"``, or
            ``"identity"``.
        verbose: log each iteration's loss to stderr.

    Returns:
        A 1-D np.ndarray of length in_dim, dtype int64, the
        learned permutation.
    """
    # Imported lazily to keep the module importable without numpy
    # aliasing issues (and to make the dependency one-way).
    from _champq_lbfgs import LBFGS

    if weight.ndim != 2:
        raise ValueError(
            f"lbfgs_permutation: weight must be 2-D, got {weight.ndim}-D"
        )
    out_dim, in_dim = weight.shape

    if init == "l2rank":
        rank_perm = compute_champq_permutation(weight, act_scales)
        M = np.zeros((in_dim, in_dim), dtype=np.float64)
        M[np.arange(in_dim), rank_perm] = 1.0
    elif init == "identity":
        M = np.eye(in_dim, dtype=np.float64)
    elif init == "random":
        rng = np.random.default_rng(seed)
        M = rng.uniform(0.0, 1.0, size=(in_dim, in_dim)).astype(np.float64)
        # A few more Sinkhorn iterations than the per-step count
        # so the init is well on the doubly-stochastic manifold.
        M = sinkhorn_project(M, n_iters=max(sinkhorn_iters, 50))
    else:
        raise ValueError(f"unknown init {init!r}")

    if verbose:
        print(
            f"lbfgs_permutation: init={init} K={in_dim} iters={n_iters} "
            f"sinkhorn={sinkhorn_iters} history={history}",
            file=sys.stderr,
        )

    opt = LBFGS(n_params=in_dim * in_dim, history=history, c1=1e-4, max_ls=12)
    M_flat = M.reshape(-1)

    def closure(m_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        m = m_flat.reshape(in_dim, in_dim)
        loss, grad = smoothness_loss_grad(m, weight, act_scales, binariness=binariness)
        return loss, grad.reshape(-1)

    loss, grad = closure(M_flat)
    gnorm = float(np.linalg.norm(grad))
    if verbose:
        print(
            f"  lbfgs iter  0/{n_iters}  loss={loss:.6e}  |g|={gnorm:.3e}",
            file=sys.stderr,
        )
    grad_tol = 1.0e-6 * max(gnorm, 1.0)
    for it in range(int(n_iters)):
        M_new_flat, loss_new, grad_new, done = opt.step(M_flat, loss, grad, closure)
        # Re-project onto the doubly-stochastic manifold.
        M_new = sinkhorn_project(M_new_flat.reshape(in_dim, in_dim), n_iters=sinkhorn_iters)
        M_new_flat = M_new.reshape(-1)
        # Recompute the loss / grad at the projected point so the
        # next LBFGS step has the right values.
        loss, grad = closure(M_new_flat)
        gnorm = float(np.linalg.norm(grad))
        M_flat = M_new_flat
        if verbose:
            print(
                f"  lbfgs iter {it + 1:2d}/{n_iters}  loss={loss:.6e}  "
                f"|g|={gnorm:.3e}",
                file=sys.stderr,
            )
        if gnorm < grad_tol:
            if verbose:
                print(f"  lbfgs converged: |g|={gnorm:.3e} < tol={grad_tol:.3e}", file=sys.stderr)
            break

    # Final projection to a permutation.
    M_final = M_flat.reshape(in_dim, in_dim)
    perm = project_to_permutation(M_final, mode=projection)
    return perm


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


PERMUTE_MODES = ("rank", "random", "ga", "lbfgs")


def compute_permutation(
    mode: str,
    weight: np.ndarray,
    act_scales: Optional[np.ndarray] = None,
    seed: int = 0,
    ga_population: int = 16,
    ga_generations: int = 20,
    lbfgs_iters: int = 30,
    lbfgs_lr: float = 1.0e-1,
    lbfgs_history: int = 8,
    lbfgs_sinkhorn_iters: int = 20,
    lbfgs_projection: str = "hungarian",
    lbfgs_init: str = "l2rank",
    lbfgs_binariness: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    """Dispatch to the requested permutation mode.

    Args:
        mode: one of ``"rank"`` (L2-norm rank, the calibration
            default), ``"random"`` (uniform random permutation),
            ``"ga"`` (small genetic algorithm), ``"lbfgs"``
            (continuous relaxation + LBFGS, the new smart mode).
        weight: weight matrix with shape (..., in_dim).
        act_scales: optional per-input-channel weight.
        seed: RNG seed for ``"random"``, ``"ga"``, and
            ``"lbfgs"`` (init = "random").
        ga_population: population size for ``"ga"``.
        ga_generations: number of generations for ``"ga"``.
        lbfgs_iters: LBFGS iterations.
        lbfgs_lr: LBFGS initial step size.
        lbfgs_history: LBFGS history length.
        lbfgs_sinkhorn_iters: Sinkhorn iterations per LBFGS step.
        lbfgs_projection: ``"hungarian"`` or ``"greedy"``.
        lbfgs_init: ``"l2rank"``, ``"random"``, or ``"identity"``.
        verbose: log progress to stderr.

    Returns:
        1-D np.ndarray of length in_dim, dtype int64.
    """
    if mode == "rank":
        return compute_champq_permutation(weight, act_scales)
    if mode == "random":
        in_dim = weight.shape[-1]
        return random_permutation(in_dim, seed=seed)
    if mode == "ga":
        return ga_permutation(
            weight,
            act_scales=act_scales,
            population=ga_population,
            generations=ga_generations,
            seed=seed,
        )
    if mode == "lbfgs":
        return lbfgs_permutation(
            weight,
            act_scales=act_scales,
            n_iters=lbfgs_iters,
            lr=lbfgs_lr,
            history=lbfgs_history,
            sinkhorn_iters=lbfgs_sinkhorn_iters,
            projection=lbfgs_projection,
            init=lbfgs_init,
            binariness=lbfgs_binariness,
            seed=seed,
            verbose=verbose,
        )
    raise ValueError(
        f"unknown permute mode {mode!r}; expected one of {PERMUTE_MODES}"
    )


# ---------------------------------------------------------------------------
# Tile640 decode (reverse of pack_tile640 + compute_scales + outliers)
# ---------------------------------------------------------------------------


def _unpack_pow3() -> np.ndarray:
    """3^k for k in [0, TILE640_LANE_SIZE). Pre-computed once."""
    return np.array(
        [3 ** i for i in range(TILE640_LANE_SIZE)], dtype=np.uint32
    )


_POW3 = _unpack_pow3()


def decode_tile640_quantized(
    packed: np.ndarray,
    page_scales: np.ndarray,
    lane_scales: np.ndarray,
    outlier_row_offsets: np.ndarray,
    outlier_cols: np.ndarray,
    outlier_vals: np.ndarray,
    out_dim: int,
    in_dim: int,
) -> np.ndarray:
    """Reverse the Tile640 encoding to a dense F32 weight in the
    AWQ-scaled space (i.e. before the input_scale is applied). Callers
    that want the original weight scale must multiply by input_scale
    afterwards.

    Mirrors pack_tile640 + compute_scales + select_repair_residuals in
    tools/tile640/quantize_v3.py. Tested for in_dim that is and is not a
    multiple of TILE640_PAGE_SIZE.
    """
    pages_per_row = (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
    padded_in_dim = pages_per_row * TILE640_PAGE_SIZE

    # 1. Unpack u32 words to ternary {-1, 0, 1}.
    # packed has shape (out_dim * pages_per_row * 32,) flattened from
    # (out_dim, pages_per_row, 32).
    words = packed.astype(np.uint32).reshape(
        out_dim, pages_per_row, TILE640_LANES_PER_PAGE
    )
    trit_indices = (words[:, :, :, None] // _POW3[None, None, None, :]) % 3
    ternary = np.where(
        trit_indices == 1,
        np.int8(1),
        np.where(trit_indices == 2, np.int8(-1), np.int8(0)),
    )

    # 2. Per-lane scale: page_scale * lane_scale_i8 / 127.
    ps = page_scales.astype(np.float32).reshape(out_dim, pages_per_row)
    ls = lane_scales.astype(np.float32).reshape(
        out_dim, pages_per_row, TILE640_LANES_PER_PAGE
    )
    lane_value_scale = (ps[:, :, None] * ls / np.float32(127.0))[:, :, :, None]

    # 3. Decode: ternary * lane_value_scale.
    decoded = (ternary.astype(np.float32) * lane_value_scale).reshape(
        out_dim, padded_in_dim
    )

    # 4. Add outliers. outlier_cols indices are in [0, in_dim), so they
    # fit inside the padded row.
    for row in range(out_dim):
        start = int(outlier_row_offsets[row])
        end = int(outlier_row_offsets[row + 1])
        if end > start:
            cols = outlier_cols[start:end].astype(np.int64)
            decoded[row, cols] = outlier_vals[start:end].astype(np.float32)

    # 5. Trim padding.
    if padded_in_dim != in_dim:
        decoded = decoded[:, :in_dim]
    return np.ascontiguousarray(decoded)


def decode_q_to_weight(q: Dict[str, np.ndarray], out_dim: int, in_dim: int) -> np.ndarray:
    """Decode a quantize_2d result dict to a dense F32 weight in the
    original weight scale (after the AWQ input_scale is applied)."""
    decoded_scaled = decode_tile640_quantized(
        q["packed"],
        q["page_scales"],
        q["lane_scales"],
        q["outlier_row_offsets"],
        q["outlier_cols"],
        q["outlier_vals"],
        out_dim,
        in_dim,
    )
    input_scale = q["input_scale"].astype(np.float32).reshape(1, -1)
    return decoded_scaled * input_scale


# ---------------------------------------------------------------------------
# Policy dataclass (debug / A-B comparison)
# ---------------------------------------------------------------------------


@dataclass
class CHAMPQPolicy:
    """Per-tensor CHAMP-Q policy. Records the input-channel permutation
    that was applied to each weight, so the output can be reproduced
    or re-applied at load time (future Option B)."""

    schema: str = SCHEMA
    tensors: Dict[str, List[int]] = field(default_factory=dict)

    def add(self, name: str, perm: np.ndarray) -> None:
        self.tensors[name] = np.asarray(perm, dtype=np.int64).tolist()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"schema": self.schema, "tensors": self.tensors},
                handle,
                separators=(",", ":"),
            )

    @staticmethod
    def load(path: str) -> "CHAMPQPolicy":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema") != SCHEMA:
            raise ValueError(
                f"unsupported CHAMP-Q policy schema: {data.get('schema')!r}"
            )
        return CHAMPQPolicy(schema=data["schema"], tensors=data["tensors"])


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _build_arg_parser() -> "argparse.ArgumentParser":
    """Build the CLI argparse parser. Kept separate so the module can
    be imported without the argparse side effects (and so other
    tools can build their own parser on top of this one)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="champq_permute",
        description=(
            "Compute an input-channel permutation for CHAMP-Q permute-then-quantize. "
            "The new ``--permute-mode lbfgs`` mode learns the permutation via LBFGS "
            "on a continuous relaxation of the smoothness proxy. ``random`` and ``ga`` "
            "are simpler alternatives; ``rank`` is the legacy L2-norm sort."
        ),
    )
    parser.add_argument(
        "--permute-mode",
        choices=PERMUTE_MODES,
        default="lbfgs",
        help="Permutation mode (default: lbfgs).",
    )
    parser.add_argument(
        "--weight-npz",
        type=str,
        default=None,
        help="Optional path to a .npz with a 'weight' (out_dim, in_dim) and optional 'act_scales' (in_dim,). "
        "When omitted, a synthetic rank-8 + Gaussian 4096x4096 weight is generated for the A/B smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0).",
    )
    parser.add_argument(
        "--lbfgs-iters",
        type=int,
        default=30,
        help="Number of LBFGS iterations (default: 30).",
    )
    parser.add_argument(
        "--lbfgs-lr",
        type=float,
        default=1.0e-1,
        help="Initial LBFGS step size (default: 1e-1). The line search shrinks as needed.",
    )
    parser.add_argument(
        "--lbfgs-history",
        type=int,
        default=8,
        help="LBFGS history length (default: 8).",
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
        help="Number of Sinkhorn alternating normalisations after each LBFGS step (default: 20).",
    )
    parser.add_argument(
        "--lbfgs-projection",
        choices=("hungarian", "greedy"),
        default="hungarian",
        help="Final projection to a permutation (default: hungarian).",
    )
    parser.add_argument(
        "--lbfgs-init",
        choices=("l2rank", "random", "identity"),
        default="l2rank",
        help="Initial doubly-stochastic matrix (default: l2rank).",
    )
    parser.add_argument(
        "--lbfgs-binariness",
        type=float,
        default=1.0e-3,
        help="Binariness penalty weight for the LBFGS objective (default: 1e-3). "
        "Penalises fractional M entries to keep the relaxation close to a permutation. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--lbfgs-subsample",
        type=int,
        default=0,
        help="If > 0, run the LBFGS on a sub-matrix of this many input channels (default: 0 = full K). "
        "Used to keep the smoke-test runtime bounded on K=4096. The remaining channels are left identity.",
    )
    parser.add_argument(
        "--ga-population",
        type=int,
        default=16,
        help="GA population size (default: 16).",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=20,
        help="GA generations (default: 20).",
    )
    parser.add_argument(
        "--output-policy",
        type=str,
        default=None,
        help="Optional path to write a CHAMPQPolicy JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log progress to stderr.",
    )
    return parser


def _synthetic_weight(out_dim: int, in_dim: int, rank: int, seed: int) -> np.ndarray:
    """Generate a synthetic weight matrix: low-rank + Gaussian noise.

    The low-rank component is the same construction used in
    champq_lbfgs_ab.py: rank-`rank` outer product of two random
    vectors, scaled to dominate the Gaussian component. This makes
    the permutation-sensitive structure (the row-space of the
    low-rank factor) visible to the smoothness proxy.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(out_dim, rank)).astype(np.float64)
    v = rng.normal(size=(in_dim, rank)).astype(np.float64)
    low_rank = u @ v.T
    noise = 0.1 * rng.normal(size=(out_dim, in_dim)).astype(np.float64)
    return (low_rank + noise).astype(np.float32)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Loads a weight (or generates a synthetic one),
    runs the requested permutation mode, and (optionally) writes a
    CHAMPQPolicy JSON. Prints the smoothness proxy and a few stats
    to stdout for smoke-testing the pipeline.
    """
    args = _build_arg_parser().parse_args(argv)

    if args.weight_npz is not None:
        with np.load(args.weight_npz, allow_pickle=False) as data:
            weight = np.asarray(data["weight"], dtype=np.float32)
            act_scales = (
                np.asarray(data["act_scales"], dtype=np.float32).reshape(-1)
                if "act_scales" in data
                else None
            )
    else:
        weight = _synthetic_weight(4096, 4096, rank=8, seed=args.seed)
        act_scales = None

    if args.permute_mode == "lbfgs" and args.lbfgs_subsample > 0 and args.lbfgs_subsample < weight.shape[-1]:
        # Sub-permutation on the first N channels; the rest stay identity.
        weight_sub = weight[:, : args.lbfgs_subsample]
        act_sub = act_scales[: args.lbfgs_subsample] if act_scales is not None else None
        perm_sub = compute_permutation(
            "lbfgs",
            weight_sub,
            act_scales=act_sub,
            seed=args.seed,
            lbfgs_iters=args.lbfgs_iters,
            lbfgs_lr=args.lbfgs_lr,
            lbfgs_history=args.lbfgs_history,
            lbfgs_sinkhorn_iters=args.sinkhorn_iters,
            lbfgs_projection=args.lbfgs_projection,
            lbfgs_init=args.lbfgs_init,
            lbfgs_binariness=args.lbfgs_binariness,
            verbose=args.verbose,
        )
        perm = np.arange(weight.shape[-1], dtype=np.int64)
        perm[: args.lbfgs_subsample] = perm_sub
        perm[args.lbfgs_subsample:] = np.arange(args.lbfgs_subsample, weight.shape[-1], dtype=np.int64)
    else:
        perm = compute_permutation(
            args.permute_mode,
            weight,
            act_scales=act_scales,
            seed=args.seed,
            ga_population=args.ga_population,
            ga_generations=args.ga_generations,
            lbfgs_iters=args.lbfgs_iters,
            lbfgs_lr=args.lbfgs_lr,
            lbfgs_history=args.lbfgs_history,
            lbfgs_sinkhorn_iters=args.sinkhorn_iters,
            lbfgs_projection=args.lbfgs_projection,
            lbfgs_init=args.lbfgs_init,
            lbfgs_binariness=args.lbfgs_binariness,
            verbose=args.verbose,
        )

    proxy_value = smoothness_proxy(weight, perm, act_scales)
    identity_count = int(np.sum(perm == np.arange(perm.shape[0])))
    print(
        f"mode={args.permute_mode} in_dim={perm.shape[0]} "
        f"smoothness={proxy_value:.6e} identity_count={identity_count}/{perm.shape[0]}"
    )

    if args.output_policy is not None:
        policy = CHAMPQPolicy()
        policy.add("<weight>", perm)
        policy.save(args.output_policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

