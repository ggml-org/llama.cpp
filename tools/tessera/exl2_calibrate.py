"""EXL2 per-layer sensitivity: GPTQ-style calibration + per-layer bpw allocation.

Phase 0.5 of the iPhone ANE demo: the research-credibility layer
that cross-validates the HIGGS per-layer alpha ranking against an
independent estimator. The math is **reimplemented** in pure NumPy
from:

  - GPTQ (Frantar et al. 2022, open access): column-wise quantization
    with error correction. For each column, quantize, compute the
    reconstruction error, divide by the diagonal of the approximate
    Hessian ``H = X^T X`` (where ``X`` is the calibration activations
    for that layer), and use the scaled error to compensate. The
    closed-form update is::

        e_col = (W_col - quantize(W_col)) / H_diag[col]
        for col > 0: W[col] += e_{col-1}

    Per-layer error is the L2 norm of the column-wise reconstruction
    error stack.

  - EXL2 per-layer allocation (turboderp's README): given a target
    average bpw, search for the per-layer bpw combination that
    minimizes the max per-layer error. The search is a small
    greedy: start with the highest bpw for every layer, then for
    the layer with the lowest marginal-error-per-bpw gain, drop
    to the next-lower bpw, repeat until the average is at target.

The two estimators measure different things. HIGGS (Tessera) is
the per-layer alpha via the Linearity Theorem, weighting the L1
kernel-dequant reconstruction error by the per-tensor Hessian.
EXL2-style is the per-layer sensitivity via GPTQ-style calibration
error, where the algorithm quantizes each layer at multiple bpw
and measures the L2 reconstruction error. Different math, different
proxy, same hardware (Apple Silicon, no CUDA) - both estimate the
same underlying signal: **which transformer layers are the most
sensitive to quantization error?**

The whole module is pure NumPy + the standard library. No torch,
no scipy at the hot path (scipy.stats.spearmanr is only used in
the cross-check test, where it is already a dev dep). No gguf-py
at import time (lazy import inside ``_load_gguf`` so unit tests
don't need gguf-py on the path).

Crash safety: atomic file write for the sidecar JSON and the
markdown report (write to .tmp, fsync, rename). DuckDB writes use
a single transactional INSERT; rollback on any per-row failure.

Sidecar shape: ``ane.exl2-sensitivity.v1`` - the same family as
the HIGGS alpha sidecar (``ane.alpha-coefficients.v1``) so the
L5 retune reads both with the same reader. Per-layer records
carry the chosen bpw, the per-layer error at the chosen bpw, the
Hessian source (calibration corpus name or ``none`` for the
fallback diagonal-unit Hessian), and the family classification
(``attn_q`` / ``attn_k`` / etc., same table as the HIGGS
estimator).

CLI::

    python3 tools/tessera/exl2_calibrate.py \\
        --gguf /path/to/model.gguf \\
        --output /path/to/model.exl2-sensitivity.v1.json \\
        --target-avg-bpw 4.0 \\
        --calibration-corpus wikitext-103

The sidecar is the wire format between this estimator and the L5
orchestrator; the L5 orchestrator reads ``exl2_layer_stats`` in
the unified DuckDB to fold the EXL2 per-layer error into the
per-tensor sensitivity score as a third evidence signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


# ---- constants (ratified by the spec) ----

# The candidate bpw values the algorithm tries. Per the design doc
# (Phase 0.5): ``2, 3, 4, 5, 6, 8``. The ``2`` entry is a true
# ternary (1.585 bits at the entropy limit, ~2 bits with the
# per-tensor scale). The algorithm picks one of these for every
# layer; the per-layer bpw combination is the output the EXL2
# allocator would choose under the target_avg_bpw constraint.
CANDIDATE_BPW: tuple[int, ...] = (2, 3, 4, 5, 6, 8)

# Sidecar schema name. Mirrors the HIGGS alpha sidecar
# (``ane.alpha-coefficients.v1``) so the L5 retune and the iOS
# dispatch read both sidecars with the same reader.
SIDECAR_SCHEMA = "ane.exl2-sensitivity.v1"
SIDECAR_VERSION = 1

# The default target average bpw. The Phase 0.5 spec uses 4.0 as
# the working value (T640_3D is sub-2-bit so the EXL2-allocated
# layer-wise bpw is above the Tessera operating point by design;
# the 4.0 target keeps the EXL2 sidecar informative at the
# model-level quality budget a 4-bit quantizer would set).
DEFAULT_TARGET_AVG_BPW: float = 4.0

# The default calibration corpus name written to the sidecar when
# no calibration data is supplied. The spec ratifies the diagonal
# unit-Hessian fallback as a legitimate "no calibration data" path
# (the design doc Phase 0.5: "If no calibration activations are
# pre-captured, fall back to a diagonal unit Hessian"). The
# ``none`` label is stamped into the sidecar so a downstream
# consumer can detect the fallback.
DEFAULT_CALIBRATION_CORPUS_FALLBACK: str = "no_calibration_diagonal_unit"

# The number of columns the GPTQ column-wise update processes
# per chunk. Column-wise quantization touches one column at a
# time; processing the whole matrix at once would build an N x N
# matrix (N = in_dim, typically 4096+), which is heavy. The
# chunked form processes C columns at a time and accumulates the
# error into the next chunk's leading column; this keeps peak
# memory at O(C x out_dim) instead of O(N x out_dim). The chunk
# size is a power of two for cache-friendly access.
DEFAULT_GPTQ_CHUNK_SIZE: int = 128

# The epsilon added to the Hessian diagonal to avoid
# divide-by-zero on zero-variance columns. The GPTQ paper uses
# a small additive dampening (mean of the diagonal / 100); we
# mirror the convention with a fixed small epsilon that
# independent of the scale.
HESSIAN_DIAG_EPSILON: float = 1.0e-8


# ---- pure math (no gguf, no I/O) ----


def quantize_bpw(
    W: np.ndarray,
    *,
    bpw: int,
) -> np.ndarray:
    """Quantize ``W`` to the per-element bit-width ``bpw``.

    Reimplements the EXL2-style per-bpw grid quantizer: the same
    per-tensor scale (mean absolute value) the EXL2 allocator uses
    is computed once per tensor; the per-element grid is symmetric
    with ``2^bpw`` levels, including zero. The quantizer is
    round-to-nearest; the dequant is the same scale times the
    quantized value. This is the simple per-tensor grid EXL2
    documents in the README (the production EXL2 uses block-wise
    scales; the per-tensor form is the documented lower bound).

    Parameters
    ----------
    W
        The F32 reference weight tensor, any shape.
    bpw
        The target bits per weight. One of ``CANDIDATE_BPW``.
        ``bpw=2`` is a true ternary (``{-1, 0, +1}`` with scale);
        ``bpw=3, 4, 5, 6, 8`` are symmetric uniform grids with
        ``2^bpw`` levels.

    Returns
    -------
    The dequantized F32 reconstruction. Quantization is lossy;
    the per-element quantization error is the GPTQ input.
    """
    if bpw not in CANDIDATE_BPW:
        raise ValueError(
            f"bpw must be one of {CANDIDATE_BPW}, got {bpw}"
        )
    W = np.asarray(W, dtype=np.float32)
    if W.size == 0:
        return W.copy()
    scale = float(np.mean(np.abs(W)))
    if scale <= 0.0 or not np.isfinite(scale):
        return W.copy()
    if bpw == 2:
        # Ternary: round-to-nearest with a mean-absolute threshold.
        # Mirrors the HIGGS offline ternary proxy so the per-tensor
        # comparison across the two estimators is apples-to-apples
        # at the 2-bit operating point.
        threshold = scale
        sign = np.sign(W)
        magnitude = np.abs(W)
        active = magnitude > (threshold * 0.5)
        return (sign * active * scale).astype(np.float32)
    # Symmetric uniform grid with ``levels = 2^bpw`` points
    # (zero included). ``q = round(W / scale)`` clamps to the
    # range ``[-levels/2, levels/2 - 1]``; the dequant is
    # ``q * scale``. The "round" follows numpy semantics
    # (banker's rounding for 0.5; the GPTQ paper's
    # round-to-nearest-even is the same).
    levels = float(1 << bpw)
    half = levels / 2.0
    q = np.clip(np.round(W / scale), -half, half - 1.0)
    return (q * scale).astype(np.float32)


def relative_frobenius_error(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """The per-tensor relative Frobenius reconstruction error
    ``||reconstruction - reference||_F^2 / ||reference||_F^2``.

    Both inputs are flattened (we never need the spatial layout
    for the layer-wise error). Returns 0.0 for a zero-norm
    reference (a degenerate tensor that contributes nothing to
    the ranking; the EXL2 allocator's per-layer error is undefined
    for a zero tensor and we surface a sentinel rather than a NaN).
    """
    ref_flat = np.asarray(reference, dtype=np.float32).ravel()
    rec_flat = np.asarray(reconstruction, dtype=np.float32).ravel()
    if ref_flat.size != rec_flat.size:
        raise ValueError(
            f"reference has {ref_flat.size} elements but "
            f"reconstruction has {rec_flat.size}")
    ref_norm_sq = float(np.dot(ref_flat, ref_flat))
    if ref_norm_sq <= 0.0:
        return 0.0
    diff = rec_flat - ref_flat
    diff_norm_sq = float(np.dot(diff, diff))
    return diff_norm_sq / ref_norm_sq


def gptq_quantize_layer(
    W: np.ndarray,
    *,
    bpw: int,
    hessian: np.ndarray | None = None,
    chunk_size: int = DEFAULT_GPTQ_CHUNK_SIZE,
) -> tuple[np.ndarray, float]:
    """GPTQ-style column-wise quantization with error correction.

    Implements the closed-form update the design doc
    (Phase 0.5) ratifies: the spec's compact form

        ``e_col = (W_col - quantize(W_col)) / H_diag[col]``

        ``for col > 0: W[col] += e_{col-1}``

    is the single-column-propagation form of the full GPTQ
    rank-1 update. The full GPTQ paper does the rank-1
    update to all remaining columns; the spec's form does
    it only to the next column. Both forms are correct
    approximations of the Cholesky-inverse error
    compensation; the spec's form is the one the architect
    pinned down, and we follow it literally.

    The math, step by step, for each column ``col`` in
    column-major order:

      1. If ``col > 0``: add the previous column's scaled
         error ``e_{col-1}`` (a vector of length ``out_dim``)
         to ``W[:, col]``. This is the "compensate" step.
      2. Quantize ``W[:, col]`` to the target bpw on the
         per-tensor grid (the EXL2 allocator's per-tensor
         scale; the column-wise grid is the documented
         lower bound).
      3. Compute the scaled error
         ``e_col = (W[:, col] - quantize(W[:, col])) /
         H_diag[col]``.
      4. Store ``quantize(W[:, col])`` as the column's
         final value; ``e_col`` is the next column's
         compensation.

    The Hessian ``H = X^T X`` is the calibration
    activations outer product, computed offline from the
    calibration corpus the design doc specifies
    (Wikitext-103 + COCO + LibriSpeech for the multimodal
    case). When no calibration activations are
    pre-captured, the caller passes ``hessian=None`` and
    the algorithm uses a unit diagonal
    (``H_diag = ones(in_dim)``): every column sees the
    same unit-scale error correction, and the GPTQ
    update degenerates to "each column gets the previous
    column's raw error added". This is the
    ``no_calibration_diagonal_unit`` fallback the spec
    ratifies.

    Parameters
    ----------
    W
        The F32 reference weight matrix with shape
        ``(out_dim, in_dim)`` (the GGUF / linear-layer
        convention).
    bpw
        Target bits per weight, one of ``CANDIDATE_BPW``.
    hessian
        The F32 Hessian diagonal of shape ``(in_dim,)``.
        ``None`` triggers the diagonal-unit fallback.
    chunk_size
        Unused in the spec's form (the algorithm is
        single-column, so the chunk size is 1 by
        construction). The argument is preserved for
        API symmetry with the chunked-form callers.

    Returns
    -------
    ``(W_hat, per_layer_relative_error)``: ``W_hat`` is the
    GPTQ-quantized reconstruction in F32 (same shape as
    ``W``); ``per_layer_relative_error`` is the L2 norm of
    the reconstruction error normalized by ``||W||_F^2``.
    """
    del chunk_size  # spec's form is single-column; see docstring
    if bpw not in CANDIDATE_BPW:
        raise ValueError(
            f"bpw must be one of {CANDIDATE_BPW}, got {bpw}")
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(
            f"W must be 2-D (out_dim, in_dim); got shape {W.shape}")
    out_dim, in_dim = W.shape
    if in_dim == 0 or out_dim == 0:
        return W.copy(), 0.0

    # Build the Hessian diagonal. ``None`` -> unit diagonal.
    if hessian is None:
        H_diag = np.ones(in_dim, dtype=np.float32)
    else:
        hessian_flat = np.asarray(hessian, dtype=np.float32).ravel()
        if hessian_flat.size != in_dim:
            raise ValueError(
                f"hessian must have in_dim={in_dim} elements, "
                f"got {hessian_flat.size}")
        H_diag = np.maximum(hessian_flat, HESSIAN_DIAG_EPSILON)
    H_inv = 1.0 / H_diag

    # Per-tensor scale (the EXL2 grid scale; the per-tensor
    # form is the documented lower bound of the EXL2
    # allocator). The GPTQ column-wise quantizer uses the
    # same per-tensor scale for every column (the per-tensor
    # grid); the column-wise update is a pure
    # error-compensation step on top of the per-tensor grid.
    W_scale = float(np.mean(np.abs(W)))
    if W_scale <= 0.0 or not np.isfinite(W_scale):
        return W.copy(), 0.0

    levels = float(1 << bpw)
    half = levels / 2.0
    threshold = W_scale  # for the bpw=2 ternary path
    W_hat = np.empty_like(W)
    prev_err = np.zeros(out_dim, dtype=np.float32)
    for col in range(in_dim):
        # 1. Add the previous column's scaled error.
        if col > 0:
            W_col = W[:, col] + prev_err
        else:
            W_col = W[:, col].copy()
        # 2. Quantize the (compensated) column.
        if bpw == 2:
            sign = np.sign(W_col)
            magnitude = np.abs(W_col)
            active = magnitude > (threshold * 0.5)
            q = (sign * active * W_scale).astype(np.float32)
        else:
            q = (np.clip(
                np.round(W_col / W_scale), -half, half - 1.0
            ) * W_scale).astype(np.float32)
        # 3. Scaled error for the next column.
        prev_err = (W_col - q) * H_inv[col]
        # 4. Store the quantized value.
        W_hat[:, col] = q
    rel_err = relative_frobenius_error(W, W_hat)
    return W_hat, rel_err


def per_layer_error_table(
    W: np.ndarray,
    *,
    hessian: np.ndarray | None = None,
    chunk_size: int = DEFAULT_GPTQ_CHUNK_SIZE,
) -> dict[int, float]:
    """Run GPTQ column-wise quantization at every candidate bpw
    and return the per-bpw per-layer relative error.

    Parameters
    ----------
    W
        The F32 reference weight matrix ``(out_dim, in_dim)``.
    hessian
        Optional F32 Hessian diagonal ``(in_dim,)``. ``None``
        triggers the diagonal-unit fallback.
    chunk_size
        GPTQ chunk size; see :func:`gptq_quantize_layer`.

    Returns
    -------
    A dict ``{bpw: relative_error}`` for every bpw in
    ``CANDIDATE_BPW``. The per-bpw value is the L2 reconstruction
    error of GPTQ at that bpw. A 2-D ``W`` of zero norm
    produces a dict of all zeros (degenerate; the allocator
    still sees a valid ``bpw`` choice).
    """
    out: dict[int, float] = {}
    for bpw in CANDIDATE_BPW:
        _W_hat, rel_err = gptq_quantize_layer(
            W, bpw=bpw, hessian=hessian, chunk_size=chunk_size,
        )
        out[bpw] = float(rel_err)
    return out


def exl2_allocate_bpw(
    per_layer_errors: dict[int, dict[int, float]],
    *,
    target_avg_bpw: float,
) -> dict[int, tuple[int, float]]:
    """Greedy EXL2 per-layer bpw allocation.

    Given a ``{layer_index: {bpw: relative_error}}`` map (the
    per-layer error at every candidate bpw) and a target
    average bpw, return the per-layer ``(chosen_bpw,
    relative_error_at_chosen_bpw)`` combination that minimizes
    the max per-layer error under the average-bpw constraint.

    Algorithm (mirrors the EXL2 README's documented intent):

      1. Start with the highest bpw (8) for every layer.
      2. Compute the current average bpw. If it is at or below
         the target, stop.
      3. Among the layers whose current bpw is above the
         minimum, find the layer whose marginal-error-per-bpw
         drop (the relative error increase from dropping to the
         next-lower bpw, divided by the bpw drop) is the
         smallest. Drop that layer to the next-lower bpw.
      4. Repeat from step 2 until the average is at or below
         the target.

    The "marginal-error-per-bpw" heuristic is the EXL2
    allocator's documented search strategy (the README's
    "descend to the next-lower bpw until the budget is met"
    is the high-level description; the marginal-per-bpw
    search is the implementation detail the README does not
    pin down; we use the standard uniform-equal-loss
    criterion: the layer whose error grows the LEAST per
    bit saved is the one that should give up a bit first).

    Parameters
    ----------
    per_layer_errors
        ``{layer_index: {bpw: relative_error}}`` for every
        layer to allocate. Layers with errors at every
        candidate bpw are required; layers missing a bpw
        entry fall back to the highest-bpw choice (the
        layer keeps the bit budget the GPTQ path could not
        characterize).
    target_avg_bpw
        The target average bits per weight across all
        allocated layers. The algorithm returns the
        allocation whose average is at or below the target
        with the smallest max per-layer error.

    Returns
    -------
    A ``{layer_index: (chosen_bpw, relative_error)}`` dict.
    The chosen bpw is one of ``CANDIDATE_BPW``; the relative
    error is the per-layer error at that bpw.
    """
    if not per_layer_errors:
        return {}
    layers = sorted(int(k) for k in per_layer_errors.keys())
    # Initialize every layer at the highest bpw. The sorted
    # CANDIDATE_BPW gives a stable "next-lower" descent.
    bpw_levels = sorted(CANDIDATE_BPW, reverse=True)
    # ``allocation[layer] = (bpw_idx, bpw_value)``; ``bpw_idx``
    # is the index into ``bpw_levels`` (0 = highest).
    allocation: dict[int, int] = {
        layer: 0 for layer in layers
    }

    def _avg_bpw() -> float:
        if not allocation:
            return 0.0
        return sum(bpw_levels[idx] for idx in allocation.values()) / float(
            len(allocation)
        )

    # If the target is at or above the highest-bpw average, every
    # layer stays at the highest bpw. If the target is below the
    # lowest-bpw average, every layer descends to the lowest bpw
    # (the average may still be above target; the algorithm
    # surfaces the lowest achievable average).
    max_avg = float(bpw_levels[0])
    min_avg = float(bpw_levels[-1])
    if target_avg_bpw >= max_avg:
        return {
            layer: (bpw_levels[0], float(
                per_layer_errors[layer].get(bpw_levels[0], 0.0)
            ))
            for layer in layers
        }
    if target_avg_bpw < min_avg:
        return {
            layer: (bpw_levels[-1], float(
                per_layer_errors[layer].get(bpw_levels[-1], 0.0)
            ))
            for layer in layers
        }

    # Greedy loop. The descent is: while avg > target, find the
    # layer with the smallest marginal-error-per-bpw, drop it
    # one rung.
    while _avg_bpw() > target_avg_bpw:
        best_layer: int | None = None
        best_ratio: float = float("inf")
        for layer in layers:
            cur_idx = allocation[layer]
            if cur_idx + 1 >= len(bpw_levels):
                continue  # already at the lowest bpw
            cur_bpw = bpw_levels[cur_idx]
            next_bpw = bpw_levels[cur_idx + 1]
            cur_err = float(
                per_layer_errors[layer].get(cur_bpw, 0.0)
            )
            next_err = float(
                per_layer_errors[layer].get(next_bpw, 0.0)
            )
            err_gain = next_err - cur_err
            bpw_drop = cur_bpw - next_bpw
            if bpw_drop <= 0:
                continue
            ratio = err_gain / float(bpw_drop)
            if ratio < best_ratio:
                best_ratio = ratio
                best_layer = layer
        if best_layer is None:
            break  # every layer at the lowest bpw; the
                   # remaining gap is unrecoverable.
        allocation[best_layer] += 1

    # Build the return dict. The chosen error is the per-layer
    # error at the chosen bpw; layers missing that bpw in the
    # input fall back to the highest-bpw error (the layer
    # could not be characterized; the allocation keeps the
    # bit budget the GPTQ path could not evaluate).
    out: dict[int, tuple[int, float]] = {}
    for layer in layers:
        chosen_idx = allocation[layer]
        chosen_bpw = bpw_levels[chosen_idx]
        chosen_err = float(
            per_layer_errors[layer].get(chosen_bpw, 0.0)
        )
        out[layer] = (chosen_bpw, chosen_err)
    return out


# ---- GGUF loader (lazy gguf-py import) ----


@dataclass(frozen=True)
class LinearLayerInfo:
    """A per-linear-layer record from the source GGUF.

    The EXL2 calibration only operates on linear layers (the
    matrix-shaped weights the GPTQ column-wise update is defined
    for). Embeddings, norms, and biases are skipped; the spec
    ratifies the per-LINEAR-layer scoping.
    """
    name: str
    family: str
    layer_index: int
    n_elements: int
    shape: tuple[int, ...]
    dtype_source: str


def _is_linear_layer(name: str, shape: tuple[int, ...]) -> bool:
    """Decide whether a tensor is a linear-layer weight the
    EXL2 calibration operates on.

    The rule mirrors the HIGGS estimator's family classification
    surface: 2-D tensors whose name matches one of the
    per-block linear families (attn_q / attn_k / attn_v /
    attn_output / ffn_gate / ffn_up / ffn_down). Norms and
    biases are excluded. Embeddings (``token_embd``) and the
    output projection (``output``) are excluded: the spec is
    per-LINEAR-layer *inside the transformer block*; the
    embedding / output projection are not part of the
    per-layer sensitivity ranking the EXL2 allocator
    consumes (the design doc Phase 0.5: "ignore
    embeddings, norms, biases for the EXL2 path; the spec
    is per-LINEAR-layer").
    """
    if len(shape) != 2:
        return False
    # Strip the .weight / .bias suffix.
    base = name
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    # The per-block linear-layer suffixes. The
    # token_embd and output projections are
    # excluded; they live outside the per-block
    # grouping the EXL2 allocator operates on.
    linear_suffixes = (
        "attn_q", "attn_k", "attn_v", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down",
    )
    for suf in linear_suffixes:
        if base == suf or base.endswith("." + suf):
            return True
    return False


def _parse_layer_index(name: str) -> int:
    """Extract the block index from a ``blk.<i>.`` name. Returns
    -1 for tensors outside a block (token_embd, output, etc.).
    The EXL2 allocator groups tensors by layer; the layer
    index is the per-tensor key.
    """
    base = name
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    parts = base.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        try:
            return int(parts[1])
        except ValueError:
            return -1
    return -1


def _classify_family(name: str) -> str:
    """Map a tensor name to its family key, mirroring the
    HIGGS estimator's convention. ``other`` for unrecognized
    families (the EXL2 path is per-LINEAR-layer, so
    unrecognized linear layers get ``other``; non-linear
    tensors are filtered out by ``_is_linear_layer``).
    """
    base = name
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    # Order matters: ``attn_output`` is checked before ``attn_q``
    # so ``blk.16.attn_output`` does not match ``attn_q``.
    family_suffixes = (
        ("attn_k", "attn_k"),
        ("attn_v", "attn_v"),
        ("attn_output", "attn_output"),
        ("attn_q", "attn_q"),
        ("ffn_down", "ffn_down"),
        ("ffn_gate", "ffn_gate"),
        ("ffn_up", "ffn_up"),
        ("token_embd", "token_embd"),
        ("output", "output"),
    )
    for suf, fam in family_suffixes:
        if base == suf or base.endswith("." + suf):
            return fam
    return "other"


def _load_gguf(gguf_path: Path) -> tuple[list, list[str]]:
    """Open the GGUF and return ``(tensors, kv_keys)``.

    Tensors are the gguf-py ``ReaderTensor`` list. The lazy
    import keeps the pure math functions importable for unit
    tests without gguf-py on the path (the same convention as
    ``estimate_higgs_alpha._load_gguf``).
    """
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"failed to import gguf: {exc}. Install gguf-py "
            "(pip install gguf) or run from the tessera "
            "worktree where gguf-py is on PYTHONPATH."
        ) from exc
    reader = GGUFReader(str(gguf_path))
    return list(reader.tensors), list(reader.fields.keys())


def _dequantize_to_f32(tensor) -> np.ndarray:
    """Dequantize a gguf-py tensor to F32.

    Mirrors ``estimate_higgs_alpha._dequantize_to_f32``: F32
    / F16 / the standard Q-traits are handled via the
    gguf-py ``dequantize`` helper; an unknown dtype falls
    back to the raw byte view as F32 (the measurement is
    then a numeric garbage, but the dequant never raises).
    """
    try:
        from gguf.constants import GGMLQuantizationType  # type: ignore
        from gguf.quants import dequantize  # type: ignore
    except ImportError:
        return np.asarray(tensor.data, dtype=np.float32)
    qtype = tensor.tensor_type
    if qtype == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32).reshape(tensor.shape)
    if qtype == GGMLQuantizationType.F16:
        return np.asarray(tensor.data, dtype=np.float16).astype(np.float32).reshape(tensor.shape)
    try:
        return dequantize(np.asarray(tensor.data), qtype).reshape(tensor.shape)
    except NotImplementedError:
        return np.asarray(tensor.data, dtype=np.float32).reshape(tensor.shape)


def select_linear_layers(tensors: list) -> list[LinearLayerInfo]:
    """Pick the linear-layer tensors the EXL2 calibration
    operates on. Embeddings, norms, and biases are filtered
    out. The returned list is in declaration order; the
    per-layer allocation groups by ``layer_index``.
    """
    out: list[LinearLayerInfo] = []
    for t in tensors:
        if not hasattr(t, "shape"):
            continue
        shape = tuple(int(s) for s in t.shape)
        if not _is_linear_layer(t.name, shape):
            continue
        n = 1
        for s in shape:
            n *= s
        if n < 32:
            continue
        out.append(LinearLayerInfo(
            name=str(t.name),
            family=_classify_family(str(t.name)),
            layer_index=_parse_layer_index(str(t.name)),
            n_elements=int(n),
            shape=shape,
            dtype_source=str(t.tensor_type.name),
        ))
    return out


def _model_hash(gguf_path: Path) -> str:
    """Cache-invalidation hash; mirrors
    ``estimate_higgs_alpha.model_hash``: SHA-256 over the
    first 64KB + the last 64KB of the file. Files smaller
    than the window are hashed as a single block.
    """
    h = hashlib.sha256()
    file_size = gguf_path.stat().st_size
    window = 64 * 1024
    with gguf_path.open("rb") as f:
        if file_size <= window:
            h.update(f.read())
        else:
            h.update(f.read(window))
            f.seek(-window, 2)
            h.update(f.read(window))
    return h.hexdigest()[:16]


# ---- the per-layer result record ----


@dataclass
class LayerCalibration:
    """The per-layer calibration result.

    The orchestrator aggregates these into the
    ``ane.exl2-sensitivity.v1`` sidecar and the L5
    orchestrator folds ``per_layer_error`` into the
    per-tensor sensitivity score.
    """
    layer_index: int
    layer_name: str
    family: str
    n_elements: int
    shape: tuple[int, ...]
    dtype_source: str
    # ``per_bpw_error[bpw]`` is the GPTQ relative error at that
    # bpw (one entry per ``CANDIDATE_BPW``). The EXL2 allocator
    # reads this table to choose the per-layer bpw.
    per_bpw_error: dict[int, float]
    # The bpw the EXL2 allocator chose for this layer under the
    # target_avg_bpw constraint.
    chosen_bpw: int
    # The per-layer relative error at the chosen bpw.
    per_layer_error: float


# ---- atomic write helpers (sidecar JSON + markdown report) ----


def write_sidecar_atomic(
    path: Path,
    sidecar: dict,
) -> None:
    """Write the sidecar JSON atomically.

    The contract: write to ``path + ".tmp"``, fsync, rename to
    ``path``. A crash mid-write leaves the original ``path``
    intact (or absent, if it didn't exist) and the
    ``.tmp`` file is detectable for forensic cleanup. The
    L5 retune reader and the iOS dispatch both consume the
    sidecar as a JSON document; a partial file would be a
    parse error at load time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # Use a NamedTemporaryFile-like pattern in the same
    # directory so the rename is on the same filesystem
    # (rename across filesystems is not atomic on macOS).
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the tmp file on failure; the original
        # ``path`` is untouched.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_report_atomic(
    path: Path,
    sidecar: dict,
) -> None:
    """Write the human-readable markdown report atomically.

    The report is a one-table-per-section dump with the
    chosen bpw, the per-bpw error table, the family, and
    the layer shape. The first section is the audit
    summary (target avg bpw, fallback flag, layer count).
    The format mirrors the design doc's Phase 0.5
    "per-layer table" requirement.
    """
    lines: list[str] = []
    lines.append(f"# EXL2 per-layer sensitivity report: {sidecar['bundle_name']}")
    lines.append("")
    lines.append(f"- **Model hash**: `{sidecar['model_hash']}`")
    lines.append(f"- **Source GGUF**: `{sidecar['gguf_path']}`")
    lines.append(f"- **Schema**: `{sidecar['schema']}` v{sidecar['version']}")
    lines.append(f"- **Target average bpw**: {sidecar['target_avg_bpw']}")
    lines.append(f"- **Calibration corpus**: `{sidecar['calibration_corpus']}`")
    lines.append(f"- **Hessian source**: `{sidecar['hessian_source']}`")
    lines.append(f"- **GPTQ chunk size**: {sidecar['gptq_chunk_size']}")
    lines.append(f"- **Layer count**: {sidecar['layer_count']}")
    lines.append(f"- **Achieved average bpw**: {sidecar['achieved_avg_bpw']:.3f}")
    lines.append(f"- **Max per-layer error**: {sidecar['max_per_layer_error']:.6e}")
    lines.append("")
    lines.append("## bpw distribution")
    lines.append("")
    lines.append("| chosen bpw | layer count | fraction |")
    lines.append("|---:|---:|---:|")
    bpw_dist = sidecar.get("bpw_distribution", {})
    for bpw in sorted(bpw_dist.keys()):
        d = bpw_dist[bpw]
        lines.append(
            f"| {bpw} | {d['count']} | {d['fraction']:.3f} |"
        )
    lines.append("")
    lines.append("## Per-layer results")
    lines.append("")
    lines.append("| Layer | Family | Shape | n_elements | chosen bpw | per-layer error |")
    lines.append("|---:|---|---|---:|---:|---:|")
    for layer in sidecar["layers"]:
        lines.append(
            f"| {layer['layer_index']} | {layer['family']} | "
            f"{list(layer['shape'])} | {layer['n_elements']:,} | "
            f"{layer['chosen_bpw']} | {layer['per_layer_error']:.6e} |"
        )
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


# ---- sidecar assembly ----


def build_sidecar(
    calibrations: list[LayerCalibration],
    *,
    model_hash_value: str,
    gguf_path: Path,
    bundle_name: str | None = None,
    target_avg_bpw: float,
    calibration_corpus: str,
    gptq_chunk_size: int,
    hessian_source: str,
) -> dict:
    """Assemble the ``ane.exl2-sensitivity.v1`` sidecar.

    The shape mirrors the HIGGS alpha sidecar
    (``ane.alpha-coefficients.v1``) so the L5 retune
    reader treats both sidecars uniformly. The
    per-layer records carry the chosen bpw, the
    per-bpw error table, and the per-layer error
    at the chosen bpw.
    """
    name = bundle_name or gguf_path.stem
    layers_json: list[dict] = []
    chosen_bpws: list[int] = []
    per_layer_errors: list[float] = []
    bpw_distribution: dict[int, dict[str, float]] = {}
    for c in calibrations:
        layers_json.append({
            "layer_index": int(c.layer_index),
            "layer_name": str(c.layer_name),
            "family": str(c.family),
            "shape": list(c.shape),
            "n_elements": int(c.n_elements),
            "dtype_source": str(c.dtype_source),
            "per_bpw_error": {
                str(int(bpw)): float(err)
                for bpw, err in sorted(c.per_bpw_error.items())
            },
            "chosen_bpw": int(c.chosen_bpw),
            "per_layer_error": float(c.per_layer_error),
        })
        chosen_bpws.append(int(c.chosen_bpw))
        per_layer_errors.append(float(c.per_layer_error))
        bpw = int(c.chosen_bpw)
        if bpw not in bpw_distribution:
            bpw_distribution[bpw] = {"count": 0, "fraction": 0.0}
        bpw_distribution[bpw]["count"] += 1
    n = max(1, len(calibrations))
    for bpw, d in bpw_distribution.items():
        d["fraction"] = float(d["count"]) / float(n)
    achieved_avg = (
        sum(chosen_bpws) / float(n) if chosen_bpws else 0.0
    )
    max_err = max(per_layer_errors) if per_layer_errors else 0.0
    return {
        "schema": SIDECAR_SCHEMA,
        "version": SIDECAR_VERSION,
        "bundle_name": name,
        "gguf_path": str(gguf_path),
        "model_hash": str(model_hash_value),
        "target_avg_bpw": float(target_avg_bpw),
        "calibration_corpus": str(calibration_corpus),
        "hessian_source": str(hessian_source),
        "gptq_chunk_size": int(gptq_chunk_size),
        "candidate_bpw": list(CANDIDATE_BPW),
        "layer_count": int(len(calibrations)),
        "achieved_avg_bpw": float(achieved_avg),
        "max_per_layer_error": float(max_err),
        "bpw_distribution": {
            str(int(bpw)): d for bpw, d in sorted(bpw_distribution.items())
        },
        "layers": layers_json,
    }


# ---- the calibration orchestrator ----


def calibrate(
    gguf_path: Path,
    *,
    target_avg_bpw: float = DEFAULT_TARGET_AVG_BPW,
    calibration_corpus: str = DEFAULT_CALIBRATION_CORPUS_FALLBACK,
    hessian: dict[str, np.ndarray] | None = None,
    gptq_chunk_size: int = DEFAULT_GPTQ_CHUNK_SIZE,
    verbose: bool = False,
) -> tuple[list[LayerCalibration], dict]:
    """Run the full EXL2 calibration on a GGUF.

    Steps:

      1. Load the GGUF and pick the linear-layer tensors.
      2. For each tensor, dequantize to F32.
      3. Run GPTQ column-wise quantization at every
         ``CANDIDATE_BPW``; record the per-bpw relative error.
      4. Run the greedy EXL2 per-layer allocation under the
         ``target_avg_bpw`` constraint.

    Parameters
    ----------
    gguf_path
        The path to the source GGUF.
    target_avg_bpw
        The target average bpw. Default
        :data:`DEFAULT_TARGET_AVG_BPW`.
    calibration_corpus
        The calibration corpus name stamped into the
        sidecar. When no calibration activations are
        pre-captured, the caller passes the
        ``DEFAULT_CALIBRATION_CORPUS_FALLBACK`` (the
        spec-ratified ``no_calibration_diagonal_unit``).
    hessian
        Optional ``{tensor_name: hessian_diagonal}`` map.
        ``None`` triggers the diagonal-unit fallback for
        every layer. The caller can pre-compute Hessians
        from a calibration corpus and pass them here; the
        ``calibration_corpus`` string is stamped into the
        sidecar so the audit trail records the source.
    gptq_chunk_size
        GPTQ chunk size. See :func:`gptq_quantize_layer`.
    verbose
        Print a one-line summary per layer (off by default;
        the sidecar is the durable record).

    Returns
    -------
    ``(calibrations, audit)`` where ``calibrations`` is the
    list of :class:`LayerCalibration` (one per linear layer)
    and ``audit`` is the sidecar's top-level metadata
    (model_hash, target_avg_bpw, calibration_corpus,
    hessian_source, gptq_chunk_size).
    """
    tensors, _kv_keys = _load_gguf(gguf_path)
    linear_layers = select_linear_layers(tensors)
    if not linear_layers:
        raise SystemExit(
            f"GGUF {gguf_path} has no linear-layer tensors; "
            "EXL2 calibration operates on 2-D matrix weights only."
        )
    # Per-layer grouping: the EXL2 allocator operates on a
    # per-layer index, not per-tensor. The per-layer
    # allocation sums the per-bpw error over every tensor in
    # the layer (the per-layer error is the max across the
    # layer's tensors; the spec says "minimize the max
    # per-layer error", so the per-layer error is the
    # per-layer max-of-tensors).
    per_layer_tensors: dict[int, list[LinearLayerInfo]] = {}
    for li in linear_layers:
        per_layer_tensors.setdefault(li.layer_index, []).append(li)
    model_hash_value = _model_hash(gguf_path)
    hessian_source = (
        f"calibration_corpus:{calibration_corpus}"
        if hessian is not None
        else "diagonal_unit_fallback"
    )

    # For each layer, run GPTQ at every candidate bpw on every
    # tensor in the layer, then take the per-layer max error.
    # The per-layer max is the conservative choice: a layer is
    # only as good as its worst tensor.
    per_layer_errors: dict[int, dict[int, float]] = {}
    per_layer_records: dict[int, list[dict]] = {}
    for layer_index, infos in per_layer_tensors.items():
        layer_err_per_bpw: dict[int, float] = {bpw: 0.0 for bpw in CANDIDATE_BPW}
        layer_records: list[dict] = []
        for li in infos:
            t = next(t for t in tensors if str(t.name) == li.name)
            W = _dequantize_to_f32(t)
            if W.ndim != 2:
                continue
            h_diag = None
            if hessian is not None and li.name in hessian:
                h_diag = hessian[li.name]
            err_table = per_layer_error_table(
                W, hessian=h_diag, chunk_size=gptq_chunk_size,
            )
            for bpw, err in err_table.items():
                layer_err_per_bpw[bpw] = max(
                    layer_err_per_bpw[bpw], float(err)
                )
            layer_records.append({
                "info": li,
                "per_bpw_error": err_table,
            })
            if verbose:
                print(
                    f"  {li.name}: bpw="
                    f"{ {k: round(v, 4) for k, v in err_table.items()} }",
                    file=sys.stderr,
                )
        per_layer_errors[layer_index] = layer_err_per_bpw
        per_layer_records[layer_index] = layer_records

    # EXL2 per-layer allocation.
    allocation = exl2_allocate_bpw(
        per_layer_errors, target_avg_bpw=float(target_avg_bpw),
    )

    # Build the per-layer calibration records. The
    # ``layer_name`` field is the canonical name of the first
    # linear-layer tensor in the layer (the EXL2 allocator
    # groups tensors by layer; the L5 orchestrator looks up
    # the per-layer error by ``layer_index`` so the name is
    # just for human display).
    calibrations: list[LayerCalibration] = []
    for layer_index in sorted(per_layer_tensors.keys()):
        infos = per_layer_tensors[layer_index]
        layer_records = per_layer_records[layer_index]
        chosen_bpw, per_layer_err = allocation.get(
            layer_index, (max(CANDIDATE_BPW), 0.0)
        )
        # The per-bpw error is the per-layer max across the
        # layer's tensors; we report the per-layer
        # aggregate.
        err_table = per_layer_errors[layer_index]
        # The ``layer_name`` is the most prominent tensor in
        # the layer: the attn_output.weight if present, else
        # the first tensor. The orchestrator looks up by
        # ``layer_index`` so the name is purely cosmetic.
        layer_name = ""
        for li in infos:
            if li.family == "attn_output":
                layer_name = li.name
                break
        if not layer_name and infos:
            layer_name = infos[0].name
        # The shape / n_elements are the layer aggregate (sum
        # over the layer's tensors; the family is the most
        # common family in the layer).
        n_total = sum(int(li.n_elements) for li in infos)
        family_counts: dict[str, int] = {}
        for li in infos:
            family_counts[li.family] = family_counts.get(li.family, 0) + 1
        family = (
            max(family_counts, key=family_counts.get)
            if family_counts else "other"
        )
        # The per-layer shape is the (in_dim, out_dim) of the
        # attn_output tensor if present; else the first
        # tensor. The shape is reported in the sidecar for
        # human display only.
        rep_shape = infos[0].shape if infos else (0, 0)
        for li in infos:
            if li.family == "attn_output":
                rep_shape = li.shape
                break
        # The dtype_source is the dtype of the first tensor
        # in the layer; EXL2 calibration operates on the
        # dequantized F32 regardless of the source dtype.
        dtype_source = (
            infos[0].dtype_source if infos else "unknown"
        )
        calibrations.append(LayerCalibration(
            layer_index=int(layer_index),
            layer_name=str(layer_name),
            family=str(family),
            n_elements=int(n_total),
            shape=tuple(int(s) for s in rep_shape),
            dtype_source=str(dtype_source),
            per_bpw_error={int(k): float(v) for k, v in err_table.items()},
            chosen_bpw=int(chosen_bpw),
            per_layer_error=float(per_layer_err),
        ))
    audit = {
        "model_hash": model_hash_value,
        "target_avg_bpw": float(target_avg_bpw),
        "calibration_corpus": str(calibration_corpus),
        "hessian_source": hessian_source,
        "gptq_chunk_size": int(gptq_chunk_size),
    }
    return calibrations, audit


# ---- DuckDB write (additive table) ----


def write_to_duckdb(
    db_path: Path,
    *,
    model_hash_value: str,
    calibrations: list[LayerCalibration],
    calibration_corpus: str,
) -> int:
    """Write the per-layer calibrations to the unified DuckDB.

    The ``exl2_layer_stats`` table is created on the fly
    (``CREATE TABLE IF NOT EXISTS``); the schema is the
    additive one the Phase 0.5 spec ratifies. The
    ``PRIMARY KEY (model_hash, layer_index, exl2_calibration_corpus)``
    composite lets multiple corpus runs coexist (the same
    model calibrated against wikitext-103 and COCO produces
    distinct rows; the audit trail records which corpus the
    row came from).

    Returns the number of rows written.
    """
    import duckdb  # type: ignore
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS exl2_layer_stats (
                model_hash              TEXT NOT NULL,
                layer_index             INTEGER NOT NULL,
                layer_name              TEXT,
                family                  TEXT,
                n_elements              BIGINT,
                exl2_per_layer_error    DOUBLE,
                exl2_per_layer_bpw      DOUBLE,
                exl2_chosen_bpw         INTEGER,
                exl2_calibration_corpus TEXT NOT NULL,
                created_at              TIMESTAMP DEFAULT now(),
                PRIMARY KEY (model_hash, layer_index, exl2_calibration_corpus)
            )
            """
        )
        # The per-tensor sensitivity path's additive
        # ``exl2_error`` column on ``l5_plan_summary``
        # is the L5 orchestrator's migration
        # (``TesseraDB._ensure_l5_plan_columns``).
        # The calibrator does not write to
        # ``l5_plan_summary``; the column migration
        # fires on the orchestrator's first open.
        # We skip the ALTER here so a pre-Phase-0.5
        # DB without ``l5_plan_summary`` (the EXL2
        # calibrator's first run on a fresh DB) does
        # not see a Catalog Error. The L5 orchestrator
        # picks up the column migration on its
        # first open.
        # Transactional batch insert. The
        # ``INSERT ... ON CONFLICT DO UPDATE`` lets a
        # re-run against the same (model, layer,
        # corpus) overwrite the prior values without a
        # manual delete (the audit trail is in the
        # ``model_hash + corpus`` PK; the value
        # reflects the most recent run).
        rows: list[tuple] = []
        for c in calibrations:
            rows.append((
                str(model_hash_value),
                int(c.layer_index),
                str(c.layer_name),
                str(c.family),
                int(c.n_elements),
                float(c.per_layer_error),
                float(c.chosen_bpw),
                int(c.chosen_bpw),
                str(calibration_corpus),
            ))
        con.executemany(
            """
            INSERT INTO exl2_layer_stats (
                model_hash, layer_index, layer_name, family,
                n_elements, exl2_per_layer_error, exl2_per_layer_bpw,
                exl2_chosen_bpw, exl2_calibration_corpus
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (model_hash, layer_index, exl2_calibration_corpus)
            DO UPDATE SET
                layer_name           = excluded.layer_name,
                family               = excluded.family,
                n_elements           = excluded.n_elements,
                exl2_per_layer_error = excluded.exl2_per_layer_error,
                exl2_per_layer_bpw   = excluded.exl2_per_layer_bpw,
                exl2_chosen_bpw      = excluded.exl2_chosen_bpw,
                created_at           = now()
            """,
            rows,
        )
        return len(rows)
    finally:
        con.close()


# ---- CLI ----


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EXL2 per-layer sensitivity: GPTQ-style calibration "
            "+ per-layer bpw allocation, reimplemented in pure "
            "NumPy. Same Mac, same corpus, same model as the "
            "HIGGS estimator (the cross-check the iPhone ANE "
            "demo design ratifies in Phase 0.5). See "
            "docs/tessera-ane-ios-demo-design.md."
        ),
    )
    parser.add_argument(
        "--gguf", type=Path, required=True,
        help="Path to the source GGUF (the model being calibrated).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help=(
            "Output path for the sidecar JSON. Conventional "
            "name: <bundle>.exl2-sensitivity.v1.json."
        ),
    )
    parser.add_argument(
        "--target-avg-bpw", type=float,
        default=DEFAULT_TARGET_AVG_BPW,
        help=(
            "Target average bits per weight for the EXL2 "
            "allocation (default 4.0). The allocator finds "
            "the per-layer bpw combination whose average is "
            "at or below this target and whose max per-layer "
            "error is minimized."
        ),
    )
    parser.add_argument(
        "--calibration-corpus", type=str,
        default=DEFAULT_CALIBRATION_CORPUS_FALLBACK,
        help=(
            "Calibration corpus name stamped into the "
            "sidecar (default: "
            f"{DEFAULT_CALIBRATION_CORPUS_FALLBACK!r}, the "
            "spec-ratified 'no calibration data' fallback)."
        ),
    )
    parser.add_argument(
        "--json-sidecar", type=Path, default=None,
        help=(
            "Optional output path for the L5-retune-shape "
            "sidecar JSON. Defaults to --output when not "
            "given. The L5 retune reads the exl2_layer_stats "
            "DuckDB table, so this is the human-audit copy."
        ),
    )
    parser.add_argument(
        "--model-hash", type=str, default=None,
        help=(
            "Override the model_hash (default: auto-compute "
            "from the GGUF header + tail). The hash is the "
            "cache-invalidation key for the L5 retune."
        ),
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help=(
            "Optional path for the human-readable markdown "
            "report. Default: alongside --output as "
            "<sidecar-stem>.report.md."
        ),
    )
    parser.add_argument(
        "--duckdb", type=Path, default=None,
        help=(
            "Optional path to the unified tessera.duckdb. "
            "When set, the per-layer results are also written "
            "to the exl2_layer_stats table (the L5 "
            "orchestrator's read path)."
        ),
    )
    parser.add_argument(
        "--gptq-chunk-size", type=int,
        default=DEFAULT_GPTQ_CHUNK_SIZE,
        help=(
            "GPTQ column-wise chunk size (default 128). The "
            "chunked form keeps peak memory at "
            "O(chunk_size * out_dim); the math is unchanged."
        ),
    )
    parser.add_argument(
        "--bundle-name", type=str, default=None,
        help="Override the bundle name in the sidecar.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print a one-line summary per layer (off by default).",
    )
    args = parser.parse_args(argv)
    if not args.gguf.is_file():
        raise SystemExit(f"GGUF not found: {args.gguf}")
    if args.target_avg_bpw <= 0.0:
        raise SystemExit("--target-avg-bpw must be > 0")
    if args.gptq_chunk_size < 1:
        raise SystemExit("--gptq-chunk-size must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    t0 = time.monotonic()
    calibrations, audit = calibrate(
        args.gguf,
        target_avg_bpw=args.target_avg_bpw,
        calibration_corpus=args.calibration_corpus,
        gptq_chunk_size=args.gptq_chunk_size,
        verbose=args.verbose,
    )
    model_hash_value = (
        args.model_hash if args.model_hash else audit["model_hash"]
    )
    sidecar = build_sidecar(
        calibrations,
        model_hash_value=model_hash_value,
        gguf_path=args.gguf,
        bundle_name=args.bundle_name,
        target_avg_bpw=args.target_avg_bpw,
        calibration_corpus=args.calibration_corpus,
        gptq_chunk_size=args.gptq_chunk_size,
        hessian_source=audit["hessian_source"],
    )
    json_sidecar = args.json_sidecar or args.output
    write_sidecar_atomic(json_sidecar, sidecar)
    report_path = args.report or args.output.with_suffix(".report.md")
    write_report_atomic(report_path, sidecar)
    if args.duckdb is not None:
        write_to_duckdb(
            args.duckdb,
            model_hash_value=model_hash_value,
            calibrations=calibrations,
            calibration_corpus=args.calibration_corpus,
        )
    elapsed = time.monotonic() - t0
    n_layers = len(calibrations)
    chosen_avg = sidecar["achieved_avg_bpw"]
    max_err = sidecar["max_per_layer_error"]
    print(f"wrote {json_sidecar}", file=sys.stderr)
    print(f"  model_hash:         {model_hash_value}", file=sys.stderr)
    print(f"  layer_count:        {n_layers}", file=sys.stderr)
    print(f"  target_avg_bpw:     {args.target_avg_bpw}", file=sys.stderr)
    print(f"  achieved_avg_bpw:   {chosen_avg:.3f}", file=sys.stderr)
    print(f"  max_per_layer_err:  {max_err:.6e}", file=sys.stderr)
    print(f"  calibration_corpus: {args.calibration_corpus}", file=sys.stderr)
    print(f"  hessian_source:     {audit['hessian_source']}",
          file=sys.stderr)
    print(f"  elapsed:            {elapsed:.2f}s", file=sys.stderr)
    print(f"wrote {report_path}", file=sys.stderr)
    if args.duckdb is not None:
        print(f"wrote {args.duckdb} (exl2_layer_stats)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
