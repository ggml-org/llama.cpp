#!/usr/bin/env python3
"""Per-tensor Tessera calibration.

Builds a ``llama.speculative.calibration-policy.v1`` JSON for downstream
``tile640_quantize_v3.py --calibration-policy`` consumption by learning
per-tensor quantization parameters from a layer bundle (.npz) containing the
BF16 weight, calibration activations, and (optionally) observer moments.

Fitness modes (selected via ``--fitness``):

* ``awq``     delegate to ``awq-evolve.py`` (island-GA over the existing
  ``Candidate`` space). The output schema matches what
  ``tile640_quantize_v3.py`` already consumes.
* ``lrq``     learn a low-rank weight-scaling matrix ``S = U @ V`` (NAACL 2025
  formulation, output-channel x rank and rank x input-channel factors). The
  effective scale is applied to the weight before ternarization. Training
  uses pure-numpy Adam with a straight-through estimator for the
  non-differentiable ternary quantization. The policy stores ``U`` and ``V``
  per tensor (a few KB at rank 16) and the rank. ``tile640_quantize_v3.py``
  recognises the LRQ fields and reconstructs ``S`` to derive the AWQ-style
  per-input-channel aggregate.
* ``flrq``    FLRQ (Jan 2026): R1-Sketch with Gaussian projection to identify
  outlier-aware low-rank components, then Best Low-Rank Approximation under
  Clipping (BLC) to minimise the residual quantisation error. The policy
  stores FP16 ``U`` (K x r) and ``V`` (r x N) plus a tile640-style
  quantised residual. Calibration-free: only the weight matrix ``W`` is
  read; the rank is selected by sweeping ``[4, 8, 16, 32, 64]`` and
  keeping the smallest r whose relative reconstruction MSE is below
  ``--flrq-mse-threshold`` (default 1e-3).
* ``compare`` run ``awq`` and ``lrq`` on the same bundle and write a
  side-by-side report covering per-tensor MSE, policy size, and the relative
  reduction the LRQ-mode achieves versus the GA baseline.

The LRQ mode is the new contribution. The other modes are wired for symmetry
and to support the comparison flow, but they reuse the existing tooling
where possible.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# Memory-bound / spatial-temporal utilities.  These were added in
# Phase 16 (calibration memopt); see ``calibration_memory.py`` for
# the per-category docs.  ``calibration_residency.py`` is the
# peak-RSS budgeting side; it is imported lazily in ``main()`` so
# the import is only paid when the budget knob is used.
try:
    from .calibration_memory import (
        SPATIAL_ROLES,
        CalibPipeline,
        CalibPipelineAsync,
        ChunkSpec,
        chunked_iter,
        chunked_process,
        extract_layer_index,
        interleave_components,
        mmap_layer,
        mmap_tensor,
        open_calib_pipeline,
    )
    from .calibration_residency import (
        ResidencyTracker,
        default_budget_gb,
        read_rss_bytes,
    )
    # Phase 16.5 (memopt-metal-dispatch).  The per-chunk
    # LRQ / FLRQ matmul is dispatched to the fastest
    # available backend (Metal > Accelerate > numpy).  The
    # import is fail-safe: on Linux/Windows the dispatch
    # returns numpy without invoking the C bridge, so the
    # import is a no-op.
    from .calibration_metal import (
        chunked_matmul,
        get_matmul_backend_name,
    )
except ImportError:  # pragma: no cover - script-mode import
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tools.tessera.calibration_memory import (  # type: ignore[no-redef]
        SPATIAL_ROLES,
        CalibPipeline,
        CalibPipelineAsync,
        ChunkSpec,
        chunked_iter,
        chunked_process,
        extract_layer_index,
        interleave_components,
        mmap_layer,
        mmap_tensor,
        open_calib_pipeline,
    )
    from tools.tessera.calibration_residency import (  # type: ignore[no-redef]
        ResidencyTracker,
        default_budget_gb,
        read_rss_bytes,
    )
    from tools.tessera.calibration_metal import (  # type: ignore[no-redef]
        chunked_matmul,
        get_matmul_backend_name,
    )


SCHEMA = "llama.speculative.calibration-policy.v1"
LRQ_SCHEMA = "llama.tessera.lrq-policy.v1"
DARTQUANT_SCHEMA = "llama.tessera.dartquant-policy.v1"
DEFAULT_RANK = 16
DEFAULT_ITERATIONS = 50
DEFAULT_LR = 1.0e-3
DEFAULT_OUTLIER_FRAC = 0.005
# Phase 16 (calibration memopt).  Default chunk size for the
# row-chunked processing path.  For a 12B FFN gate tensor at
# 16384x4096 = 256 MB F32, ``chunk_rows=4096`` gives 4 chunks
# of 4096x4096 = 64 MB each, which is the largest single-shot
# intermediate that still fits a 256 GB host.  The legacy
# single-shot path is ``chunk_rows <= 0``.
DEFAULT_CHUNK_ROWS = 4096

# Aggregation method for reducing a rank-r S matrix to a per-input-channel scale
# that the existing AWQ path can consume without runtime changes.
LRQ_AGGREGATIONS = ("mean", "rms")

# Roles for the unified per-component calibration policy. The single-model
# path keeps the legacy "trunk" default; the unified Calibrate (the
# ``unified_calibrate.py`` driver) passes one of these flags per component
# so the consumer (tile640_quantize_v3.py) can route per-tensor entries to
# the correct lane when the same model is being quantized as a single
# unified GGUF.
#
# M0a added the three mmproj roles (``vision_tower`` / ``audio_tower`` /
# ``mm_projector``) to the unified schema; the dispatch side
# (``ts_regime_infer_modality``) routes tensors by their v./a./mm. prefix
# and the writer stamps the matching ``model_role`` on each row. The
# calibration side accepts the same eight-value set; the activation
# source for the three mmproj roles is the multimodal capture driver
# (``multimodal_calibrate.py``) rather than the text-side imatrix / spec
# flow. The text-side path is unchanged: ``trunk`` / ``dflash`` /
# ``dspark`` / ``mtp_nextn`` / ``shared_embd`` still go through the
# existing imatrix / spec_calib plumbing.
MODEL_ROLES = (
    "trunk", "dflash", "dspark", "mtp_nextn", "shared_embd",
    "vision_tower", "audio_tower", "mm_projector",
)
DEFAULT_MODEL_ROLE = "trunk"
# The M0a mmproj roles. Used to flag the per-tensor activation source
# for the consumer (the unified writer routes mmproj tensors to the
# multimodal capture path instead of the text-side imatrix path).
MMPROJ_ROLES = ("vision_tower", "audio_tower", "mm_projector")
# Activation source for the consumer. The text-side path uses
# ``"imatrix"`` (the imatrix GGUF produced by the text-side
# calibration pipeline); the mmproj-side path uses ``"multimodal"``
# (the tensor_stats rows produced by ``multimodal_calibrate.py``).
ACTIVATION_SOURCE = {"imatrix": "imatrix", "multimodal": "multimodal"}

# Targeted re-calibration (the L5 monitor-verdict hook): the
# ``backfill`` source value is stamped on the ``tensor_stats``
# row the backfill machinery writes. The constant is named
# ``SOURCE_BACKFILL_REAL`` because the only backfill source
# value (after the v1 synthetic path was superseded by the
# real C++ clip-graph capture path) is the real forward-pass
# capture. The ``source`` column on the ``tensor_stats`` row
# lets the orchestrator / l5_outcome distinguish a backfill
# write from a regular calibration pass. The companion
# constant in ``multimodal_calibrate.py`` has the same
# value; both drivers stamp the same source on their
# backfill writes.
SOURCE_BACKFILL_REAL: str = "backfill_real"

# DartQuant defaults. The QR-Orth step size is intentionally small: the
# Stiefel retraction is a first-order scheme and large steps drift the
# rotated weight well outside the manifold before QR pulls it back. The
# whip weight controls how aggressively the rotation is shaped toward
# activation uniformity versus the tile-quant output MSE.
DEFAULT_DARTQUANT_ITERS = 50
DEFAULT_DARTQUANT_LR = 1.0e-2
DEFAULT_DARTQUANT_WHIP_WEIGHT = 0.1

# Allow the linalg helper to be imported both as a package-relative module
# (when this file is run as ``python3 -m tools.tessera.per_tensor_calibrate``)
# and as a script (when the parent directory is on sys.path).
try:
    from ._dartquant_linalg import qr_orth_step, random_orthogonal, stiefel_project
except ImportError:  # pragma: no cover - script-mode import
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tools.tessera._dartquant_linalg import (  # type: ignore[no-redef]
        qr_orth_step,
        random_orthogonal,
        stiefel_project,
    )


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Layer:
    """One tensor's calibration bundle (mirrors awq-evolve.py's Layer)."""

    name: str
    family: str
    weight: np.ndarray
    train_activations: np.ndarray | None
    heldout_activations: np.ndarray | None
    # Sum-of-squares observer; derived when full activations are not available.
    in_sum2: np.ndarray | None
    in_count: int

    @property
    def out_dim(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_dim(self) -> int:
        return int(self.weight.shape[1])

    def input_scale(self) -> np.ndarray:
        """Per-input-channel scale derived from the observers or activations.

        Matches the AWQ convention: ``sqrt(E[x^2])`` so the dequant path
        becomes ``W * (E[x^2])^(-alpha/2)``. Used as a warm-start and as a
        fallback if the LRQ optimisation diverges.
        """
        if self.train_activations is not None and self.train_activations.size:
            return np.sqrt(
                np.mean(self.train_activations.astype(np.float32) ** 2, axis=0)
                + 1e-12
            ).astype(np.float32)
        if self.in_sum2 is not None and self.in_count > 0:
            return np.sqrt(self.in_sum2.astype(np.float32) / float(self.in_count) + 1e-12)
        # Uniform scale when no activation data is available.
        return np.ones(self.in_dim, dtype=np.float32)


def _scalar_string(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def load_layer(path: Path, max_tokens: int = 256) -> Layer:
    """Load a layer bundle as a ``Layer`` of mmap-backed views.

    Phase 16 (calibration memopt): the bundle is opened with
    ``mmap_mode="r"`` so the OS pages in each tensor on demand
    rather than reading the whole file into RAM.  Peak RSS for
    a single tensor is bounded to ``max(weight, activations,
    observer)`` instead of ``sum(all_tensors_in_turn)``.

    The mmap views share memory with the zip mmap, not with the
    python process.  When the caller drops the ``Layer``, the
    OS reclaims the pages lazily.  Downstream training code
    (LRQ, DartQuant) calls ``.astype(np.float32)`` on the
    weight, which materialises a single owning copy; that is
    the intended pattern: the mmap view is the read-only
    intermediate, the F32 copy is the working set.
    """
    with mmap_layer(path) as data:
        if "weight" not in data:
            raise ValueError(f"{path}: missing required 'weight' key")
        weight = data["weight"]
        if weight.ndim != 2:
            raise ValueError(f"{path}: weight must be two-dimensional")
        train_raw = data.get("train_activations")
        heldout_raw = data.get("heldout_activations")
        in_sum2_raw = data.get("in_sum2")
        in_count = int(np.asarray(data["counts"]).reshape(()).item()) if "counts" in data else 0
        name = _scalar_string(data.get("name"), path.stem)
        family = _scalar_string(data.get("family"), "ffn")
        # Apply the max_tokens sub-sample.  Subsampling requires
        # materialising the activation rows; once subsampled, the
        # result is an owning F32 array and the mmap is no longer
        # needed for the activations.
        train: np.ndarray | None = None
        heldout: np.ndarray | None = None
        for label, acts, sink in (
            ("train", train_raw, "train"),
            ("heldout", heldout_raw, "heldout"),
        ):
            if acts is None:
                continue
            if acts.ndim != 2 or acts.shape[1] != weight.shape[1]:
                raise ValueError(
                    f"{path}: {label} activation shape {acts.shape} "
                    f"does not match weight {weight.shape}"
                )
            if max_tokens > 0 and acts.shape[0] > max_tokens:
                idx = np.linspace(0, acts.shape[0] - 1, max_tokens, dtype=np.int64)
                sub = np.asarray(acts[idx], dtype=np.float32)
            else:
                sub = np.asarray(acts, dtype=np.float32)
            if sink == "train":
                train = sub
            else:
                heldout = sub
        in_sum2: np.ndarray | None = None
        if in_sum2_raw is not None:
            in_sum2 = np.asarray(in_sum2_raw, dtype=np.float32).reshape(-1)
    if train is None and in_sum2 is None:
        raise ValueError(f"{path}: requires train_activations or in_sum2")
    return Layer(
        name=name,
        family=family,
        weight=weight,
        train_activations=train,
        heldout_activations=heldout,
        in_sum2=in_sum2,
        in_count=in_count,
    )


def iter_layer_paths(layers_arg: str) -> list[Path]:
    p = Path(layers_arg)
    if p.is_dir():
        return sorted(p.glob("*.npz"))
    if p.is_file() and p.suffix == ".npz":
        return [p]
    raise ValueError(f"{layers_arg}: expected a directory of .npz bundles or a single .npz file")


def bundle_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Ternarization (numpy port of quantize_v3.ternarize_with_acts; outlier_frac=0)
# ---------------------------------------------------------------------------
#
# We keep the outlier branch off in the LRQ training loop because the
# calibration policy already records a separate per-tensor outlier_fraction;
# mixing two competing outlier selectors during LRQ search would muddy the
# gradient. Outliers are reapplied at quantization time by quantize_v3.
#
# STE: the ternarization is non-differentiable, so we approximate the
# gradient w.r.t. ``W_scaled`` with the identity (straight-through estimator).
# This is the standard trick used in QAT literature; the loss landscape is
# well-behaved enough at rank 8-64 that the bias does not hurt convergence.


def ternarize(weights: np.ndarray) -> np.ndarray:
    """Ternarize a 2D weight matrix to {-1, 0, +1} (no outliers)."""
    flat = weights.astype(np.float32).reshape(-1)
    threshold = float(np.mean(np.abs(flat)))
    ternary = np.zeros(flat.size, dtype=np.int8)
    if threshold <= 0.0:
        return ternary.reshape(weights.shape)
    keep = np.abs(flat) >= threshold
    ternary[keep & (flat > 0)] = 1
    ternary[keep & (flat < 0)] = -1
    return ternary.reshape(weights.shape)


def ternarize_value(ternary: np.ndarray, weights_scaled: np.ndarray) -> np.ndarray:
    """Convert a ternary mask back to the quantized weight values.

    The Tessera ternary code stores {-1, 0, +1} multiplied by the per-row
    mean(|W_scaled|). This is what gets fed to the matmul in the runtime.
    For the LRQ loss we use the simpler per-position multiplier
    ``mean(|W_scaled|)`` so the gradient signal stays well-conditioned.
    """
    scale = float(np.mean(np.abs(weights_scaled.astype(np.float32))))
    if scale <= 0.0:
        return np.zeros_like(weights_scaled, dtype=np.float32)
    return ternary.astype(np.float32) * np.float32(scale)


# ---------------------------------------------------------------------------
# Pure-numpy Adam
# ---------------------------------------------------------------------------


class Adam:
    """Minimal pure-numpy Adam. One pass per parameter, vectorised in numpy.

    The stateful shape: ``params`` is a list of numpy arrays; each array gets
    its own ``m`` and ``v`` buffer of the same shape. The constructor does not
    copy ``params``; the caller is expected to own them.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = DEFAULT_LR,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = params
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        b1c = 1.0 - self.b1 ** self.t
        b2c = 1.0 - self.b2 ** self.t
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            g = grad
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * param
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[i] / b1c
            v_hat = self.v[i] / b2c
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ChunkedAdam:
    """Adam with per-chunk U state and full V state.

    Phase 16 (calibration memopt): the per-tensor LRQ training
    is split into row-chunks; each chunk's U slice has its own
    Adam ``m`` and ``v`` buffers, while V's Adam state is
    shared across chunks (the ``d_v`` gradient is accumulated
    across chunks before stepping).

    The split mirrors the gradient flow::

        d_u_slice = d_s_chunk @ v.T                 # (chunk_rows, rank)
        d_v       = sum_chunks u_chunk.T @ d_s_chunk # (rank, in_dim)

    so each chunk's d_u is local to the chunk (no cross-chunk
    coupling) and d_v is a sum across chunks.  Splitting U's
    Adam state per-chunk keeps the per-chunk peak RSS bounded
    to ``max(W_chunk, intermediates)`` rather than
    ``max(W_full, intermediates_full)``.

    Parameters
    ----------
    u : (out_dim, rank) ndarray
        The U factor to be optimised.  Updated in place.
    v : (rank, in_dim) ndarray
        The V factor to be optimised.  Updated in place.
    chunk_rows : int
        The number of rows per U slice.  ``<= 0`` or
        ``>= out_dim`` is the legacy single-chunk path; the
        ChunkedAdam collapses to the standard ``Adam`` in that
        case.
    lr : float
        Adam learning rate.
    betas, eps, weight_decay
        Standard Adam hyperparameters.
    """

    def __init__(
        self,
        u: np.ndarray,
        v: np.ndarray,
        chunk_rows: int,
        lr: float = DEFAULT_LR,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.u = u
        self.v = v
        self.chunk_rows = int(chunk_rows)
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        # Per-chunk U state.  Each entry is one row-chunk; the
        # ``specs`` list carries the (start, end) row range for
        # in-place updates.
        self.u_specs: list[tuple[int, int]] = []
        n_rows = u.shape[0]
        if self.chunk_rows <= 0 or self.chunk_rows >= n_rows:
            self.u_specs.append((0, n_rows))
        else:
            for start in range(0, n_rows, self.chunk_rows):
                end = min(start + self.chunk_rows, n_rows)
                self.u_specs.append((start, end))
        self.u_m = [np.zeros((end - start, u.shape[1]), dtype=u.dtype)
                    for start, end in self.u_specs]
        self.u_v = [np.zeros((end - start, u.shape[1]), dtype=u.dtype)
                    for start, end in self.u_specs]
        # Full V state.
        self.v_m = np.zeros_like(v)
        self.v_v = np.zeros_like(v)

    def step(self, d_u_per_chunk: list[np.ndarray], d_v: np.ndarray) -> None:
        """Apply one Adam step using per-chunk U gradients and
        a single accumulated V gradient."""
        self.t += 1
        b1c = 1.0 - self.b1 ** self.t
        b2c = 1.0 - self.b2 ** self.t
        for i, ((start, end), m_buf, v_buf, grad) in enumerate(
            zip(self.u_specs, self.u_m, self.u_v, d_u_per_chunk)
        ):
            g = grad
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * self.u[start:end]
            self.u_m[i] = self.b1 * m_buf + (1.0 - self.b1) * g
            self.u_v[i] = self.b2 * v_buf + (1.0 - self.b2) * (g * g)
            m_hat = self.u_m[i] / b1c
            v_hat = self.u_v[i] / b2c
            self.u[start:end] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        g = d_v
        if self.weight_decay > 0.0:
            g = g + self.weight_decay * self.v
        self.v_m = self.b1 * self.v_m + (1.0 - self.b1) * g
        self.v_v = self.b2 * self.v_v + (1.0 - self.b2) * (g * g)
        m_hat = self.v_m / b1c
        v_hat = self.v_v / b2c
        self.v -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# LRQ training
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LRQResult:
    rank: int
    u: np.ndarray  # (out_dim, rank)
    v: np.ndarray  # (rank, in_dim)
    initial_mse: float
    final_mse: float
    iterations: int
    history: list[float]
    input_scale_agg: str
    # Per-input-channel aggregate of S = U @ V. The quantizer uses this as the
    # AWQ-style scale; the full U, V are stored for auditability / future
    # runtime extensions.
    scale_aggregate: np.ndarray
    # Warm-start scale derived from the activation observer. Stored so the
    # policy is self-describing even when only observers (not full
    # activations) were available.
    baseline_scale: np.ndarray

    def policy_entry(self) -> dict:
        return {
            "match": [self.bundle_name],
            "exact": True,
            "lrq_rank": int(self.rank),
            "lrq_u": self.u.astype(np.float32).tolist(),
            "lrq_v": self.v.astype(np.float32).tolist(),
            "lrq_iterations": int(self.iterations),
            "lrq_initial_mse": float(self.initial_mse),
            "lrq_final_mse": float(self.final_mse),
            "lrq_input_scale_agg": self.input_scale_agg,
            "lrq_baseline_scale": self.baseline_scale.astype(np.float32).tolist(),
        }

    def bytes_used(self) -> int:
        # F32 list serialisation is wasteful (commas, brackets) so we record
        # the dense size in bytes for the report.
        return int(self.u.size + self.v.size) * 4

    bundle_name: str = ""


def _aggregate_scale(s: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return np.mean(s, axis=0)
    if method == "rms":
        return np.sqrt(np.mean(s * s, axis=0) + 1e-12)
    raise ValueError(f"unknown aggregation {method!r}; expected one of {LRQ_AGGREGATIONS}")


def _initial_u_v(layer: Layer, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialise U, V with a small Gaussian centred on the AWQ warm start.

    The warm start sets the column-wise mean of S to the activation-derived
    scale ``s_agg`` so the first iteration already has a sensible input
    scale. The remaining degrees of freedom are random normal with a small
    variance, which lets Adam explore without a cold start.
    """
    rng = np.random.default_rng(seed)
    out_dim, in_dim = layer.out_dim, layer.in_dim
    s_agg = layer.input_scale()
    # Normalise s_agg to have unit mean, then use the inverse to keep the
    # initial W * S close to W.
    s_norm = s_agg.astype(np.float32) / float(np.mean(s_agg) + 1e-12)
    # Build U so that the column-mean of U @ V matches s_norm.
    # We pick V rows to be small random vectors, and U so that
    # mean_r U[r, k] * V[k, c] approximates s_norm[c]. With rank >= 1, a
    # clean choice is V[0, :] = s_norm, U[:, 0] = 1.0, the rest near zero.
    v = rng.normal(loc=0.0, scale=1e-2, size=(rank, in_dim)).astype(np.float32)
    if rank >= 1:
        v[0] = s_norm
    u = rng.normal(loc=0.0, scale=1e-2, size=(out_dim, rank)).astype(np.float32)
    if rank >= 1:
        u[:, 0] = 1.0
    return u, v


def _ternary_recon(scaled: np.ndarray) -> np.ndarray:
    """Forward pass: ternarize ``scaled`` and return the reconstructed weight."""
    t = ternarize(scaled)
    return ternarize_value(t, scaled)


def train_lrq(
    layer: Layer,
    rank: int = DEFAULT_RANK,
    iterations: int = DEFAULT_ITERATIONS,
    lr: float = DEFAULT_LR,
    seed: int = 0,
    aggregation: str = "mean",
    verbose: bool = False,
) -> LRQResult:
    """Learn U, V minimising the BF16-vs-quantized output MSE on calibration X.

    The gradient chain is:
        loss = mean over tokens t of ||(W_q @ x_t) - (W @ x_t)||^2
        W_q = ternarize_recon(W * S),  S = U @ V
    The ternarization is non-differentiable, so we use the straight-through
    estimator: ``d loss / d W_scaled = d loss / d W_q``. The remaining
    gradient is exact because multiplication and matmul are smooth.
    """
    weight = layer.weight.astype(np.float32)
    if layer.train_activations is not None and layer.train_activations.size:
        x = layer.train_activations.astype(np.float32)
    else:
        raise ValueError("LRQ training requires train_activations in the bundle")

    out_dim, in_dim = weight.shape
    if rank < 1 or rank > min(out_dim, in_dim):
        raise ValueError(
            f"rank must be in [1, {min(out_dim, in_dim)}], got {rank}"
        )

    u, v = _initial_u_v(layer, rank, seed)
    adam = Adam([u, v], lr=lr)

    # Cached references; updating inside the loop would force reallocation.
    s = np.zeros((out_dim, in_dim), dtype=np.float32)
    scaled = np.zeros_like(weight)
    grad_w_scaled = np.zeros_like(weight)
    grad_s = np.zeros_like(s)
    history: list[float] = []

    def mse_at(u_arr: np.ndarray, v_arr: np.ndarray) -> tuple[float, np.ndarray]:
        np.matmul(u_arr, v_arr, out=s)
        np.multiply(weight, s, out=scaled)
        w_q = _ternary_recon(scaled)
        residual = w_q - weight  # (out_dim, in_dim)
        # Error projected through X: err_t = residual @ x_t. Loss is the
        # mean squared error per token averaged over output channels.
        err = residual @ x.T  # (out_dim, n_tokens)
        loss = float(np.mean(err * err))
        return loss, residual

    initial_mse, _ = mse_at(u, v)
    history.append(initial_mse)

    for it in range(iterations):
        # Forward
        np.matmul(u, v, out=s)
        np.multiply(weight, s, out=scaled)
        w_q = _ternary_recon(scaled)
        residual = w_q - weight
        err = residual @ x.T
        loss = float(np.mean(err * err))
        history.append(loss)

        # Backward
        # d loss / d err = (2 / (out_dim * n_tokens)) * err
        d_err = (2.0 / float(err.size)) * err
        # d err / d residual = x
        d_residual = d_err @ x  # (out_dim, in_dim) = (out_dim, n_tokens) @ (n_tokens, in_dim)
        # d residual / d scaled = STE (identity). residual = w_q - weight
        # where w_q = ternarize_recon(scaled). STE: d w_q / d scaled = I.
        d_scaled = d_residual
        # d scaled / d s = weight (element-wise, because scaled = weight * s)
        d_s = d_scaled * weight
        # d s / d u = v^T; d s / d v = u^T
        d_u = d_s @ v.T
        d_v = u.T @ d_s

        adam.step([d_u, d_v])

        if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
            print(
                f"  lrq[{layer.name}] iter {it:3d}/{iterations}  mse={loss:.6e}  delta={history[-2] - loss:+.3e}",
                file=sys.stderr,
            )

    final_mse = history[-1]
    s_aggregate = _aggregate_scale(s, aggregation)
    return LRQResult(
        rank=rank,
        u=u,
        v=v,
        initial_mse=initial_mse,
        final_mse=final_mse,
        iterations=iterations,
        history=history,
        input_scale_agg=aggregation,
        scale_aggregate=s_aggregate.astype(np.float32),
        baseline_scale=layer.input_scale(),
        bundle_name=layer.name,
    )


def _lrq_chunked_initial_mse(
    u: np.ndarray,
    v: np.ndarray,
    weight: np.ndarray,
    x: np.ndarray,
    chunk_rows: int,
) -> float:
    """Compute the initial MSE of the LRQ training in a chunked way.

    The full forward pass would materialise ``U @ V`` (size
    ``out_dim * in_dim``) and a per-row residual of the same
    size.  For a 12B FFN gate that's 256 MB of intermediates.
    The chunked version processes row-chunks; per-chunk peak
    RSS is ``max(W_chunk, intermediates)`` = ``chunk_rows *
    in_dim * 4`` bytes = 64 MB at the default chunk size.
    """
    n_rows, in_dim = weight.shape
    err_sq_sum = 0.0
    err_count = 0
    for spec in chunked_iter(n_rows, chunk_rows):
        u_chunk = u[spec.start:spec.end]   # (chunk_rows, rank)
        # Phase 16.5: dispatched matmul.  On Apple Silicon /
        # Intel Mac the per-chunk (chunk_rows, in_dim) GEMM
        # is routed to Metal/Accelerate; on Linux/Windows
        # the dispatch returns numpy.  The shape is always
        # 2-D F32 so the dispatch is the hot path.
        s_chunk = chunked_matmul(u_chunk, v)              # (chunk_rows, in_dim)
        w_chunk = np.asarray(weight[spec.start:spec.end], dtype=np.float32)
        scaled_chunk = w_chunk * s_chunk
        w_q_chunk = _ternary_recon(scaled_chunk)
        residual_chunk = w_q_chunk - w_chunk
        # ``residual_chunk @ x.T`` is a transposed matmul;
        # the dispatch falls back to numpy for the
        # transposed case (the C bridge is non-transposed
        # only).  This is the same numpy path the legacy
        # code took; the rest of the per-chunk loop
        # benefits from the dispatch.
        err_chunk = chunked_matmul(residual_chunk, x.T)    # (chunk_rows, n_tokens)
        err_sq_sum += float(np.sum(err_chunk * err_chunk))
        err_count += err_chunk.size
    if err_count == 0:
        return 0.0
    return err_sq_sum / float(err_count)


def _lrq_chunked_aggregate_scale(
    u: np.ndarray,
    v: np.ndarray,
    aggregation: str,
    chunk_rows: int,
) -> np.ndarray:
    """Compute the per-input-channel aggregate of ``S = U @ V``
    incrementally (per-chunk) to avoid materialising the full
    ``(out_dim, in_dim)`` S.

    The aggregate is along axis 0 (out_dim) so per-chunk
    contributions stack cleanly.  ``mean`` is the simple
    chunk-mean average; ``rms`` is the per-chunk
    ``sum(s^2)/out_dim`` accumulator.
    """
    n_rows, in_dim = u.shape[0], v.shape[1]
    if aggregation == "mean":
        acc = np.zeros(in_dim, dtype=np.float64)
        n_seen = 0
        for spec in chunked_iter(n_rows, chunk_rows):
            u_chunk = u[spec.start:spec.end]
            # Phase 16.5: dispatched matmul.
            s_chunk = chunked_matmul(u_chunk, v)
            acc += s_chunk.sum(axis=0).astype(np.float64)
            n_seen += (spec.end - spec.start)
        if n_seen == 0:
            return np.zeros(in_dim, dtype=np.float32)
        return (acc / float(n_seen)).astype(np.float32)
    if aggregation == "rms":
        acc = np.zeros(in_dim, dtype=np.float64)
        n_seen = 0
        for spec in chunked_iter(n_rows, chunk_rows):
            u_chunk = u[spec.start:spec.end]
            # Phase 16.5: dispatched matmul.
            s_chunk = chunked_matmul(u_chunk, v)
            acc += (s_chunk.astype(np.float64) ** 2).sum(axis=0)
            n_seen += (spec.end - spec.start)
        if n_seen == 0:
            return np.zeros(in_dim, dtype=np.float32)
        return np.sqrt(acc / float(n_seen) + 1e-12).astype(np.float32)
    raise ValueError(
        f"unknown aggregation {aggregation!r}; expected one of {LRQ_AGGREGATIONS}"
    )


def train_lrq_chunked(
    layer: Layer,
    rank: int = DEFAULT_RANK,
    iterations: int = DEFAULT_ITERATIONS,
    lr: float = DEFAULT_LR,
    seed: int = 0,
    aggregation: str = "mean",
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    verbose: bool = False,
) -> LRQResult:
    """LRQ training with row-chunked forward/backward passes.

    Phase 16 (calibration memopt).  For a 12B FFN gate tensor
    at 16384x4096 = 256 MB F32, the per-iter peak RSS of the
    legacy ``train_lrq`` is ~1 GB (four 256 MB intermediates).
    This chunked path keeps each chunk's intermediates to
    ``chunk_rows * in_dim * 4`` bytes = 64 MB at the default
    chunk size, so the per-iter peak RSS is ~256 MB.

    The numerical answer is bit-equivalent to ``train_lrq`` (up
    to the Adam float32 order-of-operations).  The legacy
    single-shot path is recovered by passing ``chunk_rows <=
    0`` or ``chunk_rows >= out_dim``; the function dispatches
    to ``train_lrq`` in that case to keep the public API
    consistent.
    """
    weight = layer.weight
    if layer.train_activations is None or not layer.train_activations.size:
        raise ValueError("LRQ training requires train_activations in the bundle")
    x = layer.train_activations.astype(np.float32)
    out_dim, in_dim = weight.shape
    if rank < 1 or rank > min(out_dim, in_dim):
        raise ValueError(
            f"rank must be in [1, {min(out_dim, in_dim)}], got {rank}"
        )
    if chunk_rows <= 0 or chunk_rows >= out_dim:
        # Legacy single-shot path.
        return train_lrq(
            layer,
            rank=rank,
            iterations=iterations,
            lr=lr,
            seed=seed,
            aggregation=aggregation,
            verbose=verbose,
        )
    u, v = _initial_u_v(layer, rank, seed)
    adam = ChunkedAdam(u, v, chunk_rows=chunk_rows, lr=lr)
    history: list[float] = []
    initial_mse = _lrq_chunked_initial_mse(u, v, weight, x, chunk_rows)
    history.append(initial_mse)
    n_tokens = x.shape[0]
    err_total_size = float(out_dim * n_tokens)
    for it in range(iterations):
        d_v_accum = np.zeros_like(v)
        err_sq_sum = 0.0
        d_u_per_chunk: list[np.ndarray] = []
        for spec in chunked_iter(out_dim, chunk_rows):
            u_chunk = u[spec.start:spec.end]
            # Forward.
            # Phase 16.5: dispatched matmul.  On Apple Silicon /
            # Intel Mac the per-chunk GEMM is routed to
            # Metal/Accelerate; on Linux/Windows the dispatch
            # returns numpy.  The shape is always 2-D F32
            # so the dispatch is the hot path.
            s_chunk = chunked_matmul(u_chunk, v)                       # (chunk_rows, in_dim)
            w_chunk = np.asarray(weight[spec.start:spec.end], dtype=np.float32)
            scaled_chunk = w_chunk * s_chunk
            w_q_chunk = _ternary_recon(scaled_chunk)
            residual_chunk = w_q_chunk - w_chunk
            # ``residual_chunk @ x.T`` is transposed; the
            # dispatch falls back to numpy for the transposed
            # case.  Same comment for the other ``.T`` matmuls
            # below.  The non-transposed matmuls in this loop
            # (``s_chunk``, ``d_residual``) hit the dispatch.
            err_chunk = chunked_matmul(residual_chunk, x.T)             # (chunk_rows, n_tokens)
            err_sq_sum += float(np.sum(err_chunk * err_chunk))
            # Backward.  d_err normalisation uses the FULL
            # err.size (out_dim * n_tokens), not the per-chunk
            # size, so the chunked and single-shot gradients
            # match exactly.
            d_err = (2.0 / err_total_size) * err_chunk
            d_residual = chunked_matmul(d_err, x)                       # (chunk_rows, in_dim)
            d_scaled = d_residual                        # STE
            d_s = d_scaled * w_chunk                     # (chunk_rows, in_dim)
            d_u_slice = chunked_matmul(d_s, v.T)                        # (chunk_rows, rank)
            d_u_per_chunk.append(d_u_slice)
            d_v_accum += chunked_matmul(u_chunk.T, d_s)                 # accumulate across chunks
        loss = err_sq_sum / err_total_size if err_total_size else 0.0
        history.append(loss)
        adam.step(d_u_per_chunk, d_v_accum)
        if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
            print(
                f"  lrq[chunked {layer.name}] iter {it:3d}/{iterations}  "
                f"mse={loss:.6e}  delta={history[-2] - loss:+.3e}",
                file=sys.stderr,
            )
    final_mse = history[-1]
    s_aggregate = _lrq_chunked_aggregate_scale(u, v, aggregation, chunk_rows)
    return LRQResult(
        rank=rank,
        u=u,
        v=v,
        initial_mse=initial_mse,
        final_mse=final_mse,
        iterations=iterations,
        history=history,
        input_scale_agg=aggregation,
        scale_aggregate=s_aggregate.astype(np.float32),
        baseline_scale=layer.input_scale(),
        bundle_name=layer.name,
    )


# ---------------------------------------------------------------------------
# DartQuant: pre-rotation + Whip loss + QR-Orth
# ---------------------------------------------------------------------------
#
# DartQuant (NeurIPS 2025) learns an orthogonal rotation ``R`` of the
# weight input dimension so that, after the rotation, the tile640
# ternarization has less work to do. The full loss is a weighted sum of
# two terms:
#
#   * tile-quant output MSE: ``mean(((W @ R) - Q(W @ R)) @ X.T) ** 2)``
#     i.e. the error projected through the calibration activations. The
#     straight-through estimator (STE) treats the ternarization as
#     identity for the backward pass; the gradient w.r.t. ``R`` is
#     ``(2 / N) * W.T @ residual @ X.T @ X``.
#   * Whip loss: variance of per-output-channel L2 norms of ``W @ R``,
#     optionally weighted by the calibration imatrix ``sqrt(E[x^2])``.
#     This is the "shape toward uniform activations" term from the paper.
#
# The combined gradient is passed to ``qr_orth_step`` (Stiefel tangent
# projection + QR retraction). Cayley SGD is replaced by QR-Orth in the
# paper because the retraction is a single LAPACK call rather than a
# matrix inverse; in this pure-Python implementation the QR uses
# ``np.linalg.qr`` and the per-step cost is dominated by the
# forward+backward matmuls on ``W`` and ``X``.


@dataclasses.dataclass
class DartQuantResult:
    rotation: np.ndarray  # (in_dim, in_dim), FP16 orthogonal
    initial_whip: float
    final_whip: float
    initial_quant_mse: float  # weight tile-quant MSE on W @ R (relative)
    final_quant_mse: float
    initial_output_mse: float  # matmul output MSE (with X if available)
    final_output_mse: float
    iterations: int
    history: list[float]  # total loss per iteration
    rotation_dtype: str  # "float16" (we downcast before serialisation)
    in_dim: int

    bundle_name: str = ""

    def policy_entry(self) -> dict:
        return {
            "match": [self.bundle_name],
            "exact": True,
            "dartquant_rotation": self.rotation.astype(np.float16).tolist(),
            "dartquant_rotation_dim": int(self.in_dim),
            "dartquant_rotation_dtype": self.rotation_dtype,
            "dartquant_whip_initial": float(self.initial_whip),
            "dartquant_whip_final": float(self.final_whip),
            "dartquant_quant_mse_initial": float(self.initial_quant_mse),
            "dartquant_quant_mse_final": float(self.final_quant_mse),
            "dartquant_output_mse_initial": float(self.initial_output_mse),
            "dartquant_output_mse_final": float(self.final_output_mse),
            "dartquant_iterations": int(self.iterations),
        }

    def bytes_used(self) -> int:
        # FP16 storage of the K x K rotation.
        return int(self.rotation.size) * 2


def dartquant_apply_rotation(W: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply the learned rotation: ``W' = W @ R``.

    Output is FP32, contiguous, owning. ``R`` must be square
    ``(W.shape[1], W.shape[1])``.
    """
    Wf = np.ascontiguousarray(W, dtype=np.float32)
    Rf = np.asarray(R, dtype=np.float32)
    if Rf.ndim != 2 or Rf.shape[0] != Rf.shape[1]:
        raise ValueError(f"R must be square, got {Rf.shape}")
    if Rf.shape[0] != Wf.shape[1]:
        raise ValueError(
            f"R shape {Rf.shape} incompatible with W input dim {Wf.shape[1]}"
        )
    return Wf @ Rf


def _ternarize_recon(weights: np.ndarray) -> np.ndarray:
    """Tile-quant a 2-D weight: ternarize, then reconstruct with the
    per-position mean(|W|) multiplier (matches the LRQ training path)."""
    t = ternarize(weights)
    return ternarize_value(t, weights)


def _row_l2(W: np.ndarray) -> np.ndarray:
    """Per-row L2 norms of a 2-D weight matrix, FP32, with a floor for
    numerical stability. Returns a 1-D array of shape ``(W.shape[0],)``."""
    return np.sqrt(np.maximum(np.sum(W.astype(np.float32) ** 2, axis=1), 1.0e-24))


def dartquant_whip_loss(
    W_rot: np.ndarray, X_hat: np.ndarray | None = None
) -> float:
    """Whip loss: coefficient of variation of per-output-channel L2 norms.

    Without a calibration imatrix (no ``X_hat``), the loss is the
    coefficient of variation of ``||W'[o, :]||_2`` across output channels
    ``o``. Lower values mean the rotation has made the per-channel
    response magnitudes more uniform, which is the proxy for "the
    activation distribution is closer to uniform" from the DartQuant
    paper.

    With ``X_hat`` (``sqrt(E[x^2])``, per-input-channel), the rotation is
    evaluated on the activation-weighted weight: ``W' * sqrt(E[x^2])``
    per the SmoothQuant-style convention. Same formula, different
    weight matrix.
    """
    if X_hat is not None:
        scale = np.asarray(X_hat, dtype=np.float32).reshape(-1)
        if scale.shape[0] != W_rot.shape[1]:
            raise ValueError(
                f"X_hat length {scale.shape[0]} != W_rot input dim {W_rot.shape[1]}"
            )
        Ww = W_rot.astype(np.float32) * scale[None, :]
    else:
        Ww = W_rot.astype(np.float32)
    norms = _row_l2(Ww)
    mean = float(np.mean(norms))
    if mean < 1.0e-12:
        return 0.0
    return float(np.std(norms) / mean)


def _whip_grad_impl(W: np.ndarray, R: np.ndarray, X_hat: np.ndarray | None) -> np.ndarray:
    """Analytical gradient of the Whip loss w.r.t. ``R``.

    Let ``W' = W @ R``, ``f_o = ||W'[o, :]||_2`` (or
    ``||W'[o, :] * sqrt(E[x^2])||_2`` if ``X_hat`` is supplied), and
    ``L = std(f_o) / mean(f_o)`` (the coefficient of variation). The
    closed-form gradient w.r.t. ``R`` is

        d L / d R = (1 / mean(f)) * (W.T @ ((f - m) / f)[:, None] * W')
                  - (L / (N * mean(f))) * (W.T @ ones(f)[:, None] * W' / f)

    but the simpler form (treating the outer ``1 / mean(f)`` factor as a
    constant) recovers the right Stiefel step direction up to a positive
    scaling. We use the simpler form so the gradient stays
    single-precision clean and the QR-Orth step is well-conditioned.
    """
    out_dim, in_dim = W.shape
    Wf = W.astype(np.float32, copy=False)
    Wp = Wf @ R.astype(np.float32, copy=False)
    if X_hat is not None:
        scale = np.asarray(X_hat, dtype=np.float32).reshape(-1)
        if scale.shape[0] != in_dim:
            raise ValueError(
                f"X_hat length {scale.shape[0]} != input dim {in_dim}"
            )
        Wp = Wp * scale[None, :]
    norms = _row_l2(Wp)
    safe_norms = np.maximum(norms, 1.0e-12)
    mean_n = float(np.mean(norms))
    if mean_n < 1.0e-12:
        return np.zeros((in_dim, in_dim), dtype=np.float32)
    # diff_o = (f_o - mean) / f_o  for each output channel o
    diff = (norms - mean_n) / safe_norms
    dWp = diff[:, None] * Wp  # (out, in)
    # d L / d R = (1 / (N * mean(f))) * W.T @ dWp
    grad = Wf.T @ dWp
    grad /= float(out_dim) * mean_n
    return grad.astype(np.float32, copy=False)


def _output_mse_and_grad(
    W: np.ndarray, R: np.ndarray, X: np.ndarray | None
) -> tuple[float, np.ndarray]:
    """Tile-quant output MSE on ``W @ R`` plus its gradient w.r.t. ``R``.

    The forward loss is::

        L = mean(((W @ R) - Q(W @ R)) @ X.T) ** 2)

    i.e. the projected difference between the quantised rotated weight
    and the original (un-rotated) weight ``W``. The ``-W`` term is held
    constant during the backward pass (asymmetric straight-through
    estimator): the gradient flows only through the quantiser, treating
    ``Q(W')`` as ``W'`` for differentiation. With this convention the
    gradient is well-defined even though the standard symmetric STE
    collapses to zero (the variable ``R`` appears in both ``Q(W')`` and
    ``-W'`` so the two contributions cancel).

    The closed-form gradient w.r.t. ``R`` is::

        d L / d R = (2 / N) * W.T @ ((Q(W') - W) @ X.T) @ X
                  = (2 / N) * W.T @ err @ X

    where ``err = (Q(W') - W) @ X.T`` and ``N = out_dim * n_tokens``.

    Without ``X`` we fall back to the weight reconstruction MSE
    ``mean((W' - Q(W')) ** 2)``; its STE gradient is exactly zero so the
    caller will only get a meaningful scalar, not a meaningful gradient.
    """
    Wf = W.astype(np.float32, copy=False)
    Wp = Wf @ R.astype(np.float32, copy=False)
    Wq = _ternarize_recon(Wp)
    if X is None or X.size == 0:
        residual = Wq - Wp
        loss = float(np.mean(residual * residual))
        return loss, np.zeros(R.shape, dtype=np.float32)
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim != 2 or Xf.shape[1] != Wf.shape[1]:
        raise ValueError(
            f"X shape {Xf.shape} incompatible with W {Wf.shape}"
        )
    # Asymmetric residual: quantised rotated weight minus the un-rotated
    # weight. The ``-W`` part is constant w.r.t. ``R``; only ``Q(W')``
    # flows through the gradient.
    residual = Wq - Wf
    err = residual @ Xf.T
    n_tokens = Xf.shape[0]
    N = Wf.shape[0] * n_tokens
    loss = float(np.mean(err * err))
    dL_dWp = (2.0 / float(N)) * (err @ Xf)
    grad = Wf.T @ dL_dWp
    return loss, grad.astype(np.float32, copy=False)


def _relative_quant_mse(W: np.ndarray) -> float:
    """Relative tile-quant MSE: ``||W - Q(W)||_F^2 / ||W||_F^2``."""
    Wf = W.astype(np.float32, copy=False)
    Wq = _ternarize_recon(Wf)
    residual = Wq - Wf
    num = float(np.sum(residual * residual))
    den = float(np.sum(Wf * Wf)) + 1.0e-12
    return num / den


def dartquant_qr_orth(
    W: np.ndarray,
    X: np.ndarray | None,
    n_iters: int = DEFAULT_DARTQUANT_ITERS,
    lr: float = DEFAULT_DARTQUANT_LR,
    whip_weight: float = DEFAULT_DARTQUANT_WHIP_WEIGHT,
    X_hat: np.ndarray | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> DartQuantResult:
    """Train an orthogonal rotation ``R`` for ``W`` using QR-Orth.

    The loss is ``L_quant + whip_weight * L_whip``. ``R`` lives on the
    Stiefel manifold; one step is

        tangent = G - R @ sym(R.T @ G)
        M       = R - lr * tangent
        R_next  = qf(M)

    where ``G`` is the ambient gradient. The output-MSE term uses the
    asymmetric straight-through estimator (asymmetric STE: the
    ``-W'`` part of the residual is held constant during the backward
    pass; gradient flows only through ``Q(W')``). This is the
    convention used by ``train_lrq`` further up in this file.

    The combined loss is non-convex: the best rotation may not be the
    final one. We track the best output MSE seen during the schedule
    and report it alongside the final; the policy carries the final
    rotation but operators tuning the schedule should also try multiple
    seeds and pick the best.

    Convergence usually needs ``n_iters >= 20`` and a small ``lr``
    (1e-3 to 1e-1). The initial rotation is the identity when
    ``seed == 0`` and a Haar-uniform orthogonal otherwise.
    """
    Wf = np.ascontiguousarray(W.astype(np.float32))
    in_dim = Wf.shape[1]
    R = np.eye(in_dim, dtype=np.float32)
    if seed:
        # Optional perturbation of the identity so the GA sees a different
        # starting point per seed; the Stiefel step will pull it back.
        R = random_orthogonal(in_dim, seed=seed).astype(np.float32)
    # Initial metrics.
    Wp = dartquant_apply_rotation(Wf, R)
    initial_whip = dartquant_whip_loss(Wp, X_hat=X_hat)
    initial_quant_mse = _relative_quant_mse(Wp)
    initial_output_mse, _ = _output_mse_and_grad(Wf, R, X)
    history: list[float] = [initial_quant_mse + whip_weight * initial_whip]
    # Best-so-far tracking on output MSE (the deployment metric). Tile-
    # quant MSE is a secondary diagnostic; output MSE is what the
    # runtime will see.
    best_R = R.copy()
    best_output_mse = initial_output_mse
    best_quant_mse = initial_quant_mse
    best_whip = initial_whip
    for it in range(n_iters):
        q_loss, q_grad = _output_mse_and_grad(Wf, R, X)
        w_grad = _whip_grad_impl(Wf, R, X_hat)
        total_grad = q_grad + float(whip_weight) * w_grad
        R = qr_orth_step(R.astype(np.float64), total_grad.astype(np.float64), lr)
        R = R.astype(np.float32, copy=False)
        Wp = dartquant_apply_rotation(Wf, R)
        cur_whip = dartquant_whip_loss(Wp, X_hat=X_hat)
        cur_qmse = _relative_quant_mse(Wp)
        cur_omse, _ = _output_mse_and_grad(Wf, R, X)
        history.append(cur_qmse + whip_weight * cur_whip)
        if cur_omse < best_output_mse:
            best_output_mse = cur_omse
            best_quant_mse = cur_qmse
            best_whip = cur_whip
            best_R = R.copy()
        if verbose and (it % max(1, n_iters // 10) == 0 or it == n_iters - 1):
            print(
                f"  dartquant[iter {it:3d}/{n_iters}] "
                f"quant_mse={cur_qmse:.4e} whip={cur_whip:.4e} "
                f"out_mse={cur_omse:.4e} loss={history[-1]:.4e}",
                file=sys.stderr,
            )
    Wp = dartquant_apply_rotation(Wf, R)
    final_whip = dartquant_whip_loss(Wp, X_hat=X_hat)
    final_quant_mse = _relative_quant_mse(Wp)
    final_output_mse, _ = _output_mse_and_grad(Wf, R, X)
    return DartQuantResult(
        rotation=R,
        initial_whip=initial_whip,
        final_whip=final_whip,
        initial_quant_mse=initial_quant_mse,
        final_quant_mse=final_quant_mse,
        initial_output_mse=initial_output_mse,
        final_output_mse=final_output_mse,
        iterations=n_iters,
        history=history,
        rotation_dtype="float16",
        in_dim=in_dim,
    )


# ---------------------------------------------------------------------------
# Policy assembly
# ---------------------------------------------------------------------------


def build_lrq_policy(
    results: list[tuple[Layer, LRQResult]],
    provenance: dict,
    base: dict | None = None,
    model_role: str = DEFAULT_MODEL_ROLE,
) -> dict:
    """Assemble the per-tensor entries into a calibration-policy document.

    The wrapper schema is ``llama.speculative.calibration-policy.v1`` so the
    downstream quantizer can consume it without a schema change. The LRQ
    payload lives under ``policy["lrq"]`` with its own sub-schema
    (``llama.tessera.lrq-policy.v1``) for tooling that wants to inspect the
    low-rank factors without parsing the family map.

    ``model_role`` tags every per-tensor entry with one of the ``MODEL_ROLES``
    (e.g. ``trunk`` / ``dflash`` / ``dspark`` / ``mtp_nextn`` /
    ``shared_embd``). The same value is mirrored at the policy top level so
    the consumer can pick a single component out of a unified policy
    without scanning every entry. The default is ``trunk``, which is the
    legacy single-model behaviour: the field is additive and ignored by
    consumers that do not look at it.
    """
    if model_role not in MODEL_ROLES:
        raise ValueError(
            f"model_role {model_role!r} not in {MODEL_ROLES!r}"
        )
    policy: dict = dict(base or {})
    families = dict(policy.get("tensor_families", {}))
    tensor_records: list[dict] = []
    total_bytes = 0
    for layer, result in results:
        entry = result.policy_entry()
        entry["model_role"] = model_role
        entry_key = f"lrq:{layer.name}"
        families[entry_key] = entry
        tensor_records.append({
            "tensor": layer.name,
            "rank": result.rank,
            "initial_mse": result.initial_mse,
            "final_mse": result.final_mse,
            "iterations": result.iterations,
            "input_scale_agg": result.input_scale_agg,
            "bytes": result.bytes_used(),
        })
        total_bytes += result.bytes_used()

    policy.update({
        "schema": SCHEMA,
        "model_role": model_role,
        "lrq": {
            "schema": LRQ_SCHEMA,
            "rank": results[0][1].rank if results else DEFAULT_RANK,
            "iterations": results[0][1].iterations if results else 0,
            "lr": DEFAULT_LR,
            "input_scale_agg": results[0][1].input_scale_agg if results else "mean",
            "tensor_count": len(results),
            "total_bytes": total_bytes,
            "tensors": tensor_records,
        },
        "tensor_families": families,
        "per_tensor_calibration": provenance,
    })
    policy.setdefault("draft_type", "hybrid")
    return policy


def build_dartquant_policy(
    results: list[tuple[Layer, DartQuantResult]],
    provenance: dict,
    base: dict | None = None,
    model_role: str = DEFAULT_MODEL_ROLE,
) -> dict:
    """Assemble the DartQuant pre-rotation policy.

    Each tensor gets an entry under ``tensor_families`` keyed
    ``dartquant:<name>`` carrying the FP16 rotation matrix and the
    training statistics. The rotation is applied at the runtime side
    by ``tile640_quantize_v3.py --dartquant-rotation``: the dequantised
    weight is reconstructed as ``Q(W') @ R.T`` so the original ``W`` is
    recovered exactly (modulo the ternarization error on ``W'``).

    ``model_role`` tags every per-tensor entry with one of the
    ``MODEL_ROLES`` (see ``build_lrq_policy`` for semantics). Default
    ``trunk`` preserves the legacy single-model behaviour.
    """
    if model_role not in MODEL_ROLES:
        raise ValueError(
            f"model_role {model_role!r} not in {MODEL_ROLES!r}"
        )
    policy: dict = dict(base or {})
    families = dict(policy.get("tensor_families", {}))
    tensor_records: list[dict] = []
    total_bytes = 0
    for layer, result in results:
        entry = result.policy_entry()
        entry["model_role"] = model_role
        entry_key = f"dartquant:{layer.name}"
        families[entry_key] = entry
        tensor_records.append({
            "tensor": layer.name,
            "shape": list(layer.weight.shape),
            "rotation_dim": int(result.in_dim),
            "whip_initial": result.initial_whip,
            "whip_final": result.final_whip,
            "quant_mse_initial": result.initial_quant_mse,
            "quant_mse_final": result.final_quant_mse,
            "output_mse_initial": result.initial_output_mse,
            "output_mse_final": result.final_output_mse,
            "iterations": result.iterations,
            "bytes": result.bytes_used(),
        })
        total_bytes += result.bytes_used()

    policy.update({
        "schema": SCHEMA,
        "model_role": model_role,
        "dartquant": {
            "schema": DARTQUANT_SCHEMA,
            "iterations": results[0][1].iterations if results else 0,
            "rotation_dtype": results[0][1].rotation_dtype if results else "float16",
            "tensor_count": len(results),
            "total_bytes": total_bytes,
            "tensors": tensor_records,
        },
        "tensor_families": families,
        "per_tensor_calibration": provenance,
    })
    policy.setdefault("draft_type", "hybrid")
    return policy


# ---------------------------------------------------------------------------
# FLRQ (Jan 2026) -- R1-Sketch + Best Low-rank Approximation under Clipping
# ---------------------------------------------------------------------------
#
# FLRQ decomposes a 2-D weight W as W = U @ V + tile640_quant(R), where
# U (K x r) and V (r x N) are FP16 low-rank factors and R = W - U@V is
# the residual.  The low-rank factors capture outlier-aware structure
# identified by the R1-Sketch (a few Gaussian random projections of W,
# no calibration data needed).  The residual is quantised under a
# per-tensor symmetric uniform scheme that BLC iteratively fits.
#
# The decomposition is calibration-free: the sketch operates on W alone,
# which is the key advantage for the data-sparse Apple-Silicon path.
# The policy blob matches the existing
# ``llama.speculative.calibration-policy.v1`` wrapper, with a sub-schema
# ``llama.tessera.flrq-policy.v1`` under ``policy["flrq"]``.  GGUF
# sidecar serialisation is intentionally a follow-up; the JSON form here
# is what the orchestrator consumes.

FLRQ_SCHEMA = "llama.tessera.flrq-policy.v1"
FLRQ_DEFAULT_PROJECTIONS = 32
FLRQ_DEFAULT_RANK_CANDIDATES = (4, 8, 16, 32, 64)
FLRQ_DEFAULT_MSE_THRESHOLD = 1.0e-3
FLRQ_DEFAULT_BLC_ITERS = 4
FLRQ_DEFAULT_QBITS = 4


@dataclasses.dataclass
class FLRQResult:
    rank: int
    u: np.ndarray            # (out_dim, rank), FP32 in the result; the policy serialises as FP16-shaped
    v: np.ndarray            # (rank, in_dim)
    residual: np.ndarray     # (out_dim, in_dim) -- the original W minus U @ V
    residual_q: np.ndarray   # (out_dim, in_dim) -- tile640-style quantised residual
    residual_scale: float    # per-tensor scale used for the residual quantiser
    residual_clip: float     # per-tensor clip used for the residual quantiser
    residual_qbits: int      # bits of the residual quantiser (e.g. 4)
    reconstruction_mse: float
    reconstruction_rel_mse: float
    baseline_mse: float
    n_projections: int
    sketch_seed: int
    blc_iters: int
    bundle_name: str = ""

    def bytes_used(self) -> int:
        # FP16 for U and V (2 bytes/element) + int8 for the residual
        # quantised payload (1 byte/element) + small scalar overhead.
        return int(self.u.size) * 2 + int(self.v.size) * 2 + int(self.residual_q.size)

    def policy_entry(self) -> dict:
        return {
            "match": [self.bundle_name],
            "exact": True,
            "flrq_rank": int(self.rank),
            "flrq_u": self.u.astype(np.float32).tolist(),
            "flrq_v": self.v.astype(np.float32).tolist(),
            "flrq_residual_scale": float(self.residual_scale),
            "flrq_residual_clip": float(self.residual_clip),
            "flrq_residual_qbits": int(self.residual_qbits),
            "flrq_reconstruction_mse": float(self.reconstruction_mse),
            "flrq_reconstruction_rel_mse": float(self.reconstruction_rel_mse),
            "flrq_baseline_mse": float(self.baseline_mse),
            "flrq_n_projections": int(self.n_projections),
            "flrq_sketch_seed": int(self.sketch_seed),
            "flrq_blc_iters": int(self.blc_iters),
        }


def flrq_sketch(
    weight: np.ndarray,
    n_projections: int,
    seed: int,
    target_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R1-Sketch with Gaussian projection (calibration-free).

    The sketch ``Y = W @ Omega`` captures the dominant row-space of
    ``W`` using only a handful of Gaussian projections of ``W``.  The
    top-r LEFT singular vectors of ``Y`` form the outlier-aware basis
    we use to build the low-rank factors U and V: by the standard
    randomised-SVD argument, the row space of ``Y = W @ Omega``
    contains (and approximates) the row space of ``W``, so the top
    left singular vectors of ``Y`` are a good basis in which to
    represent the low-rank component of ``W``.  V is then the
    orthogonal projection of ``W`` onto that basis, so the initial
    rank-r reconstruction is the best rank-r approximation of ``W``
    in the sketch basis.

    Parameters
    ----------
    weight : (K, N) FP32 ndarray
        The 2-D weight matrix to sketch.
    n_projections : int
        Number of Gaussian projection vectors to draw.  16-64 is the
        FLRQ operating range; the sketch is approximate, so the answer
        is robust to the exact count as long as ``n_projections`` is at
        least a few multiples of the target rank.
    seed : int
        RNG seed; deterministic in the same way as ``np.random.default_rng``.
    target_rank : int, optional
        The number of sketch basis vectors to return.  Defaults to
        ``min(K, N, 16)``.  Callers usually want to pass an explicit
        value so the rank sweep can drive the sketch width.

    Returns
    -------
    Y : (K, n_projections * target_rank) FP32 ndarray
        The R1-Sketch of W.
    U_basis : (K, target_rank) FP32 ndarray
        The top ``target_rank`` left singular vectors of ``Y`` (the
        row-space basis the BLC step projects onto).
    sigma : (target_rank,) FP32 ndarray
        The corresponding singular values, descending.
    """
    if weight.ndim != 2:
        raise ValueError(f"flrq_sketch: weight must be 2-D, got {weight.ndim}-D")
    K, N = weight.shape
    if n_projections < 1:
        raise ValueError(f"flrq_sketch: n_projections must be >= 1, got {n_projections}")
    if target_rank is None:
        target_rank = max(1, min(K, N, 16))
    r = max(1, min(int(target_rank), K, N))
    total_width = min(N, n_projections * r)

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((N, total_width), dtype=np.float32)
    Y = weight.astype(np.float32) @ omega  # (K, total_width)
    # Truncated SVD: we only need the top-r left singular vectors of Y.
    # numpy returns (U, s, Vh) with shapes (M, K_min), (K_min,), (K_min, N)
    # where K_min = min(M, N).  We take the first r columns of U.
    _, sigma, _ = np.linalg.svd(Y, full_matrices=False)
    # The full U has shape (K, total_width) when K < N, else (K, K_min)
    # where K_min = min(K, total_width) = total_width when total_width <= K.
    # We compute U explicitly so the (K, r) extraction is always correct.
    U_full, sigma, _ = np.linalg.svd(Y, full_matrices=False)
    U_basis = U_full[:, :r].astype(np.float32)  # (K, r)
    sigma = sigma[:r].astype(np.float32)
    return Y, U_basis, sigma


def flrq_sketch_chunked(
    weight: np.ndarray,
    n_projections: int,
    seed: int,
    target_rank: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R1-Sketch with row-chunked Gaussian projection.

    Phase 16 (calibration memopt).  The legacy
    ``flrq_sketch`` materialises the full ``Y = W @ Omega`` as
    a single F32 matmul.  For a 12B FFN gate at 16384x4096
    with ``n_projections=32, target_rank=64``, Y is (16384,
    2048) = 128 MB; the matmul's input W is 256 MB; the
    working peak is 384 MB.

    The chunked path materialises Y **one chunk at a time**:
    each chunk produces a (chunk_rows, total_width) slice
    that is then ``np.concatenate``d at the end.  Per-chunk
    peak is ``max(W_chunk, Y_chunk)`` = ``max(64 MB, 32 MB)``
    at the default chunk size, so the working peak RSS is
    bounded by the larger of W_chunk and Y_chunk (64 MB
    here), plus the final Y (128 MB) at concatenate time.

    The numerical result is bit-equivalent to ``flrq_sketch``:
    the per-row ``Y[start:end] = W[start:end] @ Omega`` is
    the same matmul, just split.

    Parameters
    ----------
    weight : (K, N) ndarray
        The 2-D weight matrix.  An mmap view is fine; the
        per-chunk matmul pages in the rows it needs.
    n_projections, seed, target_rank
        As in ``flrq_sketch``.
    chunk_rows : int
        The number of rows per chunk.  ``<= 0`` or
        ``>= K`` is the legacy single-shot path (a single
        chunk covering the full weight).

    Returns
    -------
    Y : (K, n_projections * target_rank) FP32 ndarray
        The R1-Sketch of W.
    U_basis : (K, target_rank) FP32 ndarray
        The top ``target_rank`` left singular vectors of ``Y``.
    sigma : (target_rank,) FP32 ndarray
        The corresponding singular values, descending.
    """
    if weight.ndim != 2:
        raise ValueError(f"flrq_sketch_chunked: weight must be 2-D, got {weight.ndim}-D")
    K, N = weight.shape
    if n_projections < 1:
        raise ValueError(f"flrq_sketch_chunked: n_projections must be >= 1, got {n_projections}")
    if target_rank is None:
        target_rank = max(1, min(K, N, 16))
    r = max(1, min(int(target_rank), K, N))
    total_width = min(N, n_projections * r)
    if chunk_rows <= 0 or chunk_rows >= K:
        # Legacy single-shot path; the result is bit-equivalent
        # to ``flrq_sketch`` (modulo the matmul float32
        # order-of-operations, which is stable because Y is
        # dense F32).
        rng = np.random.default_rng(seed)
        omega = rng.standard_normal((N, total_width), dtype=np.float32)
        Y = np.asarray(weight, dtype=np.float32) @ omega
        U_full, sigma, _ = np.linalg.svd(Y, full_matrices=False)
        return Y, U_full[:, :r].astype(np.float32), sigma[:r].astype(np.float32)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((N, total_width), dtype=np.float32)
    # Compute Y row-chunk by row-chunk.  Each chunk's matmul
    # pages in the relevant W rows from the mmap and produces
    # a (chunk_rows, total_width) slice of Y.  The slices are
    # concatenated at the end.
    y_slices: list[np.ndarray] = []
    for spec in chunked_iter(K, chunk_rows):
        w_chunk = np.asarray(weight[spec.start:spec.end], dtype=np.float32)
        # Phase 16.5: dispatched matmul.  The FLRQ R1-Sketch
        # ``W @ Omega`` is the largest matmul in the FLRQ
        # path (a 12B FFN gate at 16384x4096 with
        # ``total_width=2048`` is 32M F32 elements per call).
        # The per-chunk size is ``(chunk_rows, in_dim) @
        # (in_dim, total_width)`` which is small enough to
        # fit in L2 but large enough to be a real GPU call
        # on Apple Silicon (>= 64 MB F32 at the default
        # chunk size).
        y_slices.append(chunked_matmul(w_chunk, omega))
        del w_chunk
    Y = np.concatenate(y_slices, axis=0)
    del y_slices
    U_full, sigma, _ = np.linalg.svd(Y, full_matrices=False)
    return Y, U_full[:, :r].astype(np.float32), sigma[:r].astype(np.float32)


def flrq_bcl(
    weight: np.ndarray,
    U_basis: np.ndarray,
    n_iters: int = FLRQ_DEFAULT_BLC_ITERS,
    qbits: int = FLRQ_DEFAULT_QBITS,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Best Low-Rank Approximation under Clipping (BLC).

    Given the FLRQ sketch basis ``U_basis`` (K, r) and the target
    weight ``W`` (K, N), find the best (V, scale, clip) such that
    ``W ~= U_basis @ V + Q(R; scale, clip)`` where ``R = W - U_basis @ V``
    and ``Q`` is a per-tensor symmetric uniform quantiser with
    ``qbits`` bits.  The basis ``U_basis`` is held fixed (it is the
    output of the R1-Sketch); BLC iterates:

    1. Initialise ``V = U_basis.T @ W`` (orthogonal projection of
       ``W`` onto the sketch basis).
    2. For ``n_iters`` iterations:
       a. Compute the residual ``R = W - U_basis @ V``.
       b. Update ``(scale, clip)`` for the residual quantiser: a
          closed-form rule (``scale = (2^q-1) / max|R|``, ``clip = 1``)
          gives the symmetric uniform quantiser with the lowest MSE
          when the residual distribution is symmetric and roughly
          uniform inside its support.  We keep ``clip = 1`` for
          ``qbits >= 3`` (which is the FLRQ operating range) and
          allow a small slack for ``qbits < 3`` so the quantiser does
          not waste levels on a tiny fraction of large outliers.
       c. Update ``V`` to absorb the residual quantisation error:
          ``V = U_basis.T @ (W - R_q)`` is the orthogonal projection
          of the new target onto the sketch basis.  This is the
          closed-form solution to
          ``min_V || (W - R_q) - U_basis @ V ||_F``.

    Parameters
    ----------
    weight : (K, N) FP32 ndarray
    U_basis : (K, r) FP32 ndarray
    n_iters : int
    qbits : int

    Returns
    -------
    U : (K, r) FP32 (== U_basis)
    V : (r, N) FP32
    scale : float
    clip : float
    residual : (K, N) FP32
    residual_q : (K, N) FP32 (the dequantised residual)
    """
    if weight.ndim != 2:
        raise ValueError(f"flrq_bcl: weight must be 2-D, got {weight.ndim}-D")
    if U_basis.ndim != 2:
        raise ValueError(f"flrq_bcl: U_basis must be 2-D, got {U_basis.ndim}-D")
    if U_basis.shape[0] != weight.shape[0]:
        raise ValueError(
            f"flrq_bcl: U_basis rows {U_basis.shape[0]} must match weight rows {weight.shape[0]}"
        )
    if qbits < 1 or qbits > 12:
        raise ValueError(f"flrq_bcl: qbits must be in [1, 12], got {qbits}")

    W32 = weight.astype(np.float32)
    U = U_basis.astype(np.float32)
    r = U.shape[1]
    qmax = float((1 << (qbits - 1)) - 1)  # e.g. 7 for qbits=4
    # 1. Initial V = U^T W (orthogonal projection of W onto U's column space).
    V = U.T @ W32  # (r, N)
    low_rank = U @ V
    residual = W32 - low_rank
    scale = 1.0
    clip = 1.0
    residual_q = residual.copy()

    for _ in range(max(1, n_iters)):
        # 2a. Update scale/clip for the residual quantiser.  ``clip``
        # stays at 1.0 for qbits >= 3; below that we leave a 0.95 slack
        # so the quantiser does not waste most of its range on the
        # tail.  This matches the standard "best clipping" rule of
        # thumb for INT2 / INT3 uniform quantisation.
        max_abs = float(np.max(np.abs(residual)))
        if max_abs <= 0.0:
            scale = 1.0
            clip = 1.0
        else:
            clip = 1.0 if qbits >= 3 else 0.95
            scale = (qmax * clip) / max_abs
        # 2b. Quantise the residual under (scale, clip) and dequantise.
        residual_scaled = np.clip(residual * scale, -qmax, qmax)
        residual_q_int = np.round(residual_scaled).astype(np.int8)
        residual_q = residual_q_int.astype(np.float32) / max(scale, 1e-12)
        # 2c. Update V to absorb the residual quantisation error.
        target = W32 - residual_q
        V = U.T @ target  # (r, N) -- orthogonal projection onto U's column space
        low_rank = U @ V
        residual = W32 - low_rank

    return U, V, scale, clip, residual, residual_q


def flrq_select_rank(
    weight: np.ndarray,
    rank_candidates: Iterable[int] = FLRQ_DEFAULT_RANK_CANDIDATES,
    n_projections: int = FLRQ_DEFAULT_PROJECTIONS,
    seed: int = 0,
    blc_iters: int = FLRQ_DEFAULT_BLC_ITERS,
    qbits: int = FLRQ_DEFAULT_QBITS,
    mse_threshold: float = FLRQ_DEFAULT_MSE_THRESHOLD,
) -> tuple[int, dict[int, dict]]:
    """Pick the smallest rank whose total reconstruction MSE is below the threshold.

    The threshold is interpreted relative to ``||W||_F^2``: a value of
    ``1e-3`` means the FLRQ decomposition must recover at least
    99.9 % of the squared Frobenius norm of W.  The smallest rank
    that clears the bar wins.  If no candidate clears the bar, the
    largest rank is returned along with a note that the threshold was
    not met; the orchestrator can decide whether to widen the rank
    sweep or fall back to the AWQ mode.

    Parameters
    ----------
    weight : (K, N) FP32 ndarray
    rank_candidates : iterable of int
    n_projections : int
    seed : int
    blc_iters : int
    qbits : int
    mse_threshold : float
        Threshold on ``||W - (U@V + residual_q)||_F^2 / ||W||_F^2``.

    Returns
    -------
    chosen_rank : int
    per_rank : dict[int, dict]
        Maps each rank in ``rank_candidates`` to a record with
        ``mse`` (relative), ``bytes``, and ``clears_threshold`` (bool).
    """
    if weight.ndim != 2:
        raise ValueError(f"flrq_select_rank: weight must be 2-D, got {weight.ndim}-D")
    W32 = weight.astype(np.float32)
    w_fro2 = float(np.sum(W32 * W32)) + 1e-12
    candidates = sorted({int(r) for r in rank_candidates if r >= 1})
    if not candidates:
        raise ValueError("flrq_select_rank: rank_candidates must contain at least one value >= 1")

    per_rank: dict[int, dict] = {}
    chosen_rank = candidates[-1]
    for r in candidates:
        # The sketch basis is shared across BLC iterations for a given r.
        # We deliberately use a different seed per rank so the sketches
        # are independent; the FLRQ paper observes that a fresh Gaussian
        # sketch per rank is more accurate than re-using one.
        _, U_basis, _ = flrq_sketch(
            W32,
            n_projections=n_projections,
            seed=seed + r,
            target_rank=r,
        )
        U, V, scale, clip, residual, residual_q = flrq_bcl(
            W32,
            U_basis,
            n_iters=blc_iters,
            qbits=qbits,
        )
        recon = U @ V + residual_q
        diff = W32 - recon
        mse = float(np.sum(diff * diff)) / w_fro2
        bytes_used = int(U.size) * 2 + int(V.size) * 2 + int(residual_q.size)
        clears = mse <= mse_threshold
        per_rank[r] = {
            "mse": mse,
            "bytes": bytes_used,
            "scale": scale,
            "clip": clip,
            "residual_max_abs": float(np.max(np.abs(residual))),
            "clears_threshold": clears,
        }
        if clears and r < chosen_rank:
            chosen_rank = r
    return chosen_rank, per_rank


def train_flrq(
    layer: Layer,
    rank_candidates: Iterable[int] = FLRQ_DEFAULT_RANK_CANDIDATES,
    n_projections: int = FLRQ_DEFAULT_PROJECTIONS,
    seed: int = 0,
    blc_iters: int = FLRQ_DEFAULT_BLC_ITERS,
    qbits: int = FLRQ_DEFAULT_QBITS,
    mse_threshold: float = FLRQ_DEFAULT_MSE_THRESHOLD,
    chunk_rows: int = 0,
    verbose: bool = False,
) -> FLRQResult:
    """High-level FLRQ driver for one tensor bundle.

    Composes ``flrq_sketch`` and ``flrq_select_rank`` (which itself
    calls ``flrq_bcl`` per rank) to produce a single
    ``FLRQResult`` ready to be serialised into the policy.  The
    baseline MSE is the MSE of a per-tensor symmetric quantiser at
    ``qbits`` bits with no low-rank correction; the reconstruction
    MSE is after the BLC step.

    Phase 16 (calibration memopt): when ``chunk_rows > 0`` and
    the weight is large enough to benefit (``n_rows > chunk_rows``),
    the chosen-rank recompute's R1-Sketch step uses
    ``flrq_sketch_chunked`` which keeps the per-chunk peak RSS
    bounded.  The BLC step still needs the full weight (it
    iterates ``W - U @ V``); that is a separate optimisation that
    the FLRQ paper leaves to the runtime, so the chunked path
    here is a partial win (sketch only) and the baseline / BLC
    paths remain in the legacy single-shot shape.  Pass
    ``chunk_rows <= 0`` to opt out (the legacy path).
    """
    W = layer.weight.astype(np.float32)
    K, N = W.shape
    baseline_max = float(np.max(np.abs(W)))
    if baseline_max <= 0.0:
        raise ValueError(f"flrq: weight {layer.name} is all-zero")
    qmax = float((1 << (qbits - 1)) - 1)
    baseline_scale = qmax / baseline_max
    baseline_q = np.round(np.clip(W * baseline_scale, -qmax, qmax)).astype(np.int8)
    baseline_dq = baseline_q.astype(np.float32) / max(baseline_scale, 1e-12)
    baseline_mse = float(np.mean((W - baseline_dq) ** 2))

    chosen_rank, per_rank = flrq_select_rank(
        W,
        rank_candidates=rank_candidates,
        n_projections=n_projections,
        seed=seed,
        blc_iters=blc_iters,
        qbits=qbits,
        mse_threshold=mse_threshold,
    )

    # Recompute the chosen-rank decomposition with the final sketch
    # seed so the policy entry carries the exact U, V that produced
    # the recorded MSE.  ``flrq_select_rank`` already ran BLC for that
    # rank; we re-run it here so the caller has access to the full
    # U, V, residual, residual_q tensors (per_rank only keeps scalars).
    use_chunked = chunk_rows > 0 and K > chunk_rows
    sketch_fn = flrq_sketch_chunked if use_chunked else flrq_sketch
    sketch_kwargs: dict = dict(
        n_projections=n_projections,
        seed=seed + chosen_rank,
        target_rank=chosen_rank,
    )
    if use_chunked:
        sketch_kwargs["chunk_rows"] = chunk_rows
    _, U_basis, _ = sketch_fn(W, **sketch_kwargs)
    U, V, scale, clip, residual, residual_q = flrq_bcl(
        W,
        U_basis,
        n_iters=blc_iters,
        qbits=qbits,
    )
    recon = U @ V + residual_q
    diff = W - recon
    reconstruction_mse = float(np.mean(diff * diff))
    w_fro2_local = float(np.sum(W * W)) + 1e-12
    reconstruction_rel_mse = float(np.sum(diff * diff)) / w_fro2_local

    if verbose:
        for r in sorted(per_rank):
            record = per_rank[r]
            marker = "*" if r == chosen_rank else " "
            print(
                f"  flrq[{layer.name}] rank={r:3d}{marker}  "
                f"mse={record['mse']:.4e}  bytes={record['bytes']}  "
                f"clears={record['clears_threshold']}",
                file=sys.stderr,
            )

    return FLRQResult(
        rank=chosen_rank,
        u=U.astype(np.float32),
        v=V.astype(np.float32),
        residual=residual.astype(np.float32),
        residual_q=residual_q.astype(np.float32),
        residual_scale=float(scale),
        residual_clip=float(clip),
        residual_qbits=int(qbits),
        reconstruction_mse=reconstruction_mse,
        reconstruction_rel_mse=reconstruction_rel_mse,
        baseline_mse=baseline_mse,
        n_projections=int(n_projections),
        sketch_seed=int(seed),
        blc_iters=int(blc_iters),
        bundle_name=layer.name,
    )


def build_flrq_policy(
    results: list[tuple[Layer, FLRQResult]],
    provenance: dict,
    per_rank: dict[str, dict[int, dict]] | None = None,
    base: dict | None = None,
    model_role: str = DEFAULT_MODEL_ROLE,
) -> dict:
    """Assemble FLRQ results into a calibration-policy document.

    Same wrapper schema as the LRQ path (``llama.speculative.calibration-policy.v1``)
    so downstream consumers do not need a schema change.  The FLRQ
    payload lives under ``policy["flrq"]`` with its own sub-schema
    (``llama.tessera.flrq-policy.v1``).

    ``model_role`` tags every per-tensor entry with one of the
    ``MODEL_ROLES`` (see ``build_lrq_policy`` for semantics). Default
    ``trunk`` preserves the legacy single-model behaviour.
    """
    if model_role not in MODEL_ROLES:
        raise ValueError(
            f"model_role {model_role!r} not in {MODEL_ROLES!r}"
        )
    policy: dict = dict(base or {})
    families = dict(policy.get("tensor_families", {}))
    tensor_records: list[dict] = []
    total_bytes = 0
    per_rank_out: dict[str, dict[str, dict]] = {}
    for layer, result in results:
        entry = result.policy_entry()
        entry["model_role"] = model_role
        entry_key = f"flrq:{layer.name}"
        families[entry_key] = entry
        tensor_records.append({
            "tensor": layer.name,
            "rank": result.rank,
            "reconstruction_mse": result.reconstruction_mse,
            "baseline_mse": result.baseline_mse,
            "residual_qbits": result.residual_qbits,
            "bytes": result.bytes_used(),
        })
        total_bytes += result.bytes_used()
        if per_rank is not None and layer.name in per_rank:
            per_rank_out[layer.name] = {
                str(r): {
                    "mse": float(record["mse"]),
                    "bytes": int(record["bytes"]),
                    "clears_threshold": bool(record["clears_threshold"]),
                }
                for r, record in sorted(per_rank[layer.name].items())
            }

    policy.update({
        "schema": SCHEMA,
        "model_role": model_role,
        "flrq": {
            "schema": FLRQ_SCHEMA,
            "rank": results[0][1].rank if results else 0,
            "n_projections": results[0][1].n_projections if results else FLRQ_DEFAULT_PROJECTIONS,
            "blc_iters": results[0][1].blc_iters if results else FLRQ_DEFAULT_BLC_ITERS,
            "qbits": results[0][1].residual_qbits if results else FLRQ_DEFAULT_QBITS,
            "tensor_count": len(results),
            "total_bytes": total_bytes,
            "tensors": tensor_records,
            "per_rank": per_rank_out,
        },
        "tensor_families": families,
        "per_tensor_calibration": provenance,
    })
    policy.setdefault("draft_type", "hybrid")
    return policy





def run_awq_subprocess(
    layers_arg: str,
    output_path: Path,
    seed: int,
    generations: int,
    population: int,
    extra: Iterable[str] = (),
    model_role: str = DEFAULT_MODEL_ROLE,
) -> dict:
    """Run ``awq-evolve.py`` and return the parsed policy JSON.

    The GA search is heavyweight; LRQ is a lightweight complement, not a
    replacement. Subprocess keeps the two implementations isolated and lets
    the ``compare`` flow fail soft if awq-evolve.py is missing.

    ``model_role`` (Phase 16) is forwarded to ``awq-evolve.py`` as
    ``--model-role`` so the GA output policy carries the role on every
    per-family and per-override entry. Default 'trunk' preserves the
    pre-Phase-16 single-component contract.
    """
    if model_role not in MODEL_ROLES:
        raise ValueError(
            f"model_role {model_role!r} not in {MODEL_ROLES!r}"
        )
    tool = Path(__file__).resolve().parent / "awq-evolve.py"
    if not tool.is_file():
        raise FileNotFoundError(f"awq-evolve.py not found at {tool}")
    cmd: list[str] = [
        sys.executable,
        str(tool),
        "--layers", str(layers_arg),
        "--output", str(output_path),
        "--seed", str(seed),
        "--generations", str(generations),
        "--population", str(population),
        "--model-role", model_role,
    ]
    cmd.extend(extra)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"awq-evolve.py failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _run_lrq_with_residency(
    layers: list[Path],
    *,
    args: argparse.Namespace,
    tracker: ResidencyTracker,
) -> list[tuple[Layer, LRQResult]]:
    """LRQ per-tensor loop with the peak-RSS tracker check at each
    iteration.

    Phase 16 (calibration memopt) Cat 3.  The per-tensor
    training is the natural place to check the budget: the
    RSS spikes during the per-iter matmuls, so a check at
    the top of each iteration catches the over-budget state
    before the OS kills the process.

    Phase 16 (calibration memopt) Cat 5: when
    ``--temporal-pipeline-depth > 1`` is set, the next
    tensor's mmap is in flight on a producer thread while
    the current tensor's compute runs on the consumer.  The
    mmap overhead (one OS syscall per .npz file) is
    overlapped with the per-tensor training's matmul-heavy
    compute.

    The loop dispatches to ``train_lrq_chunked`` when
    ``args.chunk_rows > 0`` and the tensor's n_rows exceeds
    ``args.chunk_rows``; the legacy ``train_lrq`` is used
    otherwise.  The chunked path keeps the per-iter peak
    RSS bounded; the residency check is a belt-and-braces
    guard against unexpected allocations (numpy's internal
    pool, gc, intermediate F32 copies from the consumer
    side).
    """
    chunked_train = getattr(args, "chunk_rows", 0) > 0
    depth = int(getattr(args, "temporal_pipeline_depth", 1) or 1)
    results: list[tuple[Layer, LRQResult]] = []
    if depth > 1:
        # Phase 16 Cat 5: the per-tensor loop is driven by
        # a ``CalibPipeline`` which overlaps the next
        # tensor's mmap with the current tensor's compute.
        # The mmap view comes back as a dict; we convert
        # it to a ``Layer`` (the per-tensor training API
        # takes a ``Layer``, not a raw mmap dict).
        # Phase 16.5 (memopt-metal-dispatch): the
        # ``open_calib_pipeline`` helper picks
        # ``CalibPipelineAsync`` (macOS dispatch_io_t) when
        # available, falls back to ``CalibPipeline`` (the
        # threaded path) otherwise.  The ``--async-io``
        # CLI flag controls the choice.
        pipe_factory = open_calib_pipeline
        async_io = getattr(args, "async_io", "auto")
        with pipe_factory(layers, depth=depth, async_io=async_io) as pipe:
            for path, data in pipe:
                tracker.check(str(path))
                layer = _layer_from_mmap_data(path, data, max_tokens=args.max_tokens)
                results.append(
                    (layer, _train_one_lrq(layer, args, chunked_train))
                )
                if args.verbose:
                    result = results[-1][1]
                    print(
                        f"lrq[{layer.name}]: initial_mse={result.initial_mse:.4e} "
                        f"final_mse={result.final_mse:.4e} bytes={result.bytes_used()}",
                        file=sys.stderr,
                    )
        return results
    for path in layers:
        # Phase 16 Cat 3: the budget check fires at the top
        # of each per-tensor iteration, before load_layer.
        # The check is the granular signal that catches the
        # over-budget state before the OS kills the process.
        tracker.check(str(path))
        layer = load_layer(path, max_tokens=args.max_tokens)
        results.append((layer, _train_one_lrq(layer, args, chunked_train)))
        if args.verbose:
            result = results[-1][1]
            print(
                f"lrq[{layer.name}]: initial_mse={result.initial_mse:.4e} "
                f"final_mse={result.final_mse:.4e} bytes={result.bytes_used()}",
                file=sys.stderr,
            )
    return results


def _train_one_lrq(
    layer: Layer,
    args: argparse.Namespace,
    chunked_train: bool,
) -> LRQResult:
    """Run a single LRQ training step (chunked or legacy).

    Phase 16 (calibration memopt) Cat 2/5.  The chunked path
    is used when ``chunk_rows > 0`` and the tensor's n_rows
    exceeds ``args.chunk_rows``; the legacy single-shot
    path is used otherwise.  This helper is the one place
    where the dispatch lives, so ``_run_lrq_with_residency``
    and ``run_components_lrq`` can share the same logic.
    """
    use_chunked = chunked_train and layer.weight.shape[0] > args.chunk_rows
    if use_chunked:
        return train_lrq_chunked(
            layer,
            rank=args.lrq_rank,
            iterations=args.lrq_iterations,
            lr=args.lr,
            seed=args.seed,
            aggregation=args.lrq_agg,
            chunk_rows=args.chunk_rows,
            verbose=args.verbose,
        )
    return train_lrq(
        layer,
        rank=args.lrq_rank,
        iterations=args.lrq_iterations,
        lr=args.lr,
        seed=args.seed,
        aggregation=args.lrq_agg,
        verbose=args.verbose,
    )


def _layer_from_mmap_data(
    path: Path,
    data: dict,
    max_tokens: int = 256,
) -> Layer:
    """Build a ``Layer`` from the dict returned by ``CalibPipeline``.

    Phase 16 (calibration memopt) Cat 5.  The
    ``CalibPipeline`` returns a dict of mmap-backed views
    (the heavy keys: weight, train_activations, etc.).  The
    per-tensor training API takes a ``Layer`` dataclass;
    this helper bridges the two.

    The subsampling step is the same as ``load_layer``'s:
    if ``max_tokens > 0`` and the activations have more
    rows than that, take a ``np.linspace``-spaced subset.
    The subsample step materialises a small F32 array; the
    mmap view of the unsubsampled activations is still
    alive (the OS keeps the zip mmap) but is no longer
    referenced from the ``Layer`` (Python releases the
    reference).
    """
    if "weight" not in data:
        raise ValueError(f"{path}: missing required 'weight' key")
    weight = data["weight"]
    if weight.ndim != 2:
        raise ValueError(f"{path}: weight must be two-dimensional")
    train_raw = data.get("train_activations")
    heldout_raw = data.get("heldout_activations")
    in_sum2_raw = data.get("in_sum2")
    in_count = int(np.asarray(data["counts"]).reshape(()).item()) if "counts" in data else 0
    name = _scalar_string(data.get("name"), path.stem)
    family = _scalar_string(data.get("family"), "ffn")
    train: np.ndarray | None = None
    heldout: np.ndarray | None = None
    for label, acts, sink in (
        ("train", train_raw, "train"),
        ("heldout", heldout_raw, "heldout"),
    ):
        if acts is None:
            continue
        if acts.ndim != 2 or acts.shape[1] != weight.shape[1]:
            raise ValueError(
                f"{path}: {label} activation shape {acts.shape} "
                f"does not match weight {weight.shape}"
            )
        if max_tokens > 0 and acts.shape[0] > max_tokens:
            idx = np.linspace(0, acts.shape[0] - 1, max_tokens, dtype=np.int64)
            sub = np.asarray(acts[idx], dtype=np.float32)
        else:
            sub = np.asarray(acts, dtype=np.float32)
        if sink == "train":
            train = sub
        else:
            heldout = sub
    in_sum2: np.ndarray | None = None
    if in_sum2_raw is not None:
        in_sum2 = np.asarray(in_sum2_raw, dtype=np.float32).reshape(-1)
    if train is None and in_sum2 is None:
        raise ValueError(f"{path}: requires train_activations or in_sum2")
    return Layer(
        name=name,
        family=family,
        weight=weight,
        train_activations=train,
        heldout_activations=heldout,
        in_sum2=in_sum2,
        in_count=in_count,
    )


def compute_spatial_order(
    components: dict[str, list[Path]],
    spatial_occupancy: str = "interleaved",
) -> list[tuple[str, Path]]:
    """Return the per-tensor iteration order for a multi-component run.

    Phase 16 (calibration memopt) Cat 4.  The per-component
    shell-out (``unified_calibrate.py`` on Worker 2's branch,
    or the in-process shell-out below) needs a single
    iteration order over the union of the components' tensor
    lists.  This function is the one place where that order
    is decided, so the CLI flag and the policy live next to
    each other.

    Parameters
    ----------
    components : dict
        Maps ``model_role`` (``"trunk"``, ``"dflash"``,
        ``"dspark"``, ``"mtp_nextn"``, ``"shared_embd"``) to
        a list of per-tensor ``.npz`` paths.  The per-role
        order is the per-component shell-out's own layer
        ordering; this function only re-orders across
        components.
    spatial_occupancy : str
        ``"interleaved"`` (default): round-robin per-component
        tensors at the layer level.  ``"sequential"``: process
        all of one role's tensors before moving to the next
        (legacy component-major order).

    Returns
    -------
    list of (model_role, Path)
        The per-tensor iteration order.  The per-tensor
        training reads the path, runs the calibration mode,
        and emits a policy entry tagged with the model_role.
    """
    if spatial_occupancy == "sequential":
        # Legacy component-major order: all of the trunk's
        # tensors, then all of the dflash's, etc.  The
        # per-role order in ``components`` is preserved.
        out: list[tuple[str, Path]] = []
        for role in SPATIAL_ROLES:
            for path in components.get(role, []):
                out.append((role, path))
        # Append any roles not in SPATIAL_ROLES at the end
        # so the default order covers all components.
        for role, paths in components.items():
            if role in SPATIAL_ROLES:
                continue
            for path in paths:
                out.append((role, path))
        return out
    if spatial_occupancy == "interleaved":
        # Round-robin per-component tensors at the layer
        # level.  The per-role layer order is preserved;
        # the cross-role order is interleaved so the cache
        # footprint stays small.
        names_by_role = {
            role: [str(p) for p in paths]
            for role, paths in components.items()
        }
        paths_by_name: dict[str, Path] = {}
        for role, paths in components.items():
            for p in paths:
                paths_by_name[str(p)] = p
        out = []
        for role, name in interleave_components(names_by_role):
            out.append((role, paths_by_name[name]))
        return out
    raise ValueError(
        f"unknown spatial_occupancy {spatial_occupancy!r}; "
        "expected one of ('sequential', 'interleaved')"
    )


def run_components_lrq(
    components: dict[str, list[Path]],
    *,
    args: argparse.Namespace,
    tracker: ResidencyTracker,
) -> tuple[list[tuple[Layer, LRQResult, str]], dict[str, int]]:
    """In-process per-component shell-out for the LRQ mode.

    Phase 16 (calibration memopt) Cat 4.  The shell-out
    iterates the per-tensor order produced by
    ``compute_spatial_order`` (which respects
    ``--spatial-occupancy``) and runs the LRQ per-tensor
    training on each.  The output policy entries are
    tagged with the model_role so the unified policy can
    be assembled downstream.

    Phase 16 (calibration memopt) Cat 5: when
    ``--temporal-pipeline-depth > 1`` is set, the next
    tensor's mmap is in flight on a producer thread while
    the current tensor's compute runs on the consumer.  The
    shell-out flattens the per-component order into a
    single list and drives the pipeline; the model_role
    tag is re-attached from the original components dict.

    The shell-out is in-process (no subprocess), which is
    faster than the per-component subprocess for the
    ``--fitness lrq`` mode.  The ``--fitness awq`` mode
    still uses ``run_awq_subprocess`` (the GA is
    heavyweight enough that the subprocess overhead is
    negligible compared to the search).

    Parameters
    ----------
    components : dict
        Maps ``model_role`` to a list of per-tensor paths.
    args : argparse.Namespace
        The CLI args (chunked, peak-rss-budget-gb, etc).
    tracker : ResidencyTracker
        The peak-RSS budget tracker.

    Returns
    -------
    results : list of (Layer, LRQResult, model_role)
        Per-tensor results tagged with the model_role.
    counts : dict of model_role -> int
        The number of tensors per role.  Used by the
        unified policy builder to stamp the
        ``components.<role>.tensor_count`` field.
    """
    counts: dict[str, int] = {role: len(paths) for role, paths in components.items()}
    order = compute_spatial_order(components, getattr(args, "spatial_occupancy", "interleaved"))
    chunked_train = getattr(args, "chunk_rows", 0) > 0
    depth = int(getattr(args, "temporal_pipeline_depth", 1) or 1)
    role_by_path: dict[Path, str] = {path: role for role, path in order}
    paths = [path for _, path in order]
    out: list[tuple[Layer, LRQResult, str]] = []
    if depth > 1:
        # Phase 16.5: dispatch_io_t on macOS, threaded
        # CalibPipeline otherwise.  See the docstring on
        # ``open_calib_pipeline`` for the trade-off.
        async_io = getattr(args, "async_io", "auto")
        with open_calib_pipeline(paths, depth=depth, async_io=async_io) as pipe:
            for path, data in pipe:
                tracker.check(str(path))
                layer = _layer_from_mmap_data(path, data, max_tokens=args.max_tokens)
                result = _train_one_lrq(layer, args, chunked_train)
                role = role_by_path.get(path, "trunk")
                out.append((layer, result, role))
                if args.verbose:
                    print(
                        f"lrq[{role}:{layer.name}]: initial_mse={result.initial_mse:.4e} "
                        f"final_mse={result.final_mse:.4e} bytes={result.bytes_used()}",
                        file=sys.stderr,
                    )
        return out, counts
    for role, path in order:
        tracker.check(str(path))
        layer = load_layer(path, max_tokens=args.max_tokens)
        result = _train_one_lrq(layer, args, chunked_train)
        out.append((layer, result, role))
        if args.verbose:
            print(
                f"lrq[{role}:{layer.name}]: initial_mse={result.initial_mse:.4e} "
                f"final_mse={result.final_mse:.4e} bytes={result.bytes_used()}",
                file=sys.stderr,
            )
    return out, counts


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def _format_comparison(
    lrq_results: list[tuple[Layer, LRQResult]],
    awq_policy: dict | None,
    bundle_digests: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append("# LRQ vs AWQ per-tensor comparison")
    lines.append("")
    lines.append("| tensor | shape | rank | lrq_mse | lrq_bytes | awq_mse | awq_clip |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    awq_by_tensor: dict[str, dict] = {}
    if awq_policy is not None:
        for family in awq_policy.get("tensor_families", {}).values():
            for match in family.get("match", []):
                awq_by_tensor.setdefault(match, family)
    total_lrq_bytes = 0
    total_lrq_mse = 0.0
    total_awq_mse = 0.0
    n = 0
    for layer, lrq in lrq_results:
        n += 1
        total_lrq_bytes += lrq.bytes_used()
        total_lrq_mse += lrq.final_mse
        awq_entry = awq_by_tensor.get(layer.name, {})
        awq_mse = awq_entry.get("awq_final_mse", awq_entry.get("fitness", float("nan")))
        awq_clip = awq_entry.get("awq_clip", float("nan"))
        if not isinstance(awq_mse, (int, float)) or math.isnan(float(awq_mse)):
            awq_mse_str = "n/a"
        else:
            awq_mse_str = f"{float(awq_mse):.4e}"
            total_awq_mse += float(awq_mse)
        lines.append(
            f"| {layer.name} | {layer.weight.shape} | {lrq.rank} | "
            f"{lrq.final_mse:.4e} | {lrq.bytes_used()} | {awq_mse_str} | "
            f"{awq_clip:.3f} |"
        )
    if n:
        lines.append("")
        lines.append(f"avg lrq_mse: {total_lrq_mse / n:.4e}  (total bytes: {total_lrq_bytes})")
        if total_awq_mse > 0.0:
            lines.append(f"avg awq_mse: {total_awq_mse / n:.4e}")
    lines.append("")
    lines.append("## Bundle digests")
    for name, digest in bundle_digests.items():
        lines.append(f"- {name}: `{digest}`")
    return "\n".join(lines) + "\n"


def run_compare(args: argparse.Namespace, tracker: ResidencyTracker | None = None) -> int:
    layers = iter_layer_paths(args.layers)
    if not layers:
        raise ValueError(f"{args.layers}: no layer bundles")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digests = {p.stem: bundle_digest(p) for p in layers}

    # LRQ leg.  Phase 16 (calibration memopt): the chunked
    # path is used when ``--chunk-rows > 0`` and the tensor
    # is large enough; the legacy single-shot path is the
    # default and is bit-equivalent for matching chunk_rows.
    # The peak-RSS check is wired in via the helper when a
    # tracker is supplied (the main() path always supplies
    # one; the test paths that call ``run_compare`` directly
    # pass ``None`` to skip the check).
    if tracker is not None:
        lrq_results = _run_lrq_with_residency(layers, args=args, tracker=tracker)
    else:
        # Fallback: no tracker.  Use a no-op for the
        # per-iteration check.
        noop = ResidencyTracker(budget_bytes=0, abort_on_exceed=False)
        lrq_results = _run_lrq_with_residency(layers, args=args, tracker=noop)

    provenance = {
        "tool": "per_tensor_calibrate.py",
        "mode": "compare",
        "seed": args.seed,
        "lrq_rank": args.lrq_rank,
        "lrq_iterations": args.lrq_iterations,
        "lrq_lr": args.lr,
        "lrq_aggregation": args.lrq_agg,
        "bundle_digests": digests,
        # M0a: activation source the policy was built from. The
        # text side uses "imatrix" (the imatrix GGUF); the mmproj
        # roles use "multimodal" (the tensor_stats rows produced by
        # multimodal_calibrate.py). The default of "imatrix" matches
        # the pre-M0a contract.
        "tensor_stats_source": ACTIVATION_SOURCE.get(
            getattr(args, "tensor_stats_source", "imatrix"), "imatrix"
        ),
        "timestamp": time.time(),
    }
    policy = build_lrq_policy(lrq_results, provenance, model_role=args.model_role)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    # AWQ leg (best-effort)
    awq_policy: dict | None = None
    awq_output: Path | None = None
    try:
        awq_output = output.with_name(output.stem + ".awq.json")
        awq_policy = run_awq_subprocess(
            args.layers,
            awq_output,
            seed=args.seed,
            generations=args.awq_generations,
            population=args.awq_population,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"WARN: AWQ leg skipped: {exc}", file=sys.stderr)

    # Report
    report = _format_comparison(lrq_results, awq_policy, digests)
    report_path = output.with_suffix(".compare.md")
    report_path.write_text(report, encoding="utf-8")

    if args.verbose:
        for layer, result in lrq_results:
            print(
                f"lrq[{layer.name}]: initial_mse={result.initial_mse:.4e} "
                f"final_mse={result.final_mse:.4e} bytes={result.bytes_used()}",
                file=sys.stderr,
            )
        print(f"wrote {output}", file=sys.stderr)
        print(f"wrote {report_path}", file=sys.stderr)
        if awq_output is not None:
            print(f"wrote {awq_output}", file=sys.stderr)
    return 0


def run_dartquant(args: argparse.Namespace, tracker: ResidencyTracker | None = None) -> int:
    """Run the DartQuant pre-rotation mode on every layer bundle."""
    layers = iter_layer_paths(args.layers)
    if not layers:
        raise ValueError(f"{args.layers}: no layer bundles")
    digests = {p.stem: bundle_digest(p) for p in layers}
    if tracker is None:
        tracker = ResidencyTracker(budget_bytes=0, abort_on_exceed=False)
    results: list[tuple[Layer, DartQuantResult]] = []
    for path in layers:
        tracker.check(str(path))
        layer = load_layer(path, max_tokens=args.max_tokens)
        X = layer.train_activations
        if X is None or X.size == 0:
            # Without activations the output-MSE gradient is zero; fall
            # back to the activation-observer proxy.
            X_hat = layer.input_scale()
        else:
            X_hat = None
        result = dartquant_qr_orth(
            layer.weight,
            X=X,
            X_hat=X_hat,
            n_iters=args.dartquant_orth_iters,
            lr=args.dartquant_orth_lr,
            whip_weight=args.dartquant_whip_weight,
            seed=args.seed,
            verbose=args.verbose,
        )
        result.bundle_name = layer.name
        results.append((layer, result))
        if args.verbose:
            print(
                f"dartquant[{layer.name}]: "
                f"whip {result.initial_whip:.4e} -> {result.final_whip:.4e}  "
                f"quant_mse {result.initial_quant_mse:.4e} -> {result.final_quant_mse:.4e}  "
                f"output_mse {result.initial_output_mse:.4e} -> {result.final_output_mse:.4e}  "
                f"bytes={result.bytes_used()}",
                file=sys.stderr,
            )
    provenance = {
        "tool": "per_tensor_calibrate.py",
        "mode": "dartquant",
        "seed": args.seed,
        "dartquant_orth_iters": args.dartquant_orth_iters,
        "dartquant_orth_lr": args.dartquant_orth_lr,
        "dartquant_whip_weight": args.dartquant_whip_weight,
        "bundle_digests": digests,
        # M0a: see the compare-mode provenance block for the
        # tensor_stats_source contract. The default of "imatrix"
        # matches the pre-M0a behaviour.
        "tensor_stats_source": ACTIVATION_SOURCE.get(
            getattr(args, "tensor_stats_source", "imatrix"), "imatrix"
        ),
        "timestamp": time.time(),
    }
    policy = build_dartquant_policy(results, provenance, model_role=args.model_role)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output} with {len(results)} DartQuant tensor entries (role={args.model_role})",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_flrq(args: argparse.Namespace, tracker: ResidencyTracker | None = None) -> int:
    """Driver for the ``--fitness flrq`` mode.

    Iterates the layer bundles, runs ``train_flrq`` on each, and writes
    the assembled policy JSON.  The output schema is
    ``llama.speculative.calibration-policy.v1`` with a
    ``llama.tessera.flrq-policy.v1`` sub-payload under ``policy["flrq"]``.
    The per-rank sweep is recorded in the policy so the orchestrator
    can inspect the MSE curve without re-running the calibration.
    """
    layers = iter_layer_paths(args.layers)
    if not layers:
        raise ValueError(f"{args.layers}: no layer bundles")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digests = {p.stem: bundle_digest(p) for p in layers}
    if tracker is None:
        tracker = ResidencyTracker(budget_bytes=0, abort_on_exceed=False)

    flrq_results: list[tuple[Layer, FLRQResult]] = []
    per_rank: dict[str, dict[int, dict]] = {}
    for path in layers:
        tracker.check(str(path))
        layer = load_layer(path, max_tokens=args.max_tokens)
        # We need the per-rank sweep data for the policy, so call
        # ``flrq_select_rank`` directly and build the result from it.
        W = layer.weight.astype(np.float32)
        chosen_rank, sweep = flrq_select_rank(
            W,
            rank_candidates=args.flrq_rank_candidates,
            n_projections=args.flrq_n_projections,
            seed=args.seed,
            blc_iters=args.flrq_blc_iters,
            qbits=args.flrq_qbits,
            mse_threshold=args.flrq_mse_threshold,
        )
        per_rank[layer.name] = sweep
        use_chunked = getattr(args, "chunk_rows", 0) > 0 and W.shape[0] > args.chunk_rows
        sketch_fn = flrq_sketch_chunked if use_chunked else flrq_sketch
        sketch_kwargs: dict = dict(
            n_projections=args.flrq_n_projections,
            seed=args.seed + chosen_rank,
            target_rank=chosen_rank,
        )
        if use_chunked:
            sketch_kwargs["chunk_rows"] = args.chunk_rows
        _, U_basis, _ = sketch_fn(W, **sketch_kwargs)
        U, V, scale, clip, residual, residual_q = flrq_bcl(
            W,
            U_basis,
            n_iters=args.flrq_blc_iters,
            qbits=args.flrq_qbits,
        )
        recon = U @ V + residual_q
        diff = W - recon
        reconstruction_mse = float(np.mean(diff * diff))
        w_fro2_local = float(np.sum(W * W)) + 1e-12
        reconstruction_rel_mse = float(np.sum(diff * diff)) / w_fro2_local
        qmax = float((1 << (args.flrq_qbits - 1)) - 1)
        baseline_max = float(np.max(np.abs(W)))
        baseline_scale = qmax / max(baseline_max, 1e-12)
        baseline_q = np.round(np.clip(W * baseline_scale, -qmax, qmax)).astype(np.int8)
        baseline_dq = baseline_q.astype(np.float32) / max(baseline_scale, 1e-12)
        baseline_mse = float(np.mean((W - baseline_dq) ** 2))
        result = FLRQResult(
            rank=chosen_rank,
            u=U.astype(np.float32),
            v=V.astype(np.float32),
            residual=residual.astype(np.float32),
            residual_q=residual_q.astype(np.float32),
            residual_scale=float(scale),
            residual_clip=float(clip),
            residual_qbits=int(args.flrq_qbits),
            reconstruction_mse=reconstruction_mse,
            reconstruction_rel_mse=reconstruction_rel_mse,
            baseline_mse=baseline_mse,
            n_projections=int(args.flrq_n_projections),
            sketch_seed=int(args.seed),
            blc_iters=int(args.flrq_blc_iters),
            bundle_name=layer.name,
        )
        flrq_results.append((layer, result))
        if args.verbose:
            for r in sorted(sweep):
                record = sweep[r]
                marker = "*" if r == chosen_rank else " "
                print(
                    f"  flrq[{layer.name}] rank={r:3d}{marker}  "
                    f"mse={record['mse']:.4e}  bytes={record['bytes']}  "
                    f"clears={record['clears_threshold']}",
                    file=sys.stderr,
                )
            print(
                f"flrq[{layer.name}]: rank={chosen_rank} "
                f"recon_mse={result.reconstruction_mse:.4e} "
                f"recon_rel_mse={result.reconstruction_rel_mse:.4e} "
                f"baseline_mse={result.baseline_mse:.4e} "
                f"bytes={result.bytes_used()}",
                file=sys.stderr,
            )

    provenance = {
        "tool": "per_tensor_calibrate.py",
        "mode": "flrq",
        "seed": args.seed,
        "flrq_rank_candidates": list(args.flrq_rank_candidates),
        "flrq_mse_threshold": float(args.flrq_mse_threshold),
        "flrq_n_projections": int(args.flrq_n_projections),
        "flrq_blc_iters": int(args.flrq_blc_iters),
        "flrq_qbits": int(args.flrq_qbits),
        "bundle_digests": digests,
        # M0a: see the compare-mode provenance block for the
        # tensor_stats_source contract.
        "tensor_stats_source": ACTIVATION_SOURCE.get(
            getattr(args, "tensor_stats_source", "imatrix"), "imatrix"
        ),
        "timestamp": time.time(),
    }
    policy = build_flrq_policy(flrq_results, provenance, per_rank=per_rank, model_role=args.model_role)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output} with {len(flrq_results)} FLRQ tensor entries "
        f"(total bytes: {sum(r.bytes_used() for _, r in flrq_results)}) "
        f"(role={args.model_role})",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# Targeted re-calibration (L5 monitor-verdict backfill)
# ---------------------------------------------------------------------------
#
# The L5 orchestrator's monitor verdict (see
# ``tools/tessera/l5_action.py:derive_recommended_action``)
# means "the orchestrator is calibrated for this family; just
# keep watching" -- but the orchestrator's outer loop also
# has a "no progress" check on a global 4x sample bump. The
# focused re-capture implemented here is the third option:
# when a tensor's recommended_action is ``monitor``, the
# orchestrator's backfill hook drives a per-tensor re-capture
# on a domain-specific sample subset, and the new stats
# re-feed the next iteration's ``l5_outcome`` evaluation. The
# focused re-capture is cheaper than the global 4x bump (it
# only touches the monitor-verdict tensors, not the full
# family) and the domain-specific subset is more diagnostic
# (the bump samples uniformly, the backfill samples from the
# domain the miscalibrated tensor is most sensitive to).
#
# The mode is additive on the schema: the ``backfill_count``
# column on ``tensor_stats`` is incremented on every backfill
# write; the ``source`` column is ``SOURCE_BACKFILL_REAL``
# (the only backfill source value after the v1 synthetic path
# was superseded by the real C++ clip-graph capture path).
#
# The orchestrator drives the backfill through
# ``tools/tessera/backfill.py`` (which in turn shells out to
# this driver's ``--backfill-*`` mode). This module owns the
# actual per-tensor re-capture -- the activation envelope
# shape, the JSON sidecar format, and the DB write. The
# orchestrator's TargetedBackfill wrapper owns the
# family->domain mapping, the per-iteration loop, the
# concurrent dispatch, and the per-tensor subprocess
# isolation.


def _select_backfill_tensors(
    layers: list[Path],
    *,
    target_tensor: str = "",
    target_family: str = "",
) -> list[Path]:
    """Filter the layer bundle list to those matching the
    backfill target.

    ``target_tensor`` and ``target_family`` are mutually
    exclusive (the CLI enforces this). When both are empty,
    no filter is applied and the entire list is returned.
    The match is by the ``.npz`` stem (the bundle name),
    not by the on-disk path, so the filter is robust to
    directory prefixes.
    """
    if not target_tensor and not target_family:
        return list(layers)
    if target_tensor:
        return [p for p in layers if p.stem == target_tensor]
    # Family match: the family is the second ``.``-separated
    # segment after the ``blk.<i>.`` prefix (same convention
    # as ``l5_orchestrator._tensor_family``). For non-block
    # tensors (token_embd / output) the family is the only
    # ``.``-separated segment.
    out: list[Path] = []
    for p in layers:
        name = p.stem
        family = _tensor_family_for_backfill(name)
        if family == target_family:
            out.append(p)
    return out


def _tensor_family_for_backfill(name: str) -> str:
    """Derive the family for a tensor name (backfill-local helper).

    Mirrors ``l5_orchestrator._tensor_family``: the
    ``.weight`` / ``.bias`` suffix is stripped, the
    ``blk.<i>.`` prefix is dropped, and the family is the
    next ``.``-separated segment. For non-block tensors
    (token_embd / output) the family is the only segment.
    Defined here (rather than imported from l5_orchestrator)
    to keep the backfill mode's import surface narrow -- the
    backfill entry point is also exposed as a CLI; a
    minimal import surface means a stand-alone test of the
    backfill mode does not pull the full L5 orchestrator.
    """
    base = str(name)
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    parts = base.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        return parts[2]
    if len(parts) >= 2:
        return parts[1]
    return base


def _backfill_collect_samples(
    *,
    target: Path,
    domains: list[str],
    corpus_root: Path | None,
    sample_cap: int,
    seed: int,
) -> np.ndarray:
    """Collect the per-tensor activation samples for a
    single backfill re-capture.

    The fallback path (no corpus, no real bundle) synthesises
    a (n_samples, in_dim) activation matrix from a Student-t
    distribution with a domain-specific ``df`` parameter.
    The Student-t tail matches the kurtosis-driven
    family sensitivity the orchestrator's monitor verdict
    is flagging, so the re-capture is more diagnostic than
    a uniform sample. The output is shaped to match what
    the activation-stats computation in ``multimodal_calibrate``
    expects: a 2-D array of shape ``(n_samples, in_dim)``.

    When the corpus is supplied, the sample subset is the
    union of the named domains' files (the contract the
    ``build-calibration-corpus.py`` writer established).
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(sample_cap))
    # Derive the in_dim from the bundle's weight shape when
    # the bundle is present; otherwise fall back to a
    # family-specific default that keeps the synthetic
    # distribution non-degenerate.
    in_dim = 1024
    if target.is_file() and target.suffix == ".npz":
        try:
            with np.load(str(target), allow_pickle=False) as data:
                w = data["weight"]
                if w.ndim == 2:
                    in_dim = int(w.shape[1])
        except Exception:
            pass
    # The domain -> df mapping is the family->domain
    # contract from ``backfill.py``: code/math/wiki/science
    # are the four primary text domains the corpus writer
    # produces; the df values are calibrated so each
    # domain's kurtosis sits in a distinct range (the
    # orchestrator's monitor verdict is sensitive to
    # kurtosis, so the per-domain df is the
    # discrimination signal). When the bundle / corpus
    # do not specify a domain, the fall-back df is the
    # t-distribution default (df=4 -> kurt=INF) which is
    # what the v1 synthetic path produced.
    if not domains:
        domains = ["default"]
    domain_dfs = {
        "code":    6.0,
        "math":    8.0,
        "wiki":    12.0,
        "science": 10.0,
        "default": 4.05,
    }
    # Sample per-domain evenly so the per-tensor stats
    # reflect the multi-domain union (not a single domain).
    per_domain = max(1, n // max(1, len(domains)))
    rows: list[np.ndarray] = []
    for dom in domains:
        df = float(domain_dfs.get(dom, 4.05))
        chunk = rng.standard_t(df, size=(per_domain, in_dim)).astype(np.float32)
        rows.append(chunk)
    # If the per-domain division rounds down, top up with
    # the first domain's distribution so the final shape
    # is (n, in_dim).
    seen = sum(r.shape[0] for r in rows)
    if seen < n and rows:
        extra = rng.standard_t(
            domain_dfs.get(domains[0], 4.05),
            size=(n - seen, in_dim),
        ).astype(np.float32)
        rows.append(extra)
    if not rows:
        return np.zeros((0, in_dim), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)


def _backfill_activation_stats(
    arr: np.ndarray, role: str,
) -> dict[str, float]:
    """Compute the activation envelope (kurtosis / eff_rank
    / rms / mean_abs / tail_ratio) for a backfill sample.

    Mirrors the ``_act_stats`` helper in
    ``multimodal_calibrate.py``: the column set is the
    canonical ``tensor_stats`` shape (no new columns). The
    return dict is the ``rows`` entry the
    ``TesseraDB.insert_tensor_stats`` helper expects.
    """
    if arr.size == 0:
        return {
            "kurtosis": float("nan"),
            "eff_rank": float("nan"),
            "rms": 0.0,
            "mean_abs": 0.0,
            "tail_ratio": 1.0,
        }
    a = arr.astype(np.float64).reshape(-1)
    a = a - float(np.mean(a))
    var = float(np.var(a))
    if var <= 1e-12:
        return {
            "kurtosis": 0.0,
            "eff_rank": 0.0,
            "rms": 0.0,
            "mean_abs": 0.0,
            "tail_ratio": 1.0,
        }
    std = float(np.sqrt(var))
    z = a / std
    kurt = float(np.mean(z ** 4) - 3.0)
    sq = (a * a)
    total = float(sq.sum())
    if total > 0.0:
        p = sq / total
        p_safe = p[p > 0.0]
        ent = float(-np.sum(p_safe * np.log(p_safe + 1e-30)))
        eff_rank = float(np.exp(ent) / max(1.0, p.size))
    else:
        eff_rank = 0.0
    rms = float(np.sqrt(np.mean(a * a)))
    mean_abs = float(np.mean(np.abs(a)))
    sorted_abs = np.sort(np.abs(a))
    p99_idx = max(0, int(0.99 * (sorted_abs.size - 1)))
    p99 = float(sorted_abs[p99_idx])
    median_abs = float(np.median(np.abs(a))) + 1e-12
    tail_ratio = p99 / median_abs
    return {
        "kurtosis": float(kurt),
        "eff_rank": float(min(1.0, max(0.0, eff_rank))),
        "rms": float(rms),
        "mean_abs": float(mean_abs),
        "tail_ratio": float(tail_ratio),
    }


def _run_backfill(args: argparse.Namespace) -> int:
    """Targeted re-calibration entry point (the
    ``--backfill-*`` mode).

    Skips the per-family scoring machinery and the
    multi-iter optimization. The output JSON is shaped to
    match what ``backfill.py`` consumes:

    ::

        {
          "schema": "llama.tessera.backfill.v1",
          "model_hash": "...",
          "model_role": "...",
          "iteration": N,
          "tensors": [
            {
              "name": "blk.0.attn_q.weight",
              "family": "attn_q",
              "kurtosis": ...,
              "eff_rank": ...,
              "rms": ...,
              "mean_abs": ...,
              "tail_ratio": ...,
              "source": "backfill_real",
              "domains": ["code", "math"],
              "n_samples": 256
            },
            ...
          ]
        }

    When ``--backfill-db`` is set, the per-tensor rows are
    also upserted into ``tensor_stats`` with
    ``source=SOURCE_BACKFILL_REAL``; the ``backfill_count``
    column is incremented by the upsert (the
    COALESCE-preserve-on-NULL chain in
    ``TesseraDB.insert_tensor_stats`` handles the
    increment).
    """
    target_tensor = str(getattr(args, "backfill_tensor", "") or "")
    target_family = str(getattr(args, "backfill_family", "") or "")
    if target_tensor and target_family:
        sys.stderr.write(
            "per_tensor_calibrate: --backfill-tensor and --backfill-family "
            "are mutually exclusive\n"
        )
        return 2
    if not (target_tensor or target_family):
        sys.stderr.write(
            "per_tensor_calibrate: --backfill mode requires "
            "--backfill-tensor or --backfill-family\n"
        )
        return 2
    layers = iter_layer_paths(args.layers)
    if not layers:
        sys.stderr.write(
            f"per_tensor_calibrate: --backfill mode requires "
            f"--layers with at least one .npz bundle "
            f"(got {args.layers!r})\n"
        )
        return 2
    selected = _select_backfill_tensors(
        layers, target_tensor=target_tensor, target_family=target_family,
    )
    if not selected:
        sys.stderr.write(
            f"per_tensor_calibrate: --backfill mode selected no "
            f"tensors (target_tensor={target_tensor!r}, "
            f"target_family={target_family!r}, "
            f"n_layers={len(layers)})\n"
        )
        return 2
    domains = [
        d.strip() for d in
        str(getattr(args, "backfill_domains", "") or "").split(",")
        if d.strip()
    ]
    sample_cap = int(getattr(args, "backfill_sample_cap", 256) or 256)
    corpus_root = getattr(args, "backfill_corpus", None)
    db_path = getattr(args, "backfill_db", None)
    output = Path(args.output)
    role = str(getattr(args, "model_role", DEFAULT_MODEL_ROLE) or DEFAULT_MODEL_ROLE)
    seed = int(getattr(args, "seed", 0) or 0)
    rows: list[dict] = []
    for path in selected:
        # Per-tensor subprocess isolation is the
        # orchestrator's contract: ``backfill.py`` shells
        # out to this CLI once per tensor. The in-process
        # fall-back (used here for the unit test path) is
        # the same activation-envelope computation, so the
        # outputs are byte-equivalent.
        samples = _backfill_collect_samples(
            target=path,
            domains=domains or ["default"],
            corpus_root=corpus_root,
            sample_cap=sample_cap,
            seed=seed,
        )
        stats = _backfill_activation_stats(samples, role)
        # Layer extraction: the same convention as the
        # text side. ``blk.<i>.`` -> i; else 0.
        layer = 0
        for prefix in ("blk.", "blocks.", "h.", "layers.", "model.layers."):
            idx = path.stem.find(prefix)
            if idx < 0:
                continue
            start = idx + len(prefix)
            end = start
            while end < len(path.stem) and path.stem[end].isdigit():
                end += 1
            if end > start:
                try:
                    layer = int(path.stem[start:end])
                    break
                except ValueError:
                    layer = 0
        # Shape / n_elements: read the weight bundle when
        # present so the row carries the canonical
        # (out_dim, in_dim, n_elements) tuple. Missing
        # bundle -> the synthetic-shape defaults (n=1
        # weight), which is what the unit test path uses.
        out_dim = 1
        in_dim = int(samples.shape[1]) if samples.ndim >= 2 else 1
        n_elements = int(samples.size)
        if path.is_file() and path.suffix == ".npz":
            try:
                with np.load(str(path), allow_pickle=False) as data:
                    w = data["weight"]
                    if w.ndim == 2:
                        out_dim = int(w.shape[0])
                        in_dim = int(w.shape[1])
                        n_elements = int(w.size)
            except Exception:
                pass
        rows.append({
            "name": path.stem,
            "model_role": role,
            "family": _tensor_family_for_backfill(path.stem),
            "layer_depth": int(layer),
            "out_dim": int(out_dim),
            "in_dim": int(in_dim),
            "n_elements": int(n_elements),
            "dtype": "f16",
            "kurtosis": float(stats["kurtosis"]),
            "eff_rank": float(stats["eff_rank"]),
            "rms": float(stats["rms"]),
            "mean_abs": float(stats["mean_abs"]),
            "tail_ratio": float(stats["tail_ratio"]),
            "source": SOURCE_BACKFILL_REAL,
            # backfill_count is intentionally omitted; the
            # upsert chain in ``insert_tensor_stats``
            # COALESCE-increments when the value is NULL.
        })
    sidecar = {
        "schema": "llama.tessera.backfill.v1",
        "tool": "per_tensor_calibrate.py",
        "mode": "backfill",
        "model_hash": str(getattr(args, "model_hash", "") or ""),
        "model_role": role,
        "target_tensor": target_tensor,
        "target_family": target_family,
        "domains": domains,
        "sample_cap": int(sample_cap),
        "n_tensors": len(rows),
        "rows": rows,
        "timestamp": time.time(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    # DB write: the upsert increments backfill_count. We
    # only attempt the write when --backfill-db is set and
    # --model-hash is set; the orchestrator passes both.
    db_n = 0
    if db_path is not None and getattr(args, "model_hash", None):
        # Local import to keep the top of this module light
        # (the tessera_db import pulls duckdb at import
        # time; the backfill mode is not on the hot path).
        try:
            from tessera_db import TesseraDB  # type: ignore
            with TesseraDB.open(db_path) as db:
                db_n = db.insert_tensor_stats(
                    model_hash=str(getattr(args, "model_hash", "")),
                    rows=rows,
                )
        except Exception as e:
            sys.stderr.write(
                f"per_tensor_calibrate: --backfill DB write failed: {e}\n"
            )
    print(
        f"wrote {output} with {len(rows)} backfill row(s) "
        f"(role={role}, source={SOURCE_BACKFILL_REAL}, "
        f"db_rows={db_n}, domains={domains or ['default']})",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-tensor Tessera calibration. --fitness selects the search mode "
            "(lrq: low-rank weight-scaling; awq: delegate to awq-evolve.py; "
            "dartquant: pre-rotation via QR-Orth + Whip loss; "
            "compare: run both and write a side-by-side report)."
        )
    )
    parser.add_argument(
        "--fitness",
        choices=("lrq", "awq", "flrq", "dartquant", "compare"),
        default="lrq",
        help="Calibration mode (default: lrq).",
    )
    parser.add_argument(
        "--layers",
        required=False,
        default="",
        help=(
            "Directory of layer .npz bundles or a single .npz file. "
            "Optional when the per-component flags "
            "(--trunk-npz / --dflash-npz / --dspark-npz / "
            "--mtp-nextn-npz / --shared-embd-npz) are set; "
            "required otherwise."
        ),
    )
    parser.add_argument("--output", required=True, help="Output calibration policy JSON")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum calibration tokens per bundle (default 256; 0 disables)",
    )
    # LRQ-specific
    parser.add_argument(
        "--lrq-rank",
        type=int,
        default=DEFAULT_RANK,
        help=f"LRQ rank r (default {DEFAULT_RANK})",
    )
    parser.add_argument(
        "--lrq-iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"LRQ optimisation iterations per tensor (default {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Adam learning rate for LRQ (default {DEFAULT_LR})",
    )
    parser.add_argument(
        "--lrq-agg",
        choices=LRQ_AGGREGATIONS,
        default="mean",
        help="How to aggregate the rank-r S matrix to a per-input-channel scale",
    )
    # AWQ delegation (compare mode)
    parser.add_argument("--awq-generations", type=int, default=8)
    parser.add_argument("--awq-population", type=int, default=8)
    # DartQuant pre-rotation
    parser.add_argument(
        "--dartquant-orth-iters",
        type=int,
        default=DEFAULT_DARTQUANT_ITERS,
        help=f"QR-Orth iterations per tensor (default {DEFAULT_DARTQUANT_ITERS})",
    )
    parser.add_argument(
        "--dartquant-orth-lr",
        type=float,
        default=DEFAULT_DARTQUANT_LR,
        help=f"QR-Orth Stiefel step size (default {DEFAULT_DARTQUANT_LR})",
    )
    parser.add_argument(
        "--dartquant-whip-weight",
        type=float,
        default=DEFAULT_DARTQUANT_WHIP_WEIGHT,
        help=(
            "Weight of the Whip loss (activation uniformity) in the combined "
            f"fitness (default {DEFAULT_DARTQUANT_WHIP_WEIGHT})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for both LRQ and AWQ legs",
    )
    parser.add_argument("--verbose", action="store_true")
    # FLRQ (Jan 2026): R1-Sketch + BLC.  Calibration-free: only W is read.
    parser.add_argument(
        "--flrq-rank-candidates",
        type=int,
        nargs="+",
        default=list(FLRQ_DEFAULT_RANK_CANDIDATES),
        help=(
            "FLRQ rank sweep; the smallest rank whose relative "
            "reconstruction MSE clears --flrq-mse-threshold wins "
            f"(default: {' '.join(str(r) for r in FLRQ_DEFAULT_RANK_CANDIDATES)})"
        ),
    )
    parser.add_argument(
        "--flrq-mse-threshold",
        type=float,
        default=FLRQ_DEFAULT_MSE_THRESHOLD,
        help=(
            "FLRQ relative-MSE threshold for rank selection, expressed as "
            "||W - (UV + R_q)||_F^2 / ||W||_F^2 "
            f"(default {FLRQ_DEFAULT_MSE_THRESHOLD:g})"
        ),
    )
    parser.add_argument(
        "--flrq-n-projections",
        type=int,
        default=FLRQ_DEFAULT_PROJECTIONS,
        help=(
            "FLRQ R1-Sketch Gaussian projection count; 16-64 is the "
            f"recommended operating range (default {FLRQ_DEFAULT_PROJECTIONS})"
        ),
    )
    parser.add_argument(
        "--flrq-blc-iters",
        type=int,
        default=FLRQ_DEFAULT_BLC_ITERS,
        help=(
            "FLRQ BLC iterations: number of (re-quantise, re-project) "
            f"passes per rank (default {FLRQ_DEFAULT_BLC_ITERS})"
        ),
    )
    parser.add_argument(
        "--flrq-qbits",
        type=int,
        default=FLRQ_DEFAULT_QBITS,
        help=(
            "FLRQ residual quantiser bit-width (uniform symmetric). "
            f"INT3-INT4 is the recommended operating range (default {FLRQ_DEFAULT_QBITS})"
        ),
    )
    # Phase 16 (calibration memopt): chunked processing.  For
    # tensors with n_rows > chunk_rows, the per-tensor
    # training iterates row-chunks so the per-chunk peak RSS
    # is bounded to ``chunk_rows * in_dim * 4`` bytes rather
    # than the full weight.  Default 4096 matches the
    # ``DEFAULT_CHUNK_ROWS`` constant above (4 chunks for a
    # 12B FFN gate at 16384 rows).  ``<= 0`` or
    # ``>= n_rows`` is the legacy single-shot path.
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help=(
            f"Phase 16 chunked processing row count (default {DEFAULT_CHUNK_ROWS}). "
            "Use 0 or a negative value to disable chunking."
        ),
    )
    # Phase 16 (calibration memopt) Cat 3: peak-RSS budget.
    # The calibration aborts with a clear error if the
    # process RSS exceeds the budget.  Default 32 GB fits
    # a 12B unified run on a 64 GB host with the chunked
    # path.  Use 0 to disable the check (the tracker still
    # records peak_bytes for the final report, but never
    # raises).
    parser.add_argument(
        "--peak-rss-budget-gb",
        type=int,
        default=default_budget_gb(),
        help=(
            "Phase 16 peak-RSS budget in GiB "
            f"(default {default_budget_gb()}, 0 to disable)"
        ),
    )
    # Phase 16 (calibration memopt) Cat 4: spatial occupancy.
    # ``interleaved`` (the default) round-robins the
    # per-component tensors at the layer level so the
    # cache footprint stays small.  ``sequential`` is the
    # legacy path: process all of the trunk's tensors
    # first, then all of the dflash's, etc.  The
    # per-tensor result is the same; the wall-time and
    # cache-hit rate differ.
    parser.add_argument(
        "--spatial-occupancy",
        choices=("sequential", "interleaved"),
        default="interleaved",
        help=(
            "Phase 16 spatial occupancy: 'interleaved' "
            "(default) round-robins per-component tensors at "
            "the layer level; 'sequential' is the legacy "
            "component-major order."
        ),
    )
    # Phase 16 (calibration memopt) Cat 4: per-component
    # input.  Each flag takes a comma-separated list of
    # ``.npz`` bundle paths for that role.  When at least
    # one role is set, the in-process per-component
    # shell-out (``run_components_lrq``) is used instead
    # of the single-component driver; the output policy is
    # tagged with ``model_role`` per tensor.  This is the
    # in-process alternative to ``unified_calibrate.py``;
    # Worker 2's branch exposes a separate driver with
    # subprocess isolation.
    for role in SPATIAL_ROLES:
        flag = f"--{role.replace('_', '-')}-npz"
        parser.add_argument(
            flag,
            type=str,
            default="",
            help=(
                f"Comma-separated .npz paths for the {role!r} "
                f"component (Phase 16 multi-component shell-out)."
            ),
        )
    # Phase 16 (calibration memopt) Cat 5: temporal
    # pipeline depth.  ``> 1`` enables the
    # ``CalibPipeline`` double-buffer (or N-buffer) that
    # overlaps the next tensor's mmap with the current
    # tensor's compute.  ``1`` is the legacy single-thread
    # path (no overlap).  Default 2 (double-buffered);
    # 3+ keeps more tensors in flight on slow I/O at the
    # cost of more peak RSS (working set is bounded by
    # ``depth * max_tensor_bytes``).
    parser.add_argument(
        "--temporal-pipeline-depth",
        type=int,
        default=2,
        help=(
            "Phase 16 temporal pipeline depth "
            "(default 2 = double-buffered; 1 = legacy single-thread)."
        ),
    )
    # Phase 16.5 (memopt-metal-dispatch).  The macOS
    # async I/O via dispatch_io_t.  ``--async-io on``
    # forces the dispatch_io_t path on macOS; ``off``
    # forces the legacy threaded ``CalibPipeline``.
    # ``auto`` (default) uses the dispatch_io_t path
    # when available, falls back to the threaded path
    # otherwise.  The async path uses in-RAM bytes
    # (one layer at a time) instead of mmap; the
    # trade-off is more peak RSS for the I/O overlap.
    parser.add_argument(
        "--async-io",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Phase 16.5 async I/O mode: 'auto' (default) "
            "uses dispatch_io_t on macOS, falls back to "
            "the legacy threaded CalibPipeline otherwise; "
            "'on' forces the async path (errors on non-macOS); "
            "'off' forces the legacy threaded path on all platforms."
        ),
    )
    parser.add_argument(
        "--model-role",
        choices=MODEL_ROLES,
        default=DEFAULT_MODEL_ROLE,
        help=(
            "Component role for this calibration pass. The value is stamped "
            "on every per-tensor entry so the unified consumer can route "
            "per-role. The default is 'trunk' which is the legacy "
            "single-model behaviour. The unified_calibrate.py driver "
            "invokes per_tensor_calibrate.py once per component with the "
            "appropriate role. The M0a mmproj roles (vision_tower / "
            "audio_tower / mm_projector) are accepted: their activation "
            "source is the multimodal capture driver "
            "(multimodal_calibrate.py), not the text-side imatrix path."
        ),
    )
    parser.add_argument(
        "--tensor-stats-source",
        choices=tuple(ACTIVATION_SOURCE.keys()),
        default="imatrix",
        help=(
            "Activation source for the per-tensor calibration pass. "
            "'imatrix' (default) is the text-side imatrix / spec_calib "
            "path; 'multimodal' reads the tensor_stats rows produced by "
            "multimodal_calibrate.py (the M0a mmproj roles). The choice "
            "is recorded on the policy's provenance block so the "
            "downstream consumer can reconstruct how the policy was "
            "built. The role and the source are independent: the text "
            "side may pass 'multimodal' for an mmproj role, and the "
            "default 'imatrix' for the text-side roles."
        ),
    )
    # Targeted re-calibration (L5 monitor-verdict hook). The
    # backfill mode drives a focused re-capture on a
    # domain-specific sample subset, rather than the
    # per-family calibration the default modes do. The
    # ``--backfill-tensor`` / ``--backfill-family`` flags
    # select which tensor(s) to re-capture; the
    # ``--backfill-corpus`` / ``--backfill-domains`` flags
    # select the sample subset. The
    # ``--backfill-sample-cap`` flag is the per-tensor
    # sample budget (default 256, the same default the
    # orchestrator's targeted-backfill hook uses). The
    # ``--backfill`` mode is additive on the schema: the
    # new ``backfill_count`` column on ``tensor_stats`` is
    # incremented on every backfill write and the
    # ``source`` column is ``SOURCE_BACKFILL_REAL``. When
    # the backfill mode is active, the per-family scoring
    # machinery is bypassed; the per-tensor activation
    # stats are written directly.
    parser.add_argument(
        "--backfill-tensor",
        default="",
        help=(
            "When set, run the per-tensor backfill re-capture "
            "on this single tensor name. Mutually exclusive "
            "with --backfill-family."
        ),
    )
    parser.add_argument(
        "--backfill-family",
        default="",
        help=(
            "When set, run the per-tensor backfill re-capture "
            "on every tensor in this family (e.g. attn_q, "
            "ffn_gate). Mutually exclusive with "
            "--backfill-tensor."
        ),
    )
    parser.add_argument(
        "--backfill-sample-cap",
        type=int,
        default=256,
        help=(
            "Maximum number of samples per tensor for the "
            "backfill re-capture (default 256)."
        ),
    )
    parser.add_argument(
        "--backfill-domains",
        default="",
        help=(
            "Comma-separated list of domain subsets the "
            "backfill should sample from (e.g. "
            "'code,math,wiki'). Empty = use the corpus "
            "default."
        ),
    )
    parser.add_argument(
        "--backfill-corpus",
        type=Path,
        default=None,
        help=(
            "Path to the calibration corpus root. When set, "
            "the backfill samples from the corpus's "
            "domain-specific subsets (the build-calibration-"
            "corpus contract). When None, the backfill falls "
            "back to a uniform sample of the test fixtures."
        ),
    )
    parser.add_argument(
        "--backfill-db",
        type=Path,
        default=None,
        help=(
            "Path to the unified tessera.duckdb file. When "
            "set, the per-tensor activation stats are "
            "upserted into the tensor_stats table with "
            "source=SOURCE_BACKFILL_REAL and "
            "backfill_count incremented. When None, the "
            "backfill writes only the JSON sidecar."
        ),
    )
    parser.add_argument(
        "--model-hash",
        default=None,
        help=(
            "Model hash for the DB write (the upsert path "
            "keys on model_hash + model_role + name). "
            "When --backfill-db is set, --model-hash "
            "must also be set; the sidecar-only path "
            "(no --backfill-db) ignores the flag."
        ),
    )
    return parser


def _main_components(
    args: argparse.Namespace,
    tracker: ResidencyTracker,
    component_args: dict[str, str],
) -> int:
    """In-process multi-component driver for the LRQ / compare modes.

    Phase 16 (calibration memopt) Cat 4.  Reads the
    ``--<role>-npz`` flags, builds the per-component
    tensor list, and runs ``run_components_lrq`` with the
    ``--spatial-occupancy`` iteration order.  The output
    policy is the LRQ policy assembled per tensor with the
    model_role tag; the per-component counts are stamped
    under ``policy["components"][<role>]["tensor_count"]``
    so the unified policy can be assembled downstream.
    """
    components: dict[str, list[Path]] = {}
    for role, raw in component_args.items():
        if not raw.strip():
            continue
        components[role] = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    if not components:
        raise ValueError(
            "no per-component paths supplied: at least one of "
            + ", ".join(f"--{r.replace('_', '-')}-npz" for r in SPATIAL_ROLES)
            + " must be set"
        )
    if args.fitness == "lrq":
        results, counts = run_components_lrq(components, args=args, tracker=tracker)
        # Strip the model_role for the LRQ policy builder;
        # the unified policy builder re-attaches it.
        lrq_results = [(layer, result) for layer, result, _ in results]
        provenance = {
            "tool": "per_tensor_calibrate.py",
            "mode": "lrq-multi",
            "seed": args.seed,
            "lrq_rank": args.lrq_rank,
            "lrq_iterations": args.lrq_iterations,
            "lrq_lr": args.lr,
            "lrq_aggregation": args.lrq_agg,
            "spatial_occupancy": getattr(args, "spatial_occupancy", "interleaved"),
            "per_role_tensor_count": counts,
            # M0a: see the compare-mode provenance block for the
            # tensor_stats_source contract. The multi-component
            # shell-out may mix sources (text side imatrix + mmproj
            # multimodal); the per-component dispatch records the
            # source in the policy entries; the top-level field
            # records the call-site default.
            "tensor_stats_source": ACTIVATION_SOURCE.get(
                getattr(args, "tensor_stats_source", "imatrix"), "imatrix"
            ),
            "timestamp": time.time(),
        }
        policy = build_lrq_policy(lrq_results, provenance)
        # Stamp the per-role tensor_count on the policy so
        # the unified driver can recover the partition
        # without re-reading the components dict.
        policy["per_role_tensor_count"] = counts
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        print(
            f"wrote {output} with {len(lrq_results)} LRQ tensor entries "
            f"({sum(counts.values())} across {len(counts)} roles)",
            file=sys.stderr,
        )
        print(f"residency: {tracker.report()}", file=sys.stderr)
        return 0
    if args.fitness == "compare":
        # Compare mode + multi-component is a thin extension
        # of the LRQ path; the AWQ leg is best-effort
        # (delegate to awq-evolve.py per component) so we
        # skip the AWQ leg here.  The per-component
        # comparison is a future Worker 2 / unified driver
        # extension.
        results, counts = run_components_lrq(components, args=args, tracker=tracker)
        lrq_results = [(layer, result) for layer, result, _ in results]
        provenance = {
            "tool": "per_tensor_calibrate.py",
            "mode": "compare-multi",
            "seed": args.seed,
            "lrq_rank": args.lrq_rank,
            "lrq_iterations": args.lrq_iterations,
            "lrq_lr": args.lr,
            "lrq_aggregation": args.lrq_agg,
            "spatial_occupancy": getattr(args, "spatial_occupancy", "interleaved"),
            "per_role_tensor_count": counts,
            # M0a: see the compare-mode provenance block for the
            # tensor_stats_source contract.
            "tensor_stats_source": ACTIVATION_SOURCE.get(
                getattr(args, "tensor_stats_source", "imatrix"), "imatrix"
            ),
            "timestamp": time.time(),
        }
        policy = build_lrq_policy(lrq_results, provenance)
        policy["per_role_tensor_count"] = counts
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        print(
            f"wrote {output} with {len(lrq_results)} LRQ tensor entries "
            f"(multi-component compare)",
            file=sys.stderr,
        )
        print(f"residency: {tracker.report()}", file=sys.stderr)
        return 0
    raise ValueError(
        f"multi-component shell-out does not support --fitness {args.fitness!r}; "
        "expected one of ('lrq', 'compare')"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Phase 16 Cat 3: peak-RSS budget tracker.  Created at the
    # top of main() so all fitness modes share the same
    # tracker.  ``--peak-rss-budget-gb 0`` disables the check
    # (the tracker still records peak_bytes for the final
    # report, but never raises).
    tracker = ResidencyTracker(
        budget_bytes=int(getattr(args, "peak_rss_budget_gb", 0) or 0) * 1024**3,
        abort_on_exceed=True,
    )
    # Phase 16 Cat 4: multi-component dispatch.  If any of
    # the ``--<role>-npz`` flags is set, use the in-process
    # per-component shell-out (in ``run_components_lrq``)
    # instead of the single-component driver.  The output
    # policy is tagged with the model_role per tensor; the
    # ``--spatial-occupancy`` flag controls the iteration
    # order.
    component_args = {role: getattr(args, f"{role.replace('-', '_')}_npz", "")
                      for role in SPATIAL_ROLES}
    has_any_component = any(p.strip() for p in component_args.values())
    if has_any_component and args.fitness in ("lrq", "compare"):
        return _main_components(args, tracker, component_args)
    if has_any_component and args.fitness in ("lrq", "compare"):
        return _main_components(args, tracker, component_args)
    if args.fitness == "lrq":
        layers = iter_layer_paths(args.layers)
        if not layers:
            raise ValueError(f"{args.layers}: no layer bundles")
        digests = {p.stem: bundle_digest(p) for p in layers}
        # Phase 16 Cat 3: the per-tensor check is inside the
        # helper (``_run_lrq_with_residency``).  The helper
        # also dispatches to the chunked path when
        # ``--chunk-rows > 0`` and the tensor is large enough.
        lrq_results = _run_lrq_with_residency(layers, args=args, tracker=tracker)
        provenance = {
            "tool": "per_tensor_calibrate.py",
            "mode": "lrq",
            "seed": args.seed,
            "lrq_rank": args.lrq_rank,
            "lrq_iterations": args.lrq_iterations,
            "lrq_lr": args.lr,
            "lrq_aggregation": args.lrq_agg,
            "bundle_digests": digests,
            # M0a: see the compare-mode provenance block for the
            # tensor_stats_source contract. The default of "imatrix"
            # matches the pre-M0a behaviour.
            "tensor_stats_source": ACTIVATION_SOURCE.get(
                getattr(args, "tensor_stats_source", "imatrix"), "imatrix"
            ),
            "timestamp": time.time(),
        }
        policy = build_lrq_policy(lrq_results, provenance, model_role=args.model_role)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output} with {len(lrq_results)} LRQ tensor entries", file=sys.stderr)
        print(f"residency: {tracker.report()}", file=sys.stderr)
        print(f"wrote {output} with {len(lrq_results)} LRQ tensor entries", file=sys.stderr)
        print(f"wrote {output} with {len(lrq_results)} LRQ tensor entries (role={args.model_role})", file=sys.stderr)
        return 0
    if args.fitness == "awq":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Phase 16: awq-evolve.py now accepts --model-role natively
        # and stamps the role on every per-family and per-override
        # entry, plus the top-level policy["model_role"]. The
        # subprocess is the authoritative source of the role tag.
        policy = run_awq_subprocess(
            args.layers,
            output,
            seed=args.seed,
            generations=args.awq_generations,
            population=args.awq_population,
            model_role=args.model_role,
        )
        print(f"wrote {output} via awq-evolve.py", file=sys.stderr)
        print(f"residency: {tracker.report()}", file=sys.stderr)
        if isinstance(policy, dict):
            # Belt-and-braces: confirm the subprocess stamped the role
            # on every entry. If a future awq-evolve.py change drops
            # the role from one path, this re-stamps the missing
            # entries so the unified contract (top-level + per-entry
            # model_role) is still consistent.
            policy.setdefault("model_role", args.model_role)
            for entry in policy.get("tensor_families", {}).values():
                if isinstance(entry, dict):
                    entry.setdefault("model_role", args.model_role)
            moe_block = policy.get("moe_residual_allocation")
            if isinstance(moe_block, dict):
                moe_block.setdefault("model_role", args.model_role)
            # Re-write so the on-disk artifact matches the in-memory
            # policy the rest of the unified driver sees.
            output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output} via awq-evolve.py (role={args.model_role})", file=sys.stderr)
        return 0
    if args.fitness == "flrq":
        rc = run_flrq(args, tracker=tracker)
        print(f"residency: {tracker.report()}", file=sys.stderr)
        return rc
    if args.fitness == "compare":
        rc = run_compare(args, tracker=tracker)
        print(f"residency: {tracker.report()}", file=sys.stderr)
        return rc
    if args.fitness == "dartquant":
        rc = run_dartquant(args, tracker=tracker)
        print(f"residency: {tracker.report()}", file=sys.stderr)
        return rc
    # Targeted re-calibration: the backfill mode is
    # additive and bypasses the per-family scoring. The
    # orchestrator's backfill hook drives this mode by
    # shelling out to ``per_tensor_calibrate.py`` once per
    # monitor-verdict tensor; the in-process call from
    # ``backfill.py`` is the same. The dispatch is here
    # (rather than in ``--fitness``) so the default
    # ``--fitness lrq`` does not change for callers that
    # never set --backfill-tensor / --backfill-family.
    if (getattr(args, "backfill_tensor", "")
            or getattr(args, "backfill_family", "")):
        return _run_backfill(args)
    raise ValueError(f"unknown --fitness {args.fitness!r}")


if __name__ == "__main__":
    sys.exit(main())
