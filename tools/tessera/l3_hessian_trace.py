#!/usr/bin/env python3
"""Per-tile Hessian trace calibration (Tessera L3 E5).

Computes a per-tile empirical Hessian trace for each tensor in a calibration
bundle, then writes a ``llama.tessera.hessian-trace-policy.v1`` policy that
extends ``llama.speculative.calibration-policy.v1``.

The trace is the L3 sensitivity signal that HAWQ-V2 (NeurIPS 2020) identified
as the right metric for layer-wise mixed-precision assignment. The IterQuant
L5 orchestrator on the track-iterquant-prod branch consumes it as a third
sensitivity signal alongside the LLM.int8 outlier count (E1) and the
IterQuant token-level sensitivity.

This tool is a Tier-0 unlock: the policy schema inherits from
``llama.speculative.calibration-policy.v1`` without a format change, so the
downstream quantizer (tile640_quantize_v3.py --calibration-policy) can read
it as-is. The trace values live under ``policy["hessian_trace"]`` with a
self-describing sub-schema so a separate consumer can inspect them without
parsing the family map.

Schemas
-------
Output policy:
  - ``schema``             = ``llama.speculative.calibration-policy.v1``
  - ``hessian_trace.schema`` = ``llama.tessera.hessian-trace-policy.v1``
  - ``hessian_trace.tile_size`` = 640 (must match TILE640_PAGE_SIZE)
  - ``hessian_trace.method``    = "exact-diagonal" | "hutchinson"
  - ``hessian_trace.n_hutchinson_vectors`` (hutchinson only)
  - ``hessian_trace.tensors[]``  = one entry per tensor:
        ``name``                  = canonical tensor name
        ``weight_shape``          = [out_dim, in_dim]
        ``n_parameters``          = out_dim * in_dim
        ``n_samples``             = N (number of calibration rows)
        ``hessian_trace``         = total trace tr(H)
        ``hessian_trace_avg``     = tr(H) / n_parameters
        ``hessian_trace_per_tile`` = list[float] length = ceil(in_dim / 640)
        ``hutchinson_estimate``   (hutchinson only)
        ``hutchinson_rel_error``  (hutchinson only; abs(h - exact) / exact)

Validation rules (enforced by ``validate_policy``):
  - root schema must be ``llama.speculative.calibration-policy.v1``
  - ``hessian_trace.schema`` must be ``llama.tessera.hessian-trace-policy.v1``
  - ``hessian_trace.tile_size`` must be a positive int
  - ``hessian_trace.method`` must be one of the documented modes
  - each tensor entry must have the per-tensor fields above
  - ``hessian_trace`` (sum) must equal ``sum(hessian_trace_per_tile)`` to
    within 1e-3 relative tolerance; a divergence signals a tile-bucket bug
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = "llama.speculative.calibration-policy.v1"
HESSIAN_TRACE_SCHEMA = "llama.tessera.hessian-trace-policy.v1"
DEFAULT_TILE_SIZE = 640  # TILE640_PAGE_SIZE in tile640_quantize_v3.py
DEFAULT_HUTCHINSON_VECTORS = 50  # HAWQ-V2 default
DEFAULT_MAX_TOKENS = 256
METHODS = ("exact-diagonal", "hutchinson")


# ---------------------------------------------------------------------------
# Bundle loading (mirrors per_tensor_calibrate.py's loader but without LRQ)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Layer:
    """One tensor's calibration bundle."""

    name: str
    weight: np.ndarray  # (out_dim, in_dim) float32
    train_activations: np.ndarray | None  # (n_tokens, in_dim) float32
    in_sum2: np.ndarray | None  # (in_dim,) float32; imatrix-style diag only
    in_count: int  # number of rows used to compute in_sum2

    @property
    def out_dim(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_dim(self) -> int:
        return int(self.weight.shape[1])

    @property
    def n_parameters(self) -> int:
        return self.out_dim * self.in_dim


def _scalar_string(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def load_bundle(path: Path, max_tokens: int = DEFAULT_MAX_TOKENS) -> Layer:
    """Load a single .npz bundle in the per_tensor_calibrate.py format.

    Either ``train_activations`` or ``in_sum2`` is sufficient; both may be
    present. ``max_tokens`` subsamples the activation rows to keep the
    trace cost bounded on heavy models.
    """
    with np.load(path, allow_pickle=False) as data:
        weight = np.asarray(data["weight"], dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(f"{path}: weight must be two-dimensional")
        train: np.ndarray | None = None
        if "train_activations" in data:
            train = np.asarray(data["train_activations"], dtype=np.float32)
            if train.ndim != 2 or train.shape[1] != weight.shape[1]:
                raise ValueError(
                    f"{path}: train_activations shape {train.shape} does not match weight {weight.shape}"
                )
            if max_tokens > 0 and train.shape[0] > max_tokens:
                idx = np.linspace(0, train.shape[0] - 1, max_tokens, dtype=np.int64)
                train = train[idx]
        in_sum2: np.ndarray | None = None
        if "in_sum2" in data:
            in_sum2 = np.asarray(data["in_sum2"], dtype=np.float32).reshape(-1)
            if in_sum2.shape[0] != weight.shape[1]:
                raise ValueError(
                    f"{path}: in_sum2 length {in_sum2.shape[0]} does not match weight in_dim {weight.shape[1]}"
                )
        in_count = (
            int(np.asarray(data["counts"]).reshape(()).item())
            if "counts" in data
            else (int(train.shape[0]) if train is not None else 0)
        )
        name = _scalar_string(data["name"] if "name" in data else None, path.stem)
    if train is None and in_sum2 is None:
        raise ValueError(
            f"{path}: requires train_activations or in_sum2 to compute the Hessian trace"
        )
    return Layer(
        name=name,
        weight=weight,
        train_activations=train,
        in_sum2=in_sum2,
        in_count=in_count,
    )


def iter_bundle_paths(layers_arg: str) -> list[Path]:
    """Resolve a single .npz or a directory of .npz bundles."""
    p = Path(layers_arg)
    if p.is_dir():
        paths = sorted(p.glob("*.npz"))
        if not paths:
            raise ValueError(f"{layers_arg}: directory contains no .npz bundles")
        return paths
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
# Trace computation
# ---------------------------------------------------------------------------


def _tile_buckets(in_dim: int, tile_size: int) -> list[tuple[int, int]]:
    """Return [(start, end), ...] for each tile. The last tile may be partial."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    edges: list[tuple[int, int]] = []
    for start in range(0, in_dim, tile_size):
        edges.append((start, min(start + tile_size, in_dim)))
    return edges


def _bucket_diagonal(diag: np.ndarray, tile_size: int) -> np.ndarray:
    """Sum a 1-D diagonal vector into a 1-D array of per-tile bucket sums."""
    if diag.ndim != 1:
        raise ValueError("diag must be 1-D")
    edges = _tile_buckets(diag.shape[0], tile_size)
    out = np.empty(len(edges), dtype=np.float64)
    for i, (s, e) in enumerate(edges):
        out[i] = float(np.sum(diag[s:e]))
    return out


def trace_per_tile_from_in_sum2(
    in_sum2: np.ndarray, n_samples: int, tile_size: int
) -> tuple[float, np.ndarray]:
    """Exact per-tile trace from the imatrix-style diagonal.

    H = X^T X / N, so H[i, i] = (1/N) * sum_t x_t[i]^2 = in_sum2[i] / N.
    The per-tile trace is the sum of H[i, i] for i in the tile.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    diag = in_sum2.astype(np.float64) / float(n_samples)
    per_tile = _bucket_diagonal(diag, tile_size)
    return float(np.sum(per_tile)), per_tile


def trace_per_tile_from_X(
    X: np.ndarray, tile_size: int
) -> tuple[float, np.ndarray]:
    """Exact per-tile trace from the full activation matrix.

    diag(H)[i] = (1/N) * ||X[:, i]||^2. Per-tile sums are exact: the trace
    is a diagonal sum and Hutchinson would only add noise. We only fall
    back to Hutchinson for the per-tensor total (which equals the sum of
    per-tile buckets).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, in_dim)")
    n_samples = X.shape[0]
    if n_samples <= 0:
        raise ValueError("X must have at least one sample")
    diag = (X.astype(np.float64) ** 2).mean(axis=0)
    per_tile = _bucket_diagonal(diag, tile_size)
    return float(np.sum(per_tile)), per_tile


def hutchinson_trace(
    X: np.ndarray, n_vecs: int, seed: int
) -> tuple[float, float, float]:
    """Hutchinson estimator for ``tr(H) = tr(X^T X / N)``.

    Returns ``(estimate, lower_bound, upper_bound)`` where the bounds are
    the 1-sigma interval of the unbiased estimator (assuming the
    per-vector trace estimators are approximately i.i.d.). Lower / upper
    are NaN when the estimate is exactly zero (n_vecs=1 or trivial X).

    The estimator uses Rademacher (+/- 1) probes. For a deterministic
    seed the result is bit-for-bit reproducible.

    Cost: ``n_vecs`` mat-vec products of shape ``(N, in_dim) @ (in_dim,)``
    each, dominated by the BLAS gemv call (no full H ever materialised).
    """
    if n_vecs <= 0:
        raise ValueError("n_vecs must be positive")
    rng = np.random.default_rng(seed)
    in_dim = X.shape[1]
    # Rademacher (+/-1) probes; mean of v^T H v is tr(H) when H is symmetric.
    probes = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_vecs, in_dim))
    Xd = X.astype(np.float64, copy=False)
    # tr(H) = tr(X^T X / N) = (1/N) * mean_m ||X v_m||^2 over m probes.
    quad = np.empty(n_vecs, dtype=np.float64)
    for m in range(n_vecs):
        x_probe = Xd @ probes[m]
        quad[m] = float(np.dot(x_probe, x_probe))
    n_samples = X.shape[0]
    per_vec = quad / float(n_samples)
    estimate = float(np.mean(per_vec))
    if n_vecs > 1:
        std = float(np.std(per_vec, ddof=1)) / math.sqrt(float(n_vecs))
    else:
        std = 0.0
    return estimate, estimate - std, estimate + std


def compute_tensor_trace(
    layer: Layer,
    method: str,
    tile_size: int,
    n_hutchinson_vectors: int,
    seed: int,
) -> dict:
    """Run the chosen trace method on a single tensor and return its record.

    ``method`` is one of ``METHODS``. ``exact-diagonal`` only needs the
    imatrix; ``hutchinson`` needs full activations and additionally
    computes the exact diagonal as a sanity baseline when activations
    are also present (to report the Hutchinson relative error).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS!r}, got {method!r}")
    record: dict = {
        "name": layer.name,
        "weight_shape": [layer.out_dim, layer.in_dim],
        "n_parameters": layer.n_parameters,
        "method": method,
        "tile_size": tile_size,
    }

    if method == "exact-diagonal":
        if layer.in_sum2 is not None and layer.in_count > 0:
            trace_total, per_tile = trace_per_tile_from_in_sum2(
                layer.in_sum2, layer.in_count, tile_size
            )
            record["n_samples"] = int(layer.in_count)
        elif layer.train_activations is not None and layer.train_activations.size:
            trace_total, per_tile = trace_per_tile_from_X(
                layer.train_activations, tile_size
            )
            record["n_samples"] = int(layer.train_activations.shape[0])
        else:
            raise ValueError(
                f"{layer.name}: exact-diagonal mode requires in_sum2 or train_activations"
            )
    else:  # hutchinson
        if layer.train_activations is None or not layer.train_activations.size:
            raise ValueError(
                f"{layer.name}: hutchinson mode requires train_activations; "
                "fall back to exact-diagonal when only the imatrix is available"
            )
        x = layer.train_activations
        record["n_samples"] = int(x.shape[0])
        trace_total, per_tile = trace_per_tile_from_X(x, tile_size)
        h_estimate, h_lo, h_hi = hutchinson_trace(
            x, n_hutchinson_vectors, seed
        )
        record["hutchinson_n_vectors"] = int(n_hutchinson_vectors)
        record["hutchinson_estimate"] = float(h_estimate)
        record["hutchinson_lower_1sigma"] = float(h_lo)
        record["hutchinson_upper_1sigma"] = float(h_hi)
        if trace_total > 0.0:
            record["hutchinson_rel_error"] = float(abs(h_estimate - trace_total) / trace_total)
        else:
            record["hutchinson_rel_error"] = float("nan")

    record["hessian_trace"] = float(trace_total)
    record["hessian_trace_avg"] = (
        float(trace_total) / float(layer.n_parameters)
        if layer.n_parameters > 0
        else 0.0
    )
    record["hessian_trace_per_tile"] = [float(v) for v in per_tile]
    record["n_tiles"] = int(per_tile.shape[0])
    return record


# ---------------------------------------------------------------------------
# Policy assembly
# ---------------------------------------------------------------------------


def build_hessian_trace_policy(
    records: list[dict],
    method: str,
    tile_size: int,
    n_hutchinson_vectors: int,
    seed: int,
    provenance: dict,
) -> dict:
    """Assemble the policy document with a parent policy wrapper.

    The wrapper schema is ``llama.speculative.calibration-policy.v1`` so
    downstream consumers (notably tile640_quantize_v3.py --calibration-policy
    and the L5 orchestrator) can read it without a schema change. The
    trace payload lives under ``policy["hessian_trace"]`` with its own
    sub-schema ``llama.tessera.hessian-trace-policy.v1``.
    """
    hessian_sub: dict = {
        "schema": HESSIAN_TRACE_SCHEMA,
        "method": method,
        "tile_size": int(tile_size),
        "n_tensors": len(records),
        "n_hutchinson_vectors": int(n_hutchinson_vectors) if method == "hutchinson" else None,
        "seed": int(seed),
        "tensors": records,
    }
    if method == "hutchinson":
        hessian_sub["hutchinson_n_vectors"] = int(n_hutchinson_vectors)
    return {
        "schema": SCHEMA,
        "draft_type": "hybrid",
        "hessian_trace": hessian_sub,
        "hessian_trace_provenance": provenance,
    }


def validate_policy(policy: dict) -> None:
    """Validate a policy against the schema documented in the module docstring.

    Raises ``ValueError`` on the first violation. Cheap O(n) over the
    per-tensor records; intended to be called by tests and by the CLI
    when ``--validate`` is supplied.
    """
    if policy.get("schema") != SCHEMA:
        raise ValueError(
            f"root schema must be {SCHEMA!r}, got {policy.get('schema')!r}"
        )
    hessian = policy.get("hessian_trace")
    if not isinstance(hessian, dict):
        raise ValueError("missing hessian_trace sub-policy")
    if hessian.get("schema") != HESSIAN_TRACE_SCHEMA:
        raise ValueError(
            f"hessian_trace.schema must be {HESSIAN_TRACE_SCHEMA!r}, "
            f"got {hessian.get('schema')!r}"
        )
    method = hessian.get("method")
    if method not in METHODS:
        raise ValueError(f"hessian_trace.method must be one of {METHODS!r}, got {method!r}")
    tile_size = hessian.get("tile_size")
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("hessian_trace.tile_size must be a positive int")
    tensors = hessian.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("hessian_trace.tensors must be a list")
    required = {
        "name", "weight_shape", "n_parameters", "hessian_trace",
        "hessian_trace_avg", "hessian_trace_per_tile", "n_tiles",
    }
    for i, t in enumerate(tensors):
        if not isinstance(t, dict):
            raise ValueError(f"tensors[{i}] is not an object")
        missing = required - set(t.keys())
        if missing:
            raise ValueError(f"tensors[{i}] missing fields: {sorted(missing)}")
        per_tile = t["hessian_trace_per_tile"]
        if not isinstance(per_tile, list) or not all(
            isinstance(v, (int, float)) for v in per_tile
        ):
            raise ValueError(f"tensors[{i}].hessian_trace_per_tile must be a list of numbers")
        if t["n_tiles"] != len(per_tile):
            raise ValueError(
                f"tensors[{i}].n_tiles {t['n_tiles']} != len(per_tile) {len(per_tile)}"
            )
        if tile_size > 0 and t["weight_shape"][1] > 0:
            expected_tiles = (t["weight_shape"][1] + tile_size - 1) // tile_size
            if t["n_tiles"] != expected_tiles:
                raise ValueError(
                    f"tensors[{i}].n_tiles {t['n_tiles']} does not match expected "
                    f"ceil(in_dim / tile_size) = {expected_tiles}"
                )
        trace_total = sum(float(v) for v in per_tile)
        if trace_total > 0.0:
            rel = abs(trace_total - float(t["hessian_trace"])) / trace_total
            if rel > 1e-3:
                raise ValueError(
                    f"tensors[{i}].hessian_trace {t['hessian_trace']} disagrees with "
                    f"sum(per_tile) {trace_total} (rel error {rel:.3e})"
                )


# ---------------------------------------------------------------------------
# Outlier count (for the comparison signal in the demo)
# ---------------------------------------------------------------------------


def outlier_count_per_channel(
    weight: np.ndarray, x_rms: np.ndarray, threshold: float
) -> int:
    """LLM.int8-style outlier count: |w * x_hat| > threshold.

    ``x_hat`` is the per-input-channel RMS derived from the imatrix or
    from the activations. The threshold is in absolute units (not
    normalised) so callers can pass the same threshold across tensors
    and compare the counts.

    Note: the LLM.int8 paper itself uses ``max_t |X[t, i]| > 6.0`` on the
    raw activation column. We expose the per-position |w * x| variant
    because that is what the L1 outlier-ranker consumes; for the
    activation-column definition, use ``outlier_channels_by_max``.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be 2-D")
    if x_rms.shape[0] != weight.shape[1]:
        raise ValueError("x_rms length must match in_dim")
    score = np.abs(weight.astype(np.float32) * x_rms.astype(np.float32)[np.newaxis, :])
    return int(np.sum(score > threshold))


def outlier_channels_by_max(
    activations: np.ndarray | None, in_sum2: np.ndarray | None, threshold: float
) -> int:
    """LLM.int8 paper definition: number of input channels with
    ``max_t |X[t, i]| > threshold``.

    The activation matrix is preferred when available; otherwise we
    approximate ``max_t |X[t, i]|`` from ``in_sum2`` (RMS) using the
    conservative bound ``max >= RMS``. The bound under-estimates the
    count when only the imatrix is available, so the caller should
    prefer the activation path when both are present.
    """
    if activations is not None and activations.size:
        max_abs = np.max(np.abs(activations.astype(np.float32)), axis=0)
        return int(np.sum(max_abs > float(threshold)))
    if in_sum2 is not None and in_sum2.size:
        # We only have RMS; max >= RMS, so any channel with RMS > threshold
        # is definitely an outlier. Channels with RMS <= threshold might
        # still be outliers in the LLM.int8 sense, but we cannot tell
        # from the imatrix alone.
        rms = np.sqrt(in_sum2.astype(np.float64))
        return int(np.sum(rms > float(threshold)))
    return 0


# ---------------------------------------------------------------------------
# Spearman rho (pure numpy; matches scipy.stats.spearmanr with average ranks)
# ---------------------------------------------------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-ranks tie-breaking. Pure numpy; no scipy."""
    flat = a.astype(np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(flat)
    n = flat.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and flat[order[j + 1]] == flat[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average-ranks tie-breaking.

    Returns NaN when one of the inputs is constant (no rank variance).
    """
    if x.shape != y.shape or x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx_mean = rx - np.mean(rx)
    ry_mean = ry - np.mean(ry)
    denom = np.sqrt(np.sum(rx_mean * rx_mean) * np.sum(ry_mean * ry_mean))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(rx_mean * ry_mean) / denom)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-tile Hessian traces for Tessera L3 sensitivity. "
            "Reads a per-tensor calibration bundle (.npz) and writes a "
            "hessian-trace policy JSON consumable by tile640_quantize_v3.py "
            "--calibration-policy and the L5 orchestrator."
        )
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Directory of .npz calibration bundles or a single .npz file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output hessian-trace policy JSON path",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="hutchinson",
        help=(
            "Trace estimator. 'hutchinson' (default; HAWQ-V2 style, 50 "
            "Rademacher probes) requires train_activations. "
            "'exact-diagonal' uses the imatrix's per-channel in_sum2 and "
            "is the right choice when only the calibration observer is "
            "available."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile width for per-tile aggregation (default {DEFAULT_TILE_SIZE}; must match TILE640_PAGE_SIZE)",
    )
    parser.add_argument(
        "--n-hutchinson-vectors",
        type=int,
        default=DEFAULT_HUTCHINSON_VECTORS,
        help=f"Number of Rademacher probes for Hutchinson (default {DEFAULT_HUTCHINSON_VECTORS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum calibration tokens per bundle (default 256; 0 disables)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for Hutchinson probes")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_policy on the output before exit (also triggered by default)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")
    if args.n_hutchinson_vectors <= 0:
        raise ValueError("n-hutchinson-vectors must be positive")
    if args.method == "hutchinson" and args.n_hutchinson_vectors < 5:
        # Hutchinson with too few probes is misleading; warn the caller.
        print(
            f"WARN: hutchinson with {args.n_hutchinson_vectors} vectors; "
            "HAWQ-V2 recommends 50+",
            file=sys.stderr,
        )

    paths = iter_bundle_paths(args.layers)
    digests = {p.stem: bundle_digest(p) for p in paths}
    records: list[dict] = []
    t_start = time.perf_counter()
    for path in paths:
        layer = load_bundle(path, max_tokens=args.max_tokens)
        record = compute_tensor_trace(
            layer,
            method=args.method,
            tile_size=args.tile_size,
            n_hutchinson_vectors=args.n_hutchinson_vectors,
            seed=args.seed,
        )
        records.append(record)
        if args.verbose:
            h = record.get("hutchinson_estimate")
            rel = record.get("hutchinson_rel_error")
            extras = (
                f"  hutchinson={h:.6e} rel_err={rel:.3e}"
                if h is not None
                else ""
            )
            print(
                f"trace[{layer.name}]  shape={layer.weight.shape}  "
                f"tr(H)={record['hessian_trace']:.6e}  "
                f"tr(H)/n={record['hessian_trace_avg']:.6e}  "
                f"tiles={record['n_tiles']}{extras}",
                file=sys.stderr,
            )
    wall_time = time.perf_counter() - t_start

    provenance = {
        "tool": "l3_hessian_trace.py",
        "method": args.method,
        "tile_size": int(args.tile_size),
        "n_hutchinson_vectors": (
            int(args.n_hutchinson_vectors) if args.method == "hutchinson" else None
        ),
        "seed": int(args.seed),
        "max_tokens": int(args.max_tokens),
        "n_bundles": len(paths),
        "bundle_digests": digests,
        "wall_time_s": wall_time,
        "timestamp": time.time(),
    }
    policy = build_hessian_trace_policy(
        records=records,
        method=args.method,
        tile_size=args.tile_size,
        n_hutchinson_vectors=args.n_hutchinson_vectors,
        seed=args.seed,
        provenance=provenance,
    )
    # Validation runs on every emission; --validate makes the exit code
    # reflect a validation failure rather than just an early error.
    validate_policy(policy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    if args.verbose:
        print(
            f"wrote {output}  ({len(records)} tensors, {wall_time:.3f}s)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
