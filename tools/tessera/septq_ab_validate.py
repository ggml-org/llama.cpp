#!/usr/bin/env python3
"""A/B validation harness for the SEPTQ quantization mode.

Compares the SEPTQ (KDD 2025) two-step PTQ method against the standard
tessera Round-To-Nearest (RTN) baseline on a set of 2D weight matrices.
The comparison is at the weight level: both paths quantize a 2D weight
matrix, and we reconstruct the quantized weight from the packed output
to compute the BF16-vs-quantized MSE, max absolute error, and the number
of elements stored in each precision.

The harness is the main deliverable for the SEPTQ A/B track. The
``--septq`` mode in ``quantize_v3.py`` is the underlying mechanism; this
script provides the comparison infrastructure.

Input modes:
  * ``--bundle <path>``: load layer bundles (``.npz``) from
    ``make-awq-layer-bundles.py``. Each bundle has a ``weight`` array
    and ``in_sum2``/``counts`` for the activation observer.
  * ``--synthetic``: generate a synthetic 4096x4096 weight + 32
    calibration samples with a fixed seed for reproducibility.

Output:
  * JSON report at ``--output`` with schema
    ``llama.tessera.septq-ab-report.v1``.
  * Human-readable summary table on stdout.

The harness is deterministic: no random seeds change between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# The quantize_v3 module has heavy import side effects (gguf, mlx, etc.)
# and is normally invoked as a script. We import it lazily inside main()
# after environment tweaks so the harness is usable in CI without gguf.
HERE = Path(__file__).resolve().parent
TILE640_DIR = HERE.parent / "tile640"

SCHEMA = "llama.tessera.septq-ab-report.v1"
SYNTHETIC_SEED = 42  # fixed for determinism; do not change between runs


# ---------------------------------------------------------------------------
# Tile640 constants (must match quantize_v3.py and the C++ side)
# ---------------------------------------------------------------------------
TILE640_PAGE_SIZE = 640
TILE640_LANE_SIZE = 20
TILE640_LANES_PER_PAGE = 32


# ---------------------------------------------------------------------------
# Reconstruction utilities
# ---------------------------------------------------------------------------
def unpack_tile640(packed: np.ndarray, out_dim: int, pages_per_row: int) -> np.ndarray:
    """Unpack base-3 u32 words to a flat {-1, 0, +1} ternary array.

    Inverse of ``pack_tile640`` in quantize_v3.py. The packed format is
    20 trits per u32 word, LSB-first, with trits encoded as
    {2, 0, 1} for {-1, 0, +1}.
    """
    words = packed.view(np.uint32).reshape(
        out_dim, pages_per_row, TILE640_LANES_PER_PAGE
    )
    pow3 = np.array(
        [3 ** i for i in range(TILE640_LANE_SIZE)], dtype=np.uint32
    ).reshape(1, 1, 1, TILE640_LANE_SIZE)
    # words: [out, pages, 32]; expand to [out, pages, 32, 20]
    trits_raw = (words[:, :, :, None] // pow3) % 3
    trits = np.where(
        trits_raw == 1, 1, np.where(trits_raw == 2, -1, 0)
    ).astype(np.int8)
    return trits.reshape(out_dim, -1)


def reconstruct_weight(
    q: Dict[str, np.ndarray],
    out_dim: int,
    in_dim: int,
) -> np.ndarray:
    """Reconstruct the 2D weight from a quantize_2d/quantize_2d_septq output.

    The reconstruction is:
      ``ternary * (page_scale * lane_scale / 127) + outliers``

    For the SEPTQ path, the ``_ternary`` key is used directly to avoid
    unpacking Tile640. For the baseline path, the ternary is unpacked
    from the packed output.
    """
    pages_per_row = (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
    padded_in_dim = pages_per_row * TILE640_PAGE_SIZE

    if "_ternary" in q:
        # SEPTQ path: the ternary is exposed directly via the _ternary key.
        # It is already sized to (out_dim, in_dim) flat.
        ternary_flat = q["_ternary"]
    else:
        # Baseline path: unpack from Tile640.
        ternary_padded = unpack_tile640(q["packed"], out_dim, pages_per_row)
        # Trim the in_dim padding (the last padded lanes are zero by construction).
        ternary_flat = ternary_padded.reshape(-1)[: out_dim * in_dim]

    ternary_2d = ternary_flat.reshape(out_dim, in_dim).astype(np.float32)

    page_scales = q["page_scales"].reshape(out_dim, pages_per_row).astype(np.float32)
    lane_scales = q["lane_scales"].reshape(
        out_dim, pages_per_row, TILE640_LANES_PER_PAGE
    ).astype(np.float32)
    # Per-lane scale = page_scale * lane_scale / 127.0
    scale_lane = page_scales[:, :, None] * lane_scales / np.float32(127.0)
    # Expand to per-element scale: repeat each lane 20 times.
    scale_elem = np.repeat(scale_lane, TILE640_LANE_SIZE, axis=-1)
    scale_2d = scale_elem.reshape(out_dim, padded_in_dim)[:, :in_dim]

    reconstructed = ternary_2d * scale_2d

    # Add the outliers (full-precision residual elements). For the baseline
    # these are the "important" weights kept at FP16; for SEPTQ these are
    # the "unimportant" weights kept at FP16.
    outlier_row_offsets = q["outlier_row_offsets"]
    outlier_cols = q["outlier_cols"]
    outlier_vals = q["outlier_vals"].astype(np.float32)
    for r in range(out_dim):
        start = int(outlier_row_offsets[r])
        end = int(outlier_row_offsets[r + 1])
        if start < end:
            reconstructed[r, outlier_cols[start:end]] = outlier_vals[start:end]

    return reconstructed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class TensorSample:
    """A single 2D weight matrix and its calibration data."""

    name: str
    family: str
    weight: np.ndarray
    act_scales: Optional[np.ndarray] = None


def load_bundle(path: Path) -> TensorSample:
    """Load a single .npz layer bundle produced by make-awq-layer-bundles.py."""
    with np.load(path, allow_pickle=False) as data:
        weight = np.asarray(data["weight"], dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(
                f"{path}: expected 2D weight, got shape {weight.shape}"
            )
        # The bundle stores in_sum2 per channel; counts is a scalar.
        in_sum2 = np.asarray(data["in_sum2"], dtype=np.float32).reshape(-1)
        counts = float(np.asarray(data["counts"]).reshape(-1)[0])
        if counts > 0 and in_sum2.shape[0] == weight.shape[1]:
            act_scales = np.sqrt(in_sum2 / counts).astype(np.float32)
        else:
            act_scales = None
        name = str(np.asarray(data["name"]).item()) if "name" in data.files else path.stem
        family = (
            str(np.asarray(data["family"]).item()) if "family" in data.files else "unknown"
        )
    return TensorSample(name=name, family=family, weight=weight, act_scales=act_scales)


def load_bundles(bundle_dir: Path, limit: Optional[int] = None) -> List[TensorSample]:
    """Load all .npz bundles from a directory, sorted by name for determinism."""
    paths = sorted(bundle_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz bundles found in {bundle_dir}")
    samples = [load_bundle(p) for p in paths]
    if limit is not None:
        samples = samples[:limit]
    return samples


def synthetic_sample(
    out_dim: int = 4096, in_dim: int = 4096, n_calib: int = 32
) -> Tuple[TensorSample, np.ndarray]:
    """Generate a deterministic synthetic weight + calibration set.

    The weight is a sum of a low-rank component (rank-8) and a dense
    Gaussian component, so the importance distribution is non-trivial and
    the SEPTQ mask selection has something to chew on. The calibration
    activations are also a low-rank + Gaussian mixture to give the
    imatrix a non-uniform channel-wise profile.

    Returns:
        (TensorSample, calibration_activations) where calibration_activations
        has shape ``[n_calib, in_dim]`` and is included for documentation;
        only ``act_scales`` (RMS per channel) is consumed by the harness.
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    # Low-rank component: rank-8 outer product of random vectors.
    rank = 8
    u = rng.standard_normal((out_dim, rank)).astype(np.float32) / np.sqrt(rank)
    v = rng.standard_normal((rank, in_dim)).astype(np.float32) / np.sqrt(rank)
    dense = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * np.float32(0.1)
    weight = (u @ v + dense).astype(np.float32)
    # Calibration activations: low-rank + Gaussian, shape [n_calib, in_dim].
    x_low = rng.standard_normal((n_calib, rank)).astype(np.float32) @ v
    x_noise = rng.standard_normal((n_calib, in_dim)).astype(np.float32) * np.float32(0.5)
    x = (x_low + x_noise).astype(np.float32)
    # act_scales = RMS per channel (same convention as the imatrix loader).
    act_scales = np.sqrt(np.mean(x * x, axis=0)).astype(np.float32)
    sample = TensorSample(
        name="synthetic-4096x4096",
        family="synthetic",
        weight=weight,
        act_scales=act_scales,
    )
    return sample, x


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class TensorMetrics:
    """Per-tensor metrics for one path (baseline or SEPTQ)."""

    mse: float
    max_abs_error: float
    quantized_count: int
    residual_count: int
    wall_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse": self.mse,
            "max_abs_error": self.max_abs_error,
            "quantized_count": self.quantized_count,
            "residual_count": self.residual_count,
            "wall_time_ms": self.wall_time_ms,
        }


def compute_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    quantized_count: int,
    residual_count: int,
    wall_time_ms: float,
) -> TensorMetrics:
    """Compute per-tensor BF16-vs-quantized metrics."""
    diff = (original.astype(np.float32) - reconstructed.astype(np.float32))
    mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))
    return TensorMetrics(
        mse=mse,
        max_abs_error=max_abs,
        quantized_count=quantized_count,
        residual_count=residual_count,
        wall_time_ms=wall_time_ms,
    )


# ---------------------------------------------------------------------------
# Quantization dispatch
# ---------------------------------------------------------------------------
def quantize_with_metrics(
    quantize_fn,
    weight: np.ndarray,
    out_dim: int,
    in_dim: int,
    act_scales: Optional[np.ndarray],
    **kwargs: Any,
) -> Tuple[Dict[str, np.ndarray], TensorMetrics, float]:
    """Run a quantize function, time it, and compute metrics.

    Returns (quantize_output, metrics, wall_time_ms).
    """
    t0 = time.perf_counter()
    q = quantize_fn(weight, out_dim, in_dim, act_scales=act_scales, **kwargs)
    wall_time_ms = (time.perf_counter() - t0) * 1000.0
    reconstructed = reconstruct_weight(q, out_dim, in_dim)
    quantized_count = int(np.count_nonzero(q.get("_ternary",
        np.array([], dtype=np.int8))))
    if quantized_count == 0:
        # Baseline path: count elements not in the outlier set.
        # The ternary is {-1, 0, +1}; 0 at a position means "not quantized"
        # (i.e., the element is in the outlier set). We unpack the ternary
        # to count the non-zero entries.
        ternary = unpack_tile640(
            q["packed"], out_dim, (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
        ).reshape(-1)[: out_dim * in_dim]
        quantized_count = int(np.count_nonzero(ternary))
    residual_count = int(q["outlier_vals"].size)
    metrics = compute_metrics(
        weight, reconstructed, quantized_count, residual_count, wall_time_ms
    )
    return q, metrics, wall_time_ms


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------
@dataclass
class TensorResult:
    name: str
    family: str
    shape: List[int]
    baseline: TensorMetrics
    septq: TensorMetrics
    winner: str  # "septq", "baseline", or "tie"
    mse_improvement_pct: float  # positive = SEPTQ better

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "shape": self.shape,
            "baseline": self.baseline.to_dict(),
            "septq": self.septq.to_dict(),
            "winner": self.winner,
            "mse_improvement_pct": self.mse_improvement_pct,
        }


def compare_tensor(
    sample: TensorSample,
    baseline_fn,
    septq_fn,
    septq_ratio: float,
    septq_iterations: int,
) -> TensorResult:
    """Run baseline and SEPTQ on one tensor and return the comparison."""
    weight = sample.weight
    out_dim, in_dim = weight.shape
    act_scales = sample.act_scales

    # Baseline: standard tessera 2D path with no AWQ. outlier_frac=0.005 is
    # the production default and is a fair RTN-style baseline. No
    # imatrix-mse and no SEPTQ, so this is the "vanilla" tessera flow.
    _, baseline_metrics, _ = quantize_with_metrics(
        baseline_fn,
        weight,
        out_dim,
        in_dim,
        act_scales,
        outlier_frac=0.005,
        awq_alpha=0.0,
        awq_clip=1.0,
        tensor_name=sample.name,
        ternary_threshold=1.0,
    )

    # SEPTQ: the new --septq mode.
    _, septq_metrics, _ = quantize_with_metrics(
        septq_fn,
        weight,
        out_dim,
        in_dim,
        act_scales,
        septq_ratio=septq_ratio,
        septq_iterations=septq_iterations,
        ternary_threshold=1.0,
        tensor_name=sample.name,
    )

    # Winner: lower MSE wins. Tie if equal within 1e-9.
    if baseline_metrics.mse < septq_metrics.mse - 1e-9:
        winner = "baseline"
    elif septq_metrics.mse < baseline_metrics.mse - 1e-9:
        winner = "septq"
    else:
        winner = "tie"
    # Positive = SEPTQ better.
    if baseline_metrics.mse > 0:
        improvement = (baseline_metrics.mse - septq_metrics.mse) / baseline_metrics.mse * 100.0
    else:
        improvement = 0.0

    return TensorResult(
        name=sample.name,
        family=sample.family,
        shape=[out_dim, in_dim],
        baseline=baseline_metrics,
        septq=septq_metrics,
        winner=winner,
        mse_improvement_pct=float(improvement),
    )


# ---------------------------------------------------------------------------
# Report aggregation and emission
# ---------------------------------------------------------------------------
@dataclass
class AggregateResult:
    total_tensors: int
    septq_wins: int
    baseline_wins: int
    ties: int
    mean_baseline_mse: float
    mean_septq_mse: float
    mean_mse_improvement_pct: float
    median_mse_improvement_pct: float
    total_baseline_quantized: int
    total_septq_quantized: int
    mean_septq_ratio: float  # actual quantized / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tensors": self.total_tensors,
            "septq_wins": self.septq_wins,
            "baseline_wins": self.baseline_wins,
            "ties": self.ties,
            "mean_baseline_mse": self.mean_baseline_mse,
            "mean_septq_mse": self.mean_septq_mse,
            "mean_mse_improvement_pct": self.mean_mse_improvement_pct,
            "median_mse_improvement_pct": self.median_mse_improvement_pct,
            "total_baseline_quantized": self.total_baseline_quantized,
            "total_septq_quantized": self.total_septq_quantized,
            "mean_septq_ratio": self.mean_septq_ratio,
        }


def aggregate(results: List[TensorResult]) -> AggregateResult:
    if not results:
        return AggregateResult(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    septq_wins = sum(1 for r in results if r.winner == "septq")
    baseline_wins = sum(1 for r in results if r.winner == "baseline")
    ties = sum(1 for r in results if r.winner == "tie")
    mean_baseline = float(np.mean([r.baseline.mse for r in results]))
    mean_septq = float(np.mean([r.septq.mse for r in results]))
    improvements = [r.mse_improvement_pct for r in results]
    mean_improvement = float(np.mean(improvements))
    median_improvement = float(np.median(improvements))
    total_baseline_q = sum(r.baseline.quantized_count for r in results)
    total_septq_q = sum(r.septq.quantized_count for r in results)
    total_elements = sum(r.shape[0] * r.shape[1] for r in results)
    mean_septq_ratio = total_septq_q / total_elements if total_elements else 0.0
    return AggregateResult(
        total_tensors=len(results),
        septq_wins=septq_wins,
        baseline_wins=baseline_wins,
        ties=ties,
        mean_baseline_mse=mean_baseline,
        mean_septq_mse=mean_septq,
        mean_mse_improvement_pct=mean_improvement,
        median_mse_improvement_pct=median_improvement,
        total_baseline_quantized=total_baseline_q,
        total_septq_quantized=total_septq_q,
        mean_septq_ratio=mean_septq_ratio,
    )


def emit_report(
    results: List[TensorResult],
    aggregate_result: AggregateResult,
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "tensors": [r.to_dict() for r in results],
        "aggregate": aggregate_result.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")


def print_summary(results: List[TensorResult], aggregate_result: AggregateResult) -> None:
    """Print a human-readable summary table to stdout."""
    print()
    print("=" * 88)
    print(f"SEPTQ A/B Validation Report ({SCHEMA})")
    print("=" * 88)
    header = (
        f"{'tensor':<40s} {'shape':<12s} "
        f"{'baseline_mse':>14s} {'septq_mse':>14s} "
        f"{'improv%':>9s} {'winner':>8s}"
    )
    print(header)
    print("-" * 88)
    for r in results:
        name = r.name if len(r.name) <= 40 else r.name[:37] + "..."
        shape_str = f"{r.shape[0]}x{r.shape[1]}"
        print(
            f"{name:<40s} {shape_str:<12s} "
            f"{r.baseline.mse:>14.6e} {r.septq.mse:>14.6e} "
            f"{r.mse_improvement_pct:>8.2f}% {r.winner:>8s}"
        )
    print("-" * 88)
    print(f"tensors: {aggregate_result.total_tensors}  "
          f"septq wins: {aggregate_result.septq_wins}  "
          f"baseline wins: {aggregate_result.baseline_wins}  "
          f"ties: {aggregate_result.ties}")
    print(f"mean baseline MSE: {aggregate_result.mean_baseline_mse:.6e}")
    print(f"mean SEPTQ MSE:    {aggregate_result.mean_septq_mse:.6e}")
    print(f"mean MSE improvement: {aggregate_result.mean_mse_improvement_pct:.2f}%  "
          f"(median {aggregate_result.median_mse_improvement_pct:.2f}%)")
    print(f"baseline quantized: {aggregate_result.total_baseline_quantized}  "
          f"SEPTQ quantized: {aggregate_result.total_septq_quantized}  "
          f"actual SEPTQ ratio: {aggregate_result.mean_septq_ratio:.3f}")
    if aggregate_result.septq_wins > aggregate_result.baseline_wins:
        verdict = "SEPTQ wins overall"
    elif aggregate_result.baseline_wins > aggregate_result.septq_wins:
        verdict = "Baseline wins overall"
    else:
        verdict = "Tie"
    print(f"verdict: {verdict}")
    print("=" * 88)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="A/B validation harness for SEPTQ vs standard tessera RTN."
    )
    ap.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to a layer bundle (.npz) or directory of bundles.",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a deterministic synthetic 4096x4096 weight + 32 calibration samples.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of bundle tensors to evaluate (directory mode only).",
    )
    ap.add_argument(
        "--septq-ratio",
        type=float,
        default=0.5,
        help="Fraction of elements to quantize under SEPTQ (default 0.5).",
    )
    ap.add_argument(
        "--septq-iterations",
        type=int,
        default=1,
        help="Number of column-by-column passes for SEPTQ (default 1).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/septq_ab_report.json"),
        help="Output JSON report path (default /tmp/septq_ab_report.json).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the stdout summary table.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.bundle and not args.synthetic:
        print("error: must specify --bundle or --synthetic", file=sys.stderr)
        return 2
    if args.bundle and args.synthetic:
        print("error: --bundle and --synthetic are mutually exclusive", file=sys.stderr)
        return 2

    # Import the quantize functions lazily so the harness can show --help
    # without gguf being installed.
    if str(TILE640_DIR) not in sys.path:
        sys.path.insert(0, str(TILE640_DIR))
    os.environ.setdefault("TESSERA_ACCELERATE", "0")
    os.environ.setdefault("TESSERA_ANE_MODEL", "")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "quantize_v3", str(TILE640_DIR / "quantize_v3.py")
    )
    qmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qmod)
    baseline_fn = qmod.quantize_2d
    septq_fn = qmod.quantize_2d_septq

    # Load data.
    if args.synthetic:
        sample, _calib = synthetic_sample()
        samples = [sample]
        data_source = "synthetic-4096x4096"
    else:
        bundle_path: Path = args.bundle
        if bundle_path.is_dir():
            samples = load_bundles(bundle_path, limit=args.limit)
            data_source = f"bundle-dir:{bundle_path}"
        else:
            samples = [load_bundle(bundle_path)]
            data_source = f"bundle:{bundle_path}"

    config = {
        "data_source": data_source,
        "septq_ratio": args.septq_ratio,
        "septq_iterations": args.septq_iterations,
        "tessera_quantize_v3": str(TILE640_DIR / "quantize_v3.py"),
    }

    results: List[TensorResult] = []
    for sample in samples:
        out_dim, in_dim = sample.weight.shape
        print(
            f"  evaluating {sample.name} ({out_dim}x{in_dim})...",
            file=sys.stderr,
        )
        result = compare_tensor(
            sample,
            baseline_fn,
            septq_fn,
            args.septq_ratio,
            args.septq_iterations,
        )
        results.append(result)

    aggregate_result = aggregate(results)
    emit_report(results, aggregate_result, config, args.output)
    if not args.quiet:
        print_summary(results, aggregate_result)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
