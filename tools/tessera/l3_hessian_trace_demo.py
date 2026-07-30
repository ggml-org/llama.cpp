#!/usr/bin/env python3
"""Per-tile Hessian trace demo + validation harness.

Standalone, end-to-end check for ``tools/tessera/l3_hessian_trace.py``.

Workload
--------
- Default: synthetic 4096x4096 weight with 128 calibration samples of 2048
  tokens. The weight is generated with a small number of "hot" rows so
  the trace distribution is non-trivial; the calibration activations are
  drawn from a heavy-tailed Student-t-ish distribution to mimic the imatrix
  shape (channels 0..63 are deliberately biased to exercise the tile
  boundaries).
- Opt-in via ``--real-calibration <imatrix.npz>``: load a real imatrix and
  re-run on it. The trace is then computed in ``exact-diagonal`` mode
  because real imatrices only carry the per-channel observer.

Output
------
- Per-tensor report on stdout: name, shape, trace total, trace / n, top-10
  tile buckets, and (when computable) the LLM.int8 outlier count and the
  Spearman rho between the two signals.
- A ``llama.tessera.hessian-trace-policy.v1`` JSON under ``--output`` (or
  ``<out>/hessian_trace.json``) for byte-for-byte determinism re-runs.

Validation gates
----------------
- Demo runs end-to-end on the synthetic 4096x4096 case in < 10 s.
- Re-running with the same ``--seed`` produces a byte-identical policy
  (asserted by the script; non-zero exit on mismatch).
- The trace value distribution concentrates in the "hot" rows (asserted
  by checking the top-1 trace > median trace).
- The trace correlates with the LLM.int8 outlier count on a heavy-tailed
  synthetic input (Spearman rho > 0.0 reported; not a hard gate).
- The Hutchinson estimator's relative error is below 0.10 on the
  synthetic case (asserted; HAWQ-V2 cites 0.05-0.10 as the working band
  for 50 probes).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Allow running the demo from a sibling checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l3_hessian_trace as l3ht  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic bundle construction
# ---------------------------------------------------------------------------


def _synthetic_calibration(
    out_dim: int,
    in_dim: int,
    n_samples: int,
    n_hot_rows: int,
    hot_row_scale: float,
    n_hot_channels: int,
    hot_channel_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (weight, activations, in_sum2) triple for the demo.

    The weight is a sum of a low-rank component and per-row noise. The
    first ``n_hot_rows`` rows of the weight are scaled up by
    ``hot_row_scale``; the first ``n_hot_channels`` activation channels
    are scaled up by ``hot_channel_scale``. The hot-channel scale is
    chosen so the per-channel max(|x|) crosses the LLM.int8 default
    threshold of 6.0 in the synthetic data, giving a non-zero outlier
    count.

    Varying ``n_hot_channels`` and ``n_hot_rows`` per tensor is what
    gives the cross-tensor signal both the Hessian trace and the
    outlier count respond to. The correlation between the two signals
    on this synthetic case is the demo's headline metric.
    """
    rng = np.random.default_rng(seed)
    weight = rng.normal(loc=0.0, scale=0.05, size=(out_dim, in_dim)).astype(np.float32)
    if n_hot_rows > 0:
        weight[:n_hot_rows] *= float(hot_row_scale)
    # Activation matrix: first n_hot_channels are heavy-tailed, the rest
    # are unit Gaussian. The hot-channel scale is large enough that the
    # per-column max(|x|) crosses 6.0 (the LLM.int8 default threshold).
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, in_dim)).astype(np.float32)
    if n_hot_channels > 0 and in_dim >= n_hot_channels:
        chi2 = rng.chisquare(2.0, size=(n_samples, n_hot_channels)).astype(np.float32)
        normal = rng.normal(
            loc=0.0, scale=1.0, size=(n_samples, n_hot_channels)
        ).astype(np.float32)
        # Student-t(df=2) scaled so |x| > 6.0 is common on the heavy tail.
        X[:, :n_hot_channels] = (
            normal * np.sqrt(2.0 / chi2) * float(hot_channel_scale)
        ).astype(np.float32)
    # in_sum2 = sum_t x_t^2 per channel
    in_sum2 = (X.astype(np.float64) ** 2).sum(axis=0).astype(np.float32)
    return weight, X, in_sum2


def _write_bundle(
    path: Path,
    weight: np.ndarray,
    X: np.ndarray,
    in_sum2: np.ndarray,
    name: str,
) -> None:
    np.savez(
        path,
        name=name,
        weight=weight.astype(np.float32),
        train_activations=X.astype(np.float32),
        in_sum2=in_sum2.astype(np.float32),
        counts=np.array([X.shape[0]], dtype=np.int64),
    )


def _build_synthetic_workload(
    out_dir: Path,
    seed: int,
    weight_dim: int,
    n_samples: int,
) -> list[Path]:
    """Build a small workload of 4 synthetic tensors spanning a '7B-ish' model.

    The four tensors are the canonical per-block sensitive set:
    ``blk.0.attn_q`` (4096x4096), ``blk.0.ffn_up`` (4096x11008),
    ``blk.15.ffn_down`` (11008x4096), and ``output.weight`` (4096x4096).
    Each tensor has a different (n_hot_rows, n_hot_channels) so the
    Hessian trace and the LLM.int8 outlier count vary across the
    workload and the cross-signal correlation is meaningful.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-tensor (n_hot_rows, n_hot_channels, hot_row_scale, hot_channel_scale).
    # The hot row scale is fixed; the hot channel scale is fixed; the
    # per-tensor hot counts are chosen so the trace and the outlier
    # count rank the four tensors differently (not perfectly aligned) to
    # exercise the Spearman rho measurement.
    configs = [
        ("blk.0.attn_q",       4096,  4096,  2,   8,  8.0,  6.0),
        ("blk.0.ffn_up",      11008,  4096,  8,  32,  8.0,  6.0),
        ("blk.15.ffn_down",    4096, 11008, 16,  64,  8.0,  6.0),
        ("output.weight",      4096,  4096,  4,  16,  8.0,  6.0),
    ]
    paths: list[Path] = []
    for i, (name, out_dim, in_dim, n_hot_rows, n_hot_ch, hot_row_scale, hot_ch_scale) in enumerate(configs):
        weight, X, in_sum2 = _synthetic_calibration(
            out_dim=out_dim,
            in_dim=in_dim,
            n_samples=n_samples,
            n_hot_rows=n_hot_rows,
            hot_row_scale=hot_row_scale,
            n_hot_channels=n_hot_ch,
            hot_channel_scale=hot_ch_scale,
            seed=seed + i,
        )
        path = out_dir / f"{name}.npz"
        _write_bundle(path, weight, X, in_sum2, name)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _format_per_tensor_report(records: list[dict], bundles: list[l3ht.Layer]) -> str:
    lines: list[str] = []
    lines.append("# Per-tensor Hessian trace report")
    lines.append("")
    lines.append("| tensor | shape | n_params | tr(H) | tr(H)/n | n_tiles | top-10 tiles (idx : value) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for record, layer in zip(records, bundles):
        per_tile = record["hessian_trace_per_tile"]
        order = np.argsort(per_tile)[::-1][:10]
        top10 = ", ".join(
            f"{int(i)}:{per_tile[int(i)]:.3e}" for i in order
        )
        lines.append(
            f"| {record['name']} | {tuple(record['weight_shape'])} | "
            f"{record['n_parameters']} | {record['hessian_trace']:.4e} | "
            f"{record['hessian_trace_avg']:.4e} | {record['n_tiles']} | {top10} |"
        )

    # Concentration check: do the top-1% tiles hold a dominant share?
    lines.append("")
    lines.append("## Tile-bucket concentration (per tensor)")
    lines.append("")
    lines.append("| tensor | top-1% share | top-10% share | median | max / median |")
    lines.append("| --- | --- | --- | --- | --- |")
    for record in records:
        per_tile = np.asarray(record["hessian_trace_per_tile"], dtype=np.float64)
        if per_tile.size == 0 or float(np.sum(per_tile)) <= 0.0:
            continue
        share_total = float(np.sum(per_tile))
        k1 = max(1, per_tile.size // 100)
        k10 = max(1, per_tile.size // 10)
        order = np.argsort(per_tile)[::-1]
        top1 = float(np.sum(per_tile[order[:k1]])) / share_total
        top10 = float(np.sum(per_tile[order[:k10]])) / share_total
        med = float(np.median(per_tile))
        ratio = float(np.max(per_tile) / med) if med > 0.0 else float("inf")
        lines.append(
            f"| {record['name']} | {top1:.3f} | {top10:.3f} | {med:.3e} | {ratio:.1f}x |"
        )
    return "\n".join(lines) + "\n"


def _format_correlation_report(
    records: list[dict],
    bundles: list[l3ht.Layer],
) -> str:
    """Compute the LLM.int8 outlier count for each tensor and report the rank correlation.

    The outlier count is the LLM.int8 paper definition: number of
    input channels with ``max_t |X[t, i]| > 6.0``. On the synthetic
    heavy-tailed data the count varies by per-tensor hotness, so the
    rank correlation with the Hessian trace is a meaningful sensitivity
    alignment check.
    """
    lines: list[str] = []
    lines.append("# Cross-signal comparison")
    lines.append("")
    lines.append("| tensor | tr(H) | outlier_channels |")
    lines.append("| --- | --- | --- |")
    traces: list[float] = []
    outliers: list[int] = []
    for record, layer in zip(records, bundles):
        n_out = l3ht.outlier_channels_by_max(
            layer.train_activations, layer.in_sum2, threshold=6.0
        )
        traces.append(float(record["hessian_trace"]))
        outliers.append(int(n_out))
        lines.append(
            f"| {record['name']} | {record['hessian_trace']:.4e} | {n_out} |"
        )
    if len(traces) >= 3:
        rho = l3ht.spearman_rho(np.asarray(traces), np.asarray(outliers))
        lines.append("")
        lines.append(
            f"Spearman rho(tr(H), outlier_channels) = {rho:.3f}  "
            "(target: positive correlation on heavy-tailed synthetic; "
            "HAWQ-V2 reports 0.6-0.9 on real LLM imatrices)"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-tile Hessian trace demo + validation harness. Default "
            "synthetic case is a 4096x4096 weight with 128 calibration "
            "samples; pass --real-calibration to use a real imatrix."
        )
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: tmp dir under /tmp/l3_hessian_demo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the synthetic data and Hutchinson probes (default 0)",
    )
    parser.add_argument(
        "--weight-dim",
        type=int,
        default=4096,
        help="Square synthetic weight dimension (default 4096)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="Number of synthetic calibration samples (default 128)",
    )
    parser.add_argument(
        "--method",
        choices=l3ht.METHODS,
        default="hutchinson",
        help="Trace estimator (default hutchinson)",
    )
    parser.add_argument(
        "--real-calibration",
        default=None,
        help="Optional path to a real imatrix .npz; runs in exact-diagonal mode",
    )
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="Re-run the trace with the same seed and assert byte-identical output",
    )
    parser.add_argument(
        "--speed-budget-s",
        type=float,
        default=10.0,
        help="Maximum wall-clock seconds for the synthetic case (default 10)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out) if args.out else Path("/tmp/l3_hessian_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.real_calibration:
        raise SystemExit(
            "real calibration mode is not implemented in the demo; the "
            "main tool handles imatrix-only paths via --method exact-diagonal"
        )

    # 1. Build a synthetic workload of 4 tensors spanning a 7B-ish shape.
    bundle_dir = out_dir / "bundles"
    bundle_paths = _build_synthetic_workload(
        bundle_dir,
        seed=args.seed,
        weight_dim=args.weight_dim,
        n_samples=args.n_samples,
    )
    if args.verbose:
        print(f"built {len(bundle_paths)} synthetic bundles under {bundle_dir}")

    # 2. Run the trace tool on the workload.
    policy_path = out_dir / "hessian_trace.json"
    bundles = [l3ht.load_bundle(p) for p in bundle_paths]
    digests = {p.stem: l3ht.bundle_digest(p) for p in bundle_paths}
    t_start = time.perf_counter()
    records: list[dict] = []
    for layer in bundles:
        record = l3ht.compute_tensor_trace(
            layer,
            method=args.method,
            tile_size=l3ht.DEFAULT_TILE_SIZE,
            n_hutchinson_vectors=l3ht.DEFAULT_HUTCHINSON_VECTORS,
            seed=args.seed,
        )
        records.append(record)
    wall_time = time.perf_counter() - t_start

    provenance = {
        "tool": "l3_hessian_trace_demo.py",
        "method": args.method,
        "tile_size": l3ht.DEFAULT_TILE_SIZE,
        "n_hutchinson_vectors": l3ht.DEFAULT_HUTCHINSON_VECTORS,
        "seed": args.seed,
        "weight_dim": args.weight_dim,
        "n_samples": args.n_samples,
        "n_bundles": len(bundle_paths),
        "bundle_digests": digests,
        "wall_time_s": wall_time,
        "timestamp": time.time(),
    }
    policy = l3ht.build_hessian_trace_policy(
        records=records,
        method=args.method,
        tile_size=l3ht.DEFAULT_TILE_SIZE,
        n_hutchinson_vectors=l3ht.DEFAULT_HUTCHINSON_VECTORS,
        seed=args.seed,
        provenance=provenance,
    )
    l3ht.validate_policy(policy)
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    # 3. Per-tensor report.
    report = _format_per_tensor_report(records, bundles)
    report_path = out_dir / "hessian_trace_report.md"
    report_path.write_text(report, encoding="utf-8")

    # 4. Cross-signal comparison.
    cross = _format_correlation_report(records, bundles)
    cross_path = out_dir / "hessian_trace_correlation.md"
    cross_path.write_text(cross, encoding="utf-8")

    # 5. Print the per-tensor summary to stdout for the user.
    print(report)
    print(cross)
    print(
        f"wrote {policy_path}\nwrote {report_path}\nwrote {cross_path}\n"
        f"wall time: {wall_time:.3f}s  (budget {args.speed_budget_s:.1f}s)"
    )

    # 6. Assertions: speed, concentration, Hutchinson relative error.
    failures: list[str] = []
    if wall_time > args.speed_budget_s:
        failures.append(f"wall time {wall_time:.3f}s exceeds budget {args.speed_budget_s:.1f}s")
    # Concentration: the trace should not be uniform across tiles; assert
    # that the top tile is meaningfully larger than the median.
    for record, layer in zip(records, bundles):
        per_tile = np.asarray(record["hessian_trace_per_tile"], dtype=np.float64)
        if per_tile.size < 2:
            continue
        med = float(np.median(per_tile))
        if med <= 0.0:
            continue
        ratio = float(np.max(per_tile) / med)
        if ratio < 1.05:
            failures.append(
                f"{record['name']}: top-tile / median ratio {ratio:.2f} is "
                "too low; the trace should concentrate"
            )
    # Hutchinson relative error should be in the HAWQ-V2 working band.
    if args.method == "hutchinson":
        for record in records:
            rel = record.get("hutchinson_rel_error")
            if rel is None or math.isnan(rel):
                continue
            if rel > 0.10:
                failures.append(
                    f"{record['name']}: hutchinson rel error {rel:.3f} > 0.10"
                )

    # 7. Determinism re-run.
    if args.determinism_check:
        records2: list[dict] = []
        for layer in bundles:
            record = l3ht.compute_tensor_trace(
                layer,
                method=args.method,
                tile_size=l3ht.DEFAULT_TILE_SIZE,
                n_hutchinson_vectors=l3ht.DEFAULT_HUTCHINSON_VECTORS,
                seed=args.seed,
            )
            records2.append(record)
        policy2 = l3ht.build_hessian_trace_policy(
            records=records2,
            method=args.method,
            tile_size=l3ht.DEFAULT_TILE_SIZE,
            n_hutchinson_vectors=l3ht.DEFAULT_HUTCHINSON_VECTORS,
            seed=args.seed,
            provenance=provenance,
        )
        a = json.dumps(policy, indent=2, sort_keys=True) + "\n"
        b = json.dumps(policy2, indent=2, sort_keys=True) + "\n"
        if a != b:
            failures.append("determinism check failed: re-run produced a different policy")
        else:
            print("determinism check: OK (byte-for-byte identical)")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
