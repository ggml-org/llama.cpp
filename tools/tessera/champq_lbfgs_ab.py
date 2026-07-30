#!/usr/bin/env python3
"""A/B validation harness for the LBFGS CHAMP-Q permutation mode.

Compares four permutation strategies on the same synthetic weight:

  (a) none       no permutation (baseline)
  (b) random     uniformly random input-channel permutation
  (c) ga         genetic-algorithm search (existing CHAMP-Q path)
  (d) lbfgs      continuous relaxation + LBFGS on smoothness proxy

The synthetic is a 4096x4096 rank-8 + Gaussian tensor (per the
research-report-recommended shape). Reconstruction MSE is computed
by quantising the permuted weight with a per-row ternarisation
threshold, then applying the inverse permutation to the quantised
weight, and comparing to the original.

The harness also reports the smoothness proxy (a separate metric
the LBFGS objective directly optimises) and the per-row MSE
distribution. The ``none`` mode is the calibration default (no
permutation); ``random`` and ``ga`` are the new alternatives the
LBFGS mode is compared against.

Deterministic: a fixed seed drives both the synthetic construction
and the permutation RNGs. Re-running the harness produces the same
table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent

from champq_permute import (  # noqa: E402
    SCHEMA,
    apply_champq_permutation,
    compute_champq_permutation,
    compute_permutation,
    ga_permutation,
    invert_champq_permutation,
    lbfgs_permutation,
    random_permutation,
    smoothness_proxy,
)


SCHEMA_AB = "llama.tessera.champq-lbfgs-ab-report.v1"
DEFAULT_OUT_DIM = 4096
DEFAULT_IN_DIM = 4096
DEFAULT_RANK = 8
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic weight construction
# ---------------------------------------------------------------------------


def synthetic_weight(
    out_dim: int,
    in_dim: int,
    rank: int,
    seed: int,
) -> np.ndarray:
    """Generate a 2-D weight matrix: rank-r outer product + Gaussian noise.

    The low-rank factor carries the permutation-sensitive structure
    (the row / column magnitudes the smoothness proxy measures). The
    noise is small (1% of the low-rank norm) so the L2-rank
    heuristic has signal to lock onto.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(out_dim, rank)).astype(np.float64)
    v = rng.normal(size=(in_dim, rank)).astype(np.float64)
    low_rank = u @ v.T
    noise = 0.01 * float(np.linalg.norm(low_rank)) / float(out_dim * in_dim) ** 0.5
    noise = noise * rng.normal(size=(out_dim, in_dim)).astype(np.float64)
    return (low_rank + noise).astype(np.float32)


def synthetic_act_scales(
    in_dim: int,
    seed: int,
) -> np.ndarray:
    """Synthetic per-input-channel activation magnitudes.

    Smooth ramp + small noise, so the smoothness-proxy weighting
    varies smoothly along the input axis. Without this, the proxy is
    unweighted and the LBFGS has no per-channel signal to follow.
    """
    rng = np.random.default_rng(seed + 1)
    ramp = np.linspace(0.5, 2.0, in_dim, dtype=np.float64)
    noise = 0.1 * rng.normal(size=in_dim)
    return (ramp + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Quantisation: per-row ternarisation (a thin slice of Tile640)
# ---------------------------------------------------------------------------


def ternarize_per_row(weight: np.ndarray) -> np.ndarray:
    """Ternarise a 2-D weight with a per-row mean(|W|) threshold.

    This is the per-row half of the Tile640 path; the per-lane and
    per-page scaling are stripped out for the A/B test, where we
    only care about the permutation effect. A full Tile640 A/B
    would call quantize_v3.quantize_2d on each permuted weight,
    but that requires the gguf / mlx imports.

    The per-row mean(|W|) is invariant to the input-channel
    permutation (a permutation just reorders the elements in each
    row, preserving the row's mean absolute value). So the per-row
    ternarisation result is itself permutation-invariant. The
    Tile640 path adds per-lane scaling, which IS sensitive to the
    permutation; see ``ternarize_per_lane`` below for the A/B
    test that exposes this.
    """
    threshold = np.mean(np.abs(weight), axis=1, keepdims=True)
    ternary = np.sign(weight) * np.where(
        np.abs(weight) >= threshold, threshold, np.float32(0.0)
    )
    return ternary.astype(np.float32)


def ternarize_per_lane(weight: np.ndarray, lane_size: int = 20) -> np.ndarray:
    """Ternarise a 2-D weight with a per-(row, lane) threshold.

    Mirrors the Tile640 per-lane scale: split each row into lanes
    of ``lane_size`` contiguous input channels and apply a
    per-lane mean(|W|) threshold. The per-lane threshold is
    permutation-sensitive (a permutation that reorders the
    channels can pack high-magnitude channels into the same lane
    and reduce the per-lane variance), so the LBFGS smoothness
    proxy should improve this metric.

    Quantised value: ``sign(W) * lane_threshold`` at kept
    positions, 0 elsewhere. The output is trimmed back to the
    original in_dim (zero-padding is only used for the lane
    grouping math).
    """
    out_dim, in_dim = weight.shape
    original_in_dim = in_dim
    if in_dim % lane_size != 0:
        # Pad with zeros to a multiple of lane_size.
        pad = lane_size - (in_dim % lane_size)
        weight = np.concatenate(
            [weight, np.zeros((out_dim, pad), dtype=weight.dtype)], axis=1
        )
        in_dim = weight.shape[1]
    weight_3d = weight.reshape(out_dim, in_dim // lane_size, lane_size)
    threshold = np.mean(np.abs(weight_3d), axis=2, keepdims=True)
    ternary_3d = np.sign(weight_3d) * np.where(
        np.abs(weight_3d) >= threshold, threshold, np.float32(0.0)
    )
    ternary = ternary_3d.reshape(out_dim, in_dim)
    if ternary.shape[1] != original_in_dim:
        ternary = ternary[:, :original_in_dim]
    return ternary.astype(np.float32)


def reconstruction_mse(
    weight: np.ndarray, perm: Optional[np.ndarray], quantizer: str = "per_row"
) -> Dict[str, float]:
    """Compute reconstruction MSE for a 2-D weight with a given
    input-channel permutation.

    ``perm = None`` means no permutation (baseline). The
    reconstruction is:
      W_perm = apply_champq_permutation(W, perm)  if perm is not None
      W_q_perm = ternarize(W_perm)
      W_q = apply_champq_permutation(W_q_perm, invert_champq_permutation(perm))

    ``quantizer = "per_row"`` uses the per-row mean(|W|)
    threshold; this is the calibration-time default and is
    invariant to the input-channel permutation. ``quantizer =
    "per_lane"`` uses a per-(row, lane) threshold (groups of 20
    channels) which IS permutation-sensitive. The full Tile640
    path is a per-lane quantiser plus additional per-page and
    per-row AWQ scales; for the A/B test the per-lane quantiser
    is enough to expose the permutation effect.
    """
    if perm is None:
        w_perm = weight
        if quantizer == "per_row":
            w_q_perm = ternarize_per_row(w_perm)
        elif quantizer == "per_lane":
            w_q_perm = ternarize_per_lane(w_perm)
        else:
            raise ValueError(f"unknown quantizer {quantizer!r}")
        w_q = w_q_perm
    else:
        w_perm = apply_champq_permutation(weight, perm)
        if quantizer == "per_row":
            w_q_perm = ternarize_per_row(w_perm)
        elif quantizer == "per_lane":
            w_q_perm = ternarize_per_lane(w_perm)
        else:
            raise ValueError(f"unknown quantizer {quantizer!r}")
        inv = invert_champq_permutation(perm)
        w_q = apply_champq_permutation(w_q_perm, inv)
    diff = (weight - w_q).astype(np.float64)
    mse = float(np.mean(diff * diff))
    max_err = float(np.max(np.abs(diff)))
    per_row_mse = np.mean(diff * diff, axis=1)
    return {
        "mse": mse,
        "max_err": max_err,
        "per_row_mse_mean": float(np.mean(per_row_mse)),
        "per_row_mse_std": float(np.std(per_row_mse)),
    }


# ---------------------------------------------------------------------------
# A/B driver
# ---------------------------------------------------------------------------


def run_ab(
    out_dim: int,
    in_dim: int,
    rank: int,
    seed: int,
    ga_population: int,
    ga_generations: int,
    lbfgs_iters: int,
    lbfgs_history: int,
    lbfgs_sinkhorn_iters: int,
    lbfgs_binariness: float,
    lbfgs_init: str,
    lbfgs_subsample: Optional[int],
    verbose: bool,
) -> Dict[str, object]:
    """Run the four-mode A/B and return a JSON-serialisable report.

    Args:
        lbfgs_subsample: when set, run the LBFGS on a sub-matrix of
            the first ``lbfgs_subsample`` input channels. Useful for
            K=4096 where the M-space matmul is heavy.
    """
    t0 = time.time()
    weight = synthetic_weight(out_dim, in_dim, rank, seed)
    act_scales = synthetic_act_scales(in_dim, seed)
    if verbose:
        print(
            f"synthetic: shape=({out_dim}, {in_dim}) rank={rank} seed={seed} "
            f"|W|={float(np.linalg.norm(weight)):.3e}",
            file=sys.stderr,
        )

    report: Dict[str, object] = {
        "schema": SCHEMA_AB,
        "shape": [int(out_dim), int(in_dim)],
        "rank": int(rank),
        "seed": int(seed),
        "weight_norm": float(np.linalg.norm(weight)),
        "act_scales_norm": float(np.linalg.norm(act_scales)),
        "modes": {},
    }

    def _add_mode(name: str, perm: Optional[np.ndarray], elapsed: float, **extras: object) -> None:
        smooth = smoothness_proxy(weight, perm if perm is not None else np.arange(in_dim, dtype=np.int64), act_scales)
        mse_row = reconstruction_mse(weight, perm, quantizer="per_row")
        mse_lane = reconstruction_mse(weight, perm, quantizer="per_lane")
        report["modes"][name] = {
            "smoothness": smooth,
            "mse_per_row": mse_row["mse"],
            "mse_per_lane": mse_lane["mse"],
            "max_err_per_lane": mse_lane["max_err"],
            "per_row_mse_mean_per_lane": mse_lane["per_row_mse_mean"],
            "per_row_mse_std_per_lane": mse_lane["per_row_mse_std"],
            "elapsed_s": elapsed,
            **extras,
        }
        if verbose:
            print(
                f"  mode={name:8s}  mse_per_row={mse_row['mse']:.4e}  "
                f"mse_per_lane={mse_lane['mse']:.4e}  smooth={smooth:.4e}  "
                f"t={elapsed:.2f}s",
                file=sys.stderr,
            )

    # Mode (a): no permutation.
    t = time.time()
    _add_mode("none", None, time.time() - t)

    # Mode (b): random permutation.
    t = time.time()
    perm_r = random_permutation(in_dim, seed=seed)
    _add_mode("random", perm_r, time.time() - t, seed=seed)

    # Mode (c): GA permutation.
    t = time.time()
    perm_g = ga_permutation(
        weight,
        act_scales=act_scales,
        population=ga_population,
        generations=ga_generations,
        seed=seed,
    )
    _add_mode(
        "ga",
        perm_g,
        time.time() - t,
        population=ga_population,
        generations=ga_generations,
    )

    # Mode (d): LBFGS permutation. When ``lbfgs_subsample`` is set
    # and smaller than in_dim, the LBFGS operates on a sub-matrix
    # of the first ``lbfgs_subsample`` columns and the permutation
    # is the identity on the remaining channels.
    t = time.time()
    if lbfgs_subsample is not None and lbfgs_subsample < in_dim:
        if verbose:
            print(
                f"  lbfgs subsample: K={lbfgs_subsample} (instead of {in_dim})",
                file=sys.stderr,
            )
        weight_sub = weight[:, :lbfgs_subsample]
        act_sub = act_scales[:lbfgs_subsample]
        perm_sub = lbfgs_permutation(
            weight_sub,
            act_scales=act_sub,
            n_iters=lbfgs_iters,
            history=lbfgs_history,
            sinkhorn_iters=lbfgs_sinkhorn_iters,
            binariness=lbfgs_binariness,
            init=lbfgs_init,
            seed=seed,
            verbose=verbose,
        )
        # Embed the sub-permutation into the full channel space.
        perm_l = np.arange(in_dim, dtype=np.int64)
        perm_l[:lbfgs_subsample] = perm_sub
        perm_l[lbfgs_subsample:] = np.arange(lbfgs_subsample, in_dim, dtype=np.int64)
    else:
        perm_l = lbfgs_permutation(
            weight,
            act_scales=act_scales,
            n_iters=lbfgs_iters,
            history=lbfgs_history,
            sinkhorn_iters=lbfgs_sinkhorn_iters,
            binariness=lbfgs_binariness,
            init=lbfgs_init,
            seed=seed,
            verbose=verbose,
        )
    _add_mode(
        "lbfgs",
        perm_l,
        time.time() - t,
        iters=lbfgs_iters,
        history=lbfgs_history,
        sinkhorn_iters=lbfgs_sinkhorn_iters,
        binariness=lbfgs_binariness,
        init=lbfgs_init,
        subsample=lbfgs_subsample,
        perm_diff_vs_random=int(np.sum(perm_l != perm_r)),
    )

    report["total_elapsed_s"] = time.time() - t0
    return report


def print_table(report: Dict[str, object]) -> None:
    """Print a one-row-per-mode table to stdout."""
    modes = report["modes"]
    # Column widths chosen so the table fits in a normal terminal.
    cols = [
        ("mode", 8, "<", "s"),
        ("mse_per_row", 12, ">", "e"),
        ("mse_per_lane", 12, ">", "e"),
        ("smoothness", 13, ">", "e"),
        ("per_row_mean", 14, ">", "e"),
        ("elapsed_s", 9, ">", "f"),
    ]
    # Two-space gap between columns.
    gap = "  "
    header = gap.join(f"{name:>{w}}" for name, w, _, _ in cols)
    print(header)
    print("-" * len(header))
    for mode in ("none", "random", "ga", "lbfgs"):
        if mode not in modes:
            continue
        m = modes[mode]
        cells = [mode]
        cells.append(f"{m['mse_per_row']:.3e}")
        cells.append(f"{m['mse_per_lane']:.3e}")
        cells.append(f"{m['smoothness']:.3e}")
        cells.append(f"{m['per_row_mse_mean_per_lane']:.3e}")
        cells.append(f"{m['elapsed_s']:.2f}")
        # Apply alignment per column.
        out = []
        for (name, w, align, _fmt), val in zip(cols, cells):
            if align == "<":
                out.append(f"{val:<{w}}")
            else:
                out.append(f"{val:>{w}}")
        print(gap.join(out))
    # Verdict: did LBFGS beat the others?
    if all(k in modes for k in ("none", "random", "ga", "lbfgs")):
        for metric, label in (("mse_per_row", "per-row MSE"), ("mse_per_lane", "per-lane MSE")):
            best = min(
                ((k, modes[k][metric]) for k in ("none", "random", "ga", "lbfgs")),
                key=lambda kv: kv[1],
            )
            print()
            print(f"lowest {label}: {best[0]} ({best[1]:.3e})")
            mse_l = modes["lbfgs"][metric]
            for other in ("none", "random", "ga"):
                mse_o = modes[other][metric]
                delta = (mse_l - mse_o) / max(mse_o, 1e-30)
                print(f"  lbfgs_vs_{other}: {delta:+.3e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="champq_lbfgs_ab",
        description=(
            "A/B validation harness for the LBFGS CHAMP-Q permutation mode. "
            "Compares none / random / ga / lbfgs on a synthetic 4096x4096 rank-8 + Gaussian weight."
        ),
    )
    parser.add_argument("--out-dim", type=int, default=DEFAULT_OUT_DIM)
    parser.add_argument("--in-dim", type=int, default=DEFAULT_IN_DIM)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ga-population", type=int, default=8)
    parser.add_argument("--ga-generations", type=int, default=5)
    parser.add_argument("--lbfgs-iters", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=8)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "--lbfgs-binariness",
        type=float,
        default=1.0e-3,
        help="Binariness penalty weight for the LBFGS objective.",
    )
    parser.add_argument(
        "--lbfgs-init",
        choices=("l2rank", "random", "identity"),
        default="l2rank",
    )
    parser.add_argument(
        "--lbfgs-subsample",
        type=int,
        default=1024,
        help="Run LBFGS on a sub-matrix of this many channels (default: 1024). "
        "Use --lbfgs-subsample 0 to disable subsampling and run on the full in_dim.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the JSON report.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    lbfgs_subsample = args.lbfgs_subsample if args.lbfgs_subsample > 0 else None
    report = run_ab(
        out_dim=args.out_dim,
        in_dim=args.in_dim,
        rank=args.rank,
        seed=args.seed,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
        lbfgs_iters=args.lbfgs_iters,
        lbfgs_history=args.lbfgs_history,
        lbfgs_sinkhorn_iters=args.sinkhorn_iters,
        lbfgs_binariness=args.lbfgs_binariness,
        lbfgs_init=args.lbfgs_init,
        lbfgs_subsample=lbfgs_subsample,
        verbose=args.verbose,
    )
    print_table(report)
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
