#!/usr/bin/env python3
"""Run the SEPTQ A/B validation on the v1 synthetic + realistic regression bundles.

Runs ``septq_ab_validate.py`` in all four importance modes
(quant_error_h, inv_abs_w, inv_cdf, hybrid) and both Hessian modes
(diagonal, banded) on the synthetic, and in the four importance modes
(diagonal only, since the realistic .npz bundle does not carry raw
calibration activations) on the realistic.

Output is written to ``/tmp/septq_validation/`` as JSON and markdown.

Reference numbers:
  v1 synthetic (no tail):       +92.88% diagonal, +91.25% banded b=32
  v1 realistic (heavy tail):    -24% with original; inv_cdf recovers to +69%

Run: ``python3 tools/tessera/septq_validate_bundles.py``
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
BUNDLE_DIR = Path("/tmp/septq_bundles")
OUT_DIR = Path("/tmp/septq_validation")
SYNTHETIC = BUNDLE_DIR / "synthetic_4096x4096.npz"
REALISTIC = BUNDLE_DIR / "realistic_4096x4096.npz"
HARNESS = HERE / "septq_ab_validate.py"

MODES: List[Tuple[str, float]] = [
    ("quant_error_h", 0.0),
    ("inv_abs_w", 0.0),
    ("inv_cdf", 0.0),
    ("hybrid", 1.0),
]


def run_synthetic(mode: str, lam: float, hessian: str) -> Dict[str, Any]:
    tag = f"synthetic_{mode}_l{lam}_{hessian}"
    out_json = OUT_DIR / f"{tag}.json"
    # --synthetic generates raw calibration activations so the banded
    # mode is exercisable. The harness's synthetic construction matches
    # the .npz bundle (same seed, rank, n_calib, dims) so the diagonal
    # rows from --synthetic and the .npz bundle are identical.
    cmd = [
        sys.executable, str(HARNESS),
        "--synthetic",
        "--septq-hessian-mode", hessian,
        "--septq-hessian-bandwidth", "32",
        "--septq-importance-weight", mode,
        "--septq-importance-lambda", str(lam),
        "--output", str(out_json),
        "--quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"{tag}: harness failed (rc={r.returncode})\n"
            f"stderr: {r.stderr[-1000:]}"
        )
    with open(out_json) as f:
        rep = json.load(f)
    agg = rep["aggregate"]
    return {
        "bundle": "synthetic_4096x4096",
        "mode": mode,
        "lambda": lam,
        "hessian": hessian,
        "rtn_mse": agg["mean_baseline_mse"],
        "septq_mse": agg["mean_septq_mse"],
        "improve_pct": agg["mean_mse_improvement_pct"],
        "septq_wins": agg["septq_wins"],
        "baseline_wins": agg["baseline_wins"],
        "ties": agg["ties"],
    }


def run_realistic(mode: str, lam: float) -> Dict[str, Any]:
    tag = f"realistic_{mode}_l{lam}_diagonal"
    out_json = OUT_DIR / f"{tag}.json"
    # The realistic .npz bundle does not carry raw calibration
    # activations; only the diagonal imatrix (in_sum2) is stored. The
    # banded SEPTQ mode would silently fall back to diagonal here, so
    # we only run the diagonal mode for this bundle.
    cmd = [
        sys.executable, str(HARNESS),
        "--bundle", str(REALISTIC),
        "--septq-hessian-mode", "diagonal",
        "--septq-importance-weight", mode,
        "--septq-importance-lambda", str(lam),
        "--output", str(out_json),
        "--quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"{tag}: harness failed (rc={r.returncode})\n"
            f"stderr: {r.stderr[-1000:]}"
        )
    with open(out_json) as f:
        rep = json.load(f)
    agg = rep["aggregate"]
    return {
        "bundle": "realistic_4096x4096",
        "mode": mode,
        "lambda": lam,
        "hessian": "diagonal",
        "rtn_mse": agg["mean_baseline_mse"],
        "septq_mse": agg["mean_septq_mse"],
        "improve_pct": agg["mean_mse_improvement_pct"],
        "septq_wins": agg["septq_wins"],
        "baseline_wins": agg["baseline_wins"],
        "ties": agg["ties"],
    }


def build_synthetic_table(results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# SEPTQ v1 synthetic regression")
    lines.append("")
    lines.append("Bundle: synthetic 4096x4096 (rank-8 + Gaussian, no heavy tail).")
    lines.append("septq_ratio = 0.5. hessian_bandwidth = 32.")
    lines.append(
        "RTN baseline = standard tessera 2D (outlier_frac=0.005, "
        "awq_alpha=0.0, ternary_threshold=1.0)."
    )
    lines.append("")
    lines.append(
        f"{'mode':15s} {'lambda':>6s}  {'hessian':>9s}  "
        f"{'RTN MSE':>12s}  {'SEPTQ MSE':>12s}  "
        f"{'improve%':>10s}  {'wins (s/b/t)':>14s}"
    )
    lines.append("-" * 100)
    for r in results:
        wins = (
            f"{r['septq_wins']}/{r['baseline_wins']}/{r['ties']}"
        )
        lines.append(
            f"{r['mode']:15s} {r['lambda']:>6.2f}  "
            f"{r['hessian']:>9s}  "
            f"{r['rtn_mse']:>12.4e}  {r['septq_mse']:>12.4e}  "
            f"{r['improve_pct']:>+9.2f}%  {wins:>14s}"
        )
    lines.append("")
    lines.append(
        "Positive improve% = SEPTQ wins (lower MSE than RTN). "
        "Negative = SEPTQ loses."
    )
    lines.append("")
    lines.append(
        "Reference: v1 commit 6179dc753 reported +44% MSE at ratio=0.5 "
        "based on a column-major storage bug in the ternarize output. The "
        "corrected number on the v1 synthetic is +92.88% (diagonal) and "
        "+91.25% (banded b=32)."
    )
    return "\n".join(lines) + "\n"


def build_realistic_table(results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# SEPTQ realistic bundle regression (heavy-tailed)")
    lines.append("")
    lines.append(
        "Bundle: realistic 4096x4096 (rank-32 + Gaussian + 0.1% "
        "Student-t(3) outliers at 30x bulk std)."
    )
    lines.append("septq_ratio = 0.5. hessian = diagonal (bundles do not carry raw activations).")
    lines.append(
        "RTN baseline = standard tessera 2D (outlier_frac=0.005, "
        "awq_alpha=0.0, ternary_threshold=1.0)."
    )
    lines.append("")
    lines.append(
        f"{'mode':15s} {'lambda':>6s}  "
        f"{'RTN MSE':>12s}  {'SEPTQ MSE':>12s}  "
        f"{'improve%':>10s}  {'wins (s/b/t)':>14s}"
    )
    lines.append("-" * 90)
    for r in results:
        wins = (
            f"{r['septq_wins']}/{r['baseline_wins']}/{r['ties']}"
        )
        lines.append(
            f"{r['mode']:15s} {r['lambda']:>6.2f}  "
            f"{r['rtn_mse']:>12.4e}  {r['septq_mse']:>12.4e}  "
            f"{r['improve_pct']:>+9.2f}%  {wins:>14s}"
        )
    lines.append("")
    lines.append(
        "Positive improve% = SEPTQ wins (lower MSE than RTN). "
        "Negative = SEPTQ loses."
    )
    lines.append("")
    lines.append(
        "The original (quant_error_h) importance score loses on this "
        "bundle because (W - Q(W))^2 * h_diag is dominated by the "
        "heavy-tail outliers -- they have the largest ternarization "
        "error, so the mask picks them, and ternarization then throws "
        "away their full-precision values. The inv_cdf mode uses the "
        "per-row 1 - CDF(|W|) weight to push the outliers out of the "
        "mask, leaving them as full-precision residuals."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SYNTHETIC.exists() or not REALISTIC.exists():
        print(
            f"error: bundles not found in {BUNDLE_DIR}; run "
            f"tools/tessera/septq_build_bundles.py first",
            file=sys.stderr,
        )
        return 2
    # Synthetic: diagonal + banded (banded needs raw activations).
    print("=== synthetic (rank-8 + Gaussian, no heavy tail) ===")
    synthetic_results: List[Dict[str, Any]] = []
    for mode, lam in MODES:
        for hessian in ("diagonal", "banded"):
            tag = f"synthetic_{mode}_l{lam}_{hessian}"
            print(f"running: {tag}")
            synthetic_results.append(run_synthetic(mode, lam, hessian))
    synth_table = build_synthetic_table(synthetic_results)
    synth_path = OUT_DIR / "synthetic_ab_table.md"
    synth_path.write_text(synth_table, encoding="utf-8")
    print()
    print(synth_table)
    print(f"wrote {synth_path}")
    # Realistic: diagonal only (bundles do not carry raw activations).
    print()
    print("=== realistic (rank-32 + 0.1% Student-t(3) at 30x std) ===")
    realistic_results: List[Dict[str, Any]] = []
    for mode, lam in MODES:
        tag = f"realistic_{mode}_l{lam}_diagonal"
        print(f"running: {tag}")
        realistic_results.append(run_realistic(mode, lam))
    real_table = build_realistic_table(realistic_results)
    real_path = OUT_DIR / "realistic_ab_table.md"
    real_path.write_text(real_table, encoding="utf-8")
    print()
    print(real_table)
    print(f"wrote {real_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
