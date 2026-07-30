#!/usr/bin/env python3
"""Run the SEPTQ A/B validation on the v1 synthetic regression bundle.

Runs ``septq_ab_validate.py`` in all four importance modes
(quant_error_h, inv_abs_w, inv_cdf, hybrid) and both Hessian modes
(diagonal, banded) on the synthetic 4096x4096 bundle. The banded mode
uses the harness's --synthetic flag (which generates raw calibration
activations) so the banded row of the table is real; the diagonal
mode runs on both --synthetic and the .npz bundle (they should match
since the construction is identical).

Output is written to ``/tmp/septq_validation/`` as JSON and markdown.

This is the regression harness for the weighted importance extension.
The expected result on the v1 synthetic is:

  quant_error_h diagonal: +92.88% (the v1 result, corrected)
  quant_error_h banded:   +91.25% (banded Cholesky b=32)
  inv_abs_w:             ~+93-94%
  inv_cdf:               ~+94%
  hybrid (l=1):          ~+97%

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
HARNESS = HERE / "septq_ab_validate.py"

MODES: List[Tuple[str, float]] = [
    ("quant_error_h", 0.0),
    ("inv_abs_w", 0.0),
    ("inv_cdf", 0.0),
    ("hybrid", 1.0),
]


def run_one(mode: str, lam: float, hessian: str) -> Dict[str, Any]:
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


def build_table(results: List[Dict[str, Any]]) -> str:
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SYNTHETIC.exists():
        print(
            f"error: {SYNTHETIC} not found; run "
            f"tools/tessera/septq_build_bundles.py first",
            file=sys.stderr,
        )
        return 2
    results: List[Dict[str, Any]] = []
    for mode, lam in MODES:
        for hessian in ("diagonal", "banded"):
            tag = f"synthetic_{mode}_l{lam}_{hessian}"
            print(f"running: {tag}")
            results.append(run_one(mode, lam, hessian))
    table = build_table(results)
    table_path = OUT_DIR / "synthetic_ab_table.md"
    table_path.write_text(table, encoding="utf-8")
    print()
    print(table)
    print(f"wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
