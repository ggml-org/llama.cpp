#!/usr/bin/env python3
"""L5 orchestrator demo on a synthetic 10-tensor workload.

Runs the L5 orchestrator against a small synthetic L4 report so the
behaviour can be inspected without a real model.  The synthetic report
contains ten tensors spread across three transformer blocks plus a token
embedding and an output projection, with deliberately uneven MSE so a
visible number of tensors get requantized.

Outputs:

* The orchestrator's stdout summary (JSON, easy to grep).
* ``l5_demo_policy.json`` - the sidecar policy consumable by
  ``tile640_quantize_v3.py``.
* ``l5_demo_history.ndjson`` - the per-iteration plan history
  (one record per (plan, action) pair, conformant to
  ``common/schemas/l5_plan.schema.json``).

By default the demo writes the policy and history into a temporary
directory so it does not pollute the working tree.  Pass ``--out-dir`` to
override the destination.

Run with ``python3 tools/tessera/l5_demo.py``.  No arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

# Use the package-relative import when available, otherwise anchor sys.path
# to the directory holding this script (script-mode fallback).
try:
    from . import l5_metrics as metrics
    from . import l5_orchestrator as orch
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import l5_metrics as metrics  # type: ignore[no-redef]
    import l5_orchestrator as orch  # type: ignore[no-redef]


# Synthetic L4 report.  Ten tensors: three transformer blocks, plus a
# token embedding and an output projection.  The MSE values are chosen so
# the orchestrator's top-10% cohort hits the late FFN tensors where the
# bit width matters most, and the bottom-5% cohort hits the early
# attention projection where the AWQ scale already covers the variance.
SYNTHETIC_L4 = {
    "schema": "llama.tessera.e2e-probe.v1",
    "model": "demo-10t",
    "tensors": {
        "blk.0.attn_q.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0021,
            "mse_minus_one": 0.0024,
            "perplexity": 7.91,
            "top1_mismatch": 0.012,
            "n_weights": 4096 * 4096,
        },
        "blk.0.attn_k.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0019,
            "mse_minus_one": 0.0022,
            "perplexity": 7.88,
            "top1_mismatch": 0.010,
            "n_weights": 4096 * 1024,
        },
        "blk.1.ffn_down.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0080,
            "mse_minus_one": 0.0112,
            "perplexity": 8.05,
            "top1_mismatch": 0.034,
            "n_weights": 4096 * 14336,
        },
        "blk.2.attn_v.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0030,
            "mse_minus_one": 0.0034,
            "perplexity": 7.93,
            "top1_mismatch": 0.018,
            "n_weights": 4096 * 1024,
        },
        "blk.2.ffn_up.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0078,
            "mse_minus_one": 0.0108,
            "perplexity": 8.04,
            "top1_mismatch": 0.030,
            "n_weights": 14336 * 4096,
        },
        "blk.3.ffn_gate.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0064,
            "mse_minus_one": 0.0086,
            "perplexity": 8.00,
            "top1_mismatch": 0.024,
            "n_weights": 14336 * 4096,
        },
        "blk.3.ffn_down.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0120,
            "mse_minus_one": 0.0180,
            "perplexity": 8.18,
            "top1_mismatch": 0.044,
            "n_weights": 4096 * 14336,
        },
        "blk.4.attn_out.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0040,
            "mse_minus_one": 0.0048,
            "perplexity": 7.96,
            "top1_mismatch": 0.020,
            "n_weights": 4096 * 4096,
        },
        "token_embd.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0050,
            "mse_minus_one": 0.0056,
            "perplexity": 7.99,
            "top1_mismatch": 0.022,
            "n_weights": 32000 * 4096,
        },
        "output.weight": {
            "current_qtype": "Q4_K",
            "mse": 0.0090,
            "mse_minus_one": 0.0130,
            "perplexity": 8.10,
            "top1_mismatch": 0.036,
            "n_weights": 32000 * 4096,
        },
    },
}

# Synthetic imatrix.  Larger magnitudes align with the higher-MSE tensors
# so the imatrix component reinforces the gradient component rather than
# contradicting it.  This is intentional: in a real run the two would
# sometimes disagree, and the orchestrator's combine step is what we are
# demonstrating.
SYNTHETIC_IMATRIX = {
    "blk.0.attn_q.weight": 0.40,
    "blk.0.attn_k.weight": 0.42,
    "blk.1.ffn_down.weight": 1.00,
    "blk.2.attn_v.weight": 0.45,
    "blk.2.ffn_up.weight": 0.95,
    "blk.3.ffn_gate.weight": 0.70,
    "blk.3.ffn_down.weight": 1.20,
    "blk.4.attn_out.weight": 0.55,
    "token_embd.weight": 0.60,
    "output.weight": 1.05,
}


def _megabytes(bits: int) -> float:
    return bits / 8.0 / 1024.0 / 1024.0


def _format_table(rows: list[tuple[str, str, str, str, str]]) -> str:
    headers = ("tensor", "from", "to", "mse_delta", "storage_delta")
    body = [headers] + list(rows)
    widths = [max(len(str(row[i])) for row in body) for i in range(len(headers))]
    out: list[str] = []
    for index, row in enumerate(body):
        out.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "L5 orchestrator demo on a synthetic 10-tensor workload. "
            "Writes the policy and history into a temp directory by "
            "default so the working tree stays clean."
        )
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for the policy and history files (default: tempdir)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of requantization passes (default 5)",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Top cohort fraction per iteration (default 0.10)",
    )
    parser.add_argument(
        "--bottom-fraction",
        type=float,
        default=0.05,
        help="Bottom cohort fraction per iteration (default 0.05)",
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=0.0035,
        help="MSE threshold below which a tensor is considered converged (default 0.0035)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="l5_demo_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "l5_demo_policy.json"
    history_path = output_dir / "l5_demo_history.ndjson"

    scorer = orch.SensitivityScorer(
        decay=0.9,
        weights=metrics.DEFAULT_WEIGHTS,
        total_layers=5,
    )
    planner = orch.RequantPlanner(
        top_fraction=args.top_fraction,
        bottom_fraction=args.bottom_fraction,
        divergence_threshold=args.divergence_threshold,
    )
    loop = orch.OrchestratorLoop(
        scorer=scorer,
        planner=planner,
        apply=None,
        max_iterations=args.max_iterations,
        divergence_threshold=args.divergence_threshold,
        sidecar=policy_path,
        verbose=False,
    )

    plans = loop.run(SYNTHETIC_L4, SYNTHETIC_IMATRIX)
    orch.OrchestratorLoop.write_history(plans, history_path)

    # Print a human-readable summary.
    print("=" * 78)
    print("L5 orchestrator demo (synthetic 10-tensor workload)")
    print("=" * 78)
    print(f"policy:  {policy_path}")
    print(f"history: {history_path}")
    print()

    # Per-iteration plan dump.
    for plan in plans:
        print(
            f"iter {plan.iteration}: storage "
            f"{_megabytes(plan.storage_before_bits):.2f} MB -> "
            f"{_megabytes(plan.storage_after_bits):.2f} MB, "
            f"actions={len(plan.actions)}"
        )
        if plan.termination_reason:
            print(f"  termination_reason: {plan.termination_reason}")
        if plan.actions:
            rows = [
                (
                    action.name,
                    action.from_qtype,
                    action.to_qtype,
                    f"{action.expected_mse_delta:+.4e}",
                    f"{action.storage_delta_bits / 8 / 1024 / 1024:+.2f} MB",
                )
                for action in plan.actions
            ]
            print(_format_table(rows))
        print()

    # Sidecar JSON validity check.
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        print(
            f"sidecar policy schema={policy.get('schema')!r} "
            f"with {len(policy.get('tensor_families', {}))} tensor family entries"
        )

    # Top sensitivity check: which tensors scored highest on the final EMA.
    final = plans[-1]
    ranked = sorted(
        final.ema_sensitivity.items(), key=lambda kv: kv[1], reverse=True
    )
    print()
    print("Final EMA sensitivity ranking (top 5):")
    for name, score in ranked[:5]:
        print(f"  {name:35s}  {score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
