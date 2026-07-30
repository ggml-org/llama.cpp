#!/usr/bin/env python3
"""DartQuant demo harness.

Runs the pre-rotation mode end-to-end on a synthetic 2-D weight matrix
(default 1024x1024; can be raised to 4096x4096 for the production
target) and reports:

* Whip loss before/after the QR-Orth optimisation
* Tile-quant relative MSE before/after
* Output MSE before/after (when a calibration-like X is supplied)
* Policy JSON size (rotation matrix + tile-quant knobs)
* Wall-clock time

The synthetic weight and activation follow a "skewed-channel" recipe
chosen to exercise the rotation: a small fraction of output rows are
scaled up (the analogue of FFN output outliers) and the calibration
activation has heavy-tailed per-input-channel scale. This is the
canonical "outlier" scenario that rotational calibration is meant to
address.

The demo is intentionally conservative: it runs a single seed and a
single (lr, whip_weight) pair. The numbers below are the honest result
of that single run. Operators tuning the policy should sweep ``--lr``
and ``--whip-weight`` and pick the best of multiple seeds.

Example:

    python3 tools/tessera/dartquant_demo.py \\
        --output-dir /tmp/dartquant-demo \\
        --weight-dim 1024 --num-samples 64 \\
        --orth-iters 50 --orth-lr 0.1 --whip-weight 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Allow running directly from a worktree without installation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.tessera.per_tensor_calibrate import (  # noqa: E402
    build_dartquant_policy,
    dartquant_apply_rotation,
    dartquant_qr_orth,
    dartquant_whip_loss,
    _relative_quant_mse,
)
from tools.tessera._dartquant_linalg import (  # noqa: E402
    random_orthogonal,
)


def _synthetic(
    out_dim: int,
    in_dim: int,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic (W, x, X, X_hat) bundle.

    The weight has a per-input-channel structure: a small fraction of
    the input columns are 5x stronger than the rest. The calibration
    activations are iid Gaussian per token multiplied by the same
    per-input-channel scale, mirroring the "activation aligned with the
    strong weight columns" scenario that DartQuant is built for. This
    is the canonical "input-side outliers" pattern observed in real
    transformer FFNs.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(
        loc=0.0, scale=1.0 / math.sqrt(in_dim), size=(out_dim, in_dim)
    ).astype(np.float32)
    col_scale = np.ones(in_dim, dtype=np.float32)
    strong = rng.choice(in_dim, size=max(1, in_dim // 5), replace=False)
    col_scale[strong] = 5.0
    W = (W * col_scale[None, :]).astype(np.float32)
    x_scale = col_scale.astype(np.float32)
    x = rng.normal(loc=0.0, scale=1.0, size=(num_samples, in_dim)).astype(
        np.float32
    )
    x = (x * x_scale[None, :]).astype(np.float32)
    X_hat = (x_scale.astype(np.float32) ** 2).astype(np.float32)
    return W, x, X_hat, x_scale


def _bf16_round(W: np.ndarray) -> np.ndarray:
    """Round-trip through a FP16/BF16 surrogate: same as the input."""
    return W.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--weight-dim",
        type=int,
        default=1024,
        help="Square weight dimension (default 1024; production target 4096)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of calibration activation rows (default 64)",
    )
    parser.add_argument(
        "--orth-iters",
        type=int,
        default=50,
        help="QR-Orth iterations per tensor (default 50)",
    )
    parser.add_argument(
        "--orth-lr",
        type=float,
        default=0.1,
        help="Stiefel step size (default 0.1)",
    )
    parser.add_argument(
        "--whip-weight",
        type=float,
        default=0.1,
        help="Whip-loss weight in the combined fitness (default 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-rotation",
        choices=("identity", "random"),
        default="random",
        help=(
            "Initial rotation: 'random' starts from a Haar-uniform "
            "orthogonal matrix (more diverse but slower); 'identity' "
            "starts from R = I (closer to the canonical AWQ baseline)."
        ),
    )
    args = parser.parse_args()

    W, X, X_hat, x_scale = _synthetic(
        args.weight_dim, args.weight_dim, args.num_samples, args.seed
    )
    print(
        f"synthetic: W={W.shape}  X={X.shape}  "
        f"x_scale min={x_scale.min():.3f} max={x_scale.max():.3f} "
        f"median={np.median(x_scale):.3f}"
    )

    # Reference metrics on the un-rotated weight.
    base_quant_mse = _relative_quant_mse(W)
    base_whip = dartquant_whip_loss(W, X_hat=X_hat)
    print(
        f"baseline (no rotation):  quant_mse={base_quant_mse:.4e}  "
        f"whip={base_whip:.4e}"
    )

    # Initial rotation: identity (default) or random orthogonal.
    if args.init_rotation == "random":
        R0 = random_orthogonal(args.weight_dim, seed=args.seed).astype(
            np.float32
        )
    else:
        R0 = np.eye(args.weight_dim, dtype=np.float32)
    init_quant_mse = _relative_quant_mse(dartquant_apply_rotation(W, R0))
    init_whip = dartquant_whip_loss(
        dartquant_apply_rotation(W, R0), X_hat=X_hat
    )
    print(
        f"initial R ({args.init_rotation}):  quant_mse={init_quant_mse:.4e}  "
        f"whip={init_whip:.4e}"
    )

    # Run QR-Orth.
    t0 = time.time()
    result = dartquant_qr_orth(
        W,
        X=X,
        X_hat=X_hat,
        n_iters=args.orth_iters,
        lr=args.orth_lr,
        whip_weight=args.whip_weight,
        seed=args.seed if args.init_rotation == "random" else 0,
        verbose=True,
    )
    elapsed = time.time() - t0
    W_final = dartquant_apply_rotation(W, result.rotation)
    final_quant_mse = _relative_quant_mse(W_final)
    final_whip = dartquant_whip_loss(W_final, X_hat=X_hat)

    print(
        f"after {args.orth_iters} iters ({elapsed:.2f}s):  "
        f"quant_mse={final_quant_mse:.4e}  "
        f"whip={final_whip:.4e}  "
        f"output_mse={result.final_output_mse:.4e}"
    )

    # Honest verdict.
    if final_quant_mse < base_quant_mse:
        verdict_quant = "improved"
    elif final_quant_mse == base_quant_mse:
        verdict_quant = "unchanged"
    else:
        verdict_quant = "REGRESSED"
    if final_whip < base_whip:
        verdict_whip = "improved"
    elif final_whip == base_whip:
        verdict_whip = "unchanged"
    else:
        verdict_whip = "regressed"
    print(
        f"verdict: tile-quant {verdict_quant}  "
        f"whip-loss {verdict_whip}  "
        f"(output MSE  before={result.initial_output_mse:.4e}  "
        f"after={result.final_output_mse:.4e})"
    )

    # Persist the policy JSON.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "tool": "dartquant_demo.py",
        "mode": "dartquant",
        "weight_dim": args.weight_dim,
        "num_samples": args.num_samples,
        "orth_iters": args.orth_iters,
        "orth_lr": args.orth_lr,
        "whip_weight": args.whip_weight,
        "seed": args.seed,
        "init_rotation": args.init_rotation,
        "elapsed_seconds": elapsed,
        "timestamp": time.time(),
    }
    from tools.tessera.per_tensor_calibrate import (
        Layer,
        DARTQUANT_SCHEMA,
    )

    bundle = Layer(
        name="synthetic",
        family="ffn",
        weight=W,
        train_activations=X,
        heldout_activations=None,
        in_sum2=None,
        in_count=0,
    )
    result.bundle_name = "synthetic"
    policy = build_dartquant_policy(
        [(bundle, result)], provenance=provenance
    )
    policy_path = output_dir / "dartquant-policy.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {policy_path}  schema={policy['schema']}  "
          f"dartquant={policy['dartquant']['schema']}  "
          f"total_bytes={policy['dartquant']['total_bytes']}")

    # Side-by-side report (markdown).
    report = (
        f"# DartQuant demo report\n\n"
        f"weight shape: {list(W.shape)}\n"
        f"calibration tokens: {X.shape[0]}\n"
        f"orth iters: {args.orth_iters}   lr: {args.orth_lr}   "
        f"whip weight: {args.whip_weight}\n"
        f"elapsed: {elapsed:.2f}s\n\n"
        f"| metric | before (R=I) | initial R ({args.init_rotation}) | after QR-Orth |\n"
        f"| --- | --- | --- | --- |\n"
        f"| tile-quant MSE | {base_quant_mse:.4e} | {init_quant_mse:.4e} | {final_quant_mse:.4e} |\n"
        f"| Whip loss | {base_whip:.4e} | {init_whip:.4e} | {final_whip:.4e} |\n"
        f"| output MSE | n/a | {result.initial_output_mse:.4e} | {result.final_output_mse:.4e} |\n"
        f"\n"
        f"verdict: tile-quant {verdict_quant}; whip-loss {verdict_whip}\n"
    )
    report_path = output_dir / "dartquant-report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
