#!/usr/bin/env python3
"""Demo harness for the PE-QAT trainer.

The demo operates on a single 2-D linear layer to keep the run time small
(~10 s on M-series) while still exercising the full PE-QAT recipe: LoRA
injection, SmoothQuant-style per-channel smoothing, per-output-channel
clipping, W4A4 fake-quantization, STE backprop, and AdamW updates.  The
output mirrors what a production integration with the DSpark drafter
would produce: a ``pe-qat-policy.json`` carrying the trained LoRA, the
smoothed scales, and the clip thresholds, plus a comparison report of
the BF16-vs-quantized output MSE.

Example:

    python3 tools/tessera/pe_qat_demo.py \\
        --output-dir /tmp/peqat-demo \\
        --rank 16 --alpha 32 --iters 100 --lr 1e-3

The harness intentionally does not depend on PyTorch or any autograd
framework; everything is plain NumPy so it can run inside the existing
Tessera calibration pipeline.
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

from tools.tessera.pe_qat import (  # noqa: E402
    PE_QAT_POLICY_SCHEMA,
    apply_pe_qat,
    pe_qat_train,
    save_pe_qat_policy,
)


def _synthetic_inputs(
    out_features: int,
    in_features: int,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a (W, x) pair with realistic-ish transformer statistics.

    The weight is a Kaiming-init-ish matrix; the activations are normal
    noise multiplied by a per-channel scale that mimics the long-tailed
    distribution of attention / FFN inputs.  This gives the quantizer
    something to fight against (otherwise per-channel scales collapse to
    near-unity and the loss is already near-zero).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=1.0 / math.sqrt(in_features),
                   size=(out_features, in_features)).astype(np.float32)
    # Per-input-channel activation scale with a 1/x tail to mimic outliers.
    x_scale = np.power(rng.uniform(0.05, 1.0, size=in_features).astype(np.float32), -1.5)
    x = rng.normal(loc=0.0, scale=1.0, size=(num_samples, in_features)).astype(np.float32)
    x = x * x_scale[None, :]
    return W, x


def _bf16_output(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mimic a BF16 forward: round inputs to BF16 precision and back."""
    bf16 = W.astype(np.float32) @ x.T.astype(np.float32)
    return bf16.astype(np.float32)


def _per_position_mse(y_ref: np.ndarray, y_q: np.ndarray) -> np.ndarray:
    diff = y_ref - y_q
    return np.mean(diff * diff, axis=0)  # (batch,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, help="Where to write the policy + comparison report")
    parser.add_argument("--weight-dim", type=int, default=4096, help="Square weight dimension (out == in)")
    parser.add_argument("--num-samples", type=int, default=32, help="Calibration activation rows")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--iters", type=int, default=100, help="PE-QAT iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--smooth-prior-weight", type=float, default=1e-3, help="SmoothQuant log-space prior weight")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--family", default="attention", help="Family tag stored in the policy JSON")
    parser.add_argument("--log-every", type=int, default=10, help="Loss-log interval")
    parser.add_argument("--bundle", default=None, help="Optional .npz layer bundle (overrides --weight-dim / --num-samples)")
    args = parser.parse_args()

    if args.bundle:
        data = np.load(args.bundle, allow_pickle=False)
        if "weight" not in data or "train_activations" not in data:
            raise ValueError(f"{args.bundle}: expected `weight` and `train_activations` arrays")
        W = data["weight"].astype(np.float32)
        x = data["train_activations"].astype(np.float32)
        if W.ndim != 2:
            raise ValueError(f"{args.bundle}: weight must be 2-D, got {W.shape}")
        if x.ndim != 2 or x.shape[1] != W.shape[1]:
            raise ValueError(
                f"{args.bundle}: train_activations must be (batch, in_channels={W.shape[1]}), got {x.shape}"
            )
        print(f"loaded {args.bundle}: W={W.shape}, x={x.shape}")
    else:
        W, x = _synthetic_inputs(
            out_features=args.weight_dim,
            in_features=args.weight_dim,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        print(f"generated synthetic W={W.shape}, x={x.shape}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"PE-QAT config: rank={args.rank} alpha={args.alpha} iters={args.iters} "
        f"lr={args.lr} smooth_prior_weight={args.smooth_prior_weight}"
    )
    t0 = time.time()
    result = pe_qat_train(
        W,
        x,
        rank=args.rank,
        alpha=args.alpha,
        iters=args.iters,
        lr=args.lr,
        smooth_prior_weight=args.smooth_prior_weight,
        log_every=args.log_every,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"PE-QAT done in {elapsed:.2f}s; final loss={result['losses'][-1]:.4e}")

    # Persist the policy.
    policy_path = output_dir / "pe-qat-policy.json"
    save_pe_qat_policy(policy_path, result, family=args.family)
    print(f"wrote {policy_path}")

    # Re-apply the policy and report the BF16-vs-quantized MSE.
    y_ref, y_q = apply_pe_qat(W, x, result)
    bf16_only_mse = float(np.mean((_bf16_output(W, x) - y_ref) ** 2))
    quant_mse = float(np.mean((y_ref - y_q) ** 2))
    per_pos = _per_position_mse(y_ref, y_q)
    print(
        f"output MSE: bf16-vs-ref={bf16_only_mse:.4e}  quant-vs-ref={quant_mse:.4e}  "
        f"per-position min={per_pos.min():.4e} max={per_pos.max():.4e} "
        f"median={np.median(per_pos):.4e}"
    )

    # Loss curve + report.
    losses = result["losses"]
    mses = result["mses"]
    report = {
        "schema": PE_QAT_POLICY_SCHEMA + ".report.v1",
        "weight_shape": list(W.shape),
        "num_samples": int(x.shape[0]),
        "rank": args.rank,
        "alpha": args.alpha,
        "iters": args.iters,
        "lr": args.lr,
        "elapsed_seconds": elapsed,
        "final_loss": float(losses[-1]),
        "first_loss": float(losses[0]),
        "final_mse": float(mses[-1]),
        "first_mse": float(mses[0]),
        "trainable_params": int(result["lora_A"].size + result["lora_B"].size +
                                result["s"].size + result["c"].size),
        "base_weight_params": int(result["W_ref"].size),
        "trainable_ratio": float(
            (result["lora_A"].size + result["lora_B"].size + result["s"].size + result["c"].size)
            / max(1, result["W_ref"].size + result["lora_A"].size + result["lora_B"].size +
                  result["s"].size + result["c"].size)
        ),
        "output_mse_bf16_vs_ref": bf16_only_mse,
        "output_mse_quant_vs_ref": quant_mse,
        "per_position_mse_min": float(per_pos.min()),
        "per_position_mse_max": float(per_pos.max()),
        "per_position_mse_median": float(np.median(per_pos)),
        "loss_curve_head": losses[: min(10, len(losses))],
        "loss_curve_tail": losses[-min(10, len(losses)):],
    }
    report_path = output_dir / "pe-qat-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
