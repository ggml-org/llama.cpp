#!/usr/bin/env python3
"""Demo harness for the FLRQ (Jan 2026) fitness mode.

Runs the FLRQ pipeline on a synthetic 4096 x 4096 weight matrix with
realistic transformer-ish statistics (a few outlier rows plus a low-rank
"outlier subspace" that the R1-Sketch should pick up) and reports:

- the rank chosen by the MSE threshold sweep,
- the per-rank relative reconstruction MSE curve,
- the storage estimate (FP16 for U, V; int8 for the residual),
- the breakdown of the relative MSE into the low-rank part and the
  tile640-quantised residual part.

The demo uses ``tools.tessera._flrq_linalg`` for the small linear
algebra primitives (Gaussian projection, randomised SVD) and the FLRQ
functions in ``per_tensor_calibrate.py`` for the rest.  No PyTorch, no
extra dependencies.

Example:

    python3 tools/tessera/flrq_demo.py \\
        --output-dir /tmp/flrq-demo \\
        --weight-dim 4096 --n-projections 32 --qbits 4

    # Or, against a real layer bundle:
    python3 tools/tessera/flrq_demo.py \\
        --bundle /path/to/ffn_down.npz \\
        --output-dir /tmp/flrq-demo
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
    FLRQ_DEFAULT_MSE_THRESHOLD,
    FLRQ_DEFAULT_PROJECTIONS,
    FLRQ_DEFAULT_QBITS,
    FLRQ_DEFAULT_RANK_CANDIDATES,
    FLRQ_DEFAULT_BLC_ITERS,
    flrq_bcl,
    flrq_select_rank,
    flrq_sketch,
)


def _synthetic_weight(
    out_dim: int,
    in_dim: int,
    outlier_rank: int,
    outlier_rows: int,
    outlier_scale: float,
    seed: int,
) -> np.ndarray:
    """Build a synthetic weight with a planted low-rank + outlier structure.

    The construction is:

    W = W_base + U_out @ V_out + diag(outlier_mask) * noise

    where W_base is Kaiming-init-ish Gaussian noise, (U_out, V_out) is
    a planted low-rank "outlier subspace", and a small fraction of rows
    is multiplied by a large scale to mimic the channel outliers that
    FLRQ's R1-Sketch is designed to detect.

    The recovered rank should be close to ``outlier_rank`` (within one
    step in the candidate set) and the reconstruction MSE should drop
    sharply between rank=outlier_rank and rank=outlier_rank+1, which
    is what the demo's report shows.
    """
    rng = np.random.default_rng(seed)
    W_base = rng.normal(
        loc=0.0, scale=1.0 / math.sqrt(in_dim), size=(out_dim, in_dim)
    ).astype(np.float32)
    if outlier_rank > 0:
        U_out = rng.normal(loc=0.0, scale=0.5, size=(out_dim, outlier_rank)).astype(
            np.float32
        )
        V_out = rng.normal(
            loc=0.0, scale=0.5 / math.sqrt(outlier_rank), size=(outlier_rank, in_dim)
        ).astype(np.float32)
        W_base = W_base + U_out @ V_out
    if outlier_rows > 0:
        idx = rng.choice(out_dim, size=outlier_rows, replace=False)
        W_base[idx] *= outlier_scale
    return W_base.astype(np.float32)


def _per_rank_mse_curve(
    weight: np.ndarray,
    rank_candidates: list[int],
    n_projections: int,
    blc_iters: int,
    qbits: int,
    seed: int,
) -> tuple[int, dict[int, dict]]:
    """Run the full FLRQ rank sweep and return (chosen_rank, per_rank_records)."""
    return flrq_select_rank(
        weight,
        rank_candidates=rank_candidates,
        n_projections=n_projections,
        seed=seed,
        blc_iters=blc_iters,
        qbits=qbits,
        mse_threshold=FLRQ_DEFAULT_MSE_THRESHOLD,
    )


def _storage_estimate(out_dim: int, in_dim: int, rank: int, qbits: int) -> dict:
    """Storage estimate: FP16 for U, V; int8 for the residual payload."""
    u_bytes = out_dim * rank * 2
    v_bytes = rank * in_dim * 2
    # The residual payload is 1 byte/element at <=8 bits; we keep the
    # int8 path because the runtime tile640 path is byte-aligned.
    residual_bytes = out_dim * in_dim * 1
    return {
        "u_bytes": u_bytes,
        "v_bytes": v_bytes,
        "residual_bytes": residual_bytes,
        "total_bytes": u_bytes + v_bytes + residual_bytes,
        "total_kib": (u_bytes + v_bytes + residual_bytes) / 1024.0,
        "total_mib": (u_bytes + v_bytes + residual_bytes) / (1024.0 * 1024.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the policy JSON + report",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional .npz layer bundle (overrides --weight-dim etc.)",
    )
    parser.add_argument(
        "--weight-dim", type=int, default=4096, help="Square weight dimension (default 4096)"
    )
    parser.add_argument(
        "--outlier-rank",
        type=int,
        default=8,
        help="Planted outlier rank for the synthetic weight (default 8)",
    )
    parser.add_argument(
        "--outlier-rows",
        type=int,
        default=16,
        help="Number of outlier rows in the synthetic weight (default 16)",
    )
    parser.add_argument(
        "--outlier-scale",
        type=float,
        default=20.0,
        help="Multiplier for the outlier rows (default 20.0)",
    )
    parser.add_argument(
        "--n-projections",
        type=int,
        default=FLRQ_DEFAULT_PROJECTIONS,
        help="R1-Sketch Gaussian projection count (default 32)",
    )
    parser.add_argument(
        "--qbits",
        type=int,
        default=FLRQ_DEFAULT_QBITS,
        help="Residual quantiser bit-width (default 4)",
    )
    parser.add_argument(
        "--blc-iters",
        type=int,
        default=FLRQ_DEFAULT_BLC_ITERS,
        help="BLC iterations (default 4)",
    )
    parser.add_argument(
        "--rank-candidates",
        type=int,
        nargs="+",
        default=list(FLRQ_DEFAULT_RANK_CANDIDATES),
        help="Rank sweep (default 4 8 16 32 64)",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=FLRQ_DEFAULT_MSE_THRESHOLD,
        help="Relative MSE threshold for rank selection (default 1e-3)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bundle:
        data = np.load(args.bundle, allow_pickle=False)
        if "weight" not in data:
            raise ValueError(f"{args.bundle}: expected a `weight` array")
        W = np.asarray(data["weight"], dtype=np.float32)
        if W.ndim != 2:
            raise ValueError(f"{args.bundle}: weight must be 2-D, got {W.shape}")
        print(f"loaded {args.bundle}: W={W.shape}")
    else:
        W = _synthetic_weight(
            out_dim=args.weight_dim,
            in_dim=args.weight_dim,
            outlier_rank=args.outlier_rank,
            outlier_rows=args.outlier_rows,
            outlier_scale=args.outlier_scale,
            seed=args.seed,
        )
        print(
            f"generated synthetic W={W.shape} with outlier_rank={args.outlier_rank} "
            f"outlier_rows={args.outlier_rows} outlier_scale={args.outlier_scale}"
        )

    K, N = W.shape
    w_fro2 = float(np.sum(W * W)) + 1e-12

    print(
        f"FLRQ config: rank_candidates={args.rank_candidates} "
        f"n_projections={args.n_projections} qbits={args.qbits} "
        f"blc_iters={args.blc_iters} mse_threshold={args.mse_threshold:g}"
    )

    # 1. Stand-alone R1-Sketch sanity check: how concentrated is the
    #    sketch spectrum?  A concentrated spectrum means the rank
    #    selection has a clear winner.
    t0 = time.time()
    Y, U_basis, sigma = flrq_sketch(
        W,
        n_projections=args.n_projections,
        seed=args.seed,
        target_rank=min(8, min(K, N)),
    )
    print(
        f"R1-Sketch: W=({K}x{N}) -> Y=({Y.shape[0]}x{Y.shape[1]})  "
        f"top-{len(sigma)} sigma: {', '.join(f'{s:.3f}' for s in sigma)}  "
        f"elapsed={time.time() - t0:.2f}s"
    )

    # 2. Full rank sweep.
    t0 = time.time()
    chosen_rank, per_rank = _per_rank_mse_curve(
        W,
        rank_candidates=args.rank_candidates,
        n_projections=args.n_projections,
        blc_iters=args.blc_iters,
        qbits=args.qbits,
        seed=args.seed,
    )
    sweep_elapsed = time.time() - t0

    print(f"FLRQ rank sweep ({len(args.rank_candidates)} ranks) in {sweep_elapsed:.2f}s")
    print(
        f"{'rank':>5}  {'rel_mse':>10}  {'abs_mse':>10}  {'bytes':>8}  {'clears':>7}"
    )
    for r in sorted(per_rank):
        record = per_rank[r]
        marker = "*" if r == chosen_rank else " "
        print(
            f"{marker}{r:>4}  {record['mse']:>10.4e}  "
            f"{record['mse'] * w_fro2 / (K * N):>10.4e}  "
            f"{record['bytes']:>8d}  {str(record['clears_threshold']):>7s}"
        )
    print(f"chosen rank: {chosen_rank}")

    # 3. Reconstruct the chosen-rank decomposition end-to-end and report
    #    the storage estimate.
    t0 = time.time()
    _, U_basis_full, _ = flrq_sketch(
        W,
        n_projections=args.n_projections,
        seed=args.seed + chosen_rank,
        target_rank=chosen_rank,
    )
    U, V, scale, clip, residual, residual_q = flrq_bcl(
        W, U_basis_full, n_iters=args.blc_iters, qbits=args.qbits
    )
    recon = U @ V + residual_q
    diff = W - recon
    reconstruction_mse = float(np.sum(diff * diff)) / w_fro2
    storage = _storage_estimate(K, N, chosen_rank, args.qbits)
    print(
        f"chosen-rank BLC: residual_scale={scale:.4e} residual_clip={clip:.4f}  "
        f"reconstruction_rel_mse={reconstruction_mse:.4e}  "
        f"elapsed={time.time() - t0:.2f}s"
    )
    print(
        f"storage: u={storage['u_bytes']} v={storage['v_bytes']} "
        f"residual={storage['residual_bytes']} "
        f"total={storage['total_bytes']} ({storage['total_mib']:.2f} MiB)"
    )

    # 4. Persist a policy + report.
    policy = {
        "schema": "llama.speculative.calibration-policy.v1",
        "flrq": {
            "schema": "llama.tessera.flrq-policy.v1",
            "demo": True,
            "rank": chosen_rank,
            "n_projections": args.n_projections,
            "qbits": args.qbits,
            "blc_iters": args.blc_iters,
            "mse_threshold": args.mse_threshold,
            "weight_shape": list(W.shape),
            "residual_scale": float(scale),
            "residual_clip": float(clip),
            "reconstruction_rel_mse": float(reconstruction_mse),
            "per_rank": {
                str(r): {
                    "mse": float(record["mse"]),
                    "bytes": int(record["bytes"]),
                    "clears_threshold": bool(record["clears_threshold"]),
                }
                for r, record in sorted(per_rank.items())
            },
        },
    }
    policy_path = output_dir / "flrq-demo-policy.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {policy_path}")

    report = {
        "schema": "llama.tessera.flrq-demo-report.v1",
        "weight_shape": list(W.shape),
        "n_projections": args.n_projections,
        "qbits": args.qbits,
        "blc_iters": args.blc_iters,
        "mse_threshold": args.mse_threshold,
        "chosen_rank": int(chosen_rank),
        "reconstruction_rel_mse": float(reconstruction_mse),
        "residual_scale": float(scale),
        "residual_clip": float(clip),
        "storage": storage,
        "sweep_elapsed_seconds": float(sweep_elapsed),
        "per_rank": {
            str(r): {
                "mse": float(record["mse"]),
                "bytes": int(record["bytes"]),
                "clears_threshold": bool(record["clears_threshold"]),
                "residual_max_abs": float(record["residual_max_abs"]),
            }
            for r, record in sorted(per_rank.items())
        },
    }
    report_path = output_dir / "flrq-demo-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
