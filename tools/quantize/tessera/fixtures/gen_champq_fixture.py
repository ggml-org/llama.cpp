#!/usr/bin/env python3
#
# gen_champq_fixture.py
#
# Generates a deterministic fixture for the C++ CHAMP-Q L-BFGS parity test
# (tools/quantize/tessera/test_champq.cpp). Builds a small weight matrix
# and synthetic activation scales, runs Python's lbfgs_permutation driver
# (champq_permute.py + _champq_lbfgs.py) on it, and writes both the inputs
# and the Python-computed outputs to fixtures/champq_fixture.json:
#
#   - inputs: weight (out_dim x in_dim), act_scales (in_dim,)
#   - the L2-rank init permutation (the C++ init must match)
#   - the per-iteration loss trajectory at the projected point
#     (the parity gate; same solver + same objective -> close trajectory)
#   - the final permutation + final smoothness proxy
#   - the identity (baseline) smoothness for the improvement ratio
#
# The C++ test loads this JSON, runs ts_champq_compute on the same inputs,
# and asserts:
#   1. the loss trajectory matches within an atol/rtol that documents the
#      float64-vs-float32 gap (Python closure is float64 over f32 inputs;
#      C++ is float32 throughout).
#   2. the final smoothness proxy (recovered-permutation quality) is within
#      a tolerance of Python's. The permutation itself may differ when the
#      relaxed M has near-ties; the test falls back to quality parity.
#
# Regenerate by running once from the repo root:
#
#   python3 tools/quantize/tessera/fixtures/gen_champq_fixture.py
#
# All RNG is seeded so the fixture is byte-stable.

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent.parent  # repo root
TESSERA_PY = REPO / "tools" / "tessera"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    sys.path.insert(0, str(TESSERA_PY))
    champq = _load_module("champq_permute", TESSERA_PY / "champq_permute.py")
    lbfgs_mod = _load_module("_champq_lbfgs", TESSERA_PY / "_champq_lbfgs.py")

    LBFGS = lbfgs_mod.LBFGS
    smoothness_loss_grad = champq.smoothness_loss_grad
    sinkhorn_project = champq.sinkhorn_project
    project_to_permutation = champq.project_to_permutation
    compute_champq_permutation = champq.compute_champq_permutation
    smoothness_proxy = champq.smoothness_proxy

    # --- Build a deterministic, permutation-sensitive fixture. ---
    # in_dim = 32 keeps the K x K L-BFGS small (1024 params) so a 30-iter run
    # is instant, and large enough that the L2-rank init is non-trivial.
    rng = np.random.default_rng(0xC4A7)
    out_dim = 24
    in_dim = 32

    # Low-rank + Gaussian: the low-rank row/column structure is what the
    # smoothness proxy locks onto, so the permutation has real signal.
    rank = 4
    u = rng.normal(size=(out_dim, rank)).astype(np.float64)
    v = rng.normal(size=(in_dim, rank)).astype(np.float64)
    low_rank = u @ v.T
    noise = 0.1 * rng.normal(size=(out_dim, in_dim)).astype(np.float64)
    weight = (low_rank + noise).astype(np.float32)

    # Smooth-ramp activation scales with small noise, so the proxy is
    # per-channel weighted (matches champq_lbfgs_ab.synthetic_act_scales).
    ramp = np.linspace(0.5, 2.0, in_dim, dtype=np.float64)
    act_noise = 0.1 * rng.normal(size=in_dim).astype(np.float64)
    act_scales = (ramp + act_noise).astype(np.float32)

    # --- L-BFGS driver, transcribed from champq_permute.lbfgs_permutation ---
    # so we can record the per-iteration loss at the projected point. The
    # math is identical (same closure, same LBFGS.step, same Sinkhorn).
    n_iters = 30
    sinkhorn_iters = 20
    history = 8
    binariness = 1.0e-3

    K = in_dim
    rank_perm = compute_champq_permutation(weight, act_scales)
    M = np.zeros((K, K), dtype=np.float64)
    M[np.arange(K), rank_perm] = 1.0

    opt = LBFGS(n_params=K * K, history=history, c1=1e-4, max_ls=12)
    M_flat = M.reshape(-1)

    def closure(m_flat):
        m = m_flat.reshape(K, K)
        loss, grad = smoothness_loss_grad(m, weight, act_scales,
                                          binariness=binariness)
        return loss, grad.reshape(-1)

    loss_trajectory = []
    loss, grad = closure(M_flat)
    loss_trajectory.append(float(loss))
    for _ in range(int(n_iters)):
        M_new_flat, _, _, done = opt.step(M_flat, loss, grad, closure)
        M_new = sinkhorn_project(M_new_flat.reshape(K, K),
                                 n_iters=sinkhorn_iters)
        M_new_flat = M_new.reshape(-1)
        loss, grad = closure(M_new_flat)
        loss_trajectory.append(float(loss))
        M_flat = M_new_flat
        # NOTE: champq_permute.lbfgs_permutation does NOT break on `done`
        # (line-search failure); it only stops on the gradient-norm tol. We
        # match that here and in the C++ driver: keep iterating to n_iters.

    perm = project_to_permutation(M_flat.reshape(K, K), mode="hungarian")
    final_smooth = float(smoothness_proxy(weight, perm, act_scales))
    identity_smooth = float(smoothness_proxy(
        weight, np.arange(K, dtype=np.int64), act_scales))

    fixture = {
        "version": 1,
        "comment": "Generated by gen_champq_fixture.py; do not edit by hand.",
        "tolerance": {
            "comment": "C++ is float32 throughout; Python closure is float64 "
                       "over f32 inputs. Observed parity is ~1e-5 rel on this "
                       "fixture; the gate leaves headroom for platform/FFP "
                       "variation while still catching objective or solver "
                       "regressions.",
            "trajectory_rtol": 2e-3,
            "trajectory_atol": 5e-2,
            "smoothness_atol": 5e-3,
        },
        "params": {
            "out_dim": int(out_dim),
            "in_dim": int(in_dim),
            "n_iters": int(n_iters),
            "sinkhorn_iters": int(sinkhorn_iters),
            "history": int(history),
            "binariness": float(binariness),
            "projection": "hungarian",
            "init": "l2rank",
        },
        "inputs": {
            "weight": weight.astype(np.float32).flatten().tolist(),
            "act_scales": act_scales.astype(np.float32).flatten().tolist(),
        },
        "expected": {
            "l2rank_init_perm": rank_perm.astype(np.int64).tolist(),
            "loss_trajectory": loss_trajectory,
            "final_perm": perm.astype(np.int64).tolist(),
            "final_smoothness": final_smooth,
            "identity_smoothness": identity_smooth,
        },
    }

    out_path = HERE / "champq_fixture.json"
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=True)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"  shape: ({out_dim}, {in_dim}), iters={n_iters}")
    print(f"  loss[0]={loss_trajectory[0]:.6e} "
          f"loss[-1]={loss_trajectory[-1]:.6e} "
          f"({len(loss_trajectory)} pts)")
    print(f"  final_smoothness={final_smooth:.6e} "
          f"identity={identity_smooth:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
