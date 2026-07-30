#!/usr/bin/env python3
"""Build the v1 synthetic regression bundle for SEPTQ validation.

The synthetic bundle is the v1 (commit 6179dc753) reference: a 4096x4096
weight = low-rank-8 + Gaussian, with 32 calibration samples drawn from the
same low-rank + Gaussian distribution. SEPTQ should win ~92.88% on this
bundle with the original importance score (the v1 result, corrected for
the column-major storage bug found in the prod work).

The bundle is written to ``/tmp/septq_bundles/`` and can be loaded by
``tools/tessera/septq_ab_validate.py --bundle <path>``.

Run: ``python3 tools/tessera/septq_build_bundles.py``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT_DIR = Path("/tmp/septq_bundles")


def make_synthetic(rng_seed: int = 42, out_dim: int = 4096, in_dim: int = 4096,
                   rank: int = 8, n_calib: int = 32) -> None:
    rng = np.random.default_rng(rng_seed)
    u = rng.standard_normal((out_dim, rank)).astype(np.float32) / np.sqrt(rank)
    v = rng.standard_normal((rank, in_dim)).astype(np.float32) / np.sqrt(rank)
    dense = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.1
    W = (u @ v + dense).astype(np.float32)
    x_low = rng.standard_normal((n_calib, rank)).astype(np.float32) @ v
    x_noise = rng.standard_normal((n_calib, in_dim)).astype(np.float32) * 0.5
    X = (x_low + x_noise).astype(np.float32)
    in_sum2 = np.sum(X * X, axis=0).astype(np.float32)
    counts = np.array([float(n_calib)], dtype=np.float32)
    np.savez(
        OUT_DIR / "synthetic_4096x4096.npz",
        weight=W,
        in_sum2=in_sum2,
        counts=counts,
        name=np.array("synthetic-4096x4096"),
        family=np.array("synthetic"),
    )
    print(f"wrote {OUT_DIR / 'synthetic_4096x4096.npz'}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_synthetic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
