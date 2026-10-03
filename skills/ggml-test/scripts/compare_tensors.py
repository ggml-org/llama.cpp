#!/usr/bin/env python3
"""Compare tensors between two GGUF files (e.g. a PyTorch reference dump and a ggml test program's output dump) by name and report numerical differences.

Usage:
    python3 compare_tensors.py ref.gguf out.gguf [--rtol 1e-3] [--atol 1e-4]

Exits 0 if every tensor present in both files matches within tolerance, 1 otherwise. Tensors present in only one file are reported but don't fail the run on their own -- useful when comparing partial dumps.
"""
import argparse
import sys
from pathlib import Path

_GGUF_PY = Path(__file__).resolve().parents[3] / "gguf-py"
if _GGUF_PY.exists() and str(_GGUF_PY) not in sys.path:
    sys.path.insert(0, str(_GGUF_PY))

import numpy as np
from gguf.gguf_reader import GGUFReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    ref = {t.name: t.data for t in GGUFReader(args.ref).tensors}
    out = {t.name: t.data for t in GGUFReader(args.out).tensors}

    ok = True
    for name in sorted(set(ref) | set(out)):
        if name not in ref or name not in out:
            print(f"[MISSING] {name}: in ref={name in ref}, in out={name in out}")
            continue

        a, b = ref[name].astype(np.float64), out[name].astype(np.float64)
        if a.shape != b.shape:
            print(f"[SHAPE MISMATCH] {name}: ref={a.shape} out={b.shape}")
            ok = False
            continue

        diff = np.abs(a - b)
        max_abs = diff.max() if diff.size else 0.0
        mean_abs = diff.mean() if diff.size else 0.0
        rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(a) + 1e-12)
        passed = np.allclose(a, b, rtol=args.rtol, atol=args.atol)
        ok &= passed

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: shape={a.shape} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_l2={rel_l2:.3e}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
