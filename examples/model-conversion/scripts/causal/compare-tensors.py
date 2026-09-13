#!/usr/bin/env python3
"""Compare intermediate tensors saved by run-org-model.py (--dump-tensors)
and llama-debug (--save-tensors) to locate the first point of divergence
in the Gemma4 (or any other) forward pass.

Usage:
  python compare-tensors.py <pytorch-dir> <llamacpp-dir> [tensor ...]

If no tensor names are given, all .bin files found in pytorch-dir are compared.
"""

import sys
import os
import numpy as np
from pathlib import Path


def nmse_db(ref, test):
    mse  = float(np.mean((ref - test) ** 2))
    var  = float(np.var(ref))
    nmse = mse / var if var > 1e-30 else 0.0
    return nmse, 10 * np.log10(nmse + 1e-300)


def load_tensor(directory, name):
    path = Path(directory) / f"{name}.bin"
    if not path.exists():
        return None, None
    arr = np.fromfile(path, dtype=np.float32)
    shape_path = Path(directory) / f"{name}.shape"
    shape = None
    if shape_path.exists():
        shape = tuple(int(x) for x in shape_path.read_text().split())
    return arr, shape


def compare(ref_dir, llm_dir, tensor_names):
    ref_dir = Path(ref_dir)
    llm_dir = Path(llm_dir)

    if not tensor_names:
        tensor_names = sorted(p.stem for p in ref_dir.glob("*.bin"))

    col = 28
    print(f"\n{'Tensor':<{col}}  {'N-elem':>8}  {'NMSE':>12}  {'dB':>8}  {'MaxErr':>10}  {'MeanErr':>10}")
    print("-" * (col + 60))

    any_missing = False
    for name in tensor_names:
        ref, ref_shape = load_tensor(ref_dir, name)
        llm, llm_shape = load_tensor(llm_dir, name)

        if ref is None:
            print(f"{name:<{col}}  MISSING in pytorch dir")
            any_missing = True
            continue
        if llm is None:
            print(f"{name:<{col}}  MISSING in llamacpp dir")
            any_missing = True
            continue

        # Shapes may differ (e.g. PyTorch saves [seq, hidden] as [seq, hidden]
        # while llama.cpp saves [hidden, seq] flattened the same way), but the
        # element count must match.
        if ref.size != llm.size:
            print(f"{name:<{col}}  SIZE MISMATCH  ref={ref.size} (shape {ref_shape})  llm={llm.size} (shape {llm_shape})")
            continue

        nmse, db = nmse_db(ref, llm)
        max_err  = float(np.max(np.abs(ref - llm)))
        mean_err = float(np.mean(np.abs(ref - llm)))

        flag = ""
        if nmse > 0.1:
            flag = "  ❌ DIVERGED"
        elif nmse > 1e-2:
            flag = "  ⚠  high"
        elif nmse > 1e-4:
            flag = "  ~ ok"
        else:
            flag = "  ✓"

        print(f"{name:<{col}}  {ref.size:>8d}  {nmse:>12.3e}  {db:>8.2f}  {max_err:>10.4f}  {mean_err:>10.4f}{flag}")

    print()
    if any_missing:
        print("Some tensors were missing — check that both runs used the same prompt and filters.")


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    ref_dir  = args[0]
    llm_dir  = args[1]
    tensors  = args[2:]

    compare(ref_dir, llm_dir, tensors)


if __name__ == "__main__":
    main()
