#!/usr/bin/env python3
"""Extract tensor data from GGUF as raw f32 binary files for C++ testing.

Usage:
    python3 scripts/extract-tensor-data.py MODEL.gguf pattern1 [pattern2 ...]

Output:
    For each matching tensor, writes a .f32bin file with header:
        int64_t n_rows, int64_t row_len
    followed by n_rows * row_len float32 values.
"""
import sys
import os
import numpy as np

# Support running from build/ or repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'gguf-py'))

from gguf import GGUFReader

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} MODEL.gguf pattern1 [pattern2 ...]")
        print(f"  Extracts tensors whose names contain any of the given patterns.")
        sys.exit(1)

    model_path = sys.argv[1]
    patterns = sys.argv[2:]

    print(f"Reading {model_path}...")
    reader = GGUFReader(model_path)

    for tensor in reader.tensors:
        if not any(p in tensor.name for p in patterns):
            continue

        print(f"\nExtracting: {tensor.name}")
        print(f"  Shape: {list(tensor.shape)}, type: {tensor.tensor_type.name}")

        # Convert to f32
        raw = np.array(tensor.data, dtype=np.uint8)

        if tensor.tensor_type.name == 'BF16':
            bf16_vals = raw.view(np.uint16)
            f32_bits = bf16_vals.astype(np.uint32) << 16
            f32_vals = f32_bits.view(np.float32)
        elif tensor.tensor_type.name == 'F16':
            f16_vals = raw.view(np.float16)
            f32_vals = f16_vals.astype(np.float32)
        elif tensor.tensor_type.name == 'F32':
            f32_vals = raw.view(np.float32)
        else:
            print(f"  SKIP: unsupported type {tensor.tensor_type.name}")
            continue

        # Determine layout: GGUF stores shape as [col, row] for 2D
        row_len = int(tensor.shape[0])
        n_rows = tensor.n_elements // row_len

        fname = tensor.name.replace(".", "_") + ".f32bin"
        with open(fname, 'wb') as fp:
            fp.write(np.array([n_rows, row_len], dtype=np.int64).tobytes())
            f32_vals.tofile(fp)

        file_size = os.path.getsize(fname)
        print(f"  Wrote {fname}: {n_rows} rows x {row_len} cols = {tensor.n_elements} elements")
        print(f"  File size: {file_size / (1024*1024):.1f} MB")
        print(f"  Stats: mean={f32_vals.mean():.6f}, std={f32_vals.std():.6f}, "
              f"min={f32_vals.min():.6f}, max={f32_vals.max():.6f}")

if __name__ == "__main__":
    main()
