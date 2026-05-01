#!/usr/bin/env python3
"""Compute imatrix (importance matrix) from captured activation tensors.

The imatrix is the per-dimension sum-of-squares of the activations.
It's what upstream llama.cpp uses to weight quantization optimization.

For each activation file act_blkL_*.f32bin, produces imatrix_blkL_<role>.f32bin
where <role> matches the weight tensor it multiplies with.

Format: flat float32 array of length n_per_row, one importance value per dimension.
"""

import numpy as np
import struct
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_f32_tensor(name):
    path = os.path.join(DATA_DIR, name)
    with open(path, "rb") as f:
        nrow, ncol = struct.unpack("qq", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.float32)
        assert len(data) == nrow * ncol
        return data.reshape(nrow, ncol)


def save_imatrix(name, data):
    path = os.path.join(DATA_DIR, name)
    data.astype(np.float32).tofile(path)
    print(
        f"  Wrote {path}: {len(data)} dims, "
        f"min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}"
    )


# Mapping: activation file → imatrix files for each weight it multiplies with
# Each weight tensor's column dimension matches the activation's column dimension
mappings = [
    {
        "act_file": "act_blk0_ffn_input.f32bin",
        "imatrix_name": "imatrix_blk0_ffn_gate_up.f32bin",
        "description": "ffn_gate and ffn_up (both use ffn_input activation)",
    },
    {
        "act_file": "act_blk0_ffn_down_input.f32bin",
        "imatrix_name": "imatrix_blk0_ffn_down.f32bin",
        "description": "ffn_down (uses SwiGLU activation)",
    },
    {
        "act_file": "act_blk0_attn_input.f32bin",
        "imatrix_name": "imatrix_blk0_attn_qkv.f32bin",
        "description": "attn_q, attn_k, attn_v (all use attn_input activation)",
    },
    {
        "act_file": "act_blk0_attn_output_input.f32bin",
        "imatrix_name": "imatrix_blk0_attn_output.f32bin",
        "description": "attn_output (uses kqv_out activation)",
    },
]

print("Computing imatrix from captured activations")
print("=" * 60)

for m in mappings:
    try:
        A = load_f32_tensor(m["act_file"])
        print(f"\n{m['description']}:")
        print(f"  Activation: {A.shape[0]} tokens × {A.shape[1]} dims")

        # imatrix = sum over tokens of activation^2
        # This is the standard definition used by llama.cpp
        imatrix = np.sum(A**2, axis=0)

        # Also compute per-dim RMS for reference
        rms = np.sqrt(np.mean(A**2, axis=0))

        print(
            f"  Imatrix stats: min={imatrix.min():.6f}, max={imatrix.max():.6f}, "
            f"mean={imatrix.mean():.6f}, std={imatrix.std():.6f}"
        )
        print(
            f"  RMS stats:     min={rms.min():.6f}, max={rms.max():.6f}, "
            f"mean={rms.mean():.6f}"
        )

        # Concentration metrics
        total = imatrix.sum()
        sorted_im = np.sort(imatrix)[::-1]
        top1pct = max(1, int(len(imatrix) * 0.01))
        top10pct = max(1, int(len(imatrix) * 0.10))
        print(f"  Power concentration:")
        print(
            f"    Top 1% dims ({top1pct}): {sorted_im[:top1pct].sum() / total * 100:.1f}% of total"
        )
        print(
            f"    Top 10% dims ({top10pct}): {sorted_im[:top10pct].sum() / total * 100:.1f}% of total"
        )

        save_imatrix(m["imatrix_name"], imatrix)
    except Exception as e:
        print(f"  SKIP: {e}")

print("\nDone.")
