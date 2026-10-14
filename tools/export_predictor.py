#!/usr/bin/env python3
"""Export 31M router predictor weights from PyTorch checkpoint to flat FP16 binary.

Layout:
- w1: [48, 2048, 256] fp16 (50,331,648 bytes)
- b1: [48, 256]        fp16 (24,576 bytes)
- w2: [48, 256, 512]   fp16 (12,582,912 bytes, row-major for coalesced warps)
- b2: [48, 512]        fp16 (49,152 bytes)
Total: 62,988,288 bytes (~60.07 MB)
"""

import os
import sys
import numpy as np

try:
    import torch
except ImportError:
    venv_py = "/tmp/uv_venv/bin/python"
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    raise


def main():
    ckpt_path = "traces/router_predictor_31m.pt"
    out_dir = "data"
    out_path = os.path.join(out_dir, "router_predictor_31m.bin")

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    w1 = state["W1"].half().numpy()  # [48, 2048, 256] fp16
    b1 = state["b1"].squeeze(0).half().numpy()  # [48, 256] fp16
    w2 = state["W2"].half().numpy()  # [48, 256, 512] fp16
    b2 = state["b2"].squeeze(0).half().numpy()  # [48, 512] fp16

    print(f"W1: shape={w1.shape}, dtype={w1.dtype}, bytes={w1.nbytes:,}")
    print(f"b1: shape={b1.shape}, dtype={b1.dtype}, bytes={b1.nbytes:,}")
    print(f"W2: shape={w2.shape}, dtype={w2.dtype}, bytes={w2.nbytes:,}")
    print(f"b2: shape={b2.shape}, dtype={b2.dtype}, bytes={b2.nbytes:,}")

    total_bytes = w1.nbytes + b1.nbytes + w2.nbytes + b2.nbytes
    print(f"Total weight payload: {total_bytes:,} bytes ({total_bytes / (1024 * 1024):.2f} MB)")

    with open(out_path, "wb") as f:
        f.write(w1.tobytes())
        f.write(b1.tobytes())
        f.write(w2.tobytes())
        f.write(b2.tobytes())

    actual_size = os.path.getsize(out_path)
    print(f"Exported to {out_path} ({actual_size:,} bytes, {actual_size / (1024 * 1024):.2f} MB)")
    assert actual_size == total_bytes, f"Size mismatch: {actual_size} vs {total_bytes}"


if __name__ == "__main__":
    main()
