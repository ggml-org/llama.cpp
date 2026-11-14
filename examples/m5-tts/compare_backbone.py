#!/usr/bin/env python3
"""Generate ONNX reference rows for prefill and sequential decode, or compare a GGML run."""

import argparse
from pathlib import Path

import numpy as np


def reference(model, embeddings, prefix, output):
    import onnxruntime as ort

    x = np.fromfile(embeddings, dtype="<f4").reshape(-1, 512)
    if not 1 <= prefix <= len(x) <= 2048:
        raise ValueError("expected 1 <= prefix <= embedding rows <= 2048")
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    past = np.empty((8, 2, 1, 2, 0, 64), dtype=np.float32)
    with output.open("wb") as destination:
        for start in [0, *range(prefix, len(x))]:
            end = prefix if start == 0 else start + 1
            hidden, present = session.run(["last_hidden", "present"], {
                "hidden": x[None, start:end],
                "positions": np.arange(start, end, dtype=np.int64)[None],
                "past": past,
            })
            if hidden.shape != (1, 512) or present.shape != (8, 2, 1, 2, end, 64):
                raise ValueError(f"unexpected output shape after position {end - 1}")
            if not np.isfinite(hidden).all() or not np.isfinite(present).all():
                raise ValueError(f"nonfinite reference after position {end - 1}")
            np.testing.assert_array_equal(present[..., :start, :], past)
            past = present
            hidden.astype("<f4", copy=False).tofile(destination)


def compare(expected, actual, atol, rtol):
    expected = np.fromfile(expected, dtype="<f4").reshape(-1, 512)
    actual = np.fromfile(actual, dtype="<f4").reshape(-1, 512)
    if expected.shape != actual.shape or not expected.size:
        raise ValueError(f"output shape mismatch: {expected.shape}, {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite hidden state")
    print("row       max_abs          rms")
    for row, difference in enumerate(actual.astype(np.float64) - expected):
        print(f"{row:3}  {np.abs(difference).max():12.5g}  {np.sqrt(np.mean(difference**2)):12.5g}")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference")
    for name in ("model", "embeddings", "output"):
        ref.add_argument(name, type=Path)
    ref.add_argument("--prefix", type=int, required=True)
    check = commands.add_parser("compare")
    check.add_argument("expected", type=Path)
    check.add_argument("actual", type=Path)
    check.add_argument("--atol", type=float, required=True)
    check.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    if args.command == "reference":
        reference(args.model, args.embeddings, args.prefix, args.output)
    else:
        compare(args.expected, args.actual, args.atol, args.rtol)
