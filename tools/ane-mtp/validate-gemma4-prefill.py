#!/usr/bin/env python3
"""Check an exported Gemma 4 ANE layer slab against its source-weight PyTorch reference."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch


def load_exporter() -> object:
    path = Path(__file__).with_name("export-gemma4-prefill.py")
    spec = importlib.util.spec_from_file_location("gemma4_prefill_export", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Gemma 4 ANE exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument(
        "--bridge-output",
        type=Path,
        help=(
            "raw F32 hidden/key/value stream emitted by test-ane-prefill-slab; "
            "validates the actual IOSurface/Core ML bridge instead of the Python runtime"
        ),
    )
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--atol", type=float, default=0.20)
    parser.add_argument("--rmse", type=float, default=0.035)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    config = json.loads((args.source / "config.json").read_text())["text_config"]
    exporter = load_exporter()
    module = exporter.Gemma4InitialSlab(
        args.source,
        config["hidden_size"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
    ).eval()
    # The C++ bridge harness uses a public zero-token prompt.  Keeping the
    # vectors identical means this validates the loaded embedded artifact,
    # IOSurface input path, and output decoding rather than merely a second
    # Core ML Tools invocation.
    tokens = torch.zeros((1, args.sequence), dtype=torch.int32)
    positions = torch.arange(args.sequence, dtype=torch.int32).reshape(1, -1)
    with torch.no_grad():
        expected = module(tokens, positions)
    names = ("hidden_states", "key_states", "value_states")
    if args.bridge_output:
        counts = [int(reference.numel()) for reference in expected]
        raw = np.fromfile(args.bridge_output, dtype=np.float32)
        if raw.size != sum(counts):
            raise SystemExit(
                f"bridge output count mismatch {raw.size} != {sum(counts)}"
            )
        actual = {}
        offset = 0
        for name, reference, count in zip(names, expected, counts):
            actual[name] = raw[offset:offset + count].reshape(tuple(reference.shape))
            offset += count
    else:
        runtime = ct.models.MLModel(str(args.compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
        actual = runtime.predict({
            "token_ids": tokens.numpy(),
            "positions": positions.numpy(),
        })
    results: dict[str, dict[str, float]] = {}
    for name, reference in zip(names, expected):
        lhs = reference.float().numpy()
        rhs = np.asarray(actual[name], dtype=np.float32)
        if lhs.shape != rhs.shape:
            raise SystemExit(f"{name}: shape mismatch {lhs.shape} != {rhs.shape}")
        delta = np.abs(lhs - rhs)
        max_error = float(delta.max())
        rms_error = float(np.sqrt(np.mean((lhs - rhs) ** 2)))
        results[name] = {"max_abs_error": max_error, "rmse": rms_error}
        print(f"{name}: max={max_error:.6f} rmse={rms_error:.6f}")
        if max_error > args.atol or rms_error > args.rmse:
            raise SystemExit(f"{name}: parity tolerance exceeded")
    if args.receipt:
        args.receipt.write_text(json.dumps({
            "format": "tessera-ane-prefill-parity-v1",
            "bridge_validated": bool(args.bridge_output),
            "sequence": args.sequence,
            "max_abs_tolerance": args.atol,
            "rmse_tolerance": args.rmse,
            "outputs": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
