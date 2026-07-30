#!/usr/bin/env python3
"""Build and validate Gemma's unweighted value RMSNorm as explicit MIL.

This is the smallest parity gate for the ANE prefill compiler.  The automatic
Torch conversion of the Gemma 4 first-layer slab produced incorrect V heads
even when Core ML was restricted to CPU execution.  Prism Engine uses explicit
MIL protobuf construction for the same reason: the compiler must receive the
intended operation graph, not a converter-dependent approximation.

The program normalizes `value_states` exactly as Gemma does:

    x / sqrt(mean(x * x, axis=-1) + epsilon)

It intentionally has no learned scale.  The companion validation run compares
the package's Core ML prediction to a float32 NumPy reference and writes a
receipt.  A full Gemma slab may only use this construction after this gate
passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def build_program(sequence: int, heads: int, head_dim: int, epsilon: float):
    @mb.program(
        input_specs=[
            mb.TensorSpec(
                shape=(1, sequence, heads, head_dim),
                dtype=types.fp16,
            ),
        ],
        opset_version=ct.target.iOS18,
    )
    def gemma_vnorm(value_states):
        square = mb.mul(x=value_states, y=value_states, name="square")
        energy = mb.reduce_sum(
            x=square, axes=[-1], keep_dims=True, name="energy"
        )
        inv_width = mb.const(
            val=np.array(1.0 / head_dim, dtype=np.float16), name="inv_width"
        )
        mean = mb.mul(x=energy, y=inv_width, name="mean")
        eps = mb.const(val=np.array(epsilon, dtype=np.float16), name="epsilon")
        inv_rms = mb.rsqrt(x=mb.add(x=mean, y=eps, name="variance"), name="inv_rms")
        return mb.mul(x=value_states, y=inv_rms, name="normalized_value_states")

    return gemma_vnorm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    program = build_program(args.sequence, args.heads, args.head_dim, args.epsilon)
    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    package = args.output.with_suffix(".mlpackage")
    model.save(str(package))

    rng = np.random.default_rng(args.seed)
    source = rng.standard_normal(
        (1, args.sequence, args.heads, args.head_dim), dtype=np.float32
    ).astype(np.float16)
    # Keep the canonical float32 result for the report.  The acceptance band
    # below deliberately accounts for Core ML's fp16 reciprocal-sqrt lowering;
    # the purpose of this gate is to catch structural errors (such as the
    # converter-produced all-zero V heads), not to reject ordinary fp16 ULP
    # differences from a different rsqrt implementation.
    expected = source.astype(np.float32)
    expected /= np.sqrt(np.mean(expected * expected, axis=-1, keepdims=True) + args.epsilon)
    prediction = model.predict({"value_states": source})["normalized_value_states"]
    actual = np.asarray(prediction, dtype=np.float32)
    error = actual - expected
    receipt = {
        "format": "tessera-ane-vnorm-probe-v1",
        "implementation": "explicit_mil",
        "sequence": args.sequence,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "epsilon": args.epsilon,
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "maximum_accepted_abs_error": 0.02,
        "maximum_accepted_rmse": 0.002,
    }
    if receipt["max_abs_error"] > receipt["maximum_accepted_abs_error"] or receipt["rmse"] > receipt["maximum_accepted_rmse"]:
        raise SystemExit(json.dumps(receipt, indent=2))

    compiled_root = args.output.parent / f".{args.output.name}.compile"
    if compiled_root.exists():
        shutil.rmtree(compiled_root)
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(compiled_root)],
        check=True,
    )
    compiled = compiled_root / f"{package.stem}.mlmodelc"
    if not compiled.is_dir():
        raise SystemExit(f"compiler did not create {compiled}")
    compiled.rename(args.output)
    shutil.rmtree(compiled_root)
    shutil.rmtree(package)
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
