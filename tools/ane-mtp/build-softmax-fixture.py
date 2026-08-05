#!/usr/bin/env python3
"""Build the softmax-1x1024 ANE fixture and emit the state-layout manifest.

Same construction as build-rmsnorm-fixture.py; the bundle exports
a single functionName "main" of shape [N, 1] (decode, M=1)
implementing row-wise softmax. Used by tests/test-ane-softmax.cpp
to validate the GGML_OP_SOFT_MAX dispatch path end-to-end.

The MIL compute is the standard numerically-stable softmax:
exp(x - max(x)) / sum(exp(x - max(x))) reduced over the row.

Usage:
    python3 build-softmax-fixture.py
"""

from __future__ import annotations

import subprocess
import sys
import shutil
import os
import json
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from state_layout import StateLayout, ROLE_SOFT_MAX, DTYPE_F32, manifest_path_for


N = 1024


def _build_mlpackage(out_dir: Path) -> Path:
    """Build the .mlpackage with explicit IO names so CoreML's
    optimizer doesn't rename "y". Input is [N, 1] to match ggml's
    per-row softmax shape (decode, M=1).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "softmax-1x1024.mlpackage"

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(N, 1))],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        # Numerically-stable softmax: subtract the row max, exp,
        # then divide by the row sum. Axes are -2 (the row) to
        # keep the dims consistent with the rmsnorm fixture.
        mx    = mb.reduce_max(x=x, axes=[-2], keep_dims=True, name="mx")
        sub   = mb.sub(x=x, y=mx, name="sub")
        ex    = mb.exp(x=sub, name="ex")
        sm    = mb.reduce_sum(x=ex, axes=[-2], keep_dims=True, name="sm")
        y     = mb.real_div(x=ex, y=sm, name="y")
        return y

    model = ct.convert(
        prog,
        inputs=[ct.TensorType(name="x", shape=(N, 1))],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _compile_to_mlmodelc(mlpackage: Path, out_dir: Path) -> Path:
    mlmodelc = out_dir / "softmax-1x1024.mlmodelc"
    if mlmodelc.exists():
        shutil.rmtree(mlmodelc)
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(out_dir)],
        check=True,
    )
    if not mlmodelc.is_dir():
        raise SystemExit(f"coremlcompiler did not produce {mlmodelc}")
    return mlmodelc


def _patch_manifest_json(mlmodelc: Path) -> None:
    """Same Manifest.json patch as build-rmsnorm-fixture.py:
    xcrun coremlcompiler omits the Manifest.json for ML Program
    bundles; the Objective-C API loads without it but we add
    one anyway for consistency with the rest of the suite.
    """
    manifest = mlmodelc / "Manifest.json"
    if manifest.exists():
        return
    manifest.write_text(
        '{\n'
        '    "fileFormatVersion": "1.0.0",\n'
        '    "itemInfoEntries": {\n'
        '        "00000000-0000-0000-0000-000000000001": {\n'
        '            "author": "com.apple.CoreML",\n'
        '            "description": "CoreML Model Specification",\n'
        '            "name": "model.mlmodel",\n'
        '            "path": "model.mil"\n'
        '        },\n'
        '        "00000000-0000-0000-0000-000000000002": {\n'
        '            "author": "com.apple.CoreML",\n'
        '            "description": "CoreML Model Weights",\n'
        '            "name": "weights",\n'
        '            "path": "coremldata.bin"\n'
        '        }\n'
        '    },\n'
        '    "rootModelIdentifier": "00000000-0000-0000-0000-000000000001"\n'
        '}\n'
    )


def _emit_state_layout(mlmodelc: Path) -> Path:
    layout = StateLayout.for_body_op(
        "softmax-1x1024",
        ROLE_SOFT_MAX,
        "main",
        inputs=[("x", DTYPE_F32, [N, 1])],
        outputs=[("y", DTYPE_F32, [N, 1])],
    )
    manifest_path = manifest_path_for(mlmodelc.parent, "softmax-1x1024")
    layout.write_json(manifest_path)
    return manifest_path


def main() -> None:
    out_dir = THIS_DIR / "fixtures" / "softmax-1x1024"
    out_dir.mkdir(parents=True, exist_ok=True)

    mlpackage = _build_mlpackage(out_dir)
    print(f"built {mlpackage}", file=sys.stderr)

    mlmodelc = _compile_to_mlmodelc(mlpackage, out_dir)
    print(f"compiled {mlmodelc}", file=sys.stderr)

    _patch_manifest_json(mlmodelc)
    print(f"patched Manifest.json in {mlmodelc}", file=sys.stderr)

    manifest = _emit_state_layout(mlmodelc)
    print(f"wrote {manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
