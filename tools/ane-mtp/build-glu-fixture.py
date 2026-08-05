#!/usr/bin/env python3
"""Build the geglu-1x11008 ANE fixture and emit the state-layout manifest.

The gemma 4 FFN path is geglu: gate = GELU(x @ W_gate), then
out = (gate * up) @ W_down. The split-form GLU is what the
Phase 1 GGML_OP_GLU dispatch path is wired for: two input
tensors (gate and up), one output (the element-wise product
of GELU(gate) and up). The activation is baked into the
bundle; the manifest's role identifies the op kind so a
follow-on bundle can swap in the swiglu variant (sigma
instead of GELU) without changing the dispatch path.

Same construction as the other body-op fixtures: a single-
function .mlmodelc (functionName "main" per the coremlcompiler
validator rule), shape [N, 1] fp32 in/out, the
ane_state_layout.v1 manifest sidecar.

Usage:
    python3 build-glu-fixture.py
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
from state_layout import StateLayout, ROLE_GLU, DTYPE_F32, manifest_path_for


# gemma 4 12B's intermediate FFN dim is ~11008 (matches
# llama.cpp's standard FFN_GATE shape; the design doc's
# representative shape in Phase 1's table).
N_FF = 11008


def _build_mlpackage(out_dir: Path) -> Path:
    """Build the .mlpackage for split-form geglu: y = GELU(gate) * up.

    MIL has no high-level geglu op, so the activation +
    element-wise product is constructed from primitives.
    GELU is the standard sigmoid-based erf formulation
    (matching ggml-cpu's GELU_GATE path):
        gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    The bundle bakes the activation kind (geglu) at export
    time. swiglu (sigma(x) = x * sigmoid(x) instead of GELU)
    is a separate functionName in a follow-on bundle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "geglu-1x11008.mlpackage"

    INV_SQRT2 = np.float32(1.0 / np.sqrt(2.0))

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(N_FF, 1)),
            mb.TensorSpec(shape=(N_FF, 1)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(gate, up):
        # Standard sigmoid-based GELU:
        #   gelu(x) = 0.5 * x * (1 + erf(x * inv_sqrt2))
        inv     = mb.const(val=INV_SQRT2, name="inv_sqrt2")
        scaled  = mb.mul(x=gate, y=inv, name="scaled")
        erf_out = mb.erf(x=scaled, name="erf")
        one     = mb.const(val=np.float32(1.0), name="one")
        plus    = mb.add(x=one, y=erf_out, name="plus")
        half    = mb.const(val=np.float32(0.5), name="half")
        gelu    = mb.mul(
            x=half,
            y=mb.mul(x=gate, y=plus, name="gate_times_plus"),
            name="gelu",
        )
        # Element-wise product with up.
        y       = mb.mul(x=gelu, y=up, name="y")
        return y

    model = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="gate", shape=(N_FF, 1)),
            ct.TensorType(name="up", shape=(N_FF, 1)),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _compile_to_mlmodelc(mlpackage: Path, out_dir: Path) -> Path:
    mlmodelc = out_dir / "geglu-1x11008.mlmodelc"
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
        "geglu-1x11008",
        ROLE_GLU,
        "main",
        inputs=[("gate", DTYPE_F32, [N_FF, 1]),
                ("up", DTYPE_F32, [N_FF, 1])],
        outputs=[("y", DTYPE_F32, [N_FF, 1])],
    )
    manifest_path = manifest_path_for(mlmodelc.parent, "geglu-1x11008")
    layout.write_json(manifest_path)
    return manifest_path


def main() -> None:
    out_dir = THIS_DIR / "fixtures" / "geglu-1x11008"
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
