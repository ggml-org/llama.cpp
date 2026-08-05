#!/usr/bin/env python3
"""Build the rmsnorm-1x4096 ANE fixture and emit the state-layout manifest.

The fixture is a single-function .mlmodelc with one functionName
"rmsnorm" of shape [1, 4096] (decode, M=1). It validates the
GGML_OP_RMS_NORM dispatch case in ggml_ane_program_dispatch_op
end-to-end: ggml-ane loads the manifest, pins the input/output
slots, and the dispatch path runs the function on the ANE.

The bundle is a MIL program (Core ML ML Program spec), not a
NeuralNetwork spec, so functionName is settable on the manifest's
bound function. The model's internal compute precision is fp16;
the input/output is fp32 (Core ML handles the precision
conversion at the IOSurface boundary).

This is the W2 body-op spike for the Phase 1 RMSNorm work
(docs/tessera-ane-ios-demo-design.md). Subsequent ops
(SoftMax, RoPE, GLU, GetRows) reuse the same construction pattern
with their own per-op shapes.

Usage:
    python3 build-rmsnorm-fixture.py
        # writes to tools/ane-mtp/fixtures/rmsnorm-1x4096/
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
from google.protobuf import json_format

# Local imports. state_layout.py is the source of truth for the
# ane_state_layout.v1 manifest schema; emit the manifest through
# the same helper the multifunction bundle uses.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from state_layout import StateLayout, ROLE_RMS_NORM, DTYPE_F32, manifest_path_for


N = 4096
EPS = np.float32(1e-6)


def _build_mlpackage(out_dir: Path) -> Path:
    """Build the .mlpackage with explicit input/output names so
    CoreML's optimizer doesn't rename "y" to "mul_1" (the default
    output of the last op).

    The input is [N, 1] (a column vector) to match ggml's
    per-row RMSNorm shape: for decode, the input tensor is
    [K, 1] where K is the hidden dim. The dispatch case
    (ggml_ane_program_dispatch_op's GGML_OP_RMS_NORM arm) checks
    that op->ne[1] == 1, so the bundle's baked shape must be
    [N, 1] (the row dim first) to match what the ggml graph
    feeds in.

    Building via the MIL Builder directly (rather than via the
    Torch frontend) keeps the IO names stable across coremltools
    versions and pins the reduction to a true row-mean (no
    silent fusion with a future weight).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "rmsnorm-1x4096.mlpackage"

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(N, 1))],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        x2      = mb.mul(x=x, y=x, name="x2")
        mean    = mb.reduce_mean(x=x2, axes=[-2], keep_dims=True, name="mean")
        add     = mb.add(x=mean, y=EPS, name="add")
        rsqrt   = mb.rsqrt(x=add, name="rsqrt")
        y       = mb.mul(x=x, y=rsqrt, name="y")
        return y

    model = ct.convert(
        prog,
        inputs=[ct.TensorType(name="x", shape=(N, 1))],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    # The .mlmodelc compiler requires the ML Program to expose a
    # function literally named "main" (this is a coremlcompiler
    # validator rule, not a Core ML runtime rule). The manifest
    # sidecar therefore references "main" as the functionName,
    # and the dispatch path looks up the bound function by the
    # active_function_id. The role ("rms_norm") is what tells the
    # dispatch which op this bundle serves; the name "main" is
    # just the Core ML entry point.
    if mlpackage.exists():
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _compile_to_mlmodelc(mlpackage: Path, out_dir: Path) -> Path:
    """Compile the .mlpackage to a .mlmodelc via xcrun coremlcompiler.

    The compiler produces a directory tree with coremldata.bin +
    metadata.json + model.mil; the Core ML runtime reads it
    through the Objective-C API. The Python coremltools library
    additionally expects a Manifest.json in the .mlmodelc; we
    patch one in below (the compiler's output omits it on
    current toolchains).
    """
    mlmodelc = out_dir / "rmsnorm-1x4096.mlmodelc"
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
    """The xcrun coremlcompiler output for an ML Program omits the
    Manifest.json; without it the Python coremltools library cannot
    re-load the bundle, but the Objective-C API works fine because
    the entry model is implicit in the .mlmodelc. We add a
    Manifest.json pointing at model.mil + coremldata.bin (the
    actual files the compiler produced) so both the Python and
    Objective-C paths can read the bundle.
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
    """Emit the ane_state_layout.v1 manifest sidecar.

    The runtime (ggml/src/ggml-ane/ggml-ane.mm) reads the sidecar
    to allocate the state IOSurface and pin the function's
    input/output slots. The fixture is stateless: two slots (x, y)
    of shape [1, 4096] fp32, no STATE slots, no cross-function
    dependencies. The 64 KB state_size_bytes is the ANE minimum
    allocation (Orion #4); the actual slot data is 16 KB each,
    padded to 16 KB page boundaries.

    The function's name in the manifest matches the .mlmodelc's
    function entry point ("main" is the coremlcompiler's required
    function name for ML Program bundles). The role ("rms_norm")
    is what the dispatch path keys on; the name is what the
    MLModelConfiguration.functionName must equal.
    """
    layout = StateLayout.for_body_op(
        "rmsnorm-1x4096",
        ROLE_RMS_NORM,
        "main",
        inputs=[("x", DTYPE_F32, [N, 1])],
        outputs=[("y", DTYPE_F32, [N, 1])],
    )
    manifest_path = manifest_path_for(mlmodelc.parent, "rmsnorm-1x4096")
    layout.write_json(manifest_path)
    return manifest_path


def main() -> None:
    out_dir = THIS_DIR / "fixtures" / "rmsnorm-1x4096"
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
