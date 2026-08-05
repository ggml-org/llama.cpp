#!/usr/bin/env python3
"""Build the getrows-4x128x64 ANE fixture and emit the state-layout manifest.

GetRows is the token-embedding lookup: out[i] = table[ids[i]].
The Phase 1 spike covers the small-vocab case (vocab=128,
hidden=64) where the IOSurface write is bandwidth-light and
ANE's gather is competitive with a host-side memcpy. The
production gemma 4 use case has vocab=~256k; that goes
through the ggml-cpu memcpy path per the dispatch policy
(ANE-side gather on a 256k-row table is bandwidth-bound
and the IOSurface write is the bottleneck).

The bundle uses MIL's gather op (axis=0) which is what CoreML
lowers to ANE's IOSurface-friendly gather kernel. Input
shapes: table [vocab=128, hidden=64] fp32, ids [batch=4]
int32, output [batch=4, hidden=64] fp32.

Usage:
    python3 build-get-rows-fixture.py
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
from state_layout import StateLayout, ROLE_GET_ROWS, DTYPE_F32, DTYPE_I32, manifest_path_for


BATCH = 4
VOCAB = 128
HIDDEN = 64


def _build_mlpackage(out_dir: Path) -> Path:
    """Build the .mlpackage for the gather (embedding lookup).

    The bundle is shaped to match ggml's column-major view:
    the embedding table is [HIDDEN, VOCAB] (ggml's
    ne[0]=hidden, ne[1]=vocab) and the gather runs on axis=1
    (the vocab axis). Output is [HIDDEN, BATCH] which matches
    ggml_get_rows's output shape (ne[0]=hidden, ne[1]=batch).
    The flat data is the same; the bundle just declares the
    shape in ggml's order so the dispatch can pass
    op->src[0]->data as-is without a transpose.

    MIL has a `gather` op that does exactly this. The
    bundle is stateless: only INPUT-kind slots (table,
    ids) and one OUTPUT-kind slot (out). No STATE slots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "getrows-4x128x64.mlpackage"

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(HIDDEN, VOCAB)),
            mb.TensorSpec(shape=(BATCH,)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(table, ids):
        # Cast ids to int32 (the gather op requires integer
        # indices). The host passes the ggml-emitted i32 ids
        # directly; this cast is the bundle's safety net for
        # the path where the dispatch is asked to do an
        # int->fp conversion first.
        ids_i = mb.cast(x=ids, dtype="int32", name="ids_i")
        # axis=1 (the vocab axis): out[hidden, i] = table[hidden, ids[i]]
        # for each i in 0..batch. The output is [hidden, batch].
        # The op's name becomes the output's name; the dispatch
        # looks up "y" so we name it explicitly to match.
        out = mb.gather(x=table, indices=ids_i, axis=1, name="y")
        return out

    model = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="table", shape=(HIDDEN, VOCAB)),
            ct.TensorType(name="ids", shape=(BATCH,), dtype=int),
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
    mlmodelc = out_dir / "getrows-4x128x64.mlmodelc"
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
    # ids is rank-1 in the bundle (CoreML's gather expects a
    # 1D index vector). The table and output are in ggml's
    # column-major view: [hidden, vocab] / [hidden, batch].
    # Note: the bundle's CoreML input schema declares ids as
    # Float32 (the int32 cast happens inside the bundle via
    # mb.cast(ids, int32) before the gather). The dispatch
    # converts the ggml-emitted i32 ids to f32 before writing
    # to the pinned slot, so the manifest slot's dtype is f32
    # to match. The dispatch-side i32 -> f32 conversion is in
    # ggml-ane.mm's GGML_OP_GET_ROWS case.
    layout = StateLayout.for_body_op(
        "getrows-4x128x64",
        ROLE_GET_ROWS,
        "main",
        inputs=[("table", DTYPE_F32, [HIDDEN, VOCAB]),
                ("ids", DTYPE_F32, [BATCH])],
        outputs=[("y", DTYPE_F32, [HIDDEN, BATCH])],
    )
    manifest_path = manifest_path_for(mlmodelc.parent, "getrows-4x128x64")
    layout.write_json(manifest_path)
    return manifest_path


def main() -> None:
    out_dir = THIS_DIR / "fixtures" / "getrows-4x128x64"
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
