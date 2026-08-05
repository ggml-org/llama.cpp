#!/usr/bin/env python3
"""Build the multifunction transformer-body .mlmodelc.

Phase 1 of docs/tessera-ane-ios-demo-design.md lights up five
transformer body ops on the ANE backend:

  - GGML_OP_RMS_NORM  (per-row RMSNorm, decode [K, 1])
  - GGML_OP_SOFT_MAX  (row softmax, decode [K, 1])
  - GGML_OP_ROPE      (NORMAL mode, decode [K, 1])
  - GGML_OP_GLU       (split form, geglu, decode [K, 1])
  - GGML_OP_GET_ROWS  (embedding lookup, batch x vocab x hidden)

The multifunction bundle exports one .mlmodelc with one
functionName per op. State (current layer's weights,
activations) is supplied at runtime via IOSurface per the
architect's W0-W7 pattern; the bundle is stateless.

This is the production artifact for the iPhone ANE demo.
The single-function fixtures (rmsnorm-1x4096, softmax-1x1024,
rope-1x4096, geglu-1x11008, getrows-4x128x64) are
test-only and cover one op each; the multifunction bundle
covers all five in one .mlmodelc.

The build is a single MIL program with five functions, each
constructed from primitives (CoreML has no high-level RMSNorm,
RoPE, GLU, etc.). One .mlmodelc, one .ane_state.v1.json,
five functionName entries.

Usage:
    python3 make-transformer-body-bundle.py
        # writes to tools/ane-mtp/fixtures/transformer-body/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from google.protobuf import json_format

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from state_layout import (
    StateLayout, ROLE_RMS_NORM, ROLE_SOFT_MAX, ROLE_ROPE,
    ROLE_GLU, ROLE_GET_ROWS, DTYPE_F32, manifest_path_for,
)


# Phase 1 representative shapes. These are the same shapes the
# per-op test fixtures use; the multifunction bundle's
# dispatch (in ggml_ane_program_dispatch_op) accepts them
# directly. A follow-on commit can add a second bundle with
# different shapes if the production model needs them.
RMS_NORM_N   = 4096
SOFT_MAX_N   = 1024
ROPE_N       = 4096
GLU_N_FF     = 11008
GET_ROWS_BATCH = 4
GET_ROWS_VOCAB = 128
GET_ROWS_HIDDEN = 64
EPS = np.float32(1e-6)
INV_SQRT2 = np.float32(1.0 / np.sqrt(2.0))
FREQ_BASE = 10000.0


def _build_multifunction_mlpackage(out_dir: Path) -> Path:
    """Build the multifunction bundle as N single-function
    .mlmodelc files in one output directory, plus a single
    multifunction state-layout manifest that names all N
    functions with their expected slots.

    Why N .mlmodelc instead of one: CoreML's
    MLModelConfiguration.functionName is set at load time,
    so a single MLModel can be bound to one function only.
    For a multifunction bundle, the iOS app loads one
    MLModel per function (5 in this case) from the same
    output directory, with each load specifying a different
    functionName. The per-function .mlmodelc that this
    script produces is what the iOS app consumes.

    The multifunction manifest is the production contract:
    one .ane_state.v1.json names all 5 functions and their
    slot layouts. The dispatch path (in ggml-ane.mm) reads
    the manifest to know which function maps to which op.
    For the per-op tests, the per-op single-function
    bundles + manifests are the test fixtures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute the RoPE inv_freq table.
    half = ROPE_N // 2
    inv_freq = np.power(
        FREQ_BASE,
        (-2.0 / float(ROPE_N)) * np.arange(half, dtype=np.float32),
    ).reshape(half, 1).astype(np.float32)

    # Build each op as a separate single-function .mlpackage,
    # then compile each to its own .mlmodelc in the same
    # output directory. The iOS app loads each .mlmodelc
    # as a separate MLModel.
    builders = {
        "rmsnorm": _build_rmsnorm_mlpackage,
        "softmax": _build_softmax_mlpackage,
        "rope":    lambda d: _build_rope_mlpackage(d, inv_freq),
        "glu":     _build_glu_mlpackage,
        "get_rows":_build_getrows_mlpackage,
    }
    import shutil
    import tempfile
    for func_name, builder in builders.items():
        # The per-op builder creates <out_dir>/<func>.mlpackage;
        # we want the per-op .mlpackage directly at
        # <bundle>/<func>.mlpackage, so we build to a tempdir
        # and move the result into place.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            builder(tmp_path)
            fn_mlpackage = out_dir / f"{func_name}.mlpackage"
            if fn_mlpackage.exists():
                shutil.rmtree(fn_mlpackage)
            # builder writes tmp_path/<func>.mlpackage
            shutil.move(str(tmp_path / f"{func_name}.mlpackage"),
                        str(fn_mlpackage))
    return out_dir


def _build_rmsnorm_mlpackage(out_dir: Path) -> Path:
    """Build the rmsnorm op as a standalone single-function
    .mlpackage. The function is named "main" (the
    coremlcompiler validator rule); the dispatch sets
    functionName to the manifest's function name at load.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "rmsnorm.mlpackage"

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(RMS_NORM_N, 1))],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        x2    = mb.mul(x=x, y=x, name="x2")
        mean  = mb.reduce_mean(x=x2, axes=[-2], keep_dims=True, name="mean")
        add   = mb.add(x=mean, y=EPS, name="add")
        rsqrt = mb.rsqrt(x=add, name="rsqrt")
        y     = mb.mul(x=x, y=rsqrt, name="y")
        return y
    model = ct.convert(
        prog,
        inputs=[ct.TensorType(name="x", shape=(RMS_NORM_N, 1))],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        import shutil
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _build_softmax_mlpackage(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "softmax.mlpackage"

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(SOFT_MAX_N, 1))],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        mx    = mb.reduce_max(x=x, axes=[-2], keep_dims=True, name="mx")
        sub   = mb.sub(x=x, y=mx, name="sub")
        ex    = mb.exp(x=sub, name="ex")
        sm    = mb.reduce_sum(x=ex, axes=[-2], keep_dims=True, name="sm")
        y     = mb.real_div(x=ex, y=sm, name="y")
        return y
    model = ct.convert(
        prog,
        inputs=[ct.TensorType(name="x", shape=(SOFT_MAX_N, 1))],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        import shutil
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _build_rope_mlpackage(out_dir: Path, inv_freq) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "rope.mlpackage"

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(ROPE_N, 1)),
            mb.TensorSpec(shape=(1, 1)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(x, pos):
        inv    = mb.const(val=inv_freq, name="inv")
        theta  = mb.mul(x=pos, y=inv, name="theta")
        cos    = mb.cos(x=theta, name="cos")
        sin    = mb.sin(x=theta, name="sin")
        first  = mb.slice_by_index(
            x=x, begin=[0, 0], end=[ROPE_N // 2, 1], name="first")
        second = mb.slice_by_index(
            x=x, begin=[ROPE_N // 2, 0], end=[ROPE_N, 1], name="second")
        new_f  = mb.sub(
            x=mb.mul(x=first, y=cos, name="fcos"),
            y=mb.mul(x=second, y=sin, name="ssin"),
            name="new_first",
        )
        new_s  = mb.add(
            x=mb.mul(x=first, y=sin, name="fsin"),
            y=mb.mul(x=second, y=cos, name="scos"),
            name="new_second",
        )
        y      = mb.concat(values=[new_f, new_s], axis=0, name="y")
        return y
    model = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="x", shape=(ROPE_N, 1)),
            ct.TensorType(name="pos", shape=(1, 1)),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        import shutil
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _build_glu_mlpackage(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "glu.mlpackage"

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(GLU_N_FF, 1)),
            mb.TensorSpec(shape=(GLU_N_FF, 1)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(gate, up):
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
        y       = mb.mul(x=gelu, y=up, name="y")
        return y
    model = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="gate", shape=(GLU_N_FF, 1)),
            ct.TensorType(name="up", shape=(GLU_N_FF, 1)),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        import shutil
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _build_getrows_mlpackage(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "get_rows.mlpackage"

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(GET_ROWS_HIDDEN, GET_ROWS_VOCAB)),
            mb.TensorSpec(shape=(GET_ROWS_BATCH,)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(table, ids):
        ids_i = mb.cast(x=ids, dtype="int32", name="ids_i")
        out   = mb.gather(x=table, indices=ids_i, axis=1, name="y")
        return out
    model = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="table", shape=(GET_ROWS_HIDDEN, GET_ROWS_VOCAB)),
            ct.TensorType(name="ids", shape=(GET_ROWS_BATCH,)),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        import shutil
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _build_rmsnorm_spec():
    """Build a one-function ML Program spec for the rmsnorm
    op (shape [N, 1] fp32 in/out, eps baked). Kept for
    reference; the N-.mlmodelc path uses _build_rmsnorm_mlpackage
    instead.
    """
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(RMS_NORM_N, 1))],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        x2    = mb.mul(x=x, y=x, name="x2")
        mean  = mb.reduce_mean(x=x2, axes=[-2], keep_dims=True, name="mean")
        add   = mb.add(x=mean, y=EPS, name="add")
        rsqrt = mb.rsqrt(x=add, name="rsqrt")
        y     = mb.mul(x=x, y=rsqrt, name="rmsnorm_out")
        return y
    return _convert_single(prog, "rmsnorm",
        [("x", DTYPE_F32, [RMS_NORM_N, 1])],
        [("rmsnorm_out", DTYPE_F32, [RMS_NORM_N, 1])])


def _build_rmsnorm_spec():
    """Build a one-function ML Program spec for the rmsnorm
    op (shape [N, 1] fp32 in/out, eps baked).
    """
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(RMS_NORM_N, 1))],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        x2    = mb.mul(x=x, y=x, name="x2")
        mean  = mb.reduce_mean(x=x2, axes=[-2], keep_dims=True, name="mean")
        add   = mb.add(x=mean, y=EPS, name="add")
        rsqrt = mb.rsqrt(x=add, name="rsqrt")
        y     = mb.mul(x=x, y=rsqrt, name="rmsnorm_out")
        return y
    return _convert_single(prog, "rmsnorm",
        [("x", DTYPE_F32, [RMS_NORM_N, 1])],
        [("rmsnorm_out", DTYPE_F32, [RMS_NORM_N, 1])])


def _build_softmax_spec(n):
    """Build a one-function ML Program spec for the row
    softmax (shape [n, 1] fp32 in/out, scale=1, max_bias=0).
    """
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(n, 1))],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        mx    = mb.reduce_max(x=x, axes=[-2], keep_dims=True, name="mx")
        sub   = mb.sub(x=x, y=mx, name="sub")
        ex    = mb.exp(x=sub, name="ex")
        sm    = mb.reduce_sum(x=ex, axes=[-2], keep_dims=True, name="sm")
        y     = mb.real_div(x=ex, y=sm, name="softmax_out")
        return y
    return _convert_single(prog, "softmax",
        [("x", DTYPE_F32, [n, 1])],
        [("softmax_out", DTYPE_F32, [n, 1])])


def _build_rope_spec(n, inv_freq):
    """Build a one-function ML Program spec for NORMAL-mode
    RoPE (shape [n, 1] fp32 in, fp32 pos [1, 1] in, [n, 1] out).
    """
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(n, 1)),
            mb.TensorSpec(shape=(1, 1)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(x, pos):
        inv    = mb.const(val=inv_freq, name="inv")
        theta  = mb.mul(x=pos, y=inv, name="theta")
        cos    = mb.cos(x=theta, name="cos")
        sin    = mb.sin(x=theta, name="sin")
        first  = mb.slice_by_index(
            x=x, begin=[0, 0], end=[n // 2, 1], name="first")
        second = mb.slice_by_index(
            x=x, begin=[n // 2, 0], end=[n, 1], name="second")
        new_f  = mb.sub(
            x=mb.mul(x=first, y=cos, name="fcos"),
            y=mb.mul(x=second, y=sin, name="ssin"),
            name="new_first",
        )
        new_s  = mb.add(
            x=mb.mul(x=first, y=sin, name="fsin"),
            y=mb.mul(x=second, y=cos, name="scos"),
            name="new_second",
        )
        y      = mb.concat(values=[new_f, new_s], axis=0, name="rope_out")
        return y
    return _convert_single(prog, "rope",
        [("x", DTYPE_F32, [n, 1]),
         ("pos", DTYPE_F32, [1, 1])],
        [("rope_out", DTYPE_F32, [n, 1])])


def _build_glu_spec(n):
    """Build a one-function ML Program spec for split-form
    geglu (gate [n, 1], up [n, 1] -> y [n, 1], gelu baked).
    """
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(n, 1)),
            mb.TensorSpec(shape=(n, 1)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(gate, up):
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
        y       = mb.mul(x=gelu, y=up, name="glu_out")
        return y
    return _convert_single(prog, "glu",
        [("gate", DTYPE_F32, [n, 1]),
         ("up", DTYPE_F32, [n, 1])],
        [("glu_out", DTYPE_F32, [n, 1])])


def _build_getrows_spec(batch, vocab, hidden):
    """Build a one-function ML Program spec for axis=1 gather
    on a [hidden, vocab] table with [batch] ids (ggml's
    column-major view).
    """
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(hidden, vocab)),
            mb.TensorSpec(shape=(batch,)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(table, ids):
        ids_i = mb.cast(x=ids, dtype="int32", name="ids_i")
        out   = mb.gather(x=table, indices=ids_i, axis=1, name="getrows_out")
        return out
    return _convert_single(prog, "get_rows",
        [("table", DTYPE_F32, [hidden, vocab]),
         ("ids", DTYPE_F32, [batch,])],
        [("getrows_out", DTYPE_F32, [hidden, batch])])


def _convert_single(prog, function_name, inputs, outputs):
    """Build a single-function spec from an @mb.program.

    Returns the spec with the function renamed from the
    default "main" to function_name. The .mlmodelc compiler
    requires a "main" function; we rename it post-conversion
    via the JSON round-trip pattern (protobuf map entries
    cannot be reassigned directly).
    """
    input_specs = [
        ct.TensorType(name=name, shape=shape)
        for (name, _, shape) in inputs
    ]
    model = ct.convert(
        prog,
        inputs=input_specs,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    spec = model.get_spec()
    if function_name == "main" or "main" not in spec.mlProgram.functions:
        return spec
    # The protobuf API doesn't let us insert under a new
    # key. We serialize to JSON (with "main" still in the
    # map), rename in the JSON, then parse back. Renaming
    # requires the original entry to be present; otherwise
    # the round-trip drops the function map.
    raw = json_format.MessageToJson(spec)
    raw_obj = json.loads(raw)
    funcs = raw_obj.get("mlProgram", {}).get("functions", {})
    if "main" in funcs and function_name not in funcs:
        funcs[function_name] = funcs.pop("main")
    new_spec = json_format.Parse(json.dumps(raw_obj), type(spec)())
    return new_spec


def _build_main_passthrough(specs):
    """Stitch the per-op specs into one multifunction spec.

    Each per-op spec has its own function map; we union the
    function maps into a single spec. The result is a
    multifunction .mlmodelc with N functionName entries.
    This helper is unused in the current N-.mlmodelc output
    (see _build_multifunction_mlpackage for the rationale)
    but kept as a record of the alternative construction.
    """
    if not specs:
        raise ValueError("no per-op specs to stitch")
    raise NotImplementedError(
        "single multifunction .mlmodelc requires protobuf "
        "map-entry rename support that the CoreML tools "
        "don't expose cleanly; use _build_multifunction_mlpackage "
        "for the N-.mlmodelc output instead")


def _compile_to_mlmodelc(out_dir: Path, mlpackage_dir: Path) -> Path:
    """Compile each per-op .mlpackage in the multifunction
    bundle to a .mlmodelc via xcrun coremlcompiler. The
    output is N .mlmodelc directories, one per function,
    all under out_dir.

    xcrun coremlcompiler places the compiled .mlmodelc at
    <parent>/<source-stem>.mlmodelc, so we pass a temp
    parent for each compile and move the result to its
    final location.
    """
    import shutil
    import tempfile
    for fn_dir in sorted(mlpackage_dir.iterdir()):
        if not fn_dir.name.endswith(".mlpackage"):
            continue
        if not fn_dir.is_dir():
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            subprocess.run(
                ["xcrun", "coremlcompiler", "compile",
                 str(fn_dir), str(tmp_path)],
                check=True,
            )
            # The compiler created tmp_path/<stem>.mlmodelc
            compiled = tmp_path / (fn_dir.stem + ".mlmodelc")
            assert compiled.is_dir(), (
                f"coremlcompiler did not produce {compiled}")
            final = out_dir / (fn_dir.stem + ".mlmodelc")
            if final.exists():
                shutil.rmtree(final)
            shutil.move(str(compiled), str(final))
        _patch_manifest_json(out_dir / (fn_dir.stem + ".mlmodelc"))
    return out_dir  # The "bundle" is the directory of .mlmodelcs.


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


def _emit_state_layout(out_dir: Path) -> Path:
    """Emit the multifunction manifest: one .ane_state.v1.json
    that names all 5 functionName entries (rmsnorm, softmax,
    rope, glu, get_rows), each with its own input/output
    slots. The slot names are prefixed by the function name
    (e.g., "rmsnorm.x", "rmsnorm.y") to disambiguate across
    functions sharing the same IOSurface.

    The multifunction .mlmodelc output is a directory of N
    single-function .mlmodelc files; the manifest is the
    contract that names them and their slots. The iOS app
    reads the manifest, then loads each per-function
    .mlmodelc with MLModelConfiguration.functionName set to
    the function's name.
    """
    layout = StateLayout.for_transformer_body("transformer-body", [
        {"name": "rmsnorm", "role": ROLE_RMS_NORM,
         "inputs": [("x", DTYPE_F32, [RMS_NORM_N, 1])],
         "outputs": [("y", DTYPE_F32, [RMS_NORM_N, 1])]},
        {"name": "softmax", "role": ROLE_SOFT_MAX,
         "inputs": [("x", DTYPE_F32, [SOFT_MAX_N, 1])],
         "outputs": [("y", DTYPE_F32, [SOFT_MAX_N, 1])]},
        {"name": "rope", "role": ROLE_ROPE,
         "inputs": [("x", DTYPE_F32, [ROPE_N, 1]),
                    ("pos", DTYPE_F32, [1, 1])],
         "outputs": [("y", DTYPE_F32, [ROPE_N, 1])]},
        {"name": "glu", "role": ROLE_GLU,
         "inputs": [("gate", DTYPE_F32, [GLU_N_FF, 1]),
                    ("up", DTYPE_F32, [GLU_N_FF, 1])],
         "outputs": [("y", DTYPE_F32, [GLU_N_FF, 1])]},
        {"name": "get_rows", "role": ROLE_GET_ROWS,
         "inputs": [("table", DTYPE_F32, [GET_ROWS_HIDDEN, GET_ROWS_VOCAB]),
                    ("ids", DTYPE_F32, [GET_ROWS_BATCH])],
         "outputs": [("y", DTYPE_F32, [GET_ROWS_HIDDEN, GET_ROWS_BATCH])]},
    ])
    manifest_path = manifest_path_for(out_dir, "transformer-body")
    layout.write_json(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "fixtures" / "transformer-body",
        help="output directory for the .mlmodelc + manifest",
    )
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    mlpackage_dir = _build_multifunction_mlpackage(out_dir)
    print(f"built per-op .mlpackages under {mlpackage_dir}", file=sys.stderr)

    mlmodelc = _compile_to_mlmodelc(out_dir, mlpackage_dir)
    print(f"compiled per-op .mlmodelcs under {mlmodelc}", file=sys.stderr)

    manifest = _emit_state_layout(out_dir)
    print(f"wrote {manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
