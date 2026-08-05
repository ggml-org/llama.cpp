#!/usr/bin/env python3
"""Build the TILE640 matmul ANE fixture for Phase 0 of
docs/tessera-ane-ios-demo-design.md.

The fixture is a single-function .mlmodelc that computes
y = w @ x on the ANE: a 2-input fp16 matmul where `w` is the
host-dedequantized TILE640 weight (fp16 [out_dim, in_dim]) and
`x` is the activations (fp16 [in_dim, n_tokens]). The output
`y` is fp32 [out_dim, n_tokens].

The dispatch path (ggml_ane_program_dispatch_op's
GGML_OP_TILE640_MATMUL case) does the TILE640 dequant on the
host via the existing dequantize_row_tessera_t640 (ggml-quants.c),
writes the result into the pinned IOSurface slot for `w`, and
calls the bundle with `w` and `x` as inputs.

Why dequant-on-host and not fused dequant+matmul on ANE?

The TILE640 packing uses a 5-trit base-243 wire format (20
trits per u32, 32 u32 words per page) plus per-page fp16 scales
plus per-lane int8 scales plus sparse outlier addback. The
ANE does not have a fused-dequant matmul op (per
docs/tessera-ane-matmul-research.md Section 2.1); expressing
the 5-trit-base-243 dequant in MIL is possible but complex
(requires a 243-entry LUT plus ~50 elementwise ops per page).
For Phase 0, the host dequant is the architect's allowed
fallback: the dispatch reads the 6 TILE640 sources from
op->src[0..5], calls dequantize_row_tessera_t640 row by row
to fill an IOSurface scratch, and calls the standard ANE
matmul with the scratch + activations. The matmul is the
ANE-native fp16 matmul (mb.matmul in MIL, maps to ios18.conv
1x1 on A15+).

The fixture is built for a specific (out_dim, in_dim,
n_tokens) triple. Production graphs use the per-shape
multifunction pattern: each (out_dim, in_dim) shape gets its
own .mlmodelc, the dispatch selects the matching one. The
Phase 0 spike ships the 256x256 case; the other shape
combos (512x512, 1024x1024, 128x4096, 4096x4096) are
Phase 0.5 follow-ons that build additional fixtures with
this same script.

Usage:
    python3 build-tile640-matmul-fixture.py --out-dim 256 --in-dim 256 --n-tokens 1
        # writes to tools/ane-mtp/fixtures/tile640-matmul-256x256x1/

Environment:
    coremltools 9.x with the C++ proxy libraries (BlobWriter
    etc.) is required for the MB Builder + ct.convert path.
    The Python 3.14 install in this env (Homebrew) ships
    coremltools without the C++ extensions; the script
    refuses to run on Python 3.14 with a clear error.
    Python 3.11 / 3.12 work because coremltools has the
    matching wheels with the C++ libraries. The Phase 1
    body-op scripts (build-rmsnorm-fixture.py, etc.) have
    the same requirement.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as mil_types

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from state_layout import (
    StateLayout, ROLE_MATMUL, DTYPE_F16, DTYPE_F32,
    manifest_path_for,
)


# Guard: coremltools must be importable AND the C++ proxy
# libraries (BlobWriter) must be loadable. Python 3.14 with
# Homebrew's coremltools fails the latter check. The script
# refuses to run with a clear error in that case.
def _check_coremltools_environment() -> None:
    import importlib
    if not importlib.util.find_spec("coremltools"):
        print(
            "ERROR: coremltools is not importable. Install it via:\n"
            "  python3.11 -m pip install coremltools numpy\n"
            "The MB Builder + ct.convert path requires the\n"
            "coremltools C++ proxy libraries (BlobWriter etc.),\n"
            "which are only available in the coremltools wheel\n"
            "for Python 3.11 / 3.12.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        from coremltools.converters.mil.backend.mil import load as mil_load
        if mil_load.BlobWriter is None:
            print(
                "ERROR: coremltools' C++ proxy libraries are not\n"
                "loadable in this Python. The MB Builder + ct.convert\n"
                "path requires BlobWriter (a C++ extension) which is\n"
                "missing. This typically happens with Python 3.14 +\n"
                "Homebrew's coremltools. Switch to Python 3.11 / 3.12.",
                file=sys.stderr,
            )
            sys.exit(2)
    except Exception as e:
        print(
            f"ERROR: could not check coremltools environment: {e}",
            file=sys.stderr,
        )
        sys.exit(2)


def _build_mlpackage(out_dir: Path, out_dim: int, in_dim: int,
                     n_tokens: int) -> Path:
    """Build the TILE640 matmul as a single-function .mlpackage.

    The function takes:
      - w: fp16 [out_dim, in_dim] (the host-dedequantized weight)
      - x: fp16 [in_dim, n_tokens] (the activations)
    and produces:
      - y: fp32 [out_dim, n_tokens]

    The MIL graph does a single matmul (mb.matmul, ANE-NATIVE on
    A15+). The compute precision is FLOAT16 (the ANE's native
    fp16 path); Core ML handles the fp16->fp32 output cast.

    No baked weight: the .mlmodelc is shape-locked to (out_dim,
    in_dim, n_tokens) at compile time, but the weight `w` is a
    runtime input (per the architect's stateless multifunction
    rule). The dispatch fills the `w` slot from the IOSurface
    scratch on every call.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / f"tile640-matmul-{out_dim}x{in_dim}x{n_tokens}.mlpackage"

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(out_dim, in_dim), dtype=mil_types.fp16),
            mb.TensorSpec(shape=(in_dim, n_tokens), dtype=mil_types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(w, x):
        # y[i, t] = sum_j w[i, j] * x[j, t]
        # mb.matmul on fp16 inputs: ANE-NATIVE, maps to ios18.conv 1x1
        # on A15+ (~3x faster than the matmul op). The accumulator
        # is fp16; we cast the output to fp32 so the bundle's
        # declared y dtype matches the dispatch's expectation
        # (the dispatch writes y into op->data which is fp32).
        # Without the cast, the MIL output is fp16 and the
        # dispatch's dtype check fails.
        y_fp16 = mb.matmul(x=w, y=x, name="y_fp16")
        y = mb.cast(x=y_fp16, dtype="fp32", name="y")
        return y

    model = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="w", shape=(out_dim, in_dim), dtype=mil_types.fp16),
            ct.TensorType(name="x", shape=(in_dim, n_tokens), dtype=mil_types.fp16),
        ],
        outputs=[
            ct.TensorType(name="y", dtype=mil_types.fp32),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if mlpackage.exists():
        shutil.rmtree(mlpackage)
    model.save(str(mlpackage))
    return mlpackage


def _compile_to_mlmodelc(mlpackage: Path, out_dir: Path,
                         out_dim: int, in_dim: int, n_tokens: int) -> Path:
    """Compile the .mlpackage to .mlmodelc via xcrun coremlcompiler.

    The compiler places the .mlmodelc at <parent>/<stem>.mlmodelc,
    so we pass a tempdir parent and move the result.
    """
    mlmodelc = out_dir / f"tile640-matmul-{out_dim}x{in_dim}x{n_tokens}.mlmodelc"
    if mlmodelc.exists():
        shutil.rmtree(mlmodelc)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(tmp_path)],
            check=True,
        )
        compiled = tmp_path / f"{mlpackage.stem}.mlmodelc"
        if not compiled.is_dir():
            raise SystemExit(f"coremlcompiler did not produce {compiled}")
        shutil.move(str(compiled), str(mlmodelc))
    _patch_manifest_json(mlmodelc)
    return mlmodelc


def _patch_manifest_json(mlmodelc: Path) -> None:
    """The xcrun coremlcompiler output for an ML Program omits
    the Manifest.json; add one pointing at model.mil +
    coremldata.bin (the actual files the compiler produced).

    Same pattern as the Phase 1 body-op fixtures
    (tools/ane-mtp/build-rmsnorm-fixture.py).
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


def _emit_state_layout(out_dir: Path, out_dim: int, in_dim: int,
                       n_tokens: int) -> Path:
    """Emit the ane_state_layout.v1 manifest sidecar.

    The fixture is stateless: three slots (w, x, y), no STATE
    slots, no cross-function dependencies. The 64 KB
    state_size_bytes is the ANE minimum allocation (Orion #4);
    the actual slot data is 16 KB-aligned (ANE page size).

    The function's name in the manifest is "main" (the
    coremlcompiler's required function name for ML Program
    bundles). The role ("matmul") is what the dispatch path
    keys on.
    """
    bundle_name = f"tile640-matmul-{out_dim}x{in_dim}x{n_tokens}"
    layout = StateLayout.for_body_op(
        bundle_name,
        ROLE_MATMUL,
        "main",
        inputs=[("w", DTYPE_F16, [out_dim, in_dim]),
                ("x", DTYPE_F16, [in_dim, n_tokens])],
        outputs=[("y", DTYPE_F32, [out_dim, n_tokens])],
    )
    manifest_path = manifest_path_for(out_dir, bundle_name)
    layout.write_json(manifest_path)
    return manifest_path


def build(out_dim: int, in_dim: int, n_tokens: int,
          output: Path) -> None:
    """Build the fixture for one (out_dim, in_dim, n_tokens)
    triple. Writes the .mlmodelc, the Manifest.json, and the
    .ane_state.v1.json sidecar in `output`.
    """
    _check_coremltools_environment()
    output.mkdir(parents=True, exist_ok=True)
    print(f"building tile640-matmul fixture ({out_dim}x{in_dim}x{n_tokens}) "
          f"in {output}", file=sys.stderr)

    mlpackage = _build_mlpackage(output, out_dim, in_dim, n_tokens)
    print(f"built {mlpackage}", file=sys.stderr)

    mlmodelc = _compile_to_mlmodelc(mlpackage, output, out_dim, in_dim, n_tokens)
    print(f"compiled {mlmodelc}", file=sys.stderr)

    manifest = _emit_state_layout(output, out_dim, in_dim, n_tokens)
    print(f"wrote {manifest}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dim", type=int, default=256,
                        help="output dimension (rows of weight)")
    parser.add_argument("--in-dim", type=int, default=256,
                        help="input dimension (cols of weight)")
    parser.add_argument("--n-tokens", type=int, default=1,
                        help="number of activation vectors (M)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory for the .mlmodelc + manifest",
    )
    args = parser.parse_args()

    if args.output is None:
        bundle_name = (f"tile640-matmul-{args.out_dim}x{args.in_dim}"
                       f"x{args.n_tokens}")
        args.output = (THIS_DIR / "fixtures" / bundle_name)
    build(args.out_dim, args.in_dim, args.n_tokens, args.output)


if __name__ == "__main__":
    main()
