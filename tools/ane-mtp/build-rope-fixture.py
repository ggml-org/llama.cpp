#!/usr/bin/env python3
"""Build the rope-1x4096 ANE fixture and emit the state-layout manifest.

The RoPE bundle is the most complex of the five Phase 1 body-op
spikes: the per-head rotation has to be built from elementwise
MIL ops (CoreML has no high-level RoPE op). We bake the
rotation params (n_dims, freq_base, freq_scale, ext_factor,
attn_factor, beta_fast, beta_slow) into the bundle and
compute the cos/sin tables at runtime from the position input
(NORMAL mode, no YaRN, no freq_factors). The gemma 4 variant
covers the same ops; mrope sections and freq_factors are
Phase 1's "supported" subset and land in a follow-on bundle.

Usage:
    python3 build-rope-fixture.py
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
from state_layout import StateLayout, ROLE_ROPE, DTYPE_F32, DTYPE_I32, manifest_path_for


# Bundle shape: [n_dims, 1] = [4096, 1] (one token, the "head"
# view is folded into the 4096-dim row to keep the bundle 2D
# for the Phase 1 spike; a real gemma 4 use reshapes [head_dim,
# n_head, n_tokens] -> [n_dims, n_tokens] before RoPE).
N_DIMS = 4096
HALF   = N_DIMS // 2
FREQ_BASE    = 10000.0
FREQ_SCALE   = 1.0
EXT_FACTOR   = 0.0
ATTN_FACTOR  = 1.0
BETA_FAST    = 0.0
BETA_SLOW    = 0.0


def _build_mlpackage(out_dir: Path) -> Path:
    """Build the .mlpackage with the NORMAL-mode RoPE rotation
    baked into the bundle.

    MIL has no high-level RoPE op, so the rotation is
    constructed from elementwise ops. The per-pair math is:

        theta_i = pos * freq_base^(-2i / n_dims)
        new[i]      = x[i] * cos(theta_i) - x[i + n_dims/2] * sin(theta_i)
        new[i + n_dims/2] = x[i] * sin(theta_i) + x[i + n_dims/2] * cos(theta_i)

    For NORMAL mode the pairs are (i, i + n_dims/2), not
    interleaved. NEOX (which interleaves) and MROPE/IMROPE
    (which use sections) are follow-on bundles; this spike
    covers NORMAL only, matching the dispatch policy that
    routes other variants to the CPU/Accelerate path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = out_dir / "rope-1x4096.mlpackage"

    # Precompute the inv_freq[i] = freq_base^(-2i / n_dims) for
    # i in [0, half). The runtime computes theta = pos * inv_freq
    # then cos/sin per i. As constants of shape [half, 1] the
    # bundle never has to re-derive them.
    inv_freq = np.power(
        FREQ_BASE,
        (-2.0 / float(N_DIMS)) * np.arange(HALF, dtype=np.float32),
    ).reshape(HALF, 1).astype(np.float32)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(N_DIMS, 1)),
            mb.TensorSpec(shape=(1, 1)),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(x, pos):
        # pos is fp32 [1, 1] (the host casts i32 -> fp32 before
        # dispatch; the bundle is internally fp16 anyway).
        pos_f   = pos
        # inv_freq is a baked constant of shape [half, 1].
        inv     = mb.const(val=inv_freq, name="inv")
        # theta = pos * inv_freq; shape [half, 1].
        theta   = mb.mul(x=pos_f, y=inv, name="theta")
        # cos / sin of the angles.
        cos     = mb.cos(x=theta, name="cos")
        sin     = mb.sin(x=theta, name="sin")
        # NORMAL-mode pair split: first = x[0:half], second = x[half:].
        first   = mb.slice_by_index(
            x=x, begin=[0, 0], end=[HALF, 1], name="first")
        second  = mb.slice_by_index(
            x=x, begin=[HALF, 0], end=[N_DIMS, 1], name="second")
        # new_first = first * cos - second * sin
        new_f   = mb.sub(
            x=mb.mul(x=first, y=cos, name="fcos"),
            y=mb.mul(x=second, y=sin, name="ssin"),
            name="new_first",
        )
        # new_second = first * sin + second * cos
        new_s   = mb.add(
            x=mb.mul(x=first, y=sin, name="fsin"),
            y=mb.mul(x=second, y=cos, name="scos"),
            name="new_second",
        )
        # Concat along the row dim (axis=0): [half, 1] + [half, 1] -> [n_dims, 1].
        y       = mb.concat(values=[new_f, new_s], axis=0, name="y")
        return y

    model = ct.convert(
        prog,
        inputs=[
            ct.TensorType(name="x", shape=(N_DIMS, 1)),
            ct.TensorType(name="pos", shape=(1, 1)),
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
    mlmodelc = out_dir / "rope-1x4096.mlmodelc"
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
    # The bundle is stateless: x and pos are INPUT-kind slots,
    # y is OUTPUT-kind, no STATE-kind slots. The rotation params
    # (n_dims, freq_base, etc.) are baked constants inside the
    # bundle, not in the manifest. A future bundle that exposes
    # them as runtime inputs (for the gemma 4 mrope / freq_factors
    # variants) would add them as INPUT-kind slots here.
    # pos is fp32 in the bundle: the host casts i32 -> fp32
    # before dispatch (the bundle is internally fp16 anyway).
    layout = StateLayout.for_body_op(
        "rope-1x4096",
        ROLE_ROPE,
        "main",
        inputs=[("x", DTYPE_F32, [N_DIMS, 1]),
                ("pos", DTYPE_F32, [1, 1])],
        outputs=[("y", DTYPE_F32, [N_DIMS, 1])],
    )
    manifest_path = manifest_path_for(mlmodelc.parent, "rope-1x4096")
    layout.write_json(manifest_path)
    return manifest_path


def main() -> None:
    out_dir = THIS_DIR / "fixtures" / "rope-1x4096"
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
