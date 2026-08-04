#!/usr/bin/env python3
"""Build a tiny fp16 matmul mlmodelc for the W0 ANE spike.

The W0 spike (per docs/tessera-ane-matmul-research.md W2) needs a standalone
matmul mlpackage to validate the Core ML -> ANE integration path through
ggml-ane.mm before any TILE640 or bundle work. We build a 256x256 fp16
matmul (the smallest size that still exercises the IOSurface zero-copy
path) with a deterministic, non-trivial baked weight.

We construct the Core ML spec via coremltools.proto directly (no torch
dependency) because the local torch/coremltools combo is broken. We emit
a single .mlmodel file and compile via coremlcompiler (the .mlpackage
container path triggers a different code path in coremlcompiler that
rejects our hand-rolled structure on this toolchain version).

Usage:
    python3 make-w0-matmul.py --n 256 --output tools/ane-mtp/fixtures/w0-matmul/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from coremltools.proto import FeatureTypes_pb2 as ft
from coremltools.proto import Model_pb2 as model_pb

# Sidecar the ane_state_layout.v1 manifest next to the .mlmodelc.
# The runtime (common/ane-mtp.mm, ggml/src/ggml-ane/ggml-ane.mm) reads
# this JSON to allocate the state IOSurface and pin the function's
# input/output slots. See tools/ane-mtp/state_layout.py for the
# schema and tools/ane-mtp/test_state_layout.py for the contract.
sys.path.insert(0, str(Path(__file__).parent))
from state_layout import StateLayout, manifest_path_for  # noqa: E402


def build(n: int, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)

    # Deterministic normalized weight (fp16).
    rng = np.random.default_rng(0xA11E)
    weight = rng.standard_normal((n, n), dtype=np.float32)
    weight = weight / (np.linalg.norm(weight, axis=-1, keepdims=True) + 1.0e-6)
    weight_fp16 = weight.astype(np.float16)

    spec = build_matmul_spec(n, weight_fp16)

    # Write the .mlmodel (single file). The .mlpackage container path
    # requires a Manifest.json and other sidecars that we have to keep in
    # sync; the .mlmodel single-file path is more direct.
    mlmodel = output / f"w0-{n}x{n}.mlmodel"
    mlmodel.write_bytes(spec.SerializeToString())
    print(f"wrote {mlmodel}", file=sys.stderr)

    # Compile to .mlmodelc.
    mlmodelc_dir = output
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlmodel), str(mlmodelc_dir)],
        check=True,
    )
    mlmodelc = mlmodelc_dir / f"w0-{n}x{n}.mlmodelc"
    print(f"compiled {mlmodelc}", file=sys.stderr)

    # Also dump the weight as a sidecar .bin file in row-major fp32 order
    # so the W0 spike test can read it back and compute a CPU reference
    # without having to share an RNG with this Python script.
    weight_path = mlmodelc_dir / f"w0-{n}x{n}.weight.bin"
    weight_path.write_bytes(weight_fp16.astype(np.float32).tobytes())
    print(f"wrote {weight_path}", file=sys.stderr)

    # Emit the ane_state_layout.v1 manifest. The W0 spike is a single-
    # function matmul: one INPUT slot "x" and one OUTPUT slot "y", no
    # persistent state, no cross-function dependencies. The runtime
    # reads the manifest to allocate the state IOSurface and pin
    # the slots. See tools/ane-mtp/state_layout.py for the schema.
    bundle_stem = f"w0-{n}x{n}"
    manifest = StateLayout.for_w0_matmul(bundle_stem, n)
    manifest.write_json(manifest_path_for(mlmodelc_dir, bundle_stem))
    manifest_file = manifest_path_for(mlmodelc_dir, bundle_stem)
    print(f"wrote {manifest_file}", file=sys.stderr)
    return mlmodelc


def build_matmul_spec(n: int, weight_fp16: np.ndarray) -> model_pb.Model:
    """Build a Core ML NeuralNetwork spec v4 with a single INNER_PRODUCT layer.

    Apple picks the ANE op for an INNER_PRODUCT with these constraints:
      - input is fp16 (we declare FLOAT32 and let the precision fall through
        to fp16 via the model's compute_precision = FLOAT16)
      - weight is fp16 (we use float16Value)
      - the input/output MultiArrayType is rank-1 (vector) -- rank-2 is
        rejected by the validator ("must have dimension 1 (vector) or 3
        (image-like arrays)")
      - shape is a power of 2 in the output dim (we use n=256)

    The ANE picks `ios18.conv` 1x1 (3x faster than `matmul`) when the model
    targets iOS 18+ / macOS 15+. The runtime picks the legal device via
    MLModelConfiguration; ggml-ane.mm sets compute_units to CPU_AND_NE in
    its program_warm path.
    """
    spec = model_pb.Model()
    spec.specificationVersion = 4  # iOS 18 / macOS 15

    in_x = spec.description.input.add()
    in_x.name = "x"
    in_x.type.multiArrayType.shape.extend([n])
    in_x.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32
    out_y = spec.description.output.add()
    out_y.name = "y"
    out_y.type.multiArrayType.shape.extend([n])
    out_y.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32
    md = spec.description.metadata
    md.shortDescription = "W0 ANE matmul spike"
    md.author = "Tessera"
    md.versionString = "1"

    layer = spec.neuralNetwork.layers.add()
    layer.name = "matmul"
    layer.input.append("x")
    layer.output.append("y")
    layer_params = layer.innerProduct
    layer_params.inputChannels = n
    layer_params.outputChannels = n
    layer_params.hasBias = False
    layer_params.weights.float16Value = weight_fp16.tobytes()
    layer_params.weights.isUpdatable = False

    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256, help="matmul side length")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/ane-mtp/fixtures/w0-matmul"),
        help="output directory for the .mlmodelc",
    )
    args = parser.parse_args()
    build(args.n, args.output)


if __name__ == "__main__":
    main()
