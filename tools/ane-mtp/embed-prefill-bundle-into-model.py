#!/usr/bin/env python3
"""Embed a multifunction Tessera ANE prefill bundle into a unified GGUF model.

Wraps `embed-prefill-bundle.py` and copies the source model's tensors and
remaining GGUF metadata so the resulting file is a fully self-contained
Gemma 4 model that also carries a single multifunction prefill bundle.

This supersedes the legacy per-bucket `embed-prefill-into-model.py` once
the loader has been migrated to the multifunction format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter


def copy_source(writer: GGUFWriter, reader: GGUFReader) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        types = field.types
        if types[0] == GGUFValueType.ARRAY:
            writer.add_key_value(key, field.contents(), types[0], types[1])
        else:
            writer.add_key_value(key, field.contents(), types[0])
    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name,
            tuple(reversed(tensor.shape)),
            tensor.data.dtype,
            tensor.n_bytes,
            GGMLQuantizationType(tensor.tensor_type),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("bundle", type=Path,
            help="multifunction mlmodelc directory (e.g. prefill-bundle.mlmodelc)")
    parser.add_argument("output", type=Path)
    parser.add_argument("--causal-right-padding", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file() or args.output.exists():
        raise SystemExit("source must exist and output must not already exist")
    reader = GGUFReader(args.source)
    if reader.fields["general.architecture"].contents() != "gemma4":
        raise SystemExit("only Gemma 4 model GGUFs are accepted")

    bundle_manifest_path = args.bundle.with_name(args.bundle.stem + ".json")
    if not bundle_manifest_path.is_file():
        raise SystemExit(f"missing bundle manifest: {bundle_manifest_path}")
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    if bundle_manifest.get("format") != "tessera-ane-prefill-bundle-v1":
        raise SystemExit("unsupported bundle manifest format")
    if bundle_manifest["architecture"] != "gemma4":
        raise SystemExit("bundle architecture must be gemma4")

    writer = GGUFWriter(args.output, "gemma4")
    copy_source(writer, reader)

    # The prefill manifest fields required by the C++ loader.
    writer.add_string("tessera.ane.prefill.format", "tessera-ane-prefill-v1")
    writer.add_uint32("tessera.ane.prefill.abi_version", 1)
    writer.add_string("tessera.ane.prefill.architecture", bundle_manifest["architecture"])
    writer.add_uint32("tessera.ane.prefill.hidden_size", bundle_manifest["hidden_size"])
    writer.add_uint32("tessera.ane.prefill.layer_first", bundle_manifest["layer_first"])
    writer.add_uint32("tessera.ane.prefill.layer_last", bundle_manifest["layer_last"])
    writer.add_string("tessera.ane.prefill.execution_stage", bundle_manifest["execution_stage"])
    writer.add_string("tessera.ane.prefill.hidden_layout", bundle_manifest["hidden_layout"])
    writer.add_string("tessera.ane.prefill.kv_layout", bundle_manifest["kv_layout"])
    writer.add_string("tessera.ane.prefill.cache_requirement", bundle_manifest["cache_requirement"])
    writer.add_uint32("tessera.ane.prefill.kv_heads", bundle_manifest["kv_heads"])
    writer.add_uint32("tessera.ane.prefill.head_dim", bundle_manifest["head_dim"])
    writer.add_uint32("tessera.ane.prefill.batch_size", bundle_manifest["batch"])
    writer.add_bool("tessera.ane.prefill.causal_right_padding", bool(args.causal_right_padding))
    writer.add_array("tessera.ane.prefill.sequence_buckets", bundle_manifest["sequence_buckets"])

    # Embed the multifunction bundle as a single set of files.
    import hashlib
    digest = hashlib.sha256()
    files = sorted(p for p in args.bundle.rglob("*") if p.is_file())
    for index, path in enumerate(files):
        relative = path.relative_to(args.bundle).as_posix()
        data = np.memmap(path, mode="r", dtype=np.uint8)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(memoryview(data))
        name = f"tessera.ane.prefill.bundle.file.{index:04d}"
        writer.add_tensor(name, data.view(np.int8))
        writer.add_string(f"{name}.path", relative)
    writer.add_uint32("tessera.ane.prefill.bundle.file_count", len(files))
    writer.add_string("tessera.ane.prefill.bundle.bundle_sha256", digest.hexdigest())
    function_names = [entry["name"] for entry in bundle_manifest["functions"]]
    writer.add_array("tessera.ane.prefill.bundle.functions", function_names)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)
    # Tensor payload for the bundle follows the copied source data; emit it
    # via the writer's pending-queue without producing a second tensor-info
    # table.
    for path in files:
        data = np.memmap(path, mode="r", dtype=np.uint8)
        writer.write_tensor_data(data.view(np.int8))
    writer.close()
    print(f"embedded {len(files)} files from {args.bundle} into {args.output}")


if __name__ == "__main__":
    from gguf import GGUFValueType  # noqa: E402
    main()
