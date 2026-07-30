#!/usr/bin/env python3
"""Embed a multifunction Tessera ANE prefill bundle into a unified GGUF.

A bundle is one mlmodelc that already contains every `prefill_sN` function
the runtime needs.  This script walks the bundle, copies every file into the
GGUF as a typed tensor payload, and records the function list so the C++
loader can warm each function under its declared name without ever loading
the per-bucket format.

The on-disk file layout inside the GGUF matches the existing per-bucket
layout: a single file_count, file.NNNN tensors, and a functions array under
`tessera.ane.prefill.bundle.*`.  A top-level `tessera.ane.prefill.bundle_dir`
points the C++ loader at the relative subdirectory inside the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFWriter


def add_bundle(writer: GGUFWriter, bundle: Path) -> int:
    """Embed `bundle` as a single multifunction artifact. Returns file count."""
    if not bundle.is_dir():
        raise SystemExit(f"bundle path is not a directory: {bundle}")
    digest = hashlib.sha256()
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"bundle has no files: {bundle}")
    for index, path in enumerate(files):
        relative = path.relative_to(bundle).as_posix()
        data = np.memmap(path, mode="r", dtype=np.uint8)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(memoryview(data))
        name = f"tessera.ane.prefill.bundle.file.{index:04d}"
        writer.add_tensor(name, data.view(np.int8))
        writer.add_string(f"{name}.path", relative)
    writer.add_uint32("tessera.ane.prefill.bundle.file_count", len(files))
    writer.add_string("tessera.ane.prefill.bundle.bundle_sha256", digest.hexdigest())
    return len(files)


def add_manifest(writer: GGUFWriter, manifest: dict) -> None:
    """Copy the bundle manifest fields into the GGUF root."""
    writer.add_string("tessera.ane.prefill.format", "tessera-ane-prefill-v1")
    writer.add_uint32("tessera.ane.prefill.abi_version", 1)
    writer.add_string("tessera.ane.prefill.architecture", manifest["architecture"])
    writer.add_uint32("tessera.ane.prefill.hidden_size", manifest["hidden_size"])
    writer.add_uint32("tessera.ane.prefill.layer_first", manifest["layer_first"])
    writer.add_uint32("tessera.ane.prefill.layer_last", manifest["layer_last"])
    writer.add_string("tessera.ane.prefill.execution_stage", manifest["execution_stage"])
    writer.add_string("tessera.ane.prefill.hidden_layout", manifest["hidden_layout"])
    writer.add_string("tessera.ane.prefill.kv_layout", manifest["kv_layout"])
    writer.add_string("tessera.ane.prefill.cache_requirement", manifest["cache_requirement"])
    writer.add_uint32("tessera.ane.prefill.kv_heads", manifest["kv_heads"])
    writer.add_uint32("tessera.ane.prefill.head_dim", manifest["head_dim"])
    writer.add_uint32("tessera.ane.prefill.batch_size", manifest["batch"])
    writer.add_bool("tessera.ane.prefill.causal_right_padding", manifest.get("causal_right_padding", False))
    writer.add_array("tessera.ane.prefill.sequence_buckets", manifest["sequence_buckets"])
    function_names = [entry["name"] for entry in manifest["functions"]]
    writer.add_array("tessera.ane.prefill.bundle.functions", function_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path,
            help="multifunction mlmodelc directory (e.g. prefill-bundle.mlmodelc)")
    parser.add_argument("output", type=Path)
    parser.add_argument("--causal-right-padding", action="store_true")
    parser.add_argument("--sequence-bucket", type=int, action="append",
            help="override the declared sequence buckets")
    args = parser.parse_args()

    manifest_path = args.bundle.with_name(args.bundle.stem + ".json")
    if not manifest_path.is_file():
        raise SystemExit(f"missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "tessera-ane-prefill-bundle-v1":
        raise SystemExit("unsupported bundle manifest format")
    if args.sequence_bucket:
        manifest["sequence_buckets"] = sorted(set(args.sequence_bucket))
    manifest["causal_right_padding"] = bool(args.causal_right_padding)

    writer = GGUFWriter(args.output, "tessera-ane-prefill-test")
    add_manifest(writer, manifest)
    file_count = add_bundle(writer, args.bundle)
    writer.add_string("tessera.ane.prefill.bundle_dir", args.bundle.name)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"embedded {file_count} files from {args.bundle} into {args.output}")


if __name__ == "__main__":
    main()
