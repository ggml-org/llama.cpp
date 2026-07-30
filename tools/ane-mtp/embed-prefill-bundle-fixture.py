#!/usr/bin/env python3
"""Embed stateless ANE prefill functions in a Tessera prefill GGUF fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundles", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--architecture", default="gemma4")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--layer-first", type=int, default=0)
    parser.add_argument("--layer-last", type=int, default=0)
    parser.add_argument("--execution-stage", default="layer_slab", choices=("layer_slab",))
    parser.add_argument("--hidden-layout", default="token_major.f32.v1")
    parser.add_argument("--kv-layout", default="llama.unified.f16.v1")
    parser.add_argument("--cache-requirement", default="empty_contiguous_prompt")
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    writer = GGUFWriter(args.output, "tessera-ane-prefill-test")
    retained: list[np.ndarray] = []
    buckets: list[int] = []
    for sequence in (128, 256, 512, 1024):
        bundle = args.bundles / f"prefill-s{sequence}.mlmodelc"
        if not bundle.is_dir():
            continue
        prefix = f"tessera.ane.prefill.bucket.{sequence}"
        digest = hashlib.sha256()
        files = sorted(path for path in bundle.rglob("*") if path.is_file())
        for index, path in enumerate(files):
            relative = path.relative_to(bundle).as_posix()
            data = np.memmap(path, mode="r", dtype=np.uint8)
            retained.append(data)
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(memoryview(data))
            name = f"{prefix}.file.{index:04d}"
            writer.add_tensor(name, data.view(np.int8))
            writer.add_string(f"{name}.path", relative)
        writer.add_uint32(f"{prefix}.file_count", len(files))
        writer.add_string(f"{prefix}.bundle_sha256", digest.hexdigest())
        manifest = args.bundles / f"prefill-s{sequence}.json"
        if manifest.is_file():
            functions = json.loads(manifest.read_text()).get("functions", [])
            if functions:
                writer.add_array(f"{prefix}.functions", functions)
        buckets.append(sequence)

    if not buckets:
        raise SystemExit("no prefill-sN.mlmodelc bundles found")
    writer.add_string("tessera.ane.prefill.format", "tessera-ane-prefill-v1")
    writer.add_uint32("tessera.ane.prefill.abi_version", 1)
    writer.add_string("tessera.ane.prefill.architecture", args.architecture)
    writer.add_uint32("tessera.ane.prefill.hidden_size", args.hidden_size)
    writer.add_uint32("tessera.ane.prefill.layer_first", args.layer_first)
    writer.add_uint32("tessera.ane.prefill.layer_last", args.layer_last)
    writer.add_string("tessera.ane.prefill.execution_stage", args.execution_stage)
    writer.add_string("tessera.ane.prefill.hidden_layout", args.hidden_layout)
    writer.add_string("tessera.ane.prefill.kv_layout", args.kv_layout)
    writer.add_string("tessera.ane.prefill.cache_requirement", args.cache_requirement)
    writer.add_uint32("tessera.ane.prefill.kv_heads", args.kv_heads)
    writer.add_uint32("tessera.ane.prefill.head_dim", args.head_dim)
    writer.add_uint32("tessera.ane.prefill.batch_size", args.batch_size)
    writer.add_array("tessera.ane.prefill.sequence_buckets", buckets)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
