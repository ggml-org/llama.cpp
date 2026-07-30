#!/usr/bin/env python3
"""Create a self-contained model GGUF with a Tessera ANE prefill artifact.

GGUF metadata precedes tensor data, so an existing model cannot safely be
appended with a new Core ML payload.  This copier preserves every source KV
and tensor verbatim while adding the versioned prefill namespace.  Tensor data
is streamed from GGUFReader memmaps; the model is never materialized in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter


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
            # GGUFReader exposes dimensions in serialized order; GGUFWriter
            # accepts logical NumPy order and serializes it in reverse.
            # Preserve the original tensor descriptor exactly.
            tuple(reversed(tensor.shape)),
            tensor.data.dtype,
            tensor.n_bytes,
            GGMLQuantizationType(tensor.tensor_type),
        )


def add_bundle(writer: GGUFWriter, bundles: Path, args: argparse.Namespace) -> None:
    buckets: list[int] = []
    for sequence in (128, 256, 512, 1024):
        bundle = bundles / f"prefill-s{sequence}.mlmodelc"
        if not bundle.is_dir():
            continue
        prefix = f"tessera.ane.prefill.bucket.{sequence}"
        digest = hashlib.sha256()
        files = sorted(path for path in bundle.rglob("*") if path.is_file())
        for index, path in enumerate(files):
            relative = path.relative_to(bundle).as_posix()
            data = np.memmap(path, mode="r", dtype=np.uint8)
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(memoryview(data))
            name = f"{prefix}.file.{index:04d}"
            writer.add_tensor(name, data.view(np.int8))
            writer.add_string(f"{name}.path", relative)
        writer.add_uint32(f"{prefix}.file_count", len(files))
        writer.add_string(f"{prefix}.bundle_sha256", digest.hexdigest())
        manifest = bundles / f"prefill-s{sequence}.json"
        if manifest.is_file():
            functions = json.loads(manifest.read_text()).get("functions", [])
            if functions:
                writer.add_array(f"{prefix}.functions", functions)
        buckets.append(sequence)
    if not buckets:
        raise SystemExit("no prefill-sN.mlmodelc bundle found")
    writer.add_string("tessera.ane.prefill.format", "tessera-ane-prefill-v1")
    writer.add_uint32("tessera.ane.prefill.abi_version", 1)
    writer.add_string("tessera.ane.prefill.architecture", "gemma4")
    writer.add_uint32("tessera.ane.prefill.hidden_size", args.hidden_size)
    writer.add_uint32("tessera.ane.prefill.layer_first", 0)
    writer.add_uint32("tessera.ane.prefill.layer_last", 0)
    writer.add_string("tessera.ane.prefill.execution_stage", "layer_slab")
    writer.add_string("tessera.ane.prefill.hidden_layout", "token_major.f32.v1")
    writer.add_string("tessera.ane.prefill.kv_layout", "llama.gemma4.kv_rows.f16.v1")
    writer.add_string("tessera.ane.prefill.cache_requirement", "empty_contiguous_prompt")
    writer.add_uint32("tessera.ane.prefill.kv_heads", args.kv_heads)
    writer.add_uint32("tessera.ane.prefill.head_dim", args.head_dim)
    writer.add_uint32("tessera.ane.prefill.batch_size", 1)
    writer.add_bool("tessera.ane.prefill.causal_right_padding", args.causal_right_padding)
    writer.add_array("tessera.ane.prefill.sequence_buckets", buckets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("bundles", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hidden-size", type=int, default=3840)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument(
        "--causal-right-padding",
        action="store_true",
        help="allow a shorter causal prompt to use the next larger prefill bucket",
    )
    args = parser.parse_args()
    if not args.source.is_file() or args.output.exists():
        raise SystemExit("source must exist and output must not already exist")
    reader = GGUFReader(args.source)
    if reader.fields["general.architecture"].contents() != "gemma4":
        raise SystemExit("only Gemma 4 model GGUFs are accepted")
    writer = GGUFWriter(args.output, "gemma4")
    copy_source(writer, reader)
    add_bundle(writer, args.bundles, args)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)
    # The payload tensors follow the source tensor descriptors in the same
    # writer queue. Stream them after the copied source data without asking
    # write_tensors_to_file() to emit a second tensor-info table.
    while writer.tensors[0]:
        tensor_info = next(iter(writer.tensors[0].values()))
        writer.write_tensor_data(tensor_info.tensor)
    writer.close()


if __name__ == "__main__":
    main()
