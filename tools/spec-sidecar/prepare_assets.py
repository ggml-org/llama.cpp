#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate speculative sidecar assets from GGUF files already owned by the user.

The repository never ships model weights. Install llama.cpp's ``gguf-py``
package before running this tool:

    python -m pip install -e <llama.cpp>/gguf-py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

try:
    import gguf
    from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, quants
except ImportError as exc:
    gguf = None
    _GGUF_IMPORT_ERROR = exc
else:
    _GGUF_IMPORT_ERROR = None

from validate_assets import (
    QWEN35MOE_MTP_SCHEMA,
    dflash_schema,
    validate_schema,
)


ID_FIELDS = ("qwen35.nextn.draft_vocab_ids", "qwen3.nextn.draft_vocab_ids")


@contextmanager
def temporary_output(destination: Path) -> Iterator[Path]:
    """Yield an unpredictable same-filesystem path and remove it on failure."""
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ) as directory:
        yield Path(directory) / destination.name


def model_output_name(target: Path, requested: str | None) -> str:
    name = requested if requested is not None else f"{target.stem}-spec-sidecar.gguf"
    forbidden = '<>:"/\\|?*'
    device = name.split(".", 1)[0].upper()
    reserved_names = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved_device = device in reserved_names or (
        re.fullmatch(r"(?:COM|LPT)[1-9]", device) is not None
    )
    if (
        not name
        or name != name.strip()
        or name in {".", ".."}
        or name.endswith(".")
        or any(character in name for character in forbidden)
        or any(ord(character) < 32 for character in name)
        or reserved_device
    ):
        raise ValueError("--model-name must be a plain filename without path or drive components")
    if Path(name).name != name:
        raise ValueError("--model-name must stay inside --output")
    return name


def require_gguf() -> None:
    if gguf is None:
        raise ValueError(
            "gguf-py is required. Run `python -m pip install -e <llama.cpp>/gguf-py`."
        ) from _GGUF_IMPORT_ERROR


def open_gguf(path: Path) -> GGUFReader:
    """Open a little-endian GGUF, the only layout the raw-copy paths support."""
    require_gguf()
    if sys.byteorder != "little":
        raise ValueError("asset generation currently requires a little-endian host")
    reader = GGUFReader(str(path))
    if reader.byte_order != "I":
        raise ValueError(f"{path}: big-endian GGUF input is not supported")
    return reader


def get_tensor(reader: GGUFReader, name: str):
    for value in reader.tensors:
        if value.name == name:
            return value
    raise ValueError(f"missing required tensor: {name}")


def metadata_value(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"missing required metadata: {key}")
    return field.contents()


def require_metadata(reader: GGUFReader, expected: dict[str, object], label: str) -> None:
    for key, wanted in expected.items():
        observed = metadata_value(reader, key)
        if observed != wanted:
            raise ValueError(f"{label}: expected {key}={wanted!r}, found {observed!r}")


def tensor_entries(reader: GGUFReader) -> list[dict]:
    return [
        {
            "name": value.name,
            "dtype": str(int(value.tensor_type)),
            "shape": [int(item) for item in value.shape],
            "nbytes": int(value.data.nbytes),
        }
        for value in reader.tensors
    ]


def read_ids(path: Path | None, reader: GGUFReader) -> list[int]:
    values = None
    if path:
        if path.suffix.lower() == ".bin":
            raw = path.read_bytes()
            if len(raw) % 4:
                raise ValueError("binary ID file is not little-endian int32 data")
            values = struct.unpack(f"<{len(raw) // 4}i", raw)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload["ids"] if isinstance(payload, dict) else payload
    else:
        for key in ID_FIELDS:
            field = reader.get_field(key)
            if field is not None:
                values = field.contents()
                break
    if values is None:
        raise ValueError("pass --ids, or use a GGUF containing qwen35.nextn.draft_vocab_ids")
    ids = sorted({int(value) for value in np.asarray(values).reshape(-1).tolist()})
    if len(ids) != 40_960:
        raise ValueError(f"expected 40,960 unique IDs, found {len(ids):,}")
    if ids[0] < 0 or ids[-1] >= 248_320:
        raise ValueError("draft ID outside the Qwen3.8 248,320-token vocabulary")
    return ids


def write_ids(path: Path, ids: Sequence[int]) -> None:
    with temporary_output(path) as temporary:
        temporary.write_bytes(struct.pack(f"<{len(ids)}i", *ids))
        os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_existing(paths: Sequence[Path], force: bool) -> None:
    normalized = [os.path.normcase(str(path.resolve())) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise ValueError("output paths must be distinct")
    existing = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if existing and not force:
        raise ValueError(f"output exists; pass --force to replace it: {', '.join(existing)}")


def row_view(value) -> tuple[np.ndarray, int, int]:
    block, quant_bytes = gguf.GGML_QUANT_SIZES[value.tensor_type]
    width = int(value.shape[0])
    if width % block:
        raise ValueError(f"{value.name}: width {width} is not aligned to quant block {block}")
    row_bytes = width // block * quant_bytes
    if value.data.nbytes % row_bytes:
        raise ValueError(f"{value.name}: tensor bytes are not row-aligned")
    rows = value.data.nbytes // row_bytes
    return np.asarray(value.data).view(np.uint8).reshape(rows, row_bytes), rows, row_bytes


def write_blob(
    values,
    blob: Path,
    manifest: Path,
    source: Path,
    source_hash: str,
    metadata: dict[str, str] | None = None,
) -> None:
    offset = 0
    entries = []
    with temporary_output(blob) as blob_tmp, temporary_output(manifest) as manifest_tmp:
        with blob_tmp.open("xb") as output:
            for value in values:
                raw = np.asarray(value.data).view(np.uint8).reshape(-1)
                raw.tofile(output)
                entries.append(
                    {
                        "name": value.name,
                        "dtype": str(int(value.tensor_type)),
                        "shape": [int(item) for item in value.shape],
                        "offset": offset,
                        "nbytes": int(raw.nbytes),
                    }
                )
                offset += int(raw.nbytes)
        payload = {
            "schema": 1,
            "generator": "speculative sidecar tools/prepare_assets.py",
            "source_file": source.name,
            "source_sha256": source_hash,
            "tensors": entries,
        }
        if metadata:
            payload.update(metadata)
        manifest_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(blob_tmp, blob)
        os.replace(manifest_tmp, manifest)
    print(f"wrote {blob.name}: {offset / 1e6:.1f} MB, {len(entries)} tensors")


def copy_metadata(reader: GGUFReader, writer: GGUFWriter, omitted: set[str]) -> None:
    omitted |= {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for field in reader.fields.values():
        if field.name in omitted:
            continue
        subtype = field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), field.types[0], sub_type=subtype)


def nextn_block(reader: GGUFReader) -> int:
    blocks = {
        int(match.group(1))
        for value in reader.tensors
        if (match := re.match(r"blk\.(\d+)\.nextn\.", value.name))
    }
    if len(blocks) != 1:
        raise ValueError(f"expected one next-token block, found {sorted(blocks)}")
    return blocks.pop()


def make_mtp_target(source: Path, destination: Path, ids: Sequence[int]) -> int:
    reader = open_gguf(source)
    architecture = reader.get_field("general.architecture").contents()
    block_id = nextn_block(reader)
    if block_id != 64:
        raise ValueError(f"current MTP sidecar requires next-token block 64, found {block_id}")
    embedding = get_tensor(reader, "token_embd.weight")
    if embedding.tensor_type != GGMLQuantizationType.Q4_0 or list(map(int, embedding.shape)) != [5120, 248320]:
        raise ValueError(
            f"current MTP sidecar requires Q4_0 token_embd.weight [5120,248320], "
            f"found {embedding.tensor_type.name} {list(map(int, embedding.shape))}"
        )
    output_head = get_tensor(reader, "output.weight")
    if output_head.tensor_type != GGMLQuantizationType.Q6_K or list(map(int, output_head.shape)) != [5120, 248320]:
        raise ValueError(
            f"current MTP sidecar requires Q6_K output.weight [5120,248320], "
            f"found {output_head.tensor_type.name} {list(map(int, output_head.shape))}"
        )
    raw_head, vocab, _ = row_view(output_head)
    if ids[-1] >= vocab:
        raise ValueError("draft ID exceeds output-head row count")

    # Gather before dequantizing: only 40,960 rows need conversion to Q4_0.
    gathered = np.ascontiguousarray(raw_head[np.asarray(ids, dtype=np.int64)])
    f32 = quants.dequantize(gathered, output_head.tensor_type).reshape(len(ids), int(output_head.shape[0]))
    q4_head = quants.quantize(f32, GGMLQuantizationType.Q4_0)
    head_name = f"blk.{block_id}.nextn.shared_head_head.weight"
    id_key = f"{architecture}.nextn.draft_vocab_ids"

    with temporary_output(destination) as temporary:
        writer = GGUFWriter(str(temporary), architecture)
        try:
            copy_metadata(reader, writer, {*ID_FIELDS, id_key})
            writer.add_key_value(id_key, list(ids), GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32)
            for value in reader.tensors:
                if value.name == head_name:
                    continue
                requantize = (
                    value.name.startswith(f"blk.{block_id}.")
                    and value.name.endswith(".weight")
                    and len(value.shape) >= 2
                    and value.tensor_type != GGMLQuantizationType.Q4_0
                )
                if requantize:
                    shape = [int(item) for item in value.shape]
                    try:
                        dequantized = quants.dequantize(value.data, value.tensor_type).reshape(*reversed(shape))
                    except NotImplementedError as exc:
                        raise ValueError(
                            f"cannot convert {value.name} from {value.tensor_type.name} to Q4_0"
                        ) from exc
                    converted = quants.quantize(dequantized, GGMLQuantizationType.Q4_0)
                    writer.add_tensor(
                        value.name,
                        converted,
                        raw_shape=converted.shape,
                        raw_dtype=GGMLQuantizationType.Q4_0,
                    )
                    print(f"requantized {value.name}: {value.tensor_type.name} -> Q4_0")
                else:
                    writer.add_tensor(
                        value.name,
                        value.data,
                        raw_shape=value.data.shape,
                        raw_dtype=value.tensor_type,
                    )
            writer.add_tensor(
                head_name,
                q4_head,
                raw_shape=q4_head.shape,
                raw_dtype=GGMLQuantizationType.Q4_0,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
        finally:
            writer.close()
        os.replace(temporary, destination)
    return block_id


def prepare_mtp(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    source_resolved = args.target.resolve()
    source_reader = open_gguf(args.target)
    ids = read_ids(args.ids, source_reader)
    prepared = args.output / model_output_name(args.target, args.model_name)
    if source_resolved == prepared.resolve():
        raise ValueError("refusing to overwrite the source target GGUF")
    outputs = [
        prepared,
        args.output / "drafter_weights.bin",
        args.output / "drafter_manifest.json",
        args.output / "draft_head_ids.bin",
    ]
    refuse_existing(outputs, args.force)
    source_hash = sha256(args.target)
    id_hash = hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()
    block_id = make_mtp_target(args.target, prepared, ids)
    reader = open_gguf(prepared)
    values = [
        value
        for value in reader.tensors
        if value.name == "token_embd.weight" or value.name.startswith(f"blk.{block_id}.")
    ]
    write_blob(
        values,
        args.output / "drafter_weights.bin",
        args.output / "drafter_manifest.json",
        args.target,
        source_hash,
        {"draft_ids_sha256": id_hash},
    )
    write_ids(args.output / "draft_head_ids.bin", ids)
    print(f"prepared MTP target: {prepared}")


def _qwen35moe_q4_tensor(value):
    shape = [int(item) for item in value.shape]
    dequantized = quants.dequantize(value.data, value.tensor_type).reshape(*reversed(shape))
    return quants.quantize(dequantized, GGMLQuantizationType.Q4_0)


def _qwen35moe_f32_tensor(value):
    shape = [int(item) for item in value.shape]
    dequantized = quants.dequantize(value.data, value.tensor_type).reshape(*reversed(shape))
    return np.asarray(dequantized, dtype=np.float32)


def prepare_qwen35moe_mtp(args: argparse.Namespace) -> None:
    """Prepare the Qwen3.6 35B-A3B MoE MTP bundle."""
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output / "drafter_weights.bin",
        args.output / "drafter_manifest.json",
        args.output / "draft_head_ids.bin",
    ]
    refuse_existing(outputs, args.force)
    if args.ids is None:
        raise ValueError("Qwen3.6 35B-A3B MTP requires --ids with 40,960 output-head IDs")

    source = open_gguf(args.target)
    require_metadata(
        source,
        {
            "general.architecture": "qwen35moe",
            "general.size_label": "35B-A3B",
            "qwen35moe.block_count": 41,
            "qwen35moe.embedding_length": 2048,
            "qwen35moe.attention.head_count": 16,
            "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.attention.value_length": 256,
            "qwen35moe.expert_count": 256,
            "qwen35moe.expert_used_count": 8,
            "qwen35moe.expert_feed_forward_length": 512,
            "qwen35moe.nextn_predict_layers": 1,
            "qwen35moe.rope.dimension_count": 64,
        },
        "Qwen3.6 35B-A3B target",
    )
    if "Qwen3.6" not in str(metadata_value(source, "general.name")):
        raise ValueError("target model identity is not Qwen3.6")
    tokens = metadata_value(source, "tokenizer.ggml.tokens")
    if len(tokens) != 248_320:
        raise ValueError(f"Qwen3.6 35B-A3B target must have 248,320 tokens, found {len(tokens):,}")
    if nextn_block(source) != 40:
        raise ValueError("Qwen3.6 35B-A3B MTP block must be block 40")

    selected_names = set(QWEN35MOE_MTP_SCHEMA)
    source_names = {value.name for value in source.tensors}
    if not selected_names.issubset(source_names):
        missing = sorted(selected_names - source_names)
        raise ValueError(f"Qwen3.6 35B-A3B source is missing required tensors: {missing}")

    prepared = args.output / "qwen35moe-mtp-q4.gguf"
    prepared_outputs = [prepared]
    refuse_existing(prepared_outputs, args.force)
    source_hash = sha256(args.target)
    identity_ids = read_ids(args.ids, source)
    id_hash = hashlib.sha256(struct.pack(f"<{len(identity_ids)}i", *identity_ids)).hexdigest()

    with temporary_output(prepared) as temporary:
        writer = GGUFWriter(str(temporary), "qwen35moe")
        try:
            copy_metadata(source, writer, set(ID_FIELDS) | {"qwen35moe.nextn.draft_vocab_ids"})
            writer.add_key_value(
                "qwen35moe.nextn.draft_vocab_ids",
                identity_ids,
                GGUFValueType.ARRAY,
                sub_type=GGUFValueType.INT32,
            )
            for value in source.tensors:
                if value.name not in selected_names:
                    continue
                if value.name == "output.weight":
                    raw_head, vocabulary, _ = row_view(value)
                    if identity_ids[-1] >= vocabulary:
                        raise ValueError("draft-head ID exceeds Qwen3.6 output vocabulary")
                    gathered = np.ascontiguousarray(raw_head[np.asarray(identity_ids, dtype=np.int64)])
                    dequantized = quants.dequantize(gathered, value.tensor_type).reshape(
                        len(identity_ids), int(value.shape[0]))
                    converted = quants.quantize(dequantized, GGMLQuantizationType.Q4_0)
                    raw_shape = converted.shape
                    raw_dtype = GGMLQuantizationType.Q4_0
                elif value.tensor_type == GGMLQuantizationType.F32:
                    converted = value.data
                    raw_shape = value.data.shape
                    raw_dtype = GGMLQuantizationType.F32
                elif value.tensor_type == GGMLQuantizationType.BF16:
                    converted = _qwen35moe_f32_tensor(value)
                    raw_shape = converted.shape
                    raw_dtype = GGMLQuantizationType.F32
                else:
                    converted = _qwen35moe_q4_tensor(value)
                    raw_shape = converted.shape
                    raw_dtype = GGMLQuantizationType.Q4_0
                writer.add_tensor(value.name, converted, raw_shape=raw_shape, raw_dtype=raw_dtype)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
        finally:
            writer.close()
        os.replace(temporary, prepared)

    prepared_reader = open_gguf(prepared)
    values = [value for value in prepared_reader.tensors if value.name in selected_names]
    entries = tensor_entries(prepared_reader)
    entries = [entry for entry in entries if entry["name"] in selected_names]
    validate_schema(entries, QWEN35MOE_MTP_SCHEMA, "Qwen3.6 35B-A3B MTP")
    write_blob(
        values,
        args.output / "drafter_weights.bin",
        args.output / "drafter_manifest.json",
        args.target,
        source_hash,
        {
            "provider": "qwen35moe-mtp",
            "source_file": args.target.name,
            "draft_ids_sha256": id_hash,
        },
    )
    write_ids(args.output / "draft_head_ids.bin", identity_ids)
    print(f"prepared Qwen3.6 35B-A3B MTP bundle: {args.output}")


def prepare_dflash(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output / "dflash_weights.bin",
        args.output / "dflash_manifest.json",
        args.output / "drafter_weights.bin",
        args.output / "drafter_manifest.json",
        args.output / "target_head_sliced.bin",
        args.output / "draft_head_ids.bin",
    ]
    full_head_path = args.output / "target_head.bin"
    outputs.append(full_head_path)
    refuse_existing(outputs, args.force)
    target = open_gguf(args.target)
    draft = open_gguf(args.draft)
    ids = read_ids(args.ids, target)
    draft_entries = [
        {
            "name": value.name,
            "dtype": str(int(value.tensor_type)),
            "shape": [int(item) for item in value.shape],
            "nbytes": int(value.data.nbytes),
        }
        for value in draft.tensors
    ]
    validate_schema(draft_entries, dflash_schema(), "DFlash source")

    embedding = get_tensor(target, "token_embd.weight")
    if embedding.tensor_type != GGMLQuantizationType.Q4_0 or list(map(int, embedding.shape)) != [5120, 248320]:
        raise ValueError(
            f"expected Q4_0 token embedding [5120,248320], found "
            f"{embedding.tensor_type.name} {list(map(int, embedding.shape))}"
        )
    head = get_tensor(target, "output.weight")
    if head.tensor_type != GGMLQuantizationType.Q6_K or list(map(int, head.shape)) != [5120, 248320]:
        raise ValueError(
            f"expected Q6_K output.weight [5120,248320], found "
            f"{head.tensor_type.name} {list(map(int, head.shape))}"
        )
    raw_head, vocab, row_bytes = row_view(head)
    if ids[-1] >= vocab or row_bytes != 4_200:
        raise ValueError(f"unexpected output-head layout: rows={vocab}, row_bytes={row_bytes}")

    target_hash = sha256(args.target)
    draft_hash = sha256(args.draft)
    id_hash = hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()
    write_blob(
        draft.tensors,
        args.output / "dflash_weights.bin",
        args.output / "dflash_manifest.json",
        args.draft,
        draft_hash,
        {"target_source_sha256": target_hash, "draft_ids_sha256": id_hash},
    )
    write_blob(
        [embedding],
        args.output / "drafter_weights.bin",
        args.output / "drafter_manifest.json",
        args.target,
        target_hash,
        {"draft_source_sha256": draft_hash, "draft_ids_sha256": id_hash},
    )
    sliced = np.ascontiguousarray(raw_head[np.asarray(ids, dtype=np.int64)])
    sliced_path = args.output / "target_head_sliced.bin"
    with temporary_output(sliced_path) as sliced_tmp:
        with sliced_tmp.open("xb") as output:
            sliced.tofile(output)
        os.replace(sliced_tmp, sliced_path)
    if args.full_head:
        with temporary_output(full_head_path) as full_tmp:
            with full_tmp.open("xb") as output:
                np.asarray(raw_head).tofile(output)
            os.replace(full_tmp, full_head_path)
    elif full_head_path.exists() or full_head_path.is_symlink():
        # An explicit --force non-full build must not leave a stale optional asset.
        full_head_path.unlink()
    write_ids(args.output / "draft_head_ids.bin", ids)
    print(f"wrote sliced Q6_K head: {sliced.nbytes / 1e6:.1f} MB")


def inspect(args: argparse.Namespace) -> None:
    reader = open_gguf(args.model)
    print(f"model={args.model}\ntensors={len(reader.tensors)}")
    for name in ("token_embd.weight", "output.weight"):
        try:
            value = get_tensor(reader, name)
        except ValueError:
            continue
        print(f"{name}: {value.tensor_type.name} {list(map(int, value.shape))} {value.data.nbytes:,} bytes")
    print(f"next-token tensors={sum('.nextn.' in value.name for value in reader.tensors)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    view = commands.add_parser("inspect")
    view.add_argument("model", type=Path)
    view.set_defaults(run=inspect)

    mtp = commands.add_parser("mtp")
    mtp.add_argument("--target", type=Path, required=True)
    mtp.add_argument("--ids", type=Path)
    mtp.add_argument("--output", type=Path, required=True)
    mtp.add_argument("--model-name", help="output GGUF filename (directories are rejected)")
    mtp.add_argument("--force", action="store_true", help="replace existing outputs")
    mtp.set_defaults(run=prepare_mtp)

    qwen35moe_mtp = commands.add_parser("qwen35moe-mtp")
    qwen35moe_mtp.add_argument("--target", type=Path, required=True)
    qwen35moe_mtp.add_argument("--ids", type=Path, required=True)
    qwen35moe_mtp.add_argument("--output", type=Path, required=True)
    qwen35moe_mtp.add_argument("--force", action="store_true", help="replace existing outputs")
    qwen35moe_mtp.set_defaults(run=prepare_qwen35moe_mtp)


    dflash = commands.add_parser("dflash")
    dflash.add_argument("--target", type=Path, required=True)
    dflash.add_argument("--draft", type=Path, required=True)
    dflash.add_argument("--ids", type=Path)
    dflash.add_argument("--output", type=Path, required=True)
    dflash.add_argument("--full-head", action="store_true")
    dflash.add_argument("--force", action="store_true", help="replace existing outputs")
    dflash.set_defaults(run=prepare_dflash)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.run(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
