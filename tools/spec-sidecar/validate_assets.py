#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate speculative sidecar artifact structure without loading a model on the GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path


MTP_SCHEMA = {
    "token_embd.weight": ("2", [5120, 248320], 715161600),
    "blk.64.attn_k.weight": ("2", [5120, 1024], 2949120),
    "blk.64.attn_k_norm.weight": ("0", [256], 1024),
    "blk.64.attn_norm.weight": ("0", [5120], 20480),
    "blk.64.attn_output.weight": ("2", [6144, 5120], 17694720),
    "blk.64.attn_q.weight": ("2", [5120, 12288], 35389440),
    "blk.64.attn_q_norm.weight": ("0", [256], 1024),
    "blk.64.attn_v.weight": ("2", [5120, 1024], 2949120),
    "blk.64.ffn_down.weight": ("2", [17408, 5120], 50135040),
    "blk.64.ffn_gate.weight": ("2", [5120, 17408], 50135040),
    "blk.64.ffn_up.weight": ("2", [5120, 17408], 50135040),
    "blk.64.nextn.eh_proj.weight": ("2", [10240, 5120], 29491200),
    "blk.64.nextn.enorm.weight": ("0", [5120], 20480),
    "blk.64.nextn.hnorm.weight": ("0", [5120], 20480),
    "blk.64.nextn.shared_head_norm.weight": ("0", [5120], 20480),
    "blk.64.post_attention_norm.weight": ("0", [5120], 20480),
    "blk.64.nextn.shared_head_head.weight": ("2", [5120, 40960], 117964800),
}


def dflash_schema() -> dict[str, tuple[str, list[int], int]]:
    schema = {
        "enc.output_norm.weight": ("0", [5120], 20480),
        "fc.weight": ("12", [25600, 5120], 73728000),
        "output_norm.weight": ("0", [5120], 20480),
        "selector_hidden.weight": ("12", [5120, 256], 737280),
        "selector_predecessor.weight": ("12", [256, 248320], 35758080),
        "selector_successor.weight": ("12", [256, 248320], 35758080),
    }
    for layer in range(5):
        prefix = f"blk.{layer}."
        q6 = layer in (2, 4)
        schema.update(
            {
                prefix + "attn_conv_base": ("0", [5120, 2, 2], 81920),
                prefix + "attn_conv_proj.weight": ("12", [5120, 1280], 3686400),
                prefix + "attn_k.weight": ("12", [5120, 1024], 2949120),
                prefix + "attn_k_norm.weight": ("0", [128], 512),
                prefix + "attn_norm.weight": ("0", [5120], 20480),
                prefix + "attn_output.weight": ("12", [4096, 5120], 11796480),
                prefix + "attn_q.weight": ("12", [5120, 4096], 11796480),
                prefix + "attn_q_norm.weight": ("0", [128], 512),
                prefix + "attn_v.weight": (("14", [5120, 1024], 4300800) if q6 else ("12", [5120, 1024], 2949120)),
                prefix + "ffn_conv_base": ("0", [5120, 2, 2], 81920),
                prefix + "ffn_conv_proj.weight": ("12", [5120, 1280], 3686400),
                prefix + "ffn_down.weight": (("14", [17408, 5120], 73113600) if q6 else ("12", [17408, 5120], 50135040)),
                prefix + "ffn_gate.weight": ("12", [5120, 17408], 50135040),
                prefix + "ffn_norm.weight": ("0", [5120], 20480),
                prefix + "ffn_up.weight": ("12", [5120, 17408], 50135040),
            }
        )
    return schema


def load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "schema" in payload and payload["schema"] != 1:
        raise ValueError(f"{path}: unsupported manifest schema {payload['schema']!r}")
    for key, value in payload.items():
        if key.endswith("_sha256") and (not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None):
            raise ValueError(f"{path}: {key} must be a lowercase 64-character SHA-256 digest")
    tensors = payload.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError(f"{path}: missing tensors array")
    return tensors


def validate_blob(directory: Path, manifest_name: str, blob_name: str, expected_count: int) -> list[dict]:
    manifest_path = directory / manifest_name
    blob_path = directory / blob_name
    tensors = load_manifest(manifest_path)
    if len(tensors) != expected_count:
        raise ValueError(f"{manifest_path}: expected {expected_count} tensors, found {len(tensors)}")
    cursor = 0
    names: set[str] = set()
    for item in tensors:
        name = str(item["name"])
        if name in names:
            raise ValueError(f"{manifest_path}: duplicate tensor {name}")
        names.add(name)
        if int(item["offset"]) != cursor:
            raise ValueError(f"{manifest_path}: non-contiguous offset for {name}")
        nbytes = int(item["nbytes"])
        if nbytes <= 0:
            raise ValueError(f"{manifest_path}: invalid byte count for {name}")
        cursor += nbytes
    actual = blob_path.stat().st_size
    if actual != cursor:
        raise ValueError(f"{blob_path}: manifest says {cursor:,} bytes, file has {actual:,}")
    return tensors


def validate_schema(tensors: list[dict], expected: dict[str, tuple[str, list[int], int]], label: str) -> None:
    actual = {str(item["name"]): item for item in tensors}
    if set(actual) != set(expected):
        raise ValueError(
            f"{label} tensor set mismatch; missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for name, (dtype, shape, nbytes) in expected.items():
        item = actual[name]
        observed = (str(item["dtype"]), [int(value) for value in item["shape"]], int(item["nbytes"]))
        wanted = (dtype, shape, nbytes)
        if observed != wanted:
            raise ValueError(f"{label} schema mismatch for {name}: expected {wanted}, found {observed}")


def validate_ids(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) != 40_960 * 4:
        raise ValueError(f"{path}: expected 163,840 bytes, found {len(raw):,}")
    ids = list(struct.unpack("<40960i", raw))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: IDs are not unique")
    if min(ids) < 0 or max(ids) >= 248_320:
        raise ValueError(f"{path}: ID outside Qwen3.8 vocabulary")
    return ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mtp(directory: Path) -> list[Path]:
    tensors = validate_blob(directory, "drafter_manifest.json", "drafter_weights.bin", 17)
    validate_schema(tensors, MTP_SCHEMA, "MTP")
    validate_ids(directory / "draft_head_ids.bin")
    return [
        directory / "drafter_manifest.json",
        directory / "drafter_weights.bin",
        directory / "draft_head_ids.bin",
    ]


def validate_dflash(directory: Path) -> list[Path]:
    dflash = validate_blob(directory, "dflash_manifest.json", "dflash_weights.bin", 81)
    validate_schema(dflash, dflash_schema(), "DFlash")
    embedding_count = len(load_manifest(directory / "drafter_manifest.json"))
    if embedding_count not in (1, 17):
        raise ValueError(f"DFlash target blob must contain 1 or 17 tensors, found {embedding_count}")
    embedding = validate_blob(
        directory,
        "drafter_manifest.json",
        "drafter_weights.bin",
        embedding_count,
    )
    if embedding_count == 17:
        validate_schema(embedding, MTP_SCHEMA, "DFlash target/MTP blob")
    else:
        validate_schema(
            embedding,
            {"token_embd.weight": ("2", [5120, 248320], 715161600)},
            "DFlash target embedding",
        )
    validate_ids(directory / "draft_head_ids.bin")
    head = directory / "target_head_sliced.bin"
    expected = 40_960 * 4_200
    if head.stat().st_size != expected:
        raise ValueError(f"{head}: expected {expected:,} bytes, found {head.stat().st_size:,}")
    return [
        directory / "dflash_manifest.json",
        directory / "dflash_weights.bin",
        directory / "drafter_manifest.json",
        directory / "drafter_weights.bin",
        directory / "target_head_sliced.bin",
        directory / "draft_head_ids.bin",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("mtp", "dflash"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--hash", action="store_true", help="also calculate SHA-256 (slow for large blobs)")
    args = parser.parse_args()
    try:
        paths = validate_mtp(args.directory) if args.kind == "mtp" else validate_dflash(args.directory)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print(f"VALID: {args.kind} artifact set at {args.directory}")
    if args.hash:
        for path in paths:
            print(f"{sha256(path)}  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
