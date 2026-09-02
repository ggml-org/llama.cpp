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


QWEN35_MTP_SCHEMA = {
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


QWEN35MOE_MTP_SCHEMA = {
    "token_embd.weight": ("2", [2048, 248320], 286064640),
    "blk.40.attn_k.weight": ("2", [2048, 512], 589824),
    "blk.40.attn_k_norm.weight": ("0", [256], 1024),
    "blk.40.attn_norm.weight": ("0", [2048], 8192),
    "blk.40.attn_output.weight": ("2", [4096, 2048], 4718592),
    "blk.40.attn_q.weight": ("2", [2048, 8192], 9437184),
    "blk.40.attn_q_norm.weight": ("0", [256], 1024),
    "blk.40.attn_v.weight": ("2", [2048, 512], 589824),
    "blk.40.ffn_down_exps.weight": ("2", [512, 2048, 256], 150994944),
    "blk.40.ffn_down_shexp.weight": ("2", [512, 2048], 589824),
    "blk.40.ffn_gate_exps.weight": ("2", [2048, 512, 256], 150994944),
    "blk.40.ffn_gate_inp.weight": ("0", [2048, 256], 2097152),
    "blk.40.ffn_gate_inp_shexp.weight": ("0", [2048], 8192),
    "blk.40.ffn_gate_shexp.weight": ("2", [2048, 512], 589824),
    "blk.40.ffn_up_exps.weight": ("2", [2048, 512, 256], 150994944),
    "blk.40.ffn_up_shexp.weight": ("2", [2048, 512], 589824),
    "blk.40.nextn.eh_proj.weight": ("2", [4096, 2048], 4718592),
    "blk.40.nextn.enorm.weight": ("0", [2048], 8192),
    "blk.40.nextn.hnorm.weight": ("0", [2048], 8192),
    "blk.40.nextn.shared_head_norm.weight": ("0", [2048], 8192),
    "blk.40.post_attention_norm.weight": ("0", [2048], 8192),
    "output.weight": ("2", [2048, 40960], 47185920),
}


QWEN4EXP_MTP_SCHEMA = {
    "output.weight": ("14", [2560, 248320], 521472000),
    "output_hc_down.weight": ("2", [10240, 320], 1843200),
    "output_hc_norm.weight": ("0", [10240], 40960),
    "output_hc_up.weight": ("2", [320, 10240], 1843200),
    "token_embd.weight": ("2", [2560, 248320], 357580800),
    "blk.48.attn_k.weight": ("2", [2560, 512], 737280),
    "blk.48.attn_k_norm.weight": ("0", [256], 1024),
    "blk.48.attn_output.weight": ("2", [6144, 2560], 8847360),
    "blk.48.attn_q.weight": ("2", [2560, 12288], 17694720),
    "blk.48.attn_q_norm.weight": ("0", [256], 1024),
    "blk.48.attn_v.weight": ("2", [2560, 512], 737280),
    "blk.48.ffn_down_exps.weight": ("2", [640, 2560, 512], 471859200),
    "blk.48.ffn_down_shexp.weight": ("2", [640, 2560], 921600),
    "blk.48.ffn_gate_exps.weight": ("2", [2560, 640, 512], 471859200),
    "blk.48.ffn_gate_inp.weight": ("0", [2560, 512], 5242880),
    "blk.48.ffn_gate_inp_shexp.weight": ("0", [2560], 10240),
    "blk.48.ffn_gate_shexp.weight": ("2", [2560, 640], 921600),
    "blk.48.ffn_up_exps.weight": ("2", [2560, 640, 512], 471859200),
    "blk.48.ffn_up_shexp.weight": ("2", [2560, 640], 921600),
    "blk.48.hc_attn_down.weight": ("2", [10240, 320], 1843200),
    "blk.48.hc_attn_inject.weight": ("2", [10240, 4], 23040),
    "blk.48.hc_attn_norm.weight": ("0", [10240], 40960),
    "blk.48.hc_attn_up.weight": ("2", [320, 10240], 1843200),
    "blk.48.hc_ffn_down.weight": ("2", [10240, 320], 1843200),
    "blk.48.hc_ffn_inject.weight": ("2", [10240, 4], 23040),
    "blk.48.hc_ffn_norm.weight": ("0", [10240], 40960),
    "blk.48.hc_ffn_up.weight": ("2", [320, 10240], 1843200),
    "blk.48.indexer.k_norm.weight": ("0", [128], 512),
    "blk.48.indexer.k_proj.weight": ("1", [2560, 128], 655360),
    "blk.48.indexer.q_norm.weight": ("0", [128], 512),
    "blk.48.indexer.q_proj.weight": ("1", [2560, 512], 2621440),
    "blk.48.nextn.eh_proj.weight": ("2", [5120, 2560], 7372800),
    "blk.48.nextn.enorm.weight": ("0", [2560], 10240),
    "blk.48.nextn.hnorm.weight": ("0", [10240], 40960),
}

# Backward-compatible name for out-of-tree callers of the asset helper.
MTP_SCHEMA = QWEN35_MTP_SCHEMA


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


def validate_ids(path: Path, rows: int = 40_960) -> list[int]:
    raw = path.read_bytes()
    expected_bytes = rows * 4
    if len(raw) != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes:,} bytes, found {len(raw):,}")
    ids = list(struct.unpack(f"<{rows}i", raw))
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
    validate_schema(tensors, QWEN35_MTP_SCHEMA, "Qwen3.8-27B MTP")
    validate_ids(directory / "draft_head_ids.bin")
    return [
        directory / "drafter_manifest.json",
        directory / "drafter_weights.bin",
        directory / "draft_head_ids.bin",
    ]


def validate_qwen35moe_mtp(directory: Path) -> list[Path]:
    tensors = validate_blob(directory, "drafter_manifest.json", "drafter_weights.bin", 22)
    validate_schema(tensors, QWEN35MOE_MTP_SCHEMA, "Qwen3.6 35B-A3B MTP")
    ids = validate_ids(directory / "draft_head_ids.bin", 40_960)
    if len(ids) != 40_960:
        raise ValueError("Qwen3.6 35B-A3B MTP requires a 40,960-row draft-head ID table")
    return [
        directory / "drafter_manifest.json",
        directory / "drafter_weights.bin",
        directory / "draft_head_ids.bin",
    ]


def validate_qwen4exp_mtp(directory: Path) -> list[Path]:
    tensors = validate_blob(directory, "drafter_manifest.json", "drafter_weights.bin", 34)
    validate_schema(tensors, QWEN4EXP_MTP_SCHEMA, "Qwen3.8 Flash Next MTP")
    ids = validate_ids(directory / "draft_head_ids.bin", 248_320)
    if ids != list(range(248_320)):
        raise ValueError("Qwen3.8 Flash Next MTP requires a full-vocabulary identity ID table")
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
        validate_schema(embedding, QWEN35_MTP_SCHEMA, "DFlash target/MTP blob")
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
    parser.add_argument("kind", choices=("mtp", "qwen35moe-mtp", "qwen4exp-mtp", "dflash"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--hash", action="store_true", help="also calculate SHA-256 (slow for large blobs)")
    args = parser.parse_args()
    try:
        if args.kind == "mtp":
            paths = validate_mtp(args.directory)
        elif args.kind == "qwen35moe-mtp":
            paths = validate_qwen35moe_mtp(args.directory)
        elif args.kind == "qwen4exp-mtp":
            paths = validate_qwen4exp_mtp(args.directory)
        else:
            paths = validate_dflash(args.directory)
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
