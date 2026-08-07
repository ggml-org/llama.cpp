#!/usr/bin/env python3
"""Rank BF16 Q4_0 tensors by estimated Q4_0 -> Q8_0 error reduction.

This is an automatic, model-specific heuristic. It computes a weighted block
reconstruction-error estimate directly from BF16 weights, optionally using
imatrix in_sum2 values, then selects promotions under a final Q8 byte budget.
The result is an exact tensor override file consumed by the S8 builder.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

LINE_RE = re.compile(
    r"^\[\s*\d+\s*/\s*\d+\]\s+(\S+)\s+-\s+\[([^\]]+)\],\s+"
    r"type\s*=\s*(\S+),\s+size\s*=\s*([0-9.]+)\s+MiB"
    r"(?:\s+->\s+[0-9.]+\s+MiB\s+\(([^)]+)\))?"
)
BLOCK_BYTES = {"q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22),
               "q6_k": (256, 210), "q8_0": (32, 34)}
FLOAT_TYPES = {"f16": 2, "bf16": 2, "f32": 4}


def type_name(value: int) -> str:
    return GGMLQuantizationType(int(value)).name.lower()


def load_base_types(path: Path | None, log: Path | None) -> dict[str, str]:
    if path is not None:
        reader = GGUFReader(str(path))
        return {tensor.name: type_name(tensor.tensor_type) for tensor in reader.tensors}
    assert log is not None
    result: dict[str, str] = {}
    for line in log.read_text(errors="replace").splitlines():
        match = LINE_RE.match(line)
        if match:
            name, _shape, source, _size, final = match.groups()
            result[name] = (final or source).lower()
    return result


def tensor_bytes(shape: list[int], typ: str) -> int:
    elements = math.prod(shape)
    if typ in BLOCK_BYTES:
        block, size = BLOCK_BYTES[typ]
        if shape[0] % block:
            return 0
        return (elements // block) * size
    if typ in FLOAT_TYPES:
        return elements * FLOAT_TYPES[typ]
    return 0


def as_bf16_rows(tensor, ncols: int) -> np.ndarray:
    raw = np.asarray(tensor.data)
    if raw.dtype == np.uint8:
        words = raw.reshape(-1).view("<u2")
    elif raw.dtype == np.uint16:
        words = raw.reshape(-1).astype("<u2", copy=False)
    else:
        raise ValueError(f"{tensor.name}: expected BF16 bytes, got {raw.dtype}")
    if words.size % ncols:
        raise ValueError(f"{tensor.name}: {words.size} values do not fit {ncols}-wide rows")
    # BF16 values are stored little-endian as the high 16 bits of FP32.
    return words.reshape(-1, ncols)


def error_for_chunk(words: np.ndarray, weights: np.ndarray | None) -> tuple[float, float]:
    x = (words.astype(np.uint32) << 16).view(np.float32)
    rows, ncols = x.shape
    blocks = x.reshape(rows, ncols // 32, 32)

    # Q4_0 reference quantization: signed absolute maximum determines d.
    max_index = np.argmax(np.abs(blocks), axis=2)
    max_value = np.take_along_axis(blocks, max_index[..., None], axis=2)[..., 0]
    d4 = -max_value / 8.0
    inv4 = np.divide(1.0, d4, out=np.zeros_like(d4), where=d4 != 0)
    q4 = np.trunc(blocks * inv4[..., None] + 8.5).clip(0, 15)
    dq4 = (q4 - 8.0) * d4[..., None]

    # Q8_0 reference quantization.
    d8 = np.max(np.abs(blocks), axis=2) / 127.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv8 = np.divide(1.0, d8, out=np.zeros_like(d8), where=d8 != 0)
    q8 = np.rint(blocks * inv8[..., None]).clip(-128, 127)
    dq8 = q8 * d8[..., None]

    if weights is None:
        w = 1.0
    elif weights.ndim == 1:
        w = weights.reshape(1, ncols // 32, 32)
    else:
        w = weights.reshape(rows, ncols // 32, 32)
    e4 = np.sum((blocks - dq4) ** 2 * w, dtype=np.float64)
    e8 = np.sum((blocks - dq8) ** 2 * w, dtype=np.float64)
    return float(e4), float(e8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-map", type=Path)
    base.add_argument("--base-log", type=Path)
    parser.add_argument("--imatrix", type=Path)
    parser.add_argument("--q8-fraction", type=float, default=25.0,
                        help="maximum fraction of final quantized bytes in Q8_0 (default: 25)")
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--max-tensor-mib", type=float, default=0.0,
                        help="only consider tensors up to this Q4 size; 0 means unlimited")
    parser.add_argument("--protect-low-bandwidth", action="store_true",
                        help="always retain small SSM/shared-expert/attention-KV tensors as Q8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.q8_fraction <= 100:
        parser.error("--q8-fraction must be between 0 and 100")

    base_types = load_base_types(args.base_map, args.base_log)
    for name, typ in list(base_types.items()):
        if typ == "q4_0" and re.search(r"\.ssm_(?:alpha|beta)\.weight$", name):
            base_types[name] = "f32"
    model_reader = GGUFReader(str(args.bf16))
    tensors = {tensor.name: tensor for tensor in model_reader.tensors}
    imatrix_tensors = {}
    if args.imatrix:
        imatrix_reader = GGUFReader(str(args.imatrix))
        imatrix_tensors = {tensor.name: tensor for tensor in imatrix_reader.tensors}

    base_quant_bytes = 0
    base_q8_bytes = 0
    candidates = []
    skipped = Counter()
    for name, typ in base_types.items():
        tensor = tensors.get(name)
        if tensor is None:
            skipped["missing"] += 1
            continue
        shape = [int(x) for x in tensor.shape]
        size = tensor_bytes(shape, typ)
        if size == 0:
            skipped["unsupported"] += 1
            continue
        if typ in BLOCK_BYTES:
            base_quant_bytes += size
            if typ == "q8_0":
                base_q8_bytes += size
        if typ != "q4_0":
            continue
        if args.max_tensor_mib > 0 and tensor_bytes(shape, "q4_0") / (1024 ** 2) > args.max_tensor_mib:
            skipped["size"] += 1
            continue
        # Recurrent state controls are too sensitive for a Q4/Q8 choice;
        # preserve them as F32 in the BF16-derived builder policy.
        if re.search(r"\.ssm_(?:alpha|beta)\.weight$", name):
            skipped["ssm_state"] += 1
            continue
        if shape[0] % 32:
            skipped["shape"] += 1
            continue
        ncols = shape[0]
        rows_per_expert = shape[1] if len(shape) > 1 else 1
        experts = shape[2] if len(shape) > 2 else 1
        words = as_bf16_rows(tensor, ncols)
        imatrix = imatrix_tensors.get(name + ".in_sum2")
        importance = None
        if imatrix is not None:
            values = np.asarray(imatrix.data, dtype=np.float32).reshape(-1)
            if values.size == ncols:
                importance = values
            elif values.size == ncols * experts:
                importance = values.reshape(experts, ncols)
        e4 = e8 = 0.0
        chunk_rows = max(1, args.chunk_rows)
        for start in range(0, words.shape[0], chunk_rows):
            end = min(words.shape[0], start + chunk_rows)
            chunk_weights = None
            if importance is not None:
                row_ids = np.arange(start, end)
                expert_ids = (row_ids // rows_per_expert) % experts
                chunk_weights = importance[expert_ids] if importance.ndim == 2 else importance
            a, b = error_for_chunk(words[start:end], chunk_weights)
            e4 += a
            e8 += b
        q4_bytes = tensor_bytes(shape, "q4_0")
        q8_bytes = tensor_bytes(shape, "q8_0")
        candidates.append({
            "name": name,
            "low_bandwidth": bool(re.search(r"(?:shexp\.weight$|\.ssm_out\.weight$|\.attn_[kv]\.weight$|\.attn_output\.weight$|\.nextn\.eh_proj\.weight$)", name)),
            "q4_bytes": q4_bytes,
            "q8_bytes": q8_bytes,
            "extra_bytes": q8_bytes - q4_bytes,
            "q4_error": e4,
            "q8_error": e8,
            "gain": max(0.0, e4 - e8),
            "score": max(0.0, e4 - e8) / max(1, q8_bytes - q4_bytes),
            "weighted": importance is not None,
        })

    candidates.sort(key=lambda item: (item["score"], item["gain"]), reverse=True)
    target = args.q8_fraction / 100.0
    selected = []
    selected_names = set()
    selected_q8_bytes = base_q8_bytes
    selected_extra_bytes = 0

    def add_item(item):
        nonlocal selected_q8_bytes, selected_extra_bytes
        selected.append(item)
        selected_names.add(item["name"])
        selected_q8_bytes += item["q8_bytes"]
        selected_extra_bytes += item["extra_bytes"]

    if args.protect_low_bandwidth:
        for item in candidates:
            if item["low_bandwidth"]:
                add_item(item)

    for item in candidates:
        if item["name"] in selected_names:
            continue
        proposed_q8 = selected_q8_bytes + item["q8_bytes"]
        proposed_extra = selected_extra_bytes + item["extra_bytes"]
        proposed_fraction = proposed_q8 / max(1, base_quant_bytes + proposed_extra)
        if proposed_fraction <= target or not selected and target > 0:
            add_item(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        output.write(f"# q8 fraction target: {args.q8_fraction:.3f}%\n")
        output.write("# name\tq4_error\tq8_error\tgain\textra_bytes\tscore\n")
        for item in selected:
            output.write(
                f"{item['name']}=Q8_0\t"
                f"{item['q4_error']:.9g}\t{item['q8_error']:.9g}\t"
                f"{item['gain']:.9g}\t{item['extra_bytes']}\t{item['score']:.9g}\n"
            )

    final_q8 = selected_q8_bytes
    final_quant = base_quant_bytes + selected_extra_bytes
    weighted = sum(1 for item in candidates if item["weighted"])
    print(f"base tensors: {len(base_types)}; Q4 candidates: {len(candidates)}")
    if args.max_tensor_mib > 0:
        print(f"maximum candidate tensor size: {args.max_tensor_mib:.3f} MiB")
    print(f"selected promotions: {len(selected)}")
    print(f"estimated Q8 fraction: {100 * final_q8 / max(1, final_quant):.3f}%")
    if args.protect_low_bandwidth:
        print(f"protected low-bandwidth promotions: {sum(1 for item in selected if item['low_bandwidth'])}")
    print(f"importance-weighted candidates: {weighted}")
    if skipped:
        print(f"skipped: {dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())