#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Any

from gguf import GGUFReader


def parse_layer(name: str) -> int | None:
    m = re.search(r"(?:^|\.)blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def part_name(name: str) -> str:
    m = re.search(r"blk\.\d+\.(.*)", name)
    return m.group(1) if m else name


def find_packed_expert_axis(shape: tuple[int, ...], n_expert: int) -> int | None:
    # GGML dims are ne[0], ne[1], ... ; last dim is usually outermost.
    for axis in reversed(range(len(shape))):
        if shape[axis] == n_expert:
            return axis
    return None


def tensor_abs_offset(reader: Any, tensor: Any) -> int:
    # In llama.cpp gguf-py this is usually already the absolute data offset.
    # Keep this helper isolated because GGUFReader internals may differ by version.
    if hasattr(tensor, "data_offset"):
        return int(tensor.data_offset)

    if hasattr(tensor, "offset"):
        off = int(tensor.offset)
        # Some readers store offset relative to data section.
        if hasattr(reader, "data_offset"):
            return int(reader.data_offset) + off
        return off

    raise AttributeError(f"cannot find tensor offset field for {tensor.name}")


def tensor_nbytes(tensor: Any) -> int:
    if hasattr(tensor, "n_bytes"):
        return int(tensor.n_bytes)
    if hasattr(tensor, "nbytes"):
        return int(tensor.nbytes)
    if hasattr(tensor, "data") and hasattr(tensor.data, "nbytes"):
        return int(tensor.data.nbytes)

    raise AttributeError(f"cannot find tensor byte size for {tensor.name}")


def tensor_type_name(tensor: Any) -> str:
    t = getattr(tensor, "tensor_type", None)
    if t is None:
        t = getattr(tensor, "type", None)
    return str(t)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("gguf")
    p.add_argument("--out", required=True)
    p.add_argument("--n-expert", type=int, default=128)
    p.add_argument("--tensor-regex", default=r"blk\.\d+\..*exps.*")
    p.add_argument("--print-summary", action="store_true")
    args = p.parse_args()

    reader = GGUFReader(args.gguf)
    rx = re.compile(args.tensor_regex)

    rows = []

    for t in reader.tensors:
        name = str(t.name)
        if not rx.search(name):
            continue

        layer = parse_layer(name)
        if layer is None:
            continue

        shape = tuple(int(x) for x in t.shape)
        offset = tensor_abs_offset(reader, t)
        nbytes = tensor_nbytes(t)
        ggml_type = tensor_type_name(t)
        part = part_name(name)

        expert_axis = find_packed_expert_axis(shape, args.n_expert)

        if expert_axis == len(shape) - 1:
            if nbytes % args.n_expert != 0:
                raise RuntimeError(f"{name}: nbytes not divisible by n_expert")

            per_expert = nbytes // args.n_expert
            for expert in range(args.n_expert):
                rows.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "part": part,
                        "tensor_name": name,
                        "offset": offset + expert * per_expert,
                        "nbytes": per_expert,
                        "ggml_type": ggml_type,
                        "shape": "x".join(map(str, shape)),
                        "mode": f"packed_axis_{expert_axis}",
                    }
                )
        else:
            # Fallback: cannot represent one expert as one contiguous range.
            rows.append(
                {
                    "layer": layer,
                    "expert": -1,
                    "part": part,
                    "tensor_name": name,
                    "offset": offset,
                    "nbytes": nbytes,
                    "ggml_type": ggml_type,
                    "shape": "x".join(map(str, shape)),
                    "mode": (
                        "whole_tensor_no_expert_axis"
                        if expert_axis is None
                        else f"whole_tensor_strided_expert_axis_{expert_axis}"
                    ),
                }
            )

    rows.sort(key=lambda r: (r["layer"], r["expert"], r["offset"], r["part"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "expert",
                "part",
                "tensor_name",
                "offset",
                "nbytes",
                "ggml_type",
                "shape",
                "mode",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    if args.print_summary:
        print(f"rows: {len(rows)}")
        print(f"GiB: {sum(r['nbytes'] for r in rows) / 1024**3:.3f}")
        print(f"modes: {sorted(set(r['mode'] for r in rows))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
