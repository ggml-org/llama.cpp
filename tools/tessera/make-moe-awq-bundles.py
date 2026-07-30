#!/usr/bin/env python3
"""Build per-expert AWQ bundles from fused MoE safetensor banks."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


PATTERN = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|gate_proj|up_proj|down_proj)(?:\.weight)?$"
)
GGUF_NAMES = {
    "gate_up_proj": "ffn_gate_up_exps",
    "gate_proj": "ffn_gate_exps",
    "up_proj": "ffn_up_exps",
    "down_proj": "ffn_down_exps",
}


def import_gguf(path: str):
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from gguf import GGUFReader
    return GGUFReader


def load_observers(path: str, gguf_py: str) -> dict[str, dict[str, np.ndarray]]:
    reader = import_gguf(gguf_py)(path, "r")
    suffixes = ("in_sum2", "in_sum4", "in_maxabs", "counts")
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for tensor in reader.tensors:
        for suffix in suffixes:
            marker = f".{suffix}"
            if tensor.name.endswith(marker):
                grouped.setdefault(tensor.name[:-len(marker)], {})[suffix] = (
                    np.asarray(tensor.data, dtype=np.float32).reshape(-1)
                )
                break
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build routed-expert AWQ bundles without loading a full bank"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gguf-py",
        default="/Users/user/Developer/GitHub/llama.cpp/gguf-py",
    )
    parser.add_argument("--max-rows", type=int, default=256)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    telemetry = load_observers(args.imatrix, args.gguf_py)
    written = 0
    for shard in sorted(Path(args.model_dir).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as source:
            for tensor_name in source.keys():
                match = PATTERN.search(tensor_name)
                if match is None:
                    continue
                layer = int(match.group(1))
                projection = match.group(2)
                observer_name = (
                    f"blk.{layer}.{GGUF_NAMES[projection]}.weight"
                )
                stats = telemetry.get(observer_name)
                if stats is None and projection == "gate_up_proj":
                    split = [
                        telemetry.get(
                            observer_name.replace(
                                "ffn_gate_up_exps", replacement
                            )
                        )
                        for replacement in ("ffn_gate_exps", "ffn_up_exps")
                    ]
                    stats = next((item for item in split if item is not None), None)
                if not stats or "in_sum2" not in stats or "counts" not in stats:
                    continue
                shape = tuple(source.get_slice(tensor_name).get_shape())
                if len(shape) != 3:
                    continue
                experts, rows, channels = shape
                counts = stats["counts"].reshape(-1)
                if counts.size != experts:
                    raise ValueError(
                        f"{observer_name}: {counts.size} observer experts "
                        f"do not match weight bank of {experts}"
                    )
                selected_rows = np.linspace(
                    0, rows - 1, min(rows, args.max_rows), dtype=np.int64
                )
                sliced = source.get_slice(tensor_name)
                for expert in range(experts):
                    weight = np.stack([
                        sliced[expert, int(row)].float().numpy()
                        for row in selected_rows
                    ])

                    def expert_stat(key: str, fallback: np.ndarray) -> np.ndarray:
                        value = stats.get(key)
                        if value is None:
                            return fallback
                        return value.reshape(experts, channels)[expert]

                    sum2 = expert_stat(
                        "in_sum2", np.zeros(channels, dtype=np.float32)
                    )
                    sum4 = expert_stat(
                        "in_sum4", np.square(sum2, dtype=np.float32)
                    )
                    maxabs = expert_stat(
                        "in_maxabs", np.sqrt(np.maximum(sum2, 0.0))
                    )
                    name = f"{observer_name}.expert-{expert}"
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
                    np.savez_compressed(
                        output / f"{safe_name}.npz",
                        name=np.asarray(name),
                        family=np.asarray("routed_expert"),
                        layer=np.asarray(layer),
                        expert=np.asarray(expert),
                        weight=weight,
                        in_sum2=sum2,
                        in_sum4=sum4,
                        in_maxabs=maxabs,
                        counts=np.asarray(counts[expert]),
                    )
                    written += 1
    if written == 0:
        raise RuntimeError("no routed-expert bundles matched the model and imatrix")
    print(f"wrote {written} routed-expert bundles to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
