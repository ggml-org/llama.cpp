#!/usr/bin/env python3
"""Pair sampled Hugging Face weights with llama-imatrix telemetry."""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


PROJECTIONS = {
    "self_attn.q_proj": ("attn_q", "attention"),
    "self_attn.k_proj": ("attn_k", "attention"),
    "self_attn.v_proj": ("attn_v", "attention"),
    "self_attn.o_proj": ("attn_output", "attention"),
    "mlp.gate_proj": ("ffn_gate", "ffn"),
    "mlp.up_proj": ("ffn_up", "ffn"),
    "mlp.down_proj": ("ffn_down", "ffn"),
}
EXPERT_PROJECTIONS = {
    "gate_proj": "ffn_gate_exps",
    "up_proj": "ffn_up_exps",
    "down_proj": "ffn_down_exps",
}


@dataclasses.dataclass(frozen=True)
class TensorSource:
    shard: Path
    name: str
    observer_name: str
    family: str
    shape: tuple[int, ...]
    expert: int | None = None


def import_gguf(path: str):
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from gguf import GGUFReader
    return GGUFReader


def hf_to_observer(name: str) -> tuple[str, str, int | None] | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)\.weight$", name)
    if not match:
        if name.endswith("embed_tokens.weight"):
            return "token_embd.weight", "output_embedding", None
        return None
    layer = int(match.group(1))
    suffix = match.group(2)
    expert_match = re.fullmatch(
        r"(?:mlp\.)?experts\.(\d+)\.(gate_proj|up_proj|down_proj)",
        suffix,
    )
    if expert_match:
        expert = int(expert_match.group(1))
        return (
            f"blk.{layer}.{EXPERT_PROJECTIONS[expert_match.group(2)]}.weight",
            "routed_expert",
            expert,
        )
    router_match = re.fullmatch(r"(?:mlp\.)?(?:gate|router)", suffix)
    if router_match:
        return f"blk.{layer}.ffn_gate_inp.weight", "router", None
    shared_match = re.fullmatch(
        r"(?:mlp\.)?shared_expert\.(gate_proj|up_proj|down_proj)",
        suffix,
    )
    if shared_match:
        projection_name = {
            "gate_proj": "ffn_gate_shexp",
            "up_proj": "ffn_up_shexp",
            "down_proj": "ffn_down_shexp",
        }[shared_match.group(1)]
        return f"blk.{layer}.{projection_name}.weight", "shared_expert", None
    projection = PROJECTIONS.get(suffix)
    if projection is None:
        return None
    stem, family = projection
    return f"blk.{layer}.{stem}.weight", family, None


def load_telemetry(path: str, gguf_py: str) -> dict[str, dict[str, np.ndarray]]:
    reader = import_gguf(gguf_py)(path, "r")
    grouped: dict[str, dict[str, np.ndarray]] = {}
    suffixes = ("in_sum2", "in_sumabs", "in_sum4", "in_maxabs", "counts")
    for tensor in reader.tensors:
        for suffix in suffixes:
            marker = f".{suffix}"
            if tensor.name.endswith(marker):
                base = tensor.name[:-len(marker)]
                grouped.setdefault(base, {})[suffix] = np.asarray(tensor.data, dtype=np.float32).reshape(-1)
                break
    return grouped


def layer_index(name: str) -> int:
    match = re.match(r"blk\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def stratified_sources(sources: list[TensorSource], limit: int) -> list[TensorSource]:
    if len(sources) <= limit:
        return sources
    ordered = sorted(sources, key=lambda source: (layer_index(source.observer_name), source.observer_name))
    selected: list[TensorSource] = []
    used: set[tuple[Path, str]] = set()
    positions = np.linspace(0, len(ordered) - 1, limit, dtype=np.int64)
    for position in positions:
        source = ordered[int(position)]
        key = (source.shard, source.name)
        if key not in used:
            selected.append(source)
            used.add(key)
    if len(selected) < limit:
        for source in ordered:
            key = (source.shard, source.name)
            if key not in used:
                selected.append(source)
                used.add(key)
            if len(selected) == limit:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sampled layer inputs for AWQ evolution")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gguf-py", default="/Users/user/Developer/GitHub/llama.cpp/gguf-py")
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-per-family", type=int, default=24)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    telemetry = load_telemetry(args.imatrix, args.gguf_py)
    candidates: dict[str, list[TensorSource]] = {}
    misses = 0
    for shard in sorted(Path(args.model_dir).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as source:
            for name in source.keys():
                mapped = hf_to_observer(name)
                if mapped is None:
                    continue
                observer_name, family, expert = mapped
                stats = telemetry.get(observer_name)
                if not stats or "in_sum2" not in stats or "counts" not in stats:
                    misses += 1
                    continue
                shape = tuple(source.get_slice(name).get_shape())
                if len(shape) != 2:
                    continue
                candidates.setdefault(family, []).append(
                    TensorSource(shard, name, observer_name, family, shape, expert)
                )
    selected = {
        (source.shard, source.name): source
        for family_sources in candidates.values()
        for source in stratified_sources(family_sources, args.max_per_family)
    }
    family_counts: dict[str, int] = {}
    written = 0
    for shard in sorted(Path(args.model_dir).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as source:
            for name in source.keys():
                descriptor = selected.get((shard, name))
                if descriptor is None:
                    continue
                observer_name = descriptor.observer_name
                family = descriptor.family
                stats = telemetry.get(observer_name)
                assert stats is not None
                shape = descriptor.shape
                row_count = min(shape[0], args.max_rows)
                if row_count == shape[0]:
                    weight = source.get_tensor(name).float().numpy()
                else:
                    # Evenly spaced rows cover the whole output range without
                    # materializing a multi-gigabyte tensor.
                    indices = np.linspace(0, shape[0] - 1, row_count, dtype=np.int64)
                    full = source.get_slice(name)
                    weight = np.stack([full[index].float().numpy() for index in indices])
                channels = weight.shape[1]
                counts = stats["counts"]
                experts = max(1, counts.size)

                def pool(key: str, fallback: np.ndarray) -> np.ndarray:
                    value = stats.get(key)
                    if value is None:
                        return fallback
                    rows = value.reshape(experts, -1)
                    if rows.shape[1] != channels:
                        raise ValueError(
                            f"{observer_name}.{key}: width {rows.shape[1]} != {channels}"
                        )
                    if descriptor.expert is not None:
                        if descriptor.expert >= experts:
                            raise ValueError(
                                f"{observer_name}: expert {descriptor.expert} "
                                f"is outside telemetry bank of {experts}"
                            )
                        return rows[descriptor.expert]
                    if key == "in_maxabs":
                        return np.max(rows, axis=0)
                    return np.sum(rows, axis=0)

                sum2 = pool("in_sum2", np.zeros(channels, dtype=np.float32))
                sum4 = pool("in_sum4", np.square(sum2, dtype=np.float32))
                maxabs = pool("in_maxabs", np.sqrt(np.maximum(sum2, 0.0)))
                bundle_name = (
                    f"{observer_name}.expert-{descriptor.expert}"
                    if descriptor.expert is not None else observer_name
                )
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", bundle_name)
                np.savez_compressed(
                    output / f"{safe_name}.npz",
                    name=np.asarray(bundle_name),
                    family=np.asarray(family),
                    expert=np.asarray(
                        descriptor.expert if descriptor.expert is not None else -1
                    ),
                    weight=weight,
                    in_sum2=sum2,
                    in_sum4=sum4,
                    in_maxabs=maxabs,
                    counts=np.asarray(
                        counts[descriptor.expert]
                        if descriptor.expert is not None else counts
                    ),
                )
                family_counts[family] = family_counts.get(family, 0) + 1
                written += 1
    if written == 0:
        raise RuntimeError(
            "no bundles were written; the imatrix tensor names do not match this checkpoint"
        )
    print(
        f"wrote {written} layer bundles to {output}; "
        f"family_counts={family_counts}, telemetry_misses={misses}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
