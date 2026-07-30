#!/usr/bin/env python3
"""Adaptive, deterministic MoE calibration sampler for Tessera."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


SCHEMA = "llama.tessera.moe-sampler.v1"


def import_gguf(path: str):
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from gguf import GGUFReader
    return GGUFReader


def load_records(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or any(record.get("schema") != "llama.tessera.calibration-corpus.v1" for record in records):
        raise ValueError("unsupported or empty Tessera calibration index")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("calibration sample IDs are not unique")
    return records


def stable_order(records: list[dict], seed: int) -> list[dict]:
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}\0{record['id']}".encode()
        ).digest(),
    )


def stratified_take(
    records: list[dict],
    selected_ids: set[str],
    count: int,
    seed: int,
    round_number: int,
) -> list[dict]:
    remaining = [record for record in records if record["id"] not in selected_ids]
    if count >= len(remaining):
        return stable_order(remaining, seed + round_number)
    categories: dict[str, list[dict]] = {}
    for record in remaining:
        categories.setdefault(record["category"], []).append(record)
    for category in categories:
        categories[category] = stable_order(
            categories[category],
            seed + round_number * 1009 + int.from_bytes(category.encode()[:4], "little"),
        )
    total = len(remaining)
    quotas = {
        category: count * len(items) / total
        for category, items in categories.items()
    }
    allocation = {
        category: min(len(items), int(math.floor(quotas[category])))
        for category, items in categories.items()
    }
    while sum(allocation.values()) < count:
        candidates = [
            category for category, items in categories.items()
            if allocation[category] < len(items)
        ]
        category = max(
            candidates,
            key=lambda name: (
                quotas[name] - allocation[name],
                -allocation[name],
                name,
            ),
        )
        allocation[category] += 1
    chosen = [
        record
        for category in sorted(categories)
        for record in categories[category][:allocation[category]]
    ]
    return stable_order(chosen, seed + round_number * 7919)


def observer_snapshot(path: Path, gguf_py: str) -> tuple[np.ndarray, dict[str, list[float]]]:
    reader = import_gguf(gguf_py)(str(path), "r")
    tensors = {
        tensor.name: np.asarray(tensor.data, dtype=np.float64).reshape(-1)
        for tensor in reader.tensors
        if tensor.tensor_type.name == "F32"
    }
    coverage = []
    profiles: dict[str, list[float]] = {}
    for name, counts in tensors.items():
        if not name.endswith(".counts") or counts.size <= 1:
            continue
        base = name[:-len(".counts")]
        sum2 = tensors.get(f"{base}.in_sum2")
        if sum2 is None or sum2.size % counts.size:
            continue
        rows = sum2.reshape(counts.size, -1)
        coverage.extend(counts.tolist())
        energy = rows.sum(axis=1) / np.maximum(counts, 1.0)
        scale = max(float(np.mean(energy)), 1e-20)
        profiles[base] = (energy / scale).astype(np.float32).tolist()
    if not coverage:
        raise ValueError(f"{path}: no routed-expert observer populations")
    return np.asarray(coverage, dtype=np.float64), profiles


def stability(previous: dict[str, list[float]], current: dict[str, list[float]]) -> tuple[float, float]:
    changes = []
    for name in sorted(set(previous) & set(current)):
        left = np.asarray(previous[name], dtype=np.float64)
        right = np.asarray(current[name], dtype=np.float64)
        if left.shape != right.shape:
            continue
        changes.extend(
            (np.abs(right - left) / np.maximum(np.abs(left), 1e-6)).tolist()
        )
    if not changes:
        return math.inf, math.inf
    values = np.asarray(changes)
    return float(np.median(values)), float(np.quantile(values, 0.95))


def materialize(records: list[dict], ids: list[str], path: Path) -> None:
    by_id = {record["id"]: record for record in records}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(by_id[record_id]["text"] for record_id in ids) + "\n",
        encoding="utf-8",
    )


def initialize(args) -> None:
    records = load_records(Path(args.index))
    selected = stratified_take(records, set(), args.initial_samples, args.seed, 0)
    state = {
        "schema": SCHEMA,
        "index": str(Path(args.index).resolve()),
        "index_sha256": hashlib.sha256(Path(args.index).read_bytes()).hexdigest(),
        "seed": args.seed,
        "initial_samples": args.initial_samples,
        "step_samples": args.step_samples,
        "max_samples": min(args.max_samples, len(records)),
        "minimum_expert_count": args.minimum_expert_count,
        "coverage_percentile": args.coverage_percentile,
        "stability_p95": args.stability_p95,
        "stable_rounds_required": args.stable_rounds,
        "round": 0,
        "selected_ids": [record["id"] for record in selected],
        "pending_ids": [record["id"] for record in selected],
        "stable_rounds": 0,
        "complete": False,
        "stop_reason": None,
        "history": [],
        "observer_profile": {},
    }
    Path(args.state).parent.mkdir(parents=True, exist_ok=True)
    Path(args.state).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    materialize(records, state["pending_ids"], Path(args.batch_output))


def advance(args) -> None:
    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema") != SCHEMA:
        raise ValueError("unsupported MoE sampler state")
    records = load_records(Path(state["index"]))
    if hashlib.sha256(Path(state["index"]).read_bytes()).hexdigest() != state["index_sha256"]:
        raise ValueError("calibration index changed after sampler initialization")
    coverage, profile = observer_snapshot(Path(args.imatrix), args.gguf_py)
    median_change, p95_change = stability(state["observer_profile"], profile)
    coverage_value = float(np.quantile(
        coverage, state["coverage_percentile"] / 100.0
    ))
    coverage_ok = coverage_value >= state["minimum_expert_count"]
    stability_ok = p95_change <= state["stability_p95"]
    state["stable_rounds"] = state["stable_rounds"] + 1 if stability_ok else 0
    state["history"].append({
        "round": state["round"],
        "imatrix": str(Path(args.imatrix).resolve()),
        "samples": len(state["selected_ids"]),
        "coverage_percentile_value": coverage_value,
        "minimum_expert_count": float(np.min(coverage)),
        "median_observer_change": median_change,
        "p95_observer_change": p95_change,
        "coverage_ok": coverage_ok,
        "stability_ok": stability_ok,
    })
    state["observer_profile"] = profile
    if coverage_ok and state["stable_rounds"] >= state["stable_rounds_required"]:
        state["complete"] = True
        state["stop_reason"] = "coverage-and-observer-stability"
        state["pending_ids"] = []
    elif len(state["selected_ids"]) >= state["max_samples"]:
        state["complete"] = True
        state["stop_reason"] = "maximum-samples"
        state["pending_ids"] = []
    else:
        count = min(
            state["step_samples"],
            state["max_samples"] - len(state["selected_ids"]),
        )
        additions = stratified_take(
            records,
            set(state["selected_ids"]),
            count,
            state["seed"],
            state["round"] + 1,
        )
        state["pending_ids"] = [record["id"] for record in additions]
        state["selected_ids"].extend(state["pending_ids"])
        state["round"] += 1
        materialize(records, state["pending_ids"], Path(args.batch_output))
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Tessera MoE calibration sampler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--index", required=True)
    init.add_argument("--state", required=True)
    init.add_argument("--batch-output", required=True)
    init.add_argument("--seed", type=int, default=640)
    init.add_argument("--initial-samples", type=int, default=128)
    init.add_argument("--step-samples", type=int, default=128)
    init.add_argument("--max-samples", type=int, default=1024)
    init.add_argument("--minimum-expert-count", type=int, default=16)
    init.add_argument("--coverage-percentile", type=float, default=5.0)
    init.add_argument("--stability-p95", type=float, default=0.02)
    init.add_argument("--stable-rounds", type=int, default=2)
    init.set_defaults(func=initialize)
    observe = subparsers.add_parser("advance")
    observe.add_argument("--state", required=True)
    observe.add_argument("--imatrix", required=True)
    observe.add_argument("--batch-output", required=True)
    observe.add_argument(
        "--gguf-py",
        default="/Users/user/Developer/GitHub/llama.cpp/gguf-py",
    )
    observe.set_defaults(func=advance)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
