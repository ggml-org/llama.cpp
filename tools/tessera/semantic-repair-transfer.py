#!/usr/bin/env python3
"""Validate transfer of a family-balanced Tessera repair prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SCHEMA = "llama.tessera.semantic-repair-transfer.v1"
REPLAY_SCHEMA = "llama.tessera.replay-corpus.v1"
ADDITIVE_STATS = ("in_sum2", "in_sumabs", "in_sum4", "counts")


def import_gguf(path: str):
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from gguf import GGUFReader

    return GGUFReader


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_observers(path: Path, gguf_py: str) -> dict[str, dict[str, np.ndarray]]:
    reader = import_gguf(gguf_py)(str(path), "r")
    grouped: dict[str, dict[str, np.ndarray]] = {}
    suffixes = ADDITIVE_STATS + ("in_maxabs",)
    for tensor in reader.tensors:
        for suffix in suffixes:
            marker = f".{suffix}"
            if tensor.name.endswith(marker):
                name = tensor.name[: -len(marker)]
                grouped.setdefault(name, {})[suffix] = np.asarray(
                    tensor.data, dtype=np.float64
                ).reshape(-1)
                break
    return grouped


def normalized_windows(
    prototype: dict[str, np.ndarray],
    cumulative: dict[str, np.ndarray],
    key: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    first = prototype.get(key)
    total = cumulative.get(key)
    first_counts = prototype.get("counts")
    total_counts = cumulative.get("counts")
    if any(value is None for value in (first, total, first_counts, total_counts)):
        return None
    if first.shape != total.shape or first_counts.shape != total_counts.shape:
        return None
    experts = first_counts.size
    if experts < 1 or first.size % experts:
        return None
    channels = first.size // experts
    first_rows = first.reshape(experts, channels)
    second_rows = (total - first).reshape(experts, channels)
    second_counts = total_counts - first_counts
    valid = (first_counts > 0) & (second_counts > 0)
    if not np.any(valid):
        return None
    first_norm = first_rows[valid] / first_counts[valid, None]
    second_norm = second_rows[valid] / second_counts[valid, None]
    return first_norm.reshape(-1), second_norm.reshape(-1)


def top_overlap(first: np.ndarray, second: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(first.size * fraction)))
    first_top = set(np.argpartition(first, -count)[-count:].tolist())
    second_top = set(np.argpartition(second, -count)[-count:].tolist())
    return len(first_top & second_top) / max(1, len(first_top | second_top))


def evaluate_transfer(
    prototype: dict[str, dict[str, np.ndarray]],
    cumulative: dict[str, dict[str, np.ndarray]],
    moment_tolerance: float,
    rank_overlap_min: float,
    tensor_pass_fraction: float,
    top_fraction: float,
) -> dict:
    records = []
    for name in sorted(prototype.keys() & cumulative.keys()):
        second = normalized_windows(prototype[name], cumulative[name], "in_sum2")
        if second is None:
            continue
        first_values, next_values = second
        safe_first = np.maximum(first_values, 1e-12)
        safe_next = np.maximum(next_values, 1e-12)
        log_delta = np.abs(np.log(safe_next / safe_first))
        moment_p90 = float(np.quantile(log_delta, 0.90))
        overlap = top_overlap(first_values, next_values, top_fraction)
        passed = moment_p90 <= moment_tolerance and overlap >= rank_overlap_min
        records.append(
            {
                "tensor": name,
                "moment_log_delta_p90": moment_p90,
                "sensitive_rank_overlap": overlap,
                "channels": int(first_values.size),
                "passed": passed,
            }
        )
    passed_tensors = sum(record["passed"] for record in records)
    observed_fraction = passed_tensors / max(1, len(records))
    return {
        "transferable": bool(records) and observed_fraction >= tensor_pass_fraction,
        "evaluated_tensors": len(records),
        "passed_tensors": passed_tensors,
        "tensor_pass_fraction": observed_fraction,
        "thresholds": {
            "moment_log_delta_p90_max": moment_tolerance,
            "sensitive_rank_overlap_min": rank_overlap_min,
            "required_tensor_pass_fraction": tensor_pass_fraction,
            "sensitive_top_fraction": top_fraction,
        },
        "worst_tensors": sorted(
            records,
            key=lambda record: (
                not record["passed"],
                record["moment_log_delta_p90"],
                -record["sensitive_rank_overlap"],
            ),
            reverse=True,
        )[:32],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate semantic-family repair transfer between cumulative checkpoints"
    )
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--cumulative", required=True)
    parser.add_argument("--replay-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gguf-py", default="/Users/user/Developer/GitHub/llama.cpp/gguf-py"
    )
    parser.add_argument("--minimum-chunks", type=int, default=128)
    parser.add_argument("--prototype-chunks", type=int, default=64)
    parser.add_argument("--moment-tolerance", type=float, default=0.15)
    parser.add_argument("--rank-overlap-min", type=float, default=0.70)
    parser.add_argument("--tensor-pass-fraction", type=float, default=0.90)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    args = parser.parse_args()

    prototype_path = Path(args.prototype)
    cumulative_path = Path(args.cumulative)
    replay_path = Path(args.replay_receipt)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if replay.get("schema") != REPLAY_SCHEMA:
        raise ValueError("unsupported replay receipt schema")
    if replay.get("strategy") != "semantic-family":
        raise ValueError("repair transfer requires a semantic-family replay")
    recommended = int(replay.get("recommended_convergence_min_chunks", 0))
    if args.minimum_chunks < recommended:
        raise ValueError(
            f"minimum chunks {args.minimum_chunks} is below replay recommendation {recommended}"
        )
    if not 0.0 < args.prototype_chunks < args.minimum_chunks:
        raise ValueError("prototype chunks must be between zero and minimum chunks")

    result = evaluate_transfer(
        load_observers(prototype_path, args.gguf_py),
        load_observers(cumulative_path, args.gguf_py),
        args.moment_tolerance,
        args.rank_overlap_min,
        args.tensor_pass_fraction,
        args.top_fraction,
    )
    receipt = {
        "schema": SCHEMA,
        "prototype_chunks": args.prototype_chunks,
        "cumulative_chunks": args.minimum_chunks,
        "covered_semantic_families": replay.get("covered_semantic_families", 0),
        "prototype_sha256": digest(prototype_path),
        "cumulative_sha256": digest(cumulative_path),
        "replay_receipt_sha256": digest(replay_path),
        **result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        f"semantic repair transfer: transferable={receipt['transferable']}, "
        f"passed={receipt['passed_tensors']}/{receipt['evaluated_tensors']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
