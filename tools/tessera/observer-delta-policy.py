#!/usr/bin/env python3
"""Build a targeted Tessera repair policy from reference and quantized observers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SCHEMA = "llama.speculative.calibration-policy.v1"
REPAIR_SCHEMA = "llama.tessera.repair-policy.v1"


def import_gguf(path: str):
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from gguf import GGUFReader
    return GGUFReader


def load_observers(path: Path, gguf_py: str) -> dict[str, dict[str, np.ndarray]]:
    reader = import_gguf(gguf_py)(str(path), "r")
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for tensor in reader.tensors:
        for suffix in ("in_sum2", "in_sum4", "in_maxabs", "counts"):
            marker = f".{suffix}"
            if tensor.name.endswith(marker):
                name = tensor.name[:-len(marker)]
                grouped.setdefault(name, {})[suffix] = np.asarray(
                    tensor.data, dtype=np.float64
                ).reshape(-1)
                break
    return grouped


def normalized(stats: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    values = stats.get(key)
    counts = stats.get("counts")
    if values is None or counts is None:
        return None
    count = max(float(np.sum(counts)), 1.0)
    return values / count


def observer_delta(
    reference: dict[str, dict[str, np.ndarray]],
    candidate: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    records = []
    for name in sorted(reference.keys() & candidate.keys()):
        ref_second = normalized(reference[name], "in_sum2")
        got_second = normalized(candidate[name], "in_sum2")
        if (
            ref_second is None
            or got_second is None
            or ref_second.shape != got_second.shape
        ):
            continue
        safe_ref = np.maximum(ref_second, 1e-12)
        safe_got = np.maximum(got_second, 1e-12)
        log_delta = np.abs(np.log(safe_got / safe_ref))
        moment_delta = float(np.quantile(log_delta, 0.9))
        ref_tail = reference[name].get("in_maxabs")
        got_tail = candidate[name].get("in_maxabs")
        tail_delta = 0.0
        if (
            ref_tail is not None
            and got_tail is not None
            and ref_tail.shape == got_tail.shape
        ):
            tail_delta = float(np.quantile(
                np.abs(np.log(
                    np.maximum(got_tail, 1e-12)
                    / np.maximum(ref_tail, 1e-12)
                )),
                0.9,
            ))
        score = moment_delta + 0.25 * tail_delta
        records.append({
            "tensor": name,
            "score": score,
            "moment_delta": moment_delta,
            "tail_delta": tail_delta,
            "channels": int(ref_second.size),
        })
    return sorted(records, key=lambda record: record["score"], reverse=True)


def matching_entry(policy: dict, tensor: str) -> dict:
    for entry in policy.get("tensor_families", {}).values():
        if any(fragment in tensor for fragment in entry.get("match", [])):
            return entry
    return {}


def build_repair_policy(
    base: dict,
    records: list[dict],
    top_fraction: float,
    max_overrides: int,
    residual_boost: float,
    provenance: dict,
) -> dict:
    if base.get("schema") != SCHEMA:
        raise ValueError("base policy has an unsupported schema")
    count = min(
        max_overrides,
        max(1, int(np.ceil(len(records) * top_fraction))) if records else 0,
    )
    overrides = {}
    for record in records[:count]:
        tensor = record["tensor"]
        inherited = matching_entry(base, tensor)
        fraction = min(
            0.05,
            max(0.0001, float(inherited.get("outlier_fraction", 0.005)))
            * residual_boost,
        )
        overrides[f"repair:{tensor}"] = {
            **inherited,
            "match": [tensor],
            "exact": False,
            "outlier_fraction": fraction,
            "repair_score": record["score"],
            "repair_moment_delta": record["moment_delta"],
            "repair_tail_delta": record["tail_delta"],
        }
    return {
        **base,
        "schema": SCHEMA,
        "tensor_families": {
            **overrides,
            **base.get("tensor_families", {}),
        },
        "repair": {
            "schema": REPAIR_SCHEMA,
            **provenance,
            "evaluated_tensors": len(records),
            "selected_overrides": count,
            "top_fraction": top_fraction,
            "max_overrides": max_overrides,
            "residual_boost": residual_boost,
            "ranked_deltas": records,
        },
    }


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a targeted Tessera observer-delta repair policy")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gguf-py", default="/Users/user/Developer/GitHub/llama.cpp/gguf-py")
    parser.add_argument("--top-fraction", type=float, default=0.15)
    parser.add_argument("--max-overrides", type=int, default=16)
    parser.add_argument("--residual-boost", type=float, default=2.0)
    args = parser.parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top fraction must be in (0, 1]")
    if args.max_overrides < 1 or args.residual_boost < 1.0:
        raise ValueError("maximum overrides and residual boost must be positive")
    reference_path = Path(args.reference)
    candidate_path = Path(args.candidate)
    base_path = Path(args.base_policy)
    records = observer_delta(
        load_observers(reference_path, args.gguf_py),
        load_observers(candidate_path, args.gguf_py),
    )
    policy = build_repair_policy(
        json.loads(base_path.read_text(encoding="utf-8")),
        records,
        args.top_fraction,
        args.max_overrides,
        args.residual_boost,
        {
            "reference_imatrix_sha256": digest(reference_path),
            "candidate_imatrix_sha256": digest(candidate_path),
            "base_policy_sha256": digest(base_path),
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output} with {policy['repair']['selected_overrides']} repair overrides",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
