#!/usr/bin/env python3
"""Quantization-aware, pre-write Tessera shadow calibration.

The imatrix observer reports *where* activations are important.  This tool
adds the missing question before a GGUF is produced: how much error does the
current provisional Tessera reconstruction actually create at those sampled
layer outputs?  It evaluates every supplied layer bundle, preserves the base
policy, and emits a bounded set of depth- and family-stratified overrides.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


SCHEMA = "llama.tessera.shadow-calibration.v1"
POLICY_SCHEMA = "llama.speculative.calibration-policy.v1"


def load_evolver(path: Path):
    spec = importlib.util.spec_from_file_location("tessera_awq_evolve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load AWQ evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_for(layer, policy: dict, awq):
    """Resolve the most-specific policy entry without depending on dict order."""
    entries = policy.get("tensor_families", {})
    selected = None
    selected_rank = (-1, -1)
    for key, entry in entries.items():
        matches = entry.get("match", [])
        if not isinstance(matches, list) or not matches:
            continue
        exact = bool(entry.get("exact", False))
        if exact:
            matched = layer.name in matches
        else:
            matched = any(fragment in layer.name for fragment in matches)
        if not matched:
            continue
        rank = (int(exact), max(len(fragment) for fragment in matches))
        if rank > selected_rank:
            selected, selected_rank = entry, rank
    selected = selected or {}
    return awq.Candidate(
        float(selected.get("awq_alpha", 0.0)),
        float(selected.get("awq_clip", 1.0)),
        float(selected.get("outlier_fraction", 0.005)),
        float(selected.get("moment_mix", 0.0)),
        float(selected.get("tail_guard", 0.0)),
    ).clipped()


def score_layers(layers, policy: dict, awq) -> list[dict]:
    records = []
    for layer in layers:
        candidate = candidate_for(layer, policy, awq)
        score = awq._evaluate_layer(candidate, layer)
        # Held-out output error is the primary shadow signal. The tail term
        # prevents a calm mean error from hiding rare activation failures.
        shadow_error = score.heldout_error + 0.05 * score.tail_error
        records.append({
            "tensor": layer.name,
            "family": layer.family,
            "candidate": dataclasses.asdict(candidate),
            "train_error": score.train_error,
            "heldout_error": score.heldout_error,
            "tail_error": score.tail_error,
            "shadow_error": shadow_error,
            "has_heldout": layer.heldout_activations is not None,
            "sample_count": float(getattr(layer, "sample_count", 0.0)),
        })
    # Sparse-expert evidence has higher uncertainty.  Give a bounded prior
    # boost to under-sampled bundles instead of treating their quiet observed
    # error as proof that they need no residual budget.
    for family in {record["family"] for record in records}:
        family_records = [record for record in records if record["family"] == family]
        reference = float(np.median([
            max(record["sample_count"], 1.0) for record in family_records
        ]))
        for record in family_records:
            uncertainty = min(4.0, math.sqrt(reference / max(record["sample_count"], 1.0)))
            record["coverage_uncertainty"] = uncertainty
            record["shadow_error"] *= 1.0 + 0.10 * max(0.0, uncertainty - 1.0)
    return records


def select_stratified(records: list[dict], top_fraction: float, max_overrides: int) -> list[dict]:
    """Keep difficult layers across depth and architecture family, not one block."""
    if not records or max_overrides <= 0:
        return []
    selected: list[dict] = []
    for family in sorted({record["family"] for record in records}):
        family_records = sorted(
            (record for record in records if record["family"] == family),
            key=lambda record: record["tensor"],
        )
        count = max(1, math.ceil(len(family_records) * top_fraction))
        # Split model depth/name order into three bands, preserving an outlier
        # from early, middle, and late portions of each tensor family.
        bands = np.array_split(np.asarray(family_records, dtype=object), min(3, len(family_records)))
        family_selected = []
        for band in bands:
            if len(band):
                family_selected.append(max(band.tolist(), key=lambda record: record["shadow_error"]))
        for record in sorted(family_records, key=lambda item: item["shadow_error"], reverse=True):
            if len(family_selected) >= count:
                break
            if record not in family_selected:
                family_selected.append(record)
        selected.extend(family_selected)
    return sorted(selected, key=lambda record: record["shadow_error"], reverse=True)[:max_overrides]


def refine_policy(policy: dict, records: list[dict], selected: list[dict], boost: float) -> dict:
    refined = dict(policy)
    families = dict(policy.get("tensor_families", {}))
    baseline = float(np.median([record["shadow_error"] for record in records])) if records else 0.0
    selected_records = []
    for index, record in enumerate(selected):
        candidate = dict(record["candidate"])
        pressure = record["shadow_error"] / max(baseline, 1e-12)
        candidate["outlier_fraction"] = float(np.clip(
            candidate["outlier_fraction"] * (1.0 + boost * max(0.0, pressure - 1.0)),
            0.0001, 0.05,
        ))
        families[f"shadow_{index:03d}_{record['tensor'].replace('.', '_')}"] = {
            "match": [record["tensor"]],
            # A full tensor-name match is more specific than a family match,
            # but remains quantized. `exact=True` is reserved by Tessera for
            # intentional source-value retention.
            "exact": False,
            **candidate,
            "shadow_error": record["shadow_error"],
        }
        selected_records.append({**record, "refined_candidate": candidate})
    moe_layers: dict[str, dict] = dict(
        policy.get("moe_residual_allocation", {}).get("layers", {}))
    for record in selected_records:
        match = re.fullmatch(r"blk\.(\d+)\.(.+)\.expert-(\d+)", record["tensor"])
        if match is None:
            continue
        layer, tensor, expert = match.groups()
        layer_record = dict(moe_layers.get(layer, {}))
        tensors = dict(layer_record.get("tensors", {}))
        tensor_record = dict(tensors.get(tensor, {}))
        experts = dict(tensor_record.get("experts", {}))
        candidate = record["refined_candidate"]
        experts[expert] = {
            "outlier_fraction": candidate["outlier_fraction"],
            "awq_alpha": candidate["alpha"],
            "awq_clip": candidate["clip"],
            "shadow_error": record["shadow_error"],
            "coverage_uncertainty": record["coverage_uncertainty"],
        }
        tensor_record["experts"] = experts
        tensors[tensor] = tensor_record
        layer_record["tensors"] = tensors
        moe_layers[layer] = layer_record
    refined["tensor_families"] = families
    refined["tessera_shadow_calibration"] = {
        "schema": SCHEMA,
        "layers_scored": len(records),
        "layers_with_heldout": sum(record["has_heldout"] for record in records),
        "median_shadow_error": baseline,
        "worst_shadow_error": max((record["shadow_error"] for record in records), default=0.0),
        "selected_overrides": selected_records,
    }
    if moe_layers:
        refined["moe_residual_allocation"] = {
            "schema": "llama.tessera.moe-residual-policy.v1",
            "layers": moe_layers,
        }
    return refined


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a Tessera policy with provisional reconstruction error")
    parser.add_argument(
        "--layers", required=True, action="append",
        help="AWQ bundle directory; repeat for vision, audio, or drafter bundles",
    )
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evolver", default=str(Path(__file__).with_name("awq-evolve.py")))
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-fraction", type=float, default=0.12)
    parser.add_argument("--max-overrides", type=int, default=24)
    parser.add_argument("--residual-boost", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top fraction must be in (0, 1]")
    if args.max_overrides < 0 or args.residual_boost < 0.0:
        raise ValueError("override count and residual boost must be non-negative")
    policy = json.loads(Path(args.base_policy).read_text(encoding="utf-8"))
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("base policy has an unsupported schema")
    awq = load_evolver(Path(args.evolver))
    paths = sorted({
        path for directory in args.layers
        for path in Path(directory).glob("*.npz")
    })
    if not paths:
        raise ValueError("no AWQ layer bundles found")
    layers = [awq.load_layer(path, args.max_rows, args.max_tokens) for path in paths]
    records = score_layers(layers, policy, awq)
    selected = select_stratified(records, args.top_fraction, args.max_overrides)
    refined = refine_policy(policy, records, selected, args.residual_boost)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(refined, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}: scored={len(records)} overrides={len(selected)}", flush=True)


if __name__ == "__main__":
    main()
