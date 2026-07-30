#!/usr/bin/env python3
"""Privacy-safe router telemetry and coverage-aware Tessera MoE policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExpertEvidence:
    layer: int
    expert: int
    observations: int
    selected: int
    probability_sum: float
    confidence_sum: float
    margin_sum: float
    output_error_sum: float
    logit_divergence_sum: float

    @property
    def frequency(self) -> float:
        return self.selected / max(self.observations, 1)

    @property
    def mean_output_error(self) -> float:
        return self.output_error_sum / max(self.selected, 1)


class RouterAccumulator:
    """Accumulate sufficient statistics without retaining token-level data."""

    def __init__(self, layer: int, experts: int):
        if layer < 0 or experts <= 1:
            raise ValueError("router layer and expert count are invalid")
        self.layer = layer
        self.experts = experts
        self.observations = 0
        self.selected = np.zeros(experts, dtype=np.int64)
        self.probability_sum = np.zeros(experts, dtype=np.float64)
        self.confidence_sum = np.zeros(experts, dtype=np.float64)
        self.margin_sum = np.zeros(experts, dtype=np.float64)
        self.output_error_sum = np.zeros(experts, dtype=np.float64)
        self.logit_divergence_sum = np.zeros(experts, dtype=np.float64)

    def update(
        self,
        router_logits: np.ndarray,
        top_k: int,
        expert_output_error: np.ndarray | None = None,
        logit_divergence: np.ndarray | None = None,
    ) -> None:
        logits = np.asarray(router_logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != self.experts:
            raise ValueError("router logits must be [tokens, experts]")
        if not 0 < top_k < self.experts:
            raise ValueError("top-k must be smaller than the expert count")
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        order = np.argsort(probabilities, axis=1, kind="stable")
        selected = order[:, -top_k:]
        boundary = probabilities[
            np.arange(probabilities.shape[0]), order[:, -top_k]
        ]
        rejected = probabilities[
            np.arange(probabilities.shape[0]), order[:, -top_k - 1]
        ]
        margins = boundary - rejected
        self.observations += probabilities.shape[0]
        self.probability_sum += np.sum(probabilities, axis=0)
        for token in range(probabilities.shape[0]):
            for expert in selected[token]:
                self.selected[expert] += 1
                self.confidence_sum[expert] += probabilities[token, expert]
                self.margin_sum[expert] += margins[token]
        for values, target, label in (
            (expert_output_error, self.output_error_sum, "expert output error"),
            (logit_divergence, self.logit_divergence_sum, "logit divergence"),
        ):
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float64)
            if values.shape != selected.shape:
                raise ValueError(f"{label} must have one value per selected expert")
            for token in range(selected.shape[0]):
                target[selected[token]] += values[token]

    def evidence(self) -> list[ExpertEvidence]:
        return [
            ExpertEvidence(
                self.layer,
                expert,
                self.observations,
                int(self.selected[expert]),
                float(self.probability_sum[expert]),
                float(self.confidence_sum[expert]),
                float(self.margin_sum[expert]),
                float(self.output_error_sum[expert]),
                float(self.logit_divergence_sum[expert]),
            )
            for expert in range(self.experts)
        ]


def coverage_aware_scores(
    evidence: list[ExpertEvidence],
    prior_strength: float = 4096.0,
) -> dict[tuple[int, int], float]:
    """Return robust expert priorities with layer priors for rare experts."""
    if prior_strength < 0:
        raise ValueError("prior strength cannot be negative")
    by_layer: dict[int, list[ExpertEvidence]] = {}
    for item in evidence:
        by_layer.setdefault(item.layer, []).append(item)
    priorities: dict[tuple[int, int], float] = {}
    for layer, items in by_layer.items():
        observed_errors = [
            item.mean_output_error for item in items if item.selected > 0
        ]
        layer_error = float(np.median(observed_errors)) if observed_errors else 0.0
        total_selected = max(sum(item.selected for item in items), 1)
        for item in items:
            shrink = item.selected / (item.selected + prior_strength)
            error = shrink * item.mean_output_error + (1.0 - shrink) * layer_error
            utilization = item.selected / total_selected
            fragility = 1.0 / max(
                item.margin_sum / max(item.selected, 1),
                1e-4,
            )
            priorities[(layer, item.expert)] = (
                0.50 * utilization
                + 0.35 * error
                + 0.15 * min(fragility, 100.0) / 100.0
            )
    return priorities


def allocate_expert_residuals(
    scores: dict[tuple[int, int], float],
    total_fraction: float,
    minimum_fraction: float = 0.0001,
    maximum_fraction: float = 0.05,
) -> dict[tuple[int, int], float]:
    """Allocate one global residual budget while protecting every expert."""
    if not scores:
        return {}
    if not 0 < total_fraction <= maximum_fraction:
        raise ValueError("total fraction is outside the Tessera residual range")
    if not 0 <= minimum_fraction <= total_fraction:
        raise ValueError("minimum residual fraction is invalid")
    keys = sorted(scores)
    budget = total_fraction * len(keys)
    base = minimum_fraction * len(keys)
    remaining = max(budget - base, 0.0)
    positive = np.asarray([max(float(scores[key]), 0.0) for key in keys])
    if not np.any(positive):
        positive.fill(1.0)
    allocation = np.full(len(keys), minimum_fraction, dtype=np.float64)
    active = np.ones(len(keys), dtype=bool)
    while remaining > 1e-12 and np.any(active):
        weights = positive * active
        weights /= weights.sum()
        proposal = remaining * weights
        capacity = maximum_fraction - allocation
        accepted = np.minimum(proposal, capacity)
        allocation += accepted
        consumed = float(accepted.sum())
        remaining -= consumed
        active = capacity - accepted > 1e-12
        if consumed <= 1e-12:
            break
    return {key: float(allocation[index]) for index, key in enumerate(keys)}


def residual_policy(
    evidence: list[ExpertEvidence],
    total_fraction: float,
    prior_strength: float = 4096.0,
) -> dict:
    scores = coverage_aware_scores(evidence, prior_strength)
    allocation = allocate_expert_residuals(scores, total_fraction)
    layers: dict[str, dict] = {}
    for (layer, expert), fraction in allocation.items():
        layers.setdefault(str(layer), {"experts": {}})
        layers[str(layer)]["experts"][str(expert)] = {
            "outlier_fraction": fraction,
            "selected": next(
                item.selected
                for item in evidence
                if item.layer == layer and item.expert == expert
            ),
            "priority": scores[(layer, expert)],
        }
    return {
        "schema": "llama.tessera.moe-residual-policy.v1",
        "target_fraction": total_fraction,
        "prior_strength": prior_strength,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a coverage-aware Tessera MoE residual policy"
    )
    parser.add_argument("--router-parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-policy", default=None)
    parser.add_argument("--target-fraction", type=float, default=0.005)
    parser.add_argument("--prior-strength", type=float, default=4096.0)
    args = parser.parse_args()

    import polars as pl

    rows = pl.read_parquet(args.router_parquet).to_dicts()
    evidence = [
        ExpertEvidence(
            layer=int(row["layer"]),
            expert=int(row["expert"]),
            observations=int(row["observations"]),
            selected=int(row["selected"]),
            probability_sum=float(row["probability_sum"]),
            confidence_sum=float(row["confidence_sum"]),
            margin_sum=float(row["margin_sum"]),
            output_error_sum=float(row.get("output_error_sum", 0.0)),
            logit_divergence_sum=float(
                row.get("downstream_divergence_sum", 0.0)
            ),
        )
        for row in rows
    ]
    moe_policy = residual_policy(
        evidence, args.target_fraction, args.prior_strength
    )
    if args.base_policy:
        policy = json.loads(Path(args.base_policy).read_text(encoding="utf-8"))
        if policy.get("schema") != "llama.speculative.calibration-policy.v1":
            raise ValueError("base policy has an unsupported schema")
    else:
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {},
        }
    policy["moe_residual_allocation"] = moe_policy
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
