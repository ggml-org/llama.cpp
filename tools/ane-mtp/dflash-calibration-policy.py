#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


SCHEMA = "llama.speculative.calibration-policy.v1"
# The unified llama.tessera.spec.v1 schema is the only schema the imatrix
# spec-calibration path emits. It always carries `confidence[]` (the cheap
# per-step payload), so consumers that only look at drafted/accepted/
# confidence work without change.
EVENT_SCHEMAS = {
    "llama.tessera.spec.v1": "dflash",
    "llama.mtp.acceptance.v1": "mtp",
}


def clamp(value, low, high):
    return max(low, min(high, value))


def load_events(path):
    events = []
    draft_type = None
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = EVENT_SCHEMAS.get(event.get("schema"))
            if event_type is None:
                raise ValueError(f"{path}:{line_number}: unsupported telemetry schema")
            if draft_type is not None and event_type != draft_type:
                raise ValueError(f"{path}:{line_number}: mixed DFlash and MTP telemetry")
            draft_type = event_type
            drafted = int(event["drafted"])
            accepted = int(event["accepted"])
            confidence = [float(value) for value in event.get("confidence", [])]
            if drafted != len(confidence) or accepted < 0 or accepted > drafted:
                raise ValueError(f"{path}:{line_number}: inconsistent drafted/accepted/confidence values")
            events.append((drafted, accepted, confidence))
    if not events:
        raise ValueError(f"{path}: no DFlash acceptance events")
    return draft_type, events


def build_policy(draft_type, events, base_outlier_fraction):
    max_position = max(drafted for drafted, _, _ in events)
    proposed = [0] * max_position
    reached = [0] * max_position
    accepted_at = [0] * max_position
    confidence_sum = [0.0] * max_position

    total_drafted = 0
    total_accepted = 0
    for drafted, accepted, confidence in events:
        total_drafted += drafted
        total_accepted += accepted
        for position in range(drafted):
            proposed[position] += 1
            confidence_sum[position] += confidence[position]
            if accepted >= position:
                reached[position] += 1
            if accepted > position:
                accepted_at[position] += 1

    position_metrics = []
    impact = []
    for position in range(max_position):
        count = proposed[position]
        reach_probability = reached[position] / len(events)
        conditional_acceptance = accepted_at[position] / reached[position] if reached[position] else 0.0
        mean_confidence = confidence_sum[position] / count if count else 0.0
        # Improving a position matters only when verification reaches it.
        # Low conditional acceptance and low draft confidence increase pressure.
        sensitivity = reach_probability * (
            1.0 + (1.0 - conditional_acceptance) + 0.5 * (1.0 - mean_confidence)
        )
        impact.append(sensitivity)
        position_metrics.append({
            "position": position,
            "proposed": count,
            "reach_probability": reach_probability,
            "conditional_acceptance": conditional_acceptance,
            "mean_confidence": mean_confidence,
            "sensitivity": sensitivity,
        })

    impact_mean = sum(impact) / len(impact) if impact else 1.0
    position_weights = [
        value / impact_mean if impact_mean > 0.0 else 1.0
        for value in impact
    ]
    accepted_fraction = total_accepted / total_drafted if total_drafted else 0.0
    pressure = clamp(1.0 - accepted_fraction, 0.0, 1.0)

    def fraction(base_multiplier, pressure_multiplier):
        return clamp(
            base_outlier_fraction * (base_multiplier + pressure * pressure_multiplier),
            base_outlier_fraction,
            0.05,
        )

    families = {
        "norm": {
            "match": ["norm"],
            "exact": True,
            "awq_alpha": 0.0,
            "outlier_fraction": 1.0,
        },
        "fusion": {
            "match": ["fc.weight", "encoder_proj", "per_layer_model_proj"],
            "exact": False,
            "awq_alpha": "auto",
            "outlier_fraction": fraction(2.5, 2.5),
        },
        "output_embedding": {
            "match": ["token_embd", "output.weight", "per_layer_token_embd"],
            "exact": False,
            "awq_alpha": "auto",
            "outlier_fraction": fraction(2.0, 2.0),
        },
        "attention": {
            "match": ["attn_q", "attn_k", "attn_v", "attn_output"],
            "exact": False,
            "awq_alpha": "auto",
            "outlier_fraction": fraction(1.5, 1.5),
        },
        "ffn": {
            "match": ["ffn_gate", "ffn_up", "ffn_down"],
            "exact": False,
            "awq_alpha": "auto",
            "outlier_fraction": fraction(1.0, 1.0),
        },
    }

    return {
        "schema": SCHEMA,
        "draft_type": draft_type,
        "event_count": len(events),
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "accepted_fraction": accepted_fraction,
        "calibration_pressure": pressure,
        "position_weights": position_weights,
        "positions": position_metrics,
        "tensor_families": families,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate an acceptance-aware AWQ and Tessera policy from draft telemetry",
    )
    parser.add_argument("--telemetry", required=True, help="DFlash JSONL telemetry")
    parser.add_argument("--output", required=True, help="Policy JSON output")
    parser.add_argument("--base-outlier-frac", type=float, default=0.005)
    args = parser.parse_args()

    if not math.isfinite(args.base_outlier_frac) or not 0.0 < args.base_outlier_frac <= 0.05:
        raise ValueError("--base-outlier-frac must be in (0, 0.05]")

    draft_type, events = load_events(args.telemetry)
    policy = build_policy(draft_type, events, args.base_outlier_frac)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}: {policy['event_count']} events, "
        f"acceptance={policy['accepted_fraction']:.4f}, "
        f"pressure={policy['calibration_pressure']:.4f}"
    )


if __name__ == "__main__":
    main()
