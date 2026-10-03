from __future__ import annotations

from statistics import mean, median, stdev
from typing import Any


COUNTERS = (
    "llamacpp:spec_decode_num_draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total",
    "llamacpp:spec_decode_num_drafts_total",
)


def counter_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in COUNTERS:
        result[name] = after[name] - before[name] if name in before and name in after else None
    return result


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    end_to_end_throughput = [sample["end_to_end_output_tps"] for sample in samples if sample.get("end_to_end_output_tps") is not None]
    throughput = [sample["client_delivery_tps"] for sample in samples if sample.get("client_delivery_tps") is not None]
    ttft = [sample["ttft_ms"] for sample in samples if sample.get("ttft_ms") is not None]
    return {
        "sample_count": len(samples),
        "end_to_end_output_tps": summarize_values(end_to_end_throughput),
        "client_delivery_tps": summarize_values(throughput),
        "ttft_ms": summarize_values(ttft),
    }


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stdev": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
    }
