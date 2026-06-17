#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from speed_bench import fmt_value, print_rows


def load_summary(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary")
    if not isinstance(summary, list):
        raise ValueError(f"{path} does not contain a summary list")
    return summary


def compare_rows(baseline: list[dict[str, Any]], speculative: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_category = {row["category"]: row for row in baseline}
    comparisons = []
    for row in speculative:
        base = baseline_by_category.get(row["category"])
        if not base:
            continue
        base_speed = base.get("avg_pred_t_s")
        spec_speed = row.get("avg_pred_t_s")
        base_latency = base.get("avg_latency")
        spec_latency = row.get("avg_latency")
        comparisons.append(
            {
                "category": row["category"],
                "base_avg_pred_t_s": base_speed,
                "spec_avg_pred_t_s": spec_speed,
                "decode_speedup": (spec_speed / base_speed) if base_speed and spec_speed else None,
                "base_avg_latency": base_latency,
                "spec_avg_latency": spec_latency,
                "latency_speedup": (base_latency / spec_latency) if base_latency and spec_latency else None,
                "accept_rate": row.get("accept_rate"),
            }
        )
    return comparisons


def print_comparison(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No overlapping categories found for comparison.")
        return
    columns = [
        ("category", "category", ""),
        ("base_avg_pred_t/s", "base_avg_pred_t_s", "speed"),
        ("spec_avg_pred_t/s", "spec_avg_pred_t_s", "speed"),
        ("decode_speedup", "decode_speedup", "speedup"),
        ("base_avg_latency", "base_avg_latency", "seconds"),
        ("spec_avg_latency", "spec_avg_latency", "seconds"),
        ("latency_speedup", "latency_speedup", "speedup"),
        ("accept_rate", "accept_rate", "rate"),
    ]
    print_rows(rows, columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two SPEED-Bench runs (baseline vs speculative).")
    parser.add_argument("--baseline", required=True, help="Baseline results JSON produced by speed_bench.py --output")
    parser.add_argument("--speculative", required=True, help="Speculative decoding results JSON produced by speed_bench.py --output")
    args = parser.parse_args(argv)

    try:
        baseline = load_summary(args.baseline)
        speculative = load_summary(args.speculative)
    except Exception as exc:
        print(f"speed_bench_compare: failed to load inputs: {exc}", file=sys.stderr)
        return 2

    comparisons = compare_rows(baseline, speculative)
    print(f"Comparison: baseline={args.baseline} speculative={args.speculative}")
    print_comparison(comparisons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
