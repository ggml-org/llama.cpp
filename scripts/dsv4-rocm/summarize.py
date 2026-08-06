#!/usr/bin/env python3
"""Summarize raw llama-bench JSONL while preserving every measured sample."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import statistics
import sys
from typing import Any

TAIL_PERCENTILE_MIN_SAMPLES = 20


def parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def load_records(path: pathlib.Path, allow_trailing_partial: bool) -> tuple[list[dict[str, Any]], bool]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty result file: {path}")

    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("JSON result must be an array of objects")
        return value, False

    nonblank = [(index, line) for index, line in enumerate(text.splitlines()) if line.strip()]
    records: list[dict[str, Any]] = []
    dropped_trailing_partial = False
    for position, (line_index, line) in enumerate(nonblank):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            is_last = position == len(nonblank) - 1
            if allow_trailing_partial and is_last:
                dropped_trailing_partial = True
                break
            raise ValueError(f"malformed JSONL record at line {line_index + 1}")
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at line {line_index + 1} is not an object")
        records.append(value)
    if not records:
        raise ValueError("result has no complete JSONL records")
    return records, dropped_trailing_partial


def percentile_nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def numeric(record: dict[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=pathlib.Path)
    parser.add_argument("--json", dest="json_out", type=pathlib.Path)
    parser.add_argument("--tsv", dest="tsv_out", type=pathlib.Path)
    parser.add_argument("--expected-prompts", type=parse_csv_ints, default=[])
    parser.add_argument("--expected-ubatches", type=parse_csv_ints, default=[])
    parser.add_argument("--expected-reps", type=int, default=0)
    parser.add_argument("--truncated", action="store_true")
    parser.add_argument("--allow-trailing-partial", action="store_true")
    args = parser.parse_args()

    if args.allow_trailing_partial and not args.truncated:
        raise ValueError("--allow-trailing-partial requires --truncated")
    if args.expected_reps < 0:
        raise ValueError("--expected-reps must be non-negative")

    records, dropped_trailing_partial = load_records(args.result, args.allow_trailing_partial)
    rows: list[dict[str, Any]] = []
    completed_shapes: list[tuple[int, int]] = []
    repetitions_complete = True

    for record in records:
        samples_ts = record.get("samples_ts")
        samples_ns = record.get("samples_ns")
        if not isinstance(samples_ts, list) or not samples_ts:
            raise ValueError("record has no non-empty samples_ts array")
        if not isinstance(samples_ns, list) or len(samples_ns) != len(samples_ts):
            raise ValueError("record samples_ns is absent or does not match samples_ts")
        ts = [float(value) for value in samples_ts]
        latency_ms = [float(value) / 1_000_000.0 for value in samples_ns]
        n_prompt = numeric(record, "n_prompt", "pp")
        n_ubatch = numeric(record, "n_ubatch")
        completed_shapes.append((n_prompt, n_ubatch))
        if args.expected_reps and len(ts) != args.expected_reps:
            repetitions_complete = False

        tails_available = len(ts) >= TAIL_PERCENTILE_MIN_SAMPLES
        row = {
            "model_filename": record.get("model_filename", record.get("model", "")),
            "model_type": record.get("model_type", ""),
            "n_prompt": n_prompt,
            "n_gen": numeric(record, "n_gen", "tg"),
            "n_batch": numeric(record, "n_batch"),
            "n_ubatch": n_ubatch,
            "split_mode": record.get("split_mode", ""),
            "tensor_split": record.get("tensor_split", ""),
            "flash_attn": record.get("flash_attn", ""),
            "repetitions": len(ts),
            "mean_ts": statistics.fmean(ts),
            "median_ts": statistics.median(ts),
            "p05_ts": percentile_nearest_rank(ts, 0.05) if tails_available else None,
            "p95_ts": percentile_nearest_rank(ts, 0.95) if tails_available else None,
            "stdev_ts": statistics.stdev(ts) if len(ts) > 1 else 0.0,
            "min_ts": min(ts),
            "max_ts": max(ts),
            "mean_ms": statistics.fmean(latency_ms),
            "median_ms": statistics.median(latency_ms),
            "p05_ms": percentile_nearest_rank(latency_ms, 0.05) if tails_available else None,
            "p95_ms": percentile_nearest_rank(latency_ms, 0.95) if tails_available else None,
            "min_ms": min(latency_ms),
            "max_ms": max(latency_ms),
            "tail_percentiles_available": tails_available,
            "samples_ts": ts,
            "samples_ns": [int(value) for value in samples_ns],
        }
        rows.append(row)

    expected_shapes = list(itertools.product(args.expected_prompts, args.expected_ubatches))
    expected_set = set(expected_shapes)
    completed_set = set(completed_shapes)
    shapes_complete = (
        not expected_set
        or (completed_set == expected_set and len(completed_shapes) == len(expected_shapes))
    )
    complete = not args.truncated and not dropped_trailing_partial and shapes_complete and repetitions_complete
    missing_shapes = sorted(expected_set - completed_set)

    document = {
        "complete": complete,
        "truncated": args.truncated,
        "dropped_trailing_partial": dropped_trailing_partial,
        "expected_shapes": [
            {"n_prompt": prompt, "n_ubatch": ubatch} for prompt, ubatch in expected_shapes
        ],
        "completed_shapes": [
            {"n_prompt": prompt, "n_ubatch": ubatch} for prompt, ubatch in sorted(completed_set)
        ],
        "missing_shapes": [
            {"n_prompt": prompt, "n_ubatch": ubatch} for prompt, ubatch in missing_shapes
        ],
        "expected_repetitions": args.expected_reps,
        "repetitions_complete": repetitions_complete,
        "records": rows,
    }

    columns = [
        "run_complete", "n_prompt", "n_gen", "n_batch", "n_ubatch", "repetitions",
        "median_ts", "min_ts", "max_ts", "mean_ts", "stdev_ts", "p05_ts", "p95_ts",
        "median_ms", "min_ms", "max_ms", "p05_ms", "p95_ms",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        rendered: list[str] = []
        values = {"run_complete": complete, **row}
        for column in columns:
            value = values[column]
            if value is None:
                rendered.append("NA")
            elif isinstance(value, bool):
                rendered.append("1" if value else "0")
            elif isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        lines.append("\t".join(rendered))
    tsv = "\n".join(lines) + "\n"

    if args.tsv_out:
        args.tsv_out.write_text(tsv, encoding="utf-8")
    else:
        sys.stdout.write(tsv)
    if args.json_out:
        args.json_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)