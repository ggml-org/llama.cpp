#!/usr/bin/env python3
"""Summarize target-only llama-bench TG JSONL by starting KV depth."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from typing import Any


def csv_nonnegative_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique comma-separated non-negative integers")
    return result


def load_jsonl(path: pathlib.Path, allow_trailing_partial: bool) -> tuple[list[dict[str, Any]], bool]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty result file: {path}")
    records: list[dict[str, Any]] = []
    dropped = False
    nonblank = [(i, line) for i, line in enumerate(text.splitlines()) if line.strip()]
    for position, (line_index, line) in enumerate(nonblank):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if allow_trailing_partial and position == len(nonblank) - 1:
                dropped = True
                break
            raise ValueError(f"malformed JSONL record at line {line_index + 1}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at line {line_index + 1} is not an object")
        records.append(value)
    if not records:
        raise ValueError("result has no complete JSONL records")
    return records, dropped


def integer(record: dict[str, Any], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"record has no numeric {name}")
    return int(value)


def normalize_tensor_split(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = value.split("/")
    elif isinstance(value, list):
        parts = value
    else:
        return ()
    try:
        vals = tuple(float(item) for item in parts)
    except (TypeError, ValueError):
        return ()
    total = sum(vals)
    if total <= 0:
        return ()
    return tuple(item / total for item in vals)


def median_abs_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=pathlib.Path)
    parser.add_argument("--json", dest="json_out", type=pathlib.Path, required=True)
    parser.add_argument("--tsv", dest="tsv_out", type=pathlib.Path, required=True)
    parser.add_argument("--expected-depths", type=csv_nonnegative_ints, required=True)
    parser.add_argument("--expected-gen", type=int, required=True)
    parser.add_argument("--expected-reps", type=int, required=True, help="raw llama-bench repetitions")
    parser.add_argument("--discard-first", type=int, default=0)
    parser.add_argument("--stability-limit", type=float, default=0.03)
    parser.add_argument("--expected-batch", type=int, required=True)
    parser.add_argument("--expected-ubatch", type=int, required=True)
    parser.add_argument("--expected-tensor-split", default="1/1/1/1")
    parser.add_argument("--truncated", action="store_true")
    parser.add_argument("--allow-trailing-partial", action="store_true")
    args = parser.parse_args()

    if args.expected_gen <= 0 or args.expected_reps <= 0:
        raise ValueError("expected gen/reps must be positive")
    if not 0 <= args.discard_first < args.expected_reps:
        raise ValueError("discard-first must be smaller than expected-reps")
    if args.expected_reps - args.discard_first < 5:
        raise ValueError("at least five accepted repetitions are required")
    if not 0 < args.stability_limit < 1:
        raise ValueError("stability-limit must be between zero and one")
    if args.allow_trailing_partial and not args.truncated:
        raise ValueError("--allow-trailing-partial requires --truncated")

    records, dropped = load_jsonl(args.result, args.allow_trailing_partial)
    expected_depths = args.expected_depths
    expected_set = set(expected_depths)
    expected_split = normalize_tensor_split(args.expected_tensor_split)
    rows: list[dict[str, Any]] = []
    seen: list[int] = []
    contract_errors: list[str] = []

    for record_index, record in enumerate(records):
        depth = integer(record, "n_depth")
        n_prompt = integer(record, "n_prompt")
        n_gen = integer(record, "n_gen")
        n_batch = integer(record, "n_batch")
        n_ubatch = integer(record, "n_ubatch")
        samples_ns_raw = record.get("samples_ns")
        samples_ts_raw = record.get("samples_ts")
        if not isinstance(samples_ns_raw, list) or not all(isinstance(v, (int, float)) for v in samples_ns_raw):
            raise ValueError(f"depth {depth}: samples_ns missing or nonnumeric")
        if not isinstance(samples_ts_raw, list) or len(samples_ts_raw) != len(samples_ns_raw):
            raise ValueError(f"depth {depth}: samples_ts missing or length mismatch")
        samples_ns = [int(v) for v in samples_ns_raw]
        samples_ts = [float(v) for v in samples_ts_raw]
        seen.append(depth)

        errors: list[str] = []
        if depth not in expected_set:
            errors.append(f"unexpected depth {depth}")
        if n_prompt != 0:
            errors.append(f"n_prompt={n_prompt}, expected 0")
        if n_gen != args.expected_gen:
            errors.append(f"n_gen={n_gen}, expected {args.expected_gen}")
        if n_batch != args.expected_batch:
            errors.append(f"n_batch={n_batch}, expected {args.expected_batch}")
        if n_ubatch != args.expected_ubatch:
            errors.append(f"n_ubatch={n_ubatch}, expected {args.expected_ubatch}")
        if record.get("split_mode") != "tensor":
            errors.append(f"split_mode={record.get('split_mode')!r}, expected 'tensor'")
        actual_split = normalize_tensor_split(record.get("tensor_split"))
        if len(actual_split) != len(expected_split) or any(abs(a - b) > 1e-6 for a, b in zip(actual_split, expected_split)):
            errors.append(f"tensor_split={record.get('tensor_split')!r}, expected {args.expected_tensor_split}")
        if record.get("type_k") != "f16" or record.get("type_v") != "f16":
            errors.append(f"cache types={record.get('type_k')}/{record.get('type_v')}, expected f16/f16")
        if record.get("flash_attn") not in (1, True, "1", "on"):
            errors.append(f"flash_attn={record.get('flash_attn')!r}, expected enabled")
        if len(samples_ns) != args.expected_reps:
            errors.append(f"raw repetitions={len(samples_ns)}, expected {args.expected_reps}")
        accepted_ns = samples_ns[args.discard_first:]
        accepted_reported_ts = samples_ts[args.discard_first:]
        if not accepted_ns or any(value <= 0 for value in accepted_ns):
            raise ValueError(f"depth {depth}: accepted samples are absent or nonpositive")
        recomputed_ts = [1e9 * n_gen / value for value in accepted_ns]
        for i, (reported, recomputed) in enumerate(zip(accepted_reported_ts, recomputed_ts)):
            tolerance = max(1e-3, abs(recomputed) * 5e-6)
            if not math.isfinite(reported) or abs(reported - recomputed) > tolerance:
                errors.append(f"sample {i}: reported t/s {reported} != recomputed {recomputed}")
        if errors:
            contract_errors.extend(f"record {record_index} depth {depth}: {error}" for error in errors)

        ms_per_token = [value / n_gen / 1e6 for value in accepted_ns]
        median_ts = statistics.median(recomputed_ts)
        median_ms = statistics.median(ms_per_token)
        mad_ms = median_abs_deviation(ms_per_token)
        mad_ratio = mad_ms / median_ms if median_ms else math.inf
        stable = math.isfinite(mad_ratio) and mad_ratio <= args.stability_limit

        rows.append({
            "depth": depth,
            "actual_starting_kv_depth": depth,
            "n_gen": n_gen,
            "raw_repetitions": len(samples_ns),
            "discarded_first": args.discard_first,
            "accepted_repetitions": len(accepted_ns),
            "median_ts": median_ts,
            "min_ts": min(recomputed_ts),
            "max_ts": max(recomputed_ts),
            "median_ms_per_token": median_ms,
            "min_ms_per_token": min(ms_per_token),
            "max_ms_per_token": max(ms_per_token),
            "mad_ms_per_token": mad_ms,
            "mad_over_median": mad_ratio,
            "stability_limit": args.stability_limit,
            "stable": stable,
            "contract_errors": errors,
            "raw_samples_ns": samples_ns,
            "accepted_samples_ns": accepted_ns,
            "accepted_samples_ts": recomputed_ts,
            "accepted_samples_ms_per_token": ms_per_token,
        })

    duplicate_depths = sorted({depth for depth in seen if seen.count(depth) > 1})
    missing_depths = sorted(expected_set - set(seen))
    extra_depths = sorted(set(seen) - expected_set)
    order_matches = seen == expected_depths
    shape_complete = not missing_depths and not extra_depths and not duplicate_depths and order_matches
    complete = not args.truncated and not dropped and shape_complete and not contract_errors
    stable = complete and all(row["stable"] for row in rows)

    document = {
        "complete": complete,
        "stable": stable,
        "observational_baseline_accepted": stable,
        "post_fix_deployment_accepted": False,
        "post_fix_deployment_note": "requires separate scheduler residency summary",
        "truncated": args.truncated,
        "dropped_trailing_partial": dropped,
        "expected_depths": expected_depths,
        "seen_depths": seen,
        "missing_depths": missing_depths,
        "extra_depths": extra_depths,
        "duplicate_depths": duplicate_depths,
        "depth_order_matches": order_matches,
        "expected_gen": args.expected_gen,
        "expected_raw_repetitions": args.expected_reps,
        "discard_first": args.discard_first,
        "accepted_repetitions": args.expected_reps - args.discard_first,
        "stability_limit": args.stability_limit,
        "contract_errors": contract_errors,
        "records": rows,
    }
    args.json_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    columns = [
        "run_complete", "run_stable", "depth", "n_gen", "accepted_repetitions",
        "median_ts", "min_ts", "max_ts", "median_ms_per_token",
        "min_ms_per_token", "max_ms_per_token", "mad_over_median", "stable",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        values: dict[str, Any] = {"run_complete": complete, "run_stable": stable, **row}
        rendered: list[str] = []
        for column in columns:
            value = values[column]
            if isinstance(value, bool):
                rendered.append("1" if value else "0")
            elif isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value))
        lines.append("\t".join(rendered))
    args.tsv_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)