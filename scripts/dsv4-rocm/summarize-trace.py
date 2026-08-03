#!/usr/bin/env python3
"""Summarize rocprofv3 events inside a DSV4 harness measured interval."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter rocprofv3 CSV traces to measurement-start.ns through the last completed result"
    )
    parser.add_argument("run_dir", type=Path, help="DSV4 run directory containing rocprof/")
    parser.add_argument("--clock-offset-ns", type=int, help="realtime minus CLOCK_MONOTONIC in ns")
    parser.add_argument("--top", type=int, default=20, help="rows to print per domain (default: 20)")
    parser.add_argument("--json", type=Path, help="optional machine-readable output path")
    return parser.parse_args()


def read_ints(path: Path) -> list[int]:
    values: list[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            values.append(int(line))
    if not values:
        raise ValueError(f"no timestamps in {path}")
    return values


def read_clock_offset(run_dir: Path, override: int | None) -> int:
    if override is not None:
        return override
    path = run_dir / "clock-domain.txt"
    if not path.is_file():
        raise ValueError(f"missing {path}; supply --clock-offset-ns for a legacy trace")
    for line in path.read_text().splitlines():
        if line.startswith("realtime_minus_monotonic_ns="):
            return int(line.split("=", 1)[1])
    raise ValueError(f"missing realtime_minus_monotonic_ns in {path}")


def find_trace(rocprof_dir: Path, suffix: str) -> Path | None:
    matches = sorted(rocprof_dir.glob(f"*_{suffix}.csv"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one *_{suffix}.csv under {rocprof_dir}, found {len(matches)}")
    return matches[0]


def aggregate(path: Path | None, key_field: str, start_ns: int, end_ns: int) -> dict[str, dict[str, int]]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    if path is None:
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {key_field, "Start_Timestamp", "End_Timestamp"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            event_start = int(row["Start_Timestamp"])
            event_end = int(row["End_Timestamp"])
            clipped_start = max(event_start, start_ns)
            clipped_end = min(event_end, end_ns)
            if clipped_end <= clipped_start:
                continue
            item = totals[row[key_field]]
            item[0] += 1
            item[1] += clipped_end - clipped_start
    return {key: {"calls": value[0], "duration_ns": value[1]} for key, value in totals.items()}


def sorted_rows(values: dict[str, dict[str, int]]) -> list[tuple[str, dict[str, int]]]:
    return sorted(values.items(), key=lambda item: (-item[1]["duration_ns"], item[0]))


def print_domain(title: str, values: dict[str, dict[str, int]], top: int, wall_ns: int) -> None:
    print(f"\n[{title}]")
    if not values:
        print("no matching trace file/events")
        return
    total_ns = sum(item["duration_ns"] for item in values.values())
    total_calls = sum(item["calls"] for item in values.values())
    print(f"events={total_calls} summed_duration_ms={total_ns / 1e6:.3f} wall_equivalent_pct={100 * total_ns / wall_ns:.2f}")
    print("share_pct\tduration_ms\tcalls\tname")
    for name, item in sorted_rows(values)[:top]:
        share = 100 * item["duration_ns"] / total_ns
        print(f"{share:.2f}\t{item['duration_ns'] / 1e6:.3f}\t{item['calls']}\t{name}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rocprof_dir = run_dir / "rocprof"
    if not rocprof_dir.is_dir():
        raise ValueError(f"missing rocprof directory: {rocprof_dir}")

    offset_ns = read_clock_offset(run_dir, args.clock_offset_ns)
    measurement_epoch_ns = read_ints(run_dir / "measurement-start.ns")[0]
    completed_epoch_ns = max(read_ints(run_dir / "result-completed-at.ns"))
    start_ns = measurement_epoch_ns - offset_ns
    end_ns = completed_epoch_ns - offset_ns
    if end_ns <= start_ns:
        raise ValueError("completed timestamp does not follow measurement start")
    wall_ns = end_ns - start_ns

    domains = {
        "kernels": aggregate(find_trace(rocprof_dir, "kernel_trace"), "Kernel_Name", start_ns, end_ns),
        "memory_copies": aggregate(find_trace(rocprof_dir, "memory_copy_trace"), "Direction", start_ns, end_ns),
        "rccl_api": aggregate(find_trace(rocprof_dir, "rccl_api_trace"), "Function", start_ns, end_ns),
    }

    print(f"run_dir={run_dir}")
    print(f"clock_offset_ns={offset_ns}")
    print(f"measurement_epoch_ns={measurement_epoch_ns}")
    print(f"completed_epoch_ns={completed_epoch_ns}")
    print(f"measurement_monotonic_ns={start_ns}")
    print(f"completed_monotonic_ns={end_ns}")
    print(f"wall_ms={wall_ns / 1e6:.3f}")
    print("durations are clipped to the measured interval; kernel time is summed across devices/queues")

    print_domain("kernels", domains["kernels"], args.top, wall_ns)
    print_domain("memory copies", domains["memory_copies"], args.top, wall_ns)
    print_domain("RCCL API", domains["rccl_api"], args.top, wall_ns)

    if args.json:
        output = {
            "run_dir": str(run_dir),
            "clock_offset_ns": offset_ns,
            "measurement_epoch_ns": measurement_epoch_ns,
            "completed_epoch_ns": completed_epoch_ns,
            "measurement_monotonic_ns": start_ns,
            "completed_monotonic_ns": end_ns,
            "wall_ns": wall_ns,
            "domains": domains,
        }
        args.json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)