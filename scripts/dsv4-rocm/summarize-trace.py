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
    parser.add_argument(
        "--max-clock-drift-ns", type=int, default=1_000_000,
        help="maximum accepted start/end clock-offset change (default: 1000000 ns)",
    )
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


def read_clock_offset(
    run_dir: Path, override: int | None, max_drift_ns: int
) -> tuple[int, int | None, int | None, str]:
    if override is not None:
        return override, None, None, "command-line override"
    path = run_dir / "clock-domain.txt"
    if not path.is_file():
        raise ValueError(f"missing {path}; supply --clock-offset-ns for a legacy trace")
    values = dict(
        line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
    )
    if "start_realtime_minus_monotonic_ns" in values:
        start_offset = int(values["start_realtime_minus_monotonic_ns"])
        start_span_ns = int(values.get("start_calibration_span_ns", "0"))
        if "end_realtime_minus_monotonic_ns" not in values:
            raise ValueError("new-format clock-domain.txt is missing the run-end calibration")
        end_offset = int(values["end_realtime_minus_monotonic_ns"])
        end_span_ns = int(values.get("end_calibration_span_ns", "0"))
        drift_ns = end_offset - start_offset
        if abs(drift_ns) > max_drift_ns:
            raise ValueError(
                f"realtime/monotonic offset changed by {drift_ns} ns; "
                f"limit is {max_drift_ns} ns"
            )
        uncertainty_ns = (abs(drift_ns) + max(start_span_ns, end_span_ns) + 1) // 2
        return (start_offset + end_offset) // 2, drift_ns, uncertainty_ns, "run start/end midpoint"
    if "realtime_minus_monotonic_ns" in values:
        return int(values["realtime_minus_monotonic_ns"]), None, None, "legacy single calibration"
    raise ValueError(f"missing realtime/monotonic offset in {path}")


def find_trace(rocprof_dir: Path, suffix: str, required: bool = False) -> Path | None:
    matches = sorted(rocprof_dir.glob(f"*_{suffix}.csv"))
    if not matches:
        if required:
            raise ValueError(f"missing required *_{suffix}.csv under {rocprof_dir}")
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
    if args.max_clock_drift_ns < 0:
        raise ValueError("--max-clock-drift-ns must be non-negative")
    run_dir = args.run_dir.resolve()
    rocprof_dir = run_dir / "rocprof"
    if not rocprof_dir.is_dir():
        raise ValueError(f"missing rocprof directory: {rocprof_dir}")

    offset_ns, clock_drift_ns, clock_uncertainty_ns, clock_source = read_clock_offset(
        run_dir, args.clock_offset_ns, args.max_clock_drift_ns
    )
    measurement_epoch_ns = read_ints(run_dir / "measurement-start.ns")[0]
    completed_epoch_ns = max(read_ints(run_dir / "result-completed-at.ns"))
    start_ns = measurement_epoch_ns - offset_ns
    end_ns = completed_epoch_ns - offset_ns
    if end_ns <= start_ns:
        raise ValueError("completed timestamp does not follow measurement start")
    wall_ns = end_ns - start_ns

    trace_files = {
        "kernels": find_trace(rocprof_dir, "kernel_trace", required=True),
        "memory_copies": find_trace(rocprof_dir, "memory_copy_trace"),
        "rccl_api": find_trace(rocprof_dir, "rccl_api_trace"),
    }
    domains = {
        "kernels": aggregate(trace_files["kernels"], "Kernel_Name", start_ns, end_ns),
        "memory_copies": aggregate(trace_files["memory_copies"], "Direction", start_ns, end_ns),
        "rccl_api": aggregate(trace_files["rccl_api"], "Function", start_ns, end_ns),
    }
    if not domains["kernels"]:
        raise ValueError("required kernel trace has no events in the measured interval")

    print(f"run_dir={run_dir}")
    print(f"clock_offset_ns={offset_ns}")
    print(f"clock_source={clock_source}")
    print(f"clock_drift_ns={clock_drift_ns if clock_drift_ns is not None else 'unknown'}")
    print(f"clock_uncertainty_ns={clock_uncertainty_ns if clock_uncertainty_ns is not None else 'unknown'}")
    for domain, path in trace_files.items():
        print(f"{domain}_trace={path if path is not None else 'missing'}")
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
            "clock_source": clock_source,
            "clock_drift_ns": clock_drift_ns,
            "clock_uncertainty_ns": clock_uncertainty_ns,
            "trace_files": {key: str(value) if value is not None else None for key, value in trace_files.items()},
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