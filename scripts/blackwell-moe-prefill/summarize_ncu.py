#!/usr/bin/env python3

import csv
import pathlib
import re
import sys


METRIC_PATTERNS = (
    re.compile(r"time_duration"),
    re.compile(r"pipe_tensor"),
    re.compile(r"dram.*throughput"),
    re.compile(r"dram.*bytes"),
    re.compile(r"l1tex.*global.*(?:bytes|sectors|requests)"),
    re.compile(r"lts.*(?:hit_rate|throughput|bytes)"),
    re.compile(r"occupancy"),
    re.compile(r"warps_active"),
    re.compile(r"issue_active"),
    re.compile(r"warp_issue_stalled"),
    re.compile(r"registers_per_thread"),
    re.compile(r"shared_memory"),
    re.compile(r"waves_per_multiprocessor"),
)


def selected(name: str) -> bool:
    return any(pattern.search(name) for pattern in METRIC_PATTERNS)


def read_metrics(path: pathlib.Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    header: list[str] | None = None
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for values in csv.reader(handle):
            if "Metric Name" in values:
                header = values
                continue
            if header is None or len(values) != len(header):
                continue
            record = dict(zip(header, values))
            name = record.get("Metric Name", "")
            if selected(name):
                rows.append((
                    record.get("Kernel Name", ""),
                    name,
                    record.get("Metric Value", ""),
                    record.get("Metric Unit", ""),
                ))
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RUN_DIR", file=sys.stderr)
        return 2

    run_dir = pathlib.Path(sys.argv[1])
    print("# Nsight Compute summary")
    found = False
    for path in sorted(run_dir.glob("*.csv")):
        metrics = read_metrics(path)
        if not metrics:
            continue
        found = True
        print(f"\n## {path.stem}\n")
        print("| Kernel | Metric | Value | Unit |")
        print("|---|---|---:|---|")
        for kernel, name, value, unit in metrics:
            print(f"| {kernel} | {name} | {value} | {unit} |")
    if not found:
        print("\nNo selected metrics were found in the CSV reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
