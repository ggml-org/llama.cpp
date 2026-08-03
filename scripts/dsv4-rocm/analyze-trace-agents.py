#!/usr/bin/env python3
"""Attribute a summarized DSV4 measured-region rocprof trace by KFD agent."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable


def clipped_interval(
    row: dict[str, str], start_ns: int, end_ns: int, context: str
) -> tuple[int, int] | None:
    try:
        event_start = int(row["Start_Timestamp"])
        event_end = int(row["End_Timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid event timestamps") from exc
    if event_end < event_start:
        raise ValueError(f"{context}: event ends before it starts")
    start = max(event_start, start_ns)
    end = min(event_end, end_ns)
    return (start, end) if end > start else None


def bdf_from_location(domain_value: str, location_value: str) -> str:
    domain = int(domain_value)
    location = int(location_value)
    return f"{domain:04x}:{(location >> 8) & 0xff:02x}:{(location >> 3) & 0x1f:02x}.{location & 7}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "measured-region-summary.json"
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path} is not a JSON object")
    try:
        start_ns = int(summary["measurement_monotonic_ns"])
        end_ns = int(summary["completed_monotonic_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{summary_path} has invalid measured-region timestamps") from exc
    if end_ns <= start_ns:
        raise ValueError(f"{summary_path} has a non-positive measured interval")
    wall_ns = end_ns - start_ns
    rocprof = run_dir / "rocprof"

    kernel_paths = list(rocprof.glob("*_kernel_trace.csv"))
    agent_paths = list(rocprof.glob("*_agent_info.csv"))
    copy_paths = list(rocprof.glob("*_memory_copy_trace.csv"))
    if len(kernel_paths) != 1 or len(agent_paths) != 1:
        raise ValueError("expected exactly one kernel trace and one agent-info CSV")

    agents: dict[str, dict[str, str]] = {}
    known_agent_ids: set[str] = set()
    with agent_paths[0].open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Logical_Node_Id", "Agent_Type", "Domain", "Location_Id", "Gpu_Id", "Product_Name"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"agent-info CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                agent = f"Agent {int(row['Logical_Node_Id'])}"
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{agent_paths[0]}:{reader.line_num}: invalid agent logical node") from exc
            if agent in known_agent_ids:
                raise ValueError(f"duplicate logical node in agent-info CSV: {agent}")
            known_agent_ids.add(agent)
            if row["Agent_Type"] != "GPU":
                continue
            try:
                pci_bdf = bdf_from_location(row["Domain"], row["Location_Id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{agent_paths[0]}:{reader.line_num}: invalid GPU identity") from exc
            agents[agent] = {
                "logical_node_id": row["Logical_Node_Id"],
                "domain": row["Domain"],
                "location_id": row["Location_Id"],
                "pci_bdf": pci_bdf,
                "gpu_id": row["Gpu_Id"],
                "product_name": row["Product_Name"],
            }
    if not agents:
        raise ValueError("agent-info CSV contains no GPU agents")

    # These are deliberately name-match families, not operation-level proof.
    # Several operations can share a generic kernel name.
    category_tests: list[tuple[str, Callable[[str], bool]]] = [
        ("mul_mat_q_name_match", lambda name: "mul_mat_q<" in name),
        ("dsv4_hc_mixes_custom_name_match", lambda name: "dsv4_hc_mixes_f32" in name),
        ("rocblas_mt128x256x16_name_match", lambda name: name.startswith("Cijk_Alik_Bljk_SB_MT128x256x16_")),
        ("lightning_indexer_name_match", lambda name: "lightning_indexer_kernel" in name),
        ("flash_attn_tile_name_match", lambda name: "flash_attn_tile" in name),
        ("nccl_dev_kernel_name_match", lambda name: name.startswith("ncclDevKernel")),
    ]
    # category -> agent -> [intersecting events, summed clipped duration]
    totals: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    raw_kernel_rows = 0
    intersecting_kernel_rows = 0
    with kernel_paths[0].open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Agent_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"kernel trace is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            raw_kernel_rows += 1
            interval = clipped_interval(
                row, start_ns, end_ns, f"{kernel_paths[0]}:{reader.line_num}"
            )
            if interval is None:
                continue
            intersecting_kernel_rows += 1
            agent = row["Agent_Id"]
            if agent not in agents:
                raise ValueError(f"kernel trace references unknown GPU agent: {agent}")
            duration = interval[1] - interval[0]
            totals["all_kernels"][agent][0] += 1
            totals["all_kernels"][agent][1] += duration
            for category, test in category_tests:
                if test(row["Kernel_Name"]):
                    totals[category][agent][0] += 1
                    totals[category][agent][1] += duration
    if intersecting_kernel_rows == 0:
        raise ValueError("kernel trace has no events intersecting the measured interval")

    copies: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    raw_copy_rows = 0
    intersecting_copy_rows = 0
    if len(copy_paths) > 1:
        raise ValueError("expected at most one memory-copy trace CSV")
    if copy_paths:
        with copy_paths[0].open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Direction", "Source_Agent_Id", "Destination_Agent_Id", "Start_Timestamp", "End_Timestamp"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"memory-copy trace is missing columns: {', '.join(sorted(missing))}")
            for row in reader:
                raw_copy_rows += 1
                interval = clipped_interval(
                    row, start_ns, end_ns, f"{copy_paths[0]}:{reader.line_num}"
                )
                if interval is None:
                    continue
                intersecting_copy_rows += 1
                direction = row["Direction"]
                source = row["Source_Agent_Id"]
                destination = row["Destination_Agent_Id"]
                if not direction or not source or not destination:
                    raise ValueError(f"{copy_paths[0]}:{reader.line_num}: blank copy direction or endpoint")
                if source not in known_agent_ids or destination not in known_agent_ids:
                    raise ValueError(
                        f"{copy_paths[0]}:{reader.line_num}: copy references an unknown agent: "
                        f"{source}->{destination}"
                    )
                key = f"{direction}:{source}->{destination}"
                copies[key][0] += 1
                copies[key][1] += interval[1] - interval[0]

    copy_trace_status = (
        "trace_absent" if not copy_paths else
        "events_present" if intersecting_copy_rows else
        "trace_present_no_overlap"
    )
    rendered: dict[str, object] = {
        "run_dir": str(run_dir),
        "measurement_monotonic_ns": start_ns,
        "completed_monotonic_ns": end_ns,
        "wall_ns": wall_ns,
        "agents": agents,
        "kernel_trace_rows": raw_kernel_rows,
        "intersecting_kernel_rows": intersecting_kernel_rows,
        "kernel_categories": {},
        "memory_copy_trace_status": copy_trace_status,
        "memory_copy_trace_rows": raw_copy_rows,
        "intersecting_memory_copy_rows": intersecting_copy_rows,
        "memory_copy_paths": {},
        "caveat": "Name-match categories overlap and are not operation-level proof. Durations are clipped then summed per agent across all queues; overlapping events can make sums exceed wall time and do not alone prove PCIe causality. Calls count intersecting events, including boundary-clipped events.",
    }
    categories_out = rendered["kernel_categories"]
    assert isinstance(categories_out, dict)
    for category in ["all_kernels", *(name for name, _ in category_tests)]:
        category_rows: dict[str, object] = {}
        for agent in sorted(agents):
            calls, duration_ns = totals[category][agent]
            category_rows[agent] = {
                "calls": calls,
                "duration_ns": duration_ns,
                "mean_ns": duration_ns / calls if calls else None,
                "wall_equivalent_pct": 100 * duration_ns / wall_ns,
            }
        categories_out[category] = category_rows
    copy_out = rendered["memory_copy_paths"]
    assert isinstance(copy_out, dict)
    for key, (calls, duration_ns) in sorted(copies.items()):
        copy_out[key] = {"calls": calls, "duration_ns": duration_ns}

    print(f"run_dir={run_dir}")
    print(f"wall_ms={wall_ns / 1e6:.3f}")
    print("agent\tpci_bdf\tcategory\tcalls\tduration_ms\tmean_us\twall_equivalent_pct")
    for category in ["all_kernels", *(name for name, _ in category_tests)]:
        for agent in sorted(agents):
            item = categories_out[category][agent]
            mean_ns = item["mean_ns"]
            print(
                f"{agent}\t{agents[agent]['pci_bdf']}\t{category}\t{item['calls']}\t"
                f"{item['duration_ns'] / 1e6:.3f}\t"
                f"{mean_ns / 1e3 if mean_ns is not None else 0:.3f}\t{item['wall_equivalent_pct']:.2f}"
            )
    print(f"\n[memory copy paths: {copy_trace_status}]")
    if copy_trace_status == "trace_absent":
        print("memory-copy trace is absent; no statement about measured copies is possible")
    elif copy_trace_status == "trace_present_no_overlap":
        print("memory-copy trace is present but has no events intersecting the measured interval")
    for key, item in copy_out.items():
        print(f"{key}\t{item['calls']}\t{item['duration_ns'] / 1e6:.3f} ms")
    print(f"\ncaveat: {rendered['caveat']}")

    if args.json:
        args.json.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, csv.Error, json.JSONDecodeError) as exc:
        import sys
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)