#!/usr/bin/env python3
"""Forensic NCCL analysis for selected-region DSV4 raw-decode profiles.

This tool analyzes preserved traces only. It does not treat profiler wall timing as
throughput evidence and cannot recover RCCL message arguments that rocprof did not
record.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


EXPECTED_BLOCK_COUNT = 43
EXPECTED_ALLREDUCES_PER_BLOCK_TOKEN = 2
EXPECTED_GPU_AGENTS = 4
EXPECTED_N_GEN = 32
EXPECTED_RCCL_FUNCTIONS = {
    "ncclAllReduce",
    "ncclCommGetAsyncError",
    "ncclGroupEnd",
    "ncclGroupStart",
}
EXPECTED_RCCL_SCHEMA = {
    "Domain", "Function", "Process_Id", "Thread_Id", "Correlation_Id", "Start_Timestamp", "End_Timestamp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NCCL cadence and long device kernels in DSV4 selected-region profiles")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--top-long-kernels", type=int, default=10)
    return parser.parse_args()


def exactly_one(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern} under {directory}, found {len(paths)}")
    return paths[0]


def percentile(values: list[int], fraction: float) -> int:
    if not values or not 0 < fraction <= 1:
        raise ValueError("percentile requires values and 0 < fraction <= 1")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError(f"interval ends before it starts: {start}-{end}")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def interval_duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_duration(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    a, b = merge_intervals(left), merge_intervals(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def containing_repetition(intervals: list[tuple[int, int, int]], start: int, end: int) -> int:
    matches = [rep for left, right, rep in intervals if start >= left and end <= right]
    if len(matches) != 1:
        raise ValueError(f"event {start}-{end} belongs to {len(matches)} accepted repetitions")
    return matches[0]


def require_exact_counts(actual: Counter[str], expected: dict[str, int], context: str) -> None:
    if dict(actual) != expected:
        raise ValueError(f"{context} count mismatch: actual={dict(actual)} expected={expected}")


def analyze_run(run_dir: Path, top_long_kernels: int) -> dict[str, object]:
    run_dir = run_dir.resolve()
    contract = json.loads((run_dir / "contract.json").read_text())
    summary = json.loads((run_dir / "profile-summary.json").read_text())
    if not summary.get("complete") or not summary.get("family_attribution_complete"):
        raise ValueError(f"{run_dir}: profile summary is incomplete")
    if summary.get("profiled_throughput_eligible") or summary.get("csa_decision_eligible"):
        raise ValueError(f"{run_dir}: forensic profile unexpectedly claims throughput/CSA eligibility")
    outside = summary.get("outside_selected_interval_events")
    expected_outside_keys = {"kernel", "memory_copy", "rccl", "hip"}
    if not isinstance(outside, dict) or set(outside) != expected_outside_keys:
        raise ValueError(f"{run_dir}: missing or malformed outside-selected-interval event counts: {outside}")
    if any(not isinstance(value, int) or isinstance(value, bool) or value != 0 for value in outside.values()):
        raise ValueError(f"{run_dir}: trace contains events outside selected regions: {outside}")
    if contract.get("profile") != "kernel" or contract.get("n_gen") != EXPECTED_N_GEN:
        raise ValueError(f"{run_dir}: requires selected-region tg{EXPECTED_N_GEN} profile")
    if int(summary.get("moe_dispatch_contract", {}).get("block_count", 0)) != EXPECTED_BLOCK_COUNT:
        raise ValueError(f"{run_dir}: requires exact {EXPECTED_BLOCK_COUNT}-block DSV4 profile")
    if int(summary.get("moe_dispatch_contract", {}).get("gpu_agents", 0)) != EXPECTED_GPU_AGENTS:
        raise ValueError(f"{run_dir}: requires exact four-GPU profile")

    intervals = [
        (int(item["start"]), int(item["end"]), int(item["repetition"]))
        for item in summary["accepted_intervals_monotonic_ns"]
    ]
    expected_reps = list(range(int(contract["discard_first"]) + 1, int(contract["raw_repetitions"]) + 1))
    if [rep for _, _, rep in intervals] != expected_reps:
        raise ValueError(f"{run_dir}: accepted repetition identity mismatch")
    if int(summary.get("profiled_repetitions", -1)) != len(expected_reps):
        raise ValueError(f"{run_dir}: profiled repetition count does not match accepted intervals")
    summary_repetitions = summary.get("per_repetition")
    if not isinstance(summary_repetitions, list):
        raise ValueError(f"{run_dir}: missing per-repetition profile records")
    repetition_wall_ns: dict[int, int] = {}
    for item in summary_repetitions:
        if not isinstance(item, dict) or "repetition" not in item or "wall_ns" not in item:
            raise ValueError(f"{run_dir}: malformed per-repetition profile record: {item}")
        rep = int(item["repetition"])
        wall_ns = int(item["wall_ns"])
        if rep in repetition_wall_ns or wall_ns <= 0:
            raise ValueError(f"{run_dir}: duplicate repetition or invalid wall time for repetition {rep}")
        repetition_wall_ns[rep] = wall_ns
    if set(repetition_wall_ns) != set(expected_reps):
        raise ValueError(f"{run_dir}: per-repetition identities do not match accepted intervals")

    rocprof = run_dir / "rocprof"
    kernel_path = exactly_one(rocprof, "*_kernel_trace.csv")
    rccl_path = exactly_one(rocprof, "*_rccl_api_trace.csv")
    agent_path = exactly_one(rocprof, "*_agent_info.csv")

    agents: set[str] = set()
    agent_metadata: dict[str, dict[str, str]] = {}
    with agent_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Agent_Type") == "GPU":
                agent = f"Agent {int(row['Logical_Node_Id'])}"
                agents.add(agent)
                agent_metadata[agent] = {"product_name": row["Product_Name"], "gpu_id": row["Gpu_Id"]}
    if len(agents) != EXPECTED_GPU_AGENTS:
        raise ValueError(f"{run_dir}: found {len(agents)} GPU agents")

    nccl_events: list[dict[str, object]] = []
    compute_intervals: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    kernel_fields: set[str]
    with kernel_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        kernel_fields = set(reader.fieldnames or [])
        required = {
            "Agent_Id", "Kernel_Name", "Correlation_Id", "Start_Timestamp", "End_Timestamp",
            "Queue_Id", "Stream_Id",
        }
        missing = required.difference(kernel_fields)
        if missing:
            raise ValueError(f"{kernel_path} missing fields: {sorted(missing)}")
        for row in reader:
            start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            if end < start:
                raise ValueError(f"{kernel_path}:{reader.line_num} ends before it starts")
            rep = containing_repetition(intervals, start, end)
            agent = row["Agent_Id"]
            if agent not in agents:
                raise ValueError(f"{kernel_path}:{reader.line_num} references unknown {agent}")
            if row["Kernel_Name"].startswith("ncclDevKernel"):
                nccl_events.append({
                    "repetition": rep, "agent": agent, "queue_id": row["Queue_Id"],
                    "stream_id": row["Stream_Id"], "correlation_id": str(int(row["Correlation_Id"])),
                    "kernel_name": row["Kernel_Name"], "start_ns": start, "end_ns": end,
                    "duration_ns": end - start,
                })
            else:
                compute_intervals[(rep, agent)].append((start, end))
    if not nccl_events:
        raise ValueError(f"{run_dir}: no NCCL device kernels")

    rccl_rows: list[dict[str, object]] = []
    rccl_fields: set[str]
    with rccl_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rccl_fields = set(reader.fieldnames or [])
        if rccl_fields != EXPECTED_RCCL_SCHEMA:
            raise ValueError(
                f"{rccl_path} schema differs from the exact supported rocprof RCCL schema: "
                f"actual={sorted(rccl_fields)} expected={sorted(EXPECTED_RCCL_SCHEMA)}"
            )
        for row in reader:
            start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            if end < start:
                raise ValueError(f"{rccl_path}:{reader.line_num} ends before it starts")
            rep = containing_repetition(intervals, start, end)
            rccl_rows.append({
                "repetition": rep, "function": row["Function"], "correlation_id": str(int(row["Correlation_Id"])),
                "start_ns": start, "end_ns": end, "duration_ns": end - start,
            })
    functions = {str(row["function"]) for row in rccl_rows}
    if functions != EXPECTED_RCCL_FUNCTIONS:
        raise ValueError(f"{run_dir}: RCCL functions {sorted(functions)} != {sorted(EXPECTED_RCCL_FUNCTIONS)}")

    groups_per_token = EXPECTED_BLOCK_COUNT * EXPECTED_ALLREDUCES_PER_BLOCK_TOKEN
    expected_groups_per_rep = EXPECTED_N_GEN * groups_per_token
    expected_rank_calls_per_rep = expected_groups_per_rep * EXPECTED_GPU_AGENTS
    expected_api_counts = {
        "ncclAllReduce": expected_rank_calls_per_rep,
        "ncclCommGetAsyncError": expected_rank_calls_per_rep,
        "ncclGroupEnd": expected_groups_per_rep,
        "ncclGroupStart": expected_groups_per_rep,
    }
    expected_kernel_count = expected_rank_calls_per_rep

    repetitions: list[dict[str, object]] = []
    all_durations = [int(item["duration_ns"]) for item in nccl_events]
    for rep in expected_reps:
        rep_api = [item for item in rccl_rows if item["repetition"] == rep]
        require_exact_counts(Counter(str(item["function"]) for item in rep_api), expected_api_counts, f"repetition {rep} RCCL API")
        rep_nccl = [item for item in nccl_events if item["repetition"] == rep]
        if len(rep_nccl) != expected_kernel_count:
            raise ValueError(f"repetition {rep} NCCL kernel count {len(rep_nccl)} != {expected_kernel_count}")
        per_agent_counts = Counter(str(item["agent"]) for item in rep_nccl)
        if set(per_agent_counts) != agents or any(value != expected_groups_per_rep for value in per_agent_counts.values()):
            raise ValueError(
                f"repetition {rep} NCCL per-agent count mismatch: {dict(per_agent_counts)} "
                f"expected_each={expected_groups_per_rep}"
            )
        per_agent: dict[str, dict[str, object]] = {}
        for agent in sorted(agents):
            agent_nccl = [item for item in rep_nccl if item["agent"] == agent]
            nccl_intervals = [(int(item["start_ns"]), int(item["end_ns"])) for item in agent_nccl]
            compute = compute_intervals[(rep, agent)]
            nccl_union_ns = interval_duration(merge_intervals(nccl_intervals))
            overlap_ns = intersection_duration(nccl_intervals, compute)
            per_agent[agent] = {
                "kernels": len(agent_nccl),
                "summed_device_ns": sum(int(item["duration_ns"]) for item in agent_nccl),
                "union_device_ns": nccl_union_ns,
                "overlap_with_non_nccl_union_ns": overlap_ns,
                "overlap_fraction_of_nccl_union": overlap_ns / nccl_union_ns if nccl_union_ns else 0.0,
                "stream_ids": sorted({str(item["stream_id"]) for item in agent_nccl}),
                "queue_ids": sorted({str(item["queue_id"]) for item in agent_nccl}),
            }
        rep_durations = [int(item["duration_ns"]) for item in rep_nccl]
        repetitions.append({
            "repetition": rep,
            "wall_ns": repetition_wall_ns[rep],
            "api_counts": dict(Counter(str(item["function"]) for item in rep_api)),
            "kernel_count": len(rep_nccl),
            "summed_device_ns": sum(rep_durations),
            "median_kernel_ns": int(statistics.median(rep_durations)),
            "p95_kernel_ns": percentile(rep_durations, 0.95),
            "p99_kernel_ns": percentile(rep_durations, 0.99),
            "max_kernel_ns": max(rep_durations),
            "per_agent": per_agent,
        })

    api_correlation_ids = {str(item["correlation_id"]) for item in rccl_rows}
    kernel_correlation_ids = {str(item["correlation_id"]) for item in nccl_events}
    correlation_intersection = api_correlation_ids.intersection(kernel_correlation_ids)
    longest = sorted(nccl_events, key=lambda item: (-int(item["duration_ns"]), int(item["start_ns"])))[:top_long_kernels]
    api_stats: dict[str, dict[str, int]] = {}
    for function in sorted(functions):
        durations = [int(item["duration_ns"]) for item in rccl_rows if item["function"] == function]
        api_stats[function] = {
            "calls": len(durations), "total_duration_ns": sum(durations),
            "median_duration_ns": int(statistics.median(durations)), "max_duration_ns": max(durations),
        }

    kernel_names = sorted({str(item["kernel_name"]) for item in nccl_events})
    result: dict[str, object] = {
        "complete": True,
        "run_dir": str(run_dir),
        "depth": int(summary["depth"]),
        "profiled_repetitions": int(summary["profiled_repetitions"]),
        "profiled_wall_stable": bool(summary["profiled_wall_stable"]),
        "profiled_throughput_eligible": False,
        "csa_decision_eligible": False,
        "traced_collective_api_functions": ["ncclAllReduce"],
        "groups_per_token": groups_per_token,
        "rank_allreduce_calls_per_token": groups_per_token * EXPECTED_GPU_AGENTS,
        "rank_allreduce_calls_per_gpu_token": groups_per_token,
        "expected_counts_per_repetition": {**expected_api_counts, "nccl_device_kernels": expected_kernel_count},
        "count_contract_complete": True,
        "gpu_agents": {agent: agent_metadata[agent] for agent in sorted(agents)},
        "kernel_names": kernel_names,
        "kernel_duration_ns": {
            "count": len(all_durations), "median": int(statistics.median(all_durations)),
            "p95": percentile(all_durations, 0.95), "p99": percentile(all_durations, 0.99),
            "max": max(all_durations), "summed": sum(all_durations),
        },
        "rccl_api": api_stats,
        "message_metadata": {
            "available": False,
            "supported_rccl_schema_exact": True,
            "reason": (
                "the exact supported rocprof RCCL API schema contains no count/datatype/buffer/communicator/rank/stream/"
                "message-byte arguments; generic NCCL device-kernel launch geometry is not an attested payload size"
            ),
        },
        "api_kernel_correlation": {
            "available": bool(correlation_intersection),
            "shared_correlation_ids": len(correlation_intersection),
            "reason": "RCCL API and NCCL device-kernel correlation-id sets are disjoint in this trace",
        },
        "longest_device_kernels": longest,
        "repetitions": repetitions,
        "critical_path_proven": False,
        "overlap_caveat": (
            "Timestamp interval intersection proves temporal overlap on traced queues only; it does not prove link traffic, "
            "hardware-resource overlap, graph dependencies, rank causality, or end-to-end critical-path membership."
        ),
    }
    if correlation_intersection:
        raise ValueError(f"{run_dir}: correlation semantics changed; shared RCCL/kernel IDs require explicit validation")
    return result


def main() -> int:
    args = parse_args()
    if args.top_long_kernels < 1:
        raise ValueError("--top-long-kernels must be positive")
    resolved_paths = [path.resolve() for path in args.run_dirs]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("run directories must be distinct")
    runs = [analyze_run(path, args.top_long_kernels) for path in resolved_paths]
    cadence = {
        (
            int(run["groups_per_token"]), int(run["rank_allreduce_calls_per_token"]),
            int(run["rank_allreduce_calls_per_gpu_token"]),
        )
        for run in runs
    }
    if len(cadence) != 1:
        raise ValueError(f"communication cadence differs across runs: {sorted(cadence)}")
    output = {
        "complete": True,
        "scope": "forensic selected-region NCCL cadence/long device kernels; no throughput or critical-path acceptance",
        "runs": runs,
        "cross_run_cadence_invariant": len(runs) >= 2,
        "profiled_throughput_eligible": False,
        "csa_decision_eligible": False,
        "critical_path_proven": False,
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.json:
        temporary = args.json.with_name(args.json.name + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.json)

    print("M5.3 DSV4 NCCL FORENSICS: COMPLETE (CRITICAL PATH NOT PROVEN)")
    print("depth\treps\tgroups/token\trank_calls/token\tmedian_us\tp95_us\tp99_us\tmax_ms\tsummed_ms")
    for run in runs:
        duration = run["kernel_duration_ns"]
        print(
            f"{run['depth']}\t{run['profiled_repetitions']}\t{run['groups_per_token']}\t"
            f"{run['rank_allreduce_calls_per_token']}\t{duration['median']/1e3:.3f}\t"
            f"{duration['p95']/1e3:.3f}\t{duration['p99']/1e3:.3f}\t"
            f"{duration['max']/1e6:.3f}\t{duration['summed']/1e6:.3f}"
        )
    print("traced_collective_api=ncclAllReduce_only message_metadata_available=0 api_kernel_correlation_available=0")
    print(
        f"cross_run_cadence_invariant={int(output['cross_run_cadence_invariant'])} "
        "critical_path_proven=0 profiled_throughput_eligible=0 csa_decision_eligible=0"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)