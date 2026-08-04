#!/usr/bin/env python3
"""Parse GGML_SCHED_DEBUG=2 assignments for DSV4 raw-decode graphs."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BENCH_RE = re.compile(r"llama-bench: benchmark\s+(\d+)/(\d+):\s+starting")
DEPTH_RE = re.compile(r"llama-bench: benchmark\s+(\d+)/(\d+):\s+depth run\s+(\d+)/(\d+)")
GEN_RE = re.compile(r"llama-bench: benchmark\s+(\d+)/(\d+):\s+generation run\s+(\d+)/(\d+)")
SPLIT_RE = re.compile(r"## SPLIT #\d+:\s+(.+?)\s+#\s+(\d+)\s+inputs")
BACKEND_USE_RE = re.compile(r"\[\s*([^\s\]]+)\s+[^\]]+\]\s+use=")
BACKEND_RE = re.compile(r"\[\s*([^\s\]]+)")
TOP_K_NODE_RE = re.compile(r"node\s+#\s*\d+\s+\(\s*TOP_K\s*\):")
LID_NODE_RE = re.compile(r"node\s+#\s*\d+\s+\(\s*LIGHTNING[^)]*\):")


def csv_nonnegative_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected unique comma-separated non-negative integers")
    return values


def target_backend(name: str) -> bool:
    return name.startswith("ROCm") or name.startswith("Meta(") or name == "Meta"


def cpu_backend(name: str) -> bool:
    return name.startswith("CPU")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--depths", type=csv_nonnegative_ints, required=True)
    parser.add_argument("--json", dest="json_out", type=pathlib.Path, required=True)
    parser.add_argument("--tsv", dest="tsv_out", type=pathlib.Path, required=True)
    parser.add_argument("--require-top-k-from", type=int, default=3072)
    parser.add_argument("--expected-nodes", type=int, default=21,
                        help="expected real TOP_K and LIGHTNING_INDEXER nodes per measured graph")
    args = parser.parse_args()

    if args.require_top_k_from < 0:
        raise ValueError("require-top-k-from must be non-negative")
    if args.expected_nodes <= 0:
        raise ValueError("expected-nodes must be positive")
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    depths = args.depths
    per_depth: dict[int, dict[str, Any]] = {}
    for depth in depths:
        per_depth[depth] = {
            "depth": depth,
            "benchmark_indices": [],
            "decode_graphs": 0,
            "split_backends": collections.Counter(),
            "split_input_copies": collections.Counter(),
            "top_k_backends": collections.Counter(),
            "lid_backends": collections.Counter(),
            "top_k_examples": [],
            "lid_examples": [],
        }

    current_bench: int | None = None
    current_depth: int | None = None
    phase = "outside"
    total_benchmarks: int | None = None
    parse_warnings: list[str] = []

    for line_number, raw_line in enumerate(lines, 1):
        line = ANSI_RE.sub("", raw_line)
        match = BENCH_RE.search(line)
        if match:
            current_bench = int(match.group(1))
            total_benchmarks = int(match.group(2))
            current_depth = depths[current_bench - 1] if 1 <= current_bench <= len(depths) else None
            phase = "setup"
            if current_depth is None:
                parse_warnings.append(f"line {line_number}: benchmark index {current_bench} has no depth mapping")
            else:
                per_depth[current_depth]["benchmark_indices"].append(current_bench)
            continue
        if DEPTH_RE.search(line):
            phase = "setup"
            continue
        if GEN_RE.search(line):
            phase = "decode"
            if current_depth is not None:
                per_depth[current_depth]["decode_graphs"] += 1
            continue
        if phase != "decode" or current_depth is None:
            continue

        split = SPLIT_RE.search(line)
        if split:
            backend = split.group(1).strip()
            per_depth[current_depth]["split_backends"][backend] += 1
            per_depth[current_depth]["split_input_copies"][backend] += int(split.group(2))
            continue

        # Count only the actual operation line. Tensor names such as lid_top_k
        # also occur on CONT/SET_ROWS consumers and must not inflate attestation.
        is_top_k = TOP_K_NODE_RE.search(line) is not None
        is_lid = LID_NODE_RE.search(line) is not None
        if not is_top_k and not is_lid:
            continue
        backend_match = BACKEND_USE_RE.search(line) or BACKEND_RE.search(line)
        backend = backend_match.group(1).strip() if backend_match else "UNPARSED"
        key = "top_k_backends" if is_top_k else "lid_backends"
        examples_key = "top_k_examples" if is_top_k else "lid_examples"
        per_depth[current_depth][key][backend] += 1
        if len(per_depth[current_depth][examples_key]) < 5:
            per_depth[current_depth][examples_key].append({"line": line_number, "backend": backend, "text": line})

    records: list[dict[str, Any]] = []
    complete = True
    residency_ok = True
    for depth in depths:
        item = per_depth[depth]
        split_counts = dict(sorted(item["split_backends"].items()))
        split_input_copies = dict(sorted(item["split_input_copies"].items()))
        top_k_counts = dict(sorted(item["top_k_backends"].items()))
        lid_counts = dict(sorted(item["lid_backends"].items()))
        decode_graphs = item["decode_graphs"]
        cpu_splits = sum(count for name, count in split_counts.items() if cpu_backend(name))
        gpu_splits = sum(count for name, count in split_counts.items() if target_backend(name))
        unknown_splits = sum(count for name, count in split_counts.items() if not cpu_backend(name) and not target_backend(name))
        cpu_split_input_copies = sum(count for name, count in split_input_copies.items() if cpu_backend(name))
        gpu_split_input_copies = sum(count for name, count in split_input_copies.items() if target_backend(name))
        unknown_split_input_copies = sum(count for name, count in split_input_copies.items() if not cpu_backend(name) and not target_backend(name))
        top_k_total = sum(top_k_counts.values())
        lid_total = sum(lid_counts.values())
        top_k_cpu = sum(count for name, count in top_k_counts.items() if cpu_backend(name))
        lid_cpu = sum(count for name, count in lid_counts.items() if cpu_backend(name))
        top_k_unknown = sum(count for name, count in top_k_counts.items() if not cpu_backend(name) and not target_backend(name))
        lid_unknown = sum(count for name, count in lid_counts.items() if not cpu_backend(name) and not target_backend(name))
        expected_node_total = args.expected_nodes * decode_graphs
        if depth >= args.require_top_k_from:
            required_nodes_present = top_k_total == expected_node_total and lid_total == expected_node_total
        else:
            # Below the selector crossover, either both operation families are
            # absent or the graph must contain their complete expected counts.
            required_nodes_present = (
                top_k_total == lid_total
                and top_k_total in (0, expected_node_total)
            )
        if decode_graphs == 1 and not required_nodes_present:
            parse_warnings.append(
                f"depth {depth}: expected 0 or {expected_node_total} nodes below crossover "
                f"and exactly {expected_node_total} at/above it; observed "
                f"TOP_K={top_k_total}, LIGHTNING_INDEXER={lid_total}"
            )
        row_complete = decode_graphs == 1 and bool(split_counts) and required_nodes_present
        row_resident = row_complete and top_k_cpu == 0 and lid_cpu == 0 and top_k_unknown == 0 and lid_unknown == 0
        complete = complete and row_complete
        residency_ok = residency_ok and row_resident
        records.append({
            "depth": depth,
            "benchmark_indices": item["benchmark_indices"],
            "decode_graphs": decode_graphs,
            "split_backends": split_counts,
            "cpu_splits": cpu_splits,
            "gpu_meta_splits": gpu_splits,
            "unknown_splits": unknown_splits,
            "split_input_copies_by_backend": split_input_copies,
            "total_split_input_copies": sum(split_input_copies.values()),
            "cpu_split_input_copies": cpu_split_input_copies,
            "gpu_meta_split_input_copies": gpu_split_input_copies,
            "unknown_split_input_copies": unknown_split_input_copies,
            "top_k_backends": top_k_counts,
            "top_k_total": top_k_total,
            "top_k_cpu": top_k_cpu,
            "top_k_unknown": top_k_unknown,
            "lid_backends": lid_counts,
            "lid_total": lid_total,
            "lid_cpu": lid_cpu,
            "lid_unknown": lid_unknown,
            "expected_nodes_per_graph": args.expected_nodes,
            "expected_node_total": expected_node_total,
            "required_nodes_present": required_nodes_present,
            "complete": row_complete,
            "rocm_resident": row_resident,
            "top_k_examples": item["top_k_examples"],
            "lid_examples": item["lid_examples"],
        })

    if total_benchmarks != len(depths):
        complete = False
        residency_ok = False
        parse_warnings.append(f"progress reported {total_benchmarks} benchmarks, expected {len(depths)}")

    document = {
        "complete": complete,
        "rocm_residency_ok": complete and residency_ok,
        "expected_depths": depths,
        "reported_benchmark_count": total_benchmarks,
        "require_top_k_from": args.require_top_k_from,
        "expected_nodes_per_graph": args.expected_nodes,
        "parse_warnings": parse_warnings,
        "records": records,
    }
    args.json_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    columns = [
        "run_complete", "residency_ok", "depth", "decode_graphs", "cpu_splits",
        "gpu_meta_splits", "unknown_splits", "total_split_input_copies",
        "cpu_split_input_copies", "gpu_meta_split_input_copies",
        "unknown_split_input_copies", "top_k_total", "top_k_cpu",
        "top_k_unknown", "lid_total", "lid_cpu", "lid_unknown",
    ]
    output = ["\t".join(columns)]
    for row in records:
        values: dict[str, Any] = {"run_complete": complete, "residency_ok": document["rocm_residency_ok"], **row}
        rendered = []
        for column in columns:
            value = values[column]
            rendered.append("1" if value is True else "0" if value is False else str(value))
        output.append("\t".join(rendered))
    args.tsv_out.write_text("\n".join(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)