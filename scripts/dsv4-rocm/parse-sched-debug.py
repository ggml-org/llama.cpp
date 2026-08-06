#!/usr/bin/env python3
"""Parse and strictly attest GGML_SCHED_DEBUG=2 DSV4 decode graphs."""

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
SPLIT_RE = re.compile(r"## SPLIT #(\d+):\s+(.+?)\s+#\s+(\d+)\s+inputs")
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
    parser.add_argument("--require-top-k-from", type=int, default=2048)
    parser.add_argument("--expected-nodes", type=int, default=21,
                        help="expected real TOP_K and LIGHTNING_INDEXER nodes per measured graph")
    parser.add_argument("--expected-meta-backend", default="Meta(ROCm0,ROCm1,ROCm2,ROCm3)")
    parser.add_argument("--expected-depth-zero-inputs", type=int, default=22)
    parser.add_argument("--expected-depth-inputs", type=int, default=25)
    args = parser.parse_args()

    if args.require_top_k_from < 0:
        raise ValueError("require-top-k-from must be non-negative")
    if args.expected_nodes <= 0:
        raise ValueError("expected-nodes must be positive")
    if args.expected_depth_zero_inputs < 0 or args.expected_depth_inputs < 0:
        raise ValueError("expected split input counts must be non-negative")

    depths = args.depths
    per_depth: dict[int, dict[str, Any]] = {}
    for depth in depths:
        per_depth[depth] = {
            "depth": depth,
            "benchmark_indices": [],
            "depth_markers": [],
            "generation_markers": [],
            "decode_graphs": 0,
            "split_records": [],
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
    benchmark_sequence: list[tuple[int, int]] = []
    parse_warnings: list[str] = []

    with args.log.open(encoding="utf-8", errors="replace") as source:
        for line_number, raw_line in enumerate(source, 1):
            line = ANSI_RE.sub("", raw_line.rstrip("\n"))
            match = BENCH_RE.search(line)
            if match:
                current_bench, total = map(int, match.groups())
                benchmark_sequence.append((current_bench, total))
                current_depth = depths[current_bench - 1] if 1 <= current_bench <= len(depths) else None
                phase = "setup"
                if current_depth is None:
                    parse_warnings.append(f"line {line_number}: benchmark index {current_bench} has no depth mapping")
                else:
                    per_depth[current_depth]["benchmark_indices"].append(current_bench)
                continue

            match = DEPTH_RE.search(line)
            if match:
                marker = tuple(map(int, match.groups()))
                phase = "setup"
                if current_depth is None:
                    parse_warnings.append(f"line {line_number}: depth marker outside a mapped benchmark")
                else:
                    per_depth[current_depth]["depth_markers"].append(marker)
                continue

            match = GEN_RE.search(line)
            if match:
                marker = tuple(map(int, match.groups()))
                phase = "decode"
                if current_depth is None:
                    parse_warnings.append(f"line {line_number}: generation marker outside a mapped benchmark")
                else:
                    per_depth[current_depth]["generation_markers"].append(marker)
                    per_depth[current_depth]["decode_graphs"] += 1
                continue

            if phase != "decode" or current_depth is None:
                continue

            split = SPLIT_RE.search(line)
            if split:
                split_index = int(split.group(1))
                backend = split.group(2).strip()
                inputs = int(split.group(3))
                per_depth[current_depth]["split_records"].append({
                    "index": split_index, "backend": backend, "inputs": inputs,
                })
                per_depth[current_depth]["split_backends"][backend] += 1
                per_depth[current_depth]["split_input_copies"][backend] += inputs
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
                per_depth[current_depth][examples_key].append({
                    "line": line_number, "backend": backend, "text": line,
                })

    expected_benchmark_sequence = [(index, len(depths)) for index in range(1, len(depths) + 1)]
    benchmark_sequence_ok = benchmark_sequence == expected_benchmark_sequence
    if not benchmark_sequence_ok:
        parse_warnings.append(
            f"benchmark sequence mismatch: observed {benchmark_sequence}, expected {expected_benchmark_sequence}"
        )

    records: list[dict[str, Any]] = []
    complete = benchmark_sequence_ok
    residency_ok = benchmark_sequence_ok
    for index, depth in enumerate(depths, 1):
        item = per_depth[depth]
        split_counts = dict(sorted(item["split_backends"].items()))
        split_input_copies = dict(sorted(item["split_input_copies"].items()))
        top_k_counts = dict(sorted(item["top_k_backends"].items()))
        lid_counts = dict(sorted(item["lid_backends"].items()))
        decode_graphs = item["decode_graphs"]

        expected_depth_markers = [] if depth == 0 else [(index, len(depths), 1, 1)]
        expected_generation_markers = [(index, len(depths), 1, 1)]
        marker_structure_ok = (
            item["benchmark_indices"] == [index]
            and item["depth_markers"] == expected_depth_markers
            and item["generation_markers"] == expected_generation_markers
            and decode_graphs == 1
        )

        expected_meta_inputs = args.expected_depth_zero_inputs if depth == 0 else args.expected_depth_inputs
        expected_splits = [
            {"index": 0, "backend": "CPU", "inputs": 0},
            {"index": 1, "backend": args.expected_meta_backend, "inputs": expected_meta_inputs},
        ]
        split_structure_ok = item["split_records"] == expected_splits

        cpu_splits = sum(count for name, count in split_counts.items() if cpu_backend(name))
        gpu_splits = sum(count for name, count in split_counts.items() if target_backend(name))
        unknown_splits = sum(count for name, count in split_counts.items()
                             if not cpu_backend(name) and not target_backend(name))
        cpu_split_input_copies = sum(count for name, count in split_input_copies.items() if cpu_backend(name))
        gpu_split_input_copies = sum(count for name, count in split_input_copies.items() if target_backend(name))
        unknown_split_input_copies = sum(count for name, count in split_input_copies.items()
                                         if not cpu_backend(name) and not target_backend(name))

        top_k_total = sum(top_k_counts.values())
        lid_total = sum(lid_counts.values())
        top_k_cpu = sum(count for name, count in top_k_counts.items() if cpu_backend(name))
        lid_cpu = sum(count for name, count in lid_counts.items() if cpu_backend(name))
        top_k_unknown = sum(count for name, count in top_k_counts.items()
                            if not cpu_backend(name) and not target_backend(name))
        lid_unknown = sum(count for name, count in lid_counts.items()
                          if not cpu_backend(name) and not target_backend(name))
        expected_node_total = args.expected_nodes if depth >= args.require_top_k_from else 0
        required_nodes_present = top_k_total == expected_node_total and lid_total == expected_node_total
        op_backend_correlated = (
            (expected_node_total == 0 and not top_k_counts and not lid_counts)
            or (
                top_k_counts == {"Meta(": expected_node_total}
                and lid_counts == {"Meta(": expected_node_total}
                and split_structure_ok
            )
        )

        if not marker_structure_ok:
            parse_warnings.append(f"depth {depth}: benchmark/depth/generation marker structure mismatch")
        if not split_structure_ok:
            parse_warnings.append(
                f"depth {depth}: split structure mismatch: observed {item['split_records']}, expected {expected_splits}"
            )
        if not required_nodes_present:
            parse_warnings.append(
                f"depth {depth}: expected TOP_K={expected_node_total}, LIGHTNING_INDEXER={expected_node_total}; "
                f"observed TOP_K={top_k_total}, LIGHTNING_INDEXER={lid_total}"
            )

        row_complete = marker_structure_ok and split_structure_ok and required_nodes_present
        row_resident = (
            row_complete and op_backend_correlated
            and top_k_cpu == 0 and lid_cpu == 0
            and top_k_unknown == 0 and lid_unknown == 0
        )
        complete = complete and row_complete
        residency_ok = residency_ok and row_resident
        records.append({
            "depth": depth,
            "benchmark_indices": item["benchmark_indices"],
            "depth_markers": item["depth_markers"],
            "generation_markers": item["generation_markers"],
            "decode_graphs": decode_graphs,
            "marker_structure_ok": marker_structure_ok,
            "split_records": item["split_records"],
            "expected_split_records": expected_splits,
            "split_structure_ok": split_structure_ok,
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
            "op_backend_correlated_to_meta_split": op_backend_correlated,
            "complete": row_complete,
            "rocm_resident": row_resident,
            "top_k_examples": item["top_k_examples"],
            "lid_examples": item["lid_examples"],
        })

    # Warnings are fail-closed. A parser concern cannot coexist with attestation.
    if parse_warnings:
        complete = False
        residency_ok = False

    document = {
        "complete": complete,
        "rocm_residency_ok": complete and residency_ok,
        "expected_depths": depths,
        "benchmark_sequence": benchmark_sequence,
        "expected_benchmark_sequence": expected_benchmark_sequence,
        "benchmark_sequence_ok": benchmark_sequence_ok,
        "reported_benchmark_count": len(benchmark_sequence),
        "require_top_k_from": args.require_top_k_from,
        "expected_nodes_per_graph": args.expected_nodes,
        "expected_meta_backend": args.expected_meta_backend,
        "parse_warnings": parse_warnings,
        "records": records,
    }
    args.json_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    columns = [
        "run_complete", "residency_ok", "depth", "decode_graphs",
        "marker_structure_ok", "split_structure_ok", "cpu_splits",
        "gpu_meta_splits", "unknown_splits", "total_split_input_copies",
        "cpu_split_input_copies", "gpu_meta_split_input_copies",
        "unknown_split_input_copies", "top_k_total", "top_k_cpu",
        "top_k_unknown", "lid_total", "lid_cpu", "lid_unknown",
        "op_backend_correlated_to_meta_split",
    ]
    output = ["\t".join(columns)]
    for row in records:
        values: dict[str, Any] = {
            "run_complete": complete,
            "residency_ok": document["rocm_residency_ok"],
            **row,
        }
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