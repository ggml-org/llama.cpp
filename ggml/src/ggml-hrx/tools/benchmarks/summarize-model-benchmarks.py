#!/usr/bin/env python3
#
# Summarize generated HRX model-scoped Loom benchmark results.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_benchmark_result(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row") == "benchmark":
                return row.get("benchmark_result")
    return None


def timing_from_result(result: dict[str, Any], metric: str) -> float | None:
    if metric == "operation-p50":
        return result.get("measurement", {}).get("operation_timing_ns", {}).get("p50")
    if metric == "profile-p50":
        return (
            result.get("profile_replay", {})
            .get("dispatch_timing", {})
            .get("dispatch_distribution", {})
            .get("duration_ns", {})
            .get("p50")
        )
    fail(f"unsupported metric: {metric}")


def secondary_profile_p50(result: dict[str, Any]) -> float | None:
    return (
        result.get("profile_replay", {})
        .get("dispatch_timing", {})
        .get("dispatch_distribution", {})
        .get("duration_ns", {})
        .get("p50")
    )


def summarize(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    shape_rows = []
    for row in rows:
        count = int(row.get("count", row.get("shape_multiplicity", 1)))
        result_path = row.get("results")
        shape = {
            "benchmark": row.get("benchmark"),
            "kernel": row.get("kernel"),
            "count": count,
            "state": row.get("state", "ok"),
            "error": row.get("error"),
            "metric_ns": None,
            "profile_p50_ns": None,
            "weighted_ns": None,
        }
        if shape["state"] != "ok":
            shape_rows.append(shape)
            continue
        if not result_path:
            shape["state"] = "missing_results"
            shape["error"] = "runner did not record a results path"
            shape_rows.append(shape)
            continue
        result_file = Path(result_path)
        if not result_file.is_file():
            shape["state"] = "missing_results"
            shape["error"] = f"results file not found: {result_file}"
            shape_rows.append(shape)
            continue
        result = load_benchmark_result(result_file)
        if result is None:
            shape["state"] = "missing_benchmark_row"
            shape["error"] = "results file did not contain a benchmark row"
            shape_rows.append(shape)
            continue
        if result.get("state") != "ok":
            shape["state"] = result.get("state", "failed")
            shape["error"] = result.get("failure", {}).get("kind") or "benchmark did not complete successfully"
            shape_rows.append(shape)
            continue
        metric_ns = timing_from_result(result, metric)
        if metric_ns is None:
            shape["state"] = "missing_metric"
            shape["error"] = f"metric not found: {metric}"
            shape_rows.append(shape)
            continue
        shape["metric_ns"] = metric_ns
        shape["profile_p50_ns"] = secondary_profile_p50(result)
        shape["weighted_ns"] = metric_ns * count
        shape_rows.append(shape)

    kernels: dict[str, dict[str, Any]] = {}
    for shape in shape_rows:
        kernel = shape["kernel"]
        entry = kernels.setdefault(
            kernel,
            {
                "kernel": kernel,
                "shape_count": 0,
                "invocation_count": 0,
                "measured_shape_count": 0,
                "measured_invocation_count": 0,
                "missing_or_failed_count": 0,
                "missing_or_failed_invocations": 0,
                "weighted_ns": 0.0,
                "weighted_percent": 0.0,
                "shapes": [],
            },
        )
        entry["shape_count"] += 1
        entry["invocation_count"] += shape["count"]
        entry["shapes"].append(shape)
        if shape["weighted_ns"] is None:
            entry["missing_or_failed_count"] += 1
            entry["missing_or_failed_invocations"] += shape["count"]
            continue
        entry["measured_shape_count"] += 1
        entry["measured_invocation_count"] += shape["count"]
        entry["weighted_ns"] += shape["weighted_ns"]

    total_weighted_ns = sum(entry["weighted_ns"] for entry in kernels.values())
    for entry in kernels.values():
        if total_weighted_ns:
            entry["weighted_percent"] = entry["weighted_ns"] / total_weighted_ns * 100.0
    kernel_rows = sorted(kernels.values(), key=lambda entry: entry["weighted_ns"], reverse=True)
    return {
        "schema": "ggml-hrx-model-loom-benchmark-summary-v2",
        "metric": metric,
        "metric_description": "operation_timing_ns.p50" if metric == "operation-p50" else "profile_replay dispatch_distribution duration_ns p50",
        "shape_count": len(shape_rows),
        "measured_shape_count": sum(1 for shape in shape_rows if shape["weighted_ns"] is not None),
        "missing_or_failed_shape_count": sum(1 for shape in shape_rows if shape["weighted_ns"] is None),
        "invocation_count": sum(shape["count"] for shape in shape_rows),
        "measured_invocation_count": sum(shape["count"] for shape in shape_rows if shape["weighted_ns"] is not None),
        "missing_or_failed_invocations": sum(shape["count"] for shape in shape_rows if shape["weighted_ns"] is None),
        "total_weighted_ns": total_weighted_ns,
        "kernels": kernel_rows,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# HRX Model Loom Benchmark Summary",
        "",
        f"Metric: `{summary['metric_description']}`",
        "",
        f"Measured shapes: {summary['measured_shape_count']} / {summary['shape_count']}",
        f"Measured invocations: {summary['measured_invocation_count']} / {summary['invocation_count']}",
        "",
        "| Kernel | Shapes | Invocations | Measured Shapes | Weighted ms | Weighted Share | Missing/Failed Shapes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kernel in summary["kernels"]:
        weighted_ms = kernel["weighted_ns"] / 1_000_000.0
        lines.append(
            f"| `{kernel['kernel']}` | {kernel['shape_count']} | {kernel['invocation_count']} | "
            f"{kernel['measured_shape_count']} | {weighted_ms:.3f} | "
            f"{kernel['weighted_percent']:.2f}% | {kernel['missing_or_failed_count']} |"
        )
    lines.append("")
    lines.append("## Missing Or Failed Shapes")
    lines.append("")
    lines.append("| Kernel | Benchmark | Invocations | State | Error |")
    lines.append("| --- | --- | ---: | --- | --- |")
    missing = False
    for kernel in summary["kernels"]:
        for shape in kernel["shapes"]:
            if shape["weighted_ns"] is not None:
                continue
            missing = True
            error = str(shape.get("error") or "").replace("|", "\\|")
            lines.append(
                f"| `{shape['kernel']}` | `{shape['benchmark']}` | {shape['count']} | "
                f"`{shape['state']}` | {error} |"
            )
    if not missing:
        lines.append("| - | - | 0 | - | - |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path, help="Runner results.jsonl path.")
    parser.add_argument("--metric", choices=["operation-p50", "profile-p50"], default="operation-p50")
    parser.add_argument("--output-json", type=Path, help="Summary JSON output path.")
    parser.add_argument("--output-md", type=Path, help="Summary Markdown output path.")
    args = parser.parse_args()

    rows = load_jsonl(args.results_jsonl)
    summary = summarize(rows, args.metric)
    output_json = args.output_json or (args.results_jsonl.parent / "summary.json")
    output_md = args.output_md or (args.results_jsonl.parent / "summary.md")
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_md, summary)
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")


if __name__ == "__main__":
    main()
