from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from summarize_nsys import kernel_category, load_manifest, markdown_code, number, read_csv, read_jsonl


def decode_kernel_category(name: str) -> str:
    lower = name.lower()
    if "mul_mat_vec" in lower or "mmvq" in lower:
        return "Quantized mat-vec"
    if "setup_routed_metadata" in lower:
        return "MoE routing"
    return kernel_category(name)


def load_cases(run_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for entry in load_manifest(run_dir):
        case_dir = run_dir / entry["run_dir"]
        bench = read_jsonl(case_dir / "llama-bench.jsonl")
        if len(bench) != 1:
            raise ValueError(f"{entry['label']}: expected one benchmark row, got {len(bench)}")
        kernels = read_csv(case_dir / "cuda-kernels.csv", {"Name", "Total Time (ns)", "Instances"})
        ranges = read_csv(case_dir / "nvtx-gpu.csv", {"Range", "Total Proj Time (ns)", "Range Instances"})
        if not kernels:
            raise ValueError(f"{entry['label']}: CUDA kernel report is empty")
        steps = int(entry["tokens"]) + int(entry["warmup_tokens"])
        cases.append({"entry": entry, "bench": bench[0], "kernels": kernels, "ranges": ranges, "steps": steps})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")

    cases = load_cases(args.run_dir)
    native = next((case for case in cases if case["entry"]["backend"] == "native"), None)
    native_ts = float(native["bench"]["avg_ts"]) if native is not None else None

    print("# CUTLASS MoE decode Nsys comparison")
    print(
        "\nEach process performs the one-token llama-bench warmup followed by one timed generation run. "
        "The benchmark latency excludes warmup. Nsys totals include warmup and are normalized by the "
        "observed warmup plus measured decode steps. The CUTLASS weight transform is reported separately.\n"
    )
    print("| Case | Backend | Input scale | Output | Tokens | Latency ms | tok/s | vs native |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for case in cases:
        entry = case["entry"]
        bench = case["bench"]
        throughput = float(bench["avg_ts"])
        speedup = throughput / native_ts if native_ts is not None else None
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
        print(
            f"| {entry['label']} | {entry['backend']} | {entry['input_scale']} | {entry['output']} | "
            f"{int(bench['n_gen'])} | "
            f"{float(bench['avg_ns']) / 1.0e6:.3f} | {throughput:.3f} | "
            f"{speedup_text} |"
        )

    print("\n## CUDA time per decode step")
    print("\n| Case | Steady kernels ms | One-time repack ms | Kernel launches |")
    print("|---|---:|---:|---:|")
    for case in cases:
        steady_ns = 0.0
        repack_ns = 0.0
        launches = 0.0
        for row in case["kernels"]:
            duration = number(row["Total Time (ns)"])
            if kernel_category(row["Name"]) == "MoE weight repack (one-time)":
                repack_ns += duration
            else:
                steady_ns += duration
                launches += number(row["Instances"])
        print(
            f"| {case['entry']['label']} | {steady_ns / case['steps'] / 1.0e6:.3f} | "
            f"{repack_ns / 1.0e6:.3f} | {launches / case['steps']:.1f} |"
        )

    print("\n## Components per decode step")
    print("\n| Case | Component | GPU ms | Launches |")
    print("|---|---|---:|---:|")
    for case in cases:
        totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for row in case["kernels"]:
            category = decode_kernel_category(row["Name"])
            if category == "MoE weight repack (one-time)":
                continue
            totals[category][0] += number(row["Total Time (ns)"])
            totals[category][1] += number(row["Instances"])
        for category, (duration, launches) in sorted(totals.items(), key=lambda item: item[1][0], reverse=True):
            print(
                f"| {case['entry']['label']} | {category} | "
                f"{duration / case['steps'] / 1.0e6:.3f} | {launches / case['steps']:.1f} |"
            )

    for case in cases:
        print(f"\n## {case['entry']['label']}")
        print("\n### NVTX ranges\n")
        print("| Range | Total ms | Instances |")
        print("|---|---:|---:|")
        ranges = sorted(case["ranges"], key=lambda row: number(row["Total Proj Time (ns)"]), reverse=True)
        for row in ranges[: args.top]:
            print(
                f"| {markdown_code(row['Range'].lstrip(':'))} | "
                f"{number(row['Total Proj Time (ns)']) / 1.0e6:.3f} | "
                f"{int(number(row['Range Instances']))} |"
            )

        print("\n### CUDA kernels\n")
        print("| Kernel | Total ms | Instances |")
        print("|---|---:|---:|")
        kernels = sorted(case["kernels"], key=lambda row: number(row["Total Time (ns)"]), reverse=True)
        for row in kernels[: args.top]:
            print(
                f"| {markdown_code(row['Name'])} | {number(row['Total Time (ns)']) / 1.0e6:.3f} | "
                f"{int(number(row['Instances']))} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
