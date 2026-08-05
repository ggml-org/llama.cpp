from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from summarize_nsys import load_manifest, number, read_csv, read_jsonl


N_LAYERS = 36
CUTLASS_RANGES = (
    "ffn_moe.shared_ids_helper",
    "ffn_moe.cutlass_quant_input",
    "ffn_moe.cutlass_w13",
    "ffn_moe.cutlass_w13_epilogue_quant",
    "ffn_moe.cutlass_w2",
    "ffn_moe.cutlass_w2_finalize",
)

RANGE_ALIASES = {
    "ffn_moe.shared_ids_helper": (
        "ffn_moe.shared_ids_helper",
        "ffn_moe.cutlass_prefix_schedule",
    ),
    "ffn_moe.cutlass_quant_input": (
        "ffn_moe.cutlass_quant_input",
        "ffn_moe.cutlass_quant_input_cta",
    ),
    "ffn_moe.cutlass_w13_epilogue_quant": (
        "ffn_moe.cutlass_w13_epilogue_quant",
        "ffn_moe.cutlass_w13_epilogue_quant_cta",
    ),
}


def range_totals(rows: list[dict[str, str]]) -> dict[str, tuple[float, int]]:
    totals: dict[str, tuple[float, int]] = {}
    for row in rows:
        name = row["Range"].lstrip(":")
        duration = number(row["Total Proj Time (ns)"])
        instances = int(number(row["Range Instances"]))
        old_duration, old_instances = totals.get(name, (0.0, 0))
        totals[name] = (old_duration + duration, old_instances + instances)
    return totals


def observed_passes(entry: dict[str, str], ranges: dict[str, tuple[float, int]]) -> float:
    if entry["backend"] == "cutlass":
        instances = ranges.get("ffn_moe.cutlass_w13", (0.0, 0))[1]
        divisor = N_LAYERS
    else:
        instances = ranges.get("ffn_moe.grouped_gemm", (0.0, 0))[1]
        divisor = 2 * N_LAYERS
    if instances == 0 or instances % divisor != 0:
        raise ValueError(f"{entry['label']}: unexpected MoE range instance count {instances}")
    return instances / divisor


def ms_per_pass(ranges: dict[str, tuple[float, int]], name: str, passes: float) -> float:
    return ranges.get(name, (0.0, 0))[0] / passes / 1.0e6


def stage_ms_per_pass(ranges: dict[str, tuple[float, int]], name: str, passes: float) -> float:
    return sum(ms_per_pass(ranges, alias, passes) for alias in RANGE_ALIASES.get(name, (name,)))


def load_cases(run_dir: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in load_manifest(run_dir):
        case_dir = run_dir / entry["run_dir"]
        bench = read_jsonl(case_dir / "llama-bench.jsonl")
        if len(bench) != 1:
            raise ValueError(f"{entry['label']}: expected one benchmark row, got {len(bench)}")
        nvtx = read_csv(
            case_dir / "nvtx-gpu.csv",
            {"Range", "Total Proj Time (ns)", "Range Instances"},
        )
        ranges = range_totals(nvtx)
        passes = observed_passes(entry, ranges)
        stages = {name: stage_ms_per_pass(ranges, name, passes) for name in CUTLASS_RANGES}
        stages["ffn_moe.grouped_gemm"] = ms_per_pass(ranges, "ffn_moe.grouped_gemm", passes)
        result.append(
            {
                "entry": entry,
                "bench": bench[0],
                "ranges": ranges,
                "passes": passes,
                "stages": stages,
            }
        )
    return result


def write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = (
        "label",
        "backend",
        "pdl",
        "w13_tile",
        "w13_swap",
        "w2_tile",
        "w2_swap",
        "latency_ms",
        "tokens_per_second",
        "observed_passes",
        "one_time_repack_ms",
        "shared_plan_ms",
        "input_quant_ms",
        "w13_ms",
        "w13_epilogue_quant_ms",
        "w2_ms",
        "w2_finalize_ms",
        "cutlass_pipeline_ms",
        "native_grouped_gemm_ms",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            entry = case["entry"]
            bench = case["bench"]
            stages = case["stages"]
            pipeline = (
                sum(stages[name] for name in CUTLASS_RANGES)
                if entry["backend"] == "cutlass"
                else 0.0
            )
            writer.writerow(
                {
                    "label": entry["label"],
                    "backend": entry["backend"],
                    "pdl": entry["pdl"],
                    "w13_tile": entry["w13_tile"],
                    "w13_swap": entry["w13_swap"],
                    "w2_tile": entry["w2_tile"],
                    "w2_swap": entry["w2_swap"],
                    "latency_ms": float(bench["avg_ns"]) / 1.0e6,
                    "tokens_per_second": float(bench["avg_ts"]),
                    "observed_passes": case["passes"],
                    "one_time_repack_ms": case["ranges"].get("ffn_moe.weight_repack", (0.0, 0))[0] / 1.0e6,
                    "shared_plan_ms": stages["ffn_moe.shared_ids_helper"],
                    "input_quant_ms": stages["ffn_moe.cutlass_quant_input"],
                    "w13_ms": stages["ffn_moe.cutlass_w13"],
                    "w13_epilogue_quant_ms": stages["ffn_moe.cutlass_w13_epilogue_quant"],
                    "w2_ms": stages["ffn_moe.cutlass_w2"],
                    "w2_finalize_ms": stages["ffn_moe.cutlass_w2_finalize"],
                    "cutlass_pipeline_ms": pipeline,
                    "native_grouped_gemm_ms": stages["ffn_moe.grouped_gemm"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.run_dir)
    if args.csv is not None:
        write_csv(args.csv, cases)

    native = next((case for case in cases if case["entry"]["backend"] == "native"), None)
    native_ts = float(native["bench"]["avg_ts"]) if native is not None else None

    print("# CUTLASS MoE Nsys comparison")
    print(
        "\nEach process runs one pp8192 warmup followed by one measured pass. "
        "The latency and throughput columns use the measured pass. NVTX stage totals include both passes "
        "and are divided by the observed pass count. The one-time weight transform is reported separately.\n"
    )
    print("| Case | PDL | W13 tile/swap | W2 tile/swap | Latency ms | tok/s | vs native |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for case in cases:
        entry = case["entry"]
        bench = case["bench"]
        throughput = float(bench["avg_ts"])
        speedup = throughput / native_ts if native_ts is not None else None
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
        print(
            f"| {entry['label']} | {entry['pdl']} | {entry['w13_tile']}/{entry['w13_swap']} | "
            f"{entry['w2_tile']}/{entry['w2_swap']} | {float(bench['avg_ns']) / 1.0e6:.3f} | "
            f"{throughput:.1f} | {speedup_text} |"
        )

    print("\n## MoE stages per pp8192 pass")
    print(
        "\nThe native row reports its combined W13 and W2 grouped-GEMM range. "
        "CUTLASS rows expose W13 and W2 separately. Times are projected GPU time in milliseconds.\n"
    )
    print(
        "| Case | Shared plan | Input quant | W13 | W13 epilogue + quant | W2 | W2 finalize | "
        "CUTLASS pipeline | Native W13 + W2 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case in cases:
        stages = case["stages"]
        pipeline = (
            sum(stages[name] for name in CUTLASS_RANGES)
            if case["entry"]["backend"] == "cutlass"
            else 0.0
        )
        print(
            f"| {case['entry']['label']} | {stages['ffn_moe.shared_ids_helper']:.3f} | "
            f"{stages['ffn_moe.cutlass_quant_input']:.3f} | {stages['ffn_moe.cutlass_w13']:.3f} | "
            f"{stages['ffn_moe.cutlass_w13_epilogue_quant']:.3f} | {stages['ffn_moe.cutlass_w2']:.3f} | "
            f"{stages['ffn_moe.cutlass_w2_finalize']:.3f} | {pipeline:.3f} | "
            f"{stages['ffn_moe.grouped_gemm']:.3f} |"
        )

    print("\n## One-time weight transform")
    print("\n| Case | Total projected GPU ms | Range instances |")
    print("|---|---:|---:|")
    for case in cases:
        duration, instances = case["ranges"].get("ffn_moe.weight_repack", (0.0, 0))
        print(f"| {case['entry']['label']} | {duration / 1.0e6:.3f} | {instances} |")

    print(
        "\nThe raw `.nsys-rep` files, kernel summaries, CUDA API summaries, NVTX tables, "
        "path-selection logs, and benchmark JSON are retained under `cases/`.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
