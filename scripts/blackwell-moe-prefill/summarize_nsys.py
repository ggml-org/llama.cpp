from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def number(value: str) -> float:
    return float(value.replace(",", ""))


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for values in csv.reader(handle):
            if header is None:
                if required.issubset(values):
                    header = values
                continue
            if len(values) < len(header):
                continue
            rows.append(dict(zip(header, values)))
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def markdown_code(value: str, limit: int = 120) -> str:
    compact = " ".join(value.split())
    if len(compact) > limit:
        compact = compact[: limit - 3] + "..."
    escaped = compact.replace("|", "\\|")
    return f"`{escaped}`"


def kernel_category(name: str) -> str:
    lower = name.lower()
    if "flash_attn" in lower:
        return "Attention"
    if "rope_" in lower or "rope<" in lower:
        return "RoPE"
    if "diag_mask" in lower or "diagmask" in lower:
        return "KQ mask"
    if "moe_tma_w13" in lower or "moe_tma_persistent" in lower:
        return "MoE GEMM"
    if "cutlass::device_kernel<cutlass::gemm::kernel::gemmuniversal" in lower:
        return "MoE GEMM"
    if "mul_mat_q<(ggml_type)39" in lower or "mul_mat_q<(ggml_type)40" in lower:
        return "MoE GEMM"
    if "quantize_mmq" in lower or "moe_quantize" in lower or "moe_cutlass_quantize" in lower:
        return "MoE activation quant"
    if "moe_mmq_repack" in lower:
        return "MoE weight repack (one-time)"
    if ("mm_ids_helper" in lower or "mm_ids_prefix" in lower or "moe_cutlass_stage_routes" in lower or
            "topk" in lower or "top_k" in lower):
        return "MoE routing"
    if "moe_mmq_" in lower or "moe_cutlass_" in lower or "swiglu_oai" in lower or "add_id_kernel" in lower:
        return "MoE epilogue"
    if "rms_norm" in lower:
        return "Norm"
    return "Other"


def load_manifest(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "manifest.tsv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")

    manifest = load_manifest(args.run_dir)
    case_data: dict[str, dict[str, Any]] = {}
    for entry in manifest:
        case_dir = args.run_dir / entry["run_dir"]
        bench_rows = read_jsonl(case_dir / "llama-bench.jsonl")
        kernel_rows = read_csv(
            case_dir / "cuda-kernels.csv",
            {"Name", "Total Time (ns)", "Instances"},
        )
        nvtx_rows = read_csv(
            case_dir / "nvtx-gpu.csv",
            {"Range", "Total Proj Time (ns)", "Range Instances"},
        )
        case_data[entry["label"]] = {
            "entry": entry,
            "bench": bench_rows,
            "kernels": kernel_rows,
            "nvtx": nvtx_rows,
        }

    baseline = case_data.get("baseline")
    if baseline is None:
        baseline = next(
            (data for data in case_data.values() if data["entry"].get("backend") == "native"),
            None,
        )
    baseline_rows = {
        int(row["n_ubatch"]): row
        for row in (baseline["bench"] if baseline is not None else [])
    }

    print("# Blackwell prefill Nsys summary")
    print(
        "\nBenchmark throughput uses the measured samples and includes Nsys overhead. "
        "CUDA totals include all traced work; the first-use weight repack is reported separately.\n"
    )
    print("| Case | Validation | Tokens | Ubatch | Latency ms | tok/s | vs baseline |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for label, data in case_data.items():
        validation = data["entry"].get("validation", data["entry"].get("backend", ""))
        for row in data["bench"]:
            ubatch = int(row["n_ubatch"])
            throughput = float(row["avg_ts"])
            baseline = baseline_rows.get(ubatch)
            speedup = throughput / float(baseline["avg_ts"]) if baseline else None
            speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
            print(
                f"| {label} | {validation} | {int(row['n_prompt'])} | {ubatch} | "
                f"{float(row['avg_ns']) / 1.0e6:.3f} | {throughput:.1f} | {speedup_text} |"
            )

    traced_totals: dict[str, tuple[float, float]] = {}
    for label, data in case_data.items():
        total_ns = 0.0
        repack_ns = 0.0
        for row in data["kernels"]:
            duration = number(row["Total Time (ns)"])
            total_ns += duration
            if kernel_category(row["Name"]) == "MoE weight repack (one-time)":
                repack_ns += duration
        traced_totals[label] = (total_ns, repack_ns)

    baseline_steady_ns = None
    baseline_label = next(
        (
            label
            for label, data in case_data.items()
            if label == "baseline" or data["entry"].get("backend") == "native"
        ),
        None,
    )
    if baseline_label is not None:
        baseline_total, baseline_repack = traced_totals[baseline_label]
        baseline_steady_ns = baseline_total - baseline_repack

    print("\n## Summed CUDA kernel time")
    print("\nThe steady column excludes the one-time in-place expert-weight transform.\n")
    print("| Case | All kernels ms | One-time repack ms | Steady kernels ms | vs baseline |")
    print("|---|---:|---:|---:|---:|")
    for label, (total_ns, repack_ns) in traced_totals.items():
        steady_ns = total_ns - repack_ns
        speedup = baseline_steady_ns / steady_ns if baseline_steady_ns else None
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
        print(
            f"| {label} | {total_ns / 1.0e6:.3f} | {repack_ns / 1.0e6:.3f} | "
            f"{steady_ns / 1.0e6:.3f} | {speedup_text} |"
        )

    print("\n## CUDA component totals")
    print("\nCategories are based on kernel names. Exact kernel and NVTX tables follow.\n")
    print("| Case | Component | Total ms | Share of traced CUDA | Launches |")
    print("|---|---|---:|---:|---:|")
    for label, data in case_data.items():
        totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        traced_ns = 0.0
        for row in data["kernels"]:
            duration = number(row["Total Time (ns)"])
            instances = number(row["Instances"])
            traced_ns += duration
            category = kernel_category(row["Name"])
            totals[category][0] += duration
            totals[category][1] += instances
        for category, (duration, instances) in sorted(
            totals.items(), key=lambda item: item[1][0], reverse=True
        ):
            share = 100.0 * duration / traced_ns if traced_ns else 0.0
            print(
                f"| {label} | {category} | {duration / 1.0e6:.3f} | "
                f"{share:.1f}% | {int(instances)} |"
            )

    for label, data in case_data.items():
        print(f"\n## {label}")
        print("\n### NVTX GPU ranges\n")
        print("| Range | Total ms | Instances |")
        print("|---|---:|---:|")
        nvtx_rows = sorted(
            data["nvtx"],
            key=lambda row: number(row["Total Proj Time (ns)"]),
            reverse=True,
        )
        for row in nvtx_rows[: args.top]:
            name = row["Range"].lstrip(":")
            print(
                f"| {markdown_code(name)} | "
                f"{number(row['Total Proj Time (ns)']) / 1.0e6:.3f} | "
                f"{int(number(row['Range Instances']))} |"
            )

        print("\n### CUDA kernels\n")
        print("| Kernel | Total ms | Instances |")
        print("|---|---:|---:|")
        kernel_rows = sorted(
            data["kernels"],
            key=lambda row: number(row["Total Time (ns)"]),
            reverse=True,
        )
        for row in kernel_rows[: args.top]:
            print(
                f"| {markdown_code(row['Name'])} | "
                f"{number(row['Total Time (ns)']) / 1.0e6:.3f} | "
                f"{int(number(row['Instances']))} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
