from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize_nsys import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    with (args.run_dir / "manifest.tsv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))

    rows = []
    for entry in manifest:
        results = read_jsonl(args.run_dir / f"{entry['label']}.jsonl")
        if len(results) != 1:
            raise ValueError(f"{entry['label']}: expected one benchmark row, got {len(results)}")
        rows.append((entry, results[0]))

    reference = next((result for entry, result in rows if entry["label"] == "native-eager"), None)
    reference_ts = float(reference["avg_ts"]) if reference is not None else None

    print("# CUTLASS MoE decode benchmark")
    print()
    print("| Case | Graphs | Backend | Input scale | Output | Tokens | Latency ms | tok/s | vs native eager |")
    print("|---|---|---|---|---|---:|---:|---:|---:|")
    for entry, result in rows:
        throughput = float(result["avg_ts"])
        speedup = throughput / reference_ts if reference_ts is not None else None
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
        print(
            f"| {entry['label']} | {entry['graphs']} | {entry['backend']} | "
            f"{entry['input_scale']} | {entry['output']} | {int(result['n_gen'])} | "
            f"{float(result['avg_ns']) / 1.0e6:.3f} | {throughput:.3f} | {speedup_text} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
