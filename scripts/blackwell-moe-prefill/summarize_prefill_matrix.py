from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    with (args.run_dir / "manifest.tsv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))

    rows: list[tuple[str, str, dict[str, Any]]] = []
    for entry in manifest:
        for result in read_jsonl(args.run_dir / entry["result"]):
            rows.append((entry["label"], entry["validation"], result))

    baseline = {
        int(result["n_ubatch"]): result
        for label, _, result in rows
        if label == "baseline"
    }
    print("# Blackwell prefill performance matrix")
    print("\n| Case | Validation | Ubatch | Latency ms | tok/s | vs baseline |")
    print("|---|---|---:|---:|---:|---:|")
    for label, validation, result in rows:
        ubatch = int(result["n_ubatch"])
        throughput = float(result["avg_ts"])
        reference = baseline.get(ubatch)
        speedup = throughput / float(reference["avg_ts"]) if reference else None
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
        print(
            f"| {label} | {validation} | {ubatch} | "
            f"{float(result['avg_ns']) / 1.0e6:.3f} | {throughput:.1f} | {speedup_text} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
