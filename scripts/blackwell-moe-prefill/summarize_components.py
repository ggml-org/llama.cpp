from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
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

    files = sorted(args.run_dir.glob("*.jsonl"))
    by_case: dict[str, dict[int, dict[str, Any]]] = {}
    for path in files:
        by_case[path.stem] = {int(row["n_prompt"]): row for row in read_rows(path)}

    baseline = by_case.get("disabled", {})
    print("| case | tokens | latency ms | tok/s | vs disabled |")
    print("|---|---:|---:|---:|---:|")
    for case, rows in by_case.items():
        for tokens, row in sorted(rows.items()):
            latency_ms = float(row["avg_ns"]) / 1.0e6
            throughput = float(row["avg_ts"])
            baseline_row = baseline.get(tokens)
            speedup = throughput / float(baseline_row["avg_ts"]) if baseline_row else 1.0
            print(f"| {case} | {tokens} | {latency_ms:.3f} | {throughput:.1f} | {speedup:.3f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
