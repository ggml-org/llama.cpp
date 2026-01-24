#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[warn] {path}:{idx} invalid json ({exc})", file=sys.stderr)
    return rows


def metric_field(metric: str) -> str:
    if metric == "tops":
        return "throughput_tops"
    return "bandwidth_gbps"


def best_by_group(rows, metric: str, group_by_repeat: bool):
    field = metric_field(metric)
    best = {}
    for row in rows:
        pattern = row.get("memory_pattern", "unknown")
        repeat = row.get("repeat")
        key = (repeat, pattern) if group_by_repeat else pattern
        value = row.get(field)
        if value is None:
            continue
        current = best.get(key)
        if current is None or value > current.get(field, float("-inf")):
            best[key] = row
    return best


def format_int(value):
    return "?" if value is None else str(value)

def format_dims(row):
    return "{}x{}x{}".format(
        format_int(row.get("dim_m")),
        format_int(row.get("dim_n")),
        format_int(row.get("dim_k")),
    )


def print_table(rows, metric: str, group_by_repeat: bool, group_by_dims: bool):
    field = metric_field(metric)
    groups = [(None, rows)]
    if group_by_dims and group_by_repeat:
        groups = []
        keys = sorted({(r.get("dim_m"), r.get("dim_n"), r.get("dim_k"), r.get("repeat"))
                       for r in rows if r.get("repeat") is not None})
        for key in keys:
            dim_m, dim_n, dim_k, repeat = key
            subset = [r for r in rows if r.get("repeat") == repeat and
                      r.get("dim_m") == dim_m and r.get("dim_n") == dim_n and r.get("dim_k") == dim_k]
            label = f"repeat={repeat} dims={dim_m}x{dim_n}x{dim_k}"
            groups.append((label, subset))
    elif group_by_repeat:
        groups = []
        repeats = sorted({r.get("repeat") for r in rows if r.get("repeat") is not None})
        for repeat in repeats:
            subset = [r for r in rows if r.get("repeat") == repeat]
            groups.append((f"repeat={repeat}", subset))
    elif group_by_dims:
        groups = []
        keys = sorted({(r.get("dim_m"), r.get("dim_n"), r.get("dim_k")) for r in rows})
        for dim_m, dim_n, dim_k in keys:
            subset = [r for r in rows if r.get("dim_m") == dim_m and r.get("dim_n") == dim_n and r.get("dim_k") == dim_k]
            groups.append((f"dims={dim_m}x{dim_n}x{dim_k}", subset))

    for label, subset in groups:
        if label:
            print(f"\n{label}")
        print("pattern         ntiles prefetch  grf   acc     bandwidth_gbps  throughput_tops")
        best = best_by_group(subset, metric, group_by_repeat=False)
        ordered = sorted(best.values(), key=lambda r: r.get(field, 0.0), reverse=True)
        for row in ordered:
            print("{:<14} {:>6} {:>8} {:>4} {:>6} {:>14.3f} {:>15.3f}".format(
                row.get("memory_pattern", "unknown"),
                format_int(row.get("ntiles")),
                format_int(row.get("prefetch_dist")),
                row.get("grf_mode", "?"),
                row.get("type_acc", "?"),
                row.get("bandwidth_gbps", 0.0),
                row.get("throughput_tops", 0.0),
            ))


def main():
    parser = argparse.ArgumentParser(description="Summarize DPAS sweep JSONL results.")
    parser.add_argument("path", help="Path to DPAS sweep jsonl output")
    parser.add_argument("--metric", choices=["bandwidth", "tops"], default="bandwidth")
    parser.add_argument("--group-by-repeat", action="store_true",
                        help="Show best-per-pattern tables for each repeat value")
    parser.add_argument("--group-by-dims", action="store_true",
                        help="Show best-per-pattern tables for each dim_m/dim_n/dim_k triple")
    args = parser.parse_args()

    path = Path(args.path)
    rows = load_jsonl(path)
    if not rows:
        print("No rows found.", file=sys.stderr)
        return 1
    print_table(rows, args.metric, args.group_by_repeat, args.group_by_dims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
