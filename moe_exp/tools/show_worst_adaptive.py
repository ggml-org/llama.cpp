#!/usr/bin/env python3

import argparse
import csv

ID_COLS = [
    "prompt",
    "prompt_name",
    "prompt_path",
    "trace",
    "trace_path",
    "run",
    "run_id",
    "seed",
]

METRIC_COLS = [
    "decode_tokens",
    "hit_rate",
    "oracle_hit_rate",
    "cold_mib_per_token",
]


def parse_row(r):
    r["warmup"] = int(r["warmup"])
    r["budget"] = int(r["budget"])
    r["hit_rate"] = float(r["hit_rate"])
    r["oracle_hit_rate"] = float(r["oracle_hit_rate"])
    r["cold_mib_per_token"] = float(r["cold_mib_per_token"])
    if "decode_tokens" in r and r["decode_tokens"] != "":
        r["decode_tokens"] = int(r["decode_tokens"])
    return r


def fmt_value(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def fmt_row(values, widths):
    return " ".join(str(v).rjust(w) for v, w in zip(values, widths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = []
    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(parse_row(r))

    rows = [
        r for r in rows if r["warmup"] == args.warmup and r["budget"] == args.budget
    ]
    rows.sort(key=lambda r: r["hit_rate"])

    if not rows:
        print("no rows")
        return

    cols = [c for c in ID_COLS + METRIC_COLS if c in rows[0]]

    widths = []
    for c in cols:
        max_val_width = max(len(fmt_value(r[c])) for r in rows[: args.top])
        widths.append(max(len(c), max_val_width))

    print(fmt_row(cols, widths))

    for r in rows[: args.top]:
        print(fmt_row([fmt_value(r[c]) for c in cols], widths))


if __name__ == "__main__":
    main()
