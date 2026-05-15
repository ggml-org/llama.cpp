#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def quantile(xs, q):
    """Return quantile with linear interpolation."""
    if not xs:
        return 0.0

    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]

    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo

    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def fmt_row(values, widths):
    return " ".join(str(v).rjust(w) for v, w in zip(values, widths))


def parse_optional_float(row, name, default=0.0):
    value = row.get(name, "")
    if value == "" or value is None:
        return default
    return float(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    args = ap.parse_args()

    rows = []
    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["warmup"] = int(r["warmup"])
            r["budget"] = int(r["budget"])
            r["hit_rate"] = float(r["hit_rate"])
            r["oracle_hit_rate"] = float(r["oracle_hit_rate"])
            r["resident_expert_gib"] = float(r["resident_expert_gib"])
            r["cold_mib_per_token"] = float(r["cold_mib_per_token"])

            # Newer adaptive_summary.py outputs these fields.
            # Keep defaults so this summarizer still works on old CSVs.
            r["reload_mib_per_token"] = parse_optional_float(
                r, "reload_mib_per_token", 0.0
            )
            r["total_mib_per_token"] = parse_optional_float(
                r,
                "total_mib_per_token",
                r["cold_mib_per_token"],
            )
            r["hotset_adds_per_token"] = parse_optional_float(
                r, "hotset_adds_per_token", 0.0
            )
            r["hotset_evictions_per_token"] = parse_optional_float(
                r, "hotset_evictions_per_token", 0.0
            )

            rows.append(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["warmup"], r["budget"])].append(r)

    headers = [
        "warmup",
        "budget",
        "res_GiB",
        "n",
        "hit_avg",
        "hit_p10",
        "hit_p50",
        "hit_p90",
        "hit_min",
        "hit_max",
        "oracle",
        "cold",
        "reload",
        "total",
        "adds/tok",
        "ev/tok",
    ]

    widths = [
        6,  # warmup
        6,  # budget
        8,  # res_GiB
        3,  # n
        7,  # hit_avg
        7,  # hit_p10
        7,  # hit_p50
        7,  # hit_p90
        7,  # hit_min
        7,  # hit_max
        7,  # oracle
        8,  # cold
        8,  # reload
        8,  # total
        8,  # adds/tok
        8,  # ev/tok
    ]

    print(fmt_row(headers, widths))

    for warmup, budget in sorted(groups):
        g = groups[(warmup, budget)]

        hits = [r["hit_rate"] for r in g]
        oracles = [r["oracle_hit_rate"] for r in g]
        colds = [r["cold_mib_per_token"] for r in g]
        reloads = [r["reload_mib_per_token"] for r in g]
        totals = [r["total_mib_per_token"] for r in g]
        adds = [r["hotset_adds_per_token"] for r in g]
        evictions = [r["hotset_evictions_per_token"] for r in g]
        resident = g[0]["resident_expert_gib"]

        row = [
            warmup,
            budget,
            f"{resident:.3f}",
            len(g),
            f"{mean(hits):.3f}",
            f"{quantile(hits, 0.10):.3f}",
            f"{quantile(hits, 0.50):.3f}",
            f"{quantile(hits, 0.90):.3f}",
            f"{min(hits):.3f}",
            f"{max(hits):.3f}",
            f"{mean(oracles):.3f}",
            f"{mean(colds):.1f}",
            f"{mean(reloads):.1f}",
            f"{mean(totals):.1f}",
            f"{mean(adds):.2f}",
            f"{mean(evictions):.2f}",
        ]

        print(fmt_row(row, widths))


if __name__ == "__main__":
    main()
