#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


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
            rows.append(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["warmup"], r["budget"])].append(r)

    print(
        "warmup budget resident_GiB mean_hit min_hit max_hit mean_oracle mean_cold_MiB_tok"
    )
    for warmup, budget in sorted(groups):
        g = groups[(warmup, budget)]
        hits = [r["hit_rate"] for r in g]
        oracles = [r["oracle_hit_rate"] for r in g]
        colds = [r["cold_mib_per_token"] for r in g]
        resident = g[0]["resident_expert_gib"]

        print(
            f"{warmup:6d} "
            f"{budget:6d} "
            f"{resident:12.3f} "
            f"{mean(hits):8.3f} "
            f"{min(hits):7.3f} "
            f"{max(hits):7.3f} "
            f"{mean(oracles):11.3f} "
            f"{mean(colds):18.1f}"
        )


if __name__ == "__main__":
    main()
