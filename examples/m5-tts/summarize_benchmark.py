#!/usr/bin/env python3
"""Summarize AndroidBenchmark JSON lines, excluding the first run per mode."""

import argparse
import json
from statistics import median


def summarize(lines):
    mode = "CPU baseline"
    groups = {mode: []}
    for line in lines:
        if line.startswith("mode="):
            mode = line.strip()[len("mode="):]
            groups.setdefault(mode, [])
        elif line.startswith('{"run":'):
            groups[mode].append(json.loads(line))

    for mode, rows in groups.items():
        if len(rows) < 2:
            continue
        if [row["run"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"unexpected run numbering in {mode}")
        warm = rows[1:]
        print(json.dumps({
            "mode": mode,
            "warm_runs": len(warm),
            "rtf_min": min(row["rtf"] for row in warm),
            "rtf_max": max(row["rtf"] for row in warm),
            "realtime_runs": sum(row["rtf"] <= 1 for row in warm),
            **{key: median(row[key] for row in warm) for key in warm[0] if key != "run"},
        }, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    args = parser.parse_args()
    with open(args.log, encoding="utf-8") as stream:
        summarize(stream)
