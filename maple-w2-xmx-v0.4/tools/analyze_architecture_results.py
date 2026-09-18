# SPDX-License-Identifier: MIT
"""Paired wall-time selection. No cross-backend speedup inferred."""
import argparse
import csv
import json
import math
from pathlib import Path

def lower_median(values):
    if not values:
        raise ValueError("no paired observations")
    v = sorted(values)
    return v[(len(v)-1)//2]

def paired_stats(path):
    rounds = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["phase"] != "repeat":
                continue
            value = float(r["wall_us"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid wall time")
            by_arm = rounds.setdefault(r["round"], {})
            if r["variant"] in by_arm:
                raise ValueError("duplicate arm in paired round")
            by_arm[r["variant"]] = value
    if not rounds or any(set(v) != {"baseline", "candidate"} for v in rounds.values()):
        raise ValueError("unpaired/missing repeat samples")
    gains = [1-r["candidate"]/r["baseline"] for r in rounds.values()]
    return {"pairs": len(gains), "paired_gain_lower_median": lower_median(gains),
            "candidate_win_rate": sum(x > 0 for x in gains)/len(gains)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--min-gain", type=float, default=.05)
    ap.add_argument("--min-win-rate", type=float, default=.75)
    ap.add_argument("--min-repeats", type=int, default=8)
    a = ap.parse_args()
    if not 0 <= a.min_gain < 1 or not 0 <= a.min_win_rate <= 1 or a.min_repeats < 2:
        ap.error("invalid selection thresholds")
    rows = []
    for path in sorted(a.root.glob("*/summary.csv")):
        with path.open(newline="", encoding="utf-8-sig") as f:
            r = next((r for r in csv.DictReader(f) if r["variant"] == "candidate"), None)
        if r is None:
            raise ValueError("missing candidate: " + str(path))
        if r["kernel_pass"] != "1" or r["pair_pass"] != "1":
            continue
        stats = paired_stats(path.parent / "samples.csv")
        eligible = r.get("freeze_input") == "0" and int(r["tokens"]) > 1 and stats["pairs"] >= a.min_repeats
        recommend = eligible and stats["paired_gain_lower_median"] >= a.min_gain and stats["candidate_win_rate"] >= a.min_win_rate
        rows.append({"case": path.parent.name, **{k:r[k] for k in ("tokens","k","hidden","experts","topk","gate_tile","down_tile","fork_join","wall_median_us","expert_grouping")},
                     **stats, "candidate_recommended_at_this_measured_shape_only": recommend})
    if not rows:
        raise ValueError("No completed case summaries. A dry-run plan is not a result")
    out = a.root / "architecture-selection.json"
    out.write_text(json.dumps({"scope":"within-case paired measurements; no fitted universal crossover",
                              "auto_policy":"uncalibrated/direct remains default; measured-shape recommendations only",
                              "selection_thresholds":{"min_gain":a.min_gain,"min_win_rate":a.min_win_rate,"min_repeats":a.min_repeats},
                              "cases":rows}, indent=2) + "\n", encoding="utf-8")
    print(out)
if __name__ == "__main__":
    main()
