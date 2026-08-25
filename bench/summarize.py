#!/usr/bin/env python3
"""Turn bench/results.jsonl into a comparison table.

Reads the append-only log, drops warmup repetitions, and reports mean +/- sample stddev
across outer repetitions for each (model, test, depth, build), plus the fork-vs-mainline
delta and how many standard errors it is. Anything under 2 sigma is reported as noise
rather than as a result.
"""
import json, sys, math, statistics, collections, argparse


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_name(r):
    if r.get("n_prompt"):
        return f"pp{r['n_prompt']}"
    return f"tg{r['n_gen']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="bench/results.jsonl")
    ap.add_argument("--include-warmup", action="store_true")
    ap.add_argument("--format", choices=["md", "csv"], default="md")
    args = ap.parse_args()

    rows = [r for r in load(args.path) if args.include_warmup or not r.get("warmup")]
    if not rows:
        sys.exit("no non-warmup rows; run bench/run-depth-sweep.sh first")

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model_label"], test_name(r), r["n_depth"], r["build"])].append(r["avg_ts"])

    keys = sorted({(m, t, d) for m, t, d, _ in by},
                  key=lambda k: (k[0], k[1].startswith("tg"), k[2]))

    if args.format == "csv":
        print("model,test,depth,mainline_ts,mainline_sd,fork_ts,fork_sd,delta_pct,sigma,n")
    else:
        print("| model | test | depth | mainline t/s | fork t/s | delta | |")
        print("|---|---|---:|---:|---:|---:|---|")

    for m, t, d in keys:
        a, b = by.get((m, t, d, "mainline"), []), by.get((m, t, d, "fork"), [])
        if not a or not b:
            continue
        ma, mb = statistics.mean(a), statistics.mean(b)
        sa = statistics.stdev(a) if len(a) > 1 else 0.0
        sb = statistics.stdev(b) if len(b) > 1 else 0.0
        se = math.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b))
        pct = 100 * (mb / ma - 1)
        sig = (mb - ma) / se if se > 0 else float("inf")
        if args.format == "csv":
            print(f"{m},{t},{d},{ma:.2f},{sa:.2f},{mb:.2f},{sb:.2f},{pct:.2f},{sig:.1f},{len(b)}")
        else:
            note = "" if abs(sig) >= 2 else " *noise*"
            print(f"| {m} | {t} | {d} | {ma:.1f} +/- {sa:.1f} | {mb:.1f} +/- {sb:.1f} | **{pct:+.1f}%**{note} | {sig:+.1f} sigma |")


if __name__ == "__main__":
    main()
