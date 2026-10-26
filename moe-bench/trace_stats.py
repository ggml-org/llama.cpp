#!/usr/bin/env python3
"""Routing statistics of an expert-selection trace.

Answers the questions that decide whether prefetching or a frequency aware
cache can help:
  - how skewed is expert popularity (does a hot set exist at all)
  - how much does a layer's expert set repeat from one token to the next
    (that is the overlap a "reuse the previous token's experts" prefetch would hit)
  - how much of the whole model's expert mass is ever touched
"""

import argparse
import collections
import csv
import json


def load(path):
    """Returns per-layer list of selections in token order."""
    by_layer = collections.defaultdict(list)
    order = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                il = int(r["layer"])
                ids = [int(x) for x in r["ids"].split()] if r["ids"] else []
                n_ids = int(r["n_ids"])
            except (TypeError, ValueError):
                continue
            if ids:
                by_layer[il].append(sorted(set(ids)))
                order.append((il, n_ids))
    return by_layer, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--layout")
    ap.add_argument("--n-expert", type=int, default=256)
    args = ap.parse_args()

    for path in args.traces:
        by_layer, order = load(path)
        n_layer = len(by_layer)
        n_tok = min(len(v) for v in by_layer.values())
        print(f"\n=== {path}: {n_layer} layers, {n_tok} tokens ===")

        # 1. popularity skew, aggregated over layers
        tot_sel = 0
        touched = 0
        top_share = []
        for il, sels in by_layer.items():
            cnt = collections.Counter()
            for s in sels:
                cnt.update(s)
            tot_sel += sum(cnt.values())
            touched += len(cnt)
            ranked = sorted(cnt.values(), reverse=True)
            n = sum(ranked)
            # share of selections served by the hottest 8, 16, 32, 64 experts
            top_share.append([sum(ranked[:k]) / n for k in (8, 16, 32, 64)])
        avg = [sum(c[i] for c in top_share) / len(top_share) for i in range(4)]
        print(f"  experts touched      : {touched} of {n_layer * args.n_expert} "
              f"({100.0 * touched / (n_layer * args.n_expert):.1f}%)")
        print(f"  selections served by the hottest k experts of a layer:")
        for k, sh in zip((8, 16, 32, 64), avg):
            print(f"    k={k:3d}: {100.0 * sh:5.1f}%")

        # 2. token-to-token overlap within a layer
        for lag in (1, 2, 4, 8):
            ov = []
            for il, sels in by_layer.items():
                for t in range(lag, len(sels)):
                    a, b = set(sels[t]), set(sels[t - lag])
                    ov.append(len(a & b) / len(a))
            print(f"  overlap with token t-{lag:<2d}      : {100.0 * sum(ov) / len(ov):5.1f}% "
                  f"of this token's experts were also used {lag} token(s) ago")

        # 3. how many distinct experts a layer uses over a window of tokens
        for w in (8, 32, 128):
            tots = []
            for il, sels in by_layer.items():
                for t in range(0, max(1, len(sels) - w), w):
                    u = set()
                    for s in sels[t:t + w]:
                        u.update(s)
                    tots.append(len(u))
            if tots:
                print(f"  distinct experts per layer in {w:3d} tokens: {sum(tots) / len(tots):6.1f}")

        # 4. working set of the whole model for one token vs the hot set
        if args.layout:
            lay = json.load(open(args.layout))
            per = {}
            for g in lay["groups"]:
                for il in g["layers"]:
                    per[il] = sum(g["per_tensor_bytes"])
            one_token = sum(per.get(il, 0) * len(by_layer[il][0]) for il in by_layer)
            print(f"  experts of one token : {one_token / 2**20:.0f} MiB")
            for k in (16, 32, 64):
                hot = 0
                for il, sels in by_layer.items():
                    cnt = collections.Counter()
                    for s in sels:
                        cnt.update(s)
                    hot += per.get(il, 0) * min(k, len(cnt))
                print(f"  hottest {k:3d} per layer    : {hot / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
