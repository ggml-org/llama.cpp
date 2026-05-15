#!/usr/bin/env python3

import argparse
import os
import struct
from collections import Counter, defaultdict

REC = struct.Struct("<I H B B 8h")


def read_records(path):
    with open(path, "rb") as f:
        idx = 0
        while data := f.read(REC.size):
            token_idx, layer_idx, phase, k, *experts = REC.unpack(data)
            yield {
                "idx": idx,
                "token": token_idx,
                "layer": layer_idx,
                "phase": phase,
                "experts": experts[:k],
            }
            idx += 1


def decode_records(records):
    return [r for r in records if r["phase"] == 1]


def build_layer_hotsets(records, warmup_tokens, budget_per_layer):
    decode = decode_records(records)
    tokens = sorted({r["token"] for r in decode})
    warm_tokens = set(tokens[:warmup_tokens])

    counts = defaultdict(Counter)

    for r in decode:
        if r["token"] not in warm_tokens:
            continue
        for e in r["experts"]:
            counts[r["layer"]][e] += 1

    hotsets = {}
    for layer, c in counts.items():
        hotsets[layer] = {e for e, _ in c.most_common(budget_per_layer)}

    return hotsets, tokens


def eval_hotsets(records, hotsets, warmup_tokens):
    decode = decode_records(records)
    tokens = sorted({r["token"] for r in decode})
    future_tokens = set(tokens[warmup_tokens:])

    hits = 0
    misses = 0
    by_layer = defaultdict(lambda: [0, 0])  # hits, misses

    for r in decode:
        if r["token"] not in future_tokens:
            continue

        hs = hotsets.get(r["layer"], set())

        for e in r["experts"]:
            if e in hs:
                hits += 1
                by_layer[r["layer"]][0] += 1
            else:
                misses += 1
                by_layer[r["layer"]][1] += 1

    total = hits + misses
    hit_rate = hits / total if total else 0.0

    return hit_rate, hits, misses, by_layer, len(future_tokens)


def oracle_static_hit_rate(records, budget_per_layer):
    # Upper-ish bound for static per-layer hotset on this same trace:
    # build hotset from all decode, evaluate on all decode.
    decode = decode_records(records)
    counts = defaultdict(Counter)

    for r in decode:
        for e in r["experts"]:
            counts[r["layer"]][e] += 1

    hotsets = {
        layer: {e for e, _ in c.most_common(budget_per_layer)}
        for layer, c in counts.items()
    }

    hits = 0
    misses = 0

    for r in decode:
        hs = hotsets.get(r["layer"], set())
        for e in r["experts"]:
            if e in hs:
                hits += 1
            else:
                misses += 1

    total = hits + misses
    return hits / total if total else 0.0


def run_one(path, warmups, budgets):
    records = list(read_records(path))
    decode = decode_records(records)
    n_decode_tokens = len({r["token"] for r in decode})

    rows = []

    for budget in budgets:
        oracle = oracle_static_hit_rate(records, budget)

        for warmup in warmups:
            if warmup >= n_decode_tokens:
                continue

            hotsets, _ = build_layer_hotsets(records, warmup, budget)
            hit_rate, hits, misses, by_layer, future_n = eval_hotsets(
                records, hotsets, warmup
            )

            rows.append(
                {
                    "path": path,
                    "name": os.path.basename(os.path.dirname(path)),
                    "decode_tokens": n_decode_tokens,
                    "warmup": warmup,
                    "future_tokens": future_n,
                    "budget": budget,
                    "hit_rate": hit_rate,
                    "miss_rate": 1.0 - hit_rate,
                    "hits": hits,
                    "misses": misses,
                    "oracle_hit_rate": oracle,
                }
            )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--warmups", default="4,8,16,32,64")
    ap.add_argument("--budgets", default="8,16,24,32,48,64,96,128")
    args = ap.parse_args()

    warmups = [int(x) for x in args.warmups.split(",") if x]
    budgets = [int(x) for x in args.budgets.split(",") if x]

    all_rows = []
    for path in args.traces:
        all_rows.extend(run_one(path, warmups, budgets))

    print(
        "name,decode_tokens,warmup,future_tokens,budget,hit_rate,miss_rate,oracle_hit_rate,hits,misses"
    )

    for r in all_rows:
        print(
            f"{r['name']},"
            f"{r['decode_tokens']},"
            f"{r['warmup']},"
            f"{r['future_tokens']},"
            f"{r['budget']},"
            f"{r['hit_rate']:.6f},"
            f"{r['miss_rate']:.6f},"
            f"{r['oracle_hit_rate']:.6f},"
            f"{r['hits']},"
            f"{r['misses']}"
        )


if __name__ == "__main__":
    main()
