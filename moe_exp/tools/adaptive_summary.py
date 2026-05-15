#!/usr/bin/env python3

import argparse
import csv
import os
import sys
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


def build_hotsets_from_warmup(records, warmup_tokens, budget_per_layer):
    dec = decode_records(records)
    tokens = sorted({r["token"] for r in dec})
    warm = set(tokens[:warmup_tokens])

    counts = defaultdict(Counter)

    for r in dec:
        if r["token"] not in warm:
            continue
        for e in r["experts"]:
            counts[r["layer"]][e] += 1

    hotsets = {
        layer: {e for e, _ in c.most_common(budget_per_layer)}
        for layer, c in counts.items()
    }

    return hotsets, tokens


def eval_future(records, hotsets, warmup_tokens):
    dec = decode_records(records)
    tokens = sorted({r["token"] for r in dec})
    future = set(tokens[warmup_tokens:])

    hits = 0
    misses = 0

    for r in dec:
        if r["token"] not in future:
            continue

        hs = hotsets.get(r["layer"], set())

        for e in r["experts"]:
            if e in hs:
                hits += 1
            else:
                misses += 1

    total = hits + misses
    hit_rate = hits / total if total else 0.0

    return hit_rate, hits, misses, len(future)


def oracle_hit_rate(records, budget_per_layer):
    dec = decode_records(records)
    counts = defaultdict(Counter)

    for r in dec:
        for e in r["experts"]:
            counts[r["layer"]][e] += 1

    hotsets = {
        layer: {e for e, _ in c.most_common(budget_per_layer)}
        for layer, c in counts.items()
    }

    hits = 0
    misses = 0

    for r in dec:
        hs = hotsets.get(r["layer"], set())
        for e in r["experts"]:
            if e in hs:
                hits += 1
            else:
                misses += 1

    total = hits + misses
    return hits / total if total else 0.0


def read_metadata(run_dir):
    path = os.path.join(run_dir, "metadata.txt")
    out = {}

    if not os.path.exists(path):
        return out

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k] = v

    return out


def run_one(trace_path, warmups, budgets, expert_total_gib, n_layer, n_expert):
    run_dir = os.path.dirname(trace_path)
    meta = read_metadata(run_dir)
    records = list(read_records(trace_path))
    dec = decode_records(records)
    decode_tokens = len({r["token"] for r in dec})

    avg_layer_expert_mib = expert_total_gib * 1024.0 / (n_layer * n_expert)

    rows = []

    for budget in budgets:
        resident_expert_gib = expert_total_gib * min(budget, n_expert) / n_expert
        oracle = oracle_hit_rate(records, budget)

        for warmup in warmups:
            if warmup >= decode_tokens:
                continue

            hotsets, _ = build_hotsets_from_warmup(records, warmup, budget)
            hit_rate, hits, misses, future_tokens = eval_future(
                records, hotsets, warmup
            )

            cold_mib_per_token = (
                misses / future_tokens * avg_layer_expert_mib
                if future_tokens > 0
                else 0.0
            )

            rows.append(
                {
                    "run_name": meta.get("run_name", os.path.basename(run_dir)),
                    "prompt_name": meta.get("prompt_name", ""),
                    "seed": meta.get("seed", ""),
                    "decode_tokens": decode_tokens,
                    "warmup": warmup,
                    "future_tokens": future_tokens,
                    "budget": budget,
                    "resident_expert_gib": resident_expert_gib,
                    "hit_rate": hit_rate,
                    "miss_rate": 1.0 - hit_rate,
                    "oracle_hit_rate": oracle,
                    "hits": hits,
                    "misses": misses,
                    "cold_mib_per_token": cold_mib_per_token,
                    "trace_path": trace_path,
                }
            )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--warmups", default="8,16,32,64,128")
    ap.add_argument("--budgets", default="16,24,32,48,64,96,128")
    ap.add_argument("--expert-total-gib", type=float, default=16.348)
    ap.add_argument("--n-layer", type=int, default=48)
    ap.add_argument("--n-expert", type=int, default=128)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    warmups = [int(x) for x in args.warmups.split(",") if x]
    budgets = [int(x) for x in args.budgets.split(",") if x]

    rows = []
    for trace in args.traces:
        rows.extend(
            run_one(
                trace,
                warmups,
                budgets,
                args.expert_total_gib,
                args.n_layer,
                args.n_expert,
            )
        )

    fields = [
        "run_name",
        "prompt_name",
        "seed",
        "decode_tokens",
        "warmup",
        "future_tokens",
        "budget",
        "resident_expert_gib",
        "hit_rate",
        "miss_rate",
        "oracle_hit_rate",
        "hits",
        "misses",
        "cold_mib_per_token",
        "trace_path",
    ]

    if args.out == "-":
        f = None
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
