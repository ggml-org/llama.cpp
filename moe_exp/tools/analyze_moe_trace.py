#!/usr/bin/env python3

import argparse
import os
import struct
import math
from collections import Counter, defaultdict

REC = struct.Struct("<I H B B 8h")

PHASE = {
    0: "prefill",
    1: "decode",
}


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


def top_mass(counter, ks):
    total = sum(counter.values())
    vals = [v for _, v in counter.most_common()]
    out = {}
    for k in ks:
        out[k] = sum(vals[:k]) / total if total else 0.0
    return out


def hotset_size_for_mass(counter, targets):
    total = sum(counter.values())
    vals = [v for _, v in counter.most_common()]

    out = {}
    if total == 0:
        for target in targets:
            out[target] = 0
        return out

    for target in targets:
        acc = 0
        size = 0

        for v in vals:
            acc += v
            size += 1

            if acc / total >= target:
                break

        out[target] = size

    return out


def print_hotset_size_summary(name, records, n_layer=48):
    by_layer = layer_counters(records)
    targets = [0.50, 0.75, 0.90, 0.95]

    print()
    print(f"== {name}: hotset size by target mass ==")
    print("layer p50 p75 p90 p95 unique")

    for layer in range(n_layer):
        c = by_layer[layer]
        hs = hotset_size_for_mass(c, targets)

        print(
            f"{layer:02d}    "
            f"{hs[0.50]:3d} "
            f"{hs[0.75]:3d} "
            f"{hs[0.90]:3d} "
            f"{hs[0.95]:3d} "
            f"{len(c):3d}"
        )


def top_set(counter, k):
    return {item for item, _ in counter.most_common(k)}


def jaccard(a, b):
    if not a and not b:
        return 1.0

    union = a | b
    if not union:
        return 0.0

    return len(a & b) / len(union)


def split_decode_records(records):
    decode = [r for r in records if r["phase"] == 1]
    if not decode:
        return [], []

    tokens = sorted({r["token"] for r in decode})
    mid = len(tokens) // 2
    first_tokens = set(tokens[:mid])
    second_tokens = set(tokens[mid:])

    first = [r for r in decode if r["token"] in first_tokens]
    second = [r for r in decode if r["token"] in second_tokens]

    return first, second


def layer_counters(records):
    by_layer = defaultdict(Counter)

    for r in records:
        for e in r["experts"]:
            by_layer[r["layer"]][e] += 1

    return by_layer


def print_decode_split_stability(records, n_layer=48, ks=(8, 16, 32, 64)):
    first, second = split_decode_records(records)

    c1 = layer_counters(first)
    c2 = layer_counters(second)

    print()
    print("== decode split stability: first half vs second half ==")
    print(f"first_half_records: {len(first)}")
    print(f"second_half_records: {len(second)}")

    for k in ks:
        vals = []

        print()
        print(f"top{k} Jaccard by layer:")
        print("layer jaccard first_unique second_unique")

        for layer in range(n_layer):
            a = top_set(c1[layer], k)
            b = top_set(c2[layer], k)
            score = jaccard(a, b)
            vals.append(score)

            print(
                f"{layer:02d}    "
                f"{score:.3f}   "
                f"{len(c1[layer]):3d} "
                f"{len(c2[layer]):3d}"
            )

        avg = sum(vals) / len(vals) if vals else 0.0
        print(f"avg_top{k}_jaccard: {avg:.3f}")


def print_phase_overlap(records, n_layer=48, ks=(8, 16, 32, 64)):
    prefill = [r for r in records if r["phase"] == 0]
    decode = [r for r in records if r["phase"] == 1]

    cp = layer_counters(prefill)
    cd = layer_counters(decode)

    print()
    print("== phase overlap: prefill vs decode ==")
    print(f"prefill_records: {len(prefill)}")
    print(f"decode_records: {len(decode)}")

    for k in ks:
        vals = []

        print()
        print(f"top{k} Jaccard by layer:")
        print("layer jaccard prefill_unique decode_unique")

        for layer in range(n_layer):
            a = top_set(cp[layer], k)
            b = top_set(cd[layer], k)
            score = jaccard(a, b)
            vals.append(score)

            print(
                f"{layer:02d}    "
                f"{score:.3f}   "
                f"{len(cp[layer]):3d} "
                f"{len(cd[layer]):3d}"
            )

        avg = sum(vals) / len(vals) if vals else 0.0
        print(f"avg_top{k}_jaccard: {avg:.3f}")


def gini(counter):
    vals = sorted(counter.values())
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0

    # Gini for non-negative values.
    weighted = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * weighted) / (n * total) - (n + 1) / n


def reuse_distances(events):
    last_pos = {}
    distances = []

    # Naive but fine for first traces.
    for i, key in enumerate(events):
        if key in last_pos:
            prev = last_pos[key]
            unique_between = len(set(events[prev + 1 : i]))
            distances.append(unique_between)
        last_pos[key] = i

    return distances


def cdf_points(values, thresholds):
    if not values:
        return {t: 0.0 for t in thresholds}

    values = sorted(values)
    n = len(values)

    out = {}
    j = 0
    for t in thresholds:
        while j < n and values[j] <= t:
            j += 1
        out[t] = j / n
    return out


def summarize(records, phase_filter=None):
    if phase_filter is not None:
        records = [r for r in records if r["phase"] == phase_filter]

    layer_expert_counts = defaultdict(Counter)
    global_layer_expert = Counter()
    global_expert = Counter()
    events = []

    for r in records:
        layer = r["layer"]
        for e in r["experts"]:
            layer_expert_counts[layer][e] += 1
            global_layer_expert[(layer, e)] += 1
            global_expert[e] += 1
            events.append((layer, e))

    return {
        "records": records,
        "events": events,
        "layer_expert_counts": layer_expert_counts,
        "global_layer_expert": global_layer_expert,
        "global_expert": global_expert,
    }


def print_summary(name, s, n_layer=48, n_expert=128):
    records = s["records"]
    events = s["events"]
    layer_counts = s["layer_expert_counts"]

    print()
    print(f"== {name} ==")
    print(f"records: {len(records)}")
    print(f"expert access events: {len(events)}")
    print(
        f"unique layer-experts touched: {len(s['global_layer_expert'])} / {n_layer * n_expert}"
    )
    print(f"unique expert ids touched globally: {len(s['global_expert'])} / {n_expert}")

    print()
    print("global expert top-20:")
    for e, c in s["global_expert"].most_common(20):
        print(f"  expert {e:03d}: {c}")

    print()
    print("per-layer summary:")
    print("layer unique top1 top8 top16 top32 gini")
    for layer in range(n_layer):
        c = layer_counts[layer]
        if not c:
            print(f"{layer:02d}    0      0.000 0.000 0.000 0.000 0.000")
            continue

        mass = top_mass(c, [1, 8, 16, 32])
        print(
            f"{layer:02d}    "
            f"{len(c):3d}    "
            f"{mass[1]:.3f} "
            f"{mass[8]:.3f} "
            f"{mass[16]:.3f} "
            f"{mass[32]:.3f} "
            f"{gini(c):.3f}"
        )

    print()
    print("reuse distance CDF over (layer, expert):")
    rds = reuse_distances(events)
    thresholds = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    cdf = cdf_points(rds, thresholds)
    print(f"reuse samples: {len(rds)}")
    for t in thresholds:
        print(f"  <= {t:4d}: {cdf[t]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--n-layer", type=int, default=48)
    ap.add_argument("--n-expert", type=int, default=128)
    args = ap.parse_args()

    records = list(read_records(args.trace))

    for phase in [None, 0, 1]:
        name = "all" if phase is None else PHASE[phase]
        s = summarize(records, phase)
        print_summary(name, s, args.n_layer, args.n_expert)

    print_hotset_size_summary(
        "decode",
        [r for r in records if r["phase"] == 1],
        args.n_layer,
    )

    print_decode_split_stability(
        records,
        args.n_layer,
        ks=(8, 16, 32, 64),
    )

    print_phase_overlap(
        records,
        args.n_layer,
        ks=(8, 16, 32, 64),
    )


if __name__ == "__main__":
    main()
