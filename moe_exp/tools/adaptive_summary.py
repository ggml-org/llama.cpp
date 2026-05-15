import argparse
import csv
import os
import sys
import struct
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

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


def eval_frozen_with_churn(records, hotsets, warmup_tokens):
    hit_rate, hits, misses, future_tokens = eval_future(records, hotsets, warmup_tokens)

    # Initial install of the frozen hotsets.
    hotset_adds = sum(len(hs) for hs in hotsets.values())
    hotset_evictions = 0

    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "misses": misses,
        "future_tokens": future_tokens,
        "hotset_adds": hotset_adds,
        "hotset_evictions": hotset_evictions,
    }


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


def hotset_churn(old_hotsets, new_hotsets):
    adds = 0
    evictions = 0

    layers = set(old_hotsets.keys()) | set(new_hotsets.keys())

    for layer in layers:
        old = old_hotsets.get(layer, set())
        new = new_hotsets.get(layer, set())

        adds += len(new - old)
        evictions += len(old - new)

    return adds, evictions


def group_decode_by_token(records):
    dec = decode_records(records)
    by_token = defaultdict(list)

    for r in dec:
        by_token[r["token"]].append(r)

    tokens = sorted(by_token.keys())
    return tokens, by_token


def refresh_hotsets_from_counts(counts, budget_per_layer):
    return {
        layer: {e for e, _ in c.most_common(budget_per_layer)}
        for layer, c in counts.items()
    }


def eval_online_cumulative(records, warmup_tokens, budget_per_layer, refresh_every):
    tokens, by_token = group_decode_by_token(records)

    counts = defaultdict(Counter)
    hotsets = {}

    hits = 0
    misses = 0
    hotset_adds = 0
    hotset_evictions = 0

    for pos, tok in enumerate(tokens):
        # Evaluate current token using hotset built from previous tokens.
        if pos >= warmup_tokens:
            for r in by_token[tok]:
                hs = hotsets.get(r["layer"], set())
                for e in r["experts"]:
                    if e in hs:
                        hits += 1
                    else:
                        misses += 1

        # Observe current token.
        for r in by_token[tok]:
            for e in r["experts"]:
                counts[r["layer"]][e] += 1

        # Refresh after observing current token.
        if refresh_every <= 1 or (pos + 1) % refresh_every == 0:
            new_hotsets = refresh_hotsets_from_counts(counts, budget_per_layer)
            adds, evictions = hotset_churn(hotsets, new_hotsets)

            hotset_adds += adds
            hotset_evictions += evictions
            hotsets = new_hotsets

    future_tokens = max(0, len(tokens) - warmup_tokens)
    total = hits + misses
    hit_rate = hits / total if total else 0.0

    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "misses": misses,
        "future_tokens": future_tokens,
        "hotset_adds": hotset_adds,
        "hotset_evictions": hotset_evictions,
    }


def eval_online_sliding(
    records, warmup_tokens, budget_per_layer, window_tokens, refresh_every
):
    tokens, by_token = group_decode_by_token(records)

    counts = defaultdict(Counter)
    hotsets = {}

    hits = 0
    misses = 0
    hotset_adds = 0
    hotset_evictions = 0

    for pos, tok in enumerate(tokens):
        # Evaluate current token using hotset built from previous tokens.
        if pos >= warmup_tokens:
            for r in by_token[tok]:
                hs = hotsets.get(r["layer"], set())
                for e in r["experts"]:
                    if e in hs:
                        hits += 1
                    else:
                        misses += 1

        # Add current token to sliding counters.
        for r in by_token[tok]:
            for e in r["experts"]:
                counts[r["layer"]][e] += 1

        # Remove token that falls out of the window.
        old_pos = pos - window_tokens
        if old_pos >= 0:
            old_tok = tokens[old_pos]
            for r in by_token[old_tok]:
                for e in r["experts"]:
                    counts[r["layer"]][e] -= 1
                    if counts[r["layer"]][e] <= 0:
                        del counts[r["layer"]][e]

        # Refresh after observing current token.
        if refresh_every <= 1 or (pos + 1) % refresh_every == 0:
            new_hotsets = refresh_hotsets_from_counts(counts, budget_per_layer)
            adds, evictions = hotset_churn(hotsets, new_hotsets)

            hotset_adds += adds
            hotset_evictions += evictions
            hotsets = new_hotsets

    future_tokens = max(0, len(tokens) - warmup_tokens)
    total = hits + misses
    hit_rate = hits / total if total else 0.0

    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "misses": misses,
        "future_tokens": future_tokens,
        "hotset_adds": hotset_adds,
        "hotset_evictions": hotset_evictions,
    }


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


def run_one_job(job):
    (
        trace,
        warmups,
        budgets,
        expert_total_gib,
        n_layer,
        n_expert,
        policy,
        window_tokens,
        refresh_every,
    ) = job

    return run_one(
        trace,
        warmups,
        budgets,
        expert_total_gib,
        n_layer,
        n_expert,
        policy,
        window_tokens,
        refresh_every,
    )


def run_one(
    trace_path,
    warmups,
    budgets,
    expert_total_gib,
    n_layer,
    n_expert,
    policy,
    window_tokens,
    refresh_every,
):
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

            if policy == "frozen":
                hotsets, _ = build_hotsets_from_warmup(records, warmup, budget)
                result = eval_frozen_with_churn(records, hotsets, warmup)
            elif policy == "cumulative":
                result = eval_online_cumulative(
                    records,
                    warmup,
                    budget,
                    refresh_every,
                )
            elif policy == "sliding":
                result = eval_online_sliding(
                    records,
                    warmup,
                    budget,
                    window_tokens,
                    refresh_every,
                )
            else:
                raise ValueError(f"unknown policy: {policy}")

            hit_rate = result["hit_rate"]
            hits = result["hits"]
            misses = result["misses"]
            future_tokens = result["future_tokens"]
            hotset_adds = result["hotset_adds"]
            hotset_evictions = result["hotset_evictions"]

            cold_mib_per_token = (
                misses / future_tokens * avg_layer_expert_mib
                if future_tokens > 0
                else 0.0
            )

            reload_mib_per_token = (
                hotset_adds / future_tokens * avg_layer_expert_mib
                if future_tokens > 0
                else 0.0
            )

            total_mib_per_token = cold_mib_per_token + reload_mib_per_token

            hotset_adds_per_token = (
                hotset_adds / future_tokens if future_tokens > 0 else 0.0
            )

            hotset_evictions_per_token = (
                hotset_evictions / future_tokens if future_tokens > 0 else 0.0
            )

            rows.append(
                {
                    "run_name": meta.get("run_name", os.path.basename(run_dir)),
                    "prompt_name": meta.get("prompt_name", ""),
                    "seed": meta.get("seed", ""),
                    "decode_tokens": decode_tokens,
                    "policy": policy,
                    "window_tokens": window_tokens if policy == "sliding" else "",
                    "refresh_every": refresh_every if policy != "frozen" else "",
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
                    "hotset_adds": hotset_adds,
                    "hotset_evictions": hotset_evictions,
                    "hotset_adds_per_token": hotset_adds_per_token,
                    "hotset_evictions_per_token": hotset_evictions_per_token,
                    "reload_mib_per_token": reload_mib_per_token,
                    "total_mib_per_token": total_mib_per_token,
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
    ap.add_argument(
        "--policy",
        choices=["frozen", "cumulative", "sliding"],
        default="frozen",
    )
    ap.add_argument("--window-tokens", type=int, default=128)
    ap.add_argument("--refresh-every", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    warmups = [int(x) for x in args.warmups.split(",") if x]
    budgets = [int(x) for x in args.budgets.split(",") if x]

    jobs = [
        (
            trace,
            warmups,
            budgets,
            args.expert_total_gib,
            args.n_layer,
            args.n_expert,
            args.policy,
            args.window_tokens,
            args.refresh_every,
        )
        for trace in args.traces
    ]

    rows = []

    if args.jobs <= 1:
        for job in jobs:
            rows.extend(run_one_job(job))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for part in ex.map(run_one_job, jobs):
                rows.extend(part)

    fields = [
        "run_name",
        "prompt_name",
        "seed",
        "decode_tokens",
        "policy",
        "window_tokens",
        "refresh_every",
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
        "hotset_adds",
        "hotset_evictions",
        "hotset_adds_per_token",
        "hotset_evictions_per_token",
        "reload_mib_per_token",
        "total_mib_per_token",
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
