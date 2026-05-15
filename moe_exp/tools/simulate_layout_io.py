from __future__ import annotations

import argparse
import csv
import os
import re
import struct
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REC = struct.Struct("<I H B B 8h")
PHASE_PREFILL = 0
PHASE_DECODE = 1


@dataclass(frozen=True)
class Access:
    token: int
    layer: int
    experts: tuple[int, ...]


@dataclass(frozen=True)
class Range:
    offset: int
    nbytes: int

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class RangeStats:
    needed_bytes: int
    raw_ranges: int
    coalesced_bytes: int
    coalesced_ranges: int

    @property
    def overread_bytes(self) -> int:
        return self.coalesced_bytes - self.needed_bytes


def mib(x: float) -> float:
    return x / 1024.0 / 1024.0


def parse_run_name(path: str) -> tuple[str, str, str]:
    run_name = Path(path).parent.name
    m = re.match(r"(.+)_seed(\d+)$", run_name)
    if m:
        return run_name, m.group(1), m.group(2)
    return run_name, run_name, ""


def read_trace(path: str) -> list[Access]:
    accesses: list[Access] = []

    with open(path, "rb") as f:
        data = f.read()

    if len(data) % REC.size != 0:
        raise ValueError(
            f"{path}: bad trace size {len(data)}, not divisible by {REC.size}"
        )

    for off in range(0, len(data), REC.size):
        token, layer, phase, n_used, *experts = REC.unpack_from(data, off)
        if phase != PHASE_DECODE:
            continue

        if n_used > len(experts):
            raise ValueError(f"{path}: bad n_expert_used={n_used}")

        selected = tuple(int(x) for x in experts[:n_used])
        accesses.append(Access(token=int(token), layer=int(layer), experts=selected))

    return accesses


def group_decode_by_token(accesses: list[Access]) -> list[list[Access]]:
    by_token: dict[int, list[Access]] = defaultdict(list)
    for a in accesses:
        by_token[a.token].append(a)

    return [by_token[t] for t in sorted(by_token)]


def token_pairs(token_accesses: list[Access]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for a in token_accesses:
        for e in a.experts:
            pairs.add((a.layer, e))
    return pairs


def update_counts(
    counts: dict[int, dict[int, int]], token_accesses: list[Access]
) -> None:
    for a in token_accesses:
        layer_counts = counts[a.layer]
        for e in a.experts:
            layer_counts[e] += 1


def build_hotset_from_counts(
    counts: dict[int, dict[int, int]],
    *,
    budget: int,
    n_layer: int,
) -> dict[int, set[int]]:
    hotset: dict[int, set[int]] = {}

    for layer in range(n_layer):
        items = counts.get(layer, {})
        ranked = sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))
        hotset[layer] = {expert for expert, _count in ranked[:budget]}

    return hotset


def hotset_pairs(hotset: dict[int, set[int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for layer, experts in hotset.items():
        for expert in experts:
            out.add((layer, expert))
    return out


def load_layout(path: str) -> dict[tuple[int, int], list[Range]]:
    layout: dict[tuple[int, int], list[Range]] = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            expert = int(row["expert"])

            # expert=-1 means the extractor could not isolate contiguous expert ranges.
            if expert < 0:
                continue

            layout[(layer, expert)].append(
                Range(offset=int(row["offset"]), nbytes=int(row["nbytes"]))
            )

    for key in list(layout.keys()):
        layout[key].sort(key=lambda r: r.offset)

    return dict(layout)


def coalesce_ranges(ranges: Iterable[Range], gap_bytes: int) -> RangeStats:
    rs = sorted(ranges, key=lambda r: r.offset)

    needed = sum(r.nbytes for r in rs)
    raw_count = len(rs)

    if not rs:
        return RangeStats(
            needed_bytes=0,
            raw_ranges=0,
            coalesced_bytes=0,
            coalesced_ranges=0,
        )

    merged: list[Range] = []
    cur_start = rs[0].offset
    cur_end = rs[0].end

    for r in rs[1:]:
        if r.offset <= cur_end + gap_bytes:
            cur_end = max(cur_end, r.end)
        else:
            merged.append(Range(cur_start, cur_end - cur_start))
            cur_start = r.offset
            cur_end = r.end

    merged.append(Range(cur_start, cur_end - cur_start))

    return RangeStats(
        needed_bytes=needed,
        raw_ranges=raw_count,
        coalesced_bytes=sum(r.nbytes for r in merged),
        coalesced_ranges=len(merged),
    )


def ranges_for_pairs(
    pairs: Iterable[tuple[int, int]],
    layout: dict[tuple[int, int], list[Range]],
) -> list[Range]:
    out: list[Range] = []

    for pair in pairs:
        rs = layout.get(pair)
        if rs is None:
            raise KeyError(
                f"layout missing expert range for layer={pair[0]} expert={pair[1]}"
            )
        out.extend(rs)

    return out


def add_stats(a: RangeStats, b: RangeStats) -> RangeStats:
    return RangeStats(
        needed_bytes=a.needed_bytes + b.needed_bytes,
        raw_ranges=a.raw_ranges + b.raw_ranges,
        coalesced_bytes=a.coalesced_bytes + b.coalesced_bytes,
        coalesced_ranges=a.coalesced_ranges + b.coalesced_ranges,
    )


ZERO_STATS = RangeStats(0, 0, 0, 0)


def eval_run(
    *,
    trace_path: str,
    layout_path: str,
    policy: str,
    warmup: int,
    budget: int,
    n_layer: int,
    window_tokens: int,
    refresh_every: int,
    coalesce_gap_bytes: int,
    ssd_mib_per_sec: float,
) -> dict[str, object]:
    layout = load_layout(layout_path)
    grouped = group_decode_by_token(read_trace(trace_path))

    decode_tokens = len(grouped)
    if decode_tokens <= warmup:
        raise ValueError(
            f"{trace_path}: decode_tokens={decode_tokens} <= warmup={warmup}"
        )

    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # For sliding policy, keep token-level history so we can subtract old accesses.
    window: deque[list[Access]] = deque()

    for token in grouped[:warmup]:
        update_counts(counts, token)
        if policy == "sliding":
            window.append(token)

    if policy == "sliding":
        while len(window) > window_tokens:
            old = window.popleft()
            for a in old:
                for e in a.experts:
                    counts[a.layer][e] -= 1
                    if counts[a.layer][e] <= 0:
                        del counts[a.layer][e]

    hotset = build_hotset_from_counts(counts, budget=budget, n_layer=n_layer)

    future_tokens = decode_tokens - warmup

    cold_stats = ZERO_STATS
    reload_stats = ZERO_STATS

    hits = 0
    misses = 0

    # Count initial hotset install as reload traffic, amortized over future tokens.
    initial_pairs = hotset_pairs(hotset)
    reload_stats = add_stats(
        reload_stats,
        coalesce_ranges(
            ranges_for_pairs(initial_pairs, layout),
            gap_bytes=coalesce_gap_bytes,
        ),
    )

    for i, token in enumerate(grouped[warmup:], start=1):
        pairs = token_pairs(token)

        missing: set[tuple[int, int]] = set()
        for layer, expert in pairs:
            if expert in hotset.get(layer, set()):
                hits += 1
            else:
                misses += 1
                missing.add((layer, expert))

        cold_stats = add_stats(
            cold_stats,
            coalesce_ranges(
                ranges_for_pairs(missing, layout),
                gap_bytes=coalesce_gap_bytes,
            ),
        )

        if policy in ("cumulative", "sliding"):
            update_counts(counts, token)

            if policy == "sliding":
                window.append(token)
                while len(window) > window_tokens:
                    old = window.popleft()
                    for a in old:
                        for e in a.experts:
                            counts[a.layer][e] -= 1
                            if counts[a.layer][e] <= 0:
                                del counts[a.layer][e]

            if refresh_every <= 0:
                raise ValueError("--refresh-every must be > 0 for online policies")

            if i % refresh_every == 0:
                old_pairs = hotset_pairs(hotset)
                hotset = build_hotset_from_counts(
                    counts, budget=budget, n_layer=n_layer
                )
                new_pairs = hotset_pairs(hotset)

                added = new_pairs - old_pairs
                reload_stats = add_stats(
                    reload_stats,
                    coalesce_ranges(
                        ranges_for_pairs(added, layout),
                        gap_bytes=coalesce_gap_bytes,
                    ),
                )

    total_stats = add_stats(cold_stats, reload_stats)

    run_name, prompt_name, seed = parse_run_name(trace_path)

    total_accesses = hits + misses
    hit_rate = hits / total_accesses if total_accesses else 0.0

    def per_tok_bytes(x: int) -> float:
        return x / future_tokens

    def per_tok_count(x: int) -> float:
        return x / future_tokens

    estimated_read_ms_per_token = (
        mib(per_tok_bytes(total_stats.coalesced_bytes)) / ssd_mib_per_sec * 1000.0
        if ssd_mib_per_sec > 0
        else 0.0
    )

    return {
        "run_name": run_name,
        "prompt_name": prompt_name,
        "seed": seed,
        "trace_path": trace_path,
        "policy": policy,
        "warmup": warmup,
        "budget": budget,
        "window_tokens": window_tokens if policy == "sliding" else "",
        "refresh_every": refresh_every if policy in ("cumulative", "sliding") else "",
        "decode_tokens": decode_tokens,
        "future_tokens": future_tokens,
        "hit_rate": hit_rate,
        "miss_rate": 1.0 - hit_rate,
        "hits": hits,
        "misses": misses,
        "cold_needed_mib_per_token": mib(per_tok_bytes(cold_stats.needed_bytes)),
        "cold_raw_ranges_per_token": per_tok_count(cold_stats.raw_ranges),
        "cold_coalesced_mib_per_token": mib(per_tok_bytes(cold_stats.coalesced_bytes)),
        "cold_coalesced_ranges_per_token": per_tok_count(cold_stats.coalesced_ranges),
        "cold_overread_mib_per_token": mib(per_tok_bytes(cold_stats.overread_bytes)),
        "reload_needed_mib_per_token": mib(per_tok_bytes(reload_stats.needed_bytes)),
        "reload_raw_ranges_per_token": per_tok_count(reload_stats.raw_ranges),
        "reload_coalesced_mib_per_token": mib(
            per_tok_bytes(reload_stats.coalesced_bytes)
        ),
        "reload_coalesced_ranges_per_token": per_tok_count(
            reload_stats.coalesced_ranges
        ),
        "reload_overread_mib_per_token": mib(
            per_tok_bytes(reload_stats.overread_bytes)
        ),
        "total_needed_mib_per_token": mib(per_tok_bytes(total_stats.needed_bytes)),
        "total_raw_ranges_per_token": per_tok_count(total_stats.raw_ranges),
        "total_coalesced_mib_per_token": mib(
            per_tok_bytes(total_stats.coalesced_bytes)
        ),
        "total_coalesced_ranges_per_token": per_tok_count(total_stats.coalesced_ranges),
        "total_overread_mib_per_token": mib(per_tok_bytes(total_stats.overread_bytes)),
        "estimated_read_ms_per_token": estimated_read_ms_per_token,
        "coalesce_gap_kib": coalesce_gap_bytes / 1024.0,
        "ssd_mib_per_sec": ssd_mib_per_sec,
    }


def run_one_job(job: dict[str, object]) -> dict[str, object]:
    return eval_run(**job)  # type: ignore[arg-type]


def summarize(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    keys = [
        "hit_rate",
        "total_needed_mib_per_token",
        "total_raw_ranges_per_token",
        "total_coalesced_mib_per_token",
        "total_coalesced_ranges_per_token",
        "total_overread_mib_per_token",
        "estimated_read_ms_per_token",
    ]

    print("== IO summary ==")
    print(f"runs: {len(rows)}")

    for key in keys:
        vals = sorted(float(r[key]) for r in rows)
        avg = sum(vals) / len(vals)
        p10 = vals[max(0, int(len(vals) * 0.10) - 1)]
        p50 = vals[len(vals) // 2]
        p90 = vals[min(len(vals) - 1, int(len(vals) * 0.90))]
        print(
            f"{key:34s} "
            f"avg={avg:9.3f} p10={p10:9.3f} p50={p50:9.3f} p90={p90:9.3f} "
            f"min={vals[0]:9.3f} max={vals[-1]:9.3f}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True)
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--out", required=True)

    p.add_argument(
        "--policy", choices=["frozen", "cumulative", "sliding"], required=True
    )
    p.add_argument("--warmups", type=int, nargs="+", default=[64])
    p.add_argument("--budgets", type=int, nargs="+", default=[64])

    p.add_argument("--n-layer", type=int, default=48)
    p.add_argument("--window-tokens", type=int, default=128)
    p.add_argument("--refresh-every", type=int, default=16)

    p.add_argument("--coalesce-gap-kib", type=int, default=0)
    p.add_argument(
        "--ssd-mib-per-sec",
        type=float,
        default=4635.0,
        help="4861 MB/s decimal is about 4635 MiB/s",
    )
    p.add_argument("--jobs", type=int, default=1)

    args = p.parse_args()

    jobs: list[dict[str, object]] = []
    coalesce_gap_bytes = args.coalesce_gap_kib * 1024

    for trace_path in args.traces:
        for warmup in args.warmups:
            for budget in args.budgets:
                jobs.append(
                    {
                        "trace_path": trace_path,
                        "layout_path": args.layout,
                        "policy": args.policy,
                        "warmup": warmup,
                        "budget": budget,
                        "n_layer": args.n_layer,
                        "window_tokens": args.window_tokens,
                        "refresh_every": args.refresh_every,
                        "coalesce_gap_bytes": coalesce_gap_bytes,
                        "ssd_mib_per_sec": args.ssd_mib_per_sec,
                    }
                )

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(run_one_job, jobs))
    else:
        rows = [run_one_job(job) for job in jobs]

    rows.sort(
        key=lambda r: (
            str(r["policy"]),
            int(r["warmup"]),
            int(r["budget"]),
            str(r["run_name"]),
        )
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
