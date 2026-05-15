#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import struct
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REC = struct.Struct("<I H B B 8h")
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
    layer: int = -1
    expert: int = -1
    part: str = ""

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class MergedRange:
    offset: int
    nbytes: int
    needed_nbytes: int
    raw_ranges: int

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    @property
    def overread_nbytes(self) -> int:
        return self.nbytes - self.needed_nbytes


def parse_run_name(path: str) -> tuple[str, str, str]:
    run_name = Path(path).parent.name
    m = re.match(r"(.+)_seed(\d+)$", run_name)
    if m:
        return run_name, m.group(1), m.group(2)
    return run_name, run_name, ""


def read_trace(path: str) -> list[Access]:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) % REC.size != 0:
        raise ValueError(
            f"{path}: bad trace size {len(data)}, not divisible by {REC.size}"
        )

    accesses: list[Access] = []

    for off in range(0, len(data), REC.size):
        token, layer, phase, n_used, *experts = REC.unpack_from(data, off)
        if phase != PHASE_DECODE:
            continue

        selected = tuple(int(x) for x in experts[:n_used])
        accesses.append(Access(token=int(token), layer=int(layer), experts=selected))

    return accesses


def group_decode_by_token(accesses: list[Access]) -> list[tuple[int, list[Access]]]:
    by_token: dict[int, list[Access]] = defaultdict(list)
    for a in accesses:
        by_token[a.token].append(a)

    return [(t, by_token[t]) for t in sorted(by_token)]


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


def subtract_counts(
    counts: dict[int, dict[int, int]], token_accesses: list[Access]
) -> None:
    for a in token_accesses:
        layer_counts = counts[a.layer]
        for e in a.experts:
            layer_counts[e] -= 1
            if layer_counts[e] <= 0:
                del layer_counts[e]


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
            if expert < 0:
                continue

            layout[(layer, expert)].append(
                Range(
                    offset=int(row["offset"]),
                    nbytes=int(row["nbytes"]),
                    layer=layer,
                    expert=expert,
                    part=row.get("part", ""),
                )
            )

    for key in list(layout.keys()):
        layout[key].sort(key=lambda r: r.offset)

    return dict(layout)


def ranges_for_pairs(
    pairs: Iterable[tuple[int, int]],
    layout: dict[tuple[int, int], list[Range]],
) -> list[Range]:
    out: list[Range] = []

    for pair in sorted(pairs):
        rs = layout.get(pair)
        if rs is None:
            raise KeyError(
                f"layout missing expert range for layer={pair[0]} expert={pair[1]}"
            )
        out.extend(rs)

    out.sort(key=lambda r: r.offset)
    return out


def coalesce_ranges(ranges: Iterable[Range], gap_bytes: int) -> list[MergedRange]:
    rs = sorted(ranges, key=lambda r: r.offset)
    if not rs:
        return []

    merged: list[MergedRange] = []

    cur_start = rs[0].offset
    cur_end = rs[0].end
    cur_needed = rs[0].nbytes
    cur_raw = 1

    for r in rs[1:]:
        if r.offset <= cur_end + gap_bytes:
            cur_end = max(cur_end, r.end)
            cur_needed += r.nbytes
            cur_raw += 1
        else:
            merged.append(
                MergedRange(
                    offset=cur_start,
                    nbytes=cur_end - cur_start,
                    needed_nbytes=cur_needed,
                    raw_ranges=cur_raw,
                )
            )
            cur_start = r.offset
            cur_end = r.end
            cur_needed = r.nbytes
            cur_raw = 1

    merged.append(
        MergedRange(
            offset=cur_start,
            nbytes=cur_end - cur_start,
            needed_nbytes=cur_needed,
            raw_ranges=cur_raw,
        )
    )

    return merged


def emit_rows_for_ranges(
    *,
    writer: csv.DictWriter,
    run_name: str,
    prompt_name: str,
    seed: str,
    trace_token: int,
    decode_pos: int,
    kind: str,
    ranges: list[Range],
    coalesce_gap_bytes: int,
) -> None:
    merged = coalesce_ranges(ranges, coalesce_gap_bytes)

    for i, r in enumerate(merged):
        writer.writerow(
            {
                "run_name": run_name,
                "prompt_name": prompt_name,
                "seed": seed,
                "trace_token": trace_token,
                "decode_pos": decode_pos,
                "kind": kind,
                "range_idx": i,
                "offset": r.offset,
                "nbytes": r.nbytes,
                "needed_nbytes": r.needed_nbytes,
                "overread_nbytes": r.overread_nbytes,
                "raw_ranges": r.raw_ranges,
            }
        )


def emit_plan_for_trace(
    *,
    trace_path: str,
    layout: dict[tuple[int, int], list[Range]],
    writer: csv.DictWriter,
    policy: str,
    warmup: int,
    budget: int,
    n_layer: int,
    window_tokens: int,
    refresh_every: int,
    coalesce_gap_bytes: int,
    include_reload: bool,
) -> None:
    grouped = group_decode_by_token(read_trace(trace_path))

    if len(grouped) <= warmup:
        raise ValueError(f"{trace_path}: decode token count <= warmup")

    run_name, prompt_name, seed = parse_run_name(trace_path)

    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    window: deque[list[Access]] = deque()

    for _trace_token, token_accesses in grouped[:warmup]:
        update_counts(counts, token_accesses)
        if policy == "sliding":
            window.append(token_accesses)

    if policy == "sliding":
        while len(window) > window_tokens:
            subtract_counts(counts, window.popleft())

    hotset = build_hotset_from_counts(counts, budget=budget, n_layer=n_layer)

    if include_reload:
        initial_pairs = hotset_pairs(hotset)
        emit_rows_for_ranges(
            writer=writer,
            run_name=run_name,
            prompt_name=prompt_name,
            seed=seed,
            trace_token=grouped[warmup][0],
            decode_pos=warmup,
            kind="initial_reload",
            ranges=ranges_for_pairs(initial_pairs, layout),
            coalesce_gap_bytes=coalesce_gap_bytes,
        )

    for rel_i, (trace_token, token_accesses) in enumerate(grouped[warmup:], start=1):
        decode_pos = warmup + rel_i - 1
        pairs = token_pairs(token_accesses)

        missing = {
            (layer, expert)
            for layer, expert in pairs
            if expert not in hotset.get(layer, set())
        }

        emit_rows_for_ranges(
            writer=writer,
            run_name=run_name,
            prompt_name=prompt_name,
            seed=seed,
            trace_token=trace_token,
            decode_pos=decode_pos,
            kind="cold",
            ranges=ranges_for_pairs(missing, layout),
            coalesce_gap_bytes=coalesce_gap_bytes,
        )

        if policy in ("cumulative", "sliding"):
            update_counts(counts, token_accesses)

            if policy == "sliding":
                window.append(token_accesses)
                while len(window) > window_tokens:
                    subtract_counts(counts, window.popleft())

            if rel_i % refresh_every == 0:
                old_pairs = hotset_pairs(hotset)
                hotset = build_hotset_from_counts(
                    counts, budget=budget, n_layer=n_layer
                )
                new_pairs = hotset_pairs(hotset)

                added = new_pairs - old_pairs

                if include_reload:
                    emit_rows_for_ranges(
                        writer=writer,
                        run_name=run_name,
                        prompt_name=prompt_name,
                        seed=seed,
                        trace_token=trace_token,
                        decode_pos=decode_pos,
                        kind="refresh_reload",
                        ranges=ranges_for_pairs(added, layout),
                        coalesce_gap_bytes=coalesce_gap_bytes,
                    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True)
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--out", required=True)

    p.add_argument(
        "--policy", choices=["frozen", "cumulative", "sliding"], required=True
    )
    p.add_argument("--warmup", type=int, default=64)
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=48)

    p.add_argument("--window-tokens", type=int, default=128)
    p.add_argument("--refresh-every", type=int, default=8)
    p.add_argument("--coalesce-gap-kib", type=int, default=0)

    p.add_argument(
        "--no-reload",
        action="store_true",
        help="Emit only cold miss reads, not initial/refresh hotset loads.",
    )

    args = p.parse_args()

    layout = load_layout(args.layout)
    coalesce_gap_bytes = args.coalesce_gap_kib * 1024

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fieldnames = [
        "run_name",
        "prompt_name",
        "seed",
        "trace_token",
        "decode_pos",
        "kind",
        "range_idx",
        "offset",
        "nbytes",
        "needed_nbytes",
        "overread_nbytes",
        "raw_ranges",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trace_path in args.traces:
            emit_plan_for_trace(
                trace_path=trace_path,
                layout=layout,
                writer=writer,
                policy=args.policy,
                warmup=args.warmup,
                budget=args.budget,
                n_layer=args.n_layer,
                window_tokens=args.window_tokens,
                refresh_every=args.refresh_every,
                coalesce_gap_bytes=coalesce_gap_bytes,
                include_reload=not args.no_reload,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
