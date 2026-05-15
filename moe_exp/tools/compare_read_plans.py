#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadRange:
    offset: int
    nbytes: int


def mib(x: float) -> float:
    return x / 1024.0 / 1024.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * p))
    return ys[idx]


def plan_path(template: str, budget: int) -> str:
    return template.format(budget=budget)


def load_plan_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_stats(rows: list[dict[str, str]], kind: str) -> dict[str, float]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        if row["kind"] != kind:
            continue
        key = (row["run_name"], int(row["decode_pos"]))
        groups[key].append(row)

    if not groups:
        return {
            "groups": 0.0,
            "mib_group": 0.0,
            "ranges_group": 0.0,
        }

    mib_per_group = []
    ranges_per_group = []

    for rs in groups.values():
        mib_per_group.append(sum(int(r["nbytes"]) for r in rs) / 1024**2)
        ranges_per_group.append(len(rs))

    return {
        "groups": float(len(groups)),
        "mib_group": statistics.mean(mib_per_group),
        "ranges_group": statistics.mean(ranges_per_group),
    }


def load_bench_groups(
    rows: list[dict[str, str]],
    *,
    kind: str,
    max_groups: int | None,
) -> list[tuple[str, int, list[ReadRange]]]:
    groups: dict[tuple[str, int], list[ReadRange]] = defaultdict(list)

    for row in rows:
        if row["kind"] != kind:
            continue

        key = (row["run_name"], int(row["decode_pos"]))
        groups[key].append(
            ReadRange(
                offset=int(row["offset"]),
                nbytes=int(row["nbytes"]),
            )
        )

    out = [(run, pos, ranges) for (run, pos), ranges in groups.items()]
    out.sort(key=lambda x: (x[0], x[1]))

    if max_groups is not None:
        out = out[:max_groups]

    return out


def touch_buffer(buf: bytes) -> int:
    checksum = 0
    for i in range(0, len(buf), 4096):
        checksum ^= buf[i]
    return checksum


def bench_groups(
    *,
    gguf: str,
    groups: list[tuple[str, int, list[ReadRange]]],
    repeats: int,
    shuffle: bool,
    touch: bool,
) -> dict[str, float]:
    fd = os.open(gguf, os.O_RDONLY)

    try:
        lat_ms: list[float] = []
        total_bytes = 0
        total_ranges = 0
        checksum = 0

        order = list(range(len(groups)))

        for _ in range(repeats):
            if shuffle:
                random.shuffle(order)

            for idx in order:
                _run, _pos, ranges = groups[idx]

                t0 = time.perf_counter_ns()

                for r in ranges:
                    data = os.pread(fd, r.nbytes, r.offset)
                    if len(data) != r.nbytes:
                        raise RuntimeError(
                            f"short read: offset={r.offset}, want={r.nbytes}, got={len(data)}"
                        )

                    if touch:
                        checksum ^= touch_buffer(data)

                    total_bytes += r.nbytes
                    total_ranges += 1

                t1 = time.perf_counter_ns()
                lat_ms.append((t1 - t0) / 1_000_000.0)

        total_s = sum(lat_ms) / 1000.0
        total_mib = mib(total_bytes)

        return {
            "bench_groups": float(len(lat_ms)),
            "bench_mib_group": total_mib / len(lat_ms) if lat_ms else 0.0,
            "bench_ranges_group": total_ranges / len(lat_ms) if lat_ms else 0.0,
            "bench_throughput_mib_s": total_mib / total_s if total_s > 0 else 0.0,
            "bench_avg_ms": statistics.mean(lat_ms) if lat_ms else 0.0,
            "bench_p50_ms": percentile(lat_ms, 0.50),
            "bench_p90_ms": percentile(lat_ms, 0.90),
            "bench_p99_ms": percentile(lat_ms, 0.99),
            "bench_max_ms": max(lat_ms) if lat_ms else 0.0,
            "checksum": float(checksum),
        }
    finally:
        os.close(fd)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--template",
        required=True,
        help="Example: moe_exp/read_plan_cumulative_r16_b{budget}_gap0.csv",
    )
    p.add_argument("--budgets", type=int, nargs="+", required=True)
    p.add_argument("--refresh-every", type=int, default=16)
    p.add_argument("--expert-total-gib", type=float, default=16.348)

    p.add_argument("--gguf", default="")
    p.add_argument(
        "--bench-kind",
        default="",
        choices=["", "cold", "refresh_reload", "initial_reload"],
    )
    p.add_argument("--max-groups", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--no-touch", action="store_true")

    args = p.parse_args()

    rows_out: list[dict[str, float | int | str]] = []

    for budget in args.budgets:
        path = plan_path(args.template, budget)
        rows = load_plan_rows(path)

        initial = group_stats(rows, "initial_reload")
        refresh = group_stats(rows, "refresh_reload")
        cold = group_stats(rows, "cold")

        out: dict[str, float | int | str] = {
            "budget": budget,
            "plan": Path(path).name,
            "max_resident_gib": args.expert_total_gib * budget / 128.0,
            "initial_mib": initial["mib_group"],
            "initial_ranges": initial["ranges_group"],
            "cold_groups": cold["groups"],
            "cold_mib": cold["mib_group"],
            "cold_ranges": cold["ranges_group"],
            "refresh_events": refresh["groups"],
            "refresh_mib_event": refresh["mib_group"],
            "refresh_ranges_event": refresh["ranges_group"],
            "refresh_mib_token": refresh["mib_group"] / args.refresh_every,
            "refresh_ranges_token": refresh["ranges_group"] / args.refresh_every,
        }

        if args.bench_kind:
            if not args.gguf:
                raise RuntimeError("--gguf is required when --bench-kind is set")

            groups = load_bench_groups(
                rows,
                kind=args.bench_kind,
                max_groups=args.max_groups,
            )

            bench = bench_groups(
                gguf=args.gguf,
                groups=groups,
                repeats=args.repeats,
                shuffle=args.shuffle,
                touch=not args.no_touch,
            )
            out.update(bench)

        rows_out.append(out)

    headers = list(rows_out[0].keys())

    print(",".join(headers))
    for row in rows_out:
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print(",".join(vals))

    print("\nMarginal:")
    prev = None
    for row in rows_out:
        if prev is not None:
            print(
                f"{prev['budget']}->{row['budget']}: "
                f"resident +{float(row['max_resident_gib']) - float(prev['max_resident_gib']):.2f} GiB, "
                f"cold {float(prev['cold_mib']):.2f}->{float(row['cold_mib']):.2f} MiB "
                f"({float(row['cold_mib']) - float(prev['cold_mib']):+.2f}), "
                f"ranges {float(prev['cold_ranges']):.2f}->{float(row['cold_ranges']):.2f} "
                f"({float(row['cold_ranges']) - float(prev['cold_ranges']):+.2f})"
            )

            if args.bench_kind:
                print(
                    f"  bench p90 {float(prev['bench_p90_ms']):.2f}->{float(row['bench_p90_ms']):.2f} ms "
                    f"({float(row['bench_p90_ms']) - float(prev['bench_p90_ms']):+.2f})"
                )

        prev = row

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
