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


def load_plan(
    path: str, *, kinds: set[str], max_tokens: int | None
) -> list[tuple[str, int, str, list[ReadRange]]]:
    by_token: dict[tuple[str, int, str], list[ReadRange]] = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = row["kind"]
            if kind not in kinds:
                continue

            key = (row["run_name"], int(row["decode_pos"]), kind)
            by_token[key].append(
                ReadRange(
                    offset=int(row["offset"]),
                    nbytes=int(row["nbytes"]),
                )
            )

    tokens = [(run, pos, kind, ranges) for (run, pos, kind), ranges in by_token.items()]
    tokens.sort(key=lambda x: (x[0], x[1], x[2]))

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    return tokens


def touch_buffer(buf: bytes) -> int:
    # Force Python to actually observe the returned bytes.
    # One byte per 4 KiB page is enough for a cheap checksum.
    s = 0
    step = 4096
    for i in range(0, len(buf), step):
        s ^= buf[i]
    return s


def bench(
    *,
    gguf_path: str,
    tokens: list[tuple[str, int, list[ReadRange]]],
    repeats: int,
    shuffle_tokens: bool,
    touch: bool,
) -> dict[str, float]:
    fd = os.open(gguf_path, os.O_RDONLY)

    try:
        lat_ms: list[float] = []
        bytes_read = 0
        range_count = 0
        checksum = 0

        order = list(range(len(tokens)))

        for _rep in range(repeats):
            if shuffle_tokens:
                random.shuffle(order)

            for idx in order:
                _run, _pos, _kind, ranges = tokens[idx]

                t0 = time.perf_counter_ns()

                for r in ranges:
                    data = os.pread(fd, r.nbytes, r.offset)
                    if len(data) != r.nbytes:
                        raise RuntimeError(
                            f"short read: offset={r.offset} want={r.nbytes} got={len(data)}"
                        )

                    if touch:
                        checksum ^= touch_buffer(data)

                    bytes_read += r.nbytes
                    range_count += 1

                t1 = time.perf_counter_ns()
                lat_ms.append((t1 - t0) / 1_000_000.0)

        total_s = sum(lat_ms) / 1000.0
        mib_read = mib(bytes_read)

        return {
            "tokens": float(len(lat_ms)),
            "ranges": float(range_count),
            "mib_read": mib_read,
            "mib_per_token": mib_read / len(lat_ms) if lat_ms else 0.0,
            "ranges_per_token": range_count / len(lat_ms) if lat_ms else 0.0,
            "throughput_mib_s": mib_read / total_s if total_s > 0 else 0.0,
            "lat_avg_ms": statistics.mean(lat_ms) if lat_ms else 0.0,
            "lat_p50_ms": percentile(lat_ms, 0.50),
            "lat_p90_ms": percentile(lat_ms, 0.90),
            "lat_p99_ms": percentile(lat_ms, 0.99),
            "lat_max_ms": max(lat_ms) if lat_ms else 0.0,
            "checksum": float(checksum),
        }
    finally:
        os.close(fd)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--plan", required=True)

    p.add_argument(
        "--kinds",
        default="cold,refresh_reload,initial_reload",
        help="Comma-separated kinds from read plan",
    )
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--shuffle-tokens", action="store_true")
    p.add_argument("--no-touch", action="store_true")

    args = p.parse_args()

    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    tokens = load_plan(args.plan, kinds=kinds, max_tokens=args.max_tokens)

    print(f"plan: {args.plan}")
    print(f"gguf: {args.gguf}")
    print(f"kinds: {sorted(kinds)}")
    print(f"token groups: {len(tokens)}")

    result = bench(
        gguf_path=args.gguf,
        tokens=tokens,
        repeats=args.repeats,
        shuffle_tokens=args.shuffle_tokens,
        touch=not args.no_touch,
    )

    for k, v in result.items():
        if k == "checksum":
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
