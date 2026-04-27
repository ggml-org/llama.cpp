#!/usr/bin/env python3

import argparse
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


MOE_COPY_RE = re.compile(r"\bmoe_copy\b(?P<fields>.*)\sids=\[(?P<expert_ids>[^\]]*)\]")
FIELD_RE = re.compile(r"(\w+)=([^\s]+)")


@dataclass(frozen=True)
class MoeCopyEvent:
    key: str
    tensor: str
    dst_backend: str
    expert_size: int
    used_bytes: int
    copy_bytes: int
    expert_ids: Tuple[int, ...]


@dataclass
class SimStats:
    events: int = 0
    bypasses: int = 0
    cache_bytes: int = 0
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    baseline_bytes: int = 0
    cache_copy_bytes: int = 0


def parse_slots(value: str) -> List[int]:
    slots = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        slot_count = int(item)
        if slot_count < 0:
            raise argparse.ArgumentTypeError("slot counts must be non-negative")
        slots.append(slot_count)
    if not slots:
        raise argparse.ArgumentTypeError("at least one slot count is required")
    return slots


def parse_moe_copy_line(line: str) -> Optional[MoeCopyEvent]:
    match = MOE_COPY_RE.search(line)
    if match is None:
        return None

    fields = dict(FIELD_RE.findall(match.group("fields")))
    try:
        tensor = fields["tensor"]
        dst_backend = fields["dst_backend"]
        expert_size = int(fields["expert_size"])
        used_bytes = int(fields["used_bytes"])
        copy_bytes = int(fields["copy_bytes"])
    except KeyError as exc:
        raise ValueError(f"missing moe_copy field: {exc.args[0]}") from exc

    expert_ids_raw = match.group("expert_ids").strip()
    expert_ids = tuple(int(item) for item in expert_ids_raw.split(",") if item.strip())
    if len(expert_ids) != len(set(expert_ids)):
        raise ValueError(f"moe_copy line has duplicate expert ids: {expert_ids_raw}")
    expected_used_bytes = len(expert_ids) * expert_size
    if used_bytes != expected_used_bytes:
        raise ValueError(
            f"used_bytes={used_bytes} does not match "
            f"{len(expert_ids)} expert ids * expert_size={expert_size}"
        )
    if copy_bytes < used_bytes:
        raise ValueError(f"copy_bytes={copy_bytes} is smaller than used_bytes={used_bytes}")

    return MoeCopyEvent(
        key=f"{dst_backend}:{tensor}",
        tensor=tensor,
        dst_backend=dst_backend,
        expert_size=expert_size,
        used_bytes=used_bytes,
        copy_bytes=copy_bytes,
        expert_ids=expert_ids,
    )


def read_events(paths: Sequence[str]) -> Iterator[MoeCopyEvent]:
    if not paths:
        yield from read_events_from_lines(sys.stdin)
        return

    for path_str in paths:
        if path_str == "-":
            yield from read_events_from_lines(sys.stdin)
        else:
            with Path(path_str).open("r", encoding="utf-8") as f:
                yield from read_events_from_lines(f)


def read_events_from_lines(lines: Iterable[str]) -> Iterator[MoeCopyEvent]:
    for line_no, line in enumerate(lines, 1):
        try:
            event = parse_moe_copy_line(line)
        except ValueError as exc:
            raise ValueError(f"line {line_no}: {exc}") from exc
        if event is not None:
            yield event


def simulate_lru(events: Sequence[MoeCopyEvent], slots: Sequence[int]) -> Dict[Tuple[int, str], SimStats]:
    stats: Dict[Tuple[int, str], SimStats] = {}
    caches: Dict[Tuple[int, str], OrderedDict[int, None]] = {}
    expert_sizes: Dict[str, int] = {}

    for slot_count in slots:
        for event in events:
            previous_expert_size = expert_sizes.setdefault(event.key, event.expert_size)
            if previous_expert_size != event.expert_size:
                raise ValueError(
                    f"inconsistent expert_size for {event.key}: "
                    f"saw {event.expert_size}, expected {previous_expert_size}"
                )

            stat_key = (slot_count, event.key)
            stat = stats.setdefault(stat_key, SimStats())
            cache = caches.setdefault(stat_key, OrderedDict())

            needed = event.expert_ids
            needed_set = set(needed)

            stat.events += 1
            stat.cache_bytes = max(stat.cache_bytes, slot_count * event.expert_size)
            stat.accesses += len(needed)
            stat.baseline_bytes += event.copy_bytes

            if slot_count == 0 or len(needed) > slot_count:
                stat.bypasses += 1
                stat.misses += len(needed)
                stat.cache_copy_bytes += event.copy_bytes
                continue

            hits = [expert_id for expert_id in needed if expert_id in cache]
            misses = [expert_id for expert_id in needed if expert_id not in cache]

            stat.hits += len(hits)
            stat.misses += len(misses)
            stat.cache_copy_bytes += len(misses) * event.expert_size

            while len(cache) + len(misses) > slot_count:
                victim = next((expert_id for expert_id in cache if expert_id not in needed_set), None)
                if victim is None:
                    victim = next(iter(cache))
                del cache[victim]

            for expert_id in misses:
                cache[expert_id] = None

            for expert_id in needed:
                if expert_id in cache:
                    cache.move_to_end(expert_id)

    return stats


def aggregate_stats(stats: Dict[Tuple[int, str], SimStats]) -> Dict[int, SimStats]:
    aggregate: Dict[int, SimStats] = {}
    for (slot_count, _), stat in stats.items():
        dst = aggregate.setdefault(slot_count, SimStats())
        dst.events += stat.events
        dst.bypasses += stat.bypasses
        dst.cache_bytes += stat.cache_bytes
        dst.accesses += stat.accesses
        dst.hits += stat.hits
        dst.misses += stat.misses
        dst.baseline_bytes += stat.baseline_bytes
        dst.cache_copy_bytes += stat.cache_copy_bytes
    return aggregate


def stats_row(slot_count: int, key: str, stat: SimStats) -> str:
    hit_rate = stat.hits / stat.accesses if stat.accesses else 0.0
    saved_bytes = stat.baseline_bytes - stat.cache_copy_bytes
    saved_pct = saved_bytes / stat.baseline_bytes if stat.baseline_bytes else 0.0
    return "\t".join((
        str(slot_count),
        key,
        str(stat.cache_bytes),
        str(stat.events),
        str(stat.bypasses),
        str(stat.accesses),
        str(stat.hits),
        str(stat.misses),
        f"{hit_rate:.6f}",
        str(stat.baseline_bytes),
        str(stat.cache_copy_bytes),
        str(saved_bytes),
        f"{saved_pct:.6f}",
    ))


def print_report(stats: Dict[Tuple[int, str], SimStats], show_details: bool) -> None:
    print("slots\tkey\tcache_bytes\tevents\tbypasses\taccesses\thits\tmisses\thit_rate\tbaseline_bytes\tcache_copy_bytes\tsaved_bytes\tsaved_pct")

    for slot_count, stat in sorted(aggregate_stats(stats).items()):
        print(stats_row(slot_count, "ALL", stat))

    if not show_details:
        return

    for (slot_count, key), stat in sorted(stats.items()):
        print(stats_row(slot_count, key, stat))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate byte-weighted LRU MoE expert caches from GGML_SCHED_MOE_LOG output.",
    )
    parser.add_argument("logs", nargs="*", help="log files to parse; omit or use '-' for stdin")
    parser.add_argument("--slots", type=parse_slots, default=parse_slots("32,64,96,128"), help="comma-separated slot counts")
    parser.add_argument("--details", action="store_true", help="also print per backend/tensor stats")
    args = parser.parse_args(argv)

    events = list(read_events(args.logs))
    if not events:
        print("no moe_copy events found", file=sys.stderr)
        return 1

    print_report(simulate_lru(events, args.slots), args.details)
    return 0


if __name__ == "__main__":
    sys.exit(main())
