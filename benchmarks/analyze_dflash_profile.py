#!/usr/bin/env python3
"""Attribute a DFlash control's Nsight SQLite export by correlated CPU submission."""

import argparse
import bisect
import collections
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


def summarize_cpu_trace(rows):
    groups = collections.defaultdict(list)
    outcomes = collections.Counter()
    map_peak = history_peak = 0
    active = not any(row.get("event") == "generation_begin" for row in rows)
    phase = "unscoped"
    for row in rows:
        event = row.get("event")
        if event == "generation_begin":
            active, phase = True, "decode"
            continue
        if event == "draft_driver_begin":
            phase = "decode"
            continue
        if event == "process" and (not active or row.get("warmup")):
            continue
        if event not in ("ngram_begin", "ngram_draft", "draft_driver", "process"):
            continue
        duration = row.get("host_wall_ns")
        if type(duration) is not int or duration < 0:
            raise ValueError("invalid whole-call host duration")
        groups[event].append(duration)
        if event == "ngram_draft":
            for field in ("map_reserved_bytes", "history_capacity_bytes"):
                if type(row.get(field)) is not int or row[field] < 0:
                    raise ValueError("invalid lookup host-memory extent")
            if type(row.get("first_use")) is not bool or type(row.get("first_statistics")) is not bool:
                raise ValueError("missing first-use lookup classification")
            outcome = row.get("outcome")
            if outcome not in ("supplied", "short_history", "no_match", "minimum_hits", "ambiguous", "acceptance_suppressed", "empty"):
                raise ValueError("unknown lookup outcome")
            outcomes[outcome] += 1
            if row["first_use"]: groups["first_use"].append(duration)
            if row["first_statistics"]: groups["first_statistics"].append(duration)
            if not row["first_use"] and not row["first_statistics"]: groups["steady"].append(duration)
            map_peak = max(map_peak, row["map_reserved_bytes"])
            history_peak = max(history_peak, row["history_capacity_bytes"])
        if event == "process" and not row.get("warmup"):
            role, kind, count = row.get("role"), row.get("kind"), row.get("rows")
            if role not in ("target", "draft") or kind not in ("tokens", "features") or type(count) is not int or count < 1:
                raise ValueError("invalid process role/geometry")
            groups[f"{phase}/{role}/{kind}/{count}"].append(duration)
    def summary(values):
        ordered = sorted(values)
        return {"count":len(ordered), "total_host_us":sum(ordered)/1000,
            **{f"p{percent}_us":ordered[math.ceil(len(ordered)*percent/100)-1]/1000 if ordered else None for percent in (50,95,99)}}
    return {"lookup":{"all":summary(groups["ngram_draft"]), "first_use":summary(groups["first_use"]),
            "first_statistics":summary(groups["first_statistics"]), "steady":summary(groups["steady"]), "outcomes":dict(outcomes)},
        "begin":summary(groups["ngram_begin"]), "draft_driver":summary(groups["draft_driver"]),
        "process_calls":{name:summary(values) for name,values in groups.items() if "/" in name},
        "map_reserved_bytes_peak":map_peak, "history_capacity_bytes_peak":history_peak,
        "notes":["Whole-call host wall times, not one hash access, kernel time or uninstrumented throughput.",
            "First-use includes initial hash indexing; first-statistics can happen at a later matching key.",
            "Lookup, driver and process intervals can nest and must not be added as independent costs.",
            "Driver time includes nested diagnostic recording; process time is host submission/wait time, not GPU kernel duration.",
            "The server emits generation_begin after prefill. With that marker, process summaries cover decode only; ordinary request timings measure prefill.",
            "Nearest-rank percentiles; sample counts are explicit and small tail estimates are not precise.",
            "Vector reserved bytes exclude allocator overhead and all other server/host KV memory. History capacity is separate."]}


def interval_ns(intervals):
    total = 0
    end = None
    for first, last in sorted(intervals):
        if last < first:
            raise ValueError("reversed trace interval")
        total += max(0, last-max(first, end if end is not None else first))
        end = max(last, end if end is not None else last)
    return total


def phase_name(text):
    parts = text.split("/")
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    if parts[:2] == ["target", "tokens"]:
        return "target_prefill" if int(parts[2]) > 8 else "target_verification"
    if parts[:2] == ["draft", "features"]:
        return "draft_injection"
    if parts[:2] == ["draft", "tokens"]:
        return "draft_prediction"
    return None


def kernel_category(name):
    name = name.lower()
    attention = "flash_attn" in name or "fattn" in name
    if attention and ("fixup" in name or "fattn_combine" in name):
        return "attention_finalization"
    if "convert_rows_kernel" in name:
        return "conversion"
    if attention:
        return "attention_including_checkpoint"
    if "convert" in name or "to_fp16" in name or "cast_" in name:
        return "conversion"
    if "set_rows" in name:
        return "kv_writer"
    if "gated_delta_net" in name or "ssm" in name or "conv1d" in name:
        return "recurrent"
    if "mul_mat" in name or "gemm" in name or "gemv" in name or "mmq" in name:
        return "weight_matmul"
    return "other"


def summarize_records(ranges, runtime, kernels, copies):
    by_thread = collections.defaultdict(list)
    for first, last, thread, text in ranges:
        phase = phase_name(text)
        if phase and last is not None:
            by_thread[thread].append((first, last, phase))
    for values in by_thread.values():
        values.sort()
    starts = {thread: [v[0] for v in values] for thread, values in by_thread.items()}
    groups: collections.defaultdict[str, dict[str, Any]] = collections.defaultdict(lambda: {
        "kernel_ns": 0, "kernel_calls": 0, "cpu_wait_ns": 0, "cpu_wait_calls": 0,
        "h2d_bytes": 0, "h2d_calls": 0, "d2h_bytes": 0, "d2h_calls": 0,
        "d2d_bytes": 0, "d2d_calls": 0, "kernel_categories": {},
        "kernels": [], "copies": [], "h2d": [], "cpu_ranges": [],
    })
    for values in by_thread.values():
        for first, last, phase in values:
            groups[phase]["cpu_ranges"].append((first, last))
    correlated = {}
    for first, last, thread, correlation, name in runtime:
        phase = "unattributed"
        values = by_thread.get(thread, [])
        index = bisect.bisect_right(starts.get(thread, []), first)-1
        if index >= 0 and first <= values[index][1]:
            phase = values[index][2]
        if correlation:
            previous = correlated.get(correlation)
            correlated[correlation] = phase if previous is None or previous == phase else "unattributed"
        if "Synchronize" in name:
            groups[phase]["cpu_wait_ns"] += last-first
            groups[phase]["cpu_wait_calls"] += 1
    for first, last, correlation, name in kernels:
        group = groups[correlated.get(correlation, "unattributed")]
        group["kernel_ns"] += last-first
        group["kernel_calls"] += 1
        group["kernels"].append((first, last))
        category = group["kernel_categories"].setdefault(kernel_category(name), {"ns": 0, "calls": 0})
        category["ns"] += last-first
        category["calls"] += 1
    for first, last, correlation, size, kind in copies:
        group = groups[correlated.get(correlation, "unattributed")]
        group["copies"].append((first, last))
        direction = {1: "h2d", 2: "d2h", 8: "d2d"}.get(kind)
        if direction:
            group[direction+"_bytes"] += size
            group[direction+"_calls"] += 1
        if direction == "h2d":
            group["h2d"].append((first, last))
    for group in groups.values():
        group["kernel_active_ns"] = interval_ns(group["kernels"])
        group["device_busy_ns"] = interval_ns(group["kernels"]+group["copies"])
        group["h2d_active_ns"] = interval_ns(group["h2d"])
        group["cpu_range_ns"] = interval_ns(group["cpu_ranges"])
        group["h2d_active_gbps"] = group["h2d_bytes"]/group["h2d_active_ns"] if group["h2d_active_ns"] else None
        for key in ("kernels", "copies", "h2d", "cpu_ranges"):
            del group[key]
    return dict(groups)


def summarize_profile(path):
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        ranges = [(a,b,t,text or strings.get(text_id, "")) for a,b,t,text,text_id in connection.execute(
            "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS")]
        runtime = [(a,b,t,c,strings.get(n, "")) for a,b,t,c,n in connection.execute(
            "SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")]
        kernels = [(a,b,c,strings.get(n, "")) for a,b,c,n in connection.execute(
            "SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")]
        copies = list(connection.execute("SELECT start,end,correlationId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
    return {"source": str(path.resolve()), "phases": summarize_records(ranges, runtime, kernels, copies),
        "notes": ["GPU work follows the correlated CPU submission, even when it completes after a range ends.",
                  "Missing or conflicting correlations remain unattributed; do not infer a phase from completion time.",
                  "Kernel sums are not wall time; active intervals merge overlap.",
                  "Checkpoint work inside resumed attention is included in its kernel, not separately timed.",
                  "Inline KV conversion is included in attention; the conversion category counts only separate kernels.",
                  "Target labels use the control's query-width buckets (>8 prefill, <=8 verification); short prompt tails can share the latter bucket.",
                  "Nsight-instrumented throughput is not an uninstrumented performance result."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, nargs="?")
    parser.add_argument("--cpu-trace", type=Path, help="whole-call host trace.jsonl from the test-side preload")
    args = parser.parse_args()
    if not args.sqlite and not args.cpu_trace:
        parser.error("provide a Nsight SQLite export or --cpu-trace")
    report = summarize_profile(args.sqlite) if args.sqlite else {}
    if args.cpu_trace:
        report["cpu_source"] = str(args.cpu_trace.resolve())
        report["cpu"] = summarize_cpu_trace([json.loads(line) for line in args.cpu_trace.read_text().splitlines()])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
