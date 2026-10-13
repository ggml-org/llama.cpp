#!/usr/bin/env python3
"""Attribute a DFlash control's Nsight SQLite export by correlated CPU submission."""

import argparse
import bisect
import collections
import json
import sqlite3
from pathlib import Path


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
    groups = collections.defaultdict(lambda: {
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
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_profile(args.sqlite), indent=2))


if __name__ == "__main__":
    main()
