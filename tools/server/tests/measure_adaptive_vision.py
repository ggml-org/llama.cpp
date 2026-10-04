#!/usr/bin/env python3
"""Measure serial vision phase grants, wall latency and sampled NVIDIA device memory."""

import argparse
import contextlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from test_adaptive_vision import completion, png, request, server


def parse_phases(text):
    required = set("parent weights compute host_compute grants before_kv resumed_kv borrowed_after suspended_after diagnostics measure_us suspend_us loan_us projector_reload_us encode_us release_us text_resume_us begin_us end_us".split())
    records = []
    for line in text.splitlines():
        if "vision_phase: " not in line:
            continue
        record = {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", line.split("vision_phase: ", 1)[1])}
        if not required <= record.keys() or any(record[key] < 0 for key in required):
            raise ValueError("incomplete vision phase record")
        if (record["diagnostics"] != 1 or record["borrowed_after"] or record["suspended_after"] or
                not record["weights"] + record["compute"] <= record["grants"] <= record["parent"] or
                record["end_us"] < record["begin_us"]):
            raise ValueError("vision phase did not return its bounded grants")
        records.append(record)
    return records


def sample_window(samples, begin_us, end_us):
    values = [mib for stamp, mib in samples if begin_us <= stamp <= end_us]
    return {"samples": len(values), "min_mib": min(values) if values else None,
            "peak_mib": max(values) if values else None}


def parse_reloads(text):
    records = []
    for line in text.splitlines():
        if "KV_reload: " not in line:
            continue
        record = {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", line.split("KV_reload: ", 1)[1])}
        required = set("bytes calls padded_rows drain_us upload_us begin_us end_us".split())
        if not required <= record.keys() or any(record[key] < 0 for key in required) or record["end_us"] < record["begin_us"]:
            raise ValueError("invalid KV mirror reload record")
        records.append(record)
    return records


def group_phases(phases, rows):
    groups = [[] for row in rows]
    for phase in phases:
        owners = [i for i, row in enumerate(rows) if row["begin_us"] <= phase["begin_us"] <= phase["end_us"] <= row["end_us"]]
        if len(owners) != 1:
            raise ValueError("phase record does not belong to exactly one completed request")
        groups[owners[0]].append(phase)
    if any(not group for group in groups):
        raise ValueError("completed request has no phase records")
    return groups


@contextlib.contextmanager
def gpu_samples(gpu, interval_ms):
    command = ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits",
               "--loop-ms", str(interval_ms)]
    samples = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        def read():
            for line in process.stdout:
                try:
                    samples.append((time.monotonic_ns() // 1000, int(line.strip())))
                except ValueError:
                    pass
        worker = threading.Thread(target=read, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not samples and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.05)
            if not samples:
                process.terminate()
                raise RuntimeError("nvidia-smi supplied no memory samples: " + process.stderr.read())
            yield samples
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            worker.join(timeout=5)


def run(args, batch):
    args.batch_size = args.ubatch_size = batch
    args.output = args.root_output / f"batch-{batch}"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    images = [png(args.image_size, args.image_size, i) for i in range(args.images)]
    prompt = " green" * args.background_tokens + " Describe these patterns: " + "<__media__> " * args.images + "One sentence: "
    with gpu_samples(args.gpu, args.sample_ms) as samples, server(args, "arena", env_extra={"CUDA_VISIBLE_DEVICES": str(args.gpu)}) as port:
        ready_us = time.monotonic_ns() // 1000
        time.sleep(0.3)
        ready = sample_window(samples, ready_us, time.monotonic_ns() // 1000)
        if not ready["samples"]:
            raise RuntimeError("no ready-state device memory samples")
        status, text_control = request(port, "/completion", completion("The capital of France is", decode=args.decode))
        if status != 200:
            raise RuntimeError(text_control)
        for iteration in range(args.repeats):
            begin = time.monotonic_ns() // 1000
            status, result = request(port, "/completion", completion(prompt, images, decode=args.decode))
            end = time.monotonic_ns() // 1000
            if status != 200 or len(result.get("tokens", [])) != args.decode:
                raise RuntimeError(result)
            if rows and rows[0]["tokens"] != result["tokens"]:
                raise AssertionError("repeated uncached image output changed")
            idle_begin = time.monotonic_ns() // 1000
            time.sleep(0.3)
            rows.append({"iteration": iteration, "begin_us": begin, "end_us": end,
                         "wall_ms": (end-begin)/1000, "tokens": result["tokens"], "timings": result.get("timings"),
                         "request_memory": sample_window(samples, begin, end),
                         "idle_memory": sample_window(samples, idle_begin, time.monotonic_ns() // 1000)})
        status, final = request(port, "/completion", completion("The capital of France is", decode=args.decode))
        if status != 200 or final["tokens"] != text_control["tokens"]:
            raise AssertionError("post-image text continuation changed")
        log = (args.output / "arena.log").read_text()
        phases = parse_phases(log)
        reloads = parse_reloads(log)
        for row, group in zip(rows, group_phases(phases, rows)):
            row["phases"] = group
            row["vision_memory"] = [sample_window(samples, phase["begin_us"], phase["end_us"]) for phase in group]
            row["kv_reloads"] = [entry for entry in reloads if row["begin_us"] <= entry["begin_us"] <= entry["end_us"] <= row["end_us"]]
            row["post_vision_reloads"] = []
            for i, phase in enumerate(group):
                limit = group[i+1]["begin_us"] if i+1 < len(group) else row["end_us"]
                candidates = [entry for entry in row["kv_reloads"] if phase["end_us"] <= entry["begin_us"] <= entry["end_us"] <= limit]
                if not candidates:
                    raise AssertionError("no first-use KV refill after returning vision grants")
                row["post_vision_reloads"].append(candidates[0])
            # Counterfactual ordinary buffers, not an observed second server or a driver-allocation bound.
            row["simultaneous_buffer_estimate_mib"] = ready["min_mib"] + max(phase["weights"]+phase["compute"] for phase in group)/1048576
        warm_idle = [row["idle_memory"]["peak_mib"] for row in rows[1:]]
        if None in warm_idle or max(warm_idle)-min(warm_idle) > args.idle_tolerance_mib:
            raise AssertionError(f"post-warmup idle memory did not stabilize: {warm_idle}")
        payload = {"batch_size": batch, "context": args.context, "arena_mib": args.arena_mib,
                   "mtp_length": getattr(args, "mtp_length", 0),
                   "image_size": args.image_size, "images": args.images, "background_tokens": args.background_tokens,
                   "sample_ms": args.sample_ms, "ready_memory": ready, "requests": rows,
                   "memory_samples": samples,
                   "limits": "Sampled device-wide memory can miss transient peaks. text_resume_us is grant restoration. KV_reload measures first-use resident mirror flush wall time (host planning and synchronous copies), not pure DMA or later ring streaming. Request latency includes embedding prefill, lazy KV refill and decode."}
        (args.output / "measurements.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"batch": batch, "ready_mib": ready["min_mib"],
                      "request_peaks_mib": [row["request_memory"]["peak_mib"] for row in rows],
                      "warm_idle_mib": warm_idle}), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("server", "model", "mmproj", "output"):
        parser.add_argument("--"+option, type=Path, required=True)
    parser.add_argument("--arena-mib", type=int, default=1024)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batches", default="64,256")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--background-tokens", type=int, default=6000)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--mtp-length", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-ms", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--idle-tolerance-mib", type=int, default=32)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if sys.platform != "linux":
        parser.error("phase timestamps and device sampling are currently qualified on Linux only")
    if min(batches+[args.arena_mib, args.context, args.image_size, args.images, args.decode, args.sample_ms]) <= 0 or args.repeats < 3 or args.background_tokens < 0 or args.gpu < 0 or args.idle_tolerance_mib < 0:
        parser.error("positive sizes and at least three repeats are required")
    args.cache_ram_mib = 2048
    args.log_verbosity = 3
    args.root_output = args.output
    results = [run(args, batch) for batch in batches]
    (args.root_output / "measurements.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
