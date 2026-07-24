#!/usr/bin/env python3
"""Paired Arc A770 SYCL graph and Level Zero submission benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request


GRAPH_PROFILE_RE = re.compile(
    r"GGML_SYCL_GRAPH_PROFILE: graph_calls=(?P<graph_calls>\d+) "
    r"direct_calls=(?P<direct_calls>\d+) nodes=(?P<nodes>\d+) "
    r"prepare_us=(?P<prepare_us>\d+) record_us=(?P<record_us>\d+) "
    r"finalize_calls=(?P<finalize_calls>\d+) finalize_us=(?P<finalize_us>\d+) "
    r"update_calls=(?P<update_calls>\d+) update_fallbacks=(?P<update_fallbacks>\d+) "
    r"update_us=(?P<update_us>\d+) submit_us=(?P<submit_us>\d+) "
    r"wait_us=(?P<wait_us>\d+) direct_enqueue_us=(?P<direct_enqueue_us>\d+)"
)
QUEUE_RELEASE_RE = re.compile(
    r"urQueueRelease\(compute\) NumTimesClosedFull (?P<full>\d+), "
    r"NumTimesClosedEarly (?P<early>\d+)"
)
FAULT_RE = re.compile(
    r"i915.*(hang|reset|fault|wedg)|xe.*(hang|reset|fault|wedg)|"
    r"GPU HANG|device lost|CAT_ERROR|GT reset",
    re.IGNORECASE,
)
CONTROLLED_ENV = (
    "GGML_SYCL_DISABLE_GRAPH",
    "GGML_SYCL_GRAPH_PROFILE",
    "UR_L0_BATCH_SIZE",
    "UR_L0_USE_IMMEDIATE_COMMANDLISTS",
    "SYCL_PI_LEVEL_ZERO_BATCH_SIZE",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
    "UR_LOG_LEVEL_ZERO",
)
ARMS = {
    "graph-off": {
        "GGML_SYCL_DISABLE_GRAPH": "1",
        "GGML_SYCL_GRAPH_PROFILE": "1",
    },
    "graph-on": {
        "GGML_SYCL_DISABLE_GRAPH": "0",
        "GGML_SYCL_GRAPH_PROFILE": "1",
    },
    "batch-16": {
        "GGML_SYCL_DISABLE_GRAPH": "1",
        "GGML_SYCL_GRAPH_PROFILE": "1",
        "UR_L0_USE_IMMEDIATE_COMMANDLISTS": "0",
        "UR_L0_BATCH_SIZE": "16",
    },
    "batch-64": {
        "GGML_SYCL_DISABLE_GRAPH": "1",
        "GGML_SYCL_GRAPH_PROFILE": "1",
        "UR_L0_USE_IMMEDIATE_COMMANDLISTS": "0",
        "UR_L0_BATCH_SIZE": "64",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--depths", default="0,4096,8192,16384")
    parser.add_argument("--parallel", default="1,2,4")
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--adapter-trace", action="store_true")
    parser.add_argument("--ctx-per-slot", type=int, default=32768)
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--health-timeout", type=float, default=30.0)
    parser.add_argument("--render-node", default="/dev/dri/renderD128")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item < 0 for item in values):
        raise ValueError(f"invalid non-negative integer list: {value!r}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(
    base_url: str, method: str, path: str, payload: dict[str, Any] | None, timeout: float
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected object response from {path}")
    return result


def wait_for_health(base_url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if request_json(base_url, "GET", "/health", None, 2.0).get("status") == "ok":
                return True
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    return False


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def effective_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for name in CONTROLLED_ENV:
        env.pop(name, None)
    env.update(extra)
    env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
    env["ZES_ENABLE_SYSMAN"] = "1"
    env["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
    return env


def arm_environment(arm: str, adapter_trace: bool) -> dict[str, str]:
    extra = dict(ARMS[arm])
    if adapter_trace and arm.startswith("batch-"):
        extra["UR_LOG_LEVEL_ZERO"] = "level:debug;flush:debug;output:stderr"
    return extra


def held_env(env: dict[str, str]) -> dict[str, str]:
    names = set(CONTROLLED_ENV) | {
        "ONEAPI_DEVICE_SELECTOR",
        "ZES_ENABLE_SYSMAN",
        "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS",
    }
    return {name: env[name] for name in sorted(names) if name in env}


def server_command(
    server_bin: Path,
    model: Path,
    state_dir: Path,
    port: int,
    parallel: int,
    ctx_per_slot: int,
) -> list[str]:
    return [
        str(server_bin),
        "--model", str(model),
        "--n-gpu-layers", "99",
        "--no-mmap",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--ctx-size", str(ctx_per_slot * parallel),
        "--parallel", str(parallel),
        "--ignore-eos",
        "--threads", "12",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--slot-save-path", str(state_dir),
    ]


def token_prefix(depth: int) -> list[int]:
    return [1000 + ((index * 7919) % 28000) for index in range(depth)]


def token_suffix(slot: int) -> list[int]:
    return [1200 + ((index * 3571 + slot * 101) % 27500) for index in range(512)]


def completion_payload(depth: int, slot: int, n_predict: int) -> dict[str, Any]:
    return {
        "prompt": token_prefix(depth) + token_suffix(slot),
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 123,
        "cache_prompt": True,
        "id_slot": slot,
    }


def check_sole_tenancy(render_node: str) -> None:
    result = subprocess.run(
        ["fuser", render_node], capture_output=True, text=True, check=False
    )
    holders = (result.stdout + result.stderr).strip()
    if result.returncode == 1 and not holders:
        return
    raise RuntimeError(f"sole tenancy violated for {render_node}: {holders or result.returncode}")


def dmesg_faults(since: str) -> list[str]:
    result = subprocess.run(
        ["sudo", "-n", "dmesg", "-T", "--since", since],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"DMESG_UNAVAILABLE: {result.stderr.strip()}"]
    return [line for line in result.stdout.splitlines() if FAULT_RE.search(line)]


def parse_server_log(text: str) -> dict[str, Any]:
    graph_records = [
        {name: int(value) for name, value in match.groupdict().items()}
        for match in GRAPH_PROFILE_RE.finditer(text)
    ]
    queue_records = [
        {name: int(value) for name, value in match.groupdict().items()}
        for match in QUEUE_RELEASE_RE.finditer(text)
    ]
    return {
        "graph_profile_records": graph_records,
        "queue_release_records": queue_records,
        "regular_command_list_records": text.count("type: regular"),
        "immediate_command_list_records": text.count("type: immediate"),
    }


def run_server(
    command: list[str], env: dict[str, str], log_path: Path, base_url: str, health_timeout: float
) -> tuple[subprocess.Popen[str], Any]:
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    if not wait_for_health(base_url, health_timeout):
        stop_server(process)
        log_file.close()
        raise RuntimeError(f"server health timeout; see {log_path}")
    return process, log_file


def prepare_state(
    server_bin: Path,
    model: Path,
    state_dir: Path,
    depth: int,
    parallel: int,
    port: int,
    ctx_per_slot: int,
    request_timeout: float,
    health_timeout: float,
    render_node: str,
) -> str | None:
    if depth == 0:
        return None
    filename = f"depth-{depth}-p{parallel}.bin"
    state_path = state_dir / filename
    if state_path.is_file() and state_path.stat().st_size > 0:
        return filename
    check_sole_tenancy(render_node)
    base_url = f"http://127.0.0.1:{port}"
    command = server_command(server_bin, model, state_dir, port, parallel, ctx_per_slot)
    log_path = state_dir / f"prepare-depth-{depth}-p{parallel}.log"
    process, log_file = run_server(
        command, effective_env({"GGML_SYCL_DISABLE_GRAPH": "1"}), log_path, base_url, health_timeout
    )
    try:
        warm_payload = completion_payload(depth, 0, 0)
        warm_payload["prompt"] = token_prefix(depth)
        response = request_json(base_url, "POST", "/completion", warm_payload, request_timeout)
        prompt_n = int((response.get("timings") or {}).get("prompt_n", -1))
        if prompt_n != depth:
            raise RuntimeError(f"state preparation expected prompt_n={depth}, got {prompt_n}")
        saved = request_json(
            base_url, "POST", "/slots/0?action=save", {"filename": filename}, request_timeout
        )
        if int(saved.get("n_saved", -1)) != depth:
            raise RuntimeError(f"state save expected n_saved={depth}, got {saved.get('n_saved')}")
    finally:
        stop_server(process)
        log_file.close()
    if not state_path.is_file() or state_path.stat().st_size <= 0:
        raise RuntimeError(f"state file was not created: {state_path}")
    return filename


def run_sample(
    *,
    server_bin: Path,
    model: Path,
    state_dir: Path,
    out_dir: Path,
    arm: str,
    depth: int,
    parallel: int,
    rep: int,
    port: int,
    ctx_per_slot: int,
    request_timeout: float,
    health_timeout: float,
    render_node: str,
    adapter_trace: bool,
) -> dict[str, Any]:
    check_sole_tenancy(render_node)
    env = effective_env(arm_environment(arm, adapter_trace))
    base_url = f"http://127.0.0.1:{port}"
    command = server_command(server_bin, model, state_dir, port, parallel, ctx_per_slot)
    log_path = out_dir / "logs" / f"p{parallel}-d{depth}-{arm}-rep{rep}.log"
    process, log_file = run_server(command, env, log_path, base_url, health_timeout)
    responses: list[dict[str, Any]] = []
    error: str | None = None
    wall_start = time.monotonic()
    try:
        if depth > 0:
            filename = f"depth-{depth}-p{parallel}.bin"
            for slot in range(parallel):
                restored = request_json(
                    base_url,
                    "POST",
                    f"/slots/{slot}?action=restore",
                    {"filename": filename},
                    request_timeout,
                )
                if int(restored.get("n_restored", -1)) != depth:
                    raise RuntimeError(
                        f"slot {slot} expected n_restored={depth}, got {restored.get('n_restored')}"
                    )
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [
                pool.submit(
                    request_json,
                    base_url,
                    "POST",
                    "/completion",
                    completion_payload(depth, slot, 128),
                    request_timeout,
                )
                for slot in range(parallel)
            ]
            responses = [future.result() for future in futures]
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        error = str(exc)
    wall_s = time.monotonic() - wall_start
    stop_server(process)
    log_file.close()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    timings = [response.get("timings") or {} for response in responses]
    prompt_n = [int(item.get("prompt_n", 0)) for item in timings]
    predicted_n = [int(item.get("predicted_n", 0)) for item in timings]
    prompt_ms = [float(item.get("prompt_ms", 0.0)) for item in timings]
    predicted_ms = [float(item.get("predicted_ms", 0.0)) for item in timings]
    valid = (
        error is None
        and len(responses) == parallel
        and prompt_n == [512] * parallel
        and predicted_n == [128] * parallel
        and all(value > 0 for value in prompt_ms + predicted_ms)
        and process.returncode in (0, -15)
    )
    pp512_ts = sum(prompt_n) / (max(prompt_ms) / 1000.0) if valid else 0.0
    tg128_ts = sum(predicted_n) / (max(predicted_ms) / 1000.0) if valid else 0.0
    record = {
        "arm": arm,
        "depth": depth,
        "parallel": parallel,
        "rep": rep,
        "valid": valid,
        "error": error,
        "pp512_ts": pp512_ts,
        "tg128_ts": tg128_ts,
        "wall_s": wall_s,
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "timings": timings,
        "command": command,
        "environment": held_env(env),
        "server_returncode": process.returncode,
        "log": str(log_path),
        **parse_server_log(log_text),
    }
    sample_path = out_dir / "samples" / f"p{parallel}-d{depth}-{arm}-rep{rep}.json"
    sample_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def paired_summary(samples: list[dict[str, Any]], baseline: str, candidate: str) -> dict[str, Any]:
    by_arm = {name: {item["rep"]: item for item in samples if item["arm"] == name} for name in (baseline, candidate)}
    result: dict[str, Any] = {}
    for metric in ("pp512_ts", "tg128_ts"):
        pairs = [
            (by_arm[baseline][rep][metric], by_arm[candidate][rep][metric])
            for rep in sorted(set(by_arm[baseline]) & set(by_arm[candidate]))
            if rep > 0 and by_arm[baseline][rep]["valid"] and by_arm[candidate][rep]["valid"]
        ]
        percentages = [100.0 * (candidate_value / baseline_value - 1.0) for baseline_value, candidate_value in pairs]
        result[metric] = {
            "pairs": len(pairs),
            "baseline_median": statistics.median(value[0] for value in pairs) if pairs else 0.0,
            "candidate_median": statistics.median(value[1] for value in pairs) if pairs else 0.0,
            "paired_median_pct": statistics.median(percentages) if percentages else 0.0,
            "paired_mean_pct": statistics.mean(percentages) if percentages else 0.0,
            "paired_stddev_pct": statistics.stdev(percentages) if len(percentages) > 1 else 0.0,
        }
    return result


def main() -> int:
    args = parse_args()
    server_bin = Path(args.server_bin).resolve()
    model = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    state_dir = out_dir / "states"
    for path in (out_dir, state_dir, out_dir / "logs", out_dir / "samples"):
        path.mkdir(parents=True, exist_ok=True)
    if not server_bin.is_file() or not os.access(server_bin, os.X_OK):
        raise ValueError(f"invalid server binary: {server_bin}")
    if not model.is_file() or model.stat().st_size <= 0:
        raise ValueError(f"invalid model: {model}")
    depths = parse_int_list(args.depths)
    parallels = parse_int_list(args.parallel)
    if args.repetitions < 6:
        raise ValueError("at least six repetitions are required")
    if max(depths) + 512 + 128 > args.ctx_per_slot:
        raise ValueError("ctx-per-slot is too small for the requested depth")

    since = time.strftime("%Y-%m-%d %H:%M:%S")
    before_faults = dmesg_faults(since)
    for parallel in parallels:
        for depth in depths:
            prepare_state(
                server_bin, model, state_dir, depth, parallel, args.port, args.ctx_per_slot,
                args.request_timeout, args.health_timeout, args.render_node,
            )

    samples: list[dict[str, Any]] = []
    arm_names = list(ARMS)
    for parallel in parallels:
        for depth in depths:
            for rep in range(args.repetitions):
                order = arm_names if rep % 2 == 0 else list(reversed(arm_names))
                for arm in order:
                    print(f"[p={parallel} d={depth} rep={rep} arm={arm}]", flush=True)
                    samples.append(
                        run_sample(
                            server_bin=server_bin, model=model, state_dir=state_dir,
                            out_dir=out_dir, arm=arm, depth=depth, parallel=parallel,
                            rep=rep, port=args.port, ctx_per_slot=args.ctx_per_slot,
                            request_timeout=args.request_timeout,
                            health_timeout=args.health_timeout, render_node=args.render_node,
                            adapter_trace=args.adapter_trace,
                        )
                    )

    cells: list[dict[str, Any]] = []
    for parallel in parallels:
        for depth in depths:
            cell_samples = [item for item in samples if item["parallel"] == parallel and item["depth"] == depth]
            comparisons = {
                arm: paired_summary(cell_samples, "graph-off", arm)
                for arm in arm_names[1:]
            }
            cells.append({
                "parallel": parallel,
                "depth": depth,
                "samples": cell_samples,
                "comparisons": comparisons,
            })

    after_faults = dmesg_faults(since)
    new_faults = [line for line in after_faults if line not in before_faults]
    source_commit = subprocess.run(
        ["git", "show", "-s", "--format=%H", "HEAD"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=True,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "source_commit": source_commit,
        "server_sha256": sha256_file(server_bin),
        "model": str(model),
        "model_sha256": sha256_file(model),
        "depths": depths,
        "parallel": parallels,
        "ctx_per_slot": args.ctx_per_slot,
        "repetitions": args.repetitions,
        "cells": cells,
        "before_faults": before_faults,
        "after_faults": after_faults,
        "arms": {name: arm_environment(name, args.adapter_trace) for name in ARMS},
        "all_samples_valid": all(item["valid"] for item in samples),
    }
    result_path = out_dir / "submission.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_samples_valid": result["all_samples_valid"], "new_faults": new_faults, "result": str(result_path)}))
    return 0 if result["all_samples_valid"] and not new_faults else 1


if __name__ == "__main__":
    raise SystemExit(main())
